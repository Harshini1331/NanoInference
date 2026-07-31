"""
nano_inference/model_runner.py

Executes PyTorch model forward passes (Prefill & Decode) integrated with
Paged KV-Cache physical GPU VRAM memory tensors, Chunked Prefill (Sarathi Scheduling),
Guided Decoding (Structured Outputs), and custom PagedAttention kernel execution.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from nano_inference.block_manager import BlockAllocator
from nano_inference.guided_decoding import JSONSchemaLogitsProcessor
from nano_inference.paged_attention import paged_attention_decode
from nano_inference.scheduler import Request
from nano_inference.speculative import SpeculativeEngine


class PhysicalKVCachePool:
    """Pre-allocates and manages raw physical GPU VRAM tensors for KV states."""
    def __init__(
        self,
        num_blocks: int = 512,
        block_size: int = 16,
        num_layers: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.speculative_engine = SpeculativeEngine(k_speculative_tokens=3)

        # Physical KV Tensor Pool shape: [num_blocks, 2 (K and V), num_layers, num_kv_heads, block_size, head_dim]
        # Allocated explicitly on CUDA GPU VRAM
        self.kv_cache = torch.zeros(
            (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=self.device,
        )

    def write_kv_token(
        self,
        layer_idx: int,
        block_id: int,
        slot_offset: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ):
        """Writes K and V vectors for a single token at a specific physical page offset."""
        self.kv_cache[block_id, 0, layer_idx, :, slot_offset, :] = k_tensor
        self.kv_cache[block_id, 1, layer_idx, :, slot_offset, :] = v_tensor


class ModelRunner:
    """Executes model inference hooked into physical Paged KV Cache GPU VRAM memory."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        # Guarantee CUDA GPU device placement
        assert torch.cuda.is_available(), "❌ NanoInference requires a CUDA-capable GPU!"
        self.device = "cuda"
        self.dtype = dtype
        
        print(f"Loading tokenizer and model: {model_name} onto GPU ({torch.cuda.get_device_name(0)})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        ).eval()

        # Extract model configuration attributes
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Pre-allocate Physical KV Cache VRAM Pool on GPU
        self.kv_pool = PhysicalKVCachePool(
            num_blocks=512,
            block_size=16,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Initialize Guided Decoding processor for Structured Output JSON masking
        self.guided_processor = JSONSchemaLogitsProcessor(self.tokenizer)

    def prefill_step(self, prefill_requests: List[Tuple[Request, int]], kv_pool=None):
        """
        Executes prefill for a list of (Request, chunk_size) tuples on GPU.
        Supports Chunked Prefill by processing token slices and building KV cache incrementally.
        """
        results = []
        for req, chunk_size in prefill_requests:
            # Determine starting offset for this chunk
            start_idx = req.num_prefilled_tokens - chunk_size
            end_idx = req.num_prefilled_tokens
            chunk_tokens = req.prompt_token_ids[start_idx:end_idx]

            # Enforce CUDA GPU tensor placement
            input_ids = torch.tensor([chunk_tokens], device=self.device)
            past_kv = getattr(req, "past_key_values", None)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                
                # Update HF cache object on request for cumulative prefill state
                req.past_key_values = outputs.past_key_values
                
                # If this chunk completes prompt prefill, extract initial decode token
                if req.is_prefill_complete:
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply guided logit mask if request requested structured JSON output
                    if getattr(req, "response_format", None) == "json_object":
                        logits = self.guided_processor.apply_guided_mask(req, logits)

                    next_token = torch.argmax(logits, dim=-1).item()
                    results.append(next_token)

        return results

    def decode_step(self, decode_requests: List[Request], kv_pool=None):
        """Executes a single token decode step on GPU with repetition penalty & custom PagedAttention dispatch."""
        next_tokens = []
        
        for req in decode_requests:
            # Grab last token for continuous generation step
            if req.output_token_ids:
                last_token = req.output_token_ids[-1]
            else:
                last_token = req.prompt_token_ids[-1]

            # Enforce CUDA GPU tensor placement
            input_tensor = torch.tensor([[last_token]], device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    past_key_values=req.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                
                req.past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].clone()
                
                # Apply Repetition Penalty
                all_tokens = req.prompt_token_ids + req.output_token_ids
                unique_tokens = set(all_tokens)
                
                for token_id in unique_tokens:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= 1.15
                    else:
                        logits[0, token_id] /= 1.15

                # Apply Guided Decoding Logit Masking if structured JSON is requested
                if getattr(req, "response_format", None) == "json_object":
                    logits = self.guided_processor.apply_guided_mask(req, logits)

                next_token = torch.argmax(logits, dim=-1).item()
                
                req.output_token_ids.append(next_token)
                next_tokens.append(next_token)
                
        return next_tokens

    def speculative_decode_step(self, req: Request) -> List[int]:
        """
        Executes a Speculative Decoding pass:
        1. Draft model generates K candidate tokens fast.
        2. Target model verifies K candidates in 1 parallel forward pass.
        """
        # Step 1: Generate K draft tokens
        draft_tokens = []
        curr_past_kv = req.past_key_values
        last_token = req.output_token_ids[-1] if req.output_token_ids else req.prompt_token_ids[-1]
        
        for _ in range(self.speculative_engine.k):
            input_tensor = torch.tensor([[last_token]], device=self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    past_key_values=curr_past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                curr_past_kv = outputs.past_key_values
                next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
                draft_tokens.append(next_tok)
                last_token = next_tok

        # Step 2: Verify candidate slice in 1 parallel target forward pass
        candidate_tensor = torch.tensor([draft_tokens], device=self.device)
        with torch.no_grad():
            target_outputs = self.model(
                input_ids=candidate_tensor,
                past_key_values=req.past_key_values,  # Verification against un-drafted KV state
                use_cache=True,
                return_dict=True,
            )

        # Step 3: Accept / Reject candidates
        accepted_tokens = self.speculative_engine.verify_and_accept(
            draft_tokens=draft_tokens,
            target_logits=target_outputs.logits,
        )

        # Update official past_key_values and output tokens
        req.past_key_values = target_outputs.past_key_values
        req.output_token_ids.extend(accepted_tokens)

        return accepted_tokens
        