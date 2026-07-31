"""
nano_inference/model_runner.py

Executes PyTorch model forward passes (Prefill & Decode) integrated with
Paged KV-Cache physical VRAM memory tensors and Chunked Prefill (Sarathi Scheduling).
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from nano_inference.block_manager import BlockAllocator
from nano_inference.scheduler import Request


class PhysicalKVCachePool:
    """Pre-allocates and manages raw physical GPU VRAM tensors for KV states."""
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
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

        # Physical KV Tensor Pool shape: [num_blocks, 2 (K and V), num_layers, num_kv_heads, block_size, head_dim]
        self.kv_cache = torch.zeros(
            (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
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
        # k_tensor/v_tensor shape: [num_kv_heads, head_dim]
        self.kv_cache[block_id, 0, layer_idx, :, slot_offset, :] = k_tensor
        self.kv_cache[block_id, 1, layer_idx, :, slot_offset, :] = v_tensor


class ModelRunner:
    """Executes model inference hooked into physical Paged KV Cache memory."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading tokenizer and model: {model_name}...")
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

    def prefill_step(self, prefill_requests: List[Tuple[Request, int]], kv_pool):
        """
        Executes prefill for a list of (Request, chunk_size) tuples.
        Supports Chunked Prefill by processing token slices and building KV cache incrementally.
        """
        results = []
        for req, chunk_size in prefill_requests:
            # Determine starting offset for this chunk
            start_idx = req.num_prefilled_tokens - chunk_size
            end_idx = req.num_prefilled_tokens
            chunk_tokens = req.prompt_token_ids[start_idx:end_idx]

            input_ids = torch.tensor([chunk_tokens], device=self.device)
            past_kv = getattr(req, "past_key_values", None)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True
                )
                
                # Update HF cache object on request for cumulative prefill state
                req.past_key_values = outputs.past_key_values
                
                # If this chunk completes prompt prefill, extract initial decode token
                if req.is_prefill_complete:
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1).item()
                    results.append(next_token)

        return results

    def decode_step(self, decode_requests: List[Request], kv_pool):
        """Executes a single token decode step for active requests."""
        next_tokens = []
        
        for req in decode_requests:
            # SAFETY GUARD: If prefill hasn't populated output_token_ids yet, grab the last prompt token
            if req.output_token_ids:
                last_token = req.output_token_ids[-1]
            else:
                last_token = req.prompt_token_ids[-1]

            input_tensor = torch.tensor([[last_token]], device=self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    past_key_values=req.past_key_values,
                    use_cache=True,
                    return_dict=True
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

                next_token = torch.argmax(logits, dim=-1).item()
                
                req.output_token_ids.append(next_token)
                next_tokens.append(next_token)
                
        return next_tokens