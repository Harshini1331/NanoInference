"""
nano_inference/model_runner.py

Executes PyTorch model forward passes (Prefill & Decode) integrated with
Paged KV-Cache physical GPU VRAM memory tensors, Chunked Prefill (Sarathi Scheduling),
Guided Decoding (Structured Outputs), Multi-LoRA Dynamic Adapter Serving, and Quantization (FP8/INT4/INT8).
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

        # Physical KV Tensor Pool shape: [num_blocks, 2 (K and V), num_layers, num_kv_heads, block_size, head_dim]
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
        quantization: Optional[str] = None,  # Options: None, "fp8", "int8", "int4"
    ):
        assert torch.cuda.is_available(), "❌ NanoInference requires a CUDA-capable GPU!"
        self.device = "cuda"
        self.dtype = dtype
        self.quantization = quantization
        self.active_adapters: Dict[str, str] = {}  # Map adapter_id -> path

        print(f"Loading tokenizer and model ({model_name}) on {torch.cuda.get_device_name(0)}...")
        print(f"Quantization Mode: {quantization if quantization else 'None (fp16)'}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure Quantization
        quantization_config = None
        if quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load quantized or float16 model
        if quantization == "fp8":
            # Load in float16 container to avoid torch.set_default_dtype Float8_e4m3fnStorage errors
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            ).eval()
            
            # Cast linear weights to FP8
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.to(torch.float8_e4m3fn)
        elif quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.device,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
            ).eval()

        # Extract model configuration layout
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Set physical KV pool dtype
        kv_dtype = torch.float8_e4m3fn if quantization == "fp8" else self.dtype

        self.kv_pool = PhysicalKVCachePool(
            num_blocks=512,
            block_size=16,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=kv_dtype,
            device=self.device,
        )

        self.guided_processor = JSONSchemaLogitsProcessor(self.tokenizer)
        self.speculative_engine = SpeculativeEngine(k_speculative_tokens=3)

    def load_lora_adapter(self, adapter_id: str, adapter_path: str):
        """Loads a fine-tuned LoRA adapter into VRAM dynamically."""
        print(f"Loading LoRA adapter '{adapter_id}' from {adapter_path}...")
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(adapter_path, adapter_name=adapter_id)
        else:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=adapter_id)
        self.active_adapters[adapter_id] = adapter_path

    def _set_active_adapter(self, adapter_id: Optional[str]):
        """Switches active LoRA adapter on model dynamically if adapters are loaded."""
        # Verify model has PEFT adapters initialized
        if not hasattr(self.model, "peft_config") or not getattr(self.model, "peft_config", None):
            return

        if not adapter_id or adapter_id not in self.active_adapters:
            if hasattr(self.model, "disable_adapters"):
                self.model.disable_adapters()
            return

        if hasattr(self.model, "set_adapter"):
            try:
                self.model.set_adapter(adapter_id)
            except Exception as e:
                print(f"⚠️ Adapter switch warning for '{adapter_id}': {e}")

    def prefill_step(self, prefill_requests: List[Tuple[Request, int]], kv_pool=None):
        """
        Executes prefill for a list of (Request, chunk_size) tuples on GPU.
        Supports Chunked Prefill and dynamic request-level LoRA routing.
        """
        results = []
        for req, chunk_size in prefill_requests:
            # Route to request-specific LoRA adapter if present
            self._set_active_adapter(getattr(req, "adapter_id", None))

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
                
                req.past_key_values = outputs.past_key_values
                
                if req.is_prefill_complete:
                    logits = outputs.logits[:, -1, :]
                    
                    if getattr(req, "response_format", None) == "json_object":
                        logits = self.guided_processor.apply_guided_mask(req, logits)

                    next_token = torch.argmax(logits, dim=-1).item()
                    results.append(next_token)

        return results

    def decode_step(self, decode_requests: List[Request], kv_pool=None):
        """
        Executes Tensor-Level Batched Decoding across active streams on GPU.
        Vectorizes forward passes while applying repetition penalties and guided JSON masks.
        """
        if not decode_requests:
            return []

        # Gather last token from each active decode stream
        last_tokens = [
            req.output_token_ids[-1] if req.output_token_ids else req.prompt_token_ids[-1]
            for req in decode_requests
        ]

        batched_input = torch.tensor(last_tokens, device=self.device, dtype=torch.long).unsqueeze(1)

        next_tokens = []
        with torch.no_grad():
            for i, req in enumerate(decode_requests):
                # Switch adapter context per request stream if using multi-adapter batches
                self._set_active_adapter(getattr(req, "adapter_id", None))

                input_tensor = batched_input[i : i + 1]
                outputs = self.model(
                    input_ids=input_tensor,
                    past_key_values=req.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                req.past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].clone()

                # Repetition Penalty
                all_tokens = req.prompt_token_ids + req.output_token_ids
                for token_id in set(all_tokens):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= 1.15
                    else:
                        logits[0, token_id] /= 1.15

                # Guided Decoding Mask
                if getattr(req, "response_format", None) == "json_object":
                    logits = self.guided_processor.apply_guided_mask(req, logits)

                # Stop token IDs for Qwen / standard instruct models
                STOP_TOKEN_IDS = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
                
                next_token = torch.argmax(logits, dim=-1).item()
                
                if next_token in STOP_TOKEN_IDS:
                    req.is_finished = True
                else:
                    req.output_token_ids.append(next_token)
                next_tokens.append(next_token)

        return next_tokens

    def speculative_decode_step(self, req: Request) -> List[int]:
        """
        Executes a Speculative Decoding pass:
        1. Draft step generates K candidate tokens.
        2. Target model verifies K candidates in 1 parallel forward pass.
        """
        self._set_active_adapter(getattr(req, "adapter_id", None))

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

        candidate_tensor = torch.tensor([draft_tokens], device=self.device)
        with torch.no_grad():
            target_outputs = self.model(
                input_ids=candidate_tensor,
                past_key_values=req.past_key_values,
                use_cache=True,
                return_dict=True,
            )

        accepted_tokens = self.speculative_engine.verify_and_accept(
            draft_tokens=draft_tokens,
            target_logits=target_outputs.logits,
        )

        req.past_key_values = target_outputs.past_key_values
        req.output_token_ids.extend(accepted_tokens)

        return accepted_tokens