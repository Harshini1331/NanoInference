"""
nano_inference/server.py

Asynchronous FastAPI Gateway providing an OpenAI-compatible SSE streaming endpoint
for the NanoInference engine.
"""

import asyncio
import json
import uuid
from typing import List, Optional
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nano_inference.block_manager import BlockAllocator
from nano_inference.scheduler import Request as InferenceRequest, Scheduler
from nano_inference.model_runner import ModelRunner, PhysicalKVCachePool

app = FastAPI(title="NanoInference Serving Engine")

# Global Engine Components
allocator: Optional[BlockAllocator] = None
scheduler: Optional[Scheduler] = None
model_runner: Optional[ModelRunner] = None
kv_pool: Optional[PhysicalKVCachePool] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[ChatMessage]
    max_tokens: int = 128
    stream: bool = True


@app.on_event("startup")
async def startup_event():
    global allocator, scheduler, model_runner, kv_pool
    print("🚀 Initializing NanoInference Engine...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    allocator = BlockAllocator(total_num_blocks=128, block_size=16)
    scheduler = Scheduler(allocator=allocator, max_num_batched_tokens=2048, max_num_seqs=16)

    model_runner = ModelRunner(model_name="Qwen/Qwen2.5-0.5B-Instruct", device=device)
    kv_pool = PhysicalKVCachePool(
        num_blocks=128,
        block_size=16,
        num_layers=model_runner.num_layers,
        num_kv_heads=model_runner.num_kv_heads,
        head_dim=model_runner.head_dim,
        device=device,
    )
    print("✅ NanoInference Engine Ready!")


async def generate_stream(request_id: str, prompt_tokens: list, max_tokens: int):
    """Asynchronous token generator yielding SSE data chunks cleanly."""
    req = InferenceRequest(request_id=request_id, prompt_token_ids=prompt_tokens, max_tokens=max_tokens)
    
    scheduler.add_request(req)

    generated_count = 0
    eos_token_id = model_runner.tokenizer.eos_token_id

    while generated_count < max_tokens:
        await asyncio.sleep(0.001)

        outputs = scheduler.schedule()

        # Handle Prefill
        if outputs.prefill_requests:
            prefill_tokens = model_runner.prefill_step(outputs.prefill_requests, kv_pool)
            for r, token_id in zip([pr[0] for pr in outputs.prefill_requests], prefill_tokens):
                if r.request_id == request_id:
                    # Append to output_token_ids (matching scheduler.py)
                    r.output_token_ids.append(token_id)
                    generated_count += 1
                    
                    if token_id == eos_token_id:
                        generated_count = max_tokens
                        break
                    
                    token_str = model_runner.tokenizer.decode([token_id], skip_special_tokens=True)
                    yield f"data: {json.dumps({'token': token_str})}\n\n"

        # Handle Decode Steps
        if outputs.decode_requests:
            decode_tokens = model_runner.decode_step(outputs.decode_requests, kv_pool)
            for r, token_id in zip(outputs.decode_requests, decode_tokens):
                if r.request_id == request_id:
                    generated_count += 1
                    
                    # Stop generation if EOS or stop token hit
                    if token_id in (model_runner.tokenizer.eos_token_id, 151645): # 151645 is Qwen <|im_end|>
                        generated_count = max_tokens
                        break
                    
                    token_str = model_runner.tokenizer.decode([token_id], skip_special_tokens=True)
                    yield f"data: {json.dumps({'token': token_str})}\n\n"

    scheduler.free_finished_request(req)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    
    # Format messages using Qwen's chat template
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    # 1. Generate chat prompt string
    prompt_str = model_runner.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 2. Explicitly encode to raw integer token IDs list
    prompt_tokens = model_runner.tokenizer.encode(prompt_str)

    return StreamingResponse(
        generate_stream(request_id=request_id, prompt_tokens=prompt_tokens, max_tokens=req.max_tokens),
        media_type="text/event-stream",
    )