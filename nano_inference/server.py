"""
OpenAI-Compatible FastAPI Server for NanoInference Engine.
Includes Server-Sent Events (SSE) token streaming, Automatic Prefix Caching (APC) support,
Chunked Prefill (Sarathi), Guided Decoding (Structured Outputs), EOS stop-token detection,
Multi-LoRA adapter routing, and Prometheus observability.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Response, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# Import engine core components
from nano_inference.block_manager import BlockAllocator
from nano_inference.model_runner import ModelRunner
from nano_inference.scheduler import Request as EngineRequest, RequestStatus, Scheduler

# Import metrics from dedicated metrics module to prevent circular imports
from nano_inference.metrics import (
    KV_CACHE_USAGE,
    PREFIX_CACHE_HITS,
    PREFIX_CACHE_MISSES,
    REQUESTS_RUNNING,
    REQUESTS_WAITING,
    TOTAL_TOKENS_GENERATED,
    TTFT_HISTOGRAM,
)

# Initialize FastAPI App
app = FastAPI(
    title="NanoInference Engine API",
    description="High-performance LLM serving gateway with custom PagedAttention, Continuous Batching, Automatic Prefix Caching, Chunked Prefill, and Guided Decoding",
    version="1.2.0",
)

# -----------------------------------------------------------------------------
# Engine Hardware & Core Initialization
# -----------------------------------------------------------------------------
# Initialize 512 physical KV blocks (block size = 16 tokens -> 8,192 token capacity)
allocator = BlockAllocator(total_num_blocks=512, block_size=16)
scheduler = Scheduler(
    allocator=allocator,
    max_num_batched_tokens=2048,
    max_num_seqs=32,
    chunk_size=256,  # Sarathi Chunked Prefill threshold
)
model_runner = ModelRunner(model_name="Qwen/Qwen2.5-0.5B-Instruct")
kv_pool = {}  # Global KV-cache execution tensor pool


# -----------------------------------------------------------------------------
# Pydantic Schemas (OpenAI Spec API)
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ResponseFormat(BaseModel):
    type: str = "json_object"


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=128, ge=1, le=2048)
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    response_format: Optional[ResponseFormat] = None  # OpenAI Structured Outputs Schema
    adapter_id: Optional[str] = None  # Dynamic LoRA adapter routing ID


# -----------------------------------------------------------------------------
# Engine Streaming Generator
# -----------------------------------------------------------------------------
async def generate_stream(
    req: EngineRequest, raw_request: FastAPIRequest
) -> AsyncGenerator[str, None]:
    """
    Executes continuous batching loop per iteration, yielding SSE token chunks,
    detecting EOS stop tokens, and logging Prometheus metrics.
    """
    start_time = time.perf_counter()
    first_token_recorded = False
    
    tokenizer = model_runner.tokenizer
    eos_token_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    stop_ids = {eos_token_id, im_end_id, endoftext_id}

    try:
        while True:
            # Detect client disconnect to prevent orphan request VRAM block leaks
            if await raw_request.is_disconnected():
                scheduler.free_finished_request(req)
                break

            outputs = scheduler.schedule()

            # Execute Prefill Phase for newly scheduled requests or chunks
            if outputs.prefill_requests:
                model_runner.prefill_step(outputs.prefill_requests, kv_pool)

            # Execute Decode Phase for running streams
            if outputs.decode_requests:
                tokens = model_runner.decode_step(outputs.decode_requests, kv_pool)

                # Find generated token corresponding to this specific request stream
                if req in outputs.decode_requests:
                    req_idx = outputs.decode_requests.index(req)
                    token_id = tokens[req_idx]

                    # Convert integer token ID to text string
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)

                    # Check for EOS or chat stop sequences
                    is_eos = (token_id in stop_ids) or ("<|im_end|>" in token_str) or ("<|endoftext|>" in token_str)

                    if is_eos:
                        req.status = RequestStatus.FINISHED
                        yield "data: [DONE]\n\n"
                        break

                    # Clean special tokens for SSE stream output
                    clean_token_str = tokenizer.decode([token_id], skip_special_tokens=True)

                    # Record Time to First Token (TTFT) metric
                    if not first_token_recorded:
                        ttft = time.perf_counter() - start_time
                        TTFT_HISTOGRAM.observe(ttft)
                        first_token_recorded = True

                    # Increment global output token counter metric
                    TOTAL_TOKENS_GENERATED.inc()

                    # Yield OpenAI-formatted SSE payload chunk
                    if clean_token_str:
                        chunk = {
                            "id": f"chatcmpl-{req.request_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": clean_token_str},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            # Check if request reaches max_tokens limit or finishes
            if (
                req.status == RequestStatus.FINISHED
                or len(req.output_token_ids) >= req.max_tokens
            ):
                yield "data: [DONE]\n\n"
                break

            await asyncio.sleep(0.001)  # Yield execution back to event loop

    finally:
        # Guarantee physical KV memory blocks are freed back to allocator or LRU queue
        scheduler.free_finished_request(req)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, raw_request: FastAPIRequest):
    """OpenAI-compatible chat completion endpoint supporting SSE token streaming, Chat Templates, and Structured Outputs."""
    
    # 1. Format full chat history using tokenizer chat template (<|im_start|>user..., etc.)
    # Fallback to plain prompt_text string if model tokenizer doesn't have a chat template configured
    try:
        messages_dict = [m.model_dump() for m in body.messages]
        prompt_text = model_runner.tokenizer.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for raw text prompts
        prompt_text = body.messages[-1].content

    # 2. Encode formatted prompt into token IDs
    prompt_token_ids = model_runner.tokenizer.encode(prompt_text)

    # 3. Initialize engine request object
    req_id = f"req-{int(time.time() * 1000)}"
    req = EngineRequest(
        request_id=req_id,
        prompt_token_ids=prompt_token_ids,
        max_tokens=body.max_tokens,
    )

    # 4. Attach optional attributes
    req.prompt = prompt_text
    req.response_format = body.response_format.type if body.response_format else None
    req.adapter_id = body.adapter_id

    scheduler.add_request(req)

    return StreamingResponse(
        generate_stream(req, raw_request), media_type="text/event-stream"
    )


@app.get("/metrics")
async def get_metrics():
    """Exposes real-time Prometheus system telemetry, APC hits, and KV-cache metrics."""
    total_blocks = allocator.total_num_blocks
    free_blocks = allocator.num_free_blocks
    allocated_blocks = total_blocks - free_blocks

    # Update dynamic Prometheus system gauges
    KV_CACHE_USAGE.set(
        (allocated_blocks / total_blocks) * 100.0 if total_blocks > 0 else 0.0
    )
    REQUESTS_RUNNING.set(len(scheduler.running_queue))
    REQUESTS_WAITING.set(len(scheduler.waiting_queue))

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "engine": "NanoInference"}