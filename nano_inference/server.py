"""
OpenAI-Compatible FastAPI Server for NanoInference Engine.
Includes Server-Sent Events (SSE) token streaming and Prometheus observability metrics.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Response, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Import engine core components
from nano_inference.block_manager import BlockAllocator
from nano_inference.model_runner import ModelRunner
from nano_inference.scheduler import Request as EngineRequest, RequestStatus, Scheduler

# Initialize FastAPI App
app = FastAPI(
    title="NanoInference Engine API",
    description="High-performance LLM serving gateway with custom PagedAttention and Continuous Batching",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Engine Hardware & Core Initialization
# -----------------------------------------------------------------------------
# Initialize 512 physical KV blocks (block size = 16 tokens -> 8,192 token capacity)
allocator = BlockAllocator(total_num_blocks=512, block_size=16)
scheduler = Scheduler(allocator=allocator, max_num_batched_tokens=2048, max_num_seqs=32)
model_runner = ModelRunner(model_name="Qwen/Qwen2.5-0.5B-Instruct")
kv_pool = {}  # Global KV-cache execution tensor pool


# -----------------------------------------------------------------------------
# 📊 Prometheus Telemetry Definitions
# -----------------------------------------------------------------------------
KV_CACHE_USAGE = Gauge(
    "nanoinference_kv_cache_usage_percent",
    "Percentage of allocated physical KV blocks in GPU VRAM",
)
REQUESTS_RUNNING = Gauge(
    "nanoinference_requests_running",
    "Number of currently active decode streams",
)
REQUESTS_WAITING = Gauge(
    "nanoinference_requests_waiting",
    "Number of requests queued in prefill queue",
)
TTFT_HISTOGRAM = Histogram(
    "nanoinference_ttft_seconds",
    "Time to First Token (TTFT) in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
TOTAL_TOKENS_GENERATED = Counter(
    "nanoinference_tokens_generated_total",
    "Total output tokens generated across all streams",
)


# -----------------------------------------------------------------------------
# Pydantic Schemas (OpenAI Spec API)
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=128, ge=1, le=2048)
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True


# -----------------------------------------------------------------------------
# Engine Streaming Generator
# -----------------------------------------------------------------------------
async def generate_stream(
    req: EngineRequest, raw_request: FastAPIRequest
) -> AsyncGenerator[str, None]:
    """
    Executes continuous batching loop per iteration, yielding SSE token chunks
    and logging Prometheus metrics.
    """
    start_time = time.perf_counter()
    first_token_recorded = False

    try:
        while True:
            # Detect client disconnect to prevent orphan request VRAM block leaks
            if await raw_request.is_disconnected():
                scheduler.free_finished_request(req)
                break

            outputs = scheduler.schedule()

            # Execute Prefill Phase for newly scheduled requests
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
                    token_str = model_runner.tokenizer.decode([token_id])

                    # Record Time to First Token (TTFT) metric
                    if not first_token_recorded:
                        ttft = time.perf_counter() - start_time
                        TTFT_HISTOGRAM.observe(ttft)
                        first_token_recorded = True

                    # Increment global output token counter metric
                    TOTAL_TOKENS_GENERATED.inc()

                    # Yield OpenAI-formatted SSE payload chunk
                    chunk = {
                        "id": f"chatcmpl-{req.request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_str},
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
        # Guarantee physical KV memory blocks are freed back to allocator
        scheduler.free_finished_request(req)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, raw_request: FastAPIRequest):
    """OpenAI-compatible chat completion endpoint supporting SSE token streaming."""
    # Convert input messages into standard prompt token IDs via model tokenizer
    prompt_text = body.messages[-1].content
    prompt_token_ids = model_runner.tokenizer.encode(prompt_text)

    # Initialize new request object and enqueue into waiting queue
    req_id = f"req-{int(time.time() * 1000)}"
    req = EngineRequest(
        request_id=req_id,
        prompt_token_ids=prompt_token_ids,
        max_tokens=body.max_tokens,
    )
    scheduler.add_request(req)

    return StreamingResponse(
        generate_stream(req, raw_request), media_type="text/event-stream"
    )


@app.get("/metrics")
async def get_metrics():
    """Exposes real-time Prometheus system telemetry and KV-cache metrics."""
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