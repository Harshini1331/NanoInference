"""
benchmarks/benchmark_throughput.py

Automated benchmarking suite for NanoInference Engine.
Simulates realistic multi-user concurrent traffic (Poisson request distribution)
and calculates TTFT, ITL, Total Generation Time, and Tokens/Sec throughput.
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Tuple

import aiohttp


# Sample prompt pool for realistic inference variability
PROMPT_POOL = [
    "Explain the mechanics of KV cache paging in virtual memory architectures.",
    "Write a short Python function to calculate Fibonacci numbers recursively.",
    "Summarize the main trade-offs between continuous batching and naive batching in LLMs.",
    "Generate a structured JSON object with keys name, role, and skills.",
    "What is the difference between prefill and decode phases in Transformer inference?",
]


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int = 32,
) -> Dict[str, float]:
    """
    Sends a single SSE streaming request, measuring TTFT, total tokens,
    generation duration, and inter-token latency (ITL).
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    async with session.post(url, json=payload) as response:
        if response.status != 200:
            return {"error": response.status}

        async for line in response.content:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: ") and line_str != "data: [DONE]":
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()

    end_time = time.perf_counter()

    total_duration = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_duration
    decode_duration = (end_time - first_token_time) if first_token_time else 0.0
    itl = (decode_duration / max(1, token_count - 1)) if token_count > 1 else 0.0

    return {
        "ttft": ttft,
        "total_duration": total_duration,
        "tokens_generated": token_count,
        "itl": itl,
    }


async def run_benchmark_for_concurrency(
    url: str,
    concurrency: int,
    requests_per_worker: int = 4,
    poisson_lambda: float = 0.05,
) -> Dict[str, float]:
    """
    Executes concurrent requests simulating a Poisson process arrival distribution.
    """
    total_requests = concurrency * requests_per_worker
    print(f"\n🚀 Running benchmark: Concurrency N={concurrency} (Total Requests: {total_requests})...")

    async with aiohttp.ClientSession() as session:
        tasks = []

        for i in range(total_requests):
            prompt = random.choice(PROMPT_POOL)
            
            # Poisson request arrival interval simulation
            if poisson_lambda > 0:
                await asyncio.sleep(random.expovariate(1.0 / poisson_lambda))

            task = asyncio.create_task(
                send_streaming_request(session, url, prompt=prompt, max_tokens=32)
            )
            tasks.append(task)

        # Gather results across all concurrent streams
        results = await asyncio.gather(*tasks)

    # Filter out errored requests
    valid_results = [r for r in results if "error" not in r and r["tokens_generated"] > 0]

    if not valid_results:
        print("❌ All requests failed during run!")
        return {}

    # Aggregated Performance Metrics
    avg_ttft = sum(r["ttft"] for r in valid_results) / len(valid_results)
    avg_itl = sum(r["itl"] for r in valid_results) / len(valid_results)
    total_tokens = sum(r["tokens_generated"] for r in valid_results)
    total_wall_time = max(r["total_duration"] for r in valid_results)
    tokens_per_sec = total_tokens / total_wall_time

    print(f"  ├─ Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"  ├─ Average ITL:  {avg_itl * 1000:.2f} ms/token")
    print(f"  ├─ Total Tokens Generated: {total_tokens}")
    print(f"  └─ Total Throughput: {tokens_per_sec:.2f} Tokens/sec")

    return {
        "concurrency": concurrency,
        "avg_ttft_ms": avg_ttft * 1000,
        "avg_itl_ms": avg_itl * 1000,
        "throughput_tok_s": tokens_per_sec,
    }


async def main():
    server_url = "http://127.0.0.1:8000/v1/chat/completions"
    concurrency_levels = [1, 8, 16, 32]
    summary_table = []

    print("=" * 65)
    print("      NANOINFERENCE BENCHMARK SUITE - THROUGHPUT & LATENCY      ")
    print("=" * 65)

    for N in concurrency_levels:
        stats = await run_benchmark_for_concurrency(
            url=server_url, concurrency=N, requests_per_worker=2, poisson_lambda=0.02
        )
        if stats:
            summary_table.append(stats)

    # Output Markdown Results Table
    print("\n" + "=" * 65)
    print("                      BENCHMARK RESULTS TABLE                    ")
    print("=" * 65)
    print("| Concurrency (N) | Avg TTFT (ms) | Avg ITL (ms/tok) | Throughput (tok/s) |")
    print("|-----------------|---------------|------------------|--------------------|")
    for row in summary_table:
        print(
            f"| {row['concurrency']:<15} | {row['avg_ttft_ms']:<13.2f} | {row['avg_itl_ms']:<16.2f} | {row['throughput_tok_s']:<18.2f} |"
        )
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())