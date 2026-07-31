import asyncio
import json
import time
import argparse
import aiohttp
import numpy as np


async def send_request(session: aiohttp.ClientSession, url: str, prompt: str, max_tokens: int):
    """Sends a single request to the NanoInference engine and tracks latency metrics."""
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps = []
    generated_tokens = 0

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"❌ Error: Status {response.status}")
                return None

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                
                if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                    now = time.perf_counter()
                    
                    if first_token_time is None:
                        first_token_time = now
                    
                    token_timestamps.append(now)
                    generated_tokens += 1

        end_time = time.perf_counter()

        if generated_tokens == 0 or first_token_time is None:
            return None

        # Metrics computation
        ttft = (first_token_time - start_time) * 1000  # ms
        total_latency = end_time - start_time  # sec
        
        # Inter-Token Latencies
        itls = []
        for i in range(1, len(token_timestamps)):
            itls.append((token_timestamps[i] - token_timestamps[i - 1]) * 1000) # ms

        avg_itl = np.mean(itls) if itls else 0.0

        return {
            "ttft": ttft,
            "total_latency": total_latency,
            "generated_tokens": generated_tokens,
            "avg_itl": avg_itl
        }

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


async def run_benchmark(url: str, concurrency: int, total_requests: int, max_tokens: int):
    """Runs concurrent load test against the serving backend."""
    prompt = "Explain quantum computing in detail and why it is important for cryptography."
    
    print(f"\n🚀 Starting Benchmark...")
    print(f"• URL: {url}")
    print(f"• Concurrency Level: {concurrency}")
    print(f"• Total Requests: {total_requests}")
    print(f"• Max Tokens per Request: {max_tokens}\n")

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session):
        async with semaphore:
            return await send_request(session, url, prompt, max_tokens)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session) for _ in range(total_requests)]
        start_benchmark_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_benchmark_time = time.perf_counter() - start_benchmark_time

    # Filter valid results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("❌ All requests failed.")
        return

    # Aggregate Statistics
    ttfts = [r["ttft"] for r in valid_results]
    itls = [r["avg_itl"] for r in valid_results if r["avg_itl"] > 0]
    total_tokens = sum(r["generated_tokens"] for r in valid_results)
    system_throughput = total_tokens / total_benchmark_time

    print("=" * 50)
    print("📊 NANOINFERENCE BENCHMARK RESULTS")
    print("=" * 50)
    print(f" Successful Requests : {len(valid_results)} / {total_requests}")
    print(f" Total Execution Time: {total_benchmark_time:.2f} s")
    print(f" Total Tokens Output : {total_tokens} tokens")
    print(f" System Throughput   : {system_throughput:.2f} tokens/sec")
    print("-" * 50)
    print(f" TTFT (P50 / Median) : {np.median(ttfts):.2f} ms")
    print(f" TTFT (P95)          : {np.percentile(ttfts, 95):.2f} ms")
    print(f" TTFT (P99)          : {np.percentile(ttfts, 99):.2f} ms")
    print("-" * 50)
    if itls:
        print(f" ITL (Average)       : {np.mean(itls):.2f} ms/token")
        print(f" ITL (P95)           : {np.percentile(itls, 95):.2f} ms/token")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NanoInference Gateway")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent streams")
    parser.add_argument("--requests", type=int, default=10, help="Total number of requests")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate per request")

    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url, args.concurrency, args.requests, args.max_tokens))