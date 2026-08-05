"""
benchmarks/benchmark_triton.py

Microbenchmark comparing Triton PagedAttention vs PyTorch Baseline Gather.
Measures execution latency (ms) and achieved GPU speedup across batch sizes.
"""

import time
import torch
from nano_inference.paged_attention import (
    _paged_attention_decode_kernel,
    pytorch_paged_attention_fallback,
    TRITON_AVAILABLE,
)


def run_benchmark():
    if not torch.cuda.is_available():
        print("❌ CUDA required for Triton benchmarks.")
        return

    device = "cuda"
    dtype = torch.float16

    # Benchmark dimensions
    num_seqs_list = [1, 8, 16, 32, 64]
    num_heads = 14
    head_dim = 64
    block_size = 16
    context_len = 1024
    scale = 1.0 / (head_dim ** 0.5)

    num_blocks = (context_len + block_size - 1) // block_size
    max_batch_size = max(num_seqs_list)
    total_physical_blocks = max_batch_size * num_blocks

    print("=" * 70)
    print("⚡ NANOINFERENCE: TRITON vs PYTORCH PAGEDATTENTION BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Context Length: {context_len} tokens | Head Dim: {head_dim}")
    print("=" * 70)
    print(
        f"{'Batch Size':<12} | {'PyTorch Latency':<18} | {'Triton Latency':<18} | {'Speedup':<10}"
    )
    print("-" * 70)

    for num_seqs in num_seqs_list:
        # Dummy Query: [num_seqs, num_heads, head_dim]
        query = torch.randn(
            (num_seqs, num_heads, head_dim), dtype=dtype, device=device
        )
        key_cache = torch.randn(
            (total_physical_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        value_cache = torch.randn(
            (total_physical_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )

        # Allocate virtual block table mapping
        block_tables = torch.zeros(
            (num_seqs, num_blocks), dtype=torch.int32, device=device
        )
        for i in range(num_seqs):
            block_tables[i] = torch.arange(
                i * num_blocks, (i + 1) * num_blocks, dtype=torch.int32
            )

        context_lens = torch.full(
            (num_seqs,), context_len, dtype=torch.int32, device=device
        )

        # Warmup PyTorch
        for _ in range(10):
            _ = pytorch_paged_attention_fallback(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                scale,
                block_size,
            )
        torch.cuda.synchronize()

        # Measure PyTorch Latency
        start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            _ = pytorch_paged_attention_fallback(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                scale,
                block_size,
            )
        torch.cuda.synchronize()
        pyt_time_ms = ((time.perf_counter() - start) / iters) * 1000.0

        # Warmup Triton Kernel
        triton_out = torch.empty_like(query)
        grid = (num_seqs, num_heads)

        for _ in range(10):
            _paged_attention_decode_kernel[grid](
                triton_out,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                scale,
                num_seqs,
                num_heads,
                head_dim=head_dim,
                stride_q_seq=query.stride(0),
                stride_q_head=query.stride(1),
                stride_k_block=key_cache.stride(0),
                stride_k_head=key_cache.stride(1),
                stride_k_tok=key_cache.stride(2),
                stride_v_block=value_cache.stride(0),
                stride_v_head=value_cache.stride(1),
                stride_v_tok=value_cache.stride(2),
                stride_out_seq=triton_out.stride(0),
                stride_out_head=triton_out.stride(1),
                stride_bt_seq=block_tables.stride(0),
            )
        torch.cuda.synchronize()

        # Measure Triton Latency
        start = time.perf_counter()
        for _ in range(iters):
            _paged_attention_decode_kernel[grid](
                triton_out,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                scale,
                num_seqs,
                num_heads,
                head_dim=head_dim,
                stride_q_seq=query.stride(0),
                stride_q_head=query.stride(1),
                stride_k_block=key_cache.stride(0),
                stride_k_head=key_cache.stride(1),
                stride_k_tok=key_cache.stride(2),
                stride_v_block=value_cache.stride(0),
                stride_v_head=value_cache.stride(1),
                stride_v_tok=value_cache.stride(2),
                stride_out_seq=triton_out.stride(0),
                stride_out_head=triton_out.stride(1),
                stride_bt_seq=block_tables.stride(0),
            )
        torch.cuda.synchronize()
        triton_time_ms = ((time.perf_counter() - start) / iters) * 1000.0

        speedup = pyt_time_ms / triton_time_ms if triton_time_ms > 0 else 1.0
        print(
            f"{num_seqs:<12} | {pyt_time_ms:15.3f} ms | {triton_time_ms:15.3f} ms | {speedup:8.2f}x"
        )


if __name__ == "__main__":
    run_benchmark()