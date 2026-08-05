"""
tests/profile_nsys.py

Annotates PagedAttention execution steps using PyTorch NVTX ranges and CUDA Profiler APIs
for clean NVIDIA Nsight Systems (nsys) timeline visualization.
"""

import torch
import torch.cuda.nvtx as nvtx
from nano_inference.paged_attention import paged_attention_decode

def profile_run():
    if not torch.cuda.is_available():
        print("❌ CUDA required for profiling.")
        return

    device = "cuda"
    dtype = torch.float16
    num_seqs = 32
    num_heads = 14
    head_dim = 64
    block_size = 16
    context_len = 1024
    num_blocks = (context_len + block_size - 1) // block_size

    query = torch.randn((num_seqs, num_heads, head_dim), dtype=dtype, device=device)
    key_cache = torch.randn((num_seqs * num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
    value_cache = torch.randn((num_seqs * num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
    block_tables = torch.zeros((num_seqs, num_blocks), dtype=torch.int32, device=device)
    context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32, device=device)

    # Warmup
    for _ in range(5):
        _ = paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale=0.125)
    torch.cuda.synchronize()

    # Start CUDA Profiler and push NVTX range
    print("⚡ Starting CUDA profile collection...")
    torch.cuda.cudart().cudaProfilerStart()
    nvtx.range_push("Triton PagedAttention Decode Phase")
    
    for _ in range(50):
        _ = paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale=0.125)
    
    torch.cuda.synchronize()
    nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print("✅ Profile pass complete!")

if __name__ == "__main__":
    profile_run()