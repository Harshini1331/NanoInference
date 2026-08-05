"""
benchmarks/roofline_profiler.py

Calculates Arithmetic Intensity and measures performance against the GPU Roofline.
Computes memory loading bytes, FLOPs execution, and identifies hardware utilization gaps.
"""

import time
import torch
import pandas as pd
from nano_inference.paged_attention import paged_attention_decode


def get_gpu_hardware_roofline():
    """Returns peak FLOPs and memory bandwidth estimates based on active GPU type."""
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # Defaults: standard consumer GPU (e.g. RTX 4070 Laptop / RTX 4060)
    peak_tflops = 15.0      # FP16 peak TFLOPs
    peak_bandwidth_gbs = 200.0 # GB/s bandwidth
    
    if "h100" in gpu_name:
        peak_tflops = 1979.0
        peak_bandwidth_gbs = 3350.0
    elif "a100" in gpu_name:
        peak_tflops = 312.0
        peak_bandwidth_gbs = 2039.0
    elif "4090" in gpu_name:
        peak_tflops = 330.0
        peak_bandwidth_gbs = 1008.0
    elif "3090" in gpu_name:
        peak_tflops = 71.0
        peak_bandwidth_gbs = 936.0
    elif "5070" in gpu_name:
        peak_tflops = 30.0
        peak_bandwidth_gbs = 384.0
        
    return peak_tflops, peak_bandwidth_gbs


def profile_roofline():
    if not torch.cuda.is_available():
        print("[FAIL] CUDA required for Roofline analysis.")
        return
        
    device = "cuda"
    dtype = torch.float16
    bytes_per_el = 2 # FP16 is 2 bytes
    
    # Target benchmark parameters
    batch_sizes = [1, 8, 16, 32, 64]
    num_heads = 16
    head_dim = 64
    block_size = 16
    context_len = 1024
    scale = 1.0 / (head_dim ** 0.5)
    
    num_blocks = (context_len + block_size - 1) // block_size
    peak_tflops, peak_bandwidth_gbs = get_gpu_hardware_roofline()
    
    print("=" * 85)
    print("[ROOFLINE] NANOINFERENCE ROOFLINE MODEL & PERFORMANCE FUNNEL ANALYZER")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Target GPU Specs: Peak Compute = {peak_tflops} TFLOPs | Memory Bandwidth = {peak_bandwidth_gbs} GB/s")
    print("=" * 85)
    
    results = []
    
    for b in batch_sizes:
        # Allocations
        query = torch.randn((b, num_heads, head_dim), dtype=dtype, device=device)
        key_cache = torch.randn((b * num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        value_cache = torch.randn((b * num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        
        block_tables = torch.zeros((b, num_blocks), dtype=torch.int32, device=device)
        for i in range(b):
            block_tables[i] = torch.arange(i * num_blocks, (i + 1) * num_blocks, dtype=torch.int32)
            
        context_lens = torch.full((b,), context_len, dtype=torch.int32, device=device)
        
        # Warmup
        for _ in range(10):
            _ = paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale, block_size)
        torch.cuda.synchronize()
        
        # Latency benchmark
        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            _ = paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale, block_size)
        torch.cuda.synchronize()
        latency_ms = ((time.perf_counter() - start) / iters) * 1000.0
        
        # ----------------------------------------------------
        # ROOFLINE CALCULATIONS
        # ----------------------------------------------------
        # FLOPs = 4 * batch * context_len * num_heads * head_dim
        flops = 4 * b * context_len * num_heads * head_dim
        
        # Bytes loaded = read query, read KV cache, write output
        # Query: batch * heads * dim * bytes
        # KV Cache: 2 * batch * context_len * heads * dim * bytes
        # Output: batch * heads * dim * bytes
        bytes_transferred = (
            (b * num_heads * head_dim * bytes_per_el) +
            (2 * b * context_len * num_heads * head_dim * bytes_per_el) +
            (b * num_heads * head_dim * bytes_per_el)
        )
        
        # Arithmetic Intensity = FLOPs / Byte
        arithmetic_intensity = flops / bytes_transferred
        
        # Achieved Performance (TFLOPs) = flops / latency
        achieved_tflops = (flops / (latency_ms / 1000.0)) / 1e12
        
        # Achieved Bandwidth (GB/s) = bytes / latency
        achieved_bandwidth = (bytes_transferred / (latency_ms / 1000.0)) / 1e9
        
        # Roofline Limit (TFLOPs) = min(Peak TFLOPs, Intensity * Peak Bandwidth)
        roofline_limit_tflops = min(peak_tflops, (arithmetic_intensity * (peak_bandwidth_gbs * 1e9)) / 1e12)
        
        # Hardware utilization percent
        mfu = (achieved_tflops / peak_tflops) * 100.0
        roofline_efficiency = (achieved_tflops / roofline_limit_tflops) * 100.0
        
        # Determine bottleneck classification
        is_compute_bound = (arithmetic_intensity * peak_bandwidth_gbs * 1e9) > (peak_tflops * 1e12)
        bound = "Compute-Bound" if is_compute_bound else "Memory-Bound"
        
        results.append({
            "Batch": b,
            "Latency (ms)": round(latency_ms, 3),
            "Intensity (FLOP/B)": round(arithmetic_intensity, 2),
            "Achieved TFLOPs": round(achieved_tflops, 4),
            "Roofline Limit": round(roofline_limit_tflops, 4),
            "Achieved BW (GB/s)": round(achieved_bandwidth, 1),
            "Hardware MFU %": round(mfu, 2),
            "Roofline Efficiency %": round(roofline_efficiency, 2),
            "Bottleneck": bound
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 85)
    print("\n[INSIGHT] Decode Attention has a extremely low Arithmetic Intensity (~1.0 FLOP/Byte for FP16).")
    print("As a result, it hits the Memory Bandwidth roofline way before utilizing GPU tensor cores.")
    print("Maximizing performance requires high HBM bandwidth and minimizing global memory roundtrips.")


if __name__ == "__main__":
    profile_roofline()
