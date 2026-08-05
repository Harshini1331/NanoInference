#!/bin/bash
# profile_nsys.sh
# Profiles NanoInference Triton kernel execution using NVIDIA Nsight Systems

echo "🚀 Running Nsight Systems Profiler on NanoInference..."

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=profiles/nanoinference_triton_trace \
  --force-overwrite=true \
  python -m benchmarks.benchmark_triton

echo "✅ Profiling complete! Generated 'profiles/nanoinference_triton_trace.nsys-rep'"