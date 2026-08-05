# ⚡ NanoInference

A high-throughput LLM serving engine built from scratch in PyTorch, featuring a custom **Triton-accelerated PagedAttention** kernel, **continuous batching**, **dynamic Multi-LoRA adapter switching**, and real-time **Prometheus telemetry**.

Designed to implement systems-level LLM optimizations directly, without high-level wrappers.

---
## 🚀 Key Features & Highlights

- **Custom PagedAttention (Triton / PyTorch Fallback)**: Eliminates VRAM fragmentation by mapping logical token sequences to non-contiguous physical block allocations. Implemented via high-performance Triton GPU kernels with PyTorch fallback.
- **Continuous Batching & Chunked Prefill**: Iteration-level scheduling splits prefill (Sarathi-style chunking) and decode phases to maximize compute efficiency and minimize GPU idle time.
- **Custom Fused Triton RMSNorm**: Implements a memory-bandwidth optimized row-reduction normalization kernel, avoiding intermediate PyTorch tensor allocations.
- **Correctness Eval Gate**: A precision regression suite checking L1, L2, and L-infinity divergence across FP16, BF16, and FP8 precision boundaries.
- **Roofline & FLOPs Funnel Profiler**: Analyzes arithmetic intensity (FLOP/Byte) and maps achieved GPU performance against peak hardware bandwidth and compute limits.
- **Tail-Latency Load Balancer**: Simulates queue-aware request routing, least-connections, and round-robin under Poisson arrival streams to minimize $p95$ and $p99$ tail latency.
- **Dynamic Multi-LoRA Serving**: Hot-swaps fine-tuned LoRA adapters (PEFT) on-the-fly at the request level, avoiding redundant base model replication.
- **Guided Decoding (Structured Output)**: Custom LogitsProcessor enforces structural formats (e.g., JSON Schema/grammar) using logit-bias masking to ensure reliable API completions.
- **Speculative Decoding**: Accelerates token generation using draft model speculation verified by target model rejection sampling.
- **Engine-Level Quantization**: Out-of-the-box support for FP8 linear weights, as well as INT8 and INT4 (via `bitsandbytes`) quantization to optimize VRAM footprints.
- **Asynchronous FastAPI Gateway**: OpenAI-compatible endpoint (`/v1/chat/completions`) utilizing Server-Sent Events (SSE) for streaming. Features instant block reclamation on client disconnects to prevent memory leaks.
- **Prometheus Telemetry & Playground**: Exposes metrics (TTFT, inter-token latency, block utilization) natively for Grafana dashboarding. Includes a Gradio 5+ playground UI.

---

## 🏛 System Architecture & Lifecycle

```text
                               ┌──────────────────────────────────────────────┐
                               │       Client Layer (OpenAI SDK / UI)         │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Production API Gateway & Routing Layer (FastAPI)                                                            │
│  • OpenAI-Compatible Spec (/v1/chat/completions)                                                            │
│  • Server-Sent Events (SSE) Real-Time Token Streaming                                                       │
│  • Prometheus Observability & Metrics Endpoint (/metrics)                                                   │
│  • Client Disconnect Detection & Automatic Physical Block Reclamation                                       │
└──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Core Serving Infrastructure (NanoInference Engine)                                                         │
│  • Page Table Manager (Logical Tokens -> Physical VRAM Block IDs)                                            │
│  • Continuous Batcher (Prefill/Decode Phase Split & Queue Management)                                       │
│  • VRAM Pool Allocator (Ref Counting & Block Recycling)                                                    │
└──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Compute Execution & Infrastructure Layer                                                                    │
│  • Model Runner Engine (Qwen2.5-0.5B-Instruct Transformer Execution)                                        │
│  • CUDA-Enabled Multi-Stage Docker Container & Kubernetes Deployment Specs                                  │
└──────────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---
### Triton PagedAttention vs. PyTorch Eager Baseline

* **Environment**: NVIDIA GeForce RTX 5070 Laptop GPU (1024 Context Length, `head_dim=64`)
* **Key Finding**: Up to **28.8x speedup** during decode attention by eliminating memory-gather latency via vectorized Triton block memory loads and fused online Softmax.

| Batch Size | PyTorch Eager Latency | Custom Triton Kernel Latency | Speedup |
| :--- | :--- | :--- | :--- |
| **1** | 0.470 ms | **0.067 ms** | **7.05x** |
| **8** | 2.415 ms | **0.095 ms** | **25.45x** |
| **16** | 4.897 ms | **0.188 ms** | **26.02x** |
| **32** | 9.127 ms | **0.440 ms** | **20.72x** |
| **64** | 20.917 ms | **0.726 ms** | **28.82x** |
---

## 📂 Project Structure

- **`nano_inference/`**
  - **`block_manager.py`**: OS-style virtual page table allocator for dynamic VRAM block tracking.
  - **`paged_attention.py`**: Triton GPU kernel-accelerated decode attention layer with PyTorch reference fallback.
  - **`rmsnorm.py`**: Custom fused Triton RMSNorm GPU kernel with PyTorch fallback.
  - **`scheduler.py`**: Iteration-level continuous batcher and Sarathi-style chunked prefill scheduler.
  - **`model_runner.py`**: Prefill/decode execution, dynamic PEFT LoRA adapter routing, and quantization managers.
  - **`guided_decoding.py`**: Logits processor enforcing JSON schema structures.
  - **`speculative.py`**: Draft-target token verification engine for speculative sampling.
  - **`metrics.py`**: Custom Prometheus instrumentation (TTFT, throughput, cache utilization).
  - **`server.py`**: FastAPI server hosting SSE streaming and orphan-request VRAM reclamation.
- **`benchmarks/`**
  - **`benchmark_serving.py`**: Asynchronous concurrent client-serving load testing.
  - **`benchmark_throughput.py`**: Offline execution throughput testing.
  - **`benchmark_triton.py`**: Triton PagedAttention microbenchmark.
  - **`roofline_profile.py`**: Roofline model peak compute & memory bandwidth analyzer.
  - **`simulate_load_balancer.py`**: Queue-aware multi-instance request routing and tail-latency simulator.
- **`tests/`**
  - **`test_correctness_gate.py`**: Precision regression suite evaluating L1/L2/L-infinity difference metrics.
  - **`profile_nsys.py`**: PyTorch NVTX instrumentation script for Nsight Systems profiling.
- **`deploy/`**: Multi-stage CUDA Dockerfile & Kubernetes GPU deployment manifests.
- **`ui.py`**: Gradio-powered developer playground.

---

## ⚡ Quickstart

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Serving Gateway
```bash
# Start the FastAPI Server
uvicorn nano_inference.server:app --host 0.0.0.0 --port 8000
```

### 3. Launch the Playground UI
```bash
python ui.py
```

### 4. Run Telemetry & Performance Analysis

```bash
# 1. Run the Numerical Correctness Regression Gate
python -m tests.test_correctness_gate

# 2. Analyze Arithmetic Intensity against GPU Roofline model
python -m benchmarks.roofline_profile

# 3. Simulate Load Balancing algorithms and Tail Latencies under load
python -m benchmarks.simulate_load_balancer

# 4. Run concurrency load test against FastAPI endpoint
python -m benchmarks.benchmark_serving --concurrency 5 --requests 10 --max-tokens 64
```

---

## 🐳 Containerized Deployment

### Build & Run Docker Container
```bash
docker build -f deploy/Dockerfile -t nanoinference:latest .
docker run --gpus all -p 8000:8000 nanoinference:latest
```tion.py`**: Triton GPU kernel-accelerated decode attention layer with PyTorch reference fallback.
  - **`scheduler.py`**: Iteration-level continuous batcher and Sarathi-style chunked prefill scheduler.
  - **`model_runner.py`**: Prefill/decode execution, dynamic PEFT LoRA adapter routing, and quantization managers.
  - **`guided_decoding.py`**: Logits processor enforcing JSON schema structures.
  - **`speculative.py`**: Draft-target token verification engine for speculative sampling.
  - **`metrics.py`**: Custom Prometheus instrumentation (TTFT, throughput, cache utilization).
  - **`server.py`**: FastAPI server hosting SSE streaming and orphan-request VRAM reclamation.
- **`benchmarks/`**: Offline execution throughput testing.
- **`platform/benchmarks/`**: Asynchronous concurrent client-serving load testing.
- **`deploy/`**: Multi-stage CUDA Dockerfile & Kubernetes GPU deployment manifests.
- **`ui.py`**: Gradio-powered developer playground.

---

## ⚡ Quickstart

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Server
```bash
uvicorn nano_inference.server:app --host 0.0.0.0 --port 8000
```

### 3. Launch the Playground UI
```bash
python ui.py
```

### 4. Run Load Benchmarks
```bash
python platform/benchmarks/benchmark_serving.py --concurrency 5 --requests 10 --max-tokens 64
```

---

## 🐳 Containerized Deployment

### Build & Run Docker Container
```bash
docker build -f deploy/Dockerfile -t nanoinference:latest .
docker run --gpus all -p 8000:8000 nanoinference:latest
```
