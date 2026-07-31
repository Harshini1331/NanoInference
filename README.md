# ⚡ NanoInference

A high-throughput LLM serving engine built from scratch in PyTorch, featuring a custom **Triton-accelerated PagedAttention** kernel, **continuous batching**, **dynamic Multi-LoRA adapter switching**, and real-time **Prometheus telemetry**.

Designed to implement systems-level LLM optimizations directly, without high-level wrappers.

---

## 🚀 Key Features & Highlights

- **Custom PagedAttention (Triton / PyTorch Fallback)**: Eliminates VRAM fragmentation by mapping logical token sequences to non-contiguous physical block allocations. Implemented via high-performance Triton GPU kernels with PyTorch fallback.
- **Continuous Batching & Chunked Prefill**: Iteration-level scheduling splits prefill (Sarathi-style chunking) and decode phases to maximize compute efficiency and minimize GPU idle time.
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

## 📂 Project Structure

- **`nano_inference/`**
  - **`block_manager.py`**: OS-style virtual page table allocator for dynamic VRAM block tracking.
  - **`paged_attention.py`**: Triton GPU kernel-accelerated decode attention layer with PyTorch reference fallback.
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
