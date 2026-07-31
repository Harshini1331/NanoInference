# ⚡ NanoInference
> **A custom PyTorch LLM serving engine featuring PagedAttention KV-cache management, iteration-level continuous batching, an OpenAI-compatible SSE API gateway, and live Prometheus telemetry.**

---

## 🎯 Executive Overview

**NanoInference** is a low-level, high-throughput LLM inference server designed to implement core systems-level optimizations from scratch. 

Rather than relying on high-level wrappers around Hugging Face or vLLM, NanoInference builds a custom **PagedAttention** engine that maps logical token positions to non-contiguous physical VRAM blocks—eliminating external KV-cache fragmentation. Paired with an **iteration-level continuous scheduler**, the engine dynamically schedules batch execution across heterogeneous prompt lengths without stalling GPU compute loops.

---

## 🏛 System Architecture & Request Lifecycle

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

## 📊 System Benchmarks

Evaluated on an NVIDIA RTX 5070 GPU running concurrent requests against `Qwen/Qwen2.5-0.5B-Instruct`:

| Metric | Measured Value | Architectural Significance |
| :--- | :--- | :--- |
| **Concurrency Level** | 5 active streams | Evaluates continuous batching under multi-user contention |
| **Request Success Rate** | 100% (10/10 completed) | Zero block leaks or VRAM out-of-memory errors |
| **P50 TTFT (Time-To-First-Token)** | 1,592 ms | Latency for initial prefill pass and page table entry creation |
| **P95 TTFT** | 2,470 ms | Tail latency ceiling under parallel prefill scheduling |
| **Average ITL (Inter-Token Latency)** | 1,182 ms/token | Time per decoding iteration step across concurrent streams |
| **Engine Throughput** | 4.20 tokens/sec | Execution speed in eager-mode PyTorch continuous generation |

---

## 🛠 Core Systems Implementation

### 1. PagedAttention Memory Management (`nano_inference/block_manager.py`)
* **Virtual Page Tables:** Implements OS-style virtual memory page mapping (`BlockTable`) to decouple a sequence's logical token indices from physical non-contiguous GPU VRAM memory space.
* **Fixed Block Allocator:** Allocates memory in fixed 16-token physical blocks (`PhysicalTokenBlock`), preventing external memory fragmentation and limiting internal fragmentation to $< 16$ tokens per sequence.
* **Block Recycling:** Tracks reference counts per block to handle dynamic request creation, prompt prefix caching, and deterministic deallocation upon stream completion.

### 2. Iteration-Level Continuous Batcher (`nano_inference/scheduler.py`)
* **Prefill & Decode Decoupling:** Dynamically batches incoming sequences in prefill phase (processing prompt context) alongside active decode phases (generating next-token outputs) in a single unified execution step.
* **Dynamic Concurrency Control:** Prevents out-of-memory crashes by capping maximum batched tokens and enforcing physical block capacity limits before moving requests from `WAITING` to `RUNNING` status.

### 3. OpenAI Gateway & SSE Streaming (`nano_inference/server.py`)
* **OpenAI-Compatible Streaming:** Implements SSE protocol emitting `chat.completion.chunk` payloads compatible with native OpenAI client libraries.
* **Orphan Request Reclamation:** Listens for client HTTP dropouts (`request.is_disconnected()`) to instantly free allocated page blocks back to the memory pool, preventing memory leaks during unexpected network failures.

### 4. Telemetry & Observability (`/metrics`)
Exposes live Prometheus indicators:
* `nanoinference_kv_cache_usage_percent`: Real-time percentage of physical VRAM blocks consumed.
* `nanoinference_requests_running`: Gauge for active decode loops.
* `nanoinference_requests_waiting`: Queue depth indicator for prefill scheduling.
* `nanoinference_ttft_seconds`: Histogram tracking Time-To-First-Token latencies.
* `nanoinference_tokens_generated_total`: Total output token generation counter.

---

## 📂 Project Structure

```text
NanoInference/
├── nano_inference/
│   ├── block_manager.py     # PagedAttention page tables & physical VRAM block allocator
│   ├── scheduler.py         # Continuous batching iteration-level scheduler
│   ├── model_runner.py       # Model prefill & decode execution passes
│   └── server.py            # FastAPI gateway, SSE streaming & Prometheus metrics
├── platform/
│   └── benchmarks/
│       └── benchmark_serving.py  # Asynchronous multi-stream benchmarking suite
├── deploy/
│   ├── Dockerfile           # Multi-stage CUDA runtime container spec
│   └── k8s-deployment.yaml  # Kubernetes Deployment & Service manifests with GPU limits
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🚀 Quickstart

### Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/NanoInference.git
cd NanoInference

# Initialize environment
python -m venv venv
source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Start engine server
uvicorn nano_inference.server:app --host 0.0.0.0 --port 8000
```

### Run Benchmarks

In a second terminal, execute the async load suite:

```bash
python platform/benchmarks/benchmark_serving.py --concurrency 5 --requests 10 --max-tokens 64
```

### Inspect Prometheus Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

---

## 🐳 Containerized Deployment

### Build Container

```bash
docker build -f deploy/Dockerfile -t nanoinference:v1 .
```

### Run Container with GPU Access

```bash
docker run --gpus all -p 8000:8000 nanoinference:v1
```

### Test Inference Stream via Container

```bash
python -c "import requests; r = requests.post('http://127.0.0.1:8000/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'Explain continuous batching in one sentence.'}], 'max_tokens': 32}, stream=True); [print(line.decode('utf-8')) for line in r.iter_lines() if line]"
```
