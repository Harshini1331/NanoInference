# NanoInference ⚡

A lightweight, high-performance LLM serving engine built from scratch in PyTorch. Designed to showcase the core mechanics of modern LLM inference systems, including PagedAttention, continuous batching, and real-time Server-Sent Events (SSE) streaming.

---

## Key Features

* **PagedAttention Block Management (`block_manager.py`)**: Virtual memory allocator mapping dynamic logical sequence blocks to non-contiguous physical KV-cache VRAM blocks, eliminating external fragmentation.
* **Continuous Batching Scheduler (`scheduler.py`)**: Iteration-level prefill and decode scheduling to maintain high GPU utilization across heterogeneous sequence lengths.
* **Physical KV Cache Pool (`model_runner.py`)**: Direct PyTorch execution runner with past KV-state tracking and repetition penalty logic.
* **OpenAI-Compatible API Gateway (`server.py`)**: Async FastAPI gateway with Server-Sent Events (SSE) token-by-token streaming.
* **Interactive UI (`ui.py`)**: Real-time Streamlit chat interface.

---

## Architecture

[ Client / Streamlit UI ]
           │
           ▼ (HTTP / SSE)
┌────────────────────────────────────────────────────────┐
│ 1. FastAPI Server (server.py)                          │
│    Applies chat templates & routes SSE token streams   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ 2. Continuous Batching Scheduler (scheduler.py)       │
│    Manages iteration queues & prefill/decode batches   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ 3. PagedAttention Memory Allocator (block_manager.py)  │
│    Logical-to-physical block table mapping (16 tokens)  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ 4. Execution Model Runner (model_runner.py)           │
│    PyTorch forward pass & KV-cache state preservation  │
└────────────────────────────────────────────────────────┘

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/NanoInference.git](https://github.com/YOUR_GITHUB_USERNAME/NanoInference.git)
cd NanoInference

python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Serving Backend

Start the FastAPI inference gateway powered by Uvicorn:

```bash
uvicorn nano_inference.server:app --reload --port 8000
```

### 3. Launch the Chat Interface

In a second terminal window (with `venv` activated):

```bash
streamlit run ui.py
```

Open `http://localhost:8501` in your browser to chat!

---

## API Usage Example

Send a standard OpenAI-formatted POST request via `curl`:

```bash
curl -X POST "[http://127.0.0.1:8000/v1/chat/completions](http://127.0.0.1:8000/v1/chat/completions)" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-0.5B-Instruct",
       "messages": [{"role": "user", "content": "Why is the sky blue?"}],
       "max_tokens": 64
     }'
```

---

## Project Structure

```text
NanoInference/
├── nano_inference/
│   ├── block_manager.py  # PagedAttention VRAM page table allocator
│   ├── scheduler.py      # Iteration-level continuous batching queue
│   ├── model_runner.py   # PyTorch execution engine & KV state management
│   └── server.py         # Async FastAPI gateway + SSE streaming
├── tests/                # Unit test suite
├── ui.py                 # Streamlit chat interface
└── requirements.txt
```

---

## Tech Stack

* **Language**: Python 3.11+
* **Frameworks**: PyTorch, Hugging Face Transformers, FastAPI, Pydantic
* **Frontend**: Streamlit
* **Model Defaults**: `Qwen/Qwen2.5-0.5B-Instruct`