#!/usr/bin/env bash
set -e

echo "====================================================="
echo "        🚀 Starting NanoInference Stack             "
echo "====================================================="

# 1. Boot Docker Telemetry (Prometheus + Grafana)
if command -v docker &> /dev/null; then
    echo "📊 Booting Prometheus & Grafana telemetry..."
    docker compose up -d
else
    echo "⚠️ Docker not found, skipping Grafana/Prometheus setup."
fi

# 2. Run Test Suite
echo "🧪 Executing engine unit tests..."
python -m tests.test_speculative
python -m tests.test_quantization
python -m tests.test_multilora

# 3. Launch Server
echo "⚡ Starting NanoInference API Gateway on http://localhost:8000..."
python -m uvicorn nano_inference.server:app --host 0.0.0.0 --port 8000