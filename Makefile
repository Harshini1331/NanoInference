.PHONY: help install test dev bench docker clean

help:
	@echo "NanoInference - Available Commands:"
	@echo "  make install    Install clean pinned dependencies"
	@echo "  make test       Run unit test suite"
	@echo "  make dev        Start uvicorn server locally on port 8000"
	@echo "  make bench      Run multi-user throughput benchmark suite"
	@echo "  make docker     Boot Prometheus and Grafana telemetry stack"
	@echo "  make clean      Remove build artifacts and Python cache"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/

dev:
	python -m uvicorn nano_inference.server:app --host 0.0.0.0 --port 8000 --reload

bench:
	python benchmarks/benchmark_throughput.py

docker:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache