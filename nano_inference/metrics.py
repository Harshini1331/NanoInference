"""
nano_inference/metrics.py

Centralized Prometheus telemetry definitions for NanoInference.
Prevents circular import dependencies between API server and execution scheduler.
"""

from prometheus_client import Counter, Gauge, Histogram

# Dynamic System Gauges
KV_CACHE_USAGE = Gauge(
    "nanoinference_kv_cache_usage_percent",
    "Percentage of allocated physical KV blocks in GPU VRAM",
)
REQUESTS_RUNNING = Gauge(
    "nanoinference_requests_running",
    "Number of currently active decode streams",
)
REQUESTS_WAITING = Gauge(
    "nanoinference_requests_waiting",
    "Number of requests queued in prefill queue",
)

# Latency & Throughput Metrics
TTFT_HISTOGRAM = Histogram(
    "nanoinference_ttft_seconds",
    "Time to First Token (TTFT) in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
TOTAL_TOKENS_GENERATED = Counter(
    "nanoinference_tokens_generated_total",
    "Total output tokens generated across all streams",
)

# Automatic Prefix Caching (APC) Counters
PREFIX_CACHE_HITS = Counter(
    "nanoinference_prefix_cache_hits_total",
    "Total number of physical 16-token VRAM blocks reused via prefix caching",
)
PREFIX_CACHE_MISSES = Counter(
    "nanoinference_prefix_cache_misses_total",
    "Total number of physical VRAM blocks allocated as cache misses",
)