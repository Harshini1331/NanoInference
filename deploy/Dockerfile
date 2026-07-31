FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies & curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install requirements directly into Python 3.11
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy engine codebase
COPY nano_inference/ ./nano_inference/
COPY platform/ ./platform/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "nano_inference.server:app", "--host", "0.0.0.0", "--port", "8000"]