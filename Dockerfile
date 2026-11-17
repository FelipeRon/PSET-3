FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
