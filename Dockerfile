# AutoRecover — one image that runs the whole stack.
FROM python:3.12-slim

# build-essential: any dep without a prebuilt wheel. libgomp1: OpenMP runtime
# that onnxruntime / scikit-learn wheels load at import (missing on slim).
# curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer — cached unless pyproject/src change, so edits to the
# entrypoint, config or docs rebuild in seconds instead of re-installing.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Everything else (entrypoint, config, knowledge base, evals corpus).
COPY . .

# Only a service may spend the Razorpay link quota / SuperU credits (matches
# start.sh). Voice stays OFF unless enabled in the guardrail config.
ENV RAZORPAY_WRITES_OK=1 \
    SUPERU_CALLS_OK=1 \
    PHOENIX_WORKING_DIR=/app/data/phoenix \
    PYTHONUNBUFFERED=1

# webhook · dashboard · frontend(checkout+HUD) · Phoenix
EXPOSE 5000 5001 5002 6006

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
