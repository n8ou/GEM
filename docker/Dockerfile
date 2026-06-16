# syntax=docker/dockerfile:1.7
# ---- Multi-stage build: slim runtime, no build toolchain in final image ----

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM python:3.12-slim AS runtime
# Runtime-only OS deps for OpenCV / ONNXRuntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src ./src
COPY scripts ./scripts

# Run as non-root for security.
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/artifacts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health'); " || exit 1

CMD ["uvicorn", "facecore.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
