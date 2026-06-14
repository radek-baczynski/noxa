# syntax=docker/dockerfile:1

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------
# Shared application source (CPU base)
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS source
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version README.md main.py ./
COPY src ./src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright

# ---------------------------------------------------------------------------
# CPU build
# ---------------------------------------------------------------------------
FROM source AS builder-cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Prefer PyPI wheels; fall back to source with portable CPU flags when needed.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"

RUN uv sync --frozen --no-dev --extra ml \
    && uv run playwright install-deps chromium \
    && uv run playwright install chromium

# ---------------------------------------------------------------------------
# CUDA build (linux/amd64)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder-cuda

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.13

COPY --from=source /app /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright \
    CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=70-real;75-real;80-real;86-real;89-real;90-real" \
    FORCE_CMAKE=1

RUN uv sync --frozen --no-dev --extra ml --no-binary-package llama-cpp-python \
    && uv run playwright install-deps chromium \
    && uv run playwright install chromium

# ---------------------------------------------------------------------------
# CPU runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS cpu

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    NOXA_DOCKER_PROFILE=cloud-cpu \
    NOXA_HOST=0.0.0.0 \
    NOXA_PORT=8000 \
    NOXA_MODEL_CACHE_DIR=/data/models \
    NOXA_SQLITE_PATH=/data/noxa.db \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cpu /app /app

RUN apt-get update \
    && cd /app && /app/.venv/bin/python -m playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["/entrypoint.sh"]

# ---------------------------------------------------------------------------
# CUDA runtime (linux/amd64)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS cuda

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    NOXA_DOCKER_PROFILE=cloud-gpu \
    NOXA_HOST=0.0.0.0 \
    NOXA_PORT=8000 \
    NOXA_MODEL_CACHE_DIR=/data/models \
    NOXA_SQLITE_PATH=/data/noxa.db \
    PLAYWRIGHT_BROWSERS_PATH=/app/.playwright \
    PATH="/app/.venv/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cuda /app /app

RUN apt-get update \
    && cd /app && /app/.venv/bin/python -m playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["/entrypoint.sh"]
