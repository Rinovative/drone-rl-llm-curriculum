FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=0
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    ffmpeg \
    xvfb \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace/repo

RUN uv venv /opt/venv --python python3.10
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first for better Docker layer caching.
COPY pyproject.toml uv.lock README.md /workspace/repo/
RUN uv sync --frozen --extra dev --no-install-project

# Install the local project package.
COPY src /workspace/repo/src
COPY tests /workspace/repo/tests
RUN uv sync --frozen --extra dev

CMD ["/bin/bash"]