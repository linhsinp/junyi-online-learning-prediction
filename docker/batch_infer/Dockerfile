FROM ghcr.io/flyteorg/flytekit:py3.12-1.16.1 AS base

# Switch to root to install system packages
USER root

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -sSfL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz \
    | tar xz && mv uv-x86_64-unknown-linux-musl/uv /usr/local/bin/ && chmod +x /usr/local/bin/uv

# Copy your code
COPY . /app

# Install dependencies
RUN uv pip install -r requirements.txt --system --no-cache-dir

# ✅ Do NOT override ENTRYPOINT – use the default from base image
