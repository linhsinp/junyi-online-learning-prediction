# ---- Base Python image ----
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies (gcc for numpy/sklearn, curl, unzip)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create directory for credentials
RUN mkdir -p /opt/keys

# Copy your GCS service account key
COPY gcs-service-account.json /opt/keys/gcs-service-account.json

# Set the environment variable for GCS auth
ENV GOOGLE_APPLICATION_CREDENTIALS="/opt/keys/gcs-service-account.json"

# --- Install uv (prebuilt binary) ---
RUN curl -sSfL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz \
    | tar xz && mv uv-x86_64-unknown-linux-musl/uv /usr/local/bin/ && chmod +x /usr/local/bin/uv

# Copy the entire project
COPY . /app

# Install Python dependencies from requirements.txt via uv
RUN uv pip install -r requirements.txt --system --no-cache-dir

# Default entrypoint
ENTRYPOINT ["pyflyte"]
