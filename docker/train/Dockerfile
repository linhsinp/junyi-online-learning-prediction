# ---- Base Python image ----
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies (gcc for some packages like numpy or sklearn)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv and project dependencies ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create directory for credentials
RUN mkdir -p /opt/keys

# Copy your GCS service account key (must be in same folder as Dockerfile)
COPY gcs-service-account.json /opt/keys/gcs-service-account.json

# Set the environment variable so google-cloud-sdk uses the service account
ENV GOOGLE_APPLICATION_CREDENTIALS="/opt/keys/gcs-service-account.json"

# Copy your project into the container
COPY . /app

# Install dependencies from uv.loc (lock file)
RUN uv pip install -r requirements.txt --system --no-cache-dir

# ---- Default command ----
ENTRYPOINT ["pyflyte"]
