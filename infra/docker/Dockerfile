FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --frozen --no-dev

COPY . /app
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "-m", "orchestration.flyte_app"]
