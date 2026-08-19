FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY docs/ ./docs/
COPY mkdocs.yml ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache || uv sync --no-cache

# Expose port for API or documentation server
EXPOSE 8000 8080

# Default command: launch Knowledge CLI status check
CMD ["uv", "run", "knowledge", "status"]
