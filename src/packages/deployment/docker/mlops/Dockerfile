# MLOps Service Dockerfile
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Development stage
FROM base AS development

# Install development dependencies
RUN pip install poetry

# Copy package files
COPY ai/mlops/pyproject.toml ai/mlops/README.md ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --with dev

# Copy source code
COPY ai/mlops/src ./src

# Production stage
FROM base AS production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Copy requirements and install dependencies
COPY --chown=app:app ai/mlops/requirements.txt ./
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=app:app ai/mlops/src ./src

# Create directories for MLOps services
RUN mkdir -p /app/data/{experiments,models,configurations,service_discovery} && \
    mkdir -p /app/logs

# Expose ports
EXPOSE 8001 8002 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Command to run the application
CMD ["python", "-m", "uvicorn", "src.mlops.api.main:app", "--host", "0.0.0.0", "--port", "8001"]