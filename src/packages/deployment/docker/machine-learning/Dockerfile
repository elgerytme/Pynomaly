# Machine Learning Service Dockerfile
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies including ML libraries dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    gfortran \
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
COPY ai/machine_learning/pyproject.toml ai/machine_learning/README.md ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --with dev

# Copy source code
COPY ai/machine_learning/src ./src

# Production stage
FROM base AS production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Copy requirements and install dependencies
COPY --chown=app:app ai/machine_learning/requirements.txt ./
RUN pip install --user -r requirements.txt

# Copy application code
COPY --chown=app:app ai/machine_learning/src ./src

# Create directories for ML services
RUN mkdir -p /app/data/{models,training_jobs,predictions} && \
    mkdir -p /app/logs

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Command to run the application
CMD ["python", "-m", "uvicorn", "src.machine_learning.api.main:app", "--host", "0.0.0.0", "--port", "8004"]