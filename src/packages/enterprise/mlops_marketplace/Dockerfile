FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r marketplace && useradd -r -g marketplace marketplace

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install -e .

# Copy application code
COPY . .

# Change ownership to marketplace user
RUN chown -R marketplace:marketplace /app

# Switch to marketplace user
USER marketplace

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["mlops-marketplace", "server", "start"]

# Development stage
FROM base AS development

USER root

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER marketplace

# Override command for development
CMD ["mlops-marketplace", "server", "start", "--development", "--reload"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY --from=base /app /app

# Run as non-root user
USER marketplace

# Production command
CMD ["gunicorn", "mlops_marketplace.server:create_app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]