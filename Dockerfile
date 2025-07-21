# Production Dockerfile for Anomaly Detection Platform
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG GIT_COMMIT

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.11-slim as production

# Set build labels
LABEL maintainer="Anomaly Detection Team"
LABEL version="${VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL git-commit="${GIT_COMMIT}"
LABEL description="Anomaly Detection Platform - Production Image"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r anomaly && useradd -r -g anomaly -m -d /home/anomaly anomaly

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=anomaly:anomaly . .

# Set environment variables for production
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    WORKERS=4 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100 \
    TIMEOUT=120 \
    KEEPALIVE=5

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache && \
    chown -R anomaly:anomaly /app/logs /app/data /app/models /app/cache

# Install packages individually (no root pyproject.toml)
WORKDIR /app/src/packages/data/anomaly_detection_restructured
RUN pip install -e .

# Switch back to app directory and non-root user
WORKDIR /app
USER anomaly

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "${WORKERS}", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--max-requests", "${MAX_REQUESTS}", \
     "--max-requests-jitter", "${MAX_REQUESTS_JITTER}", \
     "--timeout", "${TIMEOUT}", \
     "--keepalive", "${KEEPALIVE}", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info", \
     "src.packages.data.anomaly_detection.main:app"]