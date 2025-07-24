# Multi-stage production Docker build for Anomaly Detection Service
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# Add metadata labels
LABEL maintainer="anomaly-detection-team@company.com" \
      version="${BUILD_VERSION}" \
      description="Production-ready anomaly detection service" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=appuser:appuser /app .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANOMALY_DETECTION_ENV=production \
    ANOMALY_DETECTION_LOG_LEVEL=INFO \
    ANOMALY_DETECTION_DATA_DIR=/app/data \
    ANOMALY_DETECTION_MODEL_DIR=/app/models \
    ANOMALY_DETECTION_LOG_DIR=/app/logs

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "anomaly_detection.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]