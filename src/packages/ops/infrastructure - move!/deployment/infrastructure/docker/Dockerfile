# Production Dockerfile for Pynomaly
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage/data \
    /app/storage/logs \
    /app/storage/uploads \
    /app/storage/cache \
    && chown -R pynomaly:pynomaly /app/storage

# Install the application
RUN pip install -e .

# Copy production configuration
COPY config/production/.env.prod /app/.env

# Set proper permissions
RUN chown -R pynomaly:pynomaly /app

# Switch to non-root user
USER pynomaly

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "pynomaly.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
