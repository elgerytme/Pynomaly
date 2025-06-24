# Production-ready Dockerfile for Pynomaly Model Server
FROM python:3.11-slim-bullseye as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R pynomaly:pynomaly /app

# Set security context
USER pynomaly

# Environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_SERVER_HOST=0.0.0.0
ENV MODEL_SERVER_PORT=8080
ENV LOG_LEVEL=INFO
ENV WORKERS=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start model server
CMD ["python", "-m", "pynomaly.infrastructure.serving.model_server_main"]