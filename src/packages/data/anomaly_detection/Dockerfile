# Multi-stage Docker build for Pynomaly Detection
# Optimized for production deployment with minimal image size

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG PYNOMALY_VERSION=0.2.0

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY setup.py README.md LICENSE ./

# Build and install the package
RUN pip install --no-cache-dir .

# Test the installation
RUN python -c "from pynomaly_detection import check_phase2_availability; print('✅ Build successful:', all(check_phase2_availability().values()))"

# Production stage
FROM python:3.11-slim as production

# Set production environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYNOMALY_ENV=production \
    PYNOMALY_LOG_LEVEL=INFO

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r pynomaly && useradd -r -g pynomaly -d /app -s /bin/bash pynomaly

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder /build/src/ ./src/
COPY docker/entrypoint.sh ./
COPY docker/healthcheck.py ./

# Copy configuration files
COPY docker/config/ ./config/

# Set proper permissions
RUN chown -R pynomaly:pynomaly /app && \
    chmod +x entrypoint.sh && \
    chmod +x healthcheck.py

# Switch to non-root user
USER pynomaly

# Expose ports
EXPOSE 8080 8081 9090

# Add labels for better container management
LABEL org.opencontainers.image.title="Pynomaly Detection" \
      org.opencontainers.image.description="Production-ready anomaly detection service" \
      org.opencontainers.image.version="${PYNOMALY_VERSION}" \
      org.opencontainers.image.vendor="Pynomaly Team" \
      org.opencontainers.image.source="https://github.com/pynomaly/pynomaly-detection" \
      org.opencontainers.image.documentation="https://docs.pynomaly-detection.io" \
      maintainer="team@pynomaly-detection.io"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["server"]