# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry==1.7.1

# Export requirements
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 pynomaly

# Set working directory
WORKDIR /app

# Copy requirements
COPY --from=builder /app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=pynomaly:pynomaly src/ ./src/
COPY --chown=pynomaly:pynomaly pyproject.toml ./

# Create directories
RUN mkdir -p /app/storage /app/logs /app/temp && \
    chown -R pynomaly:pynomaly /app

# Switch to non-root user
USER pynomaly

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "pynomaly.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]