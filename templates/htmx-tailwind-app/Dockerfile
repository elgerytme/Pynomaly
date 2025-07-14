# HTMX + Tailwind CSS App Dockerfile
FROM node:18-slim AS css-builder

# Install CSS build dependencies
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Copy source files and build CSS
COPY tailwind.config.js ./
COPY src/htmx_app/static/css/input.css ./src/htmx_app/static/css/
COPY src/htmx_app/templates/ ./src/htmx_app/templates/

# Build production CSS
RUN npm run build-css:prod

# Python runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies
COPY pyproject.toml ./
RUN pip install -e .[production]

# Copy application code
COPY src/ ./src/
COPY --from=css-builder /app/src/htmx_app/static/css/output.css ./src/htmx_app/static/css/

# Copy static assets
COPY src/htmx_app/static/ ./src/htmx_app/static/

# Create required directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["gunicorn", "htmx_app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]