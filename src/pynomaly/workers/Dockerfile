FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . .

# Install pynomaly in development mode
RUN pip install -e .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
if [ "$WORKER_TYPE" = "streaming" ]; then\n\
    exec python -m pynomaly.workers.streaming_worker --metrics-port 9091\n\
elif [ "$WORKER_TYPE" = "training" ]; then\n\
    exec python -m pynomaly.workers.training_worker --metrics-port 9092\n\
else\n\
    echo "Unknown worker type: $WORKER_TYPE"\n\
    exit 1\n\
fi\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${METRICS_PORT:-9090}/metrics || exit 1

# Expose metrics port
EXPOSE 9090 9091 9092

# Set user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly
USER pynomaly

# Start worker
ENTRYPOINT ["/entrypoint.sh"]
