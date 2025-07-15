FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies (API and Database extras)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[api,database]

# Expose API port
EXPOSE 8000

# Application environment variables
ENV PYTHONPATH=/app/src

# Start the API using Uvicorn (ASGI factory pattern)
CMD ["uvicorn", "pynomaly.presentation.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]