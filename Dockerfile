# Development Container Dockerfile
# This container mirrors the CI environment for reproducible development
# Based on CI workflow: .github/workflows/ci.yml

FROM python:3.11-slim as base

# Set environment variables to match CI
ENV PYTHON_VERSION=3.11
ENV HATCH_VERBOSE=1
ENV POETRY_CACHE_DIR=/home/dev/.cache/pypoetry
ENV PIP_CACHE_DIR=/home/dev/.cache/pip
ENV WHEEL_CACHE_DIR=/home/dev/.cache/wheels

# Install system dependencies matching CI environment
RUN apt-get update && apt-get install -y \
    # Build essentials
    gcc \
    g++ \
    build-essential \
    # Development tools
    git \
    curl \
    vim \
    nano \
    # Database clients for testing (matches CI services)
    postgresql-client \
    redis-tools \
    # System monitoring
    htop \
    procps \
    # Additional dev tools
    tree \
    less \
    && rm -rf /var/lib/apt/lists/*

# Create development user (non-root)
RUN useradd -m -u 1000 -s /bin/bash dev && \
    usermod -aG sudo dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Create cache directories
RUN mkdir -p /home/dev/.cache/pip \
             /home/dev/.cache/pypoetry \
             /home/dev/.cache/wheels \
             /home/dev/.cache/torch \
             /home/dev/.cache/tensorflow \
             /home/dev/.cache/jax && \
    chown -R dev:dev /home/dev/.cache

# Copy requirements files
COPY requirements-dev.txt ./
COPY requirements.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip wheel setuptools && \
    pip install hatch

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install core dependencies (pinned versions from CI)
RUN pip install --cache-dir /home/dev/.cache/pip \
    numpy==1.26.0 \
    scipy \
    scikit-learn

# Install heavy ML dependencies (matching CI caching strategy)
RUN pip install --cache-dir /home/dev/.cache/pip \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --cache-dir /home/dev/.cache/pip \
    tensorflow-cpu

# Switch to development user
USER dev

# Set up development environment
ENV PATH="/home/dev/.local/bin:${PATH}"
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

# Set up environment variables for testing (matching CI)
ENV PYNOMALY_ENVIRONMENT=development
ENV PYNOMALY_DB_HOST=postgres
ENV PYNOMALY_DB_PORT=5432
ENV PYNOMALY_DB_NAME=pynomaly_dev
ENV PYNOMALY_DB_USER=pynomaly
ENV PYNOMALY_DB_PASSWORD=pynomaly_dev_password
ENV PYNOMALY_REDIS_HOST=redis
ENV PYNOMALY_REDIS_PORT=6379
ENV PYNOMALY_REDIS_DB=0

# Create workspace directories
RUN mkdir -p /workspace/src \
             /workspace/tests \
             /workspace/artifacts \
             /workspace/reports \
             /workspace/coverage-reports

# Expose ports for development
EXPOSE 8000 8080 5678

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import pynomaly; print('Dev environment healthy')" || exit 1

# Default command opens a bash shell for development
CMD ["/bin/bash"]
