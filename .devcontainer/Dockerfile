# .devcontainer/Dockerfile
# VS Code Dev Container configuration for Pynomaly
# Based on the main project Dockerfile with VS Code-specific enhancements

FROM python:3.11-slim as base

# Set environment variables to match project development setup
ENV PYTHON_VERSION=3.11
ENV HATCH_VERBOSE=1
ENV POETRY_CACHE_DIR=/home/vscode/.cache/pypoetry
ENV PIP_CACHE_DIR=/home/vscode/.cache/pip
ENV WHEEL_CACHE_DIR=/home/vscode/.cache/wheels

# Install system dependencies matching main Dockerfile
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
    # VS Code development tools
    openssh-client \
    gnupg2 \
    iproute2 \
    procps \
    lsof \
    htop \
    net-tools \
    psmisc \
    # Database clients for testing
    postgresql-client \
    redis-tools \
    # Additional dev tools
    tree \
    less \
    wget \
    unzip \
    # Python development tools
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user for VS Code dev container compatibility
RUN groupadd -r vscode && useradd -m -r -g vscode vscode \
    && usermod -aG sudo vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Create cache directories with proper ownership
RUN mkdir -p /home/vscode/.cache/pip \
             /home/vscode/.cache/pypoetry \
             /home/vscode/.cache/wheels \
             /home/vscode/.cache/torch \
             /home/vscode/.cache/tensorflow \
             /home/vscode/.cache/jax \
             /home/vscode/.local \
             /home/vscode/.vscode-server \
    && chown -R vscode:vscode /home/vscode

# Install Python build tools
RUN pip install --upgrade pip wheel setuptools hatch

# Switch to vscode user
USER vscode

# Set up environment for VS Code development
ENV PATH="/home/vscode/.local/bin:${PATH}"
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

# Set up environment variables for development
ENV PYNOMALY_ENVIRONMENT=development
ENV PYNOMALY_LOG_LEVEL=DEBUG
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

# Pre-install common VS Code Python extensions requirements
RUN pip install --user \
    pylint \
    black \
    isort \
    mypy \
    pytest \
    pytest-cov \
    ruff

# Expose ports for development
EXPOSE 8000 8080 5678 5432 6379

# Health check for dev container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('Python:', sys.version); import pip; print('Dev container healthy')" || exit 1

# Set up shell environment for better development experience
RUN echo 'export PS1="\[\e[36m\]devcontainer:\[\e[m\]\[\e[33m\]\w\[\e[m\]\$ "' >> /home/vscode/.bashrc \
    && echo 'alias ll="ls -la"' >> /home/vscode/.bashrc \
    && echo 'alias la="ls -la"' >> /home/vscode/.bashrc \
    && echo 'alias ..="cd .."' >> /home/vscode/.bashrc \
    && echo 'alias pytest="python -m pytest"' >> /home/vscode/.bashrc \
    && echo 'alias pynomaly="python -m pynomaly.presentation.cli"' >> /home/vscode/.bashrc

# Default command starts bash for interactive development
CMD ["/bin/bash"]
