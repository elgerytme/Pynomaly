# Installation

This guide will help you install Pynomaly and its dependencies.

## Requirements

- Python 3.11 or higher
- Poetry (recommended) or pip
- Optional: Docker for containerized deployment

## Install with Poetry (Recommended)

Poetry provides better dependency management and virtual environment handling.

### Install Poetry

```bash
# Using the official installer
curl -sSL https://install.python-poetry.org | python3 -

# Or with pip
pip install poetry
```

### Install Pynomaly

```bash
# Clone the repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Install dependencies
poetry install

# Install with optional ML backends
poetry install -E torch      # PyTorch support
poetry install -E tensorflow # TensorFlow support
poetry install -E jax        # JAX support
poetry install -E all        # All optional dependencies
```

### Activate Virtual Environment

```bash
# Activate the poetry shell
poetry shell

# Or run commands with poetry run
poetry run pynomaly --help
```

## Install with pip

If you prefer using pip:

```bash
# Clone the repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Install with extras
pip install -e ".[torch]"      # PyTorch support
pip install -e ".[tensorflow]" # TensorFlow support
pip install -e ".[all]"        # All optional dependencies
```

## Install from PyPI

Once published to PyPI:

```bash
# Basic installation
pip install pynomaly

# With extras
pip install pynomaly[torch]
pip install pynomaly[tensorflow]
pip install pynomaly[all]
```

## Docker Installation

For containerized deployment:

```bash
# Clone the repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Build Docker image
docker-compose build

# Start services
docker-compose up -d
```

## Development Installation

For contributing to Pynomaly:

```bash
# Clone your fork
git clone https://github.com/your-username/pynomaly.git
cd pynomaly

# Install with dev dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Install Node.js dependencies for web UI
npm install
```

## Verify Installation

Check that Pynomaly is installed correctly:

```bash
# Check CLI
pynomaly --version
pynomaly --help

# Check Python import
python -c "import pynomaly; print(pynomaly.__version__)"

# Run tests
pytest

# Start API server
pynomaly server start
```

## Platform-Specific Notes

### macOS

Install system dependencies:

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install with Homebrew
brew install python@3.11
```

### Ubuntu/Debian

Install system dependencies:

```bash
# Update package list
sudo apt-get update

# Install Python and dependencies
sudo apt-get install python3.11 python3.11-dev python3-pip
```

### Windows

1. Install Python 3.11 from [python.org](https://python.org)
2. Install Visual Studio Build Tools for C++ extensions
3. Use PowerShell or WSL2 for better compatibility

## GPU Support

### NVIDIA GPU (CUDA)

For PyTorch with CUDA:

```bash
poetry install -E torch
# Or
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For TensorFlow with CUDA:

```bash
poetry install -E tensorflow
# Ensure CUDA and cuDNN are installed
```

### Apple Silicon (M1/M2)

PyTorch and TensorFlow have native support:

```bash
poetry install -E torch
poetry install -E tensorflow
```

## Environment Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- API settings
- Storage paths
- Database connections
- Security keys

## Troubleshooting

### Poetry Issues

```bash
# Clear cache
poetry cache clear pypi --all

# Update dependencies
poetry update

# Reinstall
poetry install --sync
```

### Import Errors

```bash
# Ensure you're in the correct environment
which python
which pynomaly

# Reinstall in editable mode
pip install -e .
```

### Permission Errors

```bash
# Use user installation
pip install --user pynomaly

# Or fix permissions
sudo chown -R $(whoami) ~/.cache/pypoetry
```

## Next Steps

- Follow the [Quick Start Guide](quickstart.md)
- Explore the [CLI Usage](../guide/cli.md)
- Check the [API Documentation](../api/rest.md)