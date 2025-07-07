# Installation

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Getting Started](README.md) > 📦 Installation

---


This guide will help you install Pynomaly and its dependencies using modern Python tooling.

## Requirements

- **Python 3.11 or higher** (3.12 recommended)
- **Hatch** (recommended) or pip for package management
- **Git** for version control
- **Optional**: Docker for containerized deployment

## Install with Hatch (Recommended)

Pynomaly uses **Hatch** for modern Python project management with automatic environment handling and PEP 621 compliance.

### Install Hatch

```bash
# Install Hatch (one-time setup)
pip install hatch

# Verify installation
hatch --version
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Automated setup (recommended)
make setup              # Install Hatch and create environments
make dev-install        # Install in development mode
make test               # Verify installation
```

### Manual Hatch Setup

```bash
# Create environments
hatch env create
hatch env show

# Install with specific features
hatch env run dev:setup          # Development environment
hatch env run prod:setup         # Production environment

# Install with ML backends
hatch env run -e ml pip install -e ".[torch]"
hatch env run -e ml pip install -e ".[tensorflow]"
hatch env run -e ml pip install -e ".[all]"
```

### Using Hatch Environments

```bash
# Run commands in environments
hatch env run test:run           # Run tests
hatch env run lint:style         # Check code style
hatch env run prod:serve-api     # Start API server
hatch env run cli:run --help     # CLI help
```

## Alternative: Traditional pip Installation

### Simple pip Setup

For those preferring traditional Python environment management:

**Environment Organization**: Pynomaly uses a centralized `environments/` directory structure with dot-prefix naming (`.venv`, `.test_env`) to keep the project root clean and organize all virtual environments systematically.

```bash
# Clone the repository
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Create virtual environment using organized structure
mkdir -p environments
python -m venv environments/.venv

# Activate environment
# Linux/macOS:
source environments/.venv/bin/activate
# Windows:
environments\.venv\Scripts\activate

# Install with desired features
pip install -e ".[server]"          # API + CLI + basic features
pip install -e ".[production]"      # Production-ready stack
pip install -e ".[all]"             # Everything
```

### Feature-Specific Installation

```bash
# Core functionality only
pip install -e .

# ML frameworks
pip install -e ".[torch]"           # PyTorch deep learning
pip install -e ".[tensorflow]"      # TensorFlow neural networks
pip install -e ".[jax]"             # JAX high-performance computing
pip install -e ".[graph]"           # PyGOD graph anomaly detection

# Application interfaces
pip install -e ".[api]"             # FastAPI web interface
pip install -e ".[cli]"             # Command-line interface
pip install -e ".[web]"             # Progressive Web App

# Advanced features
pip install -e ".[automl]"          # AutoML with auto-sklearn2
pip install -e ".[explainability]" # SHAP/LIME model explanation
pip install -e ".[monitoring]"      # Prometheus, OpenTelemetry
```

### Automated pip Setup

```bash
# Run the automated setup script (handles environment issues)
python scripts/setup_simple.py
```

This script will:
- Create a virtual environment
- Install core dependencies
- Set up Pynomaly in development mode
- Provide usage instructions

For more details, see [README_SIMPLE_SETUP.md](README_SIMPLE_SETUP.md)

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

### Hatch Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/pynomaly.git
cd pynomaly

# Complete development setup
make setup              # Install Hatch and create environments
make dev-install        # Install in development mode
make pre-commit         # Setup pre-commit hooks

# Install Node.js dependencies for web UI
npm install -D tailwindcss @tailwindcss/forms @tailwindcss/typography
npm install htmx.org d3 echarts
```

### Development Workflow

```bash
# Daily development commands
make format             # Auto-format code
make test               # Run core tests
make lint               # Check code quality
make ci                 # Full CI pipeline locally

# Test-Driven Development
pynomaly tdd status     # Check TDD compliance
pynomaly tdd validate   # Validate current state
pynomaly tdd require "src/module.py" "function" --desc "Test description"

# Build and package
make build              # Build wheel and source distribution
make version            # Show current version
```

**Note**: Pynomaly has **active TDD enforcement** with 85% coverage threshold. The system tracks test requirements and validates compliance automatically.

### Legacy Poetry Development

For those still using Poetry:

```bash
# Install with dev dependencies
poetry install --with dev,test
poetry shell

# Install pre-commit hooks
pre-commit install
```

## Verify Installation

Check that Pynomaly is installed correctly:

### Basic Verification

```bash
# Check CLI (after installation)
pynomaly --version
pynomaly --help

# Check Python import
python -c "import pynomaly; print('✅ Pynomaly installed successfully')"

# Test core functionality
python -c "from pynomaly.domain.entities import Dataset, Anomaly; print('✅ Core imports successful')"
```

### Testing

```bash
# With Hatch
make test               # Core tests
make test-all           # All tests
make ci                 # Full CI pipeline

# Direct Hatch commands
hatch env run test:run                    # Run tests
hatch env run test:run-cov               # With coverage

# Traditional pytest
pytest tests/domain/ tests/application/  # Core tests only
```

### API Server

```bash
# Start API server
# Method 1: Using Hatch
make prod-api-dev

# Method 2: Direct Hatch command
hatch env run prod:serve-api

# Method 3: Traditional uvicorn
uvicorn pynomaly.presentation.api.app:app --reload

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
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

### Hatch Issues

```bash
# Environment problems
make env-clean          # Clean and recreate environments
make setup              # Reinitialize project

# Build issues
hatch build --clean --verbose  # Verbose build output
hatch env prune                 # Remove unused environments

# Check project status
make status             # Project overview
hatch env show          # List environments
```

### Legacy Poetry Issues

```bash
# Clear cache (if still using Poetry)
poetry cache clear pypi --all
poetry update
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
- Explore the [CLI Usage](../developer-guides/api-integration/cli.md)
- Check the [API Documentation](../developer-guides/api-integration/rest-api.md)

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
