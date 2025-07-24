# Development Environment Setup

This directory contains tools and configurations for setting up a complete development environment for the anomaly detection platform.

## Quick Start

### Automated Setup (Recommended)

```bash
# Run the automated setup script
./dev-environment/setup-dev-environment.sh

# Validate your environment
python dev-environment/validate-dev-environment.py
```

### Manual Setup

If you prefer to set up the environment manually, follow the sections below.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL2 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB free space
- **Internet Connection**: Required for downloading dependencies

### Required Tools

- **Git**: Version control system
- **Python**: 3.11 or higher
- **curl/wget**: For downloading tools
- **Docker**: Container platform (optional but recommended)

## Development Tools

### Core Python Environment

#### Python Version Management
```bash
# Install pyenv (Python version manager)
curl https://pyenv.run | bash

# Install Python versions
pyenv install 3.11.7
pyenv install 3.12.1
pyenv global 3.11.7
```

#### Package Management
```bash
# Install Poetry (recommended)
curl -sSL https://install.python-poetry.org | python3 -

# Configure Poetry
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true
```

### Code Quality Tools

```bash
# Install essential development tools
pip install -r requirements-dev.txt

# Or install individually
pip install black ruff mypy pytest pre-commit bandit safety
```

### Build System

```bash
# Install Buck2 (Linux/macOS)
wget https://github.com/facebook/buck2/releases/download/latest/buck2-x86_64-unknown-linux-gnu.zst
zstd -d buck2-x86_64-unknown-linux-gnu.zst
chmod +x buck2-x86_64-unknown-linux-gnu
sudo mv buck2-x86_64-unknown-linux-gnu /usr/local/bin/buck2
```

### Container Platform

```bash
# Install Docker (Linux)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## IDE Configuration

### VS Code (Recommended)

#### Installation
- Download from [https://code.visualstudio.com/](https://code.virtualstudio.com/)

#### Automatic Extension Installation
```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.mypy-type-checker
code --install-extension charliermarsh.ruff
code --install-extension ms-azuretools.vscode-docker
code --install-extension eamodio.gitlens
```

#### Workspace Setup
1. Open the monorepo folder in VS Code
2. Install recommended extensions when prompted
3. Open `monorepo.code-workspace` for the full workspace experience

### Debug Configuration

The repository includes comprehensive debug configurations:

- **Debug Current File**: Debug any Python file
- **Debug API Server**: Debug the FastAPI application
- **Debug CLI**: Debug command-line interface
- **Debug Tests**: Debug pytest runs
- **Remote Debug**: Attach to running containers

### Other IDEs

#### PyCharm/IntelliJ
1. Open the repository root as a project
2. Configure Python interpreter to use pyenv Python
3. Install plugins: Docker, Makefile Support, .env files support

#### Vim/Neovim
1. Install Python language server: `pip install python-lsp-server`
2. Configure LSP client with project settings

## Pre-commit Hooks

### Installation
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

### Configuration

The repository includes comprehensive pre-commit hooks:

- **Code Formatting**: Black, isort
- **Linting**: Ruff, Bandit
- **Type Checking**: MyPy
- **Security**: detect-secrets, Safety
- **Architecture**: Domain boundary validation
- **Testing**: Critical test execution

### Manual Execution
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files
```

## Testing Environment

### Test Runner Configuration

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance
```

### Test Environment Variables

```bash
# Create test environment file
cp .env.template .env.test

# Set testing-specific variables
export ENVIRONMENT=testing
export LOG_LEVEL=DEBUG
export DATABASE_URL=sqlite:///test.db
```

## Docker Development Environment

### Development Stack
```bash
# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Start with monitoring
docker-compose -f docker-compose.dev.yml -f docker-compose.monitoring.yml up -d
```

### Service Access
- **API**: http://localhost:8000
- **Database**: localhost:5432
- **Redis**: localhost:6379
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## Environment Validation

### Automated Validation
```bash
# Run comprehensive environment validation
python dev-environment/validate-dev-environment.py
```

### Manual Validation
```bash
# Check Python
python --version  # Should be 3.11+

# Check essential tools
git --version
docker --version
pre-commit --version
pytest --version

# Test pre-commit
pre-commit run --all-files

# Test Docker
docker run --rm hello-world
```

## Common Development Workflows

### Starting Development
```bash
# 1. Activate Python environment
pyenv shell 3.11.7

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Start development services
docker-compose -f docker-compose.dev.yml up -d

# 4. Run tests to verify setup
pytest tests/health/

# 5. Start development server
python -m uvicorn anomaly_detection.api.main:app --reload
```

### Code Quality Checks
```bash
# Format code
black src/
isort src/

# Lint code
ruff check src/

# Type check
mypy src/

# Security scan
bandit -r src/
safety check

# Run all quality checks
pre-commit run --all-files
```

### Testing Workflow
```bash
# Run fast tests during development
pytest -x -vvs tests/unit/

# Run comprehensive test suite
pytest --cov=src --cov-report=html

# Run performance tests
pytest -m performance --benchmark-only
```

### Building and Deployment
```bash
# Build with Buck2
buck2 build //...

# Build Docker images
docker build -t anomaly-detection:dev .

# Run deployment validation
python scripts/validate_deployment.py
```

## Troubleshooting

### Common Issues

#### Python Environment Issues
```bash
# Problem: Python version not found
# Solution: Reinstall with pyenv
pyenv install 3.11.7
pyenv global 3.11.7

# Problem: Package import errors
# Solution: Check PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

#### Docker Issues
```bash
# Problem: Docker permission denied
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login again

# Problem: Docker daemon not running
# Solution: Start Docker daemon
sudo systemctl start docker
```

#### Pre-commit Issues
```bash
# Problem: Pre-commit hooks failing
# Solution: Update hooks and run manually
pre-commit autoupdate
pre-commit run --all-files

# Problem: Skip hooks temporarily
# Solution: Use --no-verify flag
git commit --no-verify -m "temporary commit"
```

#### VS Code Issues
```bash
# Problem: Python interpreter not found
# Solution: Set interpreter path
# Ctrl+Shift+P -> "Python: Select Interpreter"
# Choose the pyenv Python version

# Problem: Extensions not working
# Solution: Reload window
# Ctrl+Shift+P -> "Developer: Reload Window"
```

### Getting Help

1. **Check Documentation**: Review docs/ directory
2. **Run Validation**: `python dev-environment/validate-dev-environment.py`
3. **Check Logs**: Look at command output and error messages
4. **Update Tools**: Ensure all tools are up to date
5. **Clean Environment**: Remove and recreate virtual environments

### Performance Optimization

#### Fast Development Cycle
```bash
# Use file watching for fast feedback
pytest --looponfail tests/

# Use fast linting
ruff check src/ --watch

# Use incremental type checking
mypy --follow-imports=silent src/
```

#### Resource Usage
```bash
# Monitor resource usage
docker stats

# Clean up Docker resources
docker system prune -f

# Clean Python cache
find . -type d -name __pycache__ -delete
find . -name "*.pyc" -delete
```

## Advanced Configuration

### Custom Tool Configuration

#### Black Configuration
```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
```

#### Ruff Configuration
```toml
# pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "W", "I", "N", "B", "C4", "UP"]
```

#### MyPy Configuration
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
```

### Git Configuration

#### Recommended Git Settings
```bash
# Set up Git user
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set up Git aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status

# Set up pull strategy
git config --global pull.rebase true
```

#### Git Hooks
The repository includes custom Git hooks for:
- Pre-commit code quality checks
- Pre-push testing validation
- Commit message formatting

## Security Considerations

### Secure Development Practices

1. **Secrets Management**: Never commit secrets to the repository
2. **Dependency Scanning**: Regularly scan dependencies for vulnerabilities
3. **Code Scanning**: Use SAST tools for security analysis
4. **Container Security**: Scan container images for vulnerabilities

### Environment Security
```bash
# Secure file permissions
chmod 600 .env*
chmod 700 scripts/

# Use secure Python package sources
pip install --trusted-host pypi.org --trusted-host pypi.python.org

# Verify tool signatures when possible
```

This comprehensive development environment setup ensures a productive, secure, and consistent development experience across the team.