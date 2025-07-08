# Development Setup Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md)  👨‍💻 [Developer Guides](./README.md)  🛠️ Development Setup Guide

---

## Container-First Development (Recommended)

This project prioritizes **containerized development** for consistency, reproducibility, and simplified dependency management. However, we provide comprehensive alternatives for environments where Docker isn't available.

## Prerequisites

### System Requirements
- **Docker & Docker Compose** (recommended) OR **Python 3.11+** for manual setup
- **Git** for version control
- **Node.js 16+** for web UI dependencies (if developing web components)

### Environment Support
- **Primary**: Docker-based development (Linux, macOS, Windows with WSL2)
- **Enterprise**: Virtual environments (Windows, restricted Linux, air-gapped systems)
- **CI/CD**: GitHub Actions with containerized testing

## Quick Start (Docker)

### 1. Start Development Environment

```bash
# Start full development stack with hot-reload
make dev

# Alternative: Start with storage services (PostgreSQL, Redis, MinIO)
make dev-storage
```

**What this does:**
- Builds and starts the development container
- Mounts your source code for live editing
- Starts PostgreSQL and Redis services
- Exposes API on `http://localhost:8000`
- Provides API documentation at `http://localhost:8000/docs`

### 2. Run Tests

```bash
# Run all tests in Docker environment
make dev-test

# Run specific test types
make test-unit
make test-integration
```

### 3. Development Commands

```bash
# Format code
make format

# Run linting
make lint

# Build the application
make build

# Clean up containers
make dev-clean
```

## Docker-Compose Stack

### Service Architecture
```yaml
services:
  pynomaly-dev:     # Main development container
  postgres:         # Database service
  redis:           # Caching service
  minio:           # Object storage (optional)
```

### Container Profiles

#### 1. Development Profile (default)
- **Purpose**: Main development work with hot-reload
- **Services**: pynomaly-dev, postgres, redis
- **Usage**: `make dev`
- **Ports**: 8000 (API), 5432 (postgres), 6379 (redis)

#### 2. Testing Profile
- **Purpose**: Isolated testing environment
- **Services**: test-minimal, test-server, test-deep
- **Usage**: `make dev-test`
- **Features**: Deterministic settings, coverage reporting

#### 3. Production Profile
- **Purpose**: Production-like environment
- **Services**: pynomaly-prod, postgres, redis, nginx
- **Usage**: `make prod`
- **Features**: Multi-stage builds, security hardening

## VS Code Dev Containers

For users who prefer working with Visual Studio Code in a containerized environment, a `.devcontainer` configuration is provided. This allows you to open the project within a container directly from VS Code, ensuring all the necessary dependencies and configurations are automatically applied.

### Quick Start

1. **Install the Remote - Containers extension in VS Code**
2. **Clone the repository and open it in VS Code**
3. **When prompted, click "Reopen in Container"** or press `F1` and select `Remote-Containers: Reopen in Container`

### Features

The devcontainer setup provides:

- **Identical Environment**: Same Python version, dependencies, and tools across all developers
- **Pre-installed Tools**: 
  - `ruff` for linting and formatting
  - `mypy` for type checking
  - `pytest` for testing
  - `black` for code formatting
  - `hatch` for project management
- **Live Reload**: Code changes are immediately reflected
- **Port Forwarding**: Automatic forwarding of development ports (8000, 8080, 5432, 6379)
- **Volume Mounts**: Persistent storage for VS Code extensions and pip cache
- **Automated Setup**: Runs `make dev-setup` automatically on container creation

### Configuration Details

#### Environment Variables
```bash
PYNOMALY_ENVIRONMENT=development
PYNOMALY_LOG_LEVEL=DEBUG
PYTHONPATH=/workspace/src
```

#### Pre-installed VS Code Extensions
- Python support with IntelliSense
- Pylint and Ruff linting
- Black code formatting
- MyPy type checking
- Pytest testing framework
- Jupyter notebook support
- Makefile tools support
- YAML and JSON language support

#### Volume Mounts
- **Source code**: `/workspace` (live sync with host)
- **VS Code settings**: Persistent VS Code configuration
- **Extensions**: Cached to avoid re-downloading
- **Pip cache**: Faster package installation

### Development Workflow

1. **Start the container** (automatic when opening in VS Code)
2. **Run setup**: `make dev-setup` (runs automatically)
3. **Start development server**: `make dev` or `python -m pynomaly.presentation.api`
4. **Run tests**: `make test` or use VS Code's testing interface
5. **Format code**: `make format` or use VS Code's format-on-save

### Troubleshooting

#### Container Build Issues
```bash
# Rebuild container
Cmd/Ctrl + Shift + P -> "Remote-Containers: Rebuild Container"

# Or from command line
docker system prune
```

#### Port Conflicts
```bash
# Check if ports are already in use
netstat -tulpn | grep :8000

# Kill conflicting processes
sudo lsof -t -i:8000 | xargs kill -9
```

#### Permission Issues
```bash
# Fix file permissions (if needed)
sudo chown -R vscode:vscode /workspace
```

### Dev Container Requirements Summary

The `.devcontainer` configuration includes:

✅ **FROM project dev image** - Based on main `Dockerfile` with VS Code enhancements  
✅ **Volume mounts** - Source code, VS Code settings, extensions, and pip cache  
✅ **Port forwarding** - API (8000), Web UI (8080), PostgreSQL (5432), Redis (6379)  
✅ **postCreateCommand** - Runs `make dev-setup` automatically  
✅ **Identical environment** - Same Python, dependencies, and tools across all users  
✅ **Live reload** - Code changes immediately reflected in container  
✅ **Pre-installed tools** - `ruff`, `mypy`, `pytest`, `black`, `hatch` ready to use  

### Service Dependencies

The devcontainer automatically starts:
- **PostgreSQL 15** - Database service on port 5432
- **Redis 7** - Cache service on port 6379  
- **MinIO** - Object storage on ports 9000/9001 (optional)

All services are configured with health checks and persistent volumes.

## Branch-Specific Container Naming

**Convention**: `pynomaly-{branch-name}-{service}`

```bash
# Feature branch containers
pynomaly-feature-anomaly-detection-dev
pynomaly-feature-anomaly-detection-postgres

# Bugfix branch containers
pynomaly-bugfix-memory-leak-dev
pynomaly-bugfix-memory-leak-redis
```

**Implementation**: Set `COMPOSE_PROJECT_NAME` environment variable:
```bash
export COMPOSE_PROJECT_NAME=pynomaly-$(git branch --show-current | sed 's/[^a-zA-Z0-9]/-/g')
make dev
```

## Container Development Workflow

### 1. Start Development Environment
```bash
# Start the full development stack
make dev

# Monitor logs
docker-compose logs -f pynomaly-dev
```

### 2. Execute Commands in Container
```bash
# Open shell in development container
docker-compose exec pynomaly-dev bash

# Run tests directly
docker-compose exec pynomaly-dev pytest tests/unit/

# Run CLI commands
docker-compose exec pynomaly-dev python -m pynomaly.presentation.cli --help
```

### 3. Development Services
```bash
# API server runs automatically on container start
# Available at: http://localhost:8000
# API docs at: http://localhost:8000/docs

# Database access
docker-compose exec postgres psql -U pynomaly -d pynomaly_dev

# Redis access
docker-compose exec redis redis-cli
```

### 4. Testing Workflow
```bash
# Run all tests
make dev-test

# Run specific test suites
docker-compose run --rm test-minimal
docker-compose run --rm test-server
docker-compose run --rm test-deep

# Run with coverage
docker-compose exec pynomaly-dev pytest --cov=pynomaly --cov-report=html
```

### 5. Code Quality in Container
```bash
# All quality checks
make lint

# Individual tools
docker-compose exec pynomaly-dev mypy src/pynomaly/
docker-compose exec pynomaly-dev ruff check src/ tests/
docker-compose exec pynomaly-dev black src/ tests/
```

### 6. .devcontainer Setup (VS Code)

**Create `.devcontainer/devcontainer.json`:**
```json
{
  "name": "Pynomaly Development",
  "dockerComposeFile": [
    "../docker-compose.yml",
    "../docker-compose.override.yml"
  ],
  "service": "pynomaly-dev",
  "workspaceFolder": "/workspace",
  "shutdownAction": "stopCompose",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.pylint",
        "ms-python.black-formatter",
        "charliermarsh.ruff"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true,
        "python.formatting.provider": "black"
      }
    }
  },
  "forwardPorts": [8000, 5432, 6379],
  "postCreateCommand": "pip install -e .[dev]"
}
```

## Container Configuration

### Environment Variables (Docker)

**Development Environment:**
```bash
# Set in docker-compose.yml or .env file
PYNOMALY_ENVIRONMENT=development
PYNOMALY_LOG_LEVEL=DEBUG
PYNOMALY_DB_HOST=postgres
PYNOMALY_DB_PORT=5432
PYNOMALY_DB_NAME=pynomaly_dev
PYNOMALY_REDIS_HOST=redis
PYNOMALY_REDIS_PORT=6379
```

**Testing Environment:**
```bash
# Deterministic settings for reproducible tests
PYTHON_SEED=42
PYTHONHASHSEED=42
PYNOMALY_DETERMINISTIC=true
PYNOMALY_DEBUG=false
```

### Make Target Configuration

**Common Make Targets:**
```bash
make dev          # Start development environment
make dev-storage  # Start with storage services
make dev-test     # Run tests in containers
make dev-clean    # Clean up containers
make build        # Build production image
make prod         # Start production environment
```

**Advanced Make Targets:**
```bash
make security-scan    # Run security scans
make ci              # Run full CI pipeline
make docker          # Build Docker images
make buck-build      # Build with Buck2 (if available)
```

## Enterprise / Restricted Environment Setup

### For Systems Without Docker

Some enterprise environments restrict Docker usage. For these cases:

**See: [Alternative Setup Guide](ALTERNATIVE_SETUP.md)**

### Windows Enterprise Setup

**PowerShell Environment:**
```powershell
# Create virtual environment
python -m venv environments\.venv
environments\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set environment variables
$env:PYTHONPATH = "src"
$env:PYNOMALY_ENVIRONMENT = "development"

# Run application
python -m pynomaly.presentation.api
```

**WSL2 Alternative:**
```bash
# If WSL2 is available but Docker is restricted
wsl --install
# Then follow Linux setup instructions
```

### Air-Gapped / Offline Setup

1. **Download dependencies** on a connected machine
2. **Transfer wheel files** to air-gapped environment
3. **Install offline**:
   ```bash
   pip install --find-links ./wheels --no-index pynomaly
   ```

## Container Development Troubleshooting

### Common Docker Issues

#### 1. Container Won't Start
```bash
# Check container logs
docker-compose logs pynomaly-dev

# Check service health
docker-compose ps

# Restart services
make dev-clean && make dev
```

#### 2. Database Connection Fails
```bash
# Check PostgreSQL container
docker-compose exec postgres pg_isready -U pynomaly

# Check network connectivity
docker-compose exec pynomaly-dev ping postgres

# Reset database
docker-compose down -v
make dev
```

#### 3. Port Conflicts
```bash
# Check what's using ports
netstat -tulpn | grep :8000

# Use different ports
PYNOMALY_API_PORT=8001 make dev
```

#### 4. Volume Mount Issues
```bash
# Check volume permissions
docker-compose exec pynomaly-dev ls -la /workspace

# Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER .
```

#### 5. Out of Disk Space
```bash
# Clean up Docker resources
docker system prune -a -f

# Remove unused volumes
docker volume prune -f

# Check disk usage
docker system df
```

### Enterprise Network Issues

#### 1. Corporate Proxy
```bash
# Set proxy in Dockerfile
ENV HTTP_PROXY=http://proxy.company.com:8080
ENV HTTPS_PROXY=http://proxy.company.com:8080
```

#### 2. Registry Access
```bash
# Use internal registry
docker build -t internal-registry.company.com/pynomaly:dev .
```

### Performance Issues

#### 1. Slow Container Startup
```bash
# Use build cache
docker-compose build --parallel

# Pre-pull base images
docker pull python:3.11-slim
```

#### 2. Resource Constraints
```bash
# Limit resource usage
docker-compose up --scale pynomaly-dev=1 --memory=2g
```

## FAQ

### Q: Can I develop without Docker?
**A:** Yes! See [Alternative Setup Guide](ALTERNATIVE_SETUP.md) for virtual environment setup.

### Q: Why is Docker recommended?
**A:** Docker ensures:
- **Consistency** across development environments
- **Reproducible** builds and tests
- **Isolation** from system dependencies
- **CI/CD** pipeline compatibility

### Q: How do I debug in containers?
**A:** Use VS Code with the Dev Container extension or:
```bash
# Remote debugging
docker-compose exec pynomaly-dev python -m pdb script.py
```

### Q: Can I use different Python versions?
**A:** Modify the Dockerfile:
```dockerfile
FROM python:3.12-slim as base
```

### Q: How do I handle corporate firewalls?
**A:** 
1. Use internal package mirrors
2. Configure proxy settings
3. Use air-gapped installation
4. Contact IT for Docker Hub access

### Q: What about Windows containers?
**A:** Linux containers are recommended even on Windows. Use WSL2 or Docker Desktop.

## Validation Commands

```bash
# Container health check
make health

# Run full CI pipeline
make ci

# Validate environment setup
docker-compose exec pynomaly-dev python scripts/validation/validate_environment_organization.py

# Test all services
docker-compose exec pynomaly-dev python scripts/testing/test_health_check.py
```

## Next Steps

1. **Set up .devcontainer** for VS Code integration
2. **Configure branch-specific naming** for isolated development
3. **Implement pre-commit hooks** in containers
4. **Set up monitoring** for development services
5. **Create custom Docker images** for your organization

---

**Last Updated**: 2025-01-07  
**Environment**: Docker + Docker Compose (Primary), Virtual Environments (Alternative)  
**Architecture**: Container-First Development with Enterprise Alternatives
