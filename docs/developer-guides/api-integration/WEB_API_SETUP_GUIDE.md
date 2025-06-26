# Pynomaly Web API Setup Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 📄 Web_Api_Setup_Guide

---


This guide provides comprehensive instructions for setting up and running the Pynomaly web API across different environments and shells.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Multi-Shell Support](#multi-shell-support)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

## Quick Start

### Option 1: Automated Setup Script
```bash
# Clone repository and navigate to project
cd /path/to/Pynomaly

# Run automated setup (recommended for fresh environments)
./scripts/setup_fresh_environment.sh

# Start the API server
./scripts/start_api_bash.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install --break-system-packages fastapi uvicorn pydantic structlog dependency-injector \
    numpy pandas scikit-learn pyod rich typer httpx aiofiles \
    pydantic-settings redis prometheus-client \
    opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi \
    jinja2 python-multipart passlib bcrypt prometheus-fastapi-instrumentator

# Set Python path
export PYTHONPATH=/path/to/Pynomaly/src

# Start server
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Setup

### Current Environment (Development)
If you have dependencies already installed:

```bash
# Set environment variable
export PYTHONPATH=/path/to/Pynomaly/src

# Start server with auto-reload
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload

# Access API at http://localhost:8000
```

### Fresh Environment (New Installation)
For completely new environments without existing dependencies:

```bash
# Use the automated setup script
./scripts/setup_fresh_environment.sh

# Or follow manual steps:
# 1. Install Python 3.11+
# 2. Install dependencies (see Quick Start Option 2)
# 3. Set PYTHONPATH
# 4. Start server
```

### Virtual Environment (Recommended for Isolation)
```bash
# Create virtual environment
python3 -m venv .fresh_venv
source .fresh_venv/bin/activate  # Linux/Mac
# or
.fresh_venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set Python path and start
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000
```

## Multi-Shell Support

### Bash (Linux/Mac/WSL)
```bash
# Using startup script
./scripts/start_api_bash.sh [PORT] [HOST]

# Manual
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

### PowerShell (Windows)
```powershell
# Using PowerShell script
pwsh -File scripts/test_api_powershell.ps1

# Manual
$env:PYTHONPATH = "C:\Users\your-user\Pynomaly\src"
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

### Command Prompt (Windows)
```cmd
REM Set environment variable
set PYTHONPATH=C:\Users\your-user\Pynomaly\src

REM Start server
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

### Fish Shell
```fish
# Set environment variable
set -x PYTHONPATH /path/to/Pynomaly/src

# Start server
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

### Zsh
```zsh
# Set environment variable
export PYTHONPATH=/path/to/Pynomaly/src

# Start server
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

Once the server is running, you can access these endpoints:

### Core Endpoints
- **Root API**: `http://localhost:8000/`
  - Basic API information and version
- **Health Check**: `http://localhost:8000/api/health/`
  - System health status and metrics
- **Interactive Documentation**: `http://localhost:8000/api/docs`
  - Swagger UI for API exploration
- **OpenAPI Schema**: `http://localhost:8000/api/openapi.json`
  - Machine-readable API specification

### Feature Endpoints
- **Authentication**: `http://localhost:8000/api/auth/`
- **Detectors**: `http://localhost:8000/api/detectors/`
- **Datasets**: `http://localhost:8000/api/datasets/`
- **Detection**: `http://localhost:8000/api/detection/`
- **Experiments**: `http://localhost:8000/api/experiments/`
- **Performance**: `http://localhost:8000/api/performance/`
- **Export**: `http://localhost:8000/api/export/`
- **Autonomous Mode**: `http://localhost:8000/api/autonomous/`

### Testing Endpoints
```bash
# Test root endpoint
curl http://localhost:8000/

# Test health endpoint
curl http://localhost:8000/api/health/

# Test with JSON formatting (if jq is available)
curl -s http://localhost:8000/ | jq '.'
```

## Testing

### Automated Testing Suite
```bash
# Run comprehensive multi-environment tests
./scripts/test_all_environments.sh
```

### Manual Testing
```bash
# Test current environment
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8001 &
curl http://127.0.0.1:8001/
pkill -f "uvicorn.*8001"

# Test different ports
uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8002 &
uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8003 &
curl http://127.0.0.1:8002/
curl http://127.0.0.1:8003/
pkill -f uvicorn
```

### Health Check Validation
```bash
# Check API health and status
curl -s http://localhost:8000/api/health/ | grep '"overall_status"'

# Expected statuses:
# - "healthy": All systems operational
# - "degraded": Some non-critical issues
# - "unhealthy": Critical issues present
```

## Troubleshooting

### Common Issues

#### 1. Module Not Found Error
```bash
# Error: ModuleNotFoundError: No module named 'pynomaly'
# Solution: Set PYTHONPATH correctly
export PYTHONPATH=/absolute/path/to/Pynomaly/src
```

#### 2. Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'fastapi'
# Solution: Install dependencies
pip install --break-system-packages fastapi uvicorn pydantic structlog dependency-injector
```

#### 3. Port Already in Use
```bash
# Error: [Errno 98] error while attempting to bind on address
# Solution: Use different port or kill existing process
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8001
# Or kill existing processes
pkill -f uvicorn
```

#### 4. Permission Denied
```bash
# Error: externally-managed-environment
# Solution: Use --break-system-packages flag
pip install --break-system-packages <package>
```

#### 5. Import Errors in Fresh Environment
```bash
# Error: Clean environment cannot import modules
# Solution: Run fresh environment setup
./scripts/setup_fresh_environment.sh
```

### Debug Mode
```bash
# Start server with debug logging
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --log-level debug

# Check logs for detailed error information
tail -f /tmp/api.log  # If logging to file
```

### Verification Steps
```bash
# 1. Check Python version
python3 --version  # Should be 3.11+

# 2. Check PYTHONPATH
echo $PYTHONPATH  # Should include /path/to/Pynomaly/src

# 3. Test import
python3 -c "from pynomaly.presentation.api import app; print('Import successful')"

# 4. Check dependencies
python3 -c "import fastapi, uvicorn, pydantic; print('Dependencies OK')"

# 5. Test server startup
timeout 10 uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8080
```

## Production Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -f deploy/docker/Dockerfile -t pynomaly:latest .

# Run container
docker run -p 8000:8000 -e PYTHONPATH=/app/src pynomaly:latest
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes/

# Check deployment status
kubectl get pods -l app=pynomaly
```

### Systemd Service (Linux)
```bash
# Create service file
sudo tee /etc/systemd/system/pynomaly-api.service << EOF
[Unit]
Description=Pynomaly API Server
After=network.target

[Service]
Type=simple
User=pynomaly
WorkingDirectory=/opt/pynomaly
Environment=PYTHONPATH=/opt/pynomaly/src
ExecStart=/usr/local/bin/uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable pynomaly-api
sudo systemctl start pynomaly-api
sudo systemctl status pynomaly-api
```

### Environment Variables
```bash
# Production environment variables
export PYTHONPATH=/opt/pynomaly/src
export PYNOMALY_ENV=production
export PYNOMALY_LOG_LEVEL=info
export PYNOMALY_HOST=0.0.0.0
export PYNOMALY_PORT=8000
export PYNOMALY_WORKERS=4
```

### Performance Tuning
```bash
# Production server with multiple workers
uvicorn pynomaly.presentation.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-log \
    --log-level info
```

## Scripts Reference

The following scripts are available in the `scripts/` directory:

- **`setup_fresh_environment.sh`**: Automated setup for new environments
- **`start_api_bash.sh`**: Start API server in Bash/Linux
- **`test_api_powershell.ps1`**: PowerShell testing and startup script
- **`test_all_environments.sh`**: Comprehensive multi-environment testing

### Script Usage
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run with parameters
./scripts/start_api_bash.sh 8080 127.0.0.1  # port and host
./scripts/setup_fresh_environment.sh --use-venv  # with virtual environment
```

## Support

For additional help:
1. Check the [main README.md](../README.md) for general setup
2. Review the [API documentation](http://localhost:8000/api/docs) when server is running
3. Examine the health endpoint for system status
4. Check server logs for detailed error information
5. Use the testing scripts to validate your environment

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
