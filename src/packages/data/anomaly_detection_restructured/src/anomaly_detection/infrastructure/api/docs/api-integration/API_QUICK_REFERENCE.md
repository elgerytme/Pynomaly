# Pynomaly Web API Quick Reference

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 📄 Api_Quick_Reference

---


## 🚀 Quick Start Commands

### Bash/Linux/Mac/WSL
```bash
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

### PowerShell (Windows)
```powershell
$env:PYTHONPATH = "C:\Users\your-user\Pynomaly\src"
uvicorn pynomaly.presentation.api:app --host 0.0.0.0 --port 8000 --reload
```

### Automated Scripts
```bash
./scripts/setup_fresh_environment.sh    # Fresh setup
./scripts/start_api_bash.sh             # Start in Bash
pwsh -File scripts/test_api_powershell.ps1  # PowerShell test
./scripts/test_all_environments.sh      # Multi-environment test
```

## 📡 API Endpoints

| Endpoint | URL | Description |
|----------|-----|-------------|
| **Root** | `http://localhost:8000/api` | API info and version |
| **Health** | `http://localhost:8000/api/health/` | System health status |
| **Docs** | `http://localhost:8000/api/docs` | Interactive API documentation |
| **OpenAPI** | `http://localhost:8000/api/openapi.json` | API schema |
| **Auth** | `http://localhost:8000/api/auth/` | Authentication endpoints |
| **Detectors** | `http://localhost:8000/api/detectors/` | Anomaly detector management |
| **Datasets** | `http://localhost:8000/api/datasets/` | Dataset operations |
| **Detection** | `http://localhost:8000/api/detection/` | Run anomaly detection |
| **Experiments** | `http://localhost:8000/api/experiments/` | Experiment tracking |
| **Performance** | `http://localhost:8000/api/performance/` | Performance metrics |
| **Export** | `http://localhost:8000/api/export/` | Data export functionality |
| **Autonomous** | `http://localhost:8000/api/autonomous/` | Autonomous mode operations |

## 🧪 Testing Commands

### Basic API Test
```bash
# Test API root endpoint
curl http://localhost:8000/api

# Test health endpoint
curl http://localhost:8000/api/health/

# Test with JSON formatting
curl -s http://localhost:8000/api | jq '.'
```

### Health Status Check
```bash
# Get overall health status
curl -s http://localhost:8000/api/health/ | grep '"overall_status"'

# Get detailed health info
curl -s http://localhost:8000/api/health/ | jq '.summary'
```

### API Documentation Access
```bash
# Open API docs in browser
xdg-open http://localhost:8000/api/docs  # Linux
open http://localhost:8000/api/docs      # Mac
start http://localhost:8000/api/docs     # Windows
```

## 🔧 Dependency Installation

### Core Dependencies
```bash
pip install --break-system-packages \
    fastapi uvicorn pydantic structlog dependency-injector \
    numpy pandas scikit-learn pyod rich typer httpx aiofiles
```

### Additional Dependencies
```bash
pip install --break-system-packages \
    pydantic-settings redis prometheus-client \
    opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi \
    jinja2 python-multipart passlib bcrypt prometheus-fastapi-instrumentator
```

### From Requirements File
```bash
pip install --break-system-packages -r requirements.txt
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'pynomaly'` | `export PYTHONPATH=/path/to/Pynomaly/src` |
| `ModuleNotFoundError: No module named 'fastapi'` | Install dependencies with pip |
| `Address already in use` | Use different port: `--port 8001` |
| `externally-managed-environment` | Use `--break-system-packages` flag |
| Server won't start | Check Python path and dependencies |

## 🌍 Environment Variables

### Required
```bash
export PYTHONPATH=/path/to/Pynomaly/src
```

### Optional
```bash
export PYNOMALY_ENV=development
export PYNOMALY_LOG_LEVEL=info
export PYNOMALY_HOST=0.0.0.0
export PYNOMALY_PORT=8000
```

## 📊 Health Status Meanings

| Status | Description |
|--------|-------------|
| `healthy` | All systems operational |
| `degraded` | Some non-critical issues (e.g., optional adapters unavailable) |
| `unhealthy` | Critical issues present |

## 🔄 Multiple Environments

### Current Environment
```bash
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --port 8000
```

### Fresh Environment
```bash
./scripts/setup_fresh_environment.sh
```

### Virtual Environment
```bash
python3 -m venv .fresh_venv
source .fresh_venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --port 8000
```

## 🐳 Docker Quick Start

```bash
# Build image
docker build -f deploy/docker/Dockerfile -t pynomaly:latest .

# Run container
docker run -p 8000:8000 -e PYTHONPATH=/app/src pynomaly:latest
```

## 🎯 Common Use Cases

### Development
```bash
export PYTHONPATH=/path/to/Pynomaly/src
uvicorn pynomaly.presentation.api:app --reload --port 8000
```

### Testing
```bash
uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8001 &
curl http://127.0.0.1:8001/api/health/
pkill -f uvicorn
```

### Production
```bash
uvicorn pynomaly.presentation.api:app \
    --host 0.0.0.0 --port 8000 \
    --workers 4 --access-log
```

---

For complete setup instructions, see [WEB_API_SETUP_GUIDE.md](WEB_API_SETUP_GUIDE.md)

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
