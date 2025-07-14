# Python App Template

A comprehensive Python application template with build/deploy structure and production-ready features.

## Features

- **Modern Python**: Python 3.11+ with async/await support
- **FastAPI**: High-performance web framework
- **Typer**: Rich CLI interface
- **Build System**: Hatch with modern packaging
- **Deployment**: Docker, Kubernetes, and cloud-ready
- **Monitoring**: Prometheus metrics and health checks
- **Security**: Authentication, authorization, and security hardening
- **Testing**: Comprehensive test suite with coverage
- **Documentation**: Auto-generated API docs

## Directory Structure

```
my-app/
├── build/                 # Build artifacts and configuration
│   ├── docker/           # Docker configurations
│   ├── scripts/          # Build scripts
│   └── packages/         # Package outputs
├── deploy/               # Deployment configurations
│   ├── kubernetes/       # K8s manifests
│   ├── terraform/        # Infrastructure as code
│   ├── docker-compose/   # Local deployment
│   └── cloud/            # Cloud-specific configs
├── docs/                 # Documentation
│   ├── api/              # API documentation
│   ├── deployment/       # Deployment guides
│   └── development/      # Development guides
├── env/                  # Environment configurations
│   ├── development/      # Dev environment
│   ├── staging/          # Staging environment
│   └── production/       # Production environment
├── temp/                 # Temporary files and logs
├── src/                  # Source code
│   └── my_app/
│       ├── api/          # FastAPI application
│       ├── cli/          # CLI application
│       ├── core/         # Core business logic
│       ├── infrastructure/ # Infrastructure layer
│       └── shared/       # Shared utilities
├── pkg/                  # Package metadata
├── examples/             # Usage examples
├── tests/                # Test suites
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── .github/              # GitHub workflows
├── scripts/              # Automation scripts
├── pyproject.toml        # Project configuration
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local development
├── README.md            # Project documentation
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-app
   cd my-app
   ```

2. **Initialize the project**:
   ```bash
   ./scripts/init.sh
   ```

3. **Start development environment**:
   ```bash
   docker-compose up -d
   ```

4. **Run the application**:
   ```bash
   # API server
   python -m my_app.api

   # CLI
   python -m my_app.cli --help
   ```

## Development

### Local Development

```bash
# Install dependencies
pip install -e ".[dev,test]"

# Run API server
uvicorn my_app.api:app --reload

# Run CLI
my-app --help
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=src/my_app
```

### Code Quality

```bash
# Format code
ruff format

# Lint code
ruff check

# Type checking
mypy src/

# Security check
bandit -r src/
```

## Build & Deployment

### Docker Build

```bash
# Build image
docker build -t my-app:latest .

# Run container
docker run -p 8000:8000 my-app:latest
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Check deployment
kubectl get pods -l app=my-app
```

### Cloud Deployment

```bash
# Deploy to AWS
./deploy/cloud/aws/deploy.sh

# Deploy to Azure
./deploy/cloud/azure/deploy.sh

# Deploy to GCP
./deploy/cloud/gcp/deploy.sh
```

## Configuration

The application uses environment-based configuration:

```bash
# Development
export APP_ENV=development
export DEBUG=true

# Production
export APP_ENV=production
export DEBUG=false
export DATABASE_URL=postgresql://...
```

## Monitoring

- **Health Checks**: `/health`, `/ready`, `/live`
- **Metrics**: Prometheus metrics at `/metrics`
- **Logging**: Structured JSON logs
- **Tracing**: OpenTelemetry integration

## Security

- **Authentication**: JWT tokens
- **Authorization**: RBAC with scopes
- **Input Validation**: Pydantic models
- **Security Headers**: CORS, CSP, HSTS
- **Rate Limiting**: Request throttling

## License

MIT License - see LICENSE file for details