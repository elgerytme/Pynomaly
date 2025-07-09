# Pynomaly Docker Setup Guide

This guide covers the complete Docker setup for Pynomaly, including multi-stage builds for development, production, and slim deployments.

## Overview

Pynomaly provides three Docker image variants:
- **Development**: Full development environment with hot-reload, debugging tools, and all dependencies
- **Production**: Optimized production image with security hardening and performance optimizations
- **Slim**: Minimal production image with only essential dependencies

## Quick Start

### Prerequisites

- Docker Desktop or Docker Engine
- Docker Compose v2.0+
- At least 4GB RAM available for containers
- 10GB free disk space

### Development Environment

```bash
# Start full development environment
make docker-run-dev

# Or using PowerShell
.\scripts\docker\docker-helpers.ps1 start-dev
```

This starts:
- **API Server**: http://localhost:8000
- **Web UI**: http://localhost:8080
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **pgAdmin**: http://localhost:8081 (admin@pynomaly.dev/admin)
- **Redis Commander**: http://localhost:8082
- **Jaeger**: http://localhost:16686

### Production Environment

```bash
# Start production environment
make docker-run-prod

# Or using PowerShell
.\scripts\docker\docker-helpers.ps1 start-prod
```

## Building Images

### All Images

```bash
# Build all variants
make docker-build

# PowerShell
.\scripts\docker\docker-helpers.ps1 build all
```

### Specific Variants

```bash
# Development image
docker build --target development -t pynomaly:dev -f Dockerfile.multi-stage .

# Production image
docker build --target production -t pynomaly:prod -f Dockerfile.multi-stage .

# Slim image
docker build --target slim -t pynomaly:slim -f Dockerfile.multi-stage .
```

## Image Variants

### Development Image (`pynomaly:dev`)

**Features:**
- Hot-reload with uvicorn
- Full development dependencies
- Debugging tools (ipdb, pytest-sugar)
- Development environment variables
- Non-root user for security

**Usage:**
```bash
docker run -p 8000:8000 -v $(pwd)/src:/app/src pynomaly:dev
```

**Environment Variables:**
- `PYNOMALY_ENVIRONMENT=development`
- `PYNOMALY_DEBUG=true`
- `PYNOMALY_LOG_LEVEL=DEBUG`
- `PYNOMALY_RELOAD=true`

### Production Image (`pynomaly:prod`)

**Features:**
- Production-optimized
- Multi-worker uvicorn
- Security hardening
- Minimal attack surface
- Performance optimizations

**Usage:**
```bash
docker run -p 8000:8000 pynomaly:prod
```

**Environment Variables:**
- `PYNOMALY_ENVIRONMENT=production`
- `PYNOMALY_DEBUG=false`
- `PYNOMALY_LOG_LEVEL=INFO`
- `PYNOMALY_RELOAD=false`

### Slim Image (`pynomaly:slim`)

**Features:**
- Minimal dependencies
- Smallest image size
- Essential runtime only
- Optimized for resource-constrained environments

**Usage:**
```bash
docker run -p 8000:8000 pynomaly:slim
```

**Environment Variables:**
- `PYNOMALY_ENVIRONMENT=production`
- `PYNOMALY_DEBUG=false`
- `PYNOMALY_LOG_LEVEL=WARNING`

## Docker Compose Configurations

### Local Development (`docker-compose.local.yml`)

Complete development stack with:
- Hot-reload API server
- PostgreSQL database
- Redis cache
- Monitoring stack (Prometheus, Grafana, Jaeger)
- Development tools (pgAdmin, Redis Commander)
- Nginx reverse proxy

### Production (`deploy/docker/docker-compose.yml`)

Production-ready stack with:
- Multi-worker API server
- Production database
- Redis cache
- Nginx load balancer
- SSL termination
- Health checks

## Configuration

### Environment Variables

| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `PYNOMALY_ENVIRONMENT` | development | production | Runtime environment |
| `PYNOMALY_DEBUG` | true | false | Debug mode |
| `PYNOMALY_LOG_LEVEL` | DEBUG | INFO/WARNING | Logging level |
| `PYNOMALY_RELOAD` | true | false | Hot-reload |
| `PYNOMALY_DB_HOST` | postgres | postgres | Database host |
| `PYNOMALY_DB_PORT` | 5432 | 5432 | Database port |
| `PYNOMALY_REDIS_HOST` | redis | redis | Redis host |
| `PYNOMALY_REDIS_PORT` | 6379 | 6379 | Redis port |

### Database Configuration

The PostgreSQL database is automatically initialized with:
- Core tables (datasets, detectors, results, users)
- Audit logging tables
- Performance metrics tables
- Indexes for optimal performance
- Default admin user (admin/admin123)

### Redis Configuration

Redis is configured for development with:
- Memory limit: 256MB
- LRU eviction policy
- Persistence enabled
- Keyspace notifications for development

### Monitoring

#### Prometheus Metrics

Available at http://localhost:9090

Scrapes metrics from:
- Pynomaly API server
- PostgreSQL
- Redis
- Jaeger
- Grafana

#### Grafana Dashboards

Available at http://localhost:3000 (admin/admin)

Pre-configured dashboards for:
- API performance
- Database metrics
- Redis metrics
- System resources

#### Jaeger Tracing

Available at http://localhost:16686

Distributed tracing for:
- API requests
- Database queries
- Cache operations
- Background jobs

## Security

### Non-Root User

All images run as non-root user `pynomaly` (UID 1000) for security.

### Network Security

- Services communicate via internal Docker network
- External access only through reverse proxy
- Rate limiting on API endpoints
- SSL/TLS termination in production

### Secret Management

- Database credentials via environment variables
- No hardcoded secrets in images
- Secure defaults for development

## Performance Optimization

### Development

- Volume mounts for hot-reload
- Optimized for development workflow
- Debug tools enabled

### Production

- Multi-worker uvicorn
- Connection pooling
- Gzip compression
- Static file caching
- Health checks

### Slim

- Minimal dependencies
- Optimized for resource constraints
- Reduced attack surface
- Faster startup time

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 8080, 3000, 5432, 6379 are available
2. **Memory issues**: Increase Docker memory limit to at least 4GB
3. **Permission issues**: Ensure Docker daemon is running with proper permissions

### Debugging

```bash
# View logs
docker-compose -f docker-compose.local.yml logs -f pynomaly-dev

# Access container
docker exec -it pynomaly-dev /bin/bash

# Check service status
docker-compose -f docker-compose.local.yml ps
```

### Health Checks

All services include health checks:
- API server: `GET /api/v1/health`
- Database: `pg_isready`
- Redis: `redis-cli ping`

## Maintenance

### Cleanup

```bash
# Stop all containers
make docker-stop

# Clean Docker resources
make docker-clean

# PowerShell
.\scripts\docker\docker-helpers.ps1 stop
.\scripts\docker\docker-helpers.ps1 clean
```

### Updates

```bash
# Update images
docker-compose -f docker-compose.local.yml pull

# Rebuild with latest changes
docker-compose -f docker-compose.local.yml up --build
```

## Best Practices

1. **Development**: Use development image with volume mounts
2. **Testing**: Use production image for integration testing
3. **Production**: Use slim image for production deployment
4. **Security**: Regularly update base images and dependencies
5. **Monitoring**: Always enable monitoring in production
6. **Backups**: Regular database and Redis backups
7. **Logging**: Centralized logging for production

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Pynomaly API Documentation](http://localhost:8000/docs)
- [Grafana Dashboard Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/pynomaly/pynomaly/issues
- Documentation: https://pynomaly.readthedocs.io
- Community: https://discord.gg/pynomaly
