# Pynomaly Production Deployment Guide

Complete guide for deploying Pynomaly to production environments with Docker, Kubernetes, and comprehensive monitoring.

## Overview

This deployment guide covers:
- **Docker Production Build**: Multi-stage optimized containers
- **Kubernetes Deployment**: Complete production manifests with security, monitoring, and scaling
- **CI/CD Pipeline**: GitHub Actions with automated testing and deployment
- **Monitoring Stack**: Prometheus, Grafana, AlertManager with comprehensive alerting
- **Security**: Production-hardened configurations and secret management
- **Validation**: Automated smoke tests and deployment validation

## Quick Start

### Prerequisites

- Docker 20.10+
- Kubernetes 1.24+
- kubectl configured
- Helm 3.8+ (optional)
- Python 3.11+
- Poetry (for local development)

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Create production secrets
cp scripts/production_secrets.template.env deploy/production/production.env
# Edit production.env with actual secrets (see Security section)

# Verify environment
python scripts/validate_file_organization.py
```

### 2. Docker Deployment

```bash
# Build production image
docker build -f Dockerfile.production -t pynomaly:production-latest .

# Start with Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Validate deployment
scripts/validate_deployment.sh --namespace default
```

### 3. Kubernetes Deployment

```bash
# Deploy to Kubernetes
scripts/deploy.sh --environment production --namespace pynomaly-production

# Validate deployment
scripts/validate_deployment.sh --namespace pynomaly-production

# Run smoke tests
python scripts/smoke_tests.py --url http://your-api-endpoint
```

## Architecture

### Production Stack Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Load Balancer / Ingress                    │
├─────────────────────────────────────────────────────────────────┤
│  Pynomaly API (3 replicas)  │  Background Workers (2 replicas) │
├─────────────────────────────────────────────────────────────────┤
│        Redis Cache          │          PostgreSQL DB           │
├─────────────────────────────────────────────────────────────────┤
│ Prometheus │  Grafana  │ AlertManager │    Log Aggregation    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **High Availability**: Multi-replica deployments with pod disruption budgets
- **Auto Scaling**: Horizontal Pod Autoscaler based on CPU/memory metrics
- **Security**: Non-root containers, security contexts, network policies
- **Monitoring**: Comprehensive metrics collection and alerting
- **Observability**: Distributed tracing, structured logging, health checks

## Configuration

### Environment Variables

#### Core Application
```bash
PYNOMALY_ENV=production
PYNOMALY_LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://:pass@host:6379/0
```

#### Security
```bash
JWT_SECRET_KEY=your-jwt-secret
API_SECRET_KEY=your-api-secret
ENCRYPTION_KEY=your-encryption-key
```

#### Features
```bash
PROMETHEUS_ENABLED=true
HEALTH_CHECK_ENABLED=true
CACHE_ENABLED=true
STREAMING_ENABLED=true
```

### Resource Requirements

#### Minimum Production Setup
- **CPU**: 4 vCPUs total
- **Memory**: 8GB RAM total  
- **Storage**: 50GB persistent
- **Network**: 1Gbps

#### Recommended Production Setup
- **CPU**: 8 vCPUs total
- **Memory**: 16GB RAM total
- **Storage**: 200GB persistent (with backup)
- **Network**: 10Gbps with redundancy

## Security

### Secret Management

1. **Generate Secrets**:
```bash
# Generate strong passwords
openssl rand -hex 32  # For JWT/API keys
openssl rand -base64 32  # For database passwords
```

2. **Kubernetes Secrets**:
```bash
# Create secrets
kubectl create secret generic pynomaly-secrets \
  --from-env-file=deploy/production/production.env \
  -n pynomaly-production
```

3. **External Secret Management** (Recommended):
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- HashiCorp Vault

### Security Hardening

- **Non-root containers**: All containers run as non-root user (UID 1000)
- **Read-only filesystem**: Application containers use read-only root filesystem
- **Security contexts**: Drop all capabilities, prevent privilege escalation
- **Network policies**: Restrict inter-pod communication
- **RBAC**: Minimal required permissions

## Monitoring & Alerting

### Metrics Collection

**Application Metrics**:
- Request rates, latency, error rates
- Model performance, accuracy, prediction confidence
- Resource usage (CPU, memory, GPU)
- Business metrics (detections, datasets processed)

**Infrastructure Metrics**:
- Container resource usage
- Database performance
- Cache hit rates
- Network performance

### Alert Categories

| Severity | Examples | Response Time |
|----------|----------|---------------|
| Critical | API down, high error rate | Immediate |
| Warning | High latency, resource usage | 15 minutes |
| Info | Business metrics, deployments | 1 hour |

### Dashboards

1. **API Overview**: Request metrics, error rates, response times
2. **ML Performance**: Model accuracy, prediction metrics, anomaly rates
3. **Infrastructure**: Resource usage, database performance
4. **Business KPIs**: Dataset processing, user activity

## Deployment Workflows

### CI/CD Pipeline

**Triggers**:
- Push to main branch → Deploy to staging
- Git tag (v*) → Deploy to production
- Manual workflow dispatch → Deploy to specific environment

**Pipeline Stages**:
1. **Quality Assurance**: Linting, type checking, security scanning
2. **Testing**: Unit, integration, API tests
3. **Build**: Docker image with security scanning
4. **Deploy**: Blue-green deployment to Kubernetes
5. **Validate**: Smoke tests and health checks
6. **Monitor**: Enhanced monitoring for new deployment

### Manual Deployment

```bash
# Deploy to staging
scripts/deploy.sh --environment staging --namespace pynomaly-staging

# Deploy to production (requires confirmation)
scripts/deploy.sh --environment production --namespace pynomaly-production

# Deploy specific version
scripts/deploy.sh --tag v2.1.0 --environment production

# Dry run deployment
scripts/deploy.sh --dry-run --environment production
```

## Scaling

### Horizontal Pod Autoscaler

```yaml
# API Autoscaling
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%

# Worker Autoscaling  
minReplicas: 2
maxReplicas: 8
targetCPUUtilization: 80%
targetMemoryUtilization: 85%
```

### Cluster Autoscaling

Configure cluster autoscaler for automatic node scaling:
- Scale up: When pods cannot be scheduled
- Scale down: When nodes are underutilized (< 50% for 10+ minutes)

## High Availability

### Pod Distribution

- **Anti-affinity**: Spread pods across different nodes
- **Pod Disruption Budgets**: Maintain minimum available replicas
- **Rolling Updates**: Zero-downtime deployments

### Data Persistence

- **Database**: PostgreSQL with automated backups
- **Redis**: Persistence enabled with AOF
- **Storage**: Persistent volumes with backup strategies

### Disaster Recovery

1. **Backup Strategy**:
   - Database: Daily full backups, continuous WAL archiving
   - Application data: Daily persistent volume snapshots
   - Configuration: Git-based configuration management

2. **Recovery Procedures**:
   - Database restore from backup
   - Application deployment from known-good images
   - Configuration restoration from Git

## Troubleshooting

### Common Issues

**Deployment Failures**:
```bash
# Check deployment status
kubectl get deployment pynomaly-api -n pynomaly-production

# Check pod logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-production

# Check events
kubectl get events -n pynomaly-production --sort-by=.lastTimestamp
```

**Performance Issues**:
```bash
# Check resource usage
kubectl top pods -n pynomaly-production

# Check metrics
curl http://your-api/metrics

# Check database performance
kubectl exec -it postgres-pod -- psql -c "SELECT * FROM pg_stat_activity;"
```

**Network Issues**:
```bash
# Test service connectivity
kubectl run test-pod --image=busybox -it --rm -- wget -qO- http://pynomaly-api-service:8000/health

# Check service endpoints
kubectl get endpoints pynomaly-api-service -n pynomaly-production
```

### Log Analysis

**Application Logs**:
```bash
# API logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-production

# Worker logs  
kubectl logs -f deployment/pynomaly-worker -n pynomaly-production

# Aggregated logs (if using ELK stack)
# Access Kibana dashboard for log analysis
```

**System Logs**:
```bash
# Node logs
kubectl describe node <node-name>

# Event logs
kubectl get events --all-namespaces --sort-by=.lastTimestamp
```

## Validation & Testing

### Deployment Validation

```bash
# Run full validation suite
scripts/validate_deployment.sh --namespace pynomaly-production --verbose

# Run smoke tests only
python scripts/smoke_tests.py --url https://api.pynomaly.ai
```

### Health Checks

**Endpoints**:
- `/api/v1/health` - Basic health check
- `/api/v1/health/ready` - Readiness check
- `/api/v1/health/dependencies` - Dependency health
- `/metrics` - Prometheus metrics

**Expected Responses**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "uptime": 86400,
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy"
  }
}
```

## Maintenance

### Regular Tasks

**Daily**:
- Check monitoring dashboards
- Review error rates and alerts
- Monitor resource usage

**Weekly**:
- Review and rotate logs
- Check backup status
- Update dependencies (security patches)

**Monthly**:
- Rotate secrets and credentials
- Review and update monitoring thresholds
- Capacity planning review

### Updates

**Application Updates**:
```bash
# Update to new version
scripts/deploy.sh --tag v2.1.0 --environment production

# Rollback if needed
kubectl rollout undo deployment/pynomaly-api -n pynomaly-production
```

**Infrastructure Updates**:
- Kubernetes cluster updates
- Node OS patches
- Dependency updates

## Support

### Documentation
- [API Documentation](https://api.pynomaly.ai/docs)
- [User Guide](docs/user-guide.md)
- [Development Setup](docs/development.md)

### Monitoring
- [Grafana Dashboards](https://monitoring.pynomaly.ai)
- [Alert Manager](https://alerts.pynomaly.ai)
- [Status Page](https://status.pynomaly.ai)

### Contact
- Production Issues: oncall@pynomaly.ai
- Platform Team: platform-team@pynomaly.ai  
- General Support: support@pynomaly.ai

---

*Last Updated: 2024-01-15*
*Version: 2.0.0*