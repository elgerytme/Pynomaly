# Pynomaly Detection - Kubernetes Deployment Guide

## Overview

This directory contains production-ready Kubernetes manifests for deploying Pynomaly Detection at scale. The deployment includes:

- **Multi-environment support**: Development, Staging, Production
- **Auto-scaling**: Horizontal Pod Autoscaler with CPU, memory, and custom metrics
- **High availability**: Multi-replica deployment with pod anti-affinity
- **Security**: RBAC, network policies, non-root containers
- **Monitoring**: Prometheus metrics and health checks
- **Storage**: Persistent volumes for data, models, cache, and logs

## Quick Start

### Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- Kustomize (v3.8+)
- Docker registry access
- StorageClass configured (`fast-ssd` for production)

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t pynomaly/detection:v0.2.0 .

# Tag for your registry
docker tag pynomaly/detection:v0.2.0 your-registry/pynomaly/detection:v0.2.0

# Push to registry
docker push your-registry/pynomaly/detection:v0.2.0
```

### 2. Deploy to Development

```bash
# Deploy to development environment
kubectl apply -k k8s/overlays/development/

# Check deployment status
kubectl get pods -l app=pynomaly-detection

# View logs
kubectl logs -f deployment/pynomaly-detection
```

### 3. Deploy to Production

```bash
# Create production namespace
kubectl create namespace pynomaly-production

# Update secrets with production values
kubectl create secret generic pynomaly-secrets-production \
  --from-literal=PYNOMALY_DB_PASSWORD=your-production-password \
  --from-literal=PYNOMALY_REDIS_PASSWORD=your-redis-password \
  --from-literal=PYNOMALY_S3_ACCESS_KEY=your-s3-access-key \
  --from-literal=PYNOMALY_S3_SECRET_KEY=your-s3-secret-key \
  -n pynomaly-production

# Deploy to production
kubectl apply -k k8s/overlays/production/

# Verify deployment
kubectl get all -n pynomaly-production
```

## Architecture

### Components

1. **API Server Deployment**
   - Handles HTTP requests
   - Serves REST API endpoints
   - Exposes metrics on port 9090

2. **Worker Deployment** (optional)
   - Processes background tasks
   - Handles batch processing
   - Scales independently

3. **Services**
   - LoadBalancer: External access
   - ClusterIP: Internal communication
   - Headless: Service discovery

4. **Storage**
   - Data PVC: Raw data storage (10Gi)
   - Models PVC: Trained models (50Gi)
   - Cache PVC: Temporary cache (5Gi)
   - Logs PVC: Application logs (20Gi)

### Scaling Strategy

- **Horizontal Pod Autoscaler**: Scales based on CPU, memory, and custom metrics
- **Vertical Pod Autoscaler**: Adjusts resource requests/limits
- **Node Affinity**: Ensures pods run on appropriate node types
- **Pod Anti-Affinity**: Distributes pods across nodes for high availability

## Configuration

### Environment Variables

#### Core Settings
- `PYNOMALY_ENV`: Environment (development/staging/production)
- `PYNOMALY_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `PYNOMALY_WORKERS`: Number of worker processes
- `PYNOMALY_MEMORY_LIMIT_GB`: Memory limit in GB

#### Detection Settings
- `PYNOMALY_DEFAULT_ALGORITHM`: Default detection algorithm
- `PYNOMALY_DEFAULT_CONTAMINATION`: Default contamination rate
- `PYNOMALY_ENABLE_AUTOML`: Enable AutoML features
- `PYNOMALY_ENABLE_ENSEMBLE`: Enable ensemble methods

#### Performance Settings
- `PYNOMALY_BATCH_SIZE`: Batch processing size
- `PYNOMALY_PARALLEL_JOBS`: Number of parallel jobs
- `PYNOMALY_CACHE_SIZE`: Cache size limit
- `PYNOMALY_ENABLE_MEMORY_OPTIMIZATION`: Enable memory optimization

### Secrets Management

Create secrets using kubectl or external secret management:

```bash
# Database credentials
kubectl create secret generic pynomaly-db-secret \
  --from-literal=host=your-db-host \
  --from-literal=username=your-db-user \
  --from-literal=password=your-db-password

# API keys
kubectl create secret generic pynomaly-api-secret \
  --from-literal=api-key=your-api-key \
  --from-literal=secret-key=your-secret-key

# TLS certificates
kubectl create secret tls pynomaly-tls-secret \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

## Monitoring

### Health Checks

The deployment includes comprehensive health checks:

1. **Liveness Probe**: Ensures container is running
2. **Readiness Probe**: Ensures service is ready to accept traffic
3. **Startup Probe**: Handles slow container startup

### Metrics

Prometheus metrics are exposed on port 9090:

- Detection latency
- Memory usage
- CPU usage
- Request count
- Error rate
- Throughput

### Alerting

Configure alerts for:
- High error rate (>5%)
- High memory usage (>90%)
- High latency (>1000ms)
- Low disk space (<1GB)

## Security

### RBAC

- Service accounts with minimal permissions
- Role-based access control
- Network policies for traffic isolation

### Container Security

- Non-root user (UID 1000)
- Read-only root filesystem
- Security contexts
- Resource limits

### Network Security

- Network policies for ingress/egress
- TLS termination at ingress
- Service mesh compatible

## Storage

### Persistent Volumes

| Volume | Purpose | Size | Storage Class |
|--------|---------|------|---------------|
| Data | Raw data storage | 10Gi | fast-ssd |
| Models | Trained models | 50Gi | fast-ssd |
| Cache | Temporary cache | 5Gi | fast-ssd |
| Logs | Application logs | 20Gi | standard |

### Backup Strategy

1. **Scheduled backups**: Daily snapshots of persistent volumes
2. **Cross-region replication**: Models and critical data
3. **Retention policy**: 30 days for logs, 1 year for models

## Troubleshooting

### Common Issues

1. **Pod not starting**
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **High memory usage**
   ```bash
   kubectl top pods
   kubectl exec -it <pod-name> -- python healthcheck.py
   ```

3. **Service not accessible**
   ```bash
   kubectl get svc
   kubectl get ingress
   kubectl describe ingress pynomaly-detection-ingress
   ```

### Debug Commands

```bash
# Check deployment status
kubectl get deployment pynomaly-detection -o wide

# View pod logs
kubectl logs -f deployment/pynomaly-detection

# Execute health check
kubectl exec -it deployment/pynomaly-detection -- python healthcheck.py

# Check resource usage
kubectl top pods -l app=pynomaly-detection

# View HPA status
kubectl get hpa pynomaly-detection-hpa

# Check network policies
kubectl get networkpolicy
```

## Environments

### Development
- 1 replica
- Reduced resource limits
- Debug logging enabled
- Local storage

### Staging
- 2 replicas
- Production-like configuration
- Integration testing
- Shared storage

### Production
- 5+ replicas
- High availability
- Performance monitoring
- Distributed storage

## Maintenance

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/pynomaly-detection \
  pynomaly-detection=pynomaly/detection:v0.2.1

# Check rollout status
kubectl rollout status deployment/pynomaly-detection

# Rollback if needed
kubectl rollout undo deployment/pynomaly-detection
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment pynomaly-detection --replicas=10

# Update HPA
kubectl patch hpa pynomaly-detection-hpa -p '{"spec":{"maxReplicas":20}}'
```

### Backup and Recovery

```bash
# Backup persistent volumes
kubectl get pvc -o yaml > pvc-backup.yaml

# Create volume snapshots
kubectl apply -f volume-snapshot.yaml

# Restore from backup
kubectl apply -f pvc-backup.yaml
```

## Performance Tuning

### Resource Optimization

1. **CPU**: Start with 500m request, 2000m limit
2. **Memory**: Start with 1Gi request, 4Gi limit
3. **Storage**: Use fast SSDs for model storage
4. **Network**: Enable ingress caching

### Algorithm Optimization

1. **Batch Processing**: Use larger batch sizes (1000+)
2. **Parallel Processing**: Enable parallel jobs
3. **Memory Management**: Enable memory optimization
4. **Caching**: Use Redis for model caching

## Support

For issues and questions:
- Check the troubleshooting guide
- Review pod logs and metrics
- Contact the Pynomaly team
- Create GitHub issues for bugs