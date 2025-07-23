# Advanced Deployment Guide

This guide covers Docker and Kubernetes deployment configurations for the Anomaly Detection Service, including production-ready setups with monitoring, scaling, and security.

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd anomaly-detection

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f anomaly-detection-api
```

### Production Docker Setup

```bash
# Build production image
./scripts/build-docker.sh 1.0.0

# Run with environment variables
docker run -d \
  --name anomaly-detection-api \
  -p 8000:8000 \
  -e ANOMALY_DETECTION_ENV=production \
  -e ANOMALY_DETECTION_DATABASE_URL=postgresql://user:pass@db:5432/anomaly \
  -v /app/data:/app/data \
  -v /app/models:/app/models \
  anomaly-detection:1.0.0
```

### Docker Compose Services

#### Full Stack with Monitoring
```yaml
# docker-compose.yml includes:
- API Service (FastAPI)
- Worker Service (Background processing)
- PostgreSQL Database
- Redis Cache
- Kafka Streaming
- Prometheus Monitoring
- Grafana Dashboard
- Nginx Load Balancer
```

#### Development Setup
```bash
# Development environment
docker-compose -f docker-compose.dev.yml up -d

# Includes additional services:
- Jupyter Notebook (port 8889)
- Documentation Server (port 8080)
- Test Runner
- Hot reload enabled
```

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Install required tools
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

# helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Quick Deployment

```bash
# Deploy to development
./scripts/deploy-k8s.sh development

# Deploy to production
./scripts/deploy-k8s.sh production 1.0.0
```

### Manual Deployment

```bash
# Development environment
kubectl apply -k k8s/overlays/development

# Production environment
kubectl apply -k k8s/overlays/production

# Check deployment status
kubectl get pods -n anomaly-detection -w
```

### Kubernetes Architecture

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress Layer"
            ING[Nginx Ingress]
        end
        
        subgraph "Application Layer"
            API[API Pods<br/>3-20 replicas]
            WORK[Worker Pods<br/>2-15 replicas]
        end
        
        subgraph "Data Layer"
            PG[(PostgreSQL)]
            REDIS[(Redis)]
            KAFKA[(Kafka)]
        end
        
        subgraph "Storage"
            PVC1[Data Volume]
            PVC2[Model Volume]
            PVC3[Log Volume]
        end
        
        subgraph "Monitoring"
            PROM[Prometheus]
            GRAF[Grafana]
        end
    end
    
    ING --> API
    API --> WORK
    API --> PG
    API --> REDIS
    WORK --> KAFKA
    API --> PVC1
    API --> PVC2
    API --> PVC3
    PROM --> API
    PROM --> WORK
    GRAF --> PROM
```

## 🔧 Configuration Management

### Environment Variables

#### Required Configuration
```bash
# Database
ANOMALY_DETECTION_DATABASE_URL=postgresql://user:pass@host:5432/db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=anomaly_detection

# Cache
ANOMALY_DETECTION_REDIS_URL=redis://host:6379/0
REDIS_PASSWORD=redis_password

# Streaming
ANOMALY_DETECTION_KAFKA_BROKERS=kafka:9092
KAFKA_TOPIC_PREFIX=anomaly-detection

# Security
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=jwt-secret-for-tokens
```

#### Optional Configuration
```bash
# Performance
UVICORN_WORKERS=4
WORKER_CONCURRENCY=4
MAX_BATCH_SIZE=1000

# Monitoring
PROMETHEUS_METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Features
ENABLE_STREAMING=true
ENABLE_EXPLAINABILITY=true
ENABLE_ENSEMBLE=true
```

### Kubernetes ConfigMaps and Secrets

```bash
# Create ConfigMap
kubectl create configmap anomaly-detection-config \
  --from-env-file=.env.production \
  -n anomaly-detection

# Create Secret
kubectl create secret generic anomaly-detection-secrets \
  --from-literal=POSTGRES_PASSWORD=secure_password \
  --from-literal=SECRET_KEY=super-secret-key \
  -n anomaly-detection
```

## 📊 Monitoring and Observability

### Prometheus Metrics

The service exposes metrics at `/metrics`:

```prometheus
# Request metrics
anomaly_detection_requests_total
anomaly_detection_request_duration_seconds
anomaly_detection_request_size_bytes

# Business metrics
anomaly_detection_anomalies_detected_total
anomaly_detection_models_loaded_total
anomaly_detection_detection_accuracy

# System metrics
anomaly_detection_memory_usage_bytes
anomaly_detection_cpu_usage_percent
```

### Grafana Dashboards

Pre-configured dashboards include:
- API Performance Dashboard
- Anomaly Detection Metrics
- System Resource Usage
- Error Rate and SLA Monitoring

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Kubernetes health checks
kubectl get pods -n anomaly-detection
kubectl describe pod <pod-name> -n anomaly-detection
```

## 🔒 Security Configuration

### Container Security

```dockerfile
# Security features in Dockerfile:
# - Non-root user (UID 1000)
# - Read-only filesystem where possible
# - Minimal attack surface
# - Security scanning with Trivy
```

### Kubernetes Security

```yaml
# SecurityContext in deployments:
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
```

### Network Policies

```yaml
# Restrict network traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: anomaly-detection-network-policy
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: anomaly-detection
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
```

## 🚀 Scaling and Performance

### Horizontal Pod Autoscaling (HPA)

```yaml
# Automatically scale based on CPU/Memory
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-detection-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detection-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Pod Autoscaling (VPA)

```yaml
# Automatically adjust resource requests
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: anomaly-detection-api-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detection-api
  updatePolicy:
    updateMode: "Auto"
```

### Performance Tuning

#### Database Optimization
```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

#### Application Tuning
```bash
# Environment variables for performance
UVICORN_WORKERS=8
UVICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker
UVICORN_BACKLOG=2048
UVICORN_MAX_REQUESTS=1000
UVICORN_MAX_REQUESTS_JITTER=100

# Worker configuration
WORKER_CONCURRENCY=8
WORKER_PREFETCH_MULTIPLIER=1
WORKER_MAX_TASKS_PER_CHILD=1000
```

## 🔄 CI/CD Integration

### Docker Build Pipeline

```yaml
# GitHub Actions example
name: Build and Deploy
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t anomaly-detection:${{ github.sha }} .
        docker tag anomaly-detection:${{ github.sha }} anomaly-detection:latest
    
    - name: Push to registry
      run: |
        docker push anomaly-detection:${{ github.sha }}
        docker push anomaly-detection:latest
```

### Kubernetes Deployment Pipeline

```yaml
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: |
        cd k8s/overlays/production
        kustomize edit set image anomaly-detection=anomaly-detection:${{ github.sha }}
        kustomize build . | kubectl apply -f -
```

## 🛠️ Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod status
kubectl get pods -n anomaly-detection

# Describe pod for events
kubectl describe pod <pod-name> -n anomaly-detection

# Check logs
kubectl logs <pod-name> -n anomaly-detection
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n anomaly-detection

# Check HPA status
kubectl get hpa -n anomaly-detection

# Scale manually if needed
kubectl scale deployment anomaly-detection-api --replicas=10 -n anomaly-detection
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it <pod-name> -n anomaly-detection -- \
  psql postgresql://user:pass@postgres:5432/anomaly_detection -c "SELECT 1"

# Check database pod
kubectl get pods -n anomaly-detection -l app.kubernetes.io/name=postgres
kubectl logs <postgres-pod> -n anomaly-detection
```

### Debug Commands

```bash
# Port forward for local access
kubectl port-forward svc/anomaly-detection-api 8000:8000 -n anomaly-detection

# Execute commands in pod
kubectl exec -it <pod-name> -n anomaly-detection -- /bin/bash

# View resource usage
kubectl top pods -n anomaly-detection
kubectl top nodes

# Check events
kubectl get events -n anomaly-detection --sort-by='.lastTimestamp'
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Infrastructure requirements verified
- [ ] Container images built and scanned
- [ ] Configuration files reviewed
- [ ] Secrets properly configured
- [ ] Database migrations prepared
- [ ] Backup strategy in place

### Deployment
- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Verify health checks
- [ ] Check monitoring dashboards
- [ ] Validate scaling behavior
- [ ] Test failover scenarios

### Post-deployment
- [ ] Monitor application metrics
- [ ] Verify log aggregation
- [ ] Check alert configurations
- [ ] Validate backup systems
- [ ] Document any issues
- [ ] Update runbooks

## 🔗 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Guide](https://kustomize.io/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

## 📞 Support

For deployment issues or questions:
- Check the [troubleshooting guide](troubleshooting.md)
- Review [configuration documentation](configuration.md)
- Contact the platform team
- Create an issue in the repository