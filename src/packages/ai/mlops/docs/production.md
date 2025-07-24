# Production Deployment Guide

This guide provides comprehensive instructions for deploying MLOps components to production environments.

## Prerequisites

### System Requirements

- **Kubernetes**: v1.24+ cluster with sufficient resources
- **Docker**: v20.10+ for containerization
- **Helm**: v3.8+ for package management
- **Python**: 3.11+ for runtime environments
- **Storage**: Persistent storage for artifacts and metadata

### Infrastructure Components

- **Container Registry**: Docker Hub, ECR, GCR, or ACR
- **Database**: PostgreSQL 14+ for metadata storage
- **Cache**: Redis 6+ for caching and session storage
- **Message Queue**: Apache Kafka or RabbitMQ for event streaming
- **Monitoring**: Prometheus and Grafana for observability
- **Logging**: ELK Stack or similar for log aggregation

## Production Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
┌─────────┐            ┌─────────┐              ┌─────────┐
│   API   │            │   Web   │              │  Admin  │
│Gateway  │            │   UI    │              │   UI    │
└─────────┘            └─────────┘              └─────────┘
    │                         │                         │
    └─────────────────────────┼─────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Services                           │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Experiment  │ │   Model     │ │      Monitoring         │ │
│ │  Tracking   │ │ Registry    │ │       Service           │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Deployment  │ │  Feature    │ │      Pipeline           │ │
│ │  Service    │ │   Store     │ │    Orchestrator         │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ PostgreSQL  │ │    Redis    │ │        Kafka            │ │
│ │  Database   │ │   Cache     │ │     Message Queue       │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Container Images

### Building Production Images

1. **Base Image Configuration**

```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlops
USER mlops
WORKDIR /home/mlops

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=mlops:mlops . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Multi-stage Build**

```dockerfile
# Multi-stage Dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd --create-home --shell /bin/bash mlops
USER mlops
WORKDIR /home/mlops

# Copy application code
COPY --chown=mlops:mlops . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. **Build and Push Images**

```bash
#!/bin/bash
# build-images.sh

# Build images
docker build -t mlops/experiment-tracker:v1.0.0 -f Dockerfile.experiment-tracker .
docker build -t mlops/model-registry:v1.0.0 -f Dockerfile.model-registry .
docker build -t mlops/deployment-service:v1.0.0 -f Dockerfile.deployment-service .
docker build -t mlops/monitoring-service:v1.0.0 -f Dockerfile.monitoring-service .

# Tag for registry
docker tag mlops/experiment-tracker:v1.0.0 registry.company.com/mlops/experiment-tracker:v1.0.0
docker tag mlops/model-registry:v1.0.0 registry.company.com/mlops/model-registry:v1.0.0
docker tag mlops/deployment-service:v1.0.0 registry.company.com/mlops/deployment-service:v1.0.0
docker tag mlops/monitoring-service:v1.0.0 registry.company.com/mlops/monitoring-service:v1.0.0

# Push to registry
docker push registry.company.com/mlops/experiment-tracker:v1.0.0
docker push registry.company.com/mlops/model-registry:v1.0.0
docker push registry.company.com/mlops/deployment-service:v1.0.0
docker push registry.company.com/mlops/monitoring-service:v1.0.0
```

## Kubernetes Deployment

### Namespace Configuration

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops-production
  labels:
    name: mlops-production
    environment: production
```

### ConfigMap and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-config
  namespace: mlops-production
data:
  DATABASE_URL: "postgresql://mlops:password@postgres:5432/mlops"
  REDIS_URL: "redis://redis:6379"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
  MLFLOW_TRACKING_URI: "http://mlflow:5000"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"

---
apiVersion: v1
kind: Secret
metadata:
  name: mlops-secrets
  namespace: mlops-production
type: Opaque
stringData:
  database-password: "your-secure-password"
  redis-password: "your-redis-password"
  jwt-secret: "your-jwt-secret"
  aws-access-key: "your-aws-access-key"
  aws-secret-key: "your-aws-secret-key"
```

### Database Deployment

```yaml
# postgresql.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgresql
  namespace: mlops-production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
      - name: postgresql
        image: postgres:14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: mlops
        - name: POSTGRES_USER
          value: mlops
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlops-secrets
              key: database-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgresql
  namespace: mlops-production
spec:
  selector:
    app: postgresql
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: mlops-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### Redis Deployment

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: mlops-production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --requirepass
        - $(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlops-secrets
              key: redis-password
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: mlops-production
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: mlops-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### MLOps Services Deployment

```yaml
# experiment-tracker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: experiment-tracker
  namespace: mlops-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: experiment-tracker
  template:
    metadata:
      labels:
        app: experiment-tracker
    spec:
      containers:
      - name: experiment-tracker
        image: registry.company.com/mlops/experiment-tracker:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: mlops-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: mlops-config
              key: REDIS_URL
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: experiment-tracker
  namespace: mlops-production
spec:
  selector:
    app: experiment-tracker
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: experiment-tracker-hpa
  namespace: mlops-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: experiment-tracker
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlops-ingress
  namespace: mlops-production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - mlops.company.com
    secretName: mlops-tls
  rules:
  - host: mlops.company.com
    http:
      paths:
      - path: /api/experiments
        pathType: Prefix
        backend:
          service:
            name: experiment-tracker
            port:
              number: 8000
      - path: /api/models
        pathType: Prefix
        backend:
          service:
            name: model-registry
            port:
              number: 8000
      - path: /api/deployments
        pathType: Prefix
        backend:
          service:
            name: deployment-service
            port:
              number: 8000
      - path: /api/monitoring
        pathType: Prefix
        backend:
          service:
            name: monitoring-service
            port:
              number: 8000
```

## Helm Chart Deployment

### Chart Structure

```
mlops-chart/
├── Chart.yaml
├── values.yaml
├── values-production.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── hpa.yaml
└── charts/
    ├── postgresql/
    ├── redis/
    └── monitoring/
```

### Chart.yaml

```yaml
# Chart.yaml
apiVersion: v2
name: mlops
description: MLOps Platform Helm Chart
version: 1.0.0
appVersion: "1.0.0"
dependencies:
- name: postgresql
  version: 11.9.13
  repository: https://charts.bitnami.com/bitnami
- name: redis
  version: 17.3.7
  repository: https://charts.bitnami.com/bitnami
- name: kafka
  version: 18.4.3
  repository: https://charts.bitnami.com/bitnami
```

### Production Values

```yaml
# values-production.yaml
global:
  environment: production
  namespace: mlops-production
  imageRegistry: registry.company.com

experimentTracker:
  enabled: true
  replicaCount: 3
  image:
    repository: registry.company.com/mlops/experiment-tracker
    tag: v1.0.0
  resources:
    requests:
      cpu: 100m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

modelRegistry:
  enabled: true
  replicaCount: 3
  image:
    repository: registry.company.com/mlops/model-registry
    tag: v1.0.0
  resources:
    requests:
      cpu: 100m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi

deploymentService:
  enabled: true
  replicaCount: 2
  image:
    repository: registry.company.com/mlops/deployment-service
    tag: v1.0.0

monitoringService:
  enabled: true
  replicaCount: 2
  image:
    repository: registry.company.com/mlops/monitoring-service
    tag: v1.0.0

postgresql:
  enabled: true
  auth:
    username: mlops
    database: mlops
    existingSecret: mlops-secrets
    secretKeys:
      adminPasswordKey: database-password
  primary:
    persistence:
      enabled: true
      size: 20Gi
      storageClass: fast-ssd

redis:
  enabled: true
  auth:
    enabled: true
    existingSecret: mlops-secrets
    existingSecretPasswordKey: redis-password
  master:
    persistence:
      enabled: true
      size: 10Gi
      storageClass: fast-ssd

kafka:
  enabled: true
  persistence:
    enabled: true
    size: 20Gi
    storageClass: fast-ssd

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: mlops.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mlops-tls
      hosts:
        - mlops.company.com

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
  alertmanager:
    enabled: true
```

### Deployment Commands

```bash
#!/bin/bash
# deploy.sh

# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create namespace
kubectl create namespace mlops-production

# Deploy secrets
kubectl apply -f secrets.yaml

# Deploy MLOps monorepo
helm install mlops ./mlops-chart \
  --namespace mlops-production \
  --values values-production.yaml \
  --wait \
  --timeout 10m

# Verify deployment
kubectl get pods -n mlops-production
kubectl get services -n mlops-production
kubectl get ingress -n mlops-production
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: mlops-production
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'mlops-services'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - mlops-production
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
    
    - job_name: 'model-serving'
      static_configs:
      - targets: ['model-serving:8080']
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "MLOps Platform Dashboard",
    "panels": [
      {
        "title": "Experiment Tracking",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(experiment_tracker_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Model Registry",
        "type": "graph",
        "targets": [
          {
            "expr": "model_registry_models_total",
            "legendFormat": "Total Models"
          }
        ]
      },
      {
        "title": "Deployment Status",
        "type": "stat",
        "targets": [
          {
            "expr": "deployment_service_active_deployments",
            "legendFormat": "Active Deployments"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### Network Policies

```yaml
# network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mlops-network-policy
  namespace: mlops-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: experiment-tracker
    - podSelector:
        matchLabels:
          app: model-registry
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: mlops-production
  name: mlops-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlops-role-binding
  namespace: mlops-production
subjects:
- kind: ServiceAccount
  name: mlops-service-account
  namespace: mlops-production
roleRef:
  kind: Role
  name: mlops-role
  apiGroup: rbac.authorization.k8s.io

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlops-service-account
  namespace: mlops-production
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="mlops_backup_$DATE.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
kubectl exec -n mlops-production postgresql-0 -- pg_dump -U mlops mlops > $BACKUP_DIR/$BACKUP_FILE

# Compress backup
gzip $BACKUP_DIR/$BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_DIR/$BACKUP_FILE.gz s3://mlops-backups/database/

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

### Model Artifacts Backup

```bash
#!/bin/bash
# backup-models.sh

BACKUP_DIR="/backups/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup model artifacts
kubectl exec -n mlops-production model-registry-0 -- tar -czf /tmp/models_$DATE.tar.gz /app/models

# Copy backup from pod
kubectl cp mlops-production/model-registry-0:/tmp/models_$DATE.tar.gz $BACKUP_DIR/

# Upload to S3
aws s3 cp $BACKUP_DIR/models_$DATE.tar.gz s3://mlops-backups/models/

# Clean up
kubectl exec -n mlops-production model-registry-0 -- rm /tmp/models_$DATE.tar.gz
```

## Disaster Recovery

### Recovery Procedures

```bash
#!/bin/bash
# disaster-recovery.sh

# 1. Restore database
kubectl exec -n mlops-production postgresql-0 -- psql -U mlops -d mlops < /backup/mlops_backup_latest.sql

# 2. Restore model artifacts
kubectl exec -n mlops-production model-registry-0 -- tar -xzf /backup/models_latest.tar.gz -C /

# 3. Restart services
kubectl rollout restart deployment/experiment-tracker -n mlops-production
kubectl rollout restart deployment/model-registry -n mlops-production
kubectl rollout restart deployment/deployment-service -n mlops-production
kubectl rollout restart deployment/monitoring-service -n mlops-production

# 4. Verify recovery
kubectl get pods -n mlops-production
curl -f http://mlops.company.com/health
```

## Performance Optimization

### Resource Optimization

```yaml
# performance-tuning.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: performance-config
  namespace: mlops-production
data:
  # Database optimization
  postgresql.conf: |
    shared_buffers = 256MB
    effective_cache_size = 1GB
    work_mem = 4MB
    maintenance_work_mem = 64MB
    max_connections = 100
    
  # Redis optimization
  redis.conf: |
    maxmemory 512mb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
```

### Caching Strategy

```python
# caching-config.py
CACHE_CONFIG = {
    "model_cache": {
        "ttl": 3600,  # 1 hour
        "max_size": 100,
        "eviction_policy": "lru"
    },
    "experiment_cache": {
        "ttl": 1800,  # 30 minutes
        "max_size": 500,
        "eviction_policy": "lru"
    },
    "feature_cache": {
        "ttl": 300,   # 5 minutes
        "max_size": 1000,
        "eviction_policy": "lru"
    }
}
```

## Maintenance Procedures

### Regular Maintenance

```bash
#!/bin/bash
# maintenance.sh

# 1. Update images
kubectl set image deployment/experiment-tracker \
  experiment-tracker=registry.company.com/mlops/experiment-tracker:v1.0.1 \
  -n mlops-production

# 2. Check resource usage
kubectl top pods -n mlops-production
kubectl top nodes

# 3. Clean up old experiments
kubectl exec -n mlops-production experiment-tracker-0 -- python cleanup_old_experiments.py

# 4. Optimize database
kubectl exec -n mlops-production postgresql-0 -- psql -U mlops -d mlops -c "VACUUM ANALYZE;"

# 5. Check disk usage
kubectl exec -n mlops-production postgresql-0 -- df -h
```

### Health Checks

```bash
#!/bin/bash
# health-check.sh

# Check pod health
kubectl get pods -n mlops-production

# Check service endpoints
kubectl get endpoints -n mlops-production

# Check ingress
kubectl get ingress -n mlops-production

# Test API endpoints
curl -f http://mlops.company.com/api/experiments/health
curl -f http://mlops.company.com/api/models/health
curl -f http://mlops.company.com/api/deployments/health
curl -f http://mlops.company.com/api/monitoring/health
```

## Troubleshooting Production Issues

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks in applications
   - Optimize database queries
   - Adjust JVM heap sizes

2. **Database Connection Issues**
   - Check connection pool settings
   - Verify database availability
   - Review authentication configurations

3. **Model Serving Latency**
   - Optimize model loading
   - Implement model caching
   - Scale model serving pods

### Debug Commands

```bash
# Check pod logs
kubectl logs -f deployment/experiment-tracker -n mlops-production

# Access pod shell
kubectl exec -it deployment/experiment-tracker -n mlops-production -- /bin/bash

# Check resource usage
kubectl top pods -n mlops-production
kubectl describe pod <pod-name> -n mlops-production

# Check events
kubectl get events -n mlops-production --sort-by=.metadata.creationTimestamp
```

## Compliance and Governance

### Audit Logging

```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: RequestResponse
  namespaces: ["mlops-production"]
  resources:
  - group: ""
    resources: ["pods", "services"]
  - group: "apps"
    resources: ["deployments"]
```

### Data Governance

```python
# data-governance.py
from monorepo.mlops.governance import DataGovernance

governance = DataGovernance()

# Set up data classification
governance.classify_data("sensitive", ["customer_data", "financial_data"])
governance.classify_data("internal", ["model_metadata", "experiment_results"])

# Set up access controls
governance.set_access_policy("sensitive", ["data_scientist", "ml_engineer"])
governance.set_access_policy("internal", ["developer", "data_scientist", "ml_engineer"])

# Enable audit logging
governance.enable_audit_logging()
```

This production deployment guide provides a comprehensive approach to deploying MLOps components in a production environment with proper security, monitoring, and maintenance procedures.