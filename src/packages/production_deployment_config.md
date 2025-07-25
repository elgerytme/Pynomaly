# Production Deployment Configurations

## Overview

This document outlines production deployment configurations for the hexagonal architecture implementation across all packages. The configurations support scalable, maintainable, and secure deployments.

## Architecture Overview

### Package Structure
- **Data Quality Package**: ✅ Complete hexagonal architecture
- **MLOps Package**: 🟡 Partial implementation  
- **Machine Learning Package**: ❌ Incomplete (needs fixes)
- **Anomaly Detection Package**: 🟡 Basic (candidate for implementation)

### Deployment Strategy

Our hexagonal architecture enables flexible deployment patterns:

1. **Microservices Deployment**: Each package as independent service
2. **Monolithic Deployment**: All packages in single container
3. **Hybrid Deployment**: Core services together, specialized services separate

## Container Configurations

### Data Quality Service

```yaml
# docker-compose.data-quality.yml
version: '3.8'
services:
  data-quality:
    build:
      context: ./data/data_quality
      dockerfile: Dockerfile.prod
    environment:
      - ENVIRONMENT=production
      - DATA_STORAGE_PATH=/app/data
      - ENABLE_FILE_DATA_PROCESSING=true
      - ENABLE_DISTRIBUTED_PROCESSING=false
      - LOG_LEVEL=INFO
    volumes:
      - data_quality_storage:/app/data
      - ./config/data_quality:/app/config:ro
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  data_quality_storage:
    driver: local
```

### MLOps Service

```yaml
# docker-compose.mlops.yml
version: '3.8'
services:
  mlops:
    build:
      context: ./ai/mlops
      dockerfile: Dockerfile.prod
    environment:
      - ENVIRONMENT=production
      - ENABLE_FILE_SERVICE_DISCOVERY=true
      - ENABLE_FILE_CONFIGURATION=true
      - SERVICE_DISCOVERY_PATH=/app/service_discovery
      - CONFIGURATION_PATH=/app/configuration
      - DATABASE_URL=postgresql://mlops:${POSTGRES_PASSWORD}@postgres:5432/mlops
    volumes:
      - mlops_storage:/app/storage
      - ./config/mlops:/app/config:ro
    ports:
      - "8081:8080"
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mlops
      - POSTGRES_USER=mlops
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  mlops_storage:
    driver: local
  postgres_data:
    driver: local
  redis_data:
    driver: local
```

## Kubernetes Deployment

### Data Quality Service

```yaml
# k8s/data-quality/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-quality
  labels:
    app: data-quality
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-quality
  template:
    metadata:
      labels:
        app: data-quality
        version: v1
    spec:
      containers:
      - name: data-quality
        image: your-registry/data-quality:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATA_STORAGE_PATH
          value: "/app/data"
        - name: ENABLE_FILE_DATA_PROCESSING
          value: "true"
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        - name: config
          mountPath: /app/config
          readOnly: true
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-quality-pvc
      - name: config
        configMap:
          name: data-quality-config

---
apiVersion: v1
kind: Service
metadata:
  name: data-quality-service
spec:
  selector:
    app: data-quality
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-quality-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### MLOps Service

```yaml
# k8s/mlops/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops
  labels:
    app: mlops
    version: v1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlops
  template:
    metadata:
      labels:
        app: mlops
        version: v1
    spec:
      containers:
      - name: mlops
        image: your-registry/mlops:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mlops-secrets
              key: database-url
        - name: ENABLE_FILE_SERVICE_DISCOVERY
          value: "true"
        - name: ENABLE_FILE_CONFIGURATION
          value: "true"
        volumeMounts:
        - name: storage
          mountPath: /app/storage
        - name: config
          mountPath: /app/config
          readOnly: true
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: mlops-pvc
      - name: config
        configMap:
          name: mlops-config

---
apiVersion: v1
kind: Service
metadata:
  name: mlops-service
spec:
  selector:
    app: mlops
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

## Configuration Management

### Production Environment Variables

```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Data Quality Service
DATA_QUALITY_STORAGE_PATH=/app/data
DATA_QUALITY_ENABLE_FILE_PROCESSING=true
DATA_QUALITY_ENABLE_DISTRIBUTED=false

# MLOps Service
MLOPS_SERVICE_DISCOVERY_PATH=/app/service_discovery
MLOPS_CONFIGURATION_PATH=/app/configuration
MLOPS_ENABLE_FILE_SERVICE_DISCOVERY=true
MLOPS_ENABLE_FILE_CONFIGURATION=true

# Database
DATABASE_URL=postgresql://mlops:${POSTGRES_PASSWORD}@postgres:5432/mlops
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET=${JWT_SECRET}
API_KEY=${API_KEY}

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true
```

### Configuration Files

```yaml
# config/data_quality/production.yaml
data_quality:
  environment: production
  storage:
    type: file
    path: /app/data
    backup_enabled: true
    backup_interval: 3600
  processing:
    batch_size: 1000
    max_workers: 4
    timeout: 300
  quality_assessment:
    enable_real_time: true
    enable_batch: true
    thresholds:
      completeness: 0.95
      accuracy: 0.90
      consistency: 0.85
  external_systems:
    notifications:
      enabled: true
      channels: [email, slack]
    metadata:
      enabled: true
      storage_type: file
```

```yaml
# config/mlops/production.yaml
mlops:
  environment: production
  service_discovery:
    type: file
    refresh_interval: 30
    health_check_interval: 10
  configuration:
    type: file
    watch_enabled: true
    validation_enabled: true
  experiment_tracking:
    enabled: true
    storage_type: file
    artifact_retention_days: 90
  model_registry:
    enabled: true
    storage_type: file
    versioning_enabled: true
  monitoring:
    enabled: true
    metrics_retention_days: 30
    alerting_enabled: true
```

## Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# k8s/data-quality/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: data-quality-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: data-quality
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

### MLOps Scaling

```yaml
# k8s/mlops/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlops-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlops
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 85
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'data-quality'
    static_configs:
      - targets: ['data-quality-service:80']
    scrape_interval: 30s
    metrics_path: /metrics

  - job_name: 'mlops'
    static_configs:
      - targets: ['mlops-service:80']
    scrape_interval: 30s
    metrics_path: /metrics

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Hexagonal Architecture Services",
    "panels": [
      {
        "title": "Data Quality Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"data-quality\"}",
            "legendFormat": "Service Status"
          }
        ]
      },
      {
        "title": "MLOps Service Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"mlops\"}[5m])",
            "legendFormat": "Request Rate"
          }
        ]
      },
      {
        "title": "Container Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{pod=~\"data-quality.*\"}",
            "legendFormat": "Memory Usage"
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
# k8s/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: data-quality-policy
spec:
  podSelector:
    matchLabels:
      app: data-quality
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: mlops
    ports:
    - protocol: TCP
      port: 8080
```

### RBAC Configuration

```yaml
# k8s/security/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hexagonal-services
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: hexagonal-services-binding
subjects:
- kind: ServiceAccount
  name: hexagonal-services
  namespace: default
roleRef:
  kind: Role
  name: hexagonal-services
  apiGroup: rbac.authorization.k8s.io
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Hexagonal Architecture Services

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run cross-package integration tests
        run: |
          cd src/packages
          python test_cross_package_integration_summary.py
      
      - name: Run individual package tests
        run: |
          cd src/packages/data/data_quality
          python test_hexagonal_integration.py

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    strategy:
      matrix:
        service: [data-quality, mlops]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.service }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: src/packages/${{ matrix.service == 'data-quality' && 'data/data_quality' || 'ai/mlops' }}
          file: src/packages/${{ matrix.service == 'data-quality' && 'data/data_quality' || 'ai/mlops' }}/Dockerfile.prod
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/data-quality/
          kubectl apply -f k8s/mlops/
          kubectl apply -f k8s/monitoring/
          kubectl rollout status deployment/data-quality
          kubectl rollout status deployment/mlops
```

## Deployment Scripts

### Production Deployment Script

```bash
#!/bin/bash
# scripts/deploy-production.sh

set -e

echo "🚀 Deploying Hexagonal Architecture Services to Production"

# Configuration
NAMESPACE=${NAMESPACE:-production}
REGISTRY=${REGISTRY:-ghcr.io/your-org}
TAG=${TAG:-latest}

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy secrets
echo "📝 Deploying secrets..."
kubectl apply -f k8s/secrets/ -n $NAMESPACE

# Deploy ConfigMaps
echo "📋 Deploying configuration..."
kubectl apply -f k8s/config/ -n $NAMESPACE

# Deploy Data Quality Service
echo "📊 Deploying Data Quality Service..."
kubectl apply -f k8s/data-quality/ -n $NAMESPACE
kubectl rollout status deployment/data-quality -n $NAMESPACE

# Deploy MLOps Service
echo "🔧 Deploying MLOps Service..."
kubectl apply -f k8s/mlops/ -n $NAMESPACE
kubectl rollout status deployment/mlops -n $NAMESPACE

# Deploy Monitoring
echo "📈 Deploying Monitoring Stack..."
kubectl apply -f k8s/monitoring/ -n $NAMESPACE

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "🎉 Deployment completed successfully!"
echo "📊 Data Quality Service: http://data-quality-service.$NAMESPACE.svc.cluster.local"
echo "🔧 MLOps Service: http://mlops-service.$NAMESPACE.svc.cluster.local"
```

## Disaster Recovery

### Backup Configuration

```yaml
# k8s/backup/backup-job.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:latest
            command:
            - /bin/bash
            - -c
            - |
              # Backup data quality storage
              tar -czf /backup/data-quality-$(date +%Y%m%d).tar.gz /app/data
              
              # Backup MLOps storage
              tar -czf /backup/mlops-$(date +%Y%m%d).tar.gz /app/storage
              
              # Upload to cloud storage
              aws s3 cp /backup/ s3://your-backup-bucket/$(date +%Y%m%d)/ --recursive
            volumeMounts:
            - name: data-quality-storage
              mountPath: /app/data
            - name: mlops-storage
              mountPath: /app/storage
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: data-quality-storage
            persistentVolumeClaim:
              claimName: data-quality-pvc
          - name: mlops-storage
            persistentVolumeClaim:
              claimName: mlops-pvc
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

## Next Steps

1. **Complete Machine Learning Package**: Fix import issues and complete hexagonal architecture
2. **Implement Anomaly Detection**: Apply hexagonal pattern to anomaly detection package
3. **Enhanced Monitoring**: Add custom metrics for business logic
4. **Performance Optimization**: Implement caching and optimization strategies
5. **Security Hardening**: Add authentication, authorization, and encryption
6. **Multi-Cloud Support**: Add cloud provider-specific configurations

This production deployment configuration provides a solid foundation for deploying the hexagonal architecture implementation with proper scalability, monitoring, and security considerations.