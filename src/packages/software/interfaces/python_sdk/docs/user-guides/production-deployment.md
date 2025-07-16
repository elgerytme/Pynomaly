# Production Deployment Guide

This comprehensive guide covers deploying Pynomaly to production environments, including container orchestration, monitoring, scaling, and operational best practices for enterprise environments.

## 🏗️ Architecture Overview

### Production Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                    API Gateway                                  │
│                 (Authentication & Rate Limiting)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│               Application Layer (Auto-scaled)                  │
├─────────────────┬───────┼───────┬─────────────────────────────────┤
│  Web Interface  │  REST API     │   Real-time Processing         │
│                 │               │                                │
│  - Dashboard    │  - Endpoints  │   - Stream Processing          │
│  - Admin UI     │  - ML Models  │   - Event Processing          │
│  - Monitoring   │  - Governance │   - Alert Generation          │
└─────────────────┴───────┼───────┴─────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                  Data Layer                                    │
├─────────────────┬───────┼───────┬─────────────────────────────────┤
│   PostgreSQL    │   Redis       │   Object Storage               │
│                 │               │                                │
│ - User Data     │ - Cache       │ - Model Artifacts             │
│ - Metadata      │ - Sessions    │ - Training Data               │
│ - Audit Logs    │ - Queues      │ - Backups                     │
└─────────────────┴───────┼───────┴─────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────┐
│                Infrastructure Layer                            │
├─────────────────┬───────┼───────┬─────────────────────────────────┤
│   Monitoring    │   Logging     │   Message Queue                │
│                 │               │                                │
│ - Prometheus    │ - ELK Stack   │ - Apache Kafka                │
│ - Grafana       │ - Fluentd     │ - RabbitMQ                    │
│ - AlertManager  │ - Jaeger      │ - Redis Streams               │
└─────────────────┴───────────────┴─────────────────────────────────┘
```

## 🐳 Container Deployment

### Docker Configuration

#### Production Dockerfile
```dockerfile
# Production Dockerfile
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=pynomaly:pynomaly . .

# Install application
RUN pip install -e .

# Switch to non-root user
USER pynomaly

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start application
CMD ["pynomaly", "server", "start", "--host", "0.0.0.0", "--port", "8080"]
```

#### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  pynomaly:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PYNOMALY_DATABASE_URL=postgresql://pynomaly:password@postgres:5432/pynomaly
      - PYNOMALY_REDIS_URL=redis://redis:6379/0
      - PYNOMALY_LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_USER=pynomaly
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pynomaly"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

## ☸️ Kubernetes Deployment

### Core Application Deployment

#### Namespace and Configuration
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly-prod
  labels:
    name: pynomaly-prod
    environment: production

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
  namespace: pynomaly-prod
data:
  config.yaml: |
    server:
      host: 0.0.0.0
      port: 8080
      workers: 4
      timeout: 30
    
    database:
      pool_size: 20
      max_overflow: 10
      pool_timeout: 30
      pool_recycle: 3600
    
    redis:
      db: 0
      max_connections: 100
    
    ml:
      model_cache_size: 10
      prediction_timeout: 5.0
      batch_size: 1000
    
    logging:
      level: INFO
      format: json
      handlers:
        - console
        - file
    
    monitoring:
      metrics_enabled: true
      health_check_interval: 30
      alert_webhooks:
        - https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

#### Application Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-app
  namespace: pynomaly-prod
  labels:
    app: pynomaly
    component: app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: pynomaly
      component: app
  template:
    metadata:
      labels:
        app: pynomaly
        component: app
    spec:
      serviceAccountName: pynomaly
      containers:
      - name: pynomaly
        image: pynomaly/pynomaly:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: PYNOMALY_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        - name: PYNOMALY_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: redis-url
        - name: PYNOMALY_CONFIG_FILE
          value: /etc/pynomaly/config.yaml
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
        volumeMounts:
        - name: config
          mountPath: /etc/pynomaly
        - name: data
          mountPath: /app/data
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: config
        configMap:
          name: pynomaly-config
      - name: data
        persistentVolumeClaim:
          claimName: pynomaly-data-pvc
      - name: logs
        emptyDir: {}

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-hpa
  namespace: pynomaly-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-app
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

#### Services and Ingress
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-service
  namespace: pynomaly-prod
  labels:
    app: pynomaly
spec:
  selector:
    app: pynomaly
    component: app
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-ingress
  namespace: pynomaly-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - pynomaly.yourdomain.com
    secretName: pynomaly-tls
  rules:
  - host: pynomaly.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-service
            port:
              number: 80
```

### Database Deployment

#### PostgreSQL Production Setup
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: pynomaly-prod
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: pynomaly
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        ports:
        - containerPort: 5432
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        livenessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - exec pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" -h 127.0.0.1 -p 5432
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - -e
            - exec pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" -h 127.0.0.1 -p 5432
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: pynomaly-prod
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

## 🔒 Security Configuration

### RBAC and Service Accounts
```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pynomaly
  namespace: pynomaly-prod

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: pynomaly-prod
  name: pynomaly-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pynomaly-rolebinding
  namespace: pynomaly-prod
subjects:
- kind: ServiceAccount
  name: pynomaly
  namespace: pynomaly-prod
roleRef:
  kind: Role
  name: pynomaly-role
  apiGroup: rbac.authorization.k8s.io
```

### Secrets Management
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
  namespace: pynomaly-prod
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  redis-url: <base64-encoded-redis-url>
  jwt-secret: <base64-encoded-jwt-secret>
  api-key: <base64-encoded-api-key>

---
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: pynomaly-prod
type: Opaque
data:
  username: <base64-encoded-username>
  password: <base64-encoded-password>
```

### Network Policies
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-network-policy
  namespace: pynomaly-prod
spec:
  podSelector:
    matchLabels:
      app: pynomaly
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
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
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
```

## 📊 Monitoring and Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/alerts.yml"

scrape_configs:
  - job_name: 'pynomaly'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - pynomaly-prod
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

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093
```

### Application Metrics
```python
# src/pynomaly/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Application metrics
request_count = Counter('pynomaly_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('pynomaly_request_duration_seconds', 'Request duration')
active_sessions = Gauge('pynomaly_active_sessions', 'Active user sessions')
model_predictions = Counter('pynomaly_predictions_total', 'Total predictions', ['model', 'result'])
model_latency = Histogram('pynomaly_model_latency_seconds', 'Model prediction latency', ['model'])
anomaly_detection_rate = Gauge('pynomaly_anomaly_rate', 'Current anomaly detection rate')

# Infrastructure metrics
database_connections = Gauge('pynomaly_db_connections', 'Database connections')
redis_connections = Gauge('pynomaly_redis_connections', 'Redis connections')
memory_usage = Gauge('pynomaly_memory_usage_bytes', 'Memory usage')
cpu_usage = Gauge('pynomaly_cpu_usage_percent', 'CPU usage')

def metrics_middleware(app):
    """Flask middleware for collecting metrics."""
    
    @app.before_request
    def before_request():
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        duration = time.time() - request.start_time
        request_count.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=response.status_code
        ).inc()
        request_duration.observe(duration)
        return response
    
    return app

def monitor_model_prediction(model_name):
    """Decorator for monitoring model predictions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                model_predictions.labels(model=model_name, result='success').inc()
                return result
            except Exception as e:
                model_predictions.labels(model=model_name, result='error').inc()
                raise
            finally:
                latency = time.time() - start_time
                model_latency.labels(model=model_name).observe(latency)
        return wrapper
    return decorator

# Start metrics server
def start_metrics_server(port=8081):
    start_http_server(port)
    print(f"Metrics server started on port {port}")
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Pynomaly Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pynomaly_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pynomaly_predictions_total[5m])",
            "legendFormat": "{{model}} - {{result}}"
          }
        ]
      },
      {
        "title": "Anomaly Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "pynomaly_anomaly_rate",
            "legendFormat": "Current Rate"
          }
        ]
      }
    ]
  }
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

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
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=pynomaly --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
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
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v1
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/pynomaly-app pynomaly=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }} -n pynomaly-staging
        kubectl rollout status deployment/pynomaly-app -n pynomaly-staging
    
    - name: Run staging tests
      run: |
        kubectl run staging-test --rm -i --restart=Never --image=pynomaly/test-runner:latest -- pytest tests/integration/

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v1
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/pynomaly-app pynomaly=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }} -n pynomaly-prod
        kubectl rollout status deployment/pynomaly-app -n pynomaly-prod
    
    - name: Verify deployment
      run: |
        kubectl get pods -n pynomaly-prod
        kubectl logs -l app=pynomaly -n pynomaly-prod --tail=100
```

## 🔧 Configuration Management

### Environment-specific Configuration
```python
# config/production.py
import os
from urllib.parse import urlparse

class ProductionConfig:
    """Production configuration."""
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 8080))
    DEBUG = False
    TESTING = False
    
    # Database settings
    DATABASE_URL = os.environ.get('PYNOMALY_DATABASE_URL')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'pool_pre_ping': True
    }
    
    # Redis settings
    REDIS_URL = os.environ.get('PYNOMALY_REDIS_URL')
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # ML settings
    MODEL_CACHE_SIZE = 10
    PREDICTION_TIMEOUT = 5.0
    BATCH_SIZE = 1000
    MAX_MODEL_SIZE = '1GB'
    
    # Monitoring settings
    METRICS_ENABLED = True
    METRICS_PORT = 8081
    HEALTH_CHECK_INTERVAL = 30
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = 'json'
    LOG_TO_FILE = True
    LOG_FILE_PATH = '/app/logs/pynomaly.log'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Storage settings
    STORAGE_BACKEND = 's3'  # or 'gcs', 'azure', 'local'
    S3_BUCKET = os.environ.get('S3_BUCKET')
    S3_REGION = os.environ.get('S3_REGION')
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS = 1000
    RATE_LIMIT_WINDOW = 60  # seconds
    
    # CORS settings
    CORS_ENABLED = True
    CORS_ORIGINS = ['https://yourdomain.com']
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required_vars = [
            'PYNOMALY_DATABASE_URL',
            'PYNOMALY_REDIS_URL',
            'SECRET_KEY',
            'JWT_SECRET_KEY'
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
```

## 🚀 Deployment Strategies

### Blue-Green Deployment
```bash
#!/bin/bash
# scripts/deploy-blue-green.sh

set -e

NAMESPACE="pynomaly-prod"
NEW_IMAGE="$1"
CURRENT_ENV=$(kubectl get service pynomaly-service -n $NAMESPACE -o jsonpath='{.spec.selector.environment}')

if [ "$CURRENT_ENV" = "blue" ]; then
    NEW_ENV="green"
    OLD_ENV="blue"
else
    NEW_ENV="blue" 
    OLD_ENV="green"
fi

echo "Deploying to $NEW_ENV environment..."

# Update deployment with new image
kubectl set image deployment/pynomaly-$NEW_ENV pynomaly=$NEW_IMAGE -n $NAMESPACE

# Wait for rollout to complete
kubectl rollout status deployment/pynomaly-$NEW_ENV -n $NAMESPACE --timeout=300s

# Run health checks
echo "Running health checks..."
kubectl run health-check --rm -i --restart=Never --image=curlimages/curl:latest -- \
  curl -f http://pynomaly-$NEW_ENV:8080/health

if [ $? -eq 0 ]; then
    echo "Health checks passed. Switching traffic to $NEW_ENV..."
    
    # Switch service to new environment
    kubectl patch service pynomaly-service -n $NAMESPACE -p '{"spec":{"selector":{"environment":"'$NEW_ENV'"}}}'
    
    echo "Traffic switched to $NEW_ENV. Deployment complete!"
    
    # Optional: Scale down old environment after verification period
    sleep 300  # Wait 5 minutes
    kubectl scale deployment/pynomaly-$OLD_ENV --replicas=0 -n $NAMESPACE
    
else
    echo "Health checks failed. Rolling back..."
    kubectl rollout undo deployment/pynomaly-$NEW_ENV -n $NAMESPACE
    exit 1
fi
```

### Canary Deployment
```bash
#!/bin/bash
# scripts/deploy-canary.sh

set -e

NAMESPACE="pynomaly-prod"
NEW_IMAGE="$1"
CANARY_PERCENTAGE=${2:-10}  # Default 10% traffic

echo "Starting canary deployment with $CANARY_PERCENTAGE% traffic..."

# Deploy canary version
kubectl set image deployment/pynomaly-canary pynomaly=$NEW_IMAGE -n $NAMESPACE
kubectl rollout status deployment/pynomaly-canary -n $NAMESPACE --timeout=300s

# Update traffic split
kubectl patch virtualservice pynomaly-vs -n $NAMESPACE --type='merge' -p='{
  "spec": {
    "http": [{
      "match": [{"uri": {"prefix": "/"}}],
      "route": [
        {"destination": {"host": "pynomaly-stable"}, "weight": '$((100-CANARY_PERCENTAGE))'},
        {"destination": {"host": "pynomaly-canary"}, "weight": '$CANARY_PERCENTAGE'}
      ]
    }]
  }
}'

echo "Canary deployment active with $CANARY_PERCENTAGE% traffic"

# Monitor metrics for 10 minutes
echo "Monitoring canary metrics..."
sleep 600

# Check error rate and latency
ERROR_RATE=$(kubectl exec -n monitoring deployment/prometheus -- promtool query instant \
  'rate(pynomaly_requests_total{status!~"2.."}[5m]) / rate(pynomaly_requests_total[5m])' | \
  grep -o '[0-9]*\.[0-9]*' | head -1)

if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
    echo "Error rate too high ($ERROR_RATE). Rolling back canary..."
    kubectl patch virtualservice pynomaly-vs -n $NAMESPACE --type='merge' -p='{
      "spec": {
        "http": [{
          "match": [{"uri": {"prefix": "/"}}],
          "route": [{"destination": {"host": "pynomaly-stable"}, "weight": 100}]
        }]
      }
    }'
    exit 1
else
    echo "Canary metrics look good. Promoting to stable..."
    kubectl set image deployment/pynomaly-stable pynomaly=$NEW_IMAGE -n $NAMESPACE
    kubectl rollout status deployment/pynomaly-stable -n $NAMESPACE --timeout=300s
    
    # Restore 100% traffic to stable
    kubectl patch virtualservice pynomaly-vs -n $NAMESPACE --type='merge' -p='{
      "spec": {
        "http": [{
          "match": [{"uri": {"prefix": "/"}}],
          "route": [{"destination": {"host": "pynomaly-stable"}, "weight": 100}]
        }]
      }
    }'
    
    echo "Canary promotion complete!"
fi
```

## 📋 Operational Procedures

### Backup and Recovery
```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
NAMESPACE="pynomaly-prod"
S3_BUCKET="pynomaly-backups"

echo "Starting backup process..."

# Database backup
kubectl exec -n $NAMESPACE postgres-0 -- pg_dump -U pynomaly pynomaly | \
  gzip > "db_backup_$BACKUP_DATE.sql.gz"
aws s3 cp "db_backup_$BACKUP_DATE.sql.gz" "s3://$S3_BUCKET/database/"

# Model artifacts backup
kubectl exec -n $NAMESPACE deployment/pynomaly-app -- tar czf - /app/data/models | \
  aws s3 cp - "s3://$S3_BUCKET/models/models_backup_$BACKUP_DATE.tar.gz"

# Configuration backup
kubectl get configmaps,secrets -n $NAMESPACE -o yaml > "config_backup_$BACKUP_DATE.yaml"
aws s3 cp "config_backup_$BACKUP_DATE.yaml" "s3://$S3_BUCKET/config/"

echo "Backup completed: $BACKUP_DATE"
```

### Disaster Recovery
```bash
#!/bin/bash
# scripts/disaster-recovery.sh

BACKUP_DATE="$1"
NAMESPACE="pynomaly-prod"
S3_BUCKET="pynomaly-backups"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

echo "Starting disaster recovery from backup: $BACKUP_DATE"

# Restore database
aws s3 cp "s3://$S3_BUCKET/database/db_backup_$BACKUP_DATE.sql.gz" - | \
  gunzip | kubectl exec -i -n $NAMESPACE postgres-0 -- psql -U pynomaly -d pynomaly

# Restore model artifacts
aws s3 cp "s3://$S3_BUCKET/models/models_backup_$BACKUP_DATE.tar.gz" - | \
  kubectl exec -i -n $NAMESPACE deployment/pynomaly-app -- tar xzf - -C /

# Restore configuration
aws s3 cp "s3://$S3_BUCKET/config/config_backup_$BACKUP_DATE.yaml" - | \
  kubectl apply -f -

# Restart applications
kubectl rollout restart deployment/pynomaly-app -n $NAMESPACE

echo "Disaster recovery completed"
```

## 🎯 Performance Optimization

### Resource Optimization
```yaml
# k8s/performance-tuning.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-performance-config
  namespace: pynomaly-prod
data:
  nginx.conf: |
    worker_processes auto;
    worker_rlimit_nofile 65535;
    
    events {
        worker_connections 4096;
        use epoll;
        multi_accept on;
    }
    
    http {
        sendfile on;
        tcp_nopush on;
        tcp_nodelay on;
        keepalive_timeout 30;
        keepalive_requests 100;
        
        gzip on;
        gzip_comp_level 6;
        gzip_types text/plain text/css application/json application/javascript;
        
        upstream pynomaly_backend {
            least_conn;
            server pynomaly-service:80 max_fails=3 fail_timeout=30s;
        }
        
        server {
            listen 80;
            
            location / {
                proxy_pass http://pynomaly_backend;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_connect_timeout 30s;
                proxy_send_timeout 30s;
                proxy_read_timeout 30s;
            }
        }
    }
```

### Application Performance Tuning
```python
# src/pynomaly/config/performance.py
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Server configuration
    WORKERS = multiprocessing.cpu_count() * 2 + 1
    WORKER_CONNECTIONS = 1000
    MAX_REQUESTS = 10000
    MAX_REQUESTS_JITTER = 1000
    PRELOAD_APP = True
    
    # Database connection pooling
    DB_POOL_SIZE = 20
    DB_MAX_OVERFLOW = 10
    DB_POOL_TIMEOUT = 30
    DB_POOL_RECYCLE = 3600
    
    # Redis connection pooling
    REDIS_MAX_CONNECTIONS = 100
    REDIS_RETRY_ON_TIMEOUT = True
    
    # Async configuration
    THREAD_POOL_SIZE = 50
    ASYNC_TIMEOUT = 30
    
    # Caching configuration
    CACHE_TTL = 300  # 5 minutes
    MODEL_CACHE_SIZE = 10
    PREDICTION_CACHE_SIZE = 10000
    
    # Batch processing
    BATCH_SIZE = 1000
    BATCH_TIMEOUT = 5.0
    
    @classmethod
    def setup_async_pool(cls):
        """Setup async thread pool."""
        return ThreadPoolExecutor(max_workers=cls.THREAD_POOL_SIZE)
    
    @classmethod
    def setup_event_loop(cls):
        """Configure asyncio event loop."""
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
```

## 🎯 Next Steps

### Production Readiness Checklist
- [ ] Load testing completed with expected traffic
- [ ] Security scanning and penetration testing
- [ ] Disaster recovery procedures tested
- [ ] Monitoring and alerting configured
- [ ] Performance benchmarks established
- [ ] Documentation updated
- [ ] Team training completed
- [ ] On-call procedures established

### Scaling Considerations
1. **Horizontal Scaling**: Configure HPA for dynamic scaling
2. **Database Scaling**: Consider read replicas and connection pooling
3. **Cache Optimization**: Implement distributed caching strategies
4. **CDN Integration**: Use CDN for static assets and model artifacts
5. **Multi-region Deployment**: Deploy across multiple regions for HA

This production deployment guide provides a comprehensive foundation for deploying Pynomaly at enterprise scale with proper security, monitoring, and operational procedures.