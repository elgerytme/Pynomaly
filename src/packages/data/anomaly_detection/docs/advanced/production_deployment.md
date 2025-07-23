# Production Deployment Guide

This comprehensive guide covers deploying the anomaly detection system in production environments with enterprise-grade reliability, security, and scalability.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Container Deployment](#container-deployment)
4. [Kubernetes Orchestration](#kubernetes-orchestration)
5. [Load Balancing & High Availability](#load-balancing--high-availability)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security Configuration](#security-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Disaster Recovery](#disaster-recovery)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

### Production Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end
    
    subgraph "Application Tier"
        API1[API Server 1]
        API2[API Server 2]
        API3[API Server 3]
        WEB1[Web Server 1]
        WEB2[Web Server 2]
    end
    
    subgraph "Processing Tier"
        WORKER1[Worker 1]
        WORKER2[Worker 2]
        WORKER3[Worker 3]
        STREAM[Streaming Service]
    end
    
    subgraph "Data Tier"
        REDIS[(Redis Cache)]
        POSTGRES[(PostgreSQL)]
        MODELS[(Model Storage)]
    end
    
    subgraph "Message Queue"
        KAFKA[Apache Kafka]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        LOGS[ELK Stack]
    end
    
    LB --> API1
    LB --> API2  
    LB --> API3
    LB --> WEB1
    LB --> WEB2
    
    API1 --> WORKER1
    API2 --> WORKER2
    API3 --> WORKER3
    
    WORKER1 --> REDIS
    WORKER2 --> REDIS
    WORKER3 --> REDIS
    
    API1 --> POSTGRES
    API2 --> POSTGRES
    API3 --> POSTGRES
    
    STREAM --> KAFKA
    WORKER1 --> KAFKA
    
    API1 --> MODELS
    API2 --> MODELS
    API3 --> MODELS
    
    PROM --> API1
    PROM --> API2
    PROM --> API3
    GRAF --> PROM
```

### Deployment Patterns

#### 1. Microservices Architecture
```yaml
services:
  - anomaly-api: REST API service
  - anomaly-web: Web dashboard service  
  - anomaly-worker: Background processing
  - anomaly-stream: Real-time processing
  - anomaly-scheduler: Batch job scheduling
```

#### 2. Service Mesh Integration
```yaml
istio:
  enabled: true
  features:
    - traffic_management
    - security_policies
    - observability
    - fault_injection
```

## Infrastructure Requirements

### Minimum Requirements

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| API Server | 2 cores | 4GB RAM | 20GB SSD | 1Gbps |
| Worker Node | 4 cores | 8GB RAM | 50GB SSD | 1Gbps |
| Database | 4 cores | 16GB RAM | 200GB SSD | 1Gbps |
| Redis Cache | 2 cores | 8GB RAM | 20GB SSD | 1Gbps |

### Recommended Production Setup

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| API Server | 8 cores | 16GB RAM | 100GB SSD | 10Gbps |
| Worker Node | 16 cores | 32GB RAM | 500GB SSD | 10Gbps |
| Database | 16 cores | 64GB RAM | 1TB NVMe | 10Gbps |
| Redis Cache | 8 cores | 32GB RAM | 100GB SSD | 10Gbps |

### Cloud Provider Recommendations

#### AWS Configuration
```yaml
# EC2 Instance Types
api_servers:
  instance_type: "c5.2xlarge"  # 8 vCPU, 16GB RAM
  count: 3
  
worker_nodes:
  instance_type: "c5.4xlarge"  # 16 vCPU, 32GB RAM
  count: 3
  
database:
  service: "RDS PostgreSQL"
  instance_type: "db.r5.4xlarge"
  multi_az: true
  
cache:
  service: "ElastiCache Redis"
  node_type: "cache.r5.2xlarge"
  
storage:
  models: "S3 Bucket"
  logs: "CloudWatch Logs"
```

#### Azure Configuration
```yaml
# Virtual Machine Sizes
api_servers:
  vm_size: "Standard_D8s_v3"
  count: 3
  
worker_nodes:
  vm_size: "Standard_D16s_v3"
  count: 3
  
database:
  service: "Azure Database for PostgreSQL"
  tier: "General Purpose"
  compute_generation: "Gen5"
  
cache:
  service: "Azure Cache for Redis"
  tier: "Premium"
  
storage:
  models: "Azure Blob Storage"
  logs: "Azure Monitor Logs"
```

#### GCP Configuration
```yaml
# Compute Engine Machine Types
api_servers:
  machine_type: "n1-standard-8"
  count: 3
  
worker_nodes:
  machine_type: "n1-standard-16"
  count: 3
  
database:
  service: "Cloud SQL PostgreSQL"
  tier: "db-custom-16-65536"
  
cache:
  service: "Memorystore Redis"
  tier: "standard"
  
storage:
  models: "Cloud Storage"
  logs: "Cloud Logging"
```

## Container Deployment

### Docker Configuration

#### Production Dockerfile
```dockerfile
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Set security constraints
USER appuser
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "anomaly_detection.server:app"]
```

#### Multi-Stage Build for Different Services
```dockerfile
# API Service
FROM anomaly-detection-base:latest as api-service
COPY api_requirements.txt .
RUN pip install -r api_requirements.txt
CMD ["python", "-m", "anomaly_detection.server"]

# Worker Service  
FROM anomaly-detection-base:latest as worker-service
COPY worker_requirements.txt .
RUN pip install -r worker_requirements.txt
CMD ["python", "-m", "anomaly_detection.worker"]

# Streaming Service
FROM anomaly-detection-base:latest as streaming-service
COPY streaming_requirements.txt .
RUN pip install -r streaming_requirements.txt
CMD ["python", "-m", "anomaly_detection.streaming"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  anomaly-api:
    build:
      context: .
      target: api-service
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/anomaly_db
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=development
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    
  anomaly-worker:
    build:
      context: .
      target: worker-service
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/anomaly_db
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - redis
      - kafka
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    
  anomaly-stream:
    build:
      context: .
      target: streaming-service
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - kafka
      - redis
    restart: unless-stopped
    
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=anomaly_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    restart: unless-stopped
    
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Orchestration

### Namespace Configuration
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: anomaly-detection
  labels:
    name: anomaly-detection
    environment: production
```

### ConfigMap for Application Settings
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: anomaly-config
  namespace: anomaly-detection
data:
  DATABASE_HOST: "postgres-service"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "anomaly_db"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  KAFKA_BOOTSTRAP_SERVERS: "kafka-service:9092"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  WORKERS_COUNT: "4"
  MAX_MEMORY_MB: "2048"
```

### Secret Management
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: anomaly-secrets
  namespace: anomaly-detection
type: Opaque
data:
  DATABASE_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  API_SECRET_KEY: <base64-encoded-secret>
  JWT_SECRET: <base64-encoded-jwt-secret>
```

### API Service Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-api
  namespace: anomaly-detection
  labels:
    app: anomaly-api
    tier: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-api
  template:
    metadata:
      labels:
        app: anomaly-api
        tier: api
    spec:
      containers:
      - name: anomaly-api
        image: anomaly-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: anomaly-secrets
              key: DATABASE_PASSWORD
        envFrom:
        - configMapRef:
            name: anomaly-config
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      nodeSelector:
        workload: api
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - anomaly-api
              topologyKey: kubernetes.io/hostname
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: anomaly-api-service
  namespace: anomaly-detection
  labels:
    app: anomaly-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: anomaly-api
---
apiVersion: v1
kind: Service
metadata:
  name: anomaly-api-nodeport
  namespace: anomaly-detection
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 8000
    nodePort: 30080
  selector:
    app: anomaly-api
```

### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: anomaly-ingress
  namespace: anomaly-detection
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.anomaly-detection.com
    secretName: anomaly-tls
  rules:
  - host: api.anomaly-detection.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: anomaly-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-api-hpa
  namespace: anomaly-detection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-api
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
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Persistent Volume for Model Storage
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: anomaly-detection
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

## Load Balancing & High Availability

### NGINX Configuration
```nginx
upstream anomaly_api {
    least_conn;
    server anomaly-api-1:8000 max_fails=3 fail_timeout=30s;
    server anomaly-api-2:8000 max_fails=3 fail_timeout=30s;
    server anomaly-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.anomaly-detection.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/anomaly-detection.crt;
    ssl_certificate_key /etc/ssl/private/anomaly-detection.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req zone=api_limit burst=20 nodelay;
    
    # Compression
    gzip on;
    gzip_vary on;
    gzip_types text/plain application/json application/javascript text/css;
    
    location / {
        proxy_pass http://anomaly_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health checks
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
    }
    
    location /health {
        proxy_pass http://anomaly_api;
        access_log off;
    }
}
```

### HAProxy Configuration
```haproxy
global
    daemon
    log stdout local0
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option log-health-checks
    option forwardfor
    option http-server-close
    timeout connect 5000
    timeout client 50000
    timeout server 50000
    errorfile 400 /etc/haproxy/errors/400.http
    errorfile 403 /etc/haproxy/errors/403.http
    errorfile 408 /etc/haproxy/errors/408.http
    errorfile 500 /etc/haproxy/errors/500.http
    errorfile 502 /etc/haproxy/errors/502.http
    errorfile 503 /etc/haproxy/errors/503.http
    errorfile 504 /etc/haproxy/errors/504.http

frontend anomaly_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/anomaly-detection.pem
    redirect scheme https code 301 if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 100 }
    
    default_backend anomaly_api

backend anomaly_api
    balance roundrobin
    option httpchk GET /health
    server api1 anomaly-api-1:8000 check inter 5s fall 3 rise 2
    server api2 anomaly-api-2:8000 check inter 5s fall 3 rise 2
    server api3 anomaly-api-3:8000 check inter 5s fall 3 rise 2

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats hide-version
```

## Monitoring & Observability

### Prometheus Configuration
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'anomaly-api'
    static_configs:
      - targets: ['anomaly-api-1:8000', 'anomaly-api-2:8000', 'anomaly-api-3:8000']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'anomaly-worker'
    static_configs:
      - targets: ['anomaly-worker-1:9090', 'anomaly-worker-2:9090']
    metrics_path: /metrics
    scrape_interval: 15s

rule_files:
  - "anomaly_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Custom Metrics Implementation
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics definitions
DETECTION_REQUESTS = Counter(
    'anomaly_detection_requests_total',
    'Total number of detection requests',
    ['algorithm', 'status']
)

DETECTION_DURATION = Histogram(
    'anomaly_detection_duration_seconds',
    'Time spent on anomaly detection',
    ['algorithm'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

ACTIVE_MODELS = Gauge(
    'anomaly_models_active_total',
    'Number of active models in memory'
)

ANOMALIES_DETECTED = Counter(
    'anomalies_detected_total',
    'Total number of anomalies detected',
    ['algorithm', 'severity']
)

# Usage in detection service
class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
        
        # Start metrics server
        start_http_server(9090)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/api/v1/detect":
            start_time = time.time()
            
            try:
                await self.app(scope, receive, send)
                DETECTION_REQUESTS.labels(algorithm="unknown", status="success").inc()
            except Exception as e:
                DETECTION_REQUESTS.labels(algorithm="unknown", status="error").inc()
                raise
            finally:
                DETECTION_DURATION.labels(algorithm="unknown").observe(time.time() - start_time)
        else:
            await self.app(scope, receive, send)
```

### Alerting Rules
```yaml
groups:
  - name: anomaly_detection_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(anomaly_detection_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in anomaly detection API"
          description: "Error rate is {{ $value }} errors per second"
          
      - alert: SlowDetectionRequests
        expr: histogram_quantile(0.95, rate(anomaly_detection_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow anomaly detection requests"
          description: "95th percentile latency is {{ $value }} seconds"
          
      - alert: ServiceDown
        expr: up{job="anomaly-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Anomaly detection service is down"
          description: "Service {{ $labels.instance }} has been down for more than 1 minute"
          
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes{job="anomaly-api"} / 1024 / 1024 / 1024 > 4
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB on {{ $labels.instance }}"
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Anomaly Detection Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(anomaly_detection_requests_total[5m])",
            "legendFormat": "{{ algorithm }} - {{ status }}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(anomaly_detection_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(anomaly_detection_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Active Models",
        "type": "singlestat",
        "targets": [
          {
            "expr": "anomaly_models_active_total",
            "legendFormat": "Models"
          }
        ]
      },
      {
        "title": "Anomalies Detected",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(anomalies_detected_total[5m])",
            "legendFormat": "{{ algorithm }} - {{ severity }}"
          }
        ]
      }
    ]
  }
}
```

This production deployment guide provides comprehensive coverage of deploying the anomaly detection system in enterprise environments with proper monitoring, security, and scalability considerations.