# Production Deployment Guide

This comprehensive guide covers deploying Pynomaly in production environments with Docker, Kubernetes, and cloud providers.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Provider Deployments](#cloud-provider-deployments)
6. [Security Hardening](#security-hardening)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Backup and Recovery](#backup-and-recovery)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [Production Checklist](#production-checklist)

## Prerequisites

### System Requirements

**Minimum Requirements:**

- CPU: 4 cores
- RAM: 8GB
- Storage: 100GB SSD
- Network: 1Gbps

**Recommended for Production:**

- CPU: 8 cores
- RAM: 16GB  
- Storage: 500GB SSD
- Network: 10Gbps

### Software Dependencies

- Docker 24.0+
- Docker Compose 2.20+
- Kubernetes 1.27+ (for K8s deployment)
- PostgreSQL 15+ or Redis 6.2+
- Nginx or equivalent reverse proxy

### Security Requirements

- SSL/TLS certificates
- Firewall configuration
- VPN access (recommended)
- Identity management system

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Gateway   │    │   API Services  │
│     (Nginx)     │────│   (FastAPI)     │────│   (Workers)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │    Database     │    │      Cache      │
                       │  (PostgreSQL)   │    │     (Redis)     │
                       └─────────────────┘    └─────────────────┘
```

### Component Responsibilities

- **Load Balancer**: SSL termination, traffic distribution
- **Web Gateway**: Static files, API routing, authentication
- **API Services**: Core anomaly detection logic
- **Database**: Persistent data storage
- **Cache**: Session data, temporary results

## Docker Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: pynomaly
      POSTGRES_USER: pynomaly
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    secrets:
      - postgres_password
    networks:
      - backend
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --requirepass-file /run/secrets/redis_password
    volumes:
      - redis_data:/data
    secrets:
      - redis_password
    networks:
      - backend
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  pynomaly-api:
    image: pynomaly:latest
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://pynomaly@postgres:5432/pynomaly
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY_FILE=/run/secrets/app_secret_key
    secrets:
      - app_secret_key
      - postgres_password
      - redis_password
    networks:
      - backend
      - frontend
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '2'
        reservations:
          memory: 1G
          cpus: '1'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - static_files:/var/www/static:ro
    networks:
      - frontend
    depends_on:
      - pynomaly-api
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - monitoring
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - monitoring
      - frontend
    secrets:
      - grafana_password

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  static_files:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
  monitoring:
    driver: bridge

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
  app_secret_key:
    file: ./secrets/app_secret_key.txt
  grafana_password:
    file: ./secrets/grafana_password.txt
```

### Secrets Management

Create secret files:

```bash
# Create secrets directory
mkdir -p secrets

# Generate secure passwords
openssl rand -base64 32 > secrets/postgres_password.txt
openssl rand -base64 32 > secrets/redis_password.txt
openssl rand -base64 64 > secrets/app_secret_key.txt
openssl rand -base64 32 > secrets/grafana_password.txt

# Secure the secrets
chmod 600 secrets/*.txt
```

### Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream pynomaly_backend {
        least_conn;
        server pynomaly-api:8000 max_fails=3 fail_timeout=30s;
    }

    # Security headers
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Authentication endpoints with stricter rate limiting
        location /auth/ {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://pynomaly_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        # Main application
        location / {
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Deployment Commands

```bash
# Build production image
docker build -t pynomaly:latest -f Dockerfile.prod .

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale API services
docker-compose -f docker-compose.prod.yml up -d --scale pynomaly-api=5

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Health check
curl -f http://localhost/health || echo "Health check failed"
```

## Kubernetes Deployment

### Namespace and ConfigMap

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly-prod
  labels:
    name: pynomaly-prod
    environment: production
```

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
  namespace: pynomaly-prod
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  WORKER_COUNT: "4"
  DATABASE_URL: "postgresql://pynomaly@postgres:5432/pynomaly"
  REDIS_URL: "redis://redis:6379"
```

### Secrets

Create `k8s/secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
  namespace: pynomaly-prod
type: Opaque
data:
  postgres-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  app-secret-key: <base64-encoded-secret>
```

### PostgreSQL Deployment

Create `k8s/postgres.yaml`:

```yaml
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
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: pynomaly
        - name: POSTGRES_USER
          value: pynomaly
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
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
  clusterIP: None
```

### Application Deployment

Create `k8s/pynomaly-app.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  namespace: pynomaly-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
    spec:
      containers:
      - name: pynomaly-api
        image: pynomaly:latest
        envFrom:
        - configMapRef:
            name: pynomaly-config
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: app-secret-key
        ports:
        - containerPort: 8000
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
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"

---
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-api-service
  namespace: pynomaly-prod
spec:
  selector:
    app: pynomaly-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Controller

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-ingress
  namespace: pynomaly-prod
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: pynomaly-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

Create `k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-hpa
  namespace: pynomaly-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
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

### Deployment Commands

```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n pynomaly-prod

# View logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-prod

# Scale manually
kubectl scale deployment pynomaly-api --replicas=5 -n pynomaly-prod

# Rolling update
kubectl set image deployment/pynomaly-api pynomaly-api=pynomaly:v2.0 -n pynomaly-prod
```

## Cloud Provider Deployments

### AWS EKS Deployment

#### Prerequisites

```bash
# Install AWS CLI and eksctl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

#### Create EKS Cluster

```yaml
# eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: pynomaly-prod
  region: us-west-2
  version: "1.27"

vpc:
  nat:
    gateway: Single

nodeGroups:
  - name: pynomaly-workers
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 8
    volumeSize: 100
    volumeType: gp3
    ssh:
      enableSsm: true
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
        alb: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enable: true
    logTypes: ["api", "audit", "authenticator", "controllerManager", "scheduler"]
```

```bash
# Create cluster
eksctl create cluster -f eks-cluster.yaml

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=pynomaly-prod \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

#### RDS Database

```yaml
# terraform/rds.tf
resource "aws_db_instance" "pynomaly" {
  identifier = "pynomaly-prod"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r5.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "pynomaly"
  username = "pynomaly"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.pynomaly.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  deletion_protection = true
  skip_final_snapshot = false
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = {
    Name        = "pynomaly-prod"
    Environment = "production"
  }
}
```

### Azure AKS Deployment

```bash
# Create resource group
az group create --name pynomaly-prod --location eastus

# Create AKS cluster
az aks create \
  --resource-group pynomaly-prod \
  --name pynomaly-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group pynomaly-prod --name pynomaly-cluster
```

### Google GKE Deployment

```bash
# Create cluster
gcloud container clusters create pynomaly-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 8 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-cloud-logging \
  --enable-cloud-monitoring

# Get credentials
gcloud container clusters get-credentials pynomaly-prod --zone us-central1-a
```

## Security Hardening

### Container Security

#### Dockerfile Security Best Practices

```dockerfile
# Use specific version tags
FROM python:3.11-slim-bullseye

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R pynomaly:pynomaly /app

# Switch to non-root user
USER pynomaly

# Use tini for proper signal handling
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Network Security

#### Firewall Rules

```bash
# Allow only necessary ports
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable
```

#### VPN Configuration

```bash
# Install WireGuard VPN
apt install wireguard

# Generate keys
wg genkey | tee privatekey | wg pubkey > publickey

# Configure VPN server
cat > /etc/wireguard/wg0.conf << EOF
[Interface]
PrivateKey = $(cat privatekey)
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = CLIENT_PUBLIC_KEY
AllowedIPs = 10.0.0.2/32
EOF

# Start VPN
systemctl enable wg-quick@wg0
systemctl start wg-quick@wg0
```

### Authentication and Authorization

#### OAuth2 Configuration

```python
# config/auth.py
from fastapi_users.authentication import JWTAuthentication

jwt_authentication = JWTAuthentication(
    secret=settings.SECRET_KEY,
    lifetime_seconds=3600,
    tokenUrl="auth/jwt/login",
)

# Multi-factor authentication
from pyotp import TOTP

def verify_totp(user_secret: str, token: str) -> bool:
    totp = TOTP(user_secret)
    return totp.verify(token)
```

### Data Protection

#### Encryption at Rest

```yaml
# k8s/storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: encrypted-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  encrypted: "true"
  kmsKeyId: "alias/pynomaly-key"
```

#### Secrets Management

```bash
# Use HashiCorp Vault
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault

# Configure Vault with Kubernetes auth
vault auth enable kubernetes
vault write auth/kubernetes/config \
    token_reviewer_jwt="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)" \
    kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443" \
    kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
```

## Monitoring and Observability

### Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['pynomaly-api:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: pynomaly.rules
    rules:
      - alert: HighCPUUsage
        expr: (100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: APIHighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "API error rate is above 10% for more than 2 minutes"

      - alert: DatabaseConnections
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connection usage"
          description: "Database connections are above 80% of max connections"
```

### Grafana Dashboards

Create `monitoring/grafana/dashboards/pynomaly-overview.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "Pynomaly Overview",
    "tags": ["pynomaly"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Active Anomaly Detection Jobs",
        "type": "stat",
        "targets": [
          {
            "expr": "pynomaly_active_detection_jobs",
            "legendFormat": "Active Jobs"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Application Metrics

```python
# Add to your FastAPI application
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_JOBS = Gauge('pynomaly_active_detection_jobs', 'Number of active detection jobs')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## Backup and Recovery

### Database Backups

#### Automated Backup Script

Create `scripts/backup-database.sh`:

```bash
#!/bin/bash

set -e

# Configuration
BACKUP_DIR="/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="pynomaly_backup_${TIMESTAMP}.sql"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Perform backup
echo "Starting database backup..."
pg_dump -h postgres -U pynomaly -d pynomaly > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"
echo "Backup completed: ${BACKUP_FILE}.gz"

# Upload to S3 (optional)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" "s3://${AWS_S3_BUCKET}/backups/"
    echo "Backup uploaded to S3"
fi

# Clean up old backups
find $BACKUP_DIR -name "pynomaly_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
echo "Old backups cleaned up"
```

#### Kubernetes CronJob for Backups

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: pynomaly-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U pynomaly -d pynomaly | gzip > /backups/pynomaly_$(date +%Y%m%d_%H%M%S).sql.gz
              find /backups -name "pynomaly_*.sql.gz" -mtime +30 -delete
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: pynomaly-secrets
                  key: postgres-password
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Disaster Recovery

#### Recovery Procedures

```bash
# 1. Restore from backup
gunzip -c pynomaly_backup_20240101_020000.sql.gz | psql -h postgres -U pynomaly -d pynomaly

# 2. Verify data integrity
psql -h postgres -U pynomaly -d pynomaly -c "SELECT COUNT(*) FROM datasets;"

# 3. Restart services
kubectl rollout restart deployment/pynomaly-api -n pynomaly-prod

# 4. Run health checks
kubectl get pods -n pynomaly-prod
curl -f https://your-domain.com/health
```

#### Point-in-Time Recovery

```sql
-- For PostgreSQL with WAL-E or pgBackRest
SELECT pg_start_backup('disaster_recovery');
-- Copy data files
SELECT pg_stop_backup();

-- Restore to specific time
restore_command = 'cp /backups/wal/%f %p'
recovery_target_time = '2024-01-01 12:00:00'
```

## Performance Tuning

### Application Tuning

#### FastAPI Optimization

```python
# config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Worker configuration
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    max_requests: int = 1000
    max_requests_jitter: int = 100
    
    # Connection pooling
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    
    # Caching
    redis_pool_size: int = 10
    cache_ttl: int = 3600

# Database connection pool
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True
)
```

#### Caching Strategy

```python
# Implement Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            
            if cached:
                return pickle.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, pickle.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(ttl=1800)
async def get_anomaly_results(dataset_id: str):
    # Expensive computation here
    pass
```

### Database Tuning

#### PostgreSQL Configuration

```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

-- Connection limits
max_connections = 100
```

#### Database Indexing

```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX CONCURRENTLY idx_anomaly_results_dataset_id ON anomaly_results(dataset_id);
CREATE INDEX CONCURRENTLY idx_anomaly_results_timestamp ON anomaly_results(timestamp);

-- Composite indexes
CREATE INDEX CONCURRENTLY idx_anomaly_results_dataset_timestamp 
ON anomaly_results(dataset_id, timestamp DESC);

-- Partial indexes
CREATE INDEX CONCURRENTLY idx_active_jobs 
ON detection_jobs(status) WHERE status = 'running';
```

### Infrastructure Tuning

#### Kubernetes Resource Limits

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2"

# Quality of Service
priorityClassName: high-priority

# Node affinity
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: instance-type
          operator: In
          values: ["compute-optimized"]
```

#### Load Balancer Configuration

```nginx
# Nginx performance tuning
worker_processes auto;
worker_connections 1024;
worker_rlimit_nofile 2048;

# Keepalive connections
upstream pynomaly_backend {
    keepalive 32;
    server pynomaly-api:8000 max_fails=3 fail_timeout=30s;
}

# Enable compression
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript;

# Buffer sizes
client_body_buffer_size 128k;
client_max_body_size 10m;
proxy_buffers 4 256k;
proxy_buffer_size 128k;
```

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n pynomaly-prod

# Investigate memory leaks
kubectl exec -it deployment/pynomaly-api -n pynomaly-prod -- \
  python -c "
import psutil
import gc
print(f'Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB')
print(f'GC objects: {len(gc.get_objects())}')
"

# Check for memory-intensive queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;
```

#### Database Connection Issues

```bash
# Check connection pool status
kubectl exec -it deployment/pynomaly-api -n pynomaly-prod -- \
  python -c "
from sqlalchemy import create_engine
engine = create_engine('$DATABASE_URL')
print(f'Pool size: {engine.pool.size()}')
print(f'Checked out: {engine.pool.checkedout()}')
"

# Monitor active connections
SELECT count(*) as active_connections, state 
FROM pg_stat_activity 
WHERE datname = 'pynomaly' 
GROUP BY state;
```

#### Performance Issues

```bash
# Check API response times
kubectl logs -f deployment/pynomaly-api -n pynomaly-prod | \
  grep "duration" | \
  awk '{print $4}' | \
  sort -n | \
  tail -10

# Database query performance
SELECT query, total_exec_time, calls, mean_exec_time
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat%'
ORDER BY total_exec_time DESC
LIMIT 10;
```

### Debugging Tools

#### Application Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add request tracing
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

#### Container Debugging

```bash
# Debug container startup issues
kubectl describe pod <pod-name> -n pynomaly-prod

# Access container shell
kubectl exec -it <pod-name> -n pynomaly-prod -- /bin/bash

# Check container logs
kubectl logs <pod-name> -n pynomaly-prod --previous

# Debug networking
kubectl exec -it <pod-name> -n pynomaly-prod -- \
  nslookup postgres
```

## Production Checklist

### Pre-Deployment

- [ ] Security scan completed
- [ ] Load testing performed
- [ ] Database migrations tested
- [ ] Backup procedures verified
- [ ] Monitoring configured
- [ ] SSL certificates installed
- [ ] DNS records configured
- [ ] Firewall rules applied
- [ ] Secrets management configured
- [ ] Log aggregation setup

### Deployment

- [ ] Blue-green deployment strategy planned
- [ ] Rollback procedures documented
- [ ] Health checks configured
- [ ] Service discovery working
- [ ] Load balancer configured
- [ ] Auto-scaling policies set
- [ ] Resource limits defined
- [ ] Network policies applied
- [ ] Ingress controller configured
- [ ] TLS termination working

### Post-Deployment

- [ ] Application health verified
- [ ] API endpoints responding
- [ ] Database connectivity confirmed
- [ ] Cache layer functional
- [ ] Monitoring alerts active
- [ ] Log aggregation working
- [ ] Backup jobs scheduled
- [ ] Performance baseline established
- [ ] Security scanning scheduled
- [ ] Documentation updated

### Ongoing Operations

- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Capacity planning
- [ ] Disaster recovery testing
- [ ] Backup verification
- [ ] Cost optimization
- [ ] Team training completed
- [ ] Runbook documentation
- [ ] Incident response procedures
- [ ] Change management process

---

This production deployment guide provides comprehensive coverage of deploying Pynomaly in production environments. Follow the checklists and adapt the configurations to your specific infrastructure requirements.
