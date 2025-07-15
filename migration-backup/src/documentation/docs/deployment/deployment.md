# Advanced Deployment Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📄 Deployment

---


This comprehensive guide covers deploying Pynomaly in production environments, from simple single-server deployments to enterprise-grade scalable cloud architectures with advanced resilience patterns.

## Overview

Pynomaly supports multiple deployment patterns with built-in resilience:
- **Development**: Local development with hot reloading
- **Single Server**: Traditional server deployment with monitoring
- **Container**: Docker-based deployment with health checks
- **Kubernetes**: Cloud-native scalable deployment with auto-scaling
- **Serverless**: Function-as-a-Service deployment
- **Multi-Region**: Global deployment with disaster recovery
- **Hybrid Cloud**: On-premises and cloud integration
- **Edge Computing**: Distributed anomaly detection at the edge

## Quick Production Deployment

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/yourorg/pynomaly.git
cd pynomaly

# Build and run with Docker Compose
docker-compose up -d

# Access API at http://localhost:8000
# Access Web UI at http://localhost:8000/ui
```

### Option 2: Direct Installation
```bash
# Install dependencies
pip install pynomaly[production]

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/pynomaly"
export REDIS_URL="redis://localhost:6379"

# Run production server
uvicorn pynomaly.presentation.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

## Environment Configuration

### Required Environment Variables

```bash
# Database Configuration
DATABASE_URL="postgresql://username:password@host:port/database"
REDIS_URL="redis://host:port/db"

# Security
SECRET_KEY="your-secret-key-here"
JWT_SECRET_KEY="your-jwt-secret-here"
API_KEYS_ENCRYPTION_KEY="your-encryption-key"

# Application Settings
ENVIRONMENT="production"
LOG_LEVEL="INFO"
DEBUG="false"

# External Services (Optional)
PROMETHEUS_PUSHGATEWAY_URL="http://prometheus:9091"
JAEGER_AGENT_HOST="jaeger"
SENTRY_DSN="your-sentry-dsn"
```

### Configuration File (.env)
```bash
# .env file for production
DATABASE_URL=postgresql://pynomaly:secret@db:5432/pynomaly
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-very-secure-secret-key-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-this-too
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
CORS_ORIGINS=["https://yourdomain.com"]
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
MAX_REQUEST_SIZE_MB=100
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash pynomaly
RUN chown -R pynomaly:pynomaly /app
USER pynomaly

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "pynomaly.presentation.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://pynomaly:password@db:5432/pynomaly
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_USER=pynomaly
      - POSTGRES_PASSWORD=password
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

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Production docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    image: pynomaly:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    environment:
      - DATABASE_URL=postgresql://pynomaly:${DB_PASSWORD}@db:5432/pynomaly
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    depends_on:
      - db
      - redis
    restart: unless-stopped

  worker:
    image: pynomaly:latest
    command: ["python", "-m", "pynomaly.workers.detection_worker"]
    deploy:
      replicas: 2
    environment:
      - DATABASE_URL=postgresql://pynomaly:${DB_PASSWORD}@db:5432/pynomaly
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=pynomaly
      - POSTGRES_USER=pynomaly
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backup:/backup
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

## Kubernetes Deployment

### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
  namespace: pynomaly
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DEBUG: "false"
  REDIS_URL: "redis://redis:6379/0"
```

### Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
  namespace: pynomaly
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL3B5bm9tYWx5OnBhc3N3b3JkQGRiOjU0MzIvcHlub21hbHk=
  SECRET_KEY: eW91ci1zZWNyZXQta2V5LWhlcmU=
  JWT_SECRET_KEY: eW91ci1qd3Qtc2VjcmV0LWtleQ==
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  namespace: pynomaly
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
      - name: api
        image: pynomaly:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: DATABASE_URL
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: SECRET_KEY
        envFrom:
        - configMapRef:
            name: pynomaly-config
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          limits:
            cpu: 1000m
            memory: 1Gi
          requests:
            cpu: 500m
            memory: 512Mi
```

### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-api
  namespace: pynomaly
spec:
  selector:
    app: pynomaly-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-ingress
  namespace: pynomaly
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
spec:
  tls:
  - hosts:
    - api.pynomaly.com
    secretName: pynomaly-tls
  rules:
  - host: api.pynomaly.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-api
            port:
              number: 80
```

### HorizontalPodAutoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
  namespace: pynomaly
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

## Cloud Deployments

### AWS ECS with Fargate
```yaml
# ecs-task-definition.json
{
  "family": "pynomaly",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "pynomaly-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/pynomaly:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:ssm:region:account:parameter/pynomaly/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/pynomaly",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: pynomaly-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 1000
      containers:
      - image: gcr.io/project-id/pynomaly:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
```

### Azure Container Instances
```yaml
apiVersion: '2019-12-01'
location: westus2
name: pynomaly-api
properties:
  containers:
  - name: pynomaly-api
    properties:
      image: your-registry.azurecr.io/pynomaly:latest
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: postgresql://user:pass@host/db
      resources:
        requests:
          cpu: 1.0
          memoryInGb: 2.0
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: '8000'
tags: {}
type: Microsoft.ContainerInstance/containerGroups
```

## Load Balancing and Reverse Proxy

### Nginx Configuration
```nginx
upstream pynomaly_backend {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
    server api3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.pynomaly.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.pynomaly.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req zone=api burst=20 nodelay;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;

    location / {
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }

    location /health {
        access_log off;
        proxy_pass http://pynomaly_backend;
    }

    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Database Setup

### PostgreSQL Production Configuration
```sql
-- Create database and user
CREATE DATABASE pynomaly;
CREATE USER pynomaly WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE pynomaly TO pynomaly;

-- Performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();
```

### Database Migration
```bash
# Run database migrations
alembic upgrade head

# Create initial admin user
python -c "
from pynomaly.infrastructure.auth import create_user
create_user('admin', 'secure_password', ['admin'])
"
```

## Monitoring and Observability

### Prometheus Configuration
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pynomaly'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Pynomaly Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
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
        "title": "Anomaly Detection Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(anomaly_detections_total[1h])",
            "legendFormat": "Detections/hour"
          }
        ]
      }
    ]
  }
}
```

## Security Hardening

### Application Security
```python
# settings.py - Production security settings
import os

# Security
SECRET_KEY = os.getenv("SECRET_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
DEBUG = False
ALLOWED_HOSTS = ["api.pynomaly.com"]

# CORS
CORS_ALLOWED_ORIGINS = [
    "https://pynomaly.com",
    "https://app.pynomaly.com"
]

# Security headers
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE = 1000
RATE_LIMIT_BURST = 100

# Input validation
MAX_REQUEST_SIZE_MB = 100
MAX_DETECTION_BATCH_SIZE = 10000
```

### Database Security
```sql
-- Enable SSL
ALTER SYSTEM SET ssl = on;

-- Limit connections
ALTER SYSTEM SET max_connections = 100;

-- Log security events
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_failed_login_attempts = on;

-- Row level security
ALTER TABLE detectors ENABLE ROW LEVEL SECURITY;
CREATE POLICY detector_policy ON detectors
  FOR ALL TO pynomaly
  USING (user_id = current_user_id());
```

## Performance Optimization

### Application Performance
```python
# Async connection pooling
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Redis connection pooling
import aioredis

redis_pool = aioredis.ConnectionPool.from_url(
    REDIS_URL,
    max_connections=20,
    retry_on_timeout=True
)

# Caching configuration
CACHE_TTL_DETECTORS = 300  # 5 minutes
CACHE_TTL_DATASETS = 600   # 10 minutes
CACHE_TTL_RESULTS = 3600   # 1 hour
```

### Database Optimization
```sql
-- Indexes for performance
CREATE INDEX CONCURRENTLY idx_detectors_algorithm ON detectors(algorithm);
CREATE INDEX CONCURRENTLY idx_detectors_created_at ON detectors(created_at);
CREATE INDEX CONCURRENTLY idx_detection_results_detector_id ON detection_results(detector_id);
CREATE INDEX CONCURRENTLY idx_detection_results_created_at ON detection_results(created_at);

-- Partitioning for large tables
CREATE TABLE detection_results_y2024m01 PARTITION OF detection_results
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Vacuum and analyze
VACUUM ANALYZE;
```

## Backup and Disaster Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh - Automated database backup

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup"
DB_NAME="pynomaly"

# Create backup
pg_dump -h db -U pynomaly -d $DB_NAME | gzip > $BACKUP_DIR/pynomaly_$DATE.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/pynomaly_$DATE.sql.gz s3://pynomaly-backups/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "pynomaly_*.sql.gz" -mtime +30 -delete
```

### Redis Backup
```bash
# Redis backup script
redis-cli -h redis BGSAVE
cp /var/lib/redis/dump.rdb /backup/redis_$(date +%Y%m%d_%H%M%S).rdb
```

### Disaster Recovery Plan
1. **Database**: Restore from latest backup
2. **Redis**: Restore from backup or rebuild cache
3. **Application**: Deploy from latest container image
4. **DNS**: Update records to point to backup infrastructure
5. **SSL**: Ensure certificates are valid

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Monitor specific container
docker exec -it pynomaly-api ps aux

# Check for memory leaks
curl http://localhost:8000/metrics | grep memory
```

#### Database Connection Issues
```bash
# Check database connectivity
pg_isready -h db -p 5432

# Check connection pool status
curl http://localhost:8000/status | jq '.database'

# Monitor slow queries
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

#### Performance Issues
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Monitor request queue
curl http://localhost:8000/metrics | grep queue

# Check resource limits
kubectl describe pod pynomaly-api
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/db

# Cache health
curl http://localhost:8000/health/cache

# Comprehensive status
curl http://localhost:8000/status
```

## Production Checklist

### Pre-deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations applied
- [ ] Monitoring configured
- [ ] Backup system tested
- [ ] Load testing completed
- [ ] Security scan passed

### Post-deployment
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Backup verification
- [ ] Performance baseline established
- [ ] Documentation updated

## Advanced Deployment Scenarios

### Multi-Region High Availability Deployment

Deploy Pynomaly across multiple regions for maximum availability and performance:

#### Global Load Balancer Configuration
```yaml
# global-load-balancer.yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: pynomaly-ssl-cert
spec:
  domains:
    - api.pynomaly.com
---
apiVersion: compute.googleapis.com/v1
kind: GlobalAddress
metadata:
  name: pynomaly-global-ip
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-global-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: pynomaly-global-ip
    networking.gke.io/managed-certificates: pynomaly-ssl-cert
    kubernetes.io/ingress.class: gce
    kubernetes.io/ingress.allow-http: "false"
spec:
  rules:
  - host: api.pynomaly.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: pynomaly-api
            port:
              number: 80
```

#### Cross-Region Database Replication
```sql
-- Primary region database (us-east-1)
CREATE PUBLICATION pynomaly_replication FOR ALL TABLES;

-- Read replica setup (eu-west-1)
CREATE SUBSCRIPTION pynomaly_replica
CONNECTION 'host=primary-db.us-east-1.rds.amazonaws.com user=replication dbname=pynomaly'
PUBLICATION pynomaly_replication;

-- Application configuration for read replicas
```

#### Region-Specific Configuration
```python
# config/regions.py
from pynomaly.infrastructure.resilience import ml_resilient

REGION_CONFIGS = {
    "us-east-1": {
        "database_url": "postgresql://user:pass@primary-db-us-east-1/pynomaly",
        "redis_url": "redis://cache-us-east-1:6379",
        "model_storage": "s3://pynomaly-models-us-east-1",
        "backup_region": "us-west-2"
    },
    "eu-west-1": {
        "database_url": "postgresql://user:pass@replica-db-eu-west-1/pynomaly",
        "redis_url": "redis://cache-eu-west-1:6379",
        "model_storage": "s3://pynomaly-models-eu-west-1",
        "backup_region": "eu-central-1"
    }
}

@ml_resilient(timeout_seconds=300, max_attempts=2)
async def get_regional_config(region: str):
    """Get configuration for specific region with resilience."""
    if region not in REGION_CONFIGS:
        raise ConfigurationError(f"Unknown region: {region}")
    return REGION_CONFIGS[region]
```

### Microservices Architecture Deployment

Deploy Pynomaly as microservices for better scalability and maintainability:

#### Service Mesh with Istio
```yaml
# istio-gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: pynomaly-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: pynomaly-tls
    hosts:
    - api.pynomaly.com
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: pynomaly-routes
spec:
  hosts:
  - api.pynomaly.com
  gateways:
  - pynomaly-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1/detectors
    route:
    - destination:
        host: detector-service
        port:
          number: 8000
  - match:
    - uri:
        prefix: /api/v1/datasets
    route:
    - destination:
        host: dataset-service
        port:
          number: 8001
  - match:
    - uri:
        prefix: /api/v1/models
    route:
    - destination:
        host: model-service
        port:
          number: 8002
```

#### Microservice Deployment
```yaml
# detector-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: detector-service
  labels:
    app: detector-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: detector-service
  template:
    metadata:
      labels:
        app: detector-service
        version: v1
      annotations:
        sidecar.istio.io/inject: "true"
    spec:
      containers:
      - name: detector-service
        image: pynomaly/detector-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: SERVICE_NAME
          value: "detector-service"
        - name: RESILIENCE_ENABLED
          value: "true"
        - name: CIRCUIT_BREAKER_ENABLED
          value: "true"
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
```

### Edge Computing Deployment

Deploy lightweight anomaly detection at edge locations:

#### Edge Node Configuration
```yaml
# edge-deployment.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pynomaly-edge
  labels:
    app: pynomaly-edge
spec:
  selector:
    matchLabels:
      app: pynomaly-edge
  template:
    metadata:
      labels:
        app: pynomaly-edge
    spec:
      nodeSelector:
        node-type: edge
      containers:
      - name: pynomaly-edge
        image: pynomaly/edge:latest
        resources:
          limits:
            cpu: "0.5"
            memory: "512Mi"
          requests:
            cpu: "0.2"
            memory: "256Mi"
        env:
        - name: EDGE_MODE
          value: "true"
        - name: CENTRAL_API_URL
          value: "https://api.pynomaly.com"
        - name: SYNC_INTERVAL
          value: "300" # 5 minutes
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: data-buffer
          mountPath: /app/buffer
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 1Gi
      - name: data-buffer
        emptyDir:
          sizeLimit: 2Gi
```

#### Edge Synchronization Service
```python
# edge/sync_service.py
from pynomaly.infrastructure.resilience import api_resilient
import asyncio
import logging

logger = logging.getLogger(__name__)

class EdgeSyncService:
    """Service for synchronizing edge nodes with central system."""

    def __init__(self, central_api_url: str, sync_interval: int = 300):
        self.central_api_url = central_api_url
        self.sync_interval = sync_interval
        self.local_models = {}
        self.pending_results = []

    @api_resilient(timeout_seconds=30, max_attempts=3)
    async def sync_models(self):
        """Sync models from central system with resilience."""
        try:
            response = await self.api_client.get(f"{self.central_api_url}/api/v1/models/edge")
            models = response.json()

            for model_info in models:
                if model_info['version'] > self.local_models.get(model_info['id'], {}).get('version', 0):
                    await self.download_model(model_info)

        except Exception as e:
            logger.error(f"Model sync failed: {e}")
            # Continue with cached models

    @api_resilient(timeout_seconds=60, max_attempts=2)
    async def upload_results(self):
        """Upload pending results to central system."""
        if not self.pending_results:
            return

        try:
            response = await self.api_client.post(
                f"{self.central_api_url}/api/v1/results/edge",
                json={"results": self.pending_results}
            )

            if response.status_code == 200:
                self.pending_results.clear()
                logger.info(f"Uploaded {len(self.pending_results)} results")

        except Exception as e:
            logger.error(f"Result upload failed: {e}")
            # Keep results for next sync attempt
```

### Serverless Deployment with Auto-Scaling

#### AWS Lambda with API Gateway
```python
# lambda/handler.py
from pynomaly.infrastructure.resilience import ml_resilient
import json
import asyncio

@ml_resilient(timeout_seconds=30, max_attempts=1)
async def detect_anomalies_lambda(event, context):
    """Lambda function for anomaly detection with resilience."""
    try:
        # Parse input data
        body = json.loads(event['body'])
        data = body['data']
        model_id = body.get('model_id', 'default')

        # Load model (cached)
        detector = await load_cached_detector(model_id)

        # Perform detection
        results = await detector.detect_async(data)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'anomalies': results.anomalies,
                'scores': results.scores.tolist(),
                'model_id': model_id,
                'processing_time': results.processing_time
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    return asyncio.run(detect_anomalies_lambda(event, context))
```

#### Serverless Configuration
```yaml
# serverless.yml
service: pynomaly-serverless

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  timeout: 30
  memorySize: 1024
  environment:
    MODEL_CACHE_BUCKET: pynomaly-models-${self:provider.stage}
    RESULTS_TABLE: pynomaly-results-${self:provider.stage}

functions:
  detect:
    handler: handler.lambda_handler
    events:
      - http:
          path: detect
          method: post
          cors: true
    reservedConcurrency: 100

  batch-detect:
    handler: batch_handler.lambda_handler
    timeout: 300
    memorySize: 3008
    events:
      - s3:
          bucket: pynomaly-input-${self:provider.stage}
          event: s3:ObjectCreated:*
          rules:
            - suffix: .csv

resources:
  Resources:
    ModelsS3Bucket:
      Type: AWS::S3::Bucket
      Properties:
        BucketName: pynomaly-models-${self:provider.stage}

    ResultsTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: pynomaly-results-${self:provider.stage}
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH
```

### Hybrid Cloud Integration

Integrate on-premises and cloud deployments:

#### VPN Gateway Configuration
```yaml
# hybrid-networking.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vpn-config
data:
  vpn.conf: |
    [Interface]
    PrivateKey = <on-premises-private-key>
    Address = 10.0.1.1/24

    [Peer]
    PublicKey = <cloud-public-key>
    Endpoint = vpn.pynomaly.com:51820
    AllowedIPs = 10.0.0.0/16
    PersistentKeepalive = 25
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vpn-gateway
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vpn-gateway
  template:
    metadata:
      labels:
        app: vpn-gateway
    spec:
      containers:
      - name: wireguard
        image: linuxserver/wireguard:latest
        securityContext:
          capabilities:
            add:
              - NET_ADMIN
              - SYS_MODULE
        env:
        - name: PUID
          value: "1000"
        - name: PGID
          value: "1000"
        volumeMounts:
        - name: vpn-config
          mountPath: /config/wg0.conf
          subPath: vpn.conf
      volumes:
      - name: vpn-config
        configMap:
          name: vpn-config
```

#### Hybrid Data Synchronization
```python
# hybrid/sync_manager.py
from pynomaly.infrastructure.resilience import database_resilient, api_resilient

class HybridSyncManager:
    """Manages data synchronization between on-premises and cloud."""

    def __init__(self, on_prem_db_url: str, cloud_api_url: str):
        self.on_prem_db = create_async_engine(on_prem_db_url)
        self.cloud_api_url = cloud_api_url
        self.sync_queue = []

    @database_resilient(timeout_seconds=60, max_attempts=3)
    async def sync_models_to_cloud(self):
        """Sync trained models from on-premises to cloud."""
        async with self.on_prem_db.begin() as conn:
            result = await conn.execute(
                "SELECT * FROM models WHERE synced_to_cloud = false"
            )

            for model in result:
                await self.upload_model_to_cloud(model)
                await conn.execute(
                    "UPDATE models SET synced_to_cloud = true WHERE id = ?",
                    model.id
                )

    @api_resilient(timeout_seconds=120, max_attempts=2)
    async def sync_results_from_cloud(self):
        """Sync detection results from cloud to on-premises."""
        response = await self.api_client.get(
            f"{self.cloud_api_url}/api/v1/results/since/{self.last_sync_time}"
        )

        results = response.json()

        async with self.on_prem_db.begin() as conn:
            for result in results['results']:
                await conn.execute(
                    "INSERT INTO detection_results (...) VALUES (...)",
                    **result
                )
```

### Zero-Downtime Deployment Strategies

#### Blue-Green Deployment
```yaml
# blue-green-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: pynomaly-api
spec:
  replicas: 5
  strategy:
    blueGreen:
      activeService: pynomaly-active
      previewService: pynomaly-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: pynomaly-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: pynomaly-active
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
        image: pynomaly/api:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
```

#### Canary Deployment with Istio
```yaml
# canary-deployment.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: pynomaly-canary
spec:
  hosts:
  - api.pynomaly.com
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: pynomaly-api
        subset: v2
  - route:
    - destination:
        host: pynomaly-api
        subset: v1
      weight: 90
    - destination:
        host: pynomaly-api
        subset: v2
      weight: 10
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: pynomaly-api
spec:
  host: pynomaly-api
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

### Advanced Monitoring and Alerting

#### Custom Metrics for Anomaly Detection
```python
# monitoring/custom_metrics.py
from prometheus_client import Counter, Histogram, Gauge
from pynomaly.infrastructure.resilience import ml_resilient

# Custom metrics
ANOMALY_DETECTIONS_TOTAL = Counter(
    'pynomaly_anomaly_detections_total',
    'Total number of anomaly detections',
    ['model_type', 'dataset_type', 'region']
)

DETECTION_DURATION = Histogram(
    'pynomaly_detection_duration_seconds',
    'Time spent on anomaly detection',
    ['model_type', 'dataset_size_bucket']
)

MODEL_ACCURACY = Gauge(
    'pynomaly_model_accuracy',
    'Current model accuracy score',
    ['model_id', 'model_type']
)

CIRCUIT_BREAKER_STATE = Gauge(
    'pynomaly_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service', 'operation']
)

@ml_resilient(timeout_seconds=300, max_attempts=2)
async def track_detection_metrics(model_type: str, dataset_type: str, region: str, duration: float):
    """Track detection metrics with resilience."""
    ANOMALY_DETECTIONS_TOTAL.labels(
        model_type=model_type,
        dataset_type=dataset_type,
        region=region
    ).inc()

    DETECTION_DURATION.labels(
        model_type=model_type,
        dataset_size_bucket=get_size_bucket(len(data))
    ).observe(duration)
```

#### Advanced Alerting Rules
```yaml
# alerting-rules.yaml
groups:
- name: pynomaly.rules
  rules:
  - alert: HighAnomalyRate
    expr: rate(pynomaly_anomaly_detections_total[5m]) > 100
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High anomaly detection rate detected"
      description: "Anomaly detection rate is {{ $value }} per second"

  - alert: CircuitBreakerOpen
    expr: pynomaly_circuit_breaker_state > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Circuit breaker is open for {{ $labels.service }}"
      description: "Circuit breaker for {{ $labels.service }}/{{ $labels.operation }} is in state {{ $value }}"

  - alert: ModelAccuracyDegraded
    expr: pynomaly_model_accuracy < 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy has degraded"
      description: "Model {{ $labels.model_id }} accuracy is {{ $value }}"

  - alert: DatabaseConnectionPoolExhausted
    expr: pynomaly_database_connections_active / pynomaly_database_connections_max > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "Database connection usage is at {{ $value | humanizePercentage }}"
```

This comprehensive deployment guide covers enterprise-grade scenarios with advanced resilience patterns, multi-region deployments, microservices architecture, edge computing, serverless deployment, hybrid cloud integration, zero-downtime deployment strategies, and sophisticated monitoring. All deployment patterns leverage Pynomaly's built-in infrastructure resilience features including circuit breakers, retry mechanisms, and timeout handling for maximum reliability in production environments.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
