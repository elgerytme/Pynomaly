# Production Deployment Guide

**Authoritative guide for deploying Pynomaly in production environments**

## Quick Links

- [Prerequisites](#prerequisites) - System requirements and dependencies
- [Docker Deployment](#docker-deployment) - Container-based deployment
- [Kubernetes Deployment](#kubernetes-deployment) - Scalable orchestration
- [Configuration](#configuration) - Production configuration settings
- [Monitoring](#monitoring) - Health checks and observability
- [Security](#security) - Security hardening and best practices
- [Troubleshooting](#troubleshooting) - Common issues and solutions

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU**: 8+ cores, 3.0GHz+
- **Memory**: 32GB+ RAM
- **Storage**: 500GB+ NVMe SSD
- **Network**: 1Gbps+ bandwidth

**High Availability Setup:**
- **Load Balancer**: 2+ instances
- **Application Servers**: 3+ instances
- **Database**: Primary + 2 replicas
- **Cache**: Redis cluster (3+ nodes)

### Software Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | 2.20+ | Container orchestration |
| Kubernetes | 1.28+ | Container orchestration (optional) |
| PostgreSQL | 15+ | Primary database |
| Redis | 7.0+ | Caching and session storage |
| NGINX | 1.24+ | Load balancer and reverse proxy |

## Docker Deployment

### 1. Production Docker Compose

```yaml
version: '3.8'

services:
  pynomaly-app:
    image: pynomaly:latest
    ports:
      - "8000:8000"
    environment:
      - PYNOMALY_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/pynomaly
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: pynomaly
      POSTGRES_USER: pynomaly_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped

  redis:
    image: redis:7-alpine
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
      - ./ssl:/etc/ssl
    depends_on:
      - pynomaly-app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 2. Build and Deploy

```bash
# Build production image
docker build -t pynomaly:latest .

# Deploy with environment variables
export DB_PASSWORD=your_secure_password
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
docker-compose logs pynomaly-app
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly-prod
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
  namespace: pynomaly-prod
data:
  PYNOMALY_ENV: "production"
  LOG_LEVEL: "INFO"
```

### 2. Application Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-app
  namespace: pynomaly-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly-app
  template:
    metadata:
      labels:
        app: pynomaly-app
    spec:
      containers:
      - name: pynomaly
        image: pynomaly:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: pynomaly-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Configuration

### Environment Variables

```bash
# Core Application
PYNOMALY_ENV=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@host:5432/pynomaly
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://host:6379/0
REDIS_POOL_SIZE=50

# Security
SECRET_KEY=your-256-bit-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
CORS_ORIGINS=https://yourdomain.com

# Performance
WORKER_PROCESSES=8
WORKER_CONNECTIONS=1000
ASYNC_POOL_SIZE=100

# Monitoring
PROMETHEUS_ENABLED=true
HEALTH_CHECK_ENABLED=true
METRICS_ENABLED=true
```

### Production Configuration File

```yaml
# config/production.yml
app:
  name: "Pynomaly Production"
  version: "1.0.0"
  debug: false
  
database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30
  
cache:
  backend: "redis"
  url: "${REDIS_URL}"
  
security:
  secret_key: "${SECRET_KEY}"
  jwt_secret: "${JWT_SECRET_KEY}"
  cors_origins: ["${CORS_ORIGINS}"]
  
logging:
  level: "INFO"
  format: "json"
  handlers:
    - console
    - file
    - syslog
```

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/health/db

# Cache connectivity  
curl http://localhost:8000/health/cache

# Comprehensive status
curl http://localhost:8000/health/full
```

### Prometheus Metrics

Key metrics to monitor:

- **Application Performance**:
  - `pynomaly_requests_total`
  - `pynomaly_request_duration_seconds`
  - `pynomaly_active_connections`

- **Detection Performance**:
  - `pynomaly_detections_total`
  - `pynomaly_detection_duration_seconds`
  - `pynomaly_model_accuracy`

- **System Resources**:
  - `pynomaly_memory_usage_bytes`
  - `pynomaly_cpu_usage_percent`
  - `pynomaly_disk_usage_bytes`

## Security

### SSL/TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    
    location / {
        proxy_pass http://pynomaly-app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Security Hardening

1. **Network Security**:
   - Use private networks for internal communication
   - Implement proper firewall rules
   - Enable fail2ban for SSH protection

2. **Application Security**:
   - Regular security updates
   - Secure password policies
   - API rate limiting
   - Input validation and sanitization

3. **Data Security**:
   - Database encryption at rest
   - Encrypted backups
   - Secure API tokens
   - Audit logging

## Backup and Recovery

### Database Backups

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U pynomaly_user pynomaly > "${BACKUP_DIR}/pynomaly_${DATE}.sql"

# Compress and upload to S3
gzip "${BACKUP_DIR}/pynomaly_${DATE}.sql"
aws s3 cp "${BACKUP_DIR}/pynomaly_${DATE}.sql.gz" s3://your-backup-bucket/
```

### Disaster Recovery

1. **Database Recovery**:
   ```bash
   # Restore from backup
   gunzip pynomaly_backup.sql.gz
   psql -h localhost -U pynomaly_user -d pynomaly < pynomaly_backup.sql
   ```

2. **Application Recovery**:
   ```bash
   # Redeploy application
   docker-compose down
   docker-compose pull
   docker-compose up -d
   ```

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

### Application Optimization

```python
# Async configuration
ASYNC_POOL_SIZE = 100
MAX_WORKERS = 8
WORKER_TIMEOUT = 300

# Cache configuration
CACHE_TTL = 3600
CACHE_MAX_SIZE = 1000

# Connection pooling
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 30
```

## Troubleshooting

### Common Issues

1. **Application Won't Start**:
   ```bash
   # Check logs
   docker-compose logs pynomaly-app
   
   # Check environment variables
   docker-compose exec pynomaly-app env | grep PYNOMALY
   
   # Test database connection
   docker-compose exec pynomaly-app python -c "from pynomaly.database import test_connection; test_connection()"
   ```

2. **High Memory Usage**:
   ```bash
   # Monitor memory usage
   docker stats
   
   # Check for memory leaks
   docker-compose exec pynomaly-app python -c "import psutil; print(psutil.virtual_memory())"
   ```

3. **Database Connection Issues**:
   ```bash
   # Test database connectivity
   docker-compose exec db psql -U pynomaly_user -d pynomaly -c "SELECT 1;"
   
   # Check connection pool
   docker-compose logs db | grep connection
   ```

### Performance Issues

1. **Slow API Responses**:
   - Check database query performance
   - Monitor connection pool utilization
   - Review cache hit rates
   - Analyze application logs

2. **High CPU Usage**:
   - Profile application performance
   - Check for inefficient algorithms
   - Monitor background tasks
   - Review worker configuration

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review application logs
   - Check disk space usage
   - Verify backup integrity
   - Update security patches

2. **Monthly**:
   - Database maintenance (VACUUM, ANALYZE)
   - Certificate renewal check
   - Performance metrics review
   - Security audit

3. **Quarterly**:
   - Dependency updates
   - Capacity planning review
   - Disaster recovery testing
   - Security penetration testing

---

For additional support and advanced configuration options, see:
- [Configuration Reference](./configuration.md)
- [Monitoring Guide](./monitoring.md)
- [Security Hardening](./security.md)
- [Troubleshooting](../troubleshooting/common-issues.md)