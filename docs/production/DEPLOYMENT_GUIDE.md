# 🚀 Pynomaly Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Pynomaly in production environments. It covers all aspects from system requirements to monitoring and maintenance.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- RAM: 4GB
- Storage: 10GB
- CPU: 2 cores

**Recommended for Production:**
- Python 3.10+
- RAM: 16GB+
- Storage: 100GB+ (SSD recommended)
- CPU: 8+ cores
- GPU: Optional (for deep learning features)

### Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Production dependencies
pip install gunicorn uvicorn[standard] redis celery prometheus-client
```

## 🔧 Production Configuration

### 1. Environment Configuration

Create a production environment file:

```bash
# .env.production
PYNOMALY_ENV=production
PYNOMALY_DEBUG=false
PYNOMALY_LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/pynomaly_prod
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-this
API_KEY_SALT=your-api-key-salt-change-this
CORS_ORIGINS=https://your-frontend-domain.com

# Performance
MAX_WORKERS=8
WORKER_TIMEOUT=300
MAX_REQUESTS_PER_WORKER=1000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### 2. Database Setup

```bash
# PostgreSQL setup
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createdb pynomaly_prod
sudo -u postgres createuser pynomaly_user --pwprompt

# Run migrations
pynomaly db migrate
pynomaly db upgrade
```

### 3. Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf
# Set: maxmemory 1gb
# Set: maxmemory-policy allkeys-lru
```

## 🐳 Docker Deployment

### Docker Compose Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  pynomaly-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYNOMALY_ENV=production
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  pynomaly-worker:
    build: .
    command: celery -A pynomaly.infrastructure.celery worker --loglevel=info
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: pynomaly_prod
      POSTGRES_USER: pynomaly_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
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

### Deployment Commands

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Monitor deployment
docker-compose -f docker-compose.prod.yml logs -f pynomaly-api
```

## 🌐 Web Server Configuration

### Nginx Setup

```nginx
# /etc/nginx/sites-available/pynomaly
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Websocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

### SSL Certificate Setup

```bash
# Using Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'pynomaly'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Pynomaly Production Metrics",
    "panels": [
      {
        "title": "API Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Detection Accuracy",
        "targets": [
          {
            "expr": "pynomaly_detection_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### API Security

```python
# Security middleware configuration
SECURITY_CONFIG = {
    'api_key_required': True,
    'rate_limiting': {
        'enabled': True,
        'requests_per_minute': 1000,
        'burst_size': 100
    },
    'cors': {
        'allowed_origins': ['https://your-frontend-domain.com'],
        'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
        'allowed_headers': ['Content-Type', 'Authorization']
    },
    'authentication': {
        'jwt_secret': 'your-jwt-secret',
        'jwt_algorithm': 'HS256',
        'jwt_expiration': 3600
    }
}
```

### Data Protection

```bash
# Encrypt sensitive data at rest
sudo apt-get install cryptsetup
sudo cryptsetup luksFormat /dev/sdb
sudo cryptsetup luksOpen /dev/sdb encrypted-data
sudo mkfs.ext4 /dev/mapper/encrypted-data
sudo mount /dev/mapper/encrypted-data /app/data
```

## 🚀 Deployment Process

### 1. Pre-deployment Checks

```bash
# Run comprehensive tests
pynomaly validate --comprehensive --coverage-threshold 90

# Check system resources
pynomaly system check --production

# Verify configuration
pynomaly config validate --env production
```

### 2. Blue-Green Deployment

```bash
#!/bin/bash
# deploy.sh

# Build new version
docker-compose -f docker-compose.prod.yml build

# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run smoke tests
pynomaly test smoke --env staging

# Switch traffic
nginx -s reload

# Monitor for 10 minutes
sleep 600

# Cleanup old version
docker-compose -f docker-compose.prod.yml down
```

### 3. Rollback Process

```bash
#!/bin/bash
# rollback.sh

# Revert to previous version
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# Verify rollback
pynomaly health check --timeout 30
```

## 📈 Performance Optimization

### Database Optimization

```sql
-- Create indexes for performance
CREATE INDEX idx_anomaly_timestamp ON anomalies(timestamp);
CREATE INDEX idx_detector_model_id ON detectors(model_id);
CREATE INDEX idx_results_created_at ON results(created_at);

-- Configure PostgreSQL for production
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

### Application Optimization

```python
# gunicorn configuration
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2
```

## 🔧 Maintenance

### Daily Tasks

```bash
# Health check
pynomaly health check --comprehensive

# Log rotation
logrotate /etc/logrotate.d/pynomaly

# Backup
pg_dump pynomaly_prod > /backups/pynomaly_$(date +%Y%m%d).sql
```

### Weekly Tasks

```bash
# Performance review
pynomaly perf report --last-week

# Security scan
pynomaly security scan

# Update dependencies
pip list --outdated
```

### Monthly Tasks

```bash
# Full system backup
rsync -av /app/data/ /backups/monthly/

# Performance tuning
pynomaly perf optimize --analyze

# Documentation review
pynomaly docs validate
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   pynomaly system memory --detailed
   
   # Restart workers
   docker-compose restart pynomaly-worker
   ```

2. **Slow API Response**
   ```bash
   # Check performance metrics
   pynomaly perf analyze --last-hour
   
   # Scale up workers
   docker-compose up -d --scale pynomaly-api=4
   ```

3. **Database Connection Issues**
   ```bash
   # Check connections
   pynomaly db status
   
   # Reset connection pool
   pynomaly db reset-pool
   ```

### Monitoring Alerts

```yaml
# alertmanager.yml
groups:
  - name: pynomaly
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes > 8e9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
```

## 📱 API Documentation

### Health Check Endpoints

```
GET /health - Basic health check
GET /health/detailed - Detailed system status
GET /health/ready - Readiness probe
GET /health/live - Liveness probe
```

### Monitoring Endpoints

```
GET /metrics - Prometheus metrics
GET /stats - Application statistics
GET /perf - Performance metrics
```

## 🔐 Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

# Database backup
pg_dump pynomaly_prod | gzip > /backups/db_$(date +%Y%m%d_%H%M%S).sql.gz

# Data backup
tar czf /backups/data_$(date +%Y%m%d_%H%M%S).tar.gz /app/data

# Upload to S3
aws s3 cp /backups/ s3://pynomaly-backups/ --recursive

# Cleanup old backups
find /backups -name "*.gz" -mtime +7 -delete
```

### Recovery Process

```bash
# Restore database
gunzip -c /backups/db_20231201_120000.sql.gz | psql pynomaly_prod

# Restore data
tar xzf /backups/data_20231201_120000.tar.gz -C /

# Restart services
docker-compose restart
```

## 🎯 Performance Benchmarks

### Target Metrics

- **API Response Time**: <100ms (95th percentile)
- **Detection Accuracy**: >95%
- **Throughput**: 1000 requests/second
- **Memory Usage**: <8GB
- **CPU Usage**: <80%
- **Uptime**: 99.9%

### Benchmarking Commands

```bash
# Performance benchmark
pynomaly perf benchmark --duration 300 --concurrency 100

# Load test
pynomaly test load --users 1000 --duration 600

# Stress test
pynomaly test stress --ramp-up 60 --duration 300
```

## 📚 Additional Resources

- [API Documentation](./API_DOCUMENTATION.md)
- [Monitoring Guide](./MONITORING_GUIDE.md)
- [Security Best Practices](./SECURITY_GUIDE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

## 🆘 Support

For production support issues:
- Email: support@pynomaly.com
- Slack: #pynomaly-production
- On-call: +1-555-PYNOMALY

## 📝 Change Log

### v0.1.1 (Current)
- Initial production deployment guide
- Docker containerization
- Monitoring setup
- Security hardening

### Future Releases
- Kubernetes deployment
- Multi-region support
- Advanced monitoring
- Auto-scaling capabilities