# Pynomaly Production Deployment Guide

This comprehensive guide will walk you through deploying Pynomaly to production with all advanced features enabled.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows 11
- **CPU**: 8+ cores recommended
- **Memory**: 16GB+ RAM recommended
- **Storage**: 100GB+ SSD recommended
- **Network**: Stable internet connection

### Software Dependencies
- **Docker**: 24.0+ and Docker Compose 2.0+
- **Python**: 3.11+ (for development/testing)
- **Git**: 2.30+ for version control
- **OpenSSL**: For SSL certificate generation

### Cloud Provider Support
- **AWS**: ECS, EKS, or EC2 deployment
- **Google Cloud**: GKE or Compute Engine
- **Azure**: AKS or Virtual Machines
- **DigitalOcean**: Kubernetes or Droplets

## 🏗️ Quick Start Deployment

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Set up environment variables
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your production values:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_postgres_password
DATABASE_URL=postgresql://pynomaly:${POSTGRES_PASSWORD}@postgres:5432/pynomaly_prod

# Redis Configuration
REDIS_URL=redis://redis-cluster:6379

# Security
SECRET_KEY=your_very_secure_secret_key_here
GRAFANA_PASSWORD=your_grafana_admin_password

# Optional: External Services
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Cloud Storage (optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### 3. Production Deployment

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f pynomaly-api
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost/health

# API documentation
curl http://localhost/docs

# Grafana dashboard
open http://localhost:3000 (admin/your_grafana_password)
```

## 🔧 Detailed Configuration

### Database Setup

#### PostgreSQL Configuration

```bash
# Create database backup directory
mkdir -p ./backups/postgres

# Initialize database with custom settings
docker-compose -f docker-compose.production.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
"
```

#### Database Migrations

```bash
# Run database migrations
docker-compose -f docker-compose.production.yml exec pynomaly-api python -m alembic upgrade head

# Create initial admin user
docker-compose -f docker-compose.production.yml exec pynomaly-api python -m src.pynomaly.scripts.create_admin_user
```

### Redis Configuration

Create `config/redis.conf`:

```conf
# Redis Production Configuration
port 6379
timeout 0
keepalive 60
maxclients 10000
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
dbfilename dump.rdb
dir /data

# Logging
loglevel notice
logfile /var/log/redis/redis.log

# Security
requirepass your_redis_password
rename-command FLUSHALL ""
rename-command FLUSHDB ""
```

### Nginx Configuration

Create `config/nginx/nginx.conf`:

```nginx
upstream pynomaly_api {
    server pynomaly-api:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/m;
    
    # API Endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://pynomaly_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Health Check
    location /health {
        proxy_pass http://pynomaly_api;
        access_log off;
    }
    
    # Documentation
    location /docs {
        proxy_pass http://pynomaly_api;
    }
    
    # Static Files
    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Grafana Dashboard
    location /grafana/ {
        proxy_pass http://grafana:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Monitoring Setup

#### Prometheus Configuration

Create `config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['pynomaly-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-cluster:6379']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Grafana Dashboard Configuration

Create `config/grafana/dashboards/pynomaly-dashboard.json`:

```json
{
  "dashboard": {
    "title": "Pynomaly Production Dashboard",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Anomaly Detection Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(anomaly_detections_total[5m])"
          }
        ]
      },
      {
        "title": "System Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "memory_usage_percent",
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

## 🔐 Security Configuration

### SSL Certificate Setup

```bash
# Using Let's Encrypt
certbot certonly --webroot -w /var/www/html -d your-domain.com

# Copy certificates to nginx directory
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem config/nginx/ssl/
cp /etc/letsencrypt/live/your-domain.com/privkey.pem config/nginx/ssl/

# Set up auto-renewal
echo "0 2 * * * certbot renew --quiet" | crontab -
```

### Database Security

```sql
-- Create read-only user for monitoring
CREATE USER pynomaly_monitor WITH PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE pynomaly_prod TO pynomaly_monitor;
GRANT USAGE ON SCHEMA public TO pynomaly_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO pynomaly_monitor;

-- Enable row-level security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_policy ON users FOR ALL TO pynomaly USING (true);
```

### Application Security

```python
# Security middleware configuration
SECURITY_MIDDLEWARE = {
    'cors': {
        'allow_origins': ['https://your-domain.com'],
        'allow_credentials': True,
        'allow_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        'allow_headers': ['*'],
        'max_age': 3600
    },
    'rate_limiting': {
        'global': '10000/hour',
        'per_user': '1000/hour',
        'per_ip': '500/hour'
    },
    'input_validation': {
        'max_request_size': 50 * 1024 * 1024,  # 50MB
        'max_file_size': 100 * 1024 * 1024,   # 100MB
        'allowed_extensions': ['.csv', '.json', '.parquet']
    }
}
```

## 🚀 Scaling Configuration

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  pynomaly-api:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        monitor: 60s
```

### Load Balancer Configuration

```bash
# Scale API service
docker-compose -f docker-compose.production.yml -f docker-compose.scale.yml up -d --scale pynomaly-api=3

# Add load balancer
docker-compose -f docker-compose.production.yml up -d nginx
```

### Auto-scaling with Kubernetes

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 2
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

## 📊 Monitoring and Alerting

### Health Checks

```bash
# Application health
curl -f http://localhost/health || exit 1

# Database health
docker-compose -f docker-compose.production.yml exec postgres pg_isready -U pynomaly -d pynomaly_prod

# Redis health
docker-compose -f docker-compose.production.yml exec redis-cluster redis-cli ping

# Streaming health
curl -f http://localhost/api/v1/streaming/health || exit 1
```

### Alerting Rules

Create `config/prometheus/rules/pynomaly.yml`:

```yaml
groups:
  - name: pynomaly.rules
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} requests/second"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }} seconds"
          
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "PostgreSQL database is not responding"
```

## 🔄 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="pynomaly_backup_${TIMESTAMP}.sql"

# Create backup
docker-compose -f docker-compose.production.yml exec postgres pg_dump -U pynomaly -d pynomaly_prod > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Remove old backups (keep last 30 days)
find "${BACKUP_DIR}" -name "*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://your-backup-bucket/postgres/
```

### Model Backup

```bash
#!/bin/bash
# backup_models.sh

MODEL_DIR="/app/models"
BACKUP_DIR="/backups/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create model backup
tar -czf "${BACKUP_DIR}/models_backup_${TIMESTAMP}.tar.gz" -C "${MODEL_DIR}" .

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/models_backup_${TIMESTAMP}.tar.gz" s3://your-backup-bucket/models/
```

### Automated Backup Schedule

```bash
# Add to crontab
0 2 * * * /opt/pynomaly/scripts/backup_database.sh
0 3 * * * /opt/pynomaly/scripts/backup_models.sh
0 4 * * 0 /opt/pynomaly/scripts/backup_config.sh
```

## 🛠️ Troubleshooting

### Common Issues

#### API Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs pynomaly-api

# Check configuration
docker-compose -f docker-compose.production.yml exec pynomaly-api python -c "import src.pynomaly.infrastructure.config.settings; print('Config OK')"

# Check database connection
docker-compose -f docker-compose.production.yml exec pynomaly-api python -c "
from src.pynomaly.infrastructure.persistence.database import engine
print('Database connection OK')
"
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Tune garbage collection
export PYTHONHASHSEED=0
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Adjust worker processes
export WORKERS=2
export WORKER_CLASS=uvicorn.workers.UvicornWorker
```

#### Database Connection Issues

```bash
# Check database connectivity
docker-compose -f docker-compose.production.yml exec pynomaly-api nc -zv postgres 5432

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres

# Reset database connection pool
docker-compose -f docker-compose.production.yml restart pynomaly-api
```

### Performance Tuning

#### Database Optimization

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM anomaly_detections WHERE created_at > NOW() - INTERVAL '1 day';

-- Update statistics
ANALYZE;

-- Vacuum database
VACUUM ANALYZE;

-- Check slow queries
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
```

#### Application Optimization

```python
# Tune async settings
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Connection pool tuning
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
DATABASE_POOL_TIMEOUT = 30

# Cache optimization
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_ENTRIES = 10000
```

## 📈 Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh

# Update system packages
apt update && apt upgrade -y

# Clean up Docker
docker system prune -f

# Restart services
docker-compose -f docker-compose.production.yml restart

# Check disk space
df -h

# Clean up old logs
find /var/log -name "*.log" -mtime +7 -delete

# Update SSL certificates
certbot renew --quiet
```

### Monthly Maintenance

```bash
#!/bin/bash
# monthly_maintenance.sh

# Database maintenance
docker-compose -f docker-compose.production.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "VACUUM ANALYZE;"

# Update model registry
docker-compose -f docker-compose.production.yml exec pynomaly-api python -m src.pynomaly.scripts.cleanup_old_models

# Security audit
docker-compose -f docker-compose.production.yml exec pynomaly-api python -m src.pynomaly.scripts.security_audit

# Performance report
docker-compose -f docker-compose.production.yml exec pynomaly-api python -m src.pynomaly.scripts.performance_report
```

## 🎯 Performance Benchmarks

### Expected Performance Metrics

| Metric | Target | Monitoring |
|--------|--------|------------|
| API Response Time | < 100ms (95th percentile) | Prometheus |
| Throughput | > 1000 requests/second | Grafana |
| Streaming Rate | > 10,000 events/second | Application logs |
| Database Queries | < 50ms average | PostgreSQL logs |
| Memory Usage | < 80% of allocated | Docker stats |
| CPU Usage | < 70% average | System monitoring |

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost --users=100 --spawn-rate=10
```

## 📞 Support and Documentation

### Getting Help

- **Documentation**: [https://docs.pynomaly.com](https://docs.pynomaly.com)
- **API Reference**: [https://api.pynomaly.com/docs](https://api.pynomaly.com/docs)
- **GitHub Issues**: [https://github.com/your-org/pynomaly/issues](https://github.com/your-org/pynomaly/issues)
- **Community**: [https://community.pynomaly.com](https://community.pynomaly.com)

### Enterprise Support

For enterprise deployment support, contact:
- **Email**: enterprise@pynomaly.com
- **Phone**: +1-555-PYNOMALY
- **Slack**: #enterprise-support

---

## 🎉 Congratulations!

You have successfully deployed Pynomaly to production! Your anomaly detection system is now ready to:

- ✅ Handle production workloads with high availability
- ✅ Process real-time data streams
- ✅ Provide explainable AI insights
- ✅ Scale automatically based on demand
- ✅ Monitor system health and performance
- ✅ Maintain data security and compliance

Next steps:
1. Configure monitoring and alerting
2. Set up regular backups
3. Implement CI/CD pipeline
4. Train your team on system operations
5. Scale based on your specific requirements

Happy anomaly detecting! 🚀