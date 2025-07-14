# Production Deployment Guide

This comprehensive guide covers deploying Pynomaly to production environments with high availability, security, and monitoring capabilities.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Database Configuration](#database-configuration)
5. [Redis Configuration](#redis-configuration)
6. [Application Deployment](#application-deployment)
7. [Load Balancer & Reverse Proxy](#load-balancer--reverse-proxy)
8. [SSL/TLS Configuration](#ssltls-configuration)
9. [Monitoring & Logging](#monitoring--logging)
10. [Security Hardening](#security-hardening)
11. [Backup & Disaster Recovery](#backup--disaster-recovery)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Performance Optimization](#performance-optimization)
14. [Troubleshooting](#troubleshooting)
15. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04 LTS or later, CentOS 8+, or RHEL 8+
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 16GB RAM, Recommended 32GB+ RAM
- **Storage**: Minimum 100GB SSD, Recommended 500GB+ SSD
- **Network**: 1Gbps network connection

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- PostgreSQL 14+
- Redis 7.0+
- NGINX 1.20+
- Certbot (for SSL certificates)

### Cloud Provider Requirements

- **AWS**: EC2 instances, RDS, ElastiCache, Application Load Balancer
- **GCP**: Compute Engine, Cloud SQL, Memorystore, Cloud Load Balancing
- **Azure**: Virtual Machines, Database for PostgreSQL, Cache for Redis, Application Gateway

## Architecture Overview

```
Internet
    ↓
[Load Balancer/CDN]
    ↓
[NGINX Reverse Proxy]
    ↓
[Pynomaly Application Instances]
    ↓
[PostgreSQL Database Cluster]
    ↓
[Redis Cache Cluster]
    ↓
[Monitoring Stack (Grafana/Prometheus)]
```

### Components

1. **Load Balancer**: Distributes traffic across multiple application instances
2. **Reverse Proxy**: NGINX for SSL termination and static file serving
3. **Application Layer**: Multiple Pynomaly instances for high availability
4. **Database Layer**: PostgreSQL with read replicas
5. **Cache Layer**: Redis cluster for session management and caching
6. **Monitoring**: Prometheus, Grafana, and ELK stack

## Infrastructure Setup

### 1. Server Provisioning

#### AWS Setup
```bash
# Create VPC and subnets
aws ec2 create-vpc --cidr-block 10.0.0.0/16
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.1.0/24
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.2.0/24

# Launch EC2 instances
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --count 3
```

#### GCP Setup
```bash
# Create project and enable APIs
gcloud projects create pynomaly-prod
gcloud services enable compute.googleapis.com
gcloud services enable sql-admin.googleapis.com

# Create VM instances
gcloud compute instances create pynomaly-app-1 \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB
```

### 2. Network Configuration

#### Firewall Rules
```bash
# Allow HTTP/HTTPS traffic
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp  # SSH access
sudo ufw allow 5432/tcp  # PostgreSQL (internal only)
sudo ufw allow 6379/tcp  # Redis (internal only)
sudo ufw enable
```

#### Security Groups (AWS)
```bash
# Create security group
aws ec2 create-security-group \
  --group-name pynomaly-sg \
  --description "Pynomaly application security group"

# Add rules
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345678 \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```

## Database Configuration

### 1. PostgreSQL Setup

#### Installation
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-14 postgresql-client-14 postgresql-contrib-14

# Configure PostgreSQL
sudo -u postgres psql
```

#### Database Creation
```sql
-- Create database and user
CREATE DATABASE pynomaly_prod;
CREATE USER pynomaly_user WITH ENCRYPTED PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE pynomaly_prod TO pynomaly_user;

-- Grant schema permissions
\c pynomaly_prod;
GRANT ALL ON SCHEMA public TO pynomaly_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pynomaly_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pynomaly_user;
```

#### Performance Tuning
```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/14/main/postgresql.conf
```

```ini
# Memory settings
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB

# Connection settings
max_connections = 200
max_prepared_transactions = 200

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 64MB
default_statistics_target = 100

# Logging
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
```

#### High Availability Setup
```bash
# Set up streaming replication
# On primary server
sudo nano /etc/postgresql/14/main/postgresql.conf
```

```ini
# Replication settings
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64
```

### 2. Database Migration
```bash
# Run migrations
cd /path/to/pynomaly
python -m alembic upgrade head

# Create indexes for performance
python scripts/create_production_indexes.py
```

## Redis Configuration

### 1. Installation and Setup
```bash
# Install Redis
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

```ini
# Network settings
bind 127.0.0.1 10.0.1.100
port 6379
protected-mode yes

# Memory settings
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence settings
save 900 1
save 300 10
save 60 10000

# Security
requirepass your_secure_redis_password
```

### 2. Redis Cluster Setup (for high availability)
```bash
# Create Redis cluster
redis-cli --cluster create \
  10.0.1.100:7000 10.0.1.101:7000 10.0.1.102:7000 \
  10.0.1.100:7001 10.0.1.101:7001 10.0.1.102:7001 \
  --cluster-replicas 1
```

## Application Deployment

### 1. Docker Deployment

#### Production Dockerfile
```dockerfile
FROM python:3.11-slim

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
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 pynomaly
USER pynomaly

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "src.pynomaly.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose Production
```yaml
version: '3.8'

services:
  pynomaly-app:
    image: pynomaly:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - DATABASE_URL=postgresql://pynomaly_user:${DB_PASSWORD}@postgres:5432/pynomaly_prod
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - JWT_SECRET=${JWT_SECRET}
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    networks:
      - pynomaly-network

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=pynomaly_prod
      - POSTGRES_USER=pynomaly_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - pynomaly-network

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - pynomaly-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - pynomaly-app
    networks:
      - pynomaly-network

volumes:
  postgres_data:
  redis_data:

networks:
  pynomaly-network:
    driver: bridge
```

### 2. Environment Configuration

#### Production Environment File
```bash
# Create .env.production
cat > .env.production << 'EOF'
# Database
DATABASE_URL=postgresql://pynomaly_user:${DB_PASSWORD}@localhost:5432/pynomaly_prod

# Redis
REDIS_URL=redis://:${REDIS_PASSWORD}@localhost:6379/0

# Security
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
API_KEY_SALT=${API_KEY_SALT}

# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# External Services
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=${SMTP_USERNAME}
SMTP_PASSWORD=${SMTP_PASSWORD}

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring
PROMETHEUS_METRICS_ENABLED=true
GRAFANA_ENABLED=true
EOF
```

### 3. Application Startup Script
```bash
#!/bin/bash
# /opt/pynomaly/start_production.sh

set -e

# Load environment variables
source /opt/pynomaly/.env.production

# Run database migrations
python -m alembic upgrade head

# Start application with Gunicorn
exec gunicorn src.pynomaly.presentation.api.app:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --worker-connections 1000 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --preload \
  --timeout 30 \
  --keep-alive 2 \
  --access-logfile /var/log/pynomaly/access.log \
  --error-logfile /var/log/pynomaly/error.log \
  --log-level info
```

## Load Balancer & Reverse Proxy

### 1. NGINX Configuration

#### Main Configuration
```nginx
# /etc/nginx/nginx.conf
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;

    # Basic Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

    # Upstream servers
    upstream pynomaly_backend {
        least_conn;
        server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
        server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL configuration
        ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Referrer-Policy "strict-origin-when-cross-origin";

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Login endpoint with stricter rate limiting
        location /api/v1/auth/login {
            limit_req zone=login burst=3 nodelay;
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
        }

        # Static files
        location /static/ {
            alias /opt/pynomaly/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        # Documentation
        location /docs {
            proxy_pass http://pynomaly_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 2. Load Balancer Configuration (AWS ALB)

#### ALB Setup Script
```bash
#!/bin/bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name pynomaly-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678 \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4

# Create target group
aws elbv2 create-target-group \
  --name pynomaly-targets \
  --protocol HTTP \
  --port 80 \
  --vpc-id vpc-12345678 \
  --health-check-protocol HTTP \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3

# Register targets
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/pynomaly-targets/1234567890123456 \
  --targets Id=i-1234567890abcdef0 Id=i-0987654321fedcba0
```

## SSL/TLS Configuration

### 1. Let's Encrypt Setup
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Set up automatic renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Custom Certificate Setup
```bash
# Generate private key
openssl genrsa -out /etc/ssl/private/pynomaly.key 2048

# Generate certificate signing request
openssl req -new -key /etc/ssl/private/pynomaly.key -out /etc/ssl/certs/pynomaly.csr

# Generate self-signed certificate (for testing)
openssl x509 -req -days 365 -in /etc/ssl/certs/pynomaly.csr -signkey /etc/ssl/private/pynomaly.key -out /etc/ssl/certs/pynomaly.crt
```

## Monitoring & Logging

### 1. Prometheus Configuration
```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "pynomaly_rules.yml"

scrape_configs:
  - job_name: 'pynomaly'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8001', 'localhost:8002']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana Dashboard
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
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### 3. Alerting Rules
```yaml
# /etc/prometheus/pynomaly_rules.yml
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
          description: "Error rate is {{ $value }} requests/second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "PostgreSQL database is not reachable"
```

## Security Hardening

### 1. System Security
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install security tools
sudo apt install fail2ban ufw aide rkhunter

# Configure fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3
```

### 2. Application Security
```bash
# Set up proper file permissions
sudo chown -R pynomaly:pynomaly /opt/pynomaly
sudo chmod -R 750 /opt/pynomaly
sudo chmod 600 /opt/pynomaly/.env.production

# Secure database files
sudo chmod 600 /var/lib/postgresql/14/main/postgresql.conf
sudo chmod 600 /var/lib/postgresql/14/main/pg_hba.conf
```

### 3. Network Security
```bash
# Configure iptables (if not using ufw)
sudo iptables -A INPUT -i lo -j ACCEPT
sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP

# Save iptables rules
sudo iptables-save > /etc/iptables/rules.v4
```

## Backup & Disaster Recovery

### 1. Database Backup
```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup_db.sh

BACKUP_DIR="/opt/backups/postgres"
DB_NAME="pynomaly_prod"
DB_USER="pynomaly_user"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME | gzip > $BACKUP_DIR/pynomaly_backup_$TIMESTAMP.sql.gz

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/pynomaly_backup_$TIMESTAMP.sql.gz s3://pynomaly-backups/

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### 2. Application Backup
```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup_app.sh

BACKUP_DIR="/opt/backups/application"
APP_DIR="/opt/pynomaly"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf $BACKUP_DIR/pynomaly_app_$TIMESTAMP.tar.gz $APP_DIR

# Upload to S3
aws s3 cp $BACKUP_DIR/pynomaly_app_$TIMESTAMP.tar.gz s3://pynomaly-backups/

# Clean up old backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 3. Automated Backup Schedule
```bash
# Add to crontab
sudo crontab -e

# Database backup every 6 hours
0 */6 * * * /opt/pynomaly/scripts/backup_db.sh

# Application backup daily at 2 AM
0 2 * * * /opt/pynomaly/scripts/backup_app.sh

# Log rotation
0 3 * * * /usr/sbin/logrotate /etc/logrotate.d/pynomaly
```

## CI/CD Pipeline

### 1. GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

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
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src/pynomaly --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security scan
        run: |
          pip install bandit safety
          bandit -r src/
          safety check

  deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          # Build Docker image
          docker build -t pynomaly:${{ github.sha }} .
          
          # Push to registry
          docker tag pynomaly:${{ github.sha }} ${{ secrets.DOCKER_REGISTRY }}/pynomaly:${{ github.sha }}
          docker push ${{ secrets.DOCKER_REGISTRY }}/pynomaly:${{ github.sha }}
          
          # Deploy to servers
          ssh ${{ secrets.PROD_SERVER }} "docker pull ${{ secrets.DOCKER_REGISTRY }}/pynomaly:${{ github.sha }}"
          ssh ${{ secrets.PROD_SERVER }} "docker-compose -f /opt/pynomaly/docker-compose.prod.yml up -d"
```

### 2. Deployment Script
```bash
#!/bin/bash
# /opt/pynomaly/scripts/deploy.sh

set -e

IMAGE_TAG=$1
if [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <image_tag>"
    exit 1
fi

echo "Deploying Pynomaly version $IMAGE_TAG..."

# Pull new image
docker pull pynomaly:$IMAGE_TAG

# Update docker-compose file
sed -i "s/pynomaly:latest/pynomaly:$IMAGE_TAG/g" /opt/pynomaly/docker-compose.prod.yml

# Run database migrations
docker run --rm --network pynomaly_pynomaly-network \
  -e DATABASE_URL=$DATABASE_URL \
  pynomaly:$IMAGE_TAG python -m alembic upgrade head

# Rolling update
docker-compose -f /opt/pynomaly/docker-compose.prod.yml up -d --no-deps pynomaly-app

# Health check
sleep 30
curl -f http://localhost/health || exit 1

echo "Deployment completed successfully!"
```

## Performance Optimization

### 1. Application Configuration
```python
# src/pynomaly/infrastructure/config/production.py
import os
from typing import Dict, Any

class ProductionConfig:
    # Database connection pool
    DATABASE_POOL_SIZE = 20
    DATABASE_MAX_OVERFLOW = 30
    DATABASE_POOL_TIMEOUT = 30
    DATABASE_POOL_RECYCLE = 3600
    
    # Redis connection pool
    REDIS_MAX_CONNECTIONS = 50
    REDIS_SOCKET_KEEPALIVE = True
    REDIS_SOCKET_KEEPALIVE_OPTIONS = {
        "TCP_KEEPINTVL": 1,
        "TCP_KEEPCNT": 3,
        "TCP_KEEPIDLE": 1
    }
    
    # Caching settings
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_REDIS_DB = 1
    
    # Rate limiting
    RATE_LIMIT_STORAGE_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    @classmethod
    def get_database_url(cls) -> str:
        return os.getenv("DATABASE_URL", "postgresql://localhost/pynomaly_prod")
    
    @classmethod
    def get_redis_url(cls) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
```

### 2. Database Optimization
```sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX CONCURRENTLY idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX CONCURRENTLY idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX CONCURRENTLY idx_api_keys_key_hash ON api_keys(key_hash);

-- Partial indexes for active records
CREATE INDEX CONCURRENTLY idx_users_active ON users(id) WHERE is_active = true;
CREATE INDEX CONCURRENTLY idx_api_keys_active ON api_keys(id) WHERE is_active = true;

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_audit_logs_user_timestamp ON audit_logs(user_id, timestamp);
CREATE INDEX CONCURRENTLY idx_users_email_active ON users(email, is_active);
```

### 3. Caching Strategy
```python
# src/pynomaly/infrastructure/cache/redis_cache.py
import json
import pickle
from typing import Any, Optional
from redis import Redis
from src.pynomaly.infrastructure.config.production import ProductionConfig

class RedisCache:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_timeout = ProductionConfig.CACHE_DEFAULT_TIMEOUT
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception:
            return None
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            timeout = timeout or self.default_timeout
            serialized = pickle.dumps(value)
            return self.redis.setex(key, timeout, serialized)
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return self.redis.delete(key) > 0
        except Exception:
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception:
            return 0
```

## Troubleshooting

### 1. Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check application memory
docker stats pynomaly-app

# Solutions:
# - Increase server memory
# - Optimize database queries
# - Implement connection pooling
# - Add memory limits to containers
```

#### Database Connection Issues
```bash
# Check database connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check connection limits
sudo -u postgres psql -c "SHOW max_connections;"

# Monitor slow queries
sudo -u postgres psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

#### Redis Connection Issues
```bash
# Check Redis status
redis-cli ping

# Monitor Redis
redis-cli monitor

# Check memory usage
redis-cli info memory
```

### 2. Debugging Tools

#### Application Logs
```bash
# View application logs
docker logs pynomaly-app

# Real-time monitoring
tail -f /var/log/pynomaly/error.log

# Search logs
grep "ERROR" /var/log/pynomaly/error.log | tail -20
```

#### Performance Monitoring
```bash
# Check system performance
htop
iotop
nethogs

# Check application performance
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost/api/v1/health"
```

### 3. Recovery Procedures

#### Database Recovery
```bash
# Restore from backup
gunzip -c /opt/backups/postgres/pynomaly_backup_20240101_120000.sql.gz | psql -U pynomaly_user -d pynomaly_prod

# Point-in-time recovery
pg_basebackup -h localhost -D /var/lib/postgresql/recovery -U postgres -P -W
```

#### Application Recovery
```bash
# Restart application
docker-compose -f /opt/pynomaly/docker-compose.prod.yml restart pynomaly-app

# Rollback to previous version
docker-compose -f /opt/pynomaly/docker-compose.prod.yml down
sed -i "s/pynomaly:current/pynomaly:previous/g" /opt/pynomaly/docker-compose.prod.yml
docker-compose -f /opt/pynomaly/docker-compose.prod.yml up -d
```

## Maintenance

### 1. Regular Tasks

#### Daily Tasks
```bash
#!/bin/bash
# /opt/pynomaly/scripts/daily_maintenance.sh

# Check disk space
df -h | grep -E "9[0-9]%" && echo "ALERT: Disk space critical"

# Check logs for errors
grep -i error /var/log/pynomaly/error.log | tail -10

# Backup verification
test -f /opt/backups/postgres/pynomaly_backup_$(date +%Y%m%d)*.sql.gz || echo "ALERT: No backup found for today"

# Health check
curl -f http://localhost/health || echo "ALERT: Health check failed"
```

#### Weekly Tasks
```bash
#!/bin/bash
# /opt/pynomaly/scripts/weekly_maintenance.sh

# Update system packages
sudo apt update && sudo apt list --upgradable

# Analyze database performance
sudo -u postgres psql -d pynomaly_prod -c "SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables;"

# Check SSL certificate expiry
openssl x509 -in /etc/letsencrypt/live/your-domain.com/cert.pem -text -noout | grep "Not After"

# Clean old log files
find /var/log/pynomaly -name "*.log.*" -mtime +30 -delete
```

#### Monthly Tasks
```bash
#!/bin/bash
# /opt/pynomaly/scripts/monthly_maintenance.sh

# Database maintenance
sudo -u postgres psql -d pynomaly_prod -c "VACUUM ANALYZE;"

# Update dependencies
pip list --outdated

# Security audit
sudo apt install -y unattended-upgrades
sudo unattended-upgrade --dry-run

# Review and rotate API keys
python /opt/pynomaly/scripts/audit_api_keys.py
```

### 2. Monitoring Checklist

#### System Health
- [ ] CPU usage < 80%
- [ ] Memory usage < 80%
- [ ] Disk space < 80%
- [ ] Network latency < 100ms
- [ ] SSL certificates valid for > 30 days

#### Application Health
- [ ] All services running
- [ ] Database connections < 80% of limit
- [ ] Redis memory usage < 80%
- [ ] Error rate < 1%
- [ ] Response time < 1s (95th percentile)

#### Security Health
- [ ] No failed login attempts > threshold
- [ ] All security updates installed
- [ ] Firewall rules up to date
- [ ] Backup integrity verified
- [ ] Monitoring alerts functional

### 3. Scaling Procedures

#### Horizontal Scaling
```bash
# Add new application server
docker-compose -f /opt/pynomaly/docker-compose.prod.yml up -d --scale pynomaly-app=4

# Update load balancer
# Add new server to upstream block in nginx.conf
# Reload nginx configuration
sudo nginx -t && sudo nginx -s reload
```

#### Vertical Scaling
```bash
# Increase server resources
# Stop application
docker-compose -f /opt/pynomaly/docker-compose.prod.yml stop

# Resize server (cloud provider specific)
# Update docker-compose resources
# Restart application
docker-compose -f /opt/pynomaly/docker-compose.prod.yml up -d
```

## Conclusion

This production deployment guide provides a comprehensive framework for deploying Pynomaly in a production environment. Regular monitoring, maintenance, and updates are essential for maintaining a secure, performant, and reliable service.

For additional support or questions, please refer to the project documentation or contact the development team.

---

**Last Updated**: $(date)  
**Version**: 1.0  
**Maintainer**: Pynomaly Development Team