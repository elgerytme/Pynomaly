# Production Deployment Guide

This comprehensive guide covers deploying Pynomaly to production environments with high availability, security, and monitoring capabilities.

## Quick Navigation

- [Production Checklist](production-checklist.md) - Pre-deployment validation checklist
- [Docker Deployment](docker.md) - Container-based deployment
- [Kubernetes Deployment](kubernetes.md) - Kubernetes orchestration
- [Security Guide](security.md) - Security hardening and best practices
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

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

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended (8+ for high throughput)
- **Memory**: 8GB minimum (16GB+ recommended)
- **Storage**: 100GB+ SSD storage
- **Network**: High-bandwidth network connectivity

### Software Dependencies

- Docker 20.10+ or Kubernetes 1.21+
- Python 3.9+ (if deploying without containers)
- PostgreSQL 13+ or MySQL 8.0+
- Redis 6.0+
- NGINX or similar reverse proxy

### Infrastructure Dependencies

- Load balancer (AWS ALB, Google Cloud Load Balancer, etc.)
- DNS management
- SSL certificate management
- Monitoring infrastructure (Prometheus, Grafana)
- Log aggregation (ELK stack, Fluentd)

## Architecture Overview

```
Internet -> Load Balancer -> Reverse Proxy -> Pynomaly App Instances
                                          -> Database (PostgreSQL)
                                          -> Cache (Redis)
                                          -> Monitoring Stack
```

### High-Level Components

1. **Load Balancer**: Distributes traffic across multiple app instances
2. **Reverse Proxy**: NGINX for SSL termination and request routing
3. **Application**: Multiple Pynomaly instances for high availability
4. **Database**: PostgreSQL with read replicas for scalability
5. **Cache**: Redis for session storage and caching
6. **Monitoring**: Prometheus + Grafana for observability

## Infrastructure Setup

### Cloud Provider Setup

#### AWS Deployment
```bash
# Create VPC and subnets
aws ec2 create-vpc --cidr-block 10.0.0.0/16
aws ec2 create-subnet --vpc-id vpc-xxx --cidr-block 10.0.1.0/24

# Create security groups
aws ec2 create-security-group --group-name pynomaly-app --description "Pynomaly Application"
aws ec2 authorize-security-group-ingress --group-id sg-xxx --protocol tcp --port 8000 --cidr 10.0.0.0/16
```

#### Google Cloud Deployment
```bash
# Create network and firewall rules
gcloud compute networks create pynomaly-network --subnet-mode regional
gcloud compute firewall-rules create pynomaly-app --network pynomaly-network --allow tcp:8000
```

#### Azure Deployment
```bash
# Create resource group and virtual network
az group create --name pynomaly-rg --location eastus
az network vnet create --resource-group pynomaly-rg --name pynomaly-vnet --address-prefix 10.0.0.0/16
```

### Container Orchestration

For detailed container deployment instructions, see:
- [Docker Deployment Guide](docker.md)
- [Kubernetes Deployment Guide](kubernetes.md)

## Database Configuration

### PostgreSQL Setup

#### Installation and Configuration
```sql
-- Create database and user
CREATE DATABASE pynomaly;
CREATE USER pynomaly_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE pynomaly TO pynomaly_user;

-- Optimize for production
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET max_connections = '200';
ALTER SYSTEM SET work_mem = '4MB';
```

#### High Availability Setup
```bash
# Setup streaming replication
echo "host replication replicator 10.0.1.0/24 md5" >> /etc/postgresql/13/main/pg_hba.conf
echo "wal_level = replica" >> /etc/postgresql/13/main/postgresql.conf
echo "max_wal_senders = 3" >> /etc/postgresql/13/main/postgresql.conf
```

### Database Migrations
```bash
# Run database migrations
export DATABASE_URL="postgresql://pynomaly_user:password@localhost/pynomaly"
python -m pynomaly.infrastructure.persistence.migrations migrate
```

## Redis Configuration

### Redis Setup for Caching and Sessions
```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis for production
echo "maxmemory 2gb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
echo "save 900 1" >> /etc/redis/redis.conf
```

### Redis Clustering (Optional)
```bash
# For high availability Redis setup
redis-cli --cluster create 10.0.1.10:7000 10.0.1.11:7000 10.0.1.12:7000 \
10.0.1.10:7001 10.0.1.11:7001 10.0.1.12:7001 --cluster-replicas 1
```

## Application Deployment

### Environment Configuration
```bash
# Production environment variables
export PYNOMALY_ENV=production
export DATABASE_URL="postgresql://user:pass@db-host:5432/pynomaly"
export REDIS_URL="redis://redis-host:6379/0"
export SECRET_KEY="your-secret-key-here"
export ALLOWED_HOSTS="your-domain.com,www.your-domain.com"
```

### Application Startup
```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Using Kubernetes
kubectl apply -f k8s/production/
```

### Health Checks
Configure health check endpoints:
- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe

## Load Balancer & Reverse Proxy

### NGINX Configuration
```nginx
upstream pynomaly_backend {
    server app1.internal:8000;
    server app2.internal:8000;
    server app3.internal:8000;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        access_log off;
        proxy_pass http://pynomaly_backend;
    }
}
```

## SSL/TLS Configuration

### Let's Encrypt Setup
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

### SSL Best Practices
- Use TLS 1.2+ only
- Enable HSTS headers
- Configure proper cipher suites
- Implement certificate pinning

## Monitoring & Logging

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pynomaly'
    static_configs:
      - targets: ['app1:8000', 'app2:8000', 'app3:8000']
    metrics_path: '/metrics'
```

### Log Aggregation
```bash
# Configure log shipping to centralized logging
echo "*.info @@log-aggregator:514" >> /etc/rsyslog.conf
```

### Application Metrics
Pynomaly exposes metrics at `/metrics` endpoint including:
- Request latency and throughput
- Database connection pool status
- Cache hit/miss rates
- ML model performance metrics

## Security Hardening

### Network Security
- Use VPC/VNET isolation
- Configure security groups/firewall rules
- Enable DDoS protection
- Implement WAF if needed

### Application Security
- Enable CSRF protection
- Configure CORS properly
- Implement rate limiting
- Use secure session configuration

### Infrastructure Security
- Regular security updates
- Vulnerability scanning
- Secrets management (Vault, AWS Secrets Manager)
- Network segmentation

## Backup & Disaster Recovery

### Database Backups
```bash
# Automated daily backups
pg_dump -h db-host -U pynomaly_user pynomaly | gzip > backup_$(date +%Y%m%d).sql.gz

# Point-in-time recovery setup
echo "archive_mode = on" >> /etc/postgresql/13/main/postgresql.conf
echo "archive_command = 'cp %p /backup/archive/%f'" >> /etc/postgresql/13/main/postgresql.conf
```

### Application Data Backups
- Model artifacts and trained models
- Configuration files
- User data and preferences

### Disaster Recovery Plan
1. Recovery Time Objective (RTO): 1 hour
2. Recovery Point Objective (RPO): 15 minutes
3. Multi-region deployment for geographic redundancy
4. Regular disaster recovery testing

## CI/CD Pipeline

### GitHub Actions Deployment
```yaml
name: Production Deploy
on:
  push:
    branches: [main]
    
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          docker build -t pynomaly:latest .
          docker push registry.com/pynomaly:latest
          kubectl set image deployment/pynomaly pynomaly=registry.com/pynomaly:latest
```

### Deployment Strategy
- Blue-green deployments for zero-downtime
- Canary releases for gradual rollouts
- Automated rollback on health check failures

## Performance Optimization

### Application Performance
- Enable connection pooling
- Configure appropriate worker processes
- Implement caching strategies
- Optimize database queries

### Infrastructure Performance
- Auto-scaling based on metrics
- Load balancer optimization
- CDN for static assets
- Database read replicas

### Monitoring Performance
Key metrics to track:
- Response time (p95, p99)
- Throughput (requests per second)
- Error rates
- Resource utilization

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker logs pynomaly-app
kubectl logs deployment/pynomaly

# Check configuration
python -c "from pynomaly.infrastructure.config import Settings; print(Settings().dict())"
```

#### Database Connection Issues
```bash
# Test database connectivity
psql -h db-host -U pynomaly_user -d pynomaly

# Check connection pool
SELECT * FROM pg_stat_activity WHERE application_name = 'pynomaly';
```

#### Performance Issues
```bash
# Check resource usage
top
htop
docker stats

# Monitor database performance
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

### Log Analysis
- Application logs: `/var/log/pynomaly/`
- Database logs: `/var/log/postgresql/`
- NGINX logs: `/var/log/nginx/`

### Support Resources
- [Troubleshooting Guide](troubleshooting.md)
- [Performance Tuning](performance-tuning.md)
- [Security Checklist](security-checklist.md)

## Next Steps

After successful deployment:

1. **Monitoring Setup**: Configure alerts and dashboards
2. **Performance Tuning**: Optimize based on actual usage patterns
3. **Security Audit**: Conduct security assessment
4. **Backup Testing**: Verify backup and recovery procedures
5. **Documentation**: Update runbooks and procedures

For specific deployment scenarios, see:
- [Docker Deployment](docker.md)
- [Kubernetes Deployment](kubernetes.md)
- [Cloud-Specific Guides](cloud-deployment/)

---
*This guide covers production deployment best practices. For development deployment, see the [Development Setup Guide](../developer-guides/development-setup.md).*