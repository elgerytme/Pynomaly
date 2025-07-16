# Pynomaly Production Deployment Guide

This guide provides comprehensive instructions for deploying Pynomaly in a production environment with full monitoring, security, and scalability features.

## 🚀 Quick Start

### Prerequisites

- Docker >= 20.10.0
- Docker Compose >= 2.0.0
- 10GB+ available disk space
- 4GB+ available RAM
- OpenSSL (for SSL certificate generation)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Pynomaly
chmod +x scripts/deploy.sh
```

### 2. Configure Environment

```bash
# Copy and edit environment configuration
cp config/production/.env.prod config/production/.env.prod.local

# Edit the configuration file
nano config/production/.env.prod.local
```

**Required Changes:**
- `SECRET_KEY`: Generate a secure 64-character string
- `DB_PASSWORD`: Set a strong database password
- `REDIS_PASSWORD`: Set a strong Redis password
- `GRAFANA_PASSWORD`: Set a strong Grafana password

### 3. Deploy

```bash
# Full production deployment
./scripts/deploy.sh deploy
```

### 4. Verify Deployment

```bash
# Check service health
./scripts/deploy.sh health

# View logs
./scripts/deploy.sh logs
```

## 🏗️ Architecture Overview

### Services

| Service | Port | Description |
|---------|------|-------------|
| `pynomaly-api` | 8000 | Main API application |
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Redis cache |
| `nginx` | 80/443 | Reverse proxy & load balancer |
| `grafana` | 3000 | Monitoring dashboard |
| `prometheus` | 9090 | Metrics collection |
| `loki` | 3100 | Log aggregation |
| `promtail` | 9080 | Log collection |

### Network Architecture

```
Internet → NGINX (SSL Termination) → Pynomaly API → Database/Cache
                    ↓
            Monitoring Stack (Grafana/Prometheus)
```

## 🔧 Configuration

### Environment Variables

#### Core Application
```env
# Application
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secure-secret-key-here
AUTH_ENABLED=true
JWT_EXPIRATION=3600
JWT_ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://pynomaly:password@postgres:5432/pynomaly
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Cache
REDIS_URL=redis://:password@redis:6379
REDIS_MAX_CONNECTIONS=100
```

#### Performance Settings
```env
# Performance
MAX_UPLOAD_SIZE=100MB
REQUEST_TIMEOUT=30
RATE_LIMIT=1000/minute
WORKER_CONNECTIONS=1000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/pynomaly.log
```

#### Security Configuration
```env
# Security Headers
SECURITY_HEADERS_ENABLED=true
HSTS_MAX_AGE=31536000
CSP_ENABLED=true
CORS_ORIGINS=https://yourdomain.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_STORAGE=redis
RATE_LIMIT_STRATEGY=sliding_window
RATE_LIMIT_PER_MINUTE=100
```

### SSL Configuration

#### Self-Signed Certificates (Development)
The deployment script automatically generates self-signed certificates for development/testing.

#### Production SSL Certificates
Replace the generated certificates with proper CA-signed certificates:

```bash
# Copy your SSL certificates
cp your-cert.pem config/production/ssl/cert.pem
cp your-key.pem config/production/ssl/key.pem
cp your-dhparam.pem config/production/ssl/dhparam.pem

# Restart NGINX
docker-compose -f config/production/docker-compose.prod.yml restart nginx
```

## 📊 Monitoring & Observability

### Grafana Dashboard

Access Grafana at `http://localhost:3000`

- **Default credentials**: admin / (configured password)
- **Pre-configured dashboards**: API metrics, infrastructure monitoring
- **Data sources**: Prometheus, Loki, PostgreSQL

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

**Key Metrics:**
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration
- `failed_login_attempts_total`: Authentication failures
- `security_events_total`: Security events
- `database_connection_pool_active`: Database connections

### Alerts

**Critical Alerts:**
- API service down
- Database connection failures
- High error rates (>5%)
- Security incidents
- SSL certificate expiration

**Warning Alerts:**
- High response times (>2s)
- High CPU usage (>80%)
- High memory usage (>1GB)
- Database connection pool usage (>80%)

### Log Management

**Log Aggregation with Loki:**
- Application logs: JSON structured
- Audit logs: Security events
- NGINX logs: Access and error logs
- Database logs: Query performance

**Log Retention:**
- Application logs: 30 days
- Audit logs: 7 years (compliance)
- Error logs: 90 days

## 🛡️ Security Features

### Authentication & Authorization

- **JWT Authentication**: Secure token-based auth
- **Role-Based Access Control**: Admin, user, viewer roles
- **API Key Management**: Secure API access
- **Session Management**: Session tracking and invalidation

### Security Monitoring

- **Audit Logging**: All security events logged
- **Failed Login Tracking**: Brute force protection
- **SQL Injection Protection**: Input validation
- **XSS Protection**: Content security policies
- **Rate Limiting**: DDoS protection

### Data Protection

- **Encryption at Rest**: Database encryption
- **Encryption in Transit**: TLS/SSL everywhere
- **Secret Management**: Secure credential storage
- **Data Retention**: Compliance with regulations

## 🔄 Backup & Recovery

### Automated Backups

```bash
# Manual backup
./scripts/deploy.sh backup

# Automated daily backups (configured in docker-compose)
# Backups stored in /app/backups/
```

### Backup Strategy

- **Database**: Daily full backups + transaction log backups
- **Redis**: RDB snapshots every 6 hours
- **Application logs**: Archived to external storage
- **Configuration**: Version controlled

### Recovery Procedures

```bash
# Restore from backup
docker-compose -f config/production/docker-compose.prod.yml exec postgres \
  psql -U pynomaly -d pynomaly < /path/to/backup.sql

# Restart services
./scripts/deploy.sh restart
```

## 🚦 Health Checks

### Service Health Endpoints

- **API Health**: `https://localhost/health`
- **Database**: `https://localhost/api/v1/health/database`
- **Cache**: `https://localhost/api/v1/health/cache`
- **Full System**: `https://localhost/api/v1/health/`

### Health Check Script

```bash
# Run comprehensive health checks
./scripts/deploy.sh health
```

## 📈 Performance Optimization

### Resource Limits

```yaml
# Container resource limits
services:
  pynomaly-api:
    deploy:
      resources:
        limits:
          memory: 2g
          cpus: '1.5'
        reservations:
          memory: 1g
          cpus: '0.5'
```

### Database Optimization

```sql
-- PostgreSQL configuration
shared_preload_libraries = 'pg_stat_statements'
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

### Redis Optimization

```redis
# Redis configuration
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## 🔧 Maintenance Operations

### Service Management

```bash
# Start services
docker-compose -f config/production/docker-compose.prod.yml up -d

# Stop services
./scripts/deploy.sh stop

# Restart services
./scripts/deploy.sh restart

# View logs
./scripts/deploy.sh logs
```

### Database Maintenance

```bash
# Database vacuum and analyze
docker-compose -f config/production/docker-compose.prod.yml exec postgres \
  psql -U pynomaly -d pynomaly -c "VACUUM ANALYZE;"

# Reindex database
docker-compose -f config/production/docker-compose.prod.yml exec postgres \
  psql -U pynomaly -d pynomaly -c "REINDEX DATABASE pynomaly;"
```

### Log Rotation

```bash
# Rotate application logs
docker-compose -f config/production/docker-compose.prod.yml exec pynomaly-api \
  logrotate /app/config/logrotate.conf
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose -f config/production/docker-compose.prod.yml logs pynomaly-api

# Check resources
docker stats

# Check disk space
df -h
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose -f config/production/docker-compose.prod.yml exec postgres pg_isready

# Check connections
docker-compose -f config/production/docker-compose.prod.yml exec postgres \
  psql -U pynomaly -d pynomaly -c "SELECT count(*) FROM pg_stat_activity;"
```

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Clear Redis cache
docker-compose -f config/production/docker-compose.prod.yml exec redis redis-cli FLUSHALL
```

### Log Analysis

```bash
# Search for errors
docker-compose -f config/production/docker-compose.prod.yml logs pynomaly-api | grep ERROR

# Monitor real-time logs
docker-compose -f config/production/docker-compose.prod.yml logs -f pynomaly-api
```

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Database passwords changed
- [ ] Firewall rules configured
- [ ] DNS records configured
- [ ] Backup strategy tested

### Post-Deployment

- [ ] Health checks passing
- [ ] Monitoring dashboard accessible
- [ ] SSL certificate valid
- [ ] Backup process working
- [ ] Alerting configured
- [ ] Documentation updated

### Security Checklist

- [ ] Default passwords changed
- [ ] SSL/TLS enabled
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Security headers configured
- [ ] CORS properly configured
- [ ] API documentation secured

## 🆘 Support

### Getting Help

1. Check the troubleshooting section above
2. Review application logs
3. Check monitoring dashboards
4. Consult the API documentation
5. Create a support ticket with:
   - Error messages
   - Log excerpts
   - Environment details
   - Steps to reproduce

### Emergency Procedures

#### System Down
```bash
# Quick restart
./scripts/deploy.sh restart

# Full redeploy
./scripts/deploy.sh deploy
```

#### Data Loss
```bash
# Restore from backup
./scripts/deploy.sh backup
# Follow recovery procedures above
```

#### Security Incident
```bash
# Check audit logs
docker-compose -f config/production/docker-compose.prod.yml logs pynomaly-api | grep SECURITY

# Rotate secrets
# Update environment variables
# Restart services
```

## 📚 Additional Resources

- [API Documentation](https://localhost/docs)
- [Security Guidelines](../security/README.md)
- [Monitoring Guide](../monitoring/README.md)
- [Backup & Recovery](../backup/README.md)
- [Performance Tuning](../performance/README.md)