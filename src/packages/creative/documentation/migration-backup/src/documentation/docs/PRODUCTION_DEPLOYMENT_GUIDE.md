# Pynomaly Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Pynomaly in a production environment with enterprise-grade features, monitoring, and security.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Production Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│  Load Balancer (NGINX/HAProxy)                                │
│  ├── API Gateway (Rate Limiting, Authentication)              │
│  ├── Web Application (FastAPI)                                │
│  ├── Background Workers (Celery/RQ)                           │
│  └── CLI Interface                                             │
├─────────────────────────────────────────────────────────────────┤
│  Application Services                                           │
│  ├── Anomaly Detection Engine                                  │
│  ├── Model Management                                          │
│  ├── Data Processing Pipeline                                  │
│  └── Real-time Streaming                                       │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                     │
│  ├── PostgreSQL (Primary Database)                             │
│  ├── Redis (Caching & Sessions)                                │
│  ├── MongoDB (Document Storage)                                │
│  └── Object Storage (S3/MinIO)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                    │
│  ├── Prometheus (Metrics Collection)                           │
│  ├── Grafana (Dashboards & Visualization)                      │
│  ├── Alertmanager (Alert Management)                           │
│  └── Jaeger (Distributed Tracing)                              │
├─────────────────────────────────────────────────────────────────┤
│  Security & Compliance                                          │
│  ├── Authentication & Authorization                             │
│  ├── Audit Logging                                             │
│  ├── Data Encryption                                           │
│  └── Security Scanning                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 100GB+ SSD
- **Network**: 1Gbps+ bandwidth

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- NGINX (for load balancing)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/elgerytme/Pynomaly.git
cd Pynomaly

# Set up environment
make setup
make dev-install
```

### 2. Configure Environment

```bash
# Copy and configure environment variables
cp .env.example .env.production

# Edit production configuration
nano .env.production
```

### 3. Deploy with Docker Compose

```bash
# Deploy full production stack
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose ps
```

### 4. Set up Monitoring

```bash
# Start monitoring services
./scripts/setup_monitoring.sh

# Verify monitoring
curl http://localhost:9090  # Prometheus
curl http://localhost:3000  # Grafana
```

## Production Configuration

### Environment Variables

```bash
# Application Configuration
PYNOMALY_ENV=production
PYNOMALY_DEBUG=false
PYNOMALY_LOG_LEVEL=INFO
PYNOMALY_API_HOST=0.0.0.0
PYNOMALY_API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://user:password@postgres:5432/pynomaly
REDIS_URL=redis://redis:6379/0

# Security Configuration
SECRET_KEY=your-super-secret-key-change-this
JWT_SECRET_KEY=your-jwt-secret-key

# Monitoring Configuration
PROMETHEUS_ENABLED=true
METRICS_ENDPOINT=/metrics
HEALTH_CHECK_ENDPOINT=/health
```

### Database Setup

```sql
-- Create database and user
CREATE DATABASE pynomaly;
CREATE USER pynomaly_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE pynomaly TO pynomaly_user;
```

### Security Configuration

```bash
# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Run security scan
bandit -r src/pynomaly/
safety check
```

## Monitoring and Alerting

### Prometheus Metrics

The system exposes the following metrics:

- `pynomaly_detections_total` - Total number of detections
- `pynomaly_detection_duration_seconds` - Detection processing time
- `pynomaly_model_accuracy` - Model accuracy metrics
- `pynomaly_system_health_score` - Overall system health

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin123) to view:

- System health overview
- Detection performance metrics
- Model performance tracking
- Resource usage monitoring

### Alerting Rules

Critical alerts are configured for:

- System down conditions
- High error rates (>5%)
- Memory usage (>80%)
- Model performance degradation

## Backup and Recovery

### Automated Backup

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/pynomaly"

# PostgreSQL backup
pg_dump -h localhost -U pynomaly_user -d pynomaly > "$BACKUP_DIR/pynomaly_$DATE.sql"

# Redis backup
redis-cli --rdb "$BACKUP_DIR/redis_$DATE.rdb"
```

### Disaster Recovery

```bash
# Restore PostgreSQL
psql -h localhost -U pynomaly_user -d pynomaly < "$BACKUP_FILE"

# Restore Redis
redis-cli --rdb "$BACKUP_FILE"
```

## Performance Optimization

### Database Optimization

```sql
-- Create indexes for common queries
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_detections_model_id ON detections(model_id);
CREATE INDEX idx_anomalies_severity ON anomalies(severity);
```

### Application Optimization

- Enable connection pooling
- Configure Redis caching
- Optimize model loading
- Use async processing for heavy operations

## Troubleshooting

### Common Issues

1. **Application Won't Start**
   ```bash
   docker-compose logs app
   ```

2. **High Memory Usage**
   ```bash
   docker stats
   ```

3. **Database Connection Issues**
   ```bash
   docker-compose exec postgres pg_isready
   ```

### Log Analysis

```bash
# Application logs
docker-compose logs -f app

# Database logs
docker-compose logs -f postgres
```

## Maintenance

### Regular Tasks

1. **Weekly**
   - Review monitoring dashboards
   - Check backup integrity
   - Update dependencies

2. **Monthly**
   - Database maintenance
   - Security reviews
   - Performance optimization

3. **Quarterly**
   - Security audit
   - Disaster recovery testing
   - Capacity planning

## Security Checklist

- [ ] SSL/TLS certificates configured
- [ ] Firewall rules implemented
- [ ] Database credentials secured
- [ ] API rate limiting enabled
- [ ] Audit logging configured
- [ ] Security headers implemented
- [ ] Regular security scans scheduled
- [ ] Backup encryption enabled

## Support

For production support:

- **Documentation**: https://docs.pynomaly.com
- **Community**: https://github.com/elgerytme/Pynomaly/discussions
- **Issues**: https://github.com/elgerytme/Pynomaly/issues

## Conclusion

This production deployment guide provides a comprehensive setup for running Pynomaly in enterprise environments. Follow the security best practices and monitoring recommendations to ensure reliable operation.