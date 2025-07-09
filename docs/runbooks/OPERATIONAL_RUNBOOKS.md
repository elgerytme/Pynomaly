# Pynomaly Operational Runbooks

This document contains comprehensive operational runbooks for managing Pynomaly in production.

## Table of Contents
1. [System Overview](#system-overview)
2. [Common Operations](#common-operations)
3. [Troubleshooting Guides](#troubleshooting-guides)
4. [Emergency Procedures](#emergency-procedures)
5. [Maintenance Tasks](#maintenance-tasks)
6. [Monitoring and Alerting](#monitoring-and-alerting)

---

## System Overview

### Architecture Components
- **API Service**: Main FastAPI application (`pynomaly-api`)
- **Database**: PostgreSQL (`postgres`)
- **Cache**: Redis (`redis-cluster`)
- **Monitoring**: Prometheus + Grafana
- **Reverse Proxy**: Nginx
- **Load Balancer**: Docker Swarm / Kubernetes

### Key Directories
- `/app/models` - ML model artifacts
- `/app/logs` - Application logs
- `/app/config` - Configuration files
- `/backups` - Backup files
- `/etc/nginx` - Nginx configuration

---

## Common Operations

### 1. Starting the System

```bash
# Start all services
docker-compose -f docker-compose.simple.yml up -d

# Check service status
docker-compose -f docker-compose.simple.yml ps

# View logs
docker-compose -f docker-compose.simple.yml logs -f pynomaly-api
```

### 2. Stopping the System

```bash
# Stop all services
docker-compose -f docker-compose.simple.yml down

# Stop with volume cleanup
docker-compose -f docker-compose.simple.yml down -v

# Stop specific service
docker-compose -f docker-compose.simple.yml stop pynomaly-api
```

### 3. Scaling Services

```bash
# Scale API service
docker-compose -f docker-compose.simple.yml up -d --scale pynomaly-api=3

# Check scaled services
docker-compose -f docker-compose.simple.yml ps pynomaly-api
```

### 4. Updating the Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.simple.yml build pynomaly-api
docker-compose -f docker-compose.simple.yml up -d pynomaly-api

# Verify deployment
curl http://localhost:8000/health
```

### 5. Configuration Updates

```bash
# Update configuration
vi config/production.yml

# Restart affected services
docker-compose -f docker-compose.simple.yml restart pynomaly-api

# Verify configuration
docker-compose -f docker-compose.simple.yml exec pynomaly-api python -c "import yaml; print('Config OK')"
```

---

## Troubleshooting Guides

### API Service Issues

#### Problem: API Service Won't Start
**Symptoms**: Container exits immediately, health checks fail
**Diagnosis**:
```bash
# Check container logs
docker-compose -f docker-compose.simple.yml logs pynomaly-api

# Check container status
docker-compose -f docker-compose.simple.yml ps pynomaly-api

# Check configuration
docker-compose -f docker-compose.simple.yml exec pynomaly-api python -c "
import os
print('Environment:', os.getenv('ENVIRONMENT'))
print('Database URL:', os.getenv('DATABASE_URL'))
"
```

**Solutions**:
1. **Configuration Error**: Check `.env` file and `config/production.yml`
2. **Database Connection**: Verify PostgreSQL is running and accessible
3. **Port Conflicts**: Check if port 8000 is already in use
4. **Missing Dependencies**: Rebuild container with `docker-compose build`

#### Problem: High Response Times
**Symptoms**: API responses > 1000ms, timeouts
**Diagnosis**:
```bash
# Check current performance
curl -w "@curl-format.txt" http://localhost:8000/health

# Monitor resource usage
docker stats pynomaly-api

# Check database connections
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "SELECT * FROM pg_stat_activity;"
```

**Solutions**:
1. **Database Optimization**: Run `VACUUM ANALYZE` on PostgreSQL
2. **Connection Pooling**: Increase database pool size in configuration
3. **Caching**: Enable Redis caching for frequently accessed data
4. **Resource Scaling**: Increase container CPU/memory limits

#### Problem: High Error Rate
**Symptoms**: 5xx errors, application crashes
**Diagnosis**:
```bash
# Check error logs
docker-compose -f docker-compose.simple.yml logs pynomaly-api | grep ERROR

# Check system resources
docker stats

# Check database health
docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U pynomaly -d pynomaly_prod
```

**Solutions**:
1. **Application Errors**: Check logs for specific error messages
2. **Resource Exhaustion**: Scale horizontally or increase resources
3. **Database Issues**: Check PostgreSQL logs and connections
4. **Network Issues**: Verify network connectivity between services

### Database Issues

#### Problem: Database Connection Failures
**Symptoms**: "Connection refused", "Database not responding"
**Diagnosis**:
```bash
# Check database container
docker-compose -f docker-compose.simple.yml ps postgres

# Check database connectivity
docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U pynomaly -d pynomaly_prod

# Check database logs
docker-compose -f docker-compose.simple.yml logs postgres
```

**Solutions**:
1. **Container Not Running**: Start PostgreSQL container
2. **Authentication Issues**: Check credentials in `.env` file
3. **Network Issues**: Verify Docker network connectivity
4. **Database Corruption**: Restore from backup

#### Problem: Slow Database Queries
**Symptoms**: High query times, connection timeouts
**Diagnosis**:
```bash
# Check slow queries
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Check database size
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT pg_size_pretty(pg_database_size('pynomaly_prod'));
"
```

**Solutions**:
1. **Missing Indexes**: Add indexes to frequently queried columns
2. **Table Bloat**: Run `VACUUM FULL` on affected tables
3. **Query Optimization**: Optimize slow queries identified above
4. **Connection Limits**: Increase `max_connections` in PostgreSQL config

### Redis Issues

#### Problem: Redis Connection Failures
**Symptoms**: Cache misses, connection errors
**Diagnosis**:
```bash
# Check Redis container
docker-compose -f docker-compose.simple.yml ps redis-cluster

# Test Redis connectivity
docker-compose -f docker-compose.simple.yml exec redis-cluster redis-cli ping

# Check Redis logs
docker-compose -f docker-compose.simple.yml logs redis-cluster
```

**Solutions**:
1. **Container Not Running**: Start Redis container
2. **Memory Issues**: Check Redis memory usage and configure limits
3. **Configuration Issues**: Verify Redis configuration file
4. **Network Issues**: Check Docker network connectivity

---

## Emergency Procedures

### System Outage Response

#### 1. Immediate Assessment (0-5 minutes)
```bash
# Check overall system status
docker-compose -f docker-compose.simple.yml ps

# Check external availability
curl -I http://localhost:8000/health

# Check monitoring dashboard
# Access Grafana at http://localhost:3000
```

#### 2. Service Recovery (5-15 minutes)
```bash
# Restart all services
docker-compose -f docker-compose.simple.yml restart

# Check logs for errors
docker-compose -f docker-compose.simple.yml logs --tail=100

# Verify recovery
curl http://localhost:8000/health
```

#### 3. Root Cause Analysis (15-30 minutes)
```bash
# Analyze logs for root cause
docker-compose -f docker-compose.simple.yml logs --since=1h | grep ERROR

# Check system resources
docker system df
df -h

# Check database integrity
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "SELECT 1;"
```

### Data Recovery

#### Database Recovery
```bash
# Stop application
docker-compose -f docker-compose.simple.yml stop pynomaly-api

# Find latest backup
ls -la /backups/database/

# Restore from backup
docker-compose -f docker-compose.simple.yml exec -T postgres psql -U pynomaly -d pynomaly_prod < /backups/database/latest_backup.sql

# Restart application
docker-compose -f docker-compose.simple.yml start pynomaly-api
```

#### Configuration Recovery
```bash
# Restore configuration from backup
tar -xzf /backups/config/latest_config_backup.tar.gz

# Apply configuration
docker-compose -f docker-compose.simple.yml restart

# Verify configuration
docker-compose -f docker-compose.simple.yml exec pynomaly-api python -c "print('Config restored')"
```

### Security Incident Response

#### 1. Immediate Containment
```bash
# Block suspicious IP addresses
# Add to nginx configuration or firewall rules
iptables -A INPUT -s SUSPICIOUS_IP -j DROP

# Enable maintenance mode
# Update nginx configuration to return 503
```

#### 2. Investigation
```bash
# Check access logs
docker-compose -f docker-compose.simple.yml logs nginx | grep SUSPICIOUS_IP

# Check application logs for unusual activity
docker-compose -f docker-compose.simple.yml logs pynomaly-api | grep -i "auth\|error\|fail"

# Review database logs
docker-compose -f docker-compose.simple.yml logs postgres | grep -i "error\|fail"
```

#### 3. Recovery
```bash
# Reset passwords if compromised
# Update .env file with new credentials

# Restart all services with new configuration
docker-compose -f docker-compose.simple.yml down
docker-compose -f docker-compose.simple.yml up -d

# Verify system integrity
curl http://localhost:8000/health
```

---

## Maintenance Tasks

### Daily Tasks

#### System Health Check
```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Daily Health Check $(date) ==="

# Check service status
docker-compose -f docker-compose.simple.yml ps

# Check disk usage
df -h

# Check memory usage
free -h

# Check database health
docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U pynomaly -d pynomaly_prod

# Check API health
curl -f http://localhost:8000/health || echo "API health check failed"

# Check logs for errors
docker-compose -f docker-compose.simple.yml logs --since=24h | grep ERROR | wc -l
```

#### Log Rotation
```bash
# Rotate application logs
docker-compose -f docker-compose.simple.yml exec pynomaly-api logrotate /etc/logrotate.d/pynomaly

# Clean old container logs
docker system prune --filter "until=24h"
```

### Weekly Tasks

#### Database Maintenance
```bash
#!/bin/bash
# weekly_db_maintenance.sh

echo "=== Weekly Database Maintenance $(date) ==="

# Update database statistics
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "ANALYZE;"

# Check database size
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Vacuum database
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "VACUUM ANALYZE;"
```

#### Security Scan
```bash
#!/bin/bash
# weekly_security_scan.sh

echo "=== Weekly Security Scan $(date) ==="

# Check for container vulnerabilities
docker scan pynomaly:production

# Check for outdated packages
docker-compose -f docker-compose.simple.yml exec pynomaly-api pip list --outdated

# Check SSL certificate expiry
openssl x509 -enddate -noout -in /etc/ssl/certs/pynomaly.crt
```

### Monthly Tasks

#### Performance Review
```bash
#!/bin/bash
# monthly_performance_review.sh

echo "=== Monthly Performance Review $(date) ==="

# Run performance tests
python scripts/performance_testing.py

# Generate performance report
python scripts/generate_performance_report.py

# Check database performance
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 20;
"
```

#### Backup Verification
```bash
#!/bin/bash
# monthly_backup_verification.sh

echo "=== Monthly Backup Verification $(date) ==="

# Test database backup restoration
python scripts/backup_recovery.py

# Verify backup integrity
for backup in /backups/database/*.sql.gz; do
    echo "Verifying $backup"
    gunzip -t "$backup" && echo "OK" || echo "FAILED"
done

# Check backup storage usage
du -sh /backups/*
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- Response time (< 500ms average)
- Throughput (> 100 RPS)
- Error rate (< 1%)
- Active connections
- Memory usage
- CPU utilization

#### Database Metrics
- Connection count
- Query execution time
- Lock waits
- Database size
- Replication lag (if applicable)

#### System Metrics
- Disk usage (< 80%)
- Memory usage (< 85%)
- CPU usage (< 80%)
- Network I/O
- Container restarts

### Alert Thresholds

#### Critical Alerts (Immediate Response)
- API service down
- Database unavailable
- Disk usage > 95%
- Memory usage > 95%
- Error rate > 5%

#### Warning Alerts (Response within 1 hour)
- High response time (> 1000ms)
- High error rate (> 1%)
- High resource usage (> 80%)
- SSL certificate expiring (< 30 days)

### Grafana Dashboard URLs
- **Main Dashboard**: http://localhost:3000/d/pynomaly-main
- **Real-time Monitoring**: http://localhost:3000/d/pynomaly-realtime
- **System Resources**: http://localhost:3000/d/pynomaly-system

### Log Locations
- **Application Logs**: `docker-compose logs pynomaly-api`
- **Database Logs**: `docker-compose logs postgres`
- **Nginx Logs**: `docker-compose logs nginx`
- **System Logs**: `/var/log/syslog`

---

## Contact Information

### Escalation Procedures
1. **Level 1**: On-call engineer
2. **Level 2**: Senior engineer
3. **Level 3**: Engineering manager
4. **Level 4**: CTO

### Emergency Contacts
- **On-call**: +1-555-ONCALL
- **Engineering**: engineering@pynomaly.com
- **Operations**: ops@pynomaly.com
- **Security**: security@pynomaly.com

### External Resources
- **Documentation**: https://docs.pynomaly.com
- **Status Page**: https://status.pynomaly.com
- **Support**: https://support.pynomaly.com
- **GitHub**: https://github.com/pynomaly/pynomaly

---

## Runbook Maintenance

This runbook should be updated:
- After major system changes
- When new procedures are discovered
- After incident post-mortems
- At least quarterly

**Last Updated**: {current_date}
**Next Review**: {next_review_date}
**Owner**: Operations Team
**Reviewers**: Engineering Team, SRE Team