# Pynomaly Production Troubleshooting Guide

**Comprehensive guide for diagnosing and resolving production issues**

---

## 📋 Table of Contents

1. [Quick Diagnosis Tools](#quick-diagnosis-tools)
2. [Service Health Issues](#service-health-issues)
3. [Performance Problems](#performance-problems)
4. [Database Issues](#database-issues)
5. [Network and Connectivity](#network-and-connectivity)
6. [Security Issues](#security-issues)
7. [Data and Storage Problems](#data-and-storage-problems)
8. [Monitoring and Alerting Issues](#monitoring-and-alerting-issues)
9. [Emergency Procedures](#emergency-procedures)
10. [Contact Information](#contact-information)

---

## 🔍 Quick Diagnosis Tools

### System Health Check

```bash
# Quick system overview
./scripts/verify_production_deployment.sh

# Check all container status
docker-compose -f docker-compose.production.yml ps

# View recent logs from all services
docker-compose -f docker-compose.production.yml logs --tail=50

# Check system resources
docker stats --no-stream
df -h
free -m
```

### Health Endpoint Checks

```bash
# Core health checks
curl -f http://localhost/health
curl -f http://localhost/api/v1/health
curl -f http://localhost/metrics

# Service-specific health
curl -f http://localhost:9090/-/healthy  # Prometheus
curl -f http://localhost:3000/api/health # Grafana
```

### Log Analysis

```bash
# Search for errors in logs
docker-compose logs api | grep -i error | tail -20
docker-compose logs postgres | grep -i error | tail -20
docker-compose logs redis | grep -i error | tail -20

# Monitor logs in real-time
docker-compose logs -f api
```

---

## 🚨 Service Health Issues

### API Service Won't Start

**Symptoms:**

- API container exits immediately
- Health check endpoints return 502/503
- "Connection refused" errors

**Diagnosis Steps:**

```bash
# Check container status
docker-compose ps api

# View detailed logs
docker-compose logs api --tail=100

# Check configuration
docker-compose exec api python -c "
from src.pynomaly.infrastructure.config.settings import settings
print('Config validation:', settings)
"

# Test database connectivity
docker-compose exec api python -c "
from sqlalchemy import create_engine
from src.pynomaly.infrastructure.config.settings import settings
try:
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('Database connection: OK')
except Exception as e:
    print('Database error:', e)
"
```

**Common Solutions:**

1. **Environment Variables Missing:**

```bash
# Check environment file
cat .env.production | grep -v '^#' | grep -v '^$'

# Verify required variables
docker-compose config | grep -A 5 environment:
```

2. **Database Connection Issues:**

```bash
# Wait for database to be ready
until docker-compose exec postgres pg_isready -U pynomaly; do
    echo "Waiting for database..."
    sleep 2
done

# Restart API service
docker-compose restart api
```

3. **Memory/Resource Constraints:**

```bash
# Check available memory
free -m

# Increase container memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

### Database Service Issues

**Symptoms:**

- PostgreSQL container not starting
- Connection timeouts
- "Too many connections" errors

**Diagnosis Steps:**

```bash
# Check PostgreSQL status
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres --tail=50

# Test connectivity
docker-compose exec postgres pg_isready -U pynomaly -d pynomaly_prod

# Check connection count
docker-compose exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT count(*) as total_connections, state, application_name 
FROM pg_stat_activity 
GROUP BY state, application_name;
"
```

**Common Solutions:**

1. **Data Directory Permissions:**

```bash
# Fix PostgreSQL data directory permissions
sudo chown -R 999:999 ./data/postgres
sudo chmod 700 ./data/postgres
```

2. **Connection Limit Exceeded:**

```sql
-- Increase connection limit
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();

-- Kill long-running connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND query_start < NOW() - INTERVAL '1 hour';
```

3. **Disk Space Issues:**

```bash
# Check disk space
df -h

# Clean up PostgreSQL logs
docker-compose exec postgres find /var/log -name "*.log" -mtime +7 -delete
```

### Redis Service Issues

**Symptoms:**

- Redis container not responding
- Cache misses increasing
- Memory usage high

**Diagnosis Steps:**

```bash
# Check Redis status
docker-compose ps redis

# Test Redis connectivity
docker-compose exec redis redis-cli ping

# Check Redis info
docker-compose exec redis redis-cli info memory
docker-compose exec redis redis-cli info stats

# Monitor Redis operations
docker-compose exec redis redis-cli monitor
```

**Common Solutions:**

1. **Memory Issues:**

```bash
# Check memory usage
docker-compose exec redis redis-cli info memory

# Configure memory policy
docker-compose exec redis redis-cli config set maxmemory-policy allkeys-lru
docker-compose exec redis redis-cli config set maxmemory 1gb
```

2. **Connection Issues:**

```bash
# Check Redis configuration
docker-compose exec redis redis-cli config get "*"

# Restart Redis
docker-compose restart redis
```

---

## ⚡ Performance Problems

### High Response Times

**Symptoms:**

- API responses > 1 second
- Timeouts occurring
- Users reporting slow performance

**Diagnosis Steps:**

```bash
# Test API response times
time curl -f http://localhost/health
time curl -f http://localhost/api/v1/datasets

# Check system resources
top
iostat 1 5

# Analyze slow queries
docker-compose exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Check cache hit ratio
docker-compose exec redis redis-cli info stats | grep cache_hit
```

**Solutions:**

1. **Database Optimization:**

```sql
-- Update table statistics
ANALYZE;

-- Identify missing indexes
SELECT schemaname, tablename, attname, n_distinct
FROM pg_stats 
WHERE schemaname = 'public' 
AND n_distinct > 100;

-- Create performance indexes
CREATE INDEX CONCURRENTLY idx_performance_created_at 
ON your_table(created_at) 
WHERE created_at > NOW() - INTERVAL '30 days';
```

2. **Application Tuning:**

```python
# Increase connection pool size
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30

# Enable connection pooling
SQLALCHEMY_ENGINE_OPTIONS = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

3. **System Resources:**

```bash
# Scale API services horizontally
docker-compose up -d --scale api=3

# Increase memory limits
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
```

### High Memory Usage

**Symptoms:**

- Memory usage > 90%
- Out of memory errors
- Container restarts

**Diagnosis Steps:**

```bash
# Check memory usage by container
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Analyze memory usage patterns
free -m
cat /proc/meminfo

# Check for memory leaks
docker-compose exec api python -c "
import psutil
import gc
process = psutil.Process()
print(f'Memory before GC: {process.memory_info().rss / 1024 / 1024:.2f} MB')
gc.collect()
print(f'Memory after GC: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions:**

1. **Application Memory Tuning:**

```bash
# Set Python memory optimization
export PYTHONHASHSEED=0
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Tune garbage collection
export PYTHONMALLOC=malloc
```

2. **Container Resource Limits:**

```yaml
# In docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

3. **Memory Monitoring:**

```bash
# Set up memory alerts
# Add to prometheus rules:
- alert: HighMemoryUsage
  expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
  for: 10m
```

### High CPU Usage

**Symptoms:**

- CPU usage consistently > 80%
- API requests queuing up
- Slow response times

**Diagnosis Steps:**

```bash
# Check CPU usage
top -p $(pgrep -d',' -f python)
htop

# Check container CPU usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}"

# Profile application CPU usage
docker-compose exec api python -m cProfile -o /tmp/profile.out -c "
import time
time.sleep(10)
"
```

**Solutions:**

1. **Horizontal Scaling:**

```bash
# Scale API services
docker-compose up -d --scale api=5

# Update load balancer configuration
# Nginx upstream configuration
upstream pynomaly_api {
    server api_1:8000;
    server api_2:8000;
    server api_3:8000;
    server api_4:8000;
    server api_5:8000;
}
```

2. **CPU Optimization:**

```python
# Optimize async operations
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Use multiprocessing for CPU-intensive tasks
import multiprocessing
WORKERS = multiprocessing.cpu_count()
```

---

## 🗄️ Database Issues

### Connection Pool Exhaustion

**Symptoms:**

- "ConnectionPoolTimeout" errors
- New connections refused
- API requests failing

**Diagnosis Steps:**

```bash
# Check active connections
docker-compose exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT 
    count(*) as total_connections,
    state,
    application_name,
    client_addr
FROM pg_stat_activity 
GROUP BY state, application_name, client_addr
ORDER BY total_connections DESC;
"

# Check connection limits
docker-compose exec postgres psql -U pynomaly -d pynomaly_prod -c "
SHOW max_connections;
SELECT count(*) FROM pg_stat_activity;
"
```

**Solutions:**

1. **Increase Connection Limits:**

```sql
-- Increase PostgreSQL connections
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```

2. **Optimize Connection Pool:**

```python
# In application configuration
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
DATABASE_POOL_TIMEOUT = 30
DATABASE_POOL_RECYCLE = 3600
DATABASE_POOL_PRE_PING = True
```

3. **Clean Up Idle Connections:**

```sql
-- Kill idle connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND query_start < NOW() - INTERVAL '30 minutes';
```

### Slow Database Performance

**Symptoms:**

- Database queries taking > 1 second
- High database CPU usage
- Lock waits increasing

**Diagnosis Steps:**

```sql
-- Check slow queries
SELECT 
    query,
    mean_time,
    calls,
    total_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check for blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- Check table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;
```

**Solutions:**

1. **Index Optimization:**

```sql
-- Identify missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public' 
AND n_distinct > 100;

-- Create performance indexes
CREATE INDEX CONCURRENTLY idx_datasets_user_created 
ON datasets(user_id, created_at) 
WHERE active = true;

CREATE INDEX CONCURRENTLY idx_anomaly_detections_recent 
ON anomaly_detections(created_at) 
WHERE created_at > NOW() - INTERVAL '30 days';
```

2. **Database Maintenance:**

```sql
-- Update statistics
ANALYZE;

-- Vacuum tables
VACUUM ANALYZE;

-- Reindex if necessary
REINDEX DATABASE pynomaly_prod;
```

3. **Configuration Tuning:**

```sql
-- Tune PostgreSQL parameters
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
SELECT pg_reload_conf();
```

---

## 🌐 Network and Connectivity

### Load Balancer Issues

**Symptoms:**

- 502 Bad Gateway errors
- Uneven traffic distribution
- SSL certificate errors

**Diagnosis Steps:**

```bash
# Check Nginx status
docker-compose ps nginx

# Test Nginx configuration
docker-compose exec nginx nginx -t

# Check Nginx logs
docker-compose logs nginx --tail=50

# Test upstream servers
curl -H "Host: your-domain.com" http://localhost/health
```

**Solutions:**

1. **Nginx Configuration Issues:**

```bash
# Reload Nginx configuration
docker-compose exec nginx nginx -s reload

# Check upstream server health
docker-compose exec nginx cat /etc/nginx/conf.d/default.conf
```

2. **SSL Certificate Problems:**

```bash
# Check certificate validity
openssl x509 -in config/nginx/ssl/fullchain.pem -noout -dates

# Renew certificates
certbot renew --quiet
docker-compose restart nginx
```

### WebSocket Connection Issues

**Symptoms:**

- WebSocket connections failing
- Real-time features not working
- Connection drops frequently

**Diagnosis Steps:**

```bash
# Test WebSocket endpoint
wscat -c ws://localhost/ws/streaming

# Check WebSocket logs
docker-compose logs api | grep -i websocket

# Monitor WebSocket connections
docker-compose exec api python -c "
from src.pynomaly.presentation.websocket.manager import websocket_manager
print(f'Active connections: {len(websocket_manager.active_connections)}')
"
```

**Solutions:**

1. **Nginx WebSocket Configuration:**

```nginx
# Add to Nginx configuration
location /ws/ {
    proxy_pass http://pynomaly_api;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 86400;
}
```

2. **Application WebSocket Settings:**

```python
# Increase WebSocket timeout
WEBSOCKET_TIMEOUT = 60
WEBSOCKET_PING_INTERVAL = 20
WEBSOCKET_PING_TIMEOUT = 10
```

---

## 🔒 Security Issues

### Authentication Failures

**Symptoms:**

- Users cannot log in
- API returns 401 errors
- JWT token validation fails

**Diagnosis Steps:**

```bash
# Check authentication service logs
docker-compose logs api | grep -i auth

# Test authentication endpoint
curl -X POST http://localhost/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}'

# Verify JWT secret configuration
docker-compose exec api python -c "
from src.pynomaly.infrastructure.config.settings import settings
print('JWT secret configured:', bool(settings.jwt_secret_key))
"
```

**Solutions:**

1. **JWT Configuration:**

```bash
# Regenerate JWT secret
openssl rand -base64 32 > secrets/jwt_secret
docker-compose restart api
```

2. **Database User Issues:**

```sql
-- Check user accounts
SELECT id, username, email, is_active 
FROM users 
WHERE username = 'admin';

-- Reset user password
UPDATE users 
SET password_hash = '$2b$12$...' 
WHERE username = 'admin';
```

### Rate Limiting Issues

**Symptoms:**

- Legitimate requests being blocked
- 429 Too Many Requests errors
- Inconsistent rate limiting behavior

**Diagnosis Steps:**

```bash
# Check rate limiting logs
docker-compose logs nginx | grep -i rate

# Test rate limiting
for i in {1..20}; do
    curl -w "%{http_code}\n" -o /dev/null -s http://localhost/health
done

# Check Redis rate limiting data
docker-compose exec redis redis-cli keys "*rate_limit*"
```

**Solutions:**

1. **Adjust Rate Limits:**

```nginx
# Update Nginx rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=20r/s;
limit_req zone=api burst=40 nodelay;
```

2. **Application Rate Limiting:**

```python
# Adjust application rate limits
RATE_LIMIT_PER_MINUTE = 120
RATE_LIMIT_BURST = 20
```

---

## 💾 Data and Storage Problems

### Disk Space Issues

**Symptoms:**

- "No space left on device" errors
- Database write failures
- Log rotation not working

**Diagnosis Steps:**

```bash
# Check disk usage
df -h
du -sh /var/lib/docker/
du -sh ./data/
du -sh ./logs/

# Check Docker space usage
docker system df
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

**Solutions:**

1. **Clean Up Docker Resources:**

```bash
# Remove unused containers, networks, images
docker system prune -a --volumes

# Remove old Docker images
docker rmi $(docker images -f "dangling=true" -q)

# Clean up logs
find /var/log -name "*.log" -mtime +7 -delete
```

2. **Log Rotation:**

```bash
# Configure log rotation
cat > /etc/logrotate.d/pynomaly << EOF
/opt/pynomaly/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF
```

### Backup Failures

**Symptoms:**

- Backup scripts failing
- Old backups not being cleaned up
- Backup verification failures

**Diagnosis Steps:**

```bash
# Check backup script logs
cat /var/log/backup.log

# Test backup script manually
./scripts/backup_database.sh

# Verify backup integrity
gunzip -t /opt/backups/postgres/latest_backup.sql.gz
```

**Solutions:**

1. **Fix Backup Scripts:**

```bash
# Ensure backup directory exists
mkdir -p /opt/backups/postgres
chmod 755 /opt/backups/postgres

# Test database connectivity
docker-compose exec postgres pg_dump --version
```

2. **Automated Backup Monitoring:**

```bash
# Add backup monitoring to crontab
0 3 * * * /opt/pynomaly/scripts/backup_database.sh && echo "Backup successful" || echo "Backup failed" | mail -s "Backup Status" admin@company.com
```

---

## 📊 Monitoring and Alerting Issues

### Prometheus Not Collecting Metrics

**Symptoms:**

- Missing metrics in Grafana
- Prometheus targets down
- Alert rules not firing

**Diagnosis Steps:**

```bash
# Check Prometheus status
curl http://localhost:9090/-/healthy

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# View Prometheus configuration
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml

# Check Prometheus logs
docker-compose logs prometheus
```

**Solutions:**

1. **Fix Prometheus Configuration:**

```yaml
# Update prometheus.yml
scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

2. **Restart Prometheus:**

```bash
# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload

# Or restart container
docker-compose restart prometheus
```

### Grafana Dashboard Issues

**Symptoms:**

- Dashboards showing no data
- Data source connection errors
- Dashboard queries failing

**Diagnosis Steps:**

```bash
# Check Grafana logs
docker-compose logs grafana

# Test Grafana API
curl http://admin:password@localhost:3000/api/health

# Check data source configuration
curl http://admin:password@localhost:3000/api/datasources
```

**Solutions:**

1. **Fix Data Source Configuration:**

```bash
# Update Grafana data source
curl -X PUT http://admin:password@localhost:3000/api/datasources/1 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy"
  }'
```

---

## 🚨 Emergency Procedures

### Complete System Failure

**Immediate Actions:**

1. **Assess the Scope:**

```bash
# Check all services
docker-compose ps
systemctl status docker

# Check system resources
df -h
free -m
top
```

2. **Attempt Quick Recovery:**

```bash
# Restart all services
docker-compose down
docker-compose -f docker-compose.production.yml up -d

# Check logs for errors
docker-compose logs --tail=100
```

3. **If Quick Recovery Fails:**

```bash
# Enable maintenance mode
echo "System under maintenance" > /var/www/html/maintenance.html

# Implement emergency rollback
git checkout previous-stable-tag
docker-compose down
docker-compose -f docker-compose.production.yml up -d
```

### Data Corruption

**Immediate Actions:**

1. **Stop Data Writes:**

```bash
# Stop API services to prevent further corruption
docker-compose stop api worker
```

2. **Assess Damage:**

```bash
# Check database integrity
docker-compose exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT pg_database_size('pynomaly_prod');
"

# Validate recent backups
gunzip -t /opt/backups/postgres/latest_backup.sql.gz
```

3. **Recovery Process:**

```bash
# Restore from backup
./scripts/disaster_recovery.sh database YYYYMMDD_HHMMSS

# Verify data integrity
./scripts/verify_data_integrity.sh
```

### Security Incident

**Immediate Actions:**

1. **Isolate the System:**

```bash
# Block external access
ufw deny 80
ufw deny 443

# Preserve logs
cp -r /var/log /opt/incident_logs_$(date +%Y%m%d_%H%M%S)
```

2. **Assess Impact:**

```bash
# Check for unauthorized access
grep -i "unauthorized\|failed login\|brute force" /var/log/auth.log

# Review recent database changes
docker-compose exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT * FROM audit_log 
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
"
```

3. **Containment and Recovery:**

```bash
# Reset all passwords
./scripts/reset_all_passwords.sh

# Update security configurations
./scripts/harden_security.sh

# Re-enable access with enhanced monitoring
ufw allow 443
```

---

## 📞 Contact Information

### Emergency Contacts (24/7)

**Technical Team:**

- **DevOps Lead**: +1-XXX-XXX-XXXX
- **Database Administrator**: +1-XXX-XXX-XXXX
- **Security Team**: <security@company.com>
- **On-Call Engineer**: +1-XXX-XXX-XXXX

**Business Contacts:**

- **Operations Manager**: +1-XXX-XXX-XXXX
- **Product Owner**: +1-XXX-XXX-XXXX
- **Executive On-Call**: +1-XXX-XXX-XXXX

### Vendor Support

**Cloud Provider:**

- **AWS Support**: +1-XXX-XXX-XXXX (Enterprise)
- **Account Manager**: <aws-account@company.com>

**Database Vendor:**

- **PostgreSQL Support**: <support@postgresql.com>
- **Vendor Portal**: <https://support.postgresql.com>

**Monitoring Vendor:**

- **Prometheus Support**: <support@prometheus.io>
- **Grafana Support**: <support@grafana.com>

### Internal Resources

**Documentation:**

- **Wiki**: <https://wiki.company.com/pynomaly>
- **Runbooks**: <https://docs.company.com/runbooks>
- **Architecture**: <https://docs.company.com/architecture>

**Communication Channels:**

- **Slack - #pynomaly-alerts**: For automated alerts
- **Slack - #pynomaly-incidents**: For incident coordination
- **Slack - #pynomaly-team**: For general team communication

### Escalation Procedure

**Level 1 (0-15 minutes):**

- On-call engineer responds
- Initial assessment and basic troubleshooting
- Update incident channel

**Level 2 (15-30 minutes):**

- Escalate to technical lead
- Involve additional team members
- Begin detailed investigation

**Level 3 (30+ minutes):**

- Escalate to management
- Consider external vendor support
- Implement disaster recovery procedures

---

**Guide Version**: 2.0  
**Last Updated**: $(date +%Y-%m-%d)  
**Next Review**: $(date -d "+3 months" +%Y-%m-%d)

This troubleshooting guide should be your first reference for resolving production issues. Keep it updated with new issues and solutions as they are discovered.
