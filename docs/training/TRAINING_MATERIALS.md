# Pynomaly Training Materials

Comprehensive training materials for operating and maintaining Pynomaly in production.

## Table of Contents
1. [Getting Started](#getting-started)
2. [System Architecture](#system-architecture)
3. [Daily Operations](#daily-operations)
4. [Troubleshooting](#troubleshooting)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Security Best Practices](#security-best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Backup and Recovery](#backup-and-recovery)
9. [Hands-on Labs](#hands-on-labs)
10. [Certification Program](#certification-program)

---

## Getting Started

### Prerequisites
Before starting this training, you should have:
- Basic knowledge of Docker and containerization
- Understanding of Linux command line
- Familiarity with Python applications
- Basic knowledge of databases (PostgreSQL)
- Understanding of web applications and APIs

### Training Objectives
By completing this training, you will be able to:
- Deploy and manage Pynomaly in production
- Monitor system health and performance
- Troubleshoot common issues
- Implement security best practices
- Perform backup and recovery operations
- Optimize system performance

### Training Environment Setup
```bash
# Clone the repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Set up training environment
cp .env.example .env
docker-compose -f docker-compose.simple.yml up -d

# Verify setup
curl http://localhost:8000/health
```

---

## System Architecture

### Overview
Pynomaly is a microservices-based anomaly detection platform consisting of:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   Pynomaly API  │    │   PostgreSQL    │
│  (Load Balancer)│◄──►│   (FastAPI)     │◄──►│   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │     Redis       │              │
         │              │    (Cache)      │              │
         │              └─────────────────┘              │
         │                                                │
         │              ┌─────────────────┐              │
         └─────────────►│   Prometheus    │◄─────────────┘
                        │  (Monitoring)   │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │    Grafana      │
                        │  (Dashboard)    │
                        └─────────────────┘
```

### Core Components

#### 1. API Service (pynomaly-api)
- **Technology**: FastAPI (Python)
- **Purpose**: Main application serving HTTP API
- **Port**: 8000
- **Health Check**: `/health`
- **Documentation**: `/docs`

#### 2. Database (postgres)
- **Technology**: PostgreSQL 15
- **Purpose**: Persistent data storage
- **Port**: 5432
- **Database**: `pynomaly_prod`
- **User**: `pynomaly`

#### 3. Cache (redis-cluster)
- **Technology**: Redis 7
- **Purpose**: Caching and session storage
- **Port**: 6379
- **Configuration**: Cluster mode

#### 4. Monitoring (prometheus + grafana)
- **Prometheus**: Metrics collection (Port: 9090)
- **Grafana**: Visualization dashboard (Port: 3000)
- **Purpose**: System monitoring and alerting

#### 5. Load Balancer (nginx)
- **Technology**: Nginx
- **Purpose**: Reverse proxy and load balancing
- **Port**: 80/443
- **SSL**: Terminates SSL connections

### Data Flow
1. **Request**: Client → Nginx → API Service
2. **Processing**: API Service → Database/Cache
3. **Response**: Database/Cache → API Service → Nginx → Client
4. **Monitoring**: All components → Prometheus → Grafana

---

## Daily Operations

### Morning Checklist (15 minutes)

#### 1. System Health Check
```bash
# Check all services are running
docker-compose -f docker-compose.simple.yml ps

# Quick health check
curl http://localhost:8000/health

# Check Grafana dashboard
# Navigate to http://localhost:3000
```

#### 2. Resource Usage Check
```bash
# Check disk usage
df -h

# Check memory usage
free -h

# Check Docker container stats
docker stats --no-stream
```

#### 3. Log Review
```bash
# Check for errors in the last 24 hours
docker-compose -f docker-compose.simple.yml logs --since=24h | grep ERROR

# Check application logs
docker-compose -f docker-compose.simple.yml logs pynomaly-api --tail=100
```

### Afternoon Checklist (10 minutes)

#### 1. Performance Check
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Check database performance
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active';
"
```

#### 2. Backup Status
```bash
# Check latest backup
ls -la /backups/database/ | tail -5

# Verify backup integrity
python scripts/backup_recovery.py --verify-only
```

### Evening Checklist (5 minutes)

#### 1. Security Review
```bash
# Check for unusual access patterns
docker-compose -f docker-compose.simple.yml logs nginx | grep -E "(401|403|404)" | tail -10

# Check failed authentications
docker-compose -f docker-compose.simple.yml logs pynomaly-api | grep -i "auth.*fail" | tail -10
```

#### 2. Cleanup Tasks
```bash
# Clean up Docker resources
docker system prune --filter "until=24h"

# Rotate logs if needed
docker-compose -f docker-compose.simple.yml exec pynomaly-api logrotate /etc/logrotate.d/pynomaly
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: API Service Not Responding
**Symptoms**: 
- Health check returns 503 or times out
- No response from API endpoints

**Troubleshooting Steps**:
```bash
# Step 1: Check container status
docker-compose -f docker-compose.simple.yml ps pynomaly-api

# Step 2: Check logs
docker-compose -f docker-compose.simple.yml logs pynomaly-api --tail=50

# Step 3: Check resource usage
docker stats pynomaly-api --no-stream

# Step 4: Check network connectivity
docker-compose -f docker-compose.simple.yml exec pynomaly-api ping postgres
```

**Common Solutions**:
- Restart the service: `docker-compose restart pynomaly-api`
- Check configuration: Verify `.env` file and `config/production.yml`
- Increase resources: Update Docker resource limits

#### Issue 2: Database Connection Problems
**Symptoms**:
- "Connection refused" errors
- Database timeout errors

**Troubleshooting Steps**:
```bash
# Step 1: Check database container
docker-compose -f docker-compose.simple.yml ps postgres

# Step 2: Test database connectivity
docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U pynomaly -d pynomaly_prod

# Step 3: Check database logs
docker-compose -f docker-compose.simple.yml logs postgres --tail=50

# Step 4: Test connection from API
docker-compose -f docker-compose.simple.yml exec pynomaly-api python -c "
import psycopg2
conn = psycopg2.connect('postgresql://pynomaly:password@postgres:5432/pynomaly_prod')
print('Connection successful')
"
```

**Common Solutions**:
- Restart PostgreSQL: `docker-compose restart postgres`
- Check credentials: Verify `POSTGRES_PASSWORD` in `.env`
- Check network: Ensure Docker network is healthy

#### Issue 3: High Response Times
**Symptoms**:
- API responses taking > 1 second
- Timeout errors from clients

**Troubleshooting Steps**:
```bash
# Step 1: Measure response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Step 2: Check database query performance
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
"

# Step 3: Check Redis performance
docker-compose -f docker-compose.simple.yml exec redis-cluster redis-cli --latency-history -i 1

# Step 4: Check system resources
docker stats --no-stream
```

**Common Solutions**:
- Optimize database queries
- Increase database connection pool
- Enable caching for frequently accessed data
- Scale horizontally

### Troubleshooting Flowchart

```
Problem Reported
       │
       ▼
Is the API responding?
       │
   ┌───▼───┐
   │  NO   │ → Check Docker containers → Restart services
   └───────┘
       │
   ┌───▼───┐
   │  YES  │ → Check response time → If slow, check database
   └───────┘
       │
       ▼
Are errors occurring?
       │
   ┌───▼───┐
   │  YES  │ → Check logs → Identify error type → Apply fix
   └───────┘
       │
   ┌───▼───┐
   │  NO   │ → Monitor system → Check for resource issues
   └───────┘
```

---

## Monitoring and Alerting

### Key Dashboards

#### 1. System Overview Dashboard
**URL**: http://localhost:3000/d/pynomaly-main
**Key Metrics**:
- Overall system health
- Request rate and response times
- Error rates
- Resource utilization

#### 2. Database Dashboard
**Key Metrics**:
- Connection count
- Query performance
- Lock waits
- Database size growth

#### 3. Performance Dashboard
**Key Metrics**:
- Response time percentiles
- Throughput (RPS)
- Error rates by endpoint
- Cache hit rates

### Alert Levels

#### Critical Alerts (Immediate Response)
- API service down (0% uptime)
- Database unavailable
- Disk usage > 95%
- Memory usage > 95%
- Error rate > 5%

#### Warning Alerts (Response within 1 hour)
- High response time (> 1000ms)
- High error rate (> 1%)
- High resource usage (> 80%)
- SSL certificate expiring (< 30 days)

#### Info Alerts (Response within 24 hours)
- Unusual traffic patterns
- Performance degradation
- Backup failures
- Configuration changes

### Alert Response Procedures

#### Critical Alert Response
1. **Acknowledge** alert within 5 minutes
2. **Assess** system status using monitoring dashboards
3. **Escalate** if needed (Level 2 support)
4. **Document** actions taken
5. **Resolve** and close alert

#### Warning Alert Response
1. **Acknowledge** alert within 1 hour
2. **Investigate** root cause
3. **Plan** remediation actions
4. **Execute** fix during maintenance window
5. **Monitor** to ensure resolution

---

## Security Best Practices

### Access Control

#### 1. SSH Access
```bash
# Use SSH keys instead of passwords
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Configure SSH key-based authentication
ssh-copy-id user@server

# Disable password authentication
sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

#### 2. Database Access
```bash
# Create limited privilege user for application
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
CREATE USER app_user WITH PASSWORD 'secure_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
"

# Revoke unnecessary privileges
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "
REVOKE ALL ON SCHEMA public FROM public;
"
```

### Network Security

#### 1. Firewall Configuration
```bash
# Enable firewall
sudo ufw enable

# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS

# Block all other ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
```

#### 2. SSL/TLS Configuration
```bash
# Generate SSL certificate
sudo certbot certonly --webroot -w /var/www/html -d your-domain.com

# Configure nginx with SSL
# Update nginx configuration to use SSL certificates

# Test SSL configuration
curl -I https://your-domain.com
```

### Application Security

#### 1. Environment Variables
```bash
# Never commit secrets to version control
# Use environment variables for sensitive data
export SECRET_KEY="your-secret-key"
export DATABASE_PASSWORD="your-db-password"

# Use Docker secrets in production
docker secret create db_password /path/to/password/file
```

#### 2. Input Validation
```python
# Example of input validation in API
from pydantic import BaseModel, validator

class DataInput(BaseModel):
    value: float
    
    @validator('value')
    def validate_value(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('Value must be between 0 and 1000')
        return v
```

### Security Monitoring

#### 1. Log Analysis
```bash
# Monitor failed login attempts
grep "Failed password" /var/log/auth.log | tail -10

# Monitor unusual network activity
netstat -an | grep ":80\|:443" | wc -l

# Check for suspicious processes
ps aux | grep -E "(nc|netcat|wget|curl)" | grep -v grep
```

#### 2. Vulnerability Scanning
```bash
# Scan Docker images for vulnerabilities
docker scan pynomaly:production

# Check for outdated packages
docker-compose -f docker-compose.simple.yml exec pynomaly-api pip list --outdated

# Run security audit
python -m pip audit
```

---

## Performance Optimization

### Database Optimization

#### 1. Query Optimization
```sql
-- Identify slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Add indexes for frequently queried columns
CREATE INDEX idx_user_id ON user_actions(user_id);
CREATE INDEX idx_timestamp ON events(timestamp);

-- Analyze query execution plans
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'user@example.com';
```

#### 2. Connection Pooling
```python
# Configure connection pool in application
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Application Optimization

#### 1. Caching Strategy
```python
# Implement Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='redis-cluster', port=6379, db=0)

def cache_result(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(ttl=1800)
def get_user_data(user_id):
    # Expensive database query
    return database.query(f"SELECT * FROM users WHERE id = {user_id}")
```

#### 2. Async Processing
```python
# Use async/await for I/O operations
import asyncio
import aiohttp

async def fetch_external_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Concurrent processing
async def process_batch(urls):
    tasks = [fetch_external_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results
```

### Infrastructure Optimization

#### 1. Container Optimization
```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["python", "app.py"]
```

#### 2. Resource Limits
```yaml
# docker-compose.yml
version: '3.8'
services:
  pynomaly-api:
    image: pynomaly:production
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

---

## Backup and Recovery

### Backup Strategy

#### 1. Database Backup
```bash
# Daily full backup
docker-compose -f docker-compose.simple.yml exec postgres pg_dump -U pynomaly -d pynomaly_prod > backup_$(date +%Y%m%d).sql

# Incremental backup using WAL files
docker-compose -f docker-compose.simple.yml exec postgres pg_receivewal -D /backups/wal -U pynomaly
```

#### 2. Application Backup
```bash
# Backup application code and configuration
tar -czf app_backup_$(date +%Y%m%d).tar.gz src/ config/ docker-compose.yml

# Backup ML models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### Recovery Procedures

#### 1. Database Recovery
```bash
# Stop application
docker-compose -f docker-compose.simple.yml stop pynomaly-api

# Restore database
docker-compose -f docker-compose.simple.yml exec -T postgres psql -U pynomaly -d pynomaly_prod < backup_20240101.sql

# Restart application
docker-compose -f docker-compose.simple.yml start pynomaly-api
```

#### 2. Point-in-Time Recovery
```bash
# Restore to specific timestamp
docker-compose -f docker-compose.simple.yml exec postgres pg_restore -U pynomaly -d pynomaly_prod --clean --if-exists backup_file.dump

# Verify recovery
docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "SELECT COUNT(*) FROM users;"
```

---

## Hands-on Labs

### Lab 1: Deploy Pynomaly (30 minutes)

#### Objective
Deploy Pynomaly from scratch and verify all components are working.

#### Steps
1. **Environment Setup**
   ```bash
   git clone https://github.com/pynomaly/pynomaly.git
   cd pynomaly
   cp .env.example .env
   ```

2. **Configure Environment**
   ```bash
   # Edit .env file with appropriate values
   nano .env
   ```

3. **Deploy Services**
   ```bash
   docker-compose -f docker-compose.simple.yml up -d
   ```

4. **Verify Deployment**
   ```bash
   # Check service status
   docker-compose -f docker-compose.simple.yml ps
   
   # Test API
   curl http://localhost:8000/health
   
   # Test database
   docker-compose -f docker-compose.simple.yml exec postgres pg_isready -U pynomaly -d pynomaly_prod
   ```

#### Expected Results
- All services should be running
- API health check should return 200
- Database should be ready
- Grafana dashboard should be accessible

### Lab 2: Troubleshoot API Issues (45 minutes)

#### Objective
Practice troubleshooting common API issues.

#### Scenario
The API service is returning 500 errors for all requests.

#### Steps
1. **Identify the Problem**
   ```bash
   # Check API status
   curl -I http://localhost:8000/health
   
   # Check logs
   docker-compose -f docker-compose.simple.yml logs pynomaly-api
   ```

2. **Investigate Root Cause**
   ```bash
   # Check database connectivity
   docker-compose -f docker-compose.simple.yml exec pynomaly-api ping postgres
   
   # Check environment variables
   docker-compose -f docker-compose.simple.yml exec pynomaly-api env | grep DATABASE
   ```

3. **Apply Fix**
   ```bash
   # Fix configuration issue
   nano .env
   
   # Restart service
   docker-compose -f docker-compose.simple.yml restart pynomaly-api
   ```

4. **Verify Fix**
   ```bash
   # Test API again
   curl http://localhost:8000/health
   
   # Check logs for errors
   docker-compose -f docker-compose.simple.yml logs pynomaly-api --tail=10
   ```

### Lab 3: Performance Testing (60 minutes)

#### Objective
Run performance tests and analyze results.

#### Steps
1. **Run Performance Tests**
   ```bash
   python scripts/performance_testing.py
   ```

2. **Analyze Results**
   ```bash
   # Review performance report
   cat performance_test_report_*.json | jq '.performance_test_report'
   
   # Check Grafana dashboard
   # Open http://localhost:3000
   ```

3. **Optimize Performance**
   ```bash
   # Implement suggested optimizations
   # Update configuration
   # Restart services
   ```

4. **Re-test**
   ```bash
   # Run tests again
   python scripts/performance_testing.py
   
   # Compare results
   ```

### Lab 4: Backup and Recovery (45 minutes)

#### Objective
Practice backup and recovery procedures.

#### Steps
1. **Create Backup**
   ```bash
   python scripts/backup_recovery.py
   ```

2. **Simulate Data Loss**
   ```bash
   # Stop application
   docker-compose -f docker-compose.simple.yml stop pynomaly-api
   
   # Drop database (simulation)
   docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "DROP TABLE IF EXISTS users;"
   ```

3. **Restore from Backup**
   ```bash
   # Restore database
   python scripts/backup_recovery.py --restore
   
   # Restart application
   docker-compose -f docker-compose.simple.yml start pynomaly-api
   ```

4. **Verify Recovery**
   ```bash
   # Test API
   curl http://localhost:8000/health
   
   # Check data integrity
   docker-compose -f docker-compose.simple.yml exec postgres psql -U pynomaly -d pynomaly_prod -c "SELECT COUNT(*) FROM users;"
   ```

---

## Certification Program

### Pynomaly Operations Certified (POC)

#### Prerequisites
- Completed all training modules
- Hands-on experience with Pynomaly deployment
- Basic understanding of system administration

#### Certification Requirements
1. **Written Exam** (60 questions, 90 minutes)
   - System architecture (20%)
   - Daily operations (25%)
   - Troubleshooting (25%)
   - Security (15%)
   - Performance (15%)

2. **Practical Exam** (120 minutes)
   - Deploy Pynomaly from scratch
   - Troubleshoot simulated issues
   - Perform backup and recovery
   - Optimize system performance

#### Study Guide
- Review all training materials
- Complete all hands-on labs
- Practice troubleshooting scenarios
- Understand monitoring and alerting

#### Certification Validity
- Valid for 2 years
- Renewal requires continuing education
- Advanced certifications available

### Advanced Certifications

#### Pynomaly Security Specialist (PSS)
- Focus on security best practices
- Advanced threat detection
- Incident response procedures

#### Pynomaly Performance Engineer (PPE)
- Advanced performance optimization
- Scalability planning
- Load testing and analysis

#### Pynomaly DevOps Expert (PDE)
- CI/CD pipeline management
- Infrastructure as code
- Advanced monitoring and alerting

---

## Additional Resources

### Documentation
- [API Documentation](http://localhost:8000/docs)
- [System Architecture](docs/architecture.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Tools and Utilities
- [Performance Testing Script](scripts/performance_testing.py)
- [Backup Script](scripts/backup_recovery.py)
- [Security Scanner](scripts/security_hardening.py)

### Community Resources
- [Pynomaly Forum](https://forum.pynomaly.com)
- [Slack Channel](https://pynomaly.slack.com)
- [GitHub Discussions](https://github.com/pynomaly/pynomaly/discussions)

### Training Videos
- [Deployment Walkthrough](https://videos.pynomaly.com/deployment)
- [Troubleshooting Tutorial](https://videos.pynomaly.com/troubleshooting)
- [Performance Optimization](https://videos.pynomaly.com/performance)

---

**Last Updated**: {current_date}
**Version**: 1.0
**Maintainer**: Training Team
**Contact**: training@pynomaly.com