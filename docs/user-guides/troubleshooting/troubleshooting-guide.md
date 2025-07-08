# Troubleshooting Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔧 [Troubleshooting](README.md) > 📄 Troubleshooting Guide

---


## Overview

This comprehensive troubleshooting guide helps diagnose and resolve common issues encountered when deploying and operating Pynomaly. It covers installation problems, performance issues, configuration errors, and operational challenges with systematic debugging approaches.

## Table of Contents

1. [Installation and Setup Issues](#installation-and-setup-issues)
2. [Configuration Problems](#configuration-problems)
3. [Database Connectivity Issues](#database-connectivity-issues)
4. [Performance Problems](#performance-problems)
5. [Memory and Resource Issues](#memory-and-resource-issues)
6. [Authentication and Authorization](#authentication-and-authorization)
7. [API and Service Issues](#api-and-service-issues)
8. [CLI Troubleshooting](#cli-troubleshooting)
9. [Kubernetes and Container Issues](#kubernetes-and-container-issues)
10. [Debugging Methodology](#debugging-methodology)
11. [Error Reference](#error-reference)
12. [Support and Resources](#support-and-resources)

## Installation and Setup Issues

### Python Environment Problems

#### Issue: ImportError or ModuleNotFoundError

**Symptoms:**
```bash
ImportError: No module named 'pynomaly'
ModuleNotFoundError: No module named 'sklearn'
```

**Diagnosis:**
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep pynomaly
pip show pynomaly

# Check virtual environment
which python
echo $VIRTUAL_ENV
```

**Solutions:**

1. **Virtual Environment Issues:**
```bash
# Create new virtual environment
python -m venv pynomaly-env
source pynomaly-env/bin/activate  # Linux/Mac
# or
pynomaly-env\Scripts\activate  # Windows

# Reinstall Pynomaly
pip install --upgrade pip
pip install pynomaly[all]
```

2. **Dependency Conflicts:**
```bash
# Check for conflicts
pip check

# Clean installation
pip uninstall pynomaly
pip cache purge
pip install pynomaly[all]

# Use Poetry for dependency management
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

3. **System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential libpq-dev

# CentOS/RHEL
sudo yum install python3-devel gcc postgresql-devel

# macOS
brew install postgresql
xcode-select --install
```

#### Issue: Optional Dependencies Not Available

**Symptoms:**
```
TensorFlow/PyTorch not available, falling back to basic algorithms
CUDA not detected, using CPU only
```

**Solutions:**

1. **Install GPU Support:**
```bash
# NVIDIA CUDA (check CUDA version first)
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with GPU
pip install tensorflow[and-cuda]

# Verify GPU installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

2. **Install Optional Dependencies:**
```bash
# Install all optional dependencies
pip install pynomaly[all]

# Install specific framework support
pip install pynomaly[pytorch]
pip install pynomaly[tensorflow]
pip install pynomaly[jax]
```

### Poetry Installation Issues

#### Issue: Poetry Lock File Conflicts

**Symptoms:**
```
The lock file is not up to date with the latest changes in pyproject.toml
Warning: The lock file is not up to date with the latest changes
```

**Solutions:**
```bash
# Update lock file
poetry lock --no-update

# Force update
poetry lock

# Complete reinstall
rm poetry.lock
poetry install

# Clear cache
poetry cache clear pypi --all
poetry install
```

## Configuration Problems

### Configuration File Issues

#### Issue: Configuration Not Found

**Symptoms:**
```
Configuration file not found at ~/.pynomaly/config.toml
Invalid configuration format
```

**Diagnosis:**
```bash
# Check configuration location
pynomaly config show
ls -la ~/.pynomaly/

# Validate configuration syntax
pynomaly config validate
```

**Solutions:**

1. **Create Configuration:**
```bash
# Initialize configuration
pynomaly config init

# Set required values
pynomaly config set database.url "postgresql://user:pass@localhost/pynomaly"
pynomaly config set cache.redis_url "redis://localhost:6379/0"
```

2. **Fix Configuration Format:**
```toml
# Correct format (~/.pynomaly/config.toml)
[database]
url = "postgresql://user:pass@localhost/pynomaly"
pool_size = 20

[cache]
enabled = true
redis_url = "redis://localhost:6379/0"

[logging]
level = "INFO"
format = "json"
```

#### Issue: Environment Variable Override

**Symptoms:**
```
Configuration values not matching expected settings
Environment-specific settings not applied
```

**Solutions:**
```bash
# Check environment variables
env | grep PYNOMALY

# Set environment-specific configuration
export PYNOMALY_DATABASE_URL="postgresql://prod-user:pass@prod-db/pynomaly"
export PYNOMALY_LOG_LEVEL="WARNING"

# Use environment files
echo "PYNOMALY_DATABASE_URL=postgresql://..." > .env
pynomaly config load-env .env
```

### SSL/TLS Configuration Issues

#### Issue: SSL Connection Errors

**Symptoms:**
```
SSL connection error
Certificate verification failed
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solutions:**

1. **Database SSL Configuration:**
```bash
# PostgreSQL with SSL
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require"

# Skip SSL verification (development only)
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=disable"

# Custom certificate
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require&sslcert=/path/to/cert.pem"
```

2. **Redis SSL Configuration:**
```bash
# Redis with SSL
export REDIS_URL="rediss://user:pass@host:6380/0"

# Custom SSL settings
pynomaly config set cache.ssl_cert_reqs "required"
pynomaly config set cache.ssl_ca_certs "/path/to/ca.pem"
```

## Database Connectivity Issues

### Connection Pool Problems

#### Issue: Connection Pool Exhausted

**Symptoms:**
```
QueuePool limit of size 20 overflow 10 reached
Connection pool is exhausted
TimeoutError: QueuePool limit exceeded
```

**Diagnosis:**
```bash
# Check connection pool status
pynomaly perf pools

# Monitor database connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname='pynomaly';"
```

**Solutions:**

1. **Increase Pool Size:**
```python
# Configuration adjustment
pynomaly config set database.pool_size 50
pynomaly config set database.max_overflow 20
pynomaly config set database.pool_timeout 60
```

2. **Fix Connection Leaks:**
```python
# Check for unclosed connections in code
async def example_function():
    async with session_factory() as session:
        # Always use context manager
        result = await session.execute(query)
        # Session automatically closed
    return result
```

3. **Connection Recycling:**
```bash
# Enable connection recycling
pynomaly config set database.pool_recycle 3600
pynomaly config set database.pool_pre_ping true
```

### Database Performance Issues

#### Issue: Slow Database Queries

**Symptoms:**
```
Database query timeout
Slow response times
High database CPU usage
```

**Diagnosis:**
```bash
# Check slow queries
pynomaly perf queries --slow-only

# Database performance monitoring
psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**Solutions:**

1. **Index Optimization:**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_detectors_algorithm ON detectors(algorithm_name);
CREATE INDEX CONCURRENTLY idx_detection_results_created ON detection_results(created_at);

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats WHERE tablename = 'detectors';
```

2. **Query Optimization:**
```bash
# Enable query optimization
pynomaly perf optimize --database
pynomaly perf optimize --indexes

# Vacuum and analyze
pynomaly perf optimize --vacuum
```

3. **Database Configuration:**
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
SELECT pg_reload_conf();
```

### Migration Issues

#### Issue: Database Migration Failures

**Symptoms:**
```
Migration failed: column "new_column" already exists
Alembic migration error
Database schema mismatch
```

**Solutions:**

1. **Reset Migrations:**
```bash
# Check migration status
alembic current
alembic history

# Reset to latest
alembic stamp head

# Force migration
alembic upgrade head --sql > migration.sql
psql -f migration.sql
```

2. **Manual Schema Fixes:**
```sql
-- Check schema differences
\d detectors
\d detection_results

-- Fix schema manually if needed
ALTER TABLE detectors ADD COLUMN IF NOT EXISTS new_column VARCHAR(255);
```

## Performance Problems

### High Response Times

#### Issue: API Endpoints Slow

**Symptoms:**
```
Average response time > 2 seconds
Timeout errors
High CPU usage
```

**Diagnosis:**
```bash
# Performance monitoring
pynomaly perf metrics
pynomaly perf system --cpu --memory

# API endpoint analysis
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/api/detectors
```

**curl-format.txt:**
```
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
```

**Solutions:**

1. **Database Optimization:**
```bash
# Enable query caching
pynomaly config set cache.query_cache_enabled true
pynomaly config set cache.query_cache_ttl 300

# Connection pooling optimization
pynomaly config set database.pool_size 30
pynomaly config set database.echo false
```

2. **Response Caching:**
```python
# Enable response caching
pynomaly config set cache.response_cache_enabled true
pynomaly config set cache.response_cache_ttl 600
```

3. **Async Processing:**
```bash
# Increase worker processes
pynomaly server start --workers 8

# Use async endpoints
pynomaly config set api.async_enabled true
```

### Memory Leaks

#### Issue: Increasing Memory Usage

**Symptoms:**
```
Memory usage continuously increasing
Out of memory errors
Container killed by OOMKiller
```

**Diagnosis:**
```bash
# Memory monitoring
pynomaly perf system --memory --detailed
ps aux | grep pynomaly

# Python memory profiling
python -m tracemalloc -o memory_trace.log
```

**Solutions:**

1. **Garbage Collection:**
```python
# Force garbage collection
import gc
gc.collect()

# Adjust GC thresholds
import gc
gc.set_threshold(700, 10, 10)
```

2. **Memory Management:**
```bash
# Set memory limits
pynomaly config set performance.max_memory_percent 80
pynomaly config set performance.gc_threshold 0.8
```

3. **Container Memory Limits:**
```yaml
# Kubernetes memory limits
resources:
  limits:
    memory: "4Gi"
  requests:
    memory: "2Gi"
```

## Memory and Resource Issues

### Out of Memory Errors

#### Issue: Large Dataset Processing

**Symptoms:**
```
MemoryError: Unable to allocate array
Out of memory during model training
Container killed (exit code 137)
```

**Solutions:**

1. **Batch Processing:**
```python
# Process data in batches
pynomaly config set processing.batch_size 1000
pynomaly config set processing.memory_efficient true

# CLI batch processing
pynomaly predict --detector DETECTOR_ID --dataset large_data.csv --batch-size 500
```

2. **Streaming Processing:**
```bash
# Enable streaming for large datasets
pynomaly predict --detector DETECTOR_ID --input large_data.csv --stream --chunk-size 1000
```

3. **Memory-Efficient Algorithms:**
```bash
# Use memory-efficient algorithms
pynomaly detector create --algorithm "MiniBatchKMeans" --batch-size 1000
pynomaly detector create --algorithm "SGDOneClassSVM" --max-iter 1000
```

### CPU Usage Issues

#### Issue: High CPU Utilization

**Symptoms:**
```
CPU usage consistently above 90%
Slow processing times
System unresponsive
```

**Diagnosis:**
```bash
# CPU monitoring
top -p $(pgrep -f pynomaly)
pynomaly perf system --cpu --processes

# Profile CPU usage
python -m cProfile -o cpu_profile.prof script.py
```

**Solutions:**

1. **Parallel Processing:**
```bash
# Increase worker processes
pynomaly config set processing.parallel_workers 8
pynomaly config set processing.max_concurrent_jobs 4
```

2. **Algorithm Optimization:**
```bash
# Use faster algorithms for large datasets
pynomaly detector create --algorithm "IsolationForest" --n-jobs -1
pynomaly detector create --algorithm "LocalOutlierFactor" --n-jobs 4
```

3. **Resource Limits:**
```bash
# Set CPU limits
pynomaly config set performance.max_cpu_percent 80
pynomaly config set processing.cpu_count 4
```

## Authentication and Authorization

### JWT Token Issues

#### Issue: Token Validation Failures

**Symptoms:**
```
Invalid token signature
Token has expired
Malformed JWT token
```

**Diagnosis:**
```bash
# Check token
echo "TOKEN" | base64 -d

# Validate configuration
pynomaly config show auth
```

**Solutions:**

1. **Token Configuration:**
```bash
# Set correct JWT secret
pynomaly config set auth.jwt_secret_key "your-256-bit-secret"
pynomaly config set auth.jwt_algorithm "HS256"
pynomaly config set auth.jwt_expiration 3600
```

2. **Clock Synchronization:**
```bash
# Sync system time
sudo ntpdate -s time.nist.gov
sudo systemctl restart systemd-timesyncd
```

3. **Token Refresh:**
```bash
# Enable token refresh
pynomaly config set auth.refresh_token_enabled true
pynomaly config set auth.refresh_token_expiration 604800  # 7 days
```

### Permission Denied Errors

#### Issue: RBAC Authorization Failures

**Symptoms:**
```
Insufficient permissions for this operation
User role not authorized
Access denied to resource
```

**Solutions:**

1. **Check User Roles:**
```bash
# List user permissions
pynomaly auth user show USER_ID --permissions

# Update user role
pynomaly auth user update USER_ID --role data_scientist
```

2. **Role Configuration:**
```python
# Update role permissions
pynomaly auth role update data_scientist --add-permission "detector:create"
pynomaly auth role update analyst --add-permission "detection:predict"
```

## API and Service Issues

### Service Startup Failures

#### Issue: Server Won't Start

**Symptoms:**
```
Port already in use
Permission denied binding to port
Module import errors on startup
```

**Diagnosis:**
```bash
# Check port usage
netstat -tulpn | grep :8000
lsof -i :8000

# Check service status
pynomaly server status
journalctl -u pynomaly -f
```

**Solutions:**

1. **Port Configuration:**
```bash
# Use different port
pynomaly server start --port 8080

# Kill existing process
pkill -f "pynomaly server"
```

2. **Permission Issues:**
```bash
# Use non-privileged port
pynomaly server start --port 8000

# Run with sudo (not recommended for production)
sudo pynomaly server start --port 80
```

3. **Module Import Issues:**
```bash
# Check Python path
export PYTHONPATH="/path/to/pynomaly:$PYTHONPATH"

# Reinstall dependencies
pip install --force-reinstall pynomaly[all]
```

### API Response Errors

#### Issue: 500 Internal Server Errors

**Symptoms:**
```
HTTP 500 Internal Server Error
Unexpected server response
API endpoints returning errors
```

**Diagnosis:**
```bash
# Check server logs
pynomaly server logs --tail 100

# Enable debug mode
pynomaly server start --debug --log-level DEBUG
```

**Solutions:**

1. **Database Connection:**
```bash
# Test database connectivity
pynomaly config validate --database

# Reset connection pool
pynomaly perf pools --reset
```

2. **Dependency Issues:**
```bash
# Check missing dependencies
pip check
pynomaly version --dependencies
```

## CLI Troubleshooting

### Command Execution Failures

#### Issue: CLI Commands Not Working

**Symptoms:**
```
Command not found: pynomaly
Permission denied
Configuration errors
```

**Diagnosis:**
```bash
# Check installation
which pynomaly
pynomaly --version

# Check PATH
echo $PATH
```

**Solutions:**

1. **Installation Issues:**
```bash
# Reinstall CLI
pip uninstall pynomaly
pip install pynomaly[cli]

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

2. **Configuration Problems:**
```bash
# Initialize configuration
pynomaly config init --force

# Use explicit config file
pynomaly --config /path/to/config.toml command
```

### Output Format Issues

#### Issue: JSON/Table Output Problems

**Symptoms:**
```
Invalid JSON output
Table formatting broken
Output not readable
```

**Solutions:**

1. **Output Format:**
```bash
# Force specific format
pynomaly command --output json
pynomaly command --output table --no-color

# Pipe to jq for JSON processing
pynomaly command --output json | jq '.results'
```

2. **Terminal Compatibility:**
```bash
# Disable colors
export NO_COLOR=1
pynomaly command --no-color

# Set terminal width
export COLUMNS=120
pynomaly command --output table
```

## Kubernetes and Container Issues

### Pod Startup Issues

#### Issue: Pods CrashLoopBackOff

**Symptoms:**
```
CrashLoopBackOff
Pod restart count increasing
Container exit code 1 or 125
```

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -l app=pynomaly
kubectl describe pod POD_NAME

# Check logs
kubectl logs POD_NAME --previous
kubectl logs POD_NAME -c container-name
```

**Solutions:**

1. **Resource Limits:**
```yaml
# Increase resource limits
resources:
  limits:
    memory: "4Gi"
    cpu: "2000m"
  requests:
    memory: "1Gi"
    cpu: "500m"
```

2. **Startup Probes:**
```yaml
# Adjust startup probe
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10
  failureThreshold: 10
```

3. **Configuration Issues:**
```bash
# Check ConfigMap and Secrets
kubectl get configmap pynomaly-config -o yaml
kubectl get secret pynomaly-secrets -o yaml

# Update configuration
kubectl create configmap pynomaly-config --from-file=config.toml --dry-run=client -o yaml | kubectl apply -f -
```

### Service Discovery Issues

#### Issue: Services Not Accessible

**Symptoms:**
```
Connection refused
Service endpoint not found
DNS resolution failures
```

**Diagnosis:**
```bash
# Check service endpoints
kubectl get endpoints pynomaly-api
kubectl get services

# Test connectivity
kubectl run test-pod --image=busybox --rm -it -- wget -qO- http://pynomaly-api:80/health
```

**Solutions:**

1. **Service Configuration:**
```yaml
# Verify service selector
spec:
  selector:
    app: pynomaly-api
  ports:
  - port: 80
    targetPort: 8000
```

2. **Network Policies:**
```bash
# Check network policies
kubectl get networkpolicy
kubectl describe networkpolicy pynomaly-netpol
```

### Persistent Volume Issues

#### Issue: Storage Problems

**Symptoms:**
```
Pod stuck in pending state
Volume mount failures
Insufficient storage
```

**Solutions:**

1. **Storage Class:**
```bash
# Check storage classes
kubectl get storageclass
kubectl describe storageclass fast-ssd

# Create PVC with correct storage class
kubectl apply -f pvc.yaml
```

2. **Volume Permissions:**
```yaml
# Fix volume permissions
securityContext:
  runAsUser: 1000
  runAsGroup: 2000
  fsGroup: 2000
```

## Debugging Methodology

### Systematic Debugging Approach

#### 1. Problem Identification

```bash
# Gather basic information
pynomaly version
pynomaly config show
pynomaly server status

# Check system resources
pynomaly perf system
df -h
free -m
```

#### 2. Log Analysis

```bash
# Application logs
pynomaly server logs --tail 100 --level ERROR

# System logs
journalctl -u pynomaly -f
tail -f /var/log/pynomaly.log

# Container logs (if applicable)
docker logs pynomaly-container
kubectl logs -l app=pynomaly --tail=100
```

#### 3. Network Connectivity

```bash
# Test database connection
psql "postgresql://user:pass@host:5432/dbname" -c "SELECT 1;"

# Test Redis connection
redis-cli -h host -p 6379 ping

# Test API endpoints
curl -v http://localhost:8000/api/health
```

#### 4. Performance Analysis

```bash
# System performance
top
iostat -x 1
netstat -i

# Application performance
pynomaly perf metrics --detailed
pynomaly perf queries --slow-only
```

### Debug Mode Activation

```bash
# Enable debug logging
export PYNOMALY_LOG_LEVEL=DEBUG
pynomaly --log-level DEBUG command

# Enable SQL query logging
export PYNOMALY_DATABASE_ECHO=true

# Enable detailed error messages
export PYNOMALY_DEBUG=true
pynomaly server start --debug
```

### Logging Configuration

```toml
# Enhanced logging configuration
[logging]
level = "DEBUG"
format = "detailed"
file = "/var/log/pynomaly-debug.log"
rotation = "1 week"
max_size = "100 MB"

[logging.loggers.sqlalchemy]
level = "INFO"

[logging.loggers.redis]
level = "WARNING"
```

## Error Reference

### Common Error Codes

| Error Code | Description | Common Causes | Solutions |
|------------|-------------|---------------|-----------|
| DB_CONNECTION_FAILED | Database connection failed | Wrong credentials, DB down | Check connection string, verify DB status |
| CACHE_CONNECTION_FAILED | Redis connection failed | Redis down, wrong config | Verify Redis status and configuration |
| ALGORITHM_NOT_FOUND | Specified algorithm not available | Typo, missing dependency | Check algorithm name, install dependencies |
| INSUFFICIENT_MEMORY | Out of memory | Large dataset, memory leak | Reduce batch size, check for leaks |
| INVALID_CONFIGURATION | Configuration validation failed | Wrong format, missing values | Validate config file syntax |
| AUTHENTICATION_FAILED | Auth token invalid | Expired token, wrong secret | Refresh token, check JWT secret |
| PERMISSION_DENIED | Insufficient permissions | Wrong role, missing permission | Check user roles and permissions |
| DETECTOR_NOT_FITTED | Model not trained | Using untrained model | Train the model first |
| DATASET_NOT_FOUND | Dataset file missing | Wrong path, deleted file | Check file path and existence |
| API_RATE_LIMITED | Too many requests | Exceeding rate limits | Reduce request rate, check limits |

### HTTP Error Codes

| Code | Meaning | Typical Causes | Solutions |
|------|---------|----------------|-----------|
| 400 | Bad Request | Invalid JSON, missing fields | Validate request format |
| 401 | Unauthorized | Missing/invalid token | Provide valid auth token |
| 403 | Forbidden | Insufficient permissions | Check user roles |
| 404 | Not Found | Resource doesn't exist | Verify resource ID |
| 422 | Unprocessable Entity | Validation errors | Fix request data |
| 429 | Too Many Requests | Rate limit exceeded | Reduce request rate |
| 500 | Internal Server Error | Server-side error | Check server logs |
| 502 | Bad Gateway | Upstream service down | Check service status |
| 503 | Service Unavailable | Service overloaded | Scale up resources |

## Support and Resources

### Getting Help

1. **Documentation:**
   - API Reference: `/docs/api/`
   - User Guides: `/docs/guides/`
   - Examples: `/examples/`

2. **Community Support:**
   - GitHub Issues: `https://github.com/your-org/pynomaly/issues`
   - Discord Server: `https://discord.gg/pynomaly`
   - Stack Overflow: Tag `pynomaly`

3. **Professional Support:**
   - Email: `support@pynomaly.io`
   - Enterprise Support: `enterprise@pynomaly.io`

### Diagnostic Information Collection

```bash
# Generate diagnostic report
pynomaly diagnostics generate --output diagnostic_report.json

# System information
pynomaly diagnostics system --include-logs --include-config

# Performance snapshot
pynomaly diagnostics performance --duration 60
```

### Reporting Issues

When reporting issues, include:

1. **Environment Information:**
   ```bash
   pynomaly version --full
   python --version
   pip list | grep -E "(pynomaly|pandas|numpy|scikit-learn)"
   ```

2. **Configuration:**
   ```bash
   pynomaly config show --sanitized  # Removes sensitive data
   ```

3. **Error Logs:**
   ```bash
   pynomaly server logs --tail 50 --level ERROR
   ```

4. **Steps to Reproduce:**
   - Exact commands used
   - Input data characteristics
   - Expected vs actual behavior

This comprehensive troubleshooting guide provides systematic approaches to diagnosing and resolving issues across all components of the Pynomaly platform.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Developer Guides**
- **[Working While Services Are Running](../../../developer-guides/git-workflow/branching-strategy.md#working-while-services-are-running)** - Best practices for concurrent development with running services

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
