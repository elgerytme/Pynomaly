# Troubleshooting Guide

Comprehensive troubleshooting guide for common issues with Pynomaly Web UI. Find solutions to installation, configuration, and usage problems.

## 🔍 Quick Diagnostics

### Health Check Commands

Before diving into specific issues, run these quick diagnostics:

```bash
# Check Pynomaly installation
pynomaly --version

# Verify web UI components
pynomaly web check

# Test configuration
pynomaly config validate

# Check system status
pynomaly status

# View recent logs
pynomaly logs --tail 50
```

### Common Status Codes

- ✅ **200-299**: Success
- ⚠️ **400-499**: Client errors (check your request)
- ❌ **500-599**: Server errors (check logs)

## 🚀 Installation Issues

### Installation Fails

**Problem**: `pip install pynomaly[web]` fails

**Solutions**:

1. **Update pip and setuptools**:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install pynomaly[web]
   ```

2. **Check Python version**:

   ```bash
   python --version  # Should be 3.10+
   ```

3. **Use virtual environment**:

   ```bash
   python -m venv pynomaly-env
   source pynomaly-env/bin/activate  # Linux/Mac
   # or
   pynomaly-env\Scripts\activate  # Windows
   pip install pynomaly[web]
   ```

4. **Install from source** (if package issues persist):

   ```bash
   git clone https://github.com/pynomaly/pynomaly.git
   cd pynomaly
   pip install -e .[web]
   ```

### Missing Dependencies

**Problem**: ModuleNotFoundError for specific packages

**Solutions**:

1. **Install missing optional dependencies**:

   ```bash
   pip install pynomaly[all]  # Installs all optional dependencies
   ```

2. **Install specific dependency groups**:

   ```bash
   pip install pynomaly[web,automl,explainability]
   ```

3. **Manual dependency installation**:

   ```bash
   pip install fastapi uvicorn[standard] redis celery
   ```

### Platform-Specific Issues

**macOS ARM (M1/M2) Issues**:

```bash
# Use conda for better ARM support
conda create -n pynomaly python=3.10
conda activate pynomaly
conda install pandas numpy scikit-learn
pip install pynomaly[web]
```

**Windows Issues**:

```bash
# Use conda or enable long path support
pip install --user pynomaly[web]
```

**Linux Issues**:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential
pip install pynomaly[web]
```

## 🌐 Server Startup Issues

### Port Already in Use

**Problem**: "Address already in use" error

**Solutions**:

1. **Use different port**:

   ```bash
   pynomaly web start --port 8001
   ```

2. **Find and kill process using port**:

   ```bash
   # Linux/Mac
   lsof -i :8000
   kill -9 <PID>
   
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

3. **Set port in configuration**:

   ```yaml
   # config.yaml
   server:
     port: 8001
   ```

### Permission Denied

**Problem**: Cannot bind to port (usually port 80 or 443)

**Solutions**:

1. **Use higher port number**:

   ```bash
   pynomaly web start --port 8000
   ```

2. **Run with sudo** (not recommended for production):

   ```bash
   sudo pynomaly web start --port 80
   ```

3. **Use reverse proxy** (recommended):

   ```nginx
   # nginx.conf
   server {
       listen 80;
       location / {
           proxy_pass http://localhost:8000;
       }
   }
   ```

### Server Won't Start

**Problem**: Server exits immediately or fails to start

**Diagnostic Steps**:

1. **Check logs**:

   ```bash
   pynomaly logs --level DEBUG
   ```

2. **Validate configuration**:

   ```bash
   pynomaly config validate
   ```

3. **Test minimal configuration**:

   ```bash
   pynomaly web start --debug --workers 1
   ```

4. **Check system resources**:

   ```bash
   free -h  # Memory
   df -h    # Disk space
   ```

## 🗄️ Database Issues

### Database Connection Errors

**Problem**: Cannot connect to database

**Solutions**:

1. **Check database URL**:

   ```bash
   echo $PYNOMALY_DATABASE_URL
   ```

2. **Test database connection**:

   ```bash
   pynomaly config test-db
   ```

3. **Common database URLs**:

   ```bash
   # SQLite (default)
   sqlite:///pynomaly.db
   
   # PostgreSQL
   postgresql://user:pass@localhost:5432/pynomaly
   
   # MySQL
   mysql+pymysql://user:pass@localhost:3306/pynomaly
   ```

4. **Check database server status**:

   ```bash
   # PostgreSQL
   systemctl status postgresql
   
   # MySQL
   systemctl status mysql
   ```

### Migration Issues

**Problem**: Database migration fails

**Solutions**:

1. **Run migrations manually**:

   ```bash
   pynomaly db migrate
   ```

2. **Check migration status**:

   ```bash
   pynomaly db status
   ```

3. **Reset database** (⚠️ destroys data):

   ```bash
   pynomaly db reset
   pynomaly db init
   ```

4. **Backup before migration**:

   ```bash
   pynomaly db backup backup.sql
   ```

### Database Performance Issues

**Problem**: Slow database queries

**Solutions**:

1. **Check database connection pool**:

   ```yaml
   # config.yaml
   database:
     pool_size: 10
     max_overflow: 20
   ```

2. **Add database indexes** (for production):

   ```sql
   CREATE INDEX idx_detector_name ON detectors(name);
   CREATE INDEX idx_result_timestamp ON results(timestamp);
   ```

3. **Monitor database performance**:

   ```bash
   pynomaly db stats
   ```

## 🔴 Redis/Cache Issues

### Redis Connection Errors

**Problem**: Cannot connect to Redis

**Solutions**:

1. **Check Redis status**:

   ```bash
   redis-cli ping  # Should return PONG
   ```

2. **Start Redis server**:

   ```bash
   # Linux
   sudo systemctl start redis
   
   # macOS
   brew services start redis
   
   # Docker
   docker run -d -p 6379:6379 redis:alpine
   ```

3. **Use alternative cache**:

   ```yaml
   # config.yaml
   cache:
     type: "memory"  # Instead of redis
   ```

4. **Check Redis configuration**:

   ```bash
   redis-cli config get "*"
   ```

### Cache Performance Issues

**Problem**: Slow cache operations

**Solutions**:

1. **Increase Redis memory**:

   ```redis
   # redis.conf
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   ```

2. **Monitor cache hit rates**:

   ```bash
   pynomaly cache stats
   ```

3. **Clear cache if needed**:

   ```bash
   pynomaly cache clear
   ```

## 📊 Data Upload Issues

### File Upload Fails

**Problem**: Cannot upload datasets

**Solutions**:

1. **Check file size limits**:

   ```yaml
   # config.yaml
   data:
     max_file_size: "500MB"
   ```

2. **Verify file format**:

   ```bash
   # Supported formats
   file data.csv    # Should show "ASCII text"
   head -5 data.csv # Check for proper CSV format
   ```

3. **Check file permissions**:

   ```bash
   ls -la data.csv
   chmod 644 data.csv
   ```

4. **Try smaller test file**:

   ```bash
   head -100 large_file.csv > test_file.csv
   ```

### Data Processing Errors

**Problem**: Dataset processing fails

**Common Issues and Solutions**:

1. **Encoding issues**:

   ```python
   # Save with UTF-8 encoding
   df.to_csv('data.csv', encoding='utf-8')
   ```

2. **Missing values**:

   ```python
   # Check for missing values
   df.isnull().sum()
   
   # Fill missing values
   df.fillna(df.mean(), inplace=True)
   ```

3. **Invalid data types**:

   ```python
   # Convert data types
   df['column'] = pd.to_numeric(df['column'], errors='coerce')
   ```

4. **Memory issues with large files**:

   ```yaml
   # config.yaml
   performance:
     memory_limit: "8GB"
     chunk_size: 1000
   ```

## 🤖 Model Training Issues

### Training Fails

**Problem**: Detector training fails

**Diagnostic Steps**:

1. **Check training logs**:

   ```bash
   pynomaly logs --filter training
   ```

2. **Verify data quality**:
   - Check for missing values
   - Ensure numerical data types
   - Verify sufficient data samples

3. **Try different parameters**:

   ```python
   # Reduce contamination
   contamination: 0.05  # Instead of 0.1
   
   # Simplify algorithm
   algorithm: "IsolationForest"  # Instead of complex algorithms
   ```

4. **Check memory usage**:

   ```bash
   htop  # Monitor memory during training
   ```

### Training Takes Too Long

**Problem**: Training never completes

**Solutions**:

1. **Reduce data size**:

   ```python
   # Sample data for training
   sample_data = df.sample(n=10000)
   ```

2. **Optimize algorithm parameters**:

   ```yaml
   # For Isolation Forest
   n_estimators: 50  # Instead of 100
   max_samples: 1000  # Limit sample size
   ```

3. **Use faster algorithms**:
   - Isolation Forest (fast)
   - ECOD (fast)
   - Avoid: ABOD, KNN (slow on large data)

4. **Enable parallel processing**:

   ```yaml
   # config.yaml
   performance:
     max_workers: 8
   ```

### Memory Errors During Training

**Problem**: Out of memory errors

**Solutions**:

1. **Increase memory limits**:

   ```yaml
   # config.yaml
   performance:
     memory_limit: "8GB"
   ```

2. **Reduce batch size**:

   ```yaml
   performance:
     chunk_size: 500  # Smaller chunks
   ```

3. **Use memory-efficient algorithms**:

   ```python
   # Memory-efficient options
   algorithms = ["IsolationForest", "ECOD", "COPOD"]
   ```

4. **Monitor memory usage**:

   ```bash
   # Linux
   watch -n 1 'free -h'
   
   # Python memory profiling
   pip install memory-profiler
   ```

## 🔒 Authentication Issues

### Login Problems

**Problem**: Cannot log in to the interface

**Solutions**:

1. **Check if authentication is enabled**:

   ```yaml
   # config.yaml
   auth:
     enabled: true
   ```

2. **Reset user password**:

   ```bash
   pynomaly user reset-password username
   ```

3. **Check session configuration**:

   ```yaml
   security:
     session_timeout: 3600
     secret_key: "your-secret-key"
   ```

4. **Clear browser cache and cookies**

### Permission Errors

**Problem**: "Access denied" errors

**Solutions**:

1. **Check user roles**:

   ```bash
   pynomaly user list
   pynomaly user show username
   ```

2. **Grant appropriate permissions**:

   ```bash
   pynomaly user grant username admin
   ```

3. **Verify RBAC configuration**:

   ```yaml
   auth:
     rbac:
       enabled: true
       default_role: "user"
   ```

## 🌐 Browser and UI Issues

### Interface Won't Load

**Problem**: Web page doesn't load or shows errors

**Solutions**:

1. **Check server status**:

   ```bash
   curl http://localhost:8000/health
   ```

2. **Try different browser**:
   - Clear cache and cookies
   - Disable browser extensions
   - Try incognito/private mode

3. **Check browser console**:
   - Press F12 to open developer tools
   - Look for JavaScript errors in Console tab
   - Check Network tab for failed requests

4. **Verify static files**:

   ```bash
   ls -la static/css/
   ls -la static/js/
   ```

### Slow Page Loading

**Problem**: Pages load very slowly

**Solutions**:

1. **Enable caching**:

   ```yaml
   # config.yaml
   cache:
     enabled: true
     default_timeout: 300
   ```

2. **Optimize database queries**:

   ```bash
   pynomaly db optimize
   ```

3. **Check network latency**:

   ```bash
   ping localhost
   curl -w "@curl-format.txt" http://localhost:8000
   ```

4. **Enable compression**:

   ```yaml
   server:
     gzip: true
   ```

### JavaScript Errors

**Problem**: Interactive features don't work

**Solutions**:

1. **Check browser console** for errors
2. **Update browser** to latest version
3. **Disable conflicting extensions**
4. **Check Content Security Policy**:

   ```yaml
   security:
     csp: "default-src 'self' 'unsafe-inline'"
   ```

## 📊 Performance Issues

### High CPU Usage

**Problem**: Server consuming too much CPU

**Solutions**:

1. **Reduce worker processes**:

   ```bash
   pynomaly web start --workers 2
   ```

2. **Optimize algorithms**:

   ```python
   # Use faster algorithms
   contamination: 0.1
   n_estimators: 50  # Reduce from default
   ```

3. **Enable CPU limits**:

   ```yaml
   performance:
     cpu_limit: "80%"
   ```

4. **Monitor CPU usage**:

   ```bash
   htop
   pynomaly status --detailed
   ```

### High Memory Usage

**Problem**: Server using too much memory

**Solutions**:

1. **Set memory limits**:

   ```yaml
   performance:
     memory_limit: "4GB"
   ```

2. **Reduce batch sizes**:

   ```yaml
   performance:
     chunk_size: 500
   ```

3. **Enable garbage collection**:

   ```python
   import gc
   gc.collect()  # Force garbage collection
   ```

4. **Monitor memory usage**:

   ```bash
   free -h
   pynomaly status --memory
   ```

## 🔧 Configuration Issues

### Configuration Not Loading

**Problem**: Changes to config file not taking effect

**Solutions**:

1. **Restart the server**:

   ```bash
   pynomaly web restart
   ```

2. **Check configuration file location**:

   ```bash
   pynomaly config where
   ```

3. **Validate configuration syntax**:

   ```bash
   pynomaly config validate
   ```

4. **Check environment variables**:

   ```bash
   env | grep PYNOMALY
   ```

### Environment Variable Issues

**Problem**: Environment variables not working

**Solutions**:

1. **Check variable names**:

   ```bash
   echo $PYNOMALY_DATABASE_URL
   ```

2. **Export variables properly**:

   ```bash
   export PYNOMALY_SECRET_KEY="your-key"
   ```

3. **Use .env file**:

   ```bash
   # .env
   PYNOMALY_DATABASE_URL=sqlite:///pynomaly.db
   PYNOMALY_SECRET_KEY=your-secret-key
   ```

4. **Check variable precedence**:
   - Command line args (highest)
   - Environment variables
   - Config files
   - Defaults (lowest)

## 🆘 Getting Help

### Enable Debug Mode

For detailed troubleshooting information:

```bash
# Start in debug mode
pynomaly web start --debug --log-level DEBUG

# Enable verbose logging
export PYNOMALY_LOG_LEVEL=DEBUG
export PYNOMALY_DEBUG=true
```

### Collect System Information

When reporting issues, include:

```bash
# System information
pynomaly debug info

# Recent logs
pynomaly logs --tail 100 > logs.txt

# Configuration (redacted)
pynomaly config show --safe > config.txt

# System status
pynomaly status --detailed > status.txt
```

### Common Support Channels

1. **Built-in Help**: Press F1 in the interface
2. **Documentation**: Check this troubleshooting guide
3. **Community Forum**: Search for similar issues
4. **GitHub Issues**: Report bugs and feature requests
5. **Enterprise Support**: Contact support team

### Before Contacting Support

Please provide:

- [ ] Pynomaly version (`pynomaly --version`)
- [ ] Operating system and version
- [ ] Python version
- [ ] Error messages and logs
- [ ] Steps to reproduce the issue
- [ ] Configuration files (with secrets removed)

## 🔄 Recovery Procedures

### Complete System Reset

If all else fails:

```bash
# Stop all services
pynomaly web stop

# Backup important data
pynomaly db backup backup.sql

# Reset configuration
pynomaly config reset

# Reset database
pynomaly db reset

# Reinstall
pip uninstall pynomaly
pip install pynomaly[web]

# Start fresh
pynomaly web start
```

### Restore from Backup

```bash
# Restore database
pynomaly db restore backup.sql

# Restore configuration
cp backup_config.yaml ~/.pynomaly/config.yaml

# Restart services
pynomaly web restart
```

---

**Still having issues?** Don't hesitate to reach out to the community or support team. We're here to help!
