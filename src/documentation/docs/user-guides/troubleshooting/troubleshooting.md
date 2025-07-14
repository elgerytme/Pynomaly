# Troubleshooting Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔧 [Troubleshooting](README.md) > 🆘 Common Issues

---


This comprehensive troubleshooting guide covers common issues, debugging techniques, error resolution strategies, and diagnostic tools for Pynomaly deployments across development, staging, and production environments.

## Quick Diagnostics

### Health Check

```bash
# Check overall system health
pynomaly server status

# Or via API
curl http://localhost:8000/health
```

### Version Information

```bash
# Check Pynomaly version
pynomaly --version

# Check Python and dependencies
python --version
pip list | grep -E "(pynomaly|numpy|pandas|scikit-learn)"
```

### Log Analysis

```bash
# Enable debug logging
export PYNOMALY_LOG_LEVEL=DEBUG
pynomaly detectors list

# Check logs
tail -f ~/.pynomaly/logs/pynomaly.log
```

## Installation Issues

### Problem: Package Installation Fails

**Symptoms:**
```bash
pip install pynomaly
ERROR: Could not find a version that satisfies the requirement pynomaly
```

**Solutions:**

1. **Update pip and setuptools:**
```bash
pip install --upgrade pip setuptools wheel
pip install pynomaly
```

2. **Check Python version:**
```bash
python --version
# Pynomaly requires Python 3.11+
```

3. **Use virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pynomaly
```

4. **Install from source:**
```bash
git clone https://github.com/yourorg/pynomaly.git
cd pynomaly
pip install -e .
```

### Problem: Import Errors

**Symptoms:**
```python
ImportError: No module named 'pynomaly'
ModuleNotFoundError: No module named 'pynomaly.domain'
```

**Solutions:**

1. **Verify installation:**
```bash
pip show pynomaly
python -c "import pynomaly; print(pynomaly.__version__)"
```

2. **Check Python path:**
```python
import sys
print(sys.path)
# Ensure Pynomaly installation directory is in path
```

3. **Reinstall package:**
```bash
pip uninstall pynomaly
pip install pynomaly
```

### Problem: Dependency Conflicts

**Symptoms:**
```bash
ERROR: pynomaly has requirement numpy>=1.21.0, but you have numpy 1.19.0
```

**Solutions:**

1. **Update dependencies:**
```bash
pip install --upgrade numpy pandas scikit-learn
pip install pynomaly
```

2. **Use pip-tools for dependency resolution:**
```bash
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

3. **Create fresh environment:**
```bash
conda create -n pynomaly python=3.11
conda activate pynomaly
pip install pynomaly
```

## Configuration Issues

### Problem: Configuration File Not Found

**Symptoms:**
```bash
Error: Configuration file '~/.pynomaly/config.yml' not found
```

**Solutions:**

1. **Create default configuration:**
```bash
mkdir -p ~/.pynomaly
cat > ~/.pynomaly/config.yml << EOF
api:
  base_url: "http://localhost:8000"
  timeout: 30

defaults:
  output_format: "table"
  log_level: "INFO"
  contamination: 0.1

database:
  url: "sqlite:///~/.pynomaly/data.db"
EOF
```

2. **Use environment variables instead:**
```bash
export PYNOMALY_API_URL="http://localhost:8000"
export PYNOMALY_LOG_LEVEL="INFO"
```

3. **Specify config file explicitly:**
```bash
pynomaly --config /path/to/config.yml detectors list
```

### Problem: Database Connection Issues

**Symptoms:**
```bash
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
psycopg2.OperationalError: could not connect to server
```

**Solutions:**

1. **Check database URL format:**
```bash
# SQLite (default)
export PYNOMALY_DATABASE_URL="sqlite:///~/.pynomaly/data.db"

# PostgreSQL
export PYNOMALY_DATABASE_URL="postgresql://user:pass@localhost:5432/pynomaly"

# MySQL
export PYNOMALY_DATABASE_URL="mysql://user:pass@localhost:3306/pynomaly"
```

2. **Create database directory:**
```bash
mkdir -p ~/.pynomaly
chmod 755 ~/.pynomaly
```

3. **Test database connection:**
```python
from sqlalchemy import create_engine
engine = create_engine("sqlite:///~/.pynomaly/data.db")
connection = engine.connect()
print("Database connection successful")
```

4. **Run database migrations:**
```bash
pynomaly db migrate
# Or manually
python -c "from pynomaly.infrastructure.persistence import create_tables; create_tables()"
```

## Algorithm Issues

### Problem: Algorithm Not Found

**Symptoms:**
```bash
Error: Algorithm 'MyCustomAlgorithm' not found
Available algorithms: IsolationForest, LOF, OCSVM, ...
```

**Solutions:**

1. **Check available algorithms:**
```bash
pynomaly detectors algorithms
```

2. **Use correct algorithm name:**
```bash
# Correct names (case-sensitive)
pynomaly detectors create "Test" IsolationForest
pynomaly detectors create "Test" LOF
pynomaly detectors create "Test" OCSVM
```

3. **Install additional algorithm packages:**
```bash
# For PyOD algorithms
pip install pyod

# For deep learning algorithms
pip install torch tensorflow

# For graph algorithms
pip install torch-geometric
```

### Problem: Algorithm Parameters Invalid

**Symptoms:**
```bash
ValueError: contamination must be in (0, 0.5]
TypeError: fit() got an unexpected keyword argument 'n_estimators'
```

**Solutions:**

1. **Check algorithm documentation:**
```bash
pynomaly detectors algorithms --detailed
```

2. **Use valid parameter ranges:**
```bash
# IsolationForest parameters
pynomaly detectors create "Test" IsolationForest \
  --contamination 0.1 \
  --parameter n_estimators=100 \
  --parameter max_samples=256

# LOF parameters
pynomaly detectors create "Test" LOF \
  --contamination 0.1 \
  --parameter n_neighbors=20 \
  --parameter algorithm=auto
```

3. **Validate parameters before creation:**
```python
from pynomaly.domain.services import AlgorithmValidationService

validator = AlgorithmValidationService()
is_valid = validator.validate_parameters("IsolationForest", {
    "contamination": 0.1,
    "n_estimators": 100
})
print(f"Parameters valid: {is_valid}")
```

### Problem: Training Fails

**Symptoms:**
```bash
Error: Training failed: Input contains NaN, infinity or a value too large
RuntimeError: Dataset contains no valid samples
```

**Solutions:**

1. **Validate dataset before training:**
```bash
pynomaly datasets validate dataset_123 \
  --check-missing \
  --check-outliers \
  --report-file validation.json
```

2. **Clean dataset:**
```python
import pandas as pd
import numpy as np

# Load and clean data
df = pd.read_csv("data.csv")
df = df.dropna()  # Remove NaN values
df = df.replace([np.inf, -np.inf], np.nan).dropna()  # Remove infinity
df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
```

3. **Check data types:**
```bash
pynomaly datasets show dataset_123 --statistics
```

4. **Use data preprocessing:**
```bash
# Create detector with preprocessing
pynomaly detectors create "Robust Detector" IsolationForest \
  --parameter contamination=0.1 \
  --parameter preprocessing=standard_scaler
```

## Performance Issues

### Problem: Slow Training

**Symptoms:**
- Training takes unexpectedly long time
- High CPU usage during training
- Memory usage keeps growing

**Solutions:**

1. **Check dataset size:**
```bash
pynomaly datasets show dataset_123
# Large datasets (>100k samples) may need optimization
```

2. **Use sampling for large datasets:**
```bash
pynomaly datasets sample dataset_123 \
  --size 10000 \
  --method stratified \
  --output sample_dataset.csv
```

3. **Optimize algorithm parameters:**
```bash
# For IsolationForest: reduce max_samples
pynomaly detectors create "Fast Detector" IsolationForest \
  --parameter contamination=0.1 \
  --parameter max_samples=256 \
  --parameter n_estimators=50

# For LOF: reduce n_neighbors
pynomaly detectors create "Fast LOF" LOF \
  --parameter contamination=0.1 \
  --parameter n_neighbors=10
```

4. **Use faster algorithms:**
```bash
# COPOD is generally faster than IsolationForest
pynomaly detectors create "Fast Detector" COPOD

# ECOD is efficient for high-dimensional data
pynomaly detectors create "HD Detector" ECOD
```

5. **Enable parallel processing:**
```bash
# Set number of parallel jobs
export PYNOMALY_N_JOBS=4
pynomaly detectors train detector_123 dataset_456
```

### Problem: High Memory Usage

**Symptoms:**
```bash
MemoryError: Unable to allocate array
Process killed (OOM)
```

**Solutions:**

1. **Monitor memory usage:**
```bash
# Check system memory
free -h
top -p $(pgrep -f pynomaly)
```

2. **Use batch processing:**
```bash
pynomaly detect batch detector_123 dataset_456 \
  --chunk-size 1000 \
  --output-format csv
```

3. **Reduce feature dimensions:**
```python
# Use feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)
X_reduced = selector.fit_transform(X)
```

4. **Use memory-efficient data formats:**
```bash
# Convert to Parquet (more memory efficient)
pynomaly datasets export dataset_123 data.parquet --format parquet
```

### Problem: Slow Prediction

**Symptoms:**
- Real-time detection is too slow
- Batch prediction takes too long

**Solutions:**

1. **Use streaming detection:**
```bash
pynomaly detect stream detector_123 \
  --buffer-size 100 \
  --input-format json
```

2. **Optimize detector for speed:**
```bash
# Use algorithms optimized for prediction speed
pynomaly detectors create "Fast Predict" COPOD
pynomaly detectors create "Fast Predict" ECOD
```

3. **Cache predictions:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(data_hash):
    return detector.predict(data)
```

## API Issues

### Problem: API Server Won't Start

**Symptoms:**
```bash
OSError: [Errno 98] Address already in use
ConnectionError: Could not connect to API server
```

**Solutions:**

1. **Check if port is in use:**
```bash
netstat -tlnp | grep 8000
lsof -i :8000
```

2. **Kill existing process:**
```bash
pkill -f "pynomaly server"
# Or find and kill specific process
ps aux | grep pynomaly
kill <PID>
```

3. **Use different port:**
```bash
pynomaly server start --port 8001
```

4. **Check for permission issues:**
```bash
# Don't run as root in production
# Use ports > 1024 for non-root users
pynomaly server start --port 8000 --host 127.0.0.1
```

### Problem: Authentication Errors

**Symptoms:**
```bash
HTTP 401: Unauthorized
Error: Invalid API key
```

**Solutions:**

1. **Check API key:**
```bash
# List existing API keys
pynomaly auth list-keys

# Create new API key
pynomaly auth create-key --name "my-app"
```

2. **Set API key correctly:**
```bash
export PYNOMALY_API_KEY="your-api-key-here"
# Or use config file
echo "api_key: your-api-key-here" >> ~/.pynomaly/config.yml
```

3. **Check JWT token expiration:**
```bash
pynomaly auth refresh-token
```

### Problem: API Timeouts

**Symptoms:**
```bash
requests.exceptions.Timeout: HTTPSConnectionPool
Error: Request timeout after 30 seconds
```

**Solutions:**

1. **Increase timeout:**
```bash
pynomaly --config-timeout 60 detectors list
```

2. **Check network connectivity:**
```bash
curl -v http://localhost:8000/health
ping api.pynomaly.com
```

3. **Use async operations:**
```bash
pynomaly experiments run experiment_123 --async
```

## Data Issues

### Problem: Dataset Upload Fails

**Symptoms:**
```bash
Error: File format not supported
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solutions:**

1. **Check file format:**
```bash
file data.csv
head -5 data.csv

# Specify format explicitly
pynomaly datasets upload data.txt --format csv
```

2. **Fix encoding issues:**
```bash
# Detect encoding
chardet data.csv

# Convert encoding
iconv -f ISO-8859-1 -t UTF-8 data.csv > data_utf8.csv

# Upload with correct encoding
pynomaly datasets upload data.csv --encoding iso-8859-1
```

3. **Handle large files:**
```bash
# Upload sample first
pynomaly datasets upload large_file.csv \
  --sample-size 10000 \
  --name "Sample Dataset"

# Then upload full file in chunks
split -l 50000 large_file.csv chunk_
for chunk in chunk_*; do
  pynomaly datasets upload $chunk --append-to dataset_123
done
```

### Problem: Data Validation Errors

**Symptoms:**
```bash
Error: Dataset contains non-numeric data
ValueError: All features must be numeric for anomaly detection
```

**Solutions:**

1. **Check data types:**
```bash
pynomaly datasets show dataset_123 --statistics
```

2. **Convert categorical to numeric:**
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")
le = LabelEncoder()

# Convert categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))
```

3. **Remove non-numeric columns:**
```bash
pynomaly datasets preprocess dataset_123 \
  --remove-non-numeric \
  --output cleaned_dataset
```

4. **Handle missing values:**
```bash
pynomaly datasets preprocess dataset_123 \
  --fill-missing mean \
  --output complete_dataset
```

## CLI Issues

### Problem: Command Not Found

**Symptoms:**
```bash
bash: pynomaly: command not found
```

**Solutions:**

1. **Check installation:**
```bash
pip show pynomaly
which python
python -m pynomaly --help
```

2. **Add to PATH:**
```bash
export PATH="$HOME/.local/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

3. **Use full path:**
```bash
python -m pynomaly detectors list
~/.local/bin/pynomaly detectors list
```

### Problem: Permission Denied

**Symptoms:**
```bash
PermissionError: [Errno 13] Permission denied: '/var/log/pynomaly.log'
```

**Solutions:**

1. **Use user directory:**
```bash
export PYNOMALY_LOG_FILE="$HOME/.pynomaly/pynomaly.log"
mkdir -p ~/.pynomaly
```

2. **Fix permissions:**
```bash
sudo chown $USER:$USER /var/log/pynomaly.log
sudo chmod 644 /var/log/pynomaly.log
```

3. **Run without sudo:**
```bash
# Use user-writable directories
pynomaly --log-file ~/.pynomaly/app.log detectors list
```

## Web UI Issues

### Problem: Web UI Not Loading

**Symptoms:**
- Blank page in browser
- JavaScript errors in console
- CSS not loading

**Solutions:**

1. **Check server status:**
```bash
pynomaly server status
curl http://localhost:8000/ui/
```

2. **Clear browser cache:**
```javascript
// In browser developer console
localStorage.clear();
sessionStorage.clear();
location.reload(true);
```

3. **Check static files:**
```bash
# Verify static files exist
ls -la ~/.local/lib/python3.11/site-packages/pynomaly/presentation/web/static/

# Rebuild CSS if needed
cd pynomaly/presentation/web
npm run build-css
```

4. **Check browser console:**
```javascript
// Look for JavaScript errors
// Common issues:
// - CORS errors
// - Missing dependencies
// - Network connectivity
```

### Problem: Real-time Updates Not Working

**Symptoms:**
- Dashboard doesn't update automatically
- WebSocket connection errors

**Solutions:**

1. **Check WebSocket connection:**
```javascript
// In browser developer console
const ws = new WebSocket('ws://localhost:8000/ws/detections');
ws.onopen = () => console.log('WebSocket connected');
ws.onerror = (error) => console.error('WebSocket error:', error);
```

2. **Check firewall settings:**
```bash
# Allow WebSocket port
sudo ufw allow 8000
```

3. **Use polling fallback:**
```javascript
// If WebSocket fails, use polling
setInterval(() => {
  fetch('/api/v1/status')
    .then(response => response.json())
    .then(data => updateDashboard(data));
}, 5000);
```

## Getting Help

### Enable Debug Mode

```bash
# Maximum verbosity
export PYNOMALY_LOG_LEVEL=DEBUG
pynomaly --verbose detectors list
```

### Collect System Information

```bash
# Create diagnostic report
cat > diagnostic_info.txt << EOF
Pynomaly Version: $(pynomaly --version)
Python Version: $(python --version)
OS: $(uname -a)
Memory: $(free -h)
Disk: $(df -h)
Network: $(curl -s http://localhost:8000/health || echo "API not accessible")
Dependencies: $(pip list | grep -E "(pynomaly|numpy|pandas|scikit-learn)")
EOF
```

### Common Log Locations

```bash
# Default log locations
~/.pynomaly/logs/pynomaly.log
/var/log/pynomaly.log
/tmp/pynomaly.log

# View recent logs
tail -100 ~/.pynomaly/logs/pynomaly.log
```

### Contact Support

1. **Check documentation**: https://docs.pynomaly.com
2. **Search issues**: https://github.com/yourorg/pynomaly/issues
3. **Create issue**: Include diagnostic information and minimal reproduction case
4. **Community forum**: https://discuss.pynomaly.com

### Minimal Reproduction Case

When reporting issues, include:

```python
# Minimal example that reproduces the problem
import pynomaly

# Your code here
detector = pynomaly.create_detector("IsolationForest")
# ... steps to reproduce issue
```

```bash
# Command that fails
pynomaly detectors create "Test" IsolationForest --contamination 0.1

# Error output
# Include full error message and stack trace
```

## Advanced Debugging Techniques

### Memory Profiling and Analysis

```python
# Advanced memory profiling
import psutil
import tracemalloc
import gc
from datetime import datetime

class AdvancedMemoryProfiler:
    """Advanced memory profiling for Pynomaly operations."""

    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []

    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        print(f"🔍 Memory profiling started - Initial: {initial_memory:.2f} MB")

    def take_snapshot(self, label: str):
        """Take memory snapshot with label."""
        snapshot = tracemalloc.take_snapshot()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        self.snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'memory_mb': memory_mb,
            'timestamp': datetime.now()
        })

        print(f"📸 Snapshot '{label}': {memory_mb:.2f} MB")

    def analyze_top_memory_usage(self, limit=10):
        """Analyze top memory consumers."""
        if not self.snapshots:
            print("No snapshots available")
            return

        current = self.snapshots[-1]['snapshot']
        top_stats = current.statistics('lineno')

        print(f"\n🔝 Top {limit} Memory Consumers:")
        print("=" * 60)

        for i, stat in enumerate(top_stats[:limit]):
            print(f"{i+1:2d}. {stat}")

    def compare_snapshots(self, before_label: str, after_label: str):
        """Compare two memory snapshots."""
        before_snap = None
        after_snap = None

        for snap in self.snapshots:
            if snap['label'] == before_label:
                before_snap = snap
            elif snap['label'] == after_label:
                after_snap = snap

        if not before_snap or not after_snap:
            print("❌ Snapshots not found")
            return

        print(f"\n📊 Memory Comparison: {before_label} → {after_label}")
        print("=" * 60)

        memory_diff = after_snap['memory_mb'] - before_snap['memory_mb']
        print(f"Memory change: {memory_diff:+.2f} MB")

        # Compare statistics
        top_stats = after_snap['snapshot'].compare_to(
            before_snap['snapshot'], 'lineno'
        )

        print("\nTop Memory Changes:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

# Usage example
profiler = AdvancedMemoryProfiler()
profiler.start_profiling()

# Your code here
profiler.take_snapshot("before_detection")
# Run anomaly detection
profiler.take_snapshot("after_detection")

profiler.analyze_top_memory_usage()
profiler.compare_snapshots("before_detection", "after_detection")
```

### Performance Bottleneck Detection

```python
import time
import cProfile
import pstats
import io
from functools import wraps

class PerformanceProfiler:
    """Comprehensive performance profiling."""

    def __init__(self):
        self.profiler = None
        self.results = {}

    def profile_function(self, func):
        """Decorator to profile function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                profiler.disable()

                # Store results
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(20)

                self.results[func.__name__] = {
                    'duration': end_time - start_time,
                    'profile': s.getvalue()
                }

                print(f"⏱️  {func.__name__}: {end_time - start_time:.3f}s")

        return wrapper

    def print_detailed_profile(self, function_name: str):
        """Print detailed profile for function."""
        if function_name in self.results:
            print(f"\n🔍 Detailed Profile: {function_name}")
            print("=" * 60)
            print(self.results[function_name]['profile'])
        else:
            print(f"❌ No profile data for {function_name}")

# Usage
profiler = PerformanceProfiler()

@profiler.profile_function
def slow_detection_function():
    # Your detection code here
    time.sleep(1)  # Simulate work
    return "completed"

result = slow_detection_function()
profiler.print_detailed_profile("slow_detection_function")
```

### Database Query Analysis

```python
import sqlalchemy
from sqlalchemy import event
import time

class DatabaseProfiler:
    """Profile database query performance."""

    def __init__(self, engine):
        self.engine = engine
        self.queries = []
        self.setup_profiling()

    def setup_profiling(self):
        """Set up database query profiling."""

        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()

        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - context._query_start_time

            self.queries.append({
                'statement': statement,
                'parameters': parameters,
                'duration': total,
                'timestamp': time.time()
            })

            if total > 1.0:  # Log slow queries
                print(f"🐌 Slow query ({total:.3f}s): {statement[:100]}...")

    def get_slow_queries(self, threshold=0.5):
        """Get queries slower than threshold."""
        return [q for q in self.queries if q['duration'] > threshold]

    def print_query_summary(self):
        """Print query performance summary."""
        if not self.queries:
            print("No queries recorded")
            return

        total_time = sum(q['duration'] for q in self.queries)
        avg_time = total_time / len(self.queries)

        print(f"\n📊 Database Query Summary")
        print("=" * 40)
        print(f"Total queries: {len(self.queries)}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time: {avg_time:.3f}s")

        slow_queries = self.get_slow_queries(0.1)
        if slow_queries:
            print(f"Slow queries (>0.1s): {len(slow_queries)}")
            for query in slow_queries[-5:]:  # Show last 5 slow queries
                print(f"  {query['duration']:.3f}s: {query['statement'][:80]}...")

# Usage with Pynomaly
from pynomaly.infrastructure.config import create_container

container = create_container()
engine = container.database_engine()
profiler = DatabaseProfiler(engine)

# Run your operations
# ...

profiler.print_query_summary()
```

### Container and Kubernetes Diagnostics

```bash
#!/bin/bash
# comprehensive_k8s_diagnostics.sh

echo "🔍 Comprehensive Kubernetes Diagnostics for Pynomaly"
echo "=================================================="

NAMESPACE="pynomaly"

# Check namespace
echo "1. Namespace Status:"
kubectl get namespace $NAMESPACE

# Check all resources
echo -e "\n2. All Resources:"
kubectl get all -n $NAMESPACE

# Check persistent volumes
echo -e "\n3. Persistent Volumes:"
kubectl get pv,pvc -n $NAMESPACE

# Check config maps and secrets
echo -e "\n4. Configuration:"
kubectl get configmaps,secrets -n $NAMESPACE

# Check ingress
echo -e "\n5. Ingress:"
kubectl get ingress -n $NAMESPACE

# Check events (last 1 hour)
echo -e "\n6. Recent Events:"
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -20

# Detailed pod analysis
echo -e "\n7. Pod Detailed Status:"
for pod in $(kubectl get pods -n $NAMESPACE -o name); do
    echo -e "\n--- $pod ---"
    kubectl describe $pod -n $NAMESPACE | grep -A 10 -B 5 -E "(Status|Ready|Restart|Event|Error|Warning)"
done

# Resource usage
echo -e "\n8. Resource Usage:"
kubectl top pods -n $NAMESPACE

# Network diagnostics
echo -e "\n9. Network Diagnostics:"
kubectl run netshoot --image=nicolaka/netshoot --rm -i --restart=Never -- bash -c "
    echo 'DNS Resolution:'
    nslookup pynomaly-api-service.$NAMESPACE.svc.cluster.local
    echo 'Service Connectivity:'
    curl -v http://pynomaly-api-service.$NAMESPACE:8000/health
"

echo -e "\n✅ Diagnostics Complete"
```

### Log Analysis Tools

```python
import re
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict

class LogAnalyzer:
    """Advanced log analysis for Pynomaly."""

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.entries = []
        self.parse_logs()

    def parse_logs(self):
        """Parse structured JSON logs."""
        with open(self.log_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    entry['line_number'] = line_num
                    self.entries.append(entry)
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON at line {line_num}: {line.strip()[:100]}")

    def analyze_errors(self):
        """Analyze error patterns."""
        errors = [e for e in self.entries if e.get('level') == 'error']

        if not errors:
            print("✅ No errors found")
            return

        print(f"❌ Found {len(errors)} errors")

        # Group by error type
        error_types = Counter(e.get('error_type', 'Unknown') for e in errors)
        print("\nError Types:")
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count}")

        # Recent errors
        print(f"\nRecent Errors (last 5):")
        for error in errors[-5:]:
            timestamp = error.get('timestamp', 'N/A')
            message = error.get('message', 'N/A')
            print(f"  {timestamp}: {message}")

    def analyze_performance(self):
        """Analyze performance patterns."""
        perf_entries = [e for e in self.entries if 'duration_seconds' in e]

        if not perf_entries:
            print("No performance data found")
            return

        durations = [e['duration_seconds'] for e in perf_entries]

        print(f"📊 Performance Analysis ({len(perf_entries)} operations)")
        print(f"Average duration: {sum(durations)/len(durations):.3f}s")
        print(f"Max duration: {max(durations):.3f}s")
        print(f"Min duration: {min(durations):.3f}s")

        # Slow operations
        slow_ops = [e for e in perf_entries if e['duration_seconds'] > 5.0]
        if slow_ops:
            print(f"\n🐌 Slow operations (>{5.0}s): {len(slow_ops)}")
            for op in slow_ops[-3:]:  # Show last 3
                print(f"  {op.get('operation', 'N/A')}: {op['duration_seconds']:.3f}s")

    def analyze_security_events(self):
        """Analyze security-related events."""
        security_events = [e for e in self.entries if e.get('event_type') == 'security']

        if not security_events:
            print("✅ No security events found")
            return

        print(f"🔒 Found {len(security_events)} security events")

        # Group by event type
        event_types = Counter(e.get('security_event_type', 'Unknown') for e in security_events)

        for event_type, count in event_types.most_common():
            print(f"  {event_type}: {count}")

    def analyze_api_usage(self):
        """Analyze API usage patterns."""
        api_entries = [e for e in self.entries if e.get('event_type') == 'api_request']

        if not api_entries:
            print("No API request data found")
            return

        print(f"🌐 API Usage Analysis ({len(api_entries)} requests)")

        # Status code distribution
        status_codes = Counter(e.get('status_code') for e in api_entries)
        print("\nStatus Codes:")
        for code, count in status_codes.most_common():
            print(f"  {code}: {count}")

        # Popular endpoints
        endpoints = Counter(e.get('path') for e in api_entries)
        print("\nTop Endpoints:")
        for endpoint, count in endpoints.most_common(5):
            print(f"  {endpoint}: {count}")

        # Response time analysis
        response_times = [e.get('duration_seconds', 0) for e in api_entries]
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            print(f"\nAverage response time: {avg_response:.3f}s")

# Usage
analyzer = LogAnalyzer('/app/logs/pynomaly.log')
analyzer.analyze_errors()
analyzer.analyze_performance()
analyzer.analyze_security_events()
analyzer.analyze_api_usage()
```

## Emergency Procedures

### Production Incident Response

```bash
#!/bin/bash
# incident_response.sh

echo "🚨 Pynomaly Incident Response Procedure"
echo "======================================"

# 1. Immediate Assessment
echo "1. Immediate System Assessment:"
kubectl get pods -n pynomaly
kubectl get services -n pynomaly
curl -f http://api.pynomaly.com/health || echo "❌ API Health Check Failed"

# 2. Scale up resources if needed
echo "2. Emergency Scaling:"
kubectl scale deployment pynomaly-api --replicas=6 -n pynomaly
kubectl scale deployment pynomaly-worker --replicas=4 -n pynomaly

# 3. Check resource usage
echo "3. Resource Usage:"
kubectl top pods -n pynomaly
kubectl top nodes

# 4. Recent events
echo "4. Recent Events:"
kubectl get events -n pynomaly --sort-by='.lastTimestamp' | tail -10

# 5. Backup current state
echo "5. Creating State Backup:"
kubectl get all -n pynomaly -o yaml > incident_backup_$(date +%Y%m%d_%H%M%S).yaml

# 6. Collect logs
echo "6. Collecting Logs:"
mkdir -p incident_logs_$(date +%Y%m%d_%H%M%S)
for pod in $(kubectl get pods -n pynomaly -o name); do
    kubectl logs $pod -n pynomaly > incident_logs_$(date +%Y%m%d_%H%M%S)/${pod##*/}.log
done

echo "✅ Incident response procedures completed"
echo "Next steps:"
echo "  1. Analyze logs in incident_logs_* directory"
echo "  2. Check monitoring dashboards"
echo "  3. Implement specific fixes based on findings"
echo "  4. Document incident in postmortem"
```

### Rollback Procedures

```bash
#!/bin/bash
# rollback_procedure.sh

NAMESPACE="pynomaly"
PREVIOUS_VERSION="v1.2.3"  # Replace with actual version

echo "🔄 Pynomaly Rollback Procedure"
echo "============================="

# 1. Check current deployment status
echo "1. Current Deployment Status:"
kubectl rollout status deployment/pynomaly-api -n $NAMESPACE
kubectl rollout status deployment/pynomaly-worker -n $NAMESPACE

# 2. View rollout history
echo "2. Rollout History:"
kubectl rollout history deployment/pynomaly-api -n $NAMESPACE
kubectl rollout history deployment/pynomaly-worker -n $NAMESPACE

# 3. Rollback to previous version
echo "3. Rolling back to previous version..."
kubectl rollout undo deployment/pynomaly-api -n $NAMESPACE
kubectl rollout undo deployment/pynomaly-worker -n $NAMESPACE

# 4. Wait for rollback completion
echo "4. Waiting for rollback completion..."
kubectl rollout status deployment/pynomaly-api -n $NAMESPACE --timeout=300s
kubectl rollout status deployment/pynomaly-worker -n $NAMESPACE --timeout=300s

# 5. Verify rollback
echo "5. Verifying rollback..."
kubectl get pods -n $NAMESPACE
curl -f http://api.pynomaly.com/health && echo "✅ Health check passed" || echo "❌ Health check failed"

# 6. Update external services if needed
echo "6. Updating external services..."
# Add specific steps for your environment
# Example: Update load balancer, DNS, monitoring alerts

echo "✅ Rollback procedure completed"
echo "⚠️  Don't forget to:"
echo "  1. Update monitoring alerts"
echo "  2. Notify stakeholders"
echo "  3. Create incident report"
echo "  4. Plan fix for original issue"
```

This comprehensive troubleshooting guide covers advanced debugging techniques, diagnostic tools, and emergency procedures to help maintain reliable Pynomaly deployments across all environments.

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
