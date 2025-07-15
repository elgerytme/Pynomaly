# Performance Regression Testing Framework

The Pynomaly Performance Regression Testing Framework provides comprehensive automated performance monitoring and regression detection for CI/CD pipelines. It includes baseline tracking, statistical analysis, multi-channel alerting, and detailed reporting capabilities.

## Quick Start

### Basic Usage

1. **Run performance tests manually:**
```bash
cd /path/to/pynomaly
python scripts/ci/performance_regression_check.py --config performance_test_config.json
```

2. **Enable in CI/CD pipeline:**
The framework automatically runs in GitHub Actions when you push to `main` or `develop` branches.

3. **View results:**
- Check GitHub Actions logs for immediate results
- Performance reports are saved as artifacts
- Alerts are sent through configured channels

### Configuration

Create a `performance_test_config.json` file:

```json
{
  "base_url": "http://localhost:8000",
  "test_duration": 30,
  "concurrent_users": 5,
  "regression_threshold": 20.0,
  "fail_on_regression": true,
  "test_scenarios": [
    {
      "name": "health_check",
      "type": "api",
      "endpoint": "/health",
      "method": "GET",
      "concurrent_users": 2,
      "duration": 10
    },
    {
      "name": "system_resources",
      "type": "system",
      "duration": 10
    }
  ]
}
```

## Framework Components

### 1. Performance Regression Framework (`regression_framework.py`)

**Core Classes:**
- `PerformanceRegressionFramework`: Main orchestrator
- `PerformanceMetric`: Individual metric data structure
- `PerformanceBaseline`: Statistical baseline representation
- `RegressionDetector`: Analyzes metrics against baselines

**Performance Tests:**
- `APIPerformanceTest`: HTTP endpoint load testing
- `DatabasePerformanceTest`: Database query performance
- `SystemResourceTest`: CPU, memory, disk monitoring

**Example Usage:**
```python
from pynomaly.infrastructure.performance.regression_framework import *

# Create framework
framework = PerformanceRegressionFramework("./baselines")

# Add API test
api_test = APIPerformanceTest(
    name="user_login",
    endpoint="http://localhost:8000/api/v1/auth/login",
    method="POST",
    concurrent_users=3,
    duration_seconds=15
)
framework.add_test(api_test)

# Run tests
results = await framework.run_tests()
```

### 2. Adaptive Baseline Tracker (`baseline_tracker.py`)

Manages performance baselines with trend analysis and automatic adaptation.

**Key Features:**
- SQLite-backed metric storage
- Statistical outlier detection
- Trend analysis with confidence intervals
- Automatic baseline updates

**Configuration:**
```python
from pynomaly.infrastructure.performance.baseline_tracker import *

config = BaselineConfig(
    min_samples=20,              # Minimum data points for baseline
    outlier_threshold=3.0,       # Standard deviations for outliers
    trend_window_days=30,        # Days of data for trend analysis
    baseline_update_threshold=0.15  # Threshold for baseline updates
)

tracker = AdaptiveBaselineTracker("tracking.db", config)
```

### 3. Alert System (`alert_system.py`)

Multi-channel alerting with throttling and deduplication.

**Supported Channels:**
- Console/Logging
- GitHub Issues/PR Comments
- Slack Webhooks
- Email (SMTP)

**Alert Configuration:**
```python
from pynomaly.infrastructure.performance.alert_system import *

config = AlertConfig(
    enabled=True,
    channels=['console', 'github', 'slack'],
    throttle_minutes=30,
    max_alerts_per_hour=10
)

alert_manager = PerformanceAlertManager(config)
```

**Environment Variables for Channels:**
```bash
# GitHub Integration
export GITHUB_TOKEN="your_token"
export GITHUB_REPOSITORY="owner/repo"
export GITHUB_PR_NUMBER="123"

# Slack Integration
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export SLACK_CHANNEL="#alerts"

# Email Integration
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your_email@gmail.com"
export SMTP_PASSWORD="your_password"
export SMTP_FROM_EMAIL="alerts@yourcompany.com"
export ALERT_EMAIL_RECIPIENTS="dev-team@yourcompany.com,ops@yourcompany.com"
```

### 4. Reporting Service (`reporting_service.py`)

Generates comprehensive performance reports with visualizations.

**Report Types:**
- HTML reports with charts
- JSON data exports
- CSV metric files
- Markdown summaries for CI/CD

**Example:**
```python
from pynomaly.infrastructure.performance.reporting_service import *

generator = PerformanceReportGenerator("./reports")

# Generate HTML report with charts
html_file = generator.generate_html_report(performance_data, historical_data)

# Generate CSV metrics for analysis
csv_file = generator.generate_csv_metrics(performance_data)
```

## CI/CD Integration

### GitHub Actions Workflow

The framework includes a comprehensive GitHub Actions workflow (`.github/workflows/performance-regression-testing.yml`) that:

1. **Triggers automatically** on:
   - Pushes to `main`/`develop` branches
   - Pull requests
   - Daily schedule (2 AM UTC)
   - Manual workflow dispatch

2. **Runs performance tests** with matrix strategy:
   - API Performance tests
   - Database Performance tests
   - System Resource tests

3. **Provides comprehensive reporting**:
   - Uploads artifacts
   - Comments on PRs
   - Updates baselines on main branch

### CLI Tool

The `scripts/ci/performance_regression_check.py` script provides:

**Command Line Options:**
```bash
python scripts/ci/performance_regression_check.py \
  --config performance_test_config.json \
  --output results/report.json \
  --format json \
  --fail-on-regression \
  --establish-baselines \
  --verbose
```

**Exit Codes:**
- `0`: All tests passed, no regressions
- `1`: Performance regressions detected

## Configuration Reference

### Framework Configuration

```json
{
  "base_url": "http://localhost:8000",
  "test_duration": 30,
  "concurrent_users": 5,
  "regression_threshold": 20.0,
  "fail_on_regression": true,
  "baseline_storage": "performance_baselines",
  "tracking_db": "performance_baselines/tracking.db",
  "min_baseline_samples": 20,
  "outlier_threshold": 3.0,
  "trend_window_days": 30,
  "baseline_update_threshold": 0.15,
  "enable_alerts": true,
  "test_scenarios": [...]
}
```

### Test Scenarios

**API Test:**
```json
{
  "name": "api_endpoint",
  "type": "api",
  "endpoint": "/api/v1/endpoint",
  "method": "GET",
  "concurrent_users": 3,
  "duration": 15,
  "headers": {"Authorization": "Bearer token"},
  "timeout": 30
}
```

**System Test:**
```json
{
  "name": "system_monitor",
  "type": "system",
  "duration": 10,
  "metrics": ["cpu_usage", "memory_usage", "disk_io"]
}
```

**Database Test:**
```json
{
  "name": "db_query",
  "type": "database",
  "query": "SELECT COUNT(*) FROM users",
  "iterations": 100,
  "connection_string": "postgresql://..."
}
```

## Metrics and Analysis

### Collected Metrics

**API Tests:**
- Response time (mean, p50, p95, p99)
- Error rate
- Throughput (requests/second)
- Response size

**System Tests:**
- CPU usage percentage
- Memory usage (MB)
- Disk I/O (MB/s)
- Network I/O (MB/s)

**Database Tests:**
- Query execution time
- Connection time
- Row count
- Query plan cost

### Statistical Analysis

**Baseline Establishment:**
- Minimum 20 data points required
- Outlier detection using z-score (configurable threshold)
- Statistical measures: mean, std, percentiles

**Regression Detection:**
- Z-score based analysis
- Configurable severity thresholds
- Confidence intervals for trend analysis

**Severity Levels:**
- **Critical**: >4 standard deviations from baseline
- **High**: 3-4 standard deviations
- **Medium**: 2-3 standard deviations  
- **Low**: 1.5-2 standard deviations

## Best Practices

### 1. Baseline Management

- **Establish baselines** during stable periods
- **Update baselines** periodically after verified improvements
- **Monitor baseline health** - refresh if health score < 0.7
- **Use sufficient data** - minimum 20 samples for statistical significance

### 2. Test Design

- **Keep tests focused** - test specific functionality
- **Use realistic load** - match production traffic patterns
- **Include warm-up** - allow application to reach steady state
- **Monitor resources** - include system metrics alongside API tests

### 3. CI/CD Integration

- **Run on stable environments** - use dedicated test infrastructure
- **Set appropriate thresholds** - balance sensitivity vs noise
- **Review alerts regularly** - tune throttling and channels
- **Archive historical data** - maintain performance trends

### 4. Alert Configuration

- **Use multiple channels** - ensure alerts reach the right people
- **Configure throttling** - prevent alert fatigue
- **Set severity-based routing** - critical alerts to on-call, low to email
- **Include context** - environment, commit info, trends

## Troubleshooting

### Common Issues

**1. No Baselines Available**
```
WARNING: No baseline found for metric 'response_time_mean'
```
**Solution:** Run tests with `--establish-baselines` flag or wait for automatic baseline establishment.

**2. Insufficient Data for Baseline**
```
ERROR: Need at least 10 data points to establish baseline
```
**Solution:** Run more test cycles to collect sufficient data points.

**3. Tests Failing to Connect**
```
ERROR: Connection failed to http://localhost:8000/health
```
**Solution:** Ensure the application is running and accessible at the configured URL.

**4. High False Positive Rate**
```
Multiple regression alerts for stable metrics
```
**Solution:** Increase `regression_threshold` or review baseline health and update if needed.

### Debug Mode

Enable verbose logging for detailed debugging:
```bash
python scripts/ci/performance_regression_check.py --verbose
```

### Manual Baseline Reset

If baselines become outdated:
```python
from pynomaly.infrastructure.performance.baseline_tracker import AdaptiveBaselineTracker

tracker = AdaptiveBaselineTracker("tracking.db")
tracker.reset_baseline("metric_name")  # Reset specific metric
tracker.reset_all_baselines()          # Reset all baselines
```

## Integration Examples

### Docker Integration

```dockerfile
FROM python:3.12-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Run performance tests
CMD ["python", "scripts/ci/performance_regression_check.py", "--config", "config.json"]
```

### Kubernetes Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: performance-regression-test
spec:
  template:
    spec:
      containers:
      - name: performance-test
        image: pynomaly:latest
        command: ["python", "scripts/ci/performance_regression_check.py"]
        args: ["--config", "/config/performance.json", "--fail-on-regression"]
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: performance-config
      restartPolicy: Never
```

## API Reference

For detailed API documentation, see:
- [Framework API](./api/regression_framework.md)
- [Baseline Tracker API](./api/baseline_tracker.md)
- [Alert System API](./api/alert_system.md)
- [Reporting API](./api/reporting_service.md)

## Contributing

1. Run the test suite: `pytest tests/performance/`
2. Check code style: `ruff check src/`
3. Update documentation for new features
4. Add integration tests for new test types

## License

This performance testing framework is part of the Pynomaly project and follows the same license terms.