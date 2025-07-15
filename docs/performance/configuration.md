# Performance Testing Configuration Guide

This guide provides detailed information on configuring the Pynomaly Performance Regression Testing Framework for different environments and use cases.

## Configuration Files

### Main Configuration File

The primary configuration file (`performance_test_config.json`) controls all aspects of performance testing:

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
  "alert_webhooks": [],
  "test_scenarios": [
    {
      "name": "health_check",
      "type": "api",
      "endpoint": "/health",
      "method": "GET",
      "concurrent_users": 2,
      "duration": 10,
      "timeout": 30,
      "headers": {},
      "expected_status": [200]
    },
    {
      "name": "dashboard_api",
      "type": "api",
      "endpoint": "/api/v1/dashboard",
      "method": "GET",
      "concurrent_users": 3,
      "duration": 15,
      "headers": {
        "Authorization": "Bearer test-token"
      }
    },
    {
      "name": "system_resources",
      "type": "system",
      "duration": 10,
      "metrics": ["cpu_usage", "memory_usage", "disk_io"]
    },
    {
      "name": "database_performance",
      "type": "database",
      "connection_string": "postgresql://user:pass@localhost:5432/testdb",
      "query": "SELECT COUNT(*) FROM users WHERE active = true",
      "iterations": 100
    }
  ]
}
```

### Configuration Parameters

#### Global Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | `http://localhost:8000` | Base URL for API tests |
| `test_duration` | integer | `30` | Default test duration in seconds |
| `concurrent_users` | integer | `5` | Default concurrent users for load tests |
| `regression_threshold` | float | `20.0` | Percentage threshold for regression detection |
| `fail_on_regression` | boolean | `true` | Exit with error code when regressions detected |

#### Storage Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline_storage` | string | `performance_baselines` | Directory for baseline files |
| `tracking_db` | string | `performance_baselines/tracking.db` | SQLite database path |

#### Baseline Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_baseline_samples` | integer | `20` | Minimum samples to establish baseline |
| `outlier_threshold` | float | `3.0` | Standard deviations for outlier detection |
| `trend_window_days` | integer | `30` | Days of data for trend analysis |
| `baseline_update_threshold` | float | `0.15` | Threshold for automatic baseline updates |

#### Alert Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_alerts` | boolean | `true` | Enable/disable alert system |
| `alert_webhooks` | array | `[]` | Custom webhook URLs for alerts |

## Test Scenario Configuration

### API Test Scenarios

API tests perform HTTP requests with load simulation:

```json
{
  "name": "api_test_name",
  "type": "api",
  "endpoint": "/api/v1/endpoint",
  "method": "GET",
  "concurrent_users": 3,
  "duration": 15,
  "timeout": 30,
  "headers": {
    "Authorization": "Bearer token",
    "Content-Type": "application/json"
  },
  "body": {
    "key": "value"
  },
  "expected_status": [200, 201],
  "follow_redirects": true,
  "verify_ssl": true
}
```

**Parameters:**
- `name`: Unique identifier for the test
- `type`: Must be "api"
- `endpoint`: URL path (relative to base_url)
- `method`: HTTP method (GET, POST, PUT, DELETE, etc.)
- `concurrent_users`: Number of simultaneous users
- `duration`: Test duration in seconds
- `timeout`: Request timeout in seconds
- `headers`: HTTP headers to include
- `body`: Request body (for POST/PUT requests)
- `expected_status`: List of acceptable HTTP status codes
- `follow_redirects`: Whether to follow HTTP redirects
- `verify_ssl`: Whether to verify SSL certificates

### System Resource Tests

System tests monitor server resources during test execution:

```json
{
  "name": "system_monitor",
  "type": "system",
  "duration": 10,
  "interval": 1.0,
  "metrics": [
    "cpu_usage",
    "memory_usage", 
    "memory_available",
    "disk_io_read",
    "disk_io_write",
    "network_io_sent",
    "network_io_recv"
  ],
  "thresholds": {
    "cpu_usage": 80.0,
    "memory_usage": 85.0
  }
}
```

**Parameters:**
- `name`: Unique identifier for the test
- `type`: Must be "system"
- `duration`: Monitoring duration in seconds
- `interval`: Sampling interval in seconds
- `metrics`: List of metrics to collect
- `thresholds`: Warning thresholds for metrics

**Available System Metrics:**
- `cpu_usage`: CPU utilization percentage
- `memory_usage`: Memory usage percentage
- `memory_available`: Available memory in MB
- `disk_io_read`: Disk read rate in MB/s
- `disk_io_write`: Disk write rate in MB/s
- `network_io_sent`: Network send rate in MB/s
- `network_io_recv`: Network receive rate in MB/s

### Database Performance Tests

Database tests measure query execution performance:

```json
{
  "name": "db_query_test",
  "type": "database",
  "connection_string": "postgresql://user:pass@localhost:5432/db",
  "query": "SELECT * FROM users WHERE created_at > NOW() - INTERVAL '24 hours'",
  "iterations": 100,
  "connection_pool_size": 5,
  "timeout": 30,
  "parameters": {
    "param1": "value1"
  },
  "prepare_statements": true
}
```

**Parameters:**
- `name`: Unique identifier for the test
- `type`: Must be "database"
- `connection_string`: Database connection URL
- `query`: SQL query to execute
- `iterations`: Number of query executions
- `connection_pool_size`: Database connection pool size
- `timeout`: Query timeout in seconds
- `parameters`: Query parameters for prepared statements
- `prepare_statements`: Use prepared statements for performance

**Supported Database Types:**
- PostgreSQL: `postgresql://user:pass@host:port/db`
- MySQL: `mysql://user:pass@host:port/db`
- SQLite: `sqlite:///path/to/database.db`

## Environment-Specific Configurations

### Development Environment

```json
{
  "base_url": "http://localhost:8000",
  "test_duration": 10,
  "concurrent_users": 2,
  "regression_threshold": 30.0,
  "fail_on_regression": false,
  "min_baseline_samples": 5,
  "enable_alerts": false,
  "test_scenarios": [
    {
      "name": "health_check",
      "type": "api",
      "endpoint": "/health",
      "duration": 5,
      "concurrent_users": 1
    }
  ]
}
```

### Staging Environment

```json
{
  "base_url": "https://staging.example.com",
  "test_duration": 30,
  "concurrent_users": 5,
  "regression_threshold": 25.0,
  "fail_on_regression": true,
  "min_baseline_samples": 15,
  "enable_alerts": true,
  "test_scenarios": [
    {
      "name": "api_endpoints",
      "type": "api",
      "endpoint": "/api/v1/status",
      "duration": 15,
      "concurrent_users": 3,
      "headers": {
        "Authorization": "Bearer ${STAGING_API_TOKEN}"
      }
    },
    {
      "name": "system_load",
      "type": "system", 
      "duration": 20
    }
  ]
}
```

### Production Environment

```json
{
  "base_url": "https://api.example.com",
  "test_duration": 60,
  "concurrent_users": 10,
  "regression_threshold": 15.0,
  "fail_on_regression": true,
  "min_baseline_samples": 30,
  "outlier_threshold": 2.5,
  "enable_alerts": true,
  "alert_webhooks": [
    "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  ],
  "test_scenarios": [
    {
      "name": "critical_api",
      "type": "api",
      "endpoint": "/api/v1/health",
      "duration": 30,
      "concurrent_users": 5,
      "timeout": 10
    },
    {
      "name": "user_api",
      "type": "api",
      "endpoint": "/api/v1/users/profile",
      "method": "GET",
      "duration": 45,
      "concurrent_users": 8,
      "headers": {
        "Authorization": "Bearer ${PROD_API_TOKEN}"
      }
    },
    {
      "name": "database_queries",
      "type": "database",
      "connection_string": "${DATABASE_URL}",
      "query": "SELECT COUNT(*) FROM active_sessions",
      "iterations": 200
    }
  ]
}
```

## Environment Variables

### Framework Variables

```bash
# Test Configuration
export PYNOMALY_TEST_URL="http://localhost:8000"
export PERF_TEST_DURATION="30"
export PERF_CONCURRENT_USERS="5" 
export PERF_REGRESSION_THRESHOLD="20.0"
export PERF_FAIL_ON_REGRESSION="true"

# Storage Configuration
export PERF_BASELINE_STORAGE="performance_baselines"
export PERF_TRACKING_DB="performance_baselines/tracking.db"

# Baseline Configuration
export PERF_MIN_BASELINE_SAMPLES="20"
export PERF_OUTLIER_THRESHOLD="3.0"
export PERF_TREND_WINDOW_DAYS="30"
```

### Alert Configuration

```bash
# GitHub Integration
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
export GITHUB_REPOSITORY="owner/repo"
export GITHUB_PR_NUMBER="123"

# Slack Integration  
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export SLACK_CHANNEL="#performance-alerts"
export SLACK_USERNAME="Pynomaly Performance Bot"

# Email Integration
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="alerts@company.com"
export SMTP_PASSWORD="app_password"
export SMTP_FROM_EMAIL="noreply@company.com"
export ALERT_EMAIL_RECIPIENTS="dev-team@company.com,ops@company.com"

# Custom Webhooks
export CUSTOM_WEBHOOK_URL="https://api.company.com/alerts"
export CUSTOM_WEBHOOK_SECRET="webhook_secret"
```

### Database Configuration

```bash
# Primary Database
export DATABASE_URL="postgresql://user:pass@localhost:5432/pynomaly"

# Test Database
export TEST_DATABASE_URL="postgresql://test:pass@localhost:5432/pynomaly_test"

# Redis Cache
export REDIS_URL="redis://localhost:6379/0"
```

## Advanced Configuration

### Custom Test Types

You can extend the framework with custom test types:

```python
from pynomaly.infrastructure.performance.regression_framework import BasePerformanceTest

class CustomTest(BasePerformanceTest):
    def __init__(self, name: str, custom_param: str):
        super().__init__(name)
        self.custom_param = custom_param
    
    async def run(self) -> List[PerformanceMetric]:
        # Implement custom test logic
        metrics = []
        # ... custom implementation
        return metrics
```

Register custom test in configuration:

```json
{
  "name": "custom_test",
  "type": "custom",
  "custom_param": "value"
}
```

### Baseline Strategy Configuration

Configure different baseline strategies:

```python
from pynomaly.infrastructure.performance.baseline_tracker import BaselineConfig

# Conservative strategy (more stable, less sensitive)
conservative_config = BaselineConfig(
    min_samples=50,
    outlier_threshold=3.5,
    trend_window_days=60,
    baseline_update_threshold=0.20
)

# Aggressive strategy (more sensitive, faster adaptation)
aggressive_config = BaselineConfig(
    min_samples=10,
    outlier_threshold=2.0,
    trend_window_days=14,
    baseline_update_threshold=0.10
)
```

### Alert Severity Configuration

Customize alert severity thresholds:

```python
from pynomaly.infrastructure.performance.alert_system import AlertConfig

alert_config = AlertConfig(
    severity_thresholds={
        'critical': {'response_time_multiplier': 4.0, 'error_rate_threshold': 5.0},
        'high': {'response_time_multiplier': 3.0, 'error_rate_threshold': 2.0},
        'medium': {'response_time_multiplier': 2.0, 'error_rate_threshold': 1.0},
        'low': {'response_time_multiplier': 1.5, 'error_rate_threshold': 0.5}
    }
)
```

## Configuration Validation

### Schema Validation

The framework validates configuration against a JSON schema:

```bash
# Validate configuration file
python -c "
from pynomaly.infrastructure.performance.config_validator import validate_config
validate_config('performance_test_config.json')
"
```

### Common Configuration Errors

1. **Invalid URL format:**
```json
{
  "base_url": "not-a-valid-url"  // Should be "http://..." or "https://..."
}
```

2. **Missing required fields:**
```json
{
  "name": "test_name",
  "type": "api"
  // Missing "endpoint" field
}
```

3. **Invalid test duration:**
```json
{
  "duration": -5  // Should be positive integer
}
```

4. **Invalid threshold values:**
```json
{
  "regression_threshold": -10.0  // Should be positive
}
```

## Best Practices

### Configuration Management

1. **Use environment-specific configs:** Separate files for dev/staging/prod
2. **Store secrets in environment variables:** Never commit credentials
3. **Version control configurations:** Track configuration changes
4. **Validate before deployment:** Test configurations in staging first

### Performance Tuning

1. **Start with conservative settings:** Lower concurrent users, shorter durations
2. **Gradually increase load:** Monitor system impact
3. **Adjust thresholds based on historical data:** Use actual performance patterns
4. **Regular baseline maintenance:** Update baselines for known improvements

### Security Considerations

1. **Protect API tokens:** Use environment variables or secret management
2. **Limit test scope:** Don't test with production data
3. **Monitor test impact:** Ensure tests don't affect production performance
4. **Secure webhook endpoints:** Use authentication for custom alerts

## Troubleshooting Configuration

### Debug Configuration Loading

```bash
# Test configuration parsing
python scripts/ci/performance_regression_check.py --config config.json --verbose
```

### Common Issues

1. **Configuration file not found:**
   - Check file path and permissions
   - Ensure file is valid JSON

2. **Environment variables not resolved:**
   - Verify variable names and values
   - Use `env` command to check environment

3. **Database connection failures:**
   - Test connection string manually
   - Check network connectivity and credentials

4. **Alert channel setup issues:**
   - Verify webhook URLs and tokens
   - Test channels independently