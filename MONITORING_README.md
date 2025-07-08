# Enhanced Monitoring System for Pynomaly

This document describes the enhanced monitoring, metrics, and alerting system for the Pynomaly anomaly detection framework.

## Overview

The monitoring system provides comprehensive observability for both batch and streaming anomaly detection processing, including:

- **Prometheus Metrics**: Detailed performance and operational metrics
- **Alerting System**: Rule-based alerting with multiple notification channels
- **Log-based Monitoring**: Pattern-based log analysis and alerting
- **SLA Monitoring**: Service level agreement violation detection
- **Grafana Dashboard**: Pre-built visualization dashboard

## Components

### 1. Prometheus Metrics Enhanced (`prometheus_metrics_enhanced.py`)

Provides comprehensive metrics collection including:

- **Job Duration**: Processing time for different job types
- **Anomalies Found**: Count of detected anomalies with severity categorization
- **Retry Count**: Number of job retries with failure reasons
- **Memory Usage**: Current and peak memory consumption
- **Processing Throughput**: Data processing rates
- **Error Rates**: Error counts by type and component
- **SLA Violations**: Service level agreement breach tracking

### 2. Alerting System (`alerting_system.py`)

Comprehensive alerting with:

- **Rule-based Alerts**: Configurable alert rules with conditions
- **Multiple Notification Channels**: Email, webhooks, Slack, log notifications
- **Log Pattern Matching**: Real-time log analysis for anomaly detection
- **SLA Monitoring**: Automatic SLA violation detection
- **Alert Management**: Cooldown periods, severity levels, and alert history

### 3. Enhanced BatchProcessor and ProcessingOrchestrator

Extended with metrics collection and alerting integration:

- **Automatic Metrics Collection**: Job duration, memory usage, error rates
- **Chunk-level Metrics**: Processing time per chunk with size categorization
- **Session Monitoring**: Active session tracking and lifecycle metrics
- **Resource Utilization**: CPU and memory usage monitoring

## Installation

### Prerequisites

```bash
pip install prometheus_client
pip install requests  # For webhook notifications
```

### Optional Dependencies

```bash
pip install grafana-api  # For Grafana integration
pip install slack-sdk    # For enhanced Slack notifications
```

## Quick Start

### 1. Basic Setup

```python
from src.pynomaly.infrastructure.monitoring.prometheus_metrics_enhanced import get_metrics_collector
from src.pynomaly.infrastructure.monitoring.alerting_system import get_alerting_system, setup_default_alerts

# Initialize metrics collector
metrics_collector = get_metrics_collector()

# Setup alerting system
alerting_system = get_alerting_system()
setup_default_alerts()

# Start monitoring
await alerting_system.start_monitoring()
```

### 2. Batch Processing with Metrics

```python
from src.pynomaly.infrastructure.batch.batch_processor import BatchProcessor, BatchConfig

# Create batch processor (metrics are automatically collected)
batch_config = BatchConfig(
    engine=BatchEngine.MULTIPROCESSING,
    max_workers=4,
    chunk_size=1000
)

batch_processor = BatchProcessor(batch_config)

# Submit job - metrics will be automatically collected
job_id = await batch_processor.submit_job(
    name="my_batch_job",
    description="Batch processing with metrics",
    input_path="data/input.csv",
    output_path="results/output.json"
)
```

### 3. Custom Alert Rules

```python
from src.pynomaly.infrastructure.monitoring.alerting_system import AlertRule, AlertType, AlertSeverity

# Create custom alert rule
custom_rule = AlertRule(
    name="high_memory_usage",
    description="Memory usage exceeds threshold",
    alert_type=AlertType.MEMORY_USAGE,
    severity=AlertSeverity.HIGH,
    condition="memory_usage > 0.8",
    threshold=0.8,
    duration_seconds=300,
    cooldown_seconds=1800
)

alerting_system.add_rule(custom_rule)
```

### 4. Notification Configuration

```python
from src.pynomaly.infrastructure.monitoring.alerting_system import NotificationConfig, NotificationType

# Email notifications
email_config = NotificationConfig(
    type=NotificationType.EMAIL,
    enabled=True,
    config={
        "from": "alerts@yourcompany.com",
        "to": "admin@yourcompany.com",
        "smtp_server": "smtp.yourcompany.com",
        "smtp_port": 587,
        "use_tls": True,
        "username": "alerts@yourcompany.com",
        "password": "your_password"
    },
    min_severity=AlertSeverity.HIGH
)

# Webhook notifications
webhook_config = NotificationConfig(
    type=NotificationType.WEBHOOK,
    enabled=True,
    config={
        "url": "https://your-webhook-url.com/alerts",
        "headers": {"Authorization": "Bearer your-token"},
        "timeout": 30
    },
    min_severity=AlertSeverity.CRITICAL
)

alerting_system.add_notification_config(email_config)
alerting_system.add_notification_config(webhook_config)
```

## Metrics Reference

### Job Metrics

- `pynomaly_job_duration_seconds`: Job execution time
- `pynomaly_jobs_total`: Total number of jobs processed
- `pynomaly_active_jobs`: Currently active jobs
- `pynomaly_job_retries_total`: Number of job retries

### Anomaly Detection Metrics

- `pynomaly_anomalies_found_total`: Total anomalies detected
- `pynomaly_processing_latency_seconds`: Processing latency
- `pynomaly_data_throughput_records_per_second`: Data processing throughput

### Resource Metrics

- `pynomaly_memory_usage_mb`: Current memory usage
- `pynomaly_peak_memory_usage_mb`: Peak memory usage
- `pynomaly_cpu_usage_percent`: CPU utilization

### Alert Metrics

- `pynomaly_alerts_total`: Total alerts generated
- `pynomaly_notifications_total`: Total notifications sent
- `pynomaly_sla_violations_total`: SLA violations count

## Grafana Dashboard

### Installation

1. Import the provided `grafana_dashboard.json` into your Grafana instance
2. Configure Prometheus as a data source
3. Customize panels as needed

### Dashboard Features

- **Job Processing Overview**: Real-time job status and performance
- **Anomaly Detection Summary**: Anomaly counts and severity distribution
- **Resource Utilization**: Memory and CPU usage trends
- **Alert Status**: Active alerts and alert history
- **SLA Monitoring**: SLA compliance and violations

### Sample Queries

```promql
# Job completion rate
rate(pynomaly_jobs_total{status="completed"}[5m])

# Average job duration
rate(pynomaly_job_duration_seconds_sum[5m]) / rate(pynomaly_job_duration_seconds_count[5m])

# Memory usage by component
avg(pynomaly_memory_usage_mb) by (component)

# Alert rate by severity
rate(pynomaly_alerts_total[5m]) by (severity)
```

## Log-based Alerting

### Pattern Configuration

```python
# Add log patterns for monitoring
alerting_system.add_log_pattern(
    "error_pattern",
    r"ERROR|FATAL|CRITICAL",
    AlertSeverity.MEDIUM,
    threshold=5,
    window_seconds=300
)

alerting_system.add_log_pattern(
    "memory_error_pattern",
    r"OutOfMemoryError|MemoryError|out of memory",
    AlertSeverity.HIGH,
    threshold=1,
    window_seconds=60
)
```

### Log Processing

```python
# Process log entries (integrate with your logging system)
log_entry = "ERROR: Database connection failed"
alerting_system.process_log_entry(log_entry)
```

## SLA Monitoring

### Configuration

```python
# Add SLA configurations
alerting_system.add_sla_config(
    "batch_processing_sla",
    "avg_job_duration",
    threshold=1800,  # 30 minutes
    duration_seconds=300
)

alerting_system.add_sla_config(
    "streaming_latency_sla",
    "avg_processing_latency",
    threshold=5.0,  # 5 seconds
    duration_seconds=300
)
```

### Monitoring

SLA violations are automatically detected and can trigger alerts with appropriate severity levels.

## Production Deployment

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pynomaly'
    static_configs:
      - targets: ['localhost:8000']  # Your application endpoint
    scrape_interval: 30s
    metrics_path: '/metrics'
```

### Environment Variables

```bash
# Optional environment variables
export PROMETHEUS_PUSHGATEWAY_URL=http://localhost:9091
export ALERTING_WEBHOOK_URL=https://your-webhook-url.com/alerts
export SMTP_SERVER=smtp.yourcompany.com
export SMTP_PORT=587
export ALERT_EMAIL_FROM=alerts@yourcompany.com
export ALERT_EMAIL_TO=admin@yourcompany.com
```

### Docker Deployment

```dockerfile
# Add to your Dockerfile
RUN pip install prometheus_client requests

# Copy monitoring configuration
COPY monitoring_config.py /app/
COPY grafana_dashboard.json /app/

# Expose metrics port
EXPOSE 8000
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**: Check if `prometheus_client` is installed and metrics endpoint is accessible
2. **Alerts not firing**: Verify alert rules are enabled and monitoring is started
3. **Notification failures**: Check notification configuration and network connectivity
4. **High memory usage**: Monitor and adjust chunk sizes and worker counts

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check metrics availability
metrics_collector = get_metrics_collector()
if not metrics_collector.is_available():
    print("Prometheus client not available")

# Check alert system status
alerting_system = get_alerting_system()
stats = alerting_system.get_alert_statistics()
print(f"Alert system stats: {stats}")
```

## Best Practices

1. **Metric Naming**: Use consistent naming conventions with the `pynomaly_` prefix
2. **Alert Tuning**: Start with conservative thresholds and adjust based on baseline metrics
3. **Notification Management**: Use severity levels to route alerts appropriately
4. **Dashboard Design**: Focus on key metrics and avoid cluttered visualizations
5. **SLA Definition**: Define realistic SLAs based on historical performance data

## Example Usage

See `examples/monitoring_example.py` for a comprehensive example of how to use the monitoring system.

## Contributing

When adding new metrics or alert types:

1. Update the `PrometheusMetricsCollector` class
2. Add corresponding alert rules in `AlertingSystem`
3. Update the Grafana dashboard configuration
4. Add tests for new functionality
5. Update this documentation

## License

This monitoring system is part of the Pynomaly project and follows the same license terms.
