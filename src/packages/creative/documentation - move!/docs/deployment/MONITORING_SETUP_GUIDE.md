# Pynomaly Production Monitoring Setup Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📊 Monitoring Setup

This comprehensive guide covers setting up production monitoring for Pynomaly using Prometheus, Grafana, and AlertManager with complete observability across all system components.

## 📋 Table of Contents

- [Overview](#overview)
- [Prometheus Setup](#prometheus-setup)
- [Grafana Configuration](#grafana-configuration)
- [AlertManager Setup](#alertmanager-setup)
- [Custom Metrics](#custom-metrics)
- [Dashboards](#dashboards)
- [Alerting Rules](#alerting-rules)
- [Log Aggregation](#log-aggregation)
- [Distributed Tracing](#distributed-tracing)
- [Performance Monitoring](#performance-monitoring)

## 🎯 Overview

### Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pynomaly API  │    │   PostgreSQL    │    │      Redis      │
│   (Metrics)     │    │   (Metrics)     │    │   (Metrics)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │     Prometheus          │
                    │   (Metrics Storage)     │
                    └─────────────┬───────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │      Grafana           │
                    │   (Visualization)      │
                    └─────────────┬───────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │   AlertManager         │
                    │   (Alerting)           │
                    └─────────────────────────┘
```

### Key Monitoring Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and management
- **Node Exporter**: System metrics
- **Postgres Exporter**: Database metrics
- **Redis Exporter**: Cache metrics
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing

## 🔧 Prometheus Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'pynomaly-production'
    region: 'us-east-1'

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Pynomaly API Metrics
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['pynomaly-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s
    params:
      format: ['prometheus']

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s

  # PostgreSQL Metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  # Redis Metrics
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

  # Nginx Metrics
  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['nginx-exporter:9113']
    scrape_interval: 30s

  # Blackbox Monitoring
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
        - https://api.pynomaly.com/health
        - https://api.pynomaly.com/metrics
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115

  # Kubernetes Metrics (if using K8s)
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  # Custom Application Metrics
  - job_name: 'pynomaly-workers'
    static_configs:
      - targets: ['worker-1:8001', 'worker-2:8001', 'worker-3:8001']
    metrics_path: '/worker/metrics'
    scrape_interval: 30s

# Remote Write Configuration (for long-term storage)
remote_write:
  - url: "https://prometheus-remote-write.example.com/api/v1/write"
    basic_auth:
      username: "pynomaly"
      password: "secure-password"
    write_relabel_configs:
      - source_labels: [__name__]
        regex: 'pynomaly_.*'
        action: keep
```

### Recording Rules

```yaml
# recording_rules.yml
groups:
- name: pynomaly.rules
  interval: 30s
  rules:
  # API Performance Rules
  - record: pynomaly:api_request_rate_5m
    expr: rate(http_requests_total{job="pynomaly-api"}[5m])

  - record: pynomaly:api_error_rate_5m
    expr: rate(http_requests_total{job="pynomaly-api",status=~"5.."}[5m])

  - record: pynomaly:api_latency_p95_5m
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="pynomaly-api"}[5m]))

  - record: pynomaly:api_latency_p99_5m
    expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job="pynomaly-api"}[5m]))

  # Detection Performance Rules
  - record: pynomaly:detection_rate_5m
    expr: rate(pynomaly_detections_total[5m])

  - record: pynomaly:detection_latency_p95_5m
    expr: histogram_quantile(0.95, rate(pynomaly_detection_duration_seconds_bucket[5m]))

  - record: pynomaly:model_accuracy_avg_1h
    expr: avg_over_time(pynomaly_model_accuracy[1h])

  # Resource Utilization Rules
  - record: pynomaly:cpu_utilization_5m
    expr: 100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

  - record: pynomaly:memory_utilization_5m
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

  - record: pynomaly:disk_utilization_5m
    expr: (1 - (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"})) * 100

  # Database Rules
  - record: pynomaly:db_connections_utilization_5m
    expr: pg_stat_database_numbackends / pg_settings_max_connections * 100

  - record: pynomaly:db_query_duration_p95_5m
    expr: histogram_quantile(0.95, rate(pg_stat_statements_total_time_bucket[5m]))

  - record: pynomaly:db_cache_hit_ratio_5m
    expr: (sum(pg_stat_database_blks_hit) / sum(pg_stat_database_blks_hit + pg_stat_database_blks_read)) * 100
```

### Docker Compose for Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert_rules.yml:/etc/prometheus/alert_rules.yml
      - ./recording_rules.yml:/etc/prometheus/recording_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=50GB'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--alertmanager.notification-queue-capacity=10000'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure-password
      - GF_SECURITY_ADMIN_USER=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=https://alertmanager.pynomaly.com'
      - '--cluster.advertise-address=0.0.0.0:9093'
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://pynomaly:password@postgres:5432/pynomaly?sslmode=disable
    depends_on:
      - postgres
    restart: unless-stopped

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  blackbox-exporter:
    image: prom/blackbox-exporter:latest
    container_name: blackbox-exporter
    ports:
      - "9115:9115"
    volumes:
      - ./blackbox.yml:/etc/blackbox_exporter/config.yml
    restart: unless-stopped

  loki:
    image: grafana/loki:latest
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped

  promtail:
    image: grafana/promtail:latest
    container_name: promtail
    volumes:
      - ./promtail.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
  loki_data:
```

## 📊 Grafana Configuration

### Datasource Configuration

```yaml
# grafana/provisioning/datasources/datasource.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
```

### Dashboard Provisioning

```yaml
# grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1

providers:
  - name: 'pynomaly-dashboards'
    orgId: 1
    folder: 'Pynomaly'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

### Main Pynomaly Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Pynomaly Production Monitoring",
    "tags": ["pynomaly", "production"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"pynomaly-api\"}[5m])) by (method, status)",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Detection Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(rate(pynomaly_detections_total[5m]))",
            "legendFormat": "Detections/sec"
          }
        ],
        "valueName": "current",
        "format": "ops",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Model Accuracy",
        "type": "singlestat",
        "targets": [
          {
            "expr": "avg(pynomaly_model_accuracy)",
            "legendFormat": "Accuracy"
          }
        ],
        "valueName": "current",
        "format": "percentunit",
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 6,
          "y": 8
        }
      },
      {
        "id": 5,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      },
      {
        "id": 6,
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends",
            "legendFormat": "Active Connections"
          },
          {
            "expr": "pg_settings_max_connections",
            "legendFormat": "Max Connections"
          }
        ],
        "yAxes": [
          {
            "label": "Connections",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 7,
        "title": "Redis Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_memory_used_bytes",
            "legendFormat": "Used Memory"
          },
          {
            "expr": "redis_memory_max_bytes",
            "legendFormat": "Max Memory"
          }
        ],
        "yAxes": [
          {
            "label": "Bytes",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 16
        }
      }
    ]
  }
}
```

## 🚨 AlertManager Setup

### AlertManager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@pynomaly.com'
  smtp_auth_username: 'alerts@pynomaly.com'
  smtp_auth_password: 'secure-password'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 5s
      repeat_interval: 30m
    - match:
        severity: warning
      receiver: 'warning-alerts'
      group_wait: 30s
      repeat_interval: 4h

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://127.0.0.1:5001/'

  - name: 'critical-alerts'
    email_configs:
      - to: 'ops-team@pynomaly.com'
        subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-critical'
        title: 'CRITICAL: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ end }}
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
        description: '{{ .GroupLabels.alertname }}'

  - name: 'warning-alerts'
    email_configs:
      - to: 'dev-team@pynomaly.com'
        subject: 'WARNING: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-warnings'
        title: 'WARNING: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
- name: pynomaly.alerts
  rules:
  # API Health Alerts
  - alert: APIHighErrorRate
    expr: rate(http_requests_total{job="pynomaly-api",status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High API error rate detected"
      description: "API error rate is {{ $value }} errors per second"

  - alert: APIHighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="pynomaly-api"}[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      description: "95th percentile latency is {{ $value }} seconds"

  - alert: APIDown
    expr: up{job="pynomaly-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Pynomaly API is down"
      description: "API service is not responding"

  # Detection Performance Alerts
  - alert: DetectionRateAnomaly
    expr: rate(pynomaly_detections_total[5m]) > 1000 or rate(pynomaly_detections_total[5m]) < 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Anomalous detection rate"
      description: "Detection rate is {{ $value }} detections per second"

  - alert: ModelAccuracyDegraded
    expr: avg(pynomaly_model_accuracy) < 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy has degraded"
      description: "Average model accuracy is {{ $value }}"

  # System Resource Alerts
  - alert: HighCPUUsage
    expr: 100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}%"

  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}%"

  - alert: DiskSpaceLow
    expr: (1 - (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"})) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low disk space detected"
      description: "Disk usage is {{ $value }}%"

  # Database Alerts
  - alert: DatabaseDown
    expr: up{job="postgres-exporter"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "PostgreSQL database is down"
      description: "Database is not responding"

  - alert: HighDatabaseConnections
    expr: pg_stat_database_numbackends / pg_settings_max_connections * 100 > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High database connection usage"
      description: "Database connection usage is {{ $value }}%"

  - alert: SlowDatabaseQueries
    expr: histogram_quantile(0.95, rate(pg_stat_statements_total_time_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow database queries detected"
      description: "95th percentile query time is {{ $value }} seconds"

  # Redis Alerts
  - alert: RedisDown
    expr: up{job="redis-exporter"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Redis cache is down"
      description: "Redis is not responding"

  - alert: HighRedisMemoryUsage
    expr: redis_memory_used_bytes / redis_memory_max_bytes * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High Redis memory usage"
      description: "Redis memory usage is {{ $value }}%"

  # Circuit Breaker Alerts
  - alert: CircuitBreakerOpen
    expr: pynomaly_circuit_breaker_state > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Circuit breaker is open"
      description: "Circuit breaker for {{ $labels.service }}/{{ $labels.operation }} is open"

  - alert: HighCircuitBreakerFailureRate
    expr: rate(pynomaly_circuit_breaker_failures_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High circuit breaker failure rate"
      description: "Circuit breaker failure rate is {{ $value }} failures per second"
```

## 📊 Custom Metrics

### Application Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time

# Core API Metrics
API_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

API_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Detection Metrics
DETECTIONS_TOTAL = Counter(
    'pynomaly_detections_total',
    'Total anomaly detections',
    ['model_type', 'dataset_type']
)

DETECTION_DURATION = Histogram(
    'pynomaly_detection_duration_seconds',
    'Detection processing time',
    ['model_type', 'dataset_size']
)

MODEL_ACCURACY = Gauge(
    'pynomaly_model_accuracy',
    'Model accuracy score',
    ['model_id', 'model_type']
)

# Circuit Breaker Metrics
CIRCUIT_BREAKER_STATE = Gauge(
    'pynomaly_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service', 'operation']
)

CIRCUIT_BREAKER_FAILURES = Counter(
    'pynomaly_circuit_breaker_failures_total',
    'Circuit breaker failures',
    ['service', 'operation']
)

# Database Metrics
DATABASE_CONNECTIONS_ACTIVE = Gauge(
    'pynomaly_database_connections_active',
    'Active database connections'
)

DATABASE_CONNECTIONS_MAX = Gauge(
    'pynomaly_database_connections_max',
    'Maximum database connections'
)

DATABASE_QUERY_DURATION = Histogram(
    'pynomaly_database_query_duration_seconds',
    'Database query duration',
    ['operation']
)

# Cache Metrics
CACHE_OPERATIONS_TOTAL = Counter(
    'pynomaly_cache_operations_total',
    'Total cache operations',
    ['operation', 'result']
)

CACHE_HIT_RATIO = Gauge(
    'pynomaly_cache_hit_ratio',
    'Cache hit ratio'
)

# Application Info
APP_INFO = Info(
    'pynomaly_app_info',
    'Application information'
)

def track_request_metrics(f):
    """Decorator to track API request metrics."""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = kwargs.get('method', 'GET')
        endpoint = kwargs.get('endpoint', 'unknown')
        
        try:
            result = await f(*args, **kwargs)
            status = '200'
            return result
        except Exception as e:
            status = '500'
            raise
        finally:
            duration = time.time() - start_time
            API_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
            API_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return wrapper

def track_detection_metrics(f):
    """Decorator to track detection metrics."""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        model_type = kwargs.get('model_type', 'unknown')
        dataset_type = kwargs.get('dataset_type', 'unknown')
        
        try:
            result = await f(*args, **kwargs)
            DETECTIONS_TOTAL.labels(model_type=model_type, dataset_type=dataset_type).inc()
            return result
        finally:
            duration = time.time() - start_time
            dataset_size = len(kwargs.get('data', []))
            size_bucket = get_size_bucket(dataset_size)
            DETECTION_DURATION.labels(model_type=model_type, dataset_size=size_bucket).observe(duration)
    
    return wrapper

def get_size_bucket(size):
    """Get dataset size bucket for metrics."""
    if size < 100:
        return 'small'
    elif size < 1000:
        return 'medium'
    elif size < 10000:
        return 'large'
    else:
        return 'xlarge'

def update_model_accuracy(model_id: str, model_type: str, accuracy: float):
    """Update model accuracy metric."""
    MODEL_ACCURACY.labels(model_id=model_id, model_type=model_type).set(accuracy)

def update_circuit_breaker_state(service: str, operation: str, state: int):
    """Update circuit breaker state metric."""
    CIRCUIT_BREAKER_STATE.labels(service=service, operation=operation).set(state)

def increment_circuit_breaker_failures(service: str, operation: str):
    """Increment circuit breaker failures."""
    CIRCUIT_BREAKER_FAILURES.labels(service=service, operation=operation).inc()

def update_database_metrics(active_connections: int, max_connections: int):
    """Update database connection metrics."""
    DATABASE_CONNECTIONS_ACTIVE.set(active_connections)
    DATABASE_CONNECTIONS_MAX.set(max_connections)

def track_database_query(operation: str, duration: float):
    """Track database query duration."""
    DATABASE_QUERY_DURATION.labels(operation=operation).observe(duration)

def update_cache_metrics(hit_ratio: float):
    """Update cache hit ratio."""
    CACHE_HIT_RATIO.set(hit_ratio)

def track_cache_operation(operation: str, result: str):
    """Track cache operation."""
    CACHE_OPERATIONS_TOTAL.labels(operation=operation, result=result).inc()

# Initialize application info
APP_INFO.info({
    'version': '1.0.0',
    'environment': 'production',
    'build_date': '2024-01-15',
    'git_commit': 'abc123'
})
```

### Metrics Endpoint

```python
# monitoring/endpoint.py
from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}
```

## 📈 Log Aggregation

### Loki Configuration

```yaml
# loki.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s
  max_transfer_retries: 0

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  retention_period: 744h  # 31 days

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 744h  # 31 days
```

### Promtail Configuration

```yaml
# promtail.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: pynomaly-api
    static_configs:
      - targets:
          - localhost
        labels:
          job: pynomaly-api
          __path__: /var/log/pynomaly/*.log
    pipeline_stages:
      - json:
          expressions:
            timestamp: timestamp
            level: level
            message: message
            service: service
      - timestamp:
          source: timestamp
          format: RFC3339
      - labels:
          level:
          service:

  - job_name: nginx
    static_configs:
      - targets:
          - localhost
        labels:
          job: nginx
          __path__: /var/log/nginx/*.log
    pipeline_stages:
      - regex:
          expression: '^(?P<remote_addr>\S+) - (?P<remote_user>\S+) \[(?P<time_local>[^\]]+)\] "(?P<method>\S+) (?P<request_uri>\S+) (?P<server_protocol>\S+)" (?P<status>\d+) (?P<body_bytes_sent>\d+) "(?P<http_referer>[^"]*)" "(?P<http_user_agent>[^"]*)"'
      - labels:
          method:
          status:
          remote_addr:

  - job_name: docker
    static_configs:
      - targets:
          - localhost
        labels:
          job: docker
          __path__: /var/lib/docker/containers/*/*-json.log
    pipeline_stages:
      - json:
          expressions:
            log: log
            stream: stream
            time: time
      - timestamp:
          source: time
          format: RFC3339Nano
      - labels:
          stream:
      - output:
          source: log
```

## 🔍 Distributed Tracing

### Jaeger Configuration

```yaml
# jaeger-all-in-one.yml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

volumes:
  elasticsearch_data:
```

### Application Tracing

```python
# monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def setup_tracing(app):
    """Setup distributed tracing."""
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Auto-instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()
    
    # Auto-instrument Redis
    RedisInstrumentor().instrument()
    
    return tracer

# Custom tracing decorator
def trace_function(operation_name):
    """Decorator to trace function calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(operation_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.result", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.result", "error")
                    span.set_attribute("function.error", str(e))
                    raise
        return wrapper
    return decorator
```

## 🔧 Deployment Script

```bash
#!/bin/bash
# deploy-monitoring.sh

set -e

echo "🚀 Deploying Pynomaly Monitoring Stack..."

# Create directories
mkdir -p monitoring/{prometheus,grafana,alertmanager,loki}
mkdir -p monitoring/grafana/{dashboards,provisioning/{dashboards,datasources}}

# Copy configuration files
cp prometheus.yml monitoring/prometheus/
cp alert_rules.yml monitoring/prometheus/
cp recording_rules.yml monitoring/prometheus/
cp alertmanager.yml monitoring/alertmanager/
cp loki.yml monitoring/loki/
cp promtail.yml monitoring/loki/

# Copy Grafana configurations
cp grafana/provisioning/datasources/datasource.yml monitoring/grafana/provisioning/datasources/
cp grafana/provisioning/dashboards/dashboard.yml monitoring/grafana/provisioning/dashboards/
cp grafana/dashboards/*.json monitoring/grafana/dashboards/

# Set permissions
chmod 644 monitoring/prometheus/*
chmod 644 monitoring/grafana/provisioning/**/*
chmod 644 monitoring/alertmanager/*

# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Verify services
echo "🔍 Verifying services..."
curl -f http://localhost:9090/-/healthy || { echo "❌ Prometheus not ready"; exit 1; }
curl -f http://localhost:3000/api/health || { echo "❌ Grafana not ready"; exit 1; }
curl -f http://localhost:9093/-/healthy || { echo "❌ AlertManager not ready"; exit 1; }

echo "✅ Monitoring stack deployed successfully!"
echo "📊 Grafana: http://localhost:3000 (admin/secure-password)"
echo "📈 Prometheus: http://localhost:9090"
echo "🚨 AlertManager: http://localhost:9093"
```

## 📚 Performance Monitoring

### Custom Performance Metrics

```python
# monitoring/performance.py
import psutil
import time
from prometheus_client import Gauge, Histogram

# System Performance Metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    ['mountpoint']
)

SYSTEM_NETWORK_BYTES = Gauge(
    'system_network_bytes_total',
    'System network bytes',
    ['direction', 'interface']
)

# Application Performance Metrics
APP_RESPONSE_TIME = Histogram(
    'app_response_time_seconds',
    'Application response time',
    ['endpoint', 'method']
)

APP_THROUGHPUT = Gauge(
    'app_throughput_requests_per_second',
    'Application throughput'
)

APP_ERROR_RATE = Gauge(
    'app_error_rate_percent',
    'Application error rate percentage'
)

class PerformanceCollector:
    """Collects system and application performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
    def collect_system_metrics(self):
        """Collect system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU_USAGE.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.percent)
        
        # Disk usage
        for disk in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(disk.mountpoint)
                SYSTEM_DISK_USAGE.labels(mountpoint=disk.mountpoint).set(usage.percent)
            except PermissionError:
                pass
        
        # Network usage
        network = psutil.net_io_counters(pernic=True)
        for interface, stats in network.items():
            SYSTEM_NETWORK_BYTES.labels(direction='sent', interface=interface).set(stats.bytes_sent)
            SYSTEM_NETWORK_BYTES.labels(direction='recv', interface=interface).set(stats.bytes_recv)
    
    def update_app_metrics(self, request_count: int, error_count: int):
        """Update application performance metrics."""
        self.request_count = request_count
        self.error_count = error_count
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        throughput = request_count / elapsed_time if elapsed_time > 0 else 0
        APP_THROUGHPUT.set(throughput)
        
        # Calculate error rate
        error_rate = (error_count / request_count * 100) if request_count > 0 else 0
        APP_ERROR_RATE.set(error_rate)
    
    def track_response_time(self, endpoint: str, method: str, duration: float):
        """Track response time for endpoint."""
        APP_RESPONSE_TIME.labels(endpoint=endpoint, method=method).observe(duration)

# Global performance collector instance
performance_collector = PerformanceCollector()

# Background task to collect metrics
async def collect_metrics_task():
    """Background task to collect performance metrics."""
    while True:
        performance_collector.collect_system_metrics()
        await asyncio.sleep(10)  # Collect every 10 seconds
```

This comprehensive monitoring setup provides full observability for Pynomaly production deployments with metrics, logging, tracing, and alerting capabilities.

---

## 🔗 Related Documentation

- **[Production Checklist](PRODUCTION_CHECKLIST.md)** - Pre-deployment validation
- **[Deployment Guide](deployment.md)** - Complete deployment instructions
- **[Security Hardening](SECURITY_HARDENING.md)** - Security best practices
- **[Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)** - Common issues and solutions

---

*Last Updated: 2024-01-15*