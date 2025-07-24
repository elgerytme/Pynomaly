# Monitoring Stack

This directory contains comprehensive monitoring templates and configurations for the anomaly detection platform.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification management
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics

## Deployment Options

### Docker Compose (Development/Testing)

```bash
# Start the complete monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access the services
# Grafana: http://localhost:3000 (admin/grafana)
# Prometheus: http://localhost:9090
# AlertManager: http://localhost:9093
# Jaeger: http://localhost:16686
```

### Kubernetes (Production)

```bash
# Create monitoring namespace
kubectl apply -f k8s/monitoring/namespace.yaml

# Deploy RBAC
kubectl apply -f k8s/monitoring/rbac.yaml

# Deploy monitoring resources
kubectl apply -f k8s/monitoring/servicemonitor.yaml
kubectl apply -f k8s/monitoring/podmonitor.yaml
kubectl apply -f k8s/monitoring/prometheusrule.yaml
```

### Helm Chart Integration

The monitoring components are integrated into the main Helm chart:

```yaml
# values.yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
    path: /metrics
```

## Dashboards

Pre-built Grafana dashboards are available:

### 1. Anomaly Detection Overview
- Service health status
- Request rates and response times
- Error rates and anomaly detection metrics
- Model performance indicators

### 2. Infrastructure Monitoring
- CPU, Memory, and Disk usage
- Network traffic
- Database and Redis metrics
- Container resource utilization

## Alerting Rules

Comprehensive alerting rules cover:

### Critical Alerts
- Service downtime
- High error rates (>10%)
- Low model accuracy (<80%)

### Warning Alerts
- High response times (>1s)
- Resource usage thresholds
- Database connection limits
- Cache memory limits

## Custom Metrics

The application exposes custom business metrics:

```python
# Example metrics exposed
anomaly_detections_total
model_inference_duration_seconds
model_accuracy_score
data_processing_duration_seconds
```

## Configuration Files

### Prometheus Configuration
- `prometheus/prometheus.yml`: Main configuration
- `prometheus/rules/`: Alert rules directory

### Grafana Configuration
- `grafana/provisioning/datasources.yaml`: Data source configuration
- `grafana/provisioning/dashboards.yaml`: Dashboard provisioning
- `grafana/dashboards/`: Dashboard JSON files

### AlertManager Configuration
- `alertmanager/alertmanager.yml`: Notification routing

## Health Endpoints

The application provides comprehensive health endpoints:

- `/api/health/live`: Liveness check
- `/api/health/ready`: Readiness check
- `/metrics`: Prometheus metrics
- `/api/v1/monitoring/metrics`: Custom business metrics

## Observability Integration

### Distributed Tracing
Jaeger integration provides end-to-end request tracing:

```python
from anomaly_detection.infrastructure.observability import TracingService

@TracingService.trace_function("model_inference")
def predict_anomaly(data):
    # Your model inference code
    pass
```

### Structured Logging
Logs are structured for easy aggregation:

```python
import structlog
logger = structlog.get_logger()

logger.info("anomaly_detected", 
           score=0.95, 
           timestamp=datetime.utcnow(),
           model_version="v1.2.0")
```

## Monitoring Best Practices

### 1. SLI/SLO Definition
Define Service Level Indicators and Objectives:

```yaml
# Example SLOs
availability: 99.9%
response_time_p95: 500ms
error_rate: <1%
model_accuracy: >90%
```

### 2. Alert Fatigue Prevention
- Use appropriate alert thresholds
- Implement alert escalation
- Regular alert rule review

### 3. Dashboard Design
- Focus on business metrics
- Use consistent color schemes
- Implement proper time ranges

## Troubleshooting

### Common Issues

1. **Metrics not appearing**
   - Check service discovery configuration
   - Verify metric endpoint accessibility
   - Review Prometheus target status

2. **Dashboard not loading**
   - Verify data source configuration
   - Check Grafana provisioning
   - Review dashboard JSON syntax

3. **Alerts not firing**
   - Validate alert rule expressions
   - Check AlertManager configuration
   - Verify notification channels

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Validate alert rules
promtool check rules prometheus/rules/*.yml

# Test dashboard JSON
grafana-cli admin export-dashboard <dashboard-id>
```

## Scaling Considerations

### High Availability
- Deploy Prometheus in HA mode
- Use external storage (Thanos, Cortex)
- Implement Grafana clustering

### Performance Optimization
- Adjust scrape intervals based on needs
- Use recording rules for expensive queries
- Implement metric retention policies

## Integration with CI/CD

The monitoring stack integrates with the deployment pipeline:

```yaml
# .github/workflows/deployment.yml
- name: Update monitoring dashboards
  run: |
    kubectl apply -f deploy/k8s/monitoring/
```

## Security

### Access Control
- RBAC for Kubernetes resources
- Authentication for Grafana
- Network policies for service isolation

### Data Protection
- Encrypt metrics in transit
- Secure dashboard sharing
- Audit access logs