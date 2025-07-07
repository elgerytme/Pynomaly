# Pynomaly Monitoring Stack

This directory contains the complete monitoring and observability infrastructure for Pynomaly.

## Architecture

The monitoring stack includes:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification
- **Elasticsearch**: Log aggregation and search
- **Kibana**: Log visualization and analysis
- **Jaeger**: Distributed tracing
- **Logstash**: Log processing and enrichment
- **Node Exporter**: System metrics
- **Redis Exporter**: Redis metrics
- **PostgreSQL Exporter**: Database metrics

## Quick Start

### 1. Start the Monitoring Stack

```bash
# Start all monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Check service status
docker-compose -f docker-compose.monitoring.yml ps
```

### 2. Access the Services

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093
- **Kibana**: http://localhost:5601
- **Jaeger**: http://localhost:16686
- **Elasticsearch**: http://localhost:9200

### 3. Configure Pynomaly Application

Ensure your Pynomaly application is configured to send metrics and logs:

```python
# In your application configuration
from pynomaly.infrastructure.observability import (
    configure_logging,
    MetricsCollector,
    setup_tracing,
    HealthMonitor
)

# Configure observability
configure_logging(level="INFO", enable_correlation=True)
metrics = MetricsCollector()
setup_tracing({"service_name": "pynomaly", "jaeger_endpoint": "http://localhost:14268"})
health = HealthMonitor()
```

## Configuration

### Prometheus

- **Configuration**: `prometheus/prometheus.yml`
- **Alert Rules**: `prometheus/alerts/pynomaly-alerts.yml`
- **Targets**: Automatically discovers Pynomaly services
- **Retention**: 30 days (configurable)

### Grafana

- **Dashboards**: Auto-provisioned from `grafana/dashboards/`
- **Datasources**: Auto-configured for Prometheus, Elasticsearch, Jaeger
- **Alerts**: Integrated with AlertManager
- **Plugins**: Pie chart, worldmap panels included

### AlertManager

- **Configuration**: `alertmanager/alertmanager.yml`
- **Routing**: Critical, warning, and info alerts
- **Notifications**: Email, Slack, webhooks
- **Silencing**: Support for maintenance windows

### Elasticsearch & Kibana

- **Index Pattern**: `pynomaly-logs-*`
- **Retention**: Managed by ILM policies
- **Security**: Disabled for development (enable for production)
- **Memory**: 512MB heap (adjust for production)

### Logstash

- **Pipeline**: `logstash/pipeline/pynomaly-logs.conf`
- **Inputs**: Beats, HTTP, TCP, UDP
- **Processing**: Error categorization, performance extraction
- **Outputs**: Elasticsearch with index routing

## Monitoring Targets

### Application Metrics

- HTTP request rates, durations, status codes
- Detection job metrics (success/failure rates)
- Training job metrics and durations
- Database connection pools
- Cache hit/miss ratios
- Memory and CPU usage

### Infrastructure Metrics

- System resources (CPU, memory, disk)
- Network connectivity
- Service health checks
- Database performance
- Redis performance
- Container metrics

### Log Categories

- **Application Logs**: Business logic, errors, performance
- **Access Logs**: HTTP requests, authentication
- **System Logs**: Infrastructure events
- **Security Logs**: Authentication, authorization events
- **Performance Logs**: Slow queries, high latency operations

## Alerting

### Alert Severity Levels

- **Critical**: Service down, high error rates (>25%), database connectivity
- **Warning**: Performance degradation, moderate error rates (>10%)
- **Info**: Deployment events, configuration changes

### Notification Channels

- **Email**: Team notifications
- **Slack**: Real-time team alerts
- **Webhooks**: Integration with external systems
- **PagerDuty**: On-call escalation (configure in AlertManager)

## Dashboard Highlights

### System Overview
- Service health status
- Request rate and error rate
- Response time percentiles
- System resource utilization

### Anomaly Detection
- Detection job success/failure rates
- Training job performance
- Model accuracy metrics
- Data processing throughput

### Infrastructure
- Container resource usage
- Database performance
- Cache performance
- Network connectivity

## Troubleshooting

### Common Issues

1. **Elasticsearch memory errors**:
   ```bash
   # Increase memory for Elasticsearch
   docker-compose -f docker-compose.monitoring.yml stop elasticsearch
   # Edit ES_JAVA_OPTS in docker-compose.monitoring.yml
   docker-compose -f docker-compose.monitoring.yml up -d elasticsearch
   ```

2. **Grafana dashboards not loading**:
   ```bash
   # Check provisioning
   docker-compose -f docker-compose.monitoring.yml logs grafana
   ```

3. **Prometheus targets down**:
   ```bash
   # Check network connectivity
   docker-compose -f docker-compose.monitoring.yml exec prometheus wget -O- http://pynomaly-api:8000/metrics
   ```

4. **AlertManager not sending alerts**:
   ```bash
   # Check configuration
   docker-compose -f docker-compose.monitoring.yml exec alertmanager amtool config show
   ```

### Log Analysis

```bash
# View specific service logs
docker-compose -f docker-compose.monitoring.yml logs -f prometheus
docker-compose -f docker-compose.monitoring.yml logs -f grafana
docker-compose -f docker-compose.monitoring.yml logs -f elasticsearch

# Check Logstash processing
curl -X GET "localhost:9600/_node/stats/pipeline"
```

## Production Considerations

### Security

1. Enable authentication for all services
2. Use TLS for inter-service communication
3. Implement proper access controls
4. Secure sensitive configuration in secrets

### Scalability

1. Use external storage for persistence
2. Configure Elasticsearch cluster
3. Set up Prometheus federation
4. Implement log rotation and archival

### High Availability

1. Deploy services across multiple nodes
2. Use load balancers for critical services
3. Implement backup and disaster recovery
4. Monitor the monitoring stack itself

### Performance Tuning

1. Adjust scrape intervals based on needs
2. Configure appropriate retention periods
3. Optimize Elasticsearch indices
4. Tune Grafana query performance

## Development

### Adding New Metrics

1. Add metric definition in `MetricsCollector`
2. Update Prometheus configuration if needed
3. Create or update Grafana dashboards
4. Add alert rules if applicable

### Adding New Alerts

1. Define alert rule in `prometheus/alerts/`
2. Configure routing in AlertManager
3. Test alert triggering and resolution
4. Document alert runbooks

### Custom Dashboards

1. Create dashboard in Grafana UI
2. Export JSON definition
3. Save to `grafana/dashboards/`
4. Add to version control

## Support

For monitoring-related issues:

1. Check service logs first
2. Verify network connectivity
3. Review configuration files
4. Consult service documentation
5. Open GitHub issue if needed