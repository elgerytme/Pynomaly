# MLOps Platform Monitoring Setup

This directory contains the complete monitoring and alerting configuration for the MLOps platform, including Prometheus, Grafana, and Alertmanager.

## Quick Start

### Local Development Environment

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access services
open http://localhost:3000  # Grafana (admin/admin_password)
open http://localhost:9090  # Prometheus
open http://localhost:9093  # Alertmanager
```

### Kubernetes Staging Environment

```bash
# Deploy monitoring stack to staging
kubectl apply -f ../environments/staging/kubernetes/
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace mlops-staging \
  --values prometheus-values.yml
```

## Components

### Prometheus Configuration
- **File**: `prometheus.yml`
- **Purpose**: Metrics collection and storage
- **Targets**: Platform services, infrastructure, custom ML metrics
- **Retention**: 30 days

### Grafana Dashboards
- **mlops-overview.json**: High-level platform overview
- **model-performance.json**: Model-specific metrics and performance
- **business-metrics.json**: Business KPIs and ROI tracking

### Alert Rules
- **File**: `alert-rules.yml`
- **Categories**: Service health, performance, model drift, security, business metrics
- **Severity Levels**: Critical, Warning, Info

### Alertmanager Configuration
- **File**: `alertmanager.yml`
- **Routing**: Team-specific alert routing
- **Channels**: Slack, Email, PagerDuty webhooks
- **Inhibition**: Smart alert suppression

## Alert Configuration

### Severity Levels

#### Critical Alerts
- Service downtime
- Database connectivity issues
- Security violations
- Compliance violations
- Immediate PagerDuty notification

#### Warning Alerts
- High error rates (>5%)
- High latency (>100ms p95)
- Model accuracy drops
- Resource utilization issues
- Slack notifications

#### Info Alerts
- A/B test significance
- Deployment completions
- Capacity planning metrics

### Team Routing

```yaml
Critical Issues → #mlops-critical + PagerDuty + Email
Model Performance → #ml-team
Infrastructure → #devops-team  
Business Metrics → #business-metrics
Security → #security-alerts + Email
```

## Custom Metrics

### Model Performance Metrics
```prometheus
# Model accuracy tracking
model_accuracy{model_id="customer_churn", version="1.2"}

# Prediction latency
model_latency_seconds_bucket{model_id="customer_churn"}

# Prediction volume
rate(model_predictions_total{model_id="customer_churn"}[5m])

# Data drift detection
data_drift_score{model_id="customer_churn", feature_group="behavioral"}
```

### Business Metrics
```prometheus
# Revenue impact from ML models
ml_revenue_impact_daily

# Cost savings from automation
automation_cost_savings_daily

# Customer satisfaction
customer_satisfaction_score

# Platform adoption
platform_active_users
```

### Infrastructure Metrics
```prometheus
# Resource utilization
node_cpu_usage_percent
node_memory_usage_percent

# Service health
up{job="model-server"}

# Request metrics
http_requests_total{service="inference-engine"}
```

## Dashboard Access

### Grafana Login
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin_password

### Pre-configured Dashboards
1. **MLOps Platform Overview**
   - Service health status
   - Request rates and latency
   - Error rates
   - Resource utilization

2. **Model Performance**
   - Model accuracy trends
   - Latency distributions
   - Prediction volume
   - Data drift scores

3. **Business Metrics**
   - Revenue impact
   - Cost savings
   - ROI calculations
   - Customer satisfaction

## Alerting Setup

### Slack Integration
1. Create Slack webhook URLs for each channel
2. Update `alertmanager.yml` with webhook URLs
3. Test alert delivery

### Email Configuration
1. Configure SMTP settings in `alertmanager.yml`
2. Update recipient email addresses
3. Test email delivery

### PagerDuty Integration
1. Create PagerDuty integration
2. Update webhook URL in `alertmanager.yml`
3. Configure escalation policies

## Monitoring Best Practices

### Alert Fatigue Prevention
- Use appropriate thresholds
- Implement alert inhibition rules
- Regular alert rule review
- Clear runbook documentation

### Performance Optimization
- Monitor query performance
- Use recording rules for expensive queries
- Optimize dashboard refresh rates
- Implement proper retention policies

### Security Considerations
- Secure Grafana admin access
- Use TLS for metric endpoints
- Implement proper authentication
- Regular security updates

## Troubleshooting

### Common Issues

#### Prometheus Not Scraping Targets
```bash
# Check target health
curl http://localhost:9090/api/v1/targets

# Verify service discovery
kubectl get pods -n mlops-staging -l app=model-server

# Check network connectivity
kubectl exec -it prometheus-pod -- wget -O- http://model-server:8000/metrics
```

#### Grafana Dashboard Issues
```bash
# Check datasource connectivity
curl -u admin:admin_password http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up

# Verify dashboard provisioning
docker exec mlops-grafana ls -la /etc/grafana/provisioning/dashboards/
```

#### Alert Delivery Problems
```bash
# Test alertmanager config
curl -X POST http://localhost:9093/-/reload

# Check alert routing
curl http://localhost:9093/api/v1/alerts

# Verify webhook endpoints
curl -X POST your-slack-webhook-url -d '{"text": "test message"}'
```

### Log Analysis
```bash
# Prometheus logs
docker logs mlops-prometheus -f

# Grafana logs  
docker logs mlops-grafana -f

# Alertmanager logs
docker logs mlops-alertmanager -f
```

## Scaling Considerations

### High Availability
- Deploy multiple Prometheus instances
- Use external storage (Thanos/Cortex)
- Implement Grafana clustering
- Configure Alertmanager clustering

### Performance Tuning
- Optimize scrape intervals
- Use recording rules
- Implement metric filtering
- Configure proper retention

## Integration with CI/CD

### Automated Dashboard Updates
```yaml
# GitHub Actions workflow
- name: Update Grafana Dashboards
  run: |
    curl -X POST \
      -H "Authorization: Bearer $GRAFANA_API_KEY" \
      -H "Content-Type: application/json" \
      -d @grafana-dashboards/mlops-overview.json \
      http://grafana:3000/api/dashboards/db
```

### Alert Rule Testing
```bash
# Validate alert rules
promtool check rules alert-rules.yml

# Test alert firing
curl -X POST http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]={alertname="TestAlert"}
```

This monitoring setup provides comprehensive observability for the MLOps platform with proper alerting, dashboard visualization, and operational procedures.