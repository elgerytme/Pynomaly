# Advanced Monitoring and Observability Guide

This comprehensive guide covers the advanced monitoring and observability system for Pynomaly, including ML-based anomaly detection, predictive capacity planning, SLO monitoring, and business intelligence dashboards.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Components](#components)
5. [Configuration](#configuration)
6. [Dashboards](#dashboards)
7. [Alerting](#alerting)
8. [Capacity Planning](#capacity-planning)
9. [SLO Monitoring](#slo-monitoring)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)
12. [Best Practices](#best-practices)

## Overview

The Pynomaly Advanced Monitoring System provides enterprise-grade observability with the following key features:

### 🤖 **Machine Learning-Powered Monitoring**
- **Intelligent Anomaly Detection**: ML algorithms automatically detect unusual patterns in metrics
- **Predictive Capacity Planning**: Forecast resource needs and prevent capacity issues
- **Smart Alert Correlation**: Reduce alert noise through intelligent grouping and suppression

### 📊 **Comprehensive Observability**
- **Multi-layered Metrics**: System, application, and business metrics
- **Distributed Tracing**: Full request flow visibility with OpenTelemetry
- **Centralized Logging**: ELK stack for log aggregation and analysis
- **Real-time Dashboards**: Live monitoring with WebSocket updates

### 🎯 **Business Intelligence**
- **Executive Dashboards**: High-level KPIs and business metrics
- **Technical Dashboards**: Detailed performance and system health
- **SLO/SLA Monitoring**: Service level objective tracking and burn rate alerts
- **Cost Analysis**: Resource utilization and cost optimization insights

### 🔧 **Production-Ready Features**
- **High Availability**: Distributed architecture with redundancy
- **Security**: Authentication, authorization, and audit logging
- **Scalability**: Auto-scaling based on load and resource utilization
- **Integration**: APIs for custom integrations and automation

## Architecture

### System Components

```mermaid
graph TB
    subgraph "Data Sources"
        A[Application Metrics] --> D[Prometheus]
        B[Application Logs] --> E[Filebeat]
        C[Traces] --> F[Jaeger]
    end
    
    subgraph "Processing Layer"
        D --> G[Advanced Monitoring]
        E --> H[Logstash] --> I[Elasticsearch]
        F --> G
        G --> J[Anomaly Detector]
        G --> K[Capacity Planner]
        G --> L[SLO Monitor]
    end
    
    subgraph "Storage"
        I[Elasticsearch]
        M[PostgreSQL]
        N[Redis Cache]
    end
    
    subgraph "Visualization"
        O[BI Dashboard]
        P[Kibana]
        Q[Jaeger UI]
        R[Grafana]
    end
    
    G --> M
    G --> N
    J --> O
    K --> O
    L --> O
    I --> P
    F --> Q
    D --> R
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Metrics Collection** | Prometheus | Time-series metrics storage |
| **Log Aggregation** | ELK Stack | Centralized logging |
| **Distributed Tracing** | Jaeger | Request flow tracking |
| **ML Processing** | scikit-learn, pandas | Anomaly detection & forecasting |
| **Dashboards** | FastAPI, Plotly | Interactive visualizations |
| **Alerting** | AlertManager | Intelligent alert routing |
| **Database** | PostgreSQL | Metadata and historical data |
| **Caching** | Redis | High-performance caching |
| **Orchestration** | Docker Compose | Service management |

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM (for full stack)
- 50GB+ disk space

### Installation

1. **Clone and navigate to the project:**
```bash
cd /path/to/pynomaly
```

2. **Deploy the monitoring stack:**
```bash
chmod +x scripts/deployment/deploy_advanced_monitoring.sh
./scripts/deployment/deploy_advanced_monitoring.sh deploy
```

3. **Wait for services to start (5-10 minutes):**
```bash
./scripts/deployment/deploy_advanced_monitoring.sh status
```

4. **Access the dashboards:**
- **Business Intelligence Dashboard**: http://localhost:8080
- **Kibana (Logs)**: http://localhost:5601
- **Jaeger (Tracing)**: http://localhost:16686
- **Advanced Monitoring API**: http://localhost:8090

### Verification

Check that all services are running:
```bash
curl http://localhost:8080/health
curl http://localhost:8090/api/system-health
curl http://localhost:5601/api/status
curl http://localhost:16686/api/services
```

## Components

### 1. Advanced Monitoring Orchestrator

**Location**: `src/pynomaly/infrastructure/monitoring/advanced_monitoring.py`

The central component that coordinates all monitoring activities:

```python
from src.pynomaly.infrastructure.monitoring.advanced_monitoring import AdvancedMonitoringOrchestrator

# Initialize monitoring
monitor = AdvancedMonitoringOrchestrator()

# Process metrics
await monitor.process_metric_data('cpu_usage', 75.0)

# Update SLI measurements
await monitor.update_sli_measurement('api_success_rate', 0.999)

# Generate capacity report
report = await monitor.generate_capacity_report(30)
```

**Features**:
- ML-based anomaly detection
- Intelligent alert correlation
- SLO compliance monitoring
- Capacity planning automation
- Real-time metric processing

### 2. Business Intelligence Dashboard

**Location**: `src/pynomaly/infrastructure/monitoring/business_intelligence_dashboard.py`

Comprehensive dashboards for different audiences:

```python
from src.pynomaly.infrastructure.monitoring.business_intelligence_dashboard import BusinessIntelligenceDashboard

# Initialize dashboard
dashboard = BusinessIntelligenceDashboard(monitor)

# Update KPIs
await dashboard.update_kpi("system_availability", 99.95)

# Generate dashboard data
data = await dashboard.generate_dashboard_data(DashboardType.EXECUTIVE)

# Start dashboard server
dashboard.run(host="0.0.0.0", port=8080)
```

**Dashboard Types**:
- **Executive**: Business metrics, KPIs, ROI analysis
- **Technical**: Performance metrics, system health, errors
- **Operations**: Live alerts, incident management, deployments
- **Security**: Security events, compliance, audit logs

### 3. ML-Based Anomaly Detection

The system uses multiple algorithms for robust anomaly detection:

```python
from src.pynomaly.infrastructure.monitoring.advanced_monitoring import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(contamination=0.1)

# Train on historical data
detector.fit(historical_data)

# Detect anomalies in new data
anomalies = detector.detect_anomalies(new_data)
scores = detector.get_anomaly_scores(new_data)
```

**Algorithms**:
- **Isolation Forest**: Unsupervised outlier detection
- **DBSCAN**: Density-based clustering for anomalies
- **Statistical Methods**: Z-score and standard deviation based
- **Ensemble Methods**: Combining multiple algorithms

### 4. Predictive Capacity Planning

Automated forecasting and capacity recommendations:

```python
from src.pynomaly.infrastructure.monitoring.advanced_monitoring import CapacityPlanner

# Initialize planner
planner = CapacityPlanner()

# Add metrics to monitor
planner.add_metric('cpu_usage', LinearTrendPredictor(), 
                  {'warning': 70.0, 'critical': 85.0})

# Generate predictions
predictions = planner.generate_capacity_predictions(30)

for prediction in predictions:
    print(f"Metric: {prediction.metric_name}")
    print(f"Current: {prediction.current_value}")
    print(f"Predicted: {prediction.predicted_value}")
    print(f"Action: {prediction.recommended_action}")
```

**Prediction Models**:
- **Linear Trend**: Simple trend-based forecasting
- **Seasonal Decomposition**: Handles seasonal patterns
- **ARIMA**: Advanced time series forecasting
- **Prophet**: Facebook's forecasting algorithm

### 5. SLO/SLI Monitoring

Comprehensive Service Level Objective tracking:

```python
from src.pynomaly.infrastructure.monitoring.advanced_monitoring import SLOMonitor, SLO, SLI

# Initialize SLO monitor
slo_monitor = SLOMonitor()

# Define SLO
slo = SLO(
    name='api_availability',
    sli=SLI(
        name='api_success_rate',
        type=SLIType.AVAILABILITY,
        query='rate(http_requests_total{status!~"5.."}[5m]) / rate(http_requests_total[5m])',
        target_value=0.99,
        time_window='5m',
        description='API success rate'
    ),
    target_percentage=99.9,
    time_period='30d',
    description='API availability SLO'
)

# Add SLO
slo_monitor.add_slo(slo)

# Check compliance
compliance = slo_monitor.calculate_slo_compliance('api_availability')
```

## Configuration

### Main Configuration File

**Location**: `config/advanced_monitoring.yml`

The main configuration file controls all aspects of the monitoring system:

```yaml
# Example configuration snippet
global:
  service_name: "pynomaly"
  environment: "production"
  data_retention_days: 90

components:
  anomaly_detection:
    enabled: true
    algorithms:
      - name: "isolation_forest"
        contamination: 0.1
        enabled: true
    
  alert_correlation:
    enabled: true
    correlation_window_minutes: 10
    
  slo_monitoring:
    enabled: true
    
  capacity_planning:
    enabled: true
```

### Environment Variables

```bash
# Database configuration
DATABASE_URL=postgresql://pynomaly:password@postgres:5432/pynomaly_monitoring

# Service endpoints
PROMETHEUS_HOST=prometheus:9090
ELASTICSEARCH_HOST=elasticsearch:9200
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# Caching
REDIS_URL=redis://redis:6379

# Security
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PAGERDUTY_API_KEY=your-pagerduty-key
```

### SLO Configuration

Define Service Level Objectives in `config/slos.yml`:

```yaml
slos:
  - name: "api_availability"
    description: "API must be available 99.9% of the time"
    sli:
      name: "http_success_rate"
      type: "availability"
      query: "rate(http_requests_total{status!~'5..'}[5m]) / rate(http_requests_total[5m])"
      target_value: 0.99
    target_percentage: 99.9
    time_period: "30d"
    burn_rate_thresholds:
      critical: 10.0
      high: 5.0
      warning: 2.0
```

## Dashboards

### Executive Dashboard

**URL**: http://localhost:8080/dashboard/executive

Key features:
- **Business KPIs**: Revenue, user growth, conversion rates
- **System Health**: Overall availability and performance
- **Cost Analysis**: Resource usage and optimization opportunities
- **SLO Compliance**: Service level objective tracking

### Technical Dashboard

**URL**: http://localhost:8080/dashboard/technical

Key features:
- **Performance Metrics**: Response times, throughput, error rates
- **System Resources**: CPU, memory, disk, network utilization
- **Anomaly Detection**: ML-detected anomalies and confidence scores
- **Distributed Tracing**: Request flow and dependency mapping

### Operations Dashboard

**URL**: http://localhost:8080/dashboard/operations

Key features:
- **Live Alerts**: Real-time alert feed with severity filtering
- **Incident Management**: Active incidents and resolution tracking
- **Deployment Status**: Release health and rollback capabilities
- **Capacity Planning**: Resource forecasts and recommendations

## Alerting

### Alert Correlation Engine

The system includes intelligent alert correlation to reduce noise:

```python
# Configure correlation rules
correlation_engine.add_correlation_rule({
    'name': 'service_related',
    'conditions': {
        'source_pattern': 'service',
        'label_overlap': 1,
        'severity_correlation': 2
    }
})

# Configure suppression rules
correlation_engine.add_suppression_rule({
    'name': 'duplicate_suppression',
    'conditions': {
        'time_window': 300,  # 5 minutes
        'max_alerts_in_window': 1
    }
})
```

### Alert Routing

Alerts are routed based on severity and type:

```yaml
# AlertManager configuration
routes:
  - match:
      severity: "critical"
    receiver: "critical_alerts"
    group_wait: "0s"
    repeat_interval: "15m"
    
  - match:
      alertname: "SLOBurnRate"
    receiver: "slo_alerts"
    repeat_interval: "30m"
```

### Notification Channels

- **Email**: SMTP integration for email alerts
- **Slack**: Webhook integration for team notifications
- **PagerDuty**: Integration for on-call escalation
- **Webhook**: Custom webhook for integrations

## Capacity Planning

### Automated Forecasting

The system automatically generates capacity forecasts:

```bash
# Generate 30-day capacity forecast
curl http://localhost:8090/api/capacity-report?forecast_days=30
```

Response:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "forecast_days": 30,
  "predictions": [
    {
      "metric_name": "cpu_usage",
      "current_value": 65.5,
      "predicted_value": 78.2,
      "confidence": 0.85,
      "recommended_action": "plan_scale_up",
      "threshold_breach_date": "2024-02-05T14:20:00Z"
    }
  ]
}
```

### Scaling Recommendations

Based on predictions, the system provides actionable recommendations:

- **immediate_scale_up_required**: Critical action needed within days
- **plan_scale_up**: Scale up recommended within weeks
- **monitor_closely**: Watch for rapid growth
- **consider_scale_down**: Opportunity for cost optimization
- **maintain_current_capacity**: No action needed

## SLO Monitoring

### Error Budget Tracking

Monitor error budget consumption:

```bash
# Check SLO compliance
curl http://localhost:8090/api/slo-compliance
```

Response:
```json
{
  "api_availability": {
    "compliance": 99.95,
    "target": 99.9,
    "error_budget_remaining": 95.0,
    "burn_rate": 0.5,
    "status": "good"
  }
}
```

### Burn Rate Alerts

Automatic alerts when error budget consumption is too high:

- **Critical**: > 10% burn rate (budget consumed in 3 days)
- **High**: > 5% burn rate (budget consumed in 6 days)
- **Warning**: > 2% burn rate (budget consumed in 15 days)

## Troubleshooting

### Common Issues

#### 1. Services Not Starting

Check Docker logs:
```bash
./scripts/deployment/deploy_advanced_monitoring.sh logs [service-name]
```

Common causes:
- Insufficient memory (requires 8GB+)
- Port conflicts
- Configuration errors

#### 2. Elasticsearch Not Ready

Elasticsearch requires time to start. Check status:
```bash
curl http://localhost:9200/_cluster/health
```

If unhealthy:
```bash
docker-compose -f deployment/monitoring/docker-compose.advanced-monitoring.yml restart elasticsearch
```

#### 3. Dashboard Not Loading Data

Check API connectivity:
```bash
curl http://localhost:8090/api/system-health
```

Verify configuration:
```bash
docker-compose -f deployment/monitoring/docker-compose.advanced-monitoring.yml logs pynomaly-advanced-monitoring
```

#### 4. Anomaly Detection Not Working

Ensure sufficient training data:
- Minimum 100 data points for training
- At least 24 hours of historical data
- Check ML model logs for training errors

### Debug Mode

Enable debug logging:
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or modify configuration
# config/advanced_monitoring.yml
logging:
  level: "DEBUG"
```

### Performance Tuning

For large-scale deployments:

1. **Increase resource limits**:
```yaml
# docker-compose.advanced-monitoring.yml
services:
  elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms4g -Xmx4g"
```

2. **Scale horizontally**:
```bash
docker-compose -f deployment/monitoring/docker-compose.advanced-monitoring.yml up -d --scale anomaly-detector=3
```

3. **Optimize data retention**:
```yaml
# config/advanced_monitoring.yml
global:
  data_retention_days: 30  # Reduce from 90 days
```

## API Reference

### System Health

```http
GET /api/system-health
```

Response:
```json
{
  "overall_status": "healthy",
  "components": {
    "anomaly_detection": {"status": "active"},
    "slo_monitoring": {"status": "active"}
  }
}
```

### Metrics Processing

```http
POST /api/metrics
Content-Type: application/json

{
  "metric_name": "cpu_usage",
  "value": 75.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### SLI Measurements

```http
POST /api/sli-measurements
Content-Type: application/json

{
  "sli_name": "api_success_rate",
  "value": 0.999,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Capacity Reports

```http
GET /api/capacity-report?forecast_days=30
```

### Anomaly Detection

```http
GET /api/anomalies?metric_name=cpu_usage&time_range=24h
```

### Alert Correlation

```http
GET /api/alerts/active
GET /api/alerts/correlated?correlation_id=abc123
```

## Best Practices

### 1. SLO Definition

- **Start Simple**: Begin with availability and latency SLOs
- **Business Alignment**: Ensure SLOs reflect user experience
- **Realistic Targets**: Set achievable but meaningful targets
- **Regular Review**: Adjust SLOs based on business needs

### 2. Anomaly Detection

- **Quality Data**: Ensure clean, consistent metric data
- **Sufficient History**: Use at least 30 days for training
- **Regular Retraining**: Update models with new patterns
- **Human Feedback**: Incorporate false positive feedback

### 3. Capacity Planning

- **Multiple Metrics**: Monitor CPU, memory, disk, and network
- **Seasonal Patterns**: Account for business cycles
- **Safety Margins**: Plan for unexpected growth
- **Cost Optimization**: Balance performance and cost

### 4. Dashboard Design

- **Audience-Specific**: Tailor dashboards to user needs
- **Information Hierarchy**: Most important metrics first
- **Actionable Insights**: Focus on metrics that drive decisions
- **Regular Updates**: Keep dashboards current and relevant

### 5. Alert Management

- **Severity Levels**: Use consistent severity classification
- **Actionable Alerts**: Every alert should have a clear response
- **Reduce Noise**: Use correlation and suppression effectively
- **Escalation Paths**: Define clear escalation procedures

### 6. Security

- **Authentication**: Require authentication for all access
- **Authorization**: Use role-based access control
- **Audit Logging**: Log all configuration changes
- **Data Protection**: Encrypt sensitive monitoring data

### 7. Performance

- **Resource Monitoring**: Monitor the monitoring system itself
- **Data Retention**: Balance storage cost and analytical needs
- **Query Optimization**: Optimize dashboard queries
- **Caching**: Cache frequently accessed data

This advanced monitoring system provides comprehensive observability for Pynomaly with enterprise-grade features including ML-based anomaly detection, predictive capacity planning, intelligent alerting, and business intelligence dashboards. The system is designed to scale with your needs while providing actionable insights for both technical and business stakeholders.