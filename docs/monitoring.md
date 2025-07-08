# Monitoring Documentation

## Architecture Overview

The monitoring architecture in Pynomaly is a comprehensive observability stack that provides metrics collection, visualization, and alerting capabilities. The system follows a modern microservices monitoring approach with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │<--→│     Grafana     │<--→│   Pynomaly API  │
│   (Metrics)     │    │ (Visualization) │    │   (Application) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ^                       ^                       ^
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Alertmanager   │    │  OpenTelemetry  │    │    Workers      │
│   (Alerting)    │    │   Collector     │    │  (Background)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Responsibilities

- **Prometheus**: Time-series database that scrapes metrics from all services at 15-second intervals
- **Grafana**: Web-based visualization platform with pre-configured dashboards for system monitoring
- **Pynomaly API**: Main application exposing business metrics at `/api/metrics`, `/api/metrics/business`, `/api/metrics/models`, and `/api/metrics/drift`
- **Alertmanager**: Handles alerts sent by Prometheus server and routes them to notification channels
- **OpenTelemetry Collector**: Collects, processes, and exports telemetry data (metrics, logs, traces)
- **Workers**: Background processing services (training, drift monitoring) that expose their own metrics

### Data Flow

1. **Metrics Collection**: Prometheus scrapes metrics from all services every 15 seconds
2. **Data Storage**: Metrics are stored in Prometheus TSDB with 30-day retention
3. **Visualization**: Grafana queries Prometheus for dashboard rendering
4. **Alerting**: Prometheus evaluates alert rules and sends notifications via Alertmanager
5. **Telemetry**: OpenTelemetry Collector aggregates traces and forwards to external systems

## Setup Instructions

### Docker Setup

1. **Requirements**:
   - Docker
   - Docker Compose

2. **Environment Variables**:
   - Ensure that the following environment variables are set:
     ```
     PYNOMALY_ENVIRONMENT=production
     POSTGRES_PASSWORD=pynomaly_secret
     FLOWER_USER=admin
     FLOWER_PASSWORD=flower_secret
     ```

3. **Networking**:
   - The default networking setup exposes:
     - Port `8000` for the Pynomaly API
     - Port `9090` for Prometheus metrics
     - Port `5555` for Flower monitoring

4. **Running the System**:
   - Execute the following command to start the Pynomaly stack:
     ```sh
     docker-compose -f docker-compose.production.yml up -d
     ```

### Kubernetes Setup

1. **Requirements**:
   - Kubernetes Cluster
   - kubectl tool configured

2. **Deployment**:
   - Deploy the services using the following commands:
     ```sh
     kubectl apply -f kubernetes/monitoring-deployment.yaml
     ```

## Extending Metrics and Dashboards

- **Adding Custom Metrics**:
  - Implement new metrics within Pynomaly using the Prometheus metrics library.
- **Creating New Dashboards**:
  - Use Grafana to design custom dashboards and panels pointing to new metrics.

## Troubleshooting Guide

- **Prometheus Configuration Reload Failed**:
  - Make sure changes to the Prometheus configuration files are valid.
  - Verify syntax with a test reload command.

- **Grafana Dashboard Issues**:
  - Ensure that Grafana is connected to Prometheus correctly.
  - Check for typo errors in queries and data source settings.

## FAQ

- **How do I update configurations?**
  - Modify the appropriate `.yaml` or `.ini` files and restart the respective services using Docker or Kubernetes commands.

- **Where do I find the logs?**
  - Logs for each component can be found under their respective `/logs` directory bind-mounted to the host system.

This document serves as a comprehensive guide to the monitoring setup in Pynomaly. For further assistance, consult the official documentation or reach out to the support channels. 
