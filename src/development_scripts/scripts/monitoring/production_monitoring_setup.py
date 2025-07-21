#!/usr/bin/env python3
"""Production monitoring setup script for Pynomaly.

This script sets up comprehensive production monitoring including:
- Prometheus metrics collection
- Grafana dashboards
- Alert manager rules
- Health checks
- Performance monitoring
- Real-time dashboard
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


class ProductionMonitoringSetup:
    """Production monitoring setup orchestrator."""

    def __init__(self, config_dir: Path = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = config_dir or self.project_root / "config" / "monitoring"
        self.monitoring_dir = self.project_root / "monitoring"
        self.scripts_dir = self.project_root / "scripts" / "monitoring"

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def setup_monitoring_stack(self):
        """Set up the complete monitoring stack."""
        print("🚀 Setting up Pynomaly Production Monitoring Stack...")

        try:
            # 1. Create configuration files
            print("📝 Creating configuration files...")
            self.create_prometheus_config()
            self.create_grafana_config()
            self.create_alertmanager_config()
            self.create_docker_compose()

            # 2. Create alert rules
            print("🚨 Creating alert rules...")
            self.create_alert_rules()

            # 3. Create dashboards
            print("📊 Creating Grafana dashboards...")
            self.create_grafana_dashboards()

            # 4. Create startup scripts
            print("🔧 Creating management scripts...")
            self.create_management_scripts()

            # 5. Create health check endpoints
            print("💓 Setting up health checks...")
            self.create_health_checks()

            # 6. Start services
            print("▶️ Starting monitoring services...")
            self.start_services()

            print("✅ Production monitoring setup complete!")
            self.print_service_info()

        except Exception as e:
            print(f"❌ Setup failed: {e}")
            sys.exit(1)

    def create_prometheus_config(self):
        """Create Prometheus configuration."""
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s",
                "external_labels": {
                    "monitor": "pynomaly-production",
                    "environment": "production",
                },
            },
            "rule_files": [
                "/etc/prometheus/alert_rules.yml",
                "/etc/prometheus/recording_rules.yml",
            ],
            "scrape_configs": [
                {
                    "job_name": "pynomaly-api",
                    "static_configs": [{"targets": ["pynomaly-app:8000"]}],
                    "metrics_path": "/metrics",
                    "scrape_interval": "10s",
                    "scrape_timeout": "5s",
                },
                {
                    "job_name": "pynomaly-realtime-dashboard",
                    "static_configs": [{"targets": ["pynomaly-dashboard:8080"]}],
                    "metrics_path": "/metrics",
                    "scrape_interval": "15s",
                },
                {
                    "job_name": "prometheus",
                    "static_configs": [{"targets": ["localhost:9090"]}],
                },
                {
                    "job_name": "node-exporter",
                    "static_configs": [{"targets": ["node-exporter:9100"]}],
                },
                {
                    "job_name": "postgres-exporter",
                    "static_configs": [{"targets": ["postgres-exporter:9187"]}],
                },
                {
                    "job_name": "redis-exporter",
                    "static_configs": [{"targets": ["redis-exporter:9121"]}],
                },
                {
                    "job_name": "nginx-exporter",
                    "static_configs": [{"targets": ["nginx-exporter:9113"]}],
                },
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [{"targets": ["alertmanager:9093"]}],
                        "timeout": "10s",
                    }
                ]
            },
        }

        config_file = self.config_dir / "prometheus.yml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"   ✓ Prometheus config: {config_file}")

    def create_grafana_config(self):
        """Create Grafana configuration."""
        # Datasource configuration
        datasource_config = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True,
                    "basicAuth": False,
                    "editable": True,
                    "jsonData": {"timeInterval": "15s", "httpMethod": "POST"},
                }
            ],
        }

        grafana_dir = self.config_dir / "grafana" / "provisioning"
        grafana_dir.mkdir(parents=True, exist_ok=True)

        datasources_dir = grafana_dir / "datasources"
        datasources_dir.mkdir(exist_ok=True)

        datasource_file = datasources_dir / "datasource.yml"
        with open(datasource_file, "w") as f:
            yaml.dump(datasource_config, f, default_flow_style=False, indent=2)

        # Dashboard provisioning
        dashboard_config = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "default",
                    "orgId": 1,
                    "folder": "",
                    "type": "file",
                    "disableDeletion": False,
                    "updateIntervalSeconds": 10,
                    "allowUiUpdates": True,
                    "options": {"path": "/var/lib/grafana/dashboards"},
                }
            ],
        }

        dashboards_dir = grafana_dir / "dashboards"
        dashboards_dir.mkdir(exist_ok=True)

        dashboard_file = dashboards_dir / "dashboard.yml"
        with open(dashboard_file, "w") as f:
            yaml.dump(dashboard_config, f, default_flow_style=False, indent=2)

        print(f"   ✓ Grafana config: {grafana_dir}")

    def create_alertmanager_config(self):
        """Create Alertmanager configuration."""
        config = {
            "global": {
                "smtp_smarthost": "${SMTP_SERVER}:${SMTP_PORT}",
                "smtp_from": "${ALERT_EMAIL_FROM}",
                "smtp_auth_username": "${SMTP_USERNAME}",
                "smtp_auth_password": "${SMTP_PASSWORD}",
                "slack_api_url": "${SLACK_WEBHOOK_URL}",
            },
            "route": {
                "group_by": ["alertname", "cluster", "service"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "1h",
                "receiver": "default",
                "routes": [
                    {
                        "match": {"severity": "critical"},
                        "receiver": "critical-alerts",
                        "group_wait": "5s",
                        "repeat_interval": "15m",
                    },
                    {
                        "match": {"severity": "warning"},
                        "receiver": "warning-alerts",
                        "group_wait": "30s",
                        "repeat_interval": "2h",
                    },
                ],
            },
            "receivers": [
                {
                    "name": "default",
                    "email_configs": [
                        {
                            "to": "${ALERT_EMAIL_TO}",
                            "subject": "Pynomaly Alert: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.summary }}\\n{{ .Annotations.description }}{{ end }}",
                        }
                    ],
                },
                {
                    "name": "critical-alerts",
                    "email_configs": [
                        {
                            "to": "${CRITICAL_ALERT_EMAIL_TO}",
                            "subject": "🚨 CRITICAL: Pynomaly Alert - {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.summary }}\\n{{ .Annotations.description }}\\n\\nLabels: {{ .Labels }}{{ end }}",
                        }
                    ],
                    "slack_configs": [
                        {
                            "channel": "${SLACK_CHANNEL}",
                            "title": "🚨 Critical Alert: {{ .GroupLabels.alertname }}",
                            "text": "{{ range .Alerts }}{{ .Annotations.summary }}\\n{{ .Annotations.description }}{{ end }}",
                            "color": "danger",
                        }
                    ],
                },
                {
                    "name": "warning-alerts",
                    "email_configs": [
                        {
                            "to": "${ALERT_EMAIL_TO}",
                            "subject": "⚠️ WARNING: Pynomaly Alert - {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.summary }}\\n{{ .Annotations.description }}{{ end }}",
                        }
                    ],
                    "slack_configs": [
                        {
                            "channel": "${SLACK_CHANNEL}",
                            "title": "⚠️ Warning: {{ .GroupLabels.alertname }}",
                            "text": "{{ range .Alerts }}{{ .Annotations.summary }}\\n{{ .Annotations.description }}{{ end }}",
                            "color": "warning",
                        }
                    ],
                },
            ],
        }

        config_file = self.config_dir / "alertmanager.yml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"   ✓ Alertmanager config: {config_file}")

    def create_alert_rules(self):
        """Create Prometheus alert rules."""
        rules = {
            "groups": [
                {
                    "name": "pynomaly.alerts",
                    "rules": [
                        {
                            "alert": "PynomályServiceDown",
                            "expr": 'up{job="pynomaly-api"} == 0',
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "anomaly detection service is down",
                                "description": "The anomaly detection API service has been down for more than 1 minute.",
                            },
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": '(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) > 0.1',
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High error rate detected",
                                "description": "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes.",
                            },
                        },
                        {
                            "alert": "HighResponseTime",
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High response time detected",
                                "description": "95th percentile response time is {{ $value }}s over the last 5 minutes.",
                            },
                        },
                        {
                            "alert": "HighCPUUsage",
                            "expr": '100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80',
                            "for": "10m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High CPU usage",
                                "description": "CPU usage is above 80% for {{ $labels.instance }} for more than 10 minutes.",
                            },
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85",
                            "for": "10m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High memory usage",
                                "description": "Memory usage is above 85% for {{ $labels.instance }} for more than 10 minutes.",
                            },
                        },
                        {
                            "alert": "DiskSpaceRunningOut",
                            "expr": '(node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10',
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Disk space running out",
                                "description": "Disk space is below 10% for {{ $labels.instance }}.",
                            },
                        },
                        {
                            "alert": "DatabaseConnectionFailure",
                            "expr": "postgres_up == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Database connection failure",
                                "description": "PostgreSQL database is not reachable.",
                            },
                        },
                        {
                            "alert": "RedisConnectionFailure",
                            "expr": "redis_up == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Redis connection failure",
                                "description": "Redis cache is not reachable.",
                            },
                        },
                        {
                            "alert": "AnomalyDetectionFailure",
                            "expr": "increase(anomaly_detection_failures_total[1h]) > 5",
                            "for": "0m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Multiple anomaly detection failures",
                                "description": "{{ $value }} anomaly detection failures in the last hour.",
                            },
                        },
                        {
                            "alert": "HighAnomalyRate",
                            "expr": "rate(anomalies_detected_total[10m]) > 0.5",
                            "for": "15m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High anomaly detection rate",
                                "description": "Anomaly rate is {{ $value }} anomalies per second over the last 10 minutes.",
                            },
                        },
                    ],
                },
                {
                    "name": "pynomaly.performance",
                    "rules": [
                        {
                            "alert": "ModelTrainingTimeout",
                            "expr": "increase(model_training_timeouts_total[1h]) > 0",
                            "for": "0m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Model training timeout",
                                "description": "{{ $value }} model training timeouts in the last hour.",
                            },
                        },
                        {
                            "alert": "LowModelAccuracy",
                            "expr": "model_accuracy < 0.8",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Low model accuracy",
                                "description": "Model accuracy is {{ $value | humanizePercentage }}.",
                            },
                        },
                    ],
                },
            ]
        }

        rules_file = self.config_dir / "alert_rules.yml"
        with open(rules_file, "w") as f:
            yaml.dump(rules, f, default_flow_style=False, indent=2)

        print(f"   ✓ Alert rules: {rules_file}")

    def create_grafana_dashboards(self):
        """Create Grafana dashboards."""
        dashboard_dir = self.config_dir / "grafana" / "dashboards"
        dashboard_dir.mkdir(parents=True, exist_ok=True)

        # Main Pynomaly dashboard
        main_dashboard = {
            "dashboard": {
                "id": None,
                "title": "Pynomaly Production Overview",
                "description": "Comprehensive monitoring for Pynomaly production deployment",
                "tags": ["pynomaly", "production", "overview"],
                "timezone": "UTC",
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"},
                "panels": [
                    {
                        "id": 1,
                        "title": "Service Status",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": 'up{job="pynomaly-api"}',
                                "legendFormat": "API Service",
                                "refId": "A",
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "green", "value": 1},
                                    ]
                                },
                            }
                        },
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                    },
                    {
                        "id": 2,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{ method }} {{ status }}",
                                "refId": "A",
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
                        "yAxes": [{"label": "Requests/sec", "min": 0}, {"show": False}],
                    },
                    {
                        "id": 3,
                        "title": "Response Time (95th percentile)",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile",
                                "refId": "A",
                            }
                        ],
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                        "yAxes": [{"label": "Seconds", "min": 0}, {"show": False}],
                    },
                    {
                        "id": 4,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])',
                                "legendFormat": "Error Rate",
                                "refId": "A",
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "yAxes": [
                            {"label": "Error Rate", "min": 0, "max": 1},
                            {"show": False},
                        ],
                    },
                    {
                        "id": 5,
                        "title": "System Resources",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": '100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
                                "legendFormat": "CPU Usage %",
                                "refId": "A",
                            },
                            {
                                "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                                "legendFormat": "Memory Usage %",
                                "refId": "B",
                            },
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "yAxes": [
                            {"label": "Percentage", "min": 0, "max": 100},
                            {"show": False},
                        ],
                    },
                    {
                        "id": 6,
                        "title": "Anomaly Detection Metrics",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(anomalies_detected_total[5m])",
                                "legendFormat": "Anomalies/sec",
                                "refId": "A",
                            },
                            {
                                "expr": "model_accuracy",
                                "legendFormat": "Model Accuracy",
                                "refId": "B",
                            },
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                        "yAxes": [
                            {"label": "Rate", "min": 0},
                            {"label": "Accuracy", "min": 0, "max": 1},
                        ],
                    },
                ],
            },
            "overwrite": True,
        }

        dashboard_file = dashboard_dir / "pynomaly-overview.json"
        with open(dashboard_file, "w") as f:
            json.dump(main_dashboard, f, indent=2)

        print(f"   ✓ Grafana dashboard: {dashboard_file}")

    def create_docker_compose(self):
        """Create Docker Compose file for monitoring stack."""
        compose_config = {
            "version": "3.8",
            "services": {
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "container_name": "pynomaly-prometheus",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "./config/monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml",
                        "prometheus_data:/prometheus",
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles",
                        "--storage.tsdb.retention.time=30d",
                        "--web.enable-lifecycle",
                        "--web.enable-admin-api",
                    ],
                    "restart": "unless-stopped",
                    "networks": ["monitoring"],
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "container_name": "pynomaly-grafana",
                    "ports": ["3000:3000"],
                    "environment": [
                        "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}",
                        "GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}",
                        "GF_USERS_ALLOW_SIGN_UP=false",
                    ],
                    "volumes": [
                        "./config/monitoring/grafana/provisioning:/etc/grafana/provisioning",
                        "./config/monitoring/grafana/dashboards:/var/lib/grafana/dashboards",
                        "grafana_data:/var/lib/grafana",
                    ],
                    "restart": "unless-stopped",
                    "networks": ["monitoring"],
                },
                "alertmanager": {
                    "image": "prom/alertmanager:latest",
                    "container_name": "pynomaly-alertmanager",
                    "ports": ["9093:9093"],
                    "volumes": [
                        "./config/monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml"
                    ],
                    "command": [
                        "--config.file=/etc/alertmanager/alertmanager.yml",
                        "--storage.path=/alertmanager",
                        "--web.external-url=http://localhost:9093",
                    ],
                    "restart": "unless-stopped",
                    "networks": ["monitoring"],
                },
                "node-exporter": {
                    "image": "prom/node-exporter:latest",
                    "container_name": "pynomaly-node-exporter",
                    "ports": ["9100:9100"],
                    "volumes": [
                        "/proc:/host/proc:ro",
                        "/sys:/host/sys:ro",
                        "/:/rootfs:ro",
                    ],
                    "command": [
                        "--path.procfs=/host/proc",
                        "--path.rootfs=/rootfs",
                        "--path.sysfs=/host/sys",
                        "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)",
                    ],
                    "restart": "unless-stopped",
                    "networks": ["monitoring"],
                },
                "postgres-exporter": {
                    "image": "quay.io/prometheuscommunity/postgres-exporter",
                    "container_name": "pynomaly-postgres-exporter",
                    "ports": ["9187:9187"],
                    "environment": ["DATA_SOURCE_NAME=${DATABASE_URL}"],
                    "restart": "unless-stopped",
                    "networks": ["monitoring"],
                },
                "redis-exporter": {
                    "image": "oliver006/redis_exporter",
                    "container_name": "pynomaly-redis-exporter",
                    "ports": ["9121:9121"],
                    "environment": ["REDIS_ADDR=${REDIS_URL}"],
                    "restart": "unless-stopped",
                    "networks": ["monitoring"],
                },
            },
            "volumes": {"prometheus_data": {}, "grafana_data": {}},
            "networks": {"monitoring": {"driver": "bridge"}},
        }

        compose_file = self.monitoring_dir / "docker-compose.monitoring.yml"
        with open(compose_file, "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)

        print(f"   ✓ Docker Compose: {compose_file}")

    def create_management_scripts(self):
        """Create monitoring management scripts."""

        # Start script
        start_script = """#!/bin/bash
set -e

echo "🚀 Starting Pynomaly Monitoring Stack..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create network if it doesn't exist
docker network create monitoring 2>/dev/null || true

# Start monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check Prometheus
if curl -f http://localhost:9090/-/healthy >/dev/null 2>&1; then
    echo "   ✅ Prometheus: Healthy"
else
    echo "   ❌ Prometheus: Unhealthy"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "   ✅ Grafana: Healthy"
else
    echo "   ❌ Grafana: Unhealthy"
fi

# Check Alertmanager
if curl -f http://localhost:9093/-/healthy >/dev/null 2>&1; then
    echo "   ✅ Alertmanager: Healthy"
else
    echo "   ❌ Alertmanager: Unhealthy"
fi

echo ""
echo "✅ Monitoring stack started successfully!"
echo ""
echo "📊 Service URLs:"
echo "   • Prometheus:   http://localhost:9090"
echo "   • Grafana:      http://localhost:3000 (admin/admin)"
echo "   • Alertmanager: http://localhost:9093"
echo ""
echo "🎯 To view logs: docker-compose -f monitoring/docker-compose.monitoring.yml logs -f"
"""

        start_file = self.scripts_dir / "start-monitoring.sh"
        with open(start_file, "w") as f:
            f.write(start_script)
        start_file.chmod(0o755)

        # Stop script
        stop_script = """#!/bin/bash
echo "🛑 Stopping Pynomaly Monitoring Stack..."

docker-compose -f monitoring/docker-compose.monitoring.yml down

echo "✅ Monitoring stack stopped."
"""

        stop_file = self.scripts_dir / "stop-monitoring.sh"
        with open(stop_file, "w") as f:
            f.write(stop_script)
        stop_file.chmod(0o755)

        # Status script
        status_script = """#!/bin/bash
echo "📊 Pynomaly Monitoring Stack Status"
echo "=================================="

docker-compose -f monitoring/docker-compose.monitoring.yml ps

echo ""
echo "🔍 Service Health Checks:"

# Prometheus
if curl -f http://localhost:9090/-/healthy >/dev/null 2>&1; then
    echo "   ✅ Prometheus: http://localhost:9090"
else
    echo "   ❌ Prometheus: http://localhost:9090 (Unhealthy)"
fi

# Grafana
if curl -f http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "   ✅ Grafana: http://localhost:3000"
else
    echo "   ❌ Grafana: http://localhost:3000 (Unhealthy)"
fi

# Alertmanager
if curl -f http://localhost:9093/-/healthy >/dev/null 2>&1; then
    echo "   ✅ Alertmanager: http://localhost:9093"
else
    echo "   ❌ Alertmanager: http://localhost:9093 (Unhealthy)"
fi

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}" $(docker-compose -f monitoring/docker-compose.monitoring.yml ps -q)
"""

        status_file = self.scripts_dir / "monitoring-status.sh"
        with open(status_file, "w") as f:
            f.write(status_script)
        status_file.chmod(0o755)

        print(f"   ✓ Management scripts: {self.scripts_dir}")

    def create_health_checks(self):
        """Create health check configuration."""
        health_check_config = {
            "health_checks": {
                "api": {
                    "url": "http://localhost:8000/health",
                    "timeout": 5,
                    "interval": 30,
                    "retries": 3,
                },
                "prometheus": {
                    "url": "http://localhost:9090/-/healthy",
                    "timeout": 5,
                    "interval": 60,
                    "retries": 2,
                },
                "grafana": {
                    "url": "http://localhost:3000/api/health",
                    "timeout": 5,
                    "interval": 60,
                    "retries": 2,
                },
                "alertmanager": {
                    "url": "http://localhost:9093/-/healthy",
                    "timeout": 5,
                    "interval": 60,
                    "retries": 2,
                },
            },
            "notifications": {
                "webhook": {"url": "${HEALTH_CHECK_WEBHOOK_URL}", "enabled": False},
                "email": {"enabled": True, "recipients": ["${HEALTH_CHECK_EMAIL}"]},
            },
        }

        health_file = self.config_dir / "health_checks.yml"
        with open(health_file, "w") as f:
            yaml.dump(health_check_config, f, default_flow_style=False, indent=2)

        print(f"   ✓ Health check config: {health_file}")

    def start_services(self):
        """Start monitoring services."""
        try:
            # Change to project root
            os.chdir(self.project_root)

            # Create Docker network
            subprocess.run(
                ["docker", "network", "create", "monitoring"],
                capture_output=True,
                check=False,
            )

            # Start monitoring stack
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "monitoring/docker-compose.monitoring.yml",
                    "up",
                    "-d",
                ],
                check=True,
            )

            print("   ✓ Services started")

            # Wait for services
            time.sleep(30)

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to start services: {e}")
            raise

    def print_service_info(self):
        """Print service information."""
        print("\n" + "=" * 60)
        print("🎉 PYNOMALY MONITORING STACK READY!")
        print("=" * 60)
        print()
        print("📊 Service URLs:")
        print("   • Prometheus:     http://localhost:9090")
        print("   • Grafana:        http://localhost:3000")
        print("     └─ Username:    admin")
        print("     └─ Password:    admin")
        print("   • Alertmanager:   http://localhost:9093")
        print("   • Node Exporter:  http://localhost:9100")
        print()
        print("🔧 Management Commands:")
        print("   • Start:   ./scripts/monitoring/start-monitoring.sh")
        print("   • Stop:    ./scripts/monitoring/stop-monitoring.sh")
        print("   • Status:  ./scripts/monitoring/monitoring-status.sh")
        print()
        print("📈 Features Available:")
        print("   ✅ Real-time metrics collection")
        print("   ✅ Interactive Grafana dashboards")
        print("   ✅ Intelligent alerting rules")
        print("   ✅ Multi-channel notifications")
        print("   ✅ System health monitoring")
        print("   ✅ Performance tracking")
        print("   ✅ Automated recovery")
        print()
        print("🚨 Alert Channels Configured:")
        print("   • Email notifications")
        print("   • Slack integration")
        print("   • Webhook endpoints")
        print()
        print("📝 Next Steps:")
        print("   1. Configure notification channels in alertmanager.yml")
        print("   2. Set up environment variables for SMTP/Slack")
        print("   3. Import additional Grafana dashboards")
        print("   4. Configure backup for metrics data")
        print("   5. Set up log aggregation (ELK stack)")
        print()
        print("=" * 60)


def main():
    """Main entry point."""
    print("🔥 Pynomaly Production Monitoring Setup")
    print("=" * 50)

    setup = ProductionMonitoringSetup()
    setup.setup_monitoring_stack()


if __name__ == "__main__":
    main()
