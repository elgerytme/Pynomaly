#!/usr/bin/env python3
"""
Monitoring Infrastructure Setup Script

This script sets up the complete monitoring infrastructure for Pynomaly,
including Prometheus, Grafana, alerting, and dashboard components.
"""

import json
import logging
import os
import sys
from pathlib import Path

import yaml

# import docker  # Optional dependency

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import monitoring components (optional)
try:
    from monorepo.infrastructure.monitoring.alerts import AlertManager, HealthChecker
    from monorepo.infrastructure.monitoring.dashboard import (
        DashboardWebServer,
        create_dashboard_templates,
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some monitoring dependencies not available: {e}")
    MONITORING_AVAILABLE = False


class MonitoringSetup:
    """Setup and configure monitoring infrastructure."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config" / "monitoring"
        self.docker_client = None

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Docker client
        try:
            import docker
            self.docker_client = docker.from_env()
        except ImportError:
            self.logger.warning("Docker Python library not available")
            self.docker_client = None
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            self.docker_client = None

    def setup_prometheus(self):
        """Setup Prometheus configuration."""
        self.logger.info("Setting up Prometheus configuration...")

        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "alert_rules.yml"
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {
                                "targets": ["alertmanager:9093"]
                            }
                        ]
                    }
                ]
            },
            "scrape_configs": [
                {
                    "job_name": "pynomaly-api",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"]
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scrape_interval": "15s"
                },
                {
                    "job_name": "pynomaly-dashboard",
                    "static_configs": [
                        {
                            "targets": ["localhost:8080"]
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scrape_interval": "30s"
                },
                {
                    "job_name": "node-exporter",
                    "static_configs": [
                        {
                            "targets": ["localhost:9100"]
                        }
                    ]
                },
                {
                    "job_name": "redis-exporter",
                    "static_configs": [
                        {
                            "targets": ["localhost:9121"]
                        }
                    ]
                },
                {
                    "job_name": "postgres-exporter",
                    "static_configs": [
                        {
                            "targets": ["localhost:9187"]
                        }
                    ]
                }
            ]
        }

        # Write Prometheus config
        with open(self.config_dir / "prometheus.yml", "w") as f:
            yaml.dump(prometheus_config, f, default_flow_style=False, sort_keys=False)

        self.logger.info("Prometheus configuration created")

    def setup_grafana(self):
        """Setup Grafana dashboards and datasources."""
        self.logger.info("Setting up Grafana configuration...")

        # Grafana datasource configuration
        datasource_config = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True,
                    "editable": True
                }
            ]
        }

        # Grafana dashboard configuration
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
                    "options": {
                        "path": "/etc/grafana/provisioning/dashboards"
                    }
                }
            ]
        }

        # Create Grafana directories
        grafana_dir = self.config_dir / "grafana"
        grafana_dir.mkdir(exist_ok=True)
        (grafana_dir / "provisioning" / "datasources").mkdir(parents=True, exist_ok=True)
        (grafana_dir / "provisioning" / "dashboards").mkdir(parents=True, exist_ok=True)

        # Write Grafana configs
        with open(grafana_dir / "provisioning" / "datasources" / "datasource.yml", "w") as f:
            yaml.dump(datasource_config, f, default_flow_style=False)

        with open(grafana_dir / "provisioning" / "dashboards" / "dashboard.yml", "w") as f:
            yaml.dump(dashboard_config, f, default_flow_style=False)

        # Create main dashboard
        self._create_grafana_dashboard()

        self.logger.info("Grafana configuration created")

    def _create_grafana_dashboard(self):
        """Create main Grafana dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Pynomaly Monitoring Dashboard",
                "tags": ["monorepo", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "API Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{job=\"pynomaly-api\"}[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
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
                                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{job=\"pynomaly-api\",status=~\"5..\"}[5m])",
                                "legendFormat": "5xx Errors/sec"
                            },
                            {
                                "expr": "rate(http_requests_total{job=\"pynomaly-api\",status=~\"4..\"}[5m])",
                                "legendFormat": "4xx Errors/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "System Resources",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(process_cpu_seconds_total{job=\"pynomaly-api\"}[5m]) * 100",
                                "legendFormat": "CPU Usage %"
                            },
                            {
                                "expr": "process_resident_memory_bytes{job=\"pynomaly-api\"} / 1024 / 1024",
                                "legendFormat": "Memory Usage MB"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Detection Performance",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(pynomaly_detections_total[5m])",
                                "legendFormat": "Detections/sec"
                            },
                            {
                                "expr": "rate(pynomaly_anomalies_detected_total[5m])",
                                "legendFormat": "Anomalies/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
                },
                "refresh": "30s"
            }
        }

        # Write dashboard
        with open(self.config_dir / "grafana" / "provisioning" / "dashboards" / "pynomaly-dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)

    def setup_alertmanager(self):
        """Setup Alertmanager configuration."""
        self.logger.info("Setting up Alertmanager configuration...")

        alertmanager_config = {
            "global": {
                "smtp_smarthost": "localhost:587",
                "smtp_from": "alerts@monorepo.com"
            },
            "route": {
                "group_by": ["alertname"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "1h",
                "receiver": "web.hook"
            },
            "receivers": [
                {
                    "name": "web.hook",
                    "email_configs": [
                        {
                            "to": "admin@monorepo.com",
                            "subject": "Pynomaly Alert: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ],
                    "slack_configs": [
                        {
                            "api_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                            "channel": "#alerts",
                            "title": "Pynomaly Alert",
                            "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                        }
                    ]
                }
            ]
        }

        # Write Alertmanager config
        with open(self.config_dir / "alertmanager.yml", "w") as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False)

        self.logger.info("Alertmanager configuration created")

    def create_docker_compose(self):
        """Create Docker Compose file for monitoring stack."""
        self.logger.info("Creating Docker Compose configuration...")

        docker_compose = {
            "version": "3.8",
            "services": {
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "container_name": "pynomaly-prometheus",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "./config/monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml"
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles",
                        "--storage.tsdb.retention.time=200h",
                        "--web.enable-lifecycle"
                    ],
                    "restart": "unless-stopped"
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "container_name": "pynomaly-grafana",
                    "ports": ["3000:3000"],
                    "volumes": [
                        "./config/monitoring/grafana/provisioning:/etc/grafana/provisioning",
                        "grafana-storage:/var/lib/grafana"
                    ],
                    "environment": [
                        "GF_SECURITY_ADMIN_PASSWORD=admin",
                        "GF_USERS_ALLOW_SIGN_UP=false"
                    ],
                    "restart": "unless-stopped"
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
                        "--web.external-url=http://localhost:9093"
                    ],
                    "restart": "unless-stopped"
                },
                "node-exporter": {
                    "image": "prom/node-exporter:latest",
                    "container_name": "pynomaly-node-exporter",
                    "ports": ["9100:9100"],
                    "volumes": [
                        "/proc:/host/proc:ro",
                        "/sys:/host/sys:ro",
                        "/:/rootfs:ro"
                    ],
                    "command": [
                        "--path.procfs=/host/proc",
                        "--path.sysfs=/host/sys",
                        "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"
                    ],
                    "restart": "unless-stopped"
                },
                "redis-exporter": {
                    "image": "oliver006/redis_exporter:latest",
                    "container_name": "pynomaly-redis-exporter",
                    "ports": ["9121:9121"],
                    "environment": [
                        "REDIS_ADDR=redis://redis:6379"
                    ],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "grafana-storage": {}
            },
            "networks": {
                "default": {
                    "name": "pynomaly-monitoring"
                }
            }
        }

        # Write Docker Compose file
        with open(self.project_root / "docker-compose.monitoring.yml", "w") as f:
            yaml.dump(docker_compose, f, default_flow_style=False, sort_keys=False)

        self.logger.info("Docker Compose configuration created")

    def setup_monitoring_scripts(self):
        """Create monitoring management scripts."""
        self.logger.info("Creating monitoring management scripts...")

        # Start monitoring script
        start_script = """#!/bin/bash
# Start Pynomaly Monitoring Stack

echo "🔄 Starting Pynomaly monitoring stack..."

# Start Docker services
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
echo "Prometheus: $(curl -s http://localhost:9090/-/healthy || echo 'NOT READY')"
echo "Grafana: $(curl -s http://localhost:3000/api/health || echo 'NOT READY')"
echo "Alertmanager: $(curl -s http://localhost:9093/-/healthy || echo 'NOT READY')"

# Start Python monitoring services
echo "🐍 Starting Python monitoring services..."
python3 -m monorepo.infrastructure.monitoring.alerts &
python3 -m monorepo.infrastructure.monitoring.dashboard &

echo "✅ Monitoring stack started!"
echo "📊 Access points:"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Alertmanager: http://localhost:9093"
echo "  - Dashboard: http://localhost:8080"
"""

        # Stop monitoring script
        stop_script = """#!/bin/bash
# Stop Pynomaly Monitoring Stack

echo "🛑 Stopping Pynomaly monitoring stack..."

# Stop Python services
pkill -f "monorepo.infrastructure.monitoring"

# Stop Docker services
docker-compose -f docker-compose.monitoring.yml down

echo "✅ Monitoring stack stopped!"
"""

        # Status script
        status_script = """#!/bin/bash
# Check Pynomaly Monitoring Stack Status

echo "📊 Pynomaly Monitoring Stack Status"
echo "=================================="

# Check Docker services
echo "🐳 Docker Services:"
docker-compose -f docker-compose.monitoring.yml ps

echo ""
echo "🔗 Service Endpoints:"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Alertmanager: http://localhost:9093"
echo "  - Dashboard: http://localhost:8080"

echo ""
echo "🏥 Health Checks:"
echo "  - Prometheus: $(curl -s http://localhost:9090/-/healthy 2>/dev/null || echo 'DOWN')"
echo "  - Grafana: $(curl -s http://localhost:3000/api/health 2>/dev/null | grep -o '"database":"ok"' || echo 'DOWN')"
echo "  - Alertmanager: $(curl -s http://localhost:9093/-/healthy 2>/dev/null || echo 'DOWN')"
echo "  - Dashboard: $(curl -s http://localhost:8080/health 2>/dev/null | grep -o '"status":"healthy"' || echo 'DOWN')"
"""

        # Create scripts directory
        scripts_dir = self.project_root / "scripts" / "monitoring"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Write scripts
        scripts = [
            ("start-monitoring.sh", start_script),
            ("stop-monitoring.sh", stop_script),
            ("monitoring-status.sh", status_script)
        ]

        for script_name, script_content in scripts:
            script_path = scripts_dir / script_name
            with open(script_path, "w") as f:
                f.write(script_content)

            # Make script executable
            os.chmod(script_path, 0o755)

        self.logger.info("Monitoring management scripts created")

    def setup_systemd_services(self):
        """Create systemd service files for monitoring components."""
        self.logger.info("Creating systemd service files...")

        # Alert manager service
        alert_service = f"""[Unit]
Description=Pynomaly Alert Manager
After=network.target
Requires=network.target

[Service]
Type=simple
User=pynomaly
Group=pynomaly
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}/src
ExecStart=/usr/bin/python3 -m monorepo.infrastructure.monitoring.alerts
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

        # Dashboard service
        dashboard_service = f"""[Unit]
Description=Pynomaly Monitoring Dashboard
After=network.target
Requires=network.target

[Service]
Type=simple
User=pynomaly
Group=pynomaly
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}/src
ExecStart=/usr/bin/python3 -m monorepo.infrastructure.monitoring.dashboard
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

        # Create systemd directory
        systemd_dir = self.project_root / "config" / "systemd"
        systemd_dir.mkdir(parents=True, exist_ok=True)

        # Write service files
        with open(systemd_dir / "pynomaly-alerts.service", "w") as f:
            f.write(alert_service)

        with open(systemd_dir / "pynomaly-dashboard.service", "w") as f:
            f.write(dashboard_service)

        self.logger.info("Systemd service files created")

    def run_setup(self):
        """Run the complete monitoring setup."""
        self.logger.info("🔄 Starting Pynomaly monitoring setup...")

        try:
            # Setup configurations
            self.setup_prometheus()
            self.setup_grafana()
            self.setup_alertmanager()

            # Create Docker Compose
            self.create_docker_compose()

            # Create management scripts
            self.setup_monitoring_scripts()

            # Create systemd services
            self.setup_systemd_services()

            # Create dashboard templates
            if MONITORING_AVAILABLE:
                create_dashboard_templates()
            else:
                self.logger.warning("Skipping dashboard template creation due to missing dependencies")

            self.logger.info("✅ Monitoring setup completed successfully!")

            # Print next steps
            self._print_next_steps()

        except Exception as e:
            self.logger.error(f"❌ Setup failed: {e}")
            raise

    def _print_next_steps(self):
        """Print next steps for user."""
        print("\n" + "="*60)
        print("🎉 Pynomaly Monitoring Setup Complete!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Start the monitoring stack:")
        print("   ./scripts/monitoring/start-monitoring.sh")
        print("\n2. Access the monitoring interfaces:")
        print("   - Grafana Dashboard: http://localhost:3000 (admin/admin)")
        print("   - Prometheus: http://localhost:9090")
        print("   - Alertmanager: http://localhost:9093")
        print("   - Custom Dashboard: http://localhost:8080")
        print("\n3. Configure notifications:")
        print("   - Edit config/monitoring/alertmanager.yml")
        print("   - Add your email/Slack webhook URLs")
        print("\n4. Customize alert rules:")
        print("   - Edit config/monitoring/alert_rules.yml")
        print("   - Add application-specific metrics")
        print("\n5. Install as system services (optional):")
        print("   sudo cp config/systemd/*.service /etc/systemd/system/")
        print("   sudo systemctl enable pynomaly-alerts pynomaly-dashboard")
        print("   sudo systemctl start pynomaly-alerts pynomaly-dashboard")
        print("\n6. Check status:")
        print("   ./scripts/monitoring/monitoring-status.sh")
        print("\n🔧 Configuration files created:")
        print("   - config/monitoring/prometheus.yml")
        print("   - config/monitoring/grafana/")
        print("   - config/monitoring/alertmanager.yml")
        print("   - config/monitoring/alert_rules.yml")
        print("   - docker-compose.monitoring.yml")
        print("   - scripts/monitoring/")
        print("   - config/systemd/")
        print("\n" + "="*60)


def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Pynomaly monitoring setup")

    try:
        setup = MonitoringSetup()
        setup.run_setup()

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
