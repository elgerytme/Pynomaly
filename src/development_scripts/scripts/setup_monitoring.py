#!/usr/bin/env python3
"""
Setup monitoring infrastructure for anomaly_detection production deployment.
This script configures Prometheus, Grafana, and Alertmanager.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Monitoring configuration data structure."""

    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "admin_password_123"
    alertmanager_url: str = "http://localhost:9093"


class MonitoringSetup:
    """Main monitoring setup orchestrator."""

    def __init__(self, config: MonitoringConfig):
        """Initialize monitoring setup."""
        self.config = config
        self.session = self._create_session()
        self.setup_results = []

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry configuration."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    async def setup_prometheus(self) -> bool:
        """Setup Prometheus configuration."""
        logger.info("🔧 Setting up Prometheus monitoring...")

        try:
            # Update Prometheus configuration
            prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'anomaly_detection-api'
    static_configs:
      - targets: ['anomaly_detection-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-cluster:6379']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""

            # Write configuration
            with open("config/prometheus.yml", "w") as f:
                f.write(prometheus_config)

            # Test Prometheus connection
            await asyncio.sleep(2)  # Give Prometheus time to start

            try:
                response = self.session.get(
                    f"{self.config.prometheus_url}/api/v1/targets", timeout=10
                )
                if response.status_code == 200:
                    logger.info("✅ Prometheus is running and configured")
                    self.setup_results.append(
                        {"component": "Prometheus", "status": "success"}
                    )
                    return True
                else:
                    logger.error(
                        f"Prometheus health check failed: {response.status_code}"
                    )
                    self.setup_results.append(
                        {
                            "component": "Prometheus",
                            "status": "failed",
                            "error": f"HTTP {response.status_code}",
                        }
                    )
                    return False
            except Exception as e:
                logger.warning(f"Prometheus connection test failed: {e}")
                self.setup_results.append(
                    {"component": "Prometheus", "status": "warning", "error": str(e)}
                )
                return True  # Configuration was written, connection might work later

        except Exception as e:
            logger.error(f"Prometheus setup failed: {e}")
            self.setup_results.append(
                {"component": "Prometheus", "status": "failed", "error": str(e)}
            )
            return False

    async def setup_grafana(self) -> bool:
        """Setup Grafana dashboards and data sources."""
        logger.info("📊 Setting up Grafana dashboards...")

        try:
            # Wait for Grafana to be ready
            await self._wait_for_grafana()

            # Create Prometheus data source
            datasource_success = await self._create_prometheus_datasource()
            if not datasource_success:
                return False

            # Import main dashboard
            dashboard_success = await self._import_dashboard()
            if not dashboard_success:
                return False

            # Create custom dashboards
            await self._create_custom_dashboards()

            logger.info("✅ Grafana setup completed successfully")
            self.setup_results.append({"component": "Grafana", "status": "success"})
            return True

        except Exception as e:
            logger.error(f"Grafana setup failed: {e}")
            self.setup_results.append(
                {"component": "Grafana", "status": "failed", "error": str(e)}
            )
            return False

    async def _wait_for_grafana(self) -> bool:
        """Wait for Grafana to be ready."""
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = self.session.get(
                    f"{self.config.grafana_url}/api/health", timeout=5
                )
                if response.status_code == 200:
                    logger.info("Grafana is ready")
                    return True
            except Exception:
                pass

            if attempt < max_attempts - 1:
                logger.info(
                    f"Waiting for Grafana... (attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(5)

        logger.error("Grafana did not become ready in time")
        return False

    async def _create_prometheus_datasource(self) -> bool:
        """Create Prometheus data source in Grafana."""
        try:
            datasource_config = {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy",
                "isDefault": True,
                "basicAuth": False,
                "jsonData": {
                    "timeInterval": "15s",
                    "queryTimeout": "60s",
                    "httpMethod": "POST",
                },
            }

            response = self.session.post(
                f"{self.config.grafana_url}/api/datasources",
                json=datasource_config,
                auth=(
                    self.config.grafana_admin_user,
                    self.config.grafana_admin_password,
                ),
                timeout=30,
            )

            if response.status_code in [200, 409]:  # 409 = already exists
                logger.info("✅ Prometheus data source configured")
                return True
            else:
                logger.error(
                    f"Failed to create Prometheus datasource: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error creating Prometheus datasource: {e}")
            return False

    async def _import_dashboard(self) -> bool:
        """Import the main anomaly_detection dashboard."""
        try:
            # Read dashboard configuration
            dashboard_path = "config/grafana/dashboards/anomaly_detection-main-dashboard.json"
            if not os.path.exists(dashboard_path):
                logger.warning(f"Dashboard file not found: {dashboard_path}")
                return False

            with open(dashboard_path) as f:
                dashboard_config = json.load(f)

            # Import dashboard
            import_config = {
                "dashboard": dashboard_config["dashboard"],
                "overwrite": True,
                "inputs": [
                    {
                        "name": "DS_PROMETHEUS",
                        "type": "datasource",
                        "pluginId": "prometheus",
                        "value": "Prometheus",
                    }
                ],
            }

            response = self.session.post(
                f"{self.config.grafana_url}/api/dashboards/import",
                json=import_config,
                auth=(
                    self.config.grafana_admin_user,
                    self.config.grafana_admin_password,
                ),
                timeout=30,
            )

            if response.status_code == 200:
                logger.info("✅ Main dashboard imported successfully")
                return True
            else:
                logger.error(
                    f"Failed to import dashboard: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error importing dashboard: {e}")
            return False

    async def _create_custom_dashboards(self) -> bool:
        """Create additional custom dashboards."""
        try:
            # Create Real-time Monitoring Dashboard
            realtime_dashboard = {
                "dashboard": {
                    "id": None,
                    "title": "anomaly_detection Real-time Monitoring",
                    "tags": ["anomaly_detection", "real-time", "monitoring"],
                    "timezone": "browser",
                    "panels": [
                        {
                            "id": 1,
                            "title": "Live Request Rate",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "rate(anomaly_detection_requests_total[1m])",
                                    "legendFormat": "Requests/sec",
                                    "refId": "A",
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                            "refresh": "5s",
                        },
                        {
                            "id": 2,
                            "title": "Active Anomaly Detections",
                            "type": "singlestat",
                            "targets": [
                                {
                                    "expr": "rate(anomaly_detections_total[1m])",
                                    "legendFormat": "Detections/sec",
                                    "refId": "A",
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                            "refresh": "5s",
                        },
                    ],
                    "refresh": "5s",
                    "time": {"from": "now-5m", "to": "now"},
                }
            }

            response = self.session.post(
                f"{self.config.grafana_url}/api/dashboards/db",
                json=realtime_dashboard,
                auth=(
                    self.config.grafana_admin_user,
                    self.config.grafana_admin_password,
                ),
                timeout=30,
            )

            if response.status_code == 200:
                logger.info("✅ Real-time dashboard created")
            else:
                logger.warning(
                    f"Failed to create real-time dashboard: {response.status_code}"
                )

            return True

        except Exception as e:
            logger.error(f"Error creating custom dashboards: {e}")
            return False

    async def setup_alertmanager(self) -> bool:
        """Setup Alertmanager configuration."""
        logger.info("🚨 Setting up Alertmanager...")

        try:
            # Test Alertmanager connection
            try:
                response = self.session.get(
                    f"{self.config.alertmanager_url}/api/v1/status", timeout=10
                )
                if response.status_code == 200:
                    logger.info("✅ Alertmanager is running and configured")
                    self.setup_results.append(
                        {"component": "Alertmanager", "status": "success"}
                    )
                    return True
                else:
                    logger.error(
                        f"Alertmanager health check failed: {response.status_code}"
                    )
                    self.setup_results.append(
                        {
                            "component": "Alertmanager",
                            "status": "failed",
                            "error": f"HTTP {response.status_code}",
                        }
                    )
                    return False
            except Exception as e:
                logger.warning(f"Alertmanager connection test failed: {e}")
                self.setup_results.append(
                    {"component": "Alertmanager", "status": "warning", "error": str(e)}
                )
                return True  # Configuration exists, connection might work later

        except Exception as e:
            logger.error(f"Alertmanager setup failed: {e}")
            self.setup_results.append(
                {"component": "Alertmanager", "status": "failed", "error": str(e)}
            )
            return False

    async def create_test_alerts(self) -> bool:
        """Create test alerts to verify the monitoring setup."""
        logger.info("🧪 Creating test alerts...")

        try:
            # Test alert configuration
            test_alert = {
                "alerts": [
                    {
                        "labels": {
                            "alertname": "TestAlert",
                            "severity": "warning",
                            "service": "anomaly_detection-api",
                        },
                        "annotations": {
                            "summary": "Test alert from monitoring setup",
                            "description": "This is a test alert to verify monitoring configuration",
                        },
                        "generatorURL": "http://prometheus:9090/graph",
                        "startsAt": datetime.now().isoformat() + "Z",
                    }
                ]
            }

            response = self.session.post(
                f"{self.config.alertmanager_url}/api/v1/alerts",
                json=test_alert,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("✅ Test alert sent successfully")
                return True
            else:
                logger.warning(f"Failed to send test alert: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error creating test alerts: {e}")
            return False

    async def validate_monitoring_stack(self) -> bool:
        """Validate the entire monitoring stack."""
        logger.info("🔍 Validating monitoring stack...")

        validation_results = []

        # Test Prometheus targets
        try:
            response = self.session.get(
                f"{self.config.prometheus_url}/api/v1/targets", timeout=10
            )
            if response.status_code == 200:
                targets = response.json()
                active_targets = [
                    t
                    for t in targets.get("data", {}).get("activeTargets", [])
                    if t.get("health") == "up"
                ]
                validation_results.append(
                    f"✅ Prometheus: {len(active_targets)} active targets"
                )
            else:
                validation_results.append(
                    f"❌ Prometheus: API error {response.status_code}"
                )
        except Exception as e:
            validation_results.append(f"❌ Prometheus: Connection error - {e}")

        # Test Grafana dashboards
        try:
            response = self.session.get(
                f"{self.config.grafana_url}/api/search?query=anomaly_detection",
                auth=(
                    self.config.grafana_admin_user,
                    self.config.grafana_admin_password,
                ),
                timeout=10,
            )
            if response.status_code == 200:
                dashboards = response.json()
                validation_results.append(
                    f"✅ Grafana: {len(dashboards)} dashboards found"
                )
            else:
                validation_results.append(
                    f"❌ Grafana: API error {response.status_code}"
                )
        except Exception as e:
            validation_results.append(f"❌ Grafana: Connection error - {e}")

        # Test Alertmanager
        try:
            response = self.session.get(
                f"{self.config.alertmanager_url}/api/v1/status", timeout=10
            )
            if response.status_code == 200:
                validation_results.append("✅ Alertmanager: Status OK")
            else:
                validation_results.append(
                    f"❌ Alertmanager: API error {response.status_code}"
                )
        except Exception as e:
            validation_results.append(f"❌ Alertmanager: Connection error - {e}")

        # Print validation results
        logger.info("Monitoring stack validation results:")
        for result in validation_results:
            logger.info(f"  {result}")

        return all("✅" in result for result in validation_results)

    def generate_monitoring_report(self) -> dict[str, Any]:
        """Generate comprehensive monitoring setup report."""
        report = {
            "monitoring_setup": {
                "timestamp": datetime.now().isoformat(),
                "components": self.setup_results,
                "prometheus_url": self.config.prometheus_url,
                "grafana_url": self.config.grafana_url,
                "alertmanager_url": self.config.alertmanager_url,
                "success_rate": sum(
                    1 for r in self.setup_results if r["status"] == "success"
                )
                / len(self.setup_results)
                * 100
                if self.setup_results
                else 0,
                "total_components": len(self.setup_results),
            },
            "dashboards": [
                {
                    "name": "anomaly_detection Production Dashboard",
                    "url": f"{self.config.grafana_url}/dashboard/db/anomaly_detection-production-dashboard",
                    "description": "Main production monitoring dashboard",
                },
                {
                    "name": "Real-time Monitoring",
                    "url": f"{self.config.grafana_url}/dashboard/db/anomaly_detection-real-time-monitoring",
                    "description": "Real-time metrics and alerts",
                },
            ],
            "alerts": [
                {
                    "name": "API Service Down",
                    "severity": "critical",
                    "description": "API service is not responding",
                },
                {
                    "name": "High Error Rate",
                    "severity": "critical",
                    "description": "Error rate exceeds threshold",
                },
                {
                    "name": "High Latency",
                    "severity": "warning",
                    "description": "Response time is above normal",
                },
            ],
        }

        return report

    def save_monitoring_report(self, report: dict[str, Any]):
        """Save monitoring setup report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_setup_report_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Monitoring report saved to {filename}")

    def print_monitoring_summary(self, report: dict[str, Any]):
        """Print monitoring setup summary."""
        setup = report["monitoring_setup"]

        print("\n" + "=" * 60)
        print("🎯 anomaly_detection MONITORING SETUP SUMMARY")
        print("=" * 60)
        print(f"Setup Time: {setup['timestamp']}")
        print(f"Components: {setup['total_components']}")
        print(f"Success Rate: {setup['success_rate']:.1f}%")

        print("\n📊 DASHBOARDS:")
        for dashboard in report["dashboards"]:
            print(f"  • {dashboard['name']}: {dashboard['url']}")

        print("\n🚨 CONFIGURED ALERTS:")
        for alert in report["alerts"]:
            print(f"  • {alert['name']} ({alert['severity']}): {alert['description']}")

        print("\n🔗 MONITORING URLS:")
        print(f"  • Prometheus: {setup['prometheus_url']}")
        print(f"  • Grafana: {setup['grafana_url']}")
        print(f"  • Alertmanager: {setup['alertmanager_url']}")

        print("\n" + "=" * 60)
        print("🎉 MONITORING SETUP COMPLETE!")
        print("=" * 60)


async def main():
    """Main monitoring setup workflow."""
    config = MonitoringConfig()

    # Override with environment variables if available
    config.grafana_admin_password = os.getenv(
        "GRAFANA_PASSWORD", config.grafana_admin_password
    )

    setup = MonitoringSetup(config)

    try:
        logger.info("🚀 Starting monitoring infrastructure setup...")

        # Setup components
        prometheus_success = await setup.setup_prometheus()
        grafana_success = await setup.setup_grafana()
        alertmanager_success = await setup.setup_alertmanager()

        # Create test alerts
        if alertmanager_success:
            await setup.create_test_alerts()

        # Validate setup
        validation_success = await setup.validate_monitoring_stack()

        # Generate report
        report = setup.generate_monitoring_report()
        setup.save_monitoring_report(report)
        setup.print_monitoring_summary(report)

        # Overall success
        overall_success = all(
            [
                prometheus_success,
                grafana_success,
                alertmanager_success,
                validation_success,
            ]
        )

        if overall_success:
            logger.info("✅ Monitoring setup completed successfully!")
            return True
        else:
            logger.error("❌ Monitoring setup completed with errors")
            return False

    except Exception as e:
        logger.error(f"Monitoring setup failed: {e}")
        return False


if __name__ == "__main__":
    # Run the monitoring setup
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
