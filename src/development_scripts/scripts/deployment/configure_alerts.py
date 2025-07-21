#!/usr/bin/env python3
"""
Production Monitoring Alerts Configuration
Sets up comprehensive monitoring alerts for production deployment
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts"""

    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    SECURITY = "security"
    BUSINESS = "business"
    PERFORMANCE = "performance"


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    query: str
    severity: AlertSeverity
    alert_type: AlertType
    duration: str = "5m"
    description: str = ""
    runbook_url: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationChannel:
    """Notification channel configuration"""

    name: str
    type: str  # slack, email, pagerduty, webhook
    settings: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class AlertManager:
    """Manages production monitoring alerts"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.monitoring_dir = self.project_root / "config" / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_alert_config()

    def _load_alert_config(self) -> dict:
        """Load alert configuration"""
        config_file = self.monitoring_dir / "alert_config.yaml"

        default_config = {
            "global": {
                "smtp_smarthost": "smtp.company.com:587",
                "smtp_from": "alerts@anomaly_detection.com",
                "slack_api_url": os.getenv("SLACK_WEBHOOK_URL"),
            },
            "notification_channels": [
                {
                    "name": "critical-alerts",
                    "type": "slack",
                    "settings": {
                        "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                        "channel": "#alerts-critical",
                        "username": "anomaly_detection Alerts",
                    },
                },
                {
                    "name": "warning-alerts",
                    "type": "slack",
                    "settings": {
                        "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                        "channel": "#alerts-warning",
                        "username": "anomaly_detection Alerts",
                    },
                },
                {
                    "name": "pagerduty-critical",
                    "type": "pagerduty",
                    "settings": {
                        "integration_key": os.getenv("PAGERDUTY_INTEGRATION_KEY")
                    },
                },
            ],
        }

        if config_file.exists():
            try:
                with open(config_file) as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load alert config: {e}")

        return default_config

    def get_alert_rules(self) -> list[AlertRule]:
        """Define comprehensive alert rules"""
        rules = []

        # Infrastructure alerts
        rules.extend(
            [
                AlertRule(
                    name="HighCPUUsage",
                    query="rate(container_cpu_usage_seconds_total[5m]) > 0.8",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.INFRASTRUCTURE,
                    duration="5m",
                    description="Container CPU usage is above 80%",
                    runbook_url="https://runbooks.anomaly_detection.com/high-cpu",
                    labels={"team": "platform"},
                    annotations={
                        "summary": "High CPU usage detected on {{ $labels.container }}",
                        "description": "CPU usage is {{ $value | humanizePercentage }} on container {{ $labels.container }}",
                    },
                ),
                AlertRule(
                    name="HighMemoryUsage",
                    query="container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9",
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.INFRASTRUCTURE,
                    duration="5m",
                    description="Container memory usage is above 90%",
                    runbook_url="https://runbooks.anomaly_detection.com/high-memory",
                    labels={"team": "platform"},
                    annotations={
                        "summary": "High memory usage detected on {{ $labels.container }}",
                        "description": "Memory usage is {{ $value | humanizePercentage }} on container {{ $labels.container }}",
                    },
                ),
                AlertRule(
                    name="PodCrashLooping",
                    query="rate(kube_pod_container_status_restarts_total[15m]) > 0",
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.INFRASTRUCTURE,
                    duration="0m",
                    description="Pod is crash looping",
                    runbook_url="https://runbooks.anomaly_detection.com/pod-crash",
                    labels={"team": "platform"},
                    annotations={
                        "summary": "Pod {{ $labels.pod }} is crash looping",
                        "description": "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is restarting frequently",
                    },
                ),
                AlertRule(
                    name="NodeNotReady",
                    query='kube_node_status_condition{condition="Ready",status="true"} == 0',
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.INFRASTRUCTURE,
                    duration="5m",
                    description="Kubernetes node is not ready",
                    runbook_url="https://runbooks.anomaly_detection.com/node-not-ready",
                    labels={"team": "platform"},
                    annotations={
                        "summary": "Node {{ $labels.node }} is not ready",
                        "description": "Node {{ $labels.node }} has been not ready for more than 5 minutes",
                    },
                ),
                AlertRule(
                    name="DiskSpaceUsageHigh",
                    query="(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.85",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.INFRASTRUCTURE,
                    duration="5m",
                    description="Disk space usage is above 85%",
                    runbook_url="https://runbooks.anomaly_detection.com/disk-space",
                    labels={"team": "platform"},
                    annotations={
                        "summary": "High disk usage on {{ $labels.instance }}",
                        "description": "Disk usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}",
                    },
                ),
            ]
        )

        # Application alerts
        rules.extend(
            [
                AlertRule(
                    name="APIHighErrorRate",
                    query='rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05',
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.APPLICATION,
                    duration="2m",
                    description="API error rate is above 5%",
                    runbook_url="https://runbooks.anomaly_detection.com/api-errors",
                    labels={"team": "api"},
                    annotations={
                        "summary": "High error rate on API",
                        "description": "Error rate is {{ $value | humanizePercentage }} for endpoint {{ $labels.endpoint }}",
                    },
                ),
                AlertRule(
                    name="APIHighLatency",
                    query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.APPLICATION,
                    duration="5m",
                    description="API 95th percentile latency is above 2 seconds",
                    runbook_url="https://runbooks.anomaly_detection.com/api-latency",
                    labels={"team": "api"},
                    annotations={
                        "summary": "High API latency detected",
                        "description": "95th percentile latency is {{ $value }}s for endpoint {{ $labels.endpoint }}",
                    },
                ),
                AlertRule(
                    name="DatabaseConnectionPoolExhausted",
                    query="database_connection_pool_active / database_connection_pool_max > 0.9",
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.APPLICATION,
                    duration="2m",
                    description="Database connection pool is nearly exhausted",
                    runbook_url="https://runbooks.anomaly_detection.com/db-connections",
                    labels={"team": "backend"},
                    annotations={
                        "summary": "Database connection pool nearly exhausted",
                        "description": "Connection pool usage is {{ $value | humanizePercentage }}",
                    },
                ),
                AlertRule(
                    name="QueueBacklogHigh",
                    query="queue_backlog_size > 1000",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.APPLICATION,
                    duration="5m",
                    description="Message queue backlog is high",
                    runbook_url="https://runbooks.anomaly_detection.com/queue-backlog",
                    labels={"team": "backend"},
                    annotations={
                        "summary": "High queue backlog detected",
                        "description": "Queue {{ $labels.queue }} has {{ $value }} pending messages",
                    },
                ),
                AlertRule(
                    name="MLModelInferenceLatency",
                    query="histogram_quantile(0.95, rate(ml_inference_duration_seconds_bucket[5m])) > 5",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.APPLICATION,
                    duration="5m",
                    description="ML model inference latency is high",
                    runbook_url="https://runbooks.anomaly_detection.com/ml-latency",
                    labels={"team": "ml"},
                    annotations={
                        "summary": "High ML inference latency",
                        "description": "Model {{ $labels.model }} inference latency is {{ $value }}s",
                    },
                ),
            ]
        )

        # Security alerts
        rules.extend(
            [
                AlertRule(
                    name="SecurityScannerBlocked",
                    query='rate(waf_blocked_requests_total{attack_type="scanner"}[5m]) > 10',
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.SECURITY,
                    duration="2m",
                    description="High number of security scanner requests blocked",
                    runbook_url="https://runbooks.anomaly_detection.com/security-scanner",
                    labels={"team": "security"},
                    annotations={
                        "summary": "Security scanners detected",
                        "description": "{{ $value }} scanner requests per second blocked from {{ $labels.source_ip }}",
                    },
                ),
                AlertRule(
                    name="SQLInjectionAttempts",
                    query='rate(waf_blocked_requests_total{attack_type="sql_injection"}[5m]) > 1',
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.SECURITY,
                    duration="0m",
                    description="SQL injection attempts detected",
                    runbook_url="https://runbooks.anomaly_detection.com/sql-injection",
                    labels={"team": "security"},
                    annotations={
                        "summary": "SQL injection attempts detected",
                        "description": "{{ $value }} SQL injection attempts per second from {{ $labels.source_ip }}",
                    },
                ),
                AlertRule(
                    name="FailedAuthenticationHigh",
                    query="rate(authentication_failed_total[5m]) > 5",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.SECURITY,
                    duration="2m",
                    description="High number of failed authentication attempts",
                    runbook_url="https://runbooks.anomaly_detection.com/failed-auth",
                    labels={"team": "security"},
                    annotations={
                        "summary": "High failed authentication rate",
                        "description": "{{ $value }} failed authentications per second",
                    },
                ),
                AlertRule(
                    name="UnauthorizedAccessAttempt",
                    query='rate(http_requests_total{status="403"}[5m]) > 10',
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.SECURITY,
                    duration="5m",
                    description="High number of unauthorized access attempts",
                    runbook_url="https://runbooks.anomaly_detection.com/unauthorized-access",
                    labels={"team": "security"},
                    annotations={
                        "summary": "Unauthorized access attempts detected",
                        "description": "{{ $value }} unauthorized requests per second to {{ $labels.endpoint }}",
                    },
                ),
            ]
        )

        # Business alerts
        rules.extend(
            [
                AlertRule(
                    name="UserRegistrationDropped",
                    query="rate(user_registrations_total[1h]) < 0.5",
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.BUSINESS,
                    duration="30m",
                    description="User registration rate has dropped significantly",
                    runbook_url="https://runbooks.anomaly_detection.com/user-registration",
                    labels={"team": "product"},
                    annotations={
                        "summary": "User registration rate dropped",
                        "description": "Only {{ $value }} registrations per hour in the last hour",
                    },
                ),
                AlertRule(
                    name="AnomalyDetectionJobsFailing",
                    query="rate(anomaly_detection_jobs_failed_total[1h]) / rate(anomaly_detection_jobs_total[1h]) > 0.1",
                    severity=AlertSeverity.CRITICAL,
                    alert_type=AlertType.BUSINESS,
                    duration="10m",
                    description="High anomaly detection job failure rate",
                    runbook_url="https://runbooks.anomaly_detection.com/job-failures",
                    labels={"team": "ml"},
                    annotations={
                        "summary": "High anomaly detection job failure rate",
                        "description": "{{ $value | humanizePercentage }} of anomaly detection jobs are failing",
                    },
                ),
                AlertRule(
                    name="RevenueImpactingError",
                    query="rate(payment_processing_errors_total[5m]) > 0",
                    severity=AlertSeverity.EMERGENCY,
                    alert_type=AlertType.BUSINESS,
                    duration="0m",
                    description="Payment processing errors detected",
                    runbook_url="https://runbooks.anomaly_detection.com/payment-errors",
                    labels={"team": "billing"},
                    annotations={
                        "summary": "Payment processing errors detected",
                        "description": "{{ $value }} payment errors per second - immediate attention required",
                    },
                ),
            ]
        )

        return rules

    def generate_prometheus_rules(self, rules: list[AlertRule]) -> str:
        """Generate Prometheus alert rules YAML"""
        groups = {}

        # Group rules by type
        for rule in rules:
            group_name = f"anomaly_detection-{rule.alert_type.value}"
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(rule)

        # Generate YAML structure
        rule_groups = []
        for group_name, group_rules in groups.items():
            rule_group = {"name": group_name, "rules": []}

            for rule in group_rules:
                rule_config = {
                    "alert": rule.name,
                    "expr": rule.query,
                    "for": rule.duration,
                    "labels": {
                        "severity": rule.severity.value,
                        "service": "anomaly_detection",
                        **rule.labels,
                    },
                    "annotations": {
                        "description": rule.description,
                        "runbook_url": rule.runbook_url,
                        **rule.annotations,
                    },
                }
                rule_group["rules"].append(rule_config)

            rule_groups.append(rule_group)

        prometheus_rules = {"groups": rule_groups}
        return yaml.dump(prometheus_rules, default_flow_style=False)

    def generate_alertmanager_config(self) -> str:
        """Generate Alertmanager configuration"""
        config = {
            "global": self.config["global"],
            "route": {
                "group_by": ["alertname", "cluster", "service"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "1h",
                "receiver": "default-receiver",
                "routes": [
                    {
                        "match": {"severity": "critical"},
                        "receiver": "critical-alerts",
                        "group_wait": "10s",
                        "repeat_interval": "10m",
                    },
                    {
                        "match": {"severity": "emergency"},
                        "receiver": "emergency-alerts",
                        "group_wait": "0s",
                        "repeat_interval": "1m",
                    },
                    {
                        "match": {"severity": "warning"},
                        "receiver": "warning-alerts",
                        "group_wait": "30s",
                        "repeat_interval": "4h",
                    },
                ],
            },
            "receivers": [
                {
                    "name": "default-receiver",
                    "slack_configs": [
                        {
                            "api_url": self.config["global"]["slack_api_url"],
                            "channel": "#alerts-default",
                            "title": "anomaly_detection Alert",
                            "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                        }
                    ],
                },
                {
                    "name": "critical-alerts",
                    "slack_configs": [
                        {
                            "api_url": self.config["global"]["slack_api_url"],
                            "channel": "#alerts-critical",
                            "title": "🚨 CRITICAL: {{ .GroupLabels.alertname }}",
                            "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                            "color": "danger",
                        }
                    ],
                    "pagerduty_configs": [
                        {
                            "routing_key": os.getenv("PAGERDUTY_INTEGRATION_KEY"),
                            "description": "{{ .GroupLabels.alertname }}: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                        }
                    ],
                },
                {
                    "name": "emergency-alerts",
                    "slack_configs": [
                        {
                            "api_url": self.config["global"]["slack_api_url"],
                            "channel": "#alerts-emergency",
                            "title": "🚨🚨 EMERGENCY: {{ .GroupLabels.alertname }}",
                            "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                            "color": "danger",
                        }
                    ],
                    "pagerduty_configs": [
                        {
                            "routing_key": os.getenv("PAGERDUTY_INTEGRATION_KEY"),
                            "severity": "critical",
                            "description": "EMERGENCY: {{ .GroupLabels.alertname }}: {{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                        }
                    ],
                    "email_configs": [
                        {
                            "to": "oncall@anomaly_detection.com",
                            "subject": "EMERGENCY: {{ .GroupLabels.alertname }}",
                            "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                        }
                    ],
                },
                {
                    "name": "warning-alerts",
                    "slack_configs": [
                        {
                            "api_url": self.config["global"]["slack_api_url"],
                            "channel": "#alerts-warning",
                            "title": "⚠️ WARNING: {{ .GroupLabels.alertname }}",
                            "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}",
                            "color": "warning",
                        }
                    ],
                },
            ],
        }

        return yaml.dump(config, default_flow_style=False)

    def deploy_alert_rules(self):
        """Deploy alert rules to monitoring system"""
        try:
            logger.info("Deploying alert rules")

            # Generate alert rules
            rules = self.get_alert_rules()
            prometheus_rules = self.generate_prometheus_rules(rules)

            # Save Prometheus rules
            rules_file = self.monitoring_dir / "prometheus_alert_rules.yml"
            with open(rules_file, "w") as f:
                f.write(prometheus_rules)

            logger.info(f"Prometheus alert rules saved: {rules_file}")

            # Generate Alertmanager config
            alertmanager_config = self.generate_alertmanager_config()

            # Save Alertmanager config
            alertmanager_file = self.monitoring_dir / "alertmanager.yml"
            with open(alertmanager_file, "w") as f:
                f.write(alertmanager_config)

            logger.info(f"Alertmanager config saved: {alertmanager_file}")

            # Deploy to Kubernetes (if in cluster)
            self._deploy_to_kubernetes()

            # Reload Prometheus and Alertmanager
            self._reload_monitoring_services()

            logger.info("Alert rules deployed successfully")

        except Exception as e:
            logger.error(f"Failed to deploy alert rules: {e}")
            raise

    def _deploy_to_kubernetes(self):
        """Deploy alert configurations to Kubernetes"""
        try:
            # Check if running in Kubernetes
            if not os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount"):
                logger.info("Not running in Kubernetes, skipping K8s deployment")
                return

            # Create ConfigMap for Prometheus rules
            rules_file = self.monitoring_dir / "prometheus_alert_rules.yml"
            if rules_file.exists():
                cmd = [
                    "kubectl",
                    "create",
                    "configmap",
                    "prometheus-alert-rules",
                    f"--from-file={rules_file}",
                    "--namespace=monitoring",
                    "--dry-run=client",
                    "-o",
                    "yaml",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Apply the ConfigMap
                    apply_cmd = ["kubectl", "apply", "-f", "-"]
                    subprocess.run(
                        apply_cmd, input=result.stdout, text=True, check=True
                    )
                    logger.info("Prometheus alert rules ConfigMap deployed")

            # Create ConfigMap for Alertmanager config
            alertmanager_file = self.monitoring_dir / "alertmanager.yml"
            if alertmanager_file.exists():
                cmd = [
                    "kubectl",
                    "create",
                    "configmap",
                    "alertmanager-config",
                    f"--from-file={alertmanager_file}",
                    "--namespace=monitoring",
                    "--dry-run=client",
                    "-o",
                    "yaml",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Apply the ConfigMap
                    apply_cmd = ["kubectl", "apply", "-f", "-"]
                    subprocess.run(
                        apply_cmd, input=result.stdout, text=True, check=True
                    )
                    logger.info("Alertmanager config ConfigMap deployed")

        except Exception as e:
            logger.warning(f"Kubernetes deployment failed: {e}")

    def _reload_monitoring_services(self):
        """Reload Prometheus and Alertmanager services"""
        try:
            # Reload Prometheus
            prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
            reload_url = f"{prometheus_url}/-/reload"

            import requests

            response = requests.post(reload_url, timeout=30)
            if response.status_code == 200:
                logger.info("Prometheus configuration reloaded")
            else:
                logger.warning(f"Failed to reload Prometheus: {response.status_code}")

            # Reload Alertmanager
            alertmanager_url = os.getenv("ALERTMANAGER_URL", "http://alertmanager:9093")
            reload_url = f"{alertmanager_url}/-/reload"

            response = requests.post(reload_url, timeout=30)
            if response.status_code == 200:
                logger.info("Alertmanager configuration reloaded")
            else:
                logger.warning(f"Failed to reload Alertmanager: {response.status_code}")

        except Exception as e:
            logger.warning(f"Failed to reload monitoring services: {e}")

    def test_alert_rules(self):
        """Test alert rule syntax and configuration"""
        try:
            logger.info("Testing alert rules")

            rules = self.get_alert_rules()
            prometheus_rules = self.generate_prometheus_rules(rules)

            # Save to temporary file
            temp_file = self.monitoring_dir / "test_rules.yml"
            with open(temp_file, "w") as f:
                f.write(prometheus_rules)

            # Use promtool to validate rules
            cmd = ["promtool", "check", "rules", str(temp_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Alert rules validation passed")
                # Clean up temp file
                temp_file.unlink()
                return True
            else:
                logger.error(f"Alert rules validation failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Alert rule testing failed: {e}")
            return False

    def generate_alert_documentation(self):
        """Generate documentation for alert rules"""
        try:
            logger.info("Generating alert documentation")

            rules = self.get_alert_rules()

            # Group by type
            groups = {}
            for rule in rules:
                if rule.alert_type not in groups:
                    groups[rule.alert_type] = []
                groups[rule.alert_type].append(rule)

            # Generate markdown documentation
            doc_content = "# anomaly_detection Production Alert Rules\n\n"
            doc_content += "This document describes all monitoring alerts configured for anomaly_detection production environment.\n\n"

            for alert_type, type_rules in groups.items():
                doc_content += f"## {alert_type.value.title()} Alerts\n\n"

                for rule in type_rules:
                    doc_content += f"### {rule.name}\n\n"
                    doc_content += f"**Severity:** {rule.severity.value.upper()}\n\n"
                    doc_content += f"**Description:** {rule.description}\n\n"
                    doc_content += f"**Query:** `{rule.query}`\n\n"
                    doc_content += f"**Duration:** {rule.duration}\n\n"
                    if rule.runbook_url:
                        doc_content += f"**Runbook:** {rule.runbook_url}\n\n"
                    doc_content += "---\n\n"

            # Save documentation
            doc_file = self.project_root / "docs" / "runbooks" / "alert_rules.md"
            doc_file.parent.mkdir(parents=True, exist_ok=True)

            with open(doc_file, "w") as f:
                f.write(doc_content)

            logger.info(f"Alert documentation generated: {doc_file}")

        except Exception as e:
            logger.error(f"Failed to generate alert documentation: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Configure production monitoring alerts"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy alert rules to monitoring system"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test alert rule syntax and configuration"
    )
    parser.add_argument(
        "--generate-docs", action="store_true", help="Generate alert documentation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configurations without deploying",
    )

    args = parser.parse_args()

    alert_manager = AlertManager()

    try:
        if args.test:
            if alert_manager.test_alert_rules():
                logger.info("✅ Alert rules test passed")
            else:
                logger.error("❌ Alert rules test failed")
                sys.exit(1)

        if args.generate_docs:
            alert_manager.generate_alert_documentation()

        if args.deploy and not args.dry_run:
            alert_manager.deploy_alert_rules()
            logger.info("🎉 Alert rules deployed successfully!")
        elif args.dry_run:
            # Generate configurations for review
            rules = alert_manager.get_alert_rules()
            prometheus_rules = alert_manager.generate_prometheus_rules(rules)
            alertmanager_config = alert_manager.generate_alertmanager_config()

            print("Generated Prometheus Rules:")
            print("=" * 50)
            print(prometheus_rules)
            print("\nGenerated Alertmanager Config:")
            print("=" * 50)
            print(alertmanager_config)
        else:
            logger.info("Use --deploy to deploy alert rules or --test to validate them")

    except Exception as e:
        logger.error(f"Alert configuration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
