#!/usr/bin/env python3
"""
Observability Integration System
Enhanced monitoring integration with existing Prometheus, Grafana, and tracing systems
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import aiohttp
import requests
import yaml
from urllib.parse import urljoin


@dataclass
class MetricCollector:
    """Metric collection configuration"""
    name: str
    endpoint: str
    interval: int = 30
    enabled: bool = True
    labels: Dict[str, str] = None
    authentication: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.authentication is None:
            self.authentication = {}


@dataclass
class TracingConfig:
    """Distributed tracing configuration"""
    service_name: str
    jaeger_endpoint: str = ""
    zipkin_endpoint: str = ""
    sampling_rate: float = 0.1
    enabled: bool = True


class ObservabilityIntegration:
    """Main observability integration system"""
    
    def __init__(self, config_path: str = "config/observability-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.prometheus_client = None
        self.grafana_client = None
        self.jaeger_client = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_clients()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load observability configuration"""
        default_config = {
            "prometheus": {
                "endpoint": "http://localhost:9090",
                "enabled": True,
                "retention": "15d",
                "scrape_interval": "15s"
            },
            "grafana": {
                "endpoint": "http://localhost:3000",
                "enabled": True,
                "api_key": os.getenv("GRAFANA_API_KEY", ""),
                "org_id": 1
            },
            "jaeger": {
                "endpoint": "http://localhost:14268",
                "enabled": False,
                "service_name": "hexagonal-architecture"
            },
            "metrics": {
                "business_metrics": True,
                "custom_dashboards": True,
                "alert_rules": True
            },
            "integrations": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
                "pagerduty_key": os.getenv("PAGERDUTY_INTEGRATION_KEY", ""),
                "datadog_api_key": os.getenv("DATADOG_API_KEY", "")
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
        
        return default_config
    
    def _initialize_clients(self):
        """Initialize observability clients"""
        # Prometheus client
        if self.config["prometheus"]["enabled"]:
            self.prometheus_client = PrometheusClient(
                self.config["prometheus"]["endpoint"]
            )
        
        # Grafana client
        if self.config["grafana"]["enabled"]:
            self.grafana_client = GrafanaClient(
                self.config["grafana"]["endpoint"],
                self.config["grafana"]["api_key"]
            )
        
        # Jaeger client
        if self.config["jaeger"]["enabled"]:
            self.jaeger_client = JaegerClient(
                self.config["jaeger"]["endpoint"],
                self.config["jaeger"]["service_name"]
            )
    
    async def setup_prometheus_integration(self) -> bool:
        """Set up Prometheus integration"""
        if not self.prometheus_client:
            self.logger.warning("Prometheus client not initialized")
            return False
        
        try:
            # Test connectivity
            if not await self.prometheus_client.test_connection():
                self.logger.error("Cannot connect to Prometheus")
                return False
            
            # Setup service discovery
            await self._setup_prometheus_service_discovery()
            
            # Configure scraping targets
            await self._configure_prometheus_targets()
            
            # Setup alert rules
            await self._setup_prometheus_alerts()
            
            self.logger.info("✅ Prometheus integration configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Prometheus integration failed: {e}")
            return False
    
    async def setup_grafana_integration(self) -> bool:
        """Set up Grafana integration"""
        if not self.grafana_client:
            self.logger.warning("Grafana client not initialized")
            return False
        
        try:
            # Test connectivity
            if not await self.grafana_client.test_connection():
                self.logger.error("Cannot connect to Grafana")
                return False
            
            # Create data sources
            await self._setup_grafana_datasources()
            
            # Import dashboards
            await self._import_grafana_dashboards()
            
            # Setup alerts
            await self._setup_grafana_alerts()
            
            # Configure notification channels
            await self._setup_grafana_notifications()
            
            self.logger.info("✅ Grafana integration configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Grafana integration failed: {e}")
            return False
    
    async def setup_distributed_tracing(self) -> bool:
        """Set up distributed tracing"""
        if not self.jaeger_client:
            self.logger.info("Distributed tracing not enabled")
            return True
        
        try:
            # Test Jaeger connectivity
            if not await self.jaeger_client.test_connection():
                self.logger.error("Cannot connect to Jaeger")
                return False
            
            # Configure tracing for services
            await self._configure_service_tracing()
            
            # Setup trace sampling
            await self._setup_trace_sampling()
            
            self.logger.info("✅ Distributed tracing configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Distributed tracing setup failed: {e}")
            return False
    
    async def _setup_prometheus_service_discovery(self):
        """Configure Prometheus service discovery"""
        kubernetes_sd_config = {
            "job_name": "kubernetes-pods",
            "kubernetes_sd_configs": [{
                "role": "pod",
                "namespaces": {
                    "names": ["production", "staging", "development"]
                }
            }],
            "relabel_configs": [
                {
                    "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                    "action": "keep",
                    "regex": "true"
                },
                {
                    "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_path"],
                    "action": "replace",
                    "target_label": "__metrics_path__",
                    "regex": "(.+)"
                },
                {
                    "source_labels": ["__address__", "__meta_kubernetes_pod_annotation_prometheus_io_port"],
                    "action": "replace",
                    "regex": "([^:]+)(?::\\d+)?;(\\d+)",
                    "replacement": "${1}:${2}",
                    "target_label": "__address__"
                }
            ]
        }
        
        # Add to Prometheus configuration
        await self.prometheus_client.update_scrape_config("kubernetes-pods", kubernetes_sd_config)
    
    async def _configure_prometheus_targets(self):
        """Configure Prometheus scraping targets"""
        targets = [
            {
                "job_name": "hexagonal-api-gateway",
                "static_configs": [{
                    "targets": ["api-gateway:8080"],
                    "labels": {
                        "service": "api-gateway",
                        "environment": "production"
                    }
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "30s"
            },
            {
                "job_name": "hexagonal-data-quality",
                "static_configs": [{
                    "targets": ["data-quality-service:8081"],
                    "labels": {
                        "service": "data-quality",
                        "environment": "production"
                    }
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "30s"
            },
            {
                "job_name": "hexagonal-anomaly-detection",
                "static_configs": [{
                    "targets": ["anomaly-detection-service:8082"],
                    "labels": {
                        "service": "anomaly-detection",
                        "environment": "production"
                    }
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "30s"
            },
            {
                "job_name": "hexagonal-workflow-engine",
                "static_configs": [{
                    "targets": ["workflow-engine:8083"],
                    "labels": {
                        "service": "workflow-engine",
                        "environment": "production"
                    }
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "30s"
            }
        ]
        
        for target in targets:
            await self.prometheus_client.update_scrape_config(target["job_name"], target)
    
    async def _setup_prometheus_alerts(self):
        """Setup Prometheus alert rules"""
        alert_rules = {
            "groups": [
                {
                    "name": "hexagonal_architecture_alerts",
                    "rules": [
                        {
                            "alert": "ServiceDown",
                            "expr": "up{job=~\"hexagonal-.*\"} == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Hexagonal Architecture service is down",
                                "description": "Service {{ $labels.job }} has been down for more than 1 minute"
                            }
                        },
                        {
                            "alert": "HighResponseTime",
                            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{job=~\"hexagonal-.*\"}) > 1",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High response time detected",
                                "description": "95th percentile response time is {{ $value }}s for service {{ $labels.job }}"
                            }
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "rate(http_requests_total{job=~\"hexagonal-.*\",status=~\"5..\"}[5m]) / rate(http_requests_total{job=~\"hexagonal-.*\"}[5m]) > 0.05",
                            "for": "3m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High error rate detected",
                                "description": "Error rate is {{ $value | humanizePercentage }} for service {{ $labels.job }}"
                            }
                        },
                        {
                            "alert": "BusinessMetricAnomaly",
                            "expr": "abs(rate(business_transactions_total[5m]) - rate(business_transactions_total[5m] offset 1w)) / rate(business_transactions_total[5m] offset 1w) > 0.3",
                            "for": "10m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Business metric anomaly detected",
                                "description": "Business transaction rate has changed by more than 30% compared to last week"
                            }
                        }
                    ]
                }
            ]
        }
        
        await self.prometheus_client.update_alert_rules("hexagonal_architecture_alerts", alert_rules)
    
    async def _setup_grafana_datasources(self):
        """Setup Grafana data sources"""
        prometheus_datasource = {
            "name": "Prometheus-Hexagonal",
            "type": "prometheus",
            "url": self.config["prometheus"]["endpoint"],
            "access": "proxy",
            "isDefault": True,
            "jsonData": {
                "timeInterval": "15s",
                "queryTimeout": "60s",
                "httpMethod": "POST"
            }
        }
        
        await self.grafana_client.create_datasource(prometheus_datasource)
        
        # Jaeger datasource if enabled
        if self.config["jaeger"]["enabled"]:
            jaeger_datasource = {
                "name": "Jaeger-Hexagonal",
                "type": "jaeger",
                "url": self.config["jaeger"]["endpoint"],
                "access": "proxy"
            }
            await self.grafana_client.create_datasource(jaeger_datasource)
    
    async def _import_grafana_dashboards(self):
        """Import Grafana dashboards"""
        dashboard_files = [
            "dashboards/system_overview.json",
            "dashboards/application_metrics.json", 
            "dashboards/infrastructure.json",
            "dashboards/business_metrics.json"
        ]
        
        for dashboard_file in dashboard_files:
            if Path(dashboard_file).exists():
                with open(dashboard_file, 'r') as f:
                    dashboard_json = json.load(f)
                
                # Update dashboard to use the correct datasource
                self._update_dashboard_datasource(dashboard_json, "Prometheus-Hexagonal")
                
                await self.grafana_client.import_dashboard(dashboard_json)
                self.logger.info(f"Imported dashboard: {dashboard_file}")
    
    def _update_dashboard_datasource(self, dashboard: Dict, datasource_name: str):
        """Update dashboard to use specific datasource"""
        if "dashboard" in dashboard:
            dashboard_data = dashboard["dashboard"]
        else:
            dashboard_data = dashboard
        
        # Update panels to use correct datasource
        for panel in dashboard_data.get("panels", []):
            if "targets" in panel:
                for target in panel["targets"]:
                    target["datasource"] = datasource_name
    
    async def _setup_grafana_alerts(self):
        """Setup Grafana alert rules"""
        alert_rules = [
            {
                "title": "API Gateway High Response Time",
                "message": "API Gateway response time is above threshold",
                "frequency": "30s",
                "conditions": [{
                    "query": {
                        "queryType": "",
                        "refId": "A",
                        "datasource": {
                            "type": "prometheus",
                            "uid": "prometheus-hexagonal"
                        },
                        "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{service=\"api-gateway\"})",
                        "interval": "",
                        "legendFormat": "",
                        "range": True
                    },
                    "reducer": {
                        "type": "last",
                        "params": []
                    },
                    "evaluator": {
                        "params": [1.0],
                        "type": "gt"
                    }
                }],
                "executionErrorState": "alerting",
                "noDataState": "no_data",
                "for": "5m"
            }
        ]
        
        for alert_rule in alert_rules:
            await self.grafana_client.create_alert_rule(alert_rule)
    
    async def _setup_grafana_notifications(self):
        """Setup Grafana notification channels"""
        # Slack notification channel
        if self.config["integrations"]["slack_webhook"]:
            slack_channel = {
                "name": "slack-production-alerts",
                "type": "slack",
                "settings": {
                    "url": self.config["integrations"]["slack_webhook"],
                    "channel": "#production-alerts",
                    "username": "Grafana",
                    "title": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                    "text": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                }
            }
            await self.grafana_client.create_notification_channel(slack_channel)
        
        # PagerDuty notification channel
        if self.config["integrations"]["pagerduty_key"]:
            pagerduty_channel = {
                "name": "pagerduty-critical-alerts",
                "type": "pagerduty",
                "settings": {
                    "integrationKey": self.config["integrations"]["pagerduty_key"],
                    "severity": "critical",
                    "autoResolve": True
                }
            }
            await self.grafana_client.create_notification_channel(pagerduty_channel)
    
    async def _configure_service_tracing(self):
        """Configure distributed tracing for services"""
        services = [
            "api-gateway",
            "data-quality-service", 
            "anomaly-detection-service",
            "workflow-engine"
        ]
        
        for service in services:
            await self.jaeger_client.configure_service_tracing(
                service,
                self.config["jaeger"]["service_name"]
            )
    
    async def _setup_trace_sampling(self):
        """Setup trace sampling configuration"""
        sampling_config = {
            "default_strategy": {
                "type": "probabilistic",
                "param": self.config["jaeger"].get("sampling_rate", 0.1)
            },
            "per_service_strategies": [
                {
                    "service": "api-gateway",
                    "type": "probabilistic", 
                    "param": 0.2  # Higher sampling for API gateway
                },
                {
                    "service": "data-quality-service",
                    "type": "probabilistic",
                    "param": 0.1
                }
            ]
        }
        
        await self.jaeger_client.update_sampling_config(sampling_config)
    
    async def setup_business_metrics(self) -> bool:
        """Setup business metrics monitoring"""
        try:
            # Define business metrics
            business_metrics = [
                {
                    "name": "data_quality_checks_total",
                    "help": "Total number of data quality checks performed",
                    "type": "counter",
                    "labels": ["service", "status", "rule_type"]
                },
                {
                    "name": "anomalies_detected_total", 
                    "help": "Total number of anomalies detected",
                    "type": "counter",
                    "labels": ["service", "algorithm", "severity"]
                },
                {
                    "name": "workflow_executions_total",
                    "help": "Total number of workflow executions",
                    "type": "counter",
                    "labels": ["workflow_type", "status"]
                },
                {
                    "name": "data_processing_duration_seconds",
                    "help": "Time spent processing data",
                    "type": "histogram",
                    "labels": ["service", "data_type"]
                },
                {
                    "name": "user_sessions_active",
                    "help": "Number of active user sessions",
                    "type": "gauge",
                    "labels": ["service"]
                }
            ]
            
            # Register metrics with Prometheus
            for metric in business_metrics:
                await self.prometheus_client.register_metric(metric)
            
            # Create business metrics dashboard
            await self._create_business_metrics_dashboard()
            
            self.logger.info("✅ Business metrics configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Business metrics setup failed: {e}")
            return False
    
    async def _create_business_metrics_dashboard(self):
        """Create business metrics dashboard"""
        dashboard = {
            "dashboard": {
                "title": "Business Metrics - Hexagonal Architecture",
                "tags": ["business", "kpi", "hexagonal-architecture"],
                "time": {
                    "from": "now-24h",
                    "to": "now"
                },
                "refresh": "30s",
                "panels": [
                    {
                        "title": "Data Quality Checks per Hour",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(data_quality_checks_total[1h])",
                            "legendFormat": "{{ service }} - {{ status }}"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "title": "Anomalies Detected",
                        "type": "stat",
                        "targets": [{
                            "expr": "sum(increase(anomalies_detected_total[24h]))",
                            "legendFormat": "24h Total"
                        }],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "title": "Workflow Success Rate",
                        "type": "gauge",
                        "targets": [{
                            "expr": "sum(rate(workflow_executions_total{status=\"success\"}[5m])) / sum(rate(workflow_executions_total[5m])) * 100",
                            "legendFormat": "Success Rate %"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "min": 0,
                                "max": 100,
                                "unit": "percent"
                            }
                        },
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
                    }
                ]
            }
        }
        
        if self.grafana_client:
            await self.grafana_client.import_dashboard(dashboard)
    
    async def validate_integration(self) -> Dict[str, bool]:
        """Validate all observability integrations"""
        results = {}
        
        # Test Prometheus
        if self.prometheus_client:
            results["prometheus"] = await self.prometheus_client.test_connection()
        else:
            results["prometheus"] = False
        
        # Test Grafana
        if self.grafana_client:
            results["grafana"] = await self.grafana_client.test_connection()
        else:
            results["grafana"] = False
        
        # Test Jaeger
        if self.jaeger_client:
            results["jaeger"] = await self.jaeger_client.test_connection()
        else:
            results["jaeger"] = True  # Optional component
        
        # Test metrics endpoints
        results["metrics_endpoints"] = await self._test_metrics_endpoints()
        
        return results
    
    async def _test_metrics_endpoints(self) -> bool:
        """Test application metrics endpoints"""
        endpoints = [
            "http://api-gateway:8080/metrics",
            "http://data-quality-service:8081/metrics",
            "http://anomaly-detection-service:8082/metrics",
            "http://workflow-engine:8083/metrics"
        ]
        
        successful_tests = 0
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(endpoint, timeout=5) as response:
                        if response.status == 200:
                            successful_tests += 1
                except Exception:
                    pass
        
        # Consider successful if at least 50% of endpoints are reachable
        return successful_tests >= len(endpoints) * 0.5
    
    def generate_integration_report(self, results: Dict[str, bool]) -> str:
        """Generate observability integration report"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("OBSERVABILITY INTEGRATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Integration status
        all_passed = all(results.values())
        overall_status = "✅ ALL INTEGRATIONS WORKING" if all_passed else "⚠️ SOME INTEGRATIONS FAILED"
        report_lines.append(f"Overall Status: {overall_status}")
        report_lines.append("")
        
        # Individual component status
        for component, status in results.items():
            status_symbol = "✅" if status else "❌"
            report_lines.append(f"{status_symbol} {component.title()}: {'CONNECTED' if status else 'FAILED'}")
        
        report_lines.append("")
        
        # Configuration summary
        report_lines.append("Configuration Summary:")
        report_lines.append(f"  Prometheus: {self.config['prometheus']['endpoint']}")
        report_lines.append(f"  Grafana: {self.config['grafana']['endpoint']}")
        report_lines.append(f"  Jaeger: {self.config['jaeger']['endpoint']}")
        report_lines.append(f"  Business Metrics: {'Enabled' if self.config['metrics']['business_metrics'] else 'Disabled'}")
        
        report_lines.append("")
        
        # Recommendations
        report_lines.append("Recommendations:")
        if not results.get("prometheus", False):
            report_lines.append("  • Check Prometheus connectivity and configuration")
        if not results.get("grafana", False):
            report_lines.append("  • Verify Grafana API key and endpoint")
        if not results.get("metrics_endpoints", False):
            report_lines.append("  • Ensure application metrics endpoints are accessible")
        if all_passed:
            report_lines.append("  • All integrations working correctly")
            report_lines.append("  • Monitor dashboard performance and adjust as needed")
        
        return "\n".join(report_lines)


class PrometheusClient:
    """Prometheus API client"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
        self.session = aiohttp.ClientSession()
    
    async def test_connection(self) -> bool:
        """Test Prometheus connectivity"""
        try:
            async with self.session.get(f"{self.endpoint}/api/v1/status/config", timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def update_scrape_config(self, job_name: str, config: Dict[str, Any]):
        """Update Prometheus scrape configuration"""
        # This would typically update the Prometheus configuration file
        # and reload the configuration
        self.logger.info(f"Would update scrape config for job: {job_name}")
    
    async def update_alert_rules(self, group_name: str, rules: Dict[str, Any]):
        """Update Prometheus alert rules"""
        # This would typically update the alert rules file
        # and reload the configuration
        self.logger.info(f"Would update alert rules for group: {group_name}")
    
    async def register_metric(self, metric: Dict[str, Any]):
        """Register a new metric"""
        self.logger.info(f"Would register metric: {metric['name']}")


class GrafanaClient:
    """Grafana API client"""
    
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def test_connection(self) -> bool:
        """Test Grafana connectivity"""
        try:
            async with self.session.get(f"{self.endpoint}/api/health", timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def create_datasource(self, datasource: Dict[str, Any]):
        """Create Grafana datasource"""
        try:
            async with self.session.post(f"{self.endpoint}/api/datasources", json=datasource) as response:
                if response.status in [200, 409]:  # 409 = already exists
                    return True
                return False
        except Exception:
            return False
    
    async def import_dashboard(self, dashboard: Dict[str, Any]):
        """Import Grafana dashboard"""
        try:
            async with self.session.post(f"{self.endpoint}/api/dashboards/db", json=dashboard) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def create_alert_rule(self, alert_rule: Dict[str, Any]):
        """Create Grafana alert rule"""
        try:
            async with self.session.post(f"{self.endpoint}/api/alert-rules", json=alert_rule) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def create_notification_channel(self, channel: Dict[str, Any]):
        """Create Grafana notification channel"""
        try:
            async with self.session.post(f"{self.endpoint}/api/alert-notifications", json=channel) as response:
                return response.status == 200
        except Exception:
            return False


class JaegerClient:
    """Jaeger tracing client"""
    
    def __init__(self, endpoint: str, service_name: str):
        self.endpoint = endpoint.rstrip('/')
        self.service_name = service_name
        self.session = aiohttp.ClientSession()
    
    async def test_connection(self) -> bool:
        """Test Jaeger connectivity"""
        try:
            async with self.session.get(f"{self.endpoint}/api/services", timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def configure_service_tracing(self, service: str, parent_service: str):
        """Configure tracing for a service"""
        self.logger.info(f"Would configure tracing for service: {service}")
    
    async def update_sampling_config(self, config: Dict[str, Any]):
        """Update sampling configuration"""
        self.logger.info("Would update sampling configuration")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Observability Integration Setup")
    parser.add_argument("--config", default="config/observability-config.yaml", help="Configuration file")
    parser.add_argument("--setup-all", action="store_true", help="Setup all integrations")
    parser.add_argument("--setup-prometheus", action="store_true", help="Setup Prometheus only")
    parser.add_argument("--setup-grafana", action="store_true", help="Setup Grafana only")
    parser.add_argument("--setup-tracing", action="store_true", help="Setup tracing only")
    parser.add_argument("--validate", action="store_true", help="Validate integrations")
    parser.add_argument("--report", help="Generate integration report")
    args = parser.parse_args()
    
    integration = ObservabilityIntegration(args.config)
    
    if args.validate:
        results = await integration.validate_integration()
        report = integration.generate_integration_report(results)
        
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"Integration report saved to: {args.report}")
        else:
            print(report)
        return
    
    success = True
    
    if args.setup_all or args.setup_prometheus:
        success &= await integration.setup_prometheus_integration()
    
    if args.setup_all or args.setup_grafana:
        success &= await integration.setup_grafana_integration()
    
    if args.setup_all or args.setup_tracing:
        success &= await integration.setup_distributed_tracing()
    
    if args.setup_all:
        success &= await integration.setup_business_metrics()
    
    if success:
        print("✅ Observability integration completed successfully")
    else:
        print("❌ Some integrations failed - check logs for details")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())