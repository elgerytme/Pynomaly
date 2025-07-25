#!/usr/bin/env python3
"""
Monitoring Dashboard Setup and Configuration
Automated setup for production monitoring dashboards
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import requests


@dataclass
class DashboardPanel:
    """Individual dashboard panel configuration"""
    id: str
    title: str
    type: str  # graph, stat, gauge, table
    query: str
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    thresholds: List[Dict[str, Any]] = None
    position: Dict[str, int] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = []
        if self.position is None:
            self.position = {"x": 0, "y": 0, "w": 12, "h": 8}


@dataclass
class Dashboard:
    """Complete dashboard configuration"""
    id: str
    title: str
    description: str
    tags: List[str]
    panels: List[DashboardPanel]
    refresh_interval: str = "30s"
    time_range: Dict[str, str] = None
    
    def __post_init__(self):
        if self.time_range is None:
            self.time_range = {"from": "now-1h", "to": "now"}


class MonitoringDashboardSetup:
    """Main dashboard setup and configuration system"""
    
    def __init__(self, config_path: str = "config/monitoring-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_dashboards()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration if file doesn't exist
        return {
            "dashboards": {
                "system_overview": {"refresh_interval": 30},
                "application_metrics": {"refresh_interval": 30},
                "infrastructure": {"refresh_interval": 60}
            },
            "integrations": {
                "grafana": {
                    "enabled": True,
                    "endpoint": "http://localhost:3000",
                    "api_key": ""
                },
                "prometheus": {
                    "enabled": True,
                    "endpoint": "http://localhost:9090"
                }
            }
        }
    
    def _initialize_dashboards(self):
        """Initialize all monitoring dashboards"""
        
        # System Overview Dashboard
        self.dashboards["system_overview"] = Dashboard(
            id="system-overview",
            title="System Overview",
            description="High-level system health and performance metrics",
            tags=["system", "overview", "production"],
            panels=[
                DashboardPanel(
                    id="cpu-usage",
                    title="CPU Usage",
                    type="gauge",
                    query='avg(rate(cpu_usage_total[5m])) * 100',
                    unit="percent",
                    min_value=0,
                    max_value=100,
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 70},
                        {"color": "red", "value": 90}
                    ],
                    position={"x": 0, "y": 0, "w": 6, "h": 8}
                ),
                DashboardPanel(
                    id="memory-usage",
                    title="Memory Usage",
                    type="gauge",
                    query='(1 - (avg(memory_available_bytes) / avg(memory_total_bytes))) * 100',
                    unit="percent",
                    min_value=0,
                    max_value=100,
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 80},
                        {"color": "red", "value": 95}
                    ],
                    position={"x": 6, "y": 0, "w": 6, "h": 8}
                ),
                DashboardPanel(
                    id="disk-usage",
                    title="Disk Usage",
                    type="gauge",
                    query='(1 - (avg(disk_free_bytes) / avg(disk_total_bytes))) * 100',
                    unit="percent",
                    min_value=0,
                    max_value=100,
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 85},
                        {"color": "red", "value": 95}
                    ],
                    position={"x": 12, "y": 0, "w": 6, "h": 8}
                ),
                DashboardPanel(
                    id="network-io",
                    title="Network I/O",
                    type="graph",
                    query='rate(network_bytes_total[5m])',
                    unit="bytes/sec",
                    position={"x": 18, "y": 0, "w": 6, "h": 8}
                ),
                DashboardPanel(
                    id="system-load",
                    title="System Load Average",  
                    type="graph",
                    query='avg(system_load_average)',
                    unit="",
                    position={"x": 0, "y": 8, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="uptime",
                    title="System Uptime",
                    type="stat",
                    query='time() - system_boot_time',
                    unit="s",
                    position={"x": 12, "y": 8, "w": 12, "h": 4}
                )
            ]
        )
        
        # Application Metrics Dashboard
        self.dashboards["application_metrics"] = Dashboard(
            id="application-metrics",
            title="Application Metrics",
            description="Application performance and business metrics",
            tags=["application", "performance", "business"],
            panels=[
                DashboardPanel(
                    id="response-time-p95",
                    title="Response Time P95",
                    type="graph",
                    query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
                    unit="s",
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5}
                    ],
                    position={"x": 0, "y": 0, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="response-time-p99",
                    title="Response Time P99",
                    type="graph",
                    query='histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))',
                    unit="s",
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 2},
                        {"color": "red", "value": 10}
                    ],
                    position={"x": 12, "y": 0, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="requests-per-second",
                    title="Requests per Second",
                    type="graph",
                    query='sum(rate(http_requests_total[5m]))',
                    unit="reqps",
                    position={"x": 0, "y": 8, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="error-rate",
                    title="Error Rate",
                    type="graph",
                    query='sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100',
                    unit="percent",
                    thresholds=[
                        {"color": "green", "value": 0},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5}
                    ],
                    position={"x": 12, "y": 8, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="active-connections",
                    title="Active Connections",
                    type="stat",
                    query='sum(active_connections)',
                    unit="",
                    position={"x": 0, "y": 16, "w": 6, "h": 4}
                ),
                DashboardPanel(
                    id="queue-depth",
                    title="Queue Depth",
                    type="stat",
                    query='sum(queue_depth)',
                    unit="",
                    position={"x": 6, "y": 16, "w": 6, "h": 4}
                ),
                DashboardPanel(
                    id="cache-hit-rate",
                    title="Cache Hit Rate",
                    type="gauge",
                    query='sum(cache_hits) / (sum(cache_hits) + sum(cache_misses)) * 100',
                    unit="percent",
                    min_value=0,
                    max_value=100,
                    thresholds=[
                        {"color": "red", "value": 0},
                        {"color": "yellow", "value": 80},
                        {"color": "green", "value": 95}
                    ],
                    position={"x": 12, "y": 16, "w": 6, "h": 4}
                ),
                DashboardPanel(
                    id="database-connections",
                    title="Database Connections",
                    type="stat",
                    query='sum(database_connections_active)',
                    unit="",
                    position={"x": 18, "y": 16, "w": 6, "h": 4}
                )
            ]
        )
        
        # Infrastructure Dashboard
        self.dashboards["infrastructure"] = Dashboard(
            id="infrastructure",
            title="Infrastructure",
            description="Kubernetes and infrastructure metrics",
            tags=["infrastructure", "kubernetes", "containers"],
            panels=[
                DashboardPanel(
                    id="kubernetes-pods",
                    title="Pod Status",
                    type="table",
                    query='kube_pod_info',
                    position={"x": 0, "y": 0, "w": 24, "h": 8}
                ),
                DashboardPanel(
                    id="kubernetes-nodes",
                    title="Node Status",
                    type="table",
                    query='kube_node_info',
                    position={"x": 0, "y": 8, "w": 24, "h": 8}
                ),
                DashboardPanel(
                    id="pod-restarts",
                    title="Pod Restarts (Last 24h)",
                    type="graph",
                    query='increase(kube_pod_container_status_restarts_total[24h])',
                    unit="",
                    position={"x": 0, "y": 16, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="container-cpu",
                    title="Container CPU Usage",
                    type="graph",
                    query='sum(rate(container_cpu_usage_seconds_total[5m])) by (pod)',
                    unit="cores",
                    position={"x": 12, "y": 16, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="container-memory",
                    title="Container Memory Usage",
                    type="graph",
                    query='sum(container_memory_usage_bytes) by (pod)',
                    unit="bytes",
                    position={"x": 0, "y": 24, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="persistent-volumes",
                    title="Persistent Volume Usage",
                    type="graph",
                    query='(kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes) * 100',
                    unit="percent",
                    position={"x": 12, "y": 24, "w": 12, "h": 8}
                )
            ]
        )
        
        # Business Metrics Dashboard
        self.dashboards["business_metrics"] = Dashboard(
            id="business-metrics",
            title="Business Metrics",
            description="Key business performance indicators",
            tags=["business", "kpi", "metrics"],
            panels=[
                DashboardPanel(
                    id="data-quality-checks",
                    title="Data Quality Checks per Hour",
                    type="graph",
                    query='sum(rate(data_quality_checks_total[1h]))',
                    unit="checks/h",
                    position={"x": 0, "y": 0, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="anomaly-detections",
                    title="Anomalies Detected per Hour",
                    type="graph",
                    query='sum(rate(anomalies_detected_total[1h]))',
                    unit="anomalies/h",
                    position={"x": 12, "y": 0, "w": 12, "h": 8}
                ),
                DashboardPanel(
                    id="workflow-executions",
                    title="Workflow Executions",
                    type="stat",
                    query='sum(increase(workflow_executions_total[24h]))',
                    unit="",
                    position={"x": 0, "y": 8, "w": 6, "h": 4}
                ),
                DashboardPanel(
                    id="data-processed",
                    title="Data Processed (GB/day)",
                    type="stat",
                    query='sum(increase(data_bytes_processed_total[24h])) / 1024 / 1024 / 1024',
                    unit="GB",
                    position={"x": 6, "y": 8, "w": 6, "h": 4}
                ),
                DashboardPanel(
                    id="user-sessions",
                    title="Active User Sessions",
                    type="stat",
                    query='sum(active_user_sessions)',
                    unit="",
                    position={"x": 12, "y": 8, "w": 6, "h": 4}
                ),
                DashboardPanel(
                    id="api-calls",
                    title="API Calls per Minute",
                    type="graph",
                    query='sum(rate(api_calls_total[1m]))',
                    unit="calls/min",
                    position={"x": 18, "y": 8, "w": 6, "h": 4}
                )
            ]
        )
    
    def generate_grafana_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Generate Grafana dashboard JSON"""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        # Convert panels to Grafana format
        grafana_panels = []
        for i, panel in enumerate(dashboard.panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel.title,
                "type": panel.type,
                "gridPos": panel.position,
                "targets": [{
                    "expr": panel.query,
                    "refId": "A"
                }],
                "fieldConfig": {
                    "defaults": {
                        "unit": panel.unit,
                        "thresholds": {
                            "steps": panel.thresholds
                        }
                    }
                }
            }
            
            if panel.min_value is not None:
                grafana_panel["fieldConfig"]["defaults"]["min"] = panel.min_value
            if panel.max_value is not None:
                grafana_panel["fieldConfig"]["defaults"]["max"] = panel.max_value
            
            grafana_panels.append(grafana_panel)
        
        # Generate complete dashboard JSON
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags,
                "timezone": "UTC",
                "panels": grafana_panels,
                "time": dashboard.time_range,
                "refresh": dashboard.refresh_interval,
                "schemaVersion": 30,
                "version": 1,
                "links": [],
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                }
            },
            "overwrite": True
        }
        
        return grafana_dashboard
    
    def setup_grafana_dashboards(self, grafana_url: str = None, api_key: str = None):
        """Set up Grafana dashboards"""
        grafana_config = self.config.get("integrations", {}).get("grafana", {})
        
        if not grafana_config.get("enabled", True):
            self.logger.info("Grafana integration disabled, skipping dashboard setup")
            return
        
        grafana_url = grafana_url or grafana_config.get("endpoint", "http://localhost:3000")
        api_key = api_key or grafana_config.get("api_key", os.getenv("GRAFANA_API_KEY", ""))
        
        if not api_key:
            self.logger.warning("No Grafana API key provided, cannot create dashboards")
            return
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for dashboard_id, dashboard in self.dashboards.items():
            try:
                self.logger.info(f"Creating Grafana dashboard: {dashboard.title}")
                
                # Generate dashboard JSON
                dashboard_json = self.generate_grafana_dashboard(dashboard_id)
                
                # Create dashboard via API
                response = requests.post(
                    f"{grafana_url}/api/dashboards/db",
                    headers=headers,
                    json=dashboard_json,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"Dashboard created successfully: {result.get('url', 'Unknown URL')}")
                else:
                    self.logger.error(f"Failed to create dashboard: {response.status_code} - {response.text}")
                    
            except Exception as e:
                self.logger.error(f"Error creating dashboard {dashboard_id}: {e}")
    
    def export_dashboards(self, output_dir: str = "dashboards"):
        """Export all dashboards to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for dashboard_id, dashboard in self.dashboards.items():
            try:
                # Generate dashboard JSON
                dashboard_json = self.generate_grafana_dashboard(dashboard_id)
                
                # Write to file
                output_file = output_path / f"{dashboard_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(dashboard_json, f, indent=2)
                
                self.logger.info(f"Exported dashboard to: {output_file}")
                
            except Exception as e:
                self.logger.error(f"Error exporting dashboard {dashboard_id}: {e}")
    
    def setup_prometheus_monitoring(self, prometheus_url: str = None):
        """Set up Prometheus monitoring configuration"""
        prometheus_config = self.config.get("integrations", {}).get("prometheus", {})
        
        if not prometheus_config.get("enabled", True):
            self.logger.info("Prometheus integration disabled, skipping setup")
            return
        
        prometheus_url = prometheus_url or prometheus_config.get("endpoint", "http://localhost:9090")
        
        # Generate Prometheus configuration
        prometheus_config_yaml = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "hexagonal-architecture",
                    "static_configs": [{
                        "targets": [
                            f"api-gateway:8080",
                            f"data-quality-service:8081",
                            f"anomaly-detection-service:8082",
                            f"workflow-engine:8083"
                        ]
                    }],
                    "metrics_path": "/metrics",
                    "scrape_interval": "30s"
                },
                {
                    "job_name": "kubernetes-pods",
                    "kubernetes_sd_configs": [{
                        "role": "pod"
                    }],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                            "action": "keep",
                            "regex": "true"
                        }
                    ]
                },
                {
                    "job_name": "kubernetes-nodes",
                    "kubernetes_sd_configs": [{
                        "role": "node"
                    }]
                }
            ],
            "rule_files": [
                "alert_rules.yml"
            ],
            "alerting": {
                "alertmanagers": [{
                    "static_configs": [{
                        "targets": ["alertmanager:9093"]
                    }]
                }]
            }
        }
        
        # Write Prometheus configuration
        with open("prometheus.yml", 'w') as f:
            yaml.dump(prometheus_config_yaml, f, default_flow_style=False)
        
        self.logger.info("Prometheus configuration generated: prometheus.yml")
        
        # Generate alert rules
        self._generate_alert_rules()
    
    def _generate_alert_rules(self):
        """Generate Prometheus alert rules"""
        alert_rules = {
            "groups": [
                {
                    "name": "system_alerts",
                    "rules": [
                        {
                            "alert": "HighCPUUsage",
                            "expr": "avg(rate(cpu_usage_total[5m])) * 100 > 90",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High CPU usage detected",
                                "description": "CPU usage is above 90% for more than 5 minutes"
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "(1 - (avg(memory_available_bytes) / avg(memory_total_bytes))) * 100 > 95",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High memory usage detected",
                                "description": "Memory usage is above 95% for more than 5 minutes"
                            }
                        },
                        {
                            "alert": "HighDiskUsage",
                            "expr": "(1 - (avg(disk_free_bytes) / avg(disk_total_bytes))) * 100 > 95",
                            "for": "10m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High disk usage detected",
                                "description": "Disk usage is above 95% for more than 10 minutes"
                            }
                        }
                    ]
                },
                {
                    "name": "application_alerts",
                    "rules": [
                        {
                            "alert": "HighResponseTime",
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High response time detected",
                                "description": "95th percentile response time is above 1 second"
                            }
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100 > 5",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High error rate detected",
                                "description": "Error rate is above 5% for more than 5 minutes"
                            }
                        },
                        {
                            "alert": "ServiceDown",
                            "expr": "up == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Service is down",
                                "description": "Service {{ $labels.instance }} is down"
                            }
                        }
                    ]
                },
                {
                    "name": "kubernetes_alerts",
                    "rules": [
                        {
                            "alert": "PodCrashLooping",
                            "expr": "increase(kube_pod_container_status_restarts_total[1h]) > 5",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Pod is crash looping",
                                "description": "Pod {{ $labels.pod }} has restarted more than 5 times in the last hour"
                            }
                        },
                        {
                            "alert": "NodeNotReady",
                            "expr": "kube_node_status_condition{condition=\"Ready\",status=\"true\"} == 0",
                            "for": "10m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Kubernetes node not ready",
                                "description": "Node {{ $labels.node }} has been not ready for more than 10 minutes"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Write alert rules
        with open("alert_rules.yml", 'w') as f:
            yaml.dump(alert_rules, f, default_flow_style=False)
        
        self.logger.info("Alert rules generated: alert_rules.yml")
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate monitoring setup"""
        results = {}
        
        # Check Grafana connectivity
        try:
            grafana_config = self.config.get("integrations", {}).get("grafana", {})
            grafana_url = grafana_config.get("endpoint", "http://localhost:3000")
            
            response = requests.get(f"{grafana_url}/api/health", timeout=10)
            results["grafana_connectivity"] = response.status_code == 200
        except Exception:
            results["grafana_connectivity"] = False
        
        # Check Prometheus connectivity
        try:
            prometheus_config = self.config.get("integrations", {}).get("prometheus", {})
            prometheus_url = prometheus_config.get("endpoint", "http://localhost:9090")
            
            response = requests.get(f"{prometheus_url}/api/v1/status/config", timeout=10)
            results["prometheus_connectivity"] = response.status_code == 200
        except Exception:
            results["prometheus_connectivity"] = False
        
        # Check dashboard files exist
        results["dashboard_files_exist"] = all(
            Path(f"dashboards/{dashboard_id}.json").exists()
            for dashboard_id in self.dashboards.keys()
        )
        
        # Check configuration files exist
        results["prometheus_config_exists"] = Path("prometheus.yml").exists()
        results["alert_rules_exist"] = Path("alert_rules.yml").exists()
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitoring Dashboard Setup")
    parser.add_argument("--config", default="config/monitoring-config.yaml", help="Configuration file")
    parser.add_argument("--export-only", action="store_true", help="Only export dashboards, don't create in Grafana")
    parser.add_argument("--grafana-url", help="Grafana URL")
    parser.add_argument("--grafana-api-key", help="Grafana API key")
    parser.add_argument("--prometheus-url", help="Prometheus URL")
    parser.add_argument("--validate", action="store_true", help="Validate setup only")
    args = parser.parse_args()
    
    setup = MonitoringDashboardSetup(args.config)
    
    if args.validate:
        results = setup.validate_setup()
        print("Validation Results:")
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
        return
    
    # Export dashboards
    setup.export_dashboards()
    
    # Set up Prometheus configuration
    setup.setup_prometheus_monitoring(args.prometheus_url)
    
    if not args.export_only:
        # Set up Grafana dashboards
        setup.setup_grafana_dashboards(args.grafana_url, args.grafana_api_key)
    
    print("✅ Monitoring dashboard setup completed!")
    print("📊 Dashboards exported to: dashboards/")
    print("⚙️ Prometheus config generated: prometheus.yml")
    print("🚨 Alert rules generated: alert_rules.yml")


if __name__ == "__main__":
    main()