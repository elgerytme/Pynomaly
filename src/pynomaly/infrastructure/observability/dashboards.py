"""Monitoring dashboards and alerting configuration."""

import json
from typing import Any, Dict, List, Optional


class GrafanaDashboard:
    """Grafana dashboard generator for Pynomaly metrics."""
    
    def __init__(self, title: str = "Pynomaly Monitoring"):
        """Initialize dashboard generator.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.panels = []
        self.panel_id = 1
    
    def add_panel(self, panel_config: Dict[str, Any]) -> None:
        """Add a panel to the dashboard.
        
        Args:
            panel_config: Panel configuration
        """
        panel_config["id"] = self.panel_id
        self.panels.append(panel_config)
        self.panel_id += 1
    
    def create_system_overview_dashboard(self) -> Dict[str, Any]:
        """Create system overview dashboard.
        
        Returns:
            Grafana dashboard configuration
        """
        # HTTP Requests panel
        self.add_panel({
            "title": "HTTP Requests Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(pynomaly_http_requests_total[5m])",
                    "legendFormat": "{{method}} {{endpoint}} - {{status_code}}"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "yAxes": [
                {"label": "Requests/sec", "min": 0},
                {"show": False}
            ]
        })
        
        # Response Time panel
        self.add_panel({
            "title": "HTTP Response Time",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(pynomaly_http_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                },
                {
                    "expr": "histogram_quantile(0.5, rate(pynomaly_http_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "yAxes": [
                {"label": "Seconds", "min": 0},
                {"show": False}
            ]
        })
        
        # System Resources panel
        self.add_panel({
            "title": "System Resources",
            "type": "graph",
            "targets": [
                {
                    "expr": "pynomaly_cpu_usage_percent",
                    "legendFormat": "CPU Usage %"
                },
                {
                    "expr": "pynomaly_memory_usage_bytes / 1024 / 1024 / 1024",
                    "legendFormat": "Memory Usage (GB)"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "yAxes": [
                {"label": "Percent / GB", "min": 0},
                {"show": False}
            ]
        })
        
        # Active Users and Detectors
        self.add_panel({
            "title": "Active Resources",
            "type": "stat",
            "targets": [
                {
                    "expr": "pynomaly_active_users",
                    "legendFormat": "Active Users"
                },
                {
                    "expr": "pynomaly_active_detectors",
                    "legendFormat": "Active Detectors"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
        })
        
        return self._build_dashboard()
    
    def create_anomaly_detection_dashboard(self) -> Dict[str, Any]:
        """Create anomaly detection specific dashboard.
        
        Returns:
            Grafana dashboard configuration
        """
        # Detection Rate panel
        self.add_panel({
            "title": "Anomaly Detection Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(pynomaly_detections_total[5m])",
                    "legendFormat": "{{detector_type}} - {{dataset_name}}"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "yAxes": [
                {"label": "Detections/sec", "min": 0},
                {"show": False}
            ]
        })
        
        # Detection Duration panel
        self.add_panel({
            "title": "Detection Duration",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(pynomaly_detection_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                },
                {
                    "expr": "histogram_quantile(0.5, rate(pynomaly_detection_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "yAxes": [
                {"label": "Seconds", "min": 0},
                {"show": False}
            ]
        })
        
        # Anomalies Found Distribution
        self.add_panel({
            "title": "Anomalies Found Distribution",
            "type": "heatmap",
            "targets": [
                {
                    "expr": "rate(pynomaly_anomalies_found_bucket[5m])",
                    "format": "heatmap",
                    "legendFormat": "{{le}}"
                }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
        })
        
        # Training Metrics
        self.add_panel({
            "title": "Model Training Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(pynomaly_model_training_total[5m])",
                    "legendFormat": "{{detector_type}} - {{algorithm}}"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
            "yAxes": [
                {"label": "Trainings/sec", "min": 0},
                {"show": False}
            ]
        })
        
        # Training Duration
        self.add_panel({
            "title": "Training Duration",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(pynomaly_training_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                },
                {
                    "expr": "histogram_quantile(0.5, rate(pynomaly_training_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
            "yAxes": [
                {"label": "Seconds", "min": 0},
                {"show": False}
            ]
        })
        
        return self._build_dashboard()
    
    def create_infrastructure_dashboard(self) -> Dict[str, Any]:
        """Create infrastructure monitoring dashboard.
        
        Returns:
            Grafana dashboard configuration
        """
        # Database Connections
        self.add_panel({
            "title": "Database Connections",
            "type": "graph",
            "targets": [
                {
                    "expr": "pynomaly_db_connections_active",
                    "legendFormat": "Active Connections"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "yAxes": [
                {"label": "Connections", "min": 0},
                {"show": False}
            ]
        })
        
        # Database Query Duration
        self.add_panel({
            "title": "Database Query Duration",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(pynomaly_db_query_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                },
                {
                    "expr": "histogram_quantile(0.5, rate(pynomaly_db_query_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "yAxes": [
                {"label": "Seconds", "min": 0},
                {"show": False}
            ]
        })
        
        # Cache Hit Rate
        self.add_panel({
            "title": "Cache Hit Rate",
            "type": "stat",
            "targets": [
                {
                    "expr": "rate(pynomaly_cache_hits_total[5m]) / (rate(pynomaly_cache_hits_total[5m]) + rate(pynomaly_cache_misses_total[5m])) * 100",
                    "legendFormat": "{{cache_type}} Hit Rate %"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100
                }
            }
        })
        
        # Error Rate
        self.add_panel({
            "title": "Error Rate by Component",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(pynomaly_errors_total[5m])",
                    "legendFormat": "{{error_type}} - {{component}}"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            "yAxes": [
                {"label": "Errors/sec", "min": 0},
                {"show": False}
            ]
        })
        
        return self._build_dashboard()
    
    def _build_dashboard(self) -> Dict[str, Any]:
        """Build the complete dashboard configuration.
        
        Returns:
            Complete Grafana dashboard JSON
        """
        return {
            "dashboard": {
                "id": None,
                "title": self.title,
                "tags": ["pynomaly", "monitoring"],
                "timezone": "browser",
                "panels": self.panels,
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
                },
                "refresh": "30s",
                "schemaVersion": 30,
                "version": 1,
                "links": []
            },
            "folderId": 0,
            "overwrite": True
        }


class AlertManager:
    """Alert configuration manager for Prometheus AlertManager."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_rules = []
    
    def add_alert_rule(self, rule_config: Dict[str, Any]) -> None:
        """Add an alert rule.
        
        Args:
            rule_config: Alert rule configuration
        """
        self.alert_rules.append(rule_config)
    
    def create_system_alerts(self) -> Dict[str, Any]:
        """Create system-level alert rules.
        
        Returns:
            Prometheus alert rules configuration
        """
        # High error rate alert
        self.add_alert_rule({
            "alert": "HighErrorRate",
            "expr": "rate(pynomaly_errors_total[5m]) > 0.1",
            "for": "2m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "High error rate detected",
                "description": "Error rate is {{ $value }} errors per second for {{ $labels.component }}"
            }
        })
        
        # High response time alert
        self.add_alert_rule({
            "alert": "HighResponseTime",
            "expr": "histogram_quantile(0.95, rate(pynomaly_http_request_duration_seconds_bucket[5m])) > 1",
            "for": "5m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "High HTTP response time",
                "description": "95th percentile response time is {{ $value }}s"
            }
        })
        
        # High memory usage alert
        self.add_alert_rule({
            "alert": "HighMemoryUsage",
            "expr": "pynomaly_memory_usage_bytes / 1024 / 1024 / 1024 > 8",
            "for": "10m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "High memory usage",
                "description": "Memory usage is {{ $value }}GB"
            }
        })
        
        # Database connectivity alert
        self.add_alert_rule({
            "alert": "DatabaseDown",
            "expr": "up{job=\"pynomaly-db\"} == 0",
            "for": "1m",
            "labels": {
                "severity": "critical"
            },
            "annotations": {
                "summary": "Database is down",
                "description": "Database connection is unavailable"
            }
        })
        
        # Low cache hit rate alert
        self.add_alert_rule({
            "alert": "LowCacheHitRate",
            "expr": "rate(pynomaly_cache_hits_total[10m]) / (rate(pynomaly_cache_hits_total[10m]) + rate(pynomaly_cache_misses_total[10m])) < 0.8",
            "for": "15m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "Low cache hit rate",
                "description": "Cache hit rate is {{ $value | humanizePercentage }} for {{ $labels.cache_type }}"
            }
        })
        
        return self._build_alert_config()
    
    def create_anomaly_detection_alerts(self) -> Dict[str, Any]:
        """Create anomaly detection specific alerts.
        
        Returns:
            Prometheus alert rules configuration
        """
        # High detection failure rate
        self.add_alert_rule({
            "alert": "HighDetectionFailureRate",
            "expr": "rate(pynomaly_errors_total{component=\"detection\"}[5m]) > 0.05",
            "for": "3m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "High detection failure rate",
                "description": "Detection failure rate is {{ $value }} failures per second"
            }
        })
        
        # Long detection duration
        self.add_alert_rule({
            "alert": "LongDetectionDuration",
            "expr": "histogram_quantile(0.95, rate(pynomaly_detection_duration_seconds_bucket[10m])) > 60",
            "for": "5m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "Long detection duration",
                "description": "95th percentile detection duration is {{ $value }}s"
            }
        })
        
        # No detections for extended period
        self.add_alert_rule({
            "alert": "NoDetectionsRunning",
            "expr": "rate(pynomaly_detections_total[30m]) == 0",
            "for": "30m",
            "labels": {
                "severity": "warning"
            },
            "annotations": {
                "summary": "No anomaly detections running",
                "description": "No anomaly detections have been performed in the last 30 minutes"
            }
        })
        
        return self._build_alert_config()
    
    def _build_alert_config(self) -> Dict[str, Any]:
        """Build complete alert configuration.
        
        Returns:
            Prometheus alert rules configuration
        """
        return {
            "groups": [
                {
                    "name": "pynomaly-alerts",
                    "rules": self.alert_rules
                }
            ]
        }


def generate_monitoring_config() -> Dict[str, Any]:
    """Generate complete monitoring configuration.
    
    Returns:
        Complete monitoring setup configuration
    """
    # Generate dashboards
    dashboard_generator = GrafanaDashboard()
    
    dashboards = {
        "system_overview": dashboard_generator.create_system_overview_dashboard(),
        "anomaly_detection": dashboard_generator.create_anomaly_detection_dashboard(),
        "infrastructure": dashboard_generator.create_infrastructure_dashboard()
    }
    
    # Generate alerts
    alert_manager = AlertManager()
    
    alerts = {
        "system_alerts": alert_manager.create_system_alerts(),
        "detection_alerts": alert_manager.create_anomaly_detection_alerts()
    }
    
    # Prometheus configuration
    prometheus_config = {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s"
        },
        "rule_files": [
            "alerts/*.yml"
        ],
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
                "job_name": "pynomaly-db",
                "static_configs": [
                    {
                        "targets": ["localhost:5432"]
                    }
                ],
                "scrape_interval": "30s"
            }
        ],
        "alerting": {
            "alertmanagers": [
                {
                    "static_configs": [
                        {
                            "targets": ["localhost:9093"]
                        }
                    ]
                }
            ]
        }
    }
    
    return {
        "dashboards": dashboards,
        "alerts": alerts,
        "prometheus": prometheus_config
    }