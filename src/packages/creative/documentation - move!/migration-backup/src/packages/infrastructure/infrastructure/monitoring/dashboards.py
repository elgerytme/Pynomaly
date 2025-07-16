"""Grafana dashboard configurations for Pynomaly monitoring.

This module provides pre-configured Grafana dashboards for monitoring
Pynomaly's performance, health, and business metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Dashboard configuration templates
PYNOMALY_DASHBOARD_TEMPLATE = {
    "dashboard": {
        "id": None,
        "title": "Pynomaly - Anomaly Detection System",
        "tags": ["monorepo", "anomaly-detection", "ml"],
        "style": "dark",
        "timezone": "browser",
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {
            "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"],
            "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"],
        },
        "panels": [],
    }
}

OVERVIEW_PANELS = [
    {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        "targets": [
            {
                "expr": "pynomaly_active_models",
                "legendFormat": "Active Models",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 10},
                        {"color": "red", "value": 50},
                    ]
                },
            }
        },
    },
    {
        "id": 2,
        "title": "HTTP Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
            {
                "expr": "rate(pynomaly_http_requests_total[5m])",
                "legendFormat": "{{method}} {{endpoint}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Requests/sec", "min": 0}, {"show": False}],
    },
    {
        "id": 3,
        "title": "HTTP Response Times",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
        "targets": [
            {
                "expr": "histogram_quantile(0.50, rate(pynomaly_http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "P50",
                "refId": "A",
            },
            {
                "expr": "histogram_quantile(0.95, rate(pynomaly_http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "P95",
                "refId": "B",
            },
            {
                "expr": "histogram_quantile(0.99, rate(pynomaly_http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "P99",
                "refId": "C",
            },
        ],
        "yAxes": [{"label": "Duration (s)", "min": 0}, {"show": False}],
    },
]

DETECTION_PANELS = [
    {
        "id": 10,
        "title": "Detection Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
        "targets": [
            {
                "expr": "rate(pynomaly_detections_total[5m])",
                "legendFormat": "{{algorithm}} - {{status}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Detections/sec", "min": 0}, {"show": False}],
    },
    {
        "id": 11,
        "title": "Detection Duration",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
        "targets": [
            {
                "expr": "histogram_quantile(0.95, rate(pynomaly_detection_duration_seconds_bucket[5m]))",
                "legendFormat": "P95 - {{algorithm}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Duration (s)", "min": 0}, {"show": False}],
    },
    {
        "id": 12,
        "title": "Anomalies Found",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
        "targets": [
            {
                "expr": "increase(pynomaly_anomalies_found_total[1h])",
                "legendFormat": "Last Hour",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 100},
                        {"color": "red", "value": 500},
                    ]
                },
            }
        },
    },
    {
        "id": 13,
        "title": "Detection Accuracy",
        "type": "gauge",
        "gridPos": {"h": 6, "w": 6, "x": 18, "y": 0},
        "targets": [
            {
                "expr": "avg(pynomaly_detection_accuracy_ratio)",
                "legendFormat": "Average Accuracy",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "min": 0,
                "max": 1,
                "unit": "percentunit",
                "thresholds": {
                    "steps": [
                        {"color": "red", "value": 0},
                        {"color": "yellow", "value": 0.7},
                        {"color": "green", "value": 0.85},
                    ]
                },
            }
        },
    },
]

STREAMING_PANELS = [
    {
        "id": 20,
        "title": "Streaming Throughput",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
        "targets": [
            {
                "expr": "pynomaly_streaming_throughput_per_second",
                "legendFormat": "{{stream_id}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Samples/sec", "min": 0}, {"show": False}],
    },
    {
        "id": 21,
        "title": "Buffer Utilization",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 20},
        "targets": [
            {
                "expr": "pynomaly_streaming_buffer_utilization_ratio",
                "legendFormat": "{{stream_id}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Utilization %", "min": 0, "max": 1}, {"show": False}],
    },
    {
        "id": 22,
        "title": "Backpressure Events",
        "type": "graph",
        "gridPos": {"h": 6, "w": 12, "x": 0, "y": 28},
        "targets": [
            {
                "expr": "rate(pynomaly_streaming_backpressure_events_total[5m])",
                "legendFormat": "{{stream_id}} - {{strategy}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Events/sec", "min": 0}, {"show": False}],
    },
    {
        "id": 23,
        "title": "Active Streams",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
        "targets": [
            {
                "expr": "pynomaly_active_streams",
                "legendFormat": "Active Streams",
                "refId": "A",
            }
        ],
    },
]

ENSEMBLE_PANELS = [
    {
        "id": 30,
        "title": "Ensemble Predictions",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 34},
        "targets": [
            {
                "expr": "rate(pynomaly_ensemble_predictions_total[5m])",
                "legendFormat": "{{voting_strategy}} ({{detector_count}} detectors)",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Predictions/sec", "min": 0}, {"show": False}],
    },
    {
        "id": 31,
        "title": "Ensemble Agreement",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 34},
        "targets": [
            {
                "expr": "histogram_quantile(0.95, rate(pynomaly_ensemble_agreement_ratio_bucket[5m]))",
                "legendFormat": "P95 Agreement - {{voting_strategy}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Agreement Ratio", "min": 0, "max": 1}, {"show": False}],
    },
]

SYSTEM_PANELS = [
    {
        "id": 40,
        "title": "Memory Usage",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 42},
        "targets": [
            {
                "expr": "pynomaly_memory_usage_bytes",
                "legendFormat": "{{component}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Bytes", "min": 0}, {"show": False}],
    },
    {
        "id": 41,
        "title": "CPU Usage",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 42},
        "targets": [
            {
                "expr": "pynomaly_cpu_usage_ratio",
                "legendFormat": "{{component}}",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "CPU %", "min": 0, "max": 1}, {"show": False}],
    },
    {
        "id": 42,
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"h": 6, "w": 12, "x": 0, "y": 50},
        "targets": [
            {
                "expr": "rate(pynomaly_errors_total[5m])",
                "legendFormat": "{{error_type}} - {{component}} ({{severity}})",
                "refId": "A",
            }
        ],
        "yAxes": [{"label": "Errors/sec", "min": 0}, {"show": False}],
    },
    {
        "id": 43,
        "title": "Cache Hit Ratio",
        "type": "gauge",
        "gridPos": {"h": 6, "w": 12, "x": 12, "y": 50},
        "targets": [
            {
                "expr": "pynomaly_cache_hit_ratio",
                "legendFormat": "{{cache_type}}",
                "refId": "A",
            }
        ],
        "fieldConfig": {
            "defaults": {
                "min": 0,
                "max": 1,
                "unit": "percentunit",
                "thresholds": {
                    "steps": [
                        {"color": "red", "value": 0},
                        {"color": "yellow", "value": 0.8},
                        {"color": "green", "value": 0.95},
                    ]
                },
            }
        },
    },
]


@dataclass
class DashboardConfig:
    """Configuration for Grafana dashboard."""

    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    refresh_interval: str = "30s"
    time_range: str = "1h"
    panels: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        """Convert dashboard config to Grafana JSON format."""
        dashboard = PYNOMALY_DASHBOARD_TEMPLATE.copy()
        dashboard["dashboard"]["title"] = self.title
        dashboard["dashboard"]["description"] = self.description
        dashboard["dashboard"]["tags"] = self.tags
        dashboard["dashboard"]["refresh"] = self.refresh_interval
        dashboard["dashboard"]["time"]["from"] = f"now-{self.time_range}"
        dashboard["dashboard"]["panels"] = self.panels

        return json.dumps(dashboard, indent=2)


class DashboardGenerator:
    """Generator for Pynomaly Grafana dashboards."""

    @staticmethod
    def create_overview_dashboard() -> DashboardConfig:
        """Create main overview dashboard."""
        return DashboardConfig(
            title="Pynomaly - System Overview",
            description="High-level overview of Pynomaly anomaly detection system",
            tags=["monorepo", "overview", "monitoring"],
            panels=OVERVIEW_PANELS + DETECTION_PANELS[:2] + SYSTEM_PANELS[:2],
        )

    @staticmethod
    def create_detection_dashboard() -> DashboardConfig:
        """Create anomaly detection focused dashboard."""
        return DashboardConfig(
            title="Pynomaly - Anomaly Detection",
            description="Detailed metrics for anomaly detection operations",
            tags=["monorepo", "detection", "ml"],
            panels=DETECTION_PANELS,
        )

    @staticmethod
    def create_streaming_dashboard() -> DashboardConfig:
        """Create streaming operations dashboard."""
        return DashboardConfig(
            title="Pynomaly - Streaming Operations",
            description="Real-time streaming anomaly detection metrics",
            tags=["monorepo", "streaming", "realtime"],
            panels=STREAMING_PANELS,
        )

    @staticmethod
    def create_ensemble_dashboard() -> DashboardConfig:
        """Create ensemble methods dashboard."""
        return DashboardConfig(
            title="Pynomaly - Ensemble Detection",
            description="Ensemble anomaly detection methods and voting strategies",
            tags=["monorepo", "ensemble", "ml"],
            panels=ENSEMBLE_PANELS,
        )

    @staticmethod
    def create_system_dashboard() -> DashboardConfig:
        """Create system health dashboard."""
        return DashboardConfig(
            title="Pynomaly - System Health",
            description="System performance, errors, and resource utilization",
            tags=["monorepo", "system", "health"],
            panels=SYSTEM_PANELS,
        )

    @staticmethod
    def create_business_dashboard() -> DashboardConfig:
        """Create business metrics dashboard."""
        business_panels = [
            {
                "id": 50,
                "title": "Datasets Processed",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "rate(pynomaly_datasets_processed_total[1h])",
                        "legendFormat": "{{source_type}} - {{format}}",
                        "refId": "A",
                    }
                ],
            },
            {
                "id": 51,
                "title": "Data Quality Score",
                "type": "gauge",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "avg(pynomaly_data_quality_score)",
                        "legendFormat": "Average Quality",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 1,
                        "unit": "percentunit",
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.9},
                            ]
                        },
                    }
                },
            },
            {
                "id": 52,
                "title": "Prediction Confidence",
                "type": "graph",
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(pynomaly_prediction_confidence_score_bucket[5m]))",
                        "legendFormat": "P95 Confidence - {{algorithm}}",
                        "refId": "A",
                    }
                ],
            },
        ]

        return DashboardConfig(
            title="Pynomaly - Business Metrics",
            description="Business-focused metrics and KPIs",
            tags=["monorepo", "business", "kpi"],
            panels=business_panels,
        )

    @staticmethod
    def generate_all_dashboards() -> dict[str, str]:
        """Generate all Pynomaly dashboards.

        Returns:
            Dictionary mapping dashboard names to JSON configurations
        """
        generator = DashboardGenerator()

        dashboards = {
            "overview": generator.create_overview_dashboard().to_json(),
            "detection": generator.create_detection_dashboard().to_json(),
            "streaming": generator.create_streaming_dashboard().to_json(),
            "ensemble": generator.create_ensemble_dashboard().to_json(),
            "system": generator.create_system_dashboard().to_json(),
            "business": generator.create_business_dashboard().to_json(),
        }

        return dashboards

    @staticmethod
    def save_dashboards(output_dir: str = "dashboards"):
        """Save all dashboards to files.

        Args:
            output_dir: Directory to save dashboard files
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        dashboards = DashboardGenerator.generate_all_dashboards()

        for name, config in dashboards.items():
            filename = os.path.join(output_dir, f"pynomaly-{name}.json")
            with open(filename, "w") as f:
                f.write(config)

        print(f"Generated {len(dashboards)} dashboard files in {output_dir}/")


# Alert rule templates
ALERT_RULES = {
    "high_error_rate": {
        "alert": "PynomayHighErrorRate",
        "expr": "rate(pynomaly_errors_total[5m]) > 0.1",
        "for": "2m",
        "labels": {"severity": "warning", "service": "monorepo"},
        "annotations": {
            "summary": "High error rate detected in Pynomaly",
            "description": "Error rate is {{ $value }} errors/second",
        },
    },
    "low_detection_accuracy": {
        "alert": "PynomayLowAccuracy",
        "expr": "avg(pynomaly_detection_accuracy_ratio) < 0.7",
        "for": "5m",
        "labels": {"severity": "warning", "service": "monorepo"},
        "annotations": {
            "summary": "Detection accuracy below threshold",
            "description": "Average detection accuracy is {{ $value | humanizePercentage }}",
        },
    },
    "high_memory_usage": {
        "alert": "PynomayHighMemoryUsage",
        "expr": "pynomaly_memory_usage_bytes > 8e9",  # 8GB
        "for": "3m",
        "labels": {"severity": "critical", "service": "monorepo"},
        "annotations": {
            "summary": "High memory usage in Pynomaly",
            "description": "Memory usage is {{ $value | humanizeBytes }}",
        },
    },
    "streaming_backpressure": {
        "alert": "PynomayStreamingBackpressure",
        "expr": "rate(pynomaly_streaming_backpressure_events_total[1m]) > 0",
        "for": "1m",
        "labels": {"severity": "warning", "service": "monorepo"},
        "annotations": {
            "summary": "Streaming backpressure detected",
            "description": "Backpressure events occurring at {{ $value }} events/second",
        },
    },
}


def generate_alert_rules_yaml() -> str:
    """Generate Prometheus alert rules YAML configuration.

    Returns:
        YAML configuration string for Prometheus alert rules
    """
    import yaml

    rules_config = {
        "groups": [{"name": "monorepo.rules", "rules": list(ALERT_RULES.values())}]
    }

    return yaml.dump(rules_config, default_flow_style=False)


if __name__ == "__main__":
    # Generate all dashboards when run as script
    DashboardGenerator.save_dashboards()

    # Generate alert rules
    with open("pynomaly-alerts.yml", "w") as f:
        f.write(generate_alert_rules_yaml())

    print("Dashboard and alert configurations generated successfully!")
