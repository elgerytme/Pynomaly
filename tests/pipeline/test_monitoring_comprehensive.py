"""Comprehensive monitoring and observability tests.

This module contains comprehensive tests for monitoring infrastructure,
metrics collection, alerting systems, and observability features.
"""

import random
import time
import uuid
from typing import Any

import pytest


class TestMonitoringPipeline:
    """Test monitoring and observability pipeline."""

    @pytest.fixture
    def mock_monitoring_system(self):
        """Create mock monitoring system."""

        class MockMonitoringSystem:
            def __init__(self):
                self.metrics_storage = {}
                self.alert_rules = {}
                self.alert_history = []
                self.dashboards = {}
                self.log_streams = {}
                self.health_checks = {}
                self.slo_configs = {}
                self.notification_channels = {}
                self.data_retention_policies = {}

            def setup_metrics_collection(
                self, service_name: str, config: dict[str, Any]
            ) -> dict[str, Any]:
                """Setup metrics collection for a service."""
                collection_id = str(uuid.uuid4())

                metrics_config = {
                    "collection_id": collection_id,
                    "service_name": service_name,
                    "endpoints": config.get("endpoints", []),
                    "scrape_interval": config.get("scrape_interval", "30s"),
                    "timeout": config.get("timeout", "10s"),
                    "metrics_path": config.get("metrics_path", "/metrics"),
                    "labels": config.get("labels", {}),
                    "retention": config.get("retention", "30d"),
                    "enabled": config.get("enabled", True),
                }

                # Setup metric definitions
                default_metrics = [
                    {
                        "name": "http_requests_total",
                        "type": "counter",
                        "help": "Total HTTP requests",
                    },
                    {
                        "name": "http_request_duration_seconds",
                        "type": "histogram",
                        "help": "HTTP request duration",
                    },
                    {
                        "name": "http_requests_per_second",
                        "type": "gauge",
                        "help": "HTTP requests per second",
                    },
                    {
                        "name": "system_cpu_usage_percent",
                        "type": "gauge",
                        "help": "CPU usage percentage",
                    },
                    {
                        "name": "system_memory_usage_bytes",
                        "type": "gauge",
                        "help": "Memory usage in bytes",
                    },
                    {
                        "name": "system_disk_usage_percent",
                        "type": "gauge",
                        "help": "Disk usage percentage",
                    },
                    {
                        "name": "anomaly_detection_score",
                        "type": "gauge",
                        "help": "Current anomaly detection score",
                    },
                    {
                        "name": "anomaly_detection_threshold",
                        "type": "gauge",
                        "help": "Anomaly detection threshold",
                    },
                    {
                        "name": "anomalies_detected_total",
                        "type": "counter",
                        "help": "Total anomalies detected",
                    },
                    {
                        "name": "model_training_duration_seconds",
                        "type": "histogram",
                        "help": "Model training duration",
                    },
                    {
                        "name": "model_accuracy_score",
                        "type": "gauge",
                        "help": "Model accuracy score",
                    },
                ]

                metrics_config["metrics"] = (
                    config.get("custom_metrics", []) + default_metrics
                )

                self.metrics_storage[collection_id] = {
                    "config": metrics_config,
                    "data": {},
                    "last_scrape": None,
                    "scrape_count": 0,
                    "errors": [],
                }

                return {
                    "success": True,
                    "collection_id": collection_id,
                    "config": metrics_config,
                }

            def collect_metrics(self, collection_id: str) -> dict[str, Any]:
                """Collect metrics from configured endpoints."""
                if collection_id not in self.metrics_storage:
                    return {
                        "error": "Collection configuration not found",
                        "success": False,
                    }

                storage = self.metrics_storage[collection_id]
                config = storage["config"]

                # Simulate metric collection
                current_time = time.time()
                collected_metrics = {}

                # Generate sample metric data
                for metric in config["metrics"]:
                    metric_name = metric["name"]
                    metric_type = metric["type"]

                    if metric_type == "counter":
                        # Counters always increase
                        current_value = storage["data"].get(
                            metric_name, 0
                        ) + random.uniform(1, 10)
                        collected_metrics[metric_name] = {
                            "value": current_value,
                            "type": metric_type,
                            "timestamp": current_time,
                            "labels": config["labels"],
                        }

                    elif metric_type == "gauge":
                        # Gauges can fluctuate
                        if "cpu" in metric_name:
                            value = random.uniform(20, 80)  # CPU usage 20-80%
                        elif "memory" in metric_name:
                            value = random.uniform(
                                500_000_000, 2_000_000_000
                            )  # Memory in bytes
                        elif "disk" in metric_name:
                            value = random.uniform(30, 90)  # Disk usage 30-90%
                        elif "anomaly_score" in metric_name:
                            value = random.uniform(0.0, 1.0)  # Anomaly score 0-1
                        elif "threshold" in metric_name:
                            value = random.uniform(0.7, 0.9)  # Threshold 0.7-0.9
                        elif "accuracy" in metric_name:
                            value = random.uniform(0.85, 0.98)  # Accuracy 85-98%
                        elif "requests_per_second" in metric_name:
                            value = random.uniform(50, 500)  # RPS 50-500
                        else:
                            value = random.uniform(0, 100)

                        collected_metrics[metric_name] = {
                            "value": value,
                            "type": metric_type,
                            "timestamp": current_time,
                            "labels": config["labels"],
                        }

                    elif metric_type == "histogram":
                        # Histograms have buckets
                        buckets = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                        bucket_values = {}
                        total_count = random.randint(100, 1000)
                        cumulative_count = 0

                        for bucket in buckets:
                            bucket_count = random.randint(
                                0, total_count - cumulative_count
                            )
                            cumulative_count += bucket_count
                            bucket_values[f"{bucket}"] = cumulative_count

                        bucket_values["inf"] = total_count

                        collected_metrics[metric_name] = {
                            "buckets": bucket_values,
                            "count": total_count,
                            "sum": random.uniform(total_count * 0.1, total_count * 2.0),
                            "type": metric_type,
                            "timestamp": current_time,
                            "labels": config["labels"],
                        }

                # Update storage
                storage["data"].update(collected_metrics)
                storage["last_scrape"] = current_time
                storage["scrape_count"] += 1

                return {
                    "success": True,
                    "collection_id": collection_id,
                    "metrics": collected_metrics,
                    "scrape_time": current_time,
                    "metrics_count": len(collected_metrics),
                }

            def setup_alerting(self, alert_config: dict[str, Any]) -> dict[str, Any]:
                """Setup alerting rules and notifications."""
                alert_id = str(uuid.uuid4())

                alert_rule = {
                    "alert_id": alert_id,
                    "name": alert_config["name"],
                    "query": alert_config["query"],
                    "threshold": alert_config.get("threshold"),
                    "comparison": alert_config.get("comparison", "greater_than"),
                    "duration": alert_config.get("duration", "5m"),
                    "severity": alert_config.get("severity", "warning"),
                    "description": alert_config.get("description", ""),
                    "labels": alert_config.get("labels", {}),
                    "notifications": alert_config.get("notifications", []),
                    "enabled": alert_config.get("enabled", True),
                    "created_at": time.time(),
                }

                self.alert_rules[alert_id] = alert_rule

                return {"success": True, "alert_id": alert_id, "rule": alert_rule}

            def evaluate_alerts(self) -> list[dict[str, Any]]:
                """Evaluate alert rules against current metrics."""
                triggered_alerts = []
                current_time = time.time()

                for alert_id, rule in self.alert_rules.items():
                    if not rule["enabled"]:
                        continue

                    # Simulate alert evaluation
                    alert_triggered = self._evaluate_alert_rule(rule)

                    if alert_triggered:
                        alert_event = {
                            "alert_id": alert_id,
                            "rule_name": rule["name"],
                            "severity": rule["severity"],
                            "description": rule["description"],
                            "labels": rule["labels"],
                            "threshold": rule.get("threshold"),
                            "current_value": alert_triggered["value"],
                            "triggered_at": current_time,
                            "status": "firing",
                        }

                        triggered_alerts.append(alert_event)

                        # Add to alert history
                        self.alert_history.append(alert_event)

                return triggered_alerts

            def _evaluate_alert_rule(
                self, rule: dict[str, Any]
            ) -> dict[str, Any] | None:
                """Evaluate a single alert rule."""
                # Mock alert evaluation based on rule type
                query = rule["query"]
                threshold = rule.get("threshold")
                comparison = rule["comparison"]

                # Generate mock current value based on query
                if "cpu_usage" in query:
                    current_value = random.uniform(20, 95)
                    if (
                        comparison == "greater_than"
                        and threshold
                        and current_value > threshold
                    ):
                        return {"value": current_value, "threshold": threshold}

                elif "memory_usage" in query:
                    current_value = random.uniform(40, 95)
                    if (
                        comparison == "greater_than"
                        and threshold
                        and current_value > threshold
                    ):
                        return {"value": current_value, "threshold": threshold}

                elif "error_rate" in query:
                    current_value = random.uniform(0, 0.15)  # 0-15% error rate
                    if (
                        comparison == "greater_than"
                        and threshold
                        and current_value > threshold
                    ):
                        return {"value": current_value, "threshold": threshold}

                elif "response_time" in query:
                    current_value = random.uniform(100, 2000)  # 100ms to 2s
                    if (
                        comparison == "greater_than"
                        and threshold
                        and current_value > threshold
                    ):
                        return {"value": current_value, "threshold": threshold}

                elif "anomaly_score" in query:
                    current_value = random.uniform(0, 1)
                    if (
                        comparison == "greater_than"
                        and threshold
                        and current_value > threshold
                    ):
                        return {"value": current_value, "threshold": threshold}

                # Random chance of alert triggering for testing
                if random.random() < 0.1:  # 10% chance of alert
                    return {"value": random.uniform(0, 100), "threshold": threshold}

                return None

            def create_dashboard(
                self, dashboard_config: dict[str, Any]
            ) -> dict[str, Any]:
                """Create monitoring dashboard."""
                dashboard_id = str(uuid.uuid4())

                dashboard = {
                    "dashboard_id": dashboard_id,
                    "name": dashboard_config["name"],
                    "description": dashboard_config.get("description", ""),
                    "tags": dashboard_config.get("tags", []),
                    "panels": dashboard_config.get("panels", []),
                    "variables": dashboard_config.get("variables", {}),
                    "time_range": dashboard_config.get(
                        "time_range", {"from": "now-1h", "to": "now"}
                    ),
                    "refresh_interval": dashboard_config.get("refresh_interval", "30s"),
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }

                # Add default panels if none specified
                if not dashboard["panels"]:
                    dashboard["panels"] = self._create_default_panels()

                self.dashboards[dashboard_id] = dashboard

                return {
                    "success": True,
                    "dashboard_id": dashboard_id,
                    "dashboard": dashboard,
                }

            def _create_default_panels(self) -> list[dict[str, Any]]:
                """Create default dashboard panels."""
                return [
                    {
                        "id": "system_metrics",
                        "title": "System Metrics",
                        "type": "graph",
                        "metrics": [
                            "system_cpu_usage_percent",
                            "system_memory_usage_bytes",
                        ],
                        "position": {"x": 0, "y": 0, "width": 12, "height": 8},
                    },
                    {
                        "id": "http_metrics",
                        "title": "HTTP Metrics",
                        "type": "graph",
                        "metrics": [
                            "http_requests_per_second",
                            "http_request_duration_seconds",
                        ],
                        "position": {"x": 0, "y": 8, "width": 12, "height": 8},
                    },
                    {
                        "id": "anomaly_metrics",
                        "title": "Anomaly Detection",
                        "type": "graph",
                        "metrics": [
                            "anomaly_detection_score",
                            "anomaly_detection_threshold",
                        ],
                        "position": {"x": 0, "y": 16, "width": 12, "height": 8},
                    },
                    {
                        "id": "alert_summary",
                        "title": "Alert Summary",
                        "type": "stat",
                        "metrics": ["alerts_total", "alerts_critical"],
                        "position": {"x": 0, "y": 24, "width": 6, "height": 4},
                    },
                ]

            def setup_log_aggregation(
                self, log_config: dict[str, Any]
            ) -> dict[str, Any]:
                """Setup log aggregation and analysis."""
                stream_id = str(uuid.uuid4())

                log_stream = {
                    "stream_id": stream_id,
                    "name": log_config["name"],
                    "sources": log_config.get("sources", []),
                    "filters": log_config.get("filters", []),
                    "parsing_rules": log_config.get("parsing_rules", []),
                    "retention": log_config.get("retention", "30d"),
                    "compression": log_config.get("compression", "gzip"),
                    "indexing": log_config.get(
                        "indexing", ["timestamp", "level", "service"]
                    ),
                    "alerts": log_config.get("alerts", []),
                    "created_at": time.time(),
                }

                self.log_streams[stream_id] = {
                    "config": log_stream,
                    "logs": [],
                    "stats": {
                        "total_logs": 0,
                        "error_logs": 0,
                        "warning_logs": 0,
                        "info_logs": 0,
                    },
                }

                return {"success": True, "stream_id": stream_id, "config": log_stream}

            def ingest_logs(
                self, stream_id: str, logs: list[dict[str, Any]]
            ) -> dict[str, Any]:
                """Ingest logs into log stream."""
                if stream_id not in self.log_streams:
                    return {"error": "Log stream not found", "success": False}

                stream = self.log_streams[stream_id]
                ingested_count = 0
                error_count = 0

                for log_entry in logs:
                    try:
                        # Apply parsing rules and filters
                        processed_log = self._process_log_entry(
                            log_entry, stream["config"]
                        )

                        if processed_log:
                            stream["logs"].append(processed_log)
                            ingested_count += 1

                            # Update stats
                            stream["stats"]["total_logs"] += 1
                            level = processed_log.get("level", "info").lower()
                            if level == "error":
                                stream["stats"]["error_logs"] += 1
                            elif level == "warning":
                                stream["stats"]["warning_logs"] += 1
                            elif level == "info":
                                stream["stats"]["info_logs"] += 1

                    except Exception:
                        error_count += 1

                return {
                    "success": True,
                    "stream_id": stream_id,
                    "ingested_count": ingested_count,
                    "error_count": error_count,
                    "total_logs": stream["stats"]["total_logs"],
                }

            def _process_log_entry(
                self, log_entry: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any] | None:
                """Process individual log entry."""
                # Apply filters
                for filter_rule in config.get("filters", []):
                    if not self._apply_log_filter(log_entry, filter_rule):
                        return None

                # Apply parsing rules
                processed_entry = log_entry.copy()
                processed_entry["timestamp"] = processed_entry.get(
                    "timestamp", time.time()
                )
                processed_entry["stream_id"] = config["stream_id"]

                # Parse log level if not present
                if "level" not in processed_entry:
                    message = processed_entry.get("message", "").lower()
                    if "error" in message or "err" in message:
                        processed_entry["level"] = "error"
                    elif "warn" in message:
                        processed_entry["level"] = "warning"
                    else:
                        processed_entry["level"] = "info"

                return processed_entry

            def _apply_log_filter(
                self, log_entry: dict[str, Any], filter_rule: dict[str, Any]
            ) -> bool:
                """Apply log filter rule."""
                # Simple filter implementation
                field = filter_rule.get("field")
                operator = filter_rule.get("operator", "equals")
                value = filter_rule.get("value")

                if not field or value is None:
                    return True

                entry_value = log_entry.get(field)

                if operator == "equals":
                    return entry_value == value
                elif operator == "contains":
                    return value in str(entry_value)
                elif operator == "not_equals":
                    return entry_value != value

                return True

            def setup_health_checks(
                self, service_name: str, health_config: dict[str, Any]
            ) -> dict[str, Any]:
                """Setup health checks for service."""
                health_check_id = str(uuid.uuid4())

                health_check = {
                    "health_check_id": health_check_id,
                    "service_name": service_name,
                    "endpoints": health_config.get("endpoints", []),
                    "interval": health_config.get("interval", "30s"),
                    "timeout": health_config.get("timeout", "10s"),
                    "retries": health_config.get("retries", 3),
                    "expected_status": health_config.get("expected_status", 200),
                    "expected_response": health_config.get("expected_response", {}),
                    "alerts": health_config.get("alerts", []),
                    "created_at": time.time(),
                }

                self.health_checks[health_check_id] = {
                    "config": health_check,
                    "results": [],
                    "current_status": "unknown",
                }

                return {
                    "success": True,
                    "health_check_id": health_check_id,
                    "config": health_check,
                }

            def run_health_checks(self, health_check_id: str = None) -> dict[str, Any]:
                """Run health checks."""
                if health_check_id:
                    health_checks_to_run = {
                        health_check_id: self.health_checks[health_check_id]
                    }
                else:
                    health_checks_to_run = self.health_checks

                results = {}

                for hc_id, health_check in health_checks_to_run.items():
                    config = health_check["config"]
                    check_results = []

                    for endpoint in config["endpoints"]:
                        # Simulate health check
                        check_result = self._simulate_health_check(endpoint, config)
                        check_results.append(check_result)

                    # Determine overall health
                    all_healthy = all(result["healthy"] for result in check_results)

                    overall_result = {
                        "health_check_id": hc_id,
                        "service_name": config["service_name"],
                        "overall_healthy": all_healthy,
                        "endpoint_results": check_results,
                        "checked_at": time.time(),
                    }

                    # Update stored results
                    health_check["results"].append(overall_result)
                    health_check["current_status"] = (
                        "healthy" if all_healthy else "unhealthy"
                    )

                    # Keep only last 100 results
                    health_check["results"] = health_check["results"][-100:]

                    results[hc_id] = overall_result

                return {
                    "success": True,
                    "results": results,
                    "total_checks": len(results),
                    "healthy_services": sum(
                        1 for r in results.values() if r["overall_healthy"]
                    ),
                }

            def _simulate_health_check(
                self, endpoint: str, config: dict[str, Any]
            ) -> dict[str, Any]:
                """Simulate health check for endpoint."""
                # Mock health check result
                healthy = random.random() > 0.1  # 90% success rate
                response_time = random.uniform(50, 500)  # 50-500ms
                status_code = 200 if healthy else random.choice([404, 500, 503])

                return {
                    "endpoint": endpoint,
                    "healthy": healthy,
                    "status_code": status_code,
                    "response_time_ms": response_time,
                    "expected_status": config["expected_status"],
                    "message": (
                        "Health check passed" if healthy else "Health check failed"
                    ),
                }

            def setup_slo_monitoring(
                self, slo_config: dict[str, Any]
            ) -> dict[str, Any]:
                """Setup Service Level Objective monitoring."""
                slo_id = str(uuid.uuid4())

                slo = {
                    "slo_id": slo_id,
                    "name": slo_config["name"],
                    "service": slo_config["service"],
                    "objectives": slo_config[
                        "objectives"
                    ],  # e.g., [{"metric": "availability", "target": 99.9}]
                    "time_window": slo_config.get("time_window", "30d"),
                    "error_budget": slo_config.get("error_budget", {}),
                    "notifications": slo_config.get("notifications", []),
                    "created_at": time.time(),
                }

                self.slo_configs[slo_id] = {
                    "config": slo,
                    "current_performance": {},
                    "error_budget_remaining": {},
                    "history": [],
                }

                return {"success": True, "slo_id": slo_id, "config": slo}

            def calculate_slo_compliance(self, slo_id: str) -> dict[str, Any]:
                """Calculate SLO compliance and error budget."""
                if slo_id not in self.slo_configs:
                    return {"error": "SLO configuration not found", "success": False}

                slo_data = self.slo_configs[slo_id]
                config = slo_data["config"]

                compliance_results = {}

                for objective in config["objectives"]:
                    metric = objective["metric"]
                    target = objective["target"]

                    # Mock current performance
                    if metric == "availability":
                        current_value = random.uniform(99.5, 99.99)
                    elif metric == "latency":
                        current_value = random.uniform(50, 200)
                        target = objective.get("threshold", 100)  # latency threshold
                    elif metric == "error_rate":
                        current_value = random.uniform(0.001, 0.01)
                        target = objective.get("threshold", 0.005)
                    else:
                        current_value = random.uniform(90, 99.5)

                    # Calculate compliance
                    if metric == "availability":
                        compliance = (current_value / target) * 100
                        error_budget_consumed = max(
                            0, (target - current_value) / (100 - target)
                        )
                    elif metric in ["latency", "error_rate"]:
                        compliance = (
                            100
                            if current_value <= target
                            else (target / current_value) * 100
                        )
                        error_budget_consumed = max(
                            0, (current_value - target) / target
                        )
                    else:
                        compliance = (current_value / target) * 100
                        error_budget_consumed = max(
                            0, (target - current_value) / target
                        )

                    compliance_results[metric] = {
                        "current_value": current_value,
                        "target": target,
                        "compliance_percentage": min(100, compliance),
                        "error_budget_consumed_percentage": min(
                            100, error_budget_consumed * 100
                        ),
                        "status": (
                            "meeting"
                            if compliance >= 99
                            else "at_risk" if compliance >= 95 else "violated"
                        ),
                    }

                # Update stored data
                slo_data["current_performance"] = compliance_results
                slo_data["error_budget_remaining"] = {
                    metric: 100 - result["error_budget_consumed_percentage"]
                    for metric, result in compliance_results.items()
                }

                return {
                    "success": True,
                    "slo_id": slo_id,
                    "service": config["service"],
                    "compliance": compliance_results,
                    "overall_status": self._determine_overall_slo_status(
                        compliance_results
                    ),
                }

            def _determine_overall_slo_status(
                self, compliance_results: dict[str, Any]
            ) -> str:
                """Determine overall SLO status."""
                statuses = [result["status"] for result in compliance_results.values()]

                if any(status == "violated" for status in statuses):
                    return "violated"
                elif any(status == "at_risk" for status in statuses):
                    return "at_risk"
                else:
                    return "meeting"

            def get_monitoring_summary(self) -> dict[str, Any]:
                """Get comprehensive monitoring summary."""
                current_time = time.time()

                # Count metrics collections
                active_collections = sum(
                    1
                    for storage in self.metrics_storage.values()
                    if storage["config"]["enabled"]
                )

                # Count alerts
                active_alerts = sum(
                    1 for rule in self.alert_rules.values() if rule["enabled"]
                )
                recent_alerts = len(
                    [
                        alert
                        for alert in self.alert_history
                        if current_time - alert["triggered_at"] < 3600
                    ]
                )  # Last hour

                # Count dashboards
                total_dashboards = len(self.dashboards)

                # Count log streams
                active_log_streams = len(self.log_streams)
                total_logs = sum(
                    stream["stats"]["total_logs"]
                    for stream in self.log_streams.values()
                )

                # Count health checks
                total_health_checks = len(self.health_checks)
                healthy_services = sum(
                    1
                    for hc in self.health_checks.values()
                    if hc["current_status"] == "healthy"
                )

                # Count SLOs
                total_slos = len(self.slo_configs)
                slos_meeting = sum(
                    1
                    for slo in self.slo_configs.values()
                    if self._get_slo_status(slo) == "meeting"
                )

                return {
                    "timestamp": current_time,
                    "metrics": {
                        "active_collections": active_collections,
                        "total_metrics": sum(
                            len(storage["data"])
                            for storage in self.metrics_storage.values()
                        ),
                    },
                    "alerts": {
                        "active_rules": active_alerts,
                        "recent_alerts": recent_alerts,
                        "total_alert_history": len(self.alert_history),
                    },
                    "dashboards": {"total_dashboards": total_dashboards},
                    "logging": {
                        "active_streams": active_log_streams,
                        "total_logs": total_logs,
                    },
                    "health_checks": {
                        "total_checks": total_health_checks,
                        "healthy_services": healthy_services,
                        "unhealthy_services": total_health_checks - healthy_services,
                    },
                    "slo": {
                        "total_slos": total_slos,
                        "slos_meeting": slos_meeting,
                        "slos_at_risk": total_slos - slos_meeting,
                    },
                }

            def _get_slo_status(self, slo_data: dict[str, Any]) -> str:
                """Get SLO status from stored data."""
                performance = slo_data.get("current_performance", {})
                if not performance:
                    return "unknown"

                statuses = [
                    result.get("status", "unknown") for result in performance.values()
                ]
                return self._determine_overall_slo_status(
                    {"dummy": {"status": status} for status in statuses}
                )

        return MockMonitoringSystem()

    def test_metrics_collection_setup(self, mock_monitoring_system):
        """Test metrics collection setup and configuration."""
        monitoring = mock_monitoring_system

        # Setup metrics collection for Pynomaly service
        service_config = {
            "endpoints": ["http://pynomaly-api:8000", "http://pynomaly-worker:8001"],
            "scrape_interval": "15s",
            "timeout": "5s",
            "metrics_path": "/metrics",
            "labels": {
                "environment": "production",
                "service": "pynomaly",
                "version": "1.0.0",
            },
            "retention": "60d",
            "custom_metrics": [
                {
                    "name": "pynomaly_models_trained_total",
                    "type": "counter",
                    "help": "Total models trained",
                },
                {
                    "name": "pynomaly_predictions_made_total",
                    "type": "counter",
                    "help": "Total predictions made",
                },
                {
                    "name": "pynomaly_active_models",
                    "type": "gauge",
                    "help": "Number of active models",
                },
            ],
        }

        setup_result = monitoring.setup_metrics_collection("pynomaly", service_config)

        assert setup_result["success"]
        assert "collection_id" in setup_result
        assert "config" in setup_result

        # Verify configuration
        config = setup_result["config"]
        assert config["service_name"] == "pynomaly"
        assert config["scrape_interval"] == "15s"
        assert config["timeout"] == "5s"
        assert config["retention"] == "60d"
        assert config["labels"]["environment"] == "production"

        # Verify metrics include both default and custom
        metrics = config["metrics"]
        metric_names = [metric["name"] for metric in metrics]

        # Check default metrics
        assert "http_requests_total" in metric_names
        assert "system_cpu_usage_percent" in metric_names
        assert "anomaly_detection_score" in metric_names

        # Check custom metrics
        assert "pynomaly_models_trained_total" in metric_names
        assert "pynomaly_predictions_made_total" in metric_names
        assert "pynomaly_active_models" in metric_names

        # Verify metric types
        for metric in metrics:
            assert metric["type"] in ["counter", "gauge", "histogram"]
            assert "name" in metric
            assert "help" in metric

    def test_metrics_collection_execution(self, mock_monitoring_system):
        """Test actual metrics collection execution."""
        monitoring = mock_monitoring_system

        # Setup collection
        setup_result = monitoring.setup_metrics_collection(
            "test-service",
            {"endpoints": ["http://test:8000"], "scrape_interval": "30s"},
        )

        collection_id = setup_result["collection_id"]

        # Collect metrics
        collection_result = monitoring.collect_metrics(collection_id)

        assert collection_result["success"]
        assert collection_result["collection_id"] == collection_id
        assert "metrics" in collection_result
        assert "scrape_time" in collection_result
        assert "metrics_count" in collection_result

        # Verify collected metrics
        metrics = collection_result["metrics"]
        assert len(metrics) > 0

        for metric_name, metric_data in metrics.items():
            assert (
                "value" in metric_data or "buckets" in metric_data
            )  # gauge/counter or histogram
            assert "type" in metric_data
            assert "timestamp" in metric_data
            assert "labels" in metric_data

            # Verify metric types
            if metric_data["type"] == "counter":
                assert metric_data["value"] >= 0
            elif metric_data["type"] == "gauge":
                assert isinstance(metric_data["value"], int | float)
            elif metric_data["type"] == "histogram":
                assert "buckets" in metric_data
                assert "count" in metric_data
                assert "sum" in metric_data

        # Test multiple collections to verify counter increments
        second_collection = monitoring.collect_metrics(collection_id)

        # Counters should have increased
        for metric_name, metric_data in second_collection["metrics"].items():
            if metric_data["type"] == "counter":
                original_value = metrics[metric_name]["value"]
                assert metric_data["value"] >= original_value

    def test_alerting_setup_and_evaluation(self, mock_monitoring_system):
        """Test alerting rules setup and evaluation."""
        monitoring = mock_monitoring_system

        # Setup alert rules
        alert_configs = [
            {
                "name": "high_cpu_usage",
                "query": "system_cpu_usage_percent",
                "threshold": 80,
                "comparison": "greater_than",
                "duration": "5m",
                "severity": "warning",
                "description": "CPU usage is high",
                "labels": {"team": "infrastructure"},
                "notifications": ["email", "slack"],
            },
            {
                "name": "high_error_rate",
                "query": "error_rate",
                "threshold": 0.05,
                "comparison": "greater_than",
                "duration": "2m",
                "severity": "critical",
                "description": "Error rate is too high",
                "labels": {"team": "sre"},
                "notifications": ["email", "slack", "pagerduty"],
            },
            {
                "name": "anomaly_threshold_exceeded",
                "query": "anomaly_score",
                "threshold": 0.9,
                "comparison": "greater_than",
                "duration": "1m",
                "severity": "critical",
                "description": "Anomaly detection threshold exceeded",
                "labels": {"team": "ml"},
                "notifications": ["email", "slack"],
            },
        ]

        alert_ids = []
        for alert_config in alert_configs:
            alert_result = monitoring.setup_alerting(alert_config)

            assert alert_result["success"]
            assert "alert_id" in alert_result
            assert "rule" in alert_result

            alert_ids.append(alert_result["alert_id"])

            # Verify rule configuration
            rule = alert_result["rule"]
            assert rule["name"] == alert_config["name"]
            assert rule["query"] == alert_config["query"]
            assert rule["threshold"] == alert_config["threshold"]
            assert rule["severity"] == alert_config["severity"]
            assert rule["enabled"]

        # Evaluate alerts
        triggered_alerts = monitoring.evaluate_alerts()

        # Verify alert evaluation structure
        for alert in triggered_alerts:
            assert "alert_id" in alert
            assert "rule_name" in alert
            assert "severity" in alert
            assert "current_value" in alert
            assert "threshold" in alert
            assert "triggered_at" in alert
            assert "status" in alert
            assert alert["status"] == "firing"
            assert alert["severity"] in ["warning", "critical"]

        # Verify alert history
        assert len(monitoring.alert_history) >= len(triggered_alerts)

    def test_dashboard_creation(self, mock_monitoring_system):
        """Test monitoring dashboard creation."""
        monitoring = mock_monitoring_system

        # Create comprehensive dashboard
        dashboard_config = {
            "name": "Pynomaly Monitoring Dashboard",
            "description": "Main monitoring dashboard for Pynomaly service",
            "tags": ["pynomaly", "anomaly-detection", "production"],
            "time_range": {"from": "now-6h", "to": "now"},
            "refresh_interval": "30s",
            "panels": [
                {
                    "id": "request_metrics",
                    "title": "Request Metrics",
                    "type": "graph",
                    "metrics": [
                        "http_requests_per_second",
                        "http_request_duration_seconds",
                    ],
                    "position": {"x": 0, "y": 0, "width": 12, "height": 8},
                },
                {
                    "id": "anomaly_detection",
                    "title": "Anomaly Detection",
                    "type": "graph",
                    "metrics": [
                        "anomaly_detection_score",
                        "anomaly_detection_threshold",
                        "anomalies_detected_total",
                    ],
                    "position": {"x": 0, "y": 8, "width": 12, "height": 8},
                },
                {
                    "id": "model_performance",
                    "title": "Model Performance",
                    "type": "stat",
                    "metrics": [
                        "model_accuracy_score",
                        "model_training_duration_seconds",
                    ],
                    "position": {"x": 0, "y": 16, "width": 6, "height": 4},
                },
                {
                    "id": "system_health",
                    "title": "System Health",
                    "type": "graph",
                    "metrics": [
                        "system_cpu_usage_percent",
                        "system_memory_usage_bytes",
                    ],
                    "position": {"x": 6, "y": 16, "width": 6, "height": 4},
                },
            ],
            "variables": {
                "environment": {"type": "query", "values": ["production", "staging"]},
                "service": {"type": "constant", "value": "pynomaly"},
            },
        }

        dashboard_result = monitoring.create_dashboard(dashboard_config)

        assert dashboard_result["success"]
        assert "dashboard_id" in dashboard_result
        assert "dashboard" in dashboard_result

        # Verify dashboard configuration
        dashboard = dashboard_result["dashboard"]
        assert dashboard["name"] == dashboard_config["name"]
        assert dashboard["description"] == dashboard_config["description"]
        assert dashboard["tags"] == dashboard_config["tags"]
        assert dashboard["time_range"] == dashboard_config["time_range"]
        assert dashboard["refresh_interval"] == dashboard_config["refresh_interval"]

        # Verify panels
        panels = dashboard["panels"]
        assert len(panels) == len(dashboard_config["panels"])

        for panel in panels:
            assert "id" in panel
            assert "title" in panel
            assert "type" in panel
            assert "metrics" in panel
            assert "position" in panel
            assert panel["type"] in ["graph", "stat", "table", "heatmap"]

        # Verify variables
        variables = dashboard["variables"]
        assert "environment" in variables
        assert "service" in variables

        # Test dashboard with default panels
        simple_dashboard = monitoring.create_dashboard({"name": "Simple Dashboard"})
        assert simple_dashboard["success"]
        assert (
            len(simple_dashboard["dashboard"]["panels"]) > 0
        )  # Should have default panels

    def test_log_aggregation_and_analysis(self, mock_monitoring_system):
        """Test log aggregation and analysis setup."""
        monitoring = mock_monitoring_system

        # Setup log aggregation
        log_config = {
            "name": "pynomaly_application_logs",
            "sources": ["pynomaly-api", "pynomaly-worker", "pynomaly-scheduler"],
            "filters": [
                {"field": "level", "operator": "not_equals", "value": "debug"},
                {"field": "message", "operator": "contains", "value": "anomaly"},
            ],
            "parsing_rules": [
                {"field": "timestamp", "format": "ISO8601"},
                {"field": "level", "values": ["error", "warning", "info", "debug"]},
            ],
            "retention": "90d",
            "indexing": ["timestamp", "level", "service", "user_id"],
            "alerts": [
                {"pattern": "ERROR", "threshold": 10, "window": "5m"},
                {"pattern": "CRITICAL", "threshold": 1, "window": "1m"},
            ],
        }

        setup_result = monitoring.setup_log_aggregation(log_config)

        assert setup_result["success"]
        assert "stream_id" in setup_result
        assert "config" in setup_result

        # Verify configuration
        config = setup_result["config"]
        assert config["name"] == log_config["name"]
        assert config["sources"] == log_config["sources"]
        assert config["retention"] == log_config["retention"]
        assert config["indexing"] == log_config["indexing"]

        stream_id = setup_result["stream_id"]

        # Test log ingestion
        sample_logs = [
            {
                "timestamp": time.time(),
                "level": "info",
                "service": "pynomaly-api",
                "message": "Anomaly detection request processed",
                "user_id": "user123",
                "request_id": "req456",
            },
            {
                "timestamp": time.time(),
                "level": "error",
                "service": "pynomaly-worker",
                "message": "Failed to load model",
                "user_id": "user789",
                "error_code": "MODEL_LOAD_ERROR",
            },
            {
                "timestamp": time.time(),
                "level": "warning",
                "service": "pynomaly-api",
                "message": "High anomaly score detected",
                "user_id": "user123",
                "anomaly_score": 0.95,
            },
            {
                "timestamp": time.time(),
                "level": "debug",
                "service": "pynomaly-api",
                "message": "Debug trace information",
                "user_id": "user123",
            },
        ]

        ingestion_result = monitoring.ingest_logs(stream_id, sample_logs)

        assert ingestion_result["success"]
        assert ingestion_result["stream_id"] == stream_id
        assert "ingested_count" in ingestion_result
        assert "total_logs" in ingestion_result

        # Debug logs should be filtered out
        assert ingestion_result["ingested_count"] < len(sample_logs)

        # Verify log statistics
        stream_data = monitoring.log_streams[stream_id]
        stats = stream_data["stats"]

        assert stats["total_logs"] > 0
        assert stats["error_logs"] >= 1  # At least one error log
        assert stats["warning_logs"] >= 1  # At least one warning log
        assert stats["info_logs"] >= 1  # At least one info log

    def test_health_checks_setup_and_execution(self, mock_monitoring_system):
        """Test health checks setup and execution."""
        monitoring = mock_monitoring_system

        # Setup health checks
        health_config = {
            "endpoints": [
                "http://pynomaly-api:8000/health",
                "http://pynomaly-api:8000/ready",
                "http://pynomaly-worker:8001/health",
            ],
            "interval": "30s",
            "timeout": "5s",
            "retries": 3,
            "expected_status": 200,
            "expected_response": {"status": "healthy"},
            "alerts": [
                {"type": "unhealthy", "notification": ["email", "slack"]},
                {"type": "degraded", "notification": ["slack"]},
            ],
        }

        setup_result = monitoring.setup_health_checks("pynomaly", health_config)

        assert setup_result["success"]
        assert "health_check_id" in setup_result
        assert "config" in setup_result

        # Verify configuration
        config = setup_result["config"]
        assert config["service_name"] == "pynomaly"
        assert config["endpoints"] == health_config["endpoints"]
        assert config["interval"] == health_config["interval"]
        assert config["timeout"] == health_config["timeout"]
        assert config["expected_status"] == health_config["expected_status"]

        health_check_id = setup_result["health_check_id"]

        # Run health checks
        check_result = monitoring.run_health_checks(health_check_id)

        assert check_result["success"]
        assert "results" in check_result
        assert "total_checks" in check_result
        assert "healthy_services" in check_result

        # Verify check results
        results = check_result["results"]
        assert health_check_id in results

        hc_result = results[health_check_id]
        assert hc_result["health_check_id"] == health_check_id
        assert hc_result["service_name"] == "pynomaly"
        assert "overall_healthy" in hc_result
        assert "endpoint_results" in hc_result
        assert "checked_at" in hc_result

        # Verify endpoint results
        endpoint_results = hc_result["endpoint_results"]
        assert len(endpoint_results) == len(health_config["endpoints"])

        for endpoint_result in endpoint_results:
            assert "endpoint" in endpoint_result
            assert "healthy" in endpoint_result
            assert "status_code" in endpoint_result
            assert "response_time_ms" in endpoint_result
            assert "message" in endpoint_result
            assert endpoint_result["response_time_ms"] > 0

        # Run all health checks
        all_checks_result = monitoring.run_health_checks()
        assert all_checks_result["success"]
        assert len(all_checks_result["results"]) >= 1

    def test_slo_monitoring_setup_and_calculation(self, mock_monitoring_system):
        """Test SLO monitoring setup and compliance calculation."""
        monitoring = mock_monitoring_system

        # Setup SLO monitoring
        slo_config = {
            "name": "Pynomaly API SLO",
            "service": "pynomaly-api",
            "objectives": [
                {"metric": "availability", "target": 99.9},
                {"metric": "latency", "threshold": 100},  # 100ms P95
                {"metric": "error_rate", "threshold": 0.01},  # 1% max error rate
            ],
            "time_window": "30d",
            "error_budget": {"policy": "burn_rate"},
            "notifications": ["email", "slack"],
        }

        setup_result = monitoring.setup_slo_monitoring(slo_config)

        assert setup_result["success"]
        assert "slo_id" in setup_result
        assert "config" in setup_result

        # Verify configuration
        config = setup_result["config"]
        assert config["name"] == slo_config["name"]
        assert config["service"] == slo_config["service"]
        assert config["objectives"] == slo_config["objectives"]
        assert config["time_window"] == slo_config["time_window"]

        slo_id = setup_result["slo_id"]

        # Calculate SLO compliance
        compliance_result = monitoring.calculate_slo_compliance(slo_id)

        assert compliance_result["success"]
        assert compliance_result["slo_id"] == slo_id
        assert compliance_result["service"] == "pynomaly-api"
        assert "compliance" in compliance_result
        assert "overall_status" in compliance_result

        # Verify compliance results
        compliance = compliance_result["compliance"]

        for objective in slo_config["objectives"]:
            metric = objective["metric"]
            assert metric in compliance

            metric_compliance = compliance[metric]
            assert "current_value" in metric_compliance
            assert "target" in metric_compliance or "threshold" in metric_compliance
            assert "compliance_percentage" in metric_compliance
            assert "error_budget_consumed_percentage" in metric_compliance
            assert "status" in metric_compliance

            # Verify compliance percentage is between 0 and 100
            assert 0 <= metric_compliance["compliance_percentage"] <= 100
            assert 0 <= metric_compliance["error_budget_consumed_percentage"] <= 100
            assert metric_compliance["status"] in ["meeting", "at_risk", "violated"]

        # Verify overall status
        assert compliance_result["overall_status"] in ["meeting", "at_risk", "violated"]

    def test_comprehensive_monitoring_summary(self, mock_monitoring_system):
        """Test comprehensive monitoring system summary."""
        monitoring = mock_monitoring_system

        # Setup various monitoring components

        # 1. Metrics collection
        monitoring.setup_metrics_collection(
            "test-service-1", {"endpoints": ["http://test1:8000"]}
        )
        monitoring.setup_metrics_collection(
            "test-service-2", {"endpoints": ["http://test2:8000"]}
        )

        # 2. Alert rules
        monitoring.setup_alerting(
            {
                "name": "test_alert_1",
                "query": "cpu_usage",
                "threshold": 80,
                "severity": "warning",
            }
        )
        monitoring.setup_alerting(
            {
                "name": "test_alert_2",
                "query": "error_rate",
                "threshold": 0.05,
                "severity": "critical",
            }
        )

        # 3. Dashboards
        monitoring.create_dashboard({"name": "Dashboard 1"})
        monitoring.create_dashboard({"name": "Dashboard 2"})

        # 4. Log streams
        monitoring.setup_log_aggregation({"name": "logs1", "sources": ["service1"]})
        monitoring.setup_log_aggregation({"name": "logs2", "sources": ["service2"]})

        # 5. Health checks
        monitoring.setup_health_checks(
            "service1", {"endpoints": ["http://service1:8000/health"]}
        )
        monitoring.setup_health_checks(
            "service2", {"endpoints": ["http://service2:8000/health"]}
        )

        # 6. SLOs
        monitoring.setup_slo_monitoring(
            {
                "name": "SLO 1",
                "service": "service1",
                "objectives": [{"metric": "availability", "target": 99.9}],
            }
        )

        # Generate some data
        monitoring.evaluate_alerts()  # Generate some alert history

        # Get monitoring summary
        summary = monitoring.get_monitoring_summary()

        assert "timestamp" in summary
        assert "metrics" in summary
        assert "alerts" in summary
        assert "dashboards" in summary
        assert "logging" in summary
        assert "health_checks" in summary
        assert "slo" in summary

        # Verify metrics summary
        metrics_summary = summary["metrics"]
        assert "active_collections" in metrics_summary
        assert "total_metrics" in metrics_summary
        assert metrics_summary["active_collections"] == 2

        # Verify alerts summary
        alerts_summary = summary["alerts"]
        assert "active_rules" in alerts_summary
        assert "recent_alerts" in alerts_summary
        assert "total_alert_history" in alerts_summary
        assert alerts_summary["active_rules"] == 2

        # Verify dashboards summary
        dashboards_summary = summary["dashboards"]
        assert "total_dashboards" in dashboards_summary
        assert dashboards_summary["total_dashboards"] == 2

        # Verify logging summary
        logging_summary = summary["logging"]
        assert "active_streams" in logging_summary
        assert "total_logs" in logging_summary
        assert logging_summary["active_streams"] == 2

        # Verify health checks summary
        health_summary = summary["health_checks"]
        assert "total_checks" in health_summary
        assert "healthy_services" in health_summary
        assert "unhealthy_services" in health_summary
        assert health_summary["total_checks"] == 2

        # Verify SLO summary
        slo_summary = summary["slo"]
        assert "total_slos" in slo_summary
        assert "slos_meeting" in slo_summary
        assert "slos_at_risk" in slo_summary
        assert slo_summary["total_slos"] == 1
