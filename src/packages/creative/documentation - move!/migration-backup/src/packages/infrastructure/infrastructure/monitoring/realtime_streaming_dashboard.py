"""
Real-time Streaming Dashboard

Advanced real-time dashboard for monitoring streaming anomaly detection
with live updates, interactive visualizations, and intelligent alerts.
"""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots

from monorepo.application.services.streaming_anomaly_detection_service import (
    AnomalyAlert,
    AnomalyThreatLevel,
    StreamingAnomalyDetectionService,
)
from monorepo.infrastructure.logging.structured_logger import StructuredLogger


class RealtimeStreamingDashboard:
    """Real-time dashboard for streaming anomaly detection."""

    def __init__(self, anomaly_service: StreamingAnomalyDetectionService):
        self.anomaly_service = anomaly_service
        self.logger = StructuredLogger("realtime_dashboard")

        # Data storage for visualization
        self.alert_history: deque = deque(maxlen=1000)
        self.metric_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.detector_metrics: dict[str, dict] = {}

        # Dashboard state
        self.is_running = False
        self.update_interval = 1.0  # seconds
        self.last_update = datetime.now()

        # Alert aggregation
        self.alert_counts = defaultdict(int)
        self.threat_level_counts = defaultdict(int)

        # Register callback with anomaly service
        self.anomaly_service.add_global_callback(self._on_anomaly_alert)

    def _on_anomaly_alert(self, alert: AnomalyAlert):
        """Handle new anomaly alerts."""

        self.alert_history.append(alert)
        self.alert_counts[alert.data_point.source_id] += 1
        self.threat_level_counts[alert.threat_level.value] += 1

        # Update metrics
        timestamp = alert.timestamp
        self.metric_history["anomaly_scores"].append(
            {
                "timestamp": timestamp,
                "value": alert.anomaly_score,
                "threat_level": alert.threat_level.value,
            }
        )

        self.metric_history["alert_rate"].append(
            {"timestamp": timestamp, "value": self._calculate_current_alert_rate()}
        )

    def _calculate_current_alert_rate(self) -> float:
        """Calculate current alert rate (alerts per minute)."""

        if len(self.alert_history) < 2:
            return 0.0

        # Count alerts in last minute
        cutoff_time = datetime.now() - timedelta(minutes=1)
        recent_alerts = [
            alert for alert in self.alert_history if alert.timestamp >= cutoff_time
        ]

        return len(recent_alerts)

    async def start_dashboard(self, port: int = 8080):
        """Start the real-time dashboard server."""

        self.is_running = True
        self.logger.info(f"Starting real-time dashboard on port {port}")

        # Start background update task
        update_task = asyncio.create_task(self._background_update_loop())

        try:
            # In a real implementation, you'd start a web server here
            # For now, we'll simulate with continuous updates
            while self.is_running:
                await asyncio.sleep(self.update_interval)

        except KeyboardInterrupt:
            self.logger.info("Shutting down dashboard")
        finally:
            self.is_running = False
            update_task.cancel()

    async def _background_update_loop(self):
        """Background loop for updating dashboard metrics."""

        while self.is_running:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background update: {e}")
                await asyncio.sleep(5)

    async def _update_metrics(self):
        """Update dashboard metrics from anomaly service."""

        # Get service stats
        service_stats = self.anomaly_service.get_service_stats()

        timestamp = datetime.now()

        # Update system metrics
        self.metric_history["total_data_points"].append(
            {"timestamp": timestamp, "value": service_stats["total_data_points"]}
        )

        self.metric_history["active_detectors"].append(
            {"timestamp": timestamp, "value": service_stats["active_detectors"]}
        )

        self.metric_history["total_alerts"].append(
            {"timestamp": timestamp, "value": service_stats["total_alerts"]}
        )

        # Update detector-specific metrics
        for detector_id, detector_stats in service_stats.get(
            "detector_stats", {}
        ).items():
            self.detector_metrics[detector_id] = detector_stats

            # Store historical data
            self.metric_history[f"{detector_id}_anomaly_rate"].append(
                {
                    "timestamp": timestamp,
                    "value": detector_stats.get("recent_anomaly_rate", 0),
                }
            )

            self.metric_history[f"{detector_id}_processing_time"].append(
                {
                    "timestamp": timestamp,
                    "value": detector_stats.get("average_processing_time", 0),
                }
            )

        self.last_update = timestamp

    def generate_dashboard_data(self) -> dict[str, Any]:
        """Generate comprehensive dashboard data."""

        return {
            "summary": self._generate_summary(),
            "alerts": self._generate_alert_data(),
            "metrics": self._generate_metric_data(),
            "detectors": self._generate_detector_data(),
            "visualizations": self._generate_visualizations(),
            "last_update": self.last_update.isoformat(),
        }

    def _generate_summary(self) -> dict[str, Any]:
        """Generate dashboard summary statistics."""

        # Recent alert statistics
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history if alert.timestamp >= recent_cutoff
        ]

        # Threat level distribution
        threat_distribution = defaultdict(int)
        for alert in recent_alerts:
            threat_distribution[alert.threat_level.value] += 1

        return {
            "total_alerts_24h": len(self.alert_history),
            "recent_alerts_1h": len(recent_alerts),
            "current_alert_rate": self._calculate_current_alert_rate(),
            "active_detectors": len(self.detector_metrics),
            "highest_threat_level": self._get_highest_recent_threat_level(),
            "threat_distribution": dict(threat_distribution),
            "system_status": self._determine_system_status(),
        }

    def _generate_alert_data(self) -> list[dict[str, Any]]:
        """Generate alert data for dashboard."""

        # Get recent alerts (last 50)
        recent_alerts = list(self.alert_history)[-50:]

        alert_data = []
        for alert in recent_alerts:
            alert_data.append(
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "threat_level": alert.threat_level.value,
                    "anomaly_score": alert.anomaly_score,
                    "confidence": alert.confidence,
                    "source_id": alert.data_point.source_id,
                    "explanation": alert.explanation,
                    "affected_features": alert.affected_features,
                    "recommendations": alert.recommendations[
                        :3
                    ],  # Top 3 recommendations
                }
            )

        return alert_data

    def _generate_metric_data(self) -> dict[str, list[dict[str, Any]]]:
        """Generate metric time series data."""

        metric_data = {}

        for metric_name, history in self.metric_history.items():
            if history:
                metric_data[metric_name] = list(history)[-100:]  # Last 100 points

        return metric_data

    def _generate_detector_data(self) -> dict[str, dict[str, Any]]:
        """Generate detector-specific data."""

        detector_data = {}

        for detector_id, metrics in self.detector_metrics.items():
            detector_data[detector_id] = {
                "status": "active" if metrics.get("is_trained", False) else "training",
                "total_processed": metrics.get("total_processed", 0),
                "anomalies_detected": metrics.get("anomalies_detected", 0),
                "current_threshold": metrics.get("current_threshold", 0),
                "buffer_size": metrics.get("buffer_size", 0),
                "average_processing_time": metrics.get("average_processing_time", 0),
                "recent_anomaly_rate": metrics.get("recent_anomaly_rate", 0),
                "model_updates": metrics.get("model_updates", 0),
            }

        return detector_data

    def _generate_visualizations(self) -> dict[str, str]:
        """Generate Plotly visualizations as JSON."""

        visualizations = {}

        # Alert rate over time
        visualizations["alert_rate_chart"] = self._create_alert_rate_chart()

        # Anomaly score distribution
        visualizations["score_distribution"] = self._create_score_distribution_chart()

        # Threat level pie chart
        visualizations["threat_level_pie"] = self._create_threat_level_pie_chart()

        # Detector performance comparison
        visualizations["detector_performance"] = (
            self._create_detector_performance_chart()
        )

        # Real-time metrics dashboard
        visualizations["realtime_metrics"] = self._create_realtime_metrics_chart()

        return visualizations

    def _create_alert_rate_chart(self) -> str:
        """Create alert rate over time chart."""

        if not self.metric_history["alert_rate"]:
            return self._empty_chart("No alert rate data available")

        data = list(self.metric_history["alert_rate"])[-50:]  # Last 50 points

        timestamps = [point["timestamp"] for point in data]
        values = [point["value"] for point in data]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name="Alert Rate (per minute)",
                line=dict(color="red", width=2),
                marker=dict(size=4),
            )
        )

        fig.update_layout(
            title="Real-time Alert Rate",
            xaxis_title="Time",
            yaxis_title="Alerts per Minute",
            showlegend=True,
            height=400,
        )

        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    def _create_score_distribution_chart(self) -> str:
        """Create anomaly score distribution chart."""

        if not self.metric_history["anomaly_scores"]:
            return self._empty_chart("No anomaly score data available")

        data = list(self.metric_history["anomaly_scores"])[-100:]  # Last 100 points
        scores = [point["value"] for point in data]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=20,
                name="Anomaly Scores",
                marker_color="orange",
                opacity=0.7,
            )
        )

        fig.update_layout(
            title="Anomaly Score Distribution",
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            showlegend=False,
            height=400,
        )

        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    def _create_threat_level_pie_chart(self) -> str:
        """Create threat level distribution pie chart."""

        if not self.threat_level_counts:
            return self._empty_chart("No threat level data available")

        labels = list(self.threat_level_counts.keys())
        values = list(self.threat_level_counts.values())

        colors = {
            "normal": "#90EE90",
            "low": "#FFD700",
            "medium": "#FFA500",
            "high": "#FF4500",
            "critical": "#FF0000",
        }

        marker_colors = [colors.get(label, "#808080") for label in labels]

        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=marker_colors,
                textinfo="label+percent",
            )
        )

        fig.update_layout(title="Threat Level Distribution", height=400)

        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    def _create_detector_performance_chart(self) -> str:
        """Create detector performance comparison chart."""

        if not self.detector_metrics:
            return self._empty_chart("No detector data available")

        detector_ids = list(self.detector_metrics.keys())
        processing_times = [
            self.detector_metrics[d_id].get("average_processing_time", 0)
            for d_id in detector_ids
        ]
        anomaly_rates = [
            self.detector_metrics[d_id].get("recent_anomaly_rate", 0)
            for d_id in detector_ids
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Processing Time", "Anomaly Rate"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        fig.add_trace(
            go.Bar(x=detector_ids, y=processing_times, name="Processing Time (ms)"),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=detector_ids,
                y=anomaly_rates,
                name="Anomaly Rate",
                marker_color="orange",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="Detector Performance Comparison", height=400, showlegend=False
        )

        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    def _create_realtime_metrics_chart(self) -> str:
        """Create real-time metrics dashboard."""

        # Create multi-metric chart
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Data Points Processed",
                "Active Detectors",
                "Total Alerts",
                "System Health",
            ),
        )

        # Data points processed
        if self.metric_history["total_data_points"]:
            data = list(self.metric_history["total_data_points"])[-20:]
            timestamps = [point["timestamp"] for point in data]
            values = [point["value"] for point in data]

            fig.add_trace(
                go.Scatter(x=timestamps, y=values, mode="lines", name="Data Points"),
                row=1,
                col=1,
            )

        # Active detectors
        if self.metric_history["active_detectors"]:
            data = list(self.metric_history["active_detectors"])[-20:]
            timestamps = [point["timestamp"] for point in data]
            values = [point["value"] for point in data]

            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=values, mode="lines+markers", name="Detectors"
                ),
                row=1,
                col=2,
            )

        # Total alerts
        if self.metric_history["total_alerts"]:
            data = list(self.metric_history["total_alerts"])[-20:]
            timestamps = [point["timestamp"] for point in data]
            values = [point["value"] for point in data]

            fig.add_trace(
                go.Scatter(x=timestamps, y=values, mode="lines", name="Alerts"),
                row=2,
                col=1,
            )

        # System health indicator
        health_score = self._calculate_system_health_score()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Health Score"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Real-time System Metrics", height=600, showlegend=False
        )

        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    def _empty_chart(self, message: str) -> str:
        """Create an empty chart with a message."""

        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )

        fig.update_layout(
            title="No Data Available",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)

    def _get_highest_recent_threat_level(self) -> str:
        """Get the highest threat level from recent alerts."""

        if not self.alert_history:
            return "normal"

        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history if alert.timestamp >= recent_cutoff
        ]

        if not recent_alerts:
            return "normal"

        threat_levels = [alert.threat_level for alert in recent_alerts]

        # Order by severity
        level_order = {
            AnomalyThreatLevel.NORMAL: 0,
            AnomalyThreatLevel.LOW: 1,
            AnomalyThreatLevel.MEDIUM: 2,
            AnomalyThreatLevel.HIGH: 3,
            AnomalyThreatLevel.CRITICAL: 4,
        }

        highest_level = max(threat_levels, key=lambda x: level_order[x])
        return highest_level.value

    def _determine_system_status(self) -> str:
        """Determine overall system status."""

        highest_threat = self._get_highest_recent_threat_level()
        alert_rate = self._calculate_current_alert_rate()
        active_detectors = len(self.detector_metrics)

        if highest_threat == "critical" or alert_rate > 10:
            return "critical"
        elif highest_threat == "high" or alert_rate > 5:
            return "warning"
        elif active_detectors == 0:
            return "offline"
        else:
            return "healthy"

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""

        score = 100.0

        # Deduct for high alert rates
        alert_rate = self._calculate_current_alert_rate()
        if alert_rate > 5:
            score -= min(30, alert_rate * 3)

        # Deduct for high threat levels
        highest_threat = self._get_highest_recent_threat_level()
        threat_penalties = {
            "critical": 40,
            "high": 25,
            "medium": 15,
            "low": 5,
            "normal": 0,
        }
        score -= threat_penalties.get(highest_threat, 0)

        # Deduct for inactive detectors
        if len(self.detector_metrics) == 0:
            score -= 50

        return max(0, score)

    def get_alert_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get alert summary for specified time period."""

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history if alert.timestamp >= cutoff_time
        ]

        if not recent_alerts:
            return {
                "total_alerts": 0,
                "threat_distribution": {},
                "top_sources": [],
                "average_score": 0.0,
            }

        # Calculate statistics
        threat_distribution = defaultdict(int)
        source_counts = defaultdict(int)
        scores = []

        for alert in recent_alerts:
            threat_distribution[alert.threat_level.value] += 1
            source_counts[alert.data_point.source_id] += 1
            scores.append(alert.anomaly_score)

        # Top sources
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "total_alerts": len(recent_alerts),
            "threat_distribution": dict(threat_distribution),
            "top_sources": [
                {"source": source, "count": count} for source, count in top_sources
            ],
            "average_score": np.mean(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "time_period_hours": hours,
        }

    def export_dashboard_config(self) -> dict[str, Any]:
        """Export dashboard configuration."""

        return {
            "update_interval": self.update_interval,
            "max_alert_history": self.alert_history.maxlen,
            "max_metric_history": 500,  # Default maxlen for metric history
            "last_update": self.last_update.isoformat(),
            "active_detectors": list(self.detector_metrics.keys()),
        }
