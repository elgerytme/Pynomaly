"""Comprehensive monitoring dashboard for model performance visualization."""

from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import base64
from io import BytesIO

from ..logging import get_logger
from .model_performance_monitor import get_model_performance_monitor, ModelPerformanceMetrics
from .alerting_system import get_alerting_system

logger = get_logger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Plotting libraries not available - dashboard will have limited functionality")


class MonitoringDashboard:
    """Comprehensive monitoring dashboard for model performance visualization."""
    
    def __init__(self):
        self.monitor = get_model_performance_monitor()
        self.alerting = get_alerting_system()
        
        # Configure plotting style
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'library') else 'seaborn')
            sns.set_palette("husl")
        
        logger.info("Monitoring dashboard initialized")
    
    def generate_dashboard_data(
        self,
        model_ids: Optional[List[str]] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        
        dashboard_data = {
            "generation_timestamp": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "system_status": self._get_system_status(),
            "alert_summary": self._get_alert_summary(hours),
            "performance_overview": self._get_performance_overview(model_ids, hours),
            "model_summaries": {},
            "trends": {},
            "charts": {}
        }
        
        # Get model-specific data
        if model_ids:
            for model_id in model_ids:
                dashboard_data["model_summaries"][model_id] = self.monitor.get_model_performance_summary(model_id, hours)
                dashboard_data["trends"][model_id] = self._get_model_trends(model_id, hours)
        
        # Generate charts if plotting available
        if PLOTTING_AVAILABLE:
            dashboard_data["charts"] = self._generate_charts(model_ids, hours)
        
        return dashboard_data
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        
        active_alerts = self.alerting.get_active_alerts()
        
        # Determine overall health
        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        warning_alerts = [a for a in active_alerts if a.severity == "warning"]
        
        if critical_alerts:
            health_status = "critical"
            health_color = "#d73502"
        elif warning_alerts:
            health_status = "warning"
            health_color = "#ff8c00"
        else:
            health_status = "healthy"
            health_color = "#28a745"
        
        return {
            "health_status": health_status,
            "health_color": health_color,
            "active_alerts_count": len(active_alerts),
            "critical_alerts_count": len(critical_alerts),
            "warning_alerts_count": len(warning_alerts),
            "monitored_models_count": len(self.monitor._performance_history),
            "uptime_status": "operational"  # Could be enhanced with actual uptime tracking
        }
    
    def _get_alert_summary(self, hours: int) -> Dict[str, Any]:
        """Get alert summary for the specified period."""
        
        alert_stats = self.alerting.get_alert_statistics(hours)
        active_alerts = self.alerting.get_active_alerts()
        
        # Recent alert trends
        recent_alerts_by_hour = {}
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        for i in range(hours):
            hour_start = cutoff_time + timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)
            
            hour_alerts = [
                alert for alert in active_alerts
                if hour_start <= alert.timestamp < hour_end
            ]
            
            recent_alerts_by_hour[hour_start.isoformat()] = len(hour_alerts)
        
        return {
            **alert_stats,
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "model_id": alert.model_id,
                    "metric_name": alert.metric_name,
                    "severity": alert.severity,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in active_alerts
            ],
            "hourly_alert_counts": recent_alerts_by_hour
        }
    
    def _get_performance_overview(
        self,
        model_ids: Optional[List[str]],
        hours: int
    ) -> Dict[str, Any]:
        """Get performance overview across all or specified models."""
        
        if model_ids:
            comparison = self.monitor.get_model_comparison(model_ids, hours)
            models_data = comparison["models"]
        else:
            # Get data for all monitored models
            all_model_ids = list(self.monitor._performance_history.keys())
            if all_model_ids:
                comparison = self.monitor.get_model_comparison(all_model_ids, hours)
                models_data = comparison["models"]
            else:
                models_data = {}
        
        # Calculate aggregate metrics
        if not models_data:
            return {"message": "No model data available"}
        
        # Aggregate across all models
        total_predictions = sum(
            data.get("total_predictions", 0) for data in models_data.values()
        )
        total_anomalies = sum(
            data.get("total_anomalies_detected", 0) for data in models_data.values()
        )
        
        # Calculate average metrics
        avg_metrics = {}
        metric_names = ["avg_precision", "avg_recall", "avg_f1_score", "avg_prediction_time_ms"]
        
        for metric_name in metric_names:
            values = [
                data.get(metric_name) for data in models_data.values()
                if data.get(metric_name) is not None
            ]
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
        
        return {
            "total_models": len(models_data),
            "total_predictions": total_predictions,
            "total_anomalies_detected": total_anomalies,
            "overall_anomaly_rate": total_anomalies / total_predictions if total_predictions > 0 else 0,
            "average_metrics": avg_metrics,
            "model_performance_distribution": self._get_performance_distribution(models_data)
        }
    
    def _get_performance_distribution(self, models_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get performance distribution across models."""
        
        distribution = {
            "precision_ranges": {"high": 0, "medium": 0, "low": 0},
            "prediction_time_ranges": {"fast": 0, "medium": 0, "slow": 0},
            "anomaly_rate_ranges": {"low": 0, "medium": 0, "high": 0}
        }
        
        for model_data in models_data.values():
            # Precision distribution
            precision = model_data.get("avg_precision")
            if precision is not None:
                if precision >= 0.8:
                    distribution["precision_ranges"]["high"] += 1
                elif precision >= 0.6:
                    distribution["precision_ranges"]["medium"] += 1
                else:
                    distribution["precision_ranges"]["low"] += 1
            
            # Prediction time distribution
            pred_time = model_data.get("avg_prediction_time_ms")
            if pred_time is not None:
                if pred_time <= 1000:
                    distribution["prediction_time_ranges"]["fast"] += 1
                elif pred_time <= 5000:
                    distribution["prediction_time_ranges"]["medium"] += 1
                else:
                    distribution["prediction_time_ranges"]["slow"] += 1
            
            # Anomaly rate distribution
            anomaly_rate = model_data.get("avg_anomaly_rate")
            if anomaly_rate is not None:
                if anomaly_rate <= 0.05:
                    distribution["anomaly_rate_ranges"]["low"] += 1
                elif anomaly_rate <= 0.15:
                    distribution["anomaly_rate_ranges"]["medium"] += 1
                else:
                    distribution["anomaly_rate_ranges"]["high"] += 1
        
        return distribution
    
    def _get_model_trends(self, model_id: str, hours: int) -> Dict[str, Any]:
        """Get performance trends for a specific model."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.monitor._lock:
            recent_metrics = [
                m for m in self.monitor._performance_history[model_id]
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        # Sort by timestamp
        recent_metrics.sort(key=lambda x: x.timestamp)
        
        # Calculate trends over time
        trends = {
            "timestamps": [m.timestamp.isoformat() for m in recent_metrics],
            "precision_trend": [m.precision for m in recent_metrics if m.precision is not None],
            "recall_trend": [m.recall for m in recent_metrics if m.recall is not None],
            "prediction_time_trend": [m.prediction_time_ms for m in recent_metrics],
            "anomaly_rate_trend": [m.anomaly_rate for m in recent_metrics],
            "confidence_trend": [m.prediction_confidence for m in recent_metrics if m.prediction_confidence is not None]
        }
        
        # Calculate trend directions
        trend_analysis = {}
        for trend_name, values in trends.items():
            if trend_name == "timestamps" or len(values) < 2:
                continue
            
            # Simple trend calculation (first half vs second half)
            mid_point = len(values) // 2
            first_half_avg = np.mean(values[:mid_point]) if mid_point > 0 else values[0]
            second_half_avg = np.mean(values[mid_point:])
            
            if second_half_avg > first_half_avg * 1.05:
                trend_analysis[trend_name] = "improving"
            elif second_half_avg < first_half_avg * 0.95:
                trend_analysis[trend_name] = "declining"
            else:
                trend_analysis[trend_name] = "stable"
        
        return {
            "data": trends,
            "analysis": trend_analysis,
            "data_points": len(recent_metrics)
        }
    
    def _generate_charts(
        self,
        model_ids: Optional[List[str]],
        hours: int
    ) -> Dict[str, str]:
        """Generate performance charts as base64-encoded images."""
        
        if not PLOTTING_AVAILABLE:
            return {"error": "Plotting libraries not available"}
        
        charts = {}
        
        try:
            # Generate performance timeline chart
            if model_ids:
                charts["performance_timeline"] = self._create_performance_timeline_chart(model_ids, hours)
                charts["metrics_comparison"] = self._create_metrics_comparison_chart(model_ids, hours)
            
            # Generate alert frequency chart
            charts["alert_frequency"] = self._create_alert_frequency_chart(hours)
            
            # Generate system overview chart
            charts["system_overview"] = self._create_system_overview_chart()
        
        except Exception as e:
            logger.error("Error generating charts", error=str(e))
            charts["error"] = f"Chart generation failed: {str(e)}"
        
        return charts
    
    def _create_performance_timeline_chart(self, model_ids: List[str], hours: int) -> str:
        """Create performance timeline chart."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Timeline', fontsize=16)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        for model_id in model_ids:
            with self.monitor._lock:
                metrics = [
                    m for m in self.monitor._performance_history[model_id]
                    if m.timestamp >= cutoff_time
                ]
            
            if not metrics:
                continue
            
            metrics.sort(key=lambda x: x.timestamp)
            timestamps = [m.timestamp for m in metrics]
            
            # Precision/Recall over time
            precisions = [m.precision for m in metrics if m.precision is not None]
            recalls = [m.recall for m in metrics if m.recall is not None]
            
            if precisions and len(precisions) == len(timestamps):
                axes[0, 0].plot(timestamps, precisions, label=f"{model_id} Precision", marker='o')
            if recalls and len(recalls) == len(timestamps):
                axes[0, 0].plot(timestamps, recalls, label=f"{model_id} Recall", marker='s')
            
            # Prediction time over time
            pred_times = [m.prediction_time_ms for m in metrics]
            if pred_times:
                axes[0, 1].plot(timestamps, pred_times, label=f"{model_id}", marker='o')
            
            # Anomaly rate over time
            anomaly_rates = [m.anomaly_rate for m in metrics]
            if anomaly_rates:
                axes[1, 0].plot(timestamps, anomaly_rates, label=f"{model_id}", marker='o')
            
            # F1 score over time
            f1_scores = [m.f1_score for m in metrics if m.f1_score is not None]
            if f1_scores and len(f1_scores) == len(timestamps):
                axes[1, 1].plot(timestamps, f1_scores, label=f"{model_id}", marker='o')
        
        # Configure subplots
        axes[0, 0].set_title('Precision & Recall')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Prediction Time')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Anomaly Rate')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours//12)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return chart_data
    
    def _create_metrics_comparison_chart(self, model_ids: List[str], hours: int) -> str:
        """Create metrics comparison chart."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Metrics Comparison', fontsize=16)
        
        # Get latest metrics for each model
        model_metrics = {}
        for model_id in model_ids:
            summary = self.monitor.get_model_performance_summary(model_id, hours)
            model_metrics[model_id] = summary
        
        # Prepare data for plotting
        models = list(model_metrics.keys())
        
        # Accuracy metrics
        precisions = [model_metrics[m].get("avg_precision", 0) for m in models]
        recalls = [model_metrics[m].get("avg_recall", 0) for m in models]
        f1_scores = [model_metrics[m].get("avg_f1_score", 0) for m in models]
        
        # Performance metrics
        pred_times = [model_metrics[m].get("avg_prediction_time_ms", 0) for m in models]
        
        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.25
        
        axes[0].bar(x - width, precisions, width, label='Precision', alpha=0.8)
        axes[0].bar(x, recalls, width, label='Recall', alpha=0.8)
        axes[0].bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        axes[0].set_title('Accuracy Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Models')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Performance comparison
        axes[1].bar(models, pred_times, alpha=0.8, color='orange')
        axes[1].set_title('Prediction Time')
        axes[1].set_ylabel('Time (ms)')
        axes[1].set_xlabel('Models')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return chart_data
    
    def _create_alert_frequency_chart(self, hours: int) -> str:
        """Create alert frequency chart."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Alert Analysis', fontsize=16)
        
        alert_stats = self.alerting.get_alert_statistics(hours)
        
        # Alert severity distribution
        if alert_stats.get("severity_distribution"):
            severities = list(alert_stats["severity_distribution"].keys())
            counts = list(alert_stats["severity_distribution"].values())
            colors = {'critical': '#d73502', 'warning': '#ff8c00', 'info': '#0066cc'}
            
            wedges, texts, autotexts = ax1.pie(
                counts, 
                labels=severities,
                autopct='%1.1f%%',
                colors=[colors.get(s, '#gray') for s in severities],
                startangle=90
            )
            ax1.set_title('Alert Severity Distribution')
        else:
            ax1.text(0.5, 0.5, 'No alerts in period', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Alert Severity Distribution')
        
        # Alerts by model
        if alert_stats.get("model_distribution"):
            models = list(alert_stats["model_distribution"].keys())
            alert_counts = list(alert_stats["model_distribution"].values())
            
            ax2.bar(models, alert_counts, alpha=0.8, color='coral')
            ax2.set_title('Alerts by Model')
            ax2.set_ylabel('Alert Count')
            ax2.set_xlabel('Models')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No alerts in period', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Alerts by Model')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return chart_data
    
    def _create_system_overview_chart(self) -> str:
        """Create system overview chart."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('System Overview', fontsize=16)
        
        # System health status
        system_status = self._get_system_status()
        health_colors = {'healthy': '#28a745', 'warning': '#ff8c00', 'critical': '#d73502'}
        
        ax1.pie([1], colors=[health_colors[system_status["health_status"]]], startangle=90)
        ax1.set_title(f'System Health: {system_status["health_status"].upper()}')
        
        # Active alerts summary
        alert_types = ['Critical', 'Warning', 'Info']
        alert_counts = [
            system_status["critical_alerts_count"],
            system_status["warning_alerts_count"],
            system_status["active_alerts_count"] - system_status["critical_alerts_count"] - system_status["warning_alerts_count"]
        ]
        
        ax2.bar(alert_types, alert_counts, color=['#d73502', '#ff8c00', '#0066cc'], alpha=0.8)
        ax2.set_title('Active Alerts')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # Monitored models
        model_count = system_status["monitored_models_count"]
        ax3.bar(['Monitored Models'], [model_count], color='#17a2b8', alpha=0.8)
        ax3.set_title('Model Coverage')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # System metrics summary (placeholder)
        metrics = ['Availability', 'Performance', 'Reliability']
        scores = [98.5, 95.2, 99.1]  # Mock data - could be replaced with real metrics
        
        ax4.bar(metrics, scores, color='#28a745', alpha=0.8)
        ax4.set_title('System Metrics (%)')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return chart_data
    
    def generate_html_dashboard(
        self,
        model_ids: Optional[List[str]] = None,
        hours: int = 24,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a complete HTML dashboard."""
        
        dashboard_data = self.generate_dashboard_data(model_ids, hours)
        
        html_content = self._create_html_template(dashboard_data)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info("HTML dashboard generated", output_path=str(output_path))
        
        return html_content
    
    def _create_html_template(self, data: Dict[str, Any]) -> str:
        """Create HTML template for the dashboard."""
        
        system_status = data["system_status"]
        alert_summary = data["alert_summary"]
        performance_overview = data["performance_overview"]
        
        # Create charts HTML
        charts_html = ""
        if "charts" in data and data["charts"]:
            for chart_name, chart_data in data["charts"].items():
                if chart_name != "error" and chart_data:
                    charts_html += f"""
                    <div class="chart-container">
                        <h3>{chart_name.replace('_', ' ').title()}</h3>
                        <img src="data:image/png;base64,{chart_data}" alt="{chart_name}" class="chart-image">
                    </div>
                    """
        
        # Create alerts HTML
        alerts_html = ""
        for alert in alert_summary.get("active_alerts", []):
            severity_class = f"alert-{alert['severity']}"
            alerts_html += f"""
            <div class="alert {severity_class}">
                <strong>{alert['model_id']}</strong> - {alert['metric_name']}
                <br>Current: {alert['current_value']:.4f}, Threshold: {alert['threshold_value']:.4f}
                <br><small>{alert['timestamp']}</small>
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Performance Dashboard</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .status-healthy {{ color: #28a745; }}
                .status-warning {{ color: #ff8c00; }}
                .status-critical {{ color: #d73502; }}
                
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                    padding: 8px 0;
                    border-bottom: 1px solid #eee;
                }}
                
                .alert {{
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    border-left: 4px solid;
                }}
                
                .alert-critical {{
                    background-color: #f8d7da;
                    border-left-color: #d73502;
                }}
                
                .alert-warning {{
                    background-color: #fff3cd;
                    border-left-color: #ff8c00;
                }}
                
                .alert-info {{
                    background-color: #d1ecf1;
                    border-left-color: #0066cc;
                }}
                
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                
                .chart-image {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                }}
                
                .stat-item {{
                    text-align: center;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 6px;
                }}
                
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #495057;
                }}
                
                .stat-label {{
                    font-size: 0.9em;
                    color: #6c757d;
                    margin-top: 5px;
                }}
                
                .refresh-info {{
                    text-align: center;
                    color: #6c757d;
                    font-size: 0.9em;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Performance Dashboard</h1>
                <p>Generated: {data['generation_timestamp']} | Period: {data['period_hours']} hours</p>
            </div>
            
            <div class="dashboard-grid">
                <div class="card">
                    <h2>System Status</h2>
                    <div class="metric">
                        <span>Health Status:</span>
                        <span class="status-{system_status['health_status']}">{system_status['health_status'].upper()}</span>
                    </div>
                    <div class="metric">
                        <span>Active Alerts:</span>
                        <span>{system_status['active_alerts_count']}</span>
                    </div>
                    <div class="metric">
                        <span>Critical Alerts:</span>
                        <span>{system_status['critical_alerts_count']}</span>
                    </div>
                    <div class="metric">
                        <span>Monitored Models:</span>
                        <span>{system_status['monitored_models_count']}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Performance Overview</h2>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{performance_overview.get('total_predictions', 0)}</div>
                            <div class="stat-label">Total Predictions</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{performance_overview.get('total_anomalies_detected', 0)}</div>
                            <div class="stat-label">Anomalies Detected</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{performance_overview.get('overall_anomaly_rate', 0):.3f}</div>
                            <div class="stat-label">Anomaly Rate</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Active Alerts</h2>
                    {alerts_html if alerts_html else '<p>No active alerts</p>'}
                </div>
            </div>
            
            {charts_html}
            
            <div class="refresh-info">
                Dashboard auto-refreshes every 5 minutes | Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
            </div>
            
            <script>
                // Auto-refresh every 5 minutes
                setTimeout(function() {{
                    window.location.reload();
                }}, 300000);
            </script>
        </body>
        </html>
        """


# Global instance
_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    
    global _dashboard
    
    if _dashboard is None:
        _dashboard = MonitoringDashboard()
    
    return _dashboard