#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for Pynomaly Detection.

Creates comprehensive dashboards for monitoring production performance
with real-time metrics, alerts, and optimization insights.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""
    timestamp: float
    component: str
    throughput: float
    latency: float
    memory_mb: float
    cpu_percent: float
    error_rate: float
    anomaly_rate: float
    accuracy: Optional[float] = None

@dataclass
class DashboardAlert:
    """Dashboard alert."""
    timestamp: float
    severity: str  # "info", "warning", "error", "critical"
    component: str
    message: str
    metric_value: float
    threshold: float

class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""
    
    def __init__(self, max_history_points: int = 1000):
        """Initialize performance dashboard.
        
        Args:
            max_history_points: Maximum number of historical data points to keep
        """
        self.max_history_points = max_history_points
        
        # Data storage
        self.metrics_history: List[DashboardMetrics] = []
        self.alerts_history: List[DashboardAlert] = []
        self.component_status: Dict[str, str] = {}  # "healthy", "warning", "error"
        
        # Dashboard configuration
        self.update_interval = 5.0  # seconds
        self.alert_thresholds = {
            "throughput_min": 1000,
            "latency_max": 1.0,
            "memory_max": 1000,
            "cpu_max": 80.0,
            "error_rate_max": 0.05,
            "anomaly_rate_max": 0.20
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        print("📊 Performance dashboard initialized")
    
    def add_metrics(self, metrics: DashboardMetrics):
        """Add new metrics to dashboard."""
        self.metrics_history.append(metrics)
        
        # Maintain history size
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history = self.metrics_history[-self.max_history_points:]
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update component status
        self._update_component_status(metrics)
    
    def _check_alerts(self, metrics: DashboardMetrics):
        """Check if metrics trigger any alerts."""
        alerts = []
        current_time = time.time()
        
        # Throughput alert
        if metrics.throughput < self.alert_thresholds["throughput_min"]:
            severity = "critical" if metrics.throughput < self.alert_thresholds["throughput_min"] * 0.5 else "warning"
            alerts.append(DashboardAlert(
                timestamp=current_time,
                severity=severity,
                component=metrics.component,
                message=f"Low throughput: {metrics.throughput:.0f} samples/s",
                metric_value=metrics.throughput,
                threshold=self.alert_thresholds["throughput_min"]
            ))
        
        # Latency alert
        if metrics.latency > self.alert_thresholds["latency_max"]:
            severity = "critical" if metrics.latency > self.alert_thresholds["latency_max"] * 2 else "warning"
            alerts.append(DashboardAlert(
                timestamp=current_time,
                severity=severity,
                component=metrics.component,
                message=f"High latency: {metrics.latency:.3f}s",
                metric_value=metrics.latency,
                threshold=self.alert_thresholds["latency_max"]
            ))
        
        # Memory alert
        if metrics.memory_mb > self.alert_thresholds["memory_max"]:
            severity = "error" if metrics.memory_mb > self.alert_thresholds["memory_max"] * 1.5 else "warning"
            alerts.append(DashboardAlert(
                timestamp=current_time,
                severity=severity,
                component=metrics.component,
                message=f"High memory usage: {metrics.memory_mb:.1f}MB",
                metric_value=metrics.memory_mb,
                threshold=self.alert_thresholds["memory_max"]
            ))
        
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds["error_rate_max"]:
            severity = "critical" if metrics.error_rate > self.alert_thresholds["error_rate_max"] * 2 else "error"
            alerts.append(DashboardAlert(
                timestamp=current_time,
                severity=severity,
                component=metrics.component,
                message=f"High error rate: {metrics.error_rate:.2%}",
                metric_value=metrics.error_rate,
                threshold=self.alert_thresholds["error_rate_max"]
            ))
        
        # Anomaly rate alert (unusual anomaly detection rates)
        if metrics.anomaly_rate > self.alert_thresholds["anomaly_rate_max"]:
            severity = "warning"  # Anomaly rate changes are informational
            alerts.append(DashboardAlert(
                timestamp=current_time,
                severity=severity,
                component=metrics.component,
                message=f"High anomaly rate: {metrics.anomaly_rate:.2%}",
                metric_value=metrics.anomaly_rate,
                threshold=self.alert_thresholds["anomaly_rate_max"]
            ))
        
        # Add alerts to history
        for alert in alerts:
            self.alerts_history.append(alert)
            print(f"🚨 {alert.severity.upper()}: {alert.component} - {alert.message}")
        
        # Maintain alert history
        if len(self.alerts_history) > 100:
            self.alerts_history = self.alerts_history[-100:]
    
    def _update_component_status(self, metrics: DashboardMetrics):
        """Update component health status."""
        component = metrics.component
        
        # Count recent alerts for this component
        recent_time = time.time() - 300  # Last 5 minutes
        recent_alerts = [a for a in self.alerts_history 
                        if a.component == component and a.timestamp > recent_time]
        
        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
        error_alerts = [a for a in recent_alerts if a.severity == "error"] 
        warning_alerts = [a for a in recent_alerts if a.severity == "warning"]
        
        if critical_alerts:
            self.component_status[component] = "critical"
        elif error_alerts:
            self.component_status[component] = "error"
        elif warning_alerts:
            self.component_status[component] = "warning"
        else:
            self.component_status[component] = "healthy"
    
    def get_dashboard_data(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get dashboard data for specified time window."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        # Filter metrics to time window
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        recent_alerts = [a for a in self.alerts_history if a.timestamp > cutoff_time]
        
        # Group metrics by component
        component_data = {}
        for metric in recent_metrics:
            if metric.component not in component_data:
                component_data[metric.component] = []
            component_data[metric.component].append(metric)
        
        # Calculate summary statistics
        summary_stats = {}
        for component, metrics in component_data.items():
            if metrics:
                throughputs = [m.throughput for m in metrics]
                latencies = [m.latency for m in metrics]
                memories = [m.memory_mb for m in metrics]
                error_rates = [m.error_rate for m in metrics]
                
                summary_stats[component] = {
                    "avg_throughput": np.mean(throughputs),
                    "avg_latency": np.mean(latencies),
                    "avg_memory_mb": np.mean(memories),
                    "avg_error_rate": np.mean(error_rates),
                    "min_throughput": np.min(throughputs),
                    "max_latency": np.max(latencies),
                    "max_memory_mb": np.max(memories),
                    "max_error_rate": np.max(error_rates),
                    "data_points": len(metrics),
                    "status": self.component_status.get(component, "unknown")
                }
        
        # Alert summary
        alert_summary = {
            "total_alerts": len(recent_alerts),
            "by_severity": {
                "critical": len([a for a in recent_alerts if a.severity == "critical"]),
                "error": len([a for a in recent_alerts if a.severity == "error"]),
                "warning": len([a for a in recent_alerts if a.severity == "warning"]),
                "info": len([a for a in recent_alerts if a.severity == "info"])
            },
            "by_component": {}
        }
        
        for component in component_data.keys():
            component_alerts = [a for a in recent_alerts if a.component == component]
            alert_summary["by_component"][component] = len(component_alerts)
        
        return {
            "timestamp": time.time(),
            "time_window_minutes": time_window_minutes,
            "component_summary": summary_stats,
            "alert_summary": alert_summary,
            "recent_alerts": recent_alerts[-20:],  # Last 20 alerts
            "system_health": self._calculate_system_health(summary_stats)
        }
    
    def _calculate_system_health(self, summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health score."""
        if not summary_stats:
            return {"score": 0, "status": "unknown", "factors": []}
        
        health_score = 100
        health_factors = []
        
        # Analyze each component
        for component, stats in summary_stats.items():
            component_score = 100
            
            # Throughput factor
            if stats["avg_throughput"] < self.alert_thresholds["throughput_min"]:
                reduction = min(50, (self.alert_thresholds["throughput_min"] - stats["avg_throughput"]) / 100)
                component_score -= reduction
                health_factors.append(f"{component}: Low throughput (-{reduction:.0f} points)")
            
            # Latency factor
            if stats["avg_latency"] > self.alert_thresholds["latency_max"]:
                reduction = min(30, (stats["avg_latency"] - self.alert_thresholds["latency_max"]) * 30)
                component_score -= reduction
                health_factors.append(f"{component}: High latency (-{reduction:.0f} points)")
            
            # Memory factor
            if stats["avg_memory_mb"] > self.alert_thresholds["memory_max"]:
                reduction = min(20, (stats["avg_memory_mb"] - self.alert_thresholds["memory_max"]) / 50)
                component_score -= reduction
                health_factors.append(f"{component}: High memory usage (-{reduction:.0f} points)")
            
            # Error rate factor
            if stats["avg_error_rate"] > self.alert_thresholds["error_rate_max"]:
                reduction = min(40, stats["avg_error_rate"] * 1000)
                component_score -= reduction
                health_factors.append(f"{component}: High error rate (-{reduction:.0f} points)")
            
            # Update overall health with component score
            health_score = min(health_score, component_score)
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "warning"
        elif health_score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "score": max(0, health_score),
            "status": status,
            "factors": health_factors
        }
    
    def create_text_dashboard(self, time_window_minutes: int = 60) -> str:
        """Create text-based dashboard output."""
        data = self.get_dashboard_data(time_window_minutes)
        
        dashboard_text = []
        dashboard_text.append("=" * 80)
        dashboard_text.append("📊 PYNOMALY DETECTION - PERFORMANCE DASHBOARD")
        dashboard_text.append("=" * 80)
        
        # System health overview
        health = data["system_health"]
        health_emoji = {
            "excellent": "🟢",
            "good": "🟡", 
            "warning": "🟠",
            "poor": "🔴",
            "critical": "🆘"
        }.get(health["status"], "⚪")
        
        dashboard_text.append(f"\n{health_emoji} **System Health: {health['status'].upper()} ({health['score']:.0f}/100)**")
        
        if health["factors"]:
            dashboard_text.append("   Health Factors:")
            for factor in health["factors"][:3]:
                dashboard_text.append(f"   • {factor}")
        
        # Alert summary
        alert_summary = data["alert_summary"]
        dashboard_text.append(f"\n🚨 **Alerts (Last {time_window_minutes}min):** {alert_summary['total_alerts']} total")
        
        if alert_summary["total_alerts"] > 0:
            dashboard_text.append(f"   Critical: {alert_summary['by_severity']['critical']}, "
                                f"Errors: {alert_summary['by_severity']['error']}, "
                                f"Warnings: {alert_summary['by_severity']['warning']}")
        
        # Component status
        dashboard_text.append(f"\n📈 **Component Performance:**")
        dashboard_text.append("-" * 80)
        
        header = f"{'Component':<20} {'Status':<10} {'Throughput':<12} {'Latency':<10} {'Memory':<12} {'Errors':<8}"
        dashboard_text.append(header)
        dashboard_text.append("-" * 80)
        
        for component, stats in data["component_summary"].items():
            status_emoji = {
                "healthy": "🟢",
                "warning": "🟡",
                "error": "🔴",
                "critical": "🆘"
            }.get(stats["status"], "⚪")
            
            row = (f"{component:<20} "
                   f"{status_emoji}{stats['status']:<9} "
                   f"{stats['avg_throughput']:>8.0f}/s   "
                   f"{stats['avg_latency']:>6.3f}s   "
                   f"{stats['avg_memory_mb']:>8.1f}MB   "
                   f"{stats['avg_error_rate']:>6.2%}")
            dashboard_text.append(row)
        
        # Recent alerts
        if data["recent_alerts"]:
            dashboard_text.append(f"\n🚨 **Recent Alerts:**")
            for alert in data["recent_alerts"][-5:]:  # Last 5 alerts
                alert_time = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                alert_emoji = {
                    "critical": "🆘",
                    "error": "🔴",
                    "warning": "🟡", 
                    "info": "🔵"
                }.get(alert.severity, "⚪")
                
                dashboard_text.append(f"   {alert_emoji} {alert_time} - {alert.component}: {alert.message}")
        
        # Performance trends (if enough data)
        if len(self.metrics_history) > 10:
            dashboard_text.append(f"\n📊 **Performance Trends:**")
            
            # Calculate trends for each component
            for component in data["component_summary"].keys():
                component_metrics = [m for m in self.metrics_history[-50:] if m.component == component]
                if len(component_metrics) > 5:
                    recent_throughput = np.mean([m.throughput for m in component_metrics[-10:]])
                    earlier_throughput = np.mean([m.throughput for m in component_metrics[-20:-10]])
                    
                    if earlier_throughput > 0:
                        trend = (recent_throughput - earlier_throughput) / earlier_throughput
                        trend_emoji = "📈" if trend > 0.05 else "📉" if trend < -0.05 else "➡️"
                        dashboard_text.append(f"   {trend_emoji} {component}: {trend:+.1%} throughput trend")
        
        # Recommendations
        dashboard_text.append(f"\n💡 **Recommendations:**")
        recommendations = []
        
        for component, stats in data["component_summary"].items():
            if stats["status"] in ["error", "critical"]:
                recommendations.append(f"Investigate {component} performance issues immediately")
            elif stats["avg_throughput"] < self.alert_thresholds["throughput_min"]:
                recommendations.append(f"Optimize {component} throughput (current: {stats['avg_throughput']:.0f}/s)")
            elif stats["avg_latency"] > self.alert_thresholds["latency_max"]:
                recommendations.append(f"Reduce {component} latency (current: {stats['avg_latency']:.3f}s)")
        
        if not recommendations:
            recommendations = ["System performing optimally - no immediate actions needed"]
        
        for i, rec in enumerate(recommendations[:3], 1):
            dashboard_text.append(f"   {i}. {rec}")
        
        dashboard_text.append("\n" + "=" * 80)
        dashboard_text.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dashboard_text.append("=" * 80)
        
        return "\n".join(dashboard_text)
    
    def save_dashboard_html(self, filename: str = "performance_dashboard.html") -> str:
        """Save dashboard as HTML file."""
        data = self.get_dashboard_data()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly Detection - Performance Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; flex: 1; }}
        .health-excellent {{ background: #d4edda; border-left: 5px solid #28a745; }}
        .health-good {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
        .health-warning {{ background: #f8d7da; border-left: 5px solid #dc3545; }}
        .status-healthy {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
        .alert {{ margin: 10px 0; padding: 10px; border-radius: 3px; }}
        .alert-critical {{ background: #f8d7da; border-left: 3px solid #dc3545; }}
        .alert-warning {{ background: #fff3cd; border-left: 3px solid #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Pynomaly Detection - Performance Dashboard</h1>
        <p>Real-time monitoring and performance analytics</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card health-{data['system_health']['status']}">
            <h3>System Health</h3>
            <h2>{data['system_health']['score']:.0f}/100</h2>
            <p>Status: {data['system_health']['status'].upper()}</p>
        </div>
        <div class="metric-card">
            <h3>Total Alerts</h3>
            <h2>{data['alert_summary']['total_alerts']}</h2>
            <p>Last 60 minutes</p>
        </div>
        <div class="metric-card">
            <h3>Components</h3>
            <h2>{len(data['component_summary'])}</h2>
            <p>Monitored components</p>
        </div>
    </div>
    
    <h2>Component Performance</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Status</th>
            <th>Throughput</th>
            <th>Latency</th>
            <th>Memory</th>
            <th>Error Rate</th>
        </tr>
"""
        
        for component, stats in data["component_summary"].items():
            html_content += f"""
        <tr>
            <td>{component}</td>
            <td class="status-{stats['status']}">{stats['status'].upper()}</td>
            <td>{stats['avg_throughput']:,.0f}/s</td>
            <td>{stats['avg_latency']:.3f}s</td>
            <td>{stats['avg_memory_mb']:.1f}MB</td>
            <td>{stats['avg_error_rate']:.2%}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>Recent Alerts</h2>
"""
        
        if data["recent_alerts"]:
            for alert in data["recent_alerts"][-10:]:
                alert_time = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                html_content += f"""
    <div class="alert alert-{alert.severity}">
        <strong>{alert_time}</strong> - {alert.component}: {alert.message}
    </div>
"""
        else:
            html_content += "<p>No recent alerts</p>"
        
        html_content += f"""
    
    <div style="margin-top: 30px; text-align: center; color: #666;">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML dashboard saved: {filepath}")
        return str(filepath)
    
    def export_metrics_csv(self, filename: str = "performance_metrics.csv") -> str:
        """Export metrics data to CSV."""
        if not PANDAS_AVAILABLE:
            print("⚠️  Pandas not available - using manual CSV export")
            return self._manual_csv_export(filename)
        
        # Convert metrics to DataFrame
        metrics_data = []
        for metric in self.metrics_history:
            metrics_data.append({
                "timestamp": metric.timestamp,
                "datetime": datetime.fromtimestamp(metric.timestamp),
                "component": metric.component,
                "throughput": metric.throughput,
                "latency": metric.latency,
                "memory_mb": metric.memory_mb,
                "cpu_percent": metric.cpu_percent,
                "error_rate": metric.error_rate,
                "anomaly_rate": metric.anomaly_rate,
                "accuracy": metric.accuracy
            })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(filename, index=False)
        
        print(f"📊 Metrics exported to CSV: {filename}")
        return filename
    
    def _manual_csv_export(self, filename: str) -> str:
        """Manual CSV export without pandas."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'datetime', 'component', 'throughput', 'latency', 
                         'memory_mb', 'cpu_percent', 'error_rate', 'anomaly_rate', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in self.metrics_history:
                writer.writerow({
                    'timestamp': metric.timestamp,
                    'datetime': datetime.fromtimestamp(metric.timestamp).isoformat(),
                    'component': metric.component,
                    'throughput': metric.throughput,
                    'latency': metric.latency,
                    'memory_mb': metric.memory_mb,
                    'cpu_percent': metric.cpu_percent,
                    'error_rate': metric.error_rate,
                    'anomaly_rate': metric.anomaly_rate,
                    'accuracy': metric.accuracy
                })
        
        print(f"📊 Metrics exported to CSV: {filename}")
        return filename

def main():
    """Demo of performance dashboard."""
    print("📊 Pynomaly Detection - Performance Dashboard Demo")
    print("=" * 60)
    
    # Create dashboard
    dashboard = PerformanceDashboard()
    
    # Simulate performance data over time
    print("📊 Simulating performance monitoring...")
    
    components = ["CoreDetectionService", "AutoMLService", "BatchProcessor", "StreamingDetector"]
    
    # Simulate 2 hours of data (120 data points, 1 minute intervals)
    base_time = time.time() - 7200  # 2 hours ago
    
    for i in range(120):
        timestamp = base_time + (i * 60)  # 1 minute intervals
        
        for j, component in enumerate(components):
            # Simulate different performance patterns
            if component == "CoreDetectionService":
                # Stable high performance
                throughput = 15000 + np.random.normal(0, 1000)
                latency = 0.05 + np.random.normal(0, 0.01)
                memory = 200 + np.random.normal(0, 20)
                error_rate = 0.001 + np.random.normal(0, 0.0005)
                anomaly_rate = 0.05 + np.random.normal(0, 0.01)
                
            elif component == "AutoMLService":
                # Slower but comprehensive
                throughput = 2000 + np.random.normal(0, 300)
                latency = 0.8 + np.random.normal(0, 0.1)
                memory = 800 + np.random.normal(0, 100)
                error_rate = 0.005 + np.random.normal(0, 0.001)
                anomaly_rate = 0.08 + np.random.normal(0, 0.02)
                
            elif component == "BatchProcessor":
                # High throughput with occasional spikes
                base_throughput = 25000
                if i % 20 == 0:  # Spike every 20 minutes
                    base_throughput = 35000
                throughput = base_throughput + np.random.normal(0, 2000)
                latency = 2.0 + np.random.normal(0, 0.3)
                memory = 1500 + np.random.normal(0, 200)
                error_rate = 0.002 + np.random.normal(0, 0.0005)
                anomaly_rate = 0.12 + np.random.normal(0, 0.03)
                
            else:  # StreamingDetector
                # Real-time performance with some degradation over time
                degradation_factor = 1 - (i * 0.002)  # Gradual degradation
                throughput = (8000 * degradation_factor) + np.random.normal(0, 500)
                latency = 0.1 + (i * 0.001) + np.random.normal(0, 0.02)
                memory = 300 + (i * 2) + np.random.normal(0, 30)
                error_rate = 0.001 + (i * 0.00005) + np.random.normal(0, 0.0002)
                anomaly_rate = 0.06 + np.random.normal(0, 0.015)
            
            # Ensure realistic bounds
            throughput = max(100, throughput)
            latency = max(0.001, latency)
            memory = max(50, memory)
            error_rate = max(0, min(1, error_rate))
            anomaly_rate = max(0, min(1, anomaly_rate))
            
            metrics = DashboardMetrics(
                timestamp=timestamp,
                component=component,
                throughput=throughput,
                latency=latency,
                memory_mb=memory,
                cpu_percent=np.random.uniform(20, 80),
                error_rate=error_rate,
                anomaly_rate=anomaly_rate,
                accuracy=np.random.uniform(0.85, 0.98)
            )
            
            dashboard.add_metrics(metrics)
    
    # Display dashboard
    print("\n" + dashboard.create_text_dashboard())
    
    # Export data
    html_path = dashboard.save_dashboard_html()
    csv_path = dashboard.export_metrics_csv()
    
    # Save dashboard data as JSON
    dashboard_data = dashboard.get_dashboard_data()
    json_path = "dashboard_data.json"
    with open(json_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print(f"\n✅ Dashboard demo completed!")
    print(f"📄 HTML dashboard: {html_path}")
    print(f"📊 CSV export: {csv_path}")
    print(f"📋 JSON data: {json_path}")
    print(f"\n🚀 Ready for production monitoring deployment!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)