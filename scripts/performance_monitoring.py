#!/usr/bin/env python3
"""Performance monitoring management script."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynomaly.presentation.web.performance_alerts import (
    AlertSeverity,
    MetricType,
    PerformanceMetric,
    performance_monitor,
)
from pynomaly.presentation.web.performance_integration import (
    create_test_alert,
    export_prometheus_metrics,
    get_health_check,
    initialize_performance_monitoring,
    monitoring_integration,
    shutdown_performance_monitoring,
)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance monitoring management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start monitoring
    start_parser = subparsers.add_parser("start", help="Start performance monitoring")
    start_parser.add_argument(
        "--config",
        "-c",
        default="config/monitoring/performance_alerts.json",
        help="Configuration file path",
    )

    # Stop monitoring
    subparsers.add_parser("stop", help="Stop performance monitoring")

    # Status
    subparsers.add_parser("status", help="Show monitoring status")

    # Test alert
    test_parser = subparsers.add_parser("test-alert", help="Create test alert")
    test_parser.add_argument(
        "--severity",
        "-s",
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Alert severity",
    )

    # Add metric
    metric_parser = subparsers.add_parser("add-metric", help="Add test metric")
    metric_parser.add_argument(
        "--type",
        "-t",
        choices=[m.value for m in MetricType],
        required=True,
        help="Metric type",
    )
    metric_parser.add_argument(
        "--value", "-v", type=float, required=True, help="Metric value"
    )
    metric_parser.add_argument("--tags", help="Tags as JSON string")

    # List alerts
    alerts_parser = subparsers.add_parser("alerts", help="List alerts")
    alerts_parser.add_argument(
        "--active", "-a", action="store_true", help="Show only active alerts"
    )
    alerts_parser.add_argument(
        "--history", "-h", type=int, default=24, help="History hours"
    )

    # Export metrics
    export_parser = subparsers.add_parser("export", help="Export metrics")
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["prometheus", "json"],
        default="json",
        help="Export format",
    )
    export_parser.add_argument("--output", "-o", help="Output file")

    # Health check
    subparsers.add_parser("health", help="Health check")

    # Reset
    subparsers.add_parser("reset", help="Reset monitoring data")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "start":
        start_monitoring(args.config)
    elif args.command == "stop":
        stop_monitoring()
    elif args.command == "status":
        show_status()
    elif args.command == "test-alert":
        create_test_alert_cmd(args.severity)
    elif args.command == "add-metric":
        add_metric(args.type, args.value, args.tags)
    elif args.command == "alerts":
        list_alerts(args.active, args.history)
    elif args.command == "export":
        export_metrics(args.format, args.output)
    elif args.command == "health":
        health_check()
    elif args.command == "reset":
        reset_monitoring()


def start_monitoring(config_path: str):
    """Start performance monitoring."""
    print("Starting performance monitoring...")

    # Update config path
    monitoring_integration.config_path = config_path

    # Initialize monitoring
    initialize_performance_monitoring()

    if monitoring_integration.monitoring_started:
        print("✅ Performance monitoring started successfully")
        print(f"📊 Configuration: {config_path}")
        print(f"🔧 Handlers configured: {monitoring_integration.handlers_configured}")
        print(f"📈 Thresholds loaded: {len(performance_monitor.thresholds)}")
    else:
        print("❌ Failed to start performance monitoring")


def stop_monitoring():
    """Stop performance monitoring."""
    print("Stopping performance monitoring...")

    shutdown_performance_monitoring()

    if not monitoring_integration.monitoring_started:
        print("✅ Performance monitoring stopped successfully")
    else:
        print("❌ Failed to stop performance monitoring")


def show_status():
    """Show monitoring status."""
    status = monitoring_integration.get_system_status()

    print("📊 Performance Monitoring Status")
    print("=" * 40)
    print(f"Monitoring Active: {'✅' if status['monitoring_active'] else '❌'}")
    print(f"Handlers Configured: {'✅' if status['handlers_configured'] else '❌'}")
    print()

    perf_status = status["performance_monitor_status"]
    print(f"Performance Monitor Active: {'✅' if perf_status['active'] else '❌'}")
    print(f"Metrics Buffer Size: {perf_status['metrics_buffer_size']}")
    print(f"Active Alerts: {perf_status['active_alerts']}")
    print(f"Total Thresholds: {perf_status['total_thresholds']}")
    print(f"Alert Handlers: {perf_status['alert_handlers']}")
    print()

    stats = status["statistics"]
    print("📈 Statistics")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"Active Alerts: {stats['active_alerts_count']}")
    print(f"Average Resolution Time: {stats['average_resolution_time']:.2f}s")
    print()

    print("🚨 Alerts by Severity")
    for severity, count in stats["alerts_by_severity"].items():
        print(f"  {severity.title()}: {count}")
    print()

    print("📊 Alerts by Type")
    for metric_type, count in stats["alerts_by_type"].items():
        if count > 0:
            print(f"  {metric_type.replace('_', ' ').title()}: {count}")


def create_test_alert_cmd(severity: str):
    """Create test alert."""
    print(f"Creating test alert with severity: {severity}")

    severity_enum = AlertSeverity(severity)
    alert = create_test_alert(severity_enum)

    print(f"✅ Test alert created: {alert.alert_id}")
    print(f"📝 Message: {alert.message}")
    print(f"🔥 Severity: {alert.severity.value}")
    print(f"📅 Triggered: {alert.triggered_at}")


def add_metric(metric_type: str, value: float, tags_json: str = None):
    """Add test metric."""
    print(f"Adding metric: {metric_type} = {value}")

    # Parse tags
    tags = {}
    if tags_json:
        try:
            tags = json.loads(tags_json)
        except json.JSONDecodeError:
            print("❌ Invalid tags JSON")
            return

    # Create metric
    metric = PerformanceMetric(
        metric_type=MetricType(metric_type),
        value=value,
        timestamp=datetime.now(),
        tags=tags,
        metadata={"source": "test_script"},
    )

    # Record metric
    performance_monitor.record_metric(metric)

    print("✅ Metric added successfully")
    print(f"📊 Buffer size: {len(performance_monitor.metrics_buffer)}")


def list_alerts(active_only: bool, history_hours: int):
    """List alerts."""
    if active_only:
        alerts = performance_monitor.get_active_alerts()
        print(f"🚨 Active Alerts ({len(alerts)})")
    else:
        alerts = performance_monitor.get_alert_history(history_hours)
        print(f"📜 Alert History - Last {history_hours} hours ({len(alerts)})")

    print("=" * 60)

    if not alerts:
        print("No alerts found")
        return

    for alert in alerts:
        status = "🔴 ACTIVE" if not alert.resolved_at else "✅ RESOLVED"
        duration = ""
        if alert.resolved_at:
            duration = (
                f" ({(alert.resolved_at - alert.triggered_at).total_seconds():.1f}s)"
            )

        print(f"{status} {alert.severity.value.upper()} - {alert.message}")
        print(f"  📅 {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}{duration}")
        print(
            f"  🏷️  {alert.metric_type.value} | {alert.current_value} > {alert.threshold_value}"
        )
        if alert.tags:
            print(f"  🔖 Tags: {alert.tags}")
        print()


def export_metrics(format_type: str, output_file: str = None):
    """Export metrics."""
    if format_type == "prometheus":
        if not output_file:
            output_file = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prom"

        export_prometheus_metrics(output_file)
        print(f"✅ Metrics exported to {output_file} (Prometheus format)")

    elif format_type == "json":
        if not output_file:
            output_file = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Export as JSON
        data = {
            "timestamp": datetime.now().isoformat(),
            "status": monitoring_integration.get_system_status(),
            "metrics": [
                {
                    "metric_type": m.metric_type.value,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "metadata": m.metadata,
                }
                for m in performance_monitor.metrics_buffer
            ],
            "active_alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity.value,
                    "message": a.message,
                    "triggered_at": a.triggered_at.isoformat(),
                    "current_value": a.current_value,
                    "threshold_value": a.threshold_value,
                    "tags": a.tags,
                    "metadata": a.metadata,
                }
                for a in performance_monitor.get_active_alerts()
            ],
            "alert_history": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity.value,
                    "message": a.message,
                    "triggered_at": a.triggered_at.isoformat(),
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                    "current_value": a.current_value,
                    "threshold_value": a.threshold_value,
                    "tags": a.tags,
                    "metadata": a.metadata,
                }
                for a in performance_monitor.get_alert_history(24)
            ],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Metrics exported to {output_file} (JSON format)")


def health_check():
    """Perform health check."""
    health = get_health_check()

    print(f"🏥 Health Check - {health['status'].upper()}")
    print(f"📅 {health['timestamp']}")
    print()

    if health["status"] == "error":
        print(f"❌ Error: {health['error']}")
        return

    print(f"🔄 Monitoring Active: {'✅' if health['monitoring_active'] else '❌'}")
    print()

    print("📊 Metrics")
    for key, value in health["metrics"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()

    print("🔍 Checks")
    for check, status in health["checks"].items():
        emoji = "✅" if status == "pass" else "❌"
        print(f"  {emoji} {check.replace('_', ' ').title()}: {status}")


def reset_monitoring():
    """Reset monitoring data."""
    print("Resetting monitoring data...")

    # Clear metrics buffer
    performance_monitor.metrics_buffer.clear()

    # Clear active alerts
    performance_monitor.active_alerts.clear()

    # Clear alert history
    performance_monitor.alert_history.clear()

    # Reset statistics
    performance_monitor.performance_stats = {
        "total_alerts": 0,
        "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
        "alerts_by_type": {metric.value: 0 for metric in MetricType},
        "average_resolution_time": 0,
    }

    print("✅ Monitoring data reset successfully")


if __name__ == "__main__":
    main()
