"""Enterprise dashboard CLI commands for real-time monitoring and business intelligence.

This module provides command-line interface for enterprise dashboard operations,
alerting management, and business intelligence reporting.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from interfaces.application.services.enterprise_dashboard_service import (
    AlertPriority,
    DashboardMetricType,
    EnterpriseDashboardService,
    initialize_enterprise_dashboard,
)
from interfaces.application.services.enterprise_integration_service import (
    IntegrationConfig,
    initialize_enterprise_integration,
)

app = typer.Typer(
    name="enterprise",
    help="Enterprise dashboard and monitoring commands",
    no_args_is_help=True,
)
console = Console()


@app.command("dashboard")
def show_dashboard(
    refresh_interval: int = typer.Option(
        5, "--refresh", "-r", help="Refresh interval in seconds"
    ),
    duration: int = typer.Option(
        60, "--duration", "-d", help="Duration to display in seconds"
    ),
    export_file: str | None = typer.Option(
        None, "--export", help="Export data to file"
    ),
    format: str = typer.Option("json", "--format", help="Export format (json)"),
):
    """Display real-time enterprise dashboard.

    Shows executive summary, business measurements, operational status,
    and active alerts in a live updating dashboard.
    """
    try:
        dashboard_service = initialize_enterprise_dashboard()

        # Generate some sample data for demonstration
        _generate_sample_data(dashboard_service)

        if export_file:
            # Export mode
            data = dashboard_service.export_dashboard_data(format)
            Path(export_file).write_text(data)
            console.print(f"✅ Dashboard data exported to {export_file}")
            return

        # Live dashboard mode
        console.print("🎯 Enterprise Dashboard - Press Ctrl+C to exit")
        console.print()

        start_time = time.time()

        with Live(console=console, refresh_per_second=1) as live:
            while time.time() - start_time < duration:
                try:
                    layout = _create_dashboard_layout(dashboard_service)
                    live.update(layout)
                    time.sleep(refresh_interval)
                except KeyboardInterrupt:
                    break

        console.print("\n👋 Dashboard session ended")

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("summary")
def executive_summary():
    """Display executive summary for C-level reporting.

    Shows high-level KPIs, trends, and key insights
    for executive decision making.
    """
    try:
        dashboard_service = initialize_enterprise_dashboard()
        _generate_sample_data(dashboard_service)

        summary = dashboard_service.get_executive_summary()

        # Create executive summary display
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="content"))

        # Header
        header_text = Text("📊 EXECUTIVE SUMMARY", style="bold blue")
        header_text.append(
            f" - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim"
        )
        layout["header"].update(Panel(header_text, box=box.HEAVY))

        # Content
        content_layout = Layout()
        content_layout.split_row(Layout(name="left"), Layout(name="right"))

        # Left panel - Key Measurements
        measurements_table = Table(title="Key Performance Indicators", box=box.ROUNDED)
        measurements_table.add_column("Metric", style="cyan")
        measurements_table.add_column("Value", style="green")
        measurements_table.add_column("Status", style="yellow")

        measurements_table.add_row(
            "Total Detections Today", str(summary.total_processings_today), "✅ On Track"
        )
        measurements_table.add_row(
            "Anomalies Detected", str(summary.anomalies_detected_today), "📊 Normal"
        )
        measurements_table.add_row(
            "Processing Accuracy",
            f"{summary.accuracy_percentage:.1f}%",
            "🎯 Excellent" if summary.accuracy_percentage > 95 else "⚠️ Needs Attention",
        )
        measurements_table.add_row(
            "Cost Savings", f"${summary.cost_savings_usd:,.0f}", "💰 Positive ROI"
        )
        measurements_table.add_row(
            "Automation Coverage",
            f"{summary.automation_coverage_percent:.1f}%",
            "🤖 High Coverage",
        )
        measurements_table.add_row(
            "Avg Processing Time",
            f"{summary.avg_processing_time_seconds:.1f}s",
            "⚡ Fast",
        )

        layout["left"].update(measurements_table)

        # Right panel - Alerts and Insights
        right_layout = Layout()
        right_layout.split_column(
            Layout(name="alerts", size=8), Layout(name="insights")
        )

        # Alerts
        if summary.critical_alerts_count > 0:
            alerts_text = f"🚨 {summary.critical_alerts_count} Critical Alerts Active"
            alerts_style = "red"
        else:
            alerts_text = "✅ No Critical Alerts"
            alerts_style = "green"

        alerts_panel = Panel(
            Text(alerts_text, style=alerts_style), title="Alert Status", box=box.ROUNDED
        )
        right_layout["alerts"].update(alerts_panel)

        # Key Insights
        insights_text = Text()
        for i, insight in enumerate(summary.key_insights[:4], 1):
            insights_text.append(f"{i}. {insight}\n", style="white")

        insights_panel = Panel(insights_text, title="Key Insights", box=box.ROUNDED)
        right_layout["insights"].update(insights_panel)

        content_layout["left"].update(layout["left"])
        content_layout["right"].update(right_layout)
        layout["content"].update(content_layout)

        console.print(layout)

        # Compliance Score
        compliance_panel = Panel(
            f"Compliance Score: {summary.compliance_score:.1f}% | "
            f"Trend: {summary.trend_analysis.get('compliance', 'stable').title()}",
            title="🛡️ Compliance Status",
            style="blue",
        )
        console.print(compliance_panel)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("alerts")
def list_alerts(
    priority: str | None = typer.Option(
        None, "--priority", help="Filter by priority (low, medium, high, critical)"
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum alerts to display"),
    show_resolved: bool = typer.Option(
        False, "--resolved", help="Include resolved alerts"
    ),
):
    """List and manage dashboard alerts.

    Display active alerts with filtering options and
    management capabilities.
    """
    try:
        dashboard_service = initialize_enterprise_dashboard()
        _generate_sample_data(dashboard_service)

        # Get alerts
        alerts = list(dashboard_service.active_alerts.values())

        # Apply filters
        if priority:
            try:
                priority_filter = AlertPriority(priority.lower())
                alerts = [
                    alert for alert in alerts if alert.priority == priority_filter
                ]
            except ValueError:
                console.print(f"❌ Invalid priority: {priority}", style="red")
                raise typer.Exit(1)

        if not show_resolved:
            alerts = [alert for alert in alerts if not alert.resolved]

        # Apply limit
        alerts = alerts[:limit]

        if not alerts:
            console.print("✅ No alerts found matching criteria")
            return

        # Display alerts table
        table = Table(title=f"Dashboard Alerts ({len(alerts)} total)", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Title", style="white")
        table.add_column("Source", style="blue")
        table.add_column("Time", style="green")
        table.add_column("Status", style="magenta")

        for alert in alerts:
            priority_emoji = {
                AlertPriority.CRITICAL: "🚨",
                AlertPriority.HIGH: "⚠️",
                AlertPriority.MEDIUM: "📢",
                AlertPriority.LOW: "ℹ️",
            }.get(alert.priority, "📢")

            status = (
                "✅ Resolved"
                if alert.resolved
                else ("👁️ Acknowledged" if alert.acknowledged else "🔔 Active")
            )

            table.add_row(
                alert.id[:8],
                f"{priority_emoji} {alert.priority.value.title()}",
                alert.title[:50] + ("..." if len(alert.title) > 50 else ""),
                alert.source_service,
                alert.timestamp.strftime("%H:%M:%S"),
                status,
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("acknowledge")
def acknowledge_alert(
    alert_id: str = typer.Argument(..., help="Alert ID to acknowledge"),
    user: str = typer.Option("cli-user", "--user", help="User acknowledging the alert"),
):
    """Acknowledge a specific alert."""
    try:
        dashboard_service = initialize_enterprise_dashboard()

        success = dashboard_service.acknowledge_alert(alert_id, user)

        if success:
            console.print(f"✅ Alert {alert_id} acknowledged by {user}")
        else:
            console.print(f"❌ Alert {alert_id} not found")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("resolve")
def resolve_alert(
    alert_id: str = typer.Argument(..., help="Alert ID to resolve"),
    user: str = typer.Option("cli-user", "--user", help="User resolving the alert"),
):
    """Resolve a specific alert."""
    try:
        dashboard_service = initialize_enterprise_dashboard()

        success = dashboard_service.resolve_alert(alert_id, user)

        if success:
            console.print(f"✅ Alert {alert_id} resolved by {user}")
        else:
            console.print(f"❌ Alert {alert_id} not found")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("measurements")
def show_metrics(
    metric_type: str = typer.Option(
        "all", "--type", help="Metric type (business, operational, all)"
    ),
    format: str = typer.Option("table", "--format", help="Output format (table, json)"),
):
    """Display detailed measurements information.

    Show business measurements, operational measurements, or both
    in table or JSON format.
    """
    try:
        dashboard_service = initialize_enterprise_dashboard()
        _generate_sample_data(dashboard_service)

        if format == "json":
            data = dashboard_service.get_real_time_dashboard_data()

            if metric_type == "business":
                output = data.get("business_measurements", {})
            elif metric_type == "operational":
                output = data.get("operational_measurements", {})
            else:
                output = {
                    "business_measurements": data.get("business_measurements", {}),
                    "operational_measurements": data.get("operational_measurements", {}),
                }

            console.print(json.dumps(output, indent=2))
            return

        # Table format
        if metric_type in ["business", "all"]:
            business_table = Table(title="Business Measurements", box=box.ROUNDED)
            business_table.add_column("Metric", style="cyan")
            business_table.add_column("Value", style="green")
            business_table.add_column("Unit", style="yellow")
            business_table.add_column("Trend", style="blue")
            business_table.add_column("Target", style="magenta")

            for name, metric in dashboard_service.business_measurements.items():
                trend_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}.get(
                    metric.trend, "➡️"
                )

                business_table.add_row(
                    metric.name,
                    str(metric.value),
                    metric.unit,
                    f"{trend_emoji} {metric.change_percent:+.1f}%",
                    str(metric.target_value) if metric.target_value else "N/A",
                )

            console.print(business_table)
            console.print()

        if metric_type in ["operational", "all"]:
            operational_table = Table(title="Operational Measurements", box=box.ROUNDED)
            operational_table.add_column("Metric", style="cyan")
            operational_table.add_column("Current", style="green")
            operational_table.add_column("Status", style="yellow")
            operational_table.add_column("Warning", style="orange1")
            operational_table.add_column("Critical", style="red")

            for name, metric in dashboard_service.operational_measurements.items():
                status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(
                    metric.status, "❓"
                )

                operational_table.add_row(
                    name.replace("_", " ").title(),
                    f"{metric.current_value:.1f}",
                    f"{status_emoji} {metric.status.title()}",
                    f"{metric.threshold_warning:.1f}",
                    f"{metric.threshold_critical:.1f}",
                )

            console.print(operational_table)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("compliance")
def compliance_report():
    """Generate compliance and governance report.

    Display comprehensive compliance measurements and
    regulatory adherence status.
    """
    try:
        dashboard_service = initialize_enterprise_dashboard()

        report = dashboard_service.get_compliance_report()

        # Compliance scores
        scores_table = Table(title="Compliance Scores", box=box.ROUNDED)
        scores_table.add_column("Category", style="cyan")
        scores_table.add_column("Score", style="green")
        scores_table.add_column("Status", style="yellow")

        for category, score in report["compliance_scores"].items():
            status = (
                "✅ Excellent"
                if score >= 95
                else ("⚠️ Good" if score >= 85 else "🚨 Needs Attention")
            )
            scores_table.add_row(
                category.replace("_", " ").title(), f"{score:.1f}%", status
            )

        console.print(scores_table)
        console.print()

        # Audit summary
        audit_info = report["audit_summary"]
        audit_panel = Panel(
            f"""Total Detections Audited: {audit_info["total_processings_audited"]:,}
Audit Trail Completeness: {audit_info["audit_trail_completeness"]:.1f}%
Data Lineage Tracked: {"✅ Yes" if audit_info["data_lineage_tracked"] else "❌ No"}
Regulatory Violations: {audit_info["regulatory_violations"]}
Last Compliance Check: {audit_info["last_compliance_check"]}""",
            title="🔍 Audit Summary",
            style="blue",
        )
        console.print(audit_panel)
        console.print()

        # Data governance
        governance_info = report["data_governance"]
        governance_panel = Panel(
            f"""PII Processing: {"✅ Enabled" if governance_info["pii_processing_enabled"] else "❌ Disabled"}
Data Masking: {"✅ Active" if governance_info["data_masking_active"] else "❌ Inactive"}
Retention Policy: {"✅ Enforced" if governance_info["retention_policy_enforced"] else "❌ Not Enforced"}
Cross-Border Compliance: {governance_info["cross_border_compliance"]}""",
            title="🛡️ Data Governance",
            style="green",
        )
        console.print(governance_panel)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


@app.command("integration")
def integration_status():
    """Show enterprise integration status.

    Display status of dashboard, alerting, and
    autonomous processing integration.
    """
    try:
        # Initialize integration service
        config = IntegrationConfig()

        with console.status("Initializing enterprise integration..."):
            integration_service = asyncio.run(initialize_enterprise_integration(config))

        status = integration_service.get_integration_status()

        # Services status
        services_table = Table(title="Enterprise Services Status", box=box.ROUNDED)
        services_table.add_column("Service", style="cyan")
        services_table.add_column("Status", style="green")
        services_table.add_column("Configuration", style="yellow")

        for service, active in status["services"].items():
            status_text = "✅ Active" if active else "❌ Inactive"
            config_text = (
                "✅ Enabled"
                if status["configuration"].get(
                    f"{service.split('_')[0]}_integration", False
                )
                else "⚠️ Disabled"
            )

            services_table.add_row(
                service.replace("_", " ").title(), status_text, config_text
            )

        console.print(services_table)
        console.print()

        # Integration measurements
        measurements_panel = Panel(
            f"""Integration Active: {"✅ Yes" if status["integration_active"] else "❌ No"}
Dashboard Updates: {status["measurements"]["dashboard_updates"]:,}
Alerts Generated: {status["measurements"]["alerts_generated"]:,}
Detections Tracked: {status["measurements"]["autonomous_processings_tracked"]:,}
Background Tasks: {status["background_tasks"]}
Notification Providers: {status["notification_providers"]}
Last Update: {status["measurements"]["last_update"] or "Never"}""",
            title="📊 Integration Measurements",
            style="blue",
        )
        console.print(measurements_panel)

        # Shutdown integration service
        asyncio.run(integration_service.shutdown())

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


def _create_dashboard_layout(dashboard_service: EnterpriseDashboardService) -> Layout:
    """Create the live dashboard layout."""

    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="main"))

    # Header
    header_text = Text("🎯 ENTERPRISE DASHBOARD", style="bold blue")
    header_text.append(
        f" - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim"
    )
    layout["header"].update(Panel(header_text, box=box.HEAVY))

    # Main content
    main_layout = Layout()
    main_layout.split_row(Layout(name="left"), Layout(name="right"))

    # Left side - measurements
    left_layout = Layout()
    left_layout.split_column(Layout(name="business"), Layout(name="operational"))

    # Business measurements
    business_text = Text()
    for name, metric in dashboard_service.business_measurements.items():
        trend_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}.get(metric.trend, "➡️")
        business_text.append(
            f"{metric.name}: {metric.value} {metric.unit} {trend_emoji}\n",
            style="green",
        )

    left_layout["business"].update(Panel(business_text, title="💼 Business Measurements"))

    # Operational measurements
    operational_text = Text()
    for name, metric in dashboard_service.operational_measurements.items():
        status_color = {"healthy": "green", "warning": "yellow", "critical": "red"}.get(
            metric.status, "white"
        )
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(
            metric.status, "❓"
        )

        operational_text.append(
            f"{name.replace('_', ' ').title()}: {metric.current_value:.1f} {status_emoji}\n",
            style=status_color,
        )

    left_layout["operational"].update(
        Panel(operational_text, title="⚙️ Operational Measurements")
    )

    # Right side - alerts and status
    right_layout = Layout()
    right_layout.split_column(Layout(name="alerts"), Layout(name="system"))

    # Active alerts
    alerts_text = Text()
    active_alerts = list(dashboard_service.active_alerts.values())[:5]

    if not active_alerts:
        alerts_text.append("✅ No active alerts", style="green")
    else:
        for alert in active_alerts:
            priority_emoji = {
                AlertPriority.CRITICAL: "🚨",
                AlertPriority.HIGH: "⚠️",
                AlertPriority.MEDIUM: "📢",
                AlertPriority.LOW: "ℹ️",
            }.get(alert.priority, "📢")

            alerts_text.append(f"{priority_emoji} {alert.title[:40]}...\n", style="red")

    right_layout["alerts"].update(Panel(alerts_text, title="🔔 Active Alerts"))

    # System status
    stats = dashboard_service.processing_stats["today"]
    system_text = Text()
    system_text.append(f"Detections Today: {stats['total']}\n", style="cyan")
    system_text.append(f"Anomalies Found: {stats['anomalies']}\n", style="yellow")
    system_text.append(f"Success Rate: {stats['success_rate']:.1f}%\n", style="green")
    system_text.append(
        f"Active Alerts: {len(active_alerts)}\n",
        style="red" if active_alerts else "green",
    )

    right_layout["system"].update(Panel(system_text, title="📊 System Status"))

    main_layout["left"].update(left_layout)
    main_layout["right"].update(right_layout)
    layout["main"].update(main_layout)

    return layout


def _generate_sample_data(dashboard_service: EnterpriseDashboardService):
    """Generate sample data for demonstration purposes."""

    import random

    # Simulate some processing events
    algorithms = [
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM",
        "EllipticEnvelope",
    ]

    for i in range(random.randint(10, 50)):
        dashboard_service.record_processing_event(
            processing_id=f"demo_processing_{i}",
            success=random.choice([True, True, True, False]),  # 75% success rate
            execution_time=random.uniform(0.5, 15.0),
            algorithm_used=random.choice(algorithms),
            anomalies_found=random.randint(0, 25),
            data_collection_size=random.randint(100, 10000),
            cost_usd=random.uniform(0.01, 2.0),
        )

    # Create some sample alerts
    if random.random() < 0.7:  # 70% chance of having alerts
        for i in range(random.randint(1, 5)):
            dashboard_service.create_alert(
                title=f"Sample Alert {i + 1}",
                message=f"This is a sample alert message for demonstration purposes. Alert level {i + 1}.",
                priority=random.choice(list(AlertPriority)),
                metric_type=random.choice(list(DashboardMetricType)),
                source_service="demo_service",
            )
