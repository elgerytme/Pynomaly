"""CLI commands for intelligent alert management."""

import asyncio
import json
from datetime import datetime
from uuid import UUID

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly_detection.application.services.intelligent_alert_service import (
    IntelligentAlertService,
)
from pynomaly_detection.domain.entities.alert import (
    AlertCategory,
    AlertMetadata,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    NoiseClassification,
)

console = Console()


@click.group(name="alert")
def alert_commands():
    """Intelligent alert management commands."""
    pass


@alert_commands.command()
@click.option("--title", required=True, help="Alert title")
@click.option("--description", required=True, help="Alert description")
@click.option(
    "--severity",
    type=click.Choice(["critical", "high", "medium", "low", "info"]),
    default="medium",
    help="Alert severity",
)
@click.option(
    "--category",
    type=click.Choice(
        [
            "anomaly_detection",
            "system_performance",
            "security",
            "data_quality",
            "model_drift",
            "resource_usage",
            "tenant_quota",
            "infrastructure",
            "authentication",
            "compliance",
        ]
    ),
    default="anomaly_detection",
    help="Alert category",
)
@click.option(
    "--source",
    type=click.Choice(
        [
            "detector",
            "system_monitor",
            "tenant_service",
            "security_service",
            "performance_monitor",
            "data_pipeline",
            "model_service",
            "external_webhook",
        ]
    ),
    default="detector",
    help="Alert source",
)
@click.option("--tenant-id", help="Tenant ID (UUID)")
@click.option("--detector-id", help="Detector ID (UUID)")
@click.option("--anomaly-score", type=float, help="Anomaly score (0.0-1.0)")
@click.option("--confidence", type=float, help="Confidence level (0.0-1.0)")
@click.option("--affected-resources", multiple=True, help="Affected resources")
@click.option(
    "--business-impact",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Business impact level",
)
@click.option("--message", help="Alert message")
def create(
    title: str,
    description: str,
    severity: str,
    category: str,
    source: str,
    tenant_id: str | None,
    detector_id: str | None,
    anomaly_score: float | None,
    confidence: float | None,
    affected_resources: list[str],
    business_impact: str | None,
    message: str | None,
):
    """Create a new alert with intelligent processing."""

    async def create_alert():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize alert service
            task1 = progress.add_task("Initializing alert service...", total=None)
            alert_service = IntelligentAlertService()
            progress.update(task1, completed=True)

            # Prepare metadata
            task2 = progress.add_task("Preparing alert metadata...", total=None)
            metadata = AlertMetadata()

            if tenant_id:
                try:
                    metadata.tenant_id = UUID(tenant_id)
                except ValueError:
                    console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                    return

            if detector_id:
                try:
                    metadata.detector_id = UUID(detector_id)
                except ValueError:
                    console.print(
                        f"[red]Error: Invalid detector ID: {detector_id}[/red]"
                    )
                    return

            metadata.anomaly_score = anomaly_score
            metadata.confidence_level = confidence
            metadata.affected_resources = list(affected_resources)
            metadata.business_impact = business_impact

            progress.update(task2, completed=True)

            # Create alert
            task3 = progress.add_task(
                "Creating alert with intelligent processing...", total=None
            )

            try:
                alert = await alert_service.create_alert(
                    title=title,
                    description=description,
                    severity=AlertSeverity(severity),
                    category=AlertCategory(category),
                    source=AlertSource(source),
                    metadata=metadata,
                    message=message or description,
                )

                progress.update(task3, completed=True)

                # Process alert intelligence
                task4 = progress.add_task(
                    "Processing alert intelligence...", total=None
                )
                processed_alert = await alert_service.process_alert_intelligence(alert)
                progress.update(task4, completed=True)

                # Display alert information
                _display_alert_info(processed_alert)

                console.print(f"[green]✓ Alert '{title}' created successfully![/green]")
                console.print(f"Alert ID: {processed_alert.alert_id}")

                if processed_alert.is_suppressed():
                    console.print(
                        f"[yellow]⚠ Alert was automatically suppressed: {processed_alert.suppression.suppression_reason}[/yellow]"
                    )

                if processed_alert.correlation:
                    console.print(
                        f"[blue]🔗 Alert correlated with {len(processed_alert.correlation.related_alerts)} other alerts[/blue]"
                    )

                if processed_alert.noise_classification != NoiseClassification.UNKNOWN:
                    console.print(
                        f"[cyan]🤖 ML Classification: {processed_alert.noise_classification.value} "
                        f"(confidence: {processed_alert.noise_confidence:.2f})[/cyan]"
                    )

            except Exception as e:
                console.print(f"[red]Error creating alert: {e}[/red]")
                return

    asyncio.run(create_alert())


@alert_commands.command()
@click.option(
    "--status",
    type=click.Choice(
        ["open", "acknowledged", "in_progress", "resolved", "suppressed", "escalated"]
    ),
    help="Filter by status",
)
@click.option(
    "--severity",
    type=click.Choice(["critical", "high", "medium", "low", "info"]),
    help="Filter by severity",
)
@click.option(
    "--category",
    type=click.Choice(
        [
            "anomaly_detection",
            "system_performance",
            "security",
            "data_quality",
            "model_drift",
            "resource_usage",
            "tenant_quota",
            "infrastructure",
            "authentication",
            "compliance",
        ]
    ),
    help="Filter by category",
)
@click.option("--tenant-id", help="Filter by tenant ID")
@click.option(
    "--limit", type=int, default=50, help="Maximum number of alerts to display"
)
@click.option("--include-suppressed", is_flag=True, help="Include suppressed alerts")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list(
    status: str | None,
    severity: str | None,
    category: str | None,
    tenant_id: str | None,
    limit: int,
    include_suppressed: bool,
    output_format: str,
):
    """List alerts with intelligent filtering."""

    async def list_alerts():
        alert_service = IntelligentAlertService()

        # Convert string filters to enums
        status_filter = AlertStatus(status) if status else None
        severity_filter = AlertSeverity(severity) if severity else None
        category_filter = AlertCategory(category) if category else None
        tenant_id_filter = UUID(tenant_id) if tenant_id else None

        alerts = await alert_service.list_alerts(
            status_filter=status_filter,
            severity_filter=severity_filter,
            category_filter=category_filter,
            tenant_id_filter=tenant_id_filter,
            limit=limit,
            include_suppressed=include_suppressed,
        )

        if not alerts:
            console.print("[yellow]No alerts found matching the criteria[/yellow]")
            return

        if output_format == "json":
            alert_data = [alert.to_dict() for alert in alerts]
            console.print(json.dumps(alert_data, indent=2, default=str))
        else:
            _display_alerts_table(alerts)

    asyncio.run(list_alerts())


@alert_commands.command()
@click.argument("alert_id", type=str)
def show(alert_id: str):
    """Show detailed information about an alert."""

    async def show_alert():
        alert_service = IntelligentAlertService()

        try:
            alert_uuid = UUID(alert_id)
        except ValueError:
            console.print(f"[red]Error: Invalid alert ID: {alert_id}[/red]")
            return

        alert = await alert_service.get_alert(alert_uuid)

        if not alert:
            console.print(f"[red]Alert not found: {alert_id}[/red]")
            return

        # Display detailed alert information
        _display_alert_detail(alert)

    asyncio.run(show_alert())


@alert_commands.command()
@click.argument("alert_id", type=str)
@click.option("--user", required=True, help="User acknowledging the alert")
@click.option("--note", help="Acknowledgment note")
def acknowledge(alert_id: str, user: str, note: str | None):
    """Acknowledge an alert."""

    async def acknowledge_alert():
        alert_service = IntelligentAlertService()

        try:
            alert_uuid = UUID(alert_id)
        except ValueError:
            console.print(f"[red]Error: Invalid alert ID: {alert_id}[/red]")
            return

        success = await alert_service.acknowledge_alert(alert_uuid, user, note or "")

        if success:
            console.print(f"[green]✓ Alert {alert_id} acknowledged by {user}[/green]")
            if note:
                console.print(f"Note: {note}")
        else:
            console.print(f"[red]Error: Could not acknowledge alert {alert_id}[/red]")

    asyncio.run(acknowledge_alert())


@alert_commands.command()
@click.argument("alert_id", type=str)
@click.option("--user", required=True, help="User resolving the alert")
@click.option("--note", help="Resolution note")
@click.option("--quality", type=float, help="Resolution quality score (0.0-1.0)")
def resolve(alert_id: str, user: str, note: str | None, quality: float | None):
    """Resolve an alert."""

    async def resolve_alert():
        alert_service = IntelligentAlertService()

        try:
            alert_uuid = UUID(alert_id)
        except ValueError:
            console.print(f"[red]Error: Invalid alert ID: {alert_id}[/red]")
            return

        if quality is not None and (quality < 0.0 or quality > 1.0):
            console.print("[red]Error: Quality score must be between 0.0 and 1.0[/red]")
            return

        success = await alert_service.resolve_alert(
            alert_uuid, user, note or "", quality
        )

        if success:
            console.print(f"[green]✓ Alert {alert_id} resolved by {user}[/green]")
            if note:
                console.print(f"Resolution: {note}")
            if quality is not None:
                console.print(f"Quality Score: {quality:.2f}")
        else:
            console.print(f"[red]Error: Could not resolve alert {alert_id}[/red]")

    asyncio.run(resolve_alert())


@alert_commands.command()
@click.argument("alert_id", type=str)
@click.option("--user", required=True, help="User suppressing the alert")
@click.option("--reason", help="Suppression reason")
@click.option("--duration", type=int, help="Suppression duration in minutes")
def suppress(alert_id: str, user: str, reason: str | None, duration: int | None):
    """Suppress an alert."""

    async def suppress_alert():
        alert_service = IntelligentAlertService()

        try:
            alert_uuid = UUID(alert_id)
        except ValueError:
            console.print(f"[red]Error: Invalid alert ID: {alert_id}[/red]")
            return

        success = await alert_service.suppress_alert(
            alert_uuid, user, reason or "", duration
        )

        if success:
            console.print(f"[green]✓ Alert {alert_id} suppressed by {user}[/green]")
            if reason:
                console.print(f"Reason: {reason}")
            if duration:
                console.print(f"Duration: {duration} minutes")
        else:
            console.print(f"[red]Error: Could not suppress alert {alert_id}[/red]")

    asyncio.run(suppress_alert())


@alert_commands.command()
@click.argument("alert_id", type=str)
@click.option("--user", required=True, help="User escalating the alert")
@click.option("--reason", help="Escalation reason")
def escalate(alert_id: str, user: str, reason: str | None):
    """Escalate an alert."""

    async def escalate_alert():
        alert_service = IntelligentAlertService()

        try:
            alert_uuid = UUID(alert_id)
        except ValueError:
            console.print(f"[red]Error: Invalid alert ID: {alert_id}[/red]")
            return

        success = await alert_service.escalate_alert(alert_uuid, user, reason or "")

        if success:
            console.print(f"[green]✓ Alert {alert_id} escalated by {user}[/green]")
            if reason:
                console.print(f"Reason: {reason}")
        else:
            console.print(f"[red]Error: Could not escalate alert {alert_id}[/red]")

    asyncio.run(escalate_alert())


@alert_commands.command()
@click.option("--days", type=int, default=7, help="Number of days to analyze")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["console", "json"]),
    default="console",
    help="Output format",
)
@click.option("--output-file", help="Output file for analytics")
def analytics(days: int, output_format: str, output_file: str | None):
    """Show comprehensive alert analytics."""

    async def show_analytics():
        alert_service = IntelligentAlertService()

        analytics_data = await alert_service.get_alert_analytics(days)

        if output_format == "json":
            if output_file:
                with open(output_file, "w") as f:
                    json.dump(analytics_data, f, indent=2, default=str)
                console.print(f"[green]Analytics saved to {output_file}[/green]")
            else:
                console.print(json.dumps(analytics_data, indent=2, default=str))
        else:
            _display_analytics(analytics_data, days)

    asyncio.run(show_analytics())


@alert_commands.command()
@click.option(
    "--status",
    type=click.Choice(["open", "acknowledged", "in_progress"]),
    default="open",
    help="Status of alerts to check for escalation",
)
def check_escalations(status: str):
    """Check for alerts that need escalation."""

    async def check_alert_escalations():
        alert_service = IntelligentAlertService()

        status_filter = AlertStatus(status)
        alerts = await alert_service.list_alerts(
            status_filter=status_filter, limit=1000
        )

        escalation_candidates = [alert for alert in alerts if alert.should_escalate()]

        if not escalation_candidates:
            console.print(f"[green]No {status} alerts need escalation[/green]")
            return

        console.print(
            f"[yellow]Found {len(escalation_candidates)} alerts needing escalation:[/yellow]"
        )

        escalation_table = Table(title="Alerts Needing Escalation")
        escalation_table.add_column("Alert ID", style="cyan")
        escalation_table.add_column("Title", style="yellow")
        escalation_table.add_column("Severity", style="red")
        escalation_table.add_column("Age", style="blue")
        escalation_table.add_column("Current Level", style="green")
        escalation_table.add_column("Priority Score", style="magenta")

        for alert in escalation_candidates:
            age_hours = (datetime.utcnow() - alert.created_at).total_seconds() / 3600

            escalation_table.add_row(
                str(alert.alert_id)[:8] + "...",
                alert.title[:40] + ("..." if len(alert.title) > 40 else ""),
                alert.severity.value,
                f"{age_hours:.1f}h",
                alert.escalation.current_level.value,
                f"{alert.calculate_priority_score():.1f}",
            )

        console.print(escalation_table)

    asyncio.run(check_alert_escalations())


# Helper functions for display


def _display_alert_info(alert):
    """Display basic alert information."""
    console.print(
        Panel(
            f"[bold blue]Alert Created[/bold blue]\n\n"
            f"ID: {alert.alert_id}\n"
            f"Title: {alert.title}\n"
            f"Severity: {alert.severity.value}\n"
            f"Category: {alert.category.value}\n"
            f"Source: {alert.source.value}\n"
            f"Status: {alert.status.value}\n"
            f"Priority Score: {alert.calculate_priority_score():.1f}",
            title="Alert Information",
        )
    )


def _display_alerts_table(alerts):
    """Display alerts in a table format."""
    table = Table(title="Intelligent Alert Management")
    table.add_column("Alert ID", style="cyan")
    table.add_column("Title", style="yellow")
    table.add_column("Severity", style="red")
    table.add_column("Status", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Priority", style="magenta")
    table.add_column("ML Class", style="white")
    table.add_column("Created", style="dim")

    for alert in alerts:
        severity_color = {
            "critical": "red",
            "high": "orange",
            "medium": "yellow",
            "low": "green",
            "info": "blue",
        }.get(alert.severity.value, "white")

        status_color = {
            "open": "red",
            "acknowledged": "yellow",
            "in_progress": "blue",
            "resolved": "green",
            "suppressed": "dim",
            "escalated": "magenta",
        }.get(alert.status.value, "white")

        ml_class_color = {
            "signal": "green",
            "noise": "red",
            "duplicate": "yellow",
            "flapping": "orange",
            "unknown": "dim",
        }.get(alert.noise_classification.value, "white")

        table.add_row(
            str(alert.alert_id)[:8] + "...",
            alert.title[:30] + ("..." if len(alert.title) > 30 else ""),
            f"[{severity_color}]{alert.severity.value}[/{severity_color}]",
            f"[{status_color}]{alert.status.value}[/{status_color}]",
            alert.category.value,
            f"{alert.calculate_priority_score():.1f}",
            f"[{ml_class_color}]{alert.noise_classification.value}[/{ml_class_color}]",
            alert.created_at.strftime("%m-%d %H:%M"),
        )

    console.print(table)


def _display_alert_detail(alert):
    """Display detailed alert information."""
    # Basic info panel
    console.print(
        Panel(
            f"[bold blue]Alert Details[/bold blue]\n\n"
            f"ID: {alert.alert_id}\n"
            f"Title: {alert.title}\n"
            f"Description: {alert.description}\n"
            f"Message: {alert.message}\n"
            f"Severity: {alert.severity.value}\n"
            f"Status: {alert.status.value}\n"
            f"Category: {alert.category.value}\n"
            f"Source: {alert.source.value}\n"
            f"Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Priority Score: {alert.calculate_priority_score():.2f}",
            title="Basic Information",
        )
    )

    # ML Intelligence panel
    console.print(
        Panel(
            f"[bold green]ML Intelligence[/bold green]\n\n"
            f"Classification: {alert.noise_classification.value}\n"
            f"Confidence: {alert.noise_confidence:.3f}\n"
            f"Signal Probability: {alert.signal_probability:.3f}\n"
            f"Hour of Day: {alert.ml_features.hour_of_day}\n"
            f"Day of Week: {alert.ml_features.day_of_week}\n"
            f"Business Hours: {alert.ml_features.is_business_hours}\n"
            f"Similar Alerts (7d): {alert.ml_features.similar_alerts_last_week}\n"
            f"System Load Percentile: {alert.ml_features.system_load_percentile:.2f}",
            title="ML Analysis",
        )
    )

    # Metadata panel
    metadata = alert.metadata
    console.print(
        Panel(
            f"[bold cyan]Metadata[/bold cyan]\n\n"
            f"Tenant ID: {metadata.tenant_id or 'None'}\n"
            f"Detector ID: {metadata.detector_id or 'None'}\n"
            f"Anomaly Score: {metadata.anomaly_score or 'None'}\n"
            f"Confidence Level: {metadata.confidence_level or 'None'}\n"
            f"Business Impact: {metadata.business_impact or 'None'}\n"
            f"Affected Resources: {len(metadata.affected_resources)}\n"
            f"Related Metrics: {len(metadata.related_metrics)}",
            title="Metadata",
        )
    )

    # Correlation panel
    if alert.correlation:
        correlation = alert.correlation
        console.print(
            Panel(
                f"[bold magenta]Correlation Analysis[/bold magenta]\n\n"
                f"Correlation ID: {correlation.correlation_id}\n"
                f"Type: {correlation.correlation_type}\n"
                f"Strength: {correlation.correlation_strength:.3f}\n"
                f"Related Alerts: {len(correlation.related_alerts)}\n"
                f"Reason: {correlation.correlation_reason}\n"
                f"Pattern Similarity: {correlation.pattern_similarity or 'N/A'}\n"
                f"Feature Overlap: {correlation.feature_overlap or 'N/A'}",
                title="Correlation",
            )
        )

    # Escalation panel
    escalation = alert.escalation
    console.print(
        Panel(
            f"[bold yellow]Escalation Status[/bold yellow]\n\n"
            f"Current Level: {escalation.current_level.value}\n"
            f"Assigned To: {escalation.assigned_to or 'Unassigned'}\n"
            f"Assigned Team: {escalation.assigned_team or 'None'}\n"
            f"Escalation Enabled: {escalation.escalation_enabled}\n"
            f"Should Escalate: {alert.should_escalate()}\n"
            f"Escalation History: {len(escalation.escalation_history)} events",
            title="Escalation",
        )
    )


def _display_analytics(analytics_data: dict, days: int):
    """Display alert analytics."""
    console.print(
        Panel(
            f"[bold blue]Alert Analytics - Last {days} Days[/bold blue]\n\n"
            f"Total Alerts: {analytics_data['total_alerts']:,}\n"
            f"Suppressed Alerts: {analytics_data['noise_reduction_stats']['total_suppressed']:,}\n"
            f"ML Classified Noise: {analytics_data['noise_reduction_stats']['ml_classified_noise']:,}\n"
            f"Signal-to-Noise Ratio: {analytics_data['noise_reduction_stats']['signal_to_noise_ratio']:.2f}\n"
            f"Correlated Alerts: {analytics_data['correlation_stats']['correlated_alerts']:,}\n"
            f"Avg Processing Time: {analytics_data['performance_metrics']['avg_processing_time']:.3f}s",
            title="Overview",
        )
    )

    # Alert distribution by severity
    severity_table = Table(title="Distribution by Severity")
    severity_table.add_column("Severity", style="cyan")
    severity_table.add_column("Count", style="yellow")
    severity_table.add_column("Percentage", style="green")

    total_alerts = analytics_data["total_alerts"]
    for severity, count in analytics_data["alert_distribution"]["by_severity"].items():
        percentage = (count / total_alerts) * 100 if total_alerts > 0 else 0
        severity_table.add_row(severity.title(), str(count), f"{percentage:.1f}%")

    console.print(severity_table)

    # Noise classification distribution
    noise_table = Table(title="ML Noise Classification")
    noise_table.add_column("Classification", style="cyan")
    noise_table.add_column("Count", style="yellow")
    noise_table.add_column("Percentage", style="green")

    for classification, count in analytics_data["alert_distribution"][
        "by_noise_classification"
    ].items():
        percentage = (count / total_alerts) * 100 if total_alerts > 0 else 0
        noise_table.add_row(classification.title(), str(count), f"{percentage:.1f}%")

    console.print(noise_table)

    # Performance metrics
    perf_metrics = analytics_data["performance_metrics"]
    console.print(
        Panel(
            f"[bold green]Performance Metrics[/bold green]\n\n"
            f"Total Processed: {perf_metrics['total_processed']:,}\n"
            f"Suppression Rate: {perf_metrics['suppression_rate']:.1%}\n"
            f"Average Processing Time: {perf_metrics['avg_processing_time']:.3f}s",
            title="System Performance",
        )
    )


if __name__ == "__main__":
    alert_commands()
