"""CLI commands for governance framework and audit management."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.presentation.cli.async_utils import cli_runner

from pynomaly.application.services.governance_framework_service import (
    GovernanceAction,
    GovernanceFrameworkService,
    PolicyType,
    RiskLevel,
)
from pynomaly.domain.entities.security_compliance import ComplianceFramework
from pynomaly.infrastructure.config.container import Container

console = Console()


@click.group(name="governance")
def governance_commands():
    """Governance framework and audit management commands."""
    pass


@governance_commands.command()
@click.option("--start-date", help="Start date for audit report (YYYY-MM-DD)")
@click.option("--end-date", help="End date for audit report (YYYY-MM-DD)")
@click.option("--resource-types", multiple=True, help="Filter by resource types")
@click.option("--users", multiple=True, help="Filter by specific users")
@click.option("--actions", multiple=True, help="Filter by governance actions")
@click.option("--output-file", help="Save audit report to file")
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["json", "csv", "html", "pdf"]),
    default="json",
    help="Report output format",
)
def audit_report(
    start_date: str | None,
    end_date: str | None,
    resource_types: list[str],
    users: list[str],
    actions: list[str],
    output_file: str | None,
    report_format: str,
):
    """Generate comprehensive audit trail report."""

    async def run_audit_report():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Initializing governance service...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Parse dates
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
            else:
                start_dt = datetime.utcnow() - timedelta(days=30)

            if end_date:
                end_dt = datetime.fromisoformat(end_date)
            else:
                end_dt = datetime.utcnow()

            # Convert action strings to enums
            action_enums = []
            if actions:
                for action in actions:
                    try:
                        action_enums.append(GovernanceAction(action))
                    except ValueError:
                        console.print(
                            f"[yellow]Warning: Unknown action '{action}' ignored[/yellow]"
                        )

            # Generate audit report
            task2 = progress.add_task("Generating audit report...", total=None)

            try:
                report = await governance_service.generate_audit_report(
                    start_date=start_dt,
                    end_date=end_dt,
                    resource_types=list(resource_types) if resource_types else None,
                    users=list(users) if users else None,
                    actions=action_enums if action_enums else None,
                )

                progress.update(task2, completed=True)

                # Display report summary
                _display_audit_report_summary(report)

                # Save report if requested
                if output_file:
                    task3 = progress.add_task(
                        f"Saving report as {report_format}...", total=None
                    )

                    if report_format == "json":
                        with open(output_file, "w") as f:
                            json.dump(report, f, indent=2)
                    elif report_format == "csv":
                        _save_audit_report_csv(report, output_file)
                    elif report_format == "html":
                        _save_audit_report_html(report, output_file)

                    progress.update(task3, completed=True)
                    console.print(f"[green]Audit report saved to {output_file}[/green]")

            except Exception as e:
                console.print(f"[red]Error generating audit report: {e}[/red]")
                return

    cli_runner.run(run_audit_report())


@governance_commands.command()
@click.option("--policy-name", required=True, help="Name of the governance policy")
@click.option(
    "--policy-type",
    type=click.Choice(
        [
            "data_governance",
            "access_control",
            "change_management",
            "risk_management",
            "compliance",
            "quality_assurance",
            "security",
            "privacy",
        ]
    ),
    required=True,
    help="Type of governance policy",
)
@click.option("--description", required=True, help="Policy description")
@click.option("--content-file", help="JSON file with detailed policy content")
@click.option("--applicable-roles", multiple=True, help="Roles the policy applies to")
@click.option(
    "--compliance-frameworks", multiple=True, help="Applicable compliance frameworks"
)
@click.option(
    "--enforcement-level",
    type=click.Choice(["mandatory", "recommended", "optional"]),
    default="mandatory",
    help="Policy enforcement level",
)
def create_policy(
    policy_name: str,
    policy_type: str,
    description: str,
    content_file: str | None,
    applicable_roles: list[str],
    compliance_frameworks: list[str],
    enforcement_level: str,
):
    """Create new governance policy."""

    async def run_create_policy():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Initializing governance service...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Load policy content
            policy_content = {}
            if content_file:
                with open(content_file) as f:
                    policy_content = json.load(f)
            else:
                policy_content = {
                    "enforcement_level": enforcement_level,
                    "description": description,
                }

            # Convert compliance frameworks
            framework_enums = []
            for framework in compliance_frameworks:
                try:
                    framework_enums.append(ComplianceFramework(framework))
                except ValueError:
                    console.print(
                        f"[yellow]Warning: Unknown framework '{framework}' ignored[/yellow]"
                    )

            # Create policy
            task2 = progress.add_task("Creating governance policy...", total=None)

            try:
                policy_id = await governance_service.create_policy(
                    policy_name=policy_name,
                    policy_type=PolicyType(policy_type),
                    description=description,
                    policy_content=policy_content,
                    created_by="cli_user",  # Would get from auth context
                    applicable_roles=list(applicable_roles),
                    compliance_frameworks=framework_enums,
                )

                progress.update(task2, completed=True)

                console.print(
                    Panel(
                        f"[green]Governance policy created successfully[/green]\n"
                        f"Policy ID: {policy_id}\n"
                        f"Name: {policy_name}\n"
                        f"Type: {policy_type}\n"
                        f"Enforcement: {enforcement_level}\n"
                        f"Applicable roles: {', '.join(applicable_roles) if applicable_roles else 'All'}\n"
                        f"Compliance frameworks: {', '.join(compliance_frameworks) if compliance_frameworks else 'None'}",
                        title="Policy Created",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error creating policy: {e}[/red]")
                return

    cli_runner.run(run_create_policy())


@governance_commands.command()
@click.option("--risk-category", required=True, help="Category of risk being assessed")
@click.option("--description", required=True, help="Detailed risk description")
@click.option(
    "--likelihood",
    type=click.Choice(["very_low", "low", "medium", "high", "very_high", "critical"]),
    required=True,
    help="Probability of risk occurrence",
)
@click.option(
    "--impact",
    type=click.Choice(["very_low", "low", "medium", "high", "very_high", "critical"]),
    required=True,
    help="Potential impact if risk occurs",
)
@click.option("--affected-assets", multiple=True, help="Assets that could be affected")
@click.option("--threat-sources", multiple=True, help="Sources of the threat")
@click.option("--existing-controls", multiple=True, help="Existing risk controls")
def assess_risk(
    risk_category: str,
    description: str,
    likelihood: str,
    impact: str,
    affected_assets: list[str],
    threat_sources: list[str],
    existing_controls: list[str],
):
    """Conduct comprehensive risk assessment."""

    async def run_risk_assessment():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Initializing risk assessment...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Conduct risk assessment
            task2 = progress.add_task("Conducting risk assessment...", total=None)

            try:
                assessment_id = await governance_service.conduct_risk_assessment(
                    assessor="cli_user",  # Would get from auth context
                    risk_category=risk_category,
                    risk_description=description,
                    likelihood=RiskLevel(likelihood),
                    impact=RiskLevel(impact),
                    affected_assets=list(affected_assets),
                    threat_sources=list(threat_sources) if threat_sources else None,
                    existing_controls=(
                        list(existing_controls) if existing_controls else None
                    ),
                )

                progress.update(task2, completed=True)

                # Get the assessment for display
                assessment = governance_service.risk_assessments[assessment_id]
                risk_score = assessment.calculate_risk_score()

                console.print(
                    Panel(
                        f"[bold blue]Risk Assessment Completed[/bold blue]\n"
                        f"Assessment ID: {assessment_id}\n"
                        f"Category: {risk_category}\n"
                        f"Likelihood: {likelihood}\n"
                        f"Impact: {impact}\n"
                        f"Overall Risk: {assessment.overall_risk.value}\n"
                        f"Risk Score: {risk_score}\n"
                        f"Affected Assets: {len(affected_assets)}\n"
                        f"Threat Sources: {len(threat_sources)}\n"
                        f"Existing Controls: {len(existing_controls)}",
                        title="Risk Assessment Results",
                    )
                )

                # Display risk matrix
                _display_risk_matrix(likelihood, impact, assessment.overall_risk.value)

            except Exception as e:
                console.print(f"[red]Error conducting risk assessment: {e}[/red]")
                return

    cli_runner.run(run_risk_assessment())


@governance_commands.command()
@click.option("--title", required=True, help="Change request title")
@click.option("--description", required=True, help="Detailed description of the change")
@click.option(
    "--change-type",
    type=click.Choice(["configuration", "policy", "system", "data", "process"]),
    required=True,
    help="Type of change being requested",
)
@click.option(
    "--approvers",
    multiple=True,
    required=True,
    help="Required approvers for the change",
)
@click.option(
    "--urgency",
    type=click.Choice(["low", "normal", "high", "emergency"]),
    default="normal",
    help="Urgency level of the change",
)
@click.option("--impact-analysis", help="Analysis of potential impact")
@click.option("--rollback-plan", help="Plan for rolling back if needed")
@click.option("--affected-systems", multiple=True, help="Systems that will be affected")
def submit_change(
    title: str,
    description: str,
    change_type: str,
    approvers: list[str],
    urgency: str,
    impact_analysis: str | None,
    rollback_plan: str | None,
    affected_systems: list[str],
):
    """Submit change management request."""

    async def run_submit_change():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Initializing change management...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Submit change request
            task2 = progress.add_task("Submitting change request...", total=None)

            try:
                request_id = await governance_service.submit_change_request(
                    title=title,
                    description=description,
                    change_type=change_type,
                    requested_by="cli_user",  # Would get from auth context
                    approvers=list(approvers),
                    urgency=urgency,
                    impact_analysis=impact_analysis or "",
                    rollback_plan=rollback_plan or "",
                    affected_systems=list(affected_systems),
                )

                progress.update(task2, completed=True)

                console.print(
                    Panel(
                        f"[green]Change request submitted successfully[/green]\n"
                        f"Request ID: {request_id}\n"
                        f"Title: {title}\n"
                        f"Type: {change_type}\n"
                        f"Urgency: {urgency}\n"
                        f"Approvers: {', '.join(approvers)}\n"
                        f"Affected Systems: {', '.join(affected_systems) if affected_systems else 'None'}\n"
                        f"Status: Pending Approval",
                        title="Change Request Submitted",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error submitting change request: {e}[/red]")
                return

    cli_runner.run(run_submit_change())


@governance_commands.command()
@click.option("--request-id", required=True, help="Change request ID to approve/reject")
@click.option(
    "--decision",
    type=click.Choice(["approved", "rejected"]),
    required=True,
    help="Approval decision",
)
@click.option("--comments", help="Comments for the approval decision")
def approve_change(request_id: str, decision: str, comments: str | None):
    """Approve or reject change request."""

    async def run_approve_change():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Processing change approval...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Process approval
            task2 = progress.add_task(f"Processing {decision} decision...", total=None)

            try:
                is_approved = await governance_service.approve_change_request(
                    request_id=request_id,
                    approver="cli_user",  # Would get from auth context
                    decision=decision,
                    comments=comments or "",
                )

                progress.update(task2, completed=True)

                status_color = "green" if decision == "approved" else "red"
                final_status = (
                    "Fully Approved" if is_approved else f"Decision: {decision.title()}"
                )

                console.print(
                    Panel(
                        f"[{status_color}]Change request {decision}[/{status_color}]\n"
                        f"Request ID: {request_id}\n"
                        f"Decision: {decision.title()}\n"
                        f"Final Status: {final_status}\n"
                        f"Comments: {comments or 'None'}",
                        title="Approval Processed",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error processing approval: {e}[/red]")
                return

    cli_runner.run(run_approve_change())


@governance_commands.command()
@click.option("--metric-name", required=True, help="Name of the compliance metric")
@click.option(
    "--framework",
    type=click.Choice(["soc2", "gdpr", "hipaa", "pci_dss", "iso27001", "nist", "ccpa"]),
    required=True,
    help="Compliance framework",
)
@click.option("--control-id", required=True, help="Control identifier")
@click.option("--current-value", type=float, required=True, help="Current metric value")
@click.option("--target-value", type=float, default=100.0, help="Target metric value")
@click.option(
    "--responsible-party", required=True, help="Person responsible for the metric"
)
@click.option("--evidence", multiple=True, help="Supporting evidence for the metric")
def track_compliance(
    metric_name: str,
    framework: str,
    control_id: str,
    current_value: float,
    target_value: float,
    responsible_party: str,
    evidence: list[str],
):
    """Track compliance metric and performance."""

    async def run_track_compliance():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Initializing compliance tracking...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Track compliance metric
            task2 = progress.add_task("Recording compliance metric...", total=None)

            try:
                metric_id = await governance_service.track_compliance_metric(
                    metric_name=metric_name,
                    framework=ComplianceFramework(framework),
                    control_id=control_id,
                    current_value=current_value,
                    target_value=target_value,
                    responsible_party=responsible_party,
                    evidence=list(evidence) if evidence else None,
                )

                progress.update(task2, completed=True)

                # Get the metric for display
                metric = governance_service.compliance_metrics[metric_id]
                compliance_status = metric.get_compliance_status()

                status_colors = {
                    "compliant": "green",
                    "at_risk": "yellow",
                    "non_compliant": "red",
                    "critical": "red bold",
                }
                status_color = status_colors.get(compliance_status, "white")

                console.print(
                    Panel(
                        f"[bold blue]Compliance Metric Tracked[/bold blue]\n"
                        f"Metric ID: {metric_id}\n"
                        f"Name: {metric_name}\n"
                        f"Framework: {framework.upper()}\n"
                        f"Control: {control_id}\n"
                        f"Current Value: {current_value}\n"
                        f"Target Value: {target_value}\n"
                        f"Compliance Status: [{status_color}]{compliance_status.upper()}[/{status_color}]\n"
                        f"Responsible Party: {responsible_party}\n"
                        f"Evidence Count: {len(evidence)}",
                        title="Compliance Tracking",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error tracking compliance metric: {e}[/red]")
                return

    cli_runner.run(run_track_compliance())


@governance_commands.command()
def dashboard():
    """Display comprehensive governance dashboard."""

    async def run_dashboard():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Loading governance dashboard...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Get dashboard data
            task2 = progress.add_task("Generating dashboard data...", total=None)

            try:
                dashboard_data = await governance_service.get_governance_dashboard()
                progress.update(task2, completed=True)

                # Display dashboard
                _display_governance_dashboard(dashboard_data)

            except Exception as e:
                console.print(f"[red]Error generating governance dashboard: {e}[/red]")
                return

    cli_runner.run(run_dashboard())


@governance_commands.command()
@click.option("--user-id", required=True, help="User ID for the audit event")
@click.option(
    "--action",
    type=click.Choice(
        [
            "create",
            "update",
            "delete",
            "access",
            "approve",
            "reject",
            "escalate",
            "audit",
            "monitor",
            "report",
        ]
    ),
    required=True,
    help="Governance action performed",
)
@click.option("--resource-type", required=True, help="Type of resource affected")
@click.option("--resource-id", required=True, help="ID of the resource")
@click.option("--details", required=True, help="Detailed description of the action")
@click.option(
    "--risk-level",
    type=click.Choice(["very_low", "low", "medium", "high", "very_high", "critical"]),
    help="Assessed risk level",
)
@click.option(
    "--compliance-frameworks", multiple=True, help="Applicable compliance frameworks"
)
def log_event(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: str,
    risk_level: str | None,
    compliance_frameworks: list[str],
):
    """Log governance audit event."""

    async def run_log_event():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize governance service
            task1 = progress.add_task("Logging audit event...", total=None)
            storage_path = Path("./governance")
            governance_service = GovernanceFrameworkService(storage_path)
            progress.update(task1, completed=True)

            # Convert frameworks
            framework_enums = []
            for framework in compliance_frameworks:
                try:
                    framework_enums.append(ComplianceFramework(framework))
                except ValueError:
                    console.print(
                        f"[yellow]Warning: Unknown framework '{framework}' ignored[/yellow]"
                    )

            # Log event
            task2 = progress.add_task("Recording audit trail...", total=None)

            try:
                entry_id = await governance_service.log_audit_event(
                    user_id=user_id,
                    action=GovernanceAction(action),
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=details,
                    risk_level=RiskLevel(risk_level) if risk_level else None,
                    compliance_frameworks=framework_enums if framework_enums else None,
                )

                progress.update(task2, completed=True)

                console.print(
                    Panel(
                        f"[green]Audit event logged successfully[/green]\n"
                        f"Entry ID: {entry_id}\n"
                        f"User: {user_id}\n"
                        f"Action: {action}\n"
                        f"Resource: {resource_type}:{resource_id}\n"
                        f"Risk Level: {risk_level or 'Not assessed'}\n"
                        f"Compliance Frameworks: {', '.join(compliance_frameworks) if compliance_frameworks else 'None'}",
                        title="Audit Event Logged",
                    )
                )

            except Exception as e:
                console.print(f"[red]Error logging audit event: {e}[/red]")
                return

    cli_runner.run(run_log_event())


# Helper functions for display


def _display_audit_report_summary(report: dict):
    """Display audit report summary."""
    summary = report.get("summary", {})
    period = report.get("period", {})

    console.print(
        Panel(
            f"[bold blue]Audit Trail Report Summary[/bold blue]\n"
            f"Report ID: {report.get('report_id', 'N/A')}\n"
            f"Period: {period.get('start_date', 'N/A')} to {period.get('end_date', 'N/A')}\n"
            f"Duration: {period.get('duration_days', 0)} days\n"
            f"Total Events: {summary.get('total_events', 0)}\n"
            f"Unique Users: {summary.get('unique_users', 0)}\n"
            f"Unique Resources: {summary.get('unique_resources', 0)}\n"
            f"Avg Events/Day: {summary.get('average_events_per_day', 0):.1f}",
            title="Audit Report Summary",
        )
    )

    # Action distribution
    action_dist = report.get("action_distribution", {})
    if action_dist:
        action_table = Table(title="Action Distribution")
        action_table.add_column("Action", style="cyan")
        action_table.add_column("Count", style="yellow")
        action_table.add_column("Percentage", style="green")

        total_actions = sum(action_dist.values())
        for action, count in action_dist.items():
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            action_table.add_row(action, str(count), f"{percentage:.1f}%")

        console.print(action_table)


def _display_governance_dashboard(dashboard_data: dict):
    """Display comprehensive governance dashboard."""
    console.print(
        Panel(
            f"[bold blue]Governance Framework Dashboard[/bold blue]\n"
            f"Generated: {dashboard_data.get('generated_at', 'N/A')}\n"
            f"Period: {dashboard_data.get('period', 'N/A')}",
            title="Governance Overview",
        )
    )

    # Create layout with multiple sections
    layout = Layout()
    layout.split_column(
        Layout(name="audit", size=8),
        Layout(name="policy_risk", size=8),
        Layout(name="change_compliance", size=8),
    )

    # Audit activity
    audit_data = dashboard_data.get("audit_activity", {})
    audit_table = Table(title="Audit Activity (Last 30 Days)")
    audit_table.add_column("Metric", style="cyan")
    audit_table.add_column("Value", style="yellow")

    audit_table.add_row("Total Events", str(audit_data.get("total_events", 0)))
    audit_table.add_row("Unique Users", str(audit_data.get("unique_users", 0)))
    audit_table.add_row("Critical Events", str(audit_data.get("critical_events", 0)))

    layout["audit"].update(audit_table)

    # Policy and Risk split
    layout["policy_risk"].split_row(Layout(name="policy"), Layout(name="risk"))

    # Policy management
    policy_data = dashboard_data.get("policy_management", {})
    policy_table = Table(title="Policy Management")
    policy_table.add_column("Metric", style="cyan")
    policy_table.add_column("Value", style="yellow")

    policy_table.add_row("Active Policies", str(policy_data.get("active_policies", 0)))
    policy_table.add_row("Total Policies", str(policy_data.get("total_policies", 0)))
    policy_table.add_row("Violations", str(policy_data.get("policy_violations", 0)))

    layout["policy_risk"]["policy"].update(policy_table)

    # Risk management
    risk_data = dashboard_data.get("risk_management", {})
    risk_table = Table(title="Risk Management")
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", style="yellow")

    risk_table.add_row("Total Assessments", str(risk_data.get("total_assessments", 0)))
    risk_table.add_row("High Risk", str(risk_data.get("high_risk_assessments", 0)))
    risk_table.add_row("Overdue Reviews", str(risk_data.get("overdue_reviews", 0)))

    layout["policy_risk"]["risk"].update(risk_table)

    # Change and Compliance split
    layout["change_compliance"].split_row(
        Layout(name="change"), Layout(name="compliance")
    )

    # Change management
    change_data = dashboard_data.get("change_management", {})
    change_table = Table(title="Change Management")
    change_table.add_column("Metric", style="cyan")
    change_table.add_column("Value", style="yellow")

    change_table.add_row("Pending Changes", str(change_data.get("pending_changes", 0)))
    change_table.add_row("Total Requests", str(change_data.get("total_requests", 0)))
    change_table.add_row("Approved", str(change_data.get("approved_changes", 0)))

    layout["change_compliance"]["change"].update(change_table)

    # Compliance monitoring
    compliance_data = dashboard_data.get("compliance_monitoring", {})
    compliance_table = Table(title="Compliance Monitoring")
    compliance_table.add_column("Metric", style="cyan")
    compliance_table.add_column("Value", style="yellow")

    compliance_rate = compliance_data.get("compliance_rate", 0)
    rate_color = (
        "green"
        if compliance_rate >= 95
        else "yellow"
        if compliance_rate >= 80
        else "red"
    )

    compliance_table.add_row(
        "Compliance Rate", f"[{rate_color}]{compliance_rate:.1f}%[/{rate_color}]"
    )
    compliance_table.add_row(
        "Total Metrics", str(compliance_data.get("total_metrics", 0))
    )
    compliance_table.add_row("Violations", str(compliance_data.get("violations", 0)))

    layout["change_compliance"]["compliance"].update(compliance_table)

    console.print(layout)


def _display_risk_matrix(likelihood: str, impact: str, overall_risk: str):
    """Display risk assessment matrix."""
    console.print("\n[bold]Risk Assessment Matrix[/bold]")

    matrix_table = Table()
    matrix_table.add_column("", style="bold")
    matrix_table.add_column("Very Low", style="green")
    matrix_table.add_column("Low", style="green")
    matrix_table.add_column("Medium", style="yellow")
    matrix_table.add_column("High", style="red")
    matrix_table.add_column("Very High", style="red bold")
    matrix_table.add_column("Critical", style="red bold")

    impacts = ["Critical", "Very High", "High", "Medium", "Low", "Very Low"]
    likelihoods = ["very_low", "low", "medium", "high", "very_high", "critical"]

    for i, imp in enumerate(impacts):
        row = [imp]
        for j, like in enumerate(likelihoods):
            if imp.lower().replace(" ", "_") == impact and like == likelihood:
                cell = f"[bold magenta]● {overall_risk.upper()}[/bold magenta]"
            else:
                # Calculate risk level for this cell
                risk_score = (len(impacts) - i) * (j + 1)
                if risk_score <= 4:
                    cell = "Low"
                elif risk_score <= 9:
                    cell = "Medium"
                elif risk_score <= 16:
                    cell = "High"
                else:
                    cell = "Critical"
            row.append(cell)
        matrix_table.add_row(*row)

    console.print(matrix_table)


def _save_audit_report_csv(report: dict, output_file: str):
    """Save audit report as CSV."""
    import csv

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(
            [
                "Entry ID",
                "Timestamp",
                "User ID",
                "Action",
                "Resource Type",
                "Resource ID",
                "Details",
                "Risk Level",
                "Compliance Frameworks",
            ]
        )

        # Write audit entries
        for entry in report.get("audit_entries", []):
            writer.writerow(
                [
                    entry.get("entry_id", ""),
                    entry.get("timestamp", ""),
                    entry.get("user_id", ""),
                    entry.get("action", ""),
                    entry.get("resource_type", ""),
                    entry.get("resource_id", ""),
                    entry.get("details", ""),
                    entry.get("risk_assessment", ""),
                    ", ".join(entry.get("compliance_frameworks", [])),
                ]
            )


def _save_audit_report_html(report: dict, output_file: str):
    """Save audit report as HTML."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audit Trail Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Audit Trail Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Report ID:</strong> {report.get("report_id", "N/A")}</p>
            <p><strong>Total Events:</strong> {report.get("summary", {}).get("total_events", 0)}</p>
            <p><strong>Period:</strong> {report.get("period", {}).get("start_date", "N/A")} to {report.get("period", {}).get("end_date", "N/A")}</p>
        </div>
        <h2>Audit Entries</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>User</th>
                <th>Action</th>
                <th>Resource</th>
                <th>Details</th>
            </tr>
    """

    for entry in report.get("audit_entries", []):
        html_content += f"""
            <tr>
                <td>{entry.get("timestamp", "")}</td>
                <td>{entry.get("user_id", "")}</td>
                <td>{entry.get("action", "")}</td>
                <td>{entry.get("resource_type", "")}:{entry.get("resource_id", "")}</td>
                <td>{entry.get("details", "")}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    governance_commands()
