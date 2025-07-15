"""Typer-compatible wrapper for governance CLI commands."""

import typer
from pathlib import Path
from typing import List, Optional

from pynomaly.presentation.cli.governance import (
    audit_report,
    create_policy,
    assess_risk,
    submit_change,
    approve_change,
    track_compliance,
    dashboard,
    log_event,
)

# Create Typer app
app = typer.Typer(
    name="governance",
    help="🏛️ Governance framework and audit management commands",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
def audit_report_cmd(
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Start date for audit report (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date for audit report (YYYY-MM-DD)"),
    resource_types: List[str] = typer.Option([], "--resource-types", help="Filter by resource types"),
    users: List[str] = typer.Option([], "--users", help="Filter by specific users"),
    actions: List[str] = typer.Option([], "--actions", help="Filter by governance actions"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="Save audit report to file"),
    report_format: str = typer.Option("json", "--format", help="Report output format"),
):
    """Generate comprehensive audit trail report."""
    audit_report.callback(start_date, end_date, resource_types, users, actions, output_file, report_format)


@app.command()
def create_policy_cmd(
    policy_name: str = typer.Option(..., "--policy-name", help="Name of the governance policy"),
    policy_type: str = typer.Option(..., "--policy-type", help="Type of governance policy"),
    description: str = typer.Option(..., "--description", help="Policy description"),
    content_file: Optional[str] = typer.Option(None, "--content-file", help="JSON file with detailed policy content"),
    applicable_roles: List[str] = typer.Option([], "--applicable-roles", help="Roles the policy applies to"),
    compliance_frameworks: List[str] = typer.Option([], "--compliance-frameworks", help="Applicable compliance frameworks"),
    enforcement_level: str = typer.Option("mandatory", "--enforcement-level", help="Policy enforcement level"),
):
    """Create new governance policy."""
    create_policy.callback(policy_name, policy_type, description, content_file, applicable_roles, compliance_frameworks, enforcement_level)


@app.command()
def assess_risk_cmd(
    risk_category: str = typer.Option(..., "--risk-category", help="Category of risk being assessed"),
    description: str = typer.Option(..., "--description", help="Detailed risk description"),
    likelihood: str = typer.Option(..., "--likelihood", help="Probability of risk occurrence"),
    impact: str = typer.Option(..., "--impact", help="Potential impact if risk occurs"),
    affected_assets: List[str] = typer.Option([], "--affected-assets", help="Assets that could be affected"),
    threat_sources: List[str] = typer.Option([], "--threat-sources", help="Sources of the threat"),
    existing_controls: List[str] = typer.Option([], "--existing-controls", help="Existing risk controls"),
):
    """Conduct comprehensive risk assessment."""
    assess_risk.callback(risk_category, description, likelihood, impact, affected_assets, threat_sources, existing_controls)


@app.command()
def submit_change_cmd(
    title: str = typer.Option(..., "--title", help="Change request title"),
    description: str = typer.Option(..., "--description", help="Detailed description of the change"),
    change_type: str = typer.Option(..., "--change-type", help="Type of change being requested"),
    approvers: List[str] = typer.Option(..., "--approvers", help="Required approvers for the change"),
    urgency: str = typer.Option("normal", "--urgency", help="Urgency level of the change"),
    impact_analysis: Optional[str] = typer.Option(None, "--impact-analysis", help="Analysis of potential impact"),
    rollback_plan: Optional[str] = typer.Option(None, "--rollback-plan", help="Plan for rolling back if needed"),
    affected_systems: List[str] = typer.Option([], "--affected-systems", help="Systems that will be affected"),
):
    """Submit change management request."""
    submit_change.callback(title, description, change_type, approvers, urgency, impact_analysis, rollback_plan, affected_systems)


@app.command()
def approve_change_cmd(
    request_id: str = typer.Option(..., "--request-id", help="Change request ID to approve/reject"),
    decision: str = typer.Option(..., "--decision", help="Approval decision"),
    comments: Optional[str] = typer.Option(None, "--comments", help="Comments for the approval decision"),
):
    """Approve or reject change request."""
    approve_change.callback(request_id, decision, comments)


@app.command()
def track_compliance_cmd(
    metric_name: str = typer.Option(..., "--metric-name", help="Name of the compliance metric"),
    framework: str = typer.Option(..., "--framework", help="Compliance framework"),
    control_id: str = typer.Option(..., "--control-id", help="Control identifier"),
    current_value: float = typer.Option(..., "--current-value", help="Current metric value"),
    target_value: float = typer.Option(100.0, "--target-value", help="Target metric value"),
    responsible_party: str = typer.Option(..., "--responsible-party", help="Person responsible for the metric"),
    evidence: List[str] = typer.Option([], "--evidence", help="Supporting evidence for the metric"),
):
    """Track compliance metric and performance."""
    track_compliance.callback(metric_name, framework, control_id, current_value, target_value, responsible_party, evidence)


@app.command()
def dashboard_cmd():
    """Display comprehensive governance dashboard."""
    dashboard.callback()


@app.command()
def log_event_cmd(
    user_id: str = typer.Option(..., "--user-id", help="User ID for the audit event"),
    action: str = typer.Option(..., "--action", help="Governance action performed"),
    resource_type: str = typer.Option(..., "--resource-type", help="Type of resource affected"),
    resource_id: str = typer.Option(..., "--resource-id", help="ID of the resource"),
    details: str = typer.Option(..., "--details", help="Detailed description of the action"),
    risk_level: Optional[str] = typer.Option(None, "--risk-level", help="Assessed risk level"),
    compliance_frameworks: List[str] = typer.Option([], "--compliance-frameworks", help="Applicable compliance frameworks"),
):
    """Log governance audit event."""
    log_event.callback(user_id, action, resource_type, resource_id, details, risk_level, compliance_frameworks)


if __name__ == "__main__":
    app()