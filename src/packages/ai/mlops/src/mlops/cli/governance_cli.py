"""
Governance CLI Commands

Command-line interface for ML governance, compliance management,
and regulatory oversight operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import click
import yaml
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.tree import Tree
from rich import print as rprint

from ..infrastructure.governance.ml_governance_framework import (
    MLGovernanceFramework, ComplianceFramework, GovernanceRisk, AuditEventType
)
from ..infrastructure.governance.compliance_automation import ComplianceAutomationEngine
from ..infrastructure.governance.regulatory_compliance import RegulatoryComplianceManager
from ..infrastructure.governance.governance_dashboard import GovernanceDashboard


console = Console()


@click.group()
@click.pass_context
def governance(ctx):
    """ML Governance and Compliance Management CLI."""
    ctx.ensure_object(dict)
    
    # Initialize governance components
    ctx.obj['governance'] = MLGovernanceFramework()
    ctx.obj['compliance_automation'] = ComplianceAutomationEngine(ctx.obj['governance'])
    ctx.obj['regulatory_manager'] = RegulatoryComplianceManager(ctx.obj['governance'])
    ctx.obj['dashboard'] = GovernanceDashboard(
        ctx.obj['governance'],
        ctx.obj['compliance_automation'],
        ctx.obj['regulatory_manager']
    )


@governance.group()
def policies():
    """Governance policy management."""
    pass


@policies.command()
@click.option('--name', required=True, help='Policy name')
@click.option('--description', required=True, help='Policy description')
@click.option('--framework', 
              type=click.Choice([f.value for f in ComplianceFramework]),
              multiple=True,
              help='Compliance frameworks')
@click.option('--policy-type', required=True, help='Policy type')
@click.option('--created-by', required=True, help='Policy creator')
@click.pass_context
async def create(ctx, name, description, framework, policy_type, created_by):
    """Create a new governance policy."""
    
    frameworks = [ComplianceFramework(f) for f in framework] if framework else [ComplianceFramework.GDPR]
    
    with console.status(f"Creating policy '{name}'..."):
        policy_id = await ctx.obj['governance'].create_policy(
            name=name,
            description=description,
            compliance_frameworks=frameworks,
            policy_type=policy_type,
            created_by=created_by
        )
    
    console.print(f"✅ Policy created successfully with ID: [bold blue]{policy_id}[/bold blue]")


@policies.command()
@click.option('--policy-id', required=True, help='Policy ID')
@click.option('--rule-type', required=True, help='Rule type')
@click.option('--conditions', required=True, help='Rule conditions (JSON)')
@click.option('--actions', required=True, help='Rule actions (comma-separated)')
@click.pass_context
async def add_rule(ctx, policy_id, rule_type, conditions, actions):
    """Add a rule to an existing policy."""
    
    try:
        conditions_dict = json.loads(conditions)
        actions_list = [action.strip() for action in actions.split(',')]
        
        with console.status(f"Adding rule to policy {policy_id}..."):
            await ctx.obj['governance'].add_policy_rule(
                policy_id=policy_id,
                rule_type=rule_type,
                conditions=conditions_dict,
                actions=actions_list
            )
        
        console.print(f"✅ Rule added successfully to policy [bold blue]{policy_id}[/bold blue]")
        
    except json.JSONDecodeError:
        console.print("❌ Invalid JSON format for conditions")
    except Exception as e:
        console.print(f"❌ Error adding rule: {str(e)}")


@policies.command()
@click.pass_context
async def list(ctx):
    """List all governance policies."""
    
    governance = ctx.obj['governance']
    
    if not governance.policies:
        console.print("No policies found.")
        return
    
    table = Table(title="Governance Policies")
    table.add_column("Policy ID", style="cyan")
    table.add_column("Name", style="bright_white")
    table.add_column("Type", style="yellow")
    table.add_column("Frameworks", style="green")
    table.add_column("Rules", justify="right", style="magenta")
    table.add_column("Active", style="blue")
    
    for policy_id, policy in governance.policies.items():
        frameworks = ", ".join([f.value for f in policy.compliance_frameworks])
        
        table.add_row(
            policy_id[:8] + "...",
            policy.name,
            policy.policy_type,
            frameworks,
            str(len(policy.rules)),
            "✅" if policy.is_active else "❌"
        )
    
    console.print(table)


@governance.group()
def compliance():
    """Compliance automation and monitoring."""
    pass


@compliance.command()
@click.option('--name', required=True, help='Rule name')
@click.option('--description', required=True, help='Rule description')
@click.option('--framework',
              type=click.Choice([f.value for f in ComplianceFramework]),
              multiple=True,
              help='Compliance frameworks')
@click.option('--trigger', 
              type=click.Choice(['continuous', 'scheduled', 'event_driven', 'threshold_based']),
              default='continuous',
              help='Automation trigger')
@click.option('--conditions', required=True, help='Rule conditions (JSON)')
@click.option('--actions',
              type=click.Choice(['alert', 'quarantine_model', 'revoke_access', 'escalate']),
              multiple=True,
              help='Remediation actions')
@click.option('--auto-execute', is_flag=True, help='Enable automatic execution')
@click.pass_context
async def add_rule(ctx, name, description, framework, trigger, conditions, actions, auto_execute):
    """Add an automated compliance rule."""
    
    try:
        from ..infrastructure.governance.compliance_automation import AutomationTrigger, RemediationAction
        
        frameworks = [ComplianceFramework(f) for f in framework] if framework else [ComplianceFramework.GDPR]
        trigger_enum = AutomationTrigger(trigger)
        action_enums = [RemediationAction(a) for a in actions] if actions else [RemediationAction.ALERT]
        conditions_dict = json.loads(conditions)
        
        with console.status(f"Adding compliance rule '{name}'..."):
            rule_id = await ctx.obj['compliance_automation'].add_compliance_rule(
                name=name,
                description=description,
                compliance_frameworks=frameworks,
                trigger=trigger_enum,
                conditions=conditions_dict,
                remediation_actions=action_enums,
                auto_execute=auto_execute
            )
        
        console.print(f"✅ Compliance rule created with ID: [bold blue]{rule_id}[/bold blue]")
        
    except json.JSONDecodeError:
        console.print("❌ Invalid JSON format for conditions")
    except Exception as e:
        console.print(f"❌ Error creating rule: {str(e)}")


@compliance.command()
@click.option('--start', is_flag=True, help='Start compliance automation')
@click.option('--stop', is_flag=True, help='Stop compliance automation')
@click.option('--status', is_flag=True, help='Show automation status')
@click.pass_context
async def automation(ctx, start, stop, status):
    """Manage compliance automation."""
    
    automation_engine = ctx.obj['compliance_automation']
    
    if start:
        with console.status("Starting compliance automation..."):
            await automation_engine.start_automation()
        console.print("✅ Compliance automation started")
        
    elif stop:
        with console.status("Stopping compliance automation..."):
            await automation_engine.stop_automation()
        console.print("✅ Compliance automation stopped")
        
    elif status:
        status_info = {
            "running": automation_engine.is_running,
            "rules_count": len(automation_engine.compliance_rules),
            "violations_count": len(automation_engine.violations),
            "pending_tasks": len([t for t in automation_engine.remediation_tasks.values() if t.status == "pending"])
        }
        
        panel = Panel(
            f"""
[bold]Automation Status:[/bold] {'🟢 Running' if status_info['running'] else '🔴 Stopped'}
[bold]Rules:[/bold] {status_info['rules_count']}
[bold]Violations:[/bold] {status_info['violations_count']}
[bold]Pending Tasks:[/bold] {status_info['pending_tasks']}
            """.strip(),
            title="Compliance Automation Status",
            expand=False
        )
        console.print(panel)
    
    else:
        console.print("Please specify --start, --stop, or --status")


@compliance.command()
@click.option('--limit', default=10, help='Number of violations to show')
@click.option('--severity', 
              type=click.Choice(['low', 'medium', 'high', 'critical']),
              help='Filter by severity')
@click.pass_context
async def violations(ctx, limit, severity):
    """List compliance violations."""
    
    automation_engine = ctx.obj['compliance_automation']
    violations = list(automation_engine.violations.values())
    
    # Filter by severity if specified
    if severity:
        from ..infrastructure.governance.ml_governance_framework import GovernanceRisk
        severity_enum = GovernanceRisk(severity)
        violations = [v for v in violations if v.severity == severity_enum]
    
    # Sort by detection time (most recent first)
    violations.sort(key=lambda v: v.detected_at, reverse=True)
    violations = violations[:limit]
    
    if not violations:
        console.print("No violations found.")
        return
    
    table = Table(title="Compliance Violations")
    table.add_column("Violation ID", style="cyan")
    table.add_column("Type", style="bright_white")
    table.add_column("Severity", style="red")
    table.add_column("Model ID", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Detected", style="blue")
    
    for violation in violations:
        severity_color = {
            "low": "green",
            "medium": "yellow", 
            "high": "red",
            "critical": "bright_red"
        }.get(violation.severity.value, "white")
        
        table.add_row(
            violation.violation_id[:8] + "...",
            violation.violation_type,
            f"[{severity_color}]{violation.severity.value.upper()}[/{severity_color}]",
            violation.model_id[:8] + "..." if violation.model_id else "N/A",
            violation.status,
            violation.detected_at.strftime("%Y-%m-%d %H:%M")
        )
    
    console.print(table)


@governance.group()
def regulatory():
    """Regulatory compliance management."""
    pass


@regulatory.command()
@click.option('--model-id', required=True, help='Model ID to assess')
@click.option('--framework',
              type=click.Choice([f.value for f in ComplianceFramework]),
              required=True,
              help='Compliance framework')
@click.option('--scope', default='full_system', help='Assessment scope')
@click.pass_context
async def assess(ctx, model_id, framework, scope):
    """Perform regulatory compliance assessment."""
    
    framework_enum = ComplianceFramework(framework)
    regulatory_manager = ctx.obj['regulatory_manager']
    
    with Progress() as progress:
        task = progress.add_task(f"Assessing {framework} compliance...", total=100)
        
        # Simulate progress updates
        for i in range(0, 101, 20):
            progress.update(task, completed=i)
            await asyncio.sleep(0.5)
        
        assessment_id = await regulatory_manager.assess_regulatory_compliance(
            model_id=model_id,
            framework=framework_enum,
            scope=scope
        )
    
    console.print(f"✅ Assessment completed with ID: [bold blue]{assessment_id}[/bold blue]")
    
    # Show assessment results
    if assessment_id in regulatory_manager.certifications:
        cert = regulatory_manager.certifications[assessment_id]
        
        panel = Panel(
            f"""
[bold]Assessment Results[/bold]

[bold]Compliance Score:[/bold] {cert.compliance_score:.2%}
[bold]Status:[/bold] {cert.status.value}
[bold]Requirements Met:[/bold] {len(cert.requirements_met)}
[bold]Requirements Failed:[/bold] {len(cert.requirements_failed)}
[bold]Findings:[/bold] {len(cert.findings)}
            """.strip(),
            title=f"{framework} Compliance Assessment",
            expand=False
        )
        console.print(panel)


@regulatory.command()
@click.option('--data-source', required=True, help='Data source name')
@click.option('--data-type', required=True, help='Data type')
@click.option('--classification',
              type=click.Choice(['public', 'internal', 'confidential', 'restricted', 'top_secret']),
              required=True,
              help='Data classification')
@click.option('--contains-pii', is_flag=True, help='Contains personally identifiable information')
@click.option('--contains-phi', is_flag=True, help='Contains protected health information')
@click.option('--consent-required', is_flag=True, help='Requires user consent')
@click.pass_context
async def register_data(ctx, data_source, data_type, classification, contains_pii, contains_phi, consent_required):
    """Register data source in compliance inventory."""
    
    from ..infrastructure.governance.regulatory_compliance import DataClassification
    
    classification_enum = DataClassification(classification)
    regulatory_manager = ctx.obj['regulatory_manager']
    
    with console.status(f"Registering data source '{data_source}'..."):
        inventory_id = await regulatory_manager.register_data_inventory(
            data_source=data_source,
            data_type=data_type,
            classification=classification_enum,
            contains_pii=contains_pii,
            contains_phi=contains_phi,
            consent_required=consent_required
        )
    
    console.print(f"✅ Data source registered with ID: [bold blue]{inventory_id}[/bold blue]")


@regulatory.command()
@click.option('--framework',
              type=click.Choice([f.value for f in ComplianceFramework]),
              required=True,
              help='Compliance framework')
@click.option('--format', 
              type=click.Choice(['json', 'yaml', 'table']),
              default='table',
              help='Output format')
@click.option('--output', help='Output file path')
@click.pass_context
async def report(ctx, framework, format, output):
    """Generate regulatory compliance report."""
    
    framework_enum = ComplianceFramework(framework)
    regulatory_manager = ctx.obj['regulatory_manager']
    
    with console.status(f"Generating {framework} compliance report..."):
        report_data = await regulatory_manager.generate_regulatory_report(
            framework=framework_enum
        )
    
    if format == 'table':
        _display_report_as_table(report_data)
    elif format == 'json':
        if output:
            with open(output, 'w') as f:
                json.dump(report_data, f, indent=2)
            console.print(f"✅ Report saved to {output}")
        else:
            console.print_json(data=report_data)
    elif format == 'yaml':
        if output:
            with open(output, 'w') as f:
                yaml.dump(report_data, f, default_flow_style=False)
            console.print(f"✅ Report saved to {output}")
        else:
            console.print(yaml.dump(report_data, default_flow_style=False))


@governance.group()
def audit():
    """Audit trail management."""
    pass


@audit.command()
@click.option('--user-id', help='Filter by user ID')
@click.option('--model-id', help='Filter by model ID')
@click.option('--event-type', help='Filter by event type')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--limit', default=50, help='Number of events to show')
@click.pass_context
async def trail(ctx, user_id, model_id, event_type, start_date, end_date, limit):
    """View audit trail."""
    
    governance = ctx.obj['governance']
    
    # Parse dates
    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None
    event_type_enum = AuditEventType(event_type) if event_type else None
    
    with console.status("Retrieving audit trail..."):
        events = await governance.get_audit_trail(
            user_id=user_id,
            model_id=model_id,
            event_type=event_type_enum,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
    
    if not events:
        console.print("No audit events found.")
        return
    
    table = Table(title="Audit Trail")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Event Type", style="bright_white")
    table.add_column("User ID", style="yellow")
    table.add_column("Action", style="green")
    table.add_column("Resource", style="blue")
    table.add_column("Success", style="magenta")
    
    for event in events:
        table.add_row(
            event["timestamp"][:19],  # Remove milliseconds
            event["event_type"],
            event["user_id"],
            event["action"],
            event["resource"][:30] + "..." if len(event["resource"]) > 30 else event["resource"],
            "✅" if event["success"] else "❌"
        )
    
    console.print(table)


@audit.command()
@click.option('--event-type', required=True, help='Event type')
@click.option('--user-id', required=True, help='User ID')  
@click.option('--action', required=True, help='Action performed')
@click.option('--resource', required=True, help='Resource affected')
@click.option('--model-id', help='Model ID')
@click.option('--details', help='Event details (JSON)')
@click.option('--success', is_flag=True, default=True, help='Event success status')
@click.pass_context
async def log(ctx, event_type, user_id, action, resource, model_id, details, success):
    """Log an audit event."""
    
    governance = ctx.obj['governance']
    
    try:
        event_type_enum = AuditEventType(event_type)
        details_dict = json.loads(details) if details else {}
        
        with console.status("Logging audit event..."):
            event_id = await governance.log_audit_event(
                event_type=event_type_enum,
                user_id=user_id,
                action=action,
                resource=resource,
                model_id=model_id,
                details=details_dict,
                success=success
            )
        
        console.print(f"✅ Audit event logged with ID: [bold blue]{event_id}[/bold blue]")
        
    except ValueError as e:
        console.print(f"❌ Invalid event type: {event_type}")
    except json.JSONDecodeError:
        console.print("❌ Invalid JSON format for details")
    except Exception as e:
        console.print(f"❌ Error logging event: {str(e)}")


@governance.group()
def dashboard():
    """Governance dashboard management."""
    pass


@dashboard.command()
@click.option('--start', is_flag=True, help='Start dashboard monitoring')
@click.option('--stop', is_flag=True, help='Stop dashboard monitoring')
@click.option('--status', is_flag=True, help='Show dashboard status')
@click.pass_context
async def monitor(ctx, start, stop, status):
    """Manage dashboard monitoring."""
    
    dashboard = ctx.obj['dashboard']
    
    if start:
        with console.status("Starting dashboard monitoring..."):
            await dashboard.start_monitoring()
        console.print("✅ Dashboard monitoring started")
        
    elif stop:
        with console.status("Stopping dashboard monitoring..."):
            await dashboard.stop_monitoring()
        console.print("✅ Dashboard monitoring stopped")
        
    elif status:
        status_info = {
            "running": dashboard.is_running,
            "widgets": len(dashboard.widgets),
            "alerts": len(dashboard.alerts),
            "cached_data": len(dashboard.data_cache)
        }
        
        panel = Panel(
            f"""
[bold]Dashboard Status:[/bold] {'🟢 Running' if status_info['running'] else '🔴 Stopped'}
[bold]Widgets:[/bold] {status_info['widgets']}
[bold]Alerts:[/bold] {status_info['alerts']}
[bold]Cached Data:[/bold] {status_info['cached_data']} entries
            """.strip(),
            title="Dashboard Status",
            expand=False
        )
        console.print(panel)
    
    else:
        console.print("Please specify --start, --stop, or --status")


@dashboard.command()
@click.option('--format',
              type=click.Choice(['json', 'summary']),
              default='summary',
              help='Output format')
@click.pass_context
async def data(ctx, format):
    """Get dashboard data."""
    
    dashboard = ctx.obj['dashboard']
    
    with console.status("Retrieving dashboard data..."):
        dashboard_data = await dashboard.get_dashboard_data()
    
    if format == 'json':
        console.print_json(data=dashboard_data)
    else:
        _display_dashboard_summary(dashboard_data)


@governance.command()
@click.option('--component',
              type=click.Choice(['governance', 'compliance', 'regulatory', 'dashboard']),
              help='Specific component to check')
@click.pass_context
async def health(ctx, component):
    """Check governance system health."""
    
    health_status = {}
    
    if not component or component == 'governance':
        governance = ctx.obj['governance']
        health_status['governance'] = {
            "status": "healthy",
            "policies": len(governance.policies),
            "audit_events": len(governance.audit_trail),
            "approval_requests": len(governance.approval_requests)
        }
    
    if not component or component == 'compliance':
        automation = ctx.obj['compliance_automation']
        health_status['compliance'] = {
            "status": "healthy" if automation.is_running else "stopped",
            "rules": len(automation.compliance_rules),
            "violations": len(automation.violations),
            "remediation_tasks": len(automation.remediation_tasks)
        }
    
    if not component or component == 'regulatory':
        regulatory = ctx.obj['regulatory_manager']
        health_status['regulatory'] = {
            "status": "healthy",
            "requirements": len(regulatory.requirements),
            "certifications": len(regulatory.certifications),
            "data_inventory": len(regulatory.data_inventory)
        }
    
    if not component or component == 'dashboard':
        dashboard = ctx.obj['dashboard']
        health_status['dashboard'] = {
            "status": "healthy" if dashboard.is_running else "stopped",
            "widgets": len(dashboard.widgets),
            "alerts": len(dashboard.alerts)
        }
    
    _display_health_status(health_status)


def _display_report_as_table(report_data: Dict[str, Any]) -> None:
    """Display compliance report as formatted table."""
    
    # Executive Summary
    summary = report_data.get("executive_summary", {})
    
    summary_table = Table(title="Executive Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bright_white")
    
    summary_table.add_row("Overall Compliance Score", f"{summary.get('overall_compliance_score', 0):.2%}")
    summary_table.add_row("Total Requirements", str(summary.get('total_requirements', 0)))
    summary_table.add_row("Compliant Systems", str(summary.get('compliant_systems', 0)))
    summary_table.add_row("Non-Compliant Systems", str(summary.get('non_compliant_systems', 0)))
    summary_table.add_row("Pending Reviews", str(summary.get('pending_reviews', 0)))
    
    console.print(summary_table)
    console.print()
    
    # Compliance Gaps
    gaps = report_data.get("compliance_gaps", [])
    if gaps:
        gaps_table = Table(title="Compliance Gaps")
        gaps_table.add_column("Title", style="bright_white")
        gaps_table.add_column("Risk Level", style="red")
        gaps_table.add_column("Gap Type", style="yellow")
        gaps_table.add_column("Priority", style="green")
        
        for gap in gaps[:10]:  # Show top 10
            gaps_table.add_row(
                gap.get("title", "N/A"),
                gap.get("risk_level", "unknown"),
                gap.get("gap_type", "unknown"),
                gap.get("remediation_priority", "unknown")
            )
        
        console.print(gaps_table)


def _display_dashboard_summary(dashboard_data: Dict[str, Any]) -> None:
    """Display dashboard data as summary."""
    
    summary = dashboard_data.get("summary", {})
    
    # Overall Health
    health_score = summary.get("overall_health_score", 0)
    health_status = summary.get("health_status", "unknown")
    
    health_panel = Panel(
        f"""
[bold]Overall Health Score:[/bold] {health_score:.2%}
[bold]Health Status:[/bold] {health_status.upper()}
[bold]Recent Violations:[/bold] {summary.get('recent_violations', 0)}
[bold]Pending Approvals:[/bold] {summary.get('pending_approvals', 0)}
        """.strip(),
        title="Governance Health Overview",
        expand=False
    )
    console.print(health_panel)
    
    # Key Insights
    insights = summary.get("key_insights", [])
    if insights:
        console.print("\n[bold]Key Insights:[/bold]")
        for insight in insights:
            console.print(f"• {insight}")
    
    # Recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            console.print(f"• {rec}")


def _display_health_status(health_status: Dict[str, Any]) -> None:
    """Display system health status."""
    
    for component, status in health_status.items():
        status_icon = "🟢" if status["status"] == "healthy" else "🔴"
        
        panel_content = f"[bold]Status:[/bold] {status_icon} {status['status'].upper()}\n"
        
        # Add component-specific metrics
        for key, value in status.items():
            if key != "status":
                panel_content += f"[bold]{key.replace('_', ' ').title()}:[/bold] {value}\n"
        
        panel = Panel(
            panel_content.strip(),
            title=f"{component.title()} Health",
            expand=False
        )
        console.print(panel)


if __name__ == "__main__":
    # Support async CLI commands
    def async_command(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to commands that need it
    for command in [
        policies.commands['create'],
        policies.commands['add-rule'],
        compliance.commands['add-rule'],
        compliance.commands['automation'],
        regulatory.commands['assess'],
        regulatory.commands['register-data'],
        regulatory.commands['report'],
        audit.commands['trail'],
        audit.commands['log'],
        dashboard.commands['monitor'],
        dashboard.commands['data'],
        governance.commands['health']
    ]:
        command.callback = async_command(command.callback)
    
    governance()