"""CLI commands for cost optimization and resource management."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.layout import Layout

from pynomaly.application.services.cost_optimization_service import CostOptimizationService
from pynomaly.domain.entities.cost_optimization import (
    CloudResource, CostBudget, ResourceType, CloudProvider, OptimizationStrategy,
    ResourceUsageMetrics, ResourceCost, AlertMetadata
)


console = Console()


@click.group(name="cost")
def cost_commands():
    """Cost optimization and resource management commands."""
    pass


@cost_commands.command()
@click.option("--tenant-id", help="Filter by tenant ID (UUID)")
@click.option("--days", type=int, default=30, help="Analysis period in days")
@click.option("--format", "output_format", type=click.Choice(["console", "json"]), 
              default="console", help="Output format")
@click.option("--output-file", help="Output file for analysis results")
def analyze(tenant_id: Optional[str], days: int, output_format: str, output_file: Optional[str]):
    """Analyze cost trends and identify optimization opportunities."""
    
    async def run_analysis():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize service
            task1 = progress.add_task("Initializing cost optimization service...", total=None)
            cost_service = CostOptimizationService()
            progress.update(task1, completed=True)
            
            # Parse tenant ID if provided
            tenant_uuid = None
            if tenant_id:
                try:
                    tenant_uuid = UUID(tenant_id)
                except ValueError:
                    console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                    return
            
            # Run cost analysis
            task2 = progress.add_task("Analyzing cost trends and patterns...", total=None)
            analysis = await cost_service.analyze_costs(tenant_uuid)
            progress.update(task2, completed=True)
            
            # Generate output
            if output_format == "json":
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(analysis, f, indent=2, default=str)
                    console.print(f"[green]Analysis saved to {output_file}[/green]")
                else:
                    console.print(json.dumps(analysis, indent=2, default=str))
            else:
                _display_cost_analysis(analysis, days)
    
    asyncio.run(run_analysis())


@cost_commands.command()
@click.option("--strategy", type=click.Choice([
    "aggressive", "balanced", "conservative", "performance_first", "cost_first"
]), default="balanced", help="Optimization strategy")
@click.option("--tenant-id", help="Filter by tenant ID (UUID)")
@click.option("--target-savings", type=float, default=0.2, help="Target savings percentage (0.0-1.0)")
@click.option("--auto-implement", is_flag=True, help="Automatically implement low-risk recommendations")
def optimize(strategy: str, tenant_id: Optional[str], target_savings: float, auto_implement: bool):
    """Generate and optionally implement cost optimization recommendations."""
    
    async def run_optimization():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize service
            task1 = progress.add_task("Initializing cost optimization service...", total=None)
            cost_service = CostOptimizationService()
            progress.update(task1, completed=True)
            
            # Parse inputs
            tenant_uuid = None
            if tenant_id:
                try:
                    tenant_uuid = UUID(tenant_id)
                except ValueError:
                    console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                    return
            
            opt_strategy = OptimizationStrategy(strategy)
            
            # Generate optimization plan
            task2 = progress.add_task("Generating optimization plan...", total=None)
            plan = await cost_service.generate_optimization_plan(
                strategy=opt_strategy,
                tenant_id=tenant_uuid,
                target_savings_percent=target_savings
            )
            progress.update(task2, completed=True)
            
            # Display plan
            _display_optimization_plan(plan)
            
            # Auto-implement if requested
            if auto_implement:
                quick_wins = plan.get_quick_wins()
                if quick_wins:
                    task3 = progress.add_task(f"Implementing {len(quick_wins)} quick wins...", total=len(quick_wins))
                    
                    for i, rec in enumerate(quick_wins):
                        result = await cost_service.implement_recommendation(rec.recommendation_id)
                        if result["success"]:
                            console.print(f"[green]✓ Implemented: {rec.title}[/green]")
                        else:
                            console.print(f"[red]✗ Failed: {rec.title} - {result.get('message', 'Unknown error')}[/red]")
                        
                        progress.update(task3, advance=1)
                    
                    console.print(f"[green]Auto-implemented {len(quick_wins)} recommendations[/green]")
    
    asyncio.run(run_optimization())


@cost_commands.command()
@click.argument("plan_id", type=str)
@click.option("--recommendation-id", help="Implement specific recommendation by ID")
@click.option("--phase", type=int, help="Implement specific phase (1-4)")
@click.option("--dry-run", is_flag=True, help="Show what would be implemented without actually doing it")
def implement(plan_id: str, recommendation_id: Optional[str], phase: Optional[int], dry_run: bool):
    """Implement recommendations from an optimization plan."""
    
    async def run_implementation():
        cost_service = CostOptimizationService()
        
        try:
            plan_uuid = UUID(plan_id)
        except ValueError:
            console.print(f"[red]Error: Invalid plan ID: {plan_id}[/red]")
            return
        
        plan = cost_service.optimization_plans.get(plan_uuid)
        if not plan:
            console.print(f"[red]Plan not found: {plan_id}[/red]")
            return
        
        # Determine which recommendations to implement
        recommendations_to_implement = []
        
        if recommendation_id:
            # Implement specific recommendation
            try:
                rec_uuid = UUID(recommendation_id)
                rec = next((r for r in plan.recommendations if r.recommendation_id == rec_uuid), None)
                if rec:
                    recommendations_to_implement = [rec]
                else:
                    console.print(f"[red]Recommendation not found: {recommendation_id}[/red]")
                    return
            except ValueError:
                console.print(f"[red]Error: Invalid recommendation ID: {recommendation_id}[/red]")
                return
        
        elif phase is not None:
            # Implement specific phase
            phases = plan.get_implementation_phases()
            if 1 <= phase <= len(phases):
                recommendations_to_implement = phases[phase - 1]
            else:
                console.print(f"[red]Invalid phase: {phase}. Available phases: 1-{len(phases)}[/red]")
                return
        
        else:
            # Implement all recommendations
            recommendations_to_implement = plan.recommendations
        
        if not recommendations_to_implement:
            console.print("[yellow]No recommendations to implement[/yellow]")
            return
        
        # Show implementation plan
        console.print(f"[blue]Implementation Plan for {len(recommendations_to_implement)} recommendations:[/blue]")
        
        impl_table = Table(title="Recommendations to Implement")
        impl_table.add_column("ID", style="cyan")
        impl_table.add_column("Title", style="yellow")
        impl_table.add_column("Type", style="green")
        impl_table.add_column("Risk", style="red")
        impl_table.add_column("Monthly Savings", style="magenta")
        impl_table.add_column("Implementation Time", style="blue")
        
        total_savings = 0.0
        for rec in recommendations_to_implement:
            total_savings += rec.monthly_savings
            impl_table.add_row(
                str(rec.recommendation_id)[:8] + "...",
                rec.title[:40] + ("..." if len(rec.title) > 40 else ""),
                rec.recommendation_type.value,
                rec.risk_level,
                f"${rec.monthly_savings:.2f}",
                rec.estimated_implementation_time
            )
        
        console.print(impl_table)
        console.print(f"[green]Total Expected Monthly Savings: ${total_savings:.2f}[/green]")
        
        if dry_run:
            console.print("[yellow]Dry run mode - no changes will be made[/yellow]")
            return
        
        # Confirm implementation
        if not click.confirm("Proceed with implementation?"):
            console.print("Implementation cancelled")
            return
        
        # Implement recommendations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Implementing recommendations...", total=len(recommendations_to_implement))
            
            for rec in recommendations_to_implement:
                result = await cost_service.implement_recommendation(rec.recommendation_id)
                
                if result["success"]:
                    console.print(f"[green]✓ {rec.title}[/green]")
                else:
                    console.print(f"[red]✗ {rec.title} - {result.get('message', 'Failed')}[/red]")
                
                progress.update(task, advance=1)
            
            console.print(f"[green]Implementation completed![/green]")
    
    asyncio.run(run_implementation())


@cost_commands.command()
@click.option("--name", required=True, help="Budget name")
@click.option("--monthly-limit", type=float, required=True, help="Monthly budget limit")
@click.option("--annual-limit", type=float, help="Annual budget limit")
@click.option("--tenant-id", help="Tenant ID (UUID)")
@click.option("--environments", multiple=True, help="Environments to include")
@click.option("--alert-thresholds", multiple=True, type=float, help="Alert thresholds (0.0-1.0)")
@click.option("--alert-contacts", multiple=True, help="Alert contact emails")
def create_budget(
    name: str,
    monthly_limit: float,
    annual_limit: Optional[float],
    tenant_id: Optional[str],
    environments: List[str],
    alert_thresholds: List[float],
    alert_contacts: List[str]
):
    """Create a cost budget with alerts."""
    
    async def create_new_budget():
        cost_service = CostOptimizationService()
        
        # Parse tenant ID
        tenant_uuid = None
        if tenant_id:
            try:
                tenant_uuid = UUID(tenant_id)
            except ValueError:
                console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                return
        
        # Create budget
        budget = CostBudget(
            name=name,
            description=f"Budget for {name}",
            monthly_limit=monthly_limit,
            annual_limit=annual_limit or monthly_limit * 12,
            tenant_id=tenant_uuid,
            environments=set(environments),
            alert_thresholds=list(alert_thresholds) or [0.5, 0.8, 0.9, 1.0],
            alert_contacts=list(alert_contacts),
            created_by="cli_user"
        )
        
        success = await cost_service.create_budget(budget)
        
        if success:
            console.print(f"[green]✓ Budget '{name}' created successfully![/green]")
            console.print(f"Budget ID: {budget.budget_id}")
            console.print(f"Monthly Limit: ${monthly_limit:,.2f}")
            console.print(f"Annual Limit: ${budget.annual_limit:,.2f}")
            
            if alert_thresholds:
                console.print(f"Alert Thresholds: {', '.join(f'{t*100:.0f}%' for t in alert_thresholds)}")
        else:
            console.print(f"[red]Error creating budget[/red]")
    
    asyncio.run(create_new_budget())


@cost_commands.command()
@click.option("--tenant-id", help="Filter by tenant ID (UUID)")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), 
              default="table", help="Output format")
def list_budgets(tenant_id: Optional[str], output_format: str):
    """List all cost budgets and their status."""
    
    async def list_all_budgets():
        cost_service = CostOptimizationService()
        
        # Parse tenant ID
        tenant_uuid = None
        if tenant_id:
            try:
                tenant_uuid = UUID(tenant_id)
            except ValueError:
                console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                return
        
        # Filter budgets
        budgets = list(cost_service.budgets.values())
        if tenant_uuid:
            budgets = [b for b in budgets if b.tenant_id == tenant_uuid]
        
        if not budgets:
            console.print("[yellow]No budgets found[/yellow]")
            return
        
        if output_format == "json":
            budget_data = [
                {
                    "budget_id": str(b.budget_id),
                    "name": b.name,
                    "monthly_limit": b.monthly_limit,
                    "current_spend": b.current_monthly_spend,
                    "utilization": b.get_monthly_utilization(),
                    "over_budget": b.is_over_budget()
                }
                for b in budgets
            ]
            console.print(json.dumps(budget_data, indent=2))
        else:
            _display_budgets_table(budgets)
        
        # Check for alerts
        alerts = await cost_service.check_budget_alerts()
        if alerts:
            console.print(f"\n[red]⚠ {len(alerts)} budget alerts triggered[/red]")
            for alert in alerts[:3]:  # Show top 3
                console.print(f"  • {alert['budget_name']}: {alert['current_utilization']*100:.1f}% of budget used")
    
    asyncio.run(list_all_budgets())


@cost_commands.command()
@click.option("--tenant-id", help="Filter by tenant ID (UUID)")
@click.option("--resource-type", type=click.Choice([
    "compute", "memory", "storage", "network", "gpu", "database", "container", "function", "cache", "queue"
]), help="Filter by resource type")
@click.option("--environment", help="Filter by environment")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), 
              default="table", help="Output format")
def resources(tenant_id: Optional[str], resource_type: Optional[str], environment: Optional[str], output_format: str):
    """List and analyze cloud resources."""
    
    async def list_resources():
        cost_service = CostOptimizationService()
        
        # Get resource summary
        tenant_uuid = None
        if tenant_id:
            try:
                tenant_uuid = UUID(tenant_id)
            except ValueError:
                console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                return
        
        summary = await cost_service.get_resource_summary(tenant_uuid)
        
        if output_format == "json":
            console.print(json.dumps(summary, indent=2, default=str))
        else:
            _display_resource_summary(summary)
    
    asyncio.run(list_resources())


@cost_commands.command()
@click.option("--tenant-id", help="Filter by tenant ID (UUID)")
def alerts(tenant_id: Optional[str]):
    """Check for cost alerts and budget violations."""
    
    async def check_alerts():
        cost_service = CostOptimizationService()
        
        # Check budget alerts
        budget_alerts = await cost_service.check_budget_alerts()
        
        # Filter by tenant if specified
        if tenant_id:
            try:
                tenant_uuid = UUID(tenant_id)
                budget_alerts = [a for a in budget_alerts if cost_service.budgets.get(UUID(a['budget_id']), CostBudget()).tenant_id == tenant_uuid]
            except ValueError:
                console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
                return
        
        if not budget_alerts:
            console.print("[green]✓ No cost alerts[/green]")
            return
        
        # Display alerts
        console.print(f"[red]⚠ {len(budget_alerts)} active cost alerts[/red]")
        
        alerts_table = Table(title="Cost Alerts")
        alerts_table.add_column("Budget", style="cyan")
        alerts_table.add_column("Alert Type", style="yellow")
        alerts_table.add_column("Severity", style="red")
        alerts_table.add_column("Utilization", style="magenta")
        alerts_table.add_column("Current Spend", style="green")
        alerts_table.add_column("Limit", style="blue")
        alerts_table.add_column("Days to Exhaustion", style="orange")
        
        for alert in budget_alerts:
            severity_color = {
                "critical": "red",
                "high": "orange", 
                "medium": "yellow",
                "low": "green"
            }.get(alert["severity"], "white")
            
            days_left = alert.get("days_until_exhausted")
            days_text = f"{days_left}" if days_left is not None else "N/A"
            
            alerts_table.add_row(
                alert["budget_name"],
                alert["alert_type"],
                f"[{severity_color}]{alert['severity']}[/{severity_color}]",
                f"{alert['current_utilization']*100:.1f}%",
                f"${alert['current_spend']:,.2f}",
                f"${alert['budget_limit']:,.2f}",
                days_text
            )
        
        console.print(alerts_table)
    
    asyncio.run(check_alerts())


@cost_commands.command()
def metrics():
    """Display cost optimization service metrics."""
    
    async def show_metrics():
        cost_service = CostOptimizationService()
        metrics = await cost_service.get_service_metrics()
        
        console.print(Panel(
            f"[bold blue]Cost Optimization Metrics[/bold blue]\n\n"
            f"Total Resources: {metrics['total_resources']:,}\n"
            f"Total Monthly Cost: ${metrics['total_monthly_cost']:,.2f}\n"
            f"Average Cost per Resource: ${metrics['avg_cost_per_resource']:.2f}\n"
            f"Recommendations Generated: {metrics['recommendations_generated']:,}\n"
            f"Recommendations Implemented: {metrics['recommendations_implemented']:,}\n"
            f"Total Savings Identified: ${metrics['total_savings_identified']:,.2f}\n"
            f"Savings Rate: {metrics['savings_rate']:.1f}%\n"
            f"Active Optimization Plans: {metrics['optimization_plans']}\n"
            f"Active Budgets: {metrics['budgets']}",
            title="Service Metrics"
        ))
    
    asyncio.run(show_metrics())


# Helper functions for display

def _display_cost_analysis(analysis: dict, days: int):
    """Display cost analysis results."""
    console.print(Panel(
        f"[bold blue]Cost Analysis - Last {days} Days[/bold blue]\n\n"
        f"Total Monthly Cost: ${analysis.get('total_monthly_cost', 0):,.2f}\n"
        f"Projected Annual Cost: ${analysis.get('projected_annual_cost', 0):,.2f}\n"
        f"7-Day Trend: {analysis.get('cost_trends', {}).get('7d_change', 0)*100:+.1f}%\n"
        f"30-Day Trend: {analysis.get('cost_trends', {}).get('30d_change', 0)*100:+.1f}%\n"
        f"Daily Growth Rate: {analysis.get('cost_trends', {}).get('growth_rate', 0)*100:+.2f}%",
        title="Cost Overview"
    ))
    
    # Cost by resource type
    if analysis.get('cost_by_resource_type'):
        type_table = Table(title="Cost by Resource Type")
        type_table.add_column("Resource Type", style="cyan")
        type_table.add_column("Monthly Cost", style="green")
        type_table.add_column("Percentage", style="yellow")
        
        total_cost = analysis['total_monthly_cost']
        for resource_type, cost in analysis['cost_by_resource_type'].items():
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
            type_table.add_row(
                resource_type.title(),
                f"${cost:,.2f}",
                f"{percentage:.1f}%"
            )
        
        console.print(type_table)
    
    # Top cost drivers
    if analysis.get('top_cost_drivers'):
        console.print("\n[bold]Top Cost Drivers:[/bold]")
        for i, driver in enumerate(analysis['top_cost_drivers'][:5], 1):
            console.print(f"{i}. {driver['name']} ({driver['resource_type']}) - ${driver['monthly_cost']:,.2f}/month")
    
    # Inefficiency indicators
    inefficiency = analysis.get('inefficiency_indicators', {})
    if inefficiency:
        console.print(Panel(
            f"[bold red]Inefficiency Indicators[/bold red]\n\n"
            f"Idle Resources: {inefficiency.get('idle_resources', 0)}\n"
            f"Underutilized Resources: {inefficiency.get('underutilized_resources', 0)}\n"
            f"Estimated Monthly Waste: ${inefficiency.get('total_waste', 0):,.2f}",
            title="Optimization Opportunities"
        ))
    
    # Cost anomalies
    anomalies = analysis.get('cost_anomalies', [])
    if anomalies:
        console.print(f"\n[red]⚠ {len(anomalies)} cost anomalies detected[/red]")
        for anomaly in anomalies[:3]:  # Show top 3
            console.print(f"  • {anomaly['anomaly_type']}: {anomaly['description']}")


def _display_optimization_plan(plan):
    """Display optimization plan details."""
    console.print(Panel(
        f"[bold green]Optimization Plan: {plan.name}[/bold green]\n\n"
        f"Strategy: {plan.strategy.value.title()}\n"
        f"Total Recommendations: {len(plan.recommendations)}\n"
        f"Potential Annual Savings: ${plan.total_potential_savings:,.2f}\n"
        f"Implementation Cost: ${plan.total_implementation_cost:,.2f}\n"
        f"ROI: {plan.calculate_roi():.1f}%\n"
        f"Estimated Implementation: {plan.estimated_implementation_days} days",
        title="Optimization Plan"
    ))
    
    if plan.recommendations:
        # Recommendations table
        rec_table = Table(title="Recommendations")
        rec_table.add_column("Priority", style="red")
        rec_table.add_column("Type", style="cyan")
        rec_table.add_column("Title", style="yellow")
        rec_table.add_column("Monthly Savings", style="green")
        rec_table.add_column("Risk", style="orange")
        rec_table.add_column("Confidence", style="blue")
        rec_table.add_column("Automatable", style="magenta")
        
        for rec in plan.get_recommendations_by_priority()[:10]:  # Top 10
            priority_color = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow", 
                "low": "green",
                "informational": "dim"
            }.get(rec.priority.value, "white")
            
            rec_table.add_row(
                f"[{priority_color}]{rec.priority.value}[/{priority_color}]",
                rec.recommendation_type.value,
                rec.title[:40] + ("..." if len(rec.title) > 40 else ""),
                f"${rec.monthly_savings:.2f}",
                rec.risk_level,
                f"{rec.confidence_score:.2f}",
                "✓" if rec.automation_possible else "✗"
            )
        
        console.print(rec_table)
        
        # Quick wins
        quick_wins = plan.get_quick_wins()
        if quick_wins:
            console.print(f"\n[green]🚀 {len(quick_wins)} quick wins available (automated, low-risk)[/green]")
            total_quick_savings = sum(r.annual_savings for r in quick_wins)
            console.print(f"Quick wins potential: ${total_quick_savings:,.2f}/year")


def _display_budgets_table(budgets):
    """Display budgets in table format."""
    table = Table(title="Cost Budgets")
    table.add_column("Budget Name", style="cyan")
    table.add_column("Monthly Limit", style="green")
    table.add_column("Current Spend", style="yellow")
    table.add_column("Utilization", style="blue")
    table.add_column("Status", style="red")
    table.add_column("Days Left", style="orange")
    
    for budget in budgets:
        utilization = budget.get_monthly_utilization()
        status = "Over Budget" if budget.is_over_budget() else "OK"
        status_color = "red" if budget.is_over_budget() else "green"
        
        days_left = budget.days_until_budget_exhausted()
        days_text = f"{days_left}" if days_left is not None else "N/A"
        
        table.add_row(
            budget.name,
            f"${budget.monthly_limit:,.2f}",
            f"${budget.current_monthly_spend:,.2f}",
            f"{utilization*100:.1f}%",
            f"[{status_color}]{status}[/{status_color}]",
            days_text
        )
    
    console.print(table)


def _display_resource_summary(summary):
    """Display resource summary."""
    console.print(Panel(
        f"[bold blue]Resource Summary[/bold blue]\n\n"
        f"Total Resources: {summary['total_resources']:,}\n"
        f"Total Monthly Cost: ${summary['total_monthly_cost']:,.2f}\n"
        f"Optimizable Resources: {summary['optimization_summary']['optimizable_resources']:,}\n"
        f"Idle Resources: {summary['optimization_summary']['idle_resources']:,}\n"
        f"Underutilized Resources: {summary['optimization_summary']['underutilized_resources']:,}\n"
        f"Overutilized Resources: {summary['optimization_summary']['overutilized_resources']:,}\n"
        f"Potential Monthly Savings: ${summary['optimization_summary']['total_potential_savings']:,.2f}",
        title="Resource Overview"
    ))
    
    # Resource breakdown by type
    breakdown_table = Table(title="Resource Breakdown")
    breakdown_table.add_column("Category", style="cyan")
    breakdown_table.add_column("Count", style="yellow")
    breakdown_table.add_column("Percentage", style="green")
    
    total_resources = summary['total_resources']
    
    for category, counts in summary['resource_breakdown'].items():
        breakdown_table.add_row(f"[bold]{category.replace('_', ' ').title()}[/bold]", "", "")
        
        for item, count in counts.items():
            percentage = (count / total_resources) * 100 if total_resources > 0 else 0
            breakdown_table.add_row(
                f"  {item.replace('_', ' ').title()}",
                str(count),
                f"{percentage:.1f}%"
            )
    
    console.print(breakdown_table)


if __name__ == "__main__":
    cost_commands()