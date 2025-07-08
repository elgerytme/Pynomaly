"""CLI commands for multi-tenant management."""

import asyncio
import json
from uuid import UUID

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from pynomaly.application.services.multi_tenant_service import MultiTenantService
from pynomaly.domain.entities.tenant import SubscriptionTier, TenantStatus

console = Console()


@click.group(name="tenant")
def tenant_commands():
    """Multi-tenant management commands."""
    pass


@tenant_commands.command()
@click.option("--name", required=True, help="Unique tenant name")
@click.option("--display-name", help="Display name for the tenant")
@click.option("--email", required=True, help="Contact email for the tenant")
@click.option(
    "--tier",
    type=click.Choice(["free", "basic", "professional", "enterprise"]),
    default="free",
    help="Subscription tier",
)
@click.option("--description", help="Description of the tenant")
@click.option("--admin-user", help="Admin user ID (UUID)")
@click.option("--auto-activate", is_flag=True, help="Automatically activate the tenant")
def create(
    name: str,
    display_name: str | None,
    email: str,
    tier: str,
    description: str | None,
    admin_user: str | None,
    auto_activate: bool,
):
    """Create a new tenant."""

    async def create_tenant():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize multi-tenant service
            task1 = progress.add_task("Initializing tenant service...", total=None)
            tenant_service = MultiTenantService()
            progress.update(task1, completed=True)

            # Validate inputs
            task2 = progress.add_task("Validating tenant information...", total=None)

            if not display_name:
                display_name = name.title()

            admin_user_id = None
            if admin_user:
                try:
                    admin_user_id = UUID(admin_user)
                except ValueError:
                    console.print(
                        f"[red]Error: Invalid admin user UUID: {admin_user}[/red]"
                    )
                    return

            subscription_tier = SubscriptionTier(tier)
            progress.update(task2, completed=True)

            # Create tenant
            task3 = progress.add_task("Creating tenant...", total=None)

            try:
                tenant = await tenant_service.create_tenant(
                    name=name,
                    display_name=display_name,
                    contact_email=email,
                    subscription_tier=subscription_tier,
                    admin_user_id=admin_user_id,
                    description=description or "",
                )

                progress.update(task3, completed=True)

                # Auto-activate if requested
                if auto_activate:
                    task4 = progress.add_task("Activating tenant...", total=None)
                    await tenant_service.activate_tenant(tenant.tenant_id)
                    progress.update(task4, completed=True)

                # Display tenant information
                _display_tenant_info(tenant)

                console.print(f"[green]✓ Tenant '{name}' created successfully![/green]")
                console.print(f"Tenant ID: {tenant.tenant_id}")

                if auto_activate:
                    console.print("[green]✓ Tenant activated automatically[/green]")
                else:
                    console.print(
                        f"[yellow]Use 'pynomaly tenant activate {tenant.tenant_id}' to activate[/yellow]"
                    )

            except Exception as e:
                console.print(f"[red]Error creating tenant: {e}[/red]")
                return

    asyncio.run(create_tenant())


@tenant_commands.command()
@click.argument("tenant_id", type=str)
def activate(tenant_id: str):
    """Activate a tenant."""

    async def activate_tenant():
        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
            return

        tenant_service = MultiTenantService()

        success = await tenant_service.activate_tenant(tenant_uuid)

        if success:
            console.print(f"[green]✓ Tenant {tenant_id} activated successfully[/green]")
        else:
            console.print(f"[red]Error: Could not activate tenant {tenant_id}[/red]")

    asyncio.run(activate_tenant())


@tenant_commands.command()
@click.argument("tenant_id", type=str)
@click.option("--reason", help="Reason for suspension")
def suspend(tenant_id: str, reason: str | None):
    """Suspend a tenant."""

    async def suspend_tenant():
        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
            return

        tenant_service = MultiTenantService()

        # Confirm suspension
        if not Confirm.ask(f"Are you sure you want to suspend tenant {tenant_id}?"):
            console.print("[yellow]Suspension cancelled[/yellow]")
            return

        success = await tenant_service.suspend_tenant(tenant_uuid, reason or "")

        if success:
            console.print(f"[green]✓ Tenant {tenant_id} suspended successfully[/green]")
            if reason:
                console.print(f"Reason: {reason}")
        else:
            console.print(f"[red]Error: Could not suspend tenant {tenant_id}[/red]")

    asyncio.run(suspend_tenant())


@tenant_commands.command()
@click.argument("tenant_id", type=str)
def deactivate(tenant_id: str):
    """Deactivate a tenant."""

    async def deactivate_tenant():
        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            console.print(f"[red]Error: Invalid tenant ID: {tenant_id}[/red]")
            return

        tenant_service = MultiTenantService()

        # Confirm deactivation
        if not Confirm.ask(
            f"Are you sure you want to deactivate tenant {tenant_id}? This action cannot be easily undone."
        ):
            console.print("[yellow]Deactivation cancelled[/yellow]")
            return

        success = await tenant_service.deactivate_tenant(tenant_uuid)

        if success:
            console.print(
                f"[green]✓ Tenant {tenant_id} deactivated successfully[/green]"
            )
        else:
            console.print(f"[red]Error: Could not deactivate tenant {tenant_id}[/red]")

    asyncio.run(deactivate_tenant())


@tenant_commands.command()
@click.option(
    "--status",
    type=click.Choice(["active", "suspended", "deactivated", "pending_activation"]),
    help="Filter by tenant status",
)
@click.option(
    "--tier",
    type=click.Choice(["free", "basic", "professional", "enterprise"]),
    help="Filter by subscription tier",
)
@click.option(
    "--limit", type=int, default=50, help="Maximum number of tenants to display"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list(status: str | None, tier: str | None, limit: int, output_format: str):
    """List tenants with optional filtering."""

    async def list_tenants():
        tenant_service = MultiTenantService()

        # Convert string values to enums
        status_filter = TenantStatus(status) if status else None
        tier_filter = SubscriptionTier(tier) if tier else None

        tenants = await tenant_service.list_tenants(
            status=status_filter, subscription_tier=tier_filter, limit=limit
        )

        if not tenants:
            console.print("[yellow]No tenants found matching the criteria[/yellow]")
            return

        if output_format == "json":
            tenant_data = [tenant.to_dict() for tenant in tenants]
            console.print(json.dumps(tenant_data, indent=2, default=str))
        else:
            _display_tenants_table(tenants)

    asyncio.run(list_tenants())


@tenant_commands.command()
@click.argument("tenant_identifier", type=str)
def show(tenant_identifier: str):
    """Show detailed information about a tenant (by ID or name)."""

    async def show_tenant():
        tenant_service = MultiTenantService()

        # Try to get tenant by ID first, then by name
        tenant = None
        try:
            tenant_uuid = UUID(tenant_identifier)
            tenant = await tenant_service.get_tenant(tenant_uuid)
        except ValueError:
            # Not a UUID, try by name
            tenant = await tenant_service.get_tenant_by_name(tenant_identifier)

        if not tenant:
            console.print(f"[red]Tenant not found: {tenant_identifier}[/red]")
            return

        # Display detailed tenant information
        _display_tenant_detail(tenant)

        # Show usage summary
        usage_summary = await tenant_service.get_tenant_usage_summary(tenant.tenant_id)
        if usage_summary:
            _display_usage_summary(usage_summary)

    asyncio.run(show_tenant())


@tenant_commands.command()
@click.argument("tenant_identifier", type=str)
@click.argument(
    "new_tier", type=click.Choice(["free", "basic", "professional", "enterprise"])
)
def upgrade(tenant_identifier: str, new_tier: str):
    """Upgrade tenant subscription tier."""

    async def upgrade_tenant():
        tenant_service = MultiTenantService()

        # Get tenant
        tenant = None
        try:
            tenant_uuid = UUID(tenant_identifier)
            tenant = await tenant_service.get_tenant(tenant_uuid)
        except ValueError:
            tenant = await tenant_service.get_tenant_by_name(tenant_identifier)

        if not tenant:
            console.print(f"[red]Tenant not found: {tenant_identifier}[/red]")
            return

        new_subscription_tier = SubscriptionTier(new_tier)

        # Check if it's actually an upgrade
        tier_order = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.BASIC: 1,
            SubscriptionTier.PROFESSIONAL: 2,
            SubscriptionTier.ENTERPRISE: 3,
        }

        if tier_order[new_subscription_tier] <= tier_order[tenant.subscription_tier]:
            console.print(
                f"[yellow]Warning: This is not an upgrade from {tenant.subscription_tier.value} to {new_tier}[/yellow]"
            )
            if not Confirm.ask("Continue anyway?"):
                return

        success = await tenant_service.upgrade_tenant_subscription(
            tenant.tenant_id, new_subscription_tier
        )

        if success:
            console.print(
                f"[green]✓ Tenant {tenant.name} upgraded to {new_tier} tier[/green]"
            )
        else:
            console.print("[red]Error upgrading tenant subscription[/red]")

    asyncio.run(upgrade_tenant())


@tenant_commands.command()
@click.argument("tenant_identifier", type=str)
def usage(tenant_identifier: str):
    """Show detailed usage information for a tenant."""

    async def show_usage():
        tenant_service = MultiTenantService()

        # Get tenant
        tenant = None
        try:
            tenant_uuid = UUID(tenant_identifier)
            tenant = await tenant_service.get_tenant(tenant_uuid)
        except ValueError:
            tenant = await tenant_service.get_tenant_by_name(tenant_identifier)

        if not tenant:
            console.print(f"[red]Tenant not found: {tenant_identifier}[/red]")
            return

        # Get detailed metrics
        metrics = await tenant_service.get_tenant_metrics(tenant.tenant_id)

        if not metrics:
            console.print("[red]Could not retrieve tenant metrics[/red]")
            return

        _display_tenant_metrics(metrics)

    asyncio.run(show_usage())


@tenant_commands.command()
@click.argument("tenant_identifier", type=str)
def reset_quotas(tenant_identifier: str):
    """Reset resource quotas for a tenant (new billing period)."""

    async def reset_tenant_quotas():
        tenant_service = MultiTenantService()

        # Get tenant
        tenant = None
        try:
            tenant_uuid = UUID(tenant_identifier)
            tenant = await tenant_service.get_tenant(tenant_uuid)
        except ValueError:
            tenant = await tenant_service.get_tenant_by_name(tenant_identifier)

        if not tenant:
            console.print(f"[red]Tenant not found: {tenant_identifier}[/red]")
            return

        # Confirm reset
        if not Confirm.ask(f"Reset all quotas for tenant {tenant.name}?"):
            console.print("[yellow]Quota reset cancelled[/yellow]")
            return

        success = await tenant_service.reset_tenant_quotas(tenant.tenant_id)

        if success:
            console.print(f"[green]✓ Quotas reset for tenant {tenant.name}[/green]")
        else:
            console.print("[red]Error resetting quotas[/red]")

    asyncio.run(reset_tenant_quotas())


@tenant_commands.command()
def stats():
    """Show tenant statistics and overview."""

    async def show_stats():
        tenant_service = MultiTenantService()

        # Get all tenants for statistics
        all_tenants = await tenant_service.list_tenants(limit=1000)

        if not all_tenants:
            console.print("[yellow]No tenants found[/yellow]")
            return

        # Calculate statistics
        stats = {
            "total_tenants": len(all_tenants),
            "by_status": {},
            "by_tier": {},
            "active_tenants": 0,
            "total_usage": {"api_requests": 0, "cpu_hours": 0.0, "storage_gb": 0.0},
        }

        for tenant in all_tenants:
            # Count by status
            status = tenant.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by tier
            tier = tenant.subscription_tier.value
            stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1

            # Count active tenants
            if tenant.is_active():
                stats["active_tenants"] += 1

            # Aggregate usage
            stats["total_usage"]["api_requests"] += tenant.total_api_requests
            stats["total_usage"]["cpu_hours"] += tenant.total_cpu_hours
            stats["total_usage"]["storage_gb"] += tenant.total_storage_gb

        _display_tenant_stats(stats)

    asyncio.run(show_stats())


# Helper functions for display


def _display_tenant_info(tenant):
    """Display basic tenant information."""
    console.print(
        Panel(
            f"[bold blue]Tenant Created[/bold blue]\n\n"
            f"ID: {tenant.tenant_id}\n"
            f"Name: {tenant.name}\n"
            f"Display Name: {tenant.display_name}\n"
            f"Email: {tenant.contact_email}\n"
            f"Tier: {tenant.subscription_tier.value}\n"
            f"Status: {tenant.status.value}\n"
            f"Database Schema: {tenant.database_schema}",
            title="Tenant Information",
        )
    )


def _display_tenants_table(tenants):
    """Display tenants in a table format."""
    table = Table(title="Tenants")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Tier", style="blue")
    table.add_column("Email", style="magenta")
    table.add_column("Created", style="dim")

    for tenant in tenants:
        status_color = {
            "active": "green",
            "suspended": "yellow",
            "deactivated": "red",
            "pending_activation": "blue",
        }.get(tenant.status.value, "white")

        table.add_row(
            tenant.name,
            tenant.display_name,
            f"[{status_color}]{tenant.status.value}[/{status_color}]",
            tenant.subscription_tier.value,
            tenant.contact_email,
            tenant.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


def _display_tenant_detail(tenant):
    """Display detailed tenant information."""
    # Basic info panel
    console.print(
        Panel(
            f"[bold blue]Tenant Details[/bold blue]\n\n"
            f"ID: {tenant.tenant_id}\n"
            f"Name: {tenant.name}\n"
            f"Display Name: {tenant.display_name}\n"
            f"Description: {tenant.description}\n"
            f"Status: {tenant.status.value}\n"
            f"Subscription Tier: {tenant.subscription_tier.value}\n"
            f"Contact Email: {tenant.contact_email}\n"
            f"Admin User ID: {tenant.admin_user_id or 'None'}\n"
            f"Created: {tenant.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Last Activity: {tenant.last_activity.strftime('%Y-%m-%d %H:%M:%S') if tenant.last_activity else 'Never'}",
            title="Basic Information",
        )
    )

    # Configuration panel
    config = tenant.configuration
    console.print(
        Panel(
            f"[bold green]Configuration[/bold green]\n\n"
            f"Max Concurrent Jobs: {config.max_concurrent_jobs}\n"
            f"Max Model Size: {config.max_model_size_mb} MB\n"
            f"GPU Access: {'Enabled' if config.enable_gpu_access else 'Disabled'}\n"
            f"Auto Scaling: {'Enabled' if config.enable_auto_scaling else 'Disabled'}\n"
            f"Advanced Analytics: {'Enabled' if config.enable_advanced_analytics else 'Disabled'}\n"
            f"Data Retention: {config.data_retention_days} days\n"
            f"Monitoring Level: {config.monitoring_level}",
            title="Configuration",
        )
    )


def _display_usage_summary(usage_summary):
    """Display tenant usage summary."""
    quotas = usage_summary.get("quotas", {})

    quota_table = Table(title="Resource Quotas")
    quota_table.add_column("Resource", style="cyan")
    quota_table.add_column("Used", style="yellow")
    quota_table.add_column("Limit", style="green")
    quota_table.add_column("Usage %", style="blue")
    quota_table.add_column("Status", style="magenta")

    for quota_type, quota_info in quotas.items():
        usage_pct = quota_info["usage_percentage"]
        limit_str = (
            str(quota_info["limit"]) if quota_info["limit"] != "unlimited" else "∞"
        )

        status_color = (
            "red"
            if quota_info["is_exceeded"]
            else "yellow"
            if usage_pct > 80
            else "green"
        )
        status_text = "EXCEEDED" if quota_info["is_exceeded"] else "OK"

        quota_table.add_row(
            quota_type.replace("_", " ").title(),
            str(quota_info["used"]),
            limit_str,
            f"{usage_pct:.1f}%",
            f"[{status_color}]{status_text}[/{status_color}]",
        )

    console.print(quota_table)


def _display_tenant_metrics(metrics):
    """Display comprehensive tenant metrics."""
    tenant_info = metrics.get("tenant_info", {})
    metrics.get("quota_status", {})
    resource_usage = metrics.get("resource_usage", {})
    billing = metrics.get("billing", {})

    # Tenant info
    console.print(
        Panel(
            f"[bold blue]Tenant Metrics[/bold blue]\n\n"
            f"Name: {tenant_info.get('name', 'Unknown')}\n"
            f"Status: {tenant_info.get('status', 'Unknown')}\n"
            f"Tier: {tenant_info.get('subscription_tier', 'Unknown')}\n"
            f"Active Jobs: {metrics.get('active_jobs', 0)}\n"
            f"Created: {tenant_info.get('created_at', 'Unknown')}\n"
            f"Last Activity: {tenant_info.get('last_activity', 'Never')}",
            title="Overview",
        )
    )

    # Resource usage table
    if resource_usage:
        usage_table = Table(title="Real-time Resource Usage")
        usage_table.add_column("Resource", style="cyan")
        usage_table.add_column("Current Usage", style="yellow")

        for resource, usage in resource_usage.items():
            usage_table.add_row(resource.replace("_", " ").title(), f"{usage:.2f}")

        console.print(usage_table)

    # Billing information
    if billing:
        console.print(
            Panel(
                f"[bold green]Billing Information[/bold green]\n\n"
                f"Period Start: {billing.get('current_period_start', 'Unknown')}\n"
                f"Current Charges: ${billing.get('current_period_charges', 0.0):.2f}\n"
                f"Usage Events: {len(billing.get('usage_history', []))}",
                title="Billing",
            )
        )


def _display_tenant_stats(stats):
    """Display tenant statistics."""
    console.print(
        Panel(
            f"[bold blue]Tenant Statistics[/bold blue]\n\n"
            f"Total Tenants: {stats['total_tenants']}\n"
            f"Active Tenants: {stats['active_tenants']}\n"
            f"Total API Requests: {stats['total_usage']['api_requests']:,}\n"
            f"Total CPU Hours: {stats['total_usage']['cpu_hours']:.2f}\n"
            f"Total Storage: {stats['total_usage']['storage_gb']:.2f} GB",
            title="Overview",
        )
    )

    # Status distribution
    status_table = Table(title="Distribution by Status")
    status_table.add_column("Status", style="cyan")
    status_table.add_column("Count", style="yellow")
    status_table.add_column("Percentage", style="green")

    for status, count in stats["by_status"].items():
        percentage = (count / stats["total_tenants"]) * 100
        status_table.add_row(status.title(), str(count), f"{percentage:.1f}%")

    console.print(status_table)

    # Tier distribution
    tier_table = Table(title="Distribution by Subscription Tier")
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Count", style="yellow")
    tier_table.add_column("Percentage", style="green")

    for tier, count in stats["by_tier"].items():
        percentage = (count / stats["total_tenants"]) * 100
        tier_table.add_row(tier.title(), str(count), f"{percentage:.1f}%")

    console.print(tier_table)


if __name__ == "__main__":
    tenant_commands()
