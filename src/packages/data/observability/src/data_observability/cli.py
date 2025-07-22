"""
Data Observability Command-Line Interface

Provides command-line interface for data observability operations including
catalog management, lineage tracking, and quality monitoring.
"""

import click
from typing import Dict, Any
from uuid import UUID

from .application.facades.observability_facade import DataObservabilityFacade
from .infrastructure.di.container import DataObservabilityContainer


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Data Observability CLI."""
    container = DataObservabilityContainer()
    container.wire(modules=[__name__])
    facade = container.observability_facade()
    ctx.ensure_object(dict)
    ctx.obj['facade'] = facade


@cli.command()
@click.argument('asset_name')
@click.option('--asset-type', default='dataset', help='Type of data asset')
@click.option('--location', required=True, help='Asset location')
@click.option('--format', 'data_format', default='parquet', help='Data format')
@click.option('--description', help='Asset description')
@click.option('--owner', help='Asset owner')
@click.option('--domain', help='Business domain')
@click.pass_context
def register(ctx: click.Context, **kwargs) -> None:
    """Register a new data asset in the catalog."""
    facade: DataObservabilityFacade = ctx.obj['facade']
    
    try:
        asset = facade.register_data_asset(**kwargs)
        click.echo(f" Successfully registered asset: {asset.id}")
        click.echo(f"   Name: {asset.name}")
        click.echo(f"   Type: {asset.asset_type}")
        click.echo(f"   Location: {asset.location}")
    except Exception as e:
        click.echo(f"L Failed to register asset: {e}", err=True)


@cli.command()
@click.argument('query')
@click.option('--limit', default=20, help='Maximum number of results')
@click.pass_context
def search(ctx: click.Context, query: str, limit: int) -> None:
    """Search for data assets."""
    facade: DataObservabilityFacade = ctx.obj['facade']
    
    try:
        assets = facade.discover_data_assets(query, limit)
        click.echo(f"= Found {len(assets)} assets matching '{query}':")
        
        for asset in assets:
            click.echo(f"   =� {asset.name} ({asset.asset_type})")
            click.echo(f"      Location: {asset.location}")
            if asset.description:
                click.echo(f"      Description: {asset.description}")
            click.echo()
    except Exception as e:
        click.echo(f"L Search failed: {e}", err=True)


@cli.command()
@click.argument('asset_id')
@click.pass_context
def inspect(ctx: click.Context, asset_id: str) -> None:
    """Get comprehensive view of a data asset."""
    facade: DataObservabilityFacade = ctx.obj['facade']
    
    try:
        asset_uuid = UUID(asset_id)
        view = facade.get_comprehensive_asset_view(asset_uuid)
        
        if 'error' in view:
            click.echo(f"L {view['error']}", err=True)
            return
        
        click.echo(f"=� Comprehensive Asset View: {asset_id}")
        click.echo("=" * 50)
        
        # Asset info
        asset_info = view.get('asset_info', {})
        click.echo(f"=� Asset Information:")
        click.echo(f"   Name: {asset_info.get('name')}")
        click.echo(f"   Type: {asset_info.get('asset_type')}")
        click.echo(f"   Owner: {asset_info.get('owner', 'Unknown')}")
        click.echo()
        
        # Lineage
        lineage = view.get('lineage', {})
        click.echo(f"= Lineage:")
        click.echo(f"   Connected assets: {lineage.get('total_connected_assets', 0)}")
        click.echo(f"   Upstream: {lineage.get('upstream_count', 0)}")
        click.echo(f"   Downstream: {lineage.get('downstream_count', 0)}")
        click.echo()
        
        # Quality
        quality = view.get('quality', {})
        click.echo(f"( Quality:")
        click.echo(f"   Score: {quality.get('quality_score', 'N/A')}")
        click.echo(f"   Active alerts: {quality.get('active_alerts', 0)}")
        click.echo()
        
    except ValueError:
        click.echo(f"L Invalid UUID: {asset_id}", err=True)
    except Exception as e:
        click.echo(f"L Inspection failed: {e}", err=True)


@cli.command()
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """Show data health dashboard."""
    facade: DataObservabilityFacade = ctx.obj['facade']
    
    try:
        dashboard_data = facade.get_data_health_dashboard()
        
        click.echo("<� Data Health Dashboard")
        click.echo("=" * 50)
        
        # Catalog stats
        catalog = dashboard_data.get('catalog', {})
        click.echo(f"=� Catalog:")
        click.echo(f"   Total assets: {catalog.get('total_assets', 0)}")
        click.echo(f"   Domains: {catalog.get('total_domains', 0)}")
        click.echo()
        
        # Pipeline health
        pipeline_health = dashboard_data.get('pipeline_health', {})
        total_pipelines = pipeline_health.get('total_pipelines', 0)
        if total_pipelines > 0:
            click.echo(f"� Pipelines:")
            click.echo(f"   Total: {total_pipelines}")
            click.echo(f"   Healthy: {pipeline_health.get('healthy_pipelines', 0)}")
            click.echo(f"   Degraded: {pipeline_health.get('degraded_pipelines', 0)}")
            click.echo(f"   Unhealthy: {pipeline_health.get('unhealthy_pipelines', 0)}")
            click.echo()
        
        # Quality alerts
        quality = dashboard_data.get('quality_predictions', {})
        click.echo(f"=� Quality Alerts:")
        click.echo(f"   Total active: {quality.get('total_active_alerts', 0)}")
        click.echo(f"   Critical: {quality.get('critical_alerts', 0)}")
        click.echo(f"   High priority: {quality.get('high_priority_alerts', 0)}")
        click.echo()
        
        click.echo(f"Generated at: {dashboard_data.get('dashboard_generated_at')}")
        
    except Exception as e:
        click.echo(f"L Dashboard failed: {e}", err=True)


if __name__ == '__main__':
    cli()