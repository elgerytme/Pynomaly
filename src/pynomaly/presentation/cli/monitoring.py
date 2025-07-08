"""CLI commands for monitoring configuration."""

import click
import os
from typing import Optional

from pynomaly.infrastructure.config.settings import get_settings
from pynomaly.infrastructure.monitoring.service_initialization import (
    create_monitoring_service,
    initialize_monitoring_service,
)


@click.group()
def monitoring():
    """Monitoring configuration commands."""
    pass


@monitoring.command()
@click.option(
    "--provider",
    type=click.Choice(["grafana", "datadog", "newrelic", "prometheus", "webhook"]),
    required=True,
    help="Monitoring provider to configure",
)
@click.option("--enable/--disable", default=True, help="Enable or disable the provider")
@click.option("--endpoint", help="Provider endpoint URL")
@click.option("--api-key", help="API key for the provider")
def configure_provider(provider: str, enable: bool, endpoint: Optional[str], api_key: Optional[str]):
    """Configure a monitoring provider."""
    provider_map = {
        "grafana": "PYNOMALY_GRAFANA",
        "datadog": "PYNOMALY_DATADOG",
        "newrelic": "PYNOMALY_NEWRELIC",
        "prometheus": "PYNOMALY_PROMETHEUS_PUSHGATEWAY",
        "webhook": "PYNOMALY_WEBHOOK",
    }
    
    prefix = provider_map[provider]
    
    # Set environment variables
    os.environ[f"{prefix}_ENABLED"] = "true" if enable else "false"
    
    if endpoint:
        os.environ[f"{prefix}_ENDPOINT"] = endpoint
    
    if api_key:
        os.environ[f"{prefix}_API_KEY"] = api_key
    
    status = "enabled" if enable else "disabled"
    click.echo(f"Provider {provider} has been {status}")
    
    if endpoint:
        click.echo(f"Endpoint: {endpoint}")
    
    if api_key:
        click.echo(f"API key: {'*' * len(api_key)}")


@monitoring.command()
@click.option(
    "--buffer-size",
    type=int,
    default=100,
    help="Buffer size for monitoring data",
)
@click.option(
    "--flush-interval",
    type=int,
    default=60,
    help="Flush interval in seconds",
)
def configure_buffer(buffer_size: int, flush_interval: int):
    """Configure monitoring buffer settings."""
    if buffer_size <= 0:
        raise click.BadParameter("Buffer size must be positive")
    
    if flush_interval <= 0:
        raise click.BadParameter("Flush interval must be positive")
    
    os.environ["PYNOMALY_MONITORING_BUFFER_SIZE"] = str(buffer_size)
    os.environ["PYNOMALY_MONITORING_FLUSH_INTERVAL"] = str(flush_interval)
    
    click.echo(f"Buffer size set to: {buffer_size}")
    click.echo(f"Flush interval set to: {flush_interval} seconds")


@monitoring.command()
def status():
    """Show monitoring configuration status."""
    settings = get_settings()
    monitoring_config = settings.get_monitoring_config()
    
    click.echo("Monitoring Configuration Status:")
    click.echo("=" * 40)
    
    click.echo(f"Buffer size: {monitoring_config['buffer_size']}")
    click.echo(f"Flush interval: {monitoring_config['flush_interval']} seconds")
    
    click.echo("\\nConfigured Providers:")
    providers = monitoring_config["providers"]
    
    if not providers:
        click.echo("No providers configured")
        return
    
    for provider in providers:
        click.echo(f"  - {provider['provider']}")
        click.echo(f"    Endpoint: {provider['endpoint']}")
        click.echo(f"    Enabled: {provider['enabled']}")
        click.echo(f"    Has API Key: {'Yes' if provider['api_key'] else 'No'}")
        click.echo()


@monitoring.command()
@click.option(
    "--test-connections",
    is_flag=True,
    help="Test connections to all configured providers",
)
def test(test_connections: bool):
    """Test monitoring service configuration."""
    import asyncio
    
    async def run_test():
        settings = get_settings()
        monitoring_config = settings.get_monitoring_config()
        
        click.echo("Testing monitoring service configuration...")
        
        # Test service creation
        try:
            service = create_monitoring_service(settings)
            click.echo(f"✓ Service created successfully with {len(service.providers)} providers")
        except Exception as e:
            click.echo(f"✗ Failed to create service: {e}")
            return
        
        # Test service initialization
        try:
            await service.initialize()
            click.echo("✓ Service initialized successfully")
        except Exception as e:
            click.echo(f"✗ Failed to initialize service: {e}")
            return
        
        # Test provider connections if requested
        if test_connections:
            click.echo("\\nTesting provider connections...")
            try:
                results = await service.test_all_providers()
                for provider_name, result in results.items():
                    status = "✓" if result else "✗"
                    click.echo(f"{status} {provider_name}: {'Connected' if result else 'Failed'}")
            except Exception as e:
                click.echo(f"✗ Failed to test connections: {e}")
        
        # Shutdown service
        try:
            await service.shutdown()
            click.echo("✓ Service shutdown successfully")
        except Exception as e:
            click.echo(f"✗ Failed to shutdown service: {e}")
    
    asyncio.run(run_test())


if __name__ == "__main__":
    monitoring()
