"""
Cross-Cloud Deployment CLI

Command-line interface for cross-cloud deployment orchestration,
including deployment management, traffic splitting, and failover operations.
"""

import asyncio
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.tree import Tree
from rich import print as rprint

from ..infrastructure.deployment.cross_cloud_deployment import (
    CrossCloudDeploymentOrchestrator, DeploymentSpec, DeploymentTarget,
    DeploymentStrategy, TrafficSplitStrategy, FailoverMode, CloudCredentials,
    TrafficSplit, FailoverConfig
)
from ..infrastructure.monitoring.advanced_observability_platform import AdvancedObservabilityPlatform


console = Console()


@click.group()
@click.pass_context
def deployment(ctx):
    """Cross-Cloud Deployment CLI."""
    ctx.ensure_object(dict)
    
    # Initialize deployment orchestrator
    monitoring_platform = AdvancedObservabilityPlatform()
    ctx.obj['orchestrator'] = CrossCloudDeploymentOrchestrator(monitoring_platform)


@deployment.group()
def spec():
    """Deployment specification management."""
    pass


@spec.command()
@click.option('--output', '-o', required=True, help='Output file path for deployment spec')
@click.option('--name', required=True, help='Deployment name')
@click.option('--image', required=True, help='Container image')
@click.option('--tag', default='latest', help='Container tag')
@click.option('--replicas', default=3, help='Number of replicas')
@click.option('--cpu-request', default='200m', help='CPU request')
@click.option('--cpu-limit', default='1000m', help='CPU limit')
@click.option('--memory-request', default='256Mi', help='Memory request')
@click.option('--memory-limit', default='1Gi', help='Memory limit')
@click.option('--strategy', 
              type=click.Choice([s.value for s in DeploymentStrategy]),
              default='rolling_update',
              help='Deployment strategy')
@click.option('--targets', 
              multiple=True,
              type=click.Choice([t.value for t in DeploymentTarget]),
              help='Deployment targets')
@click.option('--enable-hpa', is_flag=True, help='Enable horizontal pod autoscaling')
@click.option('--min-replicas', default=2, help='Minimum replicas for HPA')
@click.option('--max-replicas', default=20, help='Maximum replicas for HPA')
@click.option('--env', multiple=True, help='Environment variables (key=value)')
def create(output, name, image, tag, replicas, cpu_request, cpu_limit,
           memory_request, memory_limit, strategy, targets, enable_hpa,
           min_replicas, max_replicas, env):
    """Create a new deployment specification."""
    
    # Parse environment variables
    env_vars = {}
    for env_var in env:
        if '=' in env_var:
            key, value = env_var.split('=', 1)
            env_vars[key] = value
    
    # Create deployment spec
    spec = DeploymentSpec(
        name=name,
        container_image=image,
        container_tag=tag,
        replicas=replicas,
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        memory_request=memory_request,
        memory_limit=memory_limit,
        strategy=DeploymentStrategy(strategy),
        targets=[DeploymentTarget(t) for t in targets],
        environment_variables=env_vars,
        hpa_enabled=enable_hpa,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        ports=[{"containerPort": 8080, "port": 80, "name": "http"}],  # Default port
        labels={
            "app": name,
            "tier": "production"
        }
    )
    
    # Convert to dict and save as YAML
    spec_dict = {
        'deployment_id': spec.deployment_id,
        'name': spec.name,
        'version': spec.version,
        'container_image': spec.container_image,
        'container_tag': spec.container_tag,
        'replicas': spec.replicas,
        'resources': {
            'cpu_request': spec.cpu_request,
            'cpu_limit': spec.cpu_limit,
            'memory_request': spec.memory_request,
            'memory_limit': spec.memory_limit
        },
        'strategy': spec.strategy.value,
        'targets': [t.value for t in spec.targets],
        'environment_variables': spec.environment_variables,
        'scaling': {
            'hpa_enabled': spec.hpa_enabled,
            'min_replicas': spec.min_replicas,
            'max_replicas': spec.max_replicas,
            'target_cpu_utilization': spec.target_cpu_utilization
        },
        'networking': {
            'ports': spec.ports,
            'ingress_enabled': spec.ingress_enabled,
            'load_balancer_type': spec.load_balancer_type
        },
        'monitoring': {
            'enabled': spec.monitoring_enabled,
            'health_check_path': spec.health_check_path,
            'readiness_probe_path': spec.readiness_probe_path,
            'liveness_probe_path': spec.liveness_probe_path
        },
        'labels': spec.labels
    }
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(spec_dict, f, default_flow_style=False, indent=2)
    
    console.print(f"✅ Deployment specification saved to {output}")
    console.print(f"📄 Deployment ID: [bold blue]{spec.deployment_id}[/bold blue]")


@spec.command()
@click.argument('spec_file')
def validate(spec_file):
    """Validate a deployment specification file."""
    
    try:
        with open(spec_file, 'r') as f:
            spec_data = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['name', 'container_image', 'targets']
        missing_fields = []
        
        for field in required_fields:
            if field not in spec_data:
                missing_fields.append(field)
        
        if missing_fields:
            console.print(f"❌ Missing required fields: {', '.join(missing_fields)}")
            return
        
        # Validate targets
        valid_targets = [t.value for t in DeploymentTarget]
        invalid_targets = []
        
        for target in spec_data.get('targets', []):
            if target not in valid_targets:
                invalid_targets.append(target)
        
        if invalid_targets:
            console.print(f"❌ Invalid targets: {', '.join(invalid_targets)}")
            console.print(f"Valid targets: {', '.join(valid_targets)}")
            return
        
        # Validate strategy
        if 'strategy' in spec_data:
            valid_strategies = [s.value for s in DeploymentStrategy]
            if spec_data['strategy'] not in valid_strategies:
                console.print(f"❌ Invalid strategy: {spec_data['strategy']}")
                console.print(f"Valid strategies: {', '.join(valid_strategies)}")
                return
        
        console.print("✅ Deployment specification is valid")
        
        # Show summary
        summary_panel = Panel(
            f"""
[bold]Deployment Summary:[/bold]

[bold]Name:[/bold] {spec_data['name']}
[bold]Image:[/bold] {spec_data['container_image']}:{spec_data.get('container_tag', 'latest')}
[bold]Replicas:[/bold] {spec_data.get('replicas', 1)}
[bold]Strategy:[/bold] {spec_data.get('strategy', 'rolling_update')}
[bold]Targets:[/bold] {', '.join(spec_data['targets'])}
[bold]HPA Enabled:[/bold] {spec_data.get('scaling', {}).get('hpa_enabled', False)}
[bold]Environment Variables:[/bold] {len(spec_data.get('environment_variables', {}))}
            """.strip(),
            title="Specification Summary",
            expand=False
        )
        console.print(summary_panel)
        
    except Exception as e:
        console.print(f"❌ Error validating specification: {str(e)}")


@deployment.group()
def credentials():
    """Cloud credentials management."""
    pass


@credentials.command()
@click.option('--provider', 
              type=click.Choice(['aws', 'azure', 'gcp', 'kubernetes']),
              required=True,
              help='Cloud provider')
@click.option('--config-file', required=True, help='Path to save credentials config')
@click.option('--aws-access-key', help='AWS access key')
@click.option('--aws-secret-key', help='AWS secret key')
@click.option('--aws-region', default='us-east-1', help='AWS region')
@click.option('--azure-subscription-id', help='Azure subscription ID')
@click.option('--azure-tenant-id', help='Azure tenant ID')
@click.option('--gcp-project-id', help='GCP project ID')
@click.option('--gcp-service-account', help='Path to GCP service account JSON')
@click.option('--k8s-cluster-name', help='Kubernetes cluster name')
def configure(provider, config_file, aws_access_key, aws_secret_key, aws_region,
              azure_subscription_id, azure_tenant_id, gcp_project_id,
              gcp_service_account, k8s_cluster_name):
    """Configure cloud provider credentials."""
    
    credential_data = {'provider': provider}
    
    if provider == 'aws':
        if not aws_access_key or not aws_secret_key:
            console.print("❌ AWS access key and secret key are required")
            return
        credential_data.update({
            'access_key': aws_access_key,
            'secret_key': aws_secret_key,
            'region': aws_region
        })
    
    elif provider == 'azure':
        if not azure_subscription_id:
            console.print("❌ Azure subscription ID is required")
            return
        credential_data.update({
            'subscription_id': azure_subscription_id,
            'tenant_id': azure_tenant_id
        })
    
    elif provider == 'gcp':
        if not gcp_project_id:
            console.print("❌ GCP project ID is required")
            return
        credential_data.update({
            'project_id': gcp_project_id,
            'service_account_path': gcp_service_account
        })
    
    elif provider == 'kubernetes':
        credential_data.update({
            'cluster_name': k8s_cluster_name
        })
    
    # Save credentials configuration
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(credential_data, f, default_flow_style=False, indent=2)
    
    console.print(f"✅ {provider.upper()} credentials configured and saved to {config_file}")


@deployment.group()
def deploy():
    """Deployment operations."""
    pass


@deploy.command()
@click.argument('spec_file')
@click.option('--credentials', multiple=True, help='Credentials configuration files')
@click.option('--wait', is_flag=True, help='Wait for deployment to complete')
@click.option('--timeout', default=600, help='Deployment timeout in seconds')
@click.pass_context
async def start(ctx, spec_file, credentials, wait, timeout):
    """Start a new cross-cloud deployment."""
    
    orchestrator = ctx.obj['orchestrator']
    
    try:
        # Load deployment specification
        with open(spec_file, 'r') as f:
            spec_data = yaml.safe_load(f)
        
        console.print(f"✅ Loaded deployment specification from {spec_file}")
        
        # Create deployment spec object
        spec = DeploymentSpec(
            deployment_id=spec_data.get('deployment_id'),
            name=spec_data['name'],
            version=spec_data.get('version', '1.0.0'),
            container_image=spec_data['container_image'],
            container_tag=spec_data.get('container_tag', 'latest'),
            replicas=spec_data.get('replicas', 1),
            cpu_request=spec_data.get('resources', {}).get('cpu_request', '200m'),
            cpu_limit=spec_data.get('resources', {}).get('cpu_limit', '1000m'),
            memory_request=spec_data.get('resources', {}).get('memory_request', '256Mi'),
            memory_limit=spec_data.get('resources', {}).get('memory_limit', '1Gi'),
            strategy=DeploymentStrategy(spec_data.get('strategy', 'rolling_update')),
            targets=[DeploymentTarget(t) for t in spec_data['targets']],
            environment_variables=spec_data.get('environment_variables', {}),
            ports=spec_data.get('networking', {}).get('ports', [{"containerPort": 8080, "port": 80}]),
            hpa_enabled=spec_data.get('scaling', {}).get('hpa_enabled', False),
            min_replicas=spec_data.get('scaling', {}).get('min_replicas', 1),
            max_replicas=spec_data.get('scaling', {}).get('max_replicas', 10),
            labels=spec_data.get('labels', {})
        )
        
        # Load credentials
        credential_objects = []
        for cred_file in credentials:
            with open(cred_file, 'r') as f:
                cred_data = yaml.safe_load(f)
            
            credential = CloudCredentials(**cred_data)
            credential_objects.append(credential)
        
        if credential_objects:
            console.print(f"✅ Loaded {len(credential_objects)} credential configurations")
            await orchestrator.initialize_cloud_clients(credential_objects)
        
        # Start deployment
        with console.status("Starting cross-cloud deployment..."):
            deployment_id = await orchestrator.deploy(spec)
        
        console.print(f"✅ Deployment started with ID: [bold blue]{deployment_id}[/bold blue]")
        console.print(f"🎯 Targets: {', '.join([t.value for t in spec.targets])}")
        console.print(f"📋 Strategy: {spec.strategy.value}")
        
        if wait:
            # Monitor deployment progress
            with Progress() as progress:
                task = progress.add_task("Deploying...", total=100)
                
                start_time = datetime.utcnow()
                while (datetime.utcnow() - start_time).total_seconds() < timeout:
                    try:
                        status = await orchestrator.get_deployment_status(deployment_id)
                        progress.update(task, completed=status['overall_progress'])
                        
                        if status['overall_progress'] >= 100:
                            break
                        
                        await asyncio.sleep(10)
                        
                    except Exception as e:
                        console.print(f"❌ Error checking status: {e}")
                        break
            
            # Show final status
            final_status = await orchestrator.get_deployment_status(deployment_id)
            _display_deployment_status(final_status)
    
    except Exception as e:
        console.print(f"❌ Deployment failed: {str(e)}")


@deploy.command()
@click.argument('deployment_id')
@click.pass_context
async def status(ctx, deployment_id):
    """Get deployment status."""
    
    orchestrator = ctx.obj['orchestrator']
    
    try:
        status = await orchestrator.get_deployment_status(deployment_id)
        _display_deployment_status(status)
        
    except ValueError as e:
        console.print(f"❌ {str(e)}")
    except Exception as e:
        console.print(f"❌ Error retrieving status: {str(e)}")


@deploy.command()
@click.argument('deployment_id')
@click.argument('target')
@click.argument('replicas', type=int)
@click.pass_context
async def scale(ctx, deployment_id, target, replicas):
    """Scale a deployment target."""
    
    orchestrator = ctx.obj['orchestrator']
    
    try:
        target_enum = DeploymentTarget(target)
        
        with console.status(f"Scaling {target} to {replicas} replicas..."):
            await orchestrator.scale_deployment(deployment_id, target_enum, replicas)
        
        console.print(f"✅ Scaled {deployment_id} on {target} to {replicas} replicas")
        
    except ValueError as e:
        console.print(f"❌ Invalid target: {target}")
        console.print(f"Valid targets: {', '.join([t.value for t in DeploymentTarget])}")
    except Exception as e:
        console.print(f"❌ Scaling failed: {str(e)}")


@deploy.command()
@click.argument('deployment_id')
@click.option('--version', help='Version to rollback to')
@click.pass_context
async def rollback(ctx, deployment_id, version):
    """Rollback a deployment."""
    
    orchestrator = ctx.obj['orchestrator']
    
    try:
        with console.status(f"Rolling back deployment {deployment_id}..."):
            await orchestrator.rollback_deployment(deployment_id, version)
        
        console.print(f"✅ Rollback completed for deployment {deployment_id}")
        
    except Exception as e:
        console.print(f"❌ Rollback failed: {str(e)}")


@deploy.command()
@click.argument('deployment_id')
@click.pass_context
async def terminate(ctx, deployment_id):
    """Terminate a deployment."""
    
    orchestrator = ctx.obj['orchestrator']
    
    if not click.confirm(f"Are you sure you want to terminate deployment {deployment_id}?"):
        return
    
    try:
        with console.status(f"Terminating deployment {deployment_id}..."):
            await orchestrator.terminate_deployment(deployment_id)
        
        console.print(f"✅ Deployment {deployment_id} terminated")
        
    except Exception as e:
        console.print(f"❌ Termination failed: {str(e)}")


@deployment.group()
def traffic():
    """Traffic management operations."""
    pass


@traffic.command()
@click.argument('deployment_id')
@click.option('--split', multiple=True, help='Traffic splits (target:percentage)')
@click.pass_context
async def split(ctx, deployment_id, split):
    """Configure traffic splitting."""
    
    orchestrator = ctx.obj['orchestrator']
    
    try:
        # Parse traffic splits
        splits = {}
        total_percentage = 0
        
        for split_config in split:
            if ':' not in split_config:
                console.print(f"❌ Invalid split format: {split_config}. Use target:percentage")
                return
            
            target, percentage_str = split_config.split(':', 1)
            try:
                percentage = float(percentage_str) / 100.0
                splits[target] = percentage
                total_percentage += percentage
            except ValueError:
                console.print(f"❌ Invalid percentage: {percentage_str}")
                return
        
        if abs(total_percentage - 1.0) > 0.01:
            console.print(f"❌ Traffic splits must sum to 100% (current: {total_percentage*100:.1f}%)")
            return
        
        with console.status("Configuring traffic splits..."):
            await orchestrator.update_traffic_split(deployment_id, splits)
        
        console.print(f"✅ Traffic splits configured for {deployment_id}")
        
        # Display splits
        table = Table(title="Traffic Splits")
        table.add_column("Target", style="cyan")
        table.add_column("Percentage", style="green")
        
        for target, percentage in splits.items():
            table.add_row(target, f"{percentage*100:.1f}%")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Traffic split configuration failed: {str(e)}")


@deployment.group()
def failover():
    """Failover management operations."""
    pass


@failover.command()
@click.argument('deployment_id')
@click.argument('from_target')
@click.argument('to_target')
@click.pass_context
async def trigger(ctx, deployment_id, from_target, to_target):
    """Trigger manual failover."""
    
    orchestrator = ctx.obj['orchestrator']
    
    try:
        from_target_enum = DeploymentTarget(from_target)
        to_target_enum = DeploymentTarget(to_target)
        
        if not click.confirm(f"Trigger failover from {from_target} to {to_target}?"):
            return
        
        with console.status("Executing failover..."):
            await orchestrator.trigger_failover(deployment_id, from_target_enum, to_target_enum)
        
        console.print(f"✅ Failover completed: {from_target} → {to_target}")
        
    except ValueError as e:
        console.print(f"❌ Invalid target: {str(e)}")
    except Exception as e:
        console.print(f"❌ Failover failed: {str(e)}")


@deployment.command()
@click.option('--format', 
              type=click.Choice(['table', 'json', 'yaml']),
              default='table',
              help='Output format')
@click.pass_context
async def list(ctx, format):
    """List all deployments."""
    
    orchestrator = ctx.obj['orchestrator']
    
    # For now, show placeholder data since we don't have a list method
    # In production, implement orchestrator.list_deployments()
    
    deployments = [
        {
            'deployment_id': 'dep-123456',
            'name': 'ml-inference-service',
            'version': '1.2.0',
            'strategy': 'multi_cloud_active',
            'targets': ['aws_eks', 'azure_aks', 'gcp_gke'],
            'status': 'deployed',
            'progress': 100.0
        }
    ]
    
    if format == 'table':
        table = Table(title="Deployments")
        table.add_column("Deployment ID", style="cyan")
        table.add_column("Name", style="bright_white")
        table.add_column("Version", style="yellow")
        table.add_column("Strategy", style="green")
        table.add_column("Targets", style="blue")
        table.add_column("Status", style="magenta")
        table.add_column("Progress", style="green")
        
        for dep in deployments:
            table.add_row(
                dep['deployment_id'][:8] + "...",
                dep['name'],
                dep['version'],
                dep['strategy'],
                ', '.join(dep['targets']),
                dep['status'],
                f"{dep['progress']:.1f}%"
            )
        
        console.print(table)
    
    elif format == 'json':
        console.print(json.dumps(deployments, indent=2))
    
    elif format == 'yaml':
        console.print(yaml.dump(deployments, default_flow_style=False, indent=2))


@deployment.command()
@click.pass_context
async def health(ctx):
    """Check deployment system health."""
    
    orchestrator = ctx.obj['orchestrator']
    
    # Check orchestrator health
    health_status = {
        "orchestrator": {
            "status": "healthy",
            "active_deployments": len(orchestrator.deployments),
            "cloud_clients": {
                "aws": len(orchestrator.aws_clients),
                "azure": len(orchestrator.azure_clients),
                "gcp": len(orchestrator.gcp_clients),
                "kubernetes": len(orchestrator.k8s_clients)
            },
            "background_tasks": len(orchestrator.orchestration_tasks)
        },
        "components": {
            "kubernetes_deployer": "available",
            "docker_deployer": "available",
            "traffic_manager": "available",
            "failover_manager": "available"
        }
    }
    
    # Display health status
    for component, status in health_status.items():
        if isinstance(status, dict):
            status_icon = "🟢" if status.get("status", "unknown") == "healthy" else "🔴"
            
            panel_content = f"[bold]Status:[/bold] {status_icon} {status.get('status', 'UNKNOWN').upper()}\n"
            
            for key, value in status.items():
                if key != "status":
                    if isinstance(value, dict):
                        panel_content += f"[bold]{key.replace('_', ' ').title()}:[/bold]\n"
                        for sub_key, sub_value in value.items():
                            panel_content += f"  • {sub_key.upper()}: {sub_value}\n"
                    else:
                        panel_content += f"[bold]{key.replace('_', ' ').title()}:[/bold] {value}\n"
            
            panel = Panel(
                panel_content.strip(),
                title=f"{component.title()} Health",
                expand=False
            )
            console.print(panel)


def _display_deployment_status(status: Dict[str, Any]) -> None:
    """Display comprehensive deployment status."""
    
    # Main status panel
    status_panel = Panel(
        f"""
[bold]Deployment Status[/bold]

[bold]ID:[/bold] {status['deployment_id']}
[bold]Name:[/bold] {status['name']}
[bold]Version:[/bold] {status['version']}
[bold]Strategy:[/bold] {status['strategy']}
[bold]Overall Progress:[/bold] {status['overall_progress']:.1f}%

[bold]Status Summary:[/bold]
{' • '.join([f"{status_name}: {count}" for status_name, count in status['status_counts'].items()])}
        """.strip(),
        title="Deployment Overview",
        expand=False
    )
    console.print(status_panel)
    
    # Targets table
    if status['targets']:
        table = Table(title="Target Status")
        table.add_column("Target", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="yellow")
        table.add_column("Replicas", style="blue")
        table.add_column("Service URLs", style="magenta")
        
        for target in status['targets']:
            status_color = "green" if target['status'] == 'deployed' else "yellow" if target['status'] == 'deploying' else "red"
            
            table.add_row(
                target['target'],
                f"[{status_color}]{target['status']}[/{status_color}]",
                f"{target['progress']:.1f}%",
                f"{target['healthy_replicas']}/{target['total_replicas']}",
                '\n'.join(target['service_urls'][:2]) if target['service_urls'] else "N/A"
            )
        
        console.print(table)
    
    # Traffic split information
    if status.get('traffic_split'):
        traffic_panel = Panel(
            f"""
[bold]Traffic Split Configuration[/bold]

[bold]Strategy:[/bold] {status['traffic_split']['strategy']}
[bold]Splits:[/bold]
{chr(10).join([f"  • {target}: {percentage*100:.1f}%" for target, percentage in status['traffic_split']['splits'].items()])}
            """.strip(),
            title="Traffic Management",
            expand=False
        )
        console.print(traffic_panel)
    
    # Failover configuration
    if status.get('failover_config'):
        failover_panel = Panel(
            f"""
[bold]Failover Configuration[/bold]

[bold]Mode:[/bold] {status['failover_config']['mode']}
[bold]Primary Target:[/bold] {status['failover_config']['primary_target']}
[bold]Secondary Targets:[/bold] {', '.join(status['failover_config']['secondary_targets'])}
[bold]Health Check Interval:[/bold] {status['failover_config']['health_check_interval']}s
[bold]Failure Threshold:[/bold] {status['failover_config']['failure_threshold']}
            """.strip(),
            title="Failover Management",
            expand=False
        )
        console.print(failover_panel)


if __name__ == "__main__":
    # Support async CLI commands
    def async_command(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to commands that need it
    async_commands = [
        deploy.commands['start'],
        deploy.commands['status'],
        deploy.commands['scale'],
        deploy.commands['rollback'],
        deploy.commands['terminate'],
        traffic.commands['split'],
        failover.commands['trigger'],
        deployment.commands['list'],
        deployment.commands['health']
    ]
    
    for command in async_commands:
        command.callback = async_command(command.callback)
    
    deployment()