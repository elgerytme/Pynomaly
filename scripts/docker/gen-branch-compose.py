#!/usr/bin/env python3
"""
Generate branch-specific docker-compose files for parallel development.

This script takes existing docker-compose files and generates branch-specific
versions by parameterizing service names, network names, and volume names
with the GIT_BRANCH environment variable to avoid conflicts when running
multiple branches simultaneously.
"""

import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def sanitize_branch_name(branch_name: str) -> str:
    """
    Sanitize branch name to be used in Docker resource names.
    
    Docker names must be valid according to:
    - Lowercase letters, digits, and separators (dashes, underscores, periods)
    - Must start with alphanumeric character
    - Cannot end with separator
    """
    # Convert to lowercase
    sanitized = branch_name.lower()
    
    # Replace invalid characters with dashes
    sanitized = re.sub(r'[^a-z0-9\-_.]', '-', sanitized)
    
    # Remove leading/trailing separators
    sanitized = sanitized.strip('-_.')
    
    # Replace multiple consecutive separators with single dash
    sanitized = re.sub(r'[-_.]+', '-', sanitized)
    
    # Ensure it starts with alphanumeric
    if sanitized and not sanitized[0].isalnum():
        sanitized = 'br-' + sanitized
    
    # Truncate if too long (Docker has limits)
    if len(sanitized) > 50:
        sanitized = sanitized[:50].rstrip('-_.')
    
    return sanitized or 'default'


def parameterize_service_names(services: Dict[str, Any], branch_suffix: str) -> Dict[str, Any]:
    """Add branch suffix to service names and update cross-references."""
    updated_services = {}
    service_mapping = {}
    
    # First pass: create mapping of old names to new names
    for service_name in services.keys():
        new_name = f"{service_name}-{branch_suffix}"
        service_mapping[service_name] = new_name
    
    # Second pass: update services with new names and fix references
    for old_name, service_config in services.items():
        new_name = service_mapping[old_name]
        
        # Update container_name if present
        if 'container_name' in service_config:
            service_config['container_name'] = f"{service_config['container_name']}-{branch_suffix}"
        
        # Update depends_on references
        if 'depends_on' in service_config:
            if isinstance(service_config['depends_on'], list):
                service_config['depends_on'] = [
                    service_mapping.get(dep, dep) for dep in service_config['depends_on']
                ]
            elif isinstance(service_config['depends_on'], dict):
                new_depends_on = {}
                for dep_name, dep_config in service_config['depends_on'].items():
                    new_dep_name = service_mapping.get(dep_name, dep_name)
                    new_depends_on[new_dep_name] = dep_config
                service_config['depends_on'] = new_depends_on
        
        # Update environment variables that reference other services
        if 'environment' in service_config:
            env_vars = service_config['environment']
            if isinstance(env_vars, list):
                # Handle list format: ["VAR=value"]
                for i, env_var in enumerate(env_vars):
                    if '=' in env_var:
                        var_name, var_value = env_var.split('=', 1)
                        for old_svc, new_svc in service_mapping.items():
                            var_value = var_value.replace(f'@{old_svc}', f'@{new_svc}')
                            var_value = var_value.replace(f'://{old_svc}:', f'://{new_svc}:')
                        env_vars[i] = f"{var_name}={var_value}"
            elif isinstance(env_vars, dict):
                # Handle dict format: {"VAR": "value"}
                for var_name, var_value in env_vars.items():
                    if isinstance(var_value, str):
                        for old_svc, new_svc in service_mapping.items():
                            var_value = var_value.replace(f'@{old_svc}', f'@{new_svc}')
                            var_value = var_value.replace(f'://{old_svc}:', f'://{new_svc}:')
                        env_vars[var_name] = var_value
        
        updated_services[new_name] = service_config
    
    return updated_services


def parameterize_networks(networks: Dict[str, Any], branch_suffix: str) -> Dict[str, Any]:
    """Add branch suffix to network names."""
    if not networks:
        return networks
    
    updated_networks = {}
    for network_name, network_config in networks.items():
        new_name = f"{network_name}-{branch_suffix}"
        
        # Update the network config
        if isinstance(network_config, dict):
            updated_config = network_config.copy()
            # Update the actual network name if specified
            if 'name' in updated_config:
                updated_config['name'] = f"{updated_config['name']}-{branch_suffix}"
            else:
                updated_config['name'] = new_name
        else:
            updated_config = network_config
        
        updated_networks[new_name] = updated_config
    
    return updated_networks


def parameterize_volumes(volumes: Dict[str, Any], branch_suffix: str) -> Dict[str, Any]:
    """Add branch suffix to volume names."""
    if not volumes:
        return volumes
    
    updated_volumes = {}
    for volume_name, volume_config in volumes.items():
        new_name = f"{volume_name}-{branch_suffix}"
        
        # Update the volume config
        if isinstance(volume_config, dict):
            updated_config = volume_config.copy()
            # Update the actual volume name if specified
            if 'name' in updated_config:
                updated_config['name'] = f"{updated_config['name']}-{branch_suffix}"
            else:
                updated_config['name'] = new_name
        else:
            updated_config = volume_config
        
        updated_volumes[new_name] = updated_config
    
    return updated_volumes


def update_service_references(services: Dict[str, Any], 
                            network_mapping: Dict[str, str],
                            volume_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Update network and volume references in services."""
    for service_name, service_config in services.items():
        
        # Update network references
        if 'networks' in service_config:
            networks = service_config['networks']
            if isinstance(networks, list):
                service_config['networks'] = [
                    network_mapping.get(net, net) for net in networks
                ]
            elif isinstance(networks, dict):
                new_networks = {}
                for net_name, net_config in networks.items():
                    new_net_name = network_mapping.get(net_name, net_name)
                    new_networks[new_net_name] = net_config
                service_config['networks'] = new_networks
        
        # Update volume references
        if 'volumes' in service_config:
            volumes = service_config['volumes']
            updated_volumes = []
            for volume_spec in volumes:
                if isinstance(volume_spec, str):
                    # Handle "volume:path" or "volume:path:mode" format
                    parts = volume_spec.split(':')
                    if len(parts) >= 2:
                        volume_name = parts[0]
                        # Only update if it's a named volume (not a bind mount)
                        if not volume_name.startswith('.') and not volume_name.startswith('/'):
                            new_volume_name = volume_mapping.get(volume_name, volume_name)
                            parts[0] = new_volume_name
                            volume_spec = ':'.join(parts)
                updated_volumes.append(volume_spec)
            service_config['volumes'] = updated_volumes
    
    return services


def generate_branch_compose(source_file: Path, output_file: Path, branch_name: str) -> None:
    """Generate branch-specific docker-compose file."""
    print(f"Processing {source_file} -> {output_file}")
    
    # Sanitize branch name
    branch_suffix = sanitize_branch_name(branch_name)
    print(f"Using branch suffix: {branch_suffix}")
    
    # Load source compose file
    with open(source_file, 'r') as f:
        compose_data = yaml.safe_load(f)
    
    # Create mappings for reference updates
    original_networks = compose_data.get('networks', {})
    original_volumes = compose_data.get('volumes', {})
    
    network_mapping = {name: f"{name}-{branch_suffix}" for name in original_networks.keys()}
    volume_mapping = {name: f"{name}-{branch_suffix}" for name in original_volumes.keys()}
    
    # Parameterize each section
    if 'services' in compose_data:
        compose_data['services'] = parameterize_service_names(
            compose_data['services'], branch_suffix
        )
        compose_data['services'] = update_service_references(
            compose_data['services'], network_mapping, volume_mapping
        )
    
    if 'networks' in compose_data:
        compose_data['networks'] = parameterize_networks(
            compose_data['networks'], branch_suffix
        )
    
    if 'volumes' in compose_data:
        compose_data['volumes'] = parameterize_volumes(
            compose_data['volumes'], branch_suffix
        )
    
    # Add branch environment variable to all services
    if 'services' in compose_data:
        for service_name, service_config in compose_data['services'].items():
            if 'environment' not in service_config:
                service_config['environment'] = []
            
            env_vars = service_config['environment']
            if isinstance(env_vars, list):
                env_vars.append(f"GIT_BRANCH={branch_name}")
            elif isinstance(env_vars, dict):
                env_vars['GIT_BRANCH'] = branch_name
    
    # Write output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"# Generated branch-specific docker-compose file for branch: {branch_name}\n")
        f.write(f"# Generated from: {source_file}\n")
        f.write(f"# Branch suffix: {branch_suffix}\n\n")
        yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated {output_file}")


def main():
    """Main function to generate branch-specific compose files."""
    # Get branch name from environment or git
    branch_name = os.environ.get('GIT_BRANCH')
    if not branch_name:
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True,
                text=True,
                check=True
            )
            branch_name = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Could not determine git branch. Set GIT_BRANCH environment variable.")
            sys.exit(1)
    
    if not branch_name:
        print("Error: No branch name found.")
        sys.exit(1)
    
    print(f"Generating branch-specific compose files for branch: {branch_name}")
    
    # Find docker-compose files to process
    project_root = Path(__file__).parent.parent.parent
    docker_dir = project_root / 'deploy' / 'docker'
    
    compose_files = [
        ('docker-compose.yml', 'docker-compose.branch.yml'),
        ('docker-compose.production.yml', 'docker-compose.production.branch.yml'),
        ('docker-compose.testing.yml', 'docker-compose.testing.branch.yml'),
        ('docker-compose.hardened.yml', 'docker-compose.hardened.branch.yml'),
        ('docker-compose.ui-testing.yml', 'docker-compose.ui-testing.branch.yml'),
    ]
    
    # Also check root directory
    root_compose_files = [
        ('docker-compose.test.yml', 'docker-compose.test.branch.yml'),
    ]
    
    generated_files = []
    
    # Process files in deploy/docker directory
    for source_name, output_name in compose_files:
        source_file = docker_dir / source_name
        output_file = docker_dir / output_name
        
        if source_file.exists():
            generate_branch_compose(source_file, output_file, branch_name)
            generated_files.append(output_file)
        else:
            print(f"Warning: {source_file} not found, skipping.")
    
    # Process files in root directory
    for source_name, output_name in root_compose_files:
        source_file = project_root / source_name
        output_file = project_root / output_name
        
        if source_file.exists():
            generate_branch_compose(source_file, output_file, branch_name)
            generated_files.append(output_file)
        else:
            print(f"Warning: {source_file} not found, skipping.")
    
    print(f"\nGenerated {len(generated_files)} branch-specific compose files:")
    for file_path in generated_files:
        print(f"  - {file_path}")
    
    print(f"\nTo use these files, run:")
    print(f"  docker-compose -f docker-compose.branch.yml up")
    print(f"  # or")
    print(f"  docker-compose -f deploy/docker/docker-compose.production.branch.yml up")


if __name__ == '__main__':
    main()
