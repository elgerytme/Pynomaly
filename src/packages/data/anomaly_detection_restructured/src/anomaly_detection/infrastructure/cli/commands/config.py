"""Configuration Management CLI Commands for system configuration and settings management."""

from __future__ import annotations

import json
import yaml
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

# Create the config CLI app
config_app = typer.Typer(
    name="config",
    help="Configuration management operations for system settings and environment configuration",
    rich_markup_mode="rich"
)


@config_app.command("get")
def get_config(
    key: str = typer.Argument(..., help="Configuration key to retrieve"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    format: str = typer.Option(
        "yaml", "--format", "-f", help="Output format: [yaml|json|plain]"
    ),
    default_value: Optional[str] = typer.Option(
        None, "--default", "-d", help="Default value if key not found"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Get configuration value by key."""
    
    try:
        # Use local configuration management
        # TODO: Implement domain-specific configuration management
        
        import json
        import yaml
        from pathlib import Path
        
        config_data = {}
        
        if config_file:
            config_manager.load_config_file(config_file)
        
        # Get configuration value
        try:
            value = config_manager.get(key, default=default_value)
            if value is None and default_value is None:
                console.print(f"[yellow]Configuration key '{key}' not found[/yellow]")
                raise typer.Exit(1)
            
            # Format output
            if format == "json":
                if isinstance(value, (dict, list)):
                    console.print(json.dumps(value, indent=2))
                else:
                    console.print(json.dumps({key: value}, indent=2))
            elif format == "yaml":
                if isinstance(value, (dict, list)):
                    console.print(yaml.dump(value, default_flow_style=False))
                else:
                    console.print(yaml.dump({key: value}, default_flow_style=False))
            else:  # plain
                console.print(str(value))
        
        except KeyError:
            if default_value:
                console.print(str(default_value))
            else:
                console.print(f"[red]Configuration key '{key}' not found[/red]")
                raise typer.Exit(1)
    
    except ImportError as e:
        console.print(f"[red]Error: Required configuration packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error retrieving configuration: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="Configuration value to set"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    value_type: str = typer.Option(
        "auto", "--type", "-t", help="Value type: [auto|string|int|float|bool|json]"
    ),
    create_path: bool = typer.Option(
        True, "--create-path/--no-create-path", help="Create nested key path if it doesn't exist"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before modification"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Set configuration value by key."""
    
    try:
        # Use local configuration management
        # TODO: Implement domain-specific configuration management
        
        import json
        import yaml
        from pathlib import Path
        
        config_data = {}
        
        if config_file:
            config_manager.load_config_file(config_file)
        
        # Parse value based on type
        parsed_value = _parse_value(value, value_type)
        
        # Create backup if requested
        if backup and config_file and config_file.exists():
            backup_file = config_file.parent / f"{config_file.name}.backup"
            import shutil
            shutil.copy2(config_file, backup_file)
            console.print(f"[blue]Backup created: {backup_file}[/blue]")
        
        # Set configuration value
        config_manager.set(key, parsed_value, create_path=create_path)
        
        # Save to file if specified
        if config_file:
            config_manager.save_config_file(config_file)
        
        console.print(f"[green]✓ Configuration '{key}' set successfully[/green]")
        console.print(f"Value: {parsed_value}")
    
    except ImportError as e:
        console.print(f"[red]Error: Required configuration packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error setting configuration: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("unset")
def unset_config(
    key: str = typer.Argument(..., help="Configuration key to remove"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before modification"
    ),
    confirm: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Confirm deletion"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Remove configuration key."""
    
    try:
        # Use local configuration management
        # TODO: Implement domain-specific configuration management
        
        import json
        import yaml
        from pathlib import Path
        
        config_data = {}
        
        if config_file:
            config_manager.load_config_file(config_file)
        
        # Check if key exists
        if not config_manager.has_key(key):
            console.print(f"[yellow]Configuration key '{key}' not found[/yellow]")
            return
        
        # Confirm deletion
        if confirm and not Confirm.ask(f"Remove configuration key '{key}'?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return
        
        # Create backup if requested
        if backup and config_file and config_file.exists():
            backup_file = config_file.parent / f"{config_file.name}.backup"
            import shutil
            shutil.copy2(config_file, backup_file)
            console.print(f"[blue]Backup created: {backup_file}[/blue]")
        
        # Remove configuration key
        config_manager.unset(key)
        
        # Save to file if specified
        if config_file:
            config_manager.save_config_file(config_file)
        
        console.print(f"[green]✓ Configuration '{key}' removed successfully[/green]")
    
    except ImportError as e:
        console.print(f"[red]Error: Required configuration packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error removing configuration: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("list")
def list_configs(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    prefix: Optional[str] = typer.Option(
        None, "--prefix", "-p", help="Filter keys by prefix"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: [table|json|yaml|keys]"
    ),
    show_values: bool = typer.Option(
        True, "--values/--no-values", help="Show configuration values"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """List all configuration keys and values."""
    
    try:
        # Use local configuration management
        # TODO: Implement domain-specific configuration management
        
        import json
        import yaml
        from pathlib import Path
        
        config_data = {}
        
        if config_file:
            config_manager.load_config_file(config_file)
        
        # Get all configurations
        all_configs = config_manager.get_all()
        
        # Filter by prefix if specified
        if prefix:
            all_configs = {
                k: v for k, v in all_configs.items() 
                if k.startswith(prefix)
            }
        
        if not all_configs:
            console.print("[yellow]No configuration keys found[/yellow]")
            return
        
        # Display based on format
        if format == "table":
            table = Table(title="Configuration Settings")
            table.add_column("Key", style="cyan")
            if show_values:
                table.add_column("Value", style="green")
                table.add_column("Type", style="blue")
            
            for key, value in all_configs.items():
                if show_values:
                    display_value = str(value)
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    table.add_row(key, display_value, type(value).__name__)
                else:
                    table.add_row(key)
            
            console.print(table)
        
        elif format == "json":
            if show_values:
                console.print(json.dumps(all_configs, indent=2, default=str))
            else:
                console.print(json.dumps(list(all_configs.keys()), indent=2))
        
        elif format == "yaml":
            if show_values:
                console.print(yaml.dump(all_configs, default_flow_style=False))
            else:
                console.print(yaml.dump(list(all_configs.keys()), default_flow_style=False))
        
        else:  # keys
            for key in all_configs.keys():
                console.print(key)
    
    except ImportError as e:
        console.print(f"[red]Error: Required configuration packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error listing configurations: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("validate")
def validate_config(
    config_file: Path = typer.Argument(..., help="Configuration file to validate"),
    schema_file: Optional[Path] = typer.Option(
        None, "--schema", "-s", help="JSON schema file for validation"
    ),
    format: str = typer.Option(
        "auto", "--format", "-f", help="Config file format: [auto|json|yaml|toml]"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Enable strict validation mode"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Validate configuration file against schema."""
    
    if not config_file.exists():
        console.print(f"[red]Configuration file {config_file} does not exist[/red]")
        raise typer.Exit(1)
    
    try:
        # Import validation packages
        # TODO: Implement within domain - from packages.shared.configuration.config_validator import ConfigValidator
        
        # Initialize validator
        validator = ConfigValidator()
        
        # Validate configuration
        validation_result = validator.validate_file(
            config_file,
            schema_file=schema_file,
            format=format,
            strict=strict
        )
        
        if validation_result["valid"]:
            console.print("[green]✓ Configuration file is valid[/green]")
        else:
            console.print("[red]✗ Configuration file has validation errors:[/red]")
            for error in validation_result["errors"]:
                console.print(f"  • {error}")
            
            if validation_result.get("warnings"):
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation_result["warnings"]:
                    console.print(f"  • {warning}")
            
            raise typer.Exit(1)
    
    except ImportError as e:
        console.print(f"[red]Error: Required validation packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error validating configuration: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("merge")
def merge_configs(
    base_config: Path = typer.Argument(..., help="Base configuration file"),
    merge_config: Path = typer.Argument(..., help="Configuration file to merge"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for merged configuration"
    ),
    strategy: str = typer.Option(
        "deep", "--strategy", "-s", help="Merge strategy: [deep|shallow|override]"
    ),
    conflict_resolution: str = typer.Option(
        "merge", "--conflicts", "-c", help="Conflict resolution: [merge|keep_base|keep_merge|prompt]"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup of base config"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Merge two configuration files."""
    
    if not base_config.exists():
        console.print(f"[red]Base configuration file {base_config} does not exist[/red]")
        raise typer.Exit(1)
    
    if not merge_config.exists():
        console.print(f"[red]Merge configuration file {merge_config} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = base_config.parent / f"{base_config.stem}_merged{base_config.suffix}"
    
    try:
        # Import merge packages
        # TODO: Implement within domain - from packages.shared.configuration.config_merger import ConfigMerger
        
        # Initialize merger
        merger = ConfigMerger()
        
        # Create backup if requested
        if backup:
            backup_file = base_config.parent / f"{base_config.name}.backup"
            import shutil
            shutil.copy2(base_config, backup_file)
            console.print(f"[blue]Backup created: {backup_file}[/blue]")
        
        # Merge configurations
        merge_result = merger.merge_files(
            base_config,
            merge_config,
            strategy=strategy,
            conflict_resolution=conflict_resolution
        )
        
        # Save merged configuration
        merger.save_merged_config(merge_result, output_file)
        
        console.print(f"[green]✓ Configuration files merged successfully[/green]")
        console.print(f"Merged configuration saved to: {output_file}")
        
        # Display merge summary
        if merge_result.get("conflicts"):
            console.print(f"\n[yellow]Conflicts resolved: {len(merge_result['conflicts'])}[/yellow]")
            for conflict in merge_result["conflicts"][:5]:  # Show first 5 conflicts
                console.print(f"  • {conflict}")
    
    except ImportError as e:
        console.print(f"[red]Error: Required merge packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error merging configurations: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("env")
def manage_environment(
    action: str = typer.Argument(..., help="Action: [list|set|unset|export|import]"),
    env_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Environment name"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    format: str = typer.Option(
        "shell", "--format", "-f", help="Export format: [shell|docker|k8s|json]"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Manage environment-specific configurations."""
    
    valid_actions = ["list", "set", "unset", "export", "import"]
    if action not in valid_actions:
        console.print(f"[red]Invalid action: {action}[/red]")
        console.print(f"Valid actions: {', '.join(valid_actions)}")
        raise typer.Exit(1)
    
    try:
        # Import environment packages
        # TODO: Implement within domain - from packages.shared.configuration.environment_manager import EnvironmentManager
        
        # Initialize environment manager
        env_manager = EnvironmentManager()
        
        if config_file:
            env_manager.load_config_file(config_file)
        
        if action == "list":
            environments = env_manager.list_environments()
            
            if not environments:
                console.print("[yellow]No environments configured[/yellow]")
                return
            
            table = Table(title="Configured Environments")
            table.add_column("Environment", style="cyan")
            table.add_column("Active", style="green")
            table.add_column("Variables", style="blue")
            
            for env in environments:
                is_active = "✓" if env["active"] else ""
                var_count = str(len(env.get("variables", {})))
                table.add_row(env["name"], is_active, var_count)
            
            console.print(table)
        
        elif action == "set":
            if not env_name:
                env_name = Prompt.ask("Environment name")
            
            # Set environment as active
            env_manager.set_active_environment(env_name)
            console.print(f"[green]✓ Environment '{env_name}' set as active[/green]")
        
        elif action == "unset":
            if not env_name:
                env_name = Prompt.ask("Environment name")
            
            # Remove environment
            env_manager.remove_environment(env_name)
            console.print(f"[green]✓ Environment '{env_name}' removed[/green]")
        
        elif action == "export":
            if not env_name:
                env_name = env_manager.get_active_environment()
            
            if not env_name:
                console.print("[red]No active environment to export[/red]")
                raise typer.Exit(1)
            
            # Export environment variables
            exported_vars = env_manager.export_environment(env_name, format=format)
            console.print(exported_vars)
        
        elif action == "import":
            if not env_name:
                env_name = Prompt.ask("Environment name")
            
            if not config_file:
                console.print("[red]Config file required for import[/red]")
                raise typer.Exit(1)
            
            # Import environment from file
            env_manager.import_environment(env_name, config_file)
            console.print(f"[green]✓ Environment '{env_name}' imported successfully[/green]")
    
    except ImportError as e:
        console.print(f"[red]Error: Required environment packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error managing environment: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@config_app.command("init")
def init_config(
    config_file: Path = typer.Argument(..., help="Configuration file to create"),
    template: str = typer.Option(
        "default", "--template", "-t", help="Configuration template: [default|minimal|advanced|custom]"
    ),
    format: str = typer.Option(
        "yaml", "--format", "-f", help="Configuration format: [yaml|json|toml]"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive configuration setup"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Initialize a new configuration file."""
    
    if config_file.exists() and not overwrite:
        console.print(f"[red]Configuration file {config_file} already exists[/red]")
        console.print("Use --overwrite to replace existing file")
        raise typer.Exit(1)
    
    try:
        # Import config initialization packages
        # TODO: Implement within domain - from packages.shared.configuration.config_initializer import ConfigInitializer
        
        # Initialize config initializer
        initializer = ConfigInitializer()
        
        # Generate configuration
        if interactive:
            config_data = initializer.interactive_setup(template)
        else:
            config_data = initializer.generate_template(template)
        
        # Save configuration file
        initializer.save_config(config_data, config_file, format=format)
        
        console.print(f"[green]✓ Configuration file created successfully[/green]")
        console.print(f"File: {config_file}")
        console.print(f"Template: {template}")
        console.print(f"Format: {format}")
        
        # Display basic configuration info
        if isinstance(config_data, dict):
            section_count = len(config_data)
            console.print(f"Sections: {section_count}")
    
    except ImportError as e:
        console.print(f"[red]Error: Required initialization packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error creating configuration: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _parse_value(value: str, value_type: str) -> Any:
    """Parse string value to appropriate type."""
    if value_type == "string":
        return value
    elif value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    elif value_type == "bool":
        return value.lower() in ("true", "yes", "1", "on")
    elif value_type == "json":
        return json.loads(value)
    else:  # auto
        # Try to automatically detect type
        try:
            # Try JSON first (handles lists, dicts, etc.)
            return json.loads(value)
        except json.JSONDecodeError:
            # Try boolean
            if value.lower() in ("true", "false", "yes", "no", "on", "off"):
                return value.lower() in ("true", "yes", "on")
            # Try integer
            try:
                return int(value)
            except ValueError:
                # Try float
                try:
                    return float(value)
                except ValueError:
                    # Default to string
                    return value


if __name__ == "__main__":
    config_app()