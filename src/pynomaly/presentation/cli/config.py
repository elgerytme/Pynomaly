"""CLI commands for configuration management."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from pynomaly.application.dto.configuration_dto import (
    ConfigurationCaptureRequestDTO,
    ConfigurationExportRequestDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    ExportFormat,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.infrastructure.persistence.configuration_repository import (
    ConfigurationRepository,
)

console = Console()
app = typer.Typer(
    name="config",
    help="Configuration management commands for experiment configurations",
    rich_markup_mode="rich",
)


@app.command("capture")
@require_feature("advanced_automl")
def capture_configuration(
    source: str = typer.Argument(
        ..., help="Configuration source (automl, cli, web_api, etc.)"
    ),
    parameters_file: Path = typer.Option(
        ..., "--params", "-p", help="JSON file with raw parameters"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for captured configuration"
    ),
    auto_save: bool = typer.Option(
        True, "--auto-save/--no-auto-save", help="Automatically save configuration"
    ),
    generate_name: bool = typer.Option(
        True,
        "--generate-name/--no-generate-name",
        help="Auto-generate configuration name",
    ),
    tags: list[str] | None = typer.Option(
        None, "--tag", "-t", help="Tags for configuration"
    ),
    user_id: str | None = typer.Option(None, "--user", "-u", help="User identifier"),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Configuration description"
    ),
):
    """Capture configuration from raw parameters."""
    try:
        # Load parameters from file
        if not parameters_file.exists():
            rprint(f"[red]Error:[/red] Parameters file not found: {parameters_file}")
            raise typer.Exit(1)

        with open(parameters_file) as f:
            raw_parameters = json.load(f)

        # Create capture service
        capture_service = ConfigurationCaptureService()

        # Create capture request
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource(source.lower()),
            raw_parameters=raw_parameters,
            source_context={"description": description, "cli_invocation": True},
            user_id=user_id,
            auto_save=auto_save,
            generate_name=generate_name,
            tags=tags or [],
        )

        # Capture configuration
        import asyncio

        response = asyncio.run(capture_service.capture_configuration(request))

        if response.success:
            config = response.configuration

            # Display success message
            rprint("[green]✓[/green] Configuration captured successfully")
            rprint(f"[blue]Name:[/blue] {config.name}")
            rprint(f"[blue]ID:[/blue] {config.id}")
            rprint(f"[blue]Algorithm:[/blue] {config.algorithm_config.algorithm_name}")

            if config.is_valid:
                rprint("[green]✓[/green] Configuration is valid")
            else:
                rprint("[yellow]⚠[/yellow] Configuration has validation issues:")
                for error in config.validation_errors:
                    rprint(f"  [red]Error:[/red] {error}")
                for warning in config.validation_warnings:
                    rprint(f"  [yellow]Warning:[/yellow] {warning}")

            # Save to output file if specified
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(config.model_dump(), f, indent=2, default=str)
                rprint(f"[blue]Saved to:[/blue] {output_file}")

        else:
            rprint(f"[red]Error:[/red] {response.message}")
            for error in response.errors:
                rprint(f"  [red]•[/red] {error}")
            raise typer.Exit(1)

    except Exception as e:
        rprint(f"[red]Error capturing configuration:[/red] {e}")
        raise typer.Exit(1)


@app.command("export")
def export_configurations(
    config_ids: list[str] = typer.Argument(..., help="Configuration IDs to export"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    export_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Export format (json, yaml, python, notebook, docker)",
    ),
    include_metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Include metadata in export"
    ),
    include_performance: bool = typer.Option(
        True, "--performance/--no-performance", help="Include performance results"
    ),
    include_lineage: bool = typer.Option(
        False, "--lineage/--no-lineage", help="Include lineage information"
    ),
    template_name: str | None = typer.Option(
        None, "--template", help="Export as template with given name"
    ),
    compress: bool = typer.Option(
        False, "--compress/--no-compress", help="Compress output file"
    ),
):
    """Export configurations to specified format."""
    try:
        # Parse configuration IDs
        parsed_ids = []
        for config_id_str in config_ids:
            try:
                parsed_ids.append(UUID(config_id_str))
            except ValueError:
                rprint(f"[red]Error:[/red] Invalid UUID format: {config_id_str}")
                raise typer.Exit(1)

        # Create capture service
        capture_service = ConfigurationCaptureService()

        # Create export request
        request = ConfigurationExportRequestDTO(
            configuration_ids=parsed_ids,
            export_format=ExportFormat(export_format.lower()),
            include_metadata=include_metadata,
            include_performance=include_performance,
            include_lineage=include_lineage,
            output_path=str(output_file),
            template_name=template_name,
            compress=compress,
        )

        # Export configurations
        import asyncio

        response = asyncio.run(capture_service.export_configurations(request))

        if response.success:
            rprint(f"[green]✓[/green] Exported {len(parsed_ids)} configurations")
            rprint(f"[blue]Format:[/blue] {export_format}")
            rprint(f"[blue]Output:[/blue] {output_file}")

            if response.export_files:
                rprint("[blue]Files created:[/blue]")
                for file_path in response.export_files:
                    rprint(f"  [green]•[/green] {file_path}")

        else:
            rprint(f"[red]Error:[/red] {response.message}")
            for error in response.errors:
                rprint(f"  [red]•[/red] {error}")
            raise typer.Exit(1)

    except Exception as e:
        rprint(f"[red]Error exporting configurations:[/red] {e}")
        raise typer.Exit(1)


@app.command("import")
def import_configurations(
    import_file: Path = typer.Argument(..., help="File to import configurations from"),
    storage_path: Path | None = typer.Option(
        None, "--storage", "-s", help="Configuration storage path"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", help="Overwrite existing configurations"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be imported without actually importing",
    ),
):
    """Import configurations from file."""
    try:
        if not import_file.exists():
            rprint(f"[red]Error:[/red] Import file not found: {import_file}")
            raise typer.Exit(1)

        # Create repository
        repo_path = storage_path or Path("data/configurations")
        repository = ConfigurationRepository(repo_path)

        if dry_run:
            # Preview import
            with open(import_file) as f:
                import_data = json.load(f)

            configurations = import_data.get("configurations", [])
            rprint("[blue]Import Preview:[/blue]")
            rprint(f"Found {len(configurations)} configurations to import")

            # Show sample configurations
            for i, config_data in enumerate(configurations[:5]):
                name = config_data.get("name", "Unknown")
                algorithm = config_data.get("algorithm_config", {}).get(
                    "algorithm_name", "Unknown"
                )
                rprint(f"  [green]{i + 1}.[/green] {name} ({algorithm})")

            if len(configurations) > 5:
                rprint(f"  ... and {len(configurations) - 5} more")

        else:
            # Perform import
            import asyncio

            imported_count = asyncio.run(
                repository.import_configurations(import_file, overwrite)
            )

            if imported_count > 0:
                rprint(
                    f"[green]✓[/green] Successfully imported {imported_count} configurations"
                )
                rprint(f"[blue]Storage:[/blue] {repo_path}")
            else:
                rprint("[yellow]⚠[/yellow] No configurations were imported")

    except Exception as e:
        rprint(f"[red]Error importing configurations:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_configurations(
    storage_path: Path | None = typer.Option(
        None, "--storage", "-s", help="Configuration storage path"
    ),
    source: str | None = typer.Option(None, "--source", help="Filter by source"),
    algorithm: str | None = typer.Option(
        None, "--algorithm", help="Filter by algorithm"
    ),
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of configurations to show"
    ),
    show_details: bool = typer.Option(
        False, "--details/--no-details", help="Show detailed information"
    ),
):
    """List stored configurations."""
    try:
        # Create repository
        repo_path = storage_path or Path("data/configurations")
        repository = ConfigurationRepository(repo_path)

        # Filter parameters
        source_filter = ConfigurationSource(source.lower()) if source else None

        # Load configurations
        import asyncio

        configurations = asyncio.run(
            repository.list_configurations(source=source_filter, limit=limit)
        )

        # Apply algorithm filter
        if algorithm:
            configurations = [
                config
                for config in configurations
                if config.algorithm_config.algorithm_name == algorithm
            ]

        if not configurations:
            rprint("[yellow]No configurations found.[/yellow]")
            return

        # Display configurations
        if show_details:
            # Detailed view
            for config in configurations:
                panel_content = f"""
[blue]ID:[/blue] {config.id}
[blue]Algorithm:[/blue] {config.algorithm_config.algorithm_name}
[blue]Source:[/blue] {config.metadata.source}
[blue]Created:[/blue] {config.metadata.created_at}
[blue]Status:[/blue] {config.status}
[blue]Valid:[/blue] {"✓" if config.is_valid else "✗"}
"""
                if config.metadata.tags:
                    panel_content += (
                        f"[blue]Tags:[/blue] {', '.join(config.metadata.tags)}\n"
                    )

                if config.performance_results and config.performance_results.accuracy:
                    panel_content += f"[blue]Accuracy:[/blue] {config.performance_results.accuracy:.3f}\n"

                rprint(Panel(panel_content, title=config.name, border_style="blue"))

        else:
            # Table view
            table = Table(title=f"Configurations ({len(configurations)} found)")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Algorithm", style="green")
            table.add_column("Source", style="blue")
            table.add_column("Created", style="yellow")
            table.add_column("Status", style="magenta")
            table.add_column("Valid", style="green")

            for config in configurations:
                table.add_row(
                    config.name[:30] + "..." if len(config.name) > 30 else config.name,
                    config.algorithm_config.algorithm_name,
                    config.metadata.source,
                    config.metadata.created_at.strftime("%Y-%m-%d %H:%M"),
                    config.status,
                    "✓" if config.is_valid else "✗",
                )

            console.print(table)

    except Exception as e:
        rprint(f"[red]Error listing configurations:[/red] {e}")
        raise typer.Exit(1)


@app.command("search")
def search_configurations(
    query: str = typer.Argument(..., help="Search query"),
    storage_path: Path | None = typer.Option(
        None, "--storage", "-s", help="Configuration storage path"
    ),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Filter by tags"),
    source: str | None = typer.Option(None, "--source", help="Filter by source"),
    algorithm: str | None = typer.Option(
        None, "--algorithm", help="Filter by algorithm"
    ),
    min_accuracy: float | None = typer.Option(
        None, "--min-accuracy", help="Minimum accuracy filter"
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results to return"),
    sort_by: str = typer.Option(
        "created_at",
        "--sort",
        help="Sort field (created_at, name, algorithm, accuracy)",
    ),
    sort_order: str = typer.Option("desc", "--order", help="Sort order (asc, desc)"),
):
    """Search configurations by query and filters."""
    try:
        # Create repository
        repo_path = storage_path or Path("data/configurations")
        repository = ConfigurationRepository(repo_path)

        # Create search request
        request = ConfigurationSearchRequestDTO(
            query=query,
            tags=tags,
            source=ConfigurationSource(source.lower()) if source else None,
            algorithm=algorithm,
            min_accuracy=min_accuracy,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        # Perform search
        import asyncio

        results = asyncio.run(repository.search_configurations(request))

        if not results:
            rprint(
                f"[yellow]No configurations found matching query: '{query}'[/yellow]"
            )
            return

        # Display results
        rprint(
            f"[green]Found {len(results)} configurations matching '{query}'[/green]\n"
        )

        table = Table()
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Algorithm", style="green")
        table.add_column("Source", style="blue")
        table.add_column("Tags", style="yellow")
        table.add_column("Accuracy", style="magenta")

        for config in results:
            accuracy_str = ""
            if config.performance_results and config.performance_results.accuracy:
                accuracy_str = f"{config.performance_results.accuracy:.3f}"

            tags_str = (
                ", ".join(config.metadata.tags[:3]) if config.metadata.tags else ""
            )
            if len(config.metadata.tags) > 3:
                tags_str += "..."

            table.add_row(
                config.name[:25] + "..." if len(config.name) > 25 else config.name,
                config.algorithm_config.algorithm_name,
                config.metadata.source,
                tags_str,
                accuracy_str,
            )

        console.print(table)

    except Exception as e:
        rprint(f"[red]Error searching configurations:[/red] {e}")
        raise typer.Exit(1)


@app.command("show")
def show_configuration(
    config_id: str = typer.Argument(..., help="Configuration ID to show"),
    storage_path: Path | None = typer.Option(
        None, "--storage", "-s", help="Configuration storage path"
    ),
    include_performance: bool = typer.Option(
        True, "--performance/--no-performance", help="Include performance details"
    ),
    include_lineage: bool = typer.Option(
        False, "--lineage/--no-lineage", help="Include lineage information"
    ),
    output_format: str = typer.Option(
        "rich", "--format", help="Output format (rich, json, yaml)"
    ),
):
    """Show detailed configuration information."""
    try:
        # Parse configuration ID
        try:
            parsed_id = UUID(config_id)
        except ValueError:
            rprint(f"[red]Error:[/red] Invalid UUID format: {config_id}")
            raise typer.Exit(1)

        # Create repository
        repo_path = storage_path or Path("data/configurations")
        repository = ConfigurationRepository(repo_path)

        # Load configuration
        import asyncio

        config = asyncio.run(repository.load_configuration(parsed_id))

        if not config:
            rprint(f"[red]Error:[/red] Configuration not found: {config_id}")
            raise typer.Exit(1)

        # Display configuration
        if output_format == "json":
            config_data = config.model_dump()
            rprint(json.dumps(config_data, indent=2, default=str))

        elif output_format == "yaml":
            import yaml

            config_data = config.model_dump()
            rprint(yaml.dump(config_data, default_flow_style=False))

        else:
            # Rich format
            tree = Tree(f"[bold cyan]{config.name}[/bold cyan]")

            # Basic information
            basic_tree = tree.add("[bold]Basic Information[/bold]")
            basic_tree.add(f"ID: {config.id}")
            basic_tree.add(f"Status: {config.status}")
            basic_tree.add(f"Valid: {'✓' if config.is_valid else '✗'}")
            basic_tree.add(f"Created: {config.metadata.created_at}")
            basic_tree.add(f"Source: {config.metadata.source}")

            # Algorithm configuration
            algo_tree = tree.add("[bold]Algorithm Configuration[/bold]")
            algo_tree.add(f"Algorithm: {config.algorithm_config.algorithm_name}")
            algo_tree.add(f"Contamination: {config.algorithm_config.contamination}")
            algo_tree.add(f"Random State: {config.algorithm_config.random_state}")
            if config.algorithm_config.hyperparameters:
                params_tree = algo_tree.add("Hyperparameters")
                for key, value in config.algorithm_config.hyperparameters.items():
                    params_tree.add(f"{key}: {value}")

            # Dataset configuration
            dataset_tree = tree.add("[bold]Dataset Configuration[/bold]")
            if config.dataset_config.dataset_path:
                dataset_tree.add(f"Path: {config.dataset_config.dataset_path}")
            if config.dataset_config.dataset_name:
                dataset_tree.add(f"Name: {config.dataset_config.dataset_name}")
            if config.dataset_config.feature_columns:
                dataset_tree.add(
                    f"Features: {len(config.dataset_config.feature_columns)} columns"
                )

            # Performance results
            if include_performance and config.performance_results:
                perf_tree = tree.add("[bold]Performance Results[/bold]")
                if config.performance_results.accuracy:
                    perf_tree.add(
                        f"Accuracy: {config.performance_results.accuracy:.4f}"
                    )
                if config.performance_results.precision:
                    perf_tree.add(
                        f"Precision: {config.performance_results.precision:.4f}"
                    )
                if config.performance_results.recall:
                    perf_tree.add(f"Recall: {config.performance_results.recall:.4f}")
                if config.performance_results.f1_score:
                    perf_tree.add(
                        f"F1 Score: {config.performance_results.f1_score:.4f}"
                    )
                if config.performance_results.training_time_seconds:
                    perf_tree.add(
                        f"Training Time: {config.performance_results.training_time_seconds:.2f}s"
                    )

            # Validation issues
            if config.validation_errors or config.validation_warnings:
                validation_tree = tree.add("[bold]Validation Issues[/bold]")
                for error in config.validation_errors:
                    validation_tree.add(f"[red]Error:[/red] {error}")
                for warning in config.validation_warnings:
                    validation_tree.add(f"[yellow]Warning:[/yellow] {warning}")

            # Tags
            if config.metadata.tags:
                tags_tree = tree.add("[bold]Tags[/bold]")
                for tag in config.metadata.tags:
                    tags_tree.add(tag)

            # Lineage information
            if include_lineage and config.lineage:
                lineage_tree = tree.add("[bold]Lineage Information[/bold]")
                if config.lineage.parent_configurations:
                    lineage_tree.add(
                        f"Parent Configs: {len(config.lineage.parent_configurations)}"
                    )
                if config.lineage.derivation_method:
                    lineage_tree.add(f"Derivation: {config.lineage.derivation_method}")
                if config.lineage.git_commit:
                    lineage_tree.add(f"Git Commit: {config.lineage.git_commit}")

            console.print(tree)

    except Exception as e:
        rprint(f"[red]Error showing configuration:[/red] {e}")
        raise typer.Exit(1)


@app.command("stats")
def show_statistics(
    storage_path: Path | None = typer.Option(
        None, "--storage", "-s", help="Configuration storage path"
    ),
):
    """Show repository statistics."""
    try:
        # Create repository
        repo_path = storage_path or Path("data/configurations")
        repository = ConfigurationRepository(repo_path)

        # Get statistics
        stats = repository.get_repository_statistics()

        if not stats:
            rprint("[yellow]No statistics available.[/yellow]")
            return

        # Display statistics
        panel_content = f"""
[blue]Total Configurations:[/blue] {stats.get("total_configurations", 0)}
[blue]Total Collections:[/blue] {stats.get("total_collections", 0)}
[blue]Total Templates:[/blue] {stats.get("total_templates", 0)}
[blue]Total Backups:[/blue] {stats.get("total_backups", 0)}
[blue]Storage Size:[/blue] {stats.get("storage_size_bytes", 0) / (1024 * 1024):.2f} MB
[blue]Storage Path:[/blue] {stats.get("storage_path")}
[blue]Versioning Enabled:[/blue] {"✓" if stats.get("versioning_enabled") else "✗"}
[blue]Compression Enabled:[/blue] {"✓" if stats.get("compression_enabled") else "✗"}
[blue]Backup Enabled:[/blue] {"✓" if stats.get("backup_enabled") else "✗"}
[blue]Last Modified:[/blue] {stats.get("last_modified", "Never")}
[blue]Repository Created:[/blue] {stats.get("repository_created", "Unknown")}
"""

        rprint(
            Panel(
                panel_content,
                title="Configuration Repository Statistics",
                border_style="blue",
            )
        )

    except Exception as e:
        rprint(f"[red]Error getting statistics:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
