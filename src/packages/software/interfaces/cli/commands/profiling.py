"""Data Profiling CLI Commands for automated data discovery and analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

# Create the profiling CLI app
profiling_app = typer.Typer(
    name="profiling",
    help="Data profiling operations for automated data discovery and schema analysis",
    rich_markup_mode="rich"
)


@profiling_app.command("profile")
def profile_dataset(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for profiling results"
    ),
    profile_type: str = typer.Option(
        "comprehensive", "--type", "-t",
        help="Profile type: [comprehensive|basic|statistical|schema|quality]"
    ),
    format: str = typer.Option(
        "json", "--format", "-f",
        help="Output format: [json|html|csv|yaml]"
    ),
    sample_size: Optional[int] = typer.Option(
        None, "--sample", "-s", help="Sample size for large datasets"
    ),
    include_correlations: bool = typer.Option(
        True, "--correlations/--no-correlations", help="Include correlation analysis"
    ),
    detect_patterns: bool = typer.Option(
        True, "--patterns/--no-patterns", help="Detect data patterns and anomalies"
    ),
    generate_report: bool = typer.Option(
        True, "--report/--no-report", help="Generate HTML report"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Use parallel processing"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Profile a data_collection to discover schema, statistics, and data quality issues."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_profile"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data_collection
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import profiling packages
            from packages.data_profiling.application.services.profiling_orchestrator import ProfilingOrchestrator
            from packages.data_profiling.application.services.schema_discovery_service import SchemaDiscoveryService
            from packages.data_profiling.application.services.statistical_profiler import StatisticalProfiler
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            if sample_size and len(data_collection) > sample_size:
                data_collection = data_collection.sample(n=sample_size, random_state=42)
                console.print(f"[yellow]Sampled {sample_size} rows from {len(data_collection)} total rows[/yellow]")
            
            progress.update(task, description="Discovering schema...")
            
            # Initialize services
            orchestrator = ProfilingOrchestrator()
            schema_service = SchemaDiscoveryService()
            stats_profiler = StatisticalProfiler()
            
            # Configure profiling
            config = {
                "include_correlations": include_correlations,
                "detect_patterns": detect_patterns,
                "parallel_processing": parallel,
                "sample_size": sample_size
            }
            
            # Perform profiling based on type
            if profile_type == "comprehensive":
                progress.update(task, description="Performing comprehensive profiling...")
                profile_results = orchestrator.profile_data_collection_comprehensive(data_collection, config)
                
            elif profile_type == "basic":
                progress.update(task, description="Performing basic profiling...")
                profile_results = orchestrator.profile_data_collection_basic(data_collection)
                
            elif profile_type == "statistical":
                progress.update(task, description="Performing statistical profiling...")
                profile_results = stats_profiler.generate_statistical_profile(data_collection)
                
            elif profile_type == "schema":
                progress.update(task, description="Discovering schema...")
                profile_results = schema_service.discover_schema(data_collection)
                
            elif profile_type == "quality":
                progress.update(task, description="Analyzing data quality...")
                from packages.data_quality.application.services.quality_assessment_service import QualityAssessmentService
                quality_service = QualityAssessmentService()
                profile_results = quality_service.assess_data_collection_quality(data_collection)
                
            else:
                console.print(f"[red]Unknown profile type: {profile_type}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Saving profiling results...")
            
            # Save results
            output_file = output_dir / f"profile.{format}"
            
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(profile_results, f, indent=2, default=str)
            elif format == "yaml":
                import yaml
                with open(output_file, 'w') as f:
                    yaml.dump(profile_results, f, default_flow_style=False)
            elif format == "csv":
                # Convert profile results to tabular format
                import pandas as pd
                df = pd.DataFrame(profile_results.get('columns', {}))
                df.to_csv(output_file, index=False)
            
            # Generate HTML report if requested
            if generate_report:
                from packages.data_profiling.application.services.profile_report_generator import ProfileReportGenerator
                report_generator = ProfileReportGenerator()
                report_file = output_dir / "profile_report.html"
                report_generator.generate_html_report(profile_results, data_collection, report_file)
            
            progress.update(task, description="Profiling complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required profiling packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during profiling: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Display results summary
    console.print("\n[green]✓ DataCollection profiling completed successfully![/green]")
    console.print(f"Profile saved to: {output_file}")
    
    if generate_report:
        console.print(f"HTML report: {output_dir / 'profile_report.html'}")
    
    # Display key statistics
    if profile_results and "summary" in profile_results:
        summary = profile_results["summary"]
        
        panel_content = f"""
[bold]DataCollection Overview[/bold]
• Rows: {summary.get('total_rows', 'N/A'):,}
• Columns: {summary.get('total_columns', 'N/A')}
• Missing Values: {summary.get('total_missing', 'N/A'):,} ({summary.get('missing_percentage', 0):.1f}%)
• Data Types: {', '.join(summary.get('data_types', []))}
• Memory Usage: {summary.get('memory_usage', 'N/A')}
        """
        
        console.print(Panel(panel_content, title="Profile Summary", border_style="green"))


@profiling_app.command("compare")
def compare_profiles(
    profile1: Path = typer.Argument(..., help="First profile file"),
    profile2: Path = typer.Argument(..., help="Second profile file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for comparison results"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: [json|html|csv]"
    ),
    threshold: float = typer.Option(
        0.05, "--threshold", "-t", help="Threshold for detecting significant changes"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Compare two data profiles to detect schema drift and changes."""
    
    if not profile1.exists():
        console.print(f"[red]Error: Profile file {profile1} does not exist[/red]")
        raise typer.Exit(1)
    
    if not profile2.exists():
        console.print(f"[red]Error: Profile file {profile2} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = profile1.parent / f"profile_comparison.{format}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading profiles...", total=None)
        
        try:
            # Load profiles
            with open(profile1, 'r') as f:
                profile_data1 = json.load(f)
            
            with open(profile2, 'r') as f:
                profile_data2 = json.load(f)
            
            progress.update(task, description="Comparing profiles...")
            
            # Import comparison service
            from packages.data_profiling.application.services.profile_comparison_service import ProfileComparisonService
            
            comparison_service = ProfileComparisonService()
            comparison_results = comparison_service.compare_profiles(
                profile_data1, profile_data2, threshold=threshold
            )
            
            progress.update(task, description="Generating comparison report...")
            
            # Save results
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(comparison_results, f, indent=2, default=str)
            elif format == "html":
                comparison_service.generate_comparison_report(
                    comparison_results, output_file
                )
            
            progress.update(task, description="Comparison complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during comparison: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Profile comparison completed successfully![/green]")
    console.print(f"Comparison saved to: {output_file}")
    
    # Display key changes
    if comparison_results and "changes" in comparison_results:
        changes = comparison_results["changes"]
        
        table = Table(title="Detected Changes")
        table.add_column("Change Type", style="cyan")
        table.add_column("Details", style="yellow")
        table.add_column("Severity", style="red")
        
        for change in changes[:10]:  # Show first 10 changes
            table.add_row(
                change.get("type", "Unknown"),
                change.get("description", ""),
                change.get("severity", "Low")
            )
        
        console.print(table)
        
        if len(changes) > 10:
            console.print(f"[yellow]... and {len(changes) - 10} more changes[/yellow]")


@profiling_app.command("schema")
def discover_schema(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for schema definition"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format: [json|yaml|sql|avro]"
    ),
    infer_relationships: bool = typer.Option(
        True, "--relationships/--no-relationships", help="Infer foreign key relationships"
    ),
    sample_size: Optional[int] = typer.Option(
        None, "--sample", "-s", help="Sample size for schema inference"
    ),
    confidence_threshold: float = typer.Option(
        0.95, "--confidence", "-c", help="Confidence threshold for type inference"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Automatically discover and generate schema definition from data_collection."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_schema.{format}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import schema discovery packages
            from packages.data_profiling.application.services.schema_discovery_service import SchemaDiscoveryService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            if sample_size and len(data_collection) > sample_size:
                data_collection = data_collection.sample(n=sample_size, random_state=42)
            
            progress.update(task, description="Discovering schema...")
            
            # Initialize schema discovery service
            schema_service = SchemaDiscoveryService()
            
            # Discover schema
            schema = schema_service.discover_schema(
                data_collection,
                infer_relationships=infer_relationships,
                confidence_threshold=confidence_threshold
            )
            
            progress.update(task, description="Generating schema definition...")
            
            # Generate schema in requested format
            if format == "json":
                schema_output = schema_service.export_schema_json(schema)
            elif format == "yaml":
                schema_output = schema_service.export_schema_yaml(schema)
            elif format == "sql":
                schema_output = schema_service.export_schema_sql(schema)
            elif format == "avro":
                schema_output = schema_service.export_schema_avro(schema)
            else:
                console.print(f"[red]Unsupported format: {format}[/red]")
                raise typer.Exit(1)
            
            # Save schema
            with open(output_file, 'w') as f:
                if isinstance(schema_output, dict):
                    json.dump(schema_output, f, indent=2)
                else:
                    f.write(schema_output)
            
            progress.update(task, description="Schema discovery complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during schema discovery: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Schema discovery completed successfully![/green]")
    console.print(f"Schema saved to: {output_file}")
    
    # Display schema summary
    if schema and "columns" in schema:
        table = Table(title="Discovered Schema")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Nullable", style="yellow")
        table.add_column("Constraints", style="blue")
        
        for col_name, col_info in schema["columns"].items():
            constraints = ", ".join(col_info.get("constraints", []))
            table.add_row(
                col_name,
                col_info.get("type", "unknown"),
                str(col_info.get("nullable", True)),
                constraints or "None"
            )
        
        console.print(table)


@profiling_app.command("patterns")
def detect_patterns(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for pattern analysis"
    ),
    pattern_types: List[str] = typer.Option(
        ["format", "semantic", "statistical"], "--type", "-t",
        help="Pattern types to detect: [format|semantic|statistical|temporal]"
    ),
    min_frequency: float = typer.Option(
        0.05, "--min-freq", help="Minimum pattern frequency to report"
    ),
    max_patterns: int = typer.Option(
        100, "--max-patterns", help="Maximum number of patterns to report per column"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Detect data patterns and classify semantic types."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_patterns.json"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import pattern processing packages
            from packages.data_profiling.application.services.pattern_recognition_service import PatternRecognitionService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Detecting patterns...")
            
            # Initialize pattern recognition service
            pattern_service = PatternRecognitionService()
            
            # Detect patterns
            patterns = pattern_service.detect_patterns(
                data_collection,
                pattern_types=pattern_types,
                min_frequency=min_frequency,
                max_patterns=max_patterns
            )
            
            progress.update(task, description="Saving pattern analysis...")
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(patterns, f, indent=2, default=str)
            
            progress.update(task, description="Pattern processing complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during pattern processing: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Pattern processing completed successfully![/green]")
    console.print(f"Patterns saved to: {output_file}")
    
    # Display pattern summary
    if patterns:
        total_patterns = sum(len(col_patterns) for col_patterns in patterns.values())
        console.print(f"[blue]Total patterns detected: {total_patterns}[/blue]")
        
        for col_name, col_patterns in list(patterns.items())[:5]:  # Show first 5 columns
            if col_patterns:
                console.print(f"\n[cyan]{col_name}[/cyan]: {len(col_patterns)} patterns")
                for pattern in col_patterns[:3]:  # Show first 3 patterns per column
                    console.print(f"  • {pattern.get('type', 'Unknown')}: {pattern.get('description', '')}")


if __name__ == "__main__":
    profiling_app()