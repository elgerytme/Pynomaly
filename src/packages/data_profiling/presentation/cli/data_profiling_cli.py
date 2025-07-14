#!/usr/bin/env python3
"""Data Profiling CLI using Typer."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
import structlog
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.tree import Tree

# Import domain entities and use cases
from ...application.use_cases.execute_data_profiling import ExecuteDataProfilingUseCase
from ...domain.entities.data_profile import ProfileId, DatasetId, ProfilingStatus
from ...infrastructure.adapters.in_memory_data_profile_repository import InMemoryDataProfileRepository

logger = structlog.get_logger(__name__)
console = Console()

# Create Typer app
app = typer.Typer(
    name="data-profiling",
    help="📊 Data Profiling and Quality Assessment Tools",
    add_completion=True,
    rich_markup_mode="rich"
)

# Dependency setup
def get_repository():
    """Get repository instance."""
    return InMemoryDataProfileRepository()

def get_use_case(repository=None):
    """Get use case instance."""
    if repository is None:
        repository = get_repository()
    return ExecuteDataProfilingUseCase(repository)

@app.command("profile")
def create_profile(
    data_file: Path = typer.Argument(..., help="Path to dataset file (CSV, JSON, Parquet)"),
    strategy: str = typer.Option(
        "full", 
        "--strategy", "-s", 
        help="Profiling strategy: full, sample, adaptive"
    ),
    sample_size: Optional[int] = typer.Option(
        None, 
        "--sample-size", "-n", 
        help="Sample size for sampling strategies"
    ),
    sample_percentage: Optional[float] = typer.Option(
        None, 
        "--sample-percentage", "-p", 
        help="Sample percentage (0-100)"
    ),
    include_patterns: bool = typer.Option(
        True, 
        "--patterns/--no-patterns", 
        help="Include pattern discovery"
    ),
    include_statistics: bool = typer.Option(
        True, 
        "--statistics/--no-statistics", 
        help="Include statistical analysis"
    ),
    include_quality: bool = typer.Option(
        True, 
        "--quality/--no-quality", 
        help="Include quality assessment"
    ),
    output: Optional[Path] = typer.Option(
        None, 
        "--output", "-o", 
        help="Output file for results (JSON)"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable verbose output"
    )
):
    """Create and execute a data profiling job."""
    try:
        console.print(f"[bold blue]📊 Data Profiling[/bold blue]")
        console.print(f"Data file: {data_file}")
        console.print(f"Strategy: {strategy}")
        
        # Validate file exists
        if not data_file.exists():
            console.print(f"[red]❌ File not found: {data_file}[/red]")
            raise typer.Exit(1)
        
        # Load data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)
            
            if data_file.suffix.lower() == '.csv':
                data = pd.read_csv(data_file)
            elif data_file.suffix.lower() == '.json':
                data = pd.read_json(data_file)
            elif data_file.suffix.lower() == '.parquet':
                data = pd.read_parquet(data_file)
            else:
                console.print(f"[red]❌ Unsupported file format: {data_file.suffix}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description=f"Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Apply sampling if specified
        if strategy == "sample":
            if sample_size and sample_size < len(data):
                data = data.sample(n=sample_size, random_state=42)
                console.print(f"Applied sampling: {len(data)} rows")
            elif sample_percentage and sample_percentage < 100:
                data = data.sample(frac=sample_percentage/100, random_state=42)
                console.print(f"Applied sampling: {len(data)} rows ({sample_percentage}%)")
        
        # Configure profiling
        profiling_config = {
            "strategy": strategy,
            "include_patterns": include_patterns,
            "include_statistical_analysis": include_statistics,
            "include_quality_assessment": include_quality
        }
        
        if sample_size:
            profiling_config["sample_size"] = sample_size
        if sample_percentage:
            profiling_config["sample_percentage"] = sample_percentage
        
        console.print(f"Configuration: {profiling_config}")
        
        # Execute profiling
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing profiling...", total=None)
            
            # Create mock dataset ID
            dataset_id = DatasetId(value=UUID('12345678-1234-5678-9012-123456789012'))
            
            # Run profiling
            use_case = get_use_case()
            profile = asyncio.run(
                use_case.execute(
                    dataset_id=dataset_id,
                    data=data,
                    source_type="file",
                    source_connection={"file_path": str(data_file)},
                    profiling_config=profiling_config
                )
            )
            
            progress.update(task, description="Profiling completed")
        
        # Display results
        _display_profile_results(profile, verbose)
        
        # Save results if output specified
        if output:
            results = _convert_profile_to_dict(profile)
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"[green]✅ Results saved to: {output}[/green]")
        
        console.print("[green]✅ Profiling completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command("list")
def list_profiles(
    dataset_id: Optional[str] = typer.Option(None, "--dataset", help="Filter by dataset ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """List data profiles."""
    try:
        use_case = get_use_case()
        
        # Get profiles
        if dataset_id:
            profiles = asyncio.run(
                use_case.get_profiles_by_dataset(
                    DatasetId(value=UUID(dataset_id))
                )
            )
        else:
            profiles = asyncio.run(use_case.repository.list_all())
        
        # Filter by status if provided
        if status:
            try:
                status_enum = ProfilingStatus(status)
                profiles = [p for p in profiles if p.status == status_enum]
            except ValueError:
                console.print(f"[red]❌ Invalid status: {status}[/red]")
                raise typer.Exit(1)
        
        # Limit results
        profiles = profiles[:limit]
        
        if not profiles:
            console.print("[yellow]No profiles found[/yellow]")
            return
        
        # Display table
        table = Table(title="Data Profiles")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Dataset", style="green", no_wrap=True)
        table.add_column("Status", style="yellow")
        table.add_column("Source", style="blue")
        table.add_column("Created", style="dim")
        
        if verbose:
            table.add_column("Quality Score", style="magenta")
            table.add_column("Patterns", style="white")
        
        for profile in profiles:
            row = [
                str(profile.profile_id.value)[:8] + "...",
                str(profile.dataset_id.value)[:8] + "...",
                profile.status.value,
                profile.source_type,
                profile.created_at.strftime("%Y-%m-%d %H:%M")
            ]
            
            if verbose:
                quality_score = "N/A"
                patterns_count = "N/A"
                
                if profile.quality_assessment:
                    quality_score = f"{profile.quality_assessment.overall_score:.2f}"
                
                if profile.schema_profile and profile.schema_profile.columns:
                    total_patterns = sum(len(col.patterns) for col in profile.schema_profile.columns)
                    patterns_count = str(total_patterns)
                
                row.extend([quality_score, patterns_count])
            
            table.add_row(*row)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("get")
def get_profile(
    profile_id: str = typer.Argument(..., help="Profile ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Get details of a specific profile."""
    try:
        use_case = get_use_case()
        
        # Get profile
        profile = asyncio.run(
            use_case.get_profile_by_id(
                ProfileId(value=UUID(profile_id))
            )
        )
        
        if not profile:
            console.print(f"[red]❌ Profile not found: {profile_id}[/red]")
            raise typer.Exit(1)
        
        # Display profile details
        _display_profile_results(profile, verbose)
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid profile ID: {profile_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("latest")
def get_latest_profile(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Get the latest profile for a dataset."""
    try:
        use_case = get_use_case()
        
        # Get latest profile
        profile = asyncio.run(
            use_case.get_latest_profile_by_dataset(
                DatasetId(value=UUID(dataset_id))
            )
        )
        
        if not profile:
            console.print(f"[red]❌ No profiles found for dataset: {dataset_id}[/red]")
            raise typer.Exit(1)
        
        # Display profile details
        _display_profile_results(profile, verbose)
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid dataset ID: {dataset_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("health")
def health_check():
    """Check the health of the data profiling service."""
    try:
        console.print("[blue]🔍 Checking service health...[/blue]")
        
        # Test repository connection
        repository = get_repository()
        
        # Basic health check
        health_status = {
            "service": "data-profiling",
            "status": "healthy",
            "repository": "connected",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Display health status
        table = Table(title="Service Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        for key, value in health_status.items():
            if key != "timestamp":
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
        console.print(f"[green]✅ Service is healthy[/green]")
        console.print(f"[dim]Checked at: {health_status['timestamp']}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Service unhealthy: {str(e)}[/red]")
        raise typer.Exit(1)

def _display_profile_results(profile, verbose: bool = False):
    """Display profile results in a formatted way."""
    
    # Basic information panel
    info_content = f"""
[bold]Profile ID:[/bold] {profile.profile_id.value}
[bold]Dataset ID:[/bold] {profile.dataset_id.value}
[bold]Status:[/bold] {profile.status.value}
[bold]Source Type:[/bold] {profile.source_type}
[bold]Created:[/bold] {profile.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    if profile.started_at:
        info_content += f"[bold]Started:[/bold] {profile.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    if profile.completed_at:
        info_content += f"[bold]Completed:[/bold] {profile.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    console.print(Panel(info_content.strip(), title="Profile Information", border_style="blue"))
    
    # Schema Profile
    if profile.schema_profile:
        schema = profile.schema_profile
        
        # Schema overview
        schema_content = f"""
[bold]Table:[/bold] {schema.table_name}
[bold]Rows:[/bold] {schema.total_rows:,}
[bold]Columns:[/bold] {schema.total_columns}
"""
        
        if schema.estimated_size_bytes:
            size_mb = schema.estimated_size_bytes / (1024 * 1024)
            schema_content += f"[bold]Estimated Size:[/bold] {size_mb:.2f} MB\n"
        
        console.print(Panel(schema_content.strip(), title="Schema Profile", border_style="green"))
        
        # Column details
        if verbose and schema.columns:
            column_table = Table(title="Column Details")
            column_table.add_column("Column", style="cyan")
            column_table.add_column("Type", style="yellow")
            column_table.add_column("Nullable", style="blue")
            column_table.add_column("Unique", style="green")
            column_table.add_column("Missing %", style="red")
            column_table.add_column("Quality", style="magenta")
            
            for col in schema.columns:
                missing_pct = (col.distribution.null_count / col.distribution.total_count) * 100 if col.distribution.total_count > 0 else 0
                
                column_table.add_row(
                    col.column_name,
                    col.data_type.value,
                    "✓" if col.nullable else "✗",
                    str(col.distribution.unique_count),
                    f"{missing_pct:.1f}%",
                    f"{col.quality_score:.2f}"
                )
            
            console.print(column_table)
    
    # Quality Assessment
    if profile.quality_assessment:
        qa = profile.quality_assessment
        
        quality_tree = Tree("📋 Quality Assessment")
        quality_tree.add(f"Overall Score: [bold]{qa.overall_score:.3f}[/bold]")
        
        dimensions = quality_tree.add("Quality Dimensions")
        dimensions.add(f"Completeness: {qa.completeness_score:.3f}")
        dimensions.add(f"Consistency: {qa.consistency_score:.3f}")
        dimensions.add(f"Accuracy: {qa.accuracy_score:.3f}")
        dimensions.add(f"Validity: {qa.validity_score:.3f}")
        dimensions.add(f"Uniqueness: {qa.uniqueness_score:.3f}")
        
        issues = quality_tree.add("Issues Found")
        issues.add(f"Critical: {qa.critical_issues}")
        issues.add(f"High: {qa.high_issues}")
        issues.add(f"Medium: {qa.medium_issues}")
        issues.add(f"Low: {qa.low_issues}")
        
        console.print(quality_tree)
        
        # Recommendations
        if qa.recommendations and verbose:
            console.print("\n[bold]💡 Recommendations:[/bold]")
            for rec in qa.recommendations:
                console.print(f"  • {rec}")
    
    # Profiling Metadata
    if profile.profiling_metadata and verbose:
        metadata = profile.profiling_metadata
        
        metadata_content = f"""
[bold]Strategy:[/bold] {metadata.profiling_strategy}
[bold]Execution Time:[/bold] {metadata.execution_time_seconds:.2f}s
[bold]Patterns Included:[/bold] {'✓' if metadata.include_patterns else '✗'}
[bold]Statistics Included:[/bold] {'✓' if metadata.include_statistical_analysis else '✗'}
[bold]Quality Assessment:[/bold] {'✓' if metadata.include_quality_assessment else '✗'}
"""
        
        if metadata.sample_size:
            metadata_content += f"[bold]Sample Size:[/bold] {metadata.sample_size:,}\n"
        
        if metadata.sample_percentage:
            metadata_content += f"[bold]Sample Percentage:[/bold] {metadata.sample_percentage:.1f}%\n"
        
        if metadata.memory_usage_mb:
            metadata_content += f"[bold]Memory Usage:[/bold] {metadata.memory_usage_mb:.2f} MB\n"
        
        console.print(Panel(metadata_content.strip(), title="Profiling Metadata", border_style="yellow"))
    
    # Error message if failed
    if profile.error_message:
        console.print(f"\n[red]❌ Error:[/red] {profile.error_message}")

def _convert_profile_to_dict(profile) -> Dict[str, Any]:
    """Convert profile entity to dictionary for JSON serialization."""
    result = {
        "profile_id": str(profile.profile_id.value),
        "dataset_id": str(profile.dataset_id.value),
        "status": profile.status.value,
        "source_type": profile.source_type,
        "source_connection": profile.source_connection,
        "source_query": profile.source_query,
        "created_at": profile.created_at.isoformat(),
        "started_at": profile.started_at.isoformat() if profile.started_at else None,
        "completed_at": profile.completed_at.isoformat() if profile.completed_at else None,
        "error_message": profile.error_message
    }
    
    if profile.schema_profile:
        result["schema_profile"] = {
            "table_name": profile.schema_profile.table_name,
            "total_rows": profile.schema_profile.total_rows,
            "total_columns": profile.schema_profile.total_columns,
            "estimated_size_bytes": profile.schema_profile.estimated_size_bytes,
            "compression_ratio": profile.schema_profile.compression_ratio,
            "primary_keys": profile.schema_profile.primary_keys,
            "foreign_keys": profile.schema_profile.foreign_keys,
            "unique_constraints": profile.schema_profile.unique_constraints,
            "check_constraints": profile.schema_profile.check_constraints,
            "columns": [
                {
                    "column_name": col.column_name,
                    "data_type": col.data_type.value,
                    "inferred_type": col.inferred_type.value if col.inferred_type else None,
                    "nullable": col.nullable,
                    "cardinality": col.cardinality.value,
                    "quality_score": col.quality_score,
                    "semantic_type": col.semantic_type,
                    "business_meaning": col.business_meaning,
                    "distribution": {
                        "unique_count": col.distribution.unique_count,
                        "null_count": col.distribution.null_count,
                        "total_count": col.distribution.total_count,
                        "completeness_ratio": col.distribution.completeness_ratio,
                        "top_values": col.distribution.top_values
                    },
                    "patterns": [
                        {
                            "pattern_type": p.pattern_type.value,
                            "regex": p.regex,
                            "frequency": p.frequency,
                            "percentage": p.percentage,
                            "examples": p.examples,
                            "confidence": p.confidence
                        }
                        for p in col.patterns
                    ],
                    "quality_issues": [
                        {
                            "issue_type": issue.issue_type.value,
                            "severity": issue.severity,
                            "description": issue.description,
                            "affected_rows": issue.affected_rows,
                            "affected_percentage": issue.affected_percentage,
                            "examples": issue.examples,
                            "suggested_action": issue.suggested_action
                        }
                        for issue in col.quality_issues
                    ]
                }
                for col in profile.schema_profile.columns
            ]
        }
    
    if profile.quality_assessment:
        result["quality_assessment"] = {
            "overall_score": profile.quality_assessment.overall_score,
            "completeness_score": profile.quality_assessment.completeness_score,
            "consistency_score": profile.quality_assessment.consistency_score,
            "accuracy_score": profile.quality_assessment.accuracy_score,
            "validity_score": profile.quality_assessment.validity_score,
            "uniqueness_score": profile.quality_assessment.uniqueness_score,
            "dimension_weights": profile.quality_assessment.dimension_weights,
            "critical_issues": profile.quality_assessment.critical_issues,
            "high_issues": profile.quality_assessment.high_issues,
            "medium_issues": profile.quality_assessment.medium_issues,
            "low_issues": profile.quality_assessment.low_issues,
            "recommendations": profile.quality_assessment.recommendations
        }
    
    if profile.profiling_metadata:
        result["profiling_metadata"] = {
            "profiling_strategy": profile.profiling_metadata.profiling_strategy,
            "sample_size": profile.profiling_metadata.sample_size,
            "sample_percentage": profile.profiling_metadata.sample_percentage,
            "execution_time_seconds": profile.profiling_metadata.execution_time_seconds,
            "memory_usage_mb": profile.profiling_metadata.memory_usage_mb,
            "include_patterns": profile.profiling_metadata.include_patterns,
            "include_statistical_analysis": profile.profiling_metadata.include_statistical_analysis,
            "include_quality_assessment": profile.profiling_metadata.include_quality_assessment
        }
    
    return result

if __name__ == "__main__":
    app()