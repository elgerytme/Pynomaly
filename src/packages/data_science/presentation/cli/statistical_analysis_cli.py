#!/usr/bin/env python3
"""Statistical Analysis CLI using Typer."""

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

# Import domain entities and use cases
from ...application.use_cases.execute_statistical_analysis import ExecuteStatisticalAnalysisUseCase
from ...domain.entities.statistical_analysis import StatisticalAnalysisId, DatasetId, UserId, AnalysisType
from ...infrastructure.adapters.in_memory_statistical_analysis_repository import InMemoryStatisticalAnalysisRepository

logger = structlog.get_logger(__name__)
console = Console()

# Create Typer app
app = typer.Typer(
    name="data-science",
    help="🧮 Statistical Analysis and Data Science Tools",
    add_completion=True,
    rich_markup_mode="rich"
)

# Dependency setup
def get_repository():
    """Get repository instance."""
    return InMemoryStatisticalAnalysisRepository()

def get_use_case(repository=None):
    """Get use case instance."""
    if repository is None:
        repository = get_repository()
    return ExecuteStatisticalAnalysisUseCase(repository)

def get_current_user():
    """Get current user (mock implementation)."""
    return UserId()

@app.command("analyze")
def create_analysis(
    data_file: Path = typer.Argument(..., help="Path to dataset file (CSV, JSON, Parquet)"),
    analysis_type: str = typer.Option(
        "descriptive_statistics", 
        "--type", "-t", 
        help="Analysis type: descriptive_statistics, correlation_analysis, hypothesis_testing"
    ),
    features: Optional[str] = typer.Option(
        None, 
        "--features", "-f", 
        help="Comma-separated list of feature columns"
    ),
    target: Optional[str] = typer.Option(
        None, 
        "--target", 
        help="Target column for supervised analysis"
    ),
    output: Optional[Path] = typer.Option(
        None, 
        "--output", "-o", 
        help="Output file for results (JSON)"
    ),
    confidence_level: float = typer.Option(
        0.95, 
        "--confidence", "-c", 
        help="Confidence level for statistical tests"
    ),
    save_results: bool = typer.Option(
        True, 
        "--save/--no-save", 
        help="Save analysis results"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable verbose output"
    )
):
    """Create and execute a statistical analysis."""
    try:
        console.print(f"[bold blue]🧮 Statistical Analysis[/bold blue]")
        console.print(f"Data file: {data_file}")
        console.print(f"Analysis type: {analysis_type}")
        
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
        
        # Parse feature columns
        if features:
            feature_columns = [col.strip() for col in features.split(',')]
            # Validate columns exist
            missing_cols = set(feature_columns) - set(data.columns)
            if missing_cols:
                console.print(f"[red]❌ Missing columns: {list(missing_cols)}[/red]")
                raise typer.Exit(1)
        else:
            # Use all numeric columns by default
            feature_columns = list(data.select_dtypes(include=['number']).columns)
            if not feature_columns:
                console.print("[red]❌ No numeric columns found for analysis[/red]")
                raise typer.Exit(1)
        
        # Validate target column if provided
        if target and target not in data.columns:
            console.print(f"[red]❌ Target column '{target}' not found[/red]")
            raise typer.Exit(1)
        
        console.print(f"Feature columns: {feature_columns}")
        if target:
            console.print(f"Target column: {target}")
        
        # Execute analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing analysis...", total=None)
            
            analysis_params = {
                "confidence_level": confidence_level,
                "save_results": save_results
            }
            
            # Create analysis type
            analysis_type_obj = AnalysisType(
                name=analysis_type,
                description=f"Statistical analysis of type: {analysis_type}",
                requires_target=target is not None
            )
            
            # Run analysis
            use_case = get_use_case()
            analysis = asyncio.run(
                use_case.execute(
                    dataset_id=DatasetId(value=UUID('12345678-1234-5678-9012-123456789012')),  # Mock dataset ID
                    user_id=get_current_user(),
                    analysis_type=analysis_type_obj,
                    data=data,
                    feature_columns=feature_columns,
                    analysis_params=analysis_params
                )
            )
            
            progress.update(task, description="Analysis completed")
        
        # Display results
        _display_analysis_results(analysis, verbose)
        
        # Save results if output specified
        if output:
            results = {
                "analysis_id": str(analysis.analysis_id.value),
                "analysis_type": analysis.analysis_type.name,
                "status": analysis.status,
                "feature_columns": analysis.feature_columns,
                "target_column": analysis.target_column,
                "insights": analysis.insights,
                "execution_time_seconds": analysis.execution_time_seconds,
                "created_at": analysis.created_at.isoformat()
            }
            
            if analysis.metrics:
                results["metrics"] = {
                    "descriptive_stats": analysis.metrics.descriptive_stats,
                    "correlation_matrix": analysis.metrics.correlation_matrix,
                    "outlier_scores": analysis.metrics.outlier_scores
                }
            
            if analysis.statistical_tests:
                results["statistical_tests"] = [
                    {
                        "test_name": test.test_name,
                        "statistic": test.statistic,
                        "p_value": test.p_value,
                        "critical_value": test.critical_value,
                        "confidence_level": test.confidence_level,
                        "interpretation": test.interpretation
                    }
                    for test in analysis.statistical_tests
                ]
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"[green]✅ Results saved to: {output}[/green]")
        
        console.print("[green]✅ Analysis completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command("list")
def list_analyses(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """List statistical analyses."""
    try:
        use_case = get_use_case()
        user_id = get_current_user()
        
        # Get analyses
        analyses = asyncio.run(use_case.get_analyses_by_user(user_id))
        
        # Filter by status if provided
        if status:
            analyses = [a for a in analyses if a.status == status]
        
        # Limit results
        analyses = analyses[:limit]
        
        if not analyses:
            console.print("[yellow]No analyses found[/yellow]")
            return
        
        # Display table
        table = Table(title="Statistical Analyses")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Features", style="blue")
        table.add_column("Created", style="dim")
        
        if verbose:
            table.add_column("Execution Time", style="magenta")
            table.add_column("Insights", style="white")
        
        for analysis in analyses:
            row = [
                str(analysis.analysis_id.value)[:8] + "...",
                analysis.analysis_type.name,
                analysis.status,
                str(len(analysis.feature_columns)),
                analysis.created_at.strftime("%Y-%m-%d %H:%M")
            ]
            
            if verbose:
                exec_time = f"{analysis.execution_time_seconds:.2f}s" if analysis.execution_time_seconds else "N/A"
                insights_count = len(analysis.insights) if analysis.insights else 0
                row.extend([exec_time, str(insights_count)])
            
            table.add_row(*row)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("get")
def get_analysis(
    analysis_id: str = typer.Argument(..., help="Analysis ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """Get details of a specific analysis."""
    try:
        use_case = get_use_case()
        
        # Get analysis
        analysis = asyncio.run(
            use_case.get_analysis_by_id(
                StatisticalAnalysisId(value=UUID(analysis_id))
            )
        )
        
        if not analysis:
            console.print(f"[red]❌ Analysis not found: {analysis_id}[/red]")
            raise typer.Exit(1)
        
        # Display analysis details
        _display_analysis_results(analysis, verbose)
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid analysis ID: {analysis_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("delete")
def delete_analysis(
    analysis_id: str = typer.Argument(..., help="Analysis ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Delete a statistical analysis."""
    try:
        if not force:
            confirmed = typer.confirm(f"Delete analysis {analysis_id}?")
            if not confirmed:
                console.print("Operation cancelled")
                return
        
        use_case = get_use_case()
        
        # Delete analysis
        asyncio.run(
            use_case.repository.delete(
                StatisticalAnalysisId(value=UUID(analysis_id))
            )
        )
        
        console.print(f"[green]✅ Analysis deleted: {analysis_id}[/green]")
        
    except ValueError as e:
        console.print(f"[red]❌ Invalid analysis ID: {analysis_id}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        raise typer.Exit(1)

@app.command("health")
def health_check():
    """Check the health of the statistical analysis service."""
    try:
        console.print("[blue]🔍 Checking service health...[/blue]")
        
        # Test repository connection
        repository = get_repository()
        
        # Basic health check
        health_status = {
            "service": "statistical-analysis",
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

def _display_analysis_results(analysis, verbose: bool = False):
    """Display analysis results in a formatted way."""
    
    # Basic information panel
    info_content = f"""
[bold]Analysis ID:[/bold] {analysis.analysis_id.value}
[bold]Type:[/bold] {analysis.analysis_type.name}
[bold]Status:[/bold] {analysis.status}
[bold]Feature Columns:[/bold] {', '.join(analysis.feature_columns)}
[bold]Target Column:[/bold] {analysis.target_column or 'None'}
[bold]Created:[/bold] {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    if analysis.execution_time_seconds:
        info_content += f"[bold]Execution Time:[/bold] {analysis.execution_time_seconds:.2f}s\n"
    
    console.print(Panel(info_content.strip(), title="Analysis Information", border_style="blue"))
    
    # Statistical tests
    if analysis.statistical_tests:
        test_table = Table(title="Statistical Tests")
        test_table.add_column("Test", style="cyan")
        test_table.add_column("Statistic", style="yellow")
        test_table.add_column("P-Value", style="green")
        test_table.add_column("Interpretation", style="white")
        
        for test in analysis.statistical_tests:
            test_table.add_row(
                test.test_name,
                f"{test.statistic:.4f}",
                f"{test.p_value:.4f}",
                test.interpretation
            )
        
        console.print(test_table)
    
    # Metrics
    if analysis.metrics and verbose:
        console.print("\n[bold]Descriptive Statistics:[/bold]")
        if analysis.metrics.descriptive_stats:
            for feature, stats in analysis.metrics.descriptive_stats.items():
                console.print(f"[cyan]{feature}:[/cyan] {stats}")
    
    # Insights
    if analysis.insights:
        console.print("\n[bold]🔍 Key Insights:[/bold]")
        for insight in analysis.insights:
            console.print(f"  • {insight}")
    
    # Error message if failed
    if analysis.error_message:
        console.print(f"\n[red]❌ Error:[/red] {analysis.error_message}")

if __name__ == "__main__":
    app()