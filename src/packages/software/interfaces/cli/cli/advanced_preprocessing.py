"""Advanced data preprocessing CLI commands with data_transformation integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List
from uuid import UUID

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

# Import existing CLI infrastructure
from monorepo.presentation.cli.container import get_cli_container

# Import data transformation components
try:
    from data_transformation.application.use_cases.data_pipeline import DataPipelineUseCase
    from data_transformation.domain.value_objects.pipeline_config import (
        PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
    )
    from data_transformation.infrastructure.adapters.data_source_adapter import DataSourceAdapter
    # Services not available in interfaces package - using basic implementations
    EnhancedDataPreprocessingService = None
    DATA_TRANSFORMATION_AVAILABLE = True
except ImportError:
    DATA_TRANSFORMATION_AVAILABLE = False

app = typer.Typer(name="advanced-preprocessing")
console = Console()


def check_data_transformation_availability():
    """Check if data transformation features are available."""
    if not DATA_TRANSFORMATION_AVAILABLE:
        console.print(
            "[red]Error: Advanced data transformation features are not available.[/red]\n"
            "Please ensure the data_transformation package is installed.",
            style="bold red"
        )
        raise typer.Exit(1)


@app.command("transform")
def transform_dataset(
    input_path: Path = typer.Argument(..., help="Path to input dataset"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for transformed dataset"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to transformation config file"),
    cleaning_strategy: str = typer.Option("auto", "--cleaning", help="Data cleaning strategy"),
    scaling_method: str = typer.Option("robust", "--scaling", help="Feature scaling method"),
    encoding_strategy: str = typer.Option("onehot", "--encoding", help="Categorical encoding strategy"),
    feature_engineering: bool = typer.Option(True, "--feature-eng/--no-feature-eng", help="Enable feature engineering"),
    validation: bool = typer.Option(True, "--validate/--no-validate", help="Enable data validation"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Enable parallel processing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Transform dataset using advanced preprocessing pipeline.
    
    Apply sophisticated data transformations optimized for anomaly detection
    workflows using the integrated data_transformation package.
    
    Examples:
        pynomaly advanced-preprocessing transform data.csv --output clean_data.csv
        pynomaly advanced-preprocessing transform data.csv --config config.yml --verbose
    """
    check_data_transformation_availability()
    
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found.[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Load configuration
            task = progress.add_task("Loading configuration...", total=None)
            if config_file and config_file.exists():
                with open(config_file) as f:
                    config_data = json.load(f)
                    config = PipelineConfig(**config_data)
            else:
                # Create config from CLI arguments
                config = PipelineConfig(
                    source_type=_detect_source_type(input_path),
                    cleaning_strategy=CleaningStrategy(cleaning_strategy),
                    scaling_method=ScalingMethod(scaling_method),
                    encoding_strategy=EncodingStrategy(encoding_strategy),
                    feature_engineering=feature_engineering,
                    validation_enabled=validation,
                    parallel_processing=parallel
                )
            
            if verbose:
                console.print(f"[blue]Configuration:[/blue]")
                console.print(f"  Cleaning: {config.cleaning_strategy.value}")
                console.print(f"  Scaling: {config.scaling_method.value}")
                console.print(f"  Encoding: {config.encoding_strategy.value}")
                console.print(f"  Feature Engineering: {config.feature_engineering}")
            
            # Execute transformation
            progress.update(task, description="Applying transformations...")
            pipeline = DataPipelineUseCase(config)
            result = pipeline.execute(str(input_path))
            
            if result.success:
                progress.update(task, description="Saving results...")
                
                # Determine output path
                if output_path is None:
                    output_path = input_path.parent / f"{input_path.stem}_transformed{input_path.suffix}"
                
                # Save transformed data
                result.data.to_csv(output_path, index=False)
                
                progress.update(task, description="Complete!", completed=True)
                
                # Display results
                console.print(f"\n[green]✓ Transformation completed successfully![/green]")
                console.print(f"[blue]Input:[/blue] {input_path}")
                console.print(f"[blue]Output:[/blue] {output_path}")
                console.print(f"[blue]Execution time:[/blue] {result.execution_time:.2f} seconds")
                
                # Show transformation summary
                _show_transformation_summary(result, verbose)
                
            else:
                console.print(f"[red]✗ Transformation failed: {result.error_message}[/red]")
                raise typer.Exit(1)
                
    except Exception as e:
        console.print(f"[red]Error during transformation: {e}[/red]")
        raise typer.Exit(1)


@app.command("analyze")
def analyze_dataset(
    input_path: Path = typer.Argument(..., help="Path to dataset for analysis"),
    output_format: str = typer.Option("table", "--format", help="Output format: table, json"),
    save_report: Optional[Path] = typer.Option(None, "--save", help="Save analysis report to file"),
    anomaly_type: str = typer.Option("unsupervised", "--anomaly-type", help="Anomaly detection type"),
) -> None:
    """Analyze dataset and provide intelligent preprocessing recommendations.
    
    Performs comprehensive data quality assessment and generates recommendations
    for optimal preprocessing configuration.
    """
    check_data_transformation_availability()
    
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found.[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Analyzing dataset...", total=None)
            
            # Load data
            df = pd.read_csv(input_path)
            
            # Get recommendations
            if EnhancedDataPreprocessingService is None:
                preprocessing_service = None
            else:
                if EnhancedDataPreprocessingService is None:
            preprocessing_service = None
        else:
            preprocessing_service = EnhancedDataPreprocessingService()
            recommendations = await preprocessing_service.get_preprocessing_recommendations(
                df, anomaly_type
            )
            
            progress.update(task, description="Complete!", completed=True)
        
        # Display results based on format
        if output_format == "json":
            if save_report:
                with open(save_report, 'w') as f:
                    json.dump(recommendations, f, indent=2)
                console.print(f"[green]Analysis saved to {save_report}[/green]")
            else:
                console.print(json.dumps(recommendations, indent=2))
        else:
            _display_analysis_table(recommendations, input_path)
            
            if save_report:
                with open(save_report, 'w') as f:
                    json.dump(recommendations, f, indent=2)
                console.print(f"\n[blue]Report saved to {save_report}[/blue]")
                
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        raise typer.Exit(1)


@app.command("quality-check")
def quality_check(
    input_path: Path = typer.Argument(..., help="Path to dataset for quality check"),
    threshold: float = typer.Option(0.7, "--threshold", help="Quality score threshold (0-1)"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed quality metrics"),
) -> None:
    """Perform comprehensive data quality assessment.
    
    Analyzes data quality and provides a detailed report with scores and recommendations.
    """
    check_data_transformation_availability()
    
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found.[/red]")
        raise typer.Exit(1)
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        
        # Perform quality assessment
        if EnhancedDataPreprocessingService is None:
            preprocessing_service = None
        else:
            preprocessing_service = EnhancedDataPreprocessingService()
        quality_report = await preprocessing_service._assess_data_quality(df, input_path.stem)
        
        # Display quality report
        _display_quality_report(quality_report, threshold, detailed)
        
        # Exit with appropriate code based on quality
        if quality_report.quality_score < threshold:
            console.print(f"\n[yellow]Warning: Quality score {quality_report.quality_score:.2f} is below threshold {threshold}[/yellow]")
            raise typer.Exit(1)
        else:
            console.print(f"\n[green]✓ Quality score {quality_report.quality_score:.2f} meets threshold {threshold}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error during quality check: {e}[/red]")
        raise typer.Exit(1)


@app.command("optimize")
def optimize_for_algorithm(
    input_path: Path = typer.Argument(..., help="Path to input dataset"),
    algorithm: str = typer.Argument(..., help="Target algorithm (isolation_forest, one_class_svm, etc.)"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for optimized dataset"),
    target_column: Optional[str] = typer.Option(None, "--target", help="Target column for supervised algorithms"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Optimize dataset preprocessing for a specific anomaly detection algorithm.
    
    Applies algorithm-specific preprocessing optimizations to improve detection performance.
    
    Supported algorithms:
    - isolation_forest: Optimized for Isolation Forest
    - one_class_svm: Optimized for One-Class SVM  
    - local_outlier_factor: Optimized for Local Outlier Factor
    - autoencoder: Optimized for Autoencoder-based detection
    """
    check_data_transformation_availability()
    
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found.[/red]")
        raise typer.Exit(1)
    
    supported_algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor", "autoencoder"]
    if algorithm not in supported_algorithms:
        console.print(f"[red]Error: Unsupported algorithm '{algorithm}'[/red]")
        console.print(f"Supported algorithms: {', '.join(supported_algorithms)}")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Optimizing for {algorithm}...", total=None)
            
            # Load data
            df = pd.read_csv(input_path)
            
            # Apply algorithm-specific optimization
            if EnhancedDataPreprocessingService is None:
                preprocessing_service = None
            else:
                if EnhancedDataPreprocessingService is None:
            preprocessing_service = None
        else:
            preprocessing_service = EnhancedDataPreprocessingService()
            optimized_data = await preprocessing_service.optimize_for_algorithm(
                df, algorithm, target_column
            )
            
            # Save optimized data
            if output_path is None:
                output_path = input_path.parent / f"{input_path.stem}_optimized_{algorithm}{input_path.suffix}"
            
            optimized_data.to_csv(output_path, index=False)
            
            progress.update(task, description="Complete!", completed=True)
        
        console.print(f"\n[green]✓ Dataset optimized for {algorithm}![/green]")
        console.print(f"[blue]Input:[/blue] {input_path}")
        console.print(f"[blue]Output:[/blue] {output_path}")
        console.print(f"[blue]Algorithm:[/blue] {algorithm}")
        if target_column:
            console.print(f"[blue]Target column:[/blue] {target_column}")
        
        if verbose:
            console.print(f"\n[blue]Dataset summary:[/blue]")
            console.print(f"  Rows: {len(optimized_data)}")
            console.print(f"  Columns: {len(optimized_data.columns)}")
            console.print(f"  Numeric features: {len(optimized_data.select_dtypes(include=['number']).columns)}")
            console.print(f"  Categorical features: {len(optimized_data.select_dtypes(include=['object']).columns)}")
            
    except Exception as e:
        console.print(f"[red]Error during optimization: {e}[/red]")
        raise typer.Exit(1)


def _detect_source_type(file_path: Path) -> SourceType:
    """Detect source type from file extension."""
    extension = file_path.suffix.lower()
    mapping = {
        '.csv': SourceType.CSV,
        '.json': SourceType.JSON,
        '.parquet': SourceType.PARQUET,
        '.xlsx': SourceType.EXCEL,
        '.xls': SourceType.EXCEL
    }
    return mapping.get(extension, SourceType.CSV)


def _show_transformation_summary(result, verbose: bool) -> None:
    """Display transformation summary."""
    table = Table(title="Transformation Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Rows processed", str(len(result.data)))
    table.add_row("Columns", str(len(result.data.columns)))
    table.add_row("Steps executed", str(len(result.steps_executed)))
    table.add_row("Execution time", f"{result.execution_time:.2f}s")
    
    if verbose and result.steps_executed:
        steps_text = ", ".join([step.step_type for step in result.steps_executed])
        table.add_row("Processing steps", steps_text)
    
    console.print(table)


def _display_analysis_table(recommendations: dict, input_path: Path) -> None:
    """Display analysis results in table format."""
    console.print(f"\n[bold blue]Analysis Report for {input_path.name}[/bold blue]")
    
    # Basic statistics
    stats = recommendations.get("basic_stats", {})
    stats_table = Table(title="Dataset Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    stats_table.add_row("Rows", str(stats.get("rows", "N/A")))
    stats_table.add_row("Columns", str(stats.get("columns", "N/A")))
    stats_table.add_row("Numeric columns", str(stats.get("numeric_columns", "N/A")))
    stats_table.add_row("Categorical columns", str(stats.get("categorical_columns", "N/A")))
    stats_table.add_row("Missing values", str(stats.get("missing_values", "N/A")))
    
    console.print(stats_table)
    
    # Quality score
    if "quality_score" in recommendations:
        score = recommendations["quality_score"]
        color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
        console.print(f"\n[bold {color}]Quality Score: {score:.2f}[/bold {color}]")
    
    # Issues and recommendations
    if recommendations.get("data_quality_issues"):
        console.print("\n[bold red]Data Quality Issues:[/bold red]")
        for issue in recommendations["data_quality_issues"]:
            console.print(f"  • {issue}")
    
    if recommendations.get("preprocessing_steps"):
        console.print("\n[bold yellow]Recommended Preprocessing Steps:[/bold yellow]")
        for step in recommendations["preprocessing_steps"]:
            console.print(f"  • {step}")
    
    if recommendations.get("optimization_suggestions"):
        console.print("\n[bold green]Optimization Suggestions:[/bold green]")
        for suggestion in recommendations["optimization_suggestions"]:
            console.print(f"  • {suggestion}")


def _display_quality_report(report, threshold: float, detailed: bool) -> None:
    """Display quality assessment report."""
    # Overall assessment
    assessment = report.get_overall_assessment()
    score = report.quality_score
    
    color = "green" if score >= threshold else "red"
    console.print(f"\n[bold {color}]Overall Assessment: {assessment} (Score: {score:.2f})[/bold {color}]")
    
    if detailed:
        # Detailed metrics table
        metrics_table = Table(title="Quality Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")
        metrics_table.add_column("Status", style="green")
        
        metrics_table.add_row(
            "Missing Values", 
            f"{report.missing_values_ratio:.2%}",
            "✓" if report.missing_values_ratio < 0.1 else "⚠"
        )
        metrics_table.add_row(
            "Duplicates", 
            f"{report.duplicate_rows_ratio:.2%}",
            "✓" if report.duplicate_rows_ratio < 0.05 else "⚠"
        )
        metrics_table.add_row(
            "Outliers", 
            f"{report.outlier_ratio:.2%}",
            "✓" if report.outlier_ratio < 0.1 else "⚠"
        )
        metrics_table.add_row(
            "Sparsity", 
            f"{report.sparsity_ratio:.2%}",
            "✓" if report.sparsity_ratio < 0.5 else "⚠"
        )
        
        console.print(metrics_table)
    
    # Issues and recommendations
    if report.issues:
        console.print("\n[bold red]Issues Found:[/bold red]")
        for issue in report.issues:
            console.print(f"  • {issue}")
    
    if report.recommendations:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for rec in report.recommendations:
            console.print(f"  • {rec}")


if __name__ == "__main__":
    app()