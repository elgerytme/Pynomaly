"""Analytics and Reporting CLI Commands for comprehensive data insights and visualization."""

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

# Create the analytics CLI app
analytics_app = typer.Typer(
    name="analytics",
    help="Advanced analytics and reporting operations for data insights and visualization",
    rich_markup_mode="rich"
)


@analytics_app.command("dashboard")
def generate_dashboard(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for dashboard"
    ),
    dashboard_type: str = typer.Option(
        "comprehensive", "--type", "-t",
        help="Dashboard type: [comprehensive|executive|technical|quality|performance]"
    ),
    format: str = typer.Option(
        "html", "--format", "-f",
        help="Dashboard format: [html|pdf|interactive|jupyter]"
    ),
    theme: str = typer.Option(
        "default", "--theme", help="Dashboard theme: [default|dark|corporate|minimal]"
    ),
    include_filters: bool = typer.Option(
        True, "--filters/--no-filters", help="Include interactive filters"
    ),
    auto_refresh: bool = typer.Option(
        False, "--auto-refresh", help="Enable auto-refresh for real-time data"
    ),
    refresh_interval: int = typer.Option(
        300, "--refresh-interval", help="Auto-refresh interval in seconds"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Generate comprehensive analytical dashboards."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_dashboard"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import analytics packages
            from packages.analytics.application.services.dashboard_generator import DashboardGenerator
            from packages.analytics.application.services.visualization_engine import VisualizationEngine
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Initializing dashboard generator...")
            
            # Initialize services
            dashboard_generator = DashboardGenerator()
            viz_engine = VisualizationEngine()
            
            # Configure dashboard
            config = {
                "dashboard_type": dashboard_type,
                "format": format,
                "theme": theme,
                "include_filters": include_filters,
                "auto_refresh": auto_refresh,
                "refresh_interval": refresh_interval
            }
            
            progress.update(task, description="Generating dashboard components...")
            
            # Generate dashboard
            dashboard = dashboard_generator.create_dashboard(data_collection, config)
            
            progress.update(task, description="Rendering dashboard...")
            
            # Save dashboard
            if format == "html":
                dashboard_file = output_dir / "dashboard.html"
                dashboard_generator.export_html(dashboard, dashboard_file)
            elif format == "pdf":
                dashboard_file = output_dir / "dashboard.pdf"
                dashboard_generator.export_pdf(dashboard, dashboard_file)
            elif format == "interactive":
                dashboard_file = output_dir / "dashboard.html"
                dashboard_generator.export_interactive(dashboard, dashboard_file)
            elif format == "jupyter":
                dashboard_file = output_dir / "dashboard.ipynb"
                dashboard_generator.export_jupyter(dashboard, dashboard_file)
            
            # Generate supporting assets
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            dashboard_generator.export_assets(dashboard, assets_dir)
            
            progress.update(task, description="Dashboard generation complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required analytics packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during dashboard generation: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Dashboard generated successfully![/green]")
    console.print(f"Dashboard saved to: {dashboard_file}")
    
    # Display dashboard info
    if dashboard:
        components = dashboard.get("components", [])
        
        panel_content = f"""
[bold]Dashboard Overview[/bold]
• Type: {dashboard_type.title()}
• Format: {format.upper()}
• Components: {len(components)}
• Theme: {theme.title()}
• Interactive: {'Yes' if include_filters else 'No'}
        """
        
        console.print(Panel(panel_content, title="Dashboard Info", border_style="blue"))


@analytics_app.command("report")
def generate_report(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
    report_type: str = typer.Option(
        "comprehensive", "--type", "-t",
        help="Report type: [comprehensive|summary|detailed|executive|technical]"
    ),
    format: str = typer.Option(
        "pdf", "--format", "-f",
        help="Report format: [pdf|html|docx|markdown]"
    ),
    template: Optional[str] = typer.Option(
        None, "--template", help="Report template to use"
    ),
    include_recommendations: bool = typer.Option(
        True, "--recommendations/--no-recommendations", help="Include data recommendations"
    ),
    include_charts: bool = typer.Option(
        True, "--charts/--no-charts", help="Include charts and visualizations"
    ),
    language: str = typer.Option(
        "en", "--language", help="Report language: [en|es|fr|de|pt]"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Generate comprehensive analytical reports."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_report.{format}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import reporting packages
            from packages.analytics.application.services.report_generator import ReportGenerator
            from packages.analytics.application.services.insights_engine import InsightsEngine
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Analyzing data_collection...")
            
            # Initialize services
            report_generator = ReportGenerator()
            insights_engine = InsightsEngine()
            
            # Generate insights
            insights = insights_engine.generate_insights(data_collection)
            
            progress.update(task, description="Generating report...")
            
            # Configure report
            config = {
                "report_type": report_type,
                "format": format,
                "template": template,
                "include_recommendations": include_recommendations,
                "include_charts": include_charts,
                "language": language
            }
            
            # Generate report
            report = report_generator.create_report(data_collection, insights, config)
            
            progress.update(task, description="Exporting report...")
            
            # Export report
            if format == "pdf":
                report_generator.export_pdf(report, output_file)
            elif format == "html":
                report_generator.export_html(report, output_file)
            elif format == "docx":
                report_generator.export_docx(report, output_file)
            elif format == "markdown":
                report_generator.export_markdown(report, output_file)
            
            progress.update(task, description="Report generation complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during report generation: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Report generated successfully![/green]")
    console.print(f"Report saved to: {output_file}")


@analytics_app.command("insights")
def generate_insights(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for insights"
    ),
    insight_types: List[str] = typer.Option(
        ["trends", "outliers", "patterns"], "--type", "-t",
        help="Insight types: [trends|outliers|patterns|correlations|anomalies]"
    ),
    confidence_threshold: float = typer.Option(
        0.8, "--confidence", "-c", help="Minimum confidence threshold for insights"
    ),
    max_insights: int = typer.Option(
        50, "--max-insights", help="Maximum number of insights to generate"
    ),
    include_explanations: bool = typer.Option(
        True, "--explanations/--no-explanations", help="Include insight explanations"
    ),
    prioritize_actionable: bool = typer.Option(
        True, "--actionable/--all", help="Prioritize actionable insights"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Generate automated data insights and recommendations."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_insights.json"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import insights packages
            from packages.analytics.application.services.insights_engine import InsightsEngine
            from packages.analytics.application.services.pattern_analyzer import PatternAnalyzer
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Analyzing data patterns...")
            
            # Initialize services
            insights_engine = InsightsEngine()
            pattern_analyzer = PatternAnalyzer()
            
            # Configure insights generation
            config = {
                "insight_types": insight_types,
                "confidence_threshold": confidence_threshold,
                "max_insights": max_insights,
                "include_explanations": include_explanations,
                "prioritize_actionable": prioritize_actionable
            }
            
            progress.update(task, description="Generating insights...")
            
            # Generate insights
            insights = insights_engine.generate_comprehensive_insights(data_collection, config)
            
            progress.update(task, description="Ranking and filtering insights...")
            
            # Rank and filter insights
            ranked_insights = insights_engine.rank_insights(insights, config)
            
            progress.update(task, description="Saving insights...")
            
            # Save insights
            with open(output_file, 'w') as f:
                json.dump(ranked_insights, f, indent=2, default=str)
            
            progress.update(task, description="Insights generation complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during insights generation: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Insights generated successfully![/green]")
    console.print(f"Insights saved to: {output_file}")
    
    # Display top insights
    if ranked_insights and "insights" in ranked_insights:
        table = Table(title="Top Data Insights")
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="yellow")
        table.add_column("Confidence", style="green")
        table.add_column("Impact", style="red")
        
        for insight in ranked_insights["insights"][:10]:  # Show top 10
            table.add_row(
                insight.get("type", "Unknown"),
                insight.get("description", "")[:50] + "...",
                f"{insight.get('confidence', 0):.2f}",
                insight.get("impact", "Low")
            )
        
        console.print(table)


@analytics_app.command("visualize")
def create_visualizations(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for visualizations"
    ),
    chart_types: List[str] = typer.Option(
        ["auto"], "--type", "-t",
        help="Chart types: [auto|histogram|scatter|line|bar|heatmap|box|violin]"
    ),
    format: str = typer.Option(
        "png", "--format", "-f",
        help="Image format: [png|svg|pdf|html]"
    ),
    style: str = typer.Option(
        "default", "--style", help="Visualization style: [default|seaborn|ggplot|bmh]"
    ),
    color_palette: str = typer.Option(
        "viridis", "--palette", help="Color palette: [viridis|plasma|tab10|husl]"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Generate interactive visualizations"
    ),
    high_resolution: bool = typer.Option(
        True, "--hires/--standard", help="Generate high-resolution images"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Create comprehensive data visualizations."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import visualization packages
            from packages.analytics.application.services.visualization_engine import VisualizationEngine
            from packages.analytics.application.services.chart_generator import ChartGenerator
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Analyzing data for optimal visualizations...")
            
            # Initialize services
            viz_engine = VisualizationEngine()
            chart_generator = ChartGenerator()
            
            # Configure visualization
            config = {
                "chart_types": chart_types,
                "format": format,
                "style": style,
                "color_palette": color_palette,
                "interactive": interactive,
                "high_resolution": high_resolution,
                "output_dir": str(output_dir)
            }
            
            progress.update(task, description="Generating visualizations...")
            
            # Generate visualizations
            if "auto" in chart_types:
                visualizations = viz_engine.auto_generate_visualizations(data_collection, config)
            else:
                visualizations = chart_generator.create_custom_charts(data_collection, config)
            
            progress.update(task, description="Saving visualizations...")
            
            # Save visualizations
            viz_engine.save_visualizations(visualizations, output_dir)
            
            # Generate visualization index
            index_file = output_dir / "index.html"
            viz_engine.create_visualization_index(visualizations, index_file)
            
            progress.update(task, description="Visualization generation complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during visualization: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Visualizations created successfully![/green]")
    console.print(f"Visualizations saved to: {output_dir}")
    console.print(f"View index: {output_dir / 'index.html'}")


@analytics_app.command("compare")
def compare_datasets(
    dataset1: Path = typer.Argument(..., help="First data_collection file"),
    dataset2: Path = typer.Argument(..., help="Second data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for comparison results"
    ),
    comparison_type: str = typer.Option(
        "comprehensive", "--type", "-t",
        help="Comparison type: [comprehensive|statistical|structural|quality]"
    ),
    significance_level: float = typer.Option(
        0.05, "--significance", help="Statistical significance level"
    ),
    generate_report: bool = typer.Option(
        True, "--report/--no-report", help="Generate comparison report"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Compare two datasets and identify differences."""
    
    if not dataset1.exists():
        console.print(f"[red]Error: DataCollection file {dataset1} does not exist[/red]")
        raise typer.Exit(1)
    
    if not dataset2.exists():
        console.print(f"[red]Error: DataCollection file {dataset2} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = dataset1.parent / f"comparison_{dataset1.stem}_{dataset2.stem}.json"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading datasets...", total=None)
        
        try:
            # Import comparison packages
            from packages.analytics.application.services.dataset_comparison_service import DatasetComparisonService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load datasets
            adapter = DataSourceAdapter()
            data1 = adapter.load_data_collection(str(dataset1))
            data2 = adapter.load_data_collection(str(dataset2))
            
            progress.update(task, description="Performing comparison analysis...")
            
            # Initialize comparison service
            comparison_service = DatasetComparisonService()
            
            # Configure comparison
            config = {
                "comparison_type": comparison_type,
                "significance_level": significance_level,
                "generate_visualizations": generate_report
            }
            
            # Perform comparison
            comparison_results = comparison_service.compare_data_collections(
                data1, data2, config
            )
            
            progress.update(task, description="Saving comparison results...")
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(comparison_results, f, indent=2, default=str)
            
            # Generate report if requested
            if generate_report:
                report_file = output_file.parent / f"{output_file.stem}_report.html"
                comparison_service.generate_comparison_report(
                    comparison_results, report_file
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
    
    console.print("\n[green]✓ DataCollection comparison completed successfully![/green]")
    console.print(f"Results saved to: {output_file}")
    
    if generate_report:
        console.print(f"Report generated: {report_file}")
    
    # Display key differences
    if comparison_results and "differences" in comparison_results:
        differences = comparison_results["differences"]
        
        table = Table(title="Key Differences")
        table.add_column("Category", style="cyan")
        table.add_column("Difference", style="yellow")
        table.add_column("Significance", style="red")
        
        for diff in differences[:10]:  # Show first 10 differences
            table.add_row(
                diff.get("category", "Unknown"),
                diff.get("description", ""),
                diff.get("significance", "Low")
            )
        
        console.print(table)


@analytics_app.command("benchmark")
def benchmark_performance(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    operations: List[str] = typer.Option(
        ["all"], "--operation", "-op",
        help="Operations to benchmark: [all|load|profile|analyze|transform|export]"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for benchmark results"
    ),
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Number of benchmark iterations"
    ),
    warmup_runs: int = typer.Option(
        1, "--warmup", help="Number of warmup runs"
    ),
    memory_profiling: bool = typer.Option(
        True, "--memory/--no-memory", help="Include memory profiling"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Benchmark data processing performance."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_benchmark.json"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing benchmark...", total=None)
        
        try:
            # Import benchmarking packages
            from packages.analytics.application.services.performance_benchmarker import PerformanceBenchmarker
            
            # Initialize benchmarker
            benchmarker = PerformanceBenchmarker()
            
            # Configure benchmark
            config = {
                "operations": operations,
                "iterations": iterations,
                "warmup_runs": warmup_runs,
                "memory_profiling": memory_profiling,
                "input_file": str(input_file)
            }
            
            progress.update(task, description="Running benchmark suite...")
            
            # Run benchmark
            benchmark_results = benchmarker.run_benchmark_suite(config)
            
            progress.update(task, description="Analyzing results...")
            
            # Analyze results
            analysis = benchmarker.analyze_benchmark_results(benchmark_results)
            
            progress.update(task, description="Saving benchmark results...")
            
            # Save results
            final_results = {
                "benchmark_results": benchmark_results,
                "analysis": analysis,
                "config": config
            }
            
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            progress.update(task, description="Benchmark complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during benchmarking: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Benchmark completed successfully![/green]")
    console.print(f"Results saved to: {output_file}")
    
    # Display performance summary
    if benchmark_results:
        table = Table(title="Performance Summary")
        table.add_column("Operation", style="cyan")
        table.add_column("Avg Time", style="green")
        table.add_column("Memory Peak", style="yellow")
        table.add_column("Throughput", style="blue")
        
        for op_name, op_results in benchmark_results.items():
            if isinstance(op_results, dict):
                avg_time = op_results.get("avg_time", "N/A")
                memory_peak = op_results.get("memory_peak", "N/A")
                throughput = op_results.get("throughput", "N/A")
                
                table.add_row(op_name, str(avg_time), str(memory_peak), str(throughput))
        
        console.print(table)


if __name__ == "__main__":
    analytics_app()