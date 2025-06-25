"""CLI commands for performance testing and benchmarking."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.tree import Tree
from rich.layout import Layout

from pynomaly.application.services.performance_testing_service import (
    PerformanceTestingService, BenchmarkSuite, StressTestConfig
)
from pynomaly.infrastructure.config.container import Container


console = Console()


@click.group(name="performance")
def performance_commands():
    """Performance testing and benchmarking commands."""
    pass


@performance_commands.command()
@click.option("--suite", type=click.Choice([
    "quick", "comprehensive", "scalability", "custom"
]), default="quick", help="Benchmark suite to run")
@click.option("--algorithms", multiple=True, help="Specific algorithms to benchmark")
@click.option("--output-dir", help="Output directory for results")
@click.option("--iterations", type=int, default=3, help="Number of benchmark iterations")
@click.option("--timeout", type=int, default=300, help="Timeout per test in seconds")
@click.option("--export-format", type=click.Choice([
    "json", "csv", "html", "excel"
]), multiple=True, default=["json"], help="Export formats for results")
def benchmark(
    suite: str,
    algorithms: List[str],
    output_dir: Optional[str],
    iterations: int,
    timeout: int,
    export_format: List[str]
):
    """Run comprehensive algorithm benchmarking suite."""
    
    async def run_benchmark():
        container = Container()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Initialize performance testing service
            task1 = progress.add_task("Initializing performance testing service...", total=None)
            storage_path = Path(output_dir) if output_dir else Path("./performance_results")
            storage_path.mkdir(parents=True, exist_ok=True)
            
            perf_service = PerformanceTestingService(
                storage_path=storage_path,
                cache_results=True,
                enable_profiling=True
            )
            progress.update(task1, completed=True)
            
            # Load detectors
            task2 = progress.add_task("Loading algorithm detectors...", total=None)
            detector_service = container.detector_service()
            
            # Get available detectors
            available_algorithms = [
                "IsolationForest", "LocalOutlierFactor", "OneClassSVM",
                "EllipticEnvelope"  # Add more as available
            ]
            
            if algorithms:
                selected_algorithms = [alg for alg in algorithms if alg in available_algorithms]
            else:
                selected_algorithms = available_algorithms
            
            detectors = {}
            for alg_name in selected_algorithms:
                try:
                    detector = await detector_service.create_detector(
                        name=f"benchmark_{alg_name}",
                        algorithm=alg_name,
                        parameters={}
                    )
                    detectors[alg_name] = detector
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {alg_name}: {e}[/yellow]")
            
            progress.update(task2, completed=True)
            
            if not detectors:
                console.print("[red]Error: No detectors available for benchmarking[/red]")
                return
            
            # Customize suite if needed
            if suite == "custom":
                # Update suite configuration
                if suite in perf_service.benchmark_suites:
                    custom_suite = perf_service.benchmark_suites[suite]
                    custom_suite.algorithms = list(detectors.keys())
                    custom_suite.iterations = iterations
                    custom_suite.timeout_seconds = timeout
            
            # Run benchmark suite
            task3 = progress.add_task(f"Running {suite} benchmark suite...", total=100)
            
            try:
                results = await perf_service.run_benchmark_suite(
                    suite_name=suite,
                    detectors=detectors
                )
                
                progress.update(task3, completed=100)
                
                # Display results summary
                _display_benchmark_summary(results)
                
                # Export results
                for fmt in export_format:
                    task4 = progress.add_task(f"Exporting results as {fmt}...", total=None)
                    
                    output_file = storage_path / f"benchmark_{suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
                    
                    if fmt == "json":
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2, default=str)
                    elif fmt == "csv":
                        _export_benchmark_csv(results, output_file)
                    elif fmt == "html":
                        _export_benchmark_html(results, output_file)
                    
                    progress.update(task4, completed=True)
                    console.print(f"[green]Results exported to {output_file}[/green]")
                
            except Exception as e:
                console.print(f"[red]Benchmark failed: {e}[/red]")
                return
    
    asyncio.run(run_benchmark())


@performance_commands.command()
@click.option("--algorithm", required=True, help="Algorithm to analyze")
@click.option("--min-size", type=int, default=1000, help="Minimum dataset size")
@click.option("--max-size", type=int, default=100000, help="Maximum dataset size")
@click.option("--min-features", type=int, default=10, help="Minimum feature count")
@click.option("--max-features", type=int, default=200, help="Maximum feature count")
@click.option("--steps", type=int, default=10, help="Number of test points")
@click.option("--output-file", help="Output file for scalability analysis")
def scalability(
    algorithm: str,
    min_size: int,
    max_size: int,
    min_features: int,
    max_features: int,
    steps: int,
    output_file: Optional[str]
):
    """Run detailed scalability analysis for an algorithm."""
    
    async def run_scalability():
        container = Container()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Initialize services
            task1 = progress.add_task("Initializing scalability analysis...", total=None)
            
            storage_path = Path("./performance_results")
            perf_service = PerformanceTestingService(storage_path)
            detector_service = container.detector_service()
            
            progress.update(task1, completed=True)
            
            # Create detector
            task2 = progress.add_task(f"Loading {algorithm} detector...", total=None)
            
            try:
                detector = await detector_service.create_detector(
                    name=f"scalability_{algorithm}",
                    algorithm=algorithm,
                    parameters={}
                )
            except Exception as e:
                console.print(f"[red]Error loading detector: {e}[/red]")
                return
            
            progress.update(task2, completed=True)
            
            # Run scalability analysis
            task3 = progress.add_task("Running scalability analysis...", total=100)
            
            try:
                results = await perf_service.run_scalability_analysis(
                    detector=detector,
                    algorithm_name=algorithm,
                    size_range=(min_size, max_size),
                    feature_range=(min_features, max_features),
                    steps=steps
                )
                
                progress.update(task3, completed=100)
                
                # Display results
                _display_scalability_results(results)
                
                # Save results
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(f"[green]Scalability analysis saved to {output_file}[/green]")
                
            except Exception as e:
                console.print(f"[red]Scalability analysis failed: {e}[/red]")
                return
    
    asyncio.run(run_scalability())


@performance_commands.command()
@click.option("--algorithm", required=True, help="Algorithm to stress test")
@click.option("--concurrent-requests", type=int, default=10, help="Number of concurrent requests")
@click.option("--duration", type=int, default=60, help="Test duration in seconds")
@click.option("--memory-pressure", type=int, default=500, help="Memory pressure in MB")
@click.option("--cpu-stress", type=int, default=1000, help="CPU intensive operations")
@click.option("--endurance-hours", type=int, default=0, help="Endurance test duration in hours")
@click.option("--output-file", help="Output file for stress test results")
def stress_test(
    algorithm: str,
    concurrent_requests: int,
    duration: int,
    memory_pressure: int,
    cpu_stress: int,
    endurance_hours: int,
    output_file: Optional[str]
):
    """Run comprehensive stress testing for an algorithm."""
    
    async def run_stress_test():
        container = Container()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Initialize services
            task1 = progress.add_task("Initializing stress testing...", total=None)
            
            storage_path = Path("./performance_results")
            perf_service = PerformanceTestingService(storage_path)
            detector_service = container.detector_service()
            
            # Create stress test configuration
            stress_config = StressTestConfig(
                concurrent_requests=concurrent_requests,
                request_duration=duration,
                memory_pressure_mb=memory_pressure,
                cpu_intensive_operations=cpu_stress,
                endurance_duration_hours=endurance_hours
            )
            
            progress.update(task1, completed=True)
            
            # Create detector
            task2 = progress.add_task(f"Loading {algorithm} detector...", total=None)
            
            try:
                detector = await detector_service.create_detector(
                    name=f"stress_{algorithm}",
                    algorithm=algorithm,
                    parameters={}
                )
            except Exception as e:
                console.print(f"[red]Error loading detector: {e}[/red]")
                return
            
            progress.update(task2, completed=True)
            
            # Run stress test
            task3 = progress.add_task("Running stress tests...", total=100)
            
            try:
                results = await perf_service.run_stress_test(
                    detector=detector,
                    algorithm_name=algorithm,
                    config=stress_config
                )
                
                progress.update(task3, completed=100)
                
                # Display results
                _display_stress_test_results(results)
                
                # Save results
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(f"[green]Stress test results saved to {output_file}[/green]")
                
            except Exception as e:
                console.print(f"[red]Stress test failed: {e}[/red]")
                return
    
    asyncio.run(run_stress_test())


@performance_commands.command()
@click.option("--algorithms", multiple=True, help="Algorithms to compare")
@click.option("--datasets", multiple=True, help="Datasets to use for comparison")
@click.option("--metrics", multiple=True, help="Metrics to compare")
@click.option("--output-file", help="Output file for comparison results")
def compare(
    algorithms: List[str],
    datasets: List[str],
    metrics: List[str],
    output_file: Optional[str]
):
    """Compare multiple algorithms across different datasets and metrics."""
    
    async def run_comparison():
        container = Container()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            # Initialize services
            task1 = progress.add_task("Initializing algorithm comparison...", total=None)
            
            storage_path = Path("./performance_results")
            perf_service = PerformanceTestingService(storage_path)
            detector_service = container.detector_service()
            dataset_service = container.dataset_service()
            
            progress.update(task1, completed=True)
            
            # Load algorithms
            task2 = progress.add_task("Loading detectors...", total=None)
            
            if not algorithms:
                algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
            
            detectors = {}
            for alg_name in algorithms:
                try:
                    detector = await detector_service.create_detector(
                        name=f"compare_{alg_name}",
                        algorithm=alg_name,
                        parameters={}
                    )
                    detectors[alg_name] = detector
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {alg_name}: {e}[/yellow]")
            
            progress.update(task2, completed=True)
            
            # Load datasets
            task3 = progress.add_task("Loading datasets...", total=None)
            
            test_datasets = []
            if datasets:
                for dataset_name in datasets:
                    try:
                        dataset = await dataset_service.get_dataset(dataset_name)
                        test_datasets.append(dataset)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not load dataset {dataset_name}: {e}[/yellow]")
            
            # Generate synthetic datasets if none provided
            if not test_datasets:
                for size in [1000, 5000]:
                    dataset = await perf_service._generate_synthetic_dataset(
                        n_samples=size, n_features=20, contamination=0.1
                    )
                    test_datasets.append(dataset)
            
            progress.update(task3, completed=True)
            
            # Run comparison
            task4 = progress.add_task("Running algorithm comparison...", total=100)
            
            try:
                results = await perf_service.compare_algorithms(
                    detectors=detectors,
                    datasets=test_datasets,
                    metrics=list(metrics) if metrics else None
                )
                
                progress.update(task4, completed=100)
                
                # Display results
                _display_comparison_results(results)
                
                # Save results
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(f"[green]Comparison results saved to {output_file}[/green]")
                
            except Exception as e:
                console.print(f"[red]Algorithm comparison failed: {e}[/red]")
                return
    
    asyncio.run(run_comparison())


@performance_commands.command()
@click.option("--results-dir", help="Directory containing benchmark results")
@click.option("--format", "report_format", type=click.Choice([
    "console", "html", "pdf"
]), default="console", help="Report format")
@click.option("--output-file", help="Output file for report")
def report(
    results_dir: Optional[str],
    report_format: str,
    output_file: Optional[str]
):
    """Generate comprehensive performance analysis report."""
    
    # Load results
    if not results_dir:
        results_dir = "./performance_results"
    
    results_path = Path(results_dir)
    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        return
    
    # Find all result files
    result_files = list(results_path.glob("benchmark_*.json"))
    
    if not result_files:
        console.print(f"[yellow]No benchmark results found in {results_dir}[/yellow]")
        return
    
    console.print(f"[green]Found {len(result_files)} benchmark result files[/green]")
    
    # Generate report based on format
    if report_format == "console":
        _generate_console_report(result_files)
    elif report_format == "html":
        _generate_html_report(result_files, output_file or "performance_report.html")
    elif report_format == "pdf":
        _generate_pdf_report(result_files, output_file or "performance_report.pdf")


@performance_commands.command()
def monitor():
    """Start real-time performance monitoring dashboard."""
    
    console.print(Panel(
        "[bold blue]Performance Monitoring Dashboard[/bold blue]\n\n"
        "Real-time monitoring of system resources during anomaly detection.\n"
        "Press Ctrl+C to stop monitoring.",
        title="Performance Monitor"
    ))
    
    # Implementation for real-time monitoring would go here
    console.print("[yellow]Real-time monitoring not yet implemented[/yellow]")


# Helper functions for display and export

def _display_benchmark_summary(results: dict):
    """Display benchmark results summary."""
    summary = results.get("summary", {})
    
    console.print(Panel(
        f"[bold blue]Benchmark Suite Results[/bold blue]\n"
        f"Suite: {results.get('suite_name', 'Unknown')}\n"
        f"Total Tests: {summary.get('total_tests', 0)}\n"
        f"Successful Tests: {summary.get('successful_tests', 0)}\n"
        f"Average ROC AUC: {summary.get('avg_roc_auc', 0):.3f}\n"
        f"Average Training Time: {summary.get('avg_training_time', 0):.2f}s",
        title="Benchmark Summary"
    ))
    
    # Results table
    if "results" in results:
        table = Table(title="Algorithm Performance Results")
        table.add_column("Algorithm", style="cyan")
        table.add_column("Dataset", style="green")
        table.add_column("ROC AUC", style="yellow")
        table.add_column("Training Time", style="blue")
        table.add_column("Memory (MB)", style="magenta")
        table.add_column("Throughput", style="red")
        
        for result in results["results"][:10]:  # Show top 10
            perf = result.get("performance", {})
            table.add_row(
                result.get("algorithm", "Unknown"),
                result.get("dataset", "Unknown"),
                f"{perf.get('roc_auc', 0):.3f}",
                f"{perf.get('training_time', 0):.2f}s",
                f"{perf.get('peak_memory_mb', 0):.1f}",
                f"{perf.get('throughput', 0):.1f}/s"
            )
        
        console.print(table)


def _display_scalability_results(results: dict):
    """Display scalability analysis results."""
    console.print(Panel(
        f"[bold blue]Scalability Analysis Results[/bold blue]\n"
        f"Algorithm: {results.get('algorithm', 'Unknown')}\n"
        f"Analysis ID: {results.get('analysis_id', 'Unknown')}\n"
        f"Complexity: {results.get('complexity_analysis', {}).get('time_complexity', 'Unknown')}",
        title="Scalability Analysis"
    ))
    
    # Size scaling results
    if "size_scaling" in results:
        size_table = Table(title="Size Scaling Performance")
        size_table.add_column("Dataset Size", style="cyan")
        size_table.add_column("Training Time (s)", style="yellow")
        size_table.add_column("Memory (MB)", style="green")
        size_table.add_column("Throughput (/s)", style="blue")
        
        for point in results["size_scaling"]:
            size_table.add_row(
                f"{point['size']:,}",
                f"{point['training_time']:.2f}",
                f"{point['memory_mb']:.1f}",
                f"{point['throughput']:.1f}"
            )
        
        console.print(size_table)


def _display_stress_test_results(results: dict):
    """Display stress test results."""
    console.print(Panel(
        f"[bold blue]Stress Test Results[/bold blue]\n"
        f"Algorithm: {results.get('algorithm', 'Unknown')}\n"
        f"Test ID: {results.get('test_id', 'Unknown')}\n"
        f"Overall Stability: {results.get('overall_stability', 0):.2f}",
        title="Stress Test Results"
    ))
    
    # Create layout for different test results
    layout = Layout()
    layout.split_column(
        Layout(name="load"),
        Layout(name="memory"),
        Layout(name="cpu")
    )
    
    # Load test results
    load_results = results.get("load_test", {})
    load_table = Table(title="Load Test Results")
    load_table.add_column("Metric", style="cyan")
    load_table.add_column("Value", style="yellow")
    
    load_table.add_row("Concurrent Requests", str(load_results.get("concurrent_requests", 0)))
    load_table.add_row("Success Rate", f"{load_results.get('success_rate', 0):.2%}")
    load_table.add_row("Avg Response Time", f"{load_results.get('avg_response_time', 0):.3f}s")
    load_table.add_row("Throughput", f"{load_results.get('throughput', 0):.1f}/s")
    
    layout["load"].update(load_table)
    
    console.print(layout)


def _display_comparison_results(results: dict):
    """Display algorithm comparison results."""
    console.print(Panel(
        f"[bold blue]Algorithm Comparison Results[/bold blue]\n"
        f"Comparison ID: {results.get('comparison_id', 'Unknown')}\n"
        f"Algorithms: {', '.join(results.get('algorithms', []))}\n"
        f"Datasets: {', '.join(results.get('datasets', []))}",
        title="Algorithm Comparison"
    ))
    
    # Rankings
    rankings = results.get("rankings", {})
    if rankings:
        console.print(f"\n[bold]Best Overall Algorithm:[/bold] {rankings.get('overall', 'Unknown')}")
        
        if "by_metric" in rankings:
            metrics_table = Table(title="Best Algorithm by Metric")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Best Algorithm", style="green")
            
            for metric, algorithm in rankings["by_metric"].items():
                metrics_table.add_row(metric, algorithm)
            
            console.print(metrics_table)


def _export_benchmark_csv(results: dict, output_file: Path):
    """Export benchmark results to CSV."""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'Algorithm', 'Dataset', 'ROC AUC', 'Training Time', 
            'Memory MB', 'Throughput', 'F1 Score'
        ])
        
        # Write results
        for result in results.get("results", []):
            perf = result.get("performance", {})
            writer.writerow([
                result.get("algorithm", ""),
                result.get("dataset", ""),
                perf.get("roc_auc", 0),
                perf.get("training_time", 0),
                perf.get("peak_memory_mb", 0),
                perf.get("throughput", 0),
                perf.get("f1_score", 0)
            ])


def _export_benchmark_html(results: dict, output_file: Path):
    """Export benchmark results to HTML."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Results</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Suite:</strong> {results.get('suite_name', 'Unknown')}</p>
            <p><strong>Total Tests:</strong> {results.get('summary', {}).get('total_tests', 0)}</p>
        </div>
        <h2>Results</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Dataset</th>
                <th>ROC AUC</th>
                <th>Training Time</th>
                <th>Memory (MB)</th>
            </tr>
    """
    
    for result in results.get("results", []):
        perf = result.get("performance", {})
        html_content += f"""
            <tr>
                <td>{result.get('algorithm', '')}</td>
                <td>{result.get('dataset', '')}</td>
                <td>{perf.get('roc_auc', 0):.3f}</td>
                <td>{perf.get('training_time', 0):.2f}s</td>
                <td>{perf.get('peak_memory_mb', 0):.1f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


def _generate_console_report(result_files: List[Path]):
    """Generate console performance report."""
    console.print("[bold]Performance Analysis Report[/bold]\n")
    
    for file_path in result_files[:5]:  # Show latest 5 results
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            console.print(f"[cyan]Results from {file_path.name}:[/cyan]")
            summary = results.get("summary", {})
            console.print(f"  Total Tests: {summary.get('total_tests', 0)}")
            console.print(f"  Avg ROC AUC: {summary.get('avg_roc_auc', 0):.3f}")
            console.print(f"  Avg Training Time: {summary.get('avg_training_time', 0):.2f}s\n")
            
        except Exception as e:
            console.print(f"[red]Error reading {file_path}: {e}[/red]")


def _generate_html_report(result_files: List[Path], output_file: str):
    """Generate HTML performance report."""
    console.print(f"[yellow]HTML report generation not yet implemented[/yellow]")
    console.print(f"Would generate: {output_file}")


def _generate_pdf_report(result_files: List[Path], output_file: str):
    """Generate PDF performance report."""
    console.print(f"[yellow]PDF report generation not yet implemented[/yellow]")
    console.print(f"Would generate: {output_file}")


if __name__ == "__main__":
    performance_commands()