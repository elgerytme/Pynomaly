"""CLI commands for performance testing and benchmarking."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from pynomaly.application.services.performance_benchmarking_service import (
    BenchmarkConfig,
    BenchmarkSuite,
    PerformanceBenchmarkingService,
)
from pynomaly.infrastructure.config.container import Container

console = Console()


@click.group(name="benchmark")
def benchmark_commands():
    """Performance testing and benchmarking commands."""
    pass


@benchmark_commands.command()
@click.option("--suite-name", required=True, help="Name for the benchmark suite")
@click.option("--description", help="Description of the benchmark suite")
@click.option("--algorithms", multiple=True, help="Algorithms to benchmark")
@click.option("--dataset-sizes", multiple=True, type=int, help="Dataset sizes to test")
@click.option(
    "--feature-dimensions", multiple=True, type=int, help="Feature dimensions to test"
)
@click.option(
    "--contamination-rates",
    multiple=True,
    type=float,
    help="Contamination rates to test",
)
@click.option("--iterations", type=int, default=5, help="Number of iterations per test")
@click.option("--timeout", type=int, default=600, help="Timeout in seconds")
@click.option("--output-dir", help="Output directory for results")
@click.option(
    "--export-format",
    type=click.Choice(["json", "csv", "html"]),
    default="html",
    help="Export format for results",
)
def comprehensive(
    suite_name: str,
    description: str | None,
    algorithms: list[str],
    dataset_sizes: list[int],
    feature_dimensions: list[int],
    contamination_rates: list[float],
    iterations: int,
    timeout: int,
    output_dir: str | None,
    export_format: str,
):
    """Run comprehensive performance benchmark suite."""

    async def run_comprehensive_benchmark():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Initialize benchmarking service
            task1 = progress.add_task("Initializing benchmark service...", total=None)
            storage_path = Path(output_dir) if output_dir else Path("./benchmarks")
            benchmark_service = PerformanceBenchmarkingService(storage_path)
            progress.update(task1, completed=True)

            # Create benchmark configuration
            config = BenchmarkConfig(
                benchmark_name=suite_name,
                description=description or f"Comprehensive benchmark: {suite_name}",
                dataset_sizes=list(dataset_sizes)
                if dataset_sizes
                else [1000, 5000, 10000],
                feature_dimensions=list(feature_dimensions)
                if feature_dimensions
                else [10, 50, 100],
                contamination_rates=list(contamination_rates)
                if contamination_rates
                else [0.05, 0.1, 0.15],
                algorithms=list(algorithms)
                if algorithms
                else ["IsolationForest", "LOF", "OneClassSVM"],
                iterations=iterations,
                timeout_seconds=timeout,
            )

            # Create benchmark suite
            task2 = progress.add_task("Creating benchmark suite...", total=None)
            suite_id = await benchmark_service.create_benchmark_suite(
                suite_name=suite_name, description=description or "", config=config
            )
            progress.update(task2, completed=True)

            # Run comprehensive benchmark
            task3 = progress.add_task(
                "Running comprehensive benchmarks...", total=len(config.algorithms)
            )

            try:
                suite = await benchmark_service.run_comprehensive_benchmark(
                    suite_id=suite_id, algorithms=config.algorithms
                )

                progress.update(task3, completed=len(config.algorithms))

                # Display results summary
                _display_benchmark_summary(suite)

                # Generate report
                if output_dir:
                    task4 = progress.add_task(
                        f"Generating {export_format} report...", total=None
                    )

                    report_path = await benchmark_service.generate_benchmark_report(
                        suite_id=suite_id,
                        output_path=storage_path,
                        format=export_format,
                    )

                    progress.update(task4, completed=True)
                    console.print(
                        f"[green]Benchmark report saved to: {report_path}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error running benchmark: {e}[/red]")
                return

    asyncio.run(run_comprehensive_benchmark())


@benchmark_commands.command()
@click.option("--algorithm", required=True, help="Algorithm to test for scalability")
@click.option("--base-size", type=int, default=1000, help="Base dataset size")
@click.option("--scale-factors", multiple=True, type=int, help="Scale factors to test")
@click.option("--feature-dimension", type=int, default=10, help="Number of features")
@click.option("--output-file", help="Output file for results")
def scalability(
    algorithm: str,
    base_size: int,
    scale_factors: list[int],
    feature_dimension: int,
    output_file: str | None,
):
    """Run scalability test for specific algorithm."""

    async def run_scalability_test():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize benchmarking service
            task1 = progress.add_task("Initializing scalability test...", total=None)
            storage_path = Path("./benchmarks")
            benchmark_service = PerformanceBenchmarkingService(storage_path)
            progress.update(task1, completed=True)

            # Run scalability test
            task2 = progress.add_task("Running scalability analysis...", total=None)

            scale_list = list(scale_factors) if scale_factors else [1, 2, 4, 8, 16, 32]

            try:
                results = await benchmark_service.run_scalability_test(
                    algorithm_name=algorithm,
                    base_dataset_size=base_size,
                    scale_factors=scale_list,
                    feature_dimension=feature_dimension,
                )

                progress.update(task2, completed=True)

                # Display scalability results
                _display_scalability_results(results)

                # Save results if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(
                        f"[green]Scalability results saved to: {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error running scalability test: {e}[/red]")
                return

    asyncio.run(run_scalability_test())


@benchmark_commands.command()
@click.option("--algorithm", required=True, help="Algorithm to test for memory usage")
@click.option(
    "--max-size", type=int, default=1000000, help="Maximum dataset size to test"
)
@click.option("--memory-limit", type=float, default=8192.0, help="Memory limit in MB")
@click.option("--output-file", help="Output file for results")
def memory_stress(
    algorithm: str, max_size: int, memory_limit: float, output_file: str | None
):
    """Run memory stress test for algorithm."""

    async def run_memory_stress():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize benchmarking service
            task1 = progress.add_task("Initializing memory stress test...", total=None)
            storage_path = Path("./benchmarks")
            benchmark_service = PerformanceBenchmarkingService(storage_path)
            progress.update(task1, completed=True)

            # Run memory stress test
            task2 = progress.add_task("Running memory stress analysis...", total=None)

            try:
                results = await benchmark_service.run_memory_stress_test(
                    algorithm_name=algorithm,
                    max_dataset_size=max_size,
                    memory_limit_mb=memory_limit,
                )

                progress.update(task2, completed=True)

                # Display memory stress results
                _display_memory_stress_results(results)

                # Save results if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(
                        f"[green]Memory stress results saved to: {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error running memory stress test: {e}[/red]")
                return

    asyncio.run(run_memory_stress())


@benchmark_commands.command()
@click.option("--algorithms", multiple=True, help="Algorithms to test for throughput")
@click.option("--dataset-sizes", multiple=True, type=int, help="Dataset sizes to test")
@click.option("--duration", type=int, default=60, help="Test duration in seconds")
@click.option("--output-file", help="Output file for results")
def throughput(
    algorithms: list[str],
    dataset_sizes: list[int],
    duration: int,
    output_file: str | None,
):
    """Run throughput benchmark for algorithms."""

    async def run_throughput_test():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize benchmarking service
            task1 = progress.add_task("Initializing throughput test...", total=None)
            storage_path = Path("./benchmarks")
            benchmark_service = PerformanceBenchmarkingService(storage_path)
            progress.update(task1, completed=True)

            # Run throughput test
            task2 = progress.add_task("Running throughput analysis...", total=None)

            algorithm_list = (
                list(algorithms)
                if algorithms
                else ["IsolationForest", "LOF", "OneClassSVM"]
            )
            size_list = (
                list(dataset_sizes) if dataset_sizes else [1000, 5000, 10000, 25000]
            )

            try:
                results = await benchmark_service.run_throughput_benchmark(
                    algorithms=algorithm_list,
                    dataset_sizes=size_list,
                    duration_seconds=duration,
                )

                progress.update(task2, completed=True)

                # Display throughput results
                _display_throughput_results(results)

                # Save results if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(
                        f"[green]Throughput results saved to: {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error running throughput test: {e}[/red]")
                return

    asyncio.run(run_throughput_test())


@benchmark_commands.command()
@click.option("--algorithms", multiple=True, help="Algorithms to compare")
@click.option(
    "--dataset-sizes", multiple=True, type=int, help="Dataset sizes for comparison"
)
@click.option("--metrics", multiple=True, help="Metrics to compare")
@click.option("--output-file", help="Output file for results")
def compare(
    algorithms: list[str],
    dataset_sizes: list[int],
    metrics: list[str],
    output_file: str | None,
):
    """Compare algorithms across multiple metrics."""

    async def run_algorithm_comparison():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize benchmarking service
            task1 = progress.add_task(
                "Initializing algorithm comparison...", total=None
            )
            storage_path = Path("./benchmarks")
            benchmark_service = PerformanceBenchmarkingService(storage_path)
            progress.update(task1, completed=True)

            # Run algorithm comparison
            task2 = progress.add_task("Running algorithm comparison...", total=None)

            algorithm_list = (
                list(algorithms)
                if algorithms
                else ["IsolationForest", "LOF", "OneClassSVM"]
            )
            size_list = list(dataset_sizes) if dataset_sizes else [1000, 5000, 10000]
            metric_list = (
                list(metrics)
                if metrics
                else ["execution_time", "memory_usage", "accuracy", "throughput"]
            )

            try:
                results = await benchmark_service.compare_algorithms(
                    algorithms=algorithm_list,
                    dataset_sizes=size_list,
                    metrics=metric_list,
                )

                progress.update(task2, completed=True)

                # Display comparison results
                _display_comparison_results(results)

                # Save results if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(
                        f"[green]Comparison results saved to: {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error running algorithm comparison: {e}[/red]")
                return

    asyncio.run(run_algorithm_comparison())


@benchmark_commands.command()
@click.option("--algorithm", help="Specific algorithm to analyze (optional)")
@click.option("--days", type=int, default=30, help="Number of days to analyze")
@click.option("--output-file", help="Output file for trend analysis")
def trends(algorithm: str | None, days: int, output_file: str | None):
    """Analyze performance trends over time."""

    async def run_trend_analysis():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize benchmarking service
            task1 = progress.add_task("Initializing trend analysis...", total=None)
            storage_path = Path("./benchmarks")
            benchmark_service = PerformanceBenchmarkingService(storage_path)
            progress.update(task1, completed=True)

            # Run trend analysis
            task2 = progress.add_task("Analyzing performance trends...", total=None)

            try:
                results = await benchmark_service.get_performance_trends(
                    algorithm_name=algorithm, days=days
                )

                progress.update(task2, completed=True)

                # Display trend analysis
                _display_trend_analysis(results)

                # Save results if requested
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    console.print(
                        f"[green]Trend analysis saved to: {output_file}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error running trend analysis: {e}[/red]")
                return

    asyncio.run(run_trend_analysis())


@benchmark_commands.command()
@click.option("--benchmark-dir", help="Directory containing benchmark results")
def list_results(benchmark_dir: str | None):
    """List available benchmark results."""

    benchmark_path = Path(benchmark_dir) if benchmark_dir else Path("./benchmarks")

    if not benchmark_path.exists():
        console.print(
            f"[yellow]Benchmark directory not found: {benchmark_path}[/yellow]"
        )
        return

    # Find benchmark files
    json_files = list(benchmark_path.glob("benchmark_report_*.json"))
    html_files = list(benchmark_path.glob("benchmark_report_*.html"))
    csv_files = list(benchmark_path.glob("benchmark_report_*.csv"))

    console.print(
        Panel(
            f"[bold blue]Available Benchmark Results[/bold blue]\n"
            f"Directory: {benchmark_path}\n"
            f"JSON Reports: {len(json_files)}\n"
            f"HTML Reports: {len(html_files)}\n"
            f"CSV Reports: {len(csv_files)}",
            title="Benchmark Results",
        )
    )

    if json_files or html_files or csv_files:
        results_table = Table(title="Recent Benchmark Reports")
        results_table.add_column("File", style="cyan")
        results_table.add_column("Format", style="yellow")
        results_table.add_column("Size", style="green")
        results_table.add_column("Modified", style="blue")

        all_files = (
            [(f, "JSON") for f in json_files]
            + [(f, "HTML") for f in html_files]
            + [(f, "CSV") for f in csv_files]
        )
        all_files.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        for file_path, file_format in all_files[:10]:  # Show latest 10
            stat = file_path.stat()
            size_kb = stat.st_size / 1024
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            results_table.add_row(
                file_path.name, file_format, f"{size_kb:.1f} KB", modified
            )

        console.print(results_table)


@benchmark_commands.command()
def system_info():
    """Display system information for benchmarking context."""

    import platform

    import psutil

    # Get system information
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    disk_usage = psutil.disk_usage("/")

    console.print(
        Panel(
            f"[bold blue]System Information[/bold blue]\n"
            f"Platform: {platform.system()} {platform.release()}\n"
            f"Architecture: {platform.machine()}\n"
            f"Python Version: {platform.python_version()}\n"
            f"CPU Cores: {cpu_count} logical, {psutil.cpu_count(logical=False)} physical\n"
            f"Memory: {memory.total / 1024 / 1024 / 1024:.1f} GB total, {memory.available / 1024 / 1024 / 1024:.1f} GB available\n"
            f"Disk Space: {disk_usage.total / 1024 / 1024 / 1024:.1f} GB total, {disk_usage.free / 1024 / 1024 / 1024:.1f} GB free\n"
            f"Current CPU Usage: {psutil.cpu_percent()}%\n"
            f"Current Memory Usage: {memory.percent}%",
            title="Benchmark Environment",
        )
    )


# Helper functions for display


def _display_benchmark_summary(suite: BenchmarkSuite):
    """Display benchmark suite summary."""
    console.print(
        Panel(
            f"[bold blue]Benchmark Suite Complete[/bold blue]\n"
            f"Suite: {suite.suite_name}\n"
            f"Duration: {suite.total_duration_seconds:.2f} seconds\n"
            f"Total Tests: {len(suite.individual_results)}\n"
            f"Algorithms: {len(suite.summary_stats)}\n"
            f"Performance Grade: {suite.performance_grade}",
            title="Benchmark Results",
        )
    )

    if suite.summary_stats:
        summary_table = Table(title="Algorithm Performance Summary")
        summary_table.add_column("Algorithm", style="cyan")
        summary_table.add_column("Avg Time (s)", style="yellow")
        summary_table.add_column("Avg Memory (MB)", style="green")
        summary_table.add_column("Avg Throughput", style="blue")
        summary_table.add_column("Avg Accuracy", style="magenta")

        for algorithm, stats in suite.summary_stats.items():
            summary_table.add_row(
                algorithm,
                f"{stats['avg_execution_time']:.3f}",
                f"{stats['avg_memory_usage']:.1f}",
                f"{stats['avg_throughput']:.1f}",
                f"{stats['avg_accuracy']:.3f}",
            )

        console.print(summary_table)

    if suite.comparative_analysis:
        console.print("\n[bold]Comparative Analysis:[/bold]")
        analysis = suite.comparative_analysis
        console.print(
            f"• Fastest Algorithm: [green]{analysis.get('fastest_algorithm', 'N/A')}[/green]"
        )
        console.print(
            f"• Most Memory Efficient: [green]{analysis.get('most_memory_efficient', 'N/A')}[/green]"
        )
        console.print(
            f"• Highest Throughput: [green]{analysis.get('highest_throughput', 'N/A')}[/green]"
        )
        console.print(
            f"• Most Accurate: [green]{analysis.get('most_accurate', 'N/A')}[/green]"
        )

    if suite.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in suite.recommendations:
            console.print(f"• {rec}")


def _display_scalability_results(results: dict):
    """Display scalability test results."""
    console.print(
        Panel(
            f"[bold blue]Scalability Test Results[/bold blue]\n"
            f"Algorithm: {results['algorithm']}\n"
            f"Scale Factors Tested: {len(results['results'])}",
            title="Scalability Analysis",
        )
    )

    scalability_table = Table(title="Scalability Performance")
    scalability_table.add_column("Scale Factor", style="cyan")
    scalability_table.add_column("Dataset Size", style="yellow")
    scalability_table.add_column("Execution Time (s)", style="green")
    scalability_table.add_column("Efficiency Ratio", style="blue")
    scalability_table.add_column("Memory Usage (MB)", style="magenta")

    for result in results["results"]:
        scalability_table.add_row(
            str(result.scalability_factor),
            str(result.dataset_size),
            f"{result.execution_time_seconds:.3f}",
            f"{result.efficiency_ratio:.3f}",
            f"{result.peak_memory_mb:.1f}",
        )

    console.print(scalability_table)

    # Display scalability summary
    if "scalability_summary" in results:
        summary = results["scalability_summary"]
        console.print("\n[bold]Scalability Summary:[/bold]")
        console.print(
            f"• Linear Scalability Score: {summary.get('linear_scalability_score', 0):.3f}"
        )
        console.print(
            f"• Time Complexity Estimate: {summary.get('time_complexity_estimate', 'unknown')}"
        )
        console.print(f"• Scalability Grade: {summary.get('scalability_grade', 'N/A')}")
        console.print(
            f"• Max Efficient Scale: {summary.get('max_efficient_scale', 'N/A')}"
        )


def _display_memory_stress_results(results: dict):
    """Display memory stress test results."""
    console.print(
        Panel(
            f"[bold blue]Memory Stress Test Results[/bold blue]\n"
            f"Algorithm: {results['algorithm']}\n"
            f"Max Dataset Size Tested: {results['max_dataset_size_tested']:,}\n"
            f"Memory Limit: {results['memory_limit_mb']:.1f} MB\n"
            f"Test Points: {len(results['results'])}",
            title="Memory Analysis",
        )
    )

    memory_table = Table(title="Memory Usage by Dataset Size")
    memory_table.add_column("Dataset Size", style="cyan")
    memory_table.add_column("Peak Memory (MB)", style="yellow")
    memory_table.add_column("Memory Growth (MB)", style="green")
    memory_table.add_column("Execution Time (s)", style="blue")
    memory_table.add_column("Success", style="magenta")

    for result in results["results"]:
        success_indicator = "✓" if result.success else "✗"
        memory_table.add_row(
            f"{result.dataset_size:,}",
            f"{result.peak_memory_mb:.1f}",
            f"{result.memory_growth_mb:.1f}",
            f"{result.execution_time_seconds:.3f}",
            success_indicator,
        )

    console.print(memory_table)

    # Display memory analysis
    if "memory_analysis" in results:
        analysis = results["memory_analysis"]
        console.print("\n[bold]Memory Analysis:[/bold]")
        console.print(
            f"• Memory Efficiency: {analysis.get('memory_efficiency', 0):.3f} MB per 1000 samples"
        )
        console.print(
            f"• Average Memory Growth: {analysis.get('memory_growth_rate', 0):.1f} MB"
        )
        console.print(
            f"• Max Memory Tested: {analysis.get('max_memory_tested', 0):.1f} MB"
        )
        console.print(
            f"• Memory Scalability: {analysis.get('memory_scalability', 'unknown')}"
        )


def _display_throughput_results(results: dict):
    """Display throughput test results."""
    console.print(
        Panel(
            f"[bold blue]Throughput Test Results[/bold blue]\n"
            f"Algorithms Tested: {len(results['results'])}\n"
            f"Overall Best: {results['throughput_analysis'].get('overall_best', 'N/A')}",
            title="Throughput Analysis",
        )
    )

    for algorithm, algorithm_results in results["results"].items():
        algorithm_table = Table(title=f"Throughput - {algorithm}")
        algorithm_table.add_column("Dataset Size", style="cyan")
        algorithm_table.add_column("Throughput (samples/s)", style="yellow")
        algorithm_table.add_column("Duration (s)", style="green")
        algorithm_table.add_column("Samples Processed", style="blue")

        for result in algorithm_results:
            algorithm_table.add_row(
                str(result["dataset_size"]),
                f"{result['throughput_samples_per_second']:.1f}",
                f"{result['duration_seconds']:.1f}",
                f"{result['samples_processed']:,}",
            )

        console.print(algorithm_table)

    # Display throughput analysis
    analysis = results["throughput_analysis"]
    console.print("\n[bold]Throughput Analysis:[/bold]")

    for algorithm, stats in analysis.items():
        if algorithm != "overall_best":
            console.print(f"\n{algorithm}:")
            console.print(
                f"  • Average Throughput: {stats.get('avg_throughput', 0):.1f} samples/s"
            )
            console.print(
                f"  • Max Throughput: {stats.get('max_throughput', 0):.1f} samples/s"
            )
            console.print(
                f"  • Throughput Stability: {stats.get('throughput_stability', 0):.3f}"
            )
            console.print(
                f"  • Best Dataset Size: {stats.get('best_dataset_size', 'N/A')}"
            )


def _display_comparison_results(results: dict):
    """Display algorithm comparison results."""
    console.print(
        Panel(
            f"[bold blue]Algorithm Comparison Results[/bold blue]\n"
            f"Algorithms: {', '.join(results['algorithms'])}\n"
            f"Dataset Sizes: {', '.join(map(str, results['dataset_sizes']))}\n"
            f"Metrics: {', '.join(results['metrics'])}",
            title="Comparison Analysis",
        )
    )

    # Display comparison by metric
    for metric, metric_comparison in results["analysis"].items():
        console.print(f"\n[bold]{metric.replace('_', ' ').title()} Comparison:[/bold]")

        metric_table = Table()
        metric_table.add_column("Algorithm", style="cyan")
        metric_table.add_column("Average", style="yellow")
        metric_table.add_column("Min", style="green")
        metric_table.add_column("Max", style="blue")
        metric_table.add_column("Std Dev", style="magenta")

        for algorithm, stats in metric_comparison.items():
            metric_table.add_row(
                algorithm,
                f"{stats['average']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['std']:.3f}",
            )

        console.print(metric_table)


def _display_trend_analysis(results: dict):
    """Display performance trend analysis."""
    if "message" in results:
        console.print(f"[yellow]{results['message']}[/yellow]")
        return

    console.print(
        Panel(
            f"[bold blue]Performance Trend Analysis[/bold blue]\n"
            f"Algorithm: {results.get('algorithm', 'All')}\n"
            f"Period: {results['period_days']} days\n"
            f"Data Points: {results['data_points']}",
            title="Trend Analysis",
        )
    )

    trends = results["trends"]

    trend_table = Table(title="Performance Trends")
    trend_table.add_column("Metric", style="cyan")
    trend_table.add_column("Direction", style="yellow")
    trend_table.add_column("Change %", style="green")
    trend_table.add_column("First Period Avg", style="blue")
    trend_table.add_column("Second Period Avg", style="magenta")

    for metric_name, trend_data in trends.items():
        if isinstance(trend_data, dict) and "direction" in trend_data:
            direction_color = {
                "increasing": "red",
                "decreasing": "green" if "time" in metric_name else "red",
                "stable": "blue",
            }.get(trend_data["direction"], "white")

            trend_table.add_row(
                metric_name.replace("_", " ").title(),
                f"[{direction_color}]{trend_data['direction']}[/{direction_color}]",
                f"{trend_data['change_percent']:+.1f}%",
                f"{trend_data['first_period_avg']:.3f}",
                f"{trend_data['second_period_avg']:.3f}",
            )

    console.print(trend_table)

    # Display recommendations
    if "recommendations" in results and results["recommendations"]:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in results["recommendations"]:
            console.print(f"• {rec}")


if __name__ == "__main__":
    benchmark_commands()
