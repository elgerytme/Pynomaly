"""CLI commands for performance benchmarking using Typer."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
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

# Create Typer app for benchmark commands
benchmark_app = typer.Typer(
    name="benchmark",
    help="🔬 Performance benchmarking and testing commands",
    add_completion=True,
)


@benchmark_app.command()
def comprehensive(
    suite_name: str = typer.Option("quick_benchmark", help="Name for the benchmark suite"),
    description: Optional[str] = typer.Option(None, help="Description of the benchmark suite"),
    algorithms: Optional[List[str]] = typer.Option(None, help="Algorithms to benchmark"),
    dataset_sizes: Optional[List[int]] = typer.Option(None, help="Dataset sizes to test"),
    feature_dimensions: Optional[List[int]] = typer.Option(None, help="Feature dimensions to test"),
    contamination_rates: Optional[List[float]] = typer.Option(None, help="Contamination rates to test"),
    iterations: int = typer.Option(5, help="Number of iterations per test"),
    timeout: int = typer.Option(600, help="Timeout in seconds"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for results"),
    export_format: str = typer.Option("html", help="Export format for results (json/csv/html)"),
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
            storage_path.mkdir(parents=True, exist_ok=True)
            
            try:
                benchmark_service = PerformanceBenchmarkingService(storage_path)
                progress.update(task1, completed=True)
            except Exception as e:
                console.print(f"[red]Error initializing benchmark service: {e}[/red]")
                return

            # Create benchmark configuration
            config = BenchmarkConfig(
                benchmark_name=suite_name,
                description=description or f"Comprehensive benchmark: {suite_name}",
                dataset_sizes=dataset_sizes or [1000, 5000, 10000],
                feature_dimensions=feature_dimensions or [10, 50, 100],
                contamination_rates=contamination_rates or [0.05, 0.1, 0.15],
                algorithms=algorithms or ["IsolationForest", "LOF", "OneClassSVM"],
                iterations=iterations,
                timeout_seconds=timeout,
            )

            # Create benchmark suite
            task2 = progress.add_task("Creating benchmark suite...", total=None)
            try:
                suite_id = await benchmark_service.create_benchmark_suite(
                    suite_name=suite_name, description=description or "", config=config
                )
                progress.update(task2, completed=True)
            except Exception as e:
                console.print(f"[red]Error creating benchmark suite: {e}[/red]")
                return

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


@benchmark_app.command()
def scalability(
    algorithm: str = typer.Argument(..., help="Algorithm to test for scalability"),
    base_size: int = typer.Option(1000, help="Base dataset size"),
    scale_factors: Optional[List[int]] = typer.Option(None, help="Scale factors to test"),
    feature_dimension: int = typer.Option(10, help="Number of features"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
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
            storage_path.mkdir(parents=True, exist_ok=True)
            
            try:
                benchmark_service = PerformanceBenchmarkingService(storage_path)
                progress.update(task1, completed=True)
            except Exception as e:
                console.print(f"[red]Error initializing benchmark service: {e}[/red]")
                return

            # Run scalability test
            task2 = progress.add_task("Running scalability analysis...", total=None)

            scale_list = scale_factors or [1, 2, 4, 8, 16, 32]

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


@benchmark_app.command()
def memory_stress(
    algorithm: str = typer.Argument(..., help="Algorithm to test for memory usage"),
    max_size: int = typer.Option(1000000, help="Maximum dataset size to test"),
    memory_limit: float = typer.Option(8192.0, help="Memory limit in MB"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
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
            storage_path.mkdir(parents=True, exist_ok=True)
            
            try:
                benchmark_service = PerformanceBenchmarkingService(storage_path)
                progress.update(task1, completed=True)
            except Exception as e:
                console.print(f"[red]Error initializing benchmark service: {e}[/red]")
                return

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


@benchmark_app.command()
def throughput(
    algorithms: Optional[List[str]] = typer.Option(None, help="Algorithms to test for throughput"),
    dataset_sizes: Optional[List[int]] = typer.Option(None, help="Dataset sizes to test"),
    duration: int = typer.Option(60, help="Test duration in seconds"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
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
            storage_path.mkdir(parents=True, exist_ok=True)
            
            try:
                benchmark_service = PerformanceBenchmarkingService(storage_path)
                progress.update(task1, completed=True)
            except Exception as e:
                console.print(f"[red]Error initializing benchmark service: {e}[/red]")
                return

            # Run throughput test
            task2 = progress.add_task("Running throughput analysis...", total=None)

            algorithm_list = algorithms or ["IsolationForest", "LOF", "OneClassSVM"]
            size_list = dataset_sizes or [1000, 5000, 10000, 25000]

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


@benchmark_app.command()
def list_results(benchmark_dir: Optional[str] = typer.Option(None, help="Directory containing benchmark results")):
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


@benchmark_app.command()
def system_info():
    """Display system information for benchmarking context."""

    import platform
    
    try:
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
    except ImportError:
        console.print(
            Panel(
                f"[bold blue]Basic System Information[/bold blue]\n"
                f"Platform: {platform.system()} {platform.release()}\n"
                f"Architecture: {platform.machine()}\n"
                f"Python Version: {platform.python_version()}\n"
                f"[yellow]Note: Install psutil for detailed system metrics[/yellow]",
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
    scalability_table.add_column("Memory Usage (MB)", style="magenta")

    for result in results["results"]:
        scalability_table.add_row(
            str(result.get("scalability_factor", "N/A")),
            str(result.get("dataset_size", "N/A")),
            f"{result.get('execution_time_seconds', 0):.3f}",
            f"{result.get('peak_memory_mb', 0):.1f}",
        )

    console.print(scalability_table)


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


def _display_throughput_results(results: dict):
    """Display throughput test results."""
    console.print(
        Panel(
            f"[bold blue]Throughput Test Results[/bold blue]\n"
            f"Algorithms Tested: {len(results['results'])}",
            title="Throughput Analysis",
        )
    )


if __name__ == "__main__":
    benchmark_app()
