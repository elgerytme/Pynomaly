"""
Performance CLI Commands for Pynomaly.

This module provides CLI commands for performance benchmarking, monitoring,
and optimization of the Pynomaly anomaly detection system.
"""

import asyncio
import json
import sys
from pathlib import Path

import click

from monorepo.infrastructure.performance.advanced_benchmarking_service import (
    AdvancedBenchmarkConfig,
    AdvancedPerformanceBenchmarkingService,
)
from monorepo.infrastructure.performance.optimization_engine import (
    create_optimization_engine,
)
from monorepo.infrastructure.performance.performance_integration import (
    get_performance_integration_manager,
    get_performance_stats,
    start_performance_monitoring,
    stop_performance_monitoring,
)


@click.group()
def performance():
    """Performance benchmarking, monitoring, and optimization commands."""
    pass


@performance.command()
@click.option(
    "--algorithms",
    "-a",
    multiple=True,
    default=["isolation_forest", "local_outlier_factor", "one_class_svm"],
    help="Algorithms to benchmark",
)
@click.option(
    "--dataset-sizes",
    "-s",
    multiple=True,
    type=int,
    default=[1000, 5000, 10000, 25000],
    help="Dataset sizes to test",
)
@click.option(
    "--feature-dimensions",
    "-f",
    multiple=True,
    type=int,
    default=[10, 20, 50, 100],
    help="Feature dimensions to test",
)
@click.option(
    "--iterations", "-i", type=int, default=5, help="Number of iterations per test"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="performance_results",
    help="Output directory for results",
)
@click.option(
    "--export-format",
    "-e",
    type=click.Choice(["json", "csv", "html", "all"]),
    default="html",
    help="Export format for results",
)
@click.option(
    "--enable-profiling", is_flag=True, default=True, help="Enable detailed profiling"
)
@click.option(
    "--parallel", is_flag=True, default=True, help="Enable parallel execution"
)
def benchmark(
    algorithms,
    dataset_sizes,
    feature_dimensions,
    iterations,
    output_dir,
    export_format,
    enable_profiling,
    parallel,
):
    """Run comprehensive performance benchmarks."""
    click.echo("🚀 Starting comprehensive performance benchmarking...")

    async def run_benchmark():
        # Setup
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create benchmark service
        benchmark_service = AdvancedPerformanceBenchmarkingService(
            storage_path=output_path
        )

        # Create benchmark configuration
        config = AdvancedBenchmarkConfig(
            benchmark_name="cli_comprehensive_benchmark",
            description="Comprehensive benchmark run from CLI",
            dataset_sizes=list(dataset_sizes),
            feature_dimensions=list(feature_dimensions),
            algorithms=list(algorithms),
            iterations=iterations,
            enable_memory_profiling=enable_profiling,
            enable_cpu_profiling=enable_profiling,
            parallel_execution=parallel,
            export_formats=["json", "csv", "html"]
            if export_format == "all"
            else [export_format],
        )

        # Create benchmark suite
        suite_id = await benchmark_service.create_benchmark_suite(
            suite_name="CLI Comprehensive Benchmark",
            description="Benchmark suite created from CLI",
            config=config,
        )

        click.echo(f"📊 Created benchmark suite: {suite_id}")

        # Run benchmarks
        with click.progressbar(
            length=len(algorithms) * len(dataset_sizes) * len(feature_dimensions),
            label="Running benchmarks",
        ) as bar:
            # Custom progress callback would go here
            # For now, we'll run the benchmark normally
            results = await benchmark_service.run_comprehensive_benchmark(
                suite_id=suite_id, algorithms=list(algorithms)
            )

            bar.update(len(algorithms) * len(dataset_sizes) * len(feature_dimensions))

        # Generate reports
        click.echo("📈 Generating performance reports...")

        formats_to_export = (
            ["json", "csv", "html"] if export_format == "all" else [export_format]
        )

        for fmt in formats_to_export:
            report_path = await benchmark_service.generate_benchmark_report(
                suite_id=suite_id, output_path=output_path, format=fmt
            )
            click.echo(f"✅ {fmt.upper()} report generated: {report_path}")

        # Display summary
        click.echo("\n📋 Benchmark Summary:")
        click.echo(f"Suite ID: {suite_id}")
        click.echo(f"Total tests: {len(results.individual_results)}")
        click.echo(f"Algorithms tested: {len(algorithms)}")
        click.echo(f"Overall performance grade: {results.performance_grade}")
        click.echo(f"Duration: {results.total_duration_seconds:.2f} seconds")

        if results.recommendations:
            click.echo("\n💡 Recommendations:")
            for rec in results.recommendations:
                click.echo(f"  • {rec}")

        return results

    # Run async benchmark
    try:
        results = asyncio.run(run_benchmark())
        click.echo("✅ Benchmark completed successfully!")
        return results
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {str(e)}", err=True)
        sys.exit(1)


@performance.command()
@click.option("--algorithm", "-a", required=True, help="Algorithm to test scalability")
@click.option("--base-size", type=int, default=1000, help="Base dataset size")
@click.option("--max-size", type=int, default=50000, help="Maximum dataset size")
@click.option("--steps", type=int, default=8, help="Number of test steps")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="scalability_results",
    help="Output directory for results",
)
def scalability(algorithm, base_size, max_size, steps, output_dir):
    """Test algorithm scalability."""
    click.echo(f"📏 Testing scalability for {algorithm}...")

    async def run_scalability():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        benchmark_service = AdvancedPerformanceBenchmarkingService(
            storage_path=output_path
        )

        # Run scalability test
        results = await benchmark_service.run_scalability_analysis(
            algorithm_name=algorithm,
            base_size=base_size,
            max_size=max_size,
            scale_steps=steps,
        )

        # Save results
        results_file = output_path / f"scalability_{algorithm}.json"
        with open(results_file, "w") as f:
            # Convert results to JSON-serializable format
            json_results = {
                "algorithm": results["algorithm"],
                "scale_range": results["scale_range"],
                "analysis": results["analysis"],
                "recommendations": results["recommendations"],
            }
            json.dump(json_results, f, indent=2)

        click.echo(f"✅ Scalability test completed: {results_file}")

        # Display analysis
        analysis = results["analysis"]
        click.echo(f"\n📊 Scalability Analysis for {algorithm}:")
        click.echo(f"Time complexity: {analysis.get('pattern', 'unknown')}")
        click.echo(f"Scalability grade: {analysis.get('scalability_grade', 'unknown')}")
        click.echo(f"Average efficiency: {analysis.get('average_efficiency', 0):.3f}")

        if "degradation_point" in analysis and analysis["degradation_point"]:
            click.echo(
                f"⚠️  Performance degradation starts at: {analysis['degradation_point']} samples"
            )

        return results

    try:
        results = asyncio.run(run_scalability())
        return results
    except Exception as e:
        click.echo(f"❌ Scalability test failed: {str(e)}", err=True)
        sys.exit(1)


@performance.command()
@click.option("--algorithm", "-a", required=True, help="Algorithm to stress test")
@click.option(
    "--max-memory", type=float, default=2048.0, help="Maximum memory limit in MB"
)
@click.option("--duration", type=int, default=30, help="Test duration in minutes")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="stress_test_results",
    help="Output directory for results",
)
def stress_test(algorithm, max_memory, duration, output_dir):
    """Run stress test for an algorithm."""
    click.echo(f"💪 Running stress test for {algorithm}...")

    async def run_stress():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        benchmark_service = AdvancedPerformanceBenchmarkingService(
            storage_path=output_path
        )

        # Run stress test
        results = await benchmark_service.run_stress_test(
            algorithm_name=algorithm,
            stress_duration_minutes=duration,
            concurrent_loads=[1, 2, 4, 8],
            memory_pressure=True,
        )

        # Save results
        results_file = output_path / f"stress_test_{algorithm}.json"
        with open(results_file, "w") as f:
            # Convert to JSON-serializable format
            json_results = {
                "algorithm": results["algorithm"],
                "duration_minutes": results["stress_duration_minutes"],
                "concurrent_loads": results["concurrent_loads_tested"],
                "analysis": results["overall_analysis"],
            }
            json.dump(json_results, f, indent=2)

        click.echo(f"✅ Stress test completed: {results_file}")

        # Display summary
        overall_analysis = results["overall_analysis"]
        click.echo(f"\n💪 Stress Test Summary for {algorithm}:")
        click.echo(f"Duration: {duration} minutes")
        click.echo(
            f"Max concurrent load tested: {max(results['concurrent_loads_tested'])}"
        )
        click.echo(f"Memory limit: {max_memory} MB")

        return results

    try:
        results = asyncio.run(run_stress())
        return results
    except Exception as e:
        click.echo(f"❌ Stress test failed: {str(e)}", err=True)
        sys.exit(1)


@performance.command()
@click.option(
    "--algorithms",
    "-a",
    multiple=True,
    default=["isolation_forest", "local_outlier_factor"],
    help="Algorithms to compare",
)
@click.option(
    "--dataset-sizes",
    "-s",
    multiple=True,
    type=int,
    default=[1000, 5000, 10000],
    help="Dataset sizes for comparison",
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    default=["execution_time", "memory_usage", "accuracy"],
    help="Metrics to compare",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="comparison_results",
    help="Output directory for results",
)
def compare(algorithms, dataset_sizes, metrics, output_dir):
    """Compare performance across algorithms."""
    click.echo(f"⚖️  Comparing {len(algorithms)} algorithms...")

    async def run_comparison():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        benchmark_service = AdvancedPerformanceBenchmarkingService(
            storage_path=output_path
        )

        # Run comparison
        results = await benchmark_service.compare_algorithm_performance(
            algorithms=list(algorithms),
            dataset_sizes=list(dataset_sizes),
            comparison_metrics=list(metrics),
            statistical_analysis=True,
        )

        # Save results
        results_file = output_path / "algorithm_comparison.json"
        with open(results_file, "w") as f:
            # Convert to JSON-serializable format
            json_results = {
                "algorithms": results["algorithms"],
                "dataset_sizes": results["dataset_sizes"],
                "metrics": results["comparison_metrics"],
                "analysis": results["comparative_analysis"],
                "recommendations": results["recommendations"],
            }
            json.dump(json_results, f, indent=2)

        click.echo(f"✅ Comparison completed: {results_file}")

        # Display analysis
        analysis = results["comparative_analysis"]
        click.echo("\n⚖️  Algorithm Comparison Results:")

        if "fastest" in analysis:
            click.echo(f"🚀 Fastest algorithm: {analysis['fastest']}")
        if "most_memory_efficient" in analysis:
            click.echo(f"💾 Most memory efficient: {analysis['most_memory_efficient']}")
        if "most_accurate" in analysis:
            click.echo(f"🎯 Most accurate: {analysis['most_accurate']}")

        if "overall_ranking" in analysis:
            click.echo(f"🏆 Overall ranking: {', '.join(analysis['overall_ranking'])}")

        # Show recommendations
        if results["recommendations"]:
            click.echo("\n💡 Recommendations:")
            for rec in results["recommendations"]:
                click.echo(f"  • {rec}")

        return results

    try:
        results = asyncio.run(run_comparison())
        return results
    except Exception as e:
        click.echo(f"❌ Comparison failed: {str(e)}", err=True)
        sys.exit(1)


@performance.command()
@click.option(
    "--start", is_flag=True, default=False, help="Start performance monitoring"
)
@click.option("--stop", is_flag=True, default=False, help="Stop performance monitoring")
@click.option("--status", is_flag=True, default=False, help="Show monitoring status")
@click.option("--alerts", is_flag=True, default=False, help="Show active alerts")
@click.option(
    "--severity",
    type=click.Choice(["low", "medium", "high", "critical"]),
    help="Filter alerts by severity",
)
def monitor(start, stop, status, alerts, severity):
    """Performance monitoring management."""

    async def run_monitor():
        manager = get_performance_integration_manager()

        if start:
            await start_performance_monitoring()
            click.echo("✅ Performance monitoring started")

        elif stop:
            await stop_performance_monitoring()
            click.echo("✅ Performance monitoring stopped")

        elif status:
            health = await manager.health_check()
            click.echo("📊 Performance System Status:")
            click.echo(f"Overall Status: {health['overall_status']}")

            for component, status_info in health["components"].items():
                click.echo(f"  {component}: {status_info['status']}")

            if health["recommendations"]:
                click.echo("\n💡 Recommendations:")
                for rec in health["recommendations"]:
                    click.echo(f"  • {rec}")

        elif alerts:
            active_alerts = manager.get_performance_alerts(severity=severity)

            if not active_alerts:
                click.echo("✅ No active performance alerts")
            else:
                click.echo(f"⚠️  {len(active_alerts)} active alerts:")

                for alert in active_alerts:
                    severity_icon = {
                        "low": "🔵",
                        "medium": "🟡",
                        "high": "🟠",
                        "critical": "🔴",
                    }.get(alert["severity"], "⚪")

                    click.echo(f"  {severity_icon} {alert['title']}")
                    click.echo(f"    Component: {alert['component']}")
                    click.echo(f"    Description: {alert['description']}")
                    click.echo(f"    Time: {alert['timestamp']}")
                    click.echo()

        else:
            # Show general stats
            stats = get_performance_stats()
            click.echo("📈 Performance Statistics:")

            if "optimization" in stats:
                opt_stats = stats["optimization"]
                cache_stats = opt_stats.get("cache_stats", {})

                click.echo(f"  Cache hit ratio: {cache_stats.get('hit_ratio', 0):.2%}")
                click.echo(
                    f"  Cache size: {cache_stats.get('total_size_mb', 0):.1f} MB"
                )

            if "monitoring" in stats:
                mon_stats = stats["monitoring"]
                click.echo(
                    f"  Total operations monitored: {mon_stats.get('total_operations', 0)}"
                )

                if "execution_time_stats" in mon_stats:
                    exec_stats = mon_stats["execution_time_stats"]
                    click.echo(
                        f"  Average execution time: {exec_stats.get('average', 0):.3f}s"
                    )
                    click.echo(
                        f"  95th percentile: {exec_stats.get('95th_percentile', 0):.3f}s"
                    )

    try:
        asyncio.run(run_monitor())
    except Exception as e:
        click.echo(f"❌ Monitoring command failed: {str(e)}", err=True)
        sys.exit(1)


@performance.command()
@click.option(
    "--time-period",
    "-t",
    default="24h",
    help="Time period for report (e.g., 24h, 7d, 30d)",
)
@click.option("--service", "-s", help="Specific service to report on")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["html", "json", "pdf"]),
    default="html",
    help="Report format",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="performance_reports",
    help="Output directory for reports",
)
@click.option(
    "--include-charts", is_flag=True, default=True, help="Include performance charts"
)
def report(time_period, service, format, output_dir, include_charts):
    """Generate performance reports."""
    click.echo(f"📊 Generating performance report for {time_period}...")

    async def generate_report():
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        manager = get_performance_integration_manager()

        # Generate report
        report_path = await manager.generate_performance_report(
            service_name=service, time_period=time_period, export_format=format
        )

        click.echo(f"✅ Report generated: {report_path}")

        # Show report summary
        if format == "json":
            with open(report_path) as f:
                report_data = json.load(f)

                click.echo("\n📋 Report Summary:")
                click.echo(
                    f"Overall Health Score: {report_data.get('overall_health_score', 0):.1f}/100"
                )
                click.echo(
                    f"Performance Grade: {report_data.get('performance_grade', 'N/A')}"
                )

                if "key_insights" in report_data:
                    click.echo("\n💡 Key Insights:")
                    for insight in report_data["key_insights"][:3]:  # Show top 3
                        click.echo(f"  • {insight}")

                if "critical_issues" in report_data and report_data["critical_issues"]:
                    click.echo("\n🚨 Critical Issues:")
                    for issue in report_data["critical_issues"]:
                        click.echo(f"  • {issue}")

        return report_path

    try:
        report_path = asyncio.run(generate_report())
        return report_path
    except Exception as e:
        click.echo(f"❌ Report generation failed: {str(e)}", err=True)
        sys.exit(1)


@performance.command()
@click.option("--cache-size", type=int, default=512, help="Cache size in MB")
@click.option(
    "--enable-parallel", is_flag=True, default=True, help="Enable parallel processing"
)
@click.option(
    "--enable-batch", is_flag=True, default=True, help="Enable batch processing"
)
@click.option(
    "--enable-memory-opt", is_flag=True, default=True, help="Enable memory optimization"
)
def optimize(cache_size, enable_parallel, enable_batch, enable_memory_opt):
    """Configure performance optimization settings."""
    click.echo("⚙️  Configuring performance optimization...")

    # Create optimization engine with specified settings
    optimization_engine = create_optimization_engine(
        cache_size_mb=cache_size,
        enable_all_optimizations=True,
        storage_path=Path("optimization_config"),
    )

    click.echo("✅ Optimization engine configured with:")
    click.echo(f"  Cache size: {cache_size} MB")
    click.echo(f"  Parallel processing: {'enabled' if enable_parallel else 'disabled'}")
    click.echo(f"  Batch processing: {'enabled' if enable_batch else 'disabled'}")
    click.echo(
        f"  Memory optimization: {'enabled' if enable_memory_opt else 'disabled'}"
    )

    # Show optimization stats
    stats = optimization_engine.get_optimization_stats()
    click.echo("\n📊 Current optimization stats:")
    cache_stats = stats.get("cache_stats", {})
    click.echo(f"  Cache entries: {cache_stats.get('total_entries', 0)}")
    click.echo(f"  Cache hit ratio: {cache_stats.get('hit_ratio', 0):.2%}")

    optimization_engine.cleanup()


if __name__ == "__main__":
    performance()
