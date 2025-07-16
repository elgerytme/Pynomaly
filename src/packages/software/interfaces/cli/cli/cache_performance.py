"""CLI commands for cache performance analysis and optimization."""

import json
import time
from typing import Any

import click

from interfaces.infrastructure.cache.optimized_key_generator import (
    OptimizedCacheKeyGenerator,
)
from interfaces.infrastructure.cache.performance_utils import (
    enable_cache_optimizations,
    get_cache_performance_report,
    get_health_monitor,
    get_performance_optimizer,
)


@click.group()
def cache_performance():
    """Cache performance analysis and optimization commands."""
    pass


@cache_performance.command()
@click.option('--format', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.option('--detailed', is_flag=True, help='Show detailed performance metrics')
def report(format: str, detailed: bool) -> None:
    """Generate cache performance report."""
    try:
        report_data = get_cache_performance_report()

        if format == 'json':
            click.echo(json.dumps(report_data, indent=2))
        else:
            _display_performance_table(report_data, detailed)

    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)


@cache_performance.command()
def enable_optimizations() -> None:
    """Enable cache performance optimizations."""
    try:
        enable_cache_optimizations()
        click.echo("✓ Cache performance optimizations enabled")

        # Get initial stats
        stats = OptimizedCacheKeyGenerator.get_performance_stats()
        if stats.get("status") != "no_data":
            click.echo(f"Cache size: {stats.get('cache_sizes', {}).get('signature_cache', 0)} cached functions")

    except Exception as e:
        click.echo(f"Error enabling optimizations: {e}", err=True)


@cache_performance.command()
def health_check() -> None:
    """Check cache system health."""
    try:
        health_monitor = get_health_monitor()
        health_report = health_monitor.check_health()

        status = health_report.get("status", "unknown")
        timestamp = health_report.get("timestamp", time.time())

        # Status indicator
        status_emoji = {
            "healthy": "✓",
            "degraded": "⚠",
            "unhealthy": "✗",
            "unknown": "?",
        }

        click.echo(f"{status_emoji.get(status, '?')} Cache Health: {status.upper()}")
        click.echo(f"Last Check: {time.ctime(timestamp)}")

        # Show statistics if available
        stats = health_report.get("statistics", {})
        if stats.get("status") != "no_data":
            avg_time = stats.get("average_generation_time_ms", 0)
            p95_time = stats.get("p95_generation_time_ms", 0)
            total_keys = stats.get("total_generated_keys", 0)

            click.echo(f"Average key generation time: {avg_time:.2f}ms")
            click.echo(f"P95 key generation time: {p95_time:.2f}ms")
            click.echo(f"Total keys generated: {total_keys}")

            # Show key size distribution
            key_dist = stats.get("key_size_distribution", {})
            if key_dist:
                click.echo("\nKey Size Distribution:")
                for size_range, count in key_dist.items():
                    click.echo(f"  {size_range}: {count}")

        # Show recent alerts
        alerts = health_report.get("recent_alerts", [])
        if alerts:
            click.echo(f"\nRecent Alerts ({len(alerts)}):")
            for alert in alerts[-3:]:  # Show last 3 alerts
                level = alert.get("level", "info")
                message = alert.get("message", "")
                timestamp = alert.get("timestamp", time.time())
                click.echo(f"  [{level.upper()}] {message} ({time.ctime(timestamp)})")

        # Show health trend
        trend = health_report.get("health_trend", "unknown")
        trend_emoji = {
            "stable_healthy": "✓",
            "stable_unhealthy": "✗",
            "improving": "↗",
            "degrading": "↘",
            "fluctuating": "↕",
            "insufficient_data": "?",
        }
        click.echo(f"Health Trend: {trend_emoji.get(trend, '?')} {trend.replace('_', ' ').title()}")

        # Show recommendations
        recommendations = stats.get("performance_recommendations", [])
        if recommendations:
            click.echo("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                click.echo(f"  {i}. {rec}")

    except Exception as e:
        click.echo(f"Error checking health: {e}", err=True)


@cache_performance.command()
@click.option('--function-name', help='Specific function to analyze')
@click.option('--top', type=int, default=10, help='Number of top functions to show')
def analyze(function_name: str, top: int) -> None:
    """Analyze cache performance by function."""
    try:
        optimizer = get_performance_optimizer()
        report = optimizer.get_performance_report()

        if not report.get("optimization_enabled", False):
            click.echo("⚠ Cache optimizations not enabled. Run 'cache-performance enable-optimizations' first.")
            return

        function_breakdown = report.get("function_breakdown", {})
        if not function_breakdown:
            click.echo("No function-specific data available yet.")
            return

        if function_name:
            # Show specific function analysis
            if function_name in function_breakdown:
                func_data = function_breakdown[function_name]
                click.echo(f"Function: {function_name}")
                click.echo(f"  Call count: {func_data.get('count', 0)}")
                click.echo(f"  Average time: {func_data.get('avg_time_ms', 0):.2f}ms")
                click.echo(f"  Maximum time: {func_data.get('max_time_ms', 0):.2f}ms")
            else:
                click.echo(f"Function '{function_name}' not found in performance data.")
        else:
            # Show top functions by performance
            sorted_functions = sorted(
                function_breakdown.items(),
                key=lambda x: x[1].get('avg_time_ms', 0),
                reverse=True
            )[:top]

            click.echo(f"Top {top} Functions by Average Key Generation Time:")
            click.echo(f"{'Function':<30} {'Calls':<8} {'Avg Time':<12} {'Max Time':<12}")
            click.echo("-" * 70)

            for func_name, func_data in sorted_functions:
                calls = func_data.get('count', 0)
                avg_time = func_data.get('avg_time_ms', 0)
                max_time = func_data.get('max_time_ms', 0)

                click.echo(f"{func_name:<30} {calls:<8} {avg_time:<12.2f} {max_time:<12.2f}")

    except Exception as e:
        click.echo(f"Error analyzing performance: {e}", err=True)


@cache_performance.command()
@click.confirmation_option(prompt='Are you sure you want to clear all cache performance data?')
def clear_stats() -> None:
    """Clear all performance statistics."""
    try:
        OptimizedCacheKeyGenerator.clear_cache()
        click.echo("✓ Performance statistics cleared")

    except Exception as e:
        click.echo(f"Error clearing statistics: {e}", err=True)


@cache_performance.command()
@click.option('--interval', type=int, default=5, help='Monitoring interval in seconds')
@click.option('--duration', type=int, default=60, help='Monitoring duration in seconds')
def monitor(interval: int, duration: int) -> None:
    """Monitor cache performance in real-time."""
    try:
        click.echo(f"Monitoring cache performance for {duration} seconds (interval: {interval}s)")
        click.echo("Press Ctrl+C to stop early")

        start_time = time.time()
        last_stats = OptimizedCacheKeyGenerator.get_performance_stats()

        while time.time() - start_time < duration:
            time.sleep(interval)

            current_stats = OptimizedCacheKeyGenerator.get_performance_stats()

            if current_stats.get("status") != "no_data":
                # Calculate deltas
                current_keys = current_stats.get("total_generated_keys", 0)
                last_keys = last_stats.get("total_generated_keys", 0) if last_stats.get("status") != "no_data" else 0
                keys_per_second = (current_keys - last_keys) / interval

                avg_time = current_stats.get("average_generation_time_ms", 0)

                elapsed = int(time.time() - start_time)
                click.echo(f"[{elapsed:3d}s] Keys/sec: {keys_per_second:6.1f}, Avg time: {avg_time:6.2f}ms")

                last_stats = current_stats
            else:
                elapsed = int(time.time() - start_time)
                click.echo(f"[{elapsed:3d}s] No data available")

        click.echo("Monitoring completed")

    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped by user")
    except Exception as e:
        click.echo(f"Error during monitoring: {e}", err=True)


@cache_performance.command()
@click.option('--output', type=click.Path(), help='Output file for benchmark results')
def benchmark() -> None:
    """Run cache performance benchmarks."""
    try:
        click.echo("Running cache performance benchmarks...")

        # Define test functions
        def simple_function(arg1: str, arg2: int) -> str:
            return f"{arg1}_{arg2}"

        def complex_function(arg1: str, arg2: list, arg3: dict) -> str:
            return f"{arg1}_{len(arg2)}_{len(arg3)}"

        # Benchmark data
        benchmark_results = {
            "timestamp": time.time(),
            "tests": {}
        }

        # Test 1: Simple function performance
        click.echo("1. Testing simple function performance...")
        times = []
        for i in range(1000):
            start_time = time.perf_counter()
            OptimizedCacheKeyGenerator.generate_key(
                simple_function, (f"test_{i}", i), {}, "bench"
            )
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        benchmark_results["tests"]["simple_function"] = {
            "iterations": 1000,
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)],
        }

        # Test 2: Complex function performance
        click.echo("2. Testing complex function performance...")
        times = []
        for i in range(500):
            large_list = list(range(i % 50))
            large_dict = {f"key_{j}": f"value_{j}" for j in range(i % 20)}

            start_time = time.perf_counter()
            OptimizedCacheKeyGenerator.generate_key(
                complex_function, (f"test_{i}", large_list, large_dict), {}, "bench"
            )
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        benchmark_results["tests"]["complex_function"] = {
            "iterations": 500,
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)],
        }

        # Test 3: Cache hit performance
        click.echo("3. Testing cache hit performance...")
        # Pre-warm cache
        OptimizedCacheKeyGenerator.optimize_for_function(simple_function)

        times = []
        for i in range(1000):
            start_time = time.perf_counter()
            OptimizedCacheKeyGenerator.generate_key(
                simple_function, ("cached_test", 42), {}, "bench"
            )
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        benchmark_results["tests"]["cache_hits"] = {
            "iterations": 1000,
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)],
        }

        # Display results
        click.echo("\nBenchmark Results:")
        click.echo("=" * 50)

        for test_name, results in benchmark_results["tests"].items():
            click.echo(f"\n{test_name.replace('_', ' ').title()}:")
            click.echo(f"  Iterations: {results['iterations']}")
            click.echo(f"  Average: {results['avg_time_ms']:.3f}ms")
            click.echo(f"  Min: {results['min_time_ms']:.3f}ms")
            click.echo(f"  Max: {results['max_time_ms']:.3f}ms")
            click.echo(f"  P95: {results['p95_time_ms']:.3f}ms")

        # Save results if requested
        if click.get_current_context().params.get('output'):
            output_file = click.get_current_context().params['output']
            with open(output_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            click.echo(f"\nResults saved to {output_file}")

    except Exception as e:
        click.echo(f"Error running benchmarks: {e}", err=True)


def _display_performance_table(report_data: dict[str, Any], detailed: bool) -> None:
    """Display performance report in table format."""
    optimizer_report = report_data.get("optimizer_report", {})
    health_report = report_data.get("health_report", {})
    key_generator_stats = report_data.get("key_generator_stats", {})

    click.echo("Cache Performance Report")
    click.echo("=" * 50)

    # Optimization status
    optimized = optimizer_report.get("optimization_enabled", False)
    click.echo(f"Optimization Status: {'✓ Enabled' if optimized else '✗ Disabled'}")

    # Health status
    health_status = health_report.get("status", "unknown")
    health_emoji = {"healthy": "✓", "degraded": "⚠", "unhealthy": "✗", "unknown": "?"}
    click.echo(f"Health Status: {health_emoji.get(health_status, '?')} {health_status.upper()}")

    # Performance metrics
    if optimizer_report.get("total_measurements", 0) > 0:
        perf_data = optimizer_report.get("key_generation_performance", {})

        click.echo("\nPerformance Metrics:")
        click.echo(f"  Total measurements: {optimizer_report.get('total_measurements', 0)}")
        click.echo(f"  Average time: {perf_data.get('avg_time_ms', 0):.2f}ms")
        click.echo(f"  P95 time: {perf_data.get('p95_time_ms', 0):.2f}ms")
        click.echo(f"  Max time: {perf_data.get('max_time_ms', 0):.2f}ms")

        # Key characteristics
        key_chars = optimizer_report.get("key_characteristics", {})
        click.echo(f"  Average key length: {key_chars.get('avg_length', 0):.0f} chars")
        click.echo(f"  Max key length: {key_chars.get('max_length', 0):.0f} chars")

    # Internal stats
    if key_generator_stats.get("status") != "no_data":
        click.echo("\nInternal Cache Stats:")
        cache_sizes = key_generator_stats.get("cache_sizes", {})
        click.echo(f"  Cached signatures: {cache_sizes.get('signature_cache', 0)}")
        click.echo(f"  Cached function names: {cache_sizes.get('func_name_cache', 0)}")

        # Key size distribution
        if detailed:
            key_dist = key_generator_stats.get("key_size_distribution", {})
            if key_dist:
                click.echo("\nKey Size Distribution:")
                for size_range, count in key_dist.items():
                    click.echo(f"  {size_range}: {count}")

    # Recommendations
    recommendations = []
    recommendations.extend(optimizer_report.get("recommendations", []))
    recommendations.extend(key_generator_stats.get("performance_recommendations", []))

    if recommendations:
        click.echo("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            click.echo(f"  {i}. {rec}")

    # Function breakdown (if detailed)
    if detailed:
        function_breakdown = optimizer_report.get("function_breakdown", {})
        if function_breakdown:
            click.echo("\nTop Functions by Performance:")
            sorted_functions = sorted(
                function_breakdown.items(),
                key=lambda x: x[1].get('avg_time_ms', 0),
                reverse=True
            )[:5]

            for func_name, func_data in sorted_functions:
                calls = func_data.get('count', 0)
                avg_time = func_data.get('avg_time_ms', 0)
                click.echo(f"  {func_name}: {calls} calls, {avg_time:.2f}ms avg")


if __name__ == "__main__":
    cache_performance()
