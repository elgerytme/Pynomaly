#!/usr/bin/env python3
"""
Performance Baseline Establishment Script

This script establishes performance baselines for the anomaly detection application by:
- Running comprehensive load tests across different scenarios
- Analyzing system performance under various loads
- Setting optimal performance thresholds based on empirical data
- Generating baseline configuration for CI/CD validation
"""

import asyncio
import json
import os
import statistics
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the performance directory to the path to import our load testing framework
sys.path.append(os.path.dirname(__file__))
from load_testing_framework import LoadTestConfig, LoadTester, LoadTestResults

console = Console()


class PerformanceBaseline:
    """Manages performance baseline establishment and validation."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: dict[str, LoadTestResults] = {}
        self.baseline_config: dict[str, Any] = {}

    async def establish_baselines(self) -> dict[str, Any]:
        """Run comprehensive baseline tests and analyze results."""
        console.print(
            Panel.fit("🎯 Establishing Performance Baselines", style="bold blue")
        )

        # Define baseline test scenarios
        test_scenarios = [
            {
                "name": "health_check_baseline",
                "scenario": "health",
                "users": 5,
                "duration": 30,
                "description": "Health check endpoint baseline",
            },
            {
                "name": "authentication_baseline",
                "scenario": "auth",
                "users": 10,
                "duration": 60,
                "description": "Authentication endpoint baseline",
            },
            {
                "name": "detection_light_load",
                "scenario": "detection",
                "users": 5,
                "duration": 120,
                "description": "Detection API under light load",
            },
            {
                "name": "detection_moderate_load",
                "scenario": "detection",
                "users": 15,
                "duration": 180,
                "description": "Detection API under moderate load",
            },
            {
                "name": "mixed_workload_baseline",
                "scenario": "mixed",
                "users": 20,
                "duration": 300,
                "description": "Mixed workload baseline",
            },
            {
                "name": "sustained_load_test",
                "scenario": "mixed",
                "users": 25,
                "duration": 600,
                "description": "Sustained load test",
            },
        ]

        # Run baseline tests
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                "Running baseline tests", total=len(test_scenarios)
            )

            for scenario_config in test_scenarios:
                progress.update(
                    overall_task, description=f"Running {scenario_config['name']}"
                )

                console.print(f"\n📊 Running {scenario_config['description']}")

                # Create load test configuration
                config = LoadTestConfig(
                    base_url=self.base_url,
                    concurrent_users=scenario_config["users"],
                    duration_seconds=scenario_config["duration"],
                    ramp_up_seconds=min(scenario_config["duration"] // 6, 30),
                    scenario=scenario_config["scenario"],
                    auth_enabled=True,
                    # Set relaxed thresholds for baseline establishment
                    thresholds={
                        "max_response_time_ms": 2000.0,
                        "max_error_rate_percent": 5.0,
                        "min_throughput_rps": 1.0,
                        "max_cpu_percent": 95.0,
                        "max_memory_percent": 95.0,
                    },
                )

                # Run the test
                try:
                    tester = LoadTester(config)
                    result = await tester.run_load_test()
                    self.results[scenario_config["name"]] = result

                    console.print(f"✅ {scenario_config['name']} completed")
                    console.print(
                        f"   • Avg Response Time: {result.avg_response_time_ms:.1f}ms"
                    )
                    console.print(f"   • Throughput: {result.throughput_rps:.1f} RPS")
                    console.print(f"   • Error Rate: {result.error_rate_percent:.1f}%")

                except Exception as e:
                    console.print(f"❌ {scenario_config['name']} failed: {e}")
                    continue

                progress.update(overall_task, advance=1)

        # Analyze results and establish baselines
        self.baseline_config = self._analyze_baseline_results()

        return self.baseline_config

    def _analyze_baseline_results(self) -> dict[str, Any]:
        """Analyze baseline test results and establish optimal thresholds."""
        console.print(Panel.fit("📈 Analyzing Baseline Results", style="bold green"))

        if not self.results:
            raise RuntimeError("No baseline results available for analysis")

        # Collect metrics from all tests
        response_times = []
        throughputs = []
        error_rates = []
        p95_times = []
        p99_times = []

        for name, result in self.results.items():
            if result.successful_requests > 0:  # Only include successful tests
                response_times.append(result.avg_response_time_ms)
                throughputs.append(result.throughput_rps)
                error_rates.append(result.error_rate_percent)
                p95_times.append(result.p95_response_time_ms)
                p99_times.append(result.p99_response_time_ms)

        if not response_times:
            raise RuntimeError("No successful baseline tests to analyze")

        # Calculate baseline statistics
        baseline_stats = {
            "response_time": {
                "avg": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p75": statistics.quantiles(response_times, n=4)[2]
                if len(response_times) > 1
                else response_times[0],
                "p95": statistics.quantiles(response_times, n=20)[18]
                if len(response_times) > 1
                else response_times[0],
                "max": max(response_times),
            },
            "throughput": {
                "avg": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "min": min(throughputs),
                "p25": statistics.quantiles(throughputs, n=4)[0]
                if len(throughputs) > 1
                else throughputs[0],
            },
            "error_rate": {
                "avg": statistics.mean(error_rates),
                "median": statistics.median(error_rates),
                "p95": statistics.quantiles(error_rates, n=20)[18]
                if len(error_rates) > 1
                else error_rates[0],
                "max": max(error_rates),
            },
        }

        # Establish optimal thresholds based on baseline data
        # Use conservative thresholds: allow for some degradation from baseline
        optimal_thresholds = {
            "max_response_time_ms": baseline_stats["response_time"]["p95"]
            * 1.5,  # 50% buffer from P95
            "max_error_rate_percent": max(
                baseline_stats["error_rate"]["p95"] * 2, 1.0
            ),  # 2x P95 error rate, min 1%
            "min_throughput_rps": baseline_stats["throughput"]["p25"]
            * 0.8,  # 80% of P25 throughput
            "max_cpu_percent": 80.0,  # Standard threshold
            "max_memory_percent": 85.0,  # Standard threshold
            "max_p95_response_time_ms": baseline_stats["response_time"]["p95"]
            * 2.0,  # 2x P95 for P95 threshold
            "max_p99_response_time_ms": max(p99_times) * 1.2
            if p99_times
            else 1000.0,  # 20% buffer from worst P99
        }

        # Create baseline configuration
        baseline_config = {
            "established_at": datetime.now().isoformat(),
            "baseline_statistics": baseline_stats,
            "optimal_thresholds": optimal_thresholds,
            "test_scenarios": list(self.results.keys()),
            "environment_info": self._get_environment_info(),
            "recommendations": self._generate_recommendations(
                baseline_stats, optimal_thresholds
            ),
        }

        # Display baseline summary
        self._display_baseline_summary(baseline_stats, optimal_thresholds)

        return baseline_config

    def _get_environment_info(self) -> dict[str, Any]:
        """Gather environment information for baseline context."""
        import platform

        import psutil

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "test_timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
        }

    def _generate_recommendations(
        self, stats: dict[str, Any], thresholds: dict[str, Any]
    ) -> list[str]:
        """Generate performance recommendations based on baseline analysis."""
        recommendations = []

        # Response time recommendations
        avg_response_time = stats["response_time"]["avg"]
        if avg_response_time > 200:
            recommendations.append(
                f"Average response time ({avg_response_time:.1f}ms) is high. "
                "Consider optimizing database queries and adding caching."
            )

        if avg_response_time > 500:
            recommendations.append(
                "Response times indicate potential performance bottlenecks. "
                "Review application profiling and consider horizontal scaling."
            )

        # Throughput recommendations
        avg_throughput = stats["throughput"]["avg"]
        if avg_throughput < 10:
            recommendations.append(
                f"Low throughput ({avg_throughput:.1f} RPS) detected. "
                "Consider connection pooling and async processing improvements."
            )

        # Error rate recommendations
        avg_error_rate = stats["error_rate"]["avg"]
        if avg_error_rate > 1.0:
            recommendations.append(
                f"Error rate ({avg_error_rate:.1f}%) is above acceptable threshold. "
                "Review error handling and system stability."
            )

        # General recommendations
        recommendations.extend(
            [
                "Monitor these thresholds in production and adjust based on actual usage patterns.",
                "Run baseline tests regularly to detect performance regressions.",
                "Consider implementing circuit breakers for external dependencies.",
                "Set up alerts for threshold violations in production monitoring.",
            ]
        )

        return recommendations

    def _display_baseline_summary(
        self, stats: dict[str, Any], thresholds: dict[str, Any]
    ):
        """Display comprehensive baseline summary."""
        # Baseline statistics table
        stats_table = Table(title="Baseline Performance Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Average", style="magenta")
        stats_table.add_column("Median", style="yellow")
        stats_table.add_column("P95", style="green")

        stats_table.add_row(
            "Response Time (ms)",
            f"{stats['response_time']['avg']:.1f}",
            f"{stats['response_time']['median']:.1f}",
            f"{stats['response_time']['p95']:.1f}",
        )
        stats_table.add_row(
            "Throughput (RPS)",
            f"{stats['throughput']['avg']:.1f}",
            f"{stats['throughput']['median']:.1f}",
            "-",
        )
        stats_table.add_row(
            "Error Rate (%)",
            f"{stats['error_rate']['avg']:.2f}",
            f"{stats['error_rate']['median']:.2f}",
            f"{stats['error_rate']['p95']:.2f}",
        )

        console.print(stats_table)

        # Recommended thresholds table
        thresholds_table = Table(title="Recommended Performance Thresholds")
        thresholds_table.add_column("Threshold", style="cyan")
        thresholds_table.add_column("Value", style="magenta")
        thresholds_table.add_column("Rationale", style="yellow")

        thresholds_table.add_row(
            "Max Response Time",
            f"{thresholds['max_response_time_ms']:.1f}ms",
            "1.5x P95 baseline",
        )
        thresholds_table.add_row(
            "Max Error Rate",
            f"{thresholds['max_error_rate_percent']:.1f}%",
            "2x P95 baseline error rate",
        )
        thresholds_table.add_row(
            "Min Throughput",
            f"{thresholds['min_throughput_rps']:.1f} RPS",
            "80% of P25 baseline throughput",
        )
        thresholds_table.add_row(
            "Max P95 Response Time",
            f"{thresholds['max_p95_response_time_ms']:.1f}ms",
            "2x baseline P95",
        )

        console.print(thresholds_table)

    def save_baseline_config(self, output_path: str):
        """Save baseline configuration to file."""
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(self.baseline_config, f, indent=2)

        console.print(f"📄 Baseline configuration saved to: {output_path}")

    def generate_ci_config(self, output_path: str):
        """Generate CI/CD configuration with baseline thresholds."""
        if not self.baseline_config:
            raise RuntimeError("No baseline configuration available")

        ci_config = {
            "load_test_config": {
                "base_url": "${PYNOMALY_TEST_URL:-http://localhost:8000}",
                "concurrent_users": 10,
                "duration_seconds": 60,
                "scenario": "mixed",
                "auth_enabled": True,
                "thresholds": self.baseline_config["optimal_thresholds"],
            },
            "performance_gates": {
                "enabled": True,
                "fail_on_threshold_violation": True,
                "baseline_comparison": True,
            },
            "baseline_metadata": {
                "established_at": self.baseline_config["established_at"],
                "environment": self.baseline_config["environment_info"],
            },
        }

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(ci_config, f, indent=2)

        console.print(f"🤖 CI/CD configuration saved to: {output_path}")


async def main():
    """Main entry point for baseline establishment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Establish Pynomaly Performance Baselines"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for baseline testing",
    )
    parser.add_argument(
        "--output-dir",
        default="performance_baselines",
        help="Output directory for baseline configuration",
    )
    parser.add_argument(
        "--skip-heavy-tests",
        action="store_true",
        help="Skip long-running tests for quick baseline",
    )

    args = parser.parse_args()

    try:
        # Create baseline manager
        baseline_manager = PerformanceBaseline(args.base_url)

        # Check if application is running
        console.print(f"🔍 Checking application availability at {args.base_url}")
        try:
            import requests

            response = requests.get(f"{args.base_url}/health", timeout=10)
            if response.status_code != 200:
                console.print("❌ Application health check failed", style="bold red")
                sys.exit(1)
            console.print("✅ Application is running and healthy")
        except Exception as e:
            console.print(f"❌ Cannot connect to application: {e}", style="bold red")
            console.print(
                "Please ensure the anomaly detection application is running at the specified URL"
            )
            sys.exit(1)

        # Establish baselines
        baseline_config = await baseline_manager.establish_baselines()

        # Save configurations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_path = f"{args.output_dir}/baseline_config_{timestamp}.json"
        ci_config_path = f"{args.output_dir}/ci_performance_config.json"

        baseline_manager.save_baseline_config(baseline_path)
        baseline_manager.generate_ci_config(ci_config_path)

        console.print(
            Panel.fit(
                "✅ Performance Baselines Established Successfully!", style="bold green"
            )
        )
        console.print("Next steps:")
        console.print("1. Review the generated thresholds and adjust as needed")
        console.print("2. Integrate the CI configuration into your pipeline")
        console.print("3. Set up monitoring alerts using these thresholds")
        console.print("4. Run regular baseline validation tests")

    except Exception as e:
        console.print(f"❌ Baseline establishment failed: {e}", style="bold red")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
