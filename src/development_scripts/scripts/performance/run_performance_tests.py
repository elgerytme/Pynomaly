#!/usr/bin/env python3
"""
Local Performance Testing Runner

This script provides a convenient way to run performance tests locally
for development and validation purposes.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

console = Console()


class LocalPerformanceRunner:
    """Manages local performance test execution."""

    def __init__(self):
        self.app_process: subprocess.Popen | None = None
        self.base_url = "http://localhost:8000"
        self.app_ready = False

    def check_application_status(self) -> bool:
        """Check if the anomaly detection application is running."""
        try:
            import requests

            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_application(self, env: str = "development") -> bool:
        """Start the anomaly detection application locally."""
        console.print("🚀 Starting anomaly detection application...")

        try:
            # Set environment variables
            env_vars = os.environ.copy()
            env_vars.update(
                {
                    "PYNOMALY_ENV": env,
                    "PYNOMALY_LOG_LEVEL": "WARNING",
                    "UVICORN_LOG_LEVEL": "warning",
                }
            )

            # Start the application
            self.app_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "pynomaly.presentation.api.app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--log-level",
                    "warning",
                ],
                env=env_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for application to start
            for i in range(30):
                if self.check_application_status():
                    console.print("✅ Application started successfully")
                    self.app_ready = True
                    return True
                time.sleep(2)

            console.print("❌ Application failed to start within 60 seconds")
            return False

        except Exception as e:
            console.print(f"❌ Failed to start application: {e}")
            return False

    def stop_application(self):
        """Stop the running application."""
        if self.app_process:
            console.print("🛑 Stopping application...")
            self.app_process.terminate()
            try:
                self.app_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.app_process.kill()
            self.app_process = None
            self.app_ready = False
            console.print("✅ Application stopped")

    def create_test_config(self, scenario: str, users: int, duration: int) -> dict:
        """Create test configuration."""
        return {
            "base_url": self.base_url,
            "concurrent_users": users,
            "duration_seconds": duration,
            "ramp_up_seconds": min(duration // 6, 10),
            "scenario": scenario,
            "auth_enabled": False,
            "thresholds": {
                "max_response_time_ms": 1000.0,
                "max_error_rate_percent": 5.0,
                "min_throughput_rps": 1.0,
                "max_cpu_percent": 90.0,
                "max_memory_percent": 90.0,
            },
        }

    async def run_performance_test(self, config: dict, output_dir: str) -> bool:
        """Run a single performance test."""
        config_file = f"{output_dir}/test_config.json"

        # Save config to file
        os.makedirs(output_dir, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Import and run the load test
        try:
            sys.path.append(str(Path(__file__).parent))
            from load_testing_framework import LoadTestConfig, LoadTester

            test_config = LoadTestConfig(**config)
            tester = LoadTester(test_config)

            results = await tester.run_load_test()

            # Save results
            from load_testing_framework import LoadTestReporter

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            LoadTestReporter.save_json_report(
                results, f"{output_dir}/load_test_{timestamp}.json"
            )
            LoadTestReporter.save_ci_report(
                results, f"{output_dir}/load_test_ci_{timestamp}.json"
            )
            LoadTestReporter.print_console_report(results)

            return results.thresholds_passed

        except Exception as e:
            console.print(f"❌ Performance test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def interactive_test_configuration(self) -> dict:
        """Interactive configuration for performance tests."""
        console.print(Panel.fit("🎯 Performance Test Configuration", style="bold blue"))

        # Test scenario
        scenario_table = Table(title="Available Test Scenarios")
        scenario_table.add_column("Scenario", style="cyan")
        scenario_table.add_column("Description", style="yellow")

        scenarios = {
            "health": "Simple health check endpoint testing",
            "detection": "Anomaly detection API testing",
            "training": "Model training API testing",
            "mixed": "Mixed workload (recommended)",
        }

        for scenario, description in scenarios.items():
            scenario_table.add_row(scenario, description)

        console.print(scenario_table)

        scenario = Prompt.ask(
            "Select test scenario", choices=list(scenarios.keys()), default="mixed"
        )

        # Test parameters
        users = IntPrompt.ask(
            "Number of concurrent users", default=5, show_default=True
        )
        duration = IntPrompt.ask(
            "Test duration (seconds)", default=30, show_default=True
        )

        # Show configuration summary
        config_table = Table(title="Test Configuration Summary")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="magenta")

        config_table.add_row("Scenario", scenario)
        config_table.add_row("Concurrent Users", str(users))
        config_table.add_row("Duration", f"{duration} seconds")
        config_table.add_row("Base URL", self.base_url)

        console.print(config_table)

        if not Confirm.ask("Proceed with this configuration?", default=True):
            return self.interactive_test_configuration()

        return self.create_test_config(scenario, users, duration)

    async def run_baseline_establishment(self, output_dir: str) -> bool:
        """Run baseline establishment tests."""
        console.print(
            Panel.fit("📊 Establishing Performance Baselines", style="bold green")
        )

        if not Confirm.ask(
            "This will run comprehensive baseline tests (may take 10-15 minutes). Continue?"
        ):
            return False

        try:
            sys.path.append(str(Path(__file__).parent))
            from establish_baselines import PerformanceBaseline

            baseline_manager = PerformanceBaseline(self.base_url)
            baseline_config = await baseline_manager.establish_baselines()

            baseline_manager.save_baseline_config(f"{output_dir}/baseline_config.json")
            baseline_manager.generate_ci_config(
                f"{output_dir}/ci_performance_config.json"
            )

            console.print("✅ Baseline establishment completed successfully")
            return True

        except Exception as e:
            console.print(f"❌ Baseline establishment failed: {e}")
            return False

    def analyze_results(self, results_dir: str):
        """Analyze performance test results."""
        try:
            sys.path.append(str(Path(__file__).parent))
            from analyze_performance_trends import PerformanceTrendAnalyzer

            analyzer = PerformanceTrendAnalyzer(results_dir)
            loaded_count = analyzer.load_test_results()

            if loaded_count < 1:
                console.print("❌ No test results found for analysis")
                return

            analyzer.analyze_trends()
            analyzer.print_analysis_summary()

            analysis_dir = f"{results_dir}/analysis"
            analyzer.save_analysis_report(f"{analysis_dir}/performance_analysis.json")

            if Confirm.ask("Generate performance charts?", default=True):
                analyzer.generate_visualizations(analysis_dir)

        except Exception as e:
            console.print(f"❌ Analysis failed: {e}")


async def main():
    """Main entry point for local performance testing."""
    parser = argparse.ArgumentParser(description="Local Performance Testing Runner")
    parser.add_argument(
        "--auto-start", action="store_true", help="Automatically start the application"
    )
    parser.add_argument(
        "--scenario",
        choices=["health", "detection", "training", "mixed"],
        help="Test scenario",
    )
    parser.add_argument("--users", type=int, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, help="Test duration in seconds")
    parser.add_argument(
        "--baseline", action="store_true", help="Run baseline establishment"
    )
    parser.add_argument("--analyze", help="Analyze results from directory")
    parser.add_argument(
        "--output-dir", default="local_performance_results", help="Output directory"
    )

    args = parser.parse_args()

    runner = LocalPerformanceRunner()

    # Handle interrupt signal
    def signal_handler(signum, frame):
        console.print("\n🛑 Interrupted by user")
        runner.stop_application()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Analysis mode
        if args.analyze:
            runner.analyze_results(args.analyze)
            return

        # Check if application is already running
        if runner.check_application_status():
            console.print("✅ Application is already running")
            runner.app_ready = True
        elif args.auto_start or Confirm.ask(
            "Application not running. Start it now?", default=True
        ):
            if not runner.start_application():
                console.print("❌ Failed to start application")
                sys.exit(1)
        else:
            console.print("❌ Application must be running to perform tests")
            sys.exit(1)

        # Baseline establishment mode
        if args.baseline:
            success = await runner.run_baseline_establishment(args.output_dir)
            if not success:
                sys.exit(1)
        else:
            # Regular performance testing
            if args.scenario and args.users and args.duration:
                # Non-interactive mode
                config = runner.create_test_config(
                    args.scenario, args.users, args.duration
                )
            else:
                # Interactive mode
                config = runner.interactive_test_configuration()

            # Run the test
            success = await runner.run_performance_test(config, args.output_dir)
            if not success:
                console.print("❌ Performance test failed or thresholds violated")
                sys.exit(1)

        # Offer to analyze results
        if Confirm.ask("Analyze test results now?", default=True):
            runner.analyze_results(args.output_dir)

        console.print(Panel.fit("✅ Performance Testing Completed", style="bold green"))

    except Exception as e:
        console.print(f"❌ Performance testing failed: {e}", style="bold red")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if args.auto_start and runner.app_process:
            runner.stop_application()


if __name__ == "__main__":
    asyncio.run(main())
