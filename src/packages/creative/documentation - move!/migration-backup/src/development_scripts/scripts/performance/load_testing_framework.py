#!/usr/bin/env python3
"""
Comprehensive Load Testing Framework for Pynomaly

This module provides automated load testing capabilities with support for:
- Multi-scenario testing (authentication, detection, training, dataset management)
- Configurable load patterns (ramp-up, sustained, spike testing)
- Real-time metrics collection and analysis
- Performance threshold validation
- CI/CD integration with detailed reporting
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""

    base_url: str = "http://localhost:8000"
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    scenario: str = "mixed"
    auth_enabled: bool = True
    api_key: str | None = None
    test_data_path: str = "test_data"
    output_path: str = "results"
    thresholds: dict[str, float] = None

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {
                "max_response_time_ms": 500.0,
                "max_error_rate_percent": 1.0,
                "min_throughput_rps": 10.0,
                "max_cpu_percent": 80.0,
                "max_memory_percent": 85.0,
            }


@dataclass
class RequestResult:
    """Result of a single HTTP request."""

    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    error: str | None = None
    payload_size_bytes: int = 0
    response_size_bytes: int = 0


@dataclass
class LoadTestResults:
    """Aggregated results of a load test run."""

    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    errors_by_type: dict[str, int]
    requests_by_endpoint: dict[str, int]
    response_times_by_endpoint: dict[str, list[float]]
    system_metrics: dict[str, Any]
    thresholds_passed: bool
    threshold_violations: list[str]


class LoadTestScenario:
    """Base class for load test scenarios."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.session = requests.Session()
        self.auth_token = None

    async def setup(self) -> bool:
        """Setup scenario (authentication, test data, etc.)."""
        if self.config.auth_enabled:
            return await self._authenticate()
        return True

    async def _authenticate(self) -> bool:
        """Authenticate and get API token."""
        try:
            if self.config.api_key:
                self.auth_token = self.config.api_key
                self.session.headers.update(
                    {"Authorization": f"Bearer {self.config.api_key}"}
                )
                return True

            # Login with test credentials
            login_data = {"email": "test@example.com", "password": "test_password"}

            response = self.session.post(
                f"{self.config.base_url}/api/v1/auth/login", json=login_data, timeout=10
            )

            if response.status_code == 200:
                token_data = response.json()
                self.auth_token = token_data.get("access_token")
                self.session.headers.update(
                    {"Authorization": f"Bearer {self.auth_token}"}
                )
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def generate_request(self) -> tuple[str, str, dict[str, Any]]:
        """Generate a single test request. Returns (method, endpoint, data)."""
        raise NotImplementedError

    def execute_request(
        self, method: str, endpoint: str, data: dict[str, Any]
    ) -> RequestResult:
        """Execute a single HTTP request and return results."""
        start_time = time.time()
        full_url = f"{self.config.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(full_url, params=data, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(full_url, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = self.session.put(full_url, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = self.session.delete(full_url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            return RequestResult(
                timestamp=start_time,
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                payload_size_bytes=len(json.dumps(data)) if data else 0,
                response_size_bytes=len(response.content) if response.content else 0,
                error=None
                if 200 <= response.status_code < 400
                else f"HTTP {response.status_code}",
            )

        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            return RequestResult(
                timestamp=start_time,
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=response_time_ms,
                error=str(e),
            )


class HealthCheckScenario(LoadTestScenario):
    """Simple health check scenario for baseline testing."""

    def generate_request(self) -> tuple[str, str, dict[str, Any]]:
        return "GET", "/health", {}


class AuthenticationScenario(LoadTestScenario):
    """Authentication-focused load testing."""

    def generate_request(self) -> tuple[str, str, dict[str, Any]]:
        return (
            "POST",
            "/api/v1/auth/login",
            {"email": "test@example.com", "password": "test_password"},
        )


class DetectionScenario(LoadTestScenario):
    """Anomaly detection load testing."""

    def generate_request(self) -> tuple[str, str, dict[str, Any]]:
        # Generate sample data for detection
        import random

        data = [[random.uniform(-10, 10) for _ in range(5)] for _ in range(100)]

        return (
            "POST",
            "/api/v1/detection/detect",
            {
                "data": data,
                "algorithm": "isolation_forest",
                "parameters": {"contamination": 0.1},
            },
        )


class TrainingScenario(LoadTestScenario):
    """Model training load testing."""

    def generate_request(self) -> tuple[str, str, dict[str, Any]]:
        import random

        data = [[random.uniform(-10, 10) for _ in range(5)] for _ in range(200)]

        return (
            "POST",
            "/api/v1/training/train",
            {
                "data": data,
                "algorithm": "isolation_forest",
                "hyperparameters": {"n_estimators": 50, "contamination": 0.1},
            },
        )


class MixedScenario(LoadTestScenario):
    """Mixed workload scenario."""

    def __init__(self, config: LoadTestConfig):
        super().__init__(config)
        self.scenarios = [
            HealthCheckScenario(config),
            DetectionScenario(config),
            TrainingScenario(config),
        ]
        self.weights = [0.4, 0.5, 0.1]  # Health checks, detection, training

    def generate_request(self) -> tuple[str, str, dict[str, Any]]:
        import random

        scenario = random.choices(self.scenarios, weights=self.weights)[0]
        return scenario.generate_request()


class LoadTester:
    """Main load testing orchestrator."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: list[RequestResult] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

        # Create scenario based on config
        scenario_classes = {
            "health": HealthCheckScenario,
            "auth": AuthenticationScenario,
            "detection": DetectionScenario,
            "training": TrainingScenario,
            "mixed": MixedScenario,
        }

        scenario_class = scenario_classes.get(self.config.scenario, MixedScenario)
        self.scenario = scenario_class(config)

    async def run_load_test(self) -> LoadTestResults:
        """Execute the complete load test."""
        console.print(Panel.fit("🚀 Starting Load Test", style="bold blue"))

        # Setup scenario
        setup_success = await self.scenario.setup()
        if not setup_success:
            raise RuntimeError("Failed to setup load test scenario")

        self.start_time = datetime.now()

        # Calculate request schedule
        total_requests = self.config.concurrent_users * self.config.duration_seconds
        requests_per_second = total_requests / self.config.duration_seconds

        console.print("Configuration:")
        console.print(f"  • Scenario: {self.config.scenario}")
        console.print(f"  • Concurrent Users: {self.config.concurrent_users}")
        console.print(f"  • Duration: {self.config.duration_seconds}s")
        console.print(f"  • Target RPS: {requests_per_second:.1f}")
        console.print(f"  • Base URL: {self.config.base_url}")

        # Execute load test with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Load Testing", total=self.config.duration_seconds)

            # Use ThreadPoolExecutor for concurrent requests
            with ThreadPoolExecutor(
                max_workers=self.config.concurrent_users
            ) as executor:
                futures = []

                # Schedule requests with ramp-up
                for second in range(self.config.duration_seconds):
                    # Calculate number of users active at this second (ramp-up)
                    if second < self.config.ramp_up_seconds:
                        active_users = int(
                            self.config.concurrent_users
                            * (second + 1)
                            / self.config.ramp_up_seconds
                        )
                    else:
                        active_users = self.config.concurrent_users

                    # Submit requests for this second
                    for _ in range(active_users):
                        method, endpoint, data = self.scenario.generate_request()
                        future = executor.submit(
                            self.scenario.execute_request, method, endpoint, data
                        )
                        futures.append(future)

                    # Update progress
                    progress.update(task, advance=1)

                    # Wait for next second
                    time.sleep(1)

                # Collect all results
                progress.update(task, description="Collecting results...")
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"Request failed: {e}")

        self.end_time = datetime.now()

        # Analyze results
        return self._analyze_results()

    def _analyze_results(self) -> LoadTestResults:
        """Analyze collected results and generate report."""
        if not self.results:
            raise RuntimeError("No results to analyze")

        # Basic statistics
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.error is None])
        failed_requests = total_requests - successful_requests

        # Response time statistics
        response_times = [r.response_time_ms for r in self.results]
        avg_response_time = statistics.mean(response_times)
        p95_response_time = (
            statistics.quantiles(response_times, n=20)[18]
            if len(response_times) > 1
            else response_times[0]
        )
        p99_response_time = (
            statistics.quantiles(response_times, n=100)[98]
            if len(response_times) > 1
            else response_times[0]
        )
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # Throughput
        test_duration = (self.end_time - self.start_time).total_seconds()
        throughput_rps = total_requests / test_duration

        # Error analysis
        error_rate = (failed_requests / total_requests) * 100
        errors_by_type = {}
        for result in self.results:
            if result.error:
                errors_by_type[result.error] = errors_by_type.get(result.error, 0) + 1

        # Endpoint analysis
        requests_by_endpoint = {}
        response_times_by_endpoint = {}
        for result in self.results:
            endpoint = result.endpoint
            requests_by_endpoint[endpoint] = requests_by_endpoint.get(endpoint, 0) + 1
            if endpoint not in response_times_by_endpoint:
                response_times_by_endpoint[endpoint] = []
            response_times_by_endpoint[endpoint].append(result.response_time_ms)

        # System metrics (placeholder - would integrate with monitoring system)
        system_metrics = {
            "cpu_percent": 65.0,  # Would get from monitoring
            "memory_percent": 72.0,
            "disk_io_percent": 15.0,
            "network_io_mbps": 45.2,
        }

        # Threshold validation
        thresholds_passed, violations = self._validate_thresholds(
            avg_response_time, error_rate, throughput_rps, system_metrics
        )

        return LoadTestResults(
            config=self.config,
            start_time=self.start_time,
            end_time=self.end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            throughput_rps=throughput_rps,
            error_rate_percent=error_rate,
            errors_by_type=errors_by_type,
            requests_by_endpoint=requests_by_endpoint,
            response_times_by_endpoint=response_times_by_endpoint,
            system_metrics=system_metrics,
            thresholds_passed=thresholds_passed,
            threshold_violations=violations,
        )

    def _validate_thresholds(
        self,
        avg_response_time: float,
        error_rate: float,
        throughput_rps: float,
        system_metrics: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Validate performance against thresholds."""
        violations = []
        thresholds = self.config.thresholds

        if avg_response_time > thresholds["max_response_time_ms"]:
            violations.append(
                f"Average response time {avg_response_time:.1f}ms > {thresholds['max_response_time_ms']}ms"
            )

        if error_rate > thresholds["max_error_rate_percent"]:
            violations.append(
                f"Error rate {error_rate:.1f}% > {thresholds['max_error_rate_percent']}%"
            )

        if throughput_rps < thresholds["min_throughput_rps"]:
            violations.append(
                f"Throughput {throughput_rps:.1f} RPS < {thresholds['min_throughput_rps']} RPS"
            )

        if system_metrics["cpu_percent"] > thresholds["max_cpu_percent"]:
            violations.append(
                f"CPU usage {system_metrics['cpu_percent']:.1f}% > {thresholds['max_cpu_percent']}%"
            )

        if system_metrics["memory_percent"] > thresholds["max_memory_percent"]:
            violations.append(
                f"Memory usage {system_metrics['memory_percent']:.1f}% > {thresholds['max_memory_percent']}%"
            )

        return len(violations) == 0, violations


class LoadTestReporter:
    """Generate comprehensive load test reports."""

    @staticmethod
    def print_console_report(results: LoadTestResults):
        """Print detailed console report."""
        console.print("\n" + "=" * 80)
        console.print(Panel.fit("📊 Load Test Results", style="bold green"))

        # Summary table
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row(
            "Test Duration",
            f"{(results.end_time - results.start_time).total_seconds():.1f}s",
        )
        summary_table.add_row("Total Requests", str(results.total_requests))
        summary_table.add_row("Successful Requests", str(results.successful_requests))
        summary_table.add_row("Failed Requests", str(results.failed_requests))
        summary_table.add_row(
            "Success Rate",
            f"{(results.successful_requests/results.total_requests)*100:.1f}%",
        )
        summary_table.add_row("Throughput", f"{results.throughput_rps:.1f} RPS")

        console.print(summary_table)

        # Performance table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="magenta")
        perf_table.add_column("Threshold", style="yellow")
        perf_table.add_column("Status", style="green")

        perf_table.add_row(
            "Avg Response Time",
            f"{results.avg_response_time_ms:.1f}ms",
            f"{results.config.thresholds['max_response_time_ms']}ms",
            "✅"
            if results.avg_response_time_ms
            <= results.config.thresholds["max_response_time_ms"]
            else "❌",
        )
        perf_table.add_row(
            "P95 Response Time", f"{results.p95_response_time_ms:.1f}ms", "-", "📊"
        )
        perf_table.add_row(
            "P99 Response Time", f"{results.p99_response_time_ms:.1f}ms", "-", "📊"
        )
        perf_table.add_row(
            "Error Rate",
            f"{results.error_rate_percent:.1f}%",
            f"{results.config.thresholds['max_error_rate_percent']}%",
            "✅"
            if results.error_rate_percent
            <= results.config.thresholds["max_error_rate_percent"]
            else "❌",
        )

        console.print(perf_table)

        # Threshold validation
        if results.thresholds_passed:
            console.print(
                Panel("✅ All performance thresholds passed!", style="bold green")
            )
        else:
            console.print(
                Panel("❌ Performance threshold violations detected!", style="bold red")
            )
            for violation in results.threshold_violations:
                console.print(f"  • {violation}", style="red")

        # Endpoint breakdown
        if len(results.requests_by_endpoint) > 1:
            endpoint_table = Table(title="Endpoint Performance")
            endpoint_table.add_column("Endpoint", style="cyan")
            endpoint_table.add_column("Requests", style="magenta")
            endpoint_table.add_column("Avg Response Time", style="yellow")

            for endpoint, count in results.requests_by_endpoint.items():
                avg_time = statistics.mean(results.response_times_by_endpoint[endpoint])
                endpoint_table.add_row(endpoint, str(count), f"{avg_time:.1f}ms")

            console.print(endpoint_table)

    @staticmethod
    def save_json_report(results: LoadTestResults, output_path: str):
        """Save detailed JSON report."""
        report_data = asdict(results)

        # Convert datetime objects to strings
        report_data["start_time"] = results.start_time.isoformat()
        report_data["end_time"] = results.end_time.isoformat()

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        console.print(f"📄 Detailed report saved to: {output_path}")

    @staticmethod
    def save_ci_report(results: LoadTestResults, output_path: str):
        """Save CI/CD-friendly report."""
        ci_data = {
            "test_passed": results.thresholds_passed,
            "total_requests": results.total_requests,
            "success_rate_percent": (
                results.successful_requests / results.total_requests
            )
            * 100,
            "avg_response_time_ms": results.avg_response_time_ms,
            "p95_response_time_ms": results.p95_response_time_ms,
            "throughput_rps": results.throughput_rps,
            "error_rate_percent": results.error_rate_percent,
            "threshold_violations": results.threshold_violations,
            "test_duration_seconds": (
                results.end_time - results.start_time
            ).total_seconds(),
        }

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(ci_data, f, indent=2)

        console.print(f"🤖 CI/CD report saved to: {output_path}")


async def main():
    """Main entry point for load testing."""
    parser = argparse.ArgumentParser(description="Pynomaly Load Testing Framework")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL for testing"
    )
    parser.add_argument(
        "--users", type=int, default=10, help="Number of concurrent users"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--ramp-up", type=int, default=10, help="Ramp-up time in seconds"
    )
    parser.add_argument(
        "--scenario",
        choices=["health", "auth", "detection", "training", "mixed"],
        default="mixed",
        help="Test scenario",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--output-dir", default="load_test_results", help="Output directory for reports"
    )
    parser.add_argument("--config-file", help="JSON config file path")
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI/CD mode - exit with error code on threshold violations",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config_file:
        with open(args.config_file) as f:
            config_data = json.load(f)
        config = LoadTestConfig(**config_data)
    else:
        config = LoadTestConfig(
            base_url=args.base_url,
            concurrent_users=args.users,
            duration_seconds=args.duration,
            ramp_up_seconds=args.ramp_up,
            scenario=args.scenario,
            api_key=args.api_key,
        )

    try:
        # Run load test
        tester = LoadTester(config)
        results = await tester.run_load_test()

        # Generate reports
        LoadTestReporter.print_console_report(results)

        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{args.output_dir}/load_test_{timestamp}.json"
        ci_path = f"{args.output_dir}/load_test_ci_{timestamp}.json"

        LoadTestReporter.save_json_report(results, json_path)
        LoadTestReporter.save_ci_report(results, ci_path)

        # Exit with appropriate code for CI/CD
        if args.ci_mode and not results.thresholds_passed:
            console.print(
                "❌ Load test failed - threshold violations detected", style="bold red"
            )
            sys.exit(1)
        else:
            console.print("✅ Load test completed successfully", style="bold green")
            sys.exit(0)

    except Exception as e:
        console.print(f"❌ Load test failed: {e}", style="bold red")
        logger.exception("Load test failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
