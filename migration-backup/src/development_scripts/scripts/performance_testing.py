#!/usr/bin/env python3
"""
Performance testing and optimization script for Pynomaly production deployment.
This script runs comprehensive performance tests and provides optimization recommendations.
"""

import asyncio
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceTest:
    """Performance test configuration."""

    name: str
    endpoint: str
    method: str = "GET"
    payload: dict[str, Any] | None = None
    concurrent_users: int = 10
    duration_seconds: int = 30
    expected_response_time_ms: int = 500
    expected_throughput_rps: int = 100


@dataclass
class PerformanceResult:
    """Performance test result."""

    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate: float
    duration_seconds: float


class PerformanceTester:
    """Main performance testing orchestrator."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize performance tester."""
        self.base_url = base_url
        self.session = self._create_session()
        self.test_results: list[PerformanceResult] = []

        # Define performance test suite
        self.test_suite = [
            PerformanceTest(
                name="Health Check",
                endpoint="/health",
                method="GET",
                concurrent_users=50,
                duration_seconds=30,
                expected_response_time_ms=100,
                expected_throughput_rps=500,
            ),
            PerformanceTest(
                name="API Documentation",
                endpoint="/docs",
                method="GET",
                concurrent_users=20,
                duration_seconds=30,
                expected_response_time_ms=200,
                expected_throughput_rps=100,
            ),
            PerformanceTest(
                name="Prometheus Metrics",
                endpoint="/metrics",
                method="GET",
                concurrent_users=10,
                duration_seconds=30,
                expected_response_time_ms=300,
                expected_throughput_rps=50,
            ),
            PerformanceTest(
                name="Anomaly Detection",
                endpoint="/api/v1/detect",
                method="POST",
                payload={
                    "data": [
                        {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
                        {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0},
                        {"feature1": 100.0, "feature2": 200.0, "feature3": 300.0},
                    ]
                },
                concurrent_users=25,
                duration_seconds=60,
                expected_response_time_ms=500,
                expected_throughput_rps=50,
            ),
            PerformanceTest(
                name="System Metrics",
                endpoint="/api/v1/metrics",
                method="GET",
                concurrent_users=15,
                duration_seconds=30,
                expected_response_time_ms=200,
                expected_throughput_rps=75,
            ),
        ]

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry configuration."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    async def run_single_request(
        self, test: PerformanceTest
    ) -> tuple[float, bool, str]:
        """Run a single request and measure performance."""
        start_time = time.time()

        try:
            url = f"{self.base_url}{test.endpoint}"

            if test.method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            elif test.method.upper() == "POST":
                response = self.session.post(url, json=test.payload, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {test.method}")

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            success = response.status_code == 200
            error_msg = "" if success else f"HTTP {response.status_code}"

            return response_time, success, error_msg

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return response_time, False, str(e)

    def run_load_test_worker(
        self, test: PerformanceTest, stop_event
    ) -> list[tuple[float, bool, str]]:
        """Worker function for load testing."""
        results = []

        while not stop_event.is_set():
            try:
                # Run synchronous version for thread pool
                start_time = time.time()
                url = f"{self.base_url}{test.endpoint}"

                if test.method.upper() == "GET":
                    response = self.session.get(url, timeout=10)
                elif test.method.upper() == "POST":
                    response = self.session.post(url, json=test.payload, timeout=10)
                else:
                    continue

                response_time = (time.time() - start_time) * 1000
                success = response.status_code == 200
                error_msg = "" if success else f"HTTP {response.status_code}"

                results.append((response_time, success, error_msg))

            except Exception as e:
                response_time = (
                    (time.time() - start_time) * 1000 if "start_time" in locals() else 0
                )
                results.append((response_time, False, str(e)))

        return results

    async def run_load_test(self, test: PerformanceTest) -> PerformanceResult:
        """Run load test for a specific endpoint."""
        logger.info(f"🚀 Running load test: {test.name}")
        logger.info(f"   Endpoint: {test.method} {test.endpoint}")
        logger.info(
            f"   Users: {test.concurrent_users}, Duration: {test.duration_seconds}s"
        )

        start_time = time.time()

        # Use threading for better concurrent request handling
        from threading import Event

        stop_event = Event()

        # Start workers
        with ThreadPoolExecutor(max_workers=test.concurrent_users) as executor:
            # Submit worker tasks
            futures = [
                executor.submit(self.run_load_test_worker, test, stop_event)
                for _ in range(test.concurrent_users)
            ]

            # Let test run for specified duration
            await asyncio.sleep(test.duration_seconds)

            # Stop all workers
            stop_event.set()

            # Collect results
            all_results = []
            for future in as_completed(futures):
                try:
                    worker_results = future.result(timeout=5)
                    all_results.extend(worker_results)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        actual_duration = time.time() - start_time

        if not all_results:
            logger.error(f"No results collected for test: {test.name}")
            return PerformanceResult(
                test_name=test.name,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                throughput_rps=0,
                error_rate=100.0,
                duration_seconds=actual_duration,
            )

        # Calculate metrics
        response_times = [r[0] for r in all_results]
        successful_requests = sum(1 for r in all_results if r[1])
        failed_requests = len(all_results) - successful_requests

        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_response_time = (
            sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
        )
        p99_response_time = (
            sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
        )

        throughput_rps = len(all_results) / actual_duration
        error_rate = (failed_requests / len(all_results)) * 100

        result = PerformanceResult(
            test_name=test.name,
            total_requests=len(all_results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            duration_seconds=actual_duration,
        )

        logger.info(f"✅ {test.name} completed:")
        logger.info(
            f"   Requests: {result.total_requests} ({result.successful_requests} success, {result.failed_requests} failed)"
        )
        logger.info(f"   Avg Response Time: {result.avg_response_time_ms:.2f}ms")
        logger.info(f"   Throughput: {result.throughput_rps:.2f} RPS")
        logger.info(f"   Error Rate: {result.error_rate:.2f}%")

        return result

    async def run_stress_test(self) -> dict[str, Any]:
        """Run stress test to find system limits."""
        logger.info("💪 Running stress test to find system limits...")

        stress_test = PerformanceTest(
            name="Stress Test",
            endpoint="/health",
            method="GET",
            concurrent_users=100,
            duration_seconds=60,
        )

        stress_results = []

        # Gradually increase load
        for users in [50, 100, 150, 200, 250, 300]:
            stress_test.concurrent_users = users
            stress_test.duration_seconds = 30

            logger.info(f"Testing with {users} concurrent users...")
            result = await self.run_load_test(stress_test)

            stress_results.append(
                {
                    "concurrent_users": users,
                    "throughput_rps": result.throughput_rps,
                    "avg_response_time_ms": result.avg_response_time_ms,
                    "error_rate": result.error_rate,
                }
            )

            # Stop if error rate is too high
            if result.error_rate > 10:
                logger.warning(
                    f"High error rate ({result.error_rate:.2f}%) detected, stopping stress test"
                )
                break

        return {
            "stress_test_results": stress_results,
            "max_throughput": max(r["throughput_rps"] for r in stress_results),
            "optimal_users": min(
                r["concurrent_users"]
                for r in stress_results
                if r["error_rate"] < 5 and r["avg_response_time_ms"] < 1000
            ),
        }

    async def run_endurance_test(self) -> dict[str, Any]:
        """Run endurance test to check system stability."""
        logger.info("⏱️ Running endurance test (10 minutes)...")

        endurance_test = PerformanceTest(
            name="Endurance Test",
            endpoint="/health",
            method="GET",
            concurrent_users=50,
            duration_seconds=600,  # 10 minutes
        )

        result = await self.run_load_test(endurance_test)

        return {
            "endurance_result": {
                "duration_minutes": result.duration_seconds / 60,
                "total_requests": result.total_requests,
                "avg_throughput_rps": result.throughput_rps,
                "avg_response_time_ms": result.avg_response_time_ms,
                "error_rate": result.error_rate,
                "stability_score": max(0, 100 - result.error_rate),
            }
        }

    async def run_spike_test(self) -> dict[str, Any]:
        """Run spike test to check system recovery."""
        logger.info("📈 Running spike test...")

        spike_results = []

        # Normal load
        normal_test = PerformanceTest(
            name="Normal Load",
            endpoint="/health",
            method="GET",
            concurrent_users=25,
            duration_seconds=30,
        )

        normal_result = await self.run_load_test(normal_test)
        spike_results.append(
            {
                "phase": "normal",
                "users": 25,
                "throughput_rps": normal_result.throughput_rps,
                "avg_response_time_ms": normal_result.avg_response_time_ms,
                "error_rate": normal_result.error_rate,
            }
        )

        # Spike load
        spike_test = PerformanceTest(
            name="Spike Load",
            endpoint="/health",
            method="GET",
            concurrent_users=200,
            duration_seconds=60,
        )

        spike_result = await self.run_load_test(spike_test)
        spike_results.append(
            {
                "phase": "spike",
                "users": 200,
                "throughput_rps": spike_result.throughput_rps,
                "avg_response_time_ms": spike_result.avg_response_time_ms,
                "error_rate": spike_result.error_rate,
            }
        )

        # Recovery phase
        recovery_test = PerformanceTest(
            name="Recovery Phase",
            endpoint="/health",
            method="GET",
            concurrent_users=25,
            duration_seconds=30,
        )

        recovery_result = await self.run_load_test(recovery_test)
        spike_results.append(
            {
                "phase": "recovery",
                "users": 25,
                "throughput_rps": recovery_result.throughput_rps,
                "avg_response_time_ms": recovery_result.avg_response_time_ms,
                "error_rate": recovery_result.error_rate,
            }
        )

        return {
            "spike_test_results": spike_results,
            "recovery_time_ms": abs(
                recovery_result.avg_response_time_ms
                - normal_result.avg_response_time_ms
            ),
            "spike_impact": spike_result.error_rate,
        }

    async def monitor_system_resources(self) -> dict[str, Any]:
        """Monitor system resources during testing."""
        logger.info("📊 Monitoring system resources...")

        try:
            # Get Docker stats
            docker_stats_cmd = [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}",
            ]
            result = subprocess.run(docker_stats_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                container_stats = []

                for line in lines:
                    if line.strip():
                        parts = line.split("\t")
                        if len(parts) >= 5:
                            container_stats.append(
                                {
                                    "container": parts[0],
                                    "cpu_percent": parts[1],
                                    "memory_usage": parts[2],
                                    "network_io": parts[3],
                                    "block_io": parts[4],
                                }
                            )

                return {
                    "system_resources": {
                        "containers": container_stats,
                        "timestamp": datetime.now().isoformat(),
                    }
                }
            else:
                logger.warning("Could not get Docker stats")
                return {"system_resources": {"error": "Could not get Docker stats"}}

        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return {"system_resources": {"error": str(e)}}

    async def run_all_tests(self) -> dict[str, Any]:
        """Run complete performance test suite."""
        logger.info("🚀 Starting comprehensive performance test suite...")

        # Run standard load tests
        for test in self.test_suite:
            result = await self.run_load_test(test)
            self.test_results.append(result)

        # Run specialized tests
        stress_results = await self.run_stress_test()
        endurance_results = await self.run_endurance_test()
        spike_results = await self.run_spike_test()
        resource_results = await self.monitor_system_resources()

        return {
            "load_test_results": self.test_results,
            "stress_test": stress_results,
            "endurance_test": endurance_results,
            "spike_test": spike_results,
            "system_resources": resource_results.get("system_resources", {}),
        }

    def analyze_performance(self, test_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze performance test results and provide recommendations."""
        load_results = test_results["load_test_results"]

        # Calculate overall metrics
        total_requests = sum(r.total_requests for r in load_results)
        total_errors = sum(r.failed_requests for r in load_results)
        avg_response_time = statistics.mean(
            [r.avg_response_time_ms for r in load_results]
        )
        max_response_time = max(r.max_response_time_ms for r in load_results)
        total_throughput = sum(r.throughput_rps for r in load_results)

        # Performance score calculation
        performance_score = 100

        # Deduct points for high response times
        if avg_response_time > 1000:
            performance_score -= 30
        elif avg_response_time > 500:
            performance_score -= 15

        # Deduct points for high error rates
        overall_error_rate = (
            (total_errors / total_requests) * 100 if total_requests > 0 else 0
        )
        if overall_error_rate > 5:
            performance_score -= 25
        elif overall_error_rate > 1:
            performance_score -= 10

        # Deduct points for low throughput
        if total_throughput < 100:
            performance_score -= 20
        elif total_throughput < 200:
            performance_score -= 10

        # Performance issues detection
        issues = []
        recommendations = []

        if avg_response_time > 500:
            issues.append("High average response time")
            recommendations.append("Consider optimizing database queries and caching")

        if overall_error_rate > 1:
            issues.append("High error rate")
            recommendations.append(
                "Investigate application errors and add error handling"
            )

        if total_throughput < 200:
            issues.append("Low throughput")
            recommendations.append(
                "Consider scaling horizontally or optimizing application"
            )

        if max_response_time > 5000:
            issues.append("Very high maximum response time")
            recommendations.append("Add request timeouts and circuit breakers")

        # Add general recommendations
        recommendations.extend(
            [
                "Implement connection pooling for databases",
                "Add caching layers for frequently accessed data",
                "Use async/await for I/O operations",
                "Implement rate limiting to prevent abuse",
                "Monitor and optimize database queries",
                "Use CDN for static assets",
                "Implement proper logging and monitoring",
            ]
        )

        return {
            "performance_analysis": {
                "overall_score": max(0, performance_score),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_error_rate": overall_error_rate,
                "avg_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time,
                "total_throughput_rps": total_throughput,
                "performance_issues": issues,
                "recommendations": recommendations,
            }
        }

    def generate_performance_report(
        self, test_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        analysis = self.analyze_performance(test_results)

        report = {
            "performance_test_report": {
                "timestamp": datetime.now().isoformat(),
                "test_duration_minutes": sum(
                    r.duration_seconds for r in test_results["load_test_results"]
                )
                / 60,
                "total_test_scenarios": len(test_results["load_test_results"]),
                **analysis["performance_analysis"],
            },
            "load_test_details": [
                {
                    "test_name": r.test_name,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "avg_response_time_ms": r.avg_response_time_ms,
                    "p95_response_time_ms": r.p95_response_time_ms,
                    "p99_response_time_ms": r.p99_response_time_ms,
                    "throughput_rps": r.throughput_rps,
                    "error_rate": r.error_rate,
                    "duration_seconds": r.duration_seconds,
                }
                for r in test_results["load_test_results"]
            ],
            "specialized_tests": {
                "stress_test": test_results.get("stress_test", {}),
                "endurance_test": test_results.get("endurance_test", {}),
                "spike_test": test_results.get("spike_test", {}),
            },
            "system_resources": test_results.get("system_resources", {}),
            "test_environment": {
                "base_url": self.base_url,
                "test_timestamp": datetime.now().isoformat(),
            },
        }

        return report

    def save_performance_report(self, report: dict[str, Any]):
        """Save performance report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_report_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Performance report saved to {filename}")

    def print_performance_summary(self, report: dict[str, Any]):
        """Print performance test summary."""
        test_info = report["performance_test_report"]

        print("\n" + "=" * 70)
        print("🚀 PYNOMALY PERFORMANCE TEST SUMMARY")
        print("=" * 70)
        print(f"Test Duration: {test_info['test_duration_minutes']:.1f} minutes")
        print(f"Test Scenarios: {test_info['total_test_scenarios']}")
        print(f"Total Requests: {test_info['total_requests']}")
        print(f"Performance Score: {test_info['overall_score']:.1f}/100")

        print("\n📊 PERFORMANCE METRICS:")
        print(f"  • Average Response Time: {test_info['avg_response_time_ms']:.2f}ms")
        print(f"  • Maximum Response Time: {test_info['max_response_time_ms']:.2f}ms")
        print(f"  • Total Throughput: {test_info['total_throughput_rps']:.2f} RPS")
        print(f"  • Error Rate: {test_info['overall_error_rate']:.2f}%")

        print("\n🔍 DETAILED RESULTS:")
        for test_detail in report["load_test_details"]:
            print(f"  • {test_detail['test_name']}:")
            print(
                f"    - Response Time: {test_detail['avg_response_time_ms']:.2f}ms (P95: {test_detail['p95_response_time_ms']:.2f}ms)"
            )
            print(f"    - Throughput: {test_detail['throughput_rps']:.2f} RPS")
            print(f"    - Error Rate: {test_detail['error_rate']:.2f}%")

        if test_info["performance_issues"]:
            print("\n⚠️  PERFORMANCE ISSUES:")
            for issue in test_info["performance_issues"]:
                print(f"  • {issue}")

        print("\n💡 RECOMMENDATIONS:")
        for recommendation in test_info["recommendations"][:8]:  # Show top 8
            print(f"  • {recommendation}")

        print("\n" + "=" * 70)

        if test_info["overall_score"] >= 80:
            print("🎉 EXCELLENT PERFORMANCE!")
        elif test_info["overall_score"] >= 60:
            print("✅ GOOD PERFORMANCE WITH ROOM FOR IMPROVEMENT")
        else:
            print("⚠️  PERFORMANCE NEEDS OPTIMIZATION")

        print("=" * 70)


async def main():
    """Main performance testing workflow."""
    base_url = os.getenv("BASE_URL", "http://localhost:8000")

    tester = PerformanceTester(base_url)

    try:
        logger.info("🚀 Starting performance testing...")

        # Run all performance tests
        test_results = await tester.run_all_tests()

        # Generate comprehensive report
        report = tester.generate_performance_report(test_results)
        tester.save_performance_report(report)
        tester.print_performance_summary(report)

        # Determine success based on performance score
        performance_score = report["performance_test_report"]["overall_score"]

        if performance_score >= 70:
            logger.info("✅ Performance testing completed successfully!")
            return True
        else:
            logger.warning("⚠️ Performance testing completed with concerns")
            return False

    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        return False


if __name__ == "__main__":
    # Run the performance testing
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
