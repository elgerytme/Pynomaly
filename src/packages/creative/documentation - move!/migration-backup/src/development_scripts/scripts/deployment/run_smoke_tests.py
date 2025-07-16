#!/usr/bin/env python3
"""
Smoke Tests for Production Deployment
Runs comprehensive smoke tests to validate core functionality
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SmokeTest:
    """Smoke test configuration"""

    name: str
    test_type: str  # "endpoint", "cli", "integration"
    command: str | None = None
    url: str | None = None
    method: str = "GET"
    payload: dict | None = None
    expected_status: int = 200
    timeout: int = 60
    required: bool = True


@dataclass
class SmokeTestResult:
    """Smoke test result"""

    test: SmokeTest
    success: bool
    duration: float
    output: str | None = None
    error: str | None = None
    status_code: int | None = None


class SmokeTestRunner:
    """Runs comprehensive smoke tests for production deployment"""

    def __init__(self):
        self.base_url = os.getenv("SMOKE_TEST_URL", "http://localhost:8000")
        self.project_root = Path(__file__).parent.parent.parent
        self.test_data_dir = self.project_root / "data" / "input"

    def get_smoke_tests(self) -> list[SmokeTest]:
        """Define all smoke tests to perform"""
        return [
            # API Endpoint Tests
            SmokeTest(
                name="Health Check API",
                test_type="endpoint",
                url=f"{self.base_url}/api/v1/health",
                timeout=30,
            ),
            SmokeTest(
                name="Metrics API",
                test_type="endpoint",
                url=f"{self.base_url}/api/v1/health/metrics",
                timeout=30,
                required=False,
            ),
            SmokeTest(
                name="Models List API",
                test_type="endpoint",
                url=f"{self.base_url}/api/v1/models",
                timeout=30,
                required=False,
            ),
            SmokeTest(
                name="OpenAPI Documentation",
                test_type="endpoint",
                url=f"{self.base_url}/openapi.json",
                timeout=30,
                required=False,
            ),
            # CLI Tests
            SmokeTest(
                name="CLI Help Command",
                test_type="cli",
                command=(
                    "./environments/.venv/bin/python -m "
                    "monorepo.presentation.cli.app --help"
                ),
                timeout=30,
            ),
            SmokeTest(
                name="CLI Version Command",
                test_type="cli",
                command=(
                    './environments/.venv/bin/python -c '
                    '"from monorepo._version import __version__; print(__version__)"'
                ),
                timeout=30,
            ),
            SmokeTest(
                name="CLI Health Check",
                test_type="cli",
                command="./environments/.venv/bin/python -c \"from monorepo.presentation.cli.app import app; print('CLI app import successful')\"",
                timeout=30,
            ),
            # Integration Tests
            SmokeTest(
                name="Core Imports Test",
                test_type="integration",
                command="./environments/.venv/bin/python -c \"from monorepo.domain.entities import Detector; print('Core imports successful')\"",
                timeout=30,
            ),
            SmokeTest(
                name="Infrastructure Test",
                test_type="integration",
                command="./environments/.venv/bin/python -c \"from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter; print('Infrastructure imports successful')\"",
                timeout=30,
            ),
            SmokeTest(
                name="Application Services Test",
                test_type="integration",
                command="./environments/.venv/bin/python -c \"from monorepo.application.services.detection_service import DetectionService; print('Application services imports successful')\"",
                timeout=30,
                required=False,
            ),
            # Basic Functionality Tests
            SmokeTest(
                name="Simple Detection Test",
                test_type="integration",
                command="./environments/.venv/bin/python -c \"import numpy as np; from sklearn.ensemble import IsolationForest; model = IsolationForest(); data = np.random.randn(10, 2); model.fit(data); result = model.predict(data); print(f'Detection test successful: {len(result)} predictions')\"",
                timeout=60,
            ),
        ]

    async def run_endpoint_test(self, test: SmokeTest) -> SmokeTestResult:
        """Run an endpoint smoke test"""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=test.timeout) as client:
                response = await client.request(
                    method=test.method, url=test.url, json=test.payload
                )

                duration = time.time() - start_time

                if response.status_code == test.expected_status:
                    return SmokeTestResult(
                        test=test,
                        success=True,
                        duration=duration,
                        status_code=response.status_code,
                        output=response.text[:500],
                    )
                else:
                    return SmokeTestResult(
                        test=test,
                        success=False,
                        duration=duration,
                        status_code=response.status_code,
                        error=f"Expected status {test.expected_status}, got {response.status_code}",
                    )

        except Exception as e:
            duration = time.time() - start_time
            return SmokeTestResult(
                test=test, success=False, duration=duration, error=str(e)
            )

    async def run_cli_test(self, test: SmokeTest) -> SmokeTestResult:
        """Run a CLI smoke test"""
        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_shell(
                test.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=test.timeout
                )

                duration = time.time() - start_time

                if process.returncode == 0:
                    return SmokeTestResult(
                        test=test,
                        success=True,
                        duration=duration,
                        output=stdout.decode() if stdout else "",
                    )
                else:
                    return SmokeTestResult(
                        test=test,
                        success=False,
                        duration=duration,
                        error=stderr.decode()
                        if stderr
                        else f"Command failed with return code {process.returncode}",
                    )

            except TimeoutError:
                process.kill()
                duration = time.time() - start_time
                return SmokeTestResult(
                    test=test,
                    success=False,
                    duration=duration,
                    error=f"Command timed out after {test.timeout} seconds",
                )

        except Exception as e:
            duration = time.time() - start_time
            return SmokeTestResult(
                test=test, success=False, duration=duration, error=str(e)
            )

    async def run_integration_test(self, test: SmokeTest) -> SmokeTestResult:
        """Run an integration smoke test"""
        # Integration tests are run as CLI commands
        return await self.run_cli_test(test)

    async def run_smoke_test(self, test: SmokeTest) -> SmokeTestResult:
        """Run a single smoke test"""
        logger.info(f"Running smoke test: {test.name}")

        if test.test_type == "endpoint":
            return await self.run_endpoint_test(test)
        elif test.test_type == "cli":
            return await self.run_cli_test(test)
        elif test.test_type == "integration":
            return await self.run_integration_test(test)
        else:
            return SmokeTestResult(
                test=test,
                success=False,
                duration=0,
                error=f"Unknown test type: {test.test_type}",
            )

    async def run_all_smoke_tests(self) -> tuple[bool, list[SmokeTestResult]]:
        """Run all smoke tests"""
        logger.info("Starting comprehensive smoke tests")

        smoke_tests = self.get_smoke_tests()

        # Run tests sequentially for better debugging
        results = []
        for test in smoke_tests:
            result = await self.run_smoke_test(test)
            results.append(result)

            # Log result
            if result.success:
                logger.info(f"✓ {test.name} - {result.duration:.2f}s")
            else:
                logger.error(f"✗ {test.name} - {result.error}")

        # Determine overall success
        required_tests = [r for r in results if r.test.required]
        required_passed = [r for r in required_tests if r.success]

        overall_success = len(required_passed) == len(required_tests)

        return overall_success, results

    def generate_smoke_test_report(self, results: list[SmokeTestResult]) -> dict:
        """Generate smoke test report"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.success])
        failed_tests = total_tests - passed_tests

        required_tests = [r for r in results if r.test.required]
        required_passed = len([r for r in required_tests if r.success])
        required_total = len(required_tests)

        avg_duration = sum(r.duration for r in results) / len(results) if results else 0

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_url": self.base_url,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "required_tests": required_total,
                "required_passed": required_passed,
                "success_rate": round(passed_tests / total_tests * 100, 2)
                if total_tests > 0
                else 0,
                "average_duration": round(avg_duration, 3),
                "overall_success": required_passed == required_total,
            },
            "tests": [
                {
                    "name": result.test.name,
                    "type": result.test.test_type,
                    "required": result.test.required,
                    "success": result.success,
                    "duration": round(result.duration, 3),
                    "status_code": result.status_code,
                    "error": result.error,
                    "output": result.output[:200] if result.output else None,
                }
                for result in results
            ],
        }

        return report

    def save_smoke_test_report(self, report: dict, output_path: str | None = None):
        """Save smoke test report to file"""
        try:
            report_path = (
                output_path
                or self.project_root
                / "artifacts"
                / "deployment"
                / "smoke_test_report.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Smoke test report saved: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save smoke test report: {e}")

    def print_smoke_test_summary(self, results: list[SmokeTestResult]):
        """Print smoke test summary"""
        print("\n" + "=" * 60)
        print("SMOKE TEST SUMMARY")
        print("=" * 60)

        # Group by test type
        test_types = {}
        for result in results:
            test_type = result.test.test_type
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)

        for test_type, type_results in test_types.items():
            print(f"\n{test_type.upper()} TESTS:")
            print("-" * 20)

            for result in type_results:
                status_symbol = "✓" if result.success else "✗"
                required_text = "[REQUIRED]" if result.test.required else "[OPTIONAL]"

                print(
                    f"{status_symbol} {result.test.name} {required_text} - {result.duration:.2f}s"
                )

                if result.error:
                    print(f"    Error: {result.error}")

        # Overall summary
        total = len(results)
        passed = len([r for r in results if r.success])
        required_tests = [r for r in results if r.test.required]
        required_passed = len([r for r in required_tests if r.success])

        print("\nOVERALL SUMMARY:")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Required tests: {len(required_tests)}")
        print(f"Required passed: {required_passed}")
        print(f"Success rate: {passed/total*100:.1f}%")

        overall_success = required_passed == len(required_tests)
        print(f"Overall status: {'PASS' if overall_success else 'FAIL'}")
        print("=" * 60)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run smoke tests for production deployment"
    )
    parser.add_argument("--url", help="Base URL for API tests")
    parser.add_argument("--output", help="Output path for smoke test report")

    args = parser.parse_args()

    # Override defaults with command line arguments
    if args.url:
        os.environ["SMOKE_TEST_URL"] = args.url

    runner = SmokeTestRunner()

    try:
        success, results = await runner.run_all_smoke_tests()

        # Generate and save report
        report = runner.generate_smoke_test_report(results)
        runner.save_smoke_test_report(report, args.output)

        # Print summary
        runner.print_smoke_test_summary(results)

        if success:
            logger.info("🎉 All required smoke tests passed!")
            sys.exit(0)
        else:
            logger.error("❌ Some required smoke tests failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Smoke tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
