#!/usr/bin/env python3
"""
Comprehensive Testing and Validation Pipeline
Automated testing pipeline for production deployments
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests in the pipeline"""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    SMOKE = "smoke"
    REGRESSION = "regression"


class TestStatus(Enum):
    """Test execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestSuite:
    """Test suite configuration"""

    name: str
    test_type: TestType
    command: str
    working_dir: str | None = None
    timeout: int = 300
    required: bool = True
    parallel_safe: bool = True
    env_vars: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result"""

    suite: TestSuite
    status: TestStatus
    start_time: float
    end_time: float | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    coverage_report: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0


class ComprehensiveTestPipeline:
    """Comprehensive testing pipeline for production validation"""

    def __init__(self, config_path: str | None = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.artifacts_dir = self.project_root / "artifacts" / "testing"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.test_suites = self._create_test_suites()
        self.results: list[TestResult] = []

    def _create_test_suites(self) -> list[TestSuite]:
        """Create comprehensive test suites"""
        return [
            # Unit Tests
            TestSuite(
                name="Core Unit Tests",
                test_type=TestType.UNIT,
                command="./environments/.venv/bin/python -m pytest tests/unit/ -v --cov=src/pynomaly --cov-report=json --cov-report=html --tb=short",
                timeout=600,
                artifacts=["htmlcov/", "coverage.json"],
            ),
            TestSuite(
                name="Domain Layer Tests",
                test_type=TestType.UNIT,
                command="./environments/.venv/bin/python -m pytest tests/unit/domain/ -v --tb=short",
                timeout=300,
            ),
            TestSuite(
                name="Application Layer Tests",
                test_type=TestType.UNIT,
                command="./environments/.venv/bin/python -m pytest tests/unit/application/ -v --tb=short",
                timeout=300,
            ),
            TestSuite(
                name="Infrastructure Layer Tests",
                test_type=TestType.UNIT,
                command="./environments/.venv/bin/python -m pytest tests/unit/infrastructure/ -v --tb=short",
                timeout=300,
            ),
            # Integration Tests
            TestSuite(
                name="API Integration Tests",
                test_type=TestType.INTEGRATION,
                command="./environments/.venv/bin/python -m pytest tests/integration/ -v --tb=short",
                timeout=900,
                env_vars={"TEST_DATABASE_URL": "sqlite:///test_integration.db"},
            ),
            TestSuite(
                name="Database Integration Tests",
                test_type=TestType.INTEGRATION,
                command="./environments/.venv/bin/python -m pytest tests/integration/test_database_integration.py -v",
                timeout=300,
                env_vars={"TEST_DATABASE_URL": "sqlite:///test_db_integration.db"},
            ),
            TestSuite(
                name="ML Pipeline Integration Tests",
                test_type=TestType.INTEGRATION,
                command="./environments/.venv/bin/python -m pytest tests/integration/test_ml_pipeline_integration.py -v",
                timeout=600,
            ),
            TestSuite(
                name="Enterprise Features Integration",
                test_type=TestType.INTEGRATION,
                command="./environments/.venv/bin/python -m pytest tests/integration/test_enterprise_integration.py -v",
                timeout=300,
                required=False,
            ),
            # End-to-End Tests
            TestSuite(
                name="API E2E Tests",
                test_type=TestType.E2E,
                command="./environments/.venv/bin/python -m pytest tests/e2e/ -v --tb=short",
                timeout=1200,
                parallel_safe=False,
                env_vars={
                    "API_BASE_URL": "http://localhost:8000",
                    "E2E_TEST_TIMEOUT": "120",
                },
            ),
            TestSuite(
                name="CLI E2E Tests",
                test_type=TestType.E2E,
                command="./environments/.venv/bin/python -m pytest tests/e2e/test_cli_e2e.py -v",
                timeout=600,
                parallel_safe=False,
            ),
            TestSuite(
                name="Web UI E2E Tests",
                test_type=TestType.E2E,
                command="./environments/.venv/bin/python -m pytest tests/e2e/test_web_ui_e2e.py -v",
                timeout=900,
                parallel_safe=False,
                required=False,
            ),
            # Performance Tests
            TestSuite(
                name="API Performance Tests",
                test_type=TestType.PERFORMANCE,
                command="./environments/.venv/bin/python scripts/performance_testing.py --test-type api --duration 120",
                timeout=300,
                parallel_safe=False,
                artifacts=["performance_report.json"],
            ),
            TestSuite(
                name="ML Model Performance Tests",
                test_type=TestType.PERFORMANCE,
                command="./environments/.venv/bin/python scripts/performance_testing.py --test-type ml --duration 60",
                timeout=180,
                parallel_safe=False,
            ),
            TestSuite(
                name="Load Testing",
                test_type=TestType.PERFORMANCE,
                command="./environments/.venv/bin/python scripts/load_testing.py --users 50 --duration 120",
                timeout=300,
                parallel_safe=False,
                required=False,
            ),
            # Security Tests
            TestSuite(
                name="Security Vulnerability Scan",
                test_type=TestType.SECURITY,
                command="./environments/.venv/bin/python scripts/security/automated_security_scanner.py",
                timeout=600,
                artifacts=["security_report.json"],
            ),
            TestSuite(
                name="Dependency Security Scan",
                test_type=TestType.SECURITY,
                command="./environments/.venv/bin/python -m safety check --json",
                timeout=120,
                artifacts=["safety_report.json"],
            ),
            TestSuite(
                name="Code Security Analysis",
                test_type=TestType.SECURITY,
                command="./environments/.venv/bin/python -m bandit -r src/ -f json -o bandit_report.json",
                timeout=180,
                artifacts=["bandit_report.json"],
            ),
            # Smoke Tests
            TestSuite(
                name="Production Smoke Tests",
                test_type=TestType.SMOKE,
                command="./environments/.venv/bin/python scripts/deployment/run_smoke_tests.py",
                timeout=300,
                parallel_safe=False,
                env_vars={"SMOKE_TEST_URL": "http://localhost:8000"},
            ),
            TestSuite(
                name="Health Check Validation",
                test_type=TestType.SMOKE,
                command="./environments/.venv/bin/python scripts/deployment/validate_health.py",
                timeout=180,
                parallel_safe=False,
                env_vars={"HEALTH_CHECK_URL": "http://localhost:8000"},
            ),
            # Code Quality Tests
            TestSuite(
                name="Code Formatting Check",
                test_type=TestType.REGRESSION,
                command="./environments/.venv/bin/python -m ruff check src/ tests/",
                timeout=120,
            ),
            TestSuite(
                name="Type Checking",
                test_type=TestType.REGRESSION,
                command="./environments/.venv/bin/python -m mypy src/pynomaly --ignore-missing-imports",
                timeout=180,
                required=False,
            ),
            TestSuite(
                name="Import Sorting Check",
                test_type=TestType.REGRESSION,
                command="./environments/.venv/bin/python -m isort --check-only src/ tests/",
                timeout=60,
                required=False,
            ),
        ]

    async def run_test_suite(self, suite: TestSuite) -> TestResult:
        """Run a single test suite"""
        logger.info(f"🧪 Running {suite.name}")

        result = TestResult(
            suite=suite, status=TestStatus.RUNNING, start_time=time.time()
        )

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(suite.env_vars)

            # Set working directory
            working_dir = suite.working_dir or self.project_root

            # Run test command
            process = await asyncio.create_subprocess_shell(
                suite.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=suite.timeout
                )

                result.end_time = time.time()
                result.exit_code = process.returncode
                result.stdout = stdout.decode() if stdout else ""
                result.stderr = stderr.decode() if stderr else ""

                if process.returncode == 0:
                    result.status = TestStatus.PASSED
                    logger.info(f"✅ {suite.name} passed ({result.duration:.2f}s)")
                else:
                    result.status = TestStatus.FAILED
                    logger.error(f"❌ {suite.name} failed ({result.duration:.2f}s)")

            except TimeoutError:
                process.kill()
                result.end_time = time.time()
                result.status = TestStatus.ERROR
                result.stderr = f"Test timed out after {suite.timeout} seconds"
                logger.error(f"⏰ {suite.name} timed out")

        except Exception as e:
            result.end_time = time.time()
            result.status = TestStatus.ERROR
            result.stderr = str(e)
            logger.error(f"💥 {suite.name} error: {e}")

        # Collect artifacts
        await self._collect_artifacts(suite, result)

        # Parse special outputs
        await self._parse_test_outputs(suite, result)

        return result

    async def _collect_artifacts(self, suite: TestSuite, result: TestResult):
        """Collect test artifacts"""
        try:
            for artifact_pattern in suite.artifacts:
                artifact_path = self.project_root / artifact_pattern

                if artifact_path.exists():
                    # Copy to artifacts directory
                    artifact_name = (
                        f"{suite.name.lower().replace(' ', '_')}_{artifact_path.name}"
                    )
                    target_path = self.artifacts_dir / artifact_name

                    if artifact_path.is_file():
                        import shutil

                        shutil.copy2(artifact_path, target_path)
                    elif artifact_path.is_dir():
                        import shutil

                        if target_path.exists():
                            shutil.rmtree(target_path)
                        shutil.copytree(artifact_path, target_path)

                    logger.info(f"📁 Collected artifact: {target_path}")

        except Exception as e:
            logger.warning(f"Failed to collect artifacts for {suite.name}: {e}")

    async def _parse_test_outputs(self, suite: TestSuite, result: TestResult):
        """Parse test outputs for specific information"""
        try:
            # Parse coverage information
            if "cov" in suite.command and result.status == TestStatus.PASSED:
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        result.coverage_report = json.load(f)

            # Parse performance metrics
            if (
                suite.test_type == TestType.PERFORMANCE
                and result.status == TestStatus.PASSED
            ):
                perf_file = self.project_root / "performance_report.json"
                if perf_file.exists():
                    with open(perf_file) as f:
                        result.performance_metrics = json.load(f)

        except Exception as e:
            logger.warning(f"Failed to parse outputs for {suite.name}: {e}")

    async def run_parallel_tests(self, suites: list[TestSuite]) -> list[TestResult]:
        """Run multiple test suites in parallel"""
        logger.info(f"🔄 Running {len(suites)} test suites in parallel")

        tasks = [self.run_test_suite(suite) for suite in suites]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = TestResult(
                    suite=suites[i],
                    status=TestStatus.ERROR,
                    start_time=time.time(),
                    end_time=time.time(),
                    stderr=str(result),
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        return processed_results

    async def run_sequential_tests(self, suites: list[TestSuite]) -> list[TestResult]:
        """Run test suites sequentially"""
        logger.info(f"📋 Running {len(suites)} test suites sequentially")

        results = []
        for suite in suites:
            result = await self.run_test_suite(suite)
            results.append(result)

            # Stop on critical failure
            if result.status == TestStatus.FAILED and suite.required:
                logger.error(f"❌ Critical test failure: {suite.name}")
                break

        return results

    async def run_full_pipeline(
        self,
        parallel_execution: bool = True,
        test_types: list[TestType] | None = None,
        stop_on_failure: bool = False,
    ) -> list[TestResult]:
        """Run the complete testing pipeline"""
        logger.info("🚀 Starting comprehensive testing pipeline")

        # Filter test suites by type if specified
        suites_to_run = self.test_suites
        if test_types:
            suites_to_run = [s for s in self.test_suites if s.test_type in test_types]

        # Separate parallel-safe and sequential tests
        parallel_suites = [s for s in suites_to_run if s.parallel_safe]
        sequential_suites = [s for s in suites_to_run if not s.parallel_safe]

        all_results = []

        # Run parallel tests first
        if parallel_suites and parallel_execution:
            parallel_results = await self.run_parallel_tests(parallel_suites)
            all_results.extend(parallel_results)

            # Check for critical failures
            if stop_on_failure:
                critical_failures = [
                    r
                    for r in parallel_results
                    if r.status == TestStatus.FAILED and r.suite.required
                ]
                if critical_failures:
                    logger.error(
                        f"❌ {len(critical_failures)} critical failures in parallel tests"
                    )
                    self.results = all_results
                    return all_results

        # Run sequential tests
        if sequential_suites:
            sequential_results = await self.run_sequential_tests(sequential_suites)
            all_results.extend(sequential_results)

        # Run all tests sequentially if parallel execution is disabled
        if not parallel_execution:
            all_results = await self.run_sequential_tests(suites_to_run)

        self.results = all_results
        return all_results

    def generate_test_report(self) -> dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}

        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in self.results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in self.results if r.status == TestStatus.ERROR])

        required_tests = [r for r in self.results if r.suite.required]
        required_passed = len(
            [r for r in required_tests if r.status == TestStatus.PASSED]
        )

        total_duration = sum(r.duration for r in self.results)

        # Group results by test type
        results_by_type = {}
        for result in self.results:
            test_type = result.suite.test_type.value
            if test_type not in results_by_type:
                results_by_type[test_type] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "error": 0,
                    "duration": 0.0,
                }

            results_by_type[test_type]["total"] += 1
            results_by_type[test_type]["duration"] += result.duration

            if result.status == TestStatus.PASSED:
                results_by_type[test_type]["passed"] += 1
            elif result.status == TestStatus.FAILED:
                results_by_type[test_type]["failed"] += 1
            elif result.status == TestStatus.ERROR:
                results_by_type[test_type]["error"] += 1

        # Calculate overall coverage
        overall_coverage = None
        coverage_results = [r for r in self.results if r.coverage_report]
        if coverage_results:
            # Take the most comprehensive coverage report
            coverage_result = max(
                coverage_results, key=lambda r: len(r.coverage_report.get("files", {}))
            )
            overall_coverage = coverage_result.coverage_report.get("totals", {})

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "required_tests": len(required_tests),
                "required_passed": required_passed,
                "success_rate": round(passed_tests / total_tests * 100, 2)
                if total_tests > 0
                else 0,
                "overall_success": required_passed == len(required_tests),
                "total_duration": round(total_duration, 2),
            },
            "results_by_type": results_by_type,
            "coverage": overall_coverage,
            "detailed_results": [
                {
                    "name": result.suite.name,
                    "type": result.suite.test_type.value,
                    "status": result.status.value,
                    "required": result.suite.required,
                    "duration": round(result.duration, 2),
                    "exit_code": result.exit_code,
                    "command": result.suite.command,
                    "stdout_lines": len(result.stdout.split("\n"))
                    if result.stdout
                    else 0,
                    "stderr_lines": len(result.stderr.split("\n"))
                    if result.stderr
                    else 0,
                }
                for result in self.results
            ],
        }

        return report

    def save_test_report(self, report: dict[str, Any], output_path: str | None = None):
        """Save test report to file"""
        try:
            report_path = (
                output_path or self.artifacts_dir / "comprehensive_test_report.json"
            )

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"📄 Test report saved: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save test report: {e}")

    def print_test_summary(self):
        """Print test execution summary"""
        if not self.results:
            print("No test results available")
            return

        print("\n" + "=" * 80)
        print("🧪 COMPREHENSIVE TEST PIPELINE SUMMARY")
        print("=" * 80)

        # Summary by test type
        results_by_type = {}
        for result in self.results:
            test_type = result.suite.test_type.value
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)

        for test_type, type_results in results_by_type.items():
            print(f"\n{test_type.upper()} TESTS:")
            print("-" * 40)

            for result in type_results:
                status_emoji = {
                    TestStatus.PASSED: "✅",
                    TestStatus.FAILED: "❌",
                    TestStatus.ERROR: "💥",
                    TestStatus.SKIPPED: "⏭️",
                }.get(result.status, "❓")

                required_text = "[REQUIRED]" if result.suite.required else "[OPTIONAL]"
                print(
                    f"{status_emoji} {result.suite.name} {required_text} - {result.duration:.2f}s"
                )

                if (
                    result.status in [TestStatus.FAILED, TestStatus.ERROR]
                    and result.stderr
                ):
                    print(f"    Error: {result.stderr[:100]}...")

        # Overall statistics
        total = len(self.results)
        passed = len([r for r in self.results if r.status == TestStatus.PASSED])
        failed = len([r for r in self.results if r.status == TestStatus.FAILED])
        errors = len([r for r in self.results if r.status == TestStatus.ERROR])

        required_tests = [r for r in self.results if r.suite.required]
        required_passed = len(
            [r for r in required_tests if r.status == TestStatus.PASSED]
        )

        total_duration = sum(r.duration for r in self.results)

        print("\n" + "=" * 40)
        print("OVERALL SUMMARY:")
        print("=" * 40)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Required tests: {len(required_tests)}")
        print(f"Required passed: {required_passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        print(f"Total duration: {total_duration:.2f}s")

        overall_success = required_passed == len(required_tests)
        status_text = "🎉 SUCCESS" if overall_success else "❌ FAILURE"
        print(f"Overall status: {status_text}")
        print("=" * 80)


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Test Pipeline")
    parser.add_argument("--config", help="Path to test configuration file")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[t.value for t in TestType],
        help="Test types to run",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run tests in parallel where possible",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run all tests sequentially"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop pipeline on first critical failure",
    )
    parser.add_argument("--output", help="Output path for test report")

    args = parser.parse_args()

    # Create pipeline
    pipeline = ComprehensiveTestPipeline(args.config)

    # Parse test types
    test_types = None
    if args.types:
        test_types = [TestType(t) for t in args.types]

    # Run pipeline
    try:
        results = await pipeline.run_full_pipeline(
            parallel_execution=not args.sequential,
            test_types=test_types,
            stop_on_failure=args.stop_on_failure,
        )

        # Generate and save report
        report = pipeline.generate_test_report()
        pipeline.save_test_report(report, args.output)

        # Print summary
        pipeline.print_test_summary()

        # Exit with appropriate code
        overall_success = report["summary"]["overall_success"]
        return 0 if overall_success else 1

    except Exception as e:
        logger.error(f"Test pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
