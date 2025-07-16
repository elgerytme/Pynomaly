#!/usr/bin/env python3
"""
Comprehensive Test Runner for Pynomaly CI/CD Pipeline.
This script provides unified test execution with intelligent test selection and reporting.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class TestResult:
    """Test result container."""

    def __init__(self, name: str, passed: bool, duration: float, output: str = "", error: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.output = output
        self.error = error
        self.timestamp = datetime.now()


class TestRunner:
    """Comprehensive test runner for Pynomaly."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: list[TestResult] = []
        self.test_suites = {
            "unit": {
                "path": "tests/unit",
                "description": "Unit tests for core functionality",
                "timeout": 300,
                "parallel": True,
            },
            "integration": {
                "path": "tests/integration",
                "description": "Integration tests for components",
                "timeout": 600,
                "parallel": True,
            },
            "api": {
                "path": "tests/api",
                "description": "API endpoint tests",
                "timeout": 300,
                "parallel": False,
            },
            "security": {
                "path": "tests/security",
                "description": "Security and vulnerability tests",
                "timeout": 300,
                "parallel": True,
            },
            "performance": {
                "path": "tests/performance",
                "description": "Performance and load tests",
                "timeout": 900,
                "parallel": False,
            },
            "e2e": {
                "path": "tests/e2e",
                "description": "End-to-end tests",
                "timeout": 1200,
                "parallel": False,
            },
        }

        # Test environment setup
        self.env_vars = {
            "PYNOMALY_ENVIRONMENT": "test",
            "PYTHONPATH": str(self.project_root / "src"),
            "LOG_LEVEL": "DEBUG",
            "DATABASE_URL": "sqlite:///test.db",
            "REDIS_URL": "redis://localhost:6379/0",
        }

        logger.info("Test runner initialized", project_root=str(project_root))

    def setup_environment(self) -> bool:
        """Setup test environment."""
        try:
            logger.info("Setting up test environment...")

            # Create test directories
            for suite_name, suite_config in self.test_suites.items():
                test_dir = self.project_root / suite_config["path"]
                test_dir.mkdir(parents=True, exist_ok=True)

                # Create __init__.py if it doesn't exist
                init_file = test_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("")

            # Setup test database
            self._setup_test_database()

            # Setup test cache
            self._setup_test_cache()

            logger.info("Test environment setup completed")
            return True

        except Exception as e:
            logger.error("Failed to setup test environment", error=str(e))
            return False

    def _setup_test_database(self):
        """Setup test database."""
        try:
            # For SQLite, just ensure directory exists
            db_path = Path("test.db")
            if db_path.exists():
                db_path.unlink()

            # Run database migrations if available
            migration_script = self.project_root / "scripts" / "db" / "migrate.py"
            if migration_script.exists():
                subprocess.run([
                    sys.executable, str(migration_script), "--env", "test"
                ], check=True, capture_output=True)

            logger.info("Test database setup completed")

        except Exception as e:
            logger.warning("Test database setup failed", error=str(e))

    def _setup_test_cache(self):
        """Setup test cache."""
        try:
            # Test Redis connection
            import redis
            r = redis.from_url(self.env_vars["REDIS_URL"])
            r.ping()
            r.flushdb()  # Clear test cache

            logger.info("Test cache setup completed")

        except Exception as e:
            logger.warning("Test cache setup failed", error=str(e))

    def run_test_suite(self, suite_name: str, coverage: bool = True) -> TestResult:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")

        suite_config = self.test_suites[suite_name]
        test_path = self.project_root / suite_config["path"]

        if not test_path.exists():
            logger.warning(f"Test path does not exist: {test_path}")
            return TestResult(
                name=suite_name,
                passed=False,
                duration=0,
                error=f"Test path does not exist: {test_path}"
            )

        logger.info(f"Running test suite: {suite_name}", path=str(test_path))

        start_time = time.time()

        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            f"--timeout={suite_config['timeout']}",
            "--junit-xml=test-results.xml",
        ]

        # Add coverage if requested
        if coverage:
            cmd.extend([
                f"--cov={self.project_root / 'src' / 'monorepo'}",
                "--cov-report=xml",
                "--cov-report=html",
                "--cov-report=term-missing",
            ])

        # Add parallel execution if supported
        if suite_config["parallel"]:
            cmd.extend(["-n", "auto"])

        # Setup environment
        env = os.environ.copy()
        env.update(self.env_vars)

        try:
            # Run tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config["timeout"],
                env=env,
                cwd=self.project_root
            )

            duration = time.time() - start_time
            passed = result.returncode == 0

            test_result = TestResult(
                name=suite_name,
                passed=passed,
                duration=duration,
                output=result.stdout,
                error=result.stderr if not passed else ""
            )

            logger.info(
                f"Test suite completed: {suite_name}",
                passed=passed,
                duration=duration,
                returncode=result.returncode
            )

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Test suite {suite_name} timed out after {suite_config['timeout']} seconds"

            logger.error(error_msg)

            return TestResult(
                name=suite_name,
                passed=False,
                duration=duration,
                error=error_msg
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Test suite {suite_name} failed with error: {str(e)}"

            logger.error(error_msg)

            return TestResult(
                name=suite_name,
                passed=False,
                duration=duration,
                error=error_msg
            )

    def run_all_tests(self, suites: list[str] | None = None, coverage: bool = True) -> dict[str, TestResult]:
        """Run all test suites or specified suites."""
        if suites is None:
            suites = list(self.test_suites.keys())

        logger.info(f"Running test suites: {suites}")

        results = {}
        total_start_time = time.time()

        for suite_name in suites:
            if suite_name not in self.test_suites:
                logger.warning(f"Unknown test suite: {suite_name}")
                continue

            result = self.run_test_suite(suite_name, coverage)
            results[suite_name] = result
            self.results.append(result)

        total_duration = time.time() - total_start_time

        # Generate summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)

        logger.info(
            "Test execution completed",
            passed=passed_count,
            total=total_count,
            duration=total_duration,
            success_rate=f"{(passed_count/total_count)*100:.1f}%"
        )

        return results

    def generate_report(self, results: dict[str, TestResult], output_path: Path):
        """Generate comprehensive test report."""
        try:
            # Create reports directory
            reports_dir = output_path / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate JSON report
            json_report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_suites": len(results),
                    "passed_suites": sum(1 for r in results.values() if r.passed),
                    "failed_suites": sum(1 for r in results.values() if not r.passed),
                    "total_duration": sum(r.duration for r in results.values()),
                    "success_rate": f"{(sum(1 for r in results.values() if r.passed) / len(results)) * 100:.1f}%"
                },
                "suites": {
                    name: {
                        "passed": result.passed,
                        "duration": result.duration,
                        "timestamp": result.timestamp.isoformat(),
                        "output": result.output,
                        "error": result.error,
                        "description": self.test_suites[name]["description"]
                    }
                    for name, result in results.items()
                }
            }

            with open(reports_dir / "test_report.json", "w") as f:
                json.dump(json_report, f, indent=2)

            # Generate HTML report
            html_report = self._generate_html_report(json_report)
            with open(reports_dir / "test_report.html", "w") as f:
                f.write(html_report)

            # Generate markdown summary
            md_report = self._generate_markdown_report(json_report)
            with open(reports_dir / "test_summary.md", "w") as f:
                f.write(md_report)

            logger.info("Test reports generated", output_dir=str(reports_dir))

        except Exception as e:
            logger.error("Failed to generate test report", error=str(e))

    def _generate_html_report(self, json_report: dict) -> str:
        """Generate HTML test report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pynomaly Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .suite {{ border: 1px solid #ddd; margin: 10px 0; border-radius: 5px; }}
                .suite-header {{ background-color: #f9f9f9; padding: 10px; }}
                .suite-content {{ padding: 10px; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                .output {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; white-space: pre-wrap; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Pynomaly Test Report</h1>
                <p>Generated: {json_report['timestamp']}</p>
            </div>

            <div class="summary">
                <div class="metric">
                    <h3>Total Suites</h3>
                    <p>{json_report['summary']['total_suites']}</p>
                </div>
                <div class="metric">
                    <h3>Passed</h3>
                    <p class="passed">{json_report['summary']['passed_suites']}</p>
                </div>
                <div class="metric">
                    <h3>Failed</h3>
                    <p class="failed">{json_report['summary']['failed_suites']}</p>
                </div>
                <div class="metric">
                    <h3>Success Rate</h3>
                    <p>{json_report['summary']['success_rate']}</p>
                </div>
                <div class="metric">
                    <h3>Duration</h3>
                    <p>{json_report['summary']['total_duration']:.2f}s</p>
                </div>
            </div>

            <h2>Test Suites</h2>
        """

        for suite_name, suite_data in json_report['suites'].items():
            status_class = "passed" if suite_data['passed'] else "failed"
            status_text = "✅ PASSED" if suite_data['passed'] else "❌ FAILED"

            html += f"""
            <div class="suite">
                <div class="suite-header">
                    <h3>{suite_name} <span class="{status_class}">{status_text}</span></h3>
                    <p>{suite_data['description']}</p>
                    <p>Duration: {suite_data['duration']:.2f}s</p>
                </div>
                <div class="suite-content">
                    {f'<div class="output">{suite_data["error"]}</div>' if suite_data['error'] else ''}
                </div>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, json_report: dict) -> str:
        """Generate markdown test report."""
        md = f"""# 🧪 Pynomaly Test Report

**Generated:** {json_report['timestamp']}

## 📊 Summary

| Metric | Value |
|--------|-------|
| Total Suites | {json_report['summary']['total_suites']} |
| Passed | {json_report['summary']['passed_suites']} |
| Failed | {json_report['summary']['failed_suites']} |
| Success Rate | {json_report['summary']['success_rate']} |
| Total Duration | {json_report['summary']['total_duration']:.2f}s |

## 🧪 Test Suites

"""

        for suite_name, suite_data in json_report['suites'].items():
            status_emoji = "✅" if suite_data['passed'] else "❌"
            status_text = "PASSED" if suite_data['passed'] else "FAILED"

            md += f"""### {status_emoji} {suite_name} - {status_text}

**Description:** {suite_data['description']}
**Duration:** {suite_data['duration']:.2f}s
**Timestamp:** {suite_data['timestamp']}

"""

            if suite_data['error']:
                md += f"""**Error:**
```
{suite_data['error']}
```

"""

        return md


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pynomaly Test Runner")
    parser.add_argument(
        "--suites",
        nargs="+",
        help="Test suites to run (default: all)",
        choices=["unit", "integration", "api", "security", "performance", "e2e"],
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-output"),
        help="Output directory for test results"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner(args.project_root)

    # Setup environment
    if not runner.setup_environment():
        logger.error("Failed to setup test environment")
        sys.exit(1)

    # Run tests
    results = runner.run_all_tests(
        suites=args.suites,
        coverage=not args.no_coverage
    )

    # Generate report
    runner.generate_report(results, args.output_dir)

    # Exit with appropriate code
    failed_count = sum(1 for r in results.values() if not r.passed)
    if failed_count > 0:
        logger.error(f"Tests failed: {failed_count} suite(s)")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
