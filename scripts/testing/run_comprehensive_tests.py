#!/usr/bin/env python3
"""Comprehensive test runner for Pynomaly with coverage analysis."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class TestRunner:
    """Comprehensive test runner with reporting."""

    def __init__(self, project_root: str):
        """Initialize test runner.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"

        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
        (self.reports_dir / "coverage").mkdir(exist_ok=True)

    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests.

        Args:
            verbose: Enable verbose output

        Returns:
            Test results dictionary
        """
        print("\n🧪 Running Unit Tests...")
        print("=" * 50)

        cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "unit or not (integration or performance or security or slow)",
            "--cov=src/pynomaly",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage/unit",
            "--cov-report=xml:reports/coverage_unit.xml",
            "--junitxml=reports/junit_unit.xml",
            "--tb=short",
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "type": "unit",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests.

        Args:
            verbose: Enable verbose output

        Returns:
            Test results dictionary
        """
        print("\n🔗 Running Integration Tests...")
        print("=" * 50)

        cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "integration",
            "--integration",
            "--cov=src/pynomaly",
            "--cov-append",
            "--cov-report=html:reports/coverage/integration",
            "--junitxml=reports/junit_integration.xml",
            "--tb=short",
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "type": "integration",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run security tests.

        Args:
            verbose: Enable verbose output

        Returns:
            Test results dictionary
        """
        print("\n🔒 Running Security Tests...")
        print("=" * 50)

        cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "security",
            "--security",
            "--cov=src/pynomaly",
            "--cov-append",
            "--cov-report=html:reports/coverage/security",
            "--junitxml=reports/junit_security.xml",
            "--tb=short",
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "type": "security",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests.

        Args:
            verbose: Enable verbose output

        Returns:
            Test results dictionary
        """
        print("\n⚡ Running Performance Tests...")
        print("=" * 50)

        cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "performance",
            "--performance",
            "--runslow",
            "--cov=src/pynomaly",
            "--cov-append",
            "--cov-report=html:reports/coverage/performance",
            "--junitxml=reports/junit_performance.xml",
            "--tb=short",
            "--timeout=300",  # 5 minute timeout for performance tests
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "type": "performance",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all tests with comprehensive coverage.

        Args:
            verbose: Enable verbose output

        Returns:
            Test results dictionary
        """
        print("\n🧪 Running All Tests with Coverage...")
        print("=" * 50)

        cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "--integration",
            "--security",
            "--performance",
            "--runslow",
            "--cov=src/pynomaly",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage",
            "--cov-report=xml:reports/coverage.xml",
            "--cov-fail-under=90",
            "--junitxml=reports/junit.xml",
            "--html=reports/pytest_report.html",
            "--self-contained-html",
            "--tb=short",
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "type": "all",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_quick_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run quick tests (unit tests only, no slow tests).

        Args:
            verbose: Enable verbose output

        Returns:
            Test results dictionary
        """
        print("\n⚡ Running Quick Tests...")
        print("=" * 50)

        cmd = [
            "python3",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "not (slow or performance)",
            "--cov=src/pynomaly",
            "--cov-report=term",
            "--tb=short",
            "--maxfail=10",  # Stop after 10 failures for quick feedback
        ]

        if verbose:
            cmd.append("-v")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "type": "quick",
            "returncode": result.returncode,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_coverage_report(self) -> Dict[str, Any]:
        """Generate coverage report.

        Returns:
            Coverage results
        """
        print("\n📊 Generating Coverage Report...")
        print("=" * 50)

        # Generate HTML report
        html_cmd = ["python3", "-m", "coverage", "html", "-d", "reports/coverage"]
        html_result = subprocess.run(
            html_cmd, cwd=self.project_root, capture_output=True, text=True
        )

        # Generate XML report
        xml_cmd = ["python3", "-m", "coverage", "xml", "-o", "reports/coverage.xml"]
        xml_result = subprocess.run(
            xml_cmd, cwd=self.project_root, capture_output=True, text=True
        )

        # Get coverage report
        report_cmd = ["python3", "-m", "coverage", "report"]
        report_result = subprocess.run(
            report_cmd, cwd=self.project_root, capture_output=True, text=True
        )

        return {
            "type": "coverage",
            "html_success": html_result.returncode == 0,
            "xml_success": xml_result.returncode == 0,
            "report": report_result.stdout,
            "report_success": report_result.returncode == 0,
        }

    def run_lint_checks(self) -> Dict[str, Any]:
        """Run linting and static analysis.

        Returns:
            Lint results
        """
        print("\n🔍 Running Lint Checks...")
        print("=" * 50)

        results = {}

        # MyPy type checking
        print("Running MyPy type checking...")
        mypy_cmd = ["python3", "-m", "mypy", "src/pynomaly", "--ignore-missing-imports"]
        mypy_result = subprocess.run(
            mypy_cmd, cwd=self.project_root, capture_output=True, text=True
        )
        results["mypy"] = {
            "returncode": mypy_result.returncode,
            "stdout": mypy_result.stdout,
            "stderr": mypy_result.stderr,
        }

        # Flake8 linting
        print("Running Flake8 linting...")
        flake8_cmd = ["python3", "-m", "flake8", "src/pynomaly", "tests/"]
        flake8_result = subprocess.run(
            flake8_cmd, cwd=self.project_root, capture_output=True, text=True
        )
        results["flake8"] = {
            "returncode": flake8_result.returncode,
            "stdout": flake8_result.stdout,
            "stderr": flake8_result.stderr,
        }

        # Black formatting check
        print("Running Black formatting check...")
        black_cmd = [
            "python3",
            "-m",
            "black",
            "--check",
            "--diff",
            "src/pynomaly",
            "tests/",
        ]
        black_result = subprocess.run(
            black_cmd, cwd=self.project_root, capture_output=True, text=True
        )
        results["black"] = {
            "returncode": black_result.returncode,
            "stdout": black_result.stdout,
            "stderr": black_result.stderr,
        }

        return results

    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print test summary.

        Args:
            results: List of test result dictionaries
        """
        print("\n" + "=" * 70)
        print("🎯 TEST SUMMARY")
        print("=" * 70)

        total_duration = sum(r.get("duration", 0) for r in results)
        passed_count = sum(1 for r in results if r.get("returncode") == 0)
        failed_count = len(results) - passed_count

        print(f"\nTotal Duration: {total_duration:.2f} seconds")
        print(f"Tests Passed: {passed_count}/{len(results)}")
        print(f"Tests Failed: {failed_count}/{len(results)}")

        print("\nDetailed Results:")
        print("-" * 40)

        for result in results:
            test_type = result.get("type", "unknown")
            returncode = result.get("returncode", -1)
            duration = result.get("duration", 0)

            status = "✅ PASSED" if returncode == 0 else "❌ FAILED"
            print(f"{test_type:12} | {status} | {duration:6.2f}s")

            # Show errors for failed tests
            if returncode != 0:
                stderr = result.get("stderr", "")
                if stderr:
                    print(f"    Error: {stderr[:200]}...")

        print("\nReports Generated:")
        print("-" * 20)
        print(f"📊 Coverage Report: {self.reports_dir}/coverage/index.html")
        print(f"📋 Test Report: {self.reports_dir}/pytest_report.html")
        print(f"📄 JUnit XML: {self.reports_dir}/junit.xml")
        print(f"📈 Coverage XML: {self.reports_dir}/coverage.xml")

        if failed_count > 0:
            print(f"\n❌ {failed_count} test suite(s) failed!")
            sys.exit(1)
        else:
            print(f"\n✅ All {passed_count} test suite(s) passed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Pynomaly"
    )
    parser.add_argument(
        "--type",
        choices=[
            "unit",
            "integration",
            "security",
            "performance",
            "all",
            "quick",
            "lint",
        ],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    # Change to project root
    project_root = Path(args.project_root).resolve()
    os.chdir(project_root)

    runner = TestRunner(project_root)
    results = []

    print(f"🚀 Starting Pynomaly Test Suite")
    print(f"📁 Project Root: {project_root}")
    print(f"🧪 Test Type: {args.type}")

    if args.type == "unit":
        results.append(runner.run_unit_tests(args.verbose))
    elif args.type == "integration":
        results.append(runner.run_integration_tests(args.verbose))
    elif args.type == "security":
        results.append(runner.run_security_tests(args.verbose))
    elif args.type == "performance":
        results.append(runner.run_performance_tests(args.verbose))
    elif args.type == "quick":
        results.append(runner.run_quick_tests(args.verbose))
    elif args.type == "lint":
        lint_results = runner.run_lint_checks()
        for tool, result in lint_results.items():
            result["type"] = f"lint-{tool}"
            results.append(result)
    else:  # all
        results.append(runner.run_unit_tests(args.verbose))
        results.append(runner.run_integration_tests(args.verbose))
        results.append(runner.run_security_tests(args.verbose))
        results.append(runner.run_performance_tests(args.verbose))

        # Generate coverage report
        coverage_result = runner.run_coverage_report()
        print(f"\n📊 Coverage Report Generated: {coverage_result['report_success']}")
        if coverage_result["report_success"]:
            print(coverage_result["report"])

    # Print summary
    runner.print_summary(results)


if __name__ == "__main__":
    main()
