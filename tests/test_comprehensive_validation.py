#!/usr/bin/env python3
"""
Comprehensive System Validation Test

This test measures the current overall system success rate by testing
core components across CLI, API, Core, and Integration layers.
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class TestResult:
    """Test result data structure."""

    category: str
    test_name: str
    passed: bool
    execution_time: float
    error_message: str = ""


class ComprehensiveValidator:
    """Comprehensive system validation."""

    def __init__(self):
        self.results: list[TestResult] = []
        self.test_data_dir = None

    def setup_test_environment(self) -> Path:
        """Setup test environment with sample data."""
        self.test_data_dir = tempfile.mkdtemp(prefix="pynomaly_validation_")

        # Create sample CSV data
        sample_data = """x,y,label
1.0,2.0,normal
2.0,3.0,normal
3.0,4.0,normal
100.0,200.0,anomaly
4.0,5.0,normal"""

        test_file = Path(self.test_data_dir) / "test_data.csv"
        with open(test_file, "w") as f:
            f.write(sample_data)

        return test_file

    def test_core_imports(self) -> None:
        """Test core system imports."""
        core_imports = [
            (
                "container",
                "from pynomaly.infrastructure.config.container import Container",
            ),
            ("entities", "from pynomaly.domain.entities import Dataset, Detector"),
            (
                "sklearn_adapter",
                "from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter",
            ),
            (
                "pyod_adapter",
                "from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter",
            ),
            ("cli_app", "from pynomaly.presentation.cli.app import main"),
            ("api_app", "from pynomaly.presentation.api.app import create_app"),
        ]

        for test_name, import_stmt in core_imports:
            start_time = time.time()
            try:
                exec(import_stmt)

                self.results.append(
                    TestResult(
                        category="Core",
                        test_name=f"import_{test_name}",
                        passed=True,
                        execution_time=time.time() - start_time,
                    )
                )
            except Exception as e:
                self.results.append(
                    TestResult(
                        category="Core",
                        test_name=f"import_{test_name}",
                        passed=False,
                        execution_time=time.time() - start_time,
                        error_message=str(e),
                    )
                )

    def test_cli_commands(self) -> None:
        """Test CLI command availability."""
        cli_commands = [
            ("help", ["python3", "-m", "pynomaly", "--help"]),
            ("version", ["python3", "-m", "pynomaly", "--version"]),
            ("detect_help", ["python3", "-m", "pynomaly", "detect", "--help"]),
        ]

        for test_name, cmd in cli_commands:
            start_time = time.time()
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT
                )

                passed = result.returncode == 0
                error_msg = result.stderr if not passed else ""

                self.results.append(
                    TestResult(
                        category="CLI",
                        test_name=test_name,
                        passed=passed,
                        execution_time=time.time() - start_time,
                        error_message=error_msg,
                    )
                )

            except subprocess.TimeoutExpired:
                self.results.append(
                    TestResult(
                        category="CLI",
                        test_name=test_name,
                        passed=False,
                        execution_time=time.time() - start_time,
                        error_message="Command timed out",
                    )
                )
            except Exception as e:
                self.results.append(
                    TestResult(
                        category="CLI",
                        test_name=test_name,
                        passed=False,
                        execution_time=time.time() - start_time,
                        error_message=str(e),
                    )
                )

    def test_api_startup(self) -> None:
        """Test API application startup."""
        start_time = time.time()
        try:
            from pynomaly.presentation.api.app import create_app

            app = create_app(testing=True)

            # Test basic app creation
            self.results.append(
                TestResult(
                    category="API",
                    test_name="app_creation",
                    passed=app is not None,
                    execution_time=time.time() - start_time,
                )
            )

        except Exception as e:
            self.results.append(
                TestResult(
                    category="API",
                    test_name="app_creation",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e),
                )
            )

    def test_integration_workflow(self, test_file: Path) -> None:
        """Test basic integration workflow."""
        start_time = time.time()

        try:
            # Test basic detection workflow using CLI
            cmd = [
                "python3",
                "-m",
                "pynomaly",
                "auto",
                "detect",
                str(test_file),
                "--output",
                str(Path(self.test_data_dir) / "results.json"),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT
            )

            # Check if command executed successfully
            passed = result.returncode == 0

            # Check if results file was created
            results_file = Path(self.test_data_dir) / "results.json"
            if passed and results_file.exists():
                try:
                    with open(results_file) as f:
                        json.load(f)
                    passed = True
                except json.JSONDecodeError:
                    passed = False

            self.results.append(
                TestResult(
                    category="Integration",
                    test_name="autonomous_detection",
                    passed=passed,
                    execution_time=time.time() - start_time,
                    error_message=result.stderr if not passed else "",
                )
            )

        except Exception as e:
            self.results.append(
                TestResult(
                    category="Integration",
                    test_name="autonomous_detection",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e),
                )
            )

    def calculate_success_rates(self) -> dict[str, dict[str, Any]]:
        """Calculate success rates by category."""
        categories = {}

        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "total": 0, "failed": 0}

            categories[result.category]["total"] += 1
            if result.passed:
                categories[result.category]["passed"] += 1
            else:
                categories[result.category]["failed"] += 1

        # Calculate success rates
        for category in categories:
            total = categories[category]["total"]
            passed = categories[category]["passed"]
            categories[category]["success_rate"] = (
                (passed / total * 100) if total > 0 else 0
            )

        # Calculate overall success rate
        total_tests = sum(cat["total"] for cat in categories.values())
        total_passed = sum(cat["passed"] for cat in categories.values())
        overall_success_rate = (
            (total_passed / total_tests * 100) if total_tests > 0 else 0
        )

        return {
            "overall": {
                "total": total_tests,
                "passed": total_passed,
                "failed": total_tests - total_passed,
                "success_rate": round(overall_success_rate, 2),
            },
            "by_category": categories,
        }

    def run_validation(self) -> dict[str, Any]:
        """Run comprehensive validation suite."""
        print("🚀 Starting Comprehensive System Validation...")
        print("=" * 60)

        # Setup test environment
        test_file = self.setup_test_environment()

        # Run all validation tests
        print("🔍 Testing Core Imports...")
        self.test_core_imports()

        print("🔍 Testing CLI Commands...")
        self.test_cli_commands()

        print("🔍 Testing API Startup...")
        self.test_api_startup()

        print("🔍 Testing Integration Workflow...")
        self.test_integration_workflow(test_file)

        # Calculate results
        results = self.calculate_success_rates()

        # Display results
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE VALIDATION RESULTS:")
        print(f"Overall Success Rate: {results['overall']['success_rate']}%")
        print(f"Total Tests: {results['overall']['total']}")
        print(f"Passed: {results['overall']['passed']}")
        print(f"Failed: {results['overall']['failed']}")

        print("\n📋 Results by Category:")
        for category, stats in results["by_category"].items():
            print(
                f"  {category}: {stats['success_rate']:.1f}% ({stats['passed']}/{stats['total']})"
            )

        # Show failed tests
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests[:10]:  # Show first 10 failures
                print(
                    f"  - {test.category}.{test.test_name}: {test.error_message[:80]}..."
                )

        return {
            "success_rates": results,
            "detailed_results": [
                (r.category, r.test_name, r.passed, r.error_message)
                for r in self.results
            ],
        }


def main():
    """Main entry point."""
    validator = ComprehensiveValidator()
    results = validator.run_validation()

    # Save results
    results_file = PROJECT_ROOT / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {results_file}")

    # Exit with appropriate code based on success rate
    overall_success_rate = results["success_rates"]["overall"]["success_rate"]
    if overall_success_rate >= 70:
        print("🎉 System validation passed!")
        return 0
    else:
        print("❌ System validation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
