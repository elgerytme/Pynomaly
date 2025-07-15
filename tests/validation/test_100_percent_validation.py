"""Comprehensive validation framework for achieving 100% test passing rate."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest


@dataclass
class ValidationResult:
    """Result of a validation check."""

    category: str
    test_name: str
    status: str  # passed, failed, skipped, error
    execution_time: float
    error_message: str | None
    stack_trace: str | None
    retry_count: int
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class ValidationSummary:
    """Summary of validation results."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    success_rate: float
    execution_time: float
    categories_tested: list[str]
    critical_failures: list[str]
    recommendations: list[str]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class TestDiscovery:
    """Discovers and categorizes all tests in the project."""

    def __init__(self, base_path: str = "tests"):
        self.base_path = Path(base_path)
        self.test_patterns = ["test_*.py", "*_test.py"]
        self.exclude_patterns = ["__pycache__", "*.pyc", ".pytest_cache", "conftest.py"]

    def discover_all_tests(self) -> dict[str, list[str]]:
        """Discover all tests organized by category."""
        test_categories = defaultdict(list)

        # Walk through test directory
        for test_file in self.base_path.rglob("test_*.py"):
            if self._should_include_file(test_file):
                category = self._categorize_test_file(test_file)
                test_categories[category].append(str(test_file))

        return dict(test_categories)

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included in test discovery."""
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in str(file_path):
                return False

        # Check if it's a valid Python test file
        return file_path.suffix == ".py" and file_path.name.startswith("test_")

    def _categorize_test_file(self, file_path: Path) -> str:
        """Categorize test file based on its path and name."""
        path_str = str(file_path)

        # Categorize based on directory structure
        if "/unit/" in path_str or "/domain/" in path_str:
            return "unit"
        elif "/integration/" in path_str:
            return "integration"
        elif "/e2e/" in path_str:
            return "e2e"
        elif "/performance/" in path_str:
            return "performance"
        elif "/security/" in path_str:
            return "security"
        elif "/ui/" in path_str:
            return "ui"
        elif "/mutation/" in path_str:
            return "mutation"
        elif "/property/" in path_str:
            return "property"
        elif "/quality_gates/" in path_str:
            return "quality"
        elif "/stability/" in path_str or "/optimization/" in path_str:
            return "infrastructure"
        else:
            return "misc"

    def get_test_count_by_category(self) -> dict[str, int]:
        """Get count of test files by category."""
        test_categories = self.discover_all_tests()
        return {category: len(files) for category, files in test_categories.items()}


class TestValidator:
    """Validates individual tests and categories."""

    def __init__(self):
        self.validation_timeouts = {
            "unit": 30,  # 30 seconds per unit test file
            "integration": 120,  # 2 minutes per integration test file
            "e2e": 300,  # 5 minutes per e2e test file
            "performance": 180,  # 3 minutes per performance test file
            "security": 60,  # 1 minute per security test file
            "ui": 180,  # 3 minutes per UI test file
            "mutation": 300,  # 5 minutes per mutation test file
            "property": 120,  # 2 minutes per property test file
            "quality": 60,  # 1 minute per quality test file
            "infrastructure": 60,  # 1 minute per infrastructure test file
            "misc": 60,  # 1 minute per misc test file
        }

        self.retry_settings = {
            "unit": 1,  # No retries for unit tests
            "integration": 2,  # 1 retry for integration tests
            "e2e": 3,  # 2 retries for e2e tests
            "performance": 1,  # No retries for performance tests
            "security": 1,  # No retries for security tests
            "ui": 3,  # 2 retries for UI tests (flaky)
            "mutation": 1,  # No retries for mutation tests
            "property": 2,  # 1 retry for property tests
            "quality": 1,  # No retries for quality tests
            "infrastructure": 1,  # No retries for infrastructure tests
            "misc": 1,  # No retries for misc tests
        }

    def validate_test_file(self, test_file: str, category: str) -> ValidationResult:
        """Validate a single test file."""
        test_name = Path(test_file).name
        max_retries = self.retry_settings.get(category, 1)
        timeout = self.validation_timeouts.get(category, 60)

        for attempt in range(max_retries):
            start_time = time.time()

            try:
                # Run pytest on the specific file
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        test_file,
                        "-v",
                        "--tb=short",
                        "--disable-warnings",
                        "--no-header",
                        "--no-summary",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                execution_time = time.time() - start_time

                # Parse pytest output
                status, error_message, stack_trace = self._parse_pytest_output(result)

                # Return successful result or retry
                if status == "passed" or attempt == max_retries - 1:
                    return ValidationResult(
                        category=category,
                        test_name=test_name,
                        status=status,
                        execution_time=execution_time,
                        error_message=error_message,
                        stack_trace=stack_trace,
                        retry_count=attempt,
                        timestamp=datetime.now(),
                    )

            except subprocess.TimeoutExpired:
                if attempt == max_retries - 1:
                    return ValidationResult(
                        category=category,
                        test_name=test_name,
                        status="timeout",
                        execution_time=timeout,
                        error_message=f"Test timed out after {timeout} seconds",
                        stack_trace=None,
                        retry_count=attempt,
                        timestamp=datetime.now(),
                    )

            except Exception as e:
                if attempt == max_retries - 1:
                    return ValidationResult(
                        category=category,
                        test_name=test_name,
                        status="error",
                        execution_time=time.time() - start_time,
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                        retry_count=attempt,
                        timestamp=datetime.now(),
                    )

            # Brief pause before retry
            time.sleep(1)

        # Should not reach here, but just in case
        return ValidationResult(
            category=category,
            test_name=test_name,
            status="error",
            execution_time=0,
            error_message="Unexpected validation failure",
            stack_trace=None,
            retry_count=max_retries,
            timestamp=datetime.now(),
        )

    def _parse_pytest_output(
        self, result: subprocess.CompletedProcess
    ) -> tuple[str, str | None, str | None]:
        """Parse pytest output to determine status and extract error information."""
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode

        # Determine status based on return code and output
        if return_code == 0:
            # Check for skipped tests
            if "SKIPPED" in stdout:
                return "skipped", None, None
            else:
                return "passed", None, None

        elif return_code == 1:
            # Test failures
            error_message = self._extract_error_message(stdout, stderr)
            stack_trace = self._extract_stack_trace(stdout, stderr)
            return "failed", error_message, stack_trace

        else:
            # Other errors (import errors, syntax errors, etc.)
            error_message = self._extract_error_message(stdout, stderr)
            stack_trace = stderr if stderr else stdout
            return "error", error_message, stack_trace

    def _extract_error_message(self, stdout: str, stderr: str) -> str:
        """Extract concise error message from pytest output."""
        combined_output = stdout + "\n" + stderr

        # Look for common error patterns
        error_patterns = [
            r"FAILED.*::(.*?) - (.+)",
            r"ERROR.*::(.*?) - (.+)",
            r"AssertionError: (.+)",
            r"Exception: (.+)",
            r"Error: (.+)",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, combined_output)
            if match:
                return match.group(1) if len(match.groups()) == 1 else match.group(2)

        # Fallback to first non-empty line that looks like an error
        lines = combined_output.split("\n")
        for line in lines:
            if any(
                keyword in line.upper() for keyword in ["FAILED", "ERROR", "EXCEPTION"]
            ):
                return line.strip()

        return "Unknown error"

    def _extract_stack_trace(self, stdout: str, stderr: str) -> str | None:
        """Extract stack trace from pytest output."""
        combined_output = stdout + "\n" + stderr

        # Look for traceback section
        traceback_start = -1
        lines = combined_output.split("\n")

        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                traceback_start = i
                break

        if traceback_start >= 0:
            # Find end of traceback
            traceback_end = len(lines)
            for i in range(traceback_start + 1, len(lines)):
                if lines[i].startswith("=") or lines[i].startswith("_"):
                    traceback_end = i
                    break

            return "\n".join(lines[traceback_start:traceback_end])

        return None


class ComprehensiveValidator:
    """Comprehensive validation framework."""

    def __init__(self):
        self.test_discovery = TestDiscovery()
        self.test_validator = TestValidator()
        self.validation_results = []
        self.start_time = None

    def validate_all_tests(self, parallel: bool = True) -> ValidationSummary:
        """Validate all tests in the project."""
        self.start_time = time.time()

        # Discover all tests
        test_categories = self.test_discovery.discover_all_tests()

        print(
            f"Discovered {sum(len(files) for files in test_categories.values())} test files across {len(test_categories)} categories"
        )

        # Validate tests
        if parallel:
            self._validate_parallel(test_categories)
        else:
            self._validate_sequential(test_categories)

        # Generate summary
        return self._generate_summary(test_categories)

    def _validate_parallel(self, test_categories: dict[str, list[str]]):
        """Validate tests in parallel."""
        max_workers = min(
            8, len([f for files in test_categories.values() for f in files])
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all validation tasks
            future_to_test = {}

            for category, test_files in test_categories.items():
                for test_file in test_files:
                    future = executor.submit(
                        self.test_validator.validate_test_file, test_file, category
                    )
                    future_to_test[future] = (test_file, category)

            # Collect results
            for future in as_completed(future_to_test):
                test_file, category = future_to_test[future]
                try:
                    result = future.result()
                    self.validation_results.append(result)

                    # Print progress
                    status_symbol = "✓" if result.status == "passed" else "✗"
                    print(
                        f"{status_symbol} {category}/{Path(test_file).name} ({result.status})"
                    )

                except Exception as e:
                    # Create error result for failed validation
                    error_result = ValidationResult(
                        category=category,
                        test_name=Path(test_file).name,
                        status="error",
                        execution_time=0,
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                        retry_count=0,
                        timestamp=datetime.now(),
                    )
                    self.validation_results.append(error_result)
                    print(f"✗ {category}/{Path(test_file).name} (validation error)")

    def _validate_sequential(self, test_categories: dict[str, list[str]]):
        """Validate tests sequentially."""
        for category, test_files in test_categories.items():
            print(f"\nValidating {category} tests ({len(test_files)} files)...")

            for test_file in test_files:
                result = self.test_validator.validate_test_file(test_file, category)
                self.validation_results.append(result)

                status_symbol = "✓" if result.status == "passed" else "✗"
                print(f"  {status_symbol} {Path(test_file).name} ({result.status})")

    def _generate_summary(
        self, test_categories: dict[str, list[str]]
    ) -> ValidationSummary:
        """Generate validation summary."""
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.status == "passed")
        failed_tests = sum(1 for r in self.validation_results if r.status == "failed")
        skipped_tests = sum(1 for r in self.validation_results if r.status == "skipped")
        error_tests = sum(
            1 for r in self.validation_results if r.status in ["error", "timeout"]
        )

        success_rate = (passed_tests / total_tests) if total_tests > 0 else 0.0
        execution_time = time.time() - self.start_time

        # Identify critical failures
        critical_failures = [
            f"{r.category}/{r.test_name}: {r.error_message}"
            for r in self.validation_results
            if r.status in ["failed", "error", "timeout"]
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations()

        return ValidationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            success_rate=success_rate,
            execution_time=execution_time,
            categories_tested=list(test_categories.keys()),
            critical_failures=critical_failures,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Analyze failure patterns
        failed_results = [
            r for r in self.validation_results if r.status in ["failed", "error"]
        ]
        category_failures = defaultdict(int)

        for result in failed_results:
            category_failures[result.category] += 1

        # Category-specific recommendations
        for category, failure_count in category_failures.items():
            if failure_count > 0:
                if category == "ui":
                    recommendations.append(
                        f"UI tests failing ({failure_count}): Check browser dependencies and wait strategies"
                    )
                elif category == "integration":
                    recommendations.append(
                        f"Integration tests failing ({failure_count}): Verify external dependencies and mocking"
                    )
                elif category == "performance":
                    recommendations.append(
                        f"Performance tests failing ({failure_count}): Review performance thresholds and system resources"
                    )
                elif category == "security":
                    recommendations.append(
                        f"Security tests failing ({failure_count}): Update security scanning tools and configurations"
                    )
                else:
                    recommendations.append(
                        f"{category.title()} tests failing ({failure_count}): Review test implementation and dependencies"
                    )

        # Timeout recommendations
        timeout_results = [r for r in self.validation_results if r.status == "timeout"]
        if timeout_results:
            recommendations.append(
                f"Tests timing out ({len(timeout_results)}): Increase timeouts or optimize test performance"
            )

        # Overall recommendations
        if self._get_success_rate() < 0.95:
            recommendations.append(
                "Overall success rate below 95%: Focus on stabilizing failing tests"
            )

        if not recommendations:
            recommendations.append(
                "All tests passing! Maintain current quality standards."
            )

        return recommendations

    def _get_success_rate(self) -> float:
        """Get current success rate."""
        if not self.validation_results:
            return 0.0

        passed = sum(1 for r in self.validation_results if r.status == "passed")
        return passed / len(self.validation_results)

    def save_detailed_report(
        self, output_path: str = "tests/validation/100_percent_validation_report.json"
    ):
        """Save detailed validation report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Create comprehensive report
        detailed_report = {
            "validation_summary": self._generate_summary({}).to_dict(),
            "detailed_results": [
                result.to_dict() for result in self.validation_results
            ],
            "category_breakdown": self._get_category_breakdown(),
            "failure_analysis": self._get_failure_analysis(),
            "performance_metrics": self._get_performance_metrics(),
            "recommendations_detailed": self._get_detailed_recommendations(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(report_path, "w") as f:
            json.dump(detailed_report, f, indent=2)

        return report_path

    def _get_category_breakdown(self) -> dict[str, dict[str, int]]:
        """Get breakdown of results by category."""
        breakdown = defaultdict(
            lambda: {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "timeout": 0}
        )

        for result in self.validation_results:
            breakdown[result.category][result.status] += 1

        return dict(breakdown)

    def _get_failure_analysis(self) -> dict[str, Any]:
        """Get detailed failure analysis."""
        failed_results = [
            r
            for r in self.validation_results
            if r.status in ["failed", "error", "timeout"]
        ]

        # Group by error type
        error_patterns = defaultdict(list)
        for result in failed_results:
            if result.error_message:
                # Extract error type
                error_type = (
                    result.error_message.split(":")[0]
                    if ":" in result.error_message
                    else result.error_message
                )
                error_patterns[error_type].append(
                    f"{result.category}/{result.test_name}"
                )

        return {
            "total_failures": len(failed_results),
            "error_patterns": dict(error_patterns),
            "longest_running_failures": sorted(
                [
                    {"test": f"{r.category}/{r.test_name}", "time": r.execution_time}
                    for r in failed_results
                ],
                key=lambda x: x["time"],
                reverse=True,
            )[:5],
        }

    def _get_performance_metrics(self) -> dict[str, float]:
        """Get performance metrics for validation."""
        if not self.validation_results:
            return {}

        execution_times = [r.execution_time for r in self.validation_results]

        return {
            "total_execution_time": sum(execution_times),
            "average_test_time": sum(execution_times) / len(execution_times),
            "median_test_time": sorted(execution_times)[len(execution_times) // 2],
            "max_test_time": max(execution_times),
            "min_test_time": min(execution_times),
        }

    def _get_detailed_recommendations(self) -> dict[str, list[str]]:
        """Get detailed recommendations by category."""
        category_results = defaultdict(list)
        for result in self.validation_results:
            category_results[result.category].append(result)

        detailed_recommendations = {}

        for category, results in category_results.items():
            category_recommendations = []

            failed_count = sum(1 for r in results if r.status in ["failed", "error"])
            total_count = len(results)

            if failed_count == 0:
                category_recommendations.append(
                    f"All {total_count} {category} tests passing - excellent!"
                )
            else:
                success_rate = (total_count - failed_count) / total_count
                category_recommendations.append(
                    f"Success rate: {success_rate:.1%} ({total_count - failed_count}/{total_count})"
                )

                if failed_count > 0:
                    category_recommendations.append(
                        f"Fix {failed_count} failing tests to achieve 100%"
                    )

                    # Add specific recommendations
                    for result in results:
                        if result.status in ["failed", "error"]:
                            category_recommendations.append(
                                f"- {result.test_name}: {result.error_message}"
                            )

            detailed_recommendations[category] = category_recommendations

        return detailed_recommendations


class Test100PercentValidation:
    """Test cases for 100% validation framework."""

    @pytest.fixture
    def comprehensive_validator(self):
        """Create comprehensive validator instance."""
        return ComprehensiveValidator()

    def test_test_discovery_functionality(self, comprehensive_validator):
        """Test test discovery functionality."""
        test_discovery = comprehensive_validator.test_discovery

        # Discover tests
        test_categories = test_discovery.discover_all_tests()

        # Verify discovery results
        assert isinstance(test_categories, dict)
        assert len(test_categories) > 0

        # Check that we have reasonable categories
        expected_categories = ["unit", "integration", "performance", "ui", "quality"]
        found_categories = set(test_categories.keys())

        # Should have at least some expected categories
        assert len(found_categories & set(expected_categories)) > 0

        # Get count by category
        count_by_category = test_discovery.get_test_count_by_category()

        assert isinstance(count_by_category, dict)
        assert sum(count_by_category.values()) > 0

        print(f"Discovered categories: {list(test_categories.keys())}")
        print(f"Total test files: {sum(count_by_category.values())}")

    def test_individual_test_validation(self, comprehensive_validator):
        """Test individual test validation."""
        test_validator = comprehensive_validator.test_validator

        # Create a simple test file for validation
        test_content = """
import pytest

def test_sample_passing():
    assert 1 + 1 == 2

def test_sample_assertion():
    assert True is True
"""

        # Write test file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_test.py", delete=False
        ) as f:
            f.write(test_content)
            test_file_path = f.name

        try:
            # Validate the test file
            result = test_validator.validate_test_file(test_file_path, "unit")

            # Verify validation result
            assert isinstance(result, ValidationResult)
            assert result.category == "unit"
            assert result.status in ["passed", "failed", "error", "skipped"]
            assert result.execution_time >= 0
            assert isinstance(result.retry_count, int)

        finally:
            # Cleanup
            os.unlink(test_file_path)

    def test_comprehensive_validation_workflow(self, comprehensive_validator):
        """Test comprehensive validation workflow."""

        # Mock test discovery to avoid running all real tests
        mock_test_categories = {
            "unit": ["tests/validation/mock_unit_test.py"],
            "integration": ["tests/validation/mock_integration_test.py"],
        }

        # Create mock test files
        mock_files = []
        for category, file_list in mock_test_categories.items():
            for file_path in file_list:
                test_content = f"""
import pytest

def test_mock_{category}_1():
    assert True

def test_mock_{category}_2():
    assert 1 == 1
"""
                file_path_obj = Path(file_path)
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path_obj, "w") as f:
                    f.write(test_content)

                mock_files.append(file_path_obj)

        try:
            # Mock the test discovery
            comprehensive_validator.test_discovery.discover_all_tests = (
                lambda: mock_test_categories
            )

            # Run validation
            summary = comprehensive_validator.validate_all_tests(parallel=False)

            # Verify summary
            assert isinstance(summary, ValidationSummary)
            assert summary.total_tests >= 2
            assert 0.0 <= summary.success_rate <= 1.0
            assert summary.execution_time > 0
            assert len(summary.categories_tested) == 2

        finally:
            # Cleanup mock files
            for mock_file in mock_files:
                try:
                    mock_file.unlink()
                except:
                    pass
                try:
                    mock_file.parent.rmdir()
                except:
                    pass

    def test_failure_analysis_and_recommendations(self, comprehensive_validator):
        """Test failure analysis and recommendation generation."""

        # Create mock failing test
        failing_test_content = """
import pytest

def test_failing_assertion():
    assert False, "This test is designed to fail"

def test_error_test():
    raise ValueError("This test raises an error")
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_test.py", delete=False
        ) as f:
            f.write(failing_test_content)
            failing_test_path = f.name

        try:
            # Validate failing test
            result = comprehensive_validator.test_validator.validate_test_file(
                failing_test_path, "unit"
            )
            comprehensive_validator.validation_results = [result]

            # Generate recommendations
            recommendations = comprehensive_validator._generate_recommendations()

            # Verify recommendations
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0

            # Should have meaningful recommendations
            assert any("tests failing" in rec.lower() for rec in recommendations)

        finally:
            os.unlink(failing_test_path)

    def test_detailed_report_generation(self, comprehensive_validator):
        """Test detailed report generation."""

        # Add some mock results
        mock_results = [
            ValidationResult(
                category="unit",
                test_name="test_sample.py",
                status="passed",
                execution_time=1.5,
                error_message=None,
                stack_trace=None,
                retry_count=0,
                timestamp=datetime.now(),
            ),
            ValidationResult(
                category="integration",
                test_name="test_integration.py",
                status="failed",
                execution_time=3.0,
                error_message="AssertionError: Expected True but got False",
                stack_trace="Traceback...",
                retry_count=1,
                timestamp=datetime.now(),
            ),
        ]

        comprehensive_validator.validation_results = mock_results
        comprehensive_validator.start_time = time.time() - 10  # 10 seconds ago

        # Generate and save report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.json"
            saved_path = comprehensive_validator.save_detailed_report(str(report_path))

            # Verify report was saved
            assert saved_path.exists()

            # Load and verify report content
            with open(saved_path) as f:
                report = json.load(f)

            assert "validation_summary" in report
            assert "detailed_results" in report
            assert "category_breakdown" in report
            assert "failure_analysis" in report
            assert "performance_metrics" in report
            assert "recommendations_detailed" in report

            # Verify structure
            assert len(report["detailed_results"]) == 2
            assert "unit" in report["category_breakdown"]
            assert "integration" in report["category_breakdown"]

    def test_performance_analysis(self, comprehensive_validator):
        """Test performance analysis of validation process."""

        # Create multiple mock results with varying execution times
        mock_results = []
        execution_times = [0.5, 1.0, 2.5, 0.8, 3.0, 1.2, 0.3, 4.0]

        for i, exec_time in enumerate(execution_times):
            result = ValidationResult(
                category="unit",
                test_name=f"test_perf_{i}.py",
                status="passed",
                execution_time=exec_time,
                error_message=None,
                stack_trace=None,
                retry_count=0,
                timestamp=datetime.now(),
            )
            mock_results.append(result)

        comprehensive_validator.validation_results = mock_results

        # Get performance metrics
        performance_metrics = comprehensive_validator._get_performance_metrics()

        # Verify performance metrics
        assert "total_execution_time" in performance_metrics
        assert "average_test_time" in performance_metrics
        assert "median_test_time" in performance_metrics
        assert "max_test_time" in performance_metrics
        assert "min_test_time" in performance_metrics

        assert performance_metrics["total_execution_time"] == sum(execution_times)
        assert performance_metrics["max_test_time"] == max(execution_times)
        assert performance_metrics["min_test_time"] == min(execution_times)

    def test_100_percent_validation_target(self, comprehensive_validator):
        """Test achievement of 100% validation target."""

        # Create comprehensive mock test results representing different scenarios
        mock_results = []

        # Mostly passing tests
        for i in range(95):
            result = ValidationResult(
                category="unit",
                test_name=f"test_passing_{i}.py",
                status="passed",
                execution_time=1.0,
                error_message=None,
                stack_trace=None,
                retry_count=0,
                timestamp=datetime.now(),
            )
            mock_results.append(result)

        # A few skipped tests (acceptable for 100% target)
        for i in range(3):
            result = ValidationResult(
                category="integration",
                test_name=f"test_skipped_{i}.py",
                status="skipped",
                execution_time=0.1,
                error_message="Test skipped due to missing dependency",
                stack_trace=None,
                retry_count=0,
                timestamp=datetime.now(),
            )
            mock_results.append(result)

        # Two failing tests (need to be fixed for 100%)
        for i in range(2):
            result = ValidationResult(
                category="integration",
                test_name=f"test_failing_{i}.py",
                status="failed",
                execution_time=2.0,
                error_message="Test failure that needs to be addressed",
                stack_trace="Stack trace...",
                retry_count=1,
                timestamp=datetime.now(),
            )
            mock_results.append(result)

        comprehensive_validator.validation_results = mock_results
        comprehensive_validator.start_time = time.time() - 60  # 1 minute execution

        # Generate summary
        summary = comprehensive_validator._generate_summary({})

        # Calculate effective success rate (passed + skipped vs total)
        # For 100% target, we count skipped as acceptable
        effective_passed = summary.passed_tests + summary.skipped_tests
        effective_success_rate = effective_passed / summary.total_tests

        # Verify metrics
        assert summary.total_tests == 100
        assert summary.passed_tests == 95
        assert summary.skipped_tests == 3
        assert summary.failed_tests == 2

        # For 100% validation target
        target_achieved = summary.failed_tests == 0 and summary.error_tests == 0

        # Generate recommendations for achieving 100%
        recommendations = []
        if not target_achieved:
            recommendations.append(
                f"Fix {summary.failed_tests + summary.error_tests} failing/error tests to achieve 100% target"
            )
            recommendations.extend(summary.recommendations)

        # Create 100% validation report
        validation_report = {
            "target_achievement": {
                "target_success_rate": 1.0,
                "actual_success_rate": summary.success_rate,
                "effective_success_rate": effective_success_rate,
                "target_achieved": target_achieved,
                "tests_to_fix": summary.failed_tests + summary.error_tests,
            },
            "summary": summary.to_dict(),
            "path_to_100_percent": recommendations,
            "critical_actions": (
                [
                    f"Fix {failure.split(':')[0]}"
                    for failure in summary.critical_failures
                ]
                if summary.critical_failures
                else ["All tests passing - maintain quality!"]
            ),
        }

        # Save 100% validation report
        report_path = Path("tests/validation/100_percent_achievement_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(validation_report, f, indent=2)

        # Verify report
        assert validation_report["target_achievement"]["actual_success_rate"] == 0.95
        assert validation_report["target_achievement"]["effective_success_rate"] == 0.98
        assert not validation_report["target_achievement"][
            "target_achieved"
        ]  # Due to failing tests
        assert validation_report["target_achievement"]["tests_to_fix"] == 2

        print(f"100% Validation Report saved to: {report_path}")
        print(f"Target achieved: {target_achieved}")
        print(f"Tests to fix: {summary.failed_tests + summary.error_tests}")
        print(f"Effective success rate: {effective_success_rate:.2%}")

    def test_validation_monitoring_dashboard_data(self, comprehensive_validator):
        """Test data generation for validation monitoring dashboard."""

        # Simulate validation results over time
        validation_history = []

        for day in range(7):
            # Simulate improving validation over time
            base_success_rate = 0.85 + (day * 0.02)  # From 85% to 97%
            total_tests = 200 + (day * 5)

            daily_results = []
            passed_count = int(total_tests * base_success_rate)
            failed_count = total_tests - passed_count

            # Create mock results for the day
            for i in range(passed_count):
                result = ValidationResult(
                    category="unit",
                    test_name=f"test_day{day}_{i}.py",
                    status="passed",
                    execution_time=1.0,
                    error_message=None,
                    stack_trace=None,
                    retry_count=0,
                    timestamp=datetime.now() - timedelta(days=6 - day),
                )
                daily_results.append(result)

            for i in range(failed_count):
                result = ValidationResult(
                    category="integration",
                    test_name=f"test_day{day}_fail_{i}.py",
                    status="failed",
                    execution_time=2.0,
                    error_message="Sample failure",
                    stack_trace=None,
                    retry_count=0,
                    timestamp=datetime.now() - timedelta(days=6 - day),
                )
                daily_results.append(result)

            # Create daily summary
            daily_summary = {
                "date": (datetime.now() - timedelta(days=6 - day)).strftime("%Y-%m-%d"),
                "total_tests": total_tests,
                "passed_tests": passed_count,
                "failed_tests": failed_count,
                "success_rate": base_success_rate,
                "execution_time": 300 - (day * 20),  # Improving execution time
            }

            validation_history.append(daily_summary)

        # Generate dashboard data
        dashboard_data = {
            "current_status": validation_history[-1],
            "trend_data": validation_history,
            "progress_towards_100_percent": {
                "current_success_rate": validation_history[-1]["success_rate"],
                "target_success_rate": 1.0,
                "gap_percentage": (1.0 - validation_history[-1]["success_rate"]) * 100,
                "tests_to_fix": validation_history[-1]["failed_tests"],
                "trend": (
                    "improving"
                    if validation_history[-1]["success_rate"]
                    > validation_history[0]["success_rate"]
                    else "declining"
                ),
            },
            "performance_metrics": {
                "average_execution_time": sum(
                    day["execution_time"] for day in validation_history
                )
                / len(validation_history),
                "total_tests_validated": sum(
                    day["total_tests"] for day in validation_history
                ),
                "improvement_rate": (
                    validation_history[-1]["success_rate"]
                    - validation_history[0]["success_rate"]
                )
                / len(validation_history),
            },
            "recommendations": [
                "Continue fixing failing tests to reach 100% target",
                "Monitor execution time to ensure performance standards",
                "Focus on integration test stability",
            ],
        }

        # Save dashboard data
        dashboard_path = Path("tests/validation/validation_dashboard_data.json")
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dashboard_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        # Verify dashboard data
        assert dashboard_data["progress_towards_100_percent"]["trend"] == "improving"
        assert dashboard_data["current_status"]["success_rate"] > 0.9
        assert dashboard_data["performance_metrics"]["improvement_rate"] > 0

        print(f"Validation dashboard data saved to: {dashboard_path}")
        print(
            f"Current success rate: {dashboard_data['current_status']['success_rate']:.2%}"
        )
        print(
            f"Gap to 100%: {dashboard_data['progress_towards_100_percent']['gap_percentage']:.1f}%"
        )
        print(
            f"Tests to fix: {dashboard_data['progress_towards_100_percent']['tests_to_fix']}"
        )

    def teardown_method(self, method):
        """Cleanup after each test method."""
        # Clean up any temporary files or state
        pass
