#!/usr/bin/env python3
"""
Test Results Template for Pynomaly Anomaly Detection

This template provides standardized formats for capturing, validating, and reporting
anomaly detection test results with comprehensive metrics and reproducibility information.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class TestStatus(Enum):
    """Test execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TestType(Enum):
    """Type of anomaly detection test."""

    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    REGRESSION = "regression"


@dataclass
class TestEnvironment:
    """Test environment information."""

    python_version: str
    pynomaly_version: str
    dependencies: dict[str, str]
    hardware: dict[str, Any]
    os_info: str
    random_seed: int | None = None
    test_runner: str | None = None


@dataclass
class TestConfiguration:
    """Test configuration details."""

    algorithm: str
    hyperparameters: dict[str, Any]
    dataset_config: dict[str, Any]
    preprocessing_config: dict[str, Any] | None = None
    validation_config: dict[str, Any] | None = None
    performance_config: dict[str, Any] | None = None


@dataclass
class PerformanceMetrics:
    """Performance and resource utilization metrics."""

    training_time_seconds: float | None = None
    prediction_time_seconds: float | None = None
    memory_peak_mb: float | None = None
    memory_average_mb: float | None = None
    cpu_usage_percent: float | None = None
    gpu_memory_mb: float | None = None
    disk_io_mb: float | None = None
    network_io_mb: float | None = None


@dataclass
class QualityMetrics:
    """Data and model quality metrics."""

    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    accuracy: float | None = None
    specificity: float | None = None
    mcc: float | None = None  # Matthews Correlation Coefficient
    balanced_accuracy: float | None = None
    log_loss: float | None = None

    # Confidence intervals (if available)
    precision_ci: list[float] | None = None
    recall_ci: list[float] | None = None
    f1_ci: list[float] | None = None

    # Additional quality indicators
    prediction_stability: float | None = None
    feature_importance_consistency: float | None = None


@dataclass
class ErrorAnalysis:
    """Detailed error analysis and diagnostics."""

    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int

    # Error patterns
    error_distribution: dict[str, int] | None = None
    misclassified_samples: list[dict[str, Any]] | None = None
    confidence_distribution: dict[str, float] | None = None

    # Error severity analysis
    critical_errors: int | None = None
    warning_errors: int | None = None
    minor_errors: int | None = None


@dataclass
class TestResult:
    """Comprehensive test result structure."""

    # Test identification
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus

    # Timestamps
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float | None = None

    # Test configuration
    configuration: TestConfiguration
    environment: TestEnvironment

    # Results
    quality_metrics: QualityMetrics | None = None
    performance_metrics: PerformanceMetrics | None = None
    error_analysis: ErrorAnalysis | None = None

    # Metadata
    dataset_hash: str | None = None
    model_hash: str | None = None
    reproducibility_info: dict[str, Any] | None = None

    # Validation and comparison
    baseline_comparison: dict[str, Any] | None = None
    cross_validation_scores: list[float] | None = None
    statistical_significance: dict[str, Any] | None = None

    # Artifacts and outputs
    model_path: str | None = None
    results_path: str | None = None
    logs_path: str | None = None
    visualizations: list[str] | None = None

    # Error handling
    error_message: str | None = None
    stack_trace: str | None = None
    warnings: list[str] | None = None

    # Additional metadata
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    created_by: str | None = None


class TestResultsManager:
    """Manage test results with validation, storage, and reporting capabilities."""

    def __init__(self, results_dir: str = "test_results"):
        """Initialize test results manager.

        Args:
            results_dir: Directory to store test results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.results_dir / "raw").mkdir(exist_ok=True)
        (self.results_dir / "processed").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        (self.results_dir / "artifacts").mkdir(exist_ok=True)

    def create_test_result(
        self,
        test_name: str,
        test_type: TestType,
        configuration: TestConfiguration,
        environment: TestEnvironment,
    ) -> TestResult:
        """Create a new test result instance.

        Args:
            test_name: Name of the test
            test_type: Type of test being performed
            configuration: Test configuration
            environment: Test environment information

        Returns:
            New TestResult instance
        """
        test_id = str(uuid.uuid4())

        return TestResult(
            test_id=test_id,
            test_name=test_name,
            test_type=test_type,
            status=TestStatus.PENDING,
            start_time=datetime.now(),
            configuration=configuration,
            environment=environment,
        )

    def update_test_result(
        self,
        test_result: TestResult,
        quality_metrics: QualityMetrics | None = None,
        performance_metrics: PerformanceMetrics | None = None,
        error_analysis: ErrorAnalysis | None = None,
        status: TestStatus | None = None,
        error_message: str | None = None,
    ) -> TestResult:
        """Update test result with new information.

        Args:
            test_result: Test result to update
            quality_metrics: Quality metrics to add
            performance_metrics: Performance metrics to add
            error_analysis: Error analysis to add
            status: New test status
            error_message: Error message if test failed

        Returns:
            Updated test result
        """
        if quality_metrics:
            test_result.quality_metrics = quality_metrics

        if performance_metrics:
            test_result.performance_metrics = performance_metrics

        if error_analysis:
            test_result.error_analysis = error_analysis

        if status:
            test_result.status = status
            if status in [
                TestStatus.COMPLETED,
                TestStatus.FAILED,
                TestStatus.CANCELLED,
            ]:
                test_result.end_time = datetime.now()
                if test_result.start_time:
                    test_result.duration_seconds = (
                        test_result.end_time - test_result.start_time
                    ).total_seconds()

        if error_message:
            test_result.error_message = error_message

        return test_result

    def save_test_result(
        self,
        test_result: TestResult,
        format_type: str = "json",
        include_artifacts: bool = True,
    ) -> str:
        """Save test result to disk.

        Args:
            test_result: Test result to save
            format_type: Output format ("json", "yaml", "pickle")
            include_artifacts: Whether to save related artifacts

        Returns:
            Path to saved test result file
        """
        # Generate filename
        timestamp = test_result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{test_result.test_name}_{timestamp}_{test_result.test_id[:8]}"

        # Save in appropriate format
        if format_type == "json":
            filepath = self.results_dir / "raw" / f"{filename}.json"
            with open(filepath, "w") as f:
                json.dump(
                    self._serialize_test_result(test_result), f, indent=2, default=str
                )

        elif format_type == "yaml":
            filepath = self.results_dir / "raw" / f"{filename}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(
                    self._serialize_test_result(test_result),
                    f,
                    default_flow_style=False,
                )

        elif format_type == "pickle":
            import pickle

            filepath = self.results_dir / "raw" / f"{filename}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(test_result, f)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Save artifacts if requested
        if include_artifacts and test_result.visualizations:
            artifacts_dir = self.results_dir / "artifacts" / test_result.test_id
            artifacts_dir.mkdir(exist_ok=True)

            # Copy visualization files (implementation would depend on file structure)
            test_result.visualizations = [
                str(artifacts_dir / Path(viz).name)
                for viz in test_result.visualizations
            ]

        return str(filepath)

    def load_test_result(self, filepath: str) -> TestResult:
        """Load test result from disk.

        Args:
            filepath: Path to test result file

        Returns:
            Loaded test result
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath) as f:
                data = json.load(f)
            return self._deserialize_test_result(data)

        elif filepath.suffix == ".yaml":
            with open(filepath) as f:
                data = yaml.safe_load(f)
            return self._deserialize_test_result(data)

        elif filepath.suffix == ".pkl":
            import pickle

            with open(filepath, "rb") as f:
                return pickle.load(f)

        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def generate_test_report(
        self,
        test_results: list[TestResult],
        output_path: str,
        report_type: str = "summary",
        include_charts: bool = True,
    ) -> str:
        """Generate comprehensive test report.

        Args:
            test_results: List of test results to include
            output_path: Output file path
            report_type: Type of report ("summary", "detailed", "comparison")
            include_charts: Whether to include visualization charts

        Returns:
            Path to generated report
        """
        if report_type == "summary":
            return self._generate_summary_report(
                test_results, output_path, include_charts
            )
        elif report_type == "detailed":
            return self._generate_detailed_report(
                test_results, output_path, include_charts
            )
        elif report_type == "comparison":
            return self._generate_comparison_report(
                test_results, output_path, include_charts
            )
        else:
            raise ValueError(f"Unsupported report type: {report_type}")

    def validate_test_result(self, test_result: TestResult) -> dict[str, Any]:
        """Validate test result for completeness and consistency.

        Args:
            test_result: Test result to validate

        Returns:
            Validation report with issues found
        """
        issues = []
        warnings = []

        # Required fields validation
        if not test_result.test_id:
            issues.append("Missing test_id")

        if not test_result.test_name:
            issues.append("Missing test_name")

        if not test_result.configuration:
            issues.append("Missing test configuration")

        if not test_result.environment:
            issues.append("Missing environment information")

        # Status consistency validation
        if test_result.status == TestStatus.COMPLETED:
            if not test_result.quality_metrics:
                warnings.append("Completed test missing quality metrics")

            if not test_result.end_time:
                issues.append("Completed test missing end_time")

        elif test_result.status == TestStatus.FAILED:
            if not test_result.error_message:
                warnings.append("Failed test missing error message")

        # Metrics validation
        if test_result.quality_metrics:
            qm = test_result.quality_metrics

            # Check metric ranges
            for metric_name in ["precision", "recall", "f1_score", "auc_roc", "auc_pr"]:
                value = getattr(qm, metric_name)
                if value < 0 or value > 1:
                    issues.append(f"Invalid {metric_name}: {value} (should be 0-1)")

            # Check confusion matrix consistency
            if test_result.error_analysis:
                ea = test_result.error_analysis
                total_predicted = (
                    ea.true_positives
                    + ea.false_positives
                    + ea.true_negatives
                    + ea.false_negatives
                )

                if total_predicted == 0:
                    issues.append("Empty confusion matrix")
                else:
                    calculated_precision = (
                        ea.true_positives / (ea.true_positives + ea.false_positives)
                        if (ea.true_positives + ea.false_positives) > 0
                        else 0
                    )
                    calculated_recall = (
                        ea.true_positives / (ea.true_positives + ea.false_negatives)
                        if (ea.true_positives + ea.false_negatives) > 0
                        else 0
                    )

                    if abs(calculated_precision - qm.precision) > 0.01:
                        warnings.append(
                            f"Precision mismatch: reported {qm.precision:.3f}, calculated {calculated_precision:.3f}"
                        )

                    if abs(calculated_recall - qm.recall) > 0.01:
                        warnings.append(
                            f"Recall mismatch: reported {qm.recall:.3f}, calculated {calculated_recall:.3f}"
                        )

        # Performance validation
        if test_result.performance_metrics:
            pm = test_result.performance_metrics

            if pm.training_time_seconds and pm.training_time_seconds < 0:
                issues.append("Negative training time")

            if pm.memory_peak_mb and pm.memory_peak_mb < 0:
                issues.append("Negative memory usage")

        # Reproducibility validation
        if test_result.environment.random_seed is None:
            warnings.append(
                "No random seed specified - results may not be reproducible"
            )

        if not test_result.dataset_hash:
            warnings.append("No dataset hash - cannot verify data consistency")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "validation_time": datetime.now().isoformat(),
        }

    def compare_test_results(
        self,
        baseline_result: TestResult,
        comparison_results: list[TestResult],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare test results against a baseline.

        Args:
            baseline_result: Baseline test result
            comparison_results: Results to compare against baseline
            metrics: Specific metrics to compare (default: all available)

        Returns:
            Comparison analysis
        """
        if metrics is None:
            metrics = ["precision", "recall", "f1_score", "auc_roc", "auc_pr"]

        comparison = {
            "baseline": {
                "test_id": baseline_result.test_id,
                "test_name": baseline_result.test_name,
                "metrics": {},
            },
            "comparisons": [],
            "statistical_analysis": {},
            "summary": {},
        }

        # Extract baseline metrics
        if baseline_result.quality_metrics:
            baseline_metrics = {
                metric: getattr(baseline_result.quality_metrics, metric)
                for metric in metrics
                if hasattr(baseline_result.quality_metrics, metric)
            }
            comparison["baseline"]["metrics"] = baseline_metrics

        # Compare each result
        for result in comparison_results:
            if not result.quality_metrics:
                continue

            result_comparison = {
                "test_id": result.test_id,
                "test_name": result.test_name,
                "metrics": {},
                "differences": {},
                "improvements": {},
            }

            for metric in metrics:
                if hasattr(result.quality_metrics, metric):
                    current_value = getattr(result.quality_metrics, metric)
                    baseline_value = baseline_metrics.get(metric, 0)

                    result_comparison["metrics"][metric] = current_value
                    result_comparison["differences"][metric] = (
                        current_value - baseline_value
                    )
                    result_comparison["improvements"][metric] = (
                        (current_value - baseline_value) / baseline_value * 100
                        if baseline_value > 0
                        else 0
                    )

            comparison["comparisons"].append(result_comparison)

        # Generate summary statistics
        if comparison["comparisons"]:
            for metric in metrics:
                differences = [
                    comp["differences"].get(metric, 0)
                    for comp in comparison["comparisons"]
                ]
                comparison["summary"][metric] = {
                    "mean_difference": np.mean(differences),
                    "std_difference": np.std(differences),
                    "min_difference": np.min(differences),
                    "max_difference": np.max(differences),
                    "improvements_count": sum(1 for d in differences if d > 0),
                    "degradations_count": sum(1 for d in differences if d < 0),
                }

        return comparison

    def _serialize_test_result(self, test_result: TestResult) -> dict[str, Any]:
        """Convert test result to serializable dictionary."""
        result_dict = asdict(test_result)

        # Convert enums to strings
        result_dict["test_type"] = test_result.test_type.value
        result_dict["status"] = test_result.status.value

        # Convert datetime objects to ISO strings
        result_dict["start_time"] = test_result.start_time.isoformat()
        if test_result.end_time:
            result_dict["end_time"] = test_result.end_time.isoformat()

        return result_dict

    def _deserialize_test_result(self, data: dict[str, Any]) -> TestResult:
        """Convert dictionary back to TestResult object."""
        # Convert string enums back to enum objects
        data["test_type"] = TestType(data["test_type"])
        data["status"] = TestStatus(data["status"])

        # Convert ISO strings back to datetime objects
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            data["end_time"] = datetime.fromisoformat(data["end_time"])

        # Reconstruct nested objects
        if data["configuration"]:
            data["configuration"] = TestConfiguration(**data["configuration"])

        if data["environment"]:
            data["environment"] = TestEnvironment(**data["environment"])

        if data["quality_metrics"]:
            data["quality_metrics"] = QualityMetrics(**data["quality_metrics"])

        if data["performance_metrics"]:
            data["performance_metrics"] = PerformanceMetrics(
                **data["performance_metrics"]
            )

        if data["error_analysis"]:
            data["error_analysis"] = ErrorAnalysis(**data["error_analysis"])

        return TestResult(**data)

    def _generate_summary_report(
        self, test_results: list[TestResult], output_path: str, include_charts: bool
    ) -> str:
        """Generate summary test report."""

        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.status == TestStatus.COMPLETED)
        failed_tests = sum(1 for r in test_results if r.status == TestStatus.FAILED)

        # Performance statistics
        if any(r.quality_metrics for r in test_results):
            f1_scores = [
                r.quality_metrics.f1_score
                for r in test_results
                if r.quality_metrics and r.quality_metrics.f1_score is not None
            ]
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            best_f1 = max(f1_scores) if f1_scores else 0
        else:
            avg_f1 = best_f1 = 0

        report_content = f"""
# Test Results Summary Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Tests:** {total_tests}

## Test Execution Summary

- **✅ Passed:** {passed_tests} ({passed_tests / total_tests * 100:.1f}%)
- **❌ Failed:** {failed_tests} ({failed_tests / total_tests * 100:.1f}%)
- **📊 Average F1 Score:** {avg_f1:.3f}
- **🏆 Best F1 Score:** {best_f1:.3f}

## Test Results Overview

| Test Name | Status | F1 Score | Duration | Algorithm |
|-----------|--------|----------|----------|-----------|
"""

        for result in test_results:
            status_emoji = "✅" if result.status == TestStatus.COMPLETED else "❌"
            f1_score = (
                result.quality_metrics.f1_score if result.quality_metrics else "N/A"
            )
            duration = (
                f"{result.duration_seconds:.1f}s" if result.duration_seconds else "N/A"
            )
            algorithm = (
                result.configuration.algorithm if result.configuration else "N/A"
            )

            report_content += f"| {result.test_name} | {status_emoji} {result.status.value} | {f1_score} | {duration} | {algorithm} |\n"

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return str(output_file)

    def _generate_detailed_report(
        self, test_results: list[TestResult], output_path: str, include_charts: bool
    ) -> str:
        """Generate detailed test report with full metrics."""
        # Implementation would be similar to summary but with much more detail
        # This is a placeholder for the full implementation
        return self._generate_summary_report(test_results, output_path, include_charts)

    def _generate_comparison_report(
        self, test_results: list[TestResult], output_path: str, include_charts: bool
    ) -> str:
        """Generate comparison report between different test results."""
        # Implementation would focus on comparing algorithms and configurations
        # This is a placeholder for the full implementation
        return self._generate_summary_report(test_results, output_path, include_charts)


# Example usage and testing
if __name__ == "__main__":
    # Create test environment
    environment = TestEnvironment(
        python_version="3.11.4",
        pynomaly_version="1.0.0",
        dependencies={"scikit-learn": "1.3.0", "pandas": "2.0.3", "numpy": "1.24.3"},
        hardware={"cpu": "Intel i7-9700K", "memory_gb": 32, "gpu": "RTX 3080"},
        os_info="Ubuntu 22.04.3 LTS",
        random_seed=42,
        test_runner="pytest",
    )

    # Create test configuration
    configuration = TestConfiguration(
        algorithm="IsolationForest",
        hyperparameters={"contamination": 0.1, "n_estimators": 100, "random_state": 42},
        dataset_config={"name": "test_dataset.csv", "size": 10000, "features": 15},
        preprocessing_config={"scaling": "standard", "missing_strategy": "median"},
        validation_config={"method": "5fold_cv", "stratified": True},
    )

    # Create test results manager
    manager = TestResultsManager("test_results_example")

    # Create a test result
    test_result = manager.create_test_result(
        test_name="isolation_forest_baseline",
        test_type=TestType.VALIDATION,
        configuration=configuration,
        environment=environment,
    )

    # Simulate test execution with results
    quality_metrics = QualityMetrics(
        precision=0.85,
        recall=0.78,
        f1_score=0.81,
        auc_roc=0.89,
        auc_pr=0.82,
        accuracy=0.94,
        precision_ci=[0.82, 0.88],
        recall_ci=[0.75, 0.81],
        f1_ci=[0.78, 0.84],
    )

    performance_metrics = PerformanceMetrics(
        training_time_seconds=45.2,
        prediction_time_seconds=2.1,
        memory_peak_mb=150.5,
        memory_average_mb=120.3,
        cpu_usage_percent=65.2,
    )

    error_analysis = ErrorAnalysis(
        false_positives=150,
        false_negatives=220,
        true_positives=780,
        true_negatives=8500,
        error_distribution={"type_1": 95, "type_2": 55, "type_3": 70},
    )

    # Update test result
    test_result = manager.update_test_result(
        test_result,
        quality_metrics=quality_metrics,
        performance_metrics=performance_metrics,
        error_analysis=error_analysis,
        status=TestStatus.COMPLETED,
    )

    # Validate test result
    validation_report = manager.validate_test_result(test_result)
    print("Validation Report:", validation_report)

    # Save test result
    result_path = manager.save_test_result(test_result, format_type="json")
    print(f"Test result saved: {result_path}")

    # Generate test report
    report_path = manager.generate_test_report(
        [test_result],
        "test_summary_report.md",
        report_type="summary",
        include_charts=False,
    )
    print(f"Test report generated: {report_path}")
