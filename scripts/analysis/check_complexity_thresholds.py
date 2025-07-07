#!/usr/bin/env python3
"""Check complexity analysis against predefined thresholds.

This script validates complexity metrics against configurable thresholds
and determines if the code quality is acceptable for CI/CD purposes.
"""

import argparse
import json
import sys
from pathlib import Path


class ComplexityThresholdChecker:
    """Checker for complexity thresholds with configurable limits."""

    def __init__(self, thresholds: dict[str, float]):
        """Initialize with threshold configuration.

        Args:
            thresholds: Dictionary of metric names to threshold values
        """
        self.thresholds = thresholds
        self.violations = []
        self.warnings = []

    def check_metric(
        self, metric_name: str, current_value: float, baseline_value: float = None
    ) -> tuple[bool, str]:
        """Check a single metric against thresholds.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            baseline_value: Baseline value for regression detection

        Returns:
            Tuple of (passed, message)
        """
        threshold = self.thresholds.get(metric_name)
        if threshold is None:
            return True, f"No threshold defined for {metric_name}"

        # Check absolute threshold
        if current_value > threshold:
            message = (
                f"{metric_name}: {current_value:.2f} exceeds threshold {threshold:.2f}"
            )
            self.violations.append(message)
            return False, message

        # Check regression (if baseline available)
        if baseline_value is not None:
            regression_threshold = baseline_value * 1.1  # 10% regression tolerance
            if current_value > regression_threshold:
                message = f"{metric_name}: {current_value:.2f} regressed from baseline {baseline_value:.2f}"
                self.warnings.append(message)
                return True, message  # Warning, not failure

        return (
            True,
            f"{metric_name}: {current_value:.2f} within threshold {threshold:.2f}",
        )

    def check_all_metrics(
        self, current_metrics: dict, baseline_metrics: dict = None
    ) -> bool:
        """Check all metrics against thresholds.

        Args:
            current_metrics: Current metric values
            baseline_metrics: Baseline metric values (optional)

        Returns:
            True if all thresholds pass
        """
        all_passed = True

        for metric_name, _threshold in self.thresholds.items():
            current_value = current_metrics.get(metric_name, 0)
            baseline_value = (
                baseline_metrics.get(metric_name) if baseline_metrics else None
            )

            passed, message = self.check_metric(
                metric_name, current_value, baseline_value
            )
            if not passed:
                all_passed = False

        return all_passed

    def get_summary(self) -> dict:
        """Get summary of threshold checking results."""
        return {
            "total_violations": len(self.violations),
            "total_warnings": len(self.warnings),
            "violations": self.violations,
            "warnings": self.warnings,
            "passed": len(self.violations) == 0,
        }


def load_analysis_data(report_file: Path) -> dict:
    """Load complexity analysis data from JSON file."""
    try:
        with open(report_file) as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load analysis data from {report_file}: {e}")


def create_default_thresholds() -> dict[str, float]:
    """Create default threshold configuration."""
    return {
        # Complexity thresholds
        "cyclomatic_complexity": 10.0,
        "cognitive_complexity": 15.0,
        "maintainability_index": 60.0,  # Minimum acceptable
        # Size thresholds
        "total_lines": 100000,
        "total_files": 1000,
        "python_files": 500,
        "average_file_length": 500,
        # Dependency thresholds
        "total_dependencies": 100,
        "direct_dependencies": 50,
        # Performance thresholds
        "startup_time": 5.0,  # seconds
        "memory_usage": 500.0,  # MB
        "import_time": 2.0,  # seconds
        # Quality thresholds
        "test_coverage": 80.0,  # percentage
        "docstring_coverage": 70.0,  # percentage
        "type_hint_coverage": 80.0,  # percentage
    }


def check_trend_regressions(
    current_data: dict, baseline_data: dict = None
) -> list[dict]:
    """Check for concerning trend regressions."""
    if not baseline_data:
        return []

    regressions = []
    trends = current_data.get("baseline_comparison", {})

    for _metric_key, trend in trends.items():
        if trend["severity"] == "critical":
            regressions.append(
                {
                    "metric": trend["name"],
                    "change_percent": trend["change_percent"],
                    "current": trend["current"],
                    "baseline": trend["baseline"],
                    "severity": "critical",
                }
            )
        elif trend["severity"] == "warning" and abs(trend["change_percent"]) > 25:
            regressions.append(
                {
                    "metric": trend["name"],
                    "change_percent": trend["change_percent"],
                    "current": trend["current"],
                    "baseline": trend["baseline"],
                    "severity": "warning",
                }
            )

    return regressions


def check_quality_gates(current_data: dict) -> list[str]:
    """Check quality gate failures."""
    quality = current_data.get("quality_assessment", {})
    failures = []

    # Critical issues are always failures
    if quality.get("critical_issues"):
        failures.extend([f"Critical: {issue}" for issue in quality["critical_issues"]])

    # Too many warnings indicate quality problems
    warnings = quality.get("warnings", [])
    if len(warnings) > 5:
        failures.append(f"Too many quality warnings: {len(warnings)} warnings detected")

    # Overall quality status
    if quality.get("overall") == "critical":
        failures.append("Overall quality status is CRITICAL")

    return failures


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check complexity thresholds")
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        required=True,
        help="Complexity analysis report JSON file",
    )
    parser.add_argument(
        "--baseline", "-b", type=Path, help="Baseline report for regression detection"
    )
    parser.add_argument(
        "--config", "-c", type=Path, help="Threshold configuration JSON file"
    )

    # Threshold options
    parser.add_argument(
        "--max-complexity",
        type=float,
        default=10.0,
        help="Maximum cyclomatic complexity",
    )
    parser.add_argument(
        "--max-total-lines", type=int, default=100000, help="Maximum total lines"
    )
    parser.add_argument(
        "--max-avg-complexity",
        type=float,
        default=5.0,
        help="Maximum average complexity",
    )
    parser.add_argument(
        "--min-maintainability",
        type=float,
        default=60.0,
        help="Minimum maintainability index",
    )
    parser.add_argument(
        "--max-dependencies",
        type=int,
        default=100,
        help="Maximum number of dependencies",
    )

    # Control options
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Fail if regression detected from baseline",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Fail on quality warnings (not just critical issues)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for results",
    )

    args = parser.parse_args()

    try:
        # Load analysis data
        current_data = load_analysis_data(args.report)
        current_metrics = current_data.get("metrics", {})

        # Load baseline data if available
        baseline_data = None
        baseline_metrics = None
        if args.baseline and args.baseline.exists():
            baseline_data = load_analysis_data(args.baseline)
            baseline_metrics = baseline_data.get("metrics", {})

        # Load or create thresholds
        if args.config and args.config.exists():
            with open(args.config) as f:
                thresholds = json.load(f)
        else:
            thresholds = create_default_thresholds()

            # Override with command-line arguments
            thresholds.update(
                {
                    "cyclomatic_complexity": args.max_complexity,
                    "total_lines": args.max_total_lines,
                    "average_complexity": args.max_avg_complexity,
                    "maintainability_index": args.min_maintainability,
                    "total_dependencies": args.max_dependencies,
                }
            )

        # Initialize checker
        checker = ComplexityThresholdChecker(thresholds)

        # Check all metrics
        passed = checker.check_all_metrics(current_metrics, baseline_metrics)

        # Check for trend regressions
        regressions = check_trend_regressions(current_data, baseline_data)

        # Check quality gates
        quality_failures = check_quality_gates(current_data)

        # Determine overall result
        has_critical_regressions = any(r["severity"] == "critical" for r in regressions)
        has_quality_failures = len(quality_failures) > 0

        overall_passed = (
            passed
            and (not args.fail_on_regression or not has_critical_regressions)
            and (not args.fail_on_warnings or not has_quality_failures)
        )

        # Generate output
        summary = checker.get_summary()
        result = {
            "passed": overall_passed,
            "threshold_check": summary,
            "regressions": regressions,
            "quality_failures": quality_failures,
            "metrics_checked": len(thresholds),
            "baseline_available": baseline_data is not None,
        }

        if args.output_format == "json":
            print(json.dumps(result, indent=2))
        else:
            # Text output
            print("=" * 60)
            print("COMPLEXITY THRESHOLD CHECK RESULTS")
            print("=" * 60)

            status = "✅ PASSED" if overall_passed else "❌ FAILED"
            print(f"Overall Status: {status}")
            print()

            # Threshold violations
            if summary["violations"]:
                print("🚨 THRESHOLD VIOLATIONS:")
                for violation in summary["violations"]:
                    print(f"  ❌ {violation}")
                print()

            # Warnings
            if summary["warnings"]:
                print("⚠️ WARNINGS:")
                for warning in summary["warnings"]:
                    print(f"  ⚠️ {warning}")
                print()

            # Regressions
            if regressions:
                print("📉 REGRESSIONS:")
                for regression in regressions:
                    icon = "🚨" if regression["severity"] == "critical" else "⚠️"
                    print(
                        f"  {icon} {regression['metric']}: {regression['change_percent']:+.1f}% "
                        f"({regression['current']} vs {regression['baseline']})"
                    )
                print()

            # Quality failures
            if quality_failures:
                print("🚫 QUALITY FAILURES:")
                for failure in quality_failures:
                    print(f"  ❌ {failure}")
                print()

            # Summary
            print(f"Metrics checked: {len(thresholds)}")
            print(f"Violations: {summary['total_violations']}")
            print(f"Warnings: {summary['total_warnings']}")
            print(f"Regressions: {len(regressions)}")
            print(f"Quality failures: {len(quality_failures)}")

            if baseline_data:
                print("Baseline comparison: Available")
            else:
                print("Baseline comparison: Not available")

        # Set output for GitHub Actions
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"failed={'true' if not overall_passed else 'false'}\n")
                f.write(f"violations={summary['total_violations']}\n")
                f.write(f"warnings={summary['total_warnings']}\n")
                f.write(f"regressions={len(regressions)}\n")

        # Exit with appropriate code
        sys.exit(0 if overall_passed else 1)

    except Exception as e:
        error_msg = f"❌ Threshold check failed: {e}"
        if args.output_format == "json":
            print(json.dumps({"error": str(e), "passed": False}))
        else:
            print(error_msg)

        sys.exit(1)


if __name__ == "__main__":
    import os

    main()
