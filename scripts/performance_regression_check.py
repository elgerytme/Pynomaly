#!/usr/bin/env python3
"""Performance regression detection for CI/CD pipelines.

This script runs performance benchmarks and compares against baseline
to detect performance regressions in the codebase.
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PerformanceRegression:
    """Performance regression detection."""

    def __init__(self, threshold_percent: float = 20.0):
        """Initialize with regression threshold.

        Args:
            threshold_percent: Percentage threshold for regression detection
        """
        self.threshold_percent = threshold_percent
        self.regressions = []
        self.improvements = []

    def compare_metrics(self, current: dict, baseline: dict) -> dict:
        """Compare current metrics against baseline.

        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics

        Returns:
            Comparison results dictionary
        """
        results = {
            "regressions": [],
            "improvements": [],
            "stable": [],
            "new_metrics": [],
            "missing_metrics": [],
        }

        # Check each baseline metric
        for metric_name, baseline_value in baseline.items():
            if metric_name in current:
                current_value = current[metric_name]
                change_percent = (
                    (current_value - baseline_value) / baseline_value
                ) * 100

                if abs(change_percent) > self.threshold_percent:
                    if change_percent > 0:
                        # Performance degraded
                        results["regressions"].append(
                            {
                                "metric": metric_name,
                                "current": current_value,
                                "baseline": baseline_value,
                                "change_percent": change_percent,
                                "severity": "critical"
                                if change_percent > 50
                                else "warning",
                            }
                        )
                    else:
                        # Performance improved
                        results["improvements"].append(
                            {
                                "metric": metric_name,
                                "current": current_value,
                                "baseline": baseline_value,
                                "change_percent": change_percent,
                            }
                        )
                else:
                    # Performance stable
                    results["stable"].append(
                        {
                            "metric": metric_name,
                            "current": current_value,
                            "baseline": baseline_value,
                            "change_percent": change_percent,
                        }
                    )
            else:
                # Metric missing in current
                results["missing_metrics"].append(metric_name)

        # Check for new metrics
        for metric_name in current:
            if metric_name not in baseline:
                results["new_metrics"].append(
                    {"metric": metric_name, "current": current[metric_name]}
                )

        return results


def run_import_performance_test() -> dict[str, float]:
    """Test import performance of core modules."""
    results = {}

    # Test core imports
    import_tests = [
        ("pynomaly.domain.entities", "domain_entities"),
        ("pynomaly.application.services", "application_services"),
        ("pynomaly.infrastructure.adapters", "infrastructure_adapters"),
        ("pynomaly.presentation.cli", "presentation_cli"),
    ]

    for module_name, test_name in import_tests:
        try:
            start_time = time.time()
            __import__(module_name)
            end_time = time.time()
            results[f"{test_name}_import_time"] = (end_time - start_time) * 1000  # ms
        except ImportError:
            results[f"{test_name}_import_time"] = -1  # Failed to import

    return results


def run_basic_functionality_test() -> dict[str, float]:
    """Test basic functionality performance."""
    results = {}

    try:
        # Test anomaly detection workflow
        start_time = time.time()

        import numpy as np

        from pynomaly.domain.entities import Dataset
        from pynomaly.infrastructure.adapters import SklearnAdapter

        # Create simple test data
        data = np.random.rand(100, 5)
        dataset = Dataset(
            name="perf_test", data=data, features=["f1", "f2", "f3", "f4", "f5"]
        )

        # Test adapter creation and fitting
        adapter = SklearnAdapter("IsolationForest")
        adapter.fit(dataset)

        end_time = time.time()
        results["basic_workflow_time"] = (end_time - start_time) * 1000  # ms

    except Exception as e:
        results["basic_workflow_time"] = -1  # Failed
        results["basic_workflow_error"] = str(e)

    return results


def run_memory_usage_test() -> dict[str, float]:
    """Test memory usage patterns."""
    results = {}

    try:
        import gc

        import psutil

        # Measure baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run memory-intensive operations
        gc.collect()
        start_memory = process.memory_info().rss / 1024 / 1024

        # Create and process data
        import numpy as np

        data = np.random.rand(1000, 20)

        # Simulate processing
        for _ in range(10):
            processed = data * 2
            del processed

        peak_memory = process.memory_info().rss / 1024 / 1024

        gc.collect()
        end_memory = process.memory_info().rss / 1024 / 1024

        results["baseline_memory_mb"] = baseline_memory
        results["peak_memory_mb"] = peak_memory
        results["end_memory_mb"] = end_memory
        results["memory_increase_mb"] = peak_memory - baseline_memory

    except ImportError:
        results["memory_test_available"] = False
    except Exception as e:
        results["memory_test_error"] = str(e)

    return results


def run_startup_performance_test() -> dict[str, float]:
    """Test application startup performance."""
    results = {}

    try:
        # Test CLI startup time
        start_time = time.time()


        end_time = time.time()
        results["cli_startup_time"] = (end_time - start_time) * 1000  # ms

    except Exception as e:
        results["cli_startup_error"] = str(e)
        results["cli_startup_time"] = -1

    try:
        # Test container initialization
        start_time = time.time()

        from pynomaly.infrastructure.config.container import create_container

        container = create_container(testing=True)

        end_time = time.time()
        results["container_init_time"] = (end_time - start_time) * 1000  # ms

    except Exception as e:
        results["container_init_error"] = str(e)
        results["container_init_time"] = -1

    return results


def run_all_performance_tests() -> dict[str, float]:
    """Run comprehensive performance test suite."""
    print("🚀 Running performance tests...")

    all_results = {}

    # Import performance
    print("  📦 Testing import performance...")
    import_results = run_import_performance_test()
    all_results.update(import_results)

    # Basic functionality
    print("  ⚙️ Testing basic functionality...")
    basic_results = run_basic_functionality_test()
    all_results.update(basic_results)

    # Memory usage
    print("  💾 Testing memory usage...")
    memory_results = run_memory_usage_test()
    all_results.update(memory_results)

    # Startup performance
    print("  🏃 Testing startup performance...")
    startup_results = run_startup_performance_test()
    all_results.update(startup_results)

    print("  ✅ Performance tests completed")
    return all_results


def load_baseline(baseline_file: Path) -> dict | None:
    """Load baseline performance metrics."""
    if not baseline_file.exists():
        return None

    try:
        with open(baseline_file) as f:
            data = json.load(f)
            return data.get("performance_metrics", {})
    except Exception as e:
        print(f"⚠️ Could not load baseline: {e}")
        return None


def save_performance_report(results: dict, output_file: Path):
    """Save performance results to file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "performance_metrics": results,
        "git_ref": get_git_ref(),
        "git_commit": get_git_commit(),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)


def get_git_ref() -> str:
    """Get current git reference."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def print_performance_summary(results: dict, comparison: dict = None):
    """Print performance test summary."""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE TEST SUMMARY")
    print("=" * 60)

    # Key metrics
    key_metrics = [
        ("cli_startup_time", "CLI Startup Time", "ms"),
        ("container_init_time", "Container Init Time", "ms"),
        ("basic_workflow_time", "Basic Workflow", "ms"),
        ("peak_memory_mb", "Peak Memory Usage", "MB"),
    ]

    for metric_key, metric_name, unit in key_metrics:
        if metric_key in results:
            value = results[metric_key]
            if value >= 0:
                print(f"📏 {metric_name}: {value:.2f} {unit}")
            else:
                print(f"❌ {metric_name}: Failed")

    # Import times
    import_metrics = {k: v for k, v in results.items() if k.endswith("_import_time")}
    if import_metrics:
        print("\n📦 Import Performance:")
        for metric, time_ms in import_metrics.items():
            name = metric.replace("_import_time", "").replace("_", " ").title()
            if time_ms >= 0:
                print(f"  {name}: {time_ms:.2f} ms")
            else:
                print(f"  {name}: Failed")

    # Regression analysis
    if comparison:
        print("\n📈 REGRESSION ANALYSIS:")

        if comparison["regressions"]:
            print("🚨 REGRESSIONS DETECTED:")
            for reg in comparison["regressions"]:
                icon = "🚨" if reg["severity"] == "critical" else "⚠️"
                print(
                    f"  {icon} {reg['metric']}: {reg['change_percent']:+.1f}% "
                    f"({reg['current']:.2f} vs {reg['baseline']:.2f})"
                )

        if comparison["improvements"]:
            print("✅ IMPROVEMENTS:")
            for imp in comparison["improvements"][:3]:  # Top 3
                print(
                    f"  ✅ {imp['metric']}: {imp['change_percent']:+.1f}% "
                    f"({imp['current']:.2f} vs {imp['baseline']:.2f})"
                )

        stable_count = len(comparison["stable"])
        print(f"➡️ Stable metrics: {stable_count}")

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance regression check")
    parser.add_argument(
        "--baseline", "-b", type=Path, help="Baseline performance data file"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=20.0,
        help="Regression threshold percentage",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output file for performance results"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Fail CI if performance regression detected",
    )

    args = parser.parse_args()

    try:
        # Run performance tests
        current_results = run_all_performance_tests()

        # Load baseline if available
        baseline_results = None
        if args.baseline:
            baseline_results = load_baseline(args.baseline)

        # Compare against baseline
        comparison = None
        has_regressions = False

        if baseline_results:
            regression_checker = PerformanceRegression(args.threshold)
            comparison = regression_checker.compare_metrics(
                current_results, baseline_results
            )
            has_regressions = len(comparison["regressions"]) > 0

            critical_regressions = [
                r for r in comparison["regressions"] if r["severity"] == "critical"
            ]
            if critical_regressions:
                print(
                    f"🚨 {len(critical_regressions)} critical performance regressions detected!"
                )

        # Save results
        if args.output:
            save_performance_report(current_results, args.output)
            print(f"💾 Performance report saved: {args.output}")

        # Print summary
        print_performance_summary(current_results, comparison)

        # Set GitHub Actions outputs
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"has_regressions={'true' if has_regressions else 'false'}\n")
                if comparison:
                    f.write(f"regression_count={len(comparison['regressions'])}\n")
                    f.write(f"improvement_count={len(comparison['improvements'])}\n")

        # Exit based on regression status
        if args.fail_on_regression and has_regressions:
            print("\n❌ Performance regression check FAILED")
            sys.exit(1)
        else:
            print("\n✅ Performance regression check PASSED")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Performance regression check failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
