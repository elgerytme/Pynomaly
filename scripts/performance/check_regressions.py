#!/usr/bin/env python3
"""
Performance Regression Checker CLI Utility for CI/CD Pipeline

This script checks for performance regressions by:
1. Comparing current_results.json against baseline.json
2. Checking for critical performance regressions based on config
3. Printing markdown summary for PR comments if needed

Usage:
    python check_regressions.py --baseline BASELINE_FILE --current CURRENT_FILE --config CONFIG_FILE
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceRegressionChecker:
    """Checks for performance regressions using JSON reports and YAML config."""

    def __init__(self, config_path: Path, baseline_path: Path, current_path: Path):
        self.config_path = config_path
        self.baseline_path = baseline_path
        self.current_path = current_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            sys.exit(1)

        if yaml is None:
            logger.error("PyYAML not available, cannot load config")
            sys.exit(1)

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            sys.exit(1)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded JSON data from {path}")
                return data
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            sys.exit(1)

    def compare_results(self) -> Dict[str, Any]:
        """Compare baseline and current results."""
        # Load JSON files
        baseline_data = self._load_json(self.baseline_path)
        current_data = self._load_json(self.current_path)

        # Compare metrics
        logger.info("Comparing performance metrics...")
        regressions = []
        improvements = []

        baseline_metrics = baseline_data.get("performance_metrics", {})
        current_metrics = current_data.get("performance_metrics", {})

        for algo, baseline_list in baseline_metrics.items():
            current_list = current_metrics.get(algo, [])

            for baseline, current in zip(baseline_list, current_list):
                regression_data = self._compare_algorithm_metrics(
                    algo, baseline, current
                )
                if regression_data.get("is_regression", False):
                    regressions.append(regression_data)
                else:
                    improvements.append(regression_data)

        # Report summary
        summary = {
            "total_regressions": len(regressions),
            "total_improvements": len(improvements),
            "critical_regressions": len(
                [r for r in regressions if r["severity"] == "critical"]
            ),
            "major_regressions": len(
                [r for r in regressions if r["severity"] == "major"]
            ),
            "minor_regressions": len(
                [r for r in regressions if r["severity"] == "minor"]
            ),
            "critical_improvements": len(
                [i for i in improvements if i["severity"] == "critical"]
            ),
            "major_improvements": len(
                [i for i in improvements if i["severity"] == "major"]
            ),
            "minor_improvements": len(
                [i for i in improvements if i["severity"] == "minor"]
            ),
        }

        return {
            "regressions": regressions,
            "improvements": improvements,
            "summary": summary,
        }

    def _compare_algorithm_metrics(
        self, algorithm: str, baseline: Dict[str, Any], current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare individual algorithm's performance metrics."""
        severity_thresholds = self.config.get("severity_thresholds", {})

        metric_comparisons = {
            "execution_time": {
                "current": current.get("execution_time_seconds", 0.0),
                "baseline": baseline.get("execution_time_seconds", 0.0),
                "unit": "seconds",
                "lower_is_better": True,
            },
            "memory_usage": {
                "current": current.get("peak_memory_mb", 0.0),
                "baseline": baseline.get("peak_memory_mb", 0.0),
                "unit": "MB",
                "lower_is_better": True,
            },
            "accuracy": {
                "current": current.get("accuracy_score", 0.0),
                "baseline": baseline.get("accuracy_score", 0.0),
                "unit": "score",
                "lower_is_better": False,
            },
            "throughput": {
                "current": current.get("training_throughput", 0.0),
                "baseline": baseline.get("training_throughput", 0.0),
                "unit": "samples/sec",
                "lower_is_better": False,
            },
        }

        regressions_data = []
        for metric_name, metric_data in metric_comparisons.items():
            current_value = metric_data["current"]
            baseline_value = metric_data["baseline"]

            if baseline_value == 0:
                continue  # Skip if no baseline available

            percent_change = ((current_value - baseline_value) / baseline_value) * 100
            absolute_change = current_value - baseline_value

            # Determine if this is a regression or improvement
            is_regression = (metric_data["lower_is_better"] and percent_change > 0) or (
                not metric_data["lower_is_better"] and percent_change < 0
            )

            # Calculate severity based on absolute percentage change
            abs_percent_change = abs(percent_change)
            severity = self._calculate_severity(abs_percent_change, severity_thresholds)

            # Record significant changes
            if (
                abs_percent_change
                >= severity_thresholds.get("minor_threshold", 0.0) * 100
            ):
                regressions_data.append(
                    {
                        "algorithm": algorithm,
                        "metric": metric_name,
                        "current_value": current_value,
                        "baseline_value": baseline_value,
                        "absolute_change": absolute_change,
                        "percent_change": percent_change,
                        "severity": severity,
                        "unit": metric_data["unit"],
                        "is_regression": is_regression,
                    }
                )

        # Select most severe regression or improvement
        if regressions_data:
            return max(regressions_data, key=lambda x: x.get("severity", "negligible"))
        else:
            return {}

    def _calculate_severity(
        self, percent_change: float, thresholds: Dict[str, float]
    ) -> str:
        """Calculate severity based on percentage change."""
        if percent_change >= thresholds.get("critical_threshold", 0.30) * 100:
            return "critical"
        elif percent_change >= thresholds.get("major_threshold", 0.15) * 100:
            return "major"
        elif percent_change >= thresholds.get("minor_threshold", 0.05) * 100:
            return "minor"
        else:
            return "negligible"

    def generate_markdown_summary(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a markdown summary for the comparison results."""
        summary = comparison_results["summary"]

        lines = [
            "# Performance Regression Summary\n",
            f"**Total Regressions:** {summary['total_regressions']}\n",
            f"**Total Improvements:** {summary['total_improvements']}\n",
            f"**Critical Regressions:** {summary['critical_regressions']}\n",
            "\n## Detailed Regressions\n",
        ]

        for regression in comparison_results["regressions"]:
            lines.append(
                f"- **Algorithm:** {regression['algorithm']}\n"
                f"  - **Metric:** {regression['metric']}\n"
                f"  - **Current Value:** {regression['current_value']} {regression['unit']}\n"
                f"  - **Baseline Value:** {regression['baseline_value']} {regression['unit']}\n"
                f"  - **Change:** {regression['absolute_change']} {regression['unit']}\n"
                f"  - **Percent Change:** {regression['percent_change']:.2f}%\n"
                f"  - **Severity:** {regression['severity']}\n\n"
            )

        return "".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check performance regressions based on JSON results and YAML config"
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Path to baseline JSON file"
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="Path to current results JSON file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to performance configuration YAML file",
    )

    args = parser.parse_args()

    # Create checker
    checker = PerformanceRegressionChecker(
        config_path=args.config, baseline_path=args.baseline, current_path=args.current
    )

    # Compare results
    comparison_results = checker.compare_results()

    # Generate markdown summary
    markdown_summary = checker.generate_markdown_summary(comparison_results)
    print(markdown_summary)

    # Exit with failure if critical regressions exist
    if comparison_results["summary"]["critical_regressions"] > 0:
        logger.error("Critical regressions found!")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
