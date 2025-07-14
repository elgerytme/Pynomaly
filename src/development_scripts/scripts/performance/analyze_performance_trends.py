#!/usr/bin/env python3
"""
Performance Trends Analysis Script

This script analyzes performance test results over time to identify trends,
regressions, and improvements in system performance.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print(
        "Warning: matplotlib and pandas not available. Visualizations will be skipped."
    )

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class PerformanceTrendAnalyzer:
    """Analyzes performance trends from test results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.test_results: list[dict[str, Any]] = []
        self.trends: dict[str, Any] = {}

    def load_test_results(self) -> int:
        """Load all performance test results from the directory."""
        console.print("🔍 Loading performance test results...")

        loaded_count = 0

        # Recursively find all JSON result files
        for json_file in self.results_dir.rglob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Check if this is a valid performance test result
                if self._is_valid_result(data):
                    # Add metadata
                    data["file_path"] = str(json_file)
                    data["file_name"] = json_file.name

                    # Parse timestamp from filename or data
                    timestamp = self._extract_timestamp(data, json_file.name)
                    if timestamp:
                        data["parsed_timestamp"] = timestamp
                        self.test_results.append(data)
                        loaded_count += 1

            except Exception as e:
                console.print(f"Warning: Could not load {json_file}: {e}")

        # Sort by timestamp
        self.test_results.sort(key=lambda x: x.get("parsed_timestamp", datetime.min))

        console.print(f"✅ Loaded {loaded_count} performance test results")
        return loaded_count

    def _is_valid_result(self, data: dict[str, Any]) -> bool:
        """Check if the JSON data is a valid performance test result."""
        required_fields = ["total_requests", "avg_response_time_ms", "throughput_rps"]
        return all(field in data for field in required_fields)

    def _extract_timestamp(
        self, data: dict[str, Any], filename: str
    ) -> datetime | None:
        """Extract timestamp from test data or filename."""
        # Try to get timestamp from data
        if "test_timestamp" in data:
            try:
                return datetime.fromisoformat(
                    data["test_timestamp"].replace("Z", "+00:00")
                )
            except:
                pass

        # Try to parse from filename (format: load_test_YYYYMMDD_HHMMSS.json)
        try:
            import re

            match = re.search(r"(\d{8}_\d{6})", filename)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            pass

        # Fall back to file modification time
        try:
            file_path = Path(data.get("file_path", ""))
            if file_path.exists():
                return datetime.fromtimestamp(file_path.stat().st_mtime)
        except:
            pass

        return None

    def analyze_trends(self) -> dict[str, Any]:
        """Analyze performance trends across all test results."""
        if not self.test_results:
            console.print("❌ No test results to analyze")
            return {}

        console.print("📈 Analyzing performance trends...")

        # Group results by scenario
        scenarios = {}
        for result in self.test_results:
            scenario = result.get("scenario", "unknown")
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(result)

        self.trends = {
            "summary": self._analyze_overall_trends(),
            "scenarios": {},
            "metrics_over_time": self._analyze_metrics_over_time(),
            "performance_regressions": self._detect_regressions(),
            "performance_improvements": self._detect_improvements(),
            "recommendations": [],
        }

        # Analyze each scenario
        for scenario, results in scenarios.items():
            self.trends["scenarios"][scenario] = self._analyze_scenario_trends(
                scenario, results
            )

        # Generate recommendations
        self.trends["recommendations"] = self._generate_recommendations()

        return self.trends

    def _analyze_overall_trends(self) -> dict[str, Any]:
        """Analyze overall performance trends."""
        if len(self.test_results) < 2:
            return {"status": "insufficient_data"}

        # Calculate metrics for first and last results
        first_result = self.test_results[0]
        last_result = self.test_results[-1]

        response_time_change = (
            (last_result["avg_response_time_ms"] - first_result["avg_response_time_ms"])
            / first_result["avg_response_time_ms"]
            * 100
        )

        throughput_change = (
            (last_result["throughput_rps"] - first_result["throughput_rps"])
            / first_result["throughput_rps"]
            * 100
        )

        error_rate_change = last_result.get("error_rate_percent", 0) - first_result.get(
            "error_rate_percent", 0
        )

        return {
            "timespan_days": (
                last_result["parsed_timestamp"] - first_result["parsed_timestamp"]
            ).days,
            "total_tests": len(self.test_results),
            "response_time_change_percent": response_time_change,
            "throughput_change_percent": throughput_change,
            "error_rate_change_percent": error_rate_change,
            "trend_direction": self._determine_trend_direction(
                response_time_change, throughput_change, error_rate_change
            ),
        }

    def _determine_trend_direction(
        self,
        response_time_change: float,
        throughput_change: float,
        error_rate_change: float,
    ) -> str:
        """Determine overall trend direction."""
        score = 0

        # Response time (lower is better)
        if response_time_change < -5:
            score += 1
        elif response_time_change > 10:
            score -= 1

        # Throughput (higher is better)
        if throughput_change > 5:
            score += 1
        elif throughput_change < -10:
            score -= 1

        # Error rate (lower is better)
        if error_rate_change < -1:
            score += 1
        elif error_rate_change > 1:
            score -= 1

        if score > 0:
            return "improving"
        elif score < 0:
            return "degrading"
        else:
            return "stable"

    def _analyze_metrics_over_time(self) -> dict[str, list[tuple[datetime, float]]]:
        """Analyze how metrics change over time."""
        metrics = {
            "response_time": [],
            "throughput": [],
            "error_rate": [],
            "p95_response_time": [],
        }

        for result in self.test_results:
            timestamp = result["parsed_timestamp"]
            metrics["response_time"].append((timestamp, result["avg_response_time_ms"]))
            metrics["throughput"].append((timestamp, result["throughput_rps"]))
            metrics["error_rate"].append(
                (timestamp, result.get("error_rate_percent", 0))
            )
            metrics["p95_response_time"].append(
                (timestamp, result.get("p95_response_time_ms", 0))
            )

        return metrics

    def _detect_regressions(self) -> list[dict[str, Any]]:
        """Detect performance regressions."""
        regressions = []

        if len(self.test_results) < 3:
            return regressions

        # Use a sliding window to detect significant changes
        window_size = min(5, len(self.test_results) // 2)

        for i in range(window_size, len(self.test_results)):
            recent_window = self.test_results[i - window_size : i]
            current_result = self.test_results[i]

            # Calculate baseline from recent window
            baseline_response_time = statistics.mean(
                [r["avg_response_time_ms"] for r in recent_window]
            )
            baseline_throughput = statistics.mean(
                [r["throughput_rps"] for r in recent_window]
            )
            baseline_error_rate = statistics.mean(
                [r.get("error_rate_percent", 0) for r in recent_window]
            )

            # Check for regressions
            response_time_increase = (
                (current_result["avg_response_time_ms"] - baseline_response_time)
                / baseline_response_time
                * 100
            )
            throughput_decrease = (
                (baseline_throughput - current_result["throughput_rps"])
                / baseline_throughput
                * 100
            )
            error_rate_increase = (
                current_result.get("error_rate_percent", 0) - baseline_error_rate
            )

            # Define regression thresholds
            if (
                response_time_increase > 20
                or throughput_decrease > 15
                or error_rate_increase > 2
            ):
                regressions.append(
                    {
                        "timestamp": current_result["parsed_timestamp"],
                        "response_time_increase_percent": response_time_increase,
                        "throughput_decrease_percent": throughput_decrease,
                        "error_rate_increase_percent": error_rate_increase,
                        "severity": self._calculate_regression_severity(
                            response_time_increase,
                            throughput_decrease,
                            error_rate_increase,
                        ),
                    }
                )

        return regressions

    def _detect_improvements(self) -> list[dict[str, Any]]:
        """Detect performance improvements."""
        improvements = []

        if len(self.test_results) < 3:
            return improvements

        window_size = min(5, len(self.test_results) // 2)

        for i in range(window_size, len(self.test_results)):
            recent_window = self.test_results[i - window_size : i]
            current_result = self.test_results[i]

            baseline_response_time = statistics.mean(
                [r["avg_response_time_ms"] for r in recent_window]
            )
            baseline_throughput = statistics.mean(
                [r["throughput_rps"] for r in recent_window]
            )
            baseline_error_rate = statistics.mean(
                [r.get("error_rate_percent", 0) for r in recent_window]
            )

            response_time_decrease = (
                (baseline_response_time - current_result["avg_response_time_ms"])
                / baseline_response_time
                * 100
            )
            throughput_increase = (
                (current_result["throughput_rps"] - baseline_throughput)
                / baseline_throughput
                * 100
            )
            error_rate_decrease = baseline_error_rate - current_result.get(
                "error_rate_percent", 0
            )

            # Define improvement thresholds
            if (
                response_time_decrease > 15
                or throughput_increase > 20
                or error_rate_decrease > 1
            ):
                improvements.append(
                    {
                        "timestamp": current_result["parsed_timestamp"],
                        "response_time_decrease_percent": response_time_decrease,
                        "throughput_increase_percent": throughput_increase,
                        "error_rate_decrease_percent": error_rate_decrease,
                        "impact": self._calculate_improvement_impact(
                            response_time_decrease,
                            throughput_increase,
                            error_rate_decrease,
                        ),
                    }
                )

        return improvements

    def _calculate_regression_severity(
        self,
        response_time_increase: float,
        throughput_decrease: float,
        error_rate_increase: float,
    ) -> str:
        """Calculate severity of performance regression."""
        if (
            response_time_increase > 50
            or throughput_decrease > 40
            or error_rate_increase > 5
        ):
            return "critical"
        elif (
            response_time_increase > 30
            or throughput_decrease > 25
            or error_rate_increase > 3
        ):
            return "high"
        elif (
            response_time_increase > 20
            or throughput_decrease > 15
            or error_rate_increase > 2
        ):
            return "medium"
        else:
            return "low"

    def _calculate_improvement_impact(
        self,
        response_time_decrease: float,
        throughput_increase: float,
        error_rate_decrease: float,
    ) -> str:
        """Calculate impact of performance improvement."""
        if (
            response_time_decrease > 40
            or throughput_increase > 50
            or error_rate_decrease > 3
        ):
            return "significant"
        elif (
            response_time_decrease > 25
            or throughput_increase > 30
            or error_rate_decrease > 2
        ):
            return "moderate"
        else:
            return "minor"

    def _analyze_scenario_trends(
        self, scenario: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze trends for a specific scenario."""
        if len(results) < 2:
            return {"status": "insufficient_data"}

        response_times = [r["avg_response_time_ms"] for r in results]
        throughputs = [r["throughput_rps"] for r in results]
        error_rates = [r.get("error_rate_percent", 0) for r in results]

        return {
            "test_count": len(results),
            "response_time_stats": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "stdev": statistics.stdev(response_times)
                if len(response_times) > 1
                else 0,
                "min": min(response_times),
                "max": max(response_times),
            },
            "throughput_stats": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                "min": min(throughputs),
                "max": max(throughputs),
            },
            "error_rate_stats": {
                "mean": statistics.mean(error_rates),
                "max": max(error_rates),
                "tests_with_errors": len([r for r in error_rates if r > 0]),
            },
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not self.trends:
            return recommendations

        summary = self.trends.get("summary", {})
        regressions = self.trends.get("performance_regressions", [])

        # Check trend direction
        if summary.get("trend_direction") == "degrading":
            recommendations.append(
                "Performance is trending downward. Consider investigating recent changes."
            )

        # Check for critical regressions
        critical_regressions = [
            r for r in regressions if r.get("severity") == "critical"
        ]
        if critical_regressions:
            recommendations.append(
                f"Critical performance regressions detected in {len(critical_regressions)} test(s). Immediate investigation required."
            )

        # Check response time
        if summary.get("response_time_change_percent", 0) > 25:
            recommendations.append(
                "Response times have increased significantly. Consider optimizing database queries and adding caching."
            )

        # Check throughput
        if summary.get("throughput_change_percent", 0) < -20:
            recommendations.append(
                "Throughput has decreased significantly. Consider horizontal scaling or connection pool optimization."
            )

        # Check error rates
        if summary.get("error_rate_change_percent", 0) > 2:
            recommendations.append(
                "Error rates are increasing. Review error handling and system stability."
            )

        # General recommendations
        if len(self.test_results) > 10:
            recommendations.append(
                "Consider establishing automated alerts for performance threshold violations."
            )

        if not recommendations:
            recommendations.append(
                "Performance appears stable. Continue regular monitoring and testing."
            )

        return recommendations

    def generate_visualizations(self, output_dir: str):
        """Generate performance trend visualizations."""
        if not VISUALIZATION_AVAILABLE:
            console.print(
                "❌ Matplotlib/pandas not available. Skipping visualizations."
            )
            return

        if not self.test_results:
            console.print("❌ No data available for visualization.")
            return

        console.print("📊 Generating performance visualizations...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data
        timestamps = [r["parsed_timestamp"] for r in self.test_results]
        response_times = [r["avg_response_time_ms"] for r in self.test_results]
        throughputs = [r["throughput_rps"] for r in self.test_results]
        error_rates = [r.get("error_rate_percent", 0) for r in self.test_results]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Performance Trends Over Time", fontsize=16)

        # Response time trend
        ax1.plot(timestamps, response_times, "b-", marker="o", markersize=3)
        ax1.set_title("Average Response Time")
        ax1.set_ylabel("Response Time (ms)")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Throughput trend
        ax2.plot(timestamps, throughputs, "g-", marker="o", markersize=3)
        ax2.set_title("Throughput")
        ax2.set_ylabel("Requests per Second")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # Error rate trend
        ax3.plot(timestamps, error_rates, "r-", marker="o", markersize=3)
        ax3.set_title("Error Rate")
        ax3.set_ylabel("Error Rate (%)")
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        # Performance score (composite metric)
        performance_scores = []
        for rt, tp, er in zip(response_times, throughputs, error_rates, strict=False):
            # Simple scoring: lower response time and error rate, higher throughput is better
            score = (1000 / rt) * (tp / 10) * (1 - er / 100)
            performance_scores.append(score)

        ax4.plot(timestamps, performance_scores, "purple", marker="o", markersize=3)
        ax4.set_title("Composite Performance Score")
        ax4.set_ylabel("Performance Score")
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            output_path / "performance_trends.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        console.print(
            f"✅ Visualization saved to {output_path / 'performance_trends.png'}"
        )

    def print_analysis_summary(self):
        """Print a comprehensive analysis summary."""
        if not self.trends:
            console.print("❌ No analysis data available")
            return

        console.print(Panel.fit("📊 Performance Trends Analysis", style="bold blue"))

        summary = self.trends.get("summary", {})

        # Overall summary table
        summary_table = Table(title="Overall Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Total Tests", str(summary.get("total_tests", 0)))
        summary_table.add_row("Timespan", f"{summary.get('timespan_days', 0)} days")
        summary_table.add_row(
            "Trend Direction", summary.get("trend_direction", "unknown").title()
        )
        summary_table.add_row(
            "Response Time Change",
            f"{summary.get('response_time_change_percent', 0):.1f}%",
        )
        summary_table.add_row(
            "Throughput Change", f"{summary.get('throughput_change_percent', 0):.1f}%"
        )
        summary_table.add_row(
            "Error Rate Change", f"{summary.get('error_rate_change_percent', 0):.1f}%"
        )

        console.print(summary_table)

        # Regressions and improvements
        regressions = self.trends.get("performance_regressions", [])
        improvements = self.trends.get("performance_improvements", [])

        if regressions:
            console.print(
                f"\n⚠️  {len(regressions)} Performance Regression(s) Detected:"
            )
            for regression in regressions[-3:]:  # Show last 3
                severity = regression.get("severity", "unknown")
                timestamp = regression.get("timestamp", "unknown")
                console.print(f"  • {timestamp} - Severity: {severity.upper()}")

        if improvements:
            console.print(
                f"\n✅ {len(improvements)} Performance Improvement(s) Detected:"
            )
            for improvement in improvements[-3:]:  # Show last 3
                impact = improvement.get("impact", "unknown")
                timestamp = improvement.get("timestamp", "unknown")
                console.print(f"  • {timestamp} - Impact: {impact.upper()}")

        # Recommendations
        recommendations = self.trends.get("recommendations", [])
        if recommendations:
            console.print("\n💡 Recommendations:")
            for recommendation in recommendations:
                console.print(f"  • {recommendation}")

    def save_analysis_report(self, output_path: str):
        """Save detailed analysis report to JSON."""
        # Convert datetime objects to strings for JSON serialization
        serializable_trends = self._make_json_serializable(self.trends)

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(serializable_trends, f, indent=2)

        console.print(f"📄 Analysis report saved to: {output_path}")

    def _make_json_serializable(self, obj):
        """Convert datetime objects to strings for JSON serialization."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


def main():
    """Main entry point for performance trend analysis."""
    parser = argparse.ArgumentParser(description="Analyze Pynomaly Performance Trends")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing performance test results",
    )
    parser.add_argument(
        "--output-dir",
        default="performance_analysis",
        help="Output directory for analysis",
    )
    parser.add_argument(
        "--visualizations", action="store_true", help="Generate visualization charts"
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=2,
        help="Minimum number of results required for analysis",
    )

    args = parser.parse_args()

    try:
        # Create analyzer
        analyzer = PerformanceTrendAnalyzer(args.results_dir)

        # Load test results
        loaded_count = analyzer.load_test_results()

        if loaded_count < args.min_results:
            console.print(
                f"❌ Insufficient data: {loaded_count} results found, {args.min_results} required"
            )
            sys.exit(1)

        # Analyze trends
        trends = analyzer.analyze_trends()

        # Print summary
        analyzer.print_analysis_summary()

        # Save analysis report
        os.makedirs(args.output_dir, exist_ok=True)
        analyzer.save_analysis_report(f"{args.output_dir}/performance_analysis.json")

        # Generate visualizations if requested
        if args.visualizations:
            analyzer.generate_visualizations(args.output_dir)

        console.print(
            Panel.fit("✅ Performance Analysis Completed", style="bold green")
        )

    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", style="bold red")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
