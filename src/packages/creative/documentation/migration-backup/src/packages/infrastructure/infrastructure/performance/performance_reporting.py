"""
Performance Reporting and Trend Analysis System.

This module provides comprehensive performance reporting, trend analysis,
and visualization capabilities for the Pynomaly system.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceTrend:
    """Performance trend analysis results."""

    metric_name: str = ""
    time_period_days: int = 0

    # Trend analysis
    trend_direction: str = "stable"  # increasing, decreasing, stable, volatile
    trend_strength: float = 0.0  # 0-1, how strong the trend is
    change_percentage: float = 0.0  # percentage change over period

    # Statistical measures
    mean_value: float = 0.0
    median_value: float = 0.0
    std_deviation: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    # Trend points
    data_points: int = 0
    significant_events: list[dict[str, Any]] = field(default_factory=list)
    anomalous_periods: list[dict[str, Any]] = field(default_factory=list)

    # Forecasting
    predicted_next_value: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    # Recommendations
    trend_assessment: str = "normal"  # excellent, good, normal, concerning, critical
    recommendations: list[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    report_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    report_type: str = "comprehensive"  # summary, detailed, comprehensive
    time_period: str = "24h"

    # Executive summary
    overall_health_score: float = 0.0
    performance_grade: str = "B"
    key_insights: list[str] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)

    # Detailed metrics
    execution_time_analysis: dict[str, Any] = field(default_factory=dict)
    memory_usage_analysis: dict[str, Any] = field(default_factory=dict)
    throughput_analysis: dict[str, Any] = field(default_factory=dict)
    error_analysis: dict[str, Any] = field(default_factory=dict)

    # Trends
    performance_trends: list[PerformanceTrend] = field(default_factory=list)

    # Comparisons
    algorithm_comparisons: dict[str, Any] = field(default_factory=dict)
    historical_comparisons: dict[str, Any] = field(default_factory=dict)

    # Recommendations
    immediate_actions: list[str] = field(default_factory=list)
    optimization_opportunities: list[str] = field(default_factory=list)
    long_term_recommendations: list[str] = field(default_factory=list)

    # Metadata
    system_info: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)


class TrendAnalyzer:
    """Advanced trend analysis for performance metrics."""

    def __init__(self):
        self.trend_models = {}
        logger.info("Trend analyzer initialized")

    def analyze_metric_trend(
        self,
        data_points: list[tuple[datetime, float]],
        metric_name: str,
        time_period_days: int,
    ) -> PerformanceTrend:
        """Analyze trend for a specific metric."""
        if len(data_points) < 2:
            return PerformanceTrend(
                metric_name=metric_name,
                time_period_days=time_period_days,
                trend_direction="insufficient_data",
            )

        # Sort by timestamp
        sorted_data = sorted(data_points, key=lambda x: x[0])
        timestamps = [point[0] for point in sorted_data]
        values = [point[1] for point in sorted_data]

        # Basic statistical analysis
        trend = PerformanceTrend(
            metric_name=metric_name,
            time_period_days=time_period_days,
            data_points=len(values),
        )

        # Calculate basic statistics
        trend.mean_value = statistics.mean(values)
        trend.median_value = statistics.median(values)
        trend.std_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        trend.min_value = min(values)
        trend.max_value = max(values)

        # Analyze trend direction and strength
        trend.trend_direction, trend.trend_strength = self._calculate_trend_direction(
            values
        )
        trend.change_percentage = self._calculate_change_percentage(values)

        # Detect anomalous periods
        trend.anomalous_periods = self._detect_anomalous_periods(timestamps, values)

        # Detect significant events
        trend.significant_events = self._detect_significant_events(timestamps, values)

        # Simple forecasting
        trend.predicted_next_value, trend.confidence_interval = self._simple_forecast(
            values
        )

        # Assessment and recommendations
        trend.trend_assessment = self._assess_trend_health(trend)
        trend.recommendations = self._generate_trend_recommendations(trend)

        return trend

    def _calculate_trend_direction(self, values: list[float]) -> tuple[str, float]:
        """Calculate trend direction and strength."""
        if len(values) < 3:
            return "stable", 0.0

        # Linear regression approach
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1] if len(values) > 1 else 0.0

        # Determine direction
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        relative_change = (
            abs(second_avg - first_avg) / first_avg if first_avg != 0 else 0
        )

        # Classify trend
        if relative_change < 0.05:
            direction = "stable"
        elif second_avg > first_avg:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Check for volatility
        if (
            trend.std_deviation / trend.mean_value > 0.3
        ):  # High coefficient of variation
            direction = "volatile"

        strength = min(abs(correlation), 1.0)

        return direction, strength

    def _calculate_change_percentage(self, values: list[float]) -> float:
        """Calculate percentage change over the period."""
        if len(values) < 2 or values[0] == 0:
            return 0.0

        return ((values[-1] - values[0]) / values[0]) * 100

    def _detect_anomalous_periods(
        self, timestamps: list[datetime], values: list[float]
    ) -> list[dict[str, Any]]:
        """Detect periods with anomalous performance."""
        if len(values) < 10:  # Need enough data points
            return []

        anomalies = []

        # Calculate rolling statistics
        window_size = min(5, len(values) // 3)
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)

        # Z-score based anomaly detection
        threshold = 2.0  # 2 standard deviations

        for i in range(len(values)):
            z_score = abs(values[i] - mean_val) / std_val if std_val > 0 else 0

            if z_score > threshold:
                anomalies.append(
                    {
                        "timestamp": timestamps[i].isoformat(),
                        "value": values[i],
                        "z_score": z_score,
                        "type": "outlier",
                    }
                )

        # Group consecutive anomalies into periods
        anomalous_periods = []
        if anomalies:
            current_period = [anomalies[0]]

            for anomaly in anomalies[1:]:
                # Check if within 1 hour of previous anomaly
                prev_time = datetime.fromisoformat(current_period[-1]["timestamp"])
                curr_time = datetime.fromisoformat(anomaly["timestamp"])

                if (curr_time - prev_time).total_seconds() <= 3600:  # 1 hour
                    current_period.append(anomaly)
                else:
                    # Finalize current period
                    if len(current_period) > 1:
                        anomalous_periods.append(
                            {
                                "start_time": current_period[0]["timestamp"],
                                "end_time": current_period[-1]["timestamp"],
                                "duration_minutes": (
                                    datetime.fromisoformat(
                                        current_period[-1]["timestamp"]
                                    )
                                    - datetime.fromisoformat(
                                        current_period[0]["timestamp"]
                                    )
                                ).total_seconds()
                                / 60,
                                "anomaly_count": len(current_period),
                                "severity": "high"
                                if len(current_period) > 5
                                else "medium",
                            }
                        )
                    current_period = [anomaly]

            # Handle last period
            if len(current_period) > 1:
                anomalous_periods.append(
                    {
                        "start_time": current_period[0]["timestamp"],
                        "end_time": current_period[-1]["timestamp"],
                        "duration_minutes": (
                            datetime.fromisoformat(current_period[-1]["timestamp"])
                            - datetime.fromisoformat(current_period[0]["timestamp"])
                        ).total_seconds()
                        / 60,
                        "anomaly_count": len(current_period),
                        "severity": "high" if len(current_period) > 5 else "medium",
                    }
                )

        return anomalous_periods

    def _detect_significant_events(
        self, timestamps: list[datetime], values: list[float]
    ) -> list[dict[str, Any]]:
        """Detect significant performance events."""
        if len(values) < 5:
            return []

        events = []

        # Look for sudden changes
        for i in range(1, len(values)):
            prev_val = values[i - 1]
            curr_val = values[i]

            if prev_val != 0:
                change_pct = abs((curr_val - prev_val) / prev_val) * 100

                if change_pct > 50:  # 50% change
                    event_type = (
                        "performance_spike"
                        if curr_val > prev_val
                        else "performance_drop"
                    )
                    events.append(
                        {
                            "timestamp": timestamps[i].isoformat(),
                            "type": event_type,
                            "change_percentage": change_pct,
                            "previous_value": prev_val,
                            "current_value": curr_val,
                            "severity": "high" if change_pct > 100 else "medium",
                        }
                    )

        # Look for sustained improvements or degradations
        window_size = min(5, len(values) // 4)
        for i in range(window_size, len(values) - window_size):
            before_window = values[i - window_size : i]
            after_window = values[i : i + window_size]

            before_avg = statistics.mean(before_window)
            after_avg = statistics.mean(after_window)

            if before_avg != 0:
                sustained_change = ((after_avg - before_avg) / before_avg) * 100

                if abs(sustained_change) > 25:  # 25% sustained change
                    event_type = (
                        "sustained_improvement"
                        if sustained_change < 0
                        else "sustained_degradation"
                    )
                    events.append(
                        {
                            "timestamp": timestamps[i].isoformat(),
                            "type": event_type,
                            "sustained_change_percentage": sustained_change,
                            "window_size": window_size,
                            "severity": "medium",
                        }
                    )

        return events

    def _simple_forecast(
        self, values: list[float]
    ) -> tuple[float, tuple[float, float]]:
        """Simple forecasting using linear trend."""
        if len(values) < 3:
            last_val = values[-1] if values else 0.0
            return last_val, (last_val, last_val)

        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope and intercept
        slope, intercept = np.polyfit(x, y, 1)

        # Predict next value
        next_x = len(values)
        predicted = slope * next_x + intercept

        # Simple confidence interval based on recent variance
        recent_values = values[-min(5, len(values)) :]
        std_recent = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0

        confidence_margin = 1.96 * std_recent  # 95% confidence interval
        lower_bound = predicted - confidence_margin
        upper_bound = predicted + confidence_margin

        return predicted, (lower_bound, upper_bound)

    def _assess_trend_health(self, trend: PerformanceTrend) -> str:
        """Assess the health of a performance trend."""
        # Assess based on direction, anomalies, and volatility
        score = 100

        # Penalize negative trends (for metrics where lower is better)
        if trend.metric_name in ["execution_time", "memory_usage", "error_rate"]:
            if trend.trend_direction == "increasing":
                score -= 30
            elif trend.trend_direction == "decreasing":
                score += 10
        else:  # For metrics where higher is better (throughput, accuracy)
            if trend.trend_direction == "decreasing":
                score -= 30
            elif trend.trend_direction == "increasing":
                score += 10

        # Penalize volatility
        if trend.trend_direction == "volatile":
            score -= 25

        # Penalize anomalous periods
        score -= len(trend.anomalous_periods) * 10

        # Penalize significant events
        high_severity_events = sum(
            1 for event in trend.significant_events if event.get("severity") == "high"
        )
        score -= high_severity_events * 15

        # Classify
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "normal"
        elif score >= 40:
            return "concerning"
        else:
            return "critical"

    def _generate_trend_recommendations(self, trend: PerformanceTrend) -> list[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []

        if trend.trend_assessment in ["concerning", "critical"]:
            recommendations.append("Immediate investigation required for this metric")

        if trend.trend_direction == "volatile":
            recommendations.append("Consider implementing stabilization measures")

        if len(trend.anomalous_periods) > 0:
            recommendations.append(
                "Investigate causes of anomalous performance periods"
            )

        high_severity_events = [
            e for e in trend.significant_events if e.get("severity") == "high"
        ]
        if high_severity_events:
            recommendations.append("Analyze high-severity performance events")

        if trend.change_percentage > 50:
            recommendations.append(
                "Significant change detected - verify system changes"
            )

        # Metric-specific recommendations
        if trend.metric_name == "execution_time":
            if trend.trend_direction == "increasing":
                recommendations.append(
                    "Consider algorithm optimization or parallel processing"
                )
        elif trend.metric_name == "memory_usage":
            if trend.trend_direction == "increasing":
                recommendations.append("Investigate potential memory leaks")
        elif trend.metric_name == "throughput":
            if trend.trend_direction == "decreasing":
                recommendations.append("Analyze bottlenecks limiting system throughput")

        return recommendations


class PerformanceReporter:
    """Comprehensive performance reporting system."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.trend_analyzer = TrendAnalyzer()
        self.report_templates = self._load_report_templates()

        logger.info("Performance reporter initialized")

    def generate_comprehensive_report(
        self,
        performance_data: list[dict[str, Any]],
        time_period: str = "24h",
        include_visualizations: bool = True,
    ) -> PerformanceReport:
        """Generate a comprehensive performance report."""
        report = PerformanceReport(
            report_id=f"perf_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            report_type="comprehensive",
            time_period=time_period,
        )

        if not performance_data:
            report.key_insights.append("No performance data available for analysis")
            return report

        # Extract time period in hours
        period_hours = self._parse_time_period(time_period)
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)

        # Filter data to time period
        filtered_data = [
            entry
            for entry in performance_data
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]

        if not filtered_data:
            report.key_insights.append(
                f"No performance data available for the last {time_period}"
            )
            return report

        # Analyze different aspects
        report.execution_time_analysis = self._analyze_execution_times(filtered_data)
        report.memory_usage_analysis = self._analyze_memory_usage(filtered_data)
        report.throughput_analysis = self._analyze_throughput(filtered_data)
        report.error_analysis = self._analyze_errors(filtered_data)

        # Trend analysis
        report.performance_trends = self._analyze_trends(filtered_data, period_hours)

        # Algorithm comparisons
        report.algorithm_comparisons = self._compare_algorithms(filtered_data)

        # Historical comparisons
        report.historical_comparisons = self._compare_with_historical(
            filtered_data, performance_data
        )

        # Calculate overall health score
        report.overall_health_score = self._calculate_overall_health_score(report)
        report.performance_grade = self._assign_performance_grade(
            report.overall_health_score
        )

        # Generate insights and recommendations
        report.key_insights = self._generate_key_insights(report)
        report.critical_issues = self._identify_critical_issues(report)
        report.immediate_actions = self._generate_immediate_actions(report)
        report.optimization_opportunities = self._identify_optimization_opportunities(
            report
        )
        report.long_term_recommendations = self._generate_long_term_recommendations(
            report
        )

        # Add system information
        report.system_info = self._get_system_info()

        return report

    def export_report(
        self,
        report: PerformanceReport,
        format: str = "html",
        include_charts: bool = True,
    ) -> Path:
        """Export report to specified format."""
        if format == "html":
            return self._export_html_report(report, include_charts)
        elif format == "json":
            return self._export_json_report(report)
        elif format == "pdf":
            return self._export_pdf_report(report, include_charts)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _analyze_execution_times(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze execution time patterns."""
        execution_times = []

        for entry in data:
            exec_time = entry.get("execution_metrics", {}).get("execution_time", 0)
            if exec_time > 0:
                execution_times.append(exec_time)

        if not execution_times:
            return {"message": "No execution time data available"}

        return {
            "total_operations": len(execution_times),
            "average_time": statistics.mean(execution_times),
            "median_time": statistics.median(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times)
            if len(execution_times) > 1
            else 0,
            "percentiles": {
                "90th": np.percentile(execution_times, 90),
                "95th": np.percentile(execution_times, 95),
                "99th": np.percentile(execution_times, 99),
            },
            "performance_distribution": self._categorize_performance(execution_times),
        }

    def _analyze_memory_usage(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_growths = []
        peak_memories = []

        for entry in data:
            exec_metrics = entry.get("execution_metrics", {})
            memory_growth = exec_metrics.get("memory_growth_mb", 0)
            peak_memory = exec_metrics.get("peak_memory_mb", 0)

            if memory_growth != 0:
                memory_growths.append(memory_growth)
            if peak_memory > 0:
                peak_memories.append(peak_memory)

        analysis = {}

        if memory_growths:
            analysis["memory_growth"] = {
                "average_growth_mb": statistics.mean(memory_growths),
                "median_growth_mb": statistics.median(memory_growths),
                "max_growth_mb": max(memory_growths),
                "total_growth_mb": sum(memory_growths),
                "leak_indicators": sum(
                    1 for g in memory_growths if g > 10
                ),  # > 10MB growth
            }

        if peak_memories:
            analysis["peak_memory"] = {
                "average_peak_mb": statistics.mean(peak_memories),
                "median_peak_mb": statistics.median(peak_memories),
                "max_peak_mb": max(peak_memories),
                "memory_efficiency": self._calculate_memory_efficiency(peak_memories),
            }

        return analysis

    def _analyze_throughput(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze throughput patterns."""
        throughputs = []

        for entry in data:
            # Calculate throughput from execution metrics
            exec_metrics = entry.get("execution_metrics", {})
            exec_time = exec_metrics.get("execution_time", 0)
            dataset_size = exec_metrics.get("dataset_size", 0)

            if exec_time > 0 and dataset_size > 0:
                throughput = dataset_size / exec_time
                throughputs.append(throughput)

        if not throughputs:
            return {"message": "No throughput data available"}

        return {
            "average_throughput": statistics.mean(throughputs),
            "median_throughput": statistics.median(throughputs),
            "min_throughput": min(throughputs),
            "max_throughput": max(throughputs),
            "throughput_stability": 1.0
            - (statistics.stdev(throughputs) / statistics.mean(throughputs)),
            "low_performance_operations": sum(
                1 for t in throughputs if t < 10
            ),  # < 10 samples/sec
        }

    def _analyze_errors(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze error patterns."""
        total_operations = len(data)
        failed_operations = sum(
            1 for entry in data if not entry.get("analysis", {}).get("success", True)
        )

        error_types = defaultdict(int)
        error_trends = []

        for entry in data:
            analysis = entry.get("analysis", {})
            if not analysis.get("success", True):
                error_msg = analysis.get("error_message", "unknown_error")
                error_types[error_msg] += 1

                timestamp = datetime.fromisoformat(entry["timestamp"])
                error_trends.append((timestamp, 1))

        return {
            "total_operations": total_operations,
            "failed_operations": failed_operations,
            "error_rate": (failed_operations / total_operations) * 100
            if total_operations > 0
            else 0,
            "error_types": dict(error_types),
            "error_trend": error_trends,
            "reliability_score": (
                (total_operations - failed_operations) / total_operations
            )
            * 100
            if total_operations > 0
            else 0,
        }

    def _analyze_trends(
        self, data: list[dict[str, Any]], period_hours: int
    ) -> list[PerformanceTrend]:
        """Analyze performance trends."""
        trends = []

        # Extract time series data for different metrics
        metrics_data = {
            "execution_time": [],
            "memory_usage": [],
            "throughput": [],
            "error_rate": [],
        }

        for entry in data:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            exec_metrics = entry.get("execution_metrics", {})

            # Execution time
            exec_time = exec_metrics.get("execution_time", 0)
            if exec_time > 0:
                metrics_data["execution_time"].append((timestamp, exec_time))

            # Memory usage
            memory_growth = exec_metrics.get("memory_growth_mb", 0)
            if memory_growth != 0:
                metrics_data["memory_usage"].append((timestamp, abs(memory_growth)))

            # Throughput (calculated)
            dataset_size = exec_metrics.get("dataset_size", 0)
            if exec_time > 0 and dataset_size > 0:
                throughput = dataset_size / exec_time
                metrics_data["throughput"].append((timestamp, throughput))

            # Error rate (binary: 1 if error, 0 if success)
            success = entry.get("analysis", {}).get("success", True)
            error_val = 0 if success else 1
            metrics_data["error_rate"].append((timestamp, error_val))

        # Analyze each metric
        for metric_name, data_points in metrics_data.items():
            if data_points:
                trend = self.trend_analyzer.analyze_metric_trend(
                    data_points, metric_name, period_hours // 24
                )
                trends.append(trend)

        return trends

    def _compare_algorithms(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare performance across different algorithms."""
        algorithm_metrics = defaultdict(lambda: defaultdict(list))

        for entry in data:
            algorithm = entry.get("execution_metrics", {}).get(
                "operation_name", "unknown"
            )
            exec_metrics = entry.get("execution_metrics", {})

            # Collect metrics by algorithm
            exec_time = exec_metrics.get("execution_time", 0)
            memory_growth = exec_metrics.get("memory_growth_mb", 0)

            if exec_time > 0:
                algorithm_metrics[algorithm]["execution_time"].append(exec_time)
            if memory_growth != 0:
                algorithm_metrics[algorithm]["memory_usage"].append(abs(memory_growth))

        # Compare algorithms
        comparison = {}

        for algorithm, metrics in algorithm_metrics.items():
            if metrics["execution_time"]:
                comparison[algorithm] = {
                    "avg_execution_time": statistics.mean(metrics["execution_time"]),
                    "operations_count": len(metrics["execution_time"]),
                }

                if metrics["memory_usage"]:
                    comparison[algorithm]["avg_memory_usage"] = statistics.mean(
                        metrics["memory_usage"]
                    )

        # Find best performers
        if comparison:
            fastest = min(comparison.items(), key=lambda x: x[1]["avg_execution_time"])
            most_efficient = min(
                [(k, v) for k, v in comparison.items() if "avg_memory_usage" in v],
                key=lambda x: x[1]["avg_memory_usage"],
                default=None,
            )

            return {
                "algorithm_performance": comparison,
                "fastest_algorithm": fastest[0],
                "most_memory_efficient": most_efficient[0] if most_efficient else None,
                "performance_variance": self._calculate_algorithm_variance(comparison),
            }

        return {"message": "Insufficient data for algorithm comparison"}

    def _compare_with_historical(
        self, current_data: list[dict[str, Any]], all_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compare current period with historical performance."""
        current_period_hours = 24  # Compare with last 24 hours
        historical_period_hours = 24 * 7  # Compare with previous week

        current_cutoff = datetime.utcnow() - timedelta(hours=current_period_hours)
        historical_cutoff = datetime.utcnow() - timedelta(hours=historical_period_hours)
        historical_end = datetime.utcnow() - timedelta(hours=current_period_hours)

        # Get historical data (previous period)
        historical_data = [
            entry
            for entry in all_data
            if historical_cutoff
            <= datetime.fromisoformat(entry["timestamp"])
            < historical_end
        ]

        if not historical_data:
            return {"message": "No historical data available for comparison"}

        # Compare key metrics
        current_avg_time = self._get_average_execution_time(current_data)
        historical_avg_time = self._get_average_execution_time(historical_data)

        current_error_rate = self._get_error_rate(current_data)
        historical_error_rate = self._get_error_rate(historical_data)

        return {
            "current_period_hours": current_period_hours,
            "historical_period_hours": historical_period_hours,
            "execution_time_comparison": {
                "current_avg": current_avg_time,
                "historical_avg": historical_avg_time,
                "change_percentage": (
                    (current_avg_time - historical_avg_time) / historical_avg_time * 100
                )
                if historical_avg_time > 0
                else 0,
                "improvement": current_avg_time < historical_avg_time,
            },
            "error_rate_comparison": {
                "current_rate": current_error_rate,
                "historical_rate": historical_error_rate,
                "change_percentage": current_error_rate - historical_error_rate,
                "improvement": current_error_rate < historical_error_rate,
            },
        }

    def _calculate_overall_health_score(self, report: PerformanceReport) -> float:
        """Calculate overall system health score."""
        score = 100.0

        # Execution time analysis
        exec_analysis = report.execution_time_analysis
        if "percentiles" in exec_analysis:
            # Penalize high percentiles
            p95 = exec_analysis["percentiles"]["95th"]
            if p95 > 60:  # > 1 minute
                score -= 20
            elif p95 > 30:  # > 30 seconds
                score -= 10

        # Memory analysis
        memory_analysis = report.memory_usage_analysis
        if "memory_growth" in memory_analysis:
            leak_indicators = memory_analysis["memory_growth"].get("leak_indicators", 0)
            score -= leak_indicators * 5  # 5 points per leak indicator

        # Error analysis
        error_analysis = report.error_analysis
        error_rate = error_analysis.get("error_rate", 0)
        if error_rate > 10:  # > 10% error rate
            score -= 30
        elif error_rate > 5:  # > 5% error rate
            score -= 15
        elif error_rate > 1:  # > 1% error rate
            score -= 5

        # Trend analysis
        concerning_trends = sum(
            1
            for trend in report.performance_trends
            if trend.trend_assessment in ["concerning", "critical"]
        )
        score -= concerning_trends * 15

        return max(0.0, min(100.0, score))

    def _assign_performance_grade(self, score: float) -> str:
        """Assign letter grade based on health score."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 55:
            return "C-"
        elif score >= 50:
            return "D"
        else:
            return "F"

    def _generate_key_insights(self, report: PerformanceReport) -> list[str]:
        """Generate key insights from the analysis."""
        insights = []

        # Overall performance insight
        insights.append(
            f"Overall system health score: {report.overall_health_score:.1f}/100 (Grade: {report.performance_grade})"
        )

        # Execution time insights
        exec_analysis = report.execution_time_analysis
        if "average_time" in exec_analysis:
            avg_time = exec_analysis["average_time"]
            insights.append(f"Average execution time: {avg_time:.2f} seconds")

            if avg_time > 60:
                insights.append("⚠️ High average execution time detected")
            elif avg_time < 1:
                insights.append("✅ Excellent execution time performance")

        # Memory insights
        memory_analysis = report.memory_usage_analysis
        if "memory_growth" in memory_analysis:
            leak_indicators = memory_analysis["memory_growth"].get("leak_indicators", 0)
            if leak_indicators > 0:
                insights.append(
                    f"⚠️ {leak_indicators} potential memory leak indicators detected"
                )

        # Error rate insights
        error_analysis = report.error_analysis
        error_rate = error_analysis.get("error_rate", 0)
        if error_rate > 5:
            insights.append(f"🚨 High error rate: {error_rate:.1f}%")
        elif error_rate < 1:
            insights.append(f"✅ Low error rate: {error_rate:.1f}%")

        # Trend insights
        negative_trends = [
            trend
            for trend in report.performance_trends
            if trend.trend_direction in ["increasing"]
            and trend.metric_name in ["execution_time", "memory_usage", "error_rate"]
        ]
        if negative_trends:
            insights.append(
                f"📈 {len(negative_trends)} concerning performance trends detected"
            )

        return insights

    def _identify_critical_issues(self, report: PerformanceReport) -> list[str]:
        """Identify critical issues requiring immediate attention."""
        issues = []

        # High error rate
        error_rate = report.error_analysis.get("error_rate", 0)
        if error_rate > 10:
            issues.append(
                f"CRITICAL: Error rate at {error_rate:.1f}% - immediate investigation required"
            )

        # Memory leaks
        memory_analysis = report.memory_usage_analysis
        if "memory_growth" in memory_analysis:
            leak_indicators = memory_analysis["memory_growth"].get("leak_indicators", 0)
            if leak_indicators > 5:
                issues.append(
                    f"CRITICAL: {leak_indicators} memory leak indicators - memory management review needed"
                )

        # Performance degradation
        exec_analysis = report.execution_time_analysis
        if "percentiles" in exec_analysis:
            p99 = exec_analysis["percentiles"]["99th"]
            if p99 > 300:  # > 5 minutes
                issues.append(
                    f"CRITICAL: 99th percentile execution time at {p99:.1f}s - severe performance degradation"
                )

        # Critical trends
        critical_trends = [
            trend
            for trend in report.performance_trends
            if trend.trend_assessment == "critical"
        ]
        for trend in critical_trends:
            issues.append(
                f"CRITICAL: {trend.metric_name} trend is critical - {trend.recommendations[0] if trend.recommendations else 'immediate attention required'}"
            )

        return issues

    def _generate_immediate_actions(self, report: PerformanceReport) -> list[str]:
        """Generate immediate action recommendations."""
        actions = []

        # Based on critical issues
        if report.critical_issues:
            actions.append("Address all critical issues immediately")

        # High error rate actions
        error_rate = report.error_analysis.get("error_rate", 0)
        if error_rate > 5:
            actions.append("Investigate and fix high error rate")

        # Performance optimization
        exec_analysis = report.execution_time_analysis
        if "average_time" in exec_analysis and exec_analysis["average_time"] > 30:
            actions.append("Optimize slow-performing operations")

        # Memory management
        memory_analysis = report.memory_usage_analysis
        if "memory_growth" in memory_analysis:
            total_growth = memory_analysis["memory_growth"].get("total_growth_mb", 0)
            if total_growth > 100:  # > 100MB total growth
                actions.append("Implement memory optimization measures")

        return actions

    def _identify_optimization_opportunities(
        self, report: PerformanceReport
    ) -> list[str]:
        """Identify optimization opportunities."""
        opportunities = []

        # Algorithm optimization
        if "algorithm_performance" in report.algorithm_comparisons:
            perf_data = report.algorithm_comparisons["algorithm_performance"]
            if len(perf_data) > 1:
                # Find performance gaps
                times = [data["avg_execution_time"] for data in perf_data.values()]
                if max(times) / min(times) > 2:  # 2x performance difference
                    opportunities.append(
                        "Significant algorithm performance differences detected - consider standardizing on fastest algorithm"
                    )

        # Caching opportunities
        exec_analysis = report.execution_time_analysis
        if (
            "total_operations" in exec_analysis
            and exec_analysis["total_operations"] > 100
        ):
            opportunities.append(
                "High operation volume detected - consider implementing caching"
            )

        # Parallel processing opportunities
        if "average_time" in exec_analysis and exec_analysis["average_time"] > 10:
            opportunities.append(
                "Long execution times detected - consider parallel processing"
            )

        # Memory optimization
        memory_analysis = report.memory_usage_analysis
        if "peak_memory" in memory_analysis:
            avg_peak = memory_analysis["peak_memory"].get("average_peak_mb", 0)
            if avg_peak > 1000:  # > 1GB average peak
                opportunities.append(
                    "High memory usage detected - consider memory optimization strategies"
                )

        return opportunities

    def _generate_long_term_recommendations(
        self, report: PerformanceReport
    ) -> list[str]:
        """Generate long-term recommendations."""
        recommendations = []

        # Infrastructure scaling
        throughput_analysis = report.throughput_analysis
        if "low_performance_operations" in throughput_analysis:
            low_perf_ops = throughput_analysis["low_performance_operations"]
            total_ops = throughput_analysis.get("total_operations", 1)
            if low_perf_ops / total_ops > 0.2:  # > 20% low performance
                recommendations.append(
                    "Consider infrastructure scaling to improve throughput"
                )

        # Monitoring enhancements
        if len(report.performance_trends) < 3:
            recommendations.append(
                "Implement more comprehensive performance monitoring"
            )

        # Automated optimization
        concerning_trends = sum(
            1
            for trend in report.performance_trends
            if trend.trend_assessment in ["concerning", "critical"]
        )
        if concerning_trends > 2:
            recommendations.append(
                "Consider implementing automated performance optimization"
            )

        # Performance testing
        error_rate = report.error_analysis.get("error_rate", 0)
        if error_rate > 1:
            recommendations.append(
                "Implement comprehensive performance testing and load testing"
            )

        return recommendations

    # Helper methods
    def _parse_time_period(self, time_period: str) -> int:
        """Parse time period string to hours."""
        if time_period.endswith("h"):
            return int(time_period[:-1])
        elif time_period.endswith("d"):
            return int(time_period[:-1]) * 24
        elif time_period.endswith("w"):
            return int(time_period[:-1]) * 24 * 7
        else:
            return 24  # Default to 24 hours

    def _categorize_performance(self, execution_times: list[float]) -> dict[str, int]:
        """Categorize operations by performance."""
        fast = sum(1 for t in execution_times if t < 1)
        medium = sum(1 for t in execution_times if 1 <= t < 10)
        slow = sum(1 for t in execution_times if 10 <= t < 60)
        very_slow = sum(1 for t in execution_times if t >= 60)

        return {
            "fast_operations": fast,
            "medium_operations": medium,
            "slow_operations": slow,
            "very_slow_operations": very_slow,
        }

    def _calculate_memory_efficiency(self, peak_memories: list[float]) -> str:
        """Calculate memory efficiency rating."""
        avg_peak = statistics.mean(peak_memories)

        if avg_peak < 100:  # < 100MB
            return "excellent"
        elif avg_peak < 500:  # < 500MB
            return "good"
        elif avg_peak < 1000:  # < 1GB
            return "fair"
        else:
            return "poor"

    def _calculate_algorithm_variance(
        self, comparison: dict[str, dict[str, float]]
    ) -> float:
        """Calculate variance in algorithm performance."""
        if len(comparison) < 2:
            return 0.0

        times = [data["avg_execution_time"] for data in comparison.values()]
        return statistics.stdev(times) if len(times) > 1 else 0.0

    def _get_average_execution_time(self, data: list[dict[str, Any]]) -> float:
        """Get average execution time from data."""
        times = [
            entry.get("execution_metrics", {}).get("execution_time", 0)
            for entry in data
        ]
        times = [t for t in times if t > 0]
        return statistics.mean(times) if times else 0.0

    def _get_error_rate(self, data: list[dict[str, Any]]) -> float:
        """Get error rate from data."""
        total = len(data)
        if total == 0:
            return 0.0

        errors = sum(
            1 for entry in data if not entry.get("analysis", {}).get("success", True)
        )

        return (errors / total) * 100

    def _load_report_templates(self) -> dict[str, str]:
        """Load report templates."""
        # This would load actual templates from files
        return {
            "html_template": """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f8ff; padding: 20px; margin-bottom: 20px; }
                    .metric { margin: 10px 0; }
                    .critical { color: red; font-weight: bold; }
                    .good { color: green; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #4CAF50; color: white; }
                </style>
            </head>
            <body>
                {content}
            </body>
            </html>
            """
        }

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for the report."""
        import psutil

        return {
            "cpu_cores": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "disk_space_gb": psutil.disk_usage("/").total / 1024**3,
            "python_version": f"{psutil.version_info}",
        }

    def _export_html_report(
        self, report: PerformanceReport, include_charts: bool
    ) -> Path:
        """Export report as HTML."""
        # Generate HTML content
        html_content = f"""
        <div class="header">
            <h1>Performance Report</h1>
            <h2>Report ID: {report.report_id}</h2>
            <p><strong>Generated:</strong> {report.generated_at.isoformat()}</p>
            <p><strong>Time Period:</strong> {report.time_period}</p>
            <p><strong>Overall Health Score:</strong> {report.overall_health_score:.1f}/100 (Grade: {report.performance_grade})</p>
        </div>

        <h3>Key Insights</h3>
        <ul>
        {''.join(f'<li>{insight}</li>' for insight in report.key_insights)}
        </ul>

        <h3>Critical Issues</h3>
        <ul>
        {''.join(f'<li class="critical">{issue}</li>' for issue in report.critical_issues)}
        </ul>

        <h3>Execution Time Analysis</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Operations</td><td>{report.execution_time_analysis.get('total_operations', 'N/A')}</td></tr>
            <tr><td>Average Time</td><td>{report.execution_time_analysis.get('average_time', 0):.3f}s</td></tr>
            <tr><td>95th Percentile</td><td>{report.execution_time_analysis.get('percentiles', {}).get('95th', 0):.3f}s</td></tr>
        </table>

        <h3>Immediate Actions Required</h3>
        <ul>
        {''.join(f'<li>{action}</li>' for action in report.immediate_actions)}
        </ul>

        <h3>Optimization Opportunities</h3>
        <ul>
        {''.join(f'<li>{opp}</li>' for opp in report.optimization_opportunities)}
        </ul>
        """

        full_html = self.report_templates["html_template"].format(content=html_content)

        # Save to file
        output_path = self.storage_path / f"{report.report_id}.html"
        with open(output_path, "w") as f:
            f.write(full_html)

        logger.info(f"HTML report exported to {output_path}")
        return output_path

    def _export_json_report(self, report: PerformanceReport) -> Path:
        """Export report as JSON."""
        # Convert dataclass to dict
        report_dict = {
            "report_id": report.report_id,
            "generated_at": report.generated_at.isoformat(),
            "report_type": report.report_type,
            "time_period": report.time_period,
            "overall_health_score": report.overall_health_score,
            "performance_grade": report.performance_grade,
            "key_insights": report.key_insights,
            "critical_issues": report.critical_issues,
            "execution_time_analysis": report.execution_time_analysis,
            "memory_usage_analysis": report.memory_usage_analysis,
            "throughput_analysis": report.throughput_analysis,
            "error_analysis": report.error_analysis,
            "algorithm_comparisons": report.algorithm_comparisons,
            "historical_comparisons": report.historical_comparisons,
            "immediate_actions": report.immediate_actions,
            "optimization_opportunities": report.optimization_opportunities,
            "long_term_recommendations": report.long_term_recommendations,
            "system_info": report.system_info,
        }

        # Save to file
        output_path = self.storage_path / f"{report.report_id}.json"
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"JSON report exported to {output_path}")
        return output_path

    def _export_pdf_report(
        self, report: PerformanceReport, include_charts: bool
    ) -> Path:
        """Export report as PDF (placeholder implementation)."""
        # This would require additional dependencies like reportlab
        # For now, export as HTML and suggest PDF conversion
        html_path = self._export_html_report(report, include_charts)

        logger.info(f"PDF export not implemented. HTML report available at {html_path}")
        logger.info("Consider using wkhtmltopdf or similar tool to convert HTML to PDF")

        return html_path


# Factory function
def create_performance_reporter(storage_path: Path = None) -> PerformanceReporter:
    """Create performance reporter with default configuration."""
    if storage_path is None:
        storage_path = Path("performance_reports")

    return PerformanceReporter(storage_path)
