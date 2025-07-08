"""High-level performance monitoring service.

This service provides application-level performance monitoring capabilities,
integrating with the infrastructure performance monitor to provide comprehensive
monitoring for anomaly detection workflows.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional

from ...domain.entities import Dataset, DetectionResult, Detector
from ...domain.entities.model_performance import ModelPerformanceMetrics, ModelPerformanceBaseline
from ...infrastructure.config.feature_flags import require_feature
from ...infrastructure.monitoring.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceTracker,
)
from ...infrastructure.repositories import ModelPerformanceRepository, PerformanceBaselineRepository


class PerformanceMonitoringService:
    """High-level service for performance monitoring."""

    def __init__(
        self,
        performance_monitor: PerformanceMonitor | None = None,
        auto_start_monitoring: bool = True,
    ):
        """Initialize performance monitoring service.

        Args:
            performance_monitor: Instance of performance monitor
            auto_start_monitoring: Whether to automatically start monitoring
        """
        self.monitor = performance_monitor or PerformanceMonitor()
        self.auto_start_monitoring = auto_start_monitoring

        # Service-specific configuration
        self.alert_handlers = []
        self.performance_baselines = {}
        self.monitoring_enabled = True

        # Initialize monitoring if auto-start is enabled
        if self.auto_start_monitoring:
            self.start_monitoring()

    @require_feature("performance_monitoring")
    def start_monitoring(self) -> None:
        """Start the performance monitoring system."""
        self.monitor.start_monitoring()
        self.monitoring_enabled = True

        # Add default alert handler
        self.monitor.add_alert_callback(self._default_alert_handler)

    def stop_monitoring(self) -> None:
        """Stop the performance monitoring system."""
        self.monitor.stop_monitoring()
        self.monitoring_enabled = False

    @require_feature("performance_monitoring")
    def monitor_detection_operation(
        self, detector: Detector, dataset: Dataset, operation_func: callable, **kwargs
    ) -> tuple[DetectionResult, PerformanceMetrics]:
        """Monitor a complete detection operation.

        Args:
            detector: The detector being used
            dataset: The dataset being processed
            operation_func: Function that performs detection
            **kwargs: Additional arguments for the operation

        Returns:
            Tuple of (detection_result, performance_metrics)
        """
        operation_name = f"detection_{detector.algorithm_name}"

        with PerformanceTracker(
            self.monitor,
            operation_name=operation_name,
            algorithm_name=detector.algorithm_name,
            dataset_size=len(dataset.data),
            metadata={
                "detector_name": detector.name,
                "dataset_name": dataset.name,
                "parameters": detector.parameters,
            },
        ) as tracker:
            # Execute the detection operation
            result = operation_func(detector, dataset, **kwargs)

            # Extract quality metrics if available
            if hasattr(result, "metadata") and result.metadata:
                quality_metrics = {}
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    if metric in result.metadata:
                        quality_metrics[metric] = result.metadata[metric]

                if quality_metrics:
                    tracker.set_quality_metrics(quality_metrics)

            # Set samples processed
            tracker.set_samples_processed(len(dataset.data))

            # Get the final metrics (will be available after context exit)
            # Check if metrics are available
            if self.monitor.metrics_history:
                return result, self.monitor.metrics_history[-1]
            else:
                # Create a dummy metrics object if none available
                from ...infrastructure.monitoring.performance_monitor import (
                    PerformanceMetrics,
                )

                dummy_metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    algorithm_name=detector.algorithm_name,
                    dataset_size=len(dataset.data),
                )
                return result, dummy_metrics

    @require_feature("performance_monitoring")
    def monitor_training_operation(
        self, detector: Detector, dataset: Dataset, training_func: callable, **kwargs
    ) -> tuple[Detector, PerformanceMetrics]:
        """Monitor a detector training operation.

        Args:
            detector: The detector being trained
            dataset: The training dataset
            training_func: Function that performs training
            **kwargs: Additional arguments for training

        Returns:
            Tuple of (trained_detector, performance_metrics)
        """
        operation_name = f"training_{detector.algorithm_name}"

        with PerformanceTracker(
            self.monitor,
            operation_name=operation_name,
            algorithm_name=detector.algorithm_name,
            dataset_size=len(dataset.data),
            metadata={
                "detector_name": detector.name,
                "dataset_name": dataset.name,
                "parameters": detector.parameters,
            },
        ) as tracker:
            # Execute the training operation
            trained_detector = training_func(detector, dataset, **kwargs)

            # Set samples processed
            tracker.set_samples_processed(len(dataset.data))

            # Check if metrics are available
            if self.monitor.metrics_history:
                return trained_detector, self.monitor.metrics_history[-1]
            else:
                # Create a dummy metrics object if none available
                from ...infrastructure.monitoring.performance_monitor import (
                    PerformanceMetrics,
                )

                dummy_metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    algorithm_name=detector.algorithm_name,
                    dataset_size=len(dataset.data),
                )
                return trained_detector, dummy_metrics

    @require_feature("performance_monitoring")
    def get_algorithm_performance_comparison(
        self, time_window: timedelta | None = None, min_operations: int = 3
    ) -> dict[str, Any]:
        """Compare performance across different algorithms.

        Args:
            time_window: Time window for analysis
            min_operations: Minimum operations required for reliable statistics

        Returns:
            Performance comparison across algorithms
        """
        # Get all metrics within time window
        metrics_list = list(self.monitor.metrics_history)
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            metrics_list = [m for m in metrics_list if m.timestamp >= cutoff_time]

        # Group by algorithm
        algorithm_metrics = {}
        for metric in metrics_list:
            if metric.algorithm_name:
                if metric.algorithm_name not in algorithm_metrics:
                    algorithm_metrics[metric.algorithm_name] = []
                algorithm_metrics[metric.algorithm_name].append(metric)

        # Calculate comparison statistics
        comparison = {
            "comparison_timestamp": datetime.utcnow().isoformat(),
            "time_window_hours": (
                time_window.total_seconds() / 3600 if time_window else None
            ),
            "algorithms": {},
        }

        for algorithm, metrics in algorithm_metrics.items():
            if len(metrics) < min_operations:
                continue

            execution_times = [m.execution_time for m in metrics]
            memory_usages = [m.memory_usage for m in metrics]
            throughputs = [
                m.samples_per_second for m in metrics if m.samples_per_second > 0
            ]

            algorithm_stats = {
                "operation_count": len(metrics),
                "avg_execution_time": (
                    sum(execution_times) / len(execution_times)
                    if execution_times
                    else 0
                ),
                "avg_memory_usage": (
                    sum(memory_usages) / len(memory_usages) if memory_usages else 0
                ),
                "avg_throughput": (
                    sum(throughputs) / len(throughputs) if throughputs else 0
                ),
                "reliability_score": self._calculate_reliability_score(metrics),
            }

            # Add quality metrics if available
            quality_metrics = [m for m in metrics if m.accuracy is not None]
            if quality_metrics:
                algorithm_stats["avg_accuracy"] = sum(
                    m.accuracy for m in quality_metrics
                ) / len(quality_metrics)
                f1_metrics = [m for m in quality_metrics if m.f1_score is not None]
                if f1_metrics:
                    algorithm_stats["avg_f1_score"] = sum(
                        m.f1_score for m in f1_metrics
                    ) / len(f1_metrics)

            comparison["algorithms"][algorithm] = algorithm_stats

        # Add rankings
        if comparison["algorithms"]:
            comparison["rankings"] = self._generate_algorithm_rankings(
                comparison["algorithms"]
            )

        return comparison

    @require_feature("performance_monitoring")
    def get_performance_trends(
        self,
        operation_name: str | None = None,
        time_window: timedelta = timedelta(hours=24),
        bucket_size: timedelta = timedelta(hours=1),
    ) -> dict[str, Any]:
        """Analyze performance trends over time.

        Args:
            operation_name: Specific operation to analyze (None for all)
            time_window: Time window for trend analysis
            bucket_size: Size of time buckets for aggregation

        Returns:
            Performance trend analysis
        """
        # Get metrics for analysis
        cutoff_time = datetime.utcnow() - time_window
        metrics_list = [
            m for m in self.monitor.metrics_history if m.timestamp >= cutoff_time
        ]

        if operation_name:
            metrics_list = [
                m for m in metrics_list if m.operation_name == operation_name
            ]

        if not metrics_list:
            return {"message": "No metrics available for trend analysis"}

        # Create time buckets
        start_time = min(m.timestamp for m in metrics_list)
        end_time = max(m.timestamp for m in metrics_list)

        buckets = []
        current_time = start_time
        while current_time < end_time:
            bucket_end = min(current_time + bucket_size, end_time)
            bucket_metrics = [
                m for m in metrics_list if current_time <= m.timestamp < bucket_end
            ]

            if bucket_metrics:
                bucket_stats = {
                    "timestamp": current_time.isoformat(),
                    "operation_count": len(bucket_metrics),
                    "avg_execution_time": sum(m.execution_time for m in bucket_metrics)
                    / len(bucket_metrics),
                    "avg_memory_usage": sum(m.memory_usage for m in bucket_metrics)
                    / len(bucket_metrics),
                    "avg_cpu_usage": sum(m.cpu_usage for m in bucket_metrics)
                    / len(bucket_metrics),
                    "total_samples_processed": sum(
                        m.samples_processed for m in bucket_metrics
                    ),
                }
                buckets.append(bucket_stats)

            current_time = bucket_end

        # Calculate trends
        trends = self._calculate_trends(buckets)

        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "time_window_hours": time_window.total_seconds() / 3600,
            "bucket_size_minutes": bucket_size.total_seconds() / 60,
            "total_operations": len(metrics_list),
            "time_buckets": buckets,
            "trends": trends,
        }

    @require_feature("performance_monitoring")
    def set_performance_baseline(
        self, operation_name: str, baseline_metrics: dict[str, float]
    ) -> None:
        """Set performance baseline for an operation.

        Args:
            operation_name: Name of the operation
            baseline_metrics: Expected baseline performance metrics
        """
        self.performance_baselines[operation_name] = {
            "metrics": baseline_metrics,
            "set_at": datetime.utcnow(),
        }

    @require_feature("performance_monitoring")
    def check_performance_regression(
        self, operation_name: str, recent_window: timedelta = timedelta(hours=1)
    ) -> dict[str, Any]:
        """Check for performance regression against baseline.

        Args:
            operation_name: Operation to check
            recent_window: Time window for recent performance

        Returns:
            Regression analysis results
        """
        if operation_name not in self.performance_baselines:
            return {"error": f"No baseline set for operation: {operation_name}"}

        baseline = self.performance_baselines[operation_name]

        # Get recent metrics
        cutoff_time = datetime.utcnow() - recent_window
        recent_metrics = [
            m
            for m in self.monitor.metrics_history
            if m.timestamp >= cutoff_time and m.operation_name == operation_name
        ]

        if not recent_metrics:
            return {"error": "No recent metrics available for comparison"}

        # Calculate current averages
        current_avg = {
            "execution_time": sum(m.execution_time for m in recent_metrics)
            / len(recent_metrics),
            "memory_usage": sum(m.memory_usage for m in recent_metrics)
            / len(recent_metrics),
            "cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
        }

        # Check for regressions
        regressions = {}
        for metric, current_value in current_avg.items():
            if metric in baseline["metrics"]:
                baseline_value = baseline["metrics"][metric]
                regression_threshold = baseline_value * 1.2  # 20% degradation threshold

                if current_value > regression_threshold:
                    regressions[metric] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "degradation_percent": (
                            (current_value - baseline_value) / baseline_value
                        )
                        * 100,
                    }

        return {
            "operation_name": operation_name,
            "check_timestamp": datetime.utcnow().isoformat(),
            "baseline_set_at": baseline["set_at"].isoformat(),
            "recent_operations": len(recent_metrics),
            "regressions_detected": len(regressions),
            "regressions": regressions,
            "current_performance": current_avg,
            "baseline_performance": baseline["metrics"],
        }

    def add_alert_handler(self, handler: callable) -> None:
        """Add custom alert handler.

        Args:
            handler: Function to handle performance alerts
        """
        self.alert_handlers.append(handler)

    def get_monitoring_dashboard_data(self) -> dict[str, Any]:
        """Get data for performance monitoring dashboard.

        Returns:
            Dashboard data including current metrics, alerts, and statistics
        """
        return {
            "current_metrics": self.monitor.get_real_time_metrics(),
            "active_alerts": [
                alert.to_dict() for alert in self.monitor.get_active_alerts()
            ],
            "recent_operations": self.monitor.get_operation_statistics(
                time_window=timedelta(hours=1)
            ),
            "system_status": {
                "monitoring_enabled": self.monitoring_enabled,
                "total_operations_monitored": self.monitor.total_operations,
                "failed_operations": self.monitor.failed_operations,
                "alert_count": len(self.monitor.active_alerts),
            },
            "performance_baselines": {
                name: data["metrics"]
                for name, data in self.performance_baselines.items()
            },
        }

    def _default_alert_handler(self, alert) -> None:
        """Default handler for performance alerts."""
        # Use lazy import to avoid circular import
        from ...infrastructure.monitoring.performance_monitor import PerformanceAlert
        
        if isinstance(alert, PerformanceAlert):
            print(f"🚨 Performance Alert: {alert.severity.upper()} - {alert.message}")

            # Call custom alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Alert handler error: {e}")

    def _calculate_reliability_score(self, metrics: list[PerformanceMetrics]) -> float:
        """Calculate reliability score based on consistency of metrics."""
        if len(metrics) < 2:
            return 1.0

        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]

        # Calculate coefficient of variation (lower is more reliable)
        exec_time_cv = (
            (
                sum(
                    (x - sum(execution_times) / len(execution_times)) ** 2
                    for x in execution_times
                )
                / len(execution_times)
            )
            ** 0.5
        ) / (sum(execution_times) / len(execution_times))

        memory_cv = (
            (
                sum(
                    (x - sum(memory_usages) / len(memory_usages)) ** 2
                    for x in memory_usages
                )
                / len(memory_usages)
            )
            ** 0.5
        ) / (
            sum(memory_usages) / len(memory_usages) + 1e-8
        )  # Avoid division by zero

        # Convert to reliability score (0-1, higher is better)
        avg_cv = (exec_time_cv + memory_cv) / 2
        reliability_score = max(0, 1 - avg_cv)

        return reliability_score

    def _generate_algorithm_rankings(
        self, algorithm_stats: dict[str, dict]
    ) -> dict[str, list[str]]:
        """Generate rankings of algorithms by different criteria."""
        algorithms = list(algorithm_stats.keys())

        rankings = {}

        # Rank by execution time (fastest first)
        rankings["fastest_execution"] = sorted(
            algorithms, key=lambda a: algorithm_stats[a]["avg_execution_time"]
        )

        # Rank by memory efficiency (lowest usage first)
        rankings["most_memory_efficient"] = sorted(
            algorithms, key=lambda a: algorithm_stats[a]["avg_memory_usage"]
        )

        # Rank by throughput (highest first)
        rankings["highest_throughput"] = sorted(
            algorithms, key=lambda a: algorithm_stats[a]["avg_throughput"], reverse=True
        )

        # Rank by reliability (most reliable first)
        rankings["most_reliable"] = sorted(
            algorithms,
            key=lambda a: algorithm_stats[a]["reliability_score"],
            reverse=True,
        )

        # Rank by accuracy if available
        accuracy_algorithms = [
            a for a in algorithms if "avg_accuracy" in algorithm_stats[a]
        ]
        if accuracy_algorithms:
            rankings["highest_accuracy"] = sorted(
                accuracy_algorithms,
                key=lambda a: algorithm_stats[a]["avg_accuracy"],
                reverse=True,
            )

        return rankings

    def _calculate_trends(self, buckets: list[dict]) -> dict[str, str]:
        """Calculate trends from time bucket data."""
        if len(buckets) < 2:
            return {"message": "Insufficient data for trend calculation"}

        trends = {}

        # Calculate execution time trend
        exec_times = [bucket["avg_execution_time"] for bucket in buckets]
        exec_trend = "increasing" if exec_times[-1] > exec_times[0] else "decreasing"
        trends["execution_time"] = exec_trend

        # Calculate memory usage trend
        memory_usages = [bucket["avg_memory_usage"] for bucket in buckets]
        memory_trend = (
            "increasing" if memory_usages[-1] > memory_usages[0] else "decreasing"
        )
        trends["memory_usage"] = memory_trend

        # Calculate throughput trend
        throughputs = [bucket["total_samples_processed"] for bucket in buckets]
        throughput_trend = (
            "increasing" if throughputs[-1] > throughputs[0] else "decreasing"
        )
        trends["throughput"] = throughput_trend

        return trends
