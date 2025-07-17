"""High-level performance monitoring service.

This service provides application-level performance monitoring capabilities,
integrating with the infrastructure performance monitor to provide comprehensive
monitoring for anomaly processing workflows.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ...domain.entities import Dataset, DetectionResult, Detector
from ...infrastructure.config.feature_flags import require_feature
from ...infrastructure.monitoring.performance_monitor import (
    PerformanceMetrics,
    PerformanceMonitor,
    PerformanceTracker,
)


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
        self, detector: Detector, data_collection: DataCollection, operation_func: callable, **kwargs
    ) -> tuple[DetectionResult, PerformanceMetrics]:
        """Monitor a complete processing operation.

        Args:
            detector: The detector being used
            data_collection: The data_collection being processed
            operation_func: Function that performs processing
            **kwargs: Additional arguments for the operation

        Returns:
            Tuple of (processing_result, performance_measurements)
        """
        operation_name = f"processing_{detector.algorithm_name}"

        with PerformanceTracker(
            self.monitor,
            operation_name=operation_name,
            algorithm_name=detector.algorithm_name,
            data_collection_size=len(data_collection.data),
            metadata={
                "detector_name": detector.name,
                "data_collection_name": data_collection.name,
                "parameters": detector.parameters,
            },
        ) as tracker:
            # Execute the processing operation
            result = operation_func(detector, data_collection, **kwargs)

            # Extract quality measurements if available
            if hasattr(result, "metadata") and result.metadata:
                quality_measurements = {}
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    if metric in result.metadata:
                        quality_measurements[metric] = result.metadata[metric]

                if quality_measurements:
                    tracker.set_quality_measurements(quality_measurements)

            # Set samples processed
            tracker.set_samples_processed(len(data_collection.data))

            # Get the final measurements (will be available after context exit)
            # Check if measurements are available
            if self.monitor.measurements_history:
                return result, self.monitor.measurements_history[-1]
            else:
                # Create a dummy measurements object if none available
                from ...infrastructure.monitoring.performance_monitor import (
                    PerformanceMetrics,
                )

                dummy_measurements = PerformanceMetrics(
                    operation_name=operation_name,
                    algorithm_name=detector.algorithm_name,
                    data_collection_size=len(data_collection.data),
                )
                return result, dummy_measurements

    @require_feature("performance_monitoring")
    def monitor_training_operation(
        self, detector: Detector, data_collection: DataCollection, training_func: callable, **kwargs
    ) -> tuple[Detector, PerformanceMetrics]:
        """Monitor a detector training operation.

        Args:
            detector: The detector being trained
            data_collection: The training data_collection
            training_func: Function that performs training
            **kwargs: Additional arguments for training

        Returns:
            Tuple of (trained_detector, performance_measurements)
        """
        operation_name = f"training_{detector.algorithm_name}"

        with PerformanceTracker(
            self.monitor,
            operation_name=operation_name,
            algorithm_name=detector.algorithm_name,
            data_collection_size=len(data_collection.data),
            metadata={
                "detector_name": detector.name,
                "data_collection_name": data_collection.name,
                "parameters": detector.parameters,
            },
        ) as tracker:
            # Execute the training operation
            trained_detector = training_func(detector, data_collection, **kwargs)

            # Set samples processed
            tracker.set_samples_processed(len(data_collection.data))

            # Check if measurements are available
            if self.monitor.measurements_history:
                return trained_detector, self.monitor.measurements_history[-1]
            else:
                # Create a dummy measurements object if none available
                from ...infrastructure.monitoring.performance_monitor import (
                    PerformanceMetrics,
                )

                dummy_measurements = PerformanceMetrics(
                    operation_name=operation_name,
                    algorithm_name=detector.algorithm_name,
                    data_collection_size=len(data_collection.data),
                )
                return trained_detector, dummy_measurements

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
        # Get all measurements within time window
        measurements_list = list(self.monitor.measurements_history)
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            measurements_list = [m for m in measurements_list if m.timestamp >= cutoff_time]

        # Group by algorithm
        algorithm_measurements = {}
        for metric in measurements_list:
            if metric.algorithm_name:
                if metric.algorithm_name not in algorithm_measurements:
                    algorithm_measurements[metric.algorithm_name] = []
                algorithm_measurements[metric.algorithm_name].append(metric)

        # Calculate comparison statistics
        comparison = {
            "comparison_timestamp": datetime.utcnow().isoformat(),
            "time_window_hours": (
                time_window.total_seconds() / 3600 if time_window else None
            ),
            "algorithms": {},
        }

        for algorithm, measurements in algorithm_measurements.items():
            if len(measurements) < min_operations:
                continue

            execution_times = [m.execution_time for m in measurements]
            memory_usages = [m.memory_usage for m in measurements]
            throughputs = [
                m.samples_per_second for m in measurements if m.samples_per_second > 0
            ]

            algorithm_stats = {
                "operation_count": len(measurements),
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
                "reliability_score": self._calculate_reliability_score(measurements),
            }

            # Add quality measurements if available
            quality_measurements = [m for m in measurements if m.accuracy is not None]
            if quality_measurements:
                algorithm_stats["avg_accuracy"] = sum(
                    m.accuracy for m in quality_measurements
                ) / len(quality_measurements)
                f1_measurements = [m for m in quality_measurements if m.f1_score is not None]
                if f1_measurements:
                    algorithm_stats["avg_f1_score"] = sum(
                        m.f1_score for m in f1_measurements
                    ) / len(f1_measurements)

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
        # Get measurements for analysis
        cutoff_time = datetime.utcnow() - time_window
        measurements_list = [
            m for m in self.monitor.measurements_history if m.timestamp >= cutoff_time
        ]

        if operation_name:
            measurements_list = [
                m for m in measurements_list if m.operation_name == operation_name
            ]

        if not measurements_list:
            return {"message": "No measurements available for trend analysis"}

        # Create time buckets
        start_time = min(m.timestamp for m in measurements_list)
        end_time = max(m.timestamp for m in measurements_list)

        buckets = []
        current_time = start_time
        while current_time < end_time:
            bucket_end = min(current_time + bucket_size, end_time)
            bucket_measurements = [
                m for m in measurements_list if current_time <= m.timestamp < bucket_end
            ]

            if bucket_measurements:
                bucket_stats = {
                    "timestamp": current_time.isoformat(),
                    "operation_count": len(bucket_measurements),
                    "avg_execution_time": sum(m.execution_time for m in bucket_measurements)
                    / len(bucket_measurements),
                    "avg_memory_usage": sum(m.memory_usage for m in bucket_measurements)
                    / len(bucket_measurements),
                    "avg_cpu_usage": sum(m.cpu_usage for m in bucket_measurements)
                    / len(bucket_measurements),
                    "total_samples_processed": sum(
                        m.samples_processed for m in bucket_measurements
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
            "total_operations": len(measurements_list),
            "time_buckets": buckets,
            "trends": trends,
        }

    @require_feature("performance_monitoring")
    def set_performance_baseline(
        self, operation_name: str, baseline_measurements: dict[str, float]
    ) -> None:
        """Set performance baseline for an operation.

        Args:
            operation_name: Name of the operation
            baseline_measurements: Expected baseline performance measurements
        """
        self.performance_baselines[operation_name] = {
            "measurements": baseline_measurements,
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

        # Get recent measurements
        cutoff_time = datetime.utcnow() - recent_window
        recent_measurements = [
            m
            for m in self.monitor.measurements_history
            if m.timestamp >= cutoff_time and m.operation_name == operation_name
        ]

        if not recent_measurements:
            return {"error": "No recent measurements available for comparison"}

        # Calculate current averages
        current_avg = {
            "execution_time": sum(m.execution_time for m in recent_measurements)
            / len(recent_measurements),
            "memory_usage": sum(m.memory_usage for m in recent_measurements)
            / len(recent_measurements),
            "cpu_usage": sum(m.cpu_usage for m in recent_measurements) / len(recent_measurements),
        }

        # Check for regressions
        regressions = {}
        for metric, current_value in current_avg.items():
            if metric in baseline["measurements"]:
                baseline_value = baseline["measurements"][metric]
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
            "recent_operations": len(recent_measurements),
            "regressions_detected": len(regressions),
            "regressions": regressions,
            "current_performance": current_avg,
            "baseline_performance": baseline["measurements"],
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
            Dashboard data including current measurements, alerts, and statistics
        """
        return {
            "current_measurements": self.monitor.get_real_time_measurements(),
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
                name: data["measurements"]
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
        """Calculate reliability score based on consistency of measurements."""
        if len(measurements) < 2:
            return 1.0

        execution_times = [m.execution_time for m in measurements]
        memory_usages = [m.memory_usage for m in measurements]

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
        ) / (sum(memory_usages) / len(memory_usages) + 1e-8)  # Avoid division by zero

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
