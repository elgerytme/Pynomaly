"""Prometheus metrics collection and management for Pynomaly.

This module provides comprehensive Prometheus metrics for monitoring
anomaly detection performance, system health, and business KPIs.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Prometheus client imports with graceful fallback
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Enum,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock implementations for when Prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, amount=1, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def time(self):
            return MockTimer()

        def observe(self, amount, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, value, **kwargs):
            pass

        def inc(self, amount=1, **kwargs):
            pass

        def dec(self, amount=1, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

    class Info:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, data, **kwargs):
            pass

    class Enum:
        def __init__(self, *args, **kwargs):
            pass

        def state(self, state, **kwargs):
            pass

    class MockTimer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None

    def generate_latest(registry=None):
        return b"# Prometheus metrics not available\n"

    def start_http_server(port, addr="", registry=None):
        pass


logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """Definition of a Prometheus metric."""

    name: str
    help_text: str
    metric_type: str  # counter, histogram, gauge, info, enum
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    states: Optional[List[str]] = None  # For enums


class PrometheusMetricsService:
    """Service for managing Prometheus metrics collection."""

    def __init__(
        self,
        enable_default_metrics: bool = True,
        custom_registry: Optional[Any] = None,
        namespace: str = "pynomaly",
        port: Optional[int] = None,
    ):
        """Initialize Prometheus metrics service.

        Args:
            enable_default_metrics: Whether to create default metrics
            custom_registry: Custom Prometheus registry
            namespace: Metric namespace prefix
            port: Port for metrics HTTP server (if None, server not started)
        """
        self.namespace = namespace
        self.registry = custom_registry or (REGISTRY if PROMETHEUS_AVAILABLE else None)
        self.port = port
        self.metrics = {}
        self.server_started = False

        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "Prometheus client not available, using mock implementations"
            )

        if enable_default_metrics:
            self._create_default_metrics()

        if port and PROMETHEUS_AVAILABLE:
            self._start_metrics_server()

    def _create_default_metrics(self):
        """Create default metrics for Pynomaly monitoring."""

        # Application info
        self.metrics["app_info"] = Info(
            f"{self.namespace}_application_info",
            "Application information",
            registry=self.registry,
        )

        # HTTP metrics
        self.metrics["http_requests_total"] = Counter(
            f"{self.namespace}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.metrics["http_request_duration"] = Histogram(
            f"{self.namespace}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Detection metrics
        self.metrics["detections_total"] = Counter(
            f"{self.namespace}_detections_total",
            "Total anomaly detections performed",
            ["algorithm", "dataset_type", "status"],
            registry=self.registry,
        )

        self.metrics["detection_duration"] = Histogram(
            f"{self.namespace}_detection_duration_seconds",
            "Anomaly detection duration in seconds",
            ["algorithm", "dataset_size_category"],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        self.metrics["anomalies_found"] = Counter(
            f"{self.namespace}_anomalies_found_total",
            "Total anomalies detected",
            ["algorithm", "confidence_level"],
            registry=self.registry,
        )

        self.metrics["detection_accuracy"] = Gauge(
            f"{self.namespace}_detection_accuracy_ratio",
            "Current detection accuracy",
            ["algorithm", "dataset_type"],
            registry=self.registry,
        )

        # Training metrics
        self.metrics["training_requests_total"] = Counter(
            f"{self.namespace}_training_requests_total",
            "Total model training requests",
            ["algorithm", "status"],
            registry=self.registry,
        )

        self.metrics["training_duration"] = Histogram(
            f"{self.namespace}_training_duration_seconds",
            "Model training duration in seconds",
            ["algorithm", "dataset_size_category"],
            buckets=[1.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
            registry=self.registry,
        )

        self.metrics["model_size"] = Histogram(
            f"{self.namespace}_model_size_bytes",
            "Trained model size in bytes",
            ["algorithm"],
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600],
            registry=self.registry,
        )

        # Streaming metrics
        self.metrics["streaming_samples_total"] = Counter(
            f"{self.namespace}_streaming_samples_total",
            "Total streaming samples processed",
            ["stream_id", "status"],
            registry=self.registry,
        )

        self.metrics["streaming_throughput"] = Gauge(
            f"{self.namespace}_streaming_throughput_per_second",
            "Current streaming throughput",
            ["stream_id"],
            registry=self.registry,
        )

        self.metrics["streaming_buffer_utilization"] = Gauge(
            f"{self.namespace}_streaming_buffer_utilization_ratio",
            "Streaming buffer utilization",
            ["stream_id"],
            registry=self.registry,
        )

        self.metrics["streaming_backpressure_events"] = Counter(
            f"{self.namespace}_streaming_backpressure_events_total",
            "Total backpressure events",
            ["stream_id", "strategy"],
            registry=self.registry,
        )

        # Ensemble metrics
        self.metrics["ensemble_predictions_total"] = Counter(
            f"{self.namespace}_ensemble_predictions_total",
            "Total ensemble predictions",
            ["voting_strategy", "detector_count"],
            registry=self.registry,
        )

        self.metrics["ensemble_agreement"] = Histogram(
            f"{self.namespace}_ensemble_agreement_ratio",
            "Ensemble detector agreement ratio",
            ["voting_strategy"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # System metrics
        self.metrics["active_models"] = Gauge(
            f"{self.namespace}_active_models",
            "Number of currently active models",
            registry=self.registry,
        )

        self.metrics["active_streams"] = Gauge(
            f"{self.namespace}_active_streams",
            "Number of currently active streams",
            registry=self.registry,
        )

        self.metrics["memory_usage"] = Gauge(
            f"{self.namespace}_memory_usage_bytes",
            "Current memory usage in bytes",
            ["component"],
            registry=self.registry,
        )

        self.metrics["cpu_usage"] = Gauge(
            f"{self.namespace}_cpu_usage_ratio",
            "Current CPU usage ratio",
            ["component"],
            registry=self.registry,
        )

        # Cache metrics
        self.metrics["cache_operations_total"] = Counter(
            f"{self.namespace}_cache_operations_total",
            "Total cache operations",
            ["cache_type", "operation", "status"],
            registry=self.registry,
        )

        self.metrics["cache_hit_ratio"] = Gauge(
            f"{self.namespace}_cache_hit_ratio",
            "Cache hit ratio",
            ["cache_type"],
            registry=self.registry,
        )

        self.metrics["cache_size"] = Gauge(
            f"{self.namespace}_cache_size_items",
            "Current cache size in items",
            ["cache_type"],
            registry=self.registry,
        )

        # Error metrics
        self.metrics["errors_total"] = Counter(
            f"{self.namespace}_errors_total",
            "Total errors by category",
            ["error_type", "component", "severity"],
            registry=self.registry,
        )

        # Quality metrics
        self.metrics["data_quality_score"] = Gauge(
            f"{self.namespace}_data_quality_score",
            "Data quality score",
            ["dataset_id", "metric_type"],
            registry=self.registry,
        )

        self.metrics["prediction_confidence"] = Histogram(
            f"{self.namespace}_prediction_confidence_score",
            "Prediction confidence score distribution",
            ["algorithm"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # Business metrics
        self.metrics["datasets_processed_total"] = Counter(
            f"{self.namespace}_datasets_processed_total",
            "Total datasets processed",
            ["source_type", "format"],
            registry=self.registry,
        )

        self.metrics["api_response_size"] = Histogram(
            f"{self.namespace}_api_response_size_bytes",
            "API response size in bytes",
            ["endpoint"],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self.registry,
        )

        logger.info(f"Created {len(self.metrics)} default Prometheus metrics")

    def _start_metrics_server(self):
        """Start HTTP server for Prometheus metrics scraping."""
        if self.server_started or not PROMETHEUS_AVAILABLE:
            return

        try:
            start_http_server(self.port, registry=self.registry)
            self.server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: Response status code
            duration: Request duration in seconds
        """
        if "http_requests_total" in self.metrics:
            self.metrics["http_requests_total"].labels(
                method=method, endpoint=endpoint, status=str(status_code)
            ).inc()

        if "http_request_duration" in self.metrics:
            self.metrics["http_request_duration"].labels(
                method=method, endpoint=endpoint
            ).observe(duration)

    def record_detection(
        self,
        algorithm: str,
        dataset_type: str,
        dataset_size: int,
        duration: float,
        anomalies_found: int,
        success: bool,
        accuracy: Optional[float] = None,
    ):
        """Record anomaly detection metrics.

        Args:
            algorithm: Algorithm used
            dataset_type: Type of dataset
            dataset_size: Size of dataset
            duration: Detection duration
            anomalies_found: Number of anomalies found
            success: Whether detection was successful
            accuracy: Detection accuracy if available
        """
        status = "success" if success else "failure"
        size_category = self._categorize_size(dataset_size)

        if "detections_total" in self.metrics:
            self.metrics["detections_total"].labels(
                algorithm=algorithm, dataset_type=dataset_type, status=status
            ).inc()

        if "detection_duration" in self.metrics:
            self.metrics["detection_duration"].labels(
                algorithm=algorithm, dataset_size_category=size_category
            ).observe(duration)

        if anomalies_found > 0 and "anomalies_found" in self.metrics:
            confidence_level = (
                "high"
                if anomalies_found > 10
                else "medium" if anomalies_found > 1 else "low"
            )
            self.metrics["anomalies_found"].labels(
                algorithm=algorithm, confidence_level=confidence_level
            ).inc(anomalies_found)

        if accuracy is not None and "detection_accuracy" in self.metrics:
            self.metrics["detection_accuracy"].labels(
                algorithm=algorithm, dataset_type=dataset_type
            ).set(accuracy)

    def record_training(
        self,
        algorithm: str,
        dataset_size: int,
        duration: float,
        model_size_bytes: int,
        success: bool,
    ):
        """Record model training metrics.

        Args:
            algorithm: Algorithm trained
            dataset_size: Training dataset size
            duration: Training duration
            model_size_bytes: Size of trained model
            success: Whether training was successful
        """
        status = "success" if success else "failure"
        size_category = self._categorize_size(dataset_size)

        if "training_requests_total" in self.metrics:
            self.metrics["training_requests_total"].labels(
                algorithm=algorithm, status=status
            ).inc()

        if "training_duration" in self.metrics:
            self.metrics["training_duration"].labels(
                algorithm=algorithm, dataset_size_category=size_category
            ).observe(duration)

        if "model_size" in self.metrics:
            self.metrics["model_size"].labels(algorithm=algorithm).observe(
                model_size_bytes
            )

    def record_streaming_metrics(
        self,
        stream_id: str,
        samples_processed: int,
        throughput: float,
        buffer_utilization: float,
        backpressure_events: int = 0,
        backpressure_strategy: str = "none",
    ):
        """Record streaming metrics.

        Args:
            stream_id: Stream identifier
            samples_processed: Number of samples processed
            throughput: Current throughput
            buffer_utilization: Buffer utilization ratio
            backpressure_events: Number of backpressure events
            backpressure_strategy: Backpressure strategy used
        """
        if "streaming_samples_total" in self.metrics:
            self.metrics["streaming_samples_total"].labels(
                stream_id=stream_id, status="processed"
            ).inc(samples_processed)

        if "streaming_throughput" in self.metrics:
            self.metrics["streaming_throughput"].labels(stream_id=stream_id).set(
                throughput
            )

        if "streaming_buffer_utilization" in self.metrics:
            self.metrics["streaming_buffer_utilization"].labels(
                stream_id=stream_id
            ).set(buffer_utilization)

        if backpressure_events > 0 and "streaming_backpressure_events" in self.metrics:
            self.metrics["streaming_backpressure_events"].labels(
                stream_id=stream_id, strategy=backpressure_strategy
            ).inc(backpressure_events)

    def record_ensemble_metrics(
        self,
        voting_strategy: str,
        detector_count: int,
        agreement_ratio: float,
        predictions_count: int = 1,
    ):
        """Record ensemble metrics.

        Args:
            voting_strategy: Voting strategy used
            detector_count: Number of detectors in ensemble
            agreement_ratio: Agreement ratio among detectors
            predictions_count: Number of predictions made
        """
        if "ensemble_predictions_total" in self.metrics:
            self.metrics["ensemble_predictions_total"].labels(
                voting_strategy=voting_strategy, detector_count=str(detector_count)
            ).inc(predictions_count)

        if "ensemble_agreement" in self.metrics:
            self.metrics["ensemble_agreement"].labels(
                voting_strategy=voting_strategy
            ).observe(agreement_ratio)

    def update_system_metrics(
        self,
        active_models: int,
        active_streams: int,
        memory_usage: Dict[str, int],
        cpu_usage: Dict[str, float],
    ):
        """Update system metrics.

        Args:
            active_models: Number of active models
            active_streams: Number of active streams
            memory_usage: Memory usage by component
            cpu_usage: CPU usage by component
        """
        if "active_models" in self.metrics:
            self.metrics["active_models"].set(active_models)

        if "active_streams" in self.metrics:
            self.metrics["active_streams"].set(active_streams)

        if "memory_usage" in self.metrics:
            for component, usage in memory_usage.items():
                self.metrics["memory_usage"].labels(component=component).set(usage)

        if "cpu_usage" in self.metrics:
            for component, usage in cpu_usage.items():
                self.metrics["cpu_usage"].labels(component=component).set(usage)

    def record_cache_metrics(
        self, cache_type: str, operation: str, hit: bool, cache_size: int
    ):
        """Record cache metrics.

        Args:
            cache_type: Type of cache
            operation: Cache operation (get, set, delete)
            hit: Whether operation was a hit
            cache_size: Current cache size
        """
        status = "hit" if hit else "miss"

        if "cache_operations_total" in self.metrics:
            self.metrics["cache_operations_total"].labels(
                cache_type=cache_type, operation=operation, status=status
            ).inc()

        if "cache_size" in self.metrics:
            self.metrics["cache_size"].labels(cache_type=cache_type).set(cache_size)

    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record error metrics.

        Args:
            error_type: Type of error
            component: Component where error occurred
            severity: Error severity (debug, info, warning, error, critical)
        """
        if "errors_total" in self.metrics:
            self.metrics["errors_total"].labels(
                error_type=error_type, component=component, severity=severity
            ).inc()

    def update_quality_metrics(
        self,
        dataset_id: str,
        quality_scores: Dict[str, float],
        prediction_confidence: Optional[float] = None,
        algorithm: Optional[str] = None,
    ):
        """Update data quality metrics.

        Args:
            dataset_id: Dataset identifier
            quality_scores: Quality scores by metric type
            prediction_confidence: Prediction confidence if available
            algorithm: Algorithm used if available
        """
        if "data_quality_score" in self.metrics:
            for metric_type, score in quality_scores.items():
                self.metrics["data_quality_score"].labels(
                    dataset_id=dataset_id, metric_type=metric_type
                ).set(score)

        if (
            prediction_confidence is not None
            and algorithm is not None
            and "prediction_confidence" in self.metrics
        ):
            self.metrics["prediction_confidence"].labels(algorithm=algorithm).observe(
                prediction_confidence
            )

    def record_dataset_processing(
        self, source_type: str, format_type: str, size_bytes: int
    ):
        """Record dataset processing metrics.

        Args:
            source_type: Source type (file, database, api, etc.)
            format_type: Data format (csv, json, parquet, etc.)
            size_bytes: Dataset size in bytes
        """
        if "datasets_processed_total" in self.metrics:
            self.metrics["datasets_processed_total"].labels(
                source_type=source_type, format=format_type
            ).inc()

    def record_api_response(self, endpoint: str, response_size_bytes: int):
        """Record API response metrics.

        Args:
            endpoint: API endpoint
            response_size_bytes: Response size in bytes
        """
        if "api_response_size" in self.metrics:
            self.metrics["api_response_size"].labels(endpoint=endpoint).observe(
                response_size_bytes
            )

    def set_application_info(
        self,
        version: str,
        environment: str,
        build_time: str,
        git_commit: str = "unknown",
    ):
        """Set application information metric.

        Args:
            version: Application version
            environment: Environment (dev, staging, prod)
            build_time: Build timestamp
            git_commit: Git commit hash
        """
        if "app_info" in self.metrics:
            self.metrics["app_info"].info(
                {
                    "version": version,
                    "environment": environment,
                    "build_time": build_time,
                    "git_commit": git_commit,
                }
            )

    def get_metrics_data(self) -> bytes:
        """Get metrics data in Prometheus format.

        Returns:
            Metrics data as bytes
        """
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        else:
            return b"# Prometheus metrics not available\n"

    def _categorize_size(self, size: int) -> str:
        """Categorize dataset size for metrics.

        Args:
            size: Dataset size

        Returns:
            Size category
        """
        if size < 1000:
            return "small"
        elif size < 10000:
            return "medium"
        elif size < 100000:
            return "large"
        else:
            return "xlarge"


# Global metrics service instance
_metrics_service: Optional[PrometheusMetricsService] = None


def initialize_metrics(
    enable_default_metrics: bool = True,
    namespace: str = "pynomaly",
    port: Optional[int] = None,
) -> PrometheusMetricsService:
    """Initialize global metrics service.

    Args:
        enable_default_metrics: Whether to create default metrics
        namespace: Metric namespace
        port: Port for metrics server

    Returns:
        Metrics service instance
    """
    global _metrics_service
    _metrics_service = PrometheusMetricsService(
        enable_default_metrics=enable_default_metrics, namespace=namespace, port=port
    )
    return _metrics_service


def get_metrics_service() -> Optional[PrometheusMetricsService]:
    """Get global metrics service instance.

    Returns:
        Metrics service instance or None if not initialized
    """
    return _metrics_service
