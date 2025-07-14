"""Tests for Prometheus metrics service."""

from unittest.mock import patch

from pynomaly.infrastructure.monitoring.prometheus_metrics import (
    MetricDefinition,
    PrometheusMetricsService,
    get_metrics_service,
    initialize_metrics,
)


class TestPrometheusMetricsService:
    """Test cases for PrometheusMetricsService."""

    def test_init_without_prometheus(self):
        """Test initialization when Prometheus client is not available."""
        with patch(
            "pynomaly.infrastructure.monitoring.prometheus_metrics.PROMETHEUS_AVAILABLE",
            False,
        ):
            service = PrometheusMetricsService()
            assert service.metrics == {}
            assert service.server_started is False

    def test_init_with_prometheus(self):
        """Test initialization with Prometheus client available."""
        service = PrometheusMetricsService(port=None)  # Don't start server
        assert len(service.metrics) > 0
        assert "http_requests_total" in service.metrics
        assert "detection_duration" in service.metrics
        assert service.server_started is False

    def test_init_with_custom_namespace(self):
        """Test initialization with custom namespace."""
        service = PrometheusMetricsService(namespace="test_app", port=None)
        assert service.namespace == "test_app"
        # Check that metrics have correct namespace prefix
        for metric_name in service.metrics.keys():
            if hasattr(service.metrics[metric_name], "_name"):
                assert service.metrics[metric_name]._name.startswith("test_app_")

    def test_record_http_request(self):
        """Test recording HTTP request metrics."""
        service = PrometheusMetricsService(port=None)

        # Record some HTTP requests
        service.record_http_request("GET", "/api/detect", 200, 0.5)
        service.record_http_request("POST", "/api/train", 201, 1.2)
        service.record_http_request("GET", "/api/detect", 400, 0.1)

        # Metrics should be recorded (can't easily verify mock metrics values)
        assert "http_requests_total" in service.metrics
        assert "http_request_duration" in service.metrics

    def test_record_detection(self):
        """Test recording detection metrics."""
        service = PrometheusMetricsService(port=None)

        # Record successful detection
        service.record_detection(
            algorithm="IsolationForest",
            dataset_type="tabular",
            dataset_size=1000,
            duration=2.5,
            anomalies_found=15,
            success=True,
            accuracy=0.92,
        )

        # Record failed detection
        service.record_detection(
            algorithm="OneClassSVM",
            dataset_type="time_series",
            dataset_size=500,
            duration=1.0,
            anomalies_found=0,
            success=False,
        )

        assert "detections_total" in service.metrics
        assert "detection_duration" in service.metrics
        assert "anomalies_found" in service.metrics
        assert "detection_accuracy" in service.metrics

    def test_record_training(self):
        """Test recording training metrics."""
        service = PrometheusMetricsService(port=None)

        service.record_training(
            algorithm="DBSCAN",
            dataset_size=5000,
            duration=45.0,
            model_size_bytes=1024000,
            success=True,
        )

        assert "training_requests_total" in service.metrics
        assert "training_duration" in service.metrics
        assert "model_size" in service.metrics

    def test_record_streaming_metrics(self):
        """Test recording streaming metrics."""
        service = PrometheusMetricsService(port=None)

        service.record_streaming_metrics(
            stream_id="stream_001",
            samples_processed=100,
            throughput=50.0,
            buffer_utilization=0.75,
            backpressure_events=2,
            backpressure_strategy="adaptive_sampling",
        )

        assert "streaming_samples_total" in service.metrics
        assert "streaming_throughput" in service.metrics
        assert "streaming_buffer_utilization" in service.metrics
        assert "streaming_backpressure_events" in service.metrics

    def test_record_ensemble_metrics(self):
        """Test recording ensemble metrics."""
        service = PrometheusMetricsService(port=None)

        service.record_ensemble_metrics(
            voting_strategy="weighted_average",
            detector_count=5,
            agreement_ratio=0.85,
            predictions_count=10,
        )

        assert "ensemble_predictions_total" in service.metrics
        assert "ensemble_agreement" in service.metrics

    def test_update_system_metrics(self):
        """Test updating system metrics."""
        service = PrometheusMetricsService(port=None)

        service.update_system_metrics(
            active_models=3,
            active_streams=2,
            memory_usage={"models": 1024000000, "streaming": 512000000},
            cpu_usage={"detection": 0.45, "training": 0.30},
        )

        assert "active_models" in service.metrics
        assert "active_streams" in service.metrics
        assert "memory_usage" in service.metrics
        assert "cpu_usage" in service.metrics

    def test_record_cache_metrics(self):
        """Test recording cache metrics."""
        service = PrometheusMetricsService(port=None)

        # Record cache hit
        service.record_cache_metrics(
            cache_type="model", operation="get", hit=True, cache_size=50
        )

        # Record cache miss
        service.record_cache_metrics(
            cache_type="result", operation="get", hit=False, cache_size=25
        )

        assert "cache_operations_total" in service.metrics
        assert "cache_size" in service.metrics

    def test_record_error(self):
        """Test recording error metrics."""
        service = PrometheusMetricsService(port=None)

        service.record_error(
            error_type="ValidationError", component="api", severity="warning"
        )

        service.record_error(
            error_type="DatabaseError", component="persistence", severity="critical"
        )

        assert "errors_total" in service.metrics

    def test_update_quality_metrics(self):
        """Test updating quality metrics."""
        service = PrometheusMetricsService(port=None)

        service.update_quality_metrics(
            dataset_id="dataset_001",
            quality_scores={
                "completeness": 0.95,
                "consistency": 0.87,
                "accuracy": 0.92,
            },
            prediction_confidence=0.88,
            algorithm="IsolationForest",
        )

        assert "data_quality_score" in service.metrics
        assert "prediction_confidence" in service.metrics

    def test_record_dataset_processing(self):
        """Test recording dataset processing metrics."""
        service = PrometheusMetricsService(port=None)

        service.record_dataset_processing(
            source_type="file", format_type="csv", size_bytes=1048576
        )

        assert "datasets_processed_total" in service.metrics

    def test_record_api_response(self):
        """Test recording API response metrics."""
        service = PrometheusMetricsService(port=None)

        service.record_api_response(endpoint="/api/detect", response_size_bytes=2048)

        assert "api_response_size" in service.metrics

    def test_set_application_info(self):
        """Test setting application info."""
        service = PrometheusMetricsService(port=None)

        service.set_application_info(
            version="1.0.0",
            environment="production",
            build_time="2024-12-26T10:30:00Z",
            git_commit="abc123def",
        )

        assert "app_info" in service.metrics

    def test_get_metrics_data(self):
        """Test getting metrics data."""
        service = PrometheusMetricsService(port=None)

        # Record some metrics first
        service.record_http_request("GET", "/test", 200, 0.1)

        metrics_data = service.get_metrics_data()
        assert isinstance(metrics_data, bytes)
        assert len(metrics_data) > 0

        # Should contain some metric information
        metrics_str = metrics_data.decode("utf-8")
        assert "pynomaly" in metrics_str

    def test_categorize_size(self):
        """Test dataset size categorization."""
        service = PrometheusMetricsService(port=None)

        assert service._categorize_size(100) == "small"
        assert service._categorize_size(5000) == "medium"
        assert service._categorize_size(50000) == "large"
        assert service._categorize_size(500000) == "xlarge"

    def test_without_prometheus_client(self):
        """Test service functionality when Prometheus client is not available."""
        with patch(
            "pynomaly.infrastructure.monitoring.prometheus_metrics.PROMETHEUS_AVAILABLE",
            False,
        ):
            service = PrometheusMetricsService()

            # All methods should work without errors
            service.record_http_request("GET", "/test", 200, 0.1)
            service.record_detection("test", "test", 100, 1.0, 5, True)
            service.record_training("test", 100, 1.0, 1000, True)

            # Get metrics should return placeholder
            metrics_data = service.get_metrics_data()
            assert b"not available" in metrics_data


class TestMetricDefinition:
    """Test cases for MetricDefinition."""

    def test_metric_definition_creation(self):
        """Test creating metric definition."""
        metric_def = MetricDefinition(
            name="test_metric",
            help_text="Test metric description",
            metric_type="counter",
            labels=["method", "status"],
            buckets=[0.1, 0.5, 1.0, 5.0],
        )

        assert metric_def.name == "test_metric"
        assert metric_def.help_text == "Test metric description"
        assert metric_def.metric_type == "counter"
        assert metric_def.labels == ["method", "status"]
        assert metric_def.buckets == [0.1, 0.5, 1.0, 5.0]

    def test_metric_definition_defaults(self):
        """Test metric definition with default values."""
        metric_def = MetricDefinition(
            name="simple_metric", help_text="Simple metric", metric_type="gauge"
        )

        assert metric_def.labels == []
        assert metric_def.buckets is None
        assert metric_def.states is None


class TestGlobalMetricsService:
    """Test cases for global metrics service functions."""

    def test_initialize_metrics(self):
        """Test global metrics service initialization."""
        service = initialize_metrics(
            enable_default_metrics=True, namespace="test_pynomaly", port=None
        )

        assert isinstance(service, PrometheusMetricsService)
        assert service.namespace == "test_pynomaly"

        # Should be retrievable via get_metrics_service
        retrieved_service = get_metrics_service()
        assert retrieved_service is service

    def test_get_metrics_service_not_initialized(self):
        """Test getting metrics service when not initialized."""
        # Reset global service
        import pynomaly.infrastructure.monitoring.prometheus_metrics as pm

        pm._metrics_service = None

        service = get_metrics_service()
        assert service is None
