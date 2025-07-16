"""Comprehensive tests for infrastructure monitoring - Phase 2 Coverage."""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from monorepo.infrastructure.monitoring import (
    AlertManager,
    HealthCheckManager,
    MetricsCollector,
    OpenTelemetryIntegration,
    PerformanceProfiler,
    PrometheusExporter,
    TelemetryService,
)


@pytest.fixture
def mock_otlp_exporter():
    """Mock OTLP exporter for testing."""
    with patch(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
    ) as mock:
        yield mock


@pytest.fixture
def mock_prometheus_client():
    """Mock Prometheus client for testing."""
    with (
        patch("prometheus_client.CollectorRegistry") as mock_registry,
        patch("prometheus_client.Counter") as mock_counter,
        patch("prometheus_client.Histogram") as mock_histogram,
        patch("prometheus_client.Gauge") as mock_gauge,
    ):
        yield {
            "registry": mock_registry,
            "counter": mock_counter,
            "histogram": mock_histogram,
            "gauge": mock_gauge,
        }


@pytest.fixture
def sample_telemetry_service():
    """Create a sample telemetry service for testing."""
    return TelemetryService(
        service_name="test_service",
        service_version="1.0.0",
        environment="test",
        otlp_endpoint="http://localhost:4317",
    )


@pytest.fixture
def sample_metrics_collector():
    """Create a sample metrics collector for testing."""
    return MetricsCollector(
        namespace="monorepo",
        subsystem="test",
        labels={"environment": "test", "service": "anomaly_detection"},
    )


class TestTelemetryService:
    """Comprehensive tests for TelemetryService functionality."""

    def test_telemetry_service_initialization(self, mock_otlp_exporter):
        """Test telemetry service initialization."""
        service = TelemetryService(
            service_name="test_app",
            service_version="2.0.0",
            environment="production",
            otlp_endpoint="http://collector:4317",
        )

        assert service.service_name == "test_app"
        assert service.service_version == "2.0.0"
        assert service.environment == "production"
        assert service.otlp_endpoint == "http://collector:4317"

    def test_telemetry_service_span_creation(self, sample_telemetry_service):
        """Test span creation and management."""
        with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
            mock_tracer = Mock()
            mock_span = Mock()
            mock_tracer.start_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            # Create span
            with sample_telemetry_service.create_span("test_operation") as span:
                span.set_attribute("operation.type", "anomaly_detection")
                span.set_attribute("dataset.size", 1000)
                span.add_event("Processing started")

            # Verify span operations
            mock_tracer.start_span.assert_called_once_with("test_operation")
            mock_span.set_attribute.assert_any_call(
                "operation.type", "anomaly_detection"
            )
            mock_span.set_attribute.assert_any_call("dataset.size", 1000)
            mock_span.add_event.assert_called_with("Processing started")

    def test_telemetry_service_custom_attributes(self, sample_telemetry_service):
        """Test custom attribute handling."""
        with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
            mock_tracer = Mock()
            mock_span = Mock()
            mock_tracer.start_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            custom_attrs = {
                "algorithm.name": "isolation_forest",
                "model.version": "v1.2",
                "dataset.format": "parquet",
                "batch.size": 500,
            }

            with sample_telemetry_service.create_span(
                "model_training", attributes=custom_attrs
            ):
                pass

            # Verify custom attributes were set
            for key, value in custom_attrs.items():
                mock_span.set_attribute.assert_any_call(key, value)

    def test_telemetry_service_error_tracking(self, sample_telemetry_service):
        """Test error tracking in spans."""
        with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
            mock_tracer = Mock()
            mock_span = Mock()
            mock_tracer.start_span.return_value = mock_span
            mock_get_tracer.return_value = mock_tracer

            try:
                with sample_telemetry_service.create_span("failing_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Verify error was recorded
            mock_span.record_exception.assert_called_once()
            mock_span.set_status.assert_called_once()

    def test_telemetry_service_batch_operations(self, sample_telemetry_service):
        """Test telemetry for batch operations."""
        with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
            mock_tracer = Mock()
            mock_get_tracer.return_value = mock_tracer

            # Mock batch span creation
            spans = []
            for i in range(5):
                mock_span = Mock()
                spans.append(mock_span)
                mock_tracer.start_span.return_value = mock_span

                with sample_telemetry_service.create_span(f"batch_item_{i}"):
                    time.sleep(0.01)  # Simulate work

            # Verify all spans were created
            assert mock_tracer.start_span.call_count == 5

            # Verify spans were properly ended
            for span in spans:
                span.__enter__.assert_called_once()
                span.__exit__.assert_called_once()

    def test_telemetry_service_metrics_integration(self, sample_telemetry_service):
        """Test integration with metrics collection."""
        with patch("opentelemetry.metrics.get_meter") as mock_get_meter:
            mock_meter = Mock()
            mock_counter = Mock()
            mock_histogram = Mock()

            mock_meter.create_counter.return_value = mock_counter
            mock_meter.create_histogram.return_value = mock_histogram
            mock_get_meter.return_value = mock_meter

            # Record metrics
            sample_telemetry_service.record_counter(
                "operations_total", 1, {"status": "success"}
            )
            sample_telemetry_service.record_histogram(
                "operation_duration", 0.5, {"operation": "detection"}
            )

            # Verify metrics were recorded
            mock_counter.add.assert_called_with(1, {"status": "success"})
            mock_histogram.record.assert_called_with(0.5, {"operation": "detection"})

    def test_telemetry_service_resource_detection(self, sample_telemetry_service):
        """Test automatic resource detection."""
        with (
            patch("platform.node") as mock_node,
            patch("os.getpid") as mock_getpid,
            patch("psutil.Process") as mock_process,
        ):
            mock_node.return_value = "test-host"
            mock_getpid.return_value = 12345
            mock_proc = Mock()
            mock_proc.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
            mock_proc.cpu_percent.return_value = 25.5
            mock_process.return_value = mock_proc

            resource_info = sample_telemetry_service.get_resource_info()

            assert resource_info["host.name"] == "test-host"
            assert resource_info["process.pid"] == 12345
            assert resource_info["process.memory.rss"] == 1024 * 1024 * 100
            assert resource_info["process.cpu.percent"] == 25.5


class TestMetricsCollector:
    """Comprehensive tests for MetricsCollector functionality."""

    def test_metrics_collector_initialization(self, mock_prometheus_client):
        """Test metrics collector initialization."""
        collector = MetricsCollector(
            namespace="monorepo",
            subsystem="detection",
            labels={"env": "prod", "version": "1.0"},
        )

        assert collector.namespace == "monorepo"
        assert collector.subsystem == "detection"
        assert collector.labels["env"] == "prod"
        assert collector.labels["version"] == "1.0"

    def test_metrics_collector_counter_operations(
        self, sample_metrics_collector, mock_prometheus_client
    ):
        """Test counter metric operations."""
        # Create and increment counter
        counter_name = "detection_requests_total"
        sample_metrics_collector.create_counter(
            counter_name, "Total number of detection requests", ["algorithm", "status"]
        )

        # Increment counter
        sample_metrics_collector.increment_counter(
            counter_name, 1, {"algorithm": "isolation_forest", "status": "success"}
        )

        sample_metrics_collector.increment_counter(
            counter_name, 2, {"algorithm": "lof", "status": "error"}
        )

        # Verify counter operations
        mock_prometheus_client["counter"].assert_called()
        counter_instance = mock_prometheus_client["counter"].return_value
        counter_instance.labels.assert_called()
        counter_instance.labels.return_value.inc.assert_called()

    def test_metrics_collector_histogram_operations(
        self, sample_metrics_collector, mock_prometheus_client
    ):
        """Test histogram metric operations."""
        # Create histogram
        histogram_name = "detection_duration_seconds"
        sample_metrics_collector.create_histogram(
            histogram_name,
            "Time spent on anomaly detection",
            ["algorithm", "dataset_size"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Record observations
        sample_metrics_collector.observe_histogram(
            histogram_name,
            1.5,
            {"algorithm": "isolation_forest", "dataset_size": "large"},
        )

        sample_metrics_collector.observe_histogram(
            histogram_name, 0.3, {"algorithm": "lof", "dataset_size": "small"}
        )

        # Verify histogram operations
        mock_prometheus_client["histogram"].assert_called()
        histogram_instance = mock_prometheus_client["histogram"].return_value
        histogram_instance.labels.assert_called()
        histogram_instance.labels.return_value.observe.assert_called()

    def test_metrics_collector_gauge_operations(
        self, sample_metrics_collector, mock_prometheus_client
    ):
        """Test gauge metric operations."""
        # Create gauge
        gauge_name = "active_models_count"
        sample_metrics_collector.create_gauge(
            gauge_name, "Number of currently active models", ["model_type"]
        )

        # Set gauge values
        sample_metrics_collector.set_gauge(
            gauge_name, 5, {"model_type": "isolation_forest"}
        )

        sample_metrics_collector.increment_gauge(gauge_name, 2, {"model_type": "lof"})

        sample_metrics_collector.decrement_gauge(
            gauge_name, 1, {"model_type": "isolation_forest"}
        )

        # Verify gauge operations
        mock_prometheus_client["gauge"].assert_called()
        gauge_instance = mock_prometheus_client["gauge"].return_value
        gauge_instance.labels.assert_called()
        gauge_instance.labels.return_value.set.assert_called()
        gauge_instance.labels.return_value.inc.assert_called()
        gauge_instance.labels.return_value.dec.assert_called()

    def test_metrics_collector_custom_metrics(
        self, sample_metrics_collector, mock_prometheus_client
    ):
        """Test custom metric creation and management."""
        # Create custom business metrics
        custom_metrics = [
            {
                "name": "anomaly_detection_accuracy",
                "type": "gauge",
                "description": "Current model accuracy",
                "labels": ["model_id", "dataset"],
            },
            {
                "name": "false_positive_rate",
                "type": "histogram",
                "description": "False positive rate distribution",
                "labels": ["threshold"],
                "buckets": [0.01, 0.05, 0.1, 0.2, 0.5],
            },
        ]

        for metric_config in custom_metrics:
            if metric_config["type"] == "gauge":
                sample_metrics_collector.create_gauge(
                    metric_config["name"],
                    metric_config["description"],
                    metric_config["labels"],
                )
            elif metric_config["type"] == "histogram":
                sample_metrics_collector.create_histogram(
                    metric_config["name"],
                    metric_config["description"],
                    metric_config["labels"],
                    metric_config.get("buckets"),
                )

        # Verify custom metrics were created
        assert mock_prometheus_client["gauge"].call_count >= 1
        assert mock_prometheus_client["histogram"].call_count >= 1

    def test_metrics_collector_batch_operations(
        self, sample_metrics_collector, mock_prometheus_client
    ):
        """Test batch metric operations."""
        # Create metrics for batch operations
        sample_metrics_collector.create_counter(
            "batch_operations_total", "Batch operations", ["status"]
        )
        sample_metrics_collector.create_histogram(
            "batch_processing_time", "Batch processing time", ["batch_size"]
        )

        # Record batch metrics
        batch_data = [
            {
                "metric": "batch_operations_total",
                "value": 1,
                "labels": {"status": "success"},
            },
            {
                "metric": "batch_operations_total",
                "value": 1,
                "labels": {"status": "success"},
            },
            {
                "metric": "batch_processing_time",
                "value": 2.5,
                "labels": {"batch_size": "100"},
            },
            {
                "metric": "batch_processing_time",
                "value": 1.8,
                "labels": {"batch_size": "50"},
            },
        ]

        sample_metrics_collector.record_batch_metrics(batch_data)

        # Verify batch recording
        mock_prometheus_client["counter"].assert_called()
        mock_prometheus_client["histogram"].assert_called()

    def test_metrics_collector_export_functionality(
        self, sample_metrics_collector, mock_prometheus_client
    ):
        """Test metrics export functionality."""
        # Set up mock registry
        mock_registry = mock_prometheus_client["registry"].return_value
        mock_registry.collect.return_value = [
            Mock(name="test_metric_1", samples=[Mock(name="sample1", value=1.0)]),
            Mock(name="test_metric_2", samples=[Mock(name="sample2", value=2.0)]),
        ]

        # Export metrics
        exported_metrics = sample_metrics_collector.export_metrics()

        # Verify export
        assert isinstance(exported_metrics, str | dict)
        mock_registry.collect.assert_called()


class TestOpenTelemetryIntegration:
    """Comprehensive tests for OpenTelemetry integration."""

    def test_otel_integration_initialization(self, mock_otlp_exporter):
        """Test OpenTelemetry integration initialization."""
        integration = OpenTelemetryIntegration(
            service_name="monorepo",
            service_version="1.0.0",
            otlp_endpoint="http://otel-collector:4317",
            sampling_rate=0.1,
        )

        assert integration.service_name == "monorepo"
        assert integration.service_version == "1.0.0"
        assert integration.sampling_rate == 0.1

    def test_otel_auto_instrumentation(self, mock_otlp_exporter):
        """Test automatic instrumentation setup."""
        with patch("opentelemetry.instrumentation.auto_instrumentation.sitecustomize"):
            integration = OpenTelemetryIntegration(
                service_name="test_service",
                auto_instrument=True,
                instrument_libraries=["requests", "sqlalchemy", "fastapi"],
            )

            integration.setup_auto_instrumentation()

            # Verify auto-instrumentation was configured
            assert integration.auto_instrument is True
            assert "requests" in integration.instrument_libraries

    def test_otel_custom_span_processors(self, mock_otlp_exporter):
        """Test custom span processor configuration."""
        integration = OpenTelemetryIntegration(service_name="test_service")

        # Mock span processors
        with (
            patch(
                "opentelemetry.sdk.trace.export.BatchSpanProcessor"
            ) as mock_batch_processor,
            patch(
                "opentelemetry.sdk.trace.export.SimpleSpanProcessor"
            ) as mock_simple_processor,
        ):
            # Configure processors
            integration.add_span_processor("batch", {"max_queue_size": 2048})
            integration.add_span_processor("simple", {})

            # Verify processors were added
            mock_batch_processor.assert_called()
            mock_simple_processor.assert_called()

    def test_otel_resource_detection(self, mock_otlp_exporter):
        """Test resource detection and attributes."""
        with patch("opentelemetry.sdk.resources.Resource") as mock_resource:
            integration = OpenTelemetryIntegration(
                service_name="test_service",
                environment="production",
                deployment_environment="kubernetes",
            )

            # Get resource attributes
            resource_attrs = integration.get_resource_attributes()

            # Verify resource attributes
            assert "service.name" in resource_attrs
            assert "service.version" in resource_attrs
            assert "deployment.environment" in resource_attrs

            mock_resource.create.assert_called()

    def test_otel_sampling_configuration(self, mock_otlp_exporter):
        """Test sampling configuration."""
        # Test different sampling strategies
        sampling_configs = [
            {"strategy": "always_on"},
            {"strategy": "always_off"},
            {"strategy": "ratio_based", "ratio": 0.1},
            {"strategy": "parent_based", "root": "ratio_based", "ratio": 0.05},
        ]

        for config in sampling_configs:
            with patch(
                "opentelemetry.sdk.trace.sampling.TraceIdRatioBasedSampler"
            ) as mock_sampler:
                integration = OpenTelemetryIntegration(
                    service_name="test_service", sampling_config=config
                )

                integration.configure_sampling()

                if config["strategy"] == "ratio_based":
                    mock_sampler.assert_called_with(config["ratio"])


class TestPrometheusExporter:
    """Comprehensive tests for Prometheus exporter."""

    def test_prometheus_exporter_initialization(self, mock_prometheus_client):
        """Test Prometheus exporter initialization."""
        exporter = PrometheusExporter(
            port=8000, endpoint="/metrics", namespace="monorepo", registry=None
        )

        assert exporter.port == 8000
        assert exporter.endpoint == "/metrics"
        assert exporter.namespace == "monorepo"

    def test_prometheus_exporter_http_server(self, mock_prometheus_client):
        """Test Prometheus HTTP server setup."""
        with patch("prometheus_client.start_http_server") as mock_start_server:
            exporter = PrometheusExporter(port=8080)
            exporter.start_server()

            mock_start_server.assert_called_with(8080, registry=exporter.registry)

    def test_prometheus_exporter_custom_collectors(self, mock_prometheus_client):
        """Test custom collector registration."""
        exporter = PrometheusExporter()

        # Mock custom collector
        class CustomCollector:
            def collect(self):
                yield Mock(name="custom_metric", samples=[Mock(value=42.0)])

        custom_collector = CustomCollector()
        exporter.register_collector(custom_collector)

        # Verify collector registration
        mock_prometheus_client["registry"].return_value.register.assert_called_with(
            custom_collector
        )

    def test_prometheus_exporter_metric_formatting(self, mock_prometheus_client):
        """Test Prometheus metric formatting."""
        exporter = PrometheusExporter(namespace="monorepo")

        # Test metric name formatting
        formatted_name = exporter.format_metric_name("detection_requests_total")
        assert formatted_name == "pynomaly_detection_requests_total"

        # Test label formatting
        labels = {"algorithm": "isolation_forest", "status": "success"}
        formatted_labels = exporter.format_labels(labels)
        assert "algorithm" in formatted_labels
        assert "status" in formatted_labels

    def test_prometheus_exporter_scrape_endpoint(self, mock_prometheus_client):
        """Test Prometheus scrape endpoint."""
        with patch("prometheus_client.generate_latest") as mock_generate:
            mock_generate.return_value = b"# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1.0\n"

            exporter = PrometheusExporter()
            metrics_output = exporter.get_metrics()

            assert b"test_metric" in metrics_output
            mock_generate.assert_called_with(exporter.registry)


class TestHealthCheckManager:
    """Comprehensive tests for health check management."""

    def test_health_check_manager_initialization(self):
        """Test health check manager initialization."""
        manager = HealthCheckManager(check_interval=30.0, timeout=5.0, max_failures=3)

        assert manager.check_interval == 30.0
        assert manager.timeout == 5.0
        assert manager.max_failures == 3

    def test_health_check_registration(self):
        """Test health check registration."""
        manager = HealthCheckManager()

        def database_health_check():
            return {"status": "healthy", "response_time": 0.05}

        def redis_health_check():
            return {"status": "healthy", "connection_count": 10}

        # Register health checks
        manager.register_check("database", database_health_check)
        manager.register_check("redis", redis_health_check)

        # Verify registration
        assert "database" in manager.checks
        assert "redis" in manager.checks

    def test_health_check_execution(self):
        """Test health check execution."""
        manager = HealthCheckManager()

        def healthy_check():
            return {"status": "healthy", "details": "All systems operational"}

        def unhealthy_check():
            return {"status": "unhealthy", "error": "Connection timeout"}

        def failing_check():
            raise Exception("Check failed")

        manager.register_check("healthy_service", healthy_check)
        manager.register_check("unhealthy_service", unhealthy_check)
        manager.register_check("failing_service", failing_check)

        # Execute health checks
        results = manager.run_all_checks()

        # Verify results
        assert results["healthy_service"]["status"] == "healthy"
        assert results["unhealthy_service"]["status"] == "unhealthy"
        assert results["failing_service"]["status"] == "error"
        assert "exception" in results["failing_service"]

    def test_health_check_aggregation(self):
        """Test health check result aggregation."""
        manager = HealthCheckManager()

        def healthy_check():
            return {"status": "healthy"}

        def degraded_check():
            return {"status": "degraded", "warning": "High latency"}

        def unhealthy_check():
            return {"status": "unhealthy", "error": "Service down"}

        manager.register_check("service_1", healthy_check)
        manager.register_check("service_2", degraded_check)
        manager.register_check("service_3", unhealthy_check)

        # Get overall health
        overall_health = manager.get_overall_health()

        # Should be unhealthy due to one unhealthy service
        assert overall_health["status"] == "unhealthy"
        assert overall_health["total_checks"] == 3
        assert overall_health["healthy_count"] == 1
        assert overall_health["degraded_count"] == 1
        assert overall_health["unhealthy_count"] == 1

    def test_health_check_persistence(self):
        """Test health check result persistence."""
        manager = HealthCheckManager(persist_results=True)

        def sample_check():
            return {"status": "healthy", "timestamp": time.time()}

        manager.register_check("sample_service", sample_check)

        # Run checks multiple times
        for _i in range(3):
            manager.run_all_checks()
            time.sleep(0.1)

        # Get history
        history = manager.get_check_history("sample_service")

        assert len(history) == 3
        assert all(result["status"] == "healthy" for result in history)

    def test_health_check_alerting(self):
        """Test health check alerting integration."""
        alert_calls = []

        def mock_alert_handler(service_name, check_result):
            alert_calls.append((service_name, check_result))

        manager = HealthCheckManager()
        manager.set_alert_handler(mock_alert_handler)

        def failing_check():
            return {"status": "unhealthy", "error": "Service down"}

        manager.register_check("critical_service", failing_check)
        manager.run_all_checks()

        # Verify alert was triggered
        assert len(alert_calls) == 1
        assert alert_calls[0][0] == "critical_service"
        assert alert_calls[0][1]["status"] == "unhealthy"


class TestAlertManager:
    """Comprehensive tests for alert management."""

    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager(
            default_severity="warning", rate_limit_window=300, max_alerts_per_window=10
        )

        assert manager.default_severity == "warning"
        assert manager.rate_limit_window == 300
        assert manager.max_alerts_per_window == 10

    def test_alert_creation_and_routing(self):
        """Test alert creation and routing."""
        manager = AlertManager()

        # Mock alert handlers
        email_alerts = []
        slack_alerts = []

        def email_handler(alert):
            email_alerts.append(alert)

        def slack_handler(alert):
            slack_alerts.append(alert)

        # Register handlers
        manager.register_handler("email", email_handler)
        manager.register_handler("slack", slack_handler)

        # Create alert
        alert = {
            "id": "test_alert_001",
            "title": "High Error Rate",
            "description": "Error rate exceeded threshold",
            "severity": "critical",
            "source": "anomaly_detection_service",
            "tags": ["performance", "errors"],
            "handlers": ["email", "slack"],
        }

        manager.create_alert(alert)

        # Verify alert routing
        assert len(email_alerts) == 1
        assert len(slack_alerts) == 1
        assert email_alerts[0]["id"] == "test_alert_001"
        assert slack_alerts[0]["severity"] == "critical"

    def test_alert_rate_limiting(self):
        """Test alert rate limiting."""
        manager = AlertManager(
            rate_limit_window=1,  # 1 second window
            max_alerts_per_window=2,
        )

        handled_alerts = []

        def test_handler(alert):
            handled_alerts.append(alert)

        manager.register_handler("test", test_handler)

        # Send alerts rapidly
        for i in range(5):
            alert = {
                "id": f"alert_{i}",
                "title": f"Alert {i}",
                "severity": "warning",
                "handlers": ["test"],
            }
            manager.create_alert(alert)

        # Should only handle first 2 alerts due to rate limiting
        assert len(handled_alerts) <= 2

    def test_alert_severity_filtering(self):
        """Test alert filtering by severity."""
        manager = AlertManager()

        critical_alerts = []
        warning_alerts = []

        def critical_handler(alert):
            if alert["severity"] == "critical":
                critical_alerts.append(alert)

        def warning_handler(alert):
            if alert["severity"] in ["warning", "critical"]:
                warning_alerts.append(alert)

        manager.register_handler("critical_only", critical_handler)
        manager.register_handler("warning_and_up", warning_handler)

        # Create alerts with different severities
        alerts = [
            {
                "id": "1",
                "severity": "info",
                "handlers": ["critical_only", "warning_and_up"],
            },
            {
                "id": "2",
                "severity": "warning",
                "handlers": ["critical_only", "warning_and_up"],
            },
            {
                "id": "3",
                "severity": "critical",
                "handlers": ["critical_only", "warning_and_up"],
            },
        ]

        for alert in alerts:
            manager.create_alert(alert)

        # Verify filtering
        assert len(critical_alerts) == 1  # Only critical alert
        assert len(warning_alerts) == 2  # Warning and critical alerts

    def test_alert_deduplication(self):
        """Test alert deduplication."""
        manager = AlertManager(deduplication_window=60)  # 1 minute

        handled_alerts = []

        def test_handler(alert):
            handled_alerts.append(alert)

        manager.register_handler("test", test_handler)

        # Send duplicate alerts
        base_alert = {
            "title": "Database Connection Failed",
            "source": "database_service",
            "severity": "critical",
            "handlers": ["test"],
        }

        # Send same alert multiple times
        for i in range(3):
            alert = base_alert.copy()
            alert["id"] = f"alert_{i}"
            alert["timestamp"] = time.time()
            manager.create_alert(alert)

        # Should deduplicate based on title and source
        assert len(handled_alerts) == 1


class TestPerformanceProfiler:
    """Comprehensive tests for performance profiling."""

    def test_performance_profiler_initialization(self):
        """Test performance profiler initialization."""
        profiler = PerformanceProfiler(
            enabled=True, sampling_interval=0.01, output_directory="./profiles"
        )

        assert profiler.enabled is True
        assert profiler.sampling_interval == 0.01
        assert profiler.output_directory == "./profiles"

    def test_performance_profiler_function_timing(self):
        """Test function timing profiling."""
        profiler = PerformanceProfiler()

        @profiler.profile_function
        def slow_function():
            time.sleep(0.1)
            return "completed"

        @profiler.profile_function
        def fast_function():
            return "completed"

        # Execute functions
        slow_function()
        fast_function()

        # Get profiling results
        results = profiler.get_function_profiles()

        assert "slow_function" in results
        assert "fast_function" in results
        assert (
            results["slow_function"]["avg_duration"]
            > results["fast_function"]["avg_duration"]
        )
        assert results["slow_function"]["call_count"] == 1
        assert results["fast_function"]["call_count"] == 1

    def test_performance_profiler_context_manager(self):
        """Test performance profiler context manager."""
        profiler = PerformanceProfiler()

        # Profile code block
        with profiler.profile_block("database_query"):
            time.sleep(0.05)
            # Simulate database query

        with profiler.profile_block("api_call"):
            time.sleep(0.02)
            # Simulate API call

        # Get block profiles
        block_profiles = profiler.get_block_profiles()

        assert "database_query" in block_profiles
        assert "api_call" in block_profiles
        assert block_profiles["database_query"]["total_duration"] > 0.04
        assert block_profiles["api_call"]["total_duration"] > 0.01

    def test_performance_profiler_memory_tracking(self):
        """Test memory usage tracking."""
        profiler = PerformanceProfiler(track_memory=True)

        @profiler.profile_function
        def memory_intensive_function():
            # Simulate memory allocation
            large_list = list(range(10000))
            return len(large_list)

        memory_intensive_function()

        # Get memory profiles
        memory_profiles = profiler.get_memory_profiles()

        assert "memory_intensive_function" in memory_profiles
        assert memory_profiles["memory_intensive_function"]["peak_memory"] > 0

    def test_performance_profiler_report_generation(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler()

        # Profile some operations
        with profiler.profile_block("operation_1"):
            time.sleep(0.01)

        with profiler.profile_block("operation_2"):
            time.sleep(0.02)

        @profiler.profile_function
        def test_function():
            time.sleep(0.005)
            return "done"

        test_function()

        # Generate report
        report = profiler.generate_report()

        assert "summary" in report
        assert "function_profiles" in report
        assert "block_profiles" in report
        assert "recommendations" in report

        # Verify summary statistics
        summary = report["summary"]
        assert "total_profiled_functions" in summary
        assert "total_profiled_blocks" in summary
        assert "total_execution_time" in summary
