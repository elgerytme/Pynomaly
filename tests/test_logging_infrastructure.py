"""Comprehensive tests for the logging and observability infrastructure."""

import tempfile
import time
from datetime import timedelta
from pathlib import Path

import pytest

from pynomaly.infrastructure.logging.log_aggregator import (
    LogAggregator,
    LogEntry,
    LogFilter,
    LogStreamType,
)
from pynomaly.infrastructure.logging.log_analysis import (
    AnomalyDetector,
    LogAnalyzer,
    PatternRule,
    PatternType,
    Severity,
)
from pynomaly.infrastructure.logging.metrics_collector import (
    MetricsCollector,
)
from pynomaly.infrastructure.logging.observability_service import (
    ObservabilityConfig,
    ObservabilityService,
)
from pynomaly.infrastructure.logging.structured_logger import (
    LogContext,
    LogLevel,
    StructuredLogger,
    get_logger,
)
from pynomaly.infrastructure.logging.tracing_manager import TracingManager


class TestStructuredLogger:
    """Test structured logging functionality."""

    def test_basic_logging(self):
        """Test basic logging functionality."""
        logger = StructuredLogger("test.logger", level=LogLevel.DEBUG)

        # Test logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Check metrics
        metrics = logger.get_metrics()
        assert metrics["logs_written"] >= 5
        assert metrics["logger_name"] == "test.logger"

    def test_context_management(self):
        """Test log context management."""
        context = LogContext(
            correlation_id="test-123", user_id="user-456", operation="test_operation"
        )

        StructuredLogger.set_context(context)
        current_context = StructuredLogger.get_current_context()

        assert current_context is not None
        assert current_context.correlation_id == "test-123"
        assert current_context.user_id == "user-456"
        assert current_context.operation == "test_operation"

        StructuredLogger.clear_context()
        assert StructuredLogger.get_current_context() is None

    def test_performance_logging(self):
        """Test performance logging context manager."""
        logger = StructuredLogger("test.perf", level=LogLevel.DEBUG)

        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir) / "test.log"

            from pynomaly.infrastructure.logging.structured_logger import (
                PerformanceLogger,
            )

            with PerformanceLogger(logger, "test_operation", min_duration_ms=0):
                time.sleep(0.01)  # Small delay to ensure measurable duration

            metrics = logger.get_metrics()
            assert metrics["performance_logs"] >= 1

    def test_global_logger_registry(self):
        """Test global logger registry."""
        logger1 = get_logger("test.app", LogLevel.INFO)
        logger2 = get_logger("test.app", LogLevel.DEBUG)

        # Should return the same instance
        assert logger1 is logger2
        assert logger1.name == "test.app"


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def test_basic_metrics(self):
        """Test basic metric collection."""
        collector = MetricsCollector(auto_flush=False, enable_system_metrics=False)

        # Test different metric types
        collector.counter("test.counter", 1)
        collector.gauge("test.gauge", 42.5)
        collector.histogram("test.histogram", 100)
        collector.timer("test.timer", 250.5)

        # Check metrics
        assert collector.get_counter_value("test.counter") == 1.0
        assert collector.get_gauge_value("test.gauge") == 42.5

        # Test histogram summary
        histogram_summary = collector.get_histogram_summary("test.histogram")
        assert histogram_summary is not None
        assert histogram_summary.count == 1
        assert histogram_summary.sum == 100.0

    def test_metrics_with_labels(self):
        """Test metrics with labels."""
        collector = MetricsCollector(auto_flush=False, enable_system_metrics=False)

        labels = {"service": "pynomaly", "env": "test"}
        collector.counter("requests.total", 5, labels)
        collector.gauge("memory.usage", 75.5, labels)

        assert collector.get_counter_value("requests.total", labels) == 5.0
        assert collector.get_gauge_value("memory.usage", labels) == 75.5

    def test_timer_context(self):
        """Test timer context manager."""
        collector = MetricsCollector(auto_flush=False, enable_system_metrics=False)

        from pynomaly.infrastructure.logging.metrics_collector import TimerContext

        with TimerContext(collector, "operation.duration"):
            time.sleep(0.01)

        timer_summary = collector.get_timer_summary("operation.duration")
        assert timer_summary is not None
        assert timer_summary.count == 1
        assert timer_summary.mean > 0

    def test_metrics_persistence(self):
        """Test metrics persistence to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "metrics.json"

            collector = MetricsCollector(
                storage_path=storage_path, auto_flush=False, enable_system_metrics=False
            )

            collector.counter("test.metric", 1)
            collector.flush_metrics()

            # Check that metric files were created
            metric_files = list(storage_path.parent.glob("metrics_*.json"))
            assert len(metric_files) > 0


class TestLogAggregator:
    """Test log aggregation and streaming."""

    def test_log_stream_creation(self):
        """Test creating and managing log streams."""
        aggregator = LogAggregator(
            enable_persistence=False,
            aggregation_interval=0,  # Disable background aggregation
        )

        # Create different types of streams
        error_stream = aggregator.create_stream(
            "errors", LogStreamType.REALTIME, LogFilter(levels=["ERROR", "CRITICAL"])
        )

        batch_stream = aggregator.create_stream(
            "batch_logs", LogStreamType.BATCH, LogFilter(), batch_size=5
        )

        assert error_stream.name == "errors"
        assert batch_stream.name == "batch_logs"
        assert aggregator.get_stream("errors") is error_stream

    def test_log_filtering(self):
        """Test log filtering functionality."""
        filter_config = LogFilter(
            levels=["ERROR", "WARNING"], loggers=["test.module"], tags=["critical"]
        )

        # Test matching entry
        matching_entry = LogEntry(
            level="ERROR",
            logger_name="test.module.submodule",
            message="Test error",
            tags=["critical", "urgent"],
        )

        # Test non-matching entry
        non_matching_entry = LogEntry(
            level="INFO",
            logger_name="other.module",
            message="Test info",
            tags=["normal"],
        )

        assert filter_config.matches(matching_entry) is True
        assert filter_config.matches(non_matching_entry) is False

    def test_log_stream_subscription(self):
        """Test log stream subscription mechanism."""
        aggregator = LogAggregator(enable_persistence=False, aggregation_interval=0)

        stream = aggregator.create_stream("test", LogStreamType.REALTIME)

        received_entries = []

        def log_subscriber(entry: LogEntry):
            received_entries.append(entry)

        stream.subscribe(log_subscriber)

        # Add log entries
        entry1 = LogEntry(level="INFO", message="Test message 1")
        entry2 = LogEntry(level="ERROR", message="Test message 2")

        aggregator.add_log_entry(entry1)
        aggregator.add_log_entry(entry2)

        # Verify entries were received
        assert len(received_entries) == 2
        assert received_entries[0].message == "Test message 1"
        assert received_entries[1].message == "Test message 2"


class TestLogAnalyzer:
    """Test log pattern analysis and anomaly detection."""

    def test_pattern_detection(self):
        """Test basic pattern detection."""
        analyzer = LogAnalyzer(enable_realtime=True, enable_background_analysis=False)

        # Create error spike scenario
        error_entries = [
            LogEntry(
                level="ERROR", message=f"Database error {i}", logger_name="db.connector"
            )
            for i in range(15)
        ]

        for entry in error_entries:
            analyzer.analyze_entry(entry)

        # Check for detected patterns
        patterns = analyzer.get_patterns()
        assert len(patterns) > 0

        # Should detect error spike pattern
        error_spike_patterns = [
            p for p in patterns if p.pattern_type == PatternType.ERROR_SPIKE
        ]
        assert len(error_spike_patterns) > 0

    def test_custom_pattern_rules(self):
        """Test adding custom pattern detection rules."""
        analyzer = LogAnalyzer(enable_realtime=True, enable_background_analysis=False)

        # Add custom rule for authentication failures
        custom_rule = PatternRule(
            id="custom_auth_fail",
            name="Custom Authentication Failure",
            pattern_type=PatternType.SECURITY_THREAT,
            conditions={"message_pattern": r"authentication.*failed", "min_count": 3},
            severity=Severity.HIGH,
            threshold=3,
            time_window=timedelta(minutes=1),
        )

        analyzer.add_rule(custom_rule)

        # Generate matching entries
        auth_entries = [
            LogEntry(level="WARNING", message=f"Authentication failed for user{i}")
            for i in range(5)
        ]

        for entry in auth_entries:
            analyzer.analyze_entry(entry)

        patterns = analyzer.get_patterns()
        custom_patterns = [p for p in patterns if "Authentication Failure" in p.title]
        assert len(custom_patterns) > 0

    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection."""
        detector = AnomalyDetector(window_size=50, sensitivity=2.0, min_samples=10)

        # Generate normal log counts
        normal_entries = []
        for i in range(20):
            # Normal range: 8-12 entries per batch
            batch_size = 10 + (i % 5) - 2
            batch = [
                LogEntry(
                    level="INFO",
                    message=f"Normal message {j}",
                    logger_name="app.service",
                )
                for j in range(batch_size)
            ]
            normal_entries.extend(batch)
            detector.update_metrics(batch)

        # Generate anomalous spike
        anomalous_batch = [
            LogEntry(
                level="INFO", message=f"Spike message {j}", logger_name="app.service"
            )
            for j in range(50)  # Significant spike
        ]
        detector.update_metrics(anomalous_batch)

        # Check for detected anomalies
        anomalies = detector.detect_anomalies()
        assert len(anomalies) > 0

        # Should detect log count anomaly
        log_count_anomalies = [a for a in anomalies if "Log Count" in a["metric"]]
        assert len(log_count_anomalies) > 0


class TestTracingManager:
    """Test distributed tracing functionality."""

    def test_basic_tracing(self):
        """Test basic span creation and management."""
        tracer = TracingManager(
            service_name="test-service",
            sampling_rate=1.0,  # Sample all traces for testing
        )

        # Create span
        span = tracer.start_span("test-operation")
        assert span is not None
        assert span.operation_name == "test-operation"
        assert span.service_name == "test-service"

        # Add attributes and events
        span.set_attribute("test.key", "test.value")
        span.add_event("test.event", {"detail": "test detail"})

        # Finish span
        tracer.finish_span(span)

        assert span.end_time is not None
        assert span.duration_ms > 0

    def test_span_context_manager(self):
        """Test span context manager functionality."""
        tracer = TracingManager(service_name="test-service")

        with tracer.span("context-operation") as span:
            span.set_attribute("test", "value")
            time.sleep(0.01)  # Small delay

        assert span.end_time is not None
        assert span.duration_ms > 0

    def test_trace_context_injection(self):
        """Test trace context injection and extraction."""
        tracer = TracingManager(service_name="test-service")

        # Create parent span
        parent_span = tracer.start_span("parent-operation")

        # Inject context
        headers = {}
        tracer.inject_context(parent_span, headers)

        assert len(headers) > 0

        # Extract context and create child span
        extracted_context = tracer.extract_context(headers)
        child_span = tracer.start_span(
            "child-operation", parent_context=extracted_context
        )

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id


class TestObservabilityService:
    """Test the comprehensive observability service."""

    def test_service_initialization(self):
        """Test observability service initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ObservabilityConfig(
                logs_storage_path=Path(temp_dir) / "logs",
                metrics_storage_path=Path(temp_dir) / "metrics",
                enable_tracing=False,  # Disable for simpler testing
                enable_background_tasks=False,
                enable_alerts=False,
            )

            service = ObservabilityService(config)
            service.initialize()

            assert service._initialized is True
            assert service.logger is not None
            assert service.metrics_collector is not None
            assert service.log_aggregator is not None
            assert service.log_analyzer is not None

    def test_unified_logging(self):
        """Test unified logging through observability service."""
        config = ObservabilityConfig(
            enable_tracing=False,
            enable_background_tasks=False,
            enable_alerts=False,
            enable_log_aggregation=False,  # Simplify for testing
        )

        service = ObservabilityService(config)
        service.initialize()

        # Test logging
        service.log("info", "Test message", test_key="test_value")
        service.log("error", "Error message", error_code="E001")

        # Check metrics
        metrics = service.get_service_metrics()
        assert metrics["total_log_entries"] >= 2
        assert metrics["logger_status"] == "active"

    def test_unified_metrics(self):
        """Test unified metrics through observability service."""
        config = ObservabilityConfig(
            enable_tracing=False,
            enable_background_tasks=False,
            enable_alerts=False,
            enable_log_analysis=False,  # Simplify for testing
        )

        service = ObservabilityService(config)
        service.initialize()

        # Test metrics
        service.record_metric("test.counter", 1, "counter", service="test")
        service.record_metric("test.gauge", 42.5, "gauge", component="test")
        service.record_metric("test.histogram", 100, "histogram")
        service.record_metric("test.timer", 250, "timer")

        # Check service metrics
        metrics = service.get_service_metrics()
        assert metrics["metrics_collector_status"] == "active"

    def test_health_monitoring(self):
        """Test health status monitoring."""
        config = ObservabilityConfig(
            enable_tracing=False, enable_background_tasks=False, enable_alerts=False
        )

        service = ObservabilityService(config)
        service.initialize()

        # Check health status
        health = service.get_health_status()

        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["running"] is True
        assert "uptime" in health
        assert "timestamp" in health

    def test_service_shutdown(self):
        """Test proper service shutdown."""
        config = ObservabilityConfig(
            enable_tracing=False, enable_background_tasks=False, enable_alerts=False
        )

        service = ObservabilityService(config)
        service.initialize()

        assert service._running is True

        service.shutdown()

        assert service._running is False


class TestIntegration:
    """Integration tests for the complete logging infrastructure."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end logging workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ObservabilityConfig(
                logs_storage_path=Path(temp_dir) / "logs",
                metrics_storage_path=Path(temp_dir) / "metrics",
                enable_tracing=True,
                enable_background_tasks=False,  # Disable for deterministic testing
                enable_alerts=False,
                aggregation_interval=0,  # Disable background aggregation
            )

            service = ObservabilityService(config)
            service.initialize()

            # Simulate application workflow

            # 1. Start a trace
            trace_span = service.start_trace("user_request", user_id="test-user")

            # 2. Log application events
            service.log(
                "info",
                "User request started",
                user_id="test-user",
                endpoint="/api/detect",
            )
            service.log("info", "Loading dataset", dataset_id="test-dataset")
            service.log("info", "Running detection", algorithm="IsolationForest")

            # 3. Record metrics
            service.record_metric("request.duration", 150.5, "timer")
            service.record_metric("detection.samples", 1000, "gauge")
            service.record_metric("detection.anomalies", 25, "gauge")

            # 4. Complete trace
            if trace_span:
                service.tracer.finish_span(trace_span)

            # 5. Verify metrics and health
            metrics = service.get_service_metrics()
            health = service.get_health_status()

            assert metrics["total_log_entries"] >= 3
            assert health["status"] == "healthy"

            # 6. Clean shutdown
            service.shutdown()

    def test_container_integration(self):
        """Test integration with dependency injection container."""
        from pynomaly.infrastructure.config.container import Container

        container = Container()

        # Check if observability services are available
        if hasattr(container, "observability_service"):
            # Test that we can create the service through DI
            obs_service = container.observability_service()
            assert obs_service is not None

            if hasattr(obs_service, "initialize"):
                obs_service.initialize()
                assert obs_service._initialized is True
                obs_service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
