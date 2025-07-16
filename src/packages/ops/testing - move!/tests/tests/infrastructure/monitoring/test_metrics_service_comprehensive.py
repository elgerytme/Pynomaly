"""Comprehensive tests for metrics service monitoring infrastructure."""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

from monorepo.domain.models.monitoring import (
    AlertSeverity,
    AlertStatus,
    MetricType,
    ServiceStatus,
)
from monorepo.infrastructure.monitoring.metrics_service import MetricsService


class TestMetricsService:
    """Test metrics service functionality."""

    def test_metrics_service_initialization(self):
        """Test metrics service initialization."""
        service = MetricsService(service_name="test-service", service_version="2.0.0")

        assert service.service_name == "test-service"
        assert service.service_version == "2.0.0"
        assert service.metrics == {}
        assert service.alert_rules == {}
        assert service.active_alerts == {}
        assert service.health_checks == {}
        assert isinstance(service.service_status, ServiceStatus)
        assert service.service_status.service_name == "test-service"
        assert service.service_status.service_version == "2.0.0"
        assert service.monitoring_tasks == set()
        assert service.is_monitoring is False

    def test_metrics_service_default_initialization(self):
        """Test metrics service with default parameters."""
        service = MetricsService()

        assert service.service_name == "monorepo"
        assert service.service_version == "1.0.0"

    def test_prometheus_metrics_initialization(self):
        """Test Prometheus metrics initialization."""
        service = MetricsService()

        # Check that Prometheus metrics are created
        assert hasattr(service, "prom_request_count")
        assert hasattr(service, "prom_request_duration")
        assert hasattr(service, "prom_error_count")
        assert hasattr(service, "prom_model_predictions")
        assert hasattr(service, "prom_model_accuracy")
        assert hasattr(service, "prom_anomaly_detection_rate")
        assert hasattr(service, "prom_cpu_usage")
        assert hasattr(service, "prom_memory_usage")
        assert hasattr(service, "prom_disk_usage")

        # Check registry is configured
        assert service.prometheus_registry is not None

    def test_create_metric(self):
        """Test creating a metric."""
        service = MetricsService()

        metric = service.create_metric(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test metric for testing",
            unit="count",
            labels={"environment": "test", "service": "api"},
        )

        assert metric.name == "test_metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.description == "Test metric for testing"
        assert metric.unit == "count"
        assert metric.labels == {"environment": "test", "service": "api"}
        assert metric.metric_id is not None

        # Check metric is stored
        assert "test_metric" in service.metrics
        assert service.metrics["test_metric"] is metric

    def test_create_metric_duplicate(self):
        """Test creating a metric with duplicate name."""
        service = MetricsService()

        # Create first metric
        metric1 = service.create_metric(
            name="duplicate_metric",
            metric_type=MetricType.COUNTER,
            description="First metric",
        )

        # Try to create duplicate - should return existing
        metric2 = service.create_metric(
            name="duplicate_metric",
            metric_type=MetricType.GAUGE,
            description="Second metric",
        )

        assert metric1 is metric2
        assert len(service.metrics) == 1

    def test_record_metric_existing(self):
        """Test recording a metric value for existing metric."""
        service = MetricsService()

        # Create metric first
        metric = service.create_metric(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            description="Test counter",
        )

        # Record value
        service.record_metric(
            name="test_counter", value=42, labels={"status": "success"}
        )

        # Check metric was updated
        latest_value = metric.get_latest_value()
        assert latest_value == 42

        # Check labels were applied
        latest_point = metric.data_points[-1]
        assert latest_point.labels == {"status": "success"}

    def test_record_metric_auto_create(self):
        """Test recording a metric value with auto-creation."""
        service = MetricsService()

        # Record value for non-existent metric
        service.record_metric(
            name="auto_created_metric", value=3.14, labels={"type": "auto"}
        )

        # Check metric was auto-created
        assert "auto_created_metric" in service.metrics
        metric = service.metrics["auto_created_metric"]
        assert metric.metric_type == MetricType.GAUGE
        assert metric.description == "Auto-created metric: auto_created_metric"

        # Check value was recorded
        latest_value = metric.get_latest_value()
        assert latest_value == 3.14

    def test_record_metric_auto_create_string(self):
        """Test recording a string metric value with auto-creation."""
        service = MetricsService()

        # Record string value
        service.record_metric(
            name="string_metric",
            value="test_value",
        )

        # Check metric was auto-created as counter
        assert "string_metric" in service.metrics
        metric = service.metrics["string_metric"]
        assert metric.metric_type == MetricType.COUNTER

    def test_record_request_metrics(self):
        """Test recording HTTP request metrics."""
        service = MetricsService()

        # Record request metrics
        service.record_request_metrics(
            method="GET", endpoint="/api/v1/users", status_code=200, duration=0.125
        )

        # Check internal metrics were recorded
        assert "http_requests_total" in service.metrics
        assert "http_request_duration_seconds" in service.metrics

        # Check values
        requests_metric = service.metrics["http_requests_total"]
        assert requests_metric.get_latest_value() == 1

        duration_metric = service.metrics["http_request_duration_seconds"]
        assert duration_metric.get_latest_value() == 0.125

    def test_record_request_metrics_error(self):
        """Test recording HTTP request metrics with error status."""
        service = MetricsService()

        # Record error request
        service.record_request_metrics(
            method="POST", endpoint="/api/v1/users", status_code=500, duration=0.250
        )

        # Check error metrics were recorded
        assert "http_errors_total" in service.metrics

        error_metric = service.metrics["http_errors_total"]
        assert error_metric.get_latest_value() == 1

        # Check labels
        latest_point = error_metric.data_points[-1]
        assert latest_point.labels["status"] == "500"
        assert latest_point.labels["endpoint"] == "/api/v1/users"

    def test_record_model_metrics(self):
        """Test recording ML model metrics."""
        service = MetricsService()

        # Record model metrics
        service.record_model_metrics(
            model_name="anomaly_detector",
            model_version="1.2.0",
            prediction_count=50,
            accuracy=0.94,
            inference_time=0.023,
        )

        # Check internal metrics were recorded
        assert "model_predictions_total" in service.metrics
        assert "model_accuracy" in service.metrics
        assert "model_inference_duration_seconds" in service.metrics

        # Check values
        predictions_metric = service.metrics["model_predictions_total"]
        assert predictions_metric.get_latest_value() == 50

        accuracy_metric = service.metrics["model_accuracy"]
        assert accuracy_metric.get_latest_value() == 0.94

        inference_metric = service.metrics["model_inference_duration_seconds"]
        assert inference_metric.get_latest_value() == 0.023

    def test_record_model_metrics_minimal(self):
        """Test recording minimal ML model metrics."""
        service = MetricsService()

        # Record minimal model metrics
        service.record_model_metrics(
            model_name="simple_model",
            model_version="1.0.0",
        )

        # Check basic metrics were recorded
        assert "model_predictions_total" in service.metrics
        predictions_metric = service.metrics["model_predictions_total"]
        assert predictions_metric.get_latest_value() == 1

        # Check optional metrics were not recorded
        assert "model_accuracy" not in service.metrics
        assert "model_inference_duration_seconds" not in service.metrics

    def test_record_anomaly_detection_metrics(self):
        """Test recording anomaly detection metrics."""
        service = MetricsService()

        # Record anomaly detection metrics
        service.record_anomaly_detection_metrics(
            detector_type="isolation_forest",
            anomalies_detected=15,
            total_samples=1000,
            detection_accuracy=0.87,
        )

        # Check internal metrics were recorded
        assert "anomalies_detected_total" in service.metrics
        assert "anomaly_detection_rate" in service.metrics
        assert "anomaly_detection_accuracy" in service.metrics

        # Check values
        anomalies_metric = service.metrics["anomalies_detected_total"]
        assert anomalies_metric.get_latest_value() == 15

        rate_metric = service.metrics["anomaly_detection_rate"]
        assert rate_metric.get_latest_value() == 1.5  # 15/1000 * 100

        accuracy_metric = service.metrics["anomaly_detection_accuracy"]
        assert accuracy_metric.get_latest_value() == 0.87

    def test_record_anomaly_detection_metrics_zero_samples(self):
        """Test recording anomaly detection metrics with zero samples."""
        service = MetricsService()

        # Record with zero samples
        service.record_anomaly_detection_metrics(
            detector_type="test_detector",
            anomalies_detected=5,
            total_samples=0,
        )

        # Check rate calculation handles zero division
        rate_metric = service.metrics["anomaly_detection_rate"]
        assert rate_metric.get_latest_value() == 500.0  # 5/1 * 100

    def test_create_alert_rule(self):
        """Test creating an alert rule."""
        service = MetricsService()

        # Create alert rule
        rule = service.create_alert_rule(
            name="High CPU Alert",
            metric_name="cpu_usage_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            evaluation_window=timedelta(minutes=5),
        )

        assert rule.name == "High CPU Alert"
        assert rule.metric_name == "cpu_usage_percent"
        assert rule.condition == "greater_than"
        assert rule.threshold == 80.0
        assert rule.severity == AlertSeverity.WARNING
        assert rule.evaluation_window == timedelta(minutes=5)
        assert rule.rule_id is not None

        # Check rule is stored
        assert rule.rule_id in service.alert_rules
        assert service.alert_rules[rule.rule_id] is rule

    def test_create_alert_rule_defaults(self):
        """Test creating an alert rule with default values."""
        service = MetricsService()

        # Create alert rule with minimal parameters
        rule = service.create_alert_rule(
            name="Memory Alert",
            metric_name="memory_usage_percent",
            condition="greater_than",
            threshold=90.0,
        )

        assert rule.severity == AlertSeverity.WARNING
        assert rule.evaluation_window == timedelta(minutes=5)

    def test_add_health_check(self):
        """Test adding a health check."""
        service = MetricsService()

        # Add health check
        health_check = service.add_health_check(
            name="Database Health",
            check_type="database",
            target="postgresql://localhost:5432/pynomaly",
            timeout=30,
            interval=60,
        )

        assert health_check.name == "Database Health"
        assert health_check.check_type == "database"
        assert health_check.target == "postgresql://localhost:5432/pynomaly"
        assert health_check.timeout == 30
        assert health_check.interval == 60
        assert health_check.check_id is not None

        # Check health check is stored
        assert health_check.check_id in service.health_checks
        assert service.health_checks[health_check.check_id] is health_check

        # Check it was added to service status
        assert len(service.service_status.health_checks) == 1

    def test_add_health_check_defaults(self):
        """Test adding a health check with default values."""
        service = MetricsService()

        # Add health check with minimal parameters
        health_check = service.add_health_check(
            name="API Health",
            check_type="http",
            target="http://localhost:8000/health",
        )

        assert health_check.timeout == 30
        assert health_check.interval == 60

    def test_get_metric_value_latest(self):
        """Test getting latest metric value."""
        service = MetricsService()

        # Create and record metric
        service.record_metric("test_metric", 42)
        service.record_metric("test_metric", 84)

        # Get latest value
        value = asyncio.run(service.get_metric_value("test_metric", "latest"))
        assert value == 84

    def test_get_metric_value_average(self):
        """Test getting average metric value."""
        service = MetricsService()

        # Create and record metric
        service.record_metric("test_metric", 10)
        service.record_metric("test_metric", 20)
        service.record_metric("test_metric", 30)

        # Get average value
        value = asyncio.run(service.get_metric_value("test_metric", "average"))
        assert value == 20.0

    def test_get_metric_value_average_with_time_range(self):
        """Test getting average metric value with time range."""
        service = MetricsService()

        # Create metric
        metric = service.create_metric("test_metric", MetricType.GAUGE, "Test")

        # Add data points with specific timestamps
        now = datetime.utcnow()

        # Old data point (outside time range)
        old_point = type(
            "DataPoint",
            (),
            {"value": 100, "timestamp": now - timedelta(hours=2), "labels": {}},
        )()
        metric.data_points.append(old_point)

        # Recent data points (within time range)
        for i, value in enumerate([10, 20, 30]):
            recent_point = type(
                "DataPoint",
                (),
                {"value": value, "timestamp": now - timedelta(minutes=i), "labels": {}},
            )()
            metric.data_points.append(recent_point)

        # Get average for last hour
        value = asyncio.run(
            service.get_metric_value(
                "test_metric", "average", time_range=timedelta(hours=1)
            )
        )
        assert value == 20.0  # Should exclude the old value (100)

    def test_get_metric_value_sum(self):
        """Test getting sum metric value."""
        service = MetricsService()

        # Create and record metric
        service.record_metric("test_metric", 10)
        service.record_metric("test_metric", 20)
        service.record_metric("test_metric", 30)

        # Get sum value
        value = asyncio.run(
            service.get_metric_value(
                "test_metric", "sum", time_range=timedelta(hours=1)
            )
        )
        assert value == 60

    def test_get_metric_value_nonexistent(self):
        """Test getting value for non-existent metric."""
        service = MetricsService()

        # Get value for non-existent metric
        value = asyncio.run(service.get_metric_value("nonexistent", "latest"))
        assert value is None

    def test_get_service_health(self):
        """Test getting service health status."""
        service = MetricsService()

        # Mock _update_service_status
        with patch.object(service, "_update_service_status") as mock_update:
            health = asyncio.run(service.get_service_health())

            # Check that update was called
            mock_update.assert_called_once()

            # Check that service status summary is returned
            assert isinstance(health, dict)
            assert "service_name" in health
            assert "service_version" in health

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        service = MetricsService()

        # Add some metrics
        service.record_metric("http_requests_total", 100)
        service.record_metric("cpu_usage_percent", 45.0)
        service.record_metric("memory_usage_percent", 60.0)

        # Add some alerts
        alert_rule = service.create_alert_rule(
            name="Test Alert",
            metric_name="cpu_usage_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
        )

        # Add health check
        health_check = service.add_health_check(
            name="Test Health", check_type="http", target="http://localhost:8000/health"
        )

        # Get summary
        summary = asyncio.run(service.get_metrics_summary())

        assert isinstance(summary, dict)
        assert "timestamp" in summary
        assert "time_range_hours" in summary
        assert "metrics" in summary
        assert "alerts" in summary
        assert "health_checks" in summary

        # Check alerts summary
        assert summary["alerts"]["active"] == 0
        assert summary["alerts"]["critical"] == 0
        assert summary["alerts"]["warning"] == 0

        # Check health checks summary
        assert summary["health_checks"]["total"] == 1
        assert summary["health_checks"]["healthy"] == 1
        assert summary["health_checks"]["unhealthy"] == 0

        # Check metrics summary
        assert "http_requests_total" in summary["metrics"]
        assert summary["metrics"]["http_requests_total"]["latest"] == 100

    def test_get_metrics_summary_custom_time_range(self):
        """Test getting metrics summary with custom time range."""
        service = MetricsService()

        # Add metrics
        service.record_metric("test_metric", 42)

        # Get summary with custom time range
        summary = asyncio.run(
            service.get_metrics_summary(time_range=timedelta(hours=6))
        )

        assert summary["time_range_hours"] == 6.0

    def test_start_monitoring(self):
        """Test starting monitoring tasks."""
        service = MetricsService()

        # Mock asyncio.create_task to track tasks
        with patch("asyncio.create_task") as mock_create_task:
            # Create mock tasks
            mock_tasks = [Mock() for _ in range(4)]
            mock_create_task.side_effect = mock_tasks

            # Start monitoring
            asyncio.run(service.start_monitoring())

            # Check that tasks were created
            assert mock_create_task.call_count == 4
            assert service.is_monitoring is True
            assert len(service.monitoring_tasks) == 4

    def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running."""
        service = MetricsService()
        service.is_monitoring = True

        # Start monitoring again
        asyncio.run(service.start_monitoring())

        # Should not create new tasks
        assert len(service.monitoring_tasks) == 0

    def test_stop_monitoring(self):
        """Test stopping monitoring tasks."""
        service = MetricsService()

        # Create mock tasks
        mock_tasks = []
        for _ in range(3):
            mock_task = Mock()
            mock_task.cancel = Mock()
            mock_tasks.append(mock_task)

        service.monitoring_tasks = set(mock_tasks)
        service.is_monitoring = True

        # Mock asyncio.gather
        with patch("asyncio.gather") as mock_gather:
            mock_gather.return_value = asyncio.Future()
            mock_gather.return_value.set_result([])

            # Stop monitoring
            asyncio.run(service.stop_monitoring())

            # Check that tasks were cancelled
            for task in mock_tasks:
                task.cancel.assert_called_once()

            # Check that gather was called
            mock_gather.assert_called_once()

            # Check state
            assert service.is_monitoring is False
            assert len(service.monitoring_tasks) == 0

    @patch("monorepo.infrastructure.monitoring.metrics_service.psutil")
    def test_system_metrics_collector(self, mock_psutil):
        """Test system metrics collector task."""
        service = MetricsService()

        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=60.0,
            used=1024 * 1024 * 1024,  # 1GB
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=500 * 1024 * 1024 * 1024,  # 500GB
            total=1000 * 1024 * 1024 * 1024,  # 1TB
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1024,
            bytes_recv=2048,
        )
        mock_psutil.Process.return_value = Mock(
            cpu_percent=Mock(return_value=25.0),
            memory_info=Mock(return_value=Mock(rss=256 * 1024 * 1024)),
            num_threads=Mock(return_value=10),
        )

        # Set monitoring flag
        service.is_monitoring = True

        # Run collector once
        async def run_once():
            # Override the while loop to run only once
            original_method = service._system_metrics_collector

            async def single_run():
                await original_method.__code__.co_consts[1]  # Get the try block
                service.is_monitoring = False  # Stop after one iteration
                return

            # Mock sleep to avoid waiting
            with patch("asyncio.sleep"):
                await service._system_metrics_collector()

        # Mock sleep to avoid waiting
        with patch("asyncio.sleep") as mock_sleep:
            # Make it run once then stop
            def stop_monitoring(*args):
                service.is_monitoring = False

            mock_sleep.side_effect = stop_monitoring

            # Run the collector
            asyncio.run(service._system_metrics_collector())

            # Check that metrics were recorded
            assert "cpu_usage_percent" in service.metrics
            assert "memory_usage_percent" in service.metrics
            assert "disk_usage_percent" in service.metrics
            assert "network_bytes_sent" in service.metrics
            assert "network_bytes_recv" in service.metrics
            assert "process_cpu_percent" in service.metrics
            assert "process_memory_rss" in service.metrics
            assert "process_num_threads" in service.metrics

            # Check service status was updated
            assert service.service_status.cpu_usage == 45.0
            assert service.service_status.memory_usage == 60.0
            assert service.service_status.disk_usage == 50.0

    @patch("monorepo.infrastructure.monitoring.metrics_service.psutil")
    def test_system_metrics_collector_exception(self, mock_psutil):
        """Test system metrics collector with exception."""
        service = MetricsService()

        # Mock psutil to raise exception
        mock_psutil.cpu_percent.side_effect = Exception("System error")

        # Set monitoring flag
        service.is_monitoring = True

        # Mock sleep to avoid waiting
        with patch("asyncio.sleep") as mock_sleep:
            # Make it run once then stop
            def stop_monitoring(*args):
                service.is_monitoring = False

            mock_sleep.side_effect = stop_monitoring

            # Run the collector - should not raise exception
            asyncio.run(service._system_metrics_collector())

            # Check that it handled the exception gracefully
            assert service.is_monitoring is False

    def test_health_check_runner(self):
        """Test health check runner task."""
        service = MetricsService()

        # Add a health check
        health_check = service.add_health_check(
            name="Test Health",
            check_type="http",
            target="http://localhost:8000/health",
            interval=1,  # 1 second interval
        )

        # Set monitoring flag
        service.is_monitoring = True

        # Mock _run_health_check
        with patch.object(service, "_run_health_check") as mock_run_check:
            # Mock sleep to avoid waiting
            with patch("asyncio.sleep") as mock_sleep:
                # Make it run once then stop
                def stop_monitoring(*args):
                    service.is_monitoring = False

                mock_sleep.side_effect = stop_monitoring

                # Run the runner
                asyncio.run(service._health_check_runner())

                # Check that health check was run
                mock_run_check.assert_called_once_with(health_check)

    def test_health_check_runner_skip_disabled(self):
        """Test health check runner skips disabled checks."""
        service = MetricsService()

        # Add a disabled health check
        health_check = service.add_health_check(
            name="Disabled Health",
            check_type="http",
            target="http://localhost:8000/health",
        )
        health_check.enabled = False

        # Set monitoring flag
        service.is_monitoring = True

        # Mock _run_health_check
        with patch.object(service, "_run_health_check") as mock_run_check:
            # Mock sleep to avoid waiting
            with patch("asyncio.sleep") as mock_sleep:
                # Make it run once then stop
                def stop_monitoring(*args):
                    service.is_monitoring = False

                mock_sleep.side_effect = stop_monitoring

                # Run the runner
                asyncio.run(service._health_check_runner())

                # Check that health check was not run
                mock_run_check.assert_not_called()

    def test_health_check_runner_skip_recent(self):
        """Test health check runner skips recently run checks."""
        service = MetricsService()

        # Add a health check
        health_check = service.add_health_check(
            name="Recent Health",
            check_type="http",
            target="http://localhost:8000/health",
            interval=60,  # 60 second interval
        )

        # Set last check time to recent
        health_check.last_check_at = datetime.utcnow() - timedelta(seconds=30)

        # Set monitoring flag
        service.is_monitoring = True

        # Mock _run_health_check
        with patch.object(service, "_run_health_check") as mock_run_check:
            # Mock sleep to avoid waiting
            with patch("asyncio.sleep") as mock_sleep:
                # Make it run once then stop
                def stop_monitoring(*args):
                    service.is_monitoring = False

                mock_sleep.side_effect = stop_monitoring

                # Run the runner
                asyncio.run(service._health_check_runner())

                # Check that health check was not run
                mock_run_check.assert_not_called()

    @patch("monorepo.infrastructure.monitoring.metrics_service.requests")
    def test_run_health_check_http_success(self, mock_requests):
        """Test running HTTP health check successfully."""
        service = MetricsService()

        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"OK"
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        # Create health check
        health_check = service.add_health_check(
            name="HTTP Health",
            check_type="http",
            target="http://localhost:8000/health",
            timeout=5,
        )

        # Run health check
        asyncio.run(service._run_health_check(health_check))

        # Check that HTTP request was made
        mock_requests.get.assert_called_once_with(
            "http://localhost:8000/health", timeout=5
        )

        # Check that health check was marked as successful
        assert health_check.is_healthy is True
        assert health_check.last_result["status_code"] == 200

    @patch("monorepo.infrastructure.monitoring.metrics_service.requests")
    def test_run_health_check_http_failure(self, mock_requests):
        """Test running HTTP health check with failure."""
        service = MetricsService()

        # Mock failed HTTP response
        mock_requests.get.side_effect = Exception("Connection refused")

        # Create health check
        health_check = service.add_health_check(
            name="HTTP Health",
            check_type="http",
            target="http://localhost:8000/health",
            timeout=5,
        )

        # Run health check
        asyncio.run(service._run_health_check(health_check))

        # Check that health check was marked as failed
        assert health_check.is_healthy is False
        assert health_check.failure_count > 0

    def test_run_health_check_tcp_success(self):
        """Test running TCP health check successfully."""
        service = MetricsService()

        # Mock successful socket connection
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 0  # Success
            mock_sock.close = Mock()
            mock_socket.return_value = mock_sock

            # Create health check
            health_check = service.add_health_check(
                name="TCP Health", check_type="tcp", target="localhost:5432", timeout=5
            )

            # Run health check
            asyncio.run(service._run_health_check(health_check))

            # Check that socket connection was attempted
            mock_sock.connect_ex.assert_called_once_with(("localhost", 5432))
            mock_sock.close.assert_called_once()

            # Check that health check was marked as successful
            assert health_check.is_healthy is True
            assert health_check.last_result["tcp_connect"] == "success"

    def test_run_health_check_tcp_failure(self):
        """Test running TCP health check with failure."""
        service = MetricsService()

        # Mock failed socket connection
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 1  # Failure
            mock_sock.close = Mock()
            mock_socket.return_value = mock_sock

            # Create health check
            health_check = service.add_health_check(
                name="TCP Health", check_type="tcp", target="localhost:5432", timeout=5
            )

            # Run health check
            asyncio.run(service._run_health_check(health_check))

            # Check that health check was marked as failed
            assert health_check.is_healthy is False
            assert health_check.failure_count > 0

    def test_run_health_check_database_success(self):
        """Test running database health check successfully."""
        service = MetricsService()

        # Create health check
        health_check = service.add_health_check(
            name="Database Health",
            check_type="database",
            target="postgresql://localhost:5432/db",
            timeout=5,
        )

        # Run health check (simulated success)
        asyncio.run(service._run_health_check(health_check))

        # Check that health check was marked as successful
        assert health_check.is_healthy is True
        assert health_check.last_result["database"] == "connected"

    def test_alert_evaluator(self):
        """Test alert evaluator task."""
        service = MetricsService()

        # Create alert rule
        rule = service.create_alert_rule(
            name="Test Alert",
            metric_name="test_metric",
            condition="greater_than",
            threshold=80.0,
        )

        # Set monitoring flag
        service.is_monitoring = True

        # Mock _evaluate_alert_rule
        with patch.object(service, "_evaluate_alert_rule") as mock_evaluate:
            # Mock sleep to avoid waiting
            with patch("asyncio.sleep") as mock_sleep:
                # Make it run once then stop
                def stop_monitoring(*args):
                    service.is_monitoring = False

                mock_sleep.side_effect = stop_monitoring

                # Run the evaluator
                asyncio.run(service._alert_evaluator())

                # Check that alert rule was evaluated
                mock_evaluate.assert_called_once_with(rule)

    def test_alert_evaluator_skip_disabled(self):
        """Test alert evaluator skips disabled rules."""
        service = MetricsService()

        # Create disabled alert rule
        rule = service.create_alert_rule(
            name="Disabled Alert",
            metric_name="test_metric",
            condition="greater_than",
            threshold=80.0,
        )
        rule.is_enabled = False

        # Set monitoring flag
        service.is_monitoring = True

        # Mock _evaluate_alert_rule
        with patch.object(service, "_evaluate_alert_rule") as mock_evaluate:
            # Mock sleep to avoid waiting
            with patch("asyncio.sleep") as mock_sleep:
                # Make it run once then stop
                def stop_monitoring(*args):
                    service.is_monitoring = False

                mock_sleep.side_effect = stop_monitoring

                # Run the evaluator
                asyncio.run(service._alert_evaluator())

                # Check that alert rule was not evaluated
                mock_evaluate.assert_not_called()

    def test_evaluate_alert_rule_no_metric(self):
        """Test evaluating alert rule when metric doesn't exist."""
        service = MetricsService()

        # Create alert rule for non-existent metric
        rule = service.create_alert_rule(
            name="Non-existent Metric Alert",
            metric_name="nonexistent_metric",
            condition="greater_than",
            threshold=80.0,
        )

        # Evaluate rule
        asyncio.run(service._evaluate_alert_rule(rule))

        # Should not create any alerts
        assert len(service.active_alerts) == 0

    def test_evaluate_alert_rule_trigger_alert(self):
        """Test evaluating alert rule that triggers alert."""
        service = MetricsService()

        # Create metric with high value
        service.record_metric("cpu_usage", 90.0)

        # Create alert rule
        rule = service.create_alert_rule(
            name="High CPU Alert",
            metric_name="cpu_usage",
            condition="greater_than",
            threshold=80.0,
        )

        # Mock rule evaluation to return True
        with patch.object(rule, "evaluate", return_value=True):
            # Mock _send_alert_notification
            with patch.object(service, "_send_alert_notification") as mock_send:
                # Evaluate rule
                asyncio.run(service._evaluate_alert_rule(rule))

                # Check that alert was created
                assert len(service.active_alerts) == 1
                alert = list(service.active_alerts.values())[0]
                assert alert.rule_id == rule.rule_id
                assert alert.status == AlertStatus.ACTIVE

                # Check that notification was sent
                mock_send.assert_called_once_with(alert)

                # Check service status was updated
                assert service.service_status.active_alerts == 1

    def test_evaluate_alert_rule_resolve_alert(self):
        """Test evaluating alert rule that resolves alert."""
        service = MetricsService()

        # Create metric with low value
        service.record_metric("cpu_usage", 50.0)

        # Create alert rule
        rule = service.create_alert_rule(
            name="High CPU Alert",
            metric_name="cpu_usage",
            condition="greater_than",
            threshold=80.0,
        )

        # Create existing active alert
        from monorepo.domain.models.monitoring import Alert

        alert = Alert(
            alert_id=uuid4(),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            metric_name=rule.metric_name,
            metric_value=90.0,
            threshold=rule.threshold,
            severity=rule.severity,
            message="CPU usage is high",
        )
        service.active_alerts[alert.alert_id] = alert
        service.service_status.active_alerts = 1

        # Mock rule evaluation to return False
        with patch.object(rule, "evaluate", return_value=False):
            # Evaluate rule
            asyncio.run(service._evaluate_alert_rule(rule))

            # Check that alert was resolved
            assert alert.status == AlertStatus.RESOLVED

            # Check service status was updated
            assert service.service_status.active_alerts == 0

    def test_send_alert_notification(self):
        """Test sending alert notification."""
        service = MetricsService()

        # Create mock alert
        from monorepo.domain.models.monitoring import Alert

        alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Test Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Test alert message",
        )

        # Send notification (currently just logs)
        asyncio.run(service._send_alert_notification(alert))

        # No exception should be raised

    def test_metrics_cleanup(self):
        """Test metrics cleanup task."""
        service = MetricsService()

        # Create metric with old data
        metric = service.create_metric("test_metric", MetricType.GAUGE, "Test")

        # Add old data points
        now = datetime.utcnow()
        old_point = type(
            "DataPoint",
            (),
            {"value": 100, "timestamp": now - timedelta(days=8), "labels": {}},
        )()
        recent_point = type(
            "DataPoint",
            (),
            {"value": 200, "timestamp": now - timedelta(days=1), "labels": {}},
        )()

        metric.data_points = [old_point, recent_point]

        # Set monitoring flag
        service.is_monitoring = True

        # Mock sleep to avoid waiting
        with patch("asyncio.sleep") as mock_sleep:
            # Make it run once then stop
            def stop_monitoring(*args):
                service.is_monitoring = False

            mock_sleep.side_effect = stop_monitoring

            # Run cleanup
            asyncio.run(service._metrics_cleanup())

            # Check that old data was removed
            assert len(metric.data_points) == 1
            assert metric.data_points[0].value == 200

    def test_metrics_cleanup_resolved_alerts(self):
        """Test cleanup of old resolved alerts."""
        service = MetricsService()

        # Create old resolved alert
        from monorepo.domain.models.monitoring import Alert

        old_alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Old Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Old alert message",
        )
        old_alert.status = AlertStatus.RESOLVED
        old_alert.resolved_at = datetime.utcnow() - timedelta(days=31)

        # Create recent resolved alert
        recent_alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Recent Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Recent alert message",
        )
        recent_alert.status = AlertStatus.RESOLVED
        recent_alert.resolved_at = datetime.utcnow() - timedelta(days=1)

        service.active_alerts[old_alert.alert_id] = old_alert
        service.active_alerts[recent_alert.alert_id] = recent_alert

        # Set monitoring flag
        service.is_monitoring = True

        # Mock sleep to avoid waiting
        with patch("asyncio.sleep") as mock_sleep:
            # Make it run once then stop
            def stop_monitoring(*args):
                service.is_monitoring = False

            mock_sleep.side_effect = stop_monitoring

            # Run cleanup
            asyncio.run(service._metrics_cleanup())

            # Check that old alert was removed
            assert old_alert.alert_id not in service.active_alerts
            assert recent_alert.alert_id in service.active_alerts

    def test_get_prometheus_metrics(self):
        """Test getting Prometheus formatted metrics."""
        service = MetricsService()

        # Record some metrics to generate Prometheus output
        service.record_request_metrics("GET", "/api/users", 200, 0.150)
        service.record_model_metrics("test_model", "1.0.0", accuracy=0.95)

        # Get Prometheus metrics
        metrics_output = service.get_prometheus_metrics()

        assert isinstance(metrics_output, str)
        assert "pynomaly_requests_total" in metrics_output
        assert "pynomaly_model_accuracy" in metrics_output

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        service = MetricsService()

        # Create alerts with different statuses
        from monorepo.domain.models.monitoring import Alert

        active_alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Active Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Active alert",
        )
        active_alert.status = AlertStatus.ACTIVE

        resolved_alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Resolved Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Resolved alert",
        )
        resolved_alert.status = AlertStatus.RESOLVED

        service.active_alerts[active_alert.alert_id] = active_alert
        service.active_alerts[resolved_alert.alert_id] = resolved_alert

        # Get active alerts
        active_alerts = service.get_active_alerts()

        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == active_alert.alert_id

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        service = MetricsService()

        # Create alert
        from monorepo.domain.models.monitoring import Alert

        alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Test Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )
        service.active_alerts[alert.alert_id] = alert

        # Acknowledge alert
        acknowledged_by = uuid4()
        result = service.acknowledge_alert(alert.alert_id, acknowledged_by)

        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == acknowledged_by

    def test_acknowledge_alert_not_found(self):
        """Test acknowledging non-existent alert."""
        service = MetricsService()

        # Try to acknowledge non-existent alert
        result = service.acknowledge_alert(uuid4(), uuid4())

        assert result is False

    def test_resolve_alert(self):
        """Test resolving an alert."""
        service = MetricsService()

        # Create alert
        from monorepo.domain.models.monitoring import Alert

        alert = Alert(
            alert_id=uuid4(),
            rule_id=uuid4(),
            rule_name="Test Alert",
            metric_name="test_metric",
            metric_value=90.0,
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )
        service.active_alerts[alert.alert_id] = alert
        service.service_status.active_alerts = 1

        # Resolve alert
        resolved_by = uuid4()
        result = service.resolve_alert(alert.alert_id, resolved_by, "Fixed the issue")

        assert result is True
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_by == resolved_by
        assert alert.resolution_note == "Fixed the issue"
        assert service.service_status.active_alerts == 0

    def test_resolve_alert_not_found(self):
        """Test resolving non-existent alert."""
        service = MetricsService()

        # Try to resolve non-existent alert
        result = service.resolve_alert(uuid4(), uuid4(), "Note")

        assert result is False

    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording."""
        service = MetricsService()

        async def record_metrics():
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    asyncio.to_thread(service.record_metric, f"metric_{i}", i * 10)
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

        # Record metrics concurrently
        asyncio.run(record_metrics())

        # Check that all metrics were recorded
        assert len(service.metrics) == 10
        for i in range(10):
            assert f"metric_{i}" in service.metrics
            assert service.metrics[f"metric_{i}"].get_latest_value() == i * 10

    def test_metrics_service_performance(self):
        """Test metrics service performance under load."""
        service = MetricsService()

        # Record many metrics quickly
        start_time = time.time()

        for i in range(1000):
            service.record_metric("load_test_metric", i)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly
        assert duration < 1.0

        # Check metric was recorded
        assert "load_test_metric" in service.metrics
        assert service.metrics["load_test_metric"].get_latest_value() == 999
