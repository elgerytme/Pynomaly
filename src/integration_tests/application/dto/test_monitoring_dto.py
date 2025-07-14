"""
Comprehensive tests for monitoring DTOs.

This module tests all monitoring-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including real-time metrics, alerts,
system health monitoring, and dashboard responses.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pynomaly.application.dto.monitoring_dto import (
    AlertLevel,
    AlertMessage,
    DetectionMetrics,
    MetricDataPoint,
    MetricSeries,
    MetricType,
    MonitoringDashboardRequest,
    MonitoringDashboardResponse,
    RealTimeUpdate,
    SystemHealth,
    SystemMetrics,
)


class TestMetricType:
    """Test suite for MetricType enum."""

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"
        assert MetricType.TIMER.value == "timer"

    def test_metric_type_completeness(self):
        """Test that all expected metric types are present."""
        expected_types = {"counter", "gauge", "histogram", "summary", "timer"}
        actual_types = {mt.value for mt in MetricType}
        assert actual_types == expected_types


class TestAlertLevel:
    """Test suite for AlertLevel enum."""

    def test_alert_level_values(self):
        """Test AlertLevel enum values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_alert_level_ordering(self):
        """Test alert level severity ordering."""
        levels = [
            AlertLevel.INFO,
            AlertLevel.WARNING,
            AlertLevel.ERROR,
            AlertLevel.CRITICAL,
        ]
        level_values = [level.value for level in levels]
        expected_order = ["info", "warning", "error", "critical"]
        assert level_values == expected_order


class TestSystemHealth:
    """Test suite for SystemHealth enum."""

    def test_system_health_values(self):
        """Test SystemHealth enum values."""
        assert SystemHealth.HEALTHY.value == "healthy"
        assert SystemHealth.DEGRADED.value == "degraded"
        assert SystemHealth.UNHEALTHY.value == "unhealthy"
        assert SystemHealth.CRITICAL.value == "critical"

    def test_system_health_states(self):
        """Test all system health states."""
        expected_states = {"healthy", "degraded", "unhealthy", "critical"}
        actual_states = {sh.value for sh in SystemHealth}
        assert actual_states == expected_states


class TestMetricDataPoint:
    """Test suite for MetricDataPoint."""

    def test_basic_creation(self):
        """Test basic metric data point creation."""
        timestamp = datetime.utcnow()
        point = MetricDataPoint(timestamp=timestamp, value=42.5)

        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.labels == {}  # Default

    def test_creation_with_labels(self):
        """Test metric data point creation with labels."""
        timestamp = datetime.utcnow()
        labels = {"service": "detection", "environment": "production"}

        point = MetricDataPoint(timestamp=timestamp, value=100.0, labels=labels)

        assert point.timestamp == timestamp
        assert point.value == 100.0
        assert point.labels == labels

    def test_to_dict(self):
        """Test metric data point to_dict conversion."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        labels = {"tag": "test"}
        point = MetricDataPoint(timestamp=timestamp, value=75.0, labels=labels)

        result = point.to_dict()
        expected = {
            "timestamp": "2024-01-01T12:00:00",
            "value": 75.0,
            "labels": {"tag": "test"},
        }

        assert result == expected

    def test_negative_values(self):
        """Test metric data point with negative values."""
        timestamp = datetime.utcnow()
        point = MetricDataPoint(timestamp=timestamp, value=-50.5)

        assert point.value == -50.5

    def test_zero_value(self):
        """Test metric data point with zero value."""
        timestamp = datetime.utcnow()
        point = MetricDataPoint(timestamp=timestamp, value=0.0)

        assert point.value == 0.0


class TestMetricSeries:
    """Test suite for MetricSeries."""

    def test_basic_creation(self):
        """Test basic metric series creation."""
        series = MetricSeries(
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
        )

        assert series.metric_name == "cpu_usage"
        assert series.metric_type == MetricType.GAUGE
        assert series.description == ""  # Default
        assert series.unit == ""  # Default
        assert series.data_points == []  # Default

    def test_complete_creation(self):
        """Test metric series creation with all fields."""
        data_points = [
            MetricDataPoint(timestamp=datetime.utcnow(), value=50.0),
            MetricDataPoint(timestamp=datetime.utcnow(), value=60.0),
        ]

        series = MetricSeries(
            metric_name="memory_usage",
            metric_type=MetricType.GAUGE,
            description="Memory usage percentage",
            unit="percent",
            data_points=data_points,
        )

        assert series.description == "Memory usage percentage"
        assert series.unit == "percent"
        assert len(series.data_points) == 2

    def test_add_data_point_default_timestamp(self):
        """Test adding data point with default timestamp."""
        series = MetricSeries(
            metric_name="test_metric",
            metric_type=MetricType.COUNTER,
        )

        series.add_data_point(value=100.0)

        assert len(series.data_points) == 1
        assert series.data_points[0].value == 100.0
        assert isinstance(series.data_points[0].timestamp, datetime)

    def test_add_data_point_custom_timestamp(self):
        """Test adding data point with custom timestamp."""
        series = MetricSeries(
            metric_name="test_metric",
            metric_type=MetricType.GAUGE,
        )

        custom_timestamp = datetime(2024, 1, 1, 10, 0, 0)
        labels = {"source": "test"}

        series.add_data_point(value=75.5, timestamp=custom_timestamp, labels=labels)

        assert len(series.data_points) == 1
        point = series.data_points[0]
        assert point.value == 75.5
        assert point.timestamp == custom_timestamp
        assert point.labels == labels

    def test_data_point_limit(self):
        """Test that data points are limited to 1000."""
        series = MetricSeries(
            metric_name="test_metric",
            metric_type=MetricType.COUNTER,
        )

        # Add more than 1000 data points
        for i in range(1100):
            series.add_data_point(value=float(i))

        # Should only keep last 1000
        assert len(series.data_points) == 1000
        assert series.data_points[0].value == 100.0  # First kept point
        assert series.data_points[-1].value == 1099.0  # Last point

    def test_get_latest_value_empty(self):
        """Test get_latest_value with empty series."""
        series = MetricSeries(
            metric_name="empty_metric",
            metric_type=MetricType.GAUGE,
        )

        assert series.get_latest_value() is None

    def test_get_latest_value_with_data(self):
        """Test get_latest_value with data."""
        series = MetricSeries(
            metric_name="test_metric",
            metric_type=MetricType.GAUGE,
        )

        series.add_data_point(value=10.0)
        series.add_data_point(value=20.0)
        series.add_data_point(value=30.0)

        assert series.get_latest_value() == 30.0

    def test_to_dict(self):
        """Test metric series to_dict conversion."""
        series = MetricSeries(
            metric_name="test_metric",
            metric_type=MetricType.HISTOGRAM,
            description="Test histogram",
            unit="ms",
        )

        series.add_data_point(value=100.0)
        series.add_data_point(value=150.0)

        result = series.to_dict()

        assert result["metric_name"] == "test_metric"
        assert result["metric_type"] == "histogram"
        assert result["description"] == "Test histogram"
        assert result["unit"] == "ms"
        assert len(result["data_points"]) == 2
        assert result["latest_value"] == 150.0


class TestSystemMetrics:
    """Test suite for SystemMetrics."""

    def test_default_creation(self):
        """Test default system metrics creation."""
        metrics = SystemMetrics()

        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_percent == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.network_io_bytes == 0.0
        assert metrics.active_connections == 0
        assert metrics.uptime_seconds == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_complete_creation(self):
        """Test system metrics creation with all values."""
        timestamp = datetime(2024, 1, 1, 15, 30, 0)

        metrics = SystemMetrics(
            cpu_usage_percent=75.5,
            memory_usage_percent=82.3,
            disk_usage_percent=45.1,
            network_io_bytes=1024000.0,
            active_connections=150,
            uptime_seconds=86400.0,
            timestamp=timestamp,
        )

        assert metrics.cpu_usage_percent == 75.5
        assert metrics.memory_usage_percent == 82.3
        assert metrics.disk_usage_percent == 45.1
        assert metrics.network_io_bytes == 1024000.0
        assert metrics.active_connections == 150
        assert metrics.uptime_seconds == 86400.0
        assert metrics.timestamp == timestamp

    def test_to_dict(self):
        """Test system metrics to_dict conversion."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metrics = SystemMetrics(
            cpu_usage_percent=50.0,
            memory_usage_percent=60.0,
            timestamp=timestamp,
        )

        result = metrics.to_dict()
        expected = {
            "cpu_usage_percent": 50.0,
            "memory_usage_percent": 60.0,
            "disk_usage_percent": 0.0,
            "network_io_bytes": 0.0,
            "active_connections": 0,
            "uptime_seconds": 0.0,
            "timestamp": "2024-01-01T12:00:00",
        }

        assert result == expected

    def test_high_usage_values(self):
        """Test system metrics with high usage values."""
        metrics = SystemMetrics(
            cpu_usage_percent=100.0,
            memory_usage_percent=99.9,
            disk_usage_percent=95.0,
        )

        assert metrics.cpu_usage_percent == 100.0
        assert metrics.memory_usage_percent == 99.9
        assert metrics.disk_usage_percent == 95.0


class TestDetectionMetrics:
    """Test suite for DetectionMetrics."""

    def test_default_creation(self):
        """Test default detection metrics creation."""
        metrics = DetectionMetrics()

        assert metrics.active_detections == 0
        assert metrics.total_detections_today == 0
        assert metrics.avg_detection_time_ms == 0.0
        assert metrics.anomaly_rate_percent == 0.0
        assert metrics.false_positive_rate_percent == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_complete_creation(self):
        """Test detection metrics creation with all values."""
        timestamp = datetime(2024, 1, 1, 16, 0, 0)

        metrics = DetectionMetrics(
            active_detections=25,
            total_detections_today=1500,
            avg_detection_time_ms=125.5,
            anomaly_rate_percent=5.2,
            false_positive_rate_percent=2.1,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            timestamp=timestamp,
        )

        assert metrics.active_detections == 25
        assert metrics.total_detections_today == 1500
        assert metrics.avg_detection_time_ms == 125.5
        assert metrics.anomaly_rate_percent == 5.2
        assert metrics.false_positive_rate_percent == 2.1
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.90
        assert metrics.timestamp == timestamp

    def test_to_dict(self):
        """Test detection metrics to_dict conversion."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metrics = DetectionMetrics(
            active_detections=10,
            precision=0.95,
            timestamp=timestamp,
        )

        result = metrics.to_dict()

        assert result["active_detections"] == 10
        assert result["precision"] == 0.95
        assert result["timestamp"] == "2024-01-01T12:00:00"
        assert "total_detections_today" in result
        assert "f1_score" in result

    def test_perfect_metrics(self):
        """Test detection metrics with perfect performance."""
        metrics = DetectionMetrics(
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            false_positive_rate_percent=0.0,
        )

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.false_positive_rate_percent == 0.0


class TestAlertMessage:
    """Test suite for AlertMessage."""

    def test_basic_creation(self):
        """Test basic alert message creation."""
        alert_id = uuid4()
        alert = AlertMessage(
            alert_id=alert_id,
            level=AlertLevel.WARNING,
            title="High CPU Usage",
            message="CPU usage has exceeded 80%",
            source="system_monitor",
        )

        assert alert.alert_id == alert_id
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "High CPU Usage"
        assert alert.message == "CPU usage has exceeded 80%"
        assert alert.source == "system_monitor"
        assert isinstance(alert.timestamp, datetime)
        assert alert.metadata == {}  # Default
        assert alert.acknowledged is False  # Default

    def test_complete_creation(self):
        """Test alert message creation with all fields."""
        alert_id = uuid4()
        timestamp = datetime(2024, 1, 1, 14, 30, 0)
        metadata = {"cpu_usage": 85.2, "threshold": 80.0}

        alert = AlertMessage(
            alert_id=alert_id,
            level=AlertLevel.CRITICAL,
            title="System Critical",
            message="Multiple thresholds exceeded",
            source="detection_service",
            timestamp=timestamp,
            metadata=metadata,
            acknowledged=True,
        )

        assert alert.level == AlertLevel.CRITICAL
        assert alert.timestamp == timestamp
        assert alert.metadata == metadata
        assert alert.acknowledged is True

    def test_to_dict(self):
        """Test alert message to_dict conversion."""
        alert_id = uuid4()
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metadata = {"value": 100}

        alert = AlertMessage(
            alert_id=alert_id,
            level=AlertLevel.ERROR,
            title="Test Alert",
            message="Test message",
            source="test",
            timestamp=timestamp,
            metadata=metadata,
            acknowledged=False,
        )

        result = alert.to_dict()
        expected = {
            "alert_id": str(alert_id),
            "level": "error",
            "title": "Test Alert",
            "message": "Test message",
            "source": "test",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {"value": 100},
            "acknowledged": False,
        }

        assert result == expected

    def test_different_alert_levels(self):
        """Test different alert levels."""
        alert_id = uuid4()
        levels = [
            AlertLevel.INFO,
            AlertLevel.WARNING,
            AlertLevel.ERROR,
            AlertLevel.CRITICAL,
        ]

        for level in levels:
            alert = AlertMessage(
                alert_id=alert_id,
                level=level,
                title=f"{level.value.title()} Alert",
                message=f"This is a {level.value} alert",
                source="test",
            )
            assert alert.level == level


class TestMonitoringDashboardRequest:
    """Test suite for MonitoringDashboardRequest."""

    def test_default_creation(self):
        """Test default monitoring dashboard request creation."""
        request = MonitoringDashboardRequest()

        assert request.time_range_minutes == 60  # Default
        assert request.include_system_metrics is True  # Default
        assert request.include_detection_metrics is True  # Default
        assert request.include_alerts is True  # Default
        assert request.metric_names == []  # Default

    def test_complete_creation(self):
        """Test monitoring dashboard request creation with all fields."""
        metric_names = ["cpu_usage", "memory_usage", "detection_rate"]

        request = MonitoringDashboardRequest(
            time_range_minutes=120,
            include_system_metrics=False,
            include_detection_metrics=True,
            include_alerts=False,
            metric_names=metric_names,
        )

        assert request.time_range_minutes == 120
        assert request.include_system_metrics is False
        assert request.include_detection_metrics is True
        assert request.include_alerts is False
        assert request.metric_names == metric_names

    def test_validate_valid_request(self):
        """Test validation of valid request."""
        request = MonitoringDashboardRequest(time_range_minutes=30)

        # Should not raise any exception
        request.validate()

    def test_validate_invalid_time_range_negative(self):
        """Test validation with negative time range."""
        request = MonitoringDashboardRequest(time_range_minutes=-10)

        with pytest.raises(ValueError, match="Time range must be positive"):
            request.validate()

    def test_validate_invalid_time_range_zero(self):
        """Test validation with zero time range."""
        request = MonitoringDashboardRequest(time_range_minutes=0)

        with pytest.raises(ValueError, match="Time range must be positive"):
            request.validate()

    def test_validate_invalid_time_range_too_large(self):
        """Test validation with time range exceeding 24 hours."""
        request = MonitoringDashboardRequest(time_range_minutes=1441)  # > 24 hours

        with pytest.raises(ValueError, match="Time range cannot exceed 24 hours"):
            request.validate()

    def test_validate_boundary_values(self):
        """Test validation with boundary values."""
        # Valid boundary values
        valid_requests = [
            MonitoringDashboardRequest(time_range_minutes=1),
            MonitoringDashboardRequest(time_range_minutes=1440),  # Exactly 24 hours
        ]

        for request in valid_requests:
            request.validate()  # Should not raise


class TestMonitoringDashboardResponse:
    """Test suite for MonitoringDashboardResponse."""

    def test_basic_creation(self):
        """Test basic monitoring dashboard response creation."""
        system_metrics = SystemMetrics(cpu_usage_percent=50.0)
        detection_metrics = DetectionMetrics(active_detections=10)

        response = MonitoringDashboardResponse(
            system_health=SystemHealth.HEALTHY,
            system_metrics=system_metrics,
            detection_metrics=detection_metrics,
        )

        assert response.system_health == SystemHealth.HEALTHY
        assert response.system_metrics == system_metrics
        assert response.detection_metrics == detection_metrics
        assert response.metric_series == []  # Default
        assert response.active_alerts == []  # Default
        assert isinstance(response.generated_at, datetime)

    def test_complete_creation(self):
        """Test monitoring dashboard response creation with all fields."""
        timestamp = datetime(2024, 1, 1, 18, 0, 0)

        system_metrics = SystemMetrics(
            cpu_usage_percent=75.0,
            memory_usage_percent=80.0,
        )

        detection_metrics = DetectionMetrics(
            active_detections=25,
            precision=0.92,
        )

        metric_series = [
            MetricSeries(
                metric_name="cpu_trend",
                metric_type=MetricType.GAUGE,
                description="CPU usage trend",
            )
        ]

        alerts = [
            AlertMessage(
                alert_id=uuid4(),
                level=AlertLevel.WARNING,
                title="High Memory",
                message="Memory usage is high",
                source="monitor",
            )
        ]

        response = MonitoringDashboardResponse(
            system_health=SystemHealth.DEGRADED,
            system_metrics=system_metrics,
            detection_metrics=detection_metrics,
            metric_series=metric_series,
            active_alerts=alerts,
            generated_at=timestamp,
        )

        assert response.system_health == SystemHealth.DEGRADED
        assert len(response.metric_series) == 1
        assert len(response.active_alerts) == 1
        assert response.generated_at == timestamp

    def test_to_dict(self):
        """Test monitoring dashboard response to_dict conversion."""
        system_metrics = SystemMetrics(cpu_usage_percent=60.0)
        detection_metrics = DetectionMetrics(active_detections=5)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        response = MonitoringDashboardResponse(
            system_health=SystemHealth.HEALTHY,
            system_metrics=system_metrics,
            detection_metrics=detection_metrics,
            generated_at=timestamp,
        )

        result = response.to_dict()

        assert result["system_health"] == "healthy"
        assert "system_metrics" in result
        assert "detection_metrics" in result
        assert result["metric_series"] == []
        assert result["active_alerts"] == []
        assert result["generated_at"] == "2024-01-01T12:00:00"


class TestRealTimeUpdate:
    """Test suite for RealTimeUpdate."""

    def test_basic_creation(self):
        """Test basic real-time update creation."""
        data = {"metric": "cpu_usage", "value": 75.0}

        update = RealTimeUpdate(
            update_type="metrics",
            data=data,
        )

        assert update.update_type == "metrics"
        assert update.data == data
        assert isinstance(update.timestamp, datetime)

    def test_complete_creation(self):
        """Test real-time update creation with all fields."""
        timestamp = datetime(2024, 1, 1, 20, 0, 0)
        data = {
            "alert_id": str(uuid4()),
            "level": "critical",
            "message": "System failure detected",
        }

        update = RealTimeUpdate(
            update_type="alert",
            data=data,
            timestamp=timestamp,
        )

        assert update.update_type == "alert"
        assert update.data == data
        assert update.timestamp == timestamp

    def test_to_dict(self):
        """Test real-time update to_dict conversion."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        data = {"health_status": "degraded"}

        update = RealTimeUpdate(
            update_type="health",
            data=data,
            timestamp=timestamp,
        )

        result = update.to_dict()
        expected = {
            "update_type": "health",
            "data": {"health_status": "degraded"},
            "timestamp": "2024-01-01T12:00:00",
        }

        assert result == expected

    def test_different_update_types(self):
        """Test different update types."""
        update_types = ["metrics", "alert", "health", "detection"]

        for update_type in update_types:
            update = RealTimeUpdate(
                update_type=update_type,
                data={"type": update_type},
            )
            assert update.update_type == update_type


class TestMonitoringIntegration:
    """Integration tests for monitoring DTOs."""

    def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow using multiple DTOs."""
        # Step 1: Create dashboard request
        request = MonitoringDashboardRequest(
            time_range_minutes=60,
            include_system_metrics=True,
            include_detection_metrics=True,
            include_alerts=True,
            metric_names=["cpu_usage", "memory_usage"],
        )

        # Step 2: Create system metrics
        system_metrics = SystemMetrics(
            cpu_usage_percent=78.5,
            memory_usage_percent=65.2,
            active_connections=125,
            uptime_seconds=86400.0,
        )

        # Step 3: Create detection metrics
        detection_metrics = DetectionMetrics(
            active_detections=15,
            total_detections_today=500,
            avg_detection_time_ms=145.2,
            precision=0.92,
            recall=0.89,
            f1_score=0.90,
        )

        # Step 4: Create metric series
        cpu_series = MetricSeries(
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
            description="CPU usage percentage",
            unit="percent",
        )

        # Add some historical data
        for i in range(10):
            cpu_series.add_data_point(
                value=70.0 + i, timestamp=datetime.utcnow() - timedelta(minutes=10 - i)
            )

        # Step 5: Create alerts
        critical_alert = AlertMessage(
            alert_id=uuid4(),
            level=AlertLevel.CRITICAL,
            title="High CPU Usage",
            message="CPU usage has been above 75% for 10 minutes",
            source="system_monitor",
            metadata={"threshold": 75.0, "current": 78.5},
        )

        # Step 6: Create dashboard response
        response = MonitoringDashboardResponse(
            system_health=SystemHealth.DEGRADED,
            system_metrics=system_metrics,
            detection_metrics=detection_metrics,
            metric_series=[cpu_series],
            active_alerts=[critical_alert],
        )

        # Step 7: Create real-time update
        update = RealTimeUpdate(
            update_type="alert",
            data=critical_alert.to_dict(),
        )

        # Verify workflow consistency
        assert response.system_metrics.cpu_usage_percent == 78.5
        assert len(response.active_alerts) == 1
        assert response.active_alerts[0].level == AlertLevel.CRITICAL
        assert len(response.metric_series) == 1
        assert response.metric_series[0].get_latest_value() == 79.0
        assert update.data["level"] == "critical"

    def test_metric_series_aggregation(self):
        """Test metric series data aggregation."""
        # Create multiple metric series
        cpu_series = MetricSeries(
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
            unit="percent",
        )

        memory_series = MetricSeries(
            metric_name="memory_usage",
            metric_type=MetricType.GAUGE,
            unit="percent",
        )

        detection_series = MetricSeries(
            metric_name="detection_rate",
            metric_type=MetricType.COUNTER,
            unit="per_minute",
        )

        # Add data points
        base_time = datetime.utcnow()
        for i in range(5):
            timestamp = base_time - timedelta(minutes=5 - i)

            cpu_series.add_data_point(value=50.0 + i * 5, timestamp=timestamp)
            memory_series.add_data_point(value=60.0 + i * 3, timestamp=timestamp)
            detection_series.add_data_point(value=float(i * 2), timestamp=timestamp)

        # Create dashboard with all series
        dashboard = MonitoringDashboardResponse(
            system_health=SystemHealth.HEALTHY,
            system_metrics=SystemMetrics(),
            detection_metrics=DetectionMetrics(),
            metric_series=[cpu_series, memory_series, detection_series],
        )

        # Verify aggregation
        assert len(dashboard.metric_series) == 3
        assert dashboard.metric_series[0].get_latest_value() == 70.0  # CPU
        assert dashboard.metric_series[1].get_latest_value() == 72.0  # Memory
        assert dashboard.metric_series[2].get_latest_value() == 8.0  # Detection

        # Test serialization
        dashboard_dict = dashboard.to_dict()
        assert len(dashboard_dict["metric_series"]) == 3
        assert all(
            "latest_value" in series for series in dashboard_dict["metric_series"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
