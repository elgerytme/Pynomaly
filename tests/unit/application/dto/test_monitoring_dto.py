"""Tests for monitoring DTOs."""

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


class TestMetricDataPoint:
    """Test MetricDataPoint."""
    
    def test_creation(self):
        """Test basic creation."""
        now = datetime.utcnow()
        point = MetricDataPoint(timestamp=now, value=42.0, labels={"env": "test"})
        
        assert point.timestamp == now
        assert point.value == 42.0
        assert point.labels == {"env": "test"}
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        now = datetime.utcnow()
        point = MetricDataPoint(timestamp=now, value=42.0, labels={"env": "test"})
        
        result = point.to_dict()
        
        assert result["timestamp"] == now.isoformat()
        assert result["value"] == 42.0
        assert result["labels"] == {"env": "test"}


class TestMetricSeries:
    """Test MetricSeries."""
    
    def test_creation(self):
        """Test basic creation."""
        series = MetricSeries(
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
            description="CPU usage percentage",
            unit="percent"
        )
        
        assert series.metric_name == "cpu_usage"
        assert series.metric_type == MetricType.GAUGE
        assert series.description == "CPU usage percentage"
        assert series.unit == "percent"
        assert len(series.data_points) == 0
    
    def test_add_data_point(self):
        """Test adding data points."""
        series = MetricSeries("cpu_usage", MetricType.GAUGE)
        
        # Add point without timestamp
        series.add_data_point(50.0)
        assert len(series.data_points) == 1
        assert series.data_points[0].value == 50.0
        
        # Add point with timestamp
        now = datetime.utcnow()
        series.add_data_point(60.0, timestamp=now, labels={"host": "server1"})
        assert len(series.data_points) == 2
        assert series.data_points[1].value == 60.0
        assert series.data_points[1].timestamp == now
        assert series.data_points[1].labels == {"host": "server1"}
    
    def test_get_latest_value(self):
        """Test getting latest value."""
        series = MetricSeries("cpu_usage", MetricType.GAUGE)
        
        # No data points
        assert series.get_latest_value() is None
        
        # Add data points
        series.add_data_point(50.0)
        series.add_data_point(60.0)
        
        assert series.get_latest_value() == 60.0
    
    def test_memory_management(self):
        """Test that series keeps only last 1000 points."""
        series = MetricSeries("test_metric", MetricType.COUNTER)
        
        # Add 1500 points
        for i in range(1500):
            series.add_data_point(float(i))
        
        # Should keep only last 1000
        assert len(series.data_points) == 1000
        assert series.data_points[0].value == 500.0  # First kept value
        assert series.data_points[-1].value == 1499.0  # Last value
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        series = MetricSeries(
            metric_name="cpu_usage",
            metric_type=MetricType.GAUGE,
            description="CPU usage",
            unit="percent"
        )
        series.add_data_point(50.0)
        series.add_data_point(60.0)
        
        result = series.to_dict()
        
        assert result["metric_name"] == "cpu_usage"
        assert result["metric_type"] == "gauge"
        assert result["description"] == "CPU usage"
        assert result["unit"] == "percent"
        assert len(result["data_points"]) == 2
        assert result["latest_value"] == 60.0


class TestSystemMetrics:
    """Test SystemMetrics."""
    
    def test_creation_with_defaults(self):
        """Test creation with default values."""
        metrics = SystemMetrics()
        
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_percent == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.network_io_bytes == 0.0
        assert metrics.active_connections == 0
        assert metrics.uptime_seconds == 0.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_creation_with_values(self):
        """Test creation with custom values."""
        now = datetime.utcnow()
        metrics = SystemMetrics(
            cpu_usage_percent=75.5,
            memory_usage_percent=60.2,
            disk_usage_percent=45.0,
            network_io_bytes=1024.0,
            active_connections=150,
            uptime_seconds=3600.0,
            timestamp=now
        )
        
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.memory_usage_percent == 60.2
        assert metrics.disk_usage_percent == 45.0
        assert metrics.network_io_bytes == 1024.0
        assert metrics.active_connections == 150
        assert metrics.uptime_seconds == 3600.0
        assert metrics.timestamp == now
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        now = datetime.utcnow()
        metrics = SystemMetrics(
            cpu_usage_percent=75.5,
            memory_usage_percent=60.2,
            timestamp=now
        )
        
        result = metrics.to_dict()
        
        assert result["cpu_usage_percent"] == 75.5
        assert result["memory_usage_percent"] == 60.2
        assert result["timestamp"] == now.isoformat()


class TestDetectionMetrics:
    """Test DetectionMetrics."""
    
    def test_creation_with_defaults(self):
        """Test creation with default values."""
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
    
    def test_creation_with_values(self):
        """Test creation with custom values."""
        now = datetime.utcnow()
        metrics = DetectionMetrics(
            active_detections=5,
            total_detections_today=150,
            avg_detection_time_ms=250.5,
            anomaly_rate_percent=5.2,
            false_positive_rate_percent=1.8,
            precision=0.95,
            recall=0.87,
            f1_score=0.91,
            timestamp=now
        )
        
        assert metrics.active_detections == 5
        assert metrics.total_detections_today == 150
        assert metrics.avg_detection_time_ms == 250.5
        assert metrics.anomaly_rate_percent == 5.2
        assert metrics.false_positive_rate_percent == 1.8
        assert metrics.precision == 0.95
        assert metrics.recall == 0.87
        assert metrics.f1_score == 0.91
        assert metrics.timestamp == now
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        now = datetime.utcnow()
        metrics = DetectionMetrics(
            active_detections=5,
            precision=0.95,
            timestamp=now
        )
        
        result = metrics.to_dict()
        
        assert result["active_detections"] == 5
        assert result["precision"] == 0.95
        assert result["timestamp"] == now.isoformat()


class TestAlertMessage:
    """Test AlertMessage."""
    
    def test_creation(self):
        """Test basic creation."""
        alert_id = uuid4()
        now = datetime.utcnow()
        alert = AlertMessage(
            alert_id=alert_id,
            level=AlertLevel.WARNING,
            title="High CPU Usage",
            message="CPU usage exceeded 80%",
            source="system_monitor",
            timestamp=now,
            metadata={"cpu": 85.2},
            acknowledged=True
        )
        
        assert alert.alert_id == alert_id
        assert alert.level == AlertLevel.WARNING
        assert alert.title == "High CPU Usage"
        assert alert.message == "CPU usage exceeded 80%"
        assert alert.source == "system_monitor"
        assert alert.timestamp == now
        assert alert.metadata == {"cpu": 85.2}
        assert alert.acknowledged is True
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        alert_id = uuid4()
        now = datetime.utcnow()
        alert = AlertMessage(
            alert_id=alert_id,
            level=AlertLevel.CRITICAL,
            title="System Error",
            message="Database connection failed",
            source="database",
            timestamp=now
        )
        
        result = alert.to_dict()
        
        assert result["alert_id"] == str(alert_id)
        assert result["level"] == "critical"
        assert result["title"] == "System Error"
        assert result["message"] == "Database connection failed"
        assert result["source"] == "database"
        assert result["timestamp"] == now.isoformat()
        assert result["metadata"] == {}
        assert result["acknowledged"] is False


class TestMonitoringDashboardRequest:
    """Test MonitoringDashboardRequest."""
    
    def test_creation_with_defaults(self):
        """Test creation with default values."""
        request = MonitoringDashboardRequest()
        
        assert request.time_range_minutes == 60
        assert request.include_system_metrics is True
        assert request.include_detection_metrics is True
        assert request.include_alerts is True
        assert request.metric_names == []
    
    def test_creation_with_values(self):
        """Test creation with custom values."""
        request = MonitoringDashboardRequest(
            time_range_minutes=120,
            include_system_metrics=False,
            include_detection_metrics=True,
            include_alerts=False,
            metric_names=["cpu_usage", "memory_usage"]
        )
        
        assert request.time_range_minutes == 120
        assert request.include_system_metrics is False
        assert request.include_detection_metrics is True
        assert request.include_alerts is False
        assert request.metric_names == ["cpu_usage", "memory_usage"]
    
    def test_validate_positive_time_range(self):
        """Test validation passes for positive time range."""
        request = MonitoringDashboardRequest(time_range_minutes=60)
        request.validate()  # Should not raise
    
    def test_validate_zero_time_range(self):
        """Test validation fails for zero time range."""
        request = MonitoringDashboardRequest(time_range_minutes=0)
        
        with pytest.raises(ValueError, match="Time range must be positive"):
            request.validate()
    
    def test_validate_negative_time_range(self):
        """Test validation fails for negative time range."""
        request = MonitoringDashboardRequest(time_range_minutes=-10)
        
        with pytest.raises(ValueError, match="Time range must be positive"):
            request.validate()
    
    def test_validate_max_time_range(self):
        """Test validation passes for maximum time range."""
        request = MonitoringDashboardRequest(time_range_minutes=1440)
        request.validate()  # Should not raise
    
    def test_validate_excessive_time_range(self):
        """Test validation fails for excessive time range."""
        request = MonitoringDashboardRequest(time_range_minutes=1500)
        
        with pytest.raises(ValueError, match="Time range cannot exceed 24 hours"):
            request.validate()


class TestMonitoringDashboardResponse:
    """Test MonitoringDashboardResponse."""
    
    def test_creation(self):
        """Test basic creation."""
        system_metrics = SystemMetrics(cpu_usage_percent=50.0)
        detection_metrics = DetectionMetrics(active_detections=3)
        alert = AlertMessage(
            alert_id=uuid4(),
            level=AlertLevel.INFO,
            title="Info",
            message="Test message",
            source="test"
        )
        
        response = MonitoringDashboardResponse(
            system_health=SystemHealth.HEALTHY,
            system_metrics=system_metrics,
            detection_metrics=detection_metrics,
            active_alerts=[alert]
        )
        
        assert response.system_health == SystemHealth.HEALTHY
        assert response.system_metrics == system_metrics
        assert response.detection_metrics == detection_metrics
        assert len(response.active_alerts) == 1
        assert response.active_alerts[0] == alert
        assert isinstance(response.generated_at, datetime)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        system_metrics = SystemMetrics(cpu_usage_percent=50.0)
        detection_metrics = DetectionMetrics(active_detections=3)
        
        response = MonitoringDashboardResponse(
            system_health=SystemHealth.DEGRADED,
            system_metrics=system_metrics,
            detection_metrics=detection_metrics
        )
        
        result = response.to_dict()
        
        assert result["system_health"] == "degraded"
        assert "system_metrics" in result
        assert "detection_metrics" in result
        assert "metric_series" in result
        assert "active_alerts" in result
        assert "generated_at" in result
        assert result["active_alerts"] == []
        assert result["metric_series"] == []


class TestRealTimeUpdate:
    """Test RealTimeUpdate."""
    
    def test_creation(self):
        """Test basic creation."""
        now = datetime.utcnow()
        data = {"cpu_usage": 75.5, "memory_usage": 60.2}
        
        update = RealTimeUpdate(
            update_type="metrics",
            data=data,
            timestamp=now
        )
        
        assert update.update_type == "metrics"
        assert update.data == data
        assert update.timestamp == now
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        now = datetime.utcnow()
        data = {"active_detections": 5}
        
        update = RealTimeUpdate(
            update_type="detection",
            data=data,
            timestamp=now
        )
        
        result = update.to_dict()
        
        assert result["update_type"] == "detection"
        assert result["data"] == data
        assert result["timestamp"] == now.isoformat()
