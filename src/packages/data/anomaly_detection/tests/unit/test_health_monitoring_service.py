"""Unit tests for HealthMonitoringService."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any

from anomaly_detection.domain.services.health_monitoring_service import (
    HealthMonitoringService,
    PerformanceTracker,
    AlertManager,
    HealthStatus,
    AlertSeverity,
    HealthMetric,
    SystemAlert,
    HealthReport
)


class TestHealthMetric:
    """Test suite for HealthMetric dataclass."""
    
    def test_health_metric_creation(self):
        """Test health metric creation."""
        timestamp = datetime.utcnow()
        metric = HealthMetric(
            name="cpu_usage",
            value=75.5,
            unit="%",
            status=HealthStatus.WARNING,
            threshold_warning=70.0,
            threshold_critical=90.0,
            timestamp=timestamp,
            description="CPU utilization"
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "%"
        assert metric.status == HealthStatus.WARNING
        assert metric.threshold_warning == 70.0
        assert metric.threshold_critical == 90.0
        assert metric.timestamp == timestamp
        assert metric.description == "CPU utilization"
    
    def test_health_metric_to_dict(self):
        """Test health metric conversion to dictionary."""
        timestamp = datetime.utcnow()
        metric = HealthMetric(
            name="memory_usage",
            value=85.0,
            unit="%",
            status=HealthStatus.CRITICAL,
            threshold_warning=80.0,
            threshold_critical=95.0,
            timestamp=timestamp
        )
        
        result = metric.to_dict()
        
        assert result["name"] == "memory_usage"
        assert result["value"] == 85.0
        assert result["unit"] == "%"
        assert result["status"] == "critical"
        assert result["threshold_warning"] == 80.0
        assert result["threshold_critical"] == 95.0
        assert result["timestamp"] == timestamp.isoformat()


class TestSystemAlert:
    """Test suite for SystemAlert dataclass."""
    
    def test_system_alert_creation(self):
        """Test system alert creation."""
        timestamp = datetime.utcnow()
        alert = SystemAlert(
            alert_id="cpu_high",
            severity=AlertSeverity.WARNING,
            title="High CPU Usage",
            message="CPU usage is above threshold",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold_value=70.0,
            timestamp=timestamp
        )
        
        assert alert.alert_id == "cpu_high"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "High CPU Usage"
        assert alert.message == "CPU usage is above threshold"
        assert alert.metric_name == "cpu_usage"
        assert alert.current_value == 85.0
        assert alert.threshold_value == 70.0
        assert alert.timestamp == timestamp
        assert alert.resolved is False
        assert alert.resolved_at is None
    
    def test_system_alert_to_dict(self):
        """Test system alert conversion to dictionary."""
        timestamp = datetime.utcnow()
        resolved_at = timestamp + timedelta(minutes=5)
        
        alert = SystemAlert(
            alert_id="memory_critical",
            severity=AlertSeverity.CRITICAL,
            title="Critical Memory Usage",
            message="Memory usage critical",
            metric_name="memory_usage",
            current_value=96.0,
            threshold_value=95.0,
            timestamp=timestamp,
            resolved=True,
            resolved_at=resolved_at
        )
        
        result = alert.to_dict()
        
        assert result["alert_id"] == "memory_critical"
        assert result["severity"] == "critical"
        assert result["title"] == "Critical Memory Usage"
        assert result["resolved"] is True
        assert result["resolved_at"] == resolved_at.isoformat()


class TestPerformanceTracker:
    """Test suite for PerformanceTracker."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Create performance tracker instance."""
        return PerformanceTracker(max_history=10)
    
    @pytest.mark.asyncio
    async def test_record_response_time(self, performance_tracker):
        """Test recording response times."""
        await performance_tracker.record_response_time(100.0)
        await performance_tracker.record_response_time(150.0)
        await performance_tracker.record_response_time(200.0)
        
        assert len(performance_tracker.response_times) == 3
        assert performance_tracker.response_times == [100.0, 150.0, 200.0]
        assert len(performance_tracker.timestamps) == 3
    
    @pytest.mark.asyncio
    async def test_record_error(self, performance_tracker):
        """Test recording errors."""
        await performance_tracker.record_error()
        await performance_tracker.record_error()
        
        assert len(performance_tracker.error_counts) == 2
        assert performance_tracker.error_counts == [1, 1]
    
    @pytest.mark.asyncio
    async def test_record_throughput(self, performance_tracker):
        """Test recording throughput."""
        await performance_tracker.record_throughput(50.0)
        await performance_tracker.record_throughput(75.0)
        
        assert len(performance_tracker.throughput_data) == 2
        assert performance_tracker.throughput_data == [50.0, 75.0]
    
    @pytest.mark.asyncio
    async def test_trim_history(self, performance_tracker):
        """Test history trimming when max_history is exceeded."""
        # Add more data than max_history
        for i in range(15):
            await performance_tracker.record_response_time(float(i))
        
        # Should be trimmed to max_history (10)
        assert len(performance_tracker.response_times) == 10
        # Should keep the most recent values
        assert performance_tracker.response_times == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    
    @pytest.mark.asyncio
    async def test_get_performance_summary_empty(self, performance_tracker):
        """Test performance summary with no data."""
        summary = await performance_tracker.get_performance_summary()
        
        assert summary["data_points"] == 0
        assert summary["response_time_stats"] == {}
        assert summary["error_stats"] == {}
        assert summary["throughput_stats"] == {}
    
    @pytest.mark.asyncio
    async def test_get_performance_summary_with_data(self, performance_tracker):
        """Test performance summary with data."""
        # Add response time data
        response_times = [100.0, 200.0, 150.0, 300.0, 250.0]
        for rt in response_times:
            await performance_tracker.record_response_time(rt)
        
        # Add error data
        for _ in range(3):
            await performance_tracker.record_error()
        
        # Add throughput data
        throughput_values = [50.0, 75.0, 60.0]
        for tp in throughput_values:
            await performance_tracker.record_throughput(tp)
        
        summary = await performance_tracker.get_performance_summary()
        
        # Check response time stats
        rt_stats = summary["response_time_stats"]
        assert rt_stats["avg_ms"] == 200.0  # (100+200+150+300+250)/5
        assert rt_stats["median_ms"] == 200.0
        assert rt_stats["min_ms"] == 100.0
        assert rt_stats["max_ms"] == 300.0
        assert rt_stats["p95_ms"] == 300.0
        assert rt_stats["p99_ms"] == 300.0
        
        # Check error stats
        error_stats = summary["error_stats"]
        assert error_stats["total_errors"] == 3
        assert error_stats["error_rate_percent"] == 100.0  # 3 errors in 3 time periods
        
        # Check throughput stats
        tp_stats = summary["throughput_stats"]
        assert tp_stats["avg_rps"] == 61.666666666666664  # (50+75+60)/3
        assert tp_stats["max_rps"] == 75.0
        assert tp_stats["min_rps"] == 50.0
    
    def test_percentile_calculation(self, performance_tracker):
        """Test percentile calculation."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        
        assert performance_tracker._percentile(data, 50) == 50.0
        assert performance_tracker._percentile(data, 95) == 100.0
        assert performance_tracker._percentile(data, 0) == 10.0
        assert performance_tracker._percentile([], 50) == 0.0


class TestAlertManager:
    """Test suite for AlertManager."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        return AlertManager()
    
    def test_add_alert_handler(self, alert_manager):
        """Test adding alert handlers."""
        handler = Mock()
        alert_manager.add_alert_handler(handler)
        
        assert handler in alert_manager.alert_handlers
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alert_manager):
        """Test creating an alert."""
        handler = Mock()
        alert_manager.add_alert_handler(handler)
        
        alert = await alert_manager.create_alert(
            alert_id="test_alert",
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            metric_name="test_metric",
            current_value=75.0,
            threshold_value=70.0
        )
        
        assert isinstance(alert, SystemAlert)
        assert alert.alert_id == "test_alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.resolved is False
        
        # Check alert is stored
        assert "test_alert" in alert_manager.active_alerts
        assert alert in alert_manager.alert_history
        
        # Check handler was called
        handler.assert_called_once_with(alert)
    
    @pytest.mark.asyncio
    async def test_create_alert_handler_exception(self, alert_manager):
        """Test alert creation when handler raises exception."""
        failing_handler = Mock(side_effect=Exception("Handler failed"))
        working_handler = Mock()
        
        alert_manager.add_alert_handler(failing_handler)
        alert_manager.add_alert_handler(working_handler)
        
        # Should not raise exception even if handler fails
        alert = await alert_manager.create_alert(
            alert_id="test_alert",
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message="Test message",
            metric_name="test_metric",
            current_value=80.0,
            threshold_value=70.0
        )
        
        assert isinstance(alert, SystemAlert)
        # Working handler should still be called
        working_handler.assert_called_once_with(alert)
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving an alert."""
        # Create alert first
        alert = await alert_manager.create_alert(
            alert_id="resolve_test",
            severity=AlertSeverity.WARNING,
            title="Resolve Test",
            message="Test message",
            metric_name="test_metric",
            current_value=75.0,
            threshold_value=70.0
        )
        
        # Resolve alert
        result = await alert_manager.resolve_alert("resolve_test")
        
        assert result is True
        assert "resolve_test" not in alert_manager.active_alerts
        assert alert.resolved is True
        assert alert.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_nonexistent_alert(self, alert_manager):
        """Test resolving non-existent alert."""
        result = await alert_manager.resolve_alert("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Create alerts with different severities
        await alert_manager.create_alert(
            alert_id="warning_alert",
            severity=AlertSeverity.WARNING,
            title="Warning",
            message="Warning message",
            metric_name="metric1",
            current_value=75.0,
            threshold_value=70.0
        )
        
        await alert_manager.create_alert(
            alert_id="critical_alert",
            severity=AlertSeverity.CRITICAL,
            title="Critical",
            message="Critical message",
            metric_name="metric2",
            current_value=95.0,
            threshold_value=90.0
        )
        
        # Get all active alerts
        all_alerts = await alert_manager.get_active_alerts()
        assert len(all_alerts) == 2
        
        # Get filtered alerts
        critical_alerts = await alert_manager.get_active_alerts(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].alert_id == "critical_alert"
    
    @pytest.mark.asyncio
    async def test_get_alert_history(self, alert_manager):
        """Test getting alert history."""
        # Create alert
        await alert_manager.create_alert(
            alert_id="history_test",
            severity=AlertSeverity.INFO,
            title="History Test",
            message="Test message",
            metric_name="test_metric",
            current_value=50.0,
            threshold_value=60.0
        )
        
        # Get recent history
        recent_alerts = await alert_manager.get_alert_history(hours=1)
        assert len(recent_alerts) == 1
        assert recent_alerts[0].alert_id == "history_test"
        
        # Get older history (should be empty)
        old_alerts = await alert_manager.get_alert_history(hours=0)
        assert len(old_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_alert_history_trimming(self, alert_manager):
        """Test alert history trimming at 1000 items."""
        # Create more than 1000 alerts to test trimming
        for i in range(1005):
            await alert_manager.create_alert(
                alert_id=f"alert_{i}",
                severity=AlertSeverity.INFO,
                title=f"Alert {i}",
                message=f"Message {i}",
                metric_name="test_metric",
                current_value=50.0,
                threshold_value=60.0
            )
        
        # History should be trimmed to 1000
        assert len(alert_manager.alert_history) == 1000
        # Should keep the most recent alerts
        assert alert_manager.alert_history[0].alert_id == "alert_5"  # First kept alert
        assert alert_manager.alert_history[-1].alert_id == "alert_1004"  # Last alert


class TestHealthMonitoringService:
    """Test suite for HealthMonitoringService."""
    
    @pytest.fixture
    def health_service(self):
        """Create health monitoring service instance."""
        return HealthMonitoringService(check_interval=1)
    
    def test_initialization(self, health_service):
        """Test health monitoring service initialization."""
        assert health_service.check_interval == 1
        assert isinstance(health_service.performance_tracker, PerformanceTracker)
        assert isinstance(health_service.alert_manager, AlertManager)
        assert health_service.monitoring_enabled is False
        assert health_service.monitoring_task is None
        
        # Check default thresholds
        assert "cpu_usage" in health_service.thresholds
        assert "memory_usage" in health_service.thresholds
        assert health_service.thresholds["cpu_usage"]["warning"] == 70.0
        assert health_service.thresholds["cpu_usage"]["critical"] == 90.0
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_service):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await health_service.start_monitoring()
        assert health_service.monitoring_enabled is True
        assert health_service.monitoring_task is not None
        
        # Stop monitoring
        await health_service.stop_monitoring()
        assert health_service.monitoring_enabled is False
    
    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self, health_service):
        """Test starting monitoring when already running."""
        await health_service.start_monitoring()
        first_task = health_service.monitoring_task
        
        # Start again - should not create new task
        await health_service.start_monitoring()
        assert health_service.monitoring_task is first_task
        
        await health_service.stop_monitoring()
    
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.cpu_percent')
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.virtual_memory')
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_check_system_resources(self, mock_disk, mock_memory, mock_cpu, health_service):
        """Test system resource checking."""
        # Mock system metrics
        mock_cpu.return_value = 85.0
        mock_memory.return_value = Mock(percent=78.5)
        mock_disk.return_value = Mock(used=80 * 1024**3, total=100 * 1024**3)  # 80% used
        
        await health_service._check_system_resources()
        
        # Verify psutil calls
        mock_cpu.assert_called_once_with(interval=1)
        mock_memory.assert_called_once()
        mock_disk.assert_called_once_with('/')
    
    @pytest.mark.asyncio
    async def test_check_performance_metrics(self, health_service):
        """Test performance metrics checking."""
        # Add some performance data
        await health_service.performance_tracker.record_response_time(1500.0)  # Above warning
        await health_service.performance_tracker.record_error()
        
        await health_service._check_performance_metrics()
        
        # Should have evaluated metrics (tested via integration)
        # This test mainly verifies no exceptions are raised
    
    @pytest.mark.asyncio
    async def test_evaluate_metric_healthy(self, health_service):
        """Test metric evaluation for healthy values."""
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=50.0,  # Below warning threshold
            unit="%",
            description="CPU usage"
        )
        
        # Should not create any alerts
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_metric_warning(self, health_service):
        """Test metric evaluation for warning values."""
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=75.0,  # Above warning threshold
            unit="%",
            description="CPU usage"
        )
        
        # Should create warning alert
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].severity == AlertSeverity.WARNING
        assert active_alerts[0].alert_id == "cpu_usage_threshold"
    
    @pytest.mark.asyncio
    async def test_evaluate_metric_critical(self, health_service):
        """Test metric evaluation for critical values."""
        await health_service._evaluate_metric(
            metric_name="memory_usage",
            value=96.0,  # Above critical threshold
            unit="%",
            description="Memory usage"
        )
        
        # Should create critical alert
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].severity == AlertSeverity.CRITICAL
        assert active_alerts[0].alert_id == "memory_usage_threshold"
    
    @pytest.mark.asyncio
    async def test_evaluate_metric_resolve_alert(self, health_service):
        """Test alert resolution when metric becomes healthy."""
        # Create warning condition
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=75.0,  # Warning
            unit="%",
            description="CPU usage"
        )
        
        # Verify alert exists
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        # Make metric healthy
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=50.0,  # Healthy
            unit="%",
            description="CPU usage"
        )
        
        # Alert should be resolved
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_metric_no_duplicate_alerts(self, health_service):
        """Test that duplicate alerts are not created."""
        # Create warning condition twice
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=75.0,  # Warning
            unit="%",
            description="CPU usage"
        )
        
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=80.0,  # Still warning
            unit="%",
            description="CPU usage"
        )
        
        # Should still have only one alert
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
    
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.cpu_percent')
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.virtual_memory')
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_collect_current_metrics(self, mock_disk, mock_memory, mock_cpu, health_service):
        """Test current metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 60.0
        mock_memory.return_value = Mock(percent=70.0)
        mock_disk.return_value = Mock(used=50 * 1024**3, total=100 * 1024**3)
        
        # Add performance data
        await health_service.performance_tracker.record_response_time(500.0)
        
        metrics = await health_service._collect_current_metrics()
        
        assert len(metrics) >= 3  # At least CPU, memory, disk
        
        # Find CPU metric
        cpu_metric = next(m for m in metrics if m.name == "cpu_usage")
        assert cpu_metric.value == 60.0
        assert cpu_metric.unit == "%"
        assert cpu_metric.status == HealthStatus.HEALTHY
    
    def test_get_metric_status(self, health_service):
        """Test metric status determination."""
        # Healthy
        status = health_service._get_metric_status("cpu_usage", 50.0)
        assert status == HealthStatus.HEALTHY
        
        # Warning
        status = health_service._get_metric_status("cpu_usage", 75.0)
        assert status == HealthStatus.WARNING
        
        # Critical
        status = health_service._get_metric_status("cpu_usage", 95.0)
        assert status == HealthStatus.CRITICAL
        
        # Unknown metric
        status = health_service._get_metric_status("unknown_metric", 50.0)
        assert status == HealthStatus.HEALTHY  # No thresholds means always healthy
    
    def test_calculate_overall_score(self, health_service):
        """Test overall score calculation."""
        metrics = [
            Mock(status=HealthStatus.HEALTHY),
            Mock(status=HealthStatus.WARNING),
            Mock(status=HealthStatus.CRITICAL),
            Mock(status=HealthStatus.UNKNOWN)
        ]
        
        score = health_service._calculate_overall_score(metrics)
        expected = (100 + 70 + 30 + 50) / 4  # 62.5
        assert score == expected
        
        # Empty metrics
        assert health_service._calculate_overall_score([]) == 0.0
    
    def test_determine_overall_status(self, health_service):
        """Test overall status determination."""
        # Critical alert present
        critical_alert = Mock(severity=AlertSeverity.CRITICAL)
        status = health_service._determine_overall_status([], [critical_alert])
        assert status == HealthStatus.CRITICAL
        
        # Critical metric present
        critical_metric = Mock(status=HealthStatus.CRITICAL)
        status = health_service._determine_overall_status([critical_metric], [])
        assert status == HealthStatus.CRITICAL
        
        # Warning conditions
        warning_alert = Mock(severity=AlertSeverity.WARNING)
        warning_metric = Mock(status=HealthStatus.WARNING)
        status = health_service._determine_overall_status([warning_metric], [warning_alert])
        assert status == HealthStatus.WARNING
        
        # All healthy
        healthy_metric = Mock(status=HealthStatus.HEALTHY)
        status = health_service._determine_overall_status([healthy_metric], [])
        assert status == HealthStatus.HEALTHY
    
    def test_generate_recommendations(self, health_service):
        """Test recommendation generation."""
        # Critical CPU metric
        cpu_metric = Mock(name="cpu_usage", status=HealthStatus.CRITICAL)
        memory_metric = Mock(name="memory_usage", status=HealthStatus.WARNING)
        
        recommendations = health_service._generate_recommendations([cpu_metric, memory_metric], [])
        
        assert len(recommendations) >= 1
        assert any("CPU" in rec or "cpu" in rec for rec in recommendations)
        assert any("memory" in rec for rec in recommendations)
    
    def test_generate_recommendations_healthy(self, health_service):
        """Test recommendation generation for healthy system."""
        healthy_metric = Mock(name="cpu_usage", status=HealthStatus.HEALTHY)
        
        recommendations = health_service._generate_recommendations([healthy_metric], [])
        
        assert "System is operating within normal parameters" in recommendations
    
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.cpu_percent')
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.virtual_memory')
    @patch('anomaly_detection.domain.services.health_monitoring_service.psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_get_health_report(self, mock_disk, mock_memory, mock_cpu, health_service):
        """Test comprehensive health report generation."""
        # Mock system metrics
        mock_cpu.return_value = 65.0
        mock_memory.return_value = Mock(percent=75.0)
        mock_disk.return_value = Mock(used=60 * 1024**3, total=100 * 1024**3)
        
        # Add performance data
        await health_service.performance_tracker.record_response_time(800.0)
        
        report = await health_service.get_health_report()
        
        assert isinstance(report, HealthReport)
        assert isinstance(report.overall_status, HealthStatus)
        assert isinstance(report.overall_score, float)
        assert len(report.metrics) >= 3
        assert isinstance(report.active_alerts, list)
        assert isinstance(report.performance_summary, dict)
        assert report.uptime_seconds > 0
        assert isinstance(report.timestamp, datetime)
        assert isinstance(report.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_record_api_call_success(self, health_service):
        """Test recording successful API call."""
        await health_service.record_api_call(response_time_ms=150.0, success=True)
        
        summary = await health_service.performance_tracker.get_performance_summary()
        assert summary["response_time_stats"]["avg_ms"] == 150.0
        assert summary["error_stats"] == {}  # No errors
    
    @pytest.mark.asyncio
    async def test_record_api_call_failure(self, health_service):
        """Test recording failed API call."""
        await health_service.record_api_call(response_time_ms=200.0, success=False)
        
        summary = await health_service.performance_tracker.get_performance_summary()
        assert summary["response_time_stats"]["avg_ms"] == 200.0
        assert summary["error_stats"]["total_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_get_alert_history(self, health_service):
        """Test getting alert history."""
        # Create an alert
        await health_service._evaluate_metric(
            metric_name="cpu_usage",
            value=85.0,  # Warning
            unit="%",
            description="CPU usage"
        )
        
        history = await health_service.get_alert_history(hours=1)
        assert len(history) == 1
        assert history[0].metric_name == "cpu_usage"
    
    def test_set_threshold(self, health_service):
        """Test setting custom thresholds."""
        health_service.set_threshold("custom_metric", warning=30.0, critical=50.0)
        
        assert "custom_metric" in health_service.thresholds
        assert health_service.thresholds["custom_metric"]["warning"] == 30.0
        assert health_service.thresholds["custom_metric"]["critical"] == 50.0
    
    def test_add_alert_handler(self, health_service):
        """Test adding custom alert handler."""
        handler = Mock()
        health_service.add_alert_handler(handler)
        
        assert handler in health_service.alert_manager.alert_handlers
    
    def test_default_alert_handler(self, health_service):
        """Test default alert handler."""
        alert = Mock(
            alert_id="test",
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message"
        )
        
        # Should not raise exception
        health_service._default_alert_handler(alert)
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_exception_handling(self, health_service):
        """Test monitoring loop handles exceptions gracefully."""
        # Mock perform_health_checks to raise exception
        original_method = health_service._perform_health_checks
        health_service._perform_health_checks = AsyncMock(side_effect=[Exception("Test error"), None])
        
        # Start monitoring for a short time
        await health_service.start_monitoring()
        await asyncio.sleep(0.1)  # Let it run briefly
        await health_service.stop_monitoring()
        
        # Should not have crashed
        health_service._perform_health_checks = original_method
    
    @pytest.mark.asyncio
    async def test_health_report_to_dict(self, health_service):
        """Test health report dictionary conversion."""
        # Create a simple health report
        metric = HealthMetric(
            name="test_metric",
            value=50.0,
            unit="test",
            status=HealthStatus.HEALTHY,
            threshold_warning=60.0,
            threshold_critical=80.0,
            timestamp=datetime.utcnow()
        )
        
        alert = SystemAlert(
            alert_id="test_alert",
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test message",
            metric_name="test_metric",
            current_value=50.0,
            threshold_value=60.0,
            timestamp=datetime.utcnow()
        )
        
        report = HealthReport(
            overall_status=HealthStatus.HEALTHY,
            overall_score=85.0,
            metrics=[metric],
            active_alerts=[alert],
            performance_summary={"test": "data"},
            uptime_seconds=300.0,
            timestamp=datetime.utcnow(),
            recommendations=["Test recommendation"]
        )
        
        result = report.to_dict()
        
        assert result["overall_status"] == "healthy"
        assert result["overall_score"] == 85.0
        assert len(result["metrics"]) == 1
        assert len(result["active_alerts"]) == 1
        assert result["performance_summary"] == {"test": "data"}
        assert result["uptime_seconds"] == 300.0
        assert result["recommendations"] == ["Test recommendation"]
    
    @pytest.mark.asyncio
    async def test_concurrent_metric_evaluation(self, health_service):
        """Test concurrent metric evaluations don't interfere."""
        # Run multiple evaluations concurrently
        tasks = []
        for i in range(5):
            task = health_service._evaluate_metric(
                metric_name=f"test_metric_{i}",
                value=80.0,  # Warning level
                unit="%",
                description=f"Test metric {i}"
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Should have created alerts for each metric
        active_alerts = await health_service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 5
    
    @pytest.mark.parametrize("cpu_value,expected_status", [
        (50.0, HealthStatus.HEALTHY),
        (75.0, HealthStatus.WARNING),
        (95.0, HealthStatus.CRITICAL),
    ])
    def test_metric_status_thresholds(self, health_service, cpu_value, expected_status):
        """Test metric status determination with different values."""
        status = health_service._get_metric_status("cpu_usage", cpu_value)
        assert status == expected_status