"""Unit tests for health monitoring API endpoints."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from anomaly_detection.api.v1.health import (
    router, HealthReportResponse, AlertResponse, PerformanceMetricsResponse,
    ThresholdUpdateRequest, get_health_service
)
from anomaly_detection.domain.services.health_monitoring_service import (
    HealthStatus, AlertSeverity
)


@pytest.fixture
def app():
    """Create FastAPI app with health router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/health")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_health_service():
    """Create mock health monitoring service."""
    service = Mock()
    service.check_interval = 30
    
    # Mock health report
    mock_report = Mock()
    mock_report.overall_status = HealthStatus.HEALTHY
    mock_report.overall_score = 85.5
    mock_report.metrics = [
        Mock(to_dict=Mock(return_value={
            "name": "cpu_usage",
            "value": 45.2,
            "status": "healthy",
            "threshold_warning": 70.0,
            "threshold_critical": 90.0
        })),
        Mock(to_dict=Mock(return_value={
            "name": "memory_usage", 
            "value": 512.8,
            "status": "healthy",
            "threshold_warning": 1024.0,
            "threshold_critical": 1536.0
        }))
    ]
    mock_report.active_alerts = [
        Mock(to_dict=Mock(return_value={
            "alert_id": "alert-123",
            "severity": "warning",
            "title": "High CPU Usage",
            "message": "CPU usage is above normal",
            "metric_name": "cpu_usage",
            "current_value": 75.2,
            "threshold_value": 70.0,
            "timestamp": "2024-01-01T12:00:00",
            "resolved": False
        }))
    ]
    mock_report.performance_summary = {
        "avg_response_time_ms": 125.5,
        "success_rate": 0.96,
        "total_requests": 1500
    }
    mock_report.uptime_seconds = 3600.0
    mock_report.timestamp = datetime(2024, 1, 1, 12, 0, 0)
    mock_report.recommendations = [
        "Consider scaling resources if CPU usage continues to be high",
        "Monitor memory usage trends"
    ]
    
    service.get_health_report = AsyncMock(return_value=mock_report)
    
    # Mock alert manager
    service.alert_manager = Mock()
    
    # Mock active alerts
    mock_alert = Mock()
    mock_alert.alert_id = "alert-456"
    mock_alert.severity = AlertSeverity.WARNING
    mock_alert.title = "High Memory Usage"
    mock_alert.message = "Memory usage is above threshold"
    mock_alert.metric_name = "memory_usage"
    mock_alert.current_value = 1100.0
    mock_alert.threshold_value = 1024.0
    mock_alert.timestamp = datetime(2024, 1, 1, 12, 5, 0)
    mock_alert.resolved = False
    mock_alert.resolved_at = None
    
    service.alert_manager.get_active_alerts = AsyncMock(return_value=[mock_alert])
    service.alert_manager.resolve_alert = AsyncMock(return_value=True)
    
    # Mock alert history
    service.get_alert_history = AsyncMock(return_value=[mock_alert])
    
    # Mock performance tracker
    service.performance_tracker = Mock()
    service.performance_tracker.get_performance_summary = AsyncMock(return_value={
        "response_time_stats": {
            "avg_ms": 125.5,
            "min_ms": 25.0,
            "max_ms": 850.0,
            "p95_ms": 275.2
        },
        "error_stats": {
            "total_errors": 25,
            "error_rate": 0.04,
            "error_types": {"timeout": 15, "validation": 10}
        },
        "throughput_stats": {
            "requests_per_second": 12.5,
            "requests_per_minute": 750.0
        },
        "data_points": 5000
    })
    
    # Mock monitoring control
    service.start_monitoring = AsyncMock()
    service.stop_monitoring = AsyncMock()
    
    # Mock thresholds
    service.thresholds = {
        "cpu_usage": {"warning": 70.0, "critical": 90.0},
        "memory_usage": {"warning": 1024.0, "critical": 1536.0}
    }
    service.set_threshold = Mock()
    
    # Mock API call recording
    service.record_api_call = AsyncMock()
    
    return service


class TestHealthReportEndpoint:
    """Test health report endpoint."""
    
    def test_get_health_report_success(self, client, mock_health_service):
        """Test successful health report retrieval."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/report")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["overall_status"] == "healthy"
            assert data["overall_score"] == 85.5
            assert len(data["metrics"]) == 2
            assert len(data["active_alerts"]) == 1
            assert data["uptime_seconds"] == 3600.0
            assert data["timestamp"] == "2024-01-01T12:00:00"
            assert len(data["recommendations"]) == 2
            
            # Check metrics
            cpu_metric = data["metrics"][0]
            assert cpu_metric["name"] == "cpu_usage"
            assert cpu_metric["value"] == 45.2
            assert cpu_metric["status"] == "healthy"
            
            # Check active alerts
            alert = data["active_alerts"][0]
            assert alert["alert_id"] == "alert-123"
            assert alert["severity"] == "warning"
            assert alert["title"] == "High CPU Usage"
            
            # Check performance summary
            performance = data["performance_summary"]
            assert performance["avg_response_time_ms"] == 125.5
            assert performance["success_rate"] == 0.96
            
            # Verify service was called correctly
            mock_health_service.get_health_report.assert_called_once_with(include_history=False)
    
    def test_get_health_report_with_history(self, client, mock_health_service):
        """Test health report with history included."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/report?include_history=true")
            
            assert response.status_code == 200
            
            # Verify service was called with history flag
            mock_health_service.get_health_report.assert_called_once_with(include_history=True)
    
    def test_get_health_report_error(self, client):
        """Test health report with service error."""
        mock_service = Mock()
        mock_service.get_health_report = AsyncMock(side_effect=Exception("Health check failed"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.get("/api/v1/health/report")
            
            assert response.status_code == 500
            assert "Failed to get health report" in response.json()["detail"]


class TestHealthStatusEndpoint:
    """Test simple health status endpoint."""
    
    def test_get_health_status_success(self, client, mock_health_service):
        """Test successful health status retrieval."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["score"] == "85.5"
            assert data["timestamp"] == "2024-01-01T12:00:00"
            assert data["uptime"] == "3600s"
    
    def test_get_health_status_error(self, client):
        """Test health status with service error."""
        mock_service = Mock()
        mock_service.get_health_report = AsyncMock(side_effect=Exception("Status check failed"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.get("/api/v1/health/status")
            
            assert response.status_code == 200  # Should not fail
            data = response.json()
            
            assert data["status"] == "unknown"
            assert "error" in data
            assert "timestamp" in data


class TestActiveAlertsEndpoint:
    """Test active alerts endpoint."""
    
    def test_get_active_alerts_success(self, client, mock_health_service):
        """Test successful active alerts retrieval."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/alerts")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            alert = data[0]
            
            assert alert["alert_id"] == "alert-456"
            assert alert["severity"] == "warning"
            assert alert["title"] == "High Memory Usage"
            assert alert["message"] == "Memory usage is above threshold"
            assert alert["metric_name"] == "memory_usage"
            assert alert["current_value"] == 1100.0
            assert alert["threshold_value"] == 1024.0
            assert alert["timestamp"] == "2024-01-01T12:05:00"
            assert alert["resolved"] is False
            assert alert["resolved_at"] is None
            
            # Verify service was called
            mock_health_service.alert_manager.get_active_alerts.assert_called_once_with(None)
    
    def test_get_active_alerts_with_severity_filter(self, client, mock_health_service):
        """Test active alerts with severity filter."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/alerts?severity=warning")
            
            assert response.status_code == 200
            
            # Verify service was called with severity filter
            call_args = mock_health_service.alert_manager.get_active_alerts.call_args
            assert call_args[0][0] == AlertSeverity.WARNING
    
    def test_get_active_alerts_invalid_severity(self, client, mock_health_service):
        """Test active alerts with invalid severity."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/alerts?severity=invalid")
            
            assert response.status_code == 400
            assert "Invalid severity" in response.json()["detail"]
    
    def test_get_active_alerts_error(self, client):
        """Test active alerts with service error."""
        mock_service = Mock()
        mock_service.alert_manager.get_active_alerts = AsyncMock(side_effect=Exception("Alerts error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.get("/api/v1/health/alerts")
            
            assert response.status_code == 500
            assert "Failed to get active alerts" in response.json()["detail"]


class TestAlertHistoryEndpoint:
    """Test alert history endpoint."""
    
    def test_get_alert_history_success(self, client, mock_health_service):
        """Test successful alert history retrieval."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/alerts/history?hours=12")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 1
            alert = data[0]
            assert alert["alert_id"] == "alert-456"
            assert alert["severity"] == "warning"
            
            # Verify service was called with correct hours
            mock_health_service.get_alert_history.assert_called_once_with(12)
    
    def test_get_alert_history_default_hours(self, client, mock_health_service):
        """Test alert history with default hours."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/alerts/history")
            
            assert response.status_code == 200
            
            # Should use default of 24 hours
            mock_health_service.get_alert_history.assert_called_once_with(24)
    
    def test_get_alert_history_error(self, client):
        """Test alert history with service error."""
        mock_service = Mock()
        mock_service.get_alert_history = AsyncMock(side_effect=Exception("History error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.get("/api/v1/health/alerts/history")
            
            assert response.status_code == 500
            assert "Failed to get alert history" in response.json()["detail"]


class TestResolveAlertEndpoint:
    """Test resolve alert endpoint."""
    
    def test_resolve_alert_success(self, client, mock_health_service):
        """Test successful alert resolution."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.post("/api/v1/health/alerts/alert-123/resolve")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "resolved successfully" in data["message"]
            assert "timestamp" in data
            
            # Verify service was called
            mock_health_service.alert_manager.resolve_alert.assert_called_once_with("alert-123")
    
    def test_resolve_alert_not_found(self, client):
        """Test resolving non-existent alert."""
        mock_service = Mock()
        mock_service.alert_manager.resolve_alert = AsyncMock(return_value=False)
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.post("/api/v1/health/alerts/nonexistent/resolve")
            
            assert response.status_code == 404
            assert "not found or already resolved" in response.json()["detail"]
    
    def test_resolve_alert_error(self, client):
        """Test resolve alert with service error."""
        mock_service = Mock()
        mock_service.alert_manager.resolve_alert = AsyncMock(side_effect=Exception("Resolve error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.post("/api/v1/health/alerts/alert-123/resolve")
            
            assert response.status_code == 500
            assert "Failed to resolve alert" in response.json()["detail"]


class TestPerformanceMetricsEndpoint:
    """Test performance metrics endpoint."""
    
    def test_get_performance_metrics_success(self, client, mock_health_service):
        """Test successful performance metrics retrieval."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/metrics/performance")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "response_time_stats" in data
            assert "error_stats" in data
            assert "throughput_stats" in data
            assert data["data_points"] == 5000
            
            # Check response time stats
            response_stats = data["response_time_stats"]
            assert response_stats["avg_ms"] == 125.5
            assert response_stats["p95_ms"] == 275.2
            
            # Check error stats
            error_stats = data["error_stats"]
            assert error_stats["total_errors"] == 25
            assert error_stats["error_rate"] == 0.04
            
            # Check throughput stats
            throughput_stats = data["throughput_stats"]
            assert throughput_stats["requests_per_second"] == 12.5
            
            # Verify service was called
            mock_health_service.performance_tracker.get_performance_summary.assert_called_once()
    
    def test_get_performance_metrics_error(self, client):
        """Test performance metrics with service error."""
        mock_service = Mock()
        mock_service.performance_tracker.get_performance_summary = AsyncMock(side_effect=Exception("Metrics error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.get("/api/v1/health/metrics/performance")
            
            assert response.status_code == 500
            assert "Failed to get performance metrics" in response.json()["detail"]


class TestMonitoringControlEndpoints:
    """Test monitoring start/stop endpoints."""
    
    def test_start_monitoring_success(self, client, mock_health_service):
        """Test successful monitoring start."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.post("/api/v1/health/monitoring/start")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "started successfully" in data["message"]
            assert data["check_interval"] == "30s"
            assert "timestamp" in data
            
            # Verify service was called
            mock_health_service.start_monitoring.assert_called_once()
    
    def test_start_monitoring_error(self, client):
        """Test monitoring start with error."""
        mock_service = Mock()
        mock_service.start_monitoring = AsyncMock(side_effect=Exception("Start error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.post("/api/v1/health/monitoring/start")
            
            assert response.status_code == 500
            assert "Failed to start health monitoring" in response.json()["detail"]
    
    def test_stop_monitoring_success(self, client, mock_health_service):
        """Test successful monitoring stop."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.post("/api/v1/health/monitoring/stop")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "stopped successfully" in data["message"]
            assert "timestamp" in data
            
            # Verify service was called
            mock_health_service.stop_monitoring.assert_called_once()
    
    def test_stop_monitoring_error(self, client):
        """Test monitoring stop with error."""
        mock_service = Mock()
        mock_service.stop_monitoring = AsyncMock(side_effect=Exception("Stop error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.post("/api/v1/health/monitoring/stop")
            
            assert response.status_code == 500
            assert "Failed to stop health monitoring" in response.json()["detail"]


class TestThresholdManagementEndpoints:
    """Test threshold management endpoints."""
    
    def test_update_threshold_success(self, client, mock_health_service):
        """Test successful threshold update."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            request_data = {
                "metric_name": "cpu_usage",
                "warning_threshold": 75.0,
                "critical_threshold": 95.0
            }
            
            response = client.post("/api/v1/health/thresholds", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "Thresholds updated" in data["message"]
            assert data["metric"] == "cpu_usage"
            assert data["warning_threshold"] == "75.0"
            assert data["critical_threshold"] == "95.0"
            assert "timestamp" in data
            
            # Verify service was called
            mock_health_service.set_threshold.assert_called_once_with(
                "cpu_usage", 75.0, 95.0
            )
    
    def test_update_threshold_invalid_values(self, client, mock_health_service):
        """Test threshold update with invalid values."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            request_data = {
                "metric_name": "cpu_usage",
                "warning_threshold": 90.0,
                "critical_threshold": 80.0  # Less than warning
            }
            
            response = client.post("/api/v1/health/thresholds", json=request_data)
            
            assert response.status_code == 400
            assert "must be less than critical threshold" in response.json()["detail"]
    
    def test_update_threshold_error(self, client):
        """Test threshold update with service error."""
        mock_service = Mock()
        mock_service.set_threshold.side_effect = Exception("Threshold error")
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            request_data = {
                "metric_name": "cpu_usage",
                "warning_threshold": 75.0,
                "critical_threshold": 95.0
            }
            
            response = client.post("/api/v1/health/thresholds", json=request_data)
            
            assert response.status_code == 500
            assert "Failed to update thresholds" in response.json()["detail"]
    
    def test_get_thresholds_success(self, client, mock_health_service):
        """Test successful threshold retrieval."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.get("/api/v1/health/thresholds")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "cpu_usage" in data
            assert "memory_usage" in data
            
            cpu_thresholds = data["cpu_usage"]
            assert cpu_thresholds["warning"] == 70.0
            assert cpu_thresholds["critical"] == 90.0
    
    def test_get_thresholds_error(self, client):
        """Test threshold retrieval with error."""
        mock_service = Mock()
        mock_service.thresholds = None  # Will cause attribute error
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            del mock_service.thresholds  # Remove attribute to cause error
            
            response = client.get("/api/v1/health/thresholds")
            
            assert response.status_code == 500
            assert "Failed to get thresholds" in response.json()["detail"]


class TestRecordMetricsEndpoint:
    """Test API call metrics recording endpoint."""
    
    def test_record_api_call_success(self, client, mock_health_service):
        """Test successful API call recording."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.post(
                "/api/v1/health/metrics/record?response_time_ms=125.5&success=true"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "metrics recorded" in data["message"]
            assert data["response_time_ms"] == "125.5"
            assert data["success"] == "True"
            assert "timestamp" in data
            
            # Verify service was called
            mock_health_service.record_api_call.assert_called_once_with(125.5, True)
    
    def test_record_api_call_failure(self, client, mock_health_service):
        """Test API call recording for failed call."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.post(
                "/api/v1/health/metrics/record?response_time_ms=2500.0&success=false"
            )
            
            assert response.status_code == 200
            
            # Verify service was called with failure
            mock_health_service.record_api_call.assert_called_once_with(2500.0, False)
    
    def test_record_api_call_defaults(self, client, mock_health_service):
        """Test API call recording with default success value."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            response = client.post(
                "/api/v1/health/metrics/record?response_time_ms=100.0"
            )
            
            assert response.status_code == 200
            
            # Should default to success=True
            mock_health_service.record_api_call.assert_called_once_with(100.0, True)
    
    def test_record_api_call_error(self, client):
        """Test API call recording with service error."""
        mock_service = Mock()
        mock_service.record_api_call = AsyncMock(side_effect=Exception("Record error"))
        
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_service):
            
            response = client.post(
                "/api/v1/health/metrics/record?response_time_ms=125.5"
            )
            
            assert response.status_code == 500
            assert "Failed to record API call metrics" in response.json()["detail"]


class TestHealthPydanticModels:
    """Test Pydantic models for health endpoints."""
    
    def test_health_report_response_model(self):
        """Test HealthReportResponse model."""
        response = HealthReportResponse(
            overall_status="healthy",
            overall_score=85.5,
            metrics=[{"name": "cpu", "value": 45.2}],
            active_alerts=[{"id": "alert-1", "severity": "warning"}],
            performance_summary={"avg_ms": 125.5},
            uptime_seconds=3600.0,
            timestamp="2024-01-01T12:00:00",
            recommendations=["Scale resources"]
        )
        
        assert response.overall_status == "healthy"
        assert response.overall_score == 85.5
        assert len(response.metrics) == 1
        assert len(response.active_alerts) == 1
        assert response.uptime_seconds == 3600.0
        assert len(response.recommendations) == 1
    
    def test_alert_response_model(self):
        """Test AlertResponse model."""
        response = AlertResponse(
            alert_id="alert-123",
            severity="critical",
            title="System Failure",
            message="Critical system error detected",
            metric_name="system_health",
            current_value=0.0,
            threshold_value=0.5,
            timestamp="2024-01-01T12:00:00",
            resolved=False,
            resolved_at=None
        )
        
        assert response.alert_id == "alert-123"
        assert response.severity == "critical"
        assert response.title == "System Failure"
        assert response.resolved is False
        assert response.resolved_at is None
    
    def test_performance_metrics_response_model(self):
        """Test PerformanceMetricsResponse model."""
        response = PerformanceMetricsResponse(
            response_time_stats={"avg_ms": 125.5, "p95_ms": 275.0},
            error_stats={"total_errors": 25, "error_rate": 0.04},
            throughput_stats={"requests_per_second": 12.5},
            data_points=5000
        )
        
        assert response.response_time_stats["avg_ms"] == 125.5
        assert response.error_stats["total_errors"] == 25
        assert response.throughput_stats["requests_per_second"] == 12.5
        assert response.data_points == 5000
    
    def test_threshold_update_request_model(self):
        """Test ThresholdUpdateRequest model."""
        request = ThresholdUpdateRequest(
            metric_name="cpu_usage",
            warning_threshold=75.0,
            critical_threshold=95.0
        )
        
        assert request.metric_name == "cpu_usage"
        assert request.warning_threshold == 75.0
        assert request.critical_threshold == 95.0


class TestHealthDependencies:
    """Test health service dependencies."""
    
    def test_get_health_service_singleton(self):
        """Test health service singleton behavior."""
        # Clear any existing instance
        import anomaly_detection.api.v1.health as health_module
        health_module._health_service = None
        
        service1 = get_health_service()
        service2 = get_health_service()
        
        assert service1 is service2


class TestHealthIntegration:
    """Integration tests for health endpoints."""
    
    def test_complete_health_monitoring_flow(self, client, mock_health_service):
        """Test complete health monitoring workflow."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            # 1. Start monitoring
            start_response = client.post("/api/v1/health/monitoring/start")
            assert start_response.status_code == 200
            
            # 2. Get health status
            status_response = client.get("/api/v1/health/status")
            assert status_response.status_code == 200
            
            # 3. Get full health report
            report_response = client.get("/api/v1/health/report")
            assert report_response.status_code == 200
            
            # 4. Check active alerts
            alerts_response = client.get("/api/v1/health/alerts")
            assert alerts_response.status_code == 200
            
            # 5. Update thresholds
            threshold_data = {
                "metric_name": "cpu_usage",
                "warning_threshold": 80.0,
                "critical_threshold": 95.0
            }
            threshold_response = client.post("/api/v1/health/thresholds", json=threshold_data)
            assert threshold_response.status_code == 200
            
            # 6. Record API call metrics
            metrics_response = client.post(
                "/api/v1/health/metrics/record?response_time_ms=150.0&success=true"
            )
            assert metrics_response.status_code == 200
            
            # 7. Stop monitoring
            stop_response = client.post("/api/v1/health/monitoring/stop")
            assert stop_response.status_code == 200
            
            # All operations should succeed
            assert all(r.status_code == 200 for r in [
                start_response, status_response, report_response, alerts_response,
                threshold_response, metrics_response, stop_response
            ])
    
    def test_alert_lifecycle_flow(self, client, mock_health_service):
        """Test alert lifecycle: create -> view -> resolve."""
        with patch('anomaly_detection.api.v1.health.get_health_service',
                  return_value=mock_health_service):
            
            # 1. Get active alerts
            alerts_response = client.get("/api/v1/health/alerts")
            assert alerts_response.status_code == 200
            alerts = alerts_response.json()
            assert len(alerts) == 1
            
            alert_id = alerts[0]["alert_id"]
            
            # 2. Resolve alert
            resolve_response = client.post(f"/api/v1/health/alerts/{alert_id}/resolve")
            assert resolve_response.status_code == 200
            
            # 3. Check alert history
            history_response = client.get("/api/v1/health/alerts/history?hours=1")
            assert history_response.status_code == 200