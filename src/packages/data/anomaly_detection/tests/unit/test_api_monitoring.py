"""Unit tests for monitoring API endpoints."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from anomaly_detection.api.v1.monitoring import router


@pytest.fixture
def app():
    """Create FastAPI app with monitoring router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/monitoring")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector."""
    collector = Mock()
    
    collector.get_summary_stats.return_value = {
        "total_requests": 1500,
        "successful_requests": 1425,
        "failed_requests": 75,
        "success_rate": 0.95,
        "avg_response_time_ms": 125.5,
        "total_anomalies_detected": 350,
        "models_created": 25,
        "active_sessions": 8
    }
    
    return collector


@pytest.fixture
def mock_performance_monitor():
    """Create mock performance monitor."""
    monitor = Mock()
    
    # Mock performance summary
    monitor.get_performance_summary.return_value = {
        "avg_cpu_percent": 45.2,
        "avg_memory_mb": 512.8,
        "peak_memory_mb": 1024.0,
        "total_operations": 2500,
        "operations_per_second": 12.5,
        "error_rate": 0.03
    }
    
    # Mock operation stats
    monitor.get_operation_stats.return_value = {
        "total_calls": 500,
        "successful_calls": 485,
        "failed_calls": 15,
        "avg_duration_ms": 150.2,
        "min_duration_ms": 25.0,
        "max_duration_ms": 2500.0,
        "p95_duration_ms": 350.0
    }
    
    # Mock recent profiles
    mock_profile = Mock()
    mock_profile.timestamp = datetime(2024, 1, 1, 12, 0, 0)
    mock_profile.total_duration_ms = 145.5
    mock_profile.success = True
    mock_profile.memory_usage_mb = 256.2
    mock_profile.peak_memory_mb = 312.8
    
    monitor.get_recent_profiles.return_value = [mock_profile] * 5
    
    # Mock resource usage
    mock_usage = Mock()
    mock_usage.timestamp = datetime(2024, 1, 1, 12, 0, 0)
    mock_usage.cpu_percent = 42.5
    mock_usage.memory_mb = 512.0
    mock_usage.memory_percent = 65.2
    mock_usage.disk_io_read_mb = 25.5
    mock_usage.disk_io_write_mb = 12.3
    mock_usage.network_sent_mb = 5.2
    mock_usage.network_received_mb = 8.7
    
    monitor.get_resource_usage.return_value = [mock_usage] * 10
    
    return monitor


@pytest.fixture
def mock_monitoring_dashboard():
    """Create mock monitoring dashboard."""
    dashboard = Mock()
    
    # Mock dashboard summary
    mock_summary = Mock()
    mock_summary.overall_health_status = "healthy"
    mock_summary.healthy_checks = 12
    mock_summary.degraded_checks = 2
    mock_summary.unhealthy_checks = 1
    mock_summary.total_operations = 5000
    mock_summary.operations_last_hour = 250
    mock_summary.avg_response_time_ms = 135.2
    mock_summary.success_rate = 0.96
    mock_summary.current_memory_mb = 512.5
    mock_summary.current_cpu_percent = 38.2
    mock_summary.peak_memory_mb = 1024.0
    mock_summary.total_models = 45
    mock_summary.active_detections = 8
    mock_summary.anomalies_detected_today = 125
    mock_summary.active_alerts = 3
    mock_summary.recent_errors = 5
    mock_summary.slow_operations = 2
    mock_summary.generated_at = datetime(2024, 1, 1, 12, 0, 0)
    
    dashboard.get_dashboard_summary.return_value = mock_summary
    
    # Mock performance trends
    dashboard.get_performance_trends.return_value = {
        "response_times": [
            {"timestamp": "2024-01-01T11:00:00", "avg_ms": 120.5},
            {"timestamp": "2024-01-01T12:00:00", "avg_ms": 135.2}
        ],
        "throughput": [
            {"timestamp": "2024-01-01T11:00:00", "requests_per_minute": 45.2},
            {"timestamp": "2024-01-01T12:00:00", "requests_per_minute": 48.7}
        ],
        "error_rates": [
            {"timestamp": "2024-01-01T11:00:00", "error_rate": 0.02},
            {"timestamp": "2024-01-01T12:00:00", "error_rate": 0.04}
        ]
    }
    
    # Mock alert summary
    dashboard.get_alert_summary.return_value = {
        "active_alerts": [
            {
                "id": "alert-1",
                "severity": "warning",
                "message": "High memory usage detected",
                "timestamp": "2024-01-01T12:00:00"
            }
        ],
        "resolved_alerts_24h": 15,
        "alert_trends": {
            "last_hour": 2,
            "last_day": 18,
            "last_week": 125
        }
    }
    
    # Mock operation breakdown
    dashboard.get_operation_breakdown.return_value = {
        "by_type": {
            "detection": {"count": 1200, "avg_ms": 145.2, "success_rate": 0.98},
            "training": {"count": 50, "avg_ms": 2500.0, "success_rate": 0.96},
            "prediction": {"count": 800, "avg_ms": 85.5, "success_rate": 0.99}
        },
        "slowest_operations": [
            {"operation": "model_training", "avg_ms": 2500.0, "count": 50},
            {"operation": "ensemble_detection", "avg_ms": 285.2, "count": 200}
        ],
        "most_frequent": [
            {"operation": "anomaly_detection", "count": 1200, "avg_ms": 145.2},
            {"operation": "model_prediction", "count": 800, "avg_ms": 85.5}
        ]
    }
    
    return dashboard


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_get_metrics_success(self, client, mock_metrics_collector, mock_performance_monitor):
        """Test successful metrics retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_metrics_collector',
                  return_value=mock_metrics_collector), \
             patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_performance_monitor):
            
            response = client.get("/api/v1/monitoring/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "metrics_summary" in data
            assert "performance_summary" in data
            assert "timestamp" in data
            
            # Check metrics summary
            metrics = data["metrics_summary"]
            assert metrics["total_requests"] == 1500
            assert metrics["success_rate"] == 0.95
            assert metrics["avg_response_time_ms"] == 125.5
            
            # Check performance summary
            performance = data["performance_summary"]
            assert performance["avg_cpu_percent"] == 45.2
            assert performance["avg_memory_mb"] == 512.8
            assert performance["total_operations"] == 2500
            
            # Verify service methods were called
            mock_metrics_collector.get_summary_stats.assert_called_once()
            mock_performance_monitor.get_performance_summary.assert_called_once()
    
    def test_get_metrics_error(self, client):
        """Test metrics retrieval with error."""
        mock_collector = Mock()
        mock_collector.get_summary_stats.side_effect = Exception("Metrics error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_metrics_collector',
                  return_value=mock_collector):
            
            response = client.get("/api/v1/monitoring/metrics")
            
            assert response.status_code == 500
            assert "Failed to retrieve metrics" in response.json()["detail"]


class TestOperationPerformanceEndpoint:
    """Test operation performance endpoint."""
    
    def test_get_operation_performance_success(self, client, mock_performance_monitor):
        """Test successful operation performance retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_performance_monitor):
            
            response = client.get("/api/v1/monitoring/metrics/performance/anomaly_detection")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["operation"] == "anomaly_detection"
            assert "statistics" in data
            assert "recent_profiles" in data
            
            # Check statistics
            stats = data["statistics"]
            assert stats["total_calls"] == 500
            assert stats["successful_calls"] == 485
            assert stats["avg_duration_ms"] == 150.2
            
            # Check recent profiles
            profiles = data["recent_profiles"]
            assert len(profiles) == 5
            profile = profiles[0]
            assert profile["timestamp"] == "2024-01-01T12:00:00"
            assert profile["duration_ms"] == 145.5
            assert profile["success"] is True
            assert profile["memory_mb"] == 256.2
            
            # Verify service methods were called
            mock_performance_monitor.get_operation_stats.assert_called_once_with("anomaly_detection")
            mock_performance_monitor.get_recent_profiles.assert_called_once_with(
                operation="anomaly_detection",
                limit=10
            )
    
    def test_get_operation_performance_error(self, client):
        """Test operation performance with error."""
        mock_monitor = Mock()
        mock_monitor.get_operation_stats.side_effect = Exception("Performance error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_monitor):
            
            response = client.get("/api/v1/monitoring/metrics/performance/test_operation")
            
            assert response.status_code == 500
            assert "Failed to retrieve performance metrics" in response.json()["detail"]


class TestResourceUsageEndpoint:
    """Test resource usage endpoint."""
    
    def test_get_resource_usage_success(self, client, mock_performance_monitor):
        """Test successful resource usage retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_performance_monitor):
            
            response = client.get("/api/v1/monitoring/resources")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "resource_usage" in data
            assert "count" in data
            assert "period_hours" in data
            
            assert data["count"] == 10
            assert data["period_hours"] == 1
            
            # Check resource usage data
            usage_data = data["resource_usage"]
            assert len(usage_data) == 10
            
            usage = usage_data[0]
            assert usage["timestamp"] == "2024-01-01T12:00:00"
            assert usage["cpu_percent"] == 42.5
            assert usage["memory_mb"] == 512.0
            assert usage["memory_percent"] == 65.2
            assert usage["disk_io_read_mb"] == 25.5
            assert usage["network_sent_mb"] == 5.2
            
            # Verify service method was called with correct parameters
            call_args = mock_performance_monitor.get_resource_usage.call_args
            assert call_args[1]["limit"] == 60
            # Check that since parameter is approximately 1 hour ago
            since_time = call_args[1]["since"]
            assert isinstance(since_time, datetime)
    
    def test_get_resource_usage_error(self, client):
        """Test resource usage with error."""
        mock_monitor = Mock()
        mock_monitor.get_resource_usage.side_effect = Exception("Resource error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_monitor):
            
            response = client.get("/api/v1/monitoring/resources")
            
            assert response.status_code == 500
            assert "Failed to retrieve resource usage" in response.json()["detail"]


class TestDashboardSummaryEndpoint:
    """Test dashboard summary endpoint."""
    
    def test_get_dashboard_summary_success(self, client, mock_monitoring_dashboard):
        """Test successful dashboard summary retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/summary")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "summary" in data
            summary = data["summary"]
            
            assert summary["overall_health_status"] == "healthy"
            assert summary["healthy_checks"] == 12
            assert summary["degraded_checks"] == 2
            assert summary["unhealthy_checks"] == 1
            assert summary["total_operations"] == 5000
            assert summary["operations_last_hour"] == 250
            assert summary["avg_response_time_ms"] == 135.2
            assert summary["success_rate"] == 0.96
            assert summary["current_memory_mb"] == 512.5
            assert summary["current_cpu_percent"] == 38.2
            assert summary["peak_memory_mb"] == 1024.0
            assert summary["total_models"] == 45
            assert summary["active_detections"] == 8
            assert summary["anomalies_detected_today"] == 125
            assert summary["active_alerts"] == 3
            assert summary["recent_errors"] == 5
            assert summary["slow_operations"] == 2
            assert summary["generated_at"] == "2024-01-01T12:00:00"
            
            # Verify service method was called
            mock_monitoring_dashboard.get_dashboard_summary.assert_called_once()
    
    def test_get_dashboard_summary_error(self, client):
        """Test dashboard summary with error."""
        mock_dashboard = Mock()
        mock_dashboard.get_dashboard_summary.side_effect = Exception("Dashboard error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/summary")
            
            assert response.status_code == 500
            assert "Failed to retrieve dashboard summary" in response.json()["detail"]


class TestPerformanceTrendsEndpoint:
    """Test performance trends endpoint."""
    
    def test_get_performance_trends_success(self, client, mock_monitoring_dashboard):
        """Test successful performance trends retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/trends?hours=12")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "response_times" in data
            assert "throughput" in data
            assert "error_rates" in data
            
            # Check response times
            response_times = data["response_times"]
            assert len(response_times) == 2
            assert response_times[0]["avg_ms"] == 120.5
            
            # Check throughput
            throughput = data["throughput"]
            assert len(throughput) == 2
            assert throughput[0]["requests_per_minute"] == 45.2
            
            # Check error rates
            error_rates = data["error_rates"]
            assert len(error_rates) == 2
            assert error_rates[0]["error_rate"] == 0.02
            
            # Verify service method was called with correct hours
            mock_monitoring_dashboard.get_performance_trends.assert_called_once_with(12)
    
    def test_get_performance_trends_default_hours(self, client, mock_monitoring_dashboard):
        """Test performance trends with default hours parameter."""
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/trends")
            
            assert response.status_code == 200
            
            # Should use default of 24 hours
            mock_monitoring_dashboard.get_performance_trends.assert_called_once_with(24)
    
    def test_get_performance_trends_invalid_hours(self, client, mock_monitoring_dashboard):
        """Test performance trends with invalid hours parameter."""
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            # Test too small
            response = client.get("/api/v1/monitoring/dashboard/trends?hours=0")
            assert response.status_code == 400
            assert "must be between 1 and 168" in response.json()["detail"]
            
            # Test too large
            response = client.get("/api/v1/monitoring/dashboard/trends?hours=200")
            assert response.status_code == 400
            assert "must be between 1 and 168" in response.json()["detail"]
    
    def test_get_performance_trends_error(self, client):
        """Test performance trends with error."""
        mock_dashboard = Mock()
        mock_dashboard.get_performance_trends.side_effect = Exception("Trends error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/trends")
            
            assert response.status_code == 500
            assert "Failed to retrieve performance trends" in response.json()["detail"]


class TestAlertsEndpoint:
    """Test alerts endpoint."""
    
    def test_get_alerts_success(self, client, mock_monitoring_dashboard):
        """Test successful alerts retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/alerts")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "active_alerts" in data
            assert "resolved_alerts_24h" in data
            assert "alert_trends" in data
            
            # Check active alerts
            active_alerts = data["active_alerts"]
            assert len(active_alerts) == 1
            alert = active_alerts[0]
            assert alert["id"] == "alert-1"
            assert alert["severity"] == "warning"
            assert alert["message"] == "High memory usage detected"
            
            # Check alert trends
            trends = data["alert_trends"]
            assert trends["last_hour"] == 2
            assert trends["last_day"] == 18
            assert trends["last_week"] == 125
            
            # Verify service method was called
            mock_monitoring_dashboard.get_alert_summary.assert_called_once()
    
    def test_get_alerts_error(self, client):
        """Test alerts retrieval with error."""
        mock_dashboard = Mock()
        mock_dashboard.get_alert_summary.side_effect = Exception("Alerts error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/alerts")
            
            assert response.status_code == 500
            assert "Failed to retrieve alerts" in response.json()["detail"]


class TestOperationBreakdownEndpoint:
    """Test operation breakdown endpoint."""
    
    def test_get_operation_breakdown_success(self, client, mock_monitoring_dashboard):
        """Test successful operation breakdown retrieval."""
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/operations")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "by_type" in data
            assert "slowest_operations" in data
            assert "most_frequent" in data
            
            # Check by_type breakdown
            by_type = data["by_type"]
            assert "detection" in by_type
            assert "training" in by_type
            assert "prediction" in by_type
            
            detection_stats = by_type["detection"]
            assert detection_stats["count"] == 1200
            assert detection_stats["avg_ms"] == 145.2
            assert detection_stats["success_rate"] == 0.98
            
            # Check slowest operations
            slowest = data["slowest_operations"]
            assert len(slowest) == 2
            assert slowest[0]["operation"] == "model_training"
            assert slowest[0]["avg_ms"] == 2500.0
            
            # Check most frequent operations
            frequent = data["most_frequent"]
            assert len(frequent) == 2
            assert frequent[0]["operation"] == "anomaly_detection"
            assert frequent[0]["count"] == 1200
            
            # Verify service method was called
            mock_monitoring_dashboard.get_operation_breakdown.assert_called_once()
    
    def test_get_operation_breakdown_error(self, client):
        """Test operation breakdown with error."""
        mock_dashboard = Mock()
        mock_dashboard.get_operation_breakdown.side_effect = Exception("Breakdown error")
        
        with patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_dashboard):
            
            response = client.get("/api/v1/monitoring/dashboard/operations")
            
            assert response.status_code == 500
            assert "Failed to retrieve operation breakdown" in response.json()["detail"]


class TestMonitoringDependencies:
    """Test monitoring dependencies initialization."""
    
    @patch('anomaly_detection.api.v1.monitoring.get_metrics_collector')
    @patch('anomaly_detection.api.v1.monitoring.get_performance_monitor')
    @patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard')
    def test_dependencies_initialization(self, mock_dashboard_func, mock_monitor_func, mock_collector_func):
        """Test that monitoring dependencies are properly initialized."""
        # Import should trigger initialization
        import anomaly_detection.api.v1.monitoring as monitoring_module
        
        # Verify all dependency functions were called
        mock_collector_func.assert_called_once()
        mock_monitor_func.assert_called_once()
        mock_dashboard_func.assert_called_once()


class TestMonitoringIntegration:
    """Integration tests for monitoring endpoints."""
    
    def test_comprehensive_monitoring_flow(self, client, mock_metrics_collector, 
                                         mock_performance_monitor, mock_monitoring_dashboard):
        """Test comprehensive monitoring data flow."""
        with patch('anomaly_detection.api.v1.monitoring.get_metrics_collector',
                  return_value=mock_metrics_collector), \
             patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_performance_monitor), \
             patch('anomaly_detection.api.v1.monitoring.get_monitoring_dashboard',
                  return_value=mock_monitoring_dashboard):
            
            # 1. Get overall metrics
            metrics_response = client.get("/api/v1/monitoring/metrics")
            assert metrics_response.status_code == 200
            
            # 2. Get dashboard summary
            summary_response = client.get("/api/v1/monitoring/dashboard/summary")
            assert summary_response.status_code == 200
            
            # 3. Get performance trends
            trends_response = client.get("/api/v1/monitoring/dashboard/trends?hours=6")
            assert trends_response.status_code == 200
            
            # 4. Get resource usage
            resources_response = client.get("/api/v1/monitoring/resources")
            assert resources_response.status_code == 200
            
            # 5. Get operation performance
            operation_response = client.get("/api/v1/monitoring/metrics/performance/detection")
            assert operation_response.status_code == 200
            
            # 6. Get alerts
            alerts_response = client.get("/api/v1/monitoring/dashboard/alerts")
            assert alerts_response.status_code == 200
            
            # 7. Get operation breakdown
            breakdown_response = client.get("/api/v1/monitoring/dashboard/operations")
            assert breakdown_response.status_code == 200
            
            # All responses should be successful and contain expected data
            assert all(r.status_code == 200 for r in [
                metrics_response, summary_response, trends_response, 
                resources_response, operation_response, alerts_response, breakdown_response
            ])
    
    def test_monitoring_error_resilience(self, client):
        """Test monitoring system resilience to individual component failures."""
        # Mock one service to fail
        mock_collector = Mock()
        mock_collector.get_summary_stats.side_effect = Exception("Collector failed")
        
        # Other services work normally
        mock_monitor = Mock()
        mock_monitor.get_performance_summary.return_value = {"test": "data"}
        
        with patch('anomaly_detection.api.v1.monitoring.get_metrics_collector',
                  return_value=mock_collector), \
             patch('anomaly_detection.api.v1.monitoring.get_performance_monitor',
                  return_value=mock_monitor):
            
            # Metrics endpoint should fail gracefully
            response = client.get("/api/v1/monitoring/metrics")
            assert response.status_code == 500
            
            # But resource endpoint should still work
            response = client.get("/api/v1/monitoring/resources")
            # This would succeed if the performance monitor works independently