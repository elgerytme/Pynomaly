"""Unit tests for web analytics API endpoints."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import Request

from anomaly_detection.web.api.analytics import router, get_analytics_service
from anomaly_detection.domain.services.analytics_service import (
    AnalyticsService,
    PerformanceMetrics,
    AlgorithmStats,
    DataQualityMetrics
)


class TestAnalyticsAPI:
    """Test suite for analytics API endpoints."""
    
    @pytest.fixture
    def mock_analytics_service(self):
        """Mock analytics service fixture."""
        service = Mock(spec=AnalyticsService)
        
        # Mock dashboard stats
        service.get_dashboard_stats.return_value = {
            'total_detections': 150,
            'total_anomalies': 23,
            'active_algorithms': 3,
            'average_detection_time': 1.45,
            'system_status': 'healthy',
            'success_rate': 96.7
        }
        
        # Mock performance metrics
        service.get_performance_metrics.return_value = PerformanceMetrics(
            total_detections=150,
            total_anomalies=23,
            average_detection_time=1.45,
            success_rate=96.7,
            throughput=10.2,
            error_rate=3.3
        )
        
        # Mock algorithm performance
        service.get_algorithm_performance.return_value = [
            AlgorithmStats(
                algorithm="isolation_forest",
                detections_count=75,
                anomalies_found=12,
                average_score=0.82,
                average_time=1.2,
                success_rate=98.0,
                last_used=datetime.now()
            ),
            AlgorithmStats(
                algorithm="lof",
                detections_count=45,
                anomalies_found=8,
                average_score=0.78,
                average_time=1.8,
                success_rate=95.5,
                last_used=datetime.now()
            )
        ]
        
        # Mock data quality metrics
        service.get_data_quality_metrics.return_value = DataQualityMetrics(
            total_samples=10000,
            missing_values=15,
            duplicate_samples=5,
            outliers_count=125,
            data_drift_events=2,
            quality_score=92.5
        )
        
        # Mock detection timeline
        service.get_detection_timeline.return_value = [
            {'timestamp': '2024-01-20T10:00:00', 'detections': 15, 'anomalies': 3},
            {'timestamp': '2024-01-20T11:00:00', 'detections': 18, 'anomalies': 2},
            {'timestamp': '2024-01-20T12:00:00', 'detections': 12, 'anomalies': 4}
        ]
        
        # Mock algorithm distribution
        service.get_algorithm_distribution.return_value = [
            {'algorithm': 'isolation_forest', 'count': 75, 'percentage': 50.0},
            {'algorithm': 'lof', 'count': 45, 'percentage': 30.0},
            {'algorithm': 'one_class_svm', 'count': 30, 'percentage': 20.0}
        ]
        
        # Mock performance trend
        service.get_performance_trend.return_value = [
            {'timestamp': '2024-01-20T10:00:00', 'processing_time': 1.2, 'throughput': 10.5, 'success_rate': 97.0},
            {'timestamp': '2024-01-20T11:00:00', 'processing_time': 1.3, 'throughput': 9.8, 'success_rate': 96.5},
            {'timestamp': '2024-01-20T12:00:00', 'processing_time': 1.1, 'throughput': 11.2, 'success_rate': 98.0}
        ]
        
        # Mock system status
        service.get_system_status.return_value = {
            'overall_status': 'healthy',
            'api_status': 'online',
            'database_status': 'connected',
            'memory_usage': '65.2%',
            'cpu_usage': '42.1%',
            'disk_usage': '28.7%',
            'active_operations': 5,
            'success_rate': '96.7%',
            'last_check': '2024-01-20T12:00:00'
        }
        
        # Mock simulation result
        service.simulate_detection.return_value = {
            'algorithm': 'isolation_forest',
            'total_samples': 1000,
            'anomalies_found': 45,
            'anomaly_rate': 4.5,
            'processing_time': 1.23
        }
        
        return service
    
    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request fixture."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        return request
    
    @pytest.fixture
    def override_dependency(self, mock_analytics_service):
        """Override the analytics service dependency."""
        def _get_analytics_service():
            return mock_analytics_service
        
        router.dependency_overrides[get_analytics_service] = _get_analytics_service
        yield
        router.dependency_overrides.clear()
    
    @pytest.fixture
    def client(self, override_dependency):
        """Test client fixture."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/htmx/analytics")
        return TestClient(app)
    
    def test_dashboard_stats_endpoint(self, client, mock_analytics_service):
        """Test dashboard stats endpoint."""
        response = client.get("/htmx/analytics/dashboard/stats")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_analytics_service.get_dashboard_stats.assert_called_once()
    
    def test_performance_metrics_endpoint(self, client, mock_analytics_service):
        """Test performance metrics endpoint."""
        response = client.get("/htmx/analytics/dashboard/performance")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_analytics_service.get_performance_metrics.assert_called_once()
    
    def test_algorithm_performance_endpoint(self, client, mock_analytics_service):
        """Test algorithm performance endpoint."""
        response = client.get("/htmx/analytics/dashboard/algorithms")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_analytics_service.get_algorithm_performance.assert_called_once()
    
    def test_data_quality_endpoint(self, client, mock_analytics_service):
        """Test data quality endpoint."""
        response = client.get("/htmx/analytics/dashboard/data-quality")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_analytics_service.get_data_quality_metrics.assert_called_once()
    
    def test_system_status_endpoint(self, client, mock_analytics_service):
        """Test system status endpoint."""
        response = client.get("/htmx/analytics/health/system-status")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_analytics_service.get_system_status.assert_called_once()
    
    def test_anomaly_timeline_chart_endpoint(self, client, mock_analytics_service):
        """Test anomaly timeline chart endpoint."""
        response = client.get("/htmx/analytics/charts/anomaly-timeline")
        
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type
        
        data = response.json()
        assert "labels" in data
        assert "datasets" in data
        
        mock_analytics_service.get_detection_timeline.assert_called_once_with(hours=24)
    
    def test_algorithm_distribution_chart_endpoint(self, client, mock_analytics_service):
        """Test algorithm distribution chart endpoint."""
        response = client.get("/htmx/analytics/charts/algorithm-distribution")
        
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type
        
        data = response.json()
        assert "labels" in data
        assert "datasets" in data
        
        mock_analytics_service.get_algorithm_distribution.assert_called_once()
    
    def test_performance_trend_chart_endpoint(self, client, mock_analytics_service):
        """Test performance trend chart endpoint."""
        response = client.get("/htmx/analytics/charts/performance-trend")
        
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type
        
        data = response.json()
        assert "labels" in data
        assert "datasets" in data
        
        mock_analytics_service.get_performance_trend.assert_called_once_with(hours=24)
    
    def test_simulate_detection_endpoint(self, client, mock_analytics_service):
        """Test simulate detection endpoint."""
        response = client.post("/htmx/analytics/simulate-detection")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_analytics_service.simulate_detection.assert_called_once()
    
    def test_dashboard_stats_error_handling(self, client):
        """Test error handling in dashboard stats endpoint."""
        with patch('anomaly_detection.web.api.analytics.get_analytics_service') as mock_get:
            mock_service = Mock()
            mock_service.get_dashboard_stats.side_effect = Exception("Database error")
            mock_get.return_value = mock_service
            
            response = client.get("/htmx/analytics/dashboard/stats")
            
            assert response.status_code == 500
            assert "text/html" in response.headers["content-type"]
    
    def test_chart_data_error_handling(self, client):
        """Test error handling in chart data endpoints."""
        with patch('anomaly_detection.web.api.analytics.get_analytics_service') as mock_get:
            mock_service = Mock()
            mock_service.get_detection_timeline.side_effect = Exception("Service unavailable")
            mock_get.return_value = mock_service
            
            response = client.get("/htmx/analytics/charts/anomaly-timeline")
            
            assert response.status_code == 500
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type
            
            data = response.json()
            assert "error" in data
    
    def test_chart_data_format_anomaly_timeline(self, client, mock_analytics_service):
        """Test chart data format for anomaly timeline."""
        response = client.get("/htmx/analytics/charts/anomaly-timeline")
        data = response.json()
        
        # Verify chart.js compatible format
        assert isinstance(data["labels"], list)
        assert isinstance(data["datasets"], list)
        assert len(data["datasets"]) == 2  # Detections and anomalies
        
        for dataset in data["datasets"]:
            assert "label" in dataset
            assert "data" in dataset
            assert "borderColor" in dataset
            assert "backgroundColor" in dataset
    
    def test_chart_data_format_algorithm_distribution(self, client, mock_analytics_service):
        """Test chart data format for algorithm distribution."""
        response = client.get("/htmx/analytics/charts/algorithm-distribution")
        data = response.json()
        
        # Verify chart.js compatible format for pie/doughnut chart
        assert isinstance(data["labels"], list)
        assert isinstance(data["datasets"], list)
        assert len(data["datasets"]) == 1
        
        dataset = data["datasets"][0]
        assert "data" in dataset
        assert "backgroundColor" in dataset
        assert isinstance(dataset["backgroundColor"], list)
    
    def test_chart_data_format_performance_trend(self, client, mock_analytics_service):
        """Test chart data format for performance trend."""
        response = client.get("/htmx/analytics/charts/performance-trend")
        data = response.json()
        
        # Verify chart.js compatible format for multi-line chart
        assert isinstance(data["labels"], list)
        assert isinstance(data["datasets"], list)
        assert len(data["datasets"]) == 3  # Processing time, throughput, success rate
        
        for dataset in data["datasets"]:
            assert "label" in dataset
            assert "data" in dataset
            assert "borderColor" in dataset
            assert "yAxisID" in dataset
    
    def test_dependency_injection(self):
        """Test that dependency injection works correctly."""
        # Test default dependency
        service = get_analytics_service()
        assert isinstance(service, AnalyticsService)
        
        # Test that it returns the same instance (singleton behavior)
        service2 = get_analytics_service()
        assert service is service2
    
    def test_cors_headers(self, client):
        """Test CORS headers in responses."""
        response = client.get("/htmx/analytics/dashboard/stats")
        
        # HTMX responses should allow cross-origin requests for partial updates
        # This depends on the FastAPI app configuration
        assert response.status_code == 200
    
    def test_empty_data_handling(self, client):
        """Test handling of empty data scenarios."""
        with patch('anomaly_detection.web.api.analytics.get_analytics_service') as mock_get:
            mock_service = Mock()
            mock_service.get_algorithm_distribution.return_value = []
            mock_service.get_detection_timeline.return_value = []
            mock_service.get_performance_trend.return_value = []
            mock_get.return_value = mock_service
            
            # Test algorithm distribution with empty data
            response = client.get("/htmx/analytics/charts/algorithm-distribution")
            assert response.status_code == 200
            data = response.json()
            assert data["labels"] == []
            assert len(data["datasets"]) == 1
            assert data["datasets"][0]["data"] == []
            
            # Test timeline with empty data
            response = client.get("/htmx/analytics/charts/anomaly-timeline")
            assert response.status_code == 200
            data = response.json()
            assert data["labels"] == []
            assert all(len(dataset["data"]) == 0 for dataset in data["datasets"])
    
    def test_time_range_parameters(self, client, mock_analytics_service):
        """Test time range parameters in chart endpoints."""
        # Test with custom hours parameter
        response = client.get("/htmx/analytics/charts/anomaly-timeline?hours=48")
        assert response.status_code == 200
        mock_analytics_service.get_detection_timeline.assert_called_with(hours=48)
        
        response = client.get("/htmx/analytics/charts/performance-trend?hours=12")
        assert response.status_code == 200
        mock_analytics_service.get_performance_trend.assert_called_with(hours=12)
    
    def test_simulation_result_format(self, client, mock_analytics_service):
        """Test simulation result format."""
        response = client.post("/htmx/analytics/simulate-detection")
        assert response.status_code == 200
        
        # The response should be HTML content for HTMX
        content = response.text
        assert "isolation_forest" in content  # Algorithm name should be in response
        assert "1000" in content  # Total samples
        assert "45" in content   # Anomalies found
        assert "4.5" in content  # Anomaly rate
        assert "1.23" in content # Processing time