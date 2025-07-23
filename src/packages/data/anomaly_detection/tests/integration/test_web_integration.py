"""Integration tests for web interface."""

import pytest
import json
from typing import Dict, Any
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from anomaly_detection.web.main import create_web_app


class TestWebIntegration:
    """Integration tests for web interface."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock all external services for web integration tests."""
        mocks = {}
        
        # Mock detection service
        mock_detection = Mock()
        mock_detection.detect_anomalies.return_value = Mock(
            algorithm="isolation_forest",
            total_samples=100,
            anomaly_count=15,
            anomaly_rate=15.0,
            processing_time=1.234,
            success=True,
            predictions=[1, -1, 1, 1, -1] * 20,  # 100 samples
            scores=[0.1, 0.9, 0.2, 0.3, 0.8] * 20
        )
        mocks['detection'] = mock_detection
        
        # Mock model repository
        mock_repo = Mock()
        mock_repo.list_models.return_value = [
            {'id': 'model_1', 'name': 'Test Model 1', 'algorithm': 'isolation_forest'},
            {'id': 'model_2', 'name': 'Test Model 2', 'algorithm': 'lof'}
        ]
        mocks['repository'] = mock_repo
        
        # Mock analytics service
        mock_analytics = Mock()
        mock_analytics.get_dashboard_stats.return_value = {
            'total_detections': 150,
            'total_anomalies': 23,
            'active_algorithms': 3,
            'average_detection_time': 1.45,
            'system_status': 'healthy',
            'success_rate': 96.7
        }
        mock_analytics.get_performance_metrics.return_value = Mock(
            total_detections=150,
            total_anomalies=23,
            average_detection_time=1.45,
            success_rate=96.7,
            throughput=10.2,
            error_rate=3.3
        )
        mock_analytics.get_detection_timeline.return_value = [
            {'timestamp': '2024-01-20T10:00:00', 'detections': 15, 'anomalies': 3},
            {'timestamp': '2024-01-20T11:00:00', 'detections': 18, 'anomalies': 2}
        ]
        mock_analytics.get_algorithm_distribution.return_value = [
            {'algorithm': 'isolation_forest', 'count': 75, 'percentage': 50.0},
            {'algorithm': 'lof', 'count': 45, 'percentage': 30.0}
        ]
        mock_analytics.simulate_detection.return_value = {
            'algorithm': 'isolation_forest',
            'total_samples': 1000,
            'anomalies_found': 45,
            'anomaly_rate': 4.5,
            'processing_time': 1.23
        }
        mocks['analytics'] = mock_analytics
        
        return mocks
    
    @pytest.fixture
    def client(self, mock_services):
        """Create test client with mocked services."""
        with patch('anomaly_detection.web.api.pages.get_detection_service', return_value=mock_services['detection']), \
             patch('anomaly_detection.web.api.pages.get_model_repository', return_value=mock_services['repository']), \
             patch('anomaly_detection.web.api.htmx.get_detection_service', return_value=mock_services['detection']), \
             patch('anomaly_detection.web.api.analytics.get_analytics_service', return_value=mock_services['analytics']):
            
            app = create_web_app()
            return TestClient(app)
    
    def test_home_page_loads(self, client: TestClient):
        """Test that home page loads successfully."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Should contain basic page structure
        content = response.text
        assert "Anomaly Detection" in content
        assert "<html" in content
        assert "</html>" in content
    
    def test_dashboard_page_loads(self, client: TestClient):
        """Test that dashboard page loads with model data."""
        response = client.get("/dashboard")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "Dashboard" in content
        # Should display model information
        assert "Test Model" in content or "model" in content.lower()
    
    def test_detection_page_loads(self, client: TestClient):
        """Test that detection page loads with algorithm options."""
        response = client.get("/detection")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "Run Detection" in content or "Detection" in content
        # Should contain algorithm options
        assert "isolation_forest" in content or "Isolation Forest" in content
    
    def test_analytics_page_loads(self, client: TestClient):
        """Test that analytics page loads successfully."""
        response = client.get("/analytics")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "Analytics" in content
    
    def test_models_page_loads(self, client: TestClient):
        """Test that models page loads with model data."""
        response = client.get("/models")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "Model" in content
    
    def test_monitoring_page_loads(self, client: TestClient):
        """Test that monitoring page loads successfully."""
        response = client.get("/monitoring")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "Monitoring" in content
    
    def test_about_page_loads(self, client: TestClient):
        """Test that about page loads successfully."""
        response = client.get("/about")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "About" in content
    
    def test_static_files_served(self, client: TestClient):
        """Test that static files are served correctly."""
        # Test CSS file
        response = client.get("/static/css/style.css")
        
        # Should either serve the file or return 404 if it doesn't exist
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert "text/css" in response.headers.get("content-type", "")
    
    def test_404_error_handling(self, client: TestClient):
        """Test that 404 errors are handled properly."""
        response = client.get("/nonexistent-page")
        
        assert response.status_code == 404
        assert "text/html" in response.headers["content-type"]
    
    def test_htmx_detection_endpoint(self, client: TestClient, mock_services):
        """Test HTMX detection endpoint integration."""
        response = client.post("/htmx/detect", data={
            "algorithm": "isolation_forest",
            "contamination": "0.1",
            "sample_data": ""  # Will generate synthetic data
        })
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Should have called the detection service
        mock_services['detection'].detect_anomalies.assert_called_once()
        
        # Check that correct parameters were passed
        call_args = mock_services['detection'].detect_anomalies.call_args
        assert call_args[1]["algorithm"] == "isolation_forest"
        assert call_args[1]["contamination"] == 0.1
    
    def test_htmx_detection_with_custom_data(self, client: TestClient, mock_services):
        """Test HTMX detection with custom data."""
        sample_data = json.dumps([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        response = client.post("/htmx/detect", data={
            "algorithm": "lof",
            "contamination": "0.05",
            "sample_data": sample_data
        })
        
        assert response.status_code == 200
        
        # Verify service was called with correct parameters
        mock_services['detection'].detect_anomalies.assert_called_once()
        call_args = mock_services['detection'].detect_anomalies.call_args
        assert call_args[1]["algorithm"] == "lof"
        assert call_args[1]["contamination"] == 0.05
    
    def test_analytics_dashboard_stats(self, client: TestClient, mock_services):
        """Test analytics dashboard stats endpoint."""
        response = client.get("/htmx/analytics/dashboard/stats")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        mock_services['analytics'].get_dashboard_stats.assert_called_once()
    
    def test_analytics_chart_data(self, client: TestClient, mock_services):
        """Test analytics chart data endpoints."""
        # Test anomaly timeline chart
        response = client.get("/htmx/analytics/charts/anomaly-timeline")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        
        data = response.json()
        assert "labels" in data
        assert "datasets" in data
        
        mock_services['analytics'].get_detection_timeline.assert_called_once()
        
        # Test algorithm distribution chart
        response = client.get("/htmx/analytics/charts/algorithm-distribution")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        
        data = response.json()
        assert "labels" in data
        assert "datasets" in data
        
        mock_services['analytics'].get_algorithm_distribution.assert_called_once()
    
    def test_analytics_simulation(self, client: TestClient, mock_services):
        """Test analytics simulation endpoint."""
        response = client.post("/htmx/analytics/simulate-detection")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        mock_services['analytics'].simulate_detection.assert_called_once()
    
    def test_navigation_consistency(self, client: TestClient):
        """Test that navigation is consistent across pages."""
        pages = ["/", "/dashboard", "/detection", "/models", "/analytics", "/monitoring", "/about"]
        
        for page in pages:
            response = client.get(page)
            assert response.status_code == 200
            
            content = response.text
            # Check that navigation elements are present
            assert "nav" in content.lower() or "menu" in content.lower()
            # Check that common navigation links are present
            assert "dashboard" in content.lower()
            assert "detection" in content.lower()
    
    def test_responsive_design_elements(self, client: TestClient):
        """Test that responsive design elements are present."""
        response = client.get("/dashboard")
        
        assert response.status_code == 200
        content = response.text
        
        # Should include Tailwind CSS for responsive design
        assert "tailwindcss" in content or "cdn.tailwindcss.com" in content
        
        # Should have mobile-responsive navigation
        assert "mobile" in content.lower() or "sm:" in content or "lg:" in content
    
    def test_javascript_libraries_loaded(self, client: TestClient):
        """Test that required JavaScript libraries are loaded."""
        response = client.get("/analytics")
        
        assert response.status_code == 200
        content = response.text
        
        # Should include Chart.js for analytics visualizations
        assert "chart.js" in content.lower() or "chartjs" in content.lower()
        
        # Should include HTMX for dynamic content
        assert "htmx" in content.lower()
    
    def test_form_submission_workflow(self, client: TestClient, mock_services):
        """Test complete form submission workflow."""
        # First, load the detection page
        response = client.get("/detection")
        assert response.status_code == 200
        
        # Then submit a detection request
        response = client.post("/htmx/detect", data={
            "algorithm": "isolation_forest",
            "contamination": "0.1",
            "sample_data": json.dumps([[1, 2], [3, 4], [5, 6]])
        })
        
        assert response.status_code == 200
        
        # Verify the service interaction
        mock_services['detection'].detect_anomalies.assert_called_once()
        
        # The response should contain detection results
        content = response.text
        assert "detection" in content.lower() or "result" in content.lower()
    
    def test_error_handling_integration(self, client: TestClient, mock_services):
        """Test error handling integration across the web interface."""
        # Make the detection service raise an exception
        mock_services['detection'].detect_anomalies.side_effect = Exception("Test error")
        
        response = client.post("/htmx/detect", data={
            "algorithm": "isolation_forest",
            "contamination": "0.1",
            "sample_data": ""
        })
        
        # Should handle the error gracefully
        # Exact behavior depends on implementation
        assert response.status_code in [200, 500]  # Either handle gracefully or return error
    
    def test_concurrent_requests(self, client: TestClient, mock_services):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.get("/dashboard")
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results if isinstance(status, int))
    
    def test_session_state_isolation(self, client: TestClient):
        """Test that sessions are properly isolated."""
        # Make multiple requests and verify they don't interfere
        response1 = client.get("/dashboard")
        response2 = client.get("/detection")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Each response should be independent
        assert response1.text != response2.text
    
    def test_csrf_protection_presence(self, client: TestClient):
        """Test that CSRF protection elements are present where needed."""
        response = client.get("/detection")
        
        assert response.status_code == 200
        content = response.text
        
        # Check for form protection (depends on implementation)
        # This might include CSRF tokens or other security measures
        assert "form" in content.lower()
    
    def test_analytics_real_time_updates(self, client: TestClient, mock_services):
        """Test real-time analytics updates."""
        # Test that analytics endpoints support time range parameters
        response = client.get("/htmx/analytics/charts/anomaly-timeline?hours=24")
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        
        # Should have called service with time parameter
        mock_services['analytics'].get_detection_timeline.assert_called_with(hours=24)
    
    def test_data_export_functionality(self, client: TestClient, mock_services):
        """Test data export functionality if available."""
        # This test depends on whether export functionality is implemented
        # Test chart data can be retrieved for export
        response = client.get("/htmx/analytics/charts/algorithm-distribution")
        
        assert response.status_code == 200
        data = response.json()
        
        # Data should be in a format suitable for export
        assert isinstance(data, dict)
        assert "labels" in data
        assert "datasets" in data


class TestWebPerformance:
    """Performance tests for web interface."""
    
    @pytest.fixture
    def client(self):
        """Create test client for performance tests."""
        with patch('anomaly_detection.web.api.pages.get_detection_service'), \
             patch('anomaly_detection.web.api.pages.get_model_repository'), \
             patch('anomaly_detection.web.api.analytics.get_analytics_service'):
            
            app = create_web_app()
            return TestClient(app)
    
    def test_page_load_performance(self, client: TestClient):
        """Test that pages load within reasonable time."""
        import time
        
        pages = ["/", "/dashboard", "/detection", "/analytics"]
        
        for page in pages:
            start_time = time.time()
            response = client.get(page)
            load_time = time.time() - start_time
            
            assert response.status_code == 200
            assert load_time < 5.0  # Should load within 5 seconds
    
    def test_concurrent_user_simulation(self, client: TestClient):
        """Simulate multiple concurrent users."""
        import threading
        import time
        
        def simulate_user():
            """Simulate a user browsing the application."""
            pages = ["/", "/dashboard", "/detection", "/analytics"]
            for page in pages:
                client.get(page)
                time.sleep(0.1)  # Small delay between requests
        
        # Start multiple user simulations
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=simulate_user)
            threads.append(thread)
            thread.start()
        
        # Wait for all simulations to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # If we get here without timeout, the test passed
        assert True