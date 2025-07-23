"""Unit tests for web pages API endpoints."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import Request

from anomaly_detection.web.api.pages import (
    router,
    get_detection_service,
    get_model_repository
)
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository


class TestPagesAPI:
    """Test suite for pages API endpoints."""
    
    @pytest.fixture
    def mock_detection_service(self):
        """Mock detection service fixture."""
        service = Mock(spec=DetectionService)
        return service
    
    @pytest.fixture
    def mock_model_repository(self):
        """Mock model repository fixture."""
        repo = Mock(spec=ModelRepository)
        repo.list_models.return_value = [
            {'id': 'model_1', 'name': 'Isolation Forest Model', 'algorithm': 'isolation_forest'},
            {'id': 'model_2', 'name': 'LOF Model', 'algorithm': 'lof'},
            {'id': 'model_3', 'name': 'SVM Model', 'algorithm': 'one_class_svm'}
        ]
        return repo
    
    @pytest.fixture
    def override_dependencies(self, mock_detection_service, mock_model_repository):
        """Override the service dependencies."""
        def _get_detection_service():
            return mock_detection_service
        
        def _get_model_repository():
            return mock_model_repository
        
        router.dependency_overrides[get_detection_service] = _get_detection_service
        router.dependency_overrides[get_model_repository] = _get_model_repository
        yield
        router.dependency_overrides.clear()
    
    @pytest.fixture
    def client(self, override_dependencies):
        """Test client fixture."""
        from fastapi import FastAPI
        from fastapi.templating import Jinja2Templates
        from pathlib import Path
        
        app = FastAPI()
        
        # Mock static files and templates
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_templates.TemplateResponse.return_value = Mock()
            mock_templates.TemplateResponse.return_value.status_code = 200
            mock_templates.TemplateResponse.return_value.headers = {"content-type": "text/html"}
            
            app.include_router(router)
            return TestClient(app)
    
    def test_home_page(self, client):
        """Test home page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/")
            
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/home.html"
            assert "request" in args[1]
            assert args[1]["title"] == "Anomaly Detection Dashboard"
    
    def test_dashboard_page(self, client, mock_model_repository):
        """Test dashboard page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/dashboard")
            
            assert response.status_code == 200
            mock_model_repository.list_models.assert_called_once()
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/dashboard.html"
            assert "request" in args[1]
            assert args[1]["title"] == "Dashboard"
            assert args[1]["total_models"] == 3
            assert "recent_detections" in args[1]
            assert "active_algorithms" in args[1]
    
    def test_dashboard_page_error_handling(self, client, mock_model_repository):
        """Test dashboard page error handling."""
        mock_model_repository.list_models.side_effect = Exception("Database error")
        
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/dashboard")
            
            # Should render error page
            mock_templates.TemplateResponse.assert_called()
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/500.html"
            assert "error" in args[1]
    
    def test_detection_page(self, client):
        """Test detection page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/detection")
            
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/detection.html"
            assert "request" in args[1]
            assert args[1]["title"] == "Run Detection"
            assert "algorithms" in args[1]
            assert "ensemble_methods" in args[1]
            
            # Check algorithm options
            algorithms = args[1]["algorithms"]
            assert len(algorithms) == 3
            algorithm_values = [alg["value"] for alg in algorithms]
            assert "isolation_forest" in algorithm_values
            assert "one_class_svm" in algorithm_values
            assert "lof" in algorithm_values
            
            # Check ensemble options
            ensemble_methods = args[1]["ensemble_methods"]
            assert len(ensemble_methods) == 4
            ensemble_values = [method["value"] for method in ensemble_methods]
            assert "majority" in ensemble_values
            assert "average" in ensemble_values
            assert "weighted_average" in ensemble_values
            assert "max" in ensemble_values
    
    def test_models_page(self, client, mock_model_repository):
        """Test models page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/models")
            
            assert response.status_code == 200
            mock_model_repository.list_models.assert_called_once()
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/models.html"
            assert "request" in args[1]
            assert args[1]["title"] == "Model Management"
            assert args[1]["models"] == mock_model_repository.list_models.return_value
    
    def test_models_page_error_handling(self, client, mock_model_repository):
        """Test models page error handling."""
        mock_model_repository.list_models.side_effect = Exception("Repository error")
        
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/models")
            
            # Should render error page
            mock_templates.TemplateResponse.assert_called()
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/500.html"
            assert "error" in args[1]
    
    def test_monitoring_page(self, client):
        """Test monitoring page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/monitoring")
            
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/monitoring.html"
            assert "request" in args[1]
            assert args[1]["title"] == "System Monitoring"
    
    def test_analytics_page(self, client):
        """Test analytics page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/analytics")
            
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/analytics_dashboard.html"
            assert "request" in args[1]
            assert args[1]["title"] == "Analytics Dashboard"
    
    def test_about_page(self, client):
        """Test about page endpoint."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/about")
            
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            # Check template call arguments
            args, kwargs = mock_templates.TemplateResponse.call_args
            assert args[0] == "pages/about.html"
            assert "request" in args[1]
            assert args[1]["title"] == "About Anomaly Detection"
    
    def test_dependency_injection_detection_service(self):
        """Test detection service dependency injection."""
        # Test default dependency
        service = get_detection_service()
        assert isinstance(service, DetectionService)
        
        # Test that it returns the same instance (singleton behavior)
        service2 = get_detection_service()
        assert service is service2
    
    def test_dependency_injection_model_repository(self):
        """Test model repository dependency injection."""
        # Test default dependency
        repo = get_model_repository()
        assert isinstance(repo, ModelRepository)
        
        # Test that it returns the same instance (singleton behavior)
        repo2 = get_model_repository()
        assert repo is repo2
    
    def test_dashboard_recent_detections_format(self, client, mock_model_repository):
        """Test the format of recent detections in dashboard."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/dashboard")
            
            args, kwargs = mock_templates.TemplateResponse.call_args
            recent_detections = args[1]["recent_detections"]
            
            # Check format of recent detections
            assert isinstance(recent_detections, list)
            assert len(recent_detections) == 3
            
            for detection in recent_detections:
                assert "id" in detection
                assert "algorithm" in detection
                assert "anomalies" in detection
                assert "timestamp" in detection
                assert detection["id"].startswith("det_")
    
    def test_dashboard_active_algorithms_format(self, client, mock_model_repository):
        """Test the format of active algorithms in dashboard."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/dashboard")
            
            args, kwargs = mock_templates.TemplateResponse.call_args
            active_algorithms = args[1]["active_algorithms"]
            
            # Check active algorithms list
            assert isinstance(active_algorithms, list)
            assert len(active_algorithms) == 4
            assert "isolation_forest" in active_algorithms
            assert "one_class_svm" in active_algorithms
            assert "lof" in active_algorithms
            assert "ensemble" in active_algorithms
    
    def test_algorithm_options_completeness(self, client):
        """Test that all algorithm options are provided correctly."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/detection")
            
            args, kwargs = mock_templates.TemplateResponse.call_args
            algorithms = args[1]["algorithms"]
            
            # Verify all algorithms have required fields
            for algorithm in algorithms:
                assert "value" in algorithm
                assert "label" in algorithm
                assert isinstance(algorithm["value"], str)
                assert isinstance(algorithm["label"], str)
                assert len(algorithm["value"]) > 0
                assert len(algorithm["label"]) > 0
    
    def test_ensemble_methods_completeness(self, client):
        """Test that all ensemble methods are provided correctly."""
        with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.get("/detection")
            
            args, kwargs = mock_templates.TemplateResponse.call_args
            ensemble_methods = args[1]["ensemble_methods"]
            
            # Verify all ensemble methods have required fields
            for method in ensemble_methods:
                assert "value" in method
                assert "label" in method
                assert isinstance(method["value"], str)
                assert isinstance(method["label"], str)
                assert len(method["value"]) > 0
                assert len(method["label"]) > 0
    
    def test_template_path_consistency(self, client):
        """Test that template paths are consistent and follow conventions."""
        endpoints_and_templates = [
            ("/", "pages/home.html"),
            ("/dashboard", "pages/dashboard.html"),
            ("/detection", "pages/detection.html"),
            ("/models", "pages/models.html"),
            ("/monitoring", "pages/monitoring.html"),
            ("/analytics", "pages/analytics_dashboard.html"),
            ("/about", "pages/about.html")
        ]
        
        for endpoint, expected_template in endpoints_and_templates:
            with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_templates.TemplateResponse.return_value = mock_response
                
                response = client.get(endpoint)
                
                # Check that correct template is used
                args, kwargs = mock_templates.TemplateResponse.call_args
                assert args[0] == expected_template
    
    def test_request_context_in_templates(self, client):
        """Test that request context is properly passed to all templates."""
        endpoints = ["/", "/dashboard", "/detection", "/models", "/monitoring", "/analytics", "/about"]
        
        for endpoint in endpoints:
            with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_templates.TemplateResponse.return_value = mock_response
                
                response = client.get(endpoint)
                
                # Check that request is in template context
                args, kwargs = mock_templates.TemplateResponse.call_args
                assert "request" in args[1]
                assert "title" in args[1]
    
    def test_error_handling_consistency(self, client, mock_model_repository):
        """Test that error handling is consistent across endpoints."""
        # Test both dashboard and models page error handling
        error_endpoints = ["/dashboard", "/models"]
        
        for endpoint in error_endpoints:
            mock_model_repository.list_models.side_effect = Exception("Test error")
            
            with patch('anomaly_detection.web.api.pages.templates') as mock_templates:
                mock_response = Mock()
                mock_response.status_code = 500
                mock_templates.TemplateResponse.return_value = mock_response
                
                response = client.get(endpoint)
                
                # Should render 500 error page
                args, kwargs = mock_templates.TemplateResponse.call_args
                assert args[0] == "pages/500.html"
                assert "request" in args[1]
                assert "error" in args[1]
            
            # Reset mock for next iteration
            mock_model_repository.list_models.side_effect = None
            mock_model_repository.list_models.return_value = []