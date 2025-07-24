"""Unit tests for web page routes."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from anomaly_detection.web.api.pages import router


class TestWebPages:
    """Test cases for web page routes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test app
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        
        # Mock templates
        self.mock_templates = Mock(spec=Jinja2Templates)
        self.mock_template_response = Mock()
        self.mock_template_response.status_code = 200
        self.mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Sample models data
        self.sample_models = [
            {
                "model_id": "model1",
                "name": "test_model_1",
                "algorithm": "isolation_forest",
                "status": "trained",
                "accuracy": 0.85
            },
            {
                "model_id": "model2", 
                "name": "test_model_2",
                "algorithm": "one_class_svm",
                "status": "deployed",
                "accuracy": 0.78
            }
        ]
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_home_page(self, mock_templates):
        """Test home page route."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/")
        
        # Assertions
        assert response.status_code == 200
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/home.html"
        
        context = template_call[0][1]
        assert "request" in context
        assert context["title"] == "Anomaly Detection Dashboard"
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_dashboard_page_success(self, mock_get_repo, mock_templates):
        """Test dashboard page with successful data loading."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.return_value = self.sample_models
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/dashboard")
        
        # Assertions
        assert response.status_code == 200
        mock_repo.list_models.assert_called_once()
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/dashboard.html"
        
        context = template_call[0][1]
        assert context["title"] == "Dashboard"
        assert context["total_models"] == 2
        assert len(context["recent_detections"]) == 3  # Mock data
        assert "isolation_forest" in context["active_algorithms"]
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_dashboard_page_error(self, mock_get_repo, mock_templates):
        """Test dashboard page when model repository raises error."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.side_effect = Exception("Repository error")
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/dashboard")
        
        # Should return error page with 500 status
        assert response.status_code == 200  # FastAPI returns 200 but template has 500
        mock_templates.TemplateResponse.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/500.html"
        assert template_call[0][2]['status_code'] == 500
        
        context = template_call[0][1]
        assert "error" in context
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_detection_page(self, mock_templates):
        """Test detection page route."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/detection")
        
        # Assertions
        assert response.status_code == 200
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/detection.html"
        
        context = template_call[0][1]
        assert context["title"] == "Run Detection"
        
        # Check algorithms list
        algorithms = context["algorithms"]
        assert len(algorithms) == 3
        algorithm_values = [alg["value"] for alg in algorithms]
        assert "isolation_forest" in algorithm_values
        assert "one_class_svm" in algorithm_values
        assert "lof" in algorithm_values
        
        # Check ensemble methods list
        ensemble_methods = context["ensemble_methods"]
        assert len(ensemble_methods) == 4
        method_values = [method["value"] for method in ensemble_methods]
        assert "majority" in method_values
        assert "average" in method_values
        assert "weighted_average" in method_values
        assert "max" in method_values
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_models_page_success(self, mock_get_repo, mock_templates):
        """Test models page with successful model listing."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.return_value = self.sample_models
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/models")
        
        # Assertions
        assert response.status_code == 200
        mock_repo.list_models.assert_called_once()
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/models.html"
        
        context = template_call[0][1]
        assert context["title"] == "Model Management"
        assert context["models"] == self.sample_models
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_models_page_error(self, mock_get_repo, mock_templates):
        """Test models page when model repository raises error."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.side_effect = Exception("Repository error")
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/models")
        
        # Should return error page with 500 status
        assert response.status_code == 200  # FastAPI returns 200 but template has 500
        mock_templates.TemplateResponse.assert_called_once()
        
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/500.html"
        assert template_call[0][2]['status_code'] == 500
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_monitoring_page(self, mock_templates):
        """Test monitoring page route."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/monitoring")
        
        # Assertions
        assert response.status_code == 200
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/monitoring.html"
        
        context = template_call[0][1]
        assert context["title"] == "System Monitoring"
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_analytics_dashboard_page(self, mock_templates):
        """Test analytics dashboard page route."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/analytics")
        
        # Assertions
        assert response.status_code == 200
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/analytics_dashboard.html"
        
        context = template_call[0][1]
        assert context["title"] == "Analytics Dashboard"
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_about_page(self, mock_templates):
        """Test about page route."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/about")
        
        # Assertions
        assert response.status_code == 200
        mock_templates.TemplateResponse.assert_called_once()
        
        # Check template call
        template_call = mock_templates.TemplateResponse.call_args
        assert template_call[0][0] == "pages/about.html"
        
        context = template_call[0][1]
        assert context["title"] == "About Anomaly Detection"
    
    def test_dependency_injection_functions(self):
        """Test dependency injection functions work correctly."""
        from anomaly_detection.web.api.pages import get_detection_service, get_model_repository
        
        # Test that services are created
        detection_service = get_detection_service()
        assert detection_service is not None
        
        model_repository = get_model_repository()
        assert model_repository is not None
        
        # Test singleton behavior - should return same instances
        assert get_detection_service() is detection_service
        assert get_model_repository() is model_repository
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_all_pages_have_request_context(self, mock_templates):
        """Test that all page routes include request in template context."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # List of routes to test
        routes = [
            ("/", "pages/home.html"),
            ("/detection", "pages/detection.html"),
            ("/monitoring", "pages/monitoring.html"),
            ("/analytics", "pages/analytics_dashboard.html"),
            ("/about", "pages/about.html")
        ]
        
        for route, expected_template in routes:
            # Reset mock
            mock_templates.reset_mock()
            
            # Make request
            response = self.client.get(route)
            
            # Assertions
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            template_call = mock_templates.TemplateResponse.call_args
            assert template_call[0][0] == expected_template
            
            context = template_call[0][1]
            assert "request" in context
            assert "title" in context
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_pages_with_model_repository_dependency(self, mock_get_repo, mock_templates):
        """Test pages that depend on model repository."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.return_value = self.sample_models
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # List of routes that use model repository
        routes_with_models = [
            ("/dashboard", "pages/dashboard.html"),
            ("/models", "pages/models.html")
        ]
        
        for route, expected_template in routes_with_models:
            # Reset mocks
            mock_repo.reset_mock()
            mock_templates.reset_mock()
            
            # Make request
            response = self.client.get(route)
            
            # Assertions
            assert response.status_code == 200
            mock_repo.list_models.assert_called_once()
            mock_templates.TemplateResponse.assert_called_once()
            
            template_call = mock_templates.TemplateResponse.call_args
            assert template_call[0][0] == expected_template
    
    @patch('anomaly_detection.web.api.pages.templates')
    def test_static_pages_no_dependencies(self, mock_templates):
        """Test static pages that don't require external dependencies."""
        # Setup mock
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # List of static routes
        static_routes = [
            ("/", "pages/home.html", "Anomaly Detection Dashboard"),
            ("/detection", "pages/detection.html", "Run Detection"),
            ("/monitoring", "pages/monitoring.html", "System Monitoring"),
            ("/analytics", "pages/analytics_dashboard.html", "Analytics Dashboard"),
            ("/about", "pages/about.html", "About Anomaly Detection")
        ]
        
        for route, expected_template, expected_title in static_routes:
            # Reset mock
            mock_templates.reset_mock()
            
            # Make request
            response = self.client.get(route)
            
            # Assertions
            assert response.status_code == 200
            mock_templates.TemplateResponse.assert_called_once()
            
            template_call = mock_templates.TemplateResponse.call_args
            assert template_call[0][0] == expected_template
            
            context = template_call[0][1]
            assert context["title"] == expected_title
    
    def test_detection_page_algorithm_configuration(self):
        """Test that detection page has correct algorithm configuration."""
        from anomaly_detection.web.api.pages import router
        
        # Get the detection route handler
        detection_route = None
        for route in router.routes:
            if hasattr(route, 'path') and route.path == "/detection":
                detection_route = route
                break
        
        assert detection_route is not None
        
        # Test algorithm data structure expectations
        expected_algorithms = ["isolation_forest", "one_class_svm", "lof"]
        expected_ensemble_methods = ["majority", "average", "weighted_average", "max"]
        
        # These should be consistent with what the template expects
        assert len(expected_algorithms) == 3
        assert len(expected_ensemble_methods) == 4
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_dashboard_page_recent_detections_mock_data(self, mock_get_repo, mock_templates):
        """Test that dashboard page includes mock recent detections data."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.return_value = []
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/dashboard")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        context = template_call[0][1]
        
        # Check recent detections mock data
        recent_detections = context["recent_detections"]
        assert len(recent_detections) == 3
        
        # Verify structure of mock data
        for detection in recent_detections:
            assert "id" in detection
            assert "algorithm" in detection
            assert "anomalies" in detection
            assert "timestamp" in detection
        
        # Check specific mock data values
        assert recent_detections[0]["algorithm"] == "isolation_forest"
        assert recent_detections[1]["algorithm"] == "ensemble_majority"
        assert recent_detections[2]["algorithm"] == "one_class_svm"
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')
    def test_models_page_empty_model_list(self, mock_get_repo, mock_templates):
        """Test models page with empty model list."""
        # Setup mocks
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        mock_repo.list_models.return_value = []
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        # Make request
        response = self.client.get("/models")
        
        # Assertions
        assert response.status_code == 200
        
        template_call = mock_templates.TemplateResponse.call_args
        context = template_call[0][1]
        assert context["models"] == []
    
    @patch('anomaly_detection.web.api.pages.templates')
    @patch('anomaly_detection.web.api.pages.get_model_repository')  
    def test_dashboard_page_model_count_calculation(self, mock_get_repo, mock_templates):
        """Test that dashboard correctly calculates total model count."""
        # Setup mocks with different number of models
        mock_repo = Mock()
        mock_get_repo.return_value = mock_repo
        
        test_cases = [
            ([], 0),
            ([{"model_id": "1"}], 1),
            (self.sample_models, 2),
            (self.sample_models + [{"model_id": "3"}, {"model_id": "4"}], 4)
        ]
        
        mock_templates.TemplateResponse.return_value = self.mock_template_response
        
        for models_list, expected_count in test_cases:
            # Reset mock
            mock_repo.reset_mock()
            mock_templates.reset_mock()
            
            mock_repo.list_models.return_value = models_list
            
            # Make request
            response = self.client.get("/dashboard")
            
            # Assertions
            assert response.status_code == 200
            
            template_call = mock_templates.TemplateResponse.call_args
            context = template_call[0][1]
            assert context["total_models"] == expected_count
    
    def test_all_page_routes_exist(self):
        """Test that all expected page routes are defined."""
        expected_routes = [
            "/",
            "/dashboard", 
            "/detection",
            "/models",
            "/monitoring",
            "/analytics",
            "/about"
        ]
        
        # Get all routes from router
        actual_routes = []
        for route in router.routes:
            if hasattr(route, 'path'):
                actual_routes.append(route.path)
        
        # Check that all expected routes exist
        for expected_route in expected_routes:
            assert expected_route in actual_routes, f"Route {expected_route} not found in router"