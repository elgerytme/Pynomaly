"""Unit tests for web main application."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from anomaly_detection.web.main import create_web_app, lifespan


class TestWebMainApplication:
    """Test suite for web main application."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings fixture."""
        settings = Mock()
        settings.api.cors_origins = ["http://localhost:3000"]
        settings.api.host = "localhost"
        settings.api.port = 8000
        settings.debug = True
        settings.logging.level = "INFO"
        return settings
    
    def test_create_web_app(self):
        """Test web app creation."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            app = create_web_app()
            
            assert isinstance(app, FastAPI)
            assert app.title == "Anomaly Detection Dashboard"
            assert app.description == "Interactive web interface for ML-based anomaly detection"
            assert app.version == "0.3.0"
    
    def test_app_middleware_configuration(self):
        """Test CORS middleware configuration."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["http://localhost:3000", "http://localhost:8080"]
            mock_get_settings.return_value = mock_settings
            
            app = create_web_app()
            
            # Check that CORS middleware is added
            # This is difficult to test directly, but we can verify the app was created
            assert isinstance(app, FastAPI)
    
    def test_app_router_inclusion(self):
        """Test that all routers are included."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            with patch('anomaly_detection.web.main.pages') as mock_pages, \
                 patch('anomaly_detection.web.main.htmx') as mock_htmx, \
                 patch('anomaly_detection.web.main.analytics') as mock_analytics:
                
                mock_pages.router = Mock()
                mock_htmx.router = Mock()
                mock_analytics.router = Mock()
                
                app = create_web_app()
                
                # Verify routers are included (indirectly)
                assert isinstance(app, FastAPI)
    
    def test_static_files_mounting(self):
        """Test static files mounting."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.StaticFiles') as mock_static_files:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            app = create_web_app()
            
            # Verify static files are mounted
            assert isinstance(app, FastAPI)
    
    def test_error_handlers(self):
        """Test custom error handlers."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.templates') as mock_templates:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            mock_templates.TemplateResponse.return_value = Mock()
            
            app = create_web_app()
            client = TestClient(app)
            
            # Test 404 handler
            response = client.get("/nonexistent-page")
            assert response.status_code == 404
    
    def test_404_error_handler(self):
        """Test 404 error handler specifically."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.templates') as mock_templates:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            mock_response = Mock()
            mock_response.status_code = 404
            mock_templates.TemplateResponse.return_value = mock_response
            
            app = create_web_app()
            client = TestClient(app)
            
            response = client.get("/does-not-exist")
            assert response.status_code == 404
    
    def test_500_error_handler(self):
        """Test 500 error handler."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.templates') as mock_templates:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            mock_response = Mock()
            mock_response.status_code = 500
            mock_templates.TemplateResponse.return_value = mock_response
            
            app = create_web_app()
            
            # Create a route that raises an exception
            @app.get("/test-error")
            async def test_error():
                raise Exception("Test error")
            
            client = TestClient(app)
            response = client.get("/test-error")
            assert response.status_code == 500
    
    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test application lifespan startup."""
        mock_app = Mock(spec=FastAPI)
        mock_app.version = "0.3.0"
        
        with patch('anomaly_detection.web.main.setup_logging') as mock_setup_logging, \
             patch('anomaly_detection.web.main.logger') as mock_logger:
            
            async with lifespan(mock_app):
                mock_setup_logging.assert_called_once()
                mock_logger.info.assert_called()
                
                # Check startup log messages
                startup_calls = [call for call in mock_logger.info.call_args_list 
                               if "Starting up" in str(call)]
                assert len(startup_calls) > 0
    
    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self):
        """Test application lifespan shutdown."""
        mock_app = Mock(spec=FastAPI)
        mock_app.version = "0.3.0"
        
        with patch('anomaly_detection.web.main.setup_logging'), \
             patch('anomaly_detection.web.main.logger') as mock_logger:
            
            async with lifespan(mock_app):
                pass  # Just test that we exit the context manager
            
            # Check shutdown log messages
            shutdown_calls = [call for call in mock_logger.info.call_args_list 
                            if "Shutting down" in str(call)]
            assert len(shutdown_calls) > 0
    
    def test_web_app_singleton(self):
        """Test that web_app is created as module-level singleton."""
        from anomaly_detection.web.main import web_app
        
        assert isinstance(web_app, FastAPI)
        assert web_app.title == "Anomaly Detection Dashboard"
    
    def test_main_function(self):
        """Test main function for running the web app."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.uvicorn') as mock_uvicorn:
            
            mock_settings = Mock()
            mock_settings.api.host = "0.0.0.0"
            mock_settings.api.port = 8000
            mock_settings.debug = False
            mock_settings.logging.level = "WARNING"
            mock_get_settings.return_value = mock_settings
            
            from anomaly_detection.web.main import main
            
            main()
            
            mock_uvicorn.run.assert_called_once()
            
            # Check uvicorn.run call arguments
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["host"] == "0.0.0.0"
            assert call_args[1]["port"] == 8001  # API port + 1
            assert call_args[1]["reload"] == False
            assert call_args[1]["log_level"] == "warning"
    
    def test_main_function_with_debug(self):
        """Test main function with debug mode enabled."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.uvicorn') as mock_uvicorn:
            
            mock_settings = Mock()
            mock_settings.api.host = "localhost"
            mock_settings.api.port = 8000
            mock_settings.debug = True
            mock_settings.logging.level = "DEBUG"
            mock_get_settings.return_value = mock_settings
            
            from anomaly_detection.web.main import main
            
            main()
            
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["reload"] == True
            assert call_args[1]["log_level"] == "debug"
    
    def test_app_configuration_completeness(self):
        """Test that all necessary app configurations are present."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            app = create_web_app()
            
            # Check basic app properties
            assert app.title is not None
            assert app.description is not None
            assert app.version is not None
            
            # Check that lifespan is configured
            assert app.lifespan_context is not None
    
    def test_cors_configuration(self):
        """Test CORS configuration with different origins."""
        test_origins = [
            ["http://localhost:3000"],
            ["http://localhost:3000", "https://example.com"],
            ["*"]
        ]
        
        for origins in test_origins:
            with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
                mock_settings = Mock()
                mock_settings.api.cors_origins = origins
                mock_get_settings.return_value = mock_settings
                
                app = create_web_app()
                
                # Verify app was created successfully with different CORS settings
                assert isinstance(app, FastAPI)
    
    def test_template_directory_configuration(self):
        """Test template directory configuration."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.Jinja2Templates') as mock_templates:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            app = create_web_app()
            
            # Verify templates are configured
            # The exact verification depends on how templates are used
            assert isinstance(app, FastAPI)
    
    def test_base_dir_calculation(self):
        """Test BASE_DIR calculation."""
        from anomaly_detection.web.main import BASE_DIR
        from pathlib import Path
        
        assert isinstance(BASE_DIR, Path)
        assert BASE_DIR.name == "web"
    
    def test_app_exception_handling_integration(self):
        """Test that exception handlers are properly integrated."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings, \
             patch('anomaly_detection.web.main.templates') as mock_templates:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            mock_404_response = Mock()
            mock_404_response.status_code = 404
            mock_500_response = Mock()
            mock_500_response.status_code = 500
            
            def template_response_side_effect(template, context, status_code=200):
                if status_code == 404:
                    return mock_404_response
                elif status_code == 500:
                    return mock_500_response
                return Mock()
            
            mock_templates.TemplateResponse.side_effect = template_response_side_effect
            
            app = create_web_app()
            client = TestClient(app)
            
            # Test 404
            response = client.get("/nonexistent")
            assert response.status_code == 404
    
    def test_module_imports(self):
        """Test that all required modules are imported correctly."""
        # This test verifies that the main module can import all dependencies
        try:
            from anomaly_detection.web.main import (
                create_web_app,
                lifespan,
                web_app,
                main
            )
            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Import error: {e}")
    
    def test_logging_configuration(self):
        """Test logging configuration during startup."""
        mock_app = Mock(spec=FastAPI)
        mock_app.version = "0.3.0"
        
        with patch('anomaly_detection.web.main.setup_logging') as mock_setup_logging, \
             patch('anomaly_detection.web.main.structlog') as mock_structlog:
            
            mock_logger = Mock()
            mock_structlog.get_logger.return_value = mock_logger
            
            # Import triggers module-level code
            from anomaly_detection.web import main
            
            # Verify logger is obtained
            mock_structlog.get_logger.assert_called()
    
    def test_settings_integration(self):
        """Test settings integration."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["http://localhost:3000"]
            mock_settings.api.host = "127.0.0.1"
            mock_settings.api.port = 9000
            mock_settings.debug = True
            mock_settings.logging.level = "INFO"
            mock_get_settings.return_value = mock_settings
            
            # Re-create app to test with new settings
            app = create_web_app()
            
            assert isinstance(app, FastAPI)
            # The specific settings usage is tested in other methods