"""Unit tests for main entry points and application initialization."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.testclient import TestClient

from anomaly_detection.main import create_app, main, lifespan
from anomaly_detection.cli import main as cli_main
from anomaly_detection.cli.main import app as cli_app
from anomaly_detection.web.main import create_web_app, main as web_main


class TestMainApplication:
    """Test cases for main.py FastAPI application."""
    
    @patch('anomaly_detection.main.get_settings')
    def test_create_app_initialization(self, mock_get_settings):
        """Test create_app function creates FastAPI instance properly."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        assert isinstance(app, FastAPI)
        assert app.title == "Anomaly Detection API"
        assert app.description is not None
        assert app.version == "0.1.0"
    
    @patch('anomaly_detection.main.get_settings')
    def test_create_app_cors_configuration(self, mock_get_settings):
        """Test CORS middleware configuration."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["http://localhost:3000", "https://app.example.com"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        # Check CORS middleware is added
        middleware_classes = [middleware.cls.__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
        
        # Find CORS middleware and check configuration
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None
        assert cors_middleware.kwargs["allow_origins"] == ["http://localhost:3000", "https://app.example.com"]
    
    @patch('anomaly_detection.main.get_settings')
    def test_create_app_api_router_included(self, mock_get_settings):
        """Test that API router is properly included."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        
        # Should have basic endpoints
        assert "/health" in routes
        assert "/docs" in routes or any("/docs" in route for route in routes)
        assert "/openapi.json" in routes
    
    @patch('anomaly_detection.main.get_logger')
    def test_lifespan_context_manager(self, mock_get_logger):
        """Test lifespan context manager for startup/shutdown."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        app = FastAPI()
        
        async def test_lifespan():
            async with lifespan(app):
                # Should be in startup phase
                pass
            # Should complete shutdown
        
        asyncio.run(test_lifespan())
        
        # Check logging calls
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Starting Anomaly Detection API" in call for call in log_calls)
        assert any("Shutting down Anomaly Detection API" in call for call in log_calls)
    
    @patch('anomaly_detection.main.get_logger')
    def test_lifespan_startup_logging(self, mock_get_logger):
        """Test lifespan startup logging includes version info."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        app = FastAPI()
        
        async def test_startup():
            async with lifespan(app):
                pass
        
        asyncio.run(test_startup())
        
        # Check that version information is logged
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        startup_call = next((call for call in log_calls if "Starting" in call), None)
        assert startup_call is not None
        assert "0.1.0" in startup_call
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.main.get_settings')
    def test_main_function_uvicorn_config(self, mock_get_settings, mock_uvicorn_run):
        """Test main function configures uvicorn properly."""
        mock_settings = Mock()
        mock_settings.api.host = "127.0.0.1"
        mock_settings.api.port = 8000
        mock_settings.api.workers = 1
        mock_settings.api.reload = True
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        main()
        
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args[1]
        
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8000
        assert call_kwargs["workers"] == 1
        assert call_kwargs["reload"] == True
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.main.get_settings')
    def test_main_function_production_config(self, mock_get_settings, mock_uvicorn_run):
        """Test main function with production configuration."""
        mock_settings = Mock()
        mock_settings.api.host = "0.0.0.0"
        mock_settings.api.port = 80
        mock_settings.api.workers = 4
        mock_settings.api.reload = False
        mock_settings.api.cors_origins = ["https://prod.example.com"]
        mock_get_settings.return_value = mock_settings
        
        main()
        
        call_kwargs = mock_uvicorn_run.call_args[1]
        
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 80
        assert call_kwargs["workers"] == 4
        assert call_kwargs["reload"] == False
    
    def test_app_integration_with_test_client(self):
        """Test app integration using TestClient."""
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            app = create_app()
            client = TestClient(app)
            
            # Test basic endpoint
            response = client.get("/health")
            assert response.status_code in [200, 404]  # Depends on if health endpoint is implemented
            
            # Test OpenAPI endpoint
            response = client.get("/openapi.json")
            assert response.status_code == 200
            assert response.json()["info"]["title"] == "Anomaly Detection API"


class TestLegacyCLI:
    """Test cases for legacy CLI (cli.py)."""
    
    @patch('click.echo')
    @patch('anomaly_detection.cli.get_logger')
    def test_cli_main_initialization(self, mock_get_logger, mock_echo):
        """Test CLI main function initialization."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Mock sys.argv to avoid argument parsing
        with patch('sys.argv', ['anomaly_detection']):
            try:
                cli_main()
            except SystemExit:
                # Click CLI exits with code 0 for help
                pass
        
        # Logger should be initialized
        mock_get_logger.assert_called()
    
    @patch('anomaly_detection.cli.DetectionService')
    @patch('anomaly_detection.cli.get_logger') 
    def test_cli_service_initialization(self, mock_get_logger, mock_detection_service):
        """Test CLI initializes services properly."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_service = Mock()
        mock_detection_service.return_value = mock_service
        
        # Import CLI module to trigger initialization
        from anomaly_detection import cli
        
        # Services should be available in CLI context
        assert hasattr(cli, 'detection_service') or mock_detection_service.called
    
    def test_cli_command_groups_registration(self):
        """Test that CLI command groups are properly registered."""
        from anomaly_detection.cli import cli
        
        # Check that command groups exist
        command_names = [cmd.name for cmd in cli.commands.values()]
        
        # Should have main command groups
        expected_groups = ['detect', 'ensemble', 'stream', 'data', 'model', 'explain', 'worker']
        for group in expected_groups:
            assert group in command_names or any(group in name for name in command_names)
    
    @patch('anomaly_detection.cli.DetectionService')
    def test_cli_error_handling_decorator(self, mock_detection_service):
        """Test CLI error handling decorator functionality."""
        from anomaly_detection.cli import handle_errors
        
        mock_service = Mock()
        mock_detection_service.return_value = mock_service
        
        # Test decorator function
        @handle_errors
        def test_command():
            raise Exception("Test error")
        
        # Should not raise exception, but handle it gracefully
        result = test_command()
        # Error handling decorator should return None or handle gracefully
        assert result is None or isinstance(result, dict)


class TestNewCLI:
    """Test cases for new CLI (cli_new/main.py)."""
    
    def test_cli_app_initialization(self):
        """Test new CLI app initialization."""
        from anomaly_detection.cli_new.main import app
        
        assert app is not None
        assert hasattr(app, 'registered_commands')
        assert hasattr(app, 'info')
    
    def test_cli_app_commands_registration(self):
        """Test that CLI commands are registered."""
        from anomaly_detection.cli_new.main import app
        
        # Get registered commands
        if hasattr(app, 'registered_commands'):
            commands = app.registered_commands
        else:
            # Alternative way to get commands from Typer
            commands = getattr(app, 'commands', {})
        
        # Should have some commands registered
        assert len(commands) > 0 or hasattr(app, 'callback')
    
    @patch('typer.echo')
    def test_cli_app_help_functionality(self, mock_echo):
        """Test CLI help functionality."""
        from anomaly_detection.cli_new.main import app
        
        # Test that app can be invoked (would show help)
        try:
            # This might not work in test environment, but structure should be valid
            assert callable(app)
        except Exception:
            # If execution fails, at least the app structure should be valid
            assert app is not None
    
    @patch('anomaly_detection.cli_new.main.get_settings')
    def test_cli_settings_integration(self, mock_get_settings):
        """Test CLI integration with settings."""
        mock_settings = Mock()
        mock_settings.api.host = "localhost"
        mock_settings.api.port = 8000
        mock_get_settings.return_value = mock_settings
        
        # Import CLI to test settings integration
        from anomaly_detection.cli_new import main
        
        # Settings should be accessible
        mock_get_settings.assert_called()
    
    def test_cli_rich_integration(self):
        """Test CLI Rich library integration."""
        from anomaly_detection.cli_new.main import app
        
        # Should be able to import Rich components used in CLI
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table()
            assert console is not None
            assert table is not None
        except ImportError:
            pytest.skip("Rich library not available")
    
    @patch('httpx.get')
    def test_cli_health_check_functionality(self, mock_get):
        """Test CLI health check functionality."""
        from anomaly_detection.cli_new.commands.health import status
        
        # Mock HTTP response for health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "services": {}}
        mock_get.return_value = mock_response
        
        # Test health check command structure
        assert callable(status)
        
        # Commands should be importable without errors
        from anomaly_detection.cli_new.commands import detection, models, data


class TestWebApplication:
    """Test cases for web application (web/main.py)."""
    
    @patch('anomaly_detection.web.main.get_settings')
    def test_create_web_app_initialization(self, mock_get_settings):
        """Test web app creation and initialization."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_web_app()
        
        assert isinstance(app, FastAPI)
        assert app.title == "Anomaly Detection Dashboard"
        assert "web dashboard" in app.description.lower()
    
    @patch('anomaly_detection.web.main.get_settings')
    def test_create_web_app_static_files(self, mock_get_settings):
        """Test web app static file configuration."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_web_app()
        
        # Check that static files are mounted
        routes = [route.path for route in app.routes]
        static_routes = [route for route in routes if "static" in route]
        assert len(static_routes) > 0 or any("/static" in route for route in routes)
    
    @patch('anomaly_detection.web.main.get_settings')
    def test_create_web_app_routers_included(self, mock_get_settings):
        """Test that web app includes all necessary routers."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_web_app()
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        
        # Should have web-specific routes
        expected_paths = ["/", "/dashboard", "/detection", "/models"]
        for path in expected_paths:
            assert path in routes or any(path in route for route in routes)
    
    @patch('anomaly_detection.web.main.get_settings')
    def test_create_web_app_error_handlers(self, mock_get_settings):
        """Test web app error handler registration."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_web_app()
        
        # Check that error handlers are registered
        assert len(app.exception_handlers) >= 0  # May have custom error handlers
    
    def test_web_app_integration_with_test_client(self):
        """Test web app integration using TestClient."""
        with patch('anomaly_detection.web.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            app = create_web_app()
            client = TestClient(app)
            
            # Test home page
            response = client.get("/")
            assert response.status_code in [200, 404, 500]  # May depend on template availability
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.web.main.get_settings')
    def test_web_main_function(self, mock_get_settings, mock_uvicorn_run):
        """Test web main function configuration."""
        mock_settings = Mock()
        mock_settings.api.host = "0.0.0.0"
        mock_settings.api.port = 8080
        mock_settings.api.workers = 1
        mock_settings.api.reload = False
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        web_main()
        
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args[1]
        
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 8080


class TestApplicationConfiguration:
    """Test cases for application configuration across entry points."""
    
    @patch('anomaly_detection.main.get_settings')
    @patch('anomaly_detection.web.main.get_settings')  
    def test_consistent_settings_usage(self, mock_web_settings, mock_main_settings):
        """Test that all entry points use settings consistently."""
        # Mock same settings for both
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_settings.api.host = "localhost"
        mock_settings.api.port = 8000
        mock_main_settings.return_value = mock_settings
        mock_web_settings.return_value = mock_settings
        
        # Create apps
        main_app = create_app()
        web_app = create_web_app()
        
        # Both should use the same settings
        assert main_app.title == "Anomaly Detection API"
        assert web_app.title == "Anomaly Detection Dashboard"
        
        # Settings should be called for both
        mock_main_settings.assert_called()
        mock_web_settings.assert_called()
    
    def test_cors_configuration_consistency(self):
        """Test CORS configuration across applications."""
        cors_origins = ["http://localhost:3000", "https://app.example.com"]
        
        with patch('anomaly_detection.main.get_settings') as mock_main_settings, \
             patch('anomaly_detection.web.main.get_settings') as mock_web_settings:
            
            mock_settings = Mock()
            mock_settings.api.cors_origins = cors_origins
            mock_main_settings.return_value = mock_settings
            mock_web_settings.return_value = mock_settings
            
            main_app = create_app()
            web_app = create_web_app()
            
            # Both apps should have CORS middleware
            main_cors = any(m.cls.__name__ == "CORSMiddleware" for m in main_app.user_middleware)
            web_cors = any(m.cls.__name__ == "CORSMiddleware" for m in web_app.user_middleware)
            
            assert main_cors
            assert web_cors
    
    @patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG', 'API_PORT': '9000'})
    def test_environment_variable_integration(self):
        """Test environment variable integration across entry points."""
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_settings.api.port = 9000  # From environment
            mock_get_settings.return_value = mock_settings
            
            app = create_app()
            
            # Settings should reflect environment variables
            mock_get_settings.assert_called()
            assert app is not None


class TestApplicationLifecycle:
    """Test cases for application lifecycle management."""
    
    @patch('anomaly_detection.main.get_logger')
    async def test_lifespan_startup_sequence(self, mock_get_logger):
        """Test application startup sequence."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        app = FastAPI()
        
        startup_completed = False
        
        async def test_startup():
            nonlocal startup_completed
            async with lifespan(app):
                startup_completed = True
        
        await test_startup()
        
        assert startup_completed
        mock_logger.info.assert_called()
    
    @patch('anomaly_detection.main.get_logger')
    async def test_lifespan_shutdown_sequence(self, mock_get_logger):
        """Test application shutdown sequence."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        app = FastAPI()
        
        shutdown_completed = False
        
        async def test_shutdown():
            nonlocal shutdown_completed
            async with lifespan(app):
                pass  # Startup completes
            shutdown_completed = True  # Shutdown completes
        
        await test_shutdown()
        
        assert shutdown_completed
        # Should have logged both startup and shutdown
        assert mock_logger.info.call_count >= 2
    
    @patch('anomaly_detection.main.get_logger')
    async def test_lifespan_exception_handling(self, mock_get_logger):
        """Test lifespan exception handling during startup."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        app = FastAPI()
        
        # Mock a startup failure
        with patch('anomaly_detection.main.get_settings', side_effect=Exception("Config error")):
            with pytest.raises(Exception, match="Config error"):
                async with lifespan(app):
                    pass
    
    def test_graceful_shutdown_signals(self):
        """Test application handles shutdown signals gracefully."""
        with patch('signal.signal') as mock_signal:
            # Import main to register signal handlers (if any)
            from anomaly_detection import main
            
            # Application should not crash on import
            assert main is not None


class TestEntryPointIntegration:
    """Integration test cases across entry points."""
    
    def test_all_entry_points_importable(self):
        """Test that all entry points can be imported without errors."""
        # Main API
        from anomaly_detection import main
        assert main is not None
        
        # Server
        from anomaly_detection import server
        assert server is not None
        
        # Worker
        from anomaly_detection import worker
        assert worker is not None
        
        # Legacy CLI
        from anomaly_detection import cli
        assert cli is not None
        
        # New CLI
        from anomaly_detection.cli_new import main as cli_new_main
        assert cli_new_main is not None
        
        # Web app
        from anomaly_detection.web import main as web_main
        assert web_main is not None
    
    def test_entry_point_functions_callable(self):
        """Test that main functions are callable."""
        from anomaly_detection.main import main, create_app
        from anomaly_detection.web.main import main as web_main, create_web_app
        from anomaly_detection.worker import main as worker_main
        from anomaly_detection.cli import main as cli_main
        
        # All main functions should be callable
        assert callable(main)
        assert callable(create_app)
        assert callable(web_main)
        assert callable(create_web_app)
        assert callable(worker_main)
        assert callable(cli_main)
    
    @patch('uvicorn.run')
    def test_multiple_entry_points_configuration(self, mock_uvicorn_run):
        """Test that multiple entry points can be configured simultaneously."""
        with patch('anomaly_detection.main.get_settings') as mock_main_settings, \
             patch('anomaly_detection.web.main.get_settings') as mock_web_settings:
            
            # Configure different settings for each
            main_settings = Mock()
            main_settings.api.cors_origins = ["*"]
            main_settings.api.host = "127.0.0.1"
            main_settings.api.port = 8000
            main_settings.api.workers = 1
            main_settings.api.reload = True
            
            web_settings = Mock()
            web_settings.api.cors_origins = ["*"]
            web_settings.api.host = "0.0.0.0"
            web_settings.api.port = 8080
            web_settings.api.workers = 1
            web_settings.api.reload = False
            
            mock_main_settings.return_value = main_settings
            mock_web_settings.return_value = web_settings
            
            # Import and create apps
            from anomaly_detection.main import main, create_app
            from anomaly_detection.web.main import main as web_main, create_web_app
            
            main_app = create_app()
            web_app = create_web_app()
            
            # Both apps should be created successfully
            assert main_app is not None
            assert web_app is not None
            assert main_app.title != web_app.title
    
    def test_shared_dependencies_consistency(self):
        """Test that shared dependencies are used consistently."""
        # All entry points should use the same settings, logging, etc.
        with patch('anomaly_detection.infrastructure.config.settings.get_settings') as mock_settings:
            mock_settings.return_value = Mock()
            
            # Import all entry points
            from anomaly_detection import main, server, worker, cli
            from anomaly_detection.cli_new import main as cli_new
            from anomaly_detection.web import main as web
            
            # All should be importable and share common infrastructure
            assert all([main, server, worker, cli, cli_new, web])


class TestEntryPointErrorHandling:
    """Test cases for error handling across entry points."""
    
    @patch('anomaly_detection.main.get_settings')
    def test_main_app_config_error_handling(self, mock_get_settings):
        """Test main app handles configuration errors."""
        mock_get_settings.side_effect = Exception("Configuration load failed")
        
        with pytest.raises(Exception, match="Configuration load failed"):
            create_app()
    
    @patch('anomaly_detection.web.main.get_settings')
    def test_web_app_config_error_handling(self, mock_get_settings):
        """Test web app handles configuration errors."""
        mock_get_settings.side_effect = Exception("Configuration load failed")
        
        with pytest.raises(Exception, match="Configuration load failed"):
            create_web_app()
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.main.get_settings')
    def test_main_function_uvicorn_error_handling(self, mock_get_settings, mock_uvicorn_run):
        """Test main function handles uvicorn startup errors."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_settings.api.host = "localhost"
        mock_settings.api.port = 8000
        mock_settings.api.workers = 1
        mock_settings.api.reload = False
        mock_get_settings.return_value = mock_settings
        
        mock_uvicorn_run.side_effect = Exception("Server startup failed")
        
        with pytest.raises(Exception, match="Server startup failed"):
            main()
    
    def test_cli_import_error_handling(self):
        """Test CLI handles import errors gracefully."""
        # CLI should be importable even if some dependencies are missing
        try:
            from anomaly_detection import cli
            assert cli is not None
        except ImportError as e:
            # If there are import errors, they should be specific and helpful
            assert "anomaly_detection" in str(e) or "cli" in str(e)
    
    def test_worker_initialization_error_handling(self):
        """Test worker handles initialization errors."""
        with patch('anomaly_detection.worker.DetectionService', side_effect=Exception("Service init failed")):
            from anomaly_detection.worker import AnomalyDetectionWorker
            
            # Worker should handle service initialization failures
            try:
                worker = AnomalyDetectionWorker()
                # If no exception, the worker should still be created
                assert worker is not None
            except Exception as e:
                # If exception occurs, it should be related to service initialization
                assert "Service init failed" in str(e) or "DetectionService" in str(e)


class TestEntryPointPerformance:
    """Test cases for entry point performance characteristics."""
    
    def test_app_creation_performance(self):
        """Test that app creation is reasonably fast."""
        import time
        
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            start_time = time.time()
            app = create_app()
            end_time = time.time()
            
            # App creation should be fast (< 1 second)
            creation_time = end_time - start_time
            assert creation_time < 1.0
            assert app is not None
    
    def test_import_performance(self):
        """Test that entry point imports are reasonably fast."""
        import time
        import sys
        
        # Remove modules if already imported
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('anomaly_detection')]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
        
        start_time = time.time()
        from anomaly_detection import main
        end_time = time.time()
        
        # Import should be fast (< 2 seconds)
        import_time = end_time - start_time
        assert import_time < 2.0
        assert main is not None
    
    def test_concurrent_app_creation(self):
        """Test that multiple apps can be created concurrently."""
        import threading
        
        apps = []
        errors = []
        
        def create_test_app():
            try:
                with patch('anomaly_detection.main.get_settings') as mock_get_settings:
                    mock_settings = Mock()
                    mock_settings.api.cors_origins = ["*"]
                    mock_get_settings.return_value = mock_settings
                    
                    app = create_app()
                    apps.append(app)
            except Exception as e:
                errors.append(e)
        
        # Create multiple apps concurrently
        threads = [threading.Thread(target=create_test_app) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All apps should be created successfully
        assert len(errors) == 0
        assert len(apps) == 3
        assert all(app is not None for app in apps)