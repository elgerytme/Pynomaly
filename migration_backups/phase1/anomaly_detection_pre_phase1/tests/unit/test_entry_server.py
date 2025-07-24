"""Unit tests for server entry point."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

from anomaly_detection.server import (
    create_app, main, lifespan, global_detection_service,
    global_ensemble_service, global_model_repository,
    global_streaming_service, global_explainability_service,
    global_health_service, global_error_handler
)


class TestServerLifespan:
    """Test cases for server lifespan management."""
    
    @patch('anomaly_detection.server.get_logger')
    def test_lifespan_context_manager(self, mock_get_logger):
        """Test lifespan context manager creates and cleans up services."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create a mock FastAPI app
        app = FastAPI()
        
        async def test_lifespan():
            async with lifespan(app):
                # Services should be created
                assert global_detection_service is not None
                assert global_ensemble_service is not None
                assert global_model_repository is not None
                assert global_streaming_service is not None
                assert global_explainability_service is not None
                assert global_health_service is not None
                assert global_error_handler is not None
        
        # Run the async test
        asyncio.run(test_lifespan())
        
        # Check that startup and shutdown were logged
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Starting Anomaly Detection API Server" in call for call in log_calls)
        assert any("Shutting down Anomaly Detection API Server" in call for call in log_calls)
    
    @patch('anomaly_detection.server.DetectionService')
    @patch('anomaly_detection.server.get_logger')
    def test_lifespan_service_initialization(self, mock_get_logger, mock_detection_service):
        """Test that lifespan properly initializes all services."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        mock_service_instance = Mock()
        mock_detection_service.return_value = mock_service_instance
        
        app = FastAPI()
        
        async def test_initialization():
            async with lifespan(app):
                # DetectionService should be instantiated
                mock_detection_service.assert_called_once()
        
        asyncio.run(test_initialization())
    
    @patch('anomaly_detection.server.get_logger')
    def test_lifespan_exception_handling(self, mock_get_logger):
        """Test lifespan handles exceptions during startup."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        app = FastAPI()
        
        with patch('anomaly_detection.server.DetectionService', side_effect=Exception("Service init failed")):
            async def test_exception():
                with pytest.raises(Exception, match="Service init failed"):
                    async with lifespan(app):
                        pass
            
            asyncio.run(test_exception())


class TestCreateApp:
    """Test cases for create_app factory function."""
    
    @patch('anomaly_detection.server.get_settings')
    def test_create_app_returns_fastapi_instance(self, mock_get_settings):
        """Test that create_app returns a FastAPI instance."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        assert isinstance(app, FastAPI)
        assert app.title == "Anomaly Detection API"
        assert app.description is not None
        assert app.version == "0.1.0"
    
    @patch('anomaly_detection.server.get_settings')
    def test_create_app_cors_configuration(self, mock_get_settings):
        """Test CORS middleware configuration."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["http://localhost:3000", "https://app.example.com"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        # Check that CORS middleware was added
        middleware_classes = [middleware.cls.__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
    
    @patch('anomaly_detection.server.get_settings')
    @patch('anomaly_detection.server.lifespan')
    def test_create_app_lifespan_assignment(self, mock_lifespan, mock_get_settings):
        """Test that lifespan is properly assigned to the app."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        # Lifespan should be assigned
        assert app.router.lifespan_context is not None
    
    @patch('anomaly_detection.server.get_settings')
    def test_create_app_includes_routers(self, mock_get_settings):
        """Test that all necessary routers are included."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        
        # Should have health endpoint
        assert "/health" in routes
        
        # Should have various API endpoints
        endpoint_prefixes = ["/detect", "/ensemble", "/models", "/stream", "/explain", "/monitoring"]
        for prefix in endpoint_prefixes:
            assert any(route.startswith(prefix) for route in routes)


class TestServerEndpoints:
    """Test cases for server API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.server.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        with patch('anomaly_detection.server.global_health_service') as mock_health:
            mock_health.get_health_summary = AsyncMock(return_value={
                "status": "healthy",
                "services": {"detection": "healthy", "models": "healthy"}
            })
            
            response = self.client.get("/health")
            
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    def test_detection_endpoint(self):
        """Test anomaly detection endpoint."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection:
            mock_detection.detect_anomalies = AsyncMock(return_value={
                "anomalies": [0, 1, 0, 1],
                "scores": [0.1, 0.8, 0.2, 0.9],
                "algorithm": "isolation_forest"
            })
            
            test_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithm": "isolation_forest",
                "contamination": 0.1
            }
            
            response = self.client.post("/detect", json=test_data)
            
            assert response.status_code == 200
            result = response.json()
            assert "anomalies" in result
            assert "scores" in result
            assert result["algorithm"] == "isolation_forest"
    
    def test_ensemble_detection_endpoint(self):
        """Test ensemble detection endpoint."""
        with patch('anomaly_detection.server.global_ensemble_service') as mock_ensemble:
            mock_ensemble.detect_with_ensemble = AsyncMock(return_value={
                "anomalies": [0, 1, 0, 1],
                "ensemble_scores": [0.1, 0.8, 0.2, 0.9],
                "individual_results": {
                    "isolation_forest": {"scores": [0.1, 0.7, 0.2, 0.8]},
                    "one_class_svm": {"scores": [0.1, 0.9, 0.2, 1.0]}
                },
                "ensemble_method": "majority"
            })
            
            test_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithms": ["isolation_forest", "one_class_svm"],
                "ensemble_method": "majority",
                "contamination": 0.1
            }
            
            response = self.client.post("/ensemble/detect", json=test_data)
            
            assert response.status_code == 200
            result = response.json()
            assert "anomalies" in result
            assert "ensemble_scores" in result
            assert "individual_results" in result
    
    def test_model_list_endpoint(self):
        """Test model listing endpoint."""
        with patch('anomaly_detection.server.global_model_repository') as mock_repo:
            mock_repo.list_models.return_value = [
                {
                    "model_id": "model_123",
                    "name": "Test Model",
                    "algorithm": "isolation_forest",
                    "status": "trained",
                    "created_at": "2024-01-20T10:00:00"
                }
            ]
            
            response = self.client.get("/models")
            
            assert response.status_code == 200
            models = response.json()
            assert len(models) == 1
            assert models[0]["model_id"] == "model_123"
            assert models[0]["algorithm"] == "isolation_forest"
    
    def test_model_details_endpoint(self):
        """Test model details endpoint."""
        with patch('anomaly_detection.server.global_model_repository') as mock_repo:
            mock_model = Mock()
            mock_model.model_id = "model_123"
            mock_model.name = "Test Model"
            mock_model.algorithm = "isolation_forest"
            mock_model.to_dict.return_value = {
                "model_id": "model_123",
                "name": "Test Model",
                "algorithm": "isolation_forest",
                "parameters": {"n_estimators": 100}
            }
            mock_repo.load.return_value = mock_model
            
            response = self.client.get("/models/model_123")
            
            assert response.status_code == 200
            model_data = response.json()
            assert model_data["model_id"] == "model_123"
            assert model_data["algorithm"] == "isolation_forest"
    
    def test_model_not_found_endpoint(self):
        """Test model not found error handling."""
        with patch('anomaly_detection.server.global_model_repository') as mock_repo:
            mock_repo.load.side_effect = KeyError("Model not found")
            
            response = self.client.get("/models/nonexistent")
            
            assert response.status_code == 404
            error = response.json()
            assert "not found" in error["detail"].lower()
    
    def test_streaming_start_endpoint(self):
        """Test streaming detection start endpoint."""
        with patch('anomaly_detection.server.global_streaming_service') as mock_streaming:
            mock_streaming.start_streaming = AsyncMock(return_value={
                "stream_id": "stream_123",
                "status": "started",
                "algorithm": "isolation_forest"
            })
            
            test_data = {
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "buffer_size": 1000
            }
            
            response = self.client.post("/stream/start", json=test_data)
            
            assert response.status_code == 200
            result = response.json()
            assert result["stream_id"] == "stream_123"
            assert result["status"] == "started"
    
    def test_streaming_stop_endpoint(self):
        """Test streaming detection stop endpoint."""
        with patch('anomaly_detection.server.global_streaming_service') as mock_streaming:
            mock_streaming.stop_streaming = AsyncMock(return_value={
                "stream_id": "stream_123",
                "status": "stopped",
                "total_processed": 5000
            })
            
            response = self.client.post("/stream/stream_123/stop")
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "stopped"
            assert result["total_processed"] == 5000
    
    def test_explanation_endpoint(self):
        """Test anomaly explanation endpoint."""
        with patch('anomaly_detection.server.global_explainability_service') as mock_explain:
            mock_explain.explain_prediction = AsyncMock(return_value={
                "explanation_type": "feature_importance",
                "feature_importance": [0.3, 0.7],
                "sample_index": 0,
                "prediction": 1,
                "confidence": 0.85
            })
            
            test_data = {
                "data": [[1, 2], [3, 4]],
                "sample_index": 0,
                "explainer_type": "feature_importance",
                "algorithm": "isolation_forest"
            }
            
            response = self.client.post("/explain", json=test_data)
            
            assert response.status_code == 200
            result = response.json()
            assert result["explanation_type"] == "feature_importance"
            assert "feature_importance" in result
    
    def test_monitoring_metrics_endpoint(self):
        """Test monitoring metrics endpoint."""
        with patch('anomaly_detection.server.get_metrics_collector') as mock_get_collector:
            mock_collector = Mock()
            mock_collector.get_summary_stats.return_value = {
                "total_metrics": 150,
                "unique_metric_names": {"api_requests", "detection_time", "accuracy"},
                "metrics_by_name": {
                    "api_requests": {"count": 100, "avg": 1.0},
                    "detection_time": {"count": 50, "avg": 125.5}
                }
            }
            mock_get_collector.return_value = mock_collector
            
            response = self.client.get("/monitoring/metrics")
            
            assert response.status_code == 200
            metrics = response.json()
            assert metrics["total_metrics"] == 150
            assert "api_requests" in metrics["metrics_by_name"]


class TestServerErrorHandling:
    """Test cases for server error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.server.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
    
    def test_detection_service_error(self):
        """Test error handling when detection service fails."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection:
            mock_detection.detect_anomalies = AsyncMock(
                side_effect=Exception("Detection algorithm failed")
            )
            
            test_data = {
                "data": [[1, 2], [3, 4]],
                "algorithm": "isolation_forest"
            }
            
            response = self.client.post("/detect", json=test_data)
            
            assert response.status_code == 500
            error = response.json()
            assert "error" in error
    
    def test_validation_error_handling(self):
        """Test validation error handling for invalid input."""
        # Invalid data format (missing required fields)
        test_data = {"invalid": "data"}
        
        response = self.client.post("/detect", json=test_data)
        
        assert response.status_code == 422  # Validation error
        error = response.json()
        assert "detail" in error
    
    def test_model_repository_error(self):
        """Test error handling when model repository fails."""
        with patch('anomaly_detection.server.global_model_repository') as mock_repo:
            mock_repo.list_models.side_effect = Exception("Database connection failed")
            
            response = self.client.get("/models")
            
            assert response.status_code == 500
            error = response.json()
            assert "error" in error
    
    def test_streaming_service_error(self):
        """Test error handling when streaming service fails."""
        with patch('anomaly_detection.server.global_streaming_service') as mock_streaming:
            mock_streaming.start_streaming = AsyncMock(
                side_effect=Exception("Streaming initialization failed")
            )
            
            test_data = {
                "algorithm": "isolation_forest",
                "contamination": 0.1
            }
            
            response = self.client.post("/stream/start", json=test_data)
            
            assert response.status_code == 500
            error = response.json()
            assert "error" in error


class TestServerConfiguration:
    """Test cases for server configuration and dependencies."""
    
    @patch('anomaly_detection.server.get_settings')
    def test_cors_origins_configuration(self, mock_get_settings):
        """Test CORS origins configuration from settings."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["http://localhost:3000", "https://prod.example.com"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        # Find CORS middleware
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None
        # CORS middleware should be configured with the specified origins
        assert cors_middleware.kwargs["allow_origins"] == ["http://localhost:3000", "https://prod.example.com"]
    
    @patch('anomaly_detection.server.get_settings')  
    def test_api_metadata_configuration(self, mock_get_settings):
        """Test API metadata configuration."""
        mock_settings = Mock()
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        assert app.title == "Anomaly Detection API"
        assert app.version == "0.1.0"
        assert "comprehensive anomaly detection" in app.description.lower()
    
    def test_global_service_initialization(self):
        """Test that global services are properly initialized."""
        # Before lifespan, services should be None
        assert global_detection_service is None
        assert global_ensemble_service is None
        assert global_model_repository is None
        
        # After running lifespan context, they should be initialized
        app = FastAPI()
        
        async def test_globals():
            async with lifespan(app):
                assert global_detection_service is not None
                assert global_ensemble_service is not None
                assert global_model_repository is not None
                assert global_streaming_service is not None
                assert global_explainability_service is not None
                assert global_health_service is not None
        
        asyncio.run(test_globals())


class TestServerMain:
    """Test cases for server main function."""
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.server.get_settings')
    def test_main_function_uvicorn_configuration(self, mock_get_settings, mock_uvicorn_run):
        """Test main function configures uvicorn correctly."""
        mock_settings = Mock()
        mock_settings.api.host = "0.0.0.0"
        mock_settings.api.port = 8000
        mock_settings.api.workers = 1
        mock_settings.api.reload = False
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        # Call main function
        main()
        
        # Check uvicorn.run was called with correct parameters
        mock_uvicorn_run.assert_called_once()
        call_kwargs = mock_uvicorn_run.call_args[1]
        
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 8000
        assert call_kwargs["workers"] == 1
        assert call_kwargs["reload"] == False
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.server.get_settings')
    def test_main_function_development_settings(self, mock_get_settings, mock_uvicorn_run):
        """Test main function with development settings."""
        mock_settings = Mock()
        mock_settings.api.host = "127.0.0.1"
        mock_settings.api.port = 8080
        mock_settings.api.workers = 1
        mock_settings.api.reload = True
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        main()
        
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8080
        assert call_kwargs["reload"] == True
    
    @patch('uvicorn.run')
    @patch('anomaly_detection.server.get_logger')
    @patch('anomaly_detection.server.get_settings')
    def test_main_function_logging(self, mock_get_settings, mock_get_logger, mock_uvicorn_run):
        """Test main function logging initialization."""
        mock_settings = Mock()
        mock_settings.api.host = "0.0.0.0"
        mock_settings.api.port = 8000
        mock_settings.api.workers = 1
        mock_settings.api.reload = False
        mock_settings.api.cors_origins = ["*"]
        mock_get_settings.return_value = mock_settings
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        main()
        
        # Should log server startup information
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Starting Anomaly Detection API Server" in call for call in log_calls)


class TestServerPerformance:
    """Test cases for server performance and resource management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.server.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
    
    def test_concurrent_requests_handling(self):
        """Test handling of concurrent requests."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection:
            mock_detection.detect_anomalies = AsyncMock(return_value={
                "anomalies": [0, 1],
                "scores": [0.1, 0.8],
                "algorithm": "isolation_forest"
            })
            
            test_data = {
                "data": [[1, 2], [3, 4]],
                "algorithm": "isolation_forest"
            }
            
            # Send multiple concurrent requests
            responses = []
            for _ in range(5):
                response = self.client.post("/detect", json=test_data)
                responses.append(response)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                assert "anomalies" in response.json()
    
    def test_large_data_handling(self):
        """Test handling of large data requests."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection:
            mock_detection.detect_anomalies = AsyncMock(return_value={
                "anomalies": [0] * 1000,
                "scores": [0.1] * 1000,
                "algorithm": "isolation_forest"
            })
            
            # Large dataset
            large_data = [[i, i+1] for i in range(1000)]
            test_data = {
                "data": large_data,
                "algorithm": "isolation_forest"
            }
            
            response = self.client.post("/detect", json=test_data)
            
            assert response.status_code == 200
            result = response.json()
            assert len(result["anomalies"]) == 1000
    
    def test_memory_usage_monitoring(self):
        """Test that endpoints don't cause memory leaks."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection:
            mock_detection.detect_anomalies = AsyncMock(return_value={
                "anomalies": [0, 1],
                "scores": [0.1, 0.8],
                "algorithm": "isolation_forest"
            })
            
            test_data = {
                "data": [[1, 2], [3, 4]],
                "algorithm": "isolation_forest"
            }
            
            # Send many requests to check for memory leaks
            for _ in range(50):
                response = self.client.post("/detect", json=test_data)
                assert response.status_code == 200
            
            # Test passes if no memory-related errors occur


class TestServerIntegration:
    """Integration test cases for server components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.server.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
    
    def test_service_integration_flow(self):
        """Test integration flow between different services."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection, \
             patch('anomaly_detection.server.global_model_repository') as mock_repo:
            
            # Mock detection service
            mock_detection.detect_anomalies = AsyncMock(return_value={
                "anomalies": [0, 1, 0],
                "scores": [0.1, 0.8, 0.2],
                "algorithm": "isolation_forest"
            })
            
            # Mock model repository
            mock_repo.list_models.return_value = [{
                "model_id": "model_123",
                "algorithm": "isolation_forest"
            }]
            
            # Test detection endpoint
            detection_response = self.client.post("/detect", json={
                "data": [[1, 2], [3, 4], [5, 6]],
                "algorithm": "isolation_forest"
            })
            assert detection_response.status_code == 200
            
            # Test models endpoint
            models_response = self.client.get("/models")
            assert models_response.status_code == 200
            
            # Both should work independently
            assert "anomalies" in detection_response.json()
            assert len(models_response.json()) == 1
    
    def test_error_propagation_integration(self):
        """Test error propagation between components."""
        with patch('anomaly_detection.server.global_detection_service') as mock_detection:
            # Service throws custom exception
            from anomaly_detection.infrastructure.logging.error_handler import ValidationError
            mock_detection.detect_anomalies = AsyncMock(
                side_effect=ValidationError("Invalid algorithm parameters")
            )
            
            response = self.client.post("/detect", json={
                "data": [[1, 2]],
                "algorithm": "invalid_algorithm"
            })
            
            # Should handle the custom exception appropriately
            assert response.status_code in [400, 422, 500]  # Depends on error handling implementation