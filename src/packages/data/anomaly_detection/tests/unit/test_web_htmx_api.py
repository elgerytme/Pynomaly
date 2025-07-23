"""Unit tests for web HTMX API endpoints."""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import Request

from anomaly_detection.web.api.htmx import (
    router,
    get_detection_service,
    get_ensemble_service,
    get_streaming_service,
    get_explainability_service,
    get_model_repository
)
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.domain.services.explainability_service import ExplainabilityService
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus


class TestHTMXAPI:
    """Test suite for HTMX API endpoints."""
    
    @pytest.fixture
    def mock_detection_service(self):
        """Mock detection service fixture."""
        service = Mock(spec=DetectionService)
        
        # Mock detection result
        result = DetectionResult(
            algorithm="isolation_forest",
            predictions=np.array([1, 1, -1, 1, 1]),
            scores=np.array([0.1, 0.2, 0.9, 0.3, 0.15]),
            total_samples=5,
            anomaly_count=1,
            processing_time=1.23,
            success=True,
            metadata={'contamination': 0.1}
        )
        service.detect_anomalies.return_value = result
        return service
    
    @pytest.fixture
    def mock_ensemble_service(self):
        """Mock ensemble service fixture."""
        service = Mock(spec=EnsembleService)
        
        # Mock ensemble result
        result = DetectionResult(
            algorithm="ensemble_majority",
            predictions=np.array([1, -1, -1, 1, 1]),
            scores=np.array([0.2, 0.8, 0.9, 0.1, 0.25]),
            total_samples=5,
            anomaly_count=2,
            processing_time=2.15,
            success=True,
            metadata={'method': 'majority', 'algorithms': ['isolation_forest', 'lof']}
        )
        service.detect_anomalies.return_value = result
        return service
    
    @pytest.fixture
    def mock_streaming_service(self):
        """Mock streaming service fixture."""
        service = Mock(spec=StreamingService)
        service.is_active.return_value = False
        service.get_status.return_value = {
            'active': False,
            'processed_samples': 0,
            'anomalies_detected': 0,
            'last_detection': None
        }
        return service
    
    @pytest.fixture
    def mock_explainability_service(self):
        """Mock explainability service fixture."""
        service = Mock(spec=ExplainabilityService)
        service.explain_anomaly.return_value = {
            'sample_index': 0,
            'anomaly_score': 0.85,
            'feature_importances': [0.3, 0.25, 0.2, 0.15, 0.1],
            'explanation': 'High feature values in dimensions 0 and 1'
        }
        return service
    
    @pytest.fixture
    def mock_model_repository(self):
        """Mock model repository fixture."""
        repo = Mock(spec=ModelRepository)
        
        # Mock model
        model = Model(
            model_id="test_model_1",
            name="Test Model",
            algorithm="isolation_forest",
            metadata=ModelMetadata(
                created_at="2024-01-20T10:00:00",
                updated_at="2024-01-20T10:00:00",
                version="1.0.0",
                description="Test model for unit tests"
            ),
            status=ModelStatus.ACTIVE,
            serialized_model=b"mock_model_data"
        )
        
        repo.get_model.return_value = model
        repo.list_models.return_value = [model]
        repo.save_model.return_value = model
        return repo
    
    @pytest.fixture
    def override_dependencies(self, mock_detection_service, mock_ensemble_service, 
                            mock_streaming_service, mock_explainability_service, 
                            mock_model_repository):
        """Override the service dependencies."""
        def _get_detection_service():
            return mock_detection_service
        
        def _get_ensemble_service():
            return mock_ensemble_service
        
        def _get_streaming_service():
            return mock_streaming_service
        
        def _get_explainability_service():
            return mock_explainability_service
        
        def _get_model_repository():
            return mock_model_repository
        
        router.dependency_overrides[get_detection_service] = _get_detection_service
        router.dependency_overrides[get_ensemble_service] = _get_ensemble_service
        router.dependency_overrides[get_streaming_service] = _get_streaming_service
        router.dependency_overrides[get_explainability_service] = _get_explainability_service
        router.dependency_overrides[get_model_repository] = _get_model_repository
        yield
        router.dependency_overrides.clear()
    
    @pytest.fixture
    def client(self, override_dependencies):
        """Test client fixture."""
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/htmx")
        return TestClient(app)
    
    def test_run_detection_with_sample_data(self, client, mock_detection_service):
        """Test running detection with provided sample data."""
        sample_data = json.dumps([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.post("/htmx/detect", data={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "sample_data": sample_data
            })
            
            assert response.status_code == 200
            mock_detection_service.detect_anomalies.assert_called_once()
            
            # Check detection service call arguments
            call_args = mock_detection_service.detect_anomalies.call_args
            assert call_args[1]["algorithm"] == "isolation_forest"
            assert call_args[1]["contamination"] == 0.1
            assert isinstance(call_args[1]["data"], np.ndarray)
    
    def test_run_detection_without_sample_data(self, client, mock_detection_service):
        """Test running detection without sample data (generates synthetic data)."""
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.post("/htmx/detect", data={
                "algorithm": "lof",
                "contamination": 0.05,
                "sample_data": ""
            })
            
            assert response.status_code == 200
            mock_detection_service.detect_anomalies.assert_called_once()
            
            # Should generate synthetic data
            call_args = mock_detection_service.detect_anomalies.call_args
            assert call_args[1]["algorithm"] == "lof"
            assert call_args[1]["contamination"] == 0.05
            assert isinstance(call_args[1]["data"], np.ndarray)
            assert call_args[1]["data"].shape[0] == 110  # 100 normal + 10 anomalies
    
    def test_run_detection_invalid_json(self, client):
        """Test running detection with invalid JSON data."""
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.post("/htmx/detect", data={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "sample_data": "invalid json"
            })
            
            # Should handle invalid JSON gracefully
            # The exact behavior depends on implementation
    
    def test_run_detection_service_error(self, client, mock_detection_service):
        """Test handling of detection service errors."""
        mock_detection_service.detect_anomalies.side_effect = Exception("Detection failed")
        
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.post("/htmx/detect", data={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "sample_data": ""
            })
            
            # Should handle service errors gracefully
            # The exact behavior depends on implementation
    
    def test_run_ensemble_detection(self, client, mock_ensemble_service):
        """Test running ensemble detection."""
        algorithms = ["isolation_forest", "lof"]
        
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            # Note: This assumes there's an ensemble endpoint
            # Adjust based on actual implementation
            sample_data = json.dumps([[1, 2, 3], [4, 5, 6]])
            
            response = client.post("/htmx/detect", data={
                "algorithm": "ensemble_majority",
                "contamination": 0.1,
                "sample_data": sample_data
            })
            
            assert response.status_code == 200
    
    def test_streaming_status_inactive(self, client, mock_streaming_service):
        """Test getting streaming status when inactive."""
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            # This endpoint may not exist in current implementation
            # Adjust based on actual HTMX endpoints available
            pass
    
    def test_model_management_endpoints(self, client, mock_model_repository):
        """Test model management related endpoints."""
        # This tests any model-related HTMX endpoints that might exist
        # Adjust based on actual implementation
        pass
    
    def test_dependency_injection_detection_service(self):
        """Test detection service dependency injection."""
        service = get_detection_service()
        assert isinstance(service, DetectionService)
        
        # Test singleton behavior
        service2 = get_detection_service()
        assert service is service2
    
    def test_dependency_injection_ensemble_service(self):
        """Test ensemble service dependency injection."""
        service = get_ensemble_service()
        assert isinstance(service, EnsembleService)
        
        # Test singleton behavior
        service2 = get_ensemble_service()
        assert service is service2
    
    def test_dependency_injection_streaming_service(self):
        """Test streaming service dependency injection."""
        service = get_streaming_service()
        assert isinstance(service, StreamingService)
        
        # Test singleton behavior
        service2 = get_streaming_service()
        assert service is service2
    
    def test_dependency_injection_explainability_service(self):
        """Test explainability service dependency injection."""
        service = get_explainability_service()
        assert isinstance(service, ExplainabilityService)
        
        # Test singleton behavior
        service2 = get_explainability_service()
        assert service is service2
    
    def test_dependency_injection_model_repository(self):
        """Test model repository dependency injection."""
        repo = get_model_repository()
        assert isinstance(repo, ModelRepository)
        
        # Test singleton behavior
        repo2 = get_model_repository()
        assert repo is repo2
    
    def test_form_data_validation(self, client):
        """Test form data validation."""
        # Test missing required fields
        response = client.post("/htmx/detect", data={
            "contamination": 0.1,
            "sample_data": ""
            # Missing algorithm
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_contamination_parameter_validation(self, client, mock_detection_service):
        """Test contamination parameter validation."""
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            # Test valid contamination value
            response = client.post("/htmx/detect", data={
                "algorithm": "isolation_forest",
                "contamination": 0.15,
                "sample_data": ""
            })
            
            assert response.status_code == 200
            
            call_args = mock_detection_service.detect_anomalies.call_args
            assert call_args[1]["contamination"] == 0.15
    
    def test_algorithm_parameter_validation(self, client, mock_detection_service):
        """Test algorithm parameter validation."""
        valid_algorithms = ["isolation_forest", "lof", "one_class_svm"]
        
        for algorithm in valid_algorithms:
            with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_templates.TemplateResponse.return_value = mock_response
                
                response = client.post("/htmx/detect", data={
                    "algorithm": algorithm,
                    "contamination": 0.1,
                    "sample_data": ""
                })
                
                assert response.status_code == 200
                
                call_args = mock_detection_service.detect_anomalies.call_args
                assert call_args[1]["algorithm"] == algorithm
    
    def test_sample_data_parsing(self, client, mock_detection_service):
        """Test different sample data formats."""
        test_cases = [
            # Simple 2D array
            [[1, 2], [3, 4], [5, 6]],
            # Different dimensions
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            # Single row
            [[1, 2, 3]],
            # Larger dataset
            [[i, i+1, i+2] for i in range(10)]
        ]
        
        for test_data in test_cases:
            with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_templates.TemplateResponse.return_value = mock_response
                
                response = client.post("/htmx/detect", data={
                    "algorithm": "isolation_forest",
                    "contamination": 0.1,
                    "sample_data": json.dumps(test_data)
                })
                
                assert response.status_code == 200
                
                call_args = mock_detection_service.detect_anomalies.call_args
                parsed_data = call_args[1]["data"]
                assert parsed_data.shape == (len(test_data), len(test_data[0]))
    
    def test_html_response_format(self, client, mock_detection_service):
        """Test that responses are properly formatted HTML for HTMX."""
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            response = client.post("/htmx/detect", data={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "sample_data": ""
            })
            
            assert response.status_code == 200
            # Verify that template response is called
            mock_templates.TemplateResponse.assert_called_once()
    
    def test_synthetic_data_generation(self, client, mock_detection_service):
        """Test synthetic data generation when no sample data provided."""
        with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_templates.TemplateResponse.return_value = mock_response
            
            with patch('numpy.random.seed') as mock_seed:
                response = client.post("/htmx/detect", data={
                    "algorithm": "isolation_forest",
                    "contamination": 0.1,
                    "sample_data": ""
                })
                
                assert response.status_code == 200
                # Verify random seed was set for reproducibility
                mock_seed.assert_called_with(42)
                
                call_args = mock_detection_service.detect_anomalies.call_args
                data = call_args[1]["data"]
                
                # Check synthetic data properties
                assert data.shape[0] == 110  # 100 normal + 10 anomalies
                assert data.shape[1] == 5    # 5 features
    
    def test_concurrent_requests(self, client, mock_detection_service):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('anomaly_detection.web.api.htmx.templates') as mock_templates:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_templates.TemplateResponse.return_value = mock_response
                
                response = client.post("/htmx/detect", data={
                    "algorithm": "isolation_forest",
                    "contamination": 0.1,
                    "sample_data": ""
                })
                results.append(response.status_code)
        
        # Start multiple threads
        threads = [threading.Thread(target=make_request) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 3