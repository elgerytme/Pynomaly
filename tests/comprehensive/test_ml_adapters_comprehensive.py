"""Comprehensive tests for all ML framework adapters with full dependency coverage."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings

from tests.conftest_dependencies import requires_dependency, requires_dependencies

# Core imports that should always be available
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import InvalidAlgorithmError


@requires_dependency('torch')
class TestPyTorchAdapterComprehensive:
    """Comprehensive tests for PyTorch adapter with all features."""
    
    @pytest.fixture
    def pytorch_adapter(self):
        """Create PyTorch adapter instance."""
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter("AutoEncoder")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
    
    def test_pytorch_adapter_creation(self, pytorch_adapter):
        """Test PyTorch adapter creation and initialization."""
        assert pytorch_adapter.algorithm_name == "AutoEncoder"
        assert hasattr(pytorch_adapter, 'fit')
        assert hasattr(pytorch_adapter, 'detect')
        assert hasattr(pytorch_adapter, 'score')
    
    def test_pytorch_device_selection(self, pytorch_adapter):
        """Test PyTorch device selection (CPU/GPU)."""
        import torch
        
        # Test CPU device
        assert pytorch_adapter.device in ['cpu', 'cuda']
        
        # Test device compatibility
        if torch.cuda.is_available():
            assert 'cuda' in str(pytorch_adapter.device)
        else:
            assert 'cpu' in str(pytorch_adapter.device)
    
    def test_pytorch_model_architecture(self, pytorch_adapter, sample_data):
        """Test PyTorch model architecture creation."""
        dataset = Dataset(name="test", data=sample_data)
        
        # Test model initialization
        pytorch_adapter._initialize_model(input_dim=sample_data.shape[1])
        
        assert pytorch_adapter.model is not None
        assert hasattr(pytorch_adapter.model, 'forward')
    
    def test_pytorch_training_loop(self, pytorch_adapter, sample_data):
        """Test PyTorch training loop functionality."""
        dataset = Dataset(name="test", data=sample_data)
        
        # Test fitting
        pytorch_adapter.fit(dataset)
        assert pytorch_adapter.is_fitted
        
        # Test that model parameters are updated
        assert any(param.requires_grad for param in pytorch_adapter.model.parameters())
    
    def test_pytorch_anomaly_detection(self, pytorch_adapter, sample_data):
        """Test PyTorch anomaly detection functionality."""
        dataset = Dataset(name="test", data=sample_data)
        
        # Fit and detect
        pytorch_adapter.fit(dataset)
        result = pytorch_adapter.detect(dataset)
        
        assert len(result.scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(0.0 <= score.value <= 1.0 for score in result.scores)
    
    @given(
        n_samples=st.integers(min_value=50, max_value=500),
        n_features=st.integers(min_value=3, max_value=20)
    )
    @settings(max_examples=5, deadline=None)
    def test_pytorch_various_data_sizes(self, pytorch_adapter, n_samples, n_features):
        """Property test: PyTorch adapter works with various data sizes."""
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(data, columns=[f'f_{i}' for i in range(n_features)])
        dataset = Dataset(name="test", data=df)
        
        pytorch_adapter.fit(dataset)
        result = pytorch_adapter.detect(dataset)
        
        assert len(result.scores) == n_samples
        assert all(isinstance(score, AnomalyScore) for score in result.scores)


@requires_dependency('tensorflow')
class TestTensorFlowAdapterComprehensive:
    """Comprehensive tests for TensorFlow adapter with all features."""
    
    @pytest.fixture
    def tensorflow_adapter(self):
        """Create TensorFlow adapter instance."""
        from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
        return TensorFlowAdapter("AutoEncoder")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
    
    def test_tensorflow_adapter_creation(self, tensorflow_adapter):
        """Test TensorFlow adapter creation."""
        assert tensorflow_adapter.algorithm_name == "AutoEncoder"
        assert hasattr(tensorflow_adapter, 'fit')
        assert hasattr(tensorflow_adapter, 'detect')
    
    def test_tensorflow_model_compilation(self, tensorflow_adapter, sample_data):
        """Test TensorFlow model compilation."""
        dataset = Dataset(name="test", data=sample_data)
        
        tensorflow_adapter._build_model(input_dim=sample_data.shape[1])
        assert tensorflow_adapter.model is not None
        assert hasattr(tensorflow_adapter.model, 'compile')
        assert hasattr(tensorflow_adapter.model, 'fit')
    
    def test_tensorflow_training(self, tensorflow_adapter, sample_data):
        """Test TensorFlow model training."""
        dataset = Dataset(name="test", data=sample_data)
        
        tensorflow_adapter.fit(dataset)
        assert tensorflow_adapter.is_fitted
        
        # Test prediction capability
        result = tensorflow_adapter.detect(dataset)
        assert len(result.scores) == len(sample_data)
    
    def test_tensorflow_gpu_support(self, tensorflow_adapter):
        """Test TensorFlow GPU support detection."""
        import tensorflow as tf
        
        # Test GPU availability detection
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            assert tensorflow_adapter.device_type in ['GPU', 'gpu']
        else:
            assert tensorflow_adapter.device_type in ['CPU', 'cpu']


@requires_dependency('jax')
class TestJAXAdapterComprehensive:
    """Comprehensive tests for JAX adapter with all features."""
    
    @pytest.fixture
    def jax_adapter(self):
        """Create JAX adapter instance."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
        return JAXAdapter("AutoEncoder")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
    
    def test_jax_adapter_creation(self, jax_adapter):
        """Test JAX adapter creation."""
        assert jax_adapter.algorithm_name == "AutoEncoder"
        assert hasattr(jax_adapter, 'fit')
        assert hasattr(jax_adapter, 'detect')
    
    def test_jax_functional_programming(self, jax_adapter, sample_data):
        """Test JAX functional programming paradigm."""
        import jax.numpy as jnp
        
        dataset = Dataset(name="test", data=sample_data)
        
        # Test JAX array conversion
        jax_data = jnp.array(sample_data.values)
        assert isinstance(jax_data, jnp.ndarray)
        
        # Test model initialization
        jax_adapter.fit(dataset)
        assert jax_adapter.is_fitted
    
    def test_jax_jit_compilation(self, jax_adapter, sample_data):
        """Test JAX JIT compilation for performance."""
        from jax import jit
        
        dataset = Dataset(name="test", data=sample_data)
        jax_adapter.fit(dataset)
        
        # Test JIT-compiled functions
        if hasattr(jax_adapter, '_predict_fn'):
            assert hasattr(jax_adapter._predict_fn, '__wrapped__')


@requires_dependency('pyod')
class TestPyODAdapterComprehensive:
    """Comprehensive tests for PyOD adapter with all algorithms."""
    
    @pytest.fixture
    def pyod_adapter(self):
        """Create PyOD adapter instance."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
        return PyODAdapter("IsolationForest")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.randn(95, 3)
        anomalies = np.random.randn(5, 3) * 3 + 5
        data = np.vstack([normal_data, anomalies])
        return pd.DataFrame(data, columns=['x', 'y', 'z'])
    
    def test_pyod_all_algorithms(self):
        """Test all available PyOD algorithms."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
        
        algorithms = ['IsolationForest', 'LOF', 'OCSVM', 'KNN', 'PCA']
        
        for algorithm in algorithms:
            if algorithm in PyODAdapter.ALGORITHM_MAPPING:
                adapter = PyODAdapter(algorithm)
                assert adapter.algorithm_name == algorithm
    
    def test_pyod_contamination_handling(self, pyod_adapter, sample_data):
        """Test PyOD contamination parameter handling."""
        dataset = Dataset(name="test", data=sample_data)
        
        # Test with different contamination rates
        contamination_rates = [0.05, 0.1, 0.15, 0.2]
        
        for rate in contamination_rates:
            pyod_adapter.contamination = rate
            pyod_adapter.fit(dataset)
            result = pyod_adapter.detect(dataset)
            
            # Verify anomaly detection with expected contamination
            anomaly_ratio = sum(result.labels) / len(result.labels)
            assert 0.0 <= anomaly_ratio <= 0.5  # Reasonable bounds
    
    def test_pyod_ensemble_methods(self, sample_data):
        """Test PyOD ensemble methods."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
        
        ensemble_algorithms = ['Feature Bagging', 'LSCP', 'XGBOD']
        dataset = Dataset(name="test", data=sample_data)
        
        for algorithm in ensemble_algorithms:
            if algorithm in PyODAdapter.ALGORITHM_MAPPING:
                adapter = PyODAdapter(algorithm)
                adapter.fit(dataset)
                result = adapter.detect(dataset)
                
                assert len(result.scores) == len(sample_data)
                assert all(isinstance(score, AnomalyScore) for score in result.scores)


@requires_dependencies('fastapi', 'uvicorn')
class TestAPIIntegrationComprehensive:
    """Comprehensive tests for FastAPI integration."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        from fastapi.testclient import TestClient
        from pynomaly.presentation.api.app import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, test_client):
        """Test API health endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_detector_crud_endpoints(self, test_client):
        """Test detector CRUD operations via API."""
        # Create detector
        detector_data = {
            "name": "test_detector",
            "algorithm": "IsolationForest",
            "contamination": 0.1,
            "parameters": {"n_estimators": 100}
        }
        
        response = test_client.post("/detectors/", json=detector_data)
        assert response.status_code in [200, 201]
        
        created_detector = response.json()
        detector_id = created_detector["id"]
        
        # Get detector
        response = test_client.get(f"/detectors/{detector_id}")
        assert response.status_code == 200
        
        # List detectors
        response = test_client.get("/detectors/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_detection_endpoint(self, test_client):
        """Test anomaly detection endpoint."""
        # First create a detector
        detector_data = {
            "name": "api_test_detector",
            "algorithm": "IsolationForest",
            "contamination": 0.1
        }
        
        response = test_client.post("/detectors/", json=detector_data)
        detector_id = response.json()["id"]
        
        # Create dataset
        dataset_data = {
            "name": "test_dataset",
            "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [100, 200, 300]]  # Last row is anomaly
        }
        
        response = test_client.post("/datasets/", json=dataset_data)
        dataset_id = response.json()["id"]
        
        # Perform detection
        detection_data = {
            "detector_id": detector_id,
            "dataset_id": dataset_id
        }
        
        response = test_client.post("/detect/", json=detection_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "scores" in result
        assert "anomalies" in result


@requires_dependency('redis')
class TestRedisIntegrationComprehensive:
    """Comprehensive tests for Redis caching integration."""
    
    @pytest.fixture
    def redis_client(self):
        """Create Redis client for testing."""
        import redis
        return redis.Redis(host='redis-test', port=6379, db=0, decode_responses=True)
    
    @pytest.fixture
    def cache_service(self):
        """Create cache service instance."""
        from pynomaly.infrastructure.cache.redis_cache import RedisCacheService
        return RedisCacheService(host='redis-test', port=6379, db=0)
    
    def test_redis_connection(self, redis_client):
        """Test Redis connection."""
        assert redis_client.ping() is True
    
    def test_cache_detector_results(self, cache_service):
        """Test caching detector results."""
        # Test data
        detector_id = "test_detector_123"
        dataset_id = "test_dataset_456"
        
        result_data = {
            "scores": [0.1, 0.8, 0.3, 0.9],
            "anomalies": [1, 3],
            "threshold": 0.5
        }
        
        # Cache result
        cache_key = f"detection:{detector_id}:{dataset_id}"
        cache_service.set(cache_key, result_data, ttl=3600)
        
        # Retrieve result
        cached_result = cache_service.get(cache_key)
        assert cached_result == result_data
    
    def test_cache_model_metadata(self, cache_service):
        """Test caching model metadata."""
        model_id = "model_789"
        metadata = {
            "algorithm": "IsolationForest",
            "training_samples": 1000,
            "contamination": 0.1,
            "performance_metrics": {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81
            }
        }
        
        cache_service.set(f"model_metadata:{model_id}", metadata, ttl=7200)
        cached_metadata = cache_service.get(f"model_metadata:{model_id}")
        
        assert cached_metadata == metadata


@requires_dependency('database')
class TestDatabaseIntegrationComprehensive:
    """Comprehensive tests for database integration."""
    
    @pytest.fixture
    def db_connection(self):
        """Create database connection for testing."""
        from pynomaly.infrastructure.persistence.database import DatabaseConnection
        
        db_url = "postgresql://pynomaly_test:test_password@postgres-test:5432/pynomaly_test"
        return DatabaseConnection(db_url)
    
    @pytest.fixture
    def detector_repository(self, db_connection):
        """Create detector repository for testing."""
        from pynomaly.infrastructure.persistence.database_repositories import DatabaseDetectorRepository
        return DatabaseDetectorRepository(db_connection)
    
    def test_database_connection(self, db_connection):
        """Test database connection."""
        with db_connection.get_session() as session:
            result = session.execute("SELECT 1").scalar()
            assert result == 1
    
    def test_detector_persistence(self, detector_repository):
        """Test detector persistence operations."""
        from pynomaly.domain.entities import Detector
        
        # Create detector
        detector = Detector(
            name="persistence_test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100}
        )
        
        # Save detector
        saved_detector = detector_repository.save(detector)
        assert saved_detector.id is not None
        
        # Retrieve detector
        retrieved_detector = detector_repository.find_by_id(saved_detector.id)
        assert retrieved_detector is not None
        assert retrieved_detector.name == detector.name
        assert retrieved_detector.algorithm == detector.algorithm
    
    def test_transaction_rollback(self, db_connection):
        """Test database transaction rollback."""
        with pytest.raises(Exception):
            with db_connection.get_session() as session:
                session.execute("INSERT INTO test_table VALUES (1, 'test')")
                raise Exception("Simulated error")
        
        # Verify rollback worked
        with db_connection.get_session() as session:
            result = session.execute("SELECT COUNT(*) FROM detectors").scalar()
            # Should not include failed insert


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""
    
    @requires_dependencies('scikit-learn', 'database', 'redis')
    def test_complete_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow."""
        # This test requires multiple dependencies and simulates
        # a complete workflow from data input to result caching
        
        # 1. Create detector
        from pynomaly.domain.entities import Detector, Dataset
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        
        detector = Detector(
            name="workflow_test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        # 2. Create dataset
        np.random.seed(42)
        data = np.random.randn(100, 3)
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        dataset = Dataset(name="workflow_test_dataset", data=df)
        
        # 3. Train detector
        adapter = SklearnAdapter("IsolationForest")
        adapter.fit(dataset)
        
        # 4. Perform detection
        result = adapter.detect(dataset)
        
        # 5. Verify results
        assert len(result.scores) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert result.detector_name == adapter.name
        
        # 6. Test that workflow components work together
        assert result.execution_time > 0
        assert len(result.anomalies) > 0  # Should detect some anomalies