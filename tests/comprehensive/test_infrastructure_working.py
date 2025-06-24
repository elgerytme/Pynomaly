"""Working comprehensive infrastructure testing for coverage boost."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

# Strategic imports for infrastructure components that exist
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.config.container import Container
from pynomaly.infrastructure.repositories.in_memory_repositories import (
    InMemoryDetectorRepository, 
    InMemoryDatasetRepository,
    InMemoryResultRepository
)


class TestInMemoryRepositoriesWorking:
    """Working comprehensive tests for in-memory repositories."""
    
    def test_detector_repository_comprehensive(self):
        """Test InMemoryDetectorRepository comprehensively."""
        repo = InMemoryDetectorRepository()
        
        # Create mock detector using actual Detector structure
        from pynomaly.domain.entities.detector import Detector
        from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
        
        # Since Detector is abstract, we'll mock it
        mock_detector = Mock(spec=Detector)
        mock_detector.id = uuid.uuid4()
        mock_detector.name = "test_detector"
        mock_detector.algorithm_name = "IsolationForest"
        mock_detector.contamination_rate = ContaminationRate(0.1)
        mock_detector.is_fitted = False
        
        # Test save operation
        repo.save(mock_detector)
        
        # Test find_by_id
        found_detector = repo.find_by_id(mock_detector.id)
        assert found_detector is not None
        assert found_detector.name == "test_detector"
        
        # Test find_by_name
        found_by_name = repo.find_by_name("test_detector")
        assert found_by_name is not None
        assert found_by_name.id == mock_detector.id
        
        # Test find_all
        all_detectors = repo.find_all()
        assert len(all_detectors) == 1
        assert all_detectors[0].id == mock_detector.id
        
        # Test exists
        assert repo.exists(mock_detector.id) is True
        assert repo.exists(uuid.uuid4()) is False
        
        # Test count
        assert repo.count() == 1
        
        # Test find_by_algorithm
        algorithm_detectors = repo.find_by_algorithm("IsolationForest")
        assert len(algorithm_detectors) == 1
        
        # Test find_fitted
        fitted_detectors = repo.find_fitted()
        assert len(fitted_detectors) == 0  # Not fitted
        
        # Test model artifact operations
        test_artifact = b"test_model_data"
        repo.save_model_artifact(mock_detector.id, test_artifact)
        
        loaded_artifact = repo.load_model_artifact(mock_detector.id)
        assert loaded_artifact == test_artifact
        
        # Test delete
        deleted = repo.delete(mock_detector.id)
        assert deleted is True
        assert repo.find_by_id(mock_detector.id) is None
        assert repo.count() == 0
        
        # Test delete non-existent
        assert repo.delete(uuid.uuid4()) is False
    
    def test_dataset_repository_comprehensive(self):
        """Test InMemoryDatasetRepository comprehensively."""
        repo = InMemoryDatasetRepository()
        
        # Create real dataset
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        dataset = Dataset(name="test_dataset", data=data)
        
        # Test save
        repo.save(dataset)
        
        # Test find operations
        found = repo.find_by_id(dataset.id)
        assert found is not None
        assert found.name == "test_dataset"
        
        found_by_name = repo.find_by_name("test_dataset")
        assert found_by_name is not None
        assert found_by_name.id == dataset.id
        
        all_datasets = repo.find_all()
        assert len(all_datasets) == 1
        
        # Test metadata search
        dataset.add_metadata("source", "test")
        repo.save(dataset)  # Update
        
        metadata_results = repo.find_by_metadata("source", "test")
        assert len(metadata_results) == 1
        
        # Test data persistence
        data_path = repo.save_data(dataset.id, "parquet")
        assert data_path.startswith("memory://")
        
        loaded_dataset = repo.load_data(dataset.id)
        assert loaded_dataset is not None
        assert loaded_dataset.n_samples == 5
        
        # Test existence and count
        assert repo.exists(dataset.id) is True
        assert repo.count() == 1
        
        # Test delete
        assert repo.delete(dataset.id) is True
        assert repo.count() == 0
    
    def test_result_repository_comprehensive(self):
        """Test InMemoryResultRepository comprehensively."""
        repo = InMemoryResultRepository()
        
        # Create mock detection result
        from pynomaly.domain.entities.detection_result import DetectionResult
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.id = uuid.uuid4()
        mock_result.detector_id = uuid.uuid4()
        mock_result.dataset_id = uuid.uuid4()
        mock_result.timestamp = datetime.utcnow()
        mock_result.n_samples = 100
        mock_result.n_anomalies = 5
        mock_result.anomaly_rate = 0.05
        mock_result.threshold = 0.5
        mock_result.execution_time_ms = 1500
        mock_result.score_statistics = {"mean": 0.3, "std": 0.2}
        mock_result.has_confidence_intervals = True
        
        # Test save
        repo.save(mock_result)
        
        # Test find operations
        found = repo.find_by_id(mock_result.id)
        assert found is not None
        assert found.detector_id == mock_result.detector_id
        
        all_results = repo.find_all()
        assert len(all_results) == 1
        
        # Test find by detector and dataset
        detector_results = repo.find_by_detector(mock_result.detector_id)
        assert len(detector_results) == 1
        
        dataset_results = repo.find_by_dataset(mock_result.dataset_id)
        assert len(dataset_results) == 1
        
        # Test recent results
        recent_results = repo.find_recent(limit=5)
        assert len(recent_results) == 1
        
        # Test summary statistics
        summary = repo.get_summary_stats(mock_result.id)
        assert summary["n_samples"] == 100
        assert summary["n_anomalies"] == 5
        assert summary["anomaly_rate"] == 0.05
        
        # Test existence and count
        assert repo.exists(mock_result.id) is True
        assert repo.count() == 1
        
        # Test delete
        assert repo.delete(mock_result.id) is True
        assert repo.count() == 0


class TestContainerWorking:
    """Working comprehensive tests for dependency injection container."""
    
    def test_container_settings_provider(self):
        """Test container settings provider."""
        container = Container()
        
        # Test settings creation
        try:
            settings = container.settings()
            assert settings is not None
            assert hasattr(settings, 'api_host')
            assert hasattr(settings, 'api_port')
            assert hasattr(settings, 'database_url')
            
            # Test settings properties
            assert settings.api_host in ["0.0.0.0", "127.0.0.1", "localhost"]
            assert isinstance(settings.api_port, int)
            assert settings.api_port > 0
            
        except Exception as e:
            # If settings creation fails, ensure it's for expected reasons
            assert any(term in str(e).lower() for term in ["dependency", "import", "module"])
    
    def test_container_repository_providers(self):
        """Test container repository providers."""
        container = Container()
        
        # Mock settings to control repository type
        with patch.object(container, 'settings') as mock_settings:
            mock_settings_instance = Mock()
            mock_settings_instance.use_database_repositories = False
            mock_settings.return_value = mock_settings_instance
            
            # Test in-memory repository creation
            detector_repo = container.detector_repository()
            assert detector_repo is not None
            assert isinstance(detector_repo, InMemoryDetectorRepository)
            
            dataset_repo = container.dataset_repository()
            assert dataset_repo is not None
            assert isinstance(dataset_repo, InMemoryDatasetRepository)
            
            result_repo = container.result_repository()
            assert result_repo is not None
            assert isinstance(result_repo, InMemoryResultRepository)
    
    def test_container_adapter_providers(self):
        """Test container adapter providers."""
        container = Container()
        
        # Test adapter factory creation
        try:
            adapter_factory = container.adapter_factory()
            assert adapter_factory is not None
            
        except Exception as e:
            # Expected if dependencies are missing
            assert any(term in str(e).lower() for term in ["dependency", "import", "module"])
    
    def test_container_singleton_behavior(self):
        """Test container singleton behavior."""
        container = Container()
        
        # Test singleton behavior for repositories
        with patch.object(container, 'settings') as mock_settings:
            mock_settings_instance = Mock()
            mock_settings_instance.use_database_repositories = False
            mock_settings.return_value = mock_settings_instance
            
            repo1 = container.detector_repository()
            repo2 = container.detector_repository()
            
            # Should be the same instance
            assert repo1 is repo2


class TestAdapterRegistryWorking:
    """Working tests for adapter registry functionality."""
    
    def test_adapter_registry_basic_operations(self):
        """Test basic adapter registry operations."""
        from pynomaly.infrastructure.adapters import AdapterRegistry
        
        registry = AdapterRegistry()
        
        # Test initial state
        available_algorithms = registry.list_available()
        assert isinstance(available_algorithms, list)
        
        # Test registration
        mock_adapter_class = Mock()
        registry.register("TestAlgorithm", mock_adapter_class)
        
        # Test retrieval
        retrieved = registry.get_adapter_class("TestAlgorithm")
        assert retrieved is mock_adapter_class
        
        # Test availability
        assert registry.is_available("TestAlgorithm") is True
        assert registry.is_available("NonExistentAlgorithm") is False
        
        # Test updated list
        updated_algorithms = registry.list_available()
        assert "TestAlgorithm" in updated_algorithms
        
        # Test metadata registration
        registry.register("TestAlgorithm2", mock_adapter_class, {
            "type": "outlier_detection",
            "supports_streaming": True,
            "supports_multivariate": True
        })
        
        metadata = registry.get_metadata("TestAlgorithm2")
        assert metadata["type"] == "outlier_detection"
        assert metadata["supports_streaming"] is True
        assert metadata["supports_multivariate"] is True
        
        # Test metadata for non-existent algorithm
        empty_metadata = registry.get_metadata("NonExistent")
        assert empty_metadata == {}


class TestDataLoadersWorking:
    """Working tests for data loaders."""
    
    def test_csv_loader_with_mocking(self):
        """Test CSV loader with comprehensive mocking."""
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        
        loader = CSVLoader()
        
        # Mock pandas read_csv
        with patch('pandas.read_csv') as mock_read_csv:
            # Create mock dataframe
            mock_df = pd.DataFrame({
                'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature_2': [2.0, 4.0, 6.0, 8.0, 10.0],
                'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 0, 1, 0, 1]
            })
            mock_read_csv.return_value = mock_df
            
            # Test loading with target column
            dataset = loader.load('test_data.csv', target_column='target')
            
            assert dataset.name == 'test_data.csv'
            assert dataset.n_samples == 5
            assert dataset.n_features == 3  # Excludes target
            assert dataset.has_target is True
            assert dataset.target_column == 'target'
            
            # Test loading without target
            dataset_no_target = loader.load('no_target.csv')
            assert dataset_no_target.has_target is False
            
            # Test custom parameters
            loader.load('custom.csv', 
                       separator=';',
                       encoding='utf-8',
                       skip_rows=1)
            
            # Verify pandas was called correctly
            mock_read_csv.assert_called_with(
                'custom.csv',
                sep=';',
                encoding='utf-8',
                skiprows=1
            )
    
    def test_data_loader_error_handling(self):
        """Test data loader error handling."""
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        
        loader = CSVLoader()
        
        # Test file not found error
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                loader.load('nonexistent.csv')
        
        # Test parsing error
        with patch('pandas.read_csv', side_effect=pd.errors.ParserError("Bad CSV")):
            with pytest.raises(pd.errors.ParserError):
                loader.load('malformed.csv')


class TestSecurityComponentsWorking:
    """Working tests for security components."""
    
    def test_encryption_service_mocked(self):
        """Test encryption service with mocking."""
        try:
            from pynomaly.infrastructure.security.encryption import EncryptionService
            
            # Mock the Fernet class
            with patch('pynomaly.infrastructure.security.encryption.Fernet') as mock_fernet:
                mock_cipher = Mock()
                mock_cipher.encrypt.return_value = b'encrypted_test_data'
                mock_cipher.decrypt.return_value = b'original_test_data'
                mock_fernet.return_value = mock_cipher
                
                # Create service
                service = EncryptionService()
                
                # Test encryption
                plaintext = "sensitive data to encrypt"
                encrypted = service.encrypt(plaintext)
                assert encrypted is not None
                mock_cipher.encrypt.assert_called()
                
                # Test decryption
                decrypted = service.decrypt(encrypted)
                assert decrypted is not None
                mock_cipher.decrypt.assert_called()
                
                # Test multiple operations
                for i in range(3):
                    test_data = f"test_data_{i}"
                    enc_result = service.encrypt(test_data)
                    dec_result = service.decrypt(enc_result)
                    assert enc_result is not None
                    assert dec_result is not None
                
        except ImportError:
            pytest.skip("EncryptionService not available")
    
    def test_input_sanitizer_comprehensive(self):
        """Test input sanitizer comprehensively."""
        try:
            from pynomaly.infrastructure.security.input_sanitizer import InputSanitizer
            
            sanitizer = InputSanitizer()
            
            # Test safe inputs
            safe_inputs = [
                "normal text",
                "user@example.com",
                "filename.txt",
                "data123",
                "normal-filename_v2.csv"
            ]
            
            for safe_input in safe_inputs:
                sanitized = sanitizer.sanitize_text(safe_input)
                assert isinstance(sanitized, str)
                # Safe inputs should remain largely unchanged
                assert len(sanitized) > 0
            
            # Test potentially dangerous inputs
            dangerous_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "javascript:alert(1)",
                "data:text/html,<script>alert(1)</script>",
                "<img src=x onerror=alert(1)>",
                "SELECT * FROM users WHERE id = 1; --"
            ]
            
            for dangerous_input in dangerous_inputs:
                sanitized = sanitizer.sanitize_text(dangerous_input)
                # Should be sanitized (different from original or empty)
                assert sanitized != dangerous_input or len(sanitized) == 0
                
        except ImportError:
            pytest.skip("InputSanitizer not available")


class TestPerformanceComponentsWorking:
    """Working tests for performance and resilience components."""
    
    def test_circuit_breaker_basic(self):
        """Test circuit breaker basic functionality."""
        try:
            from pynomaly.infrastructure.resilience.circuit_breaker import CircuitBreaker
            
            # Create circuit breaker with low thresholds for testing
            cb = CircuitBreaker(
                failure_threshold=2,
                recovery_timeout=5,
                expected_exception=Exception
            )
            
            # Test initial state
            assert cb.state == "closed"
            assert cb.failure_count == 0
            
            # Test successful call
            call_count = 0
            
            @cb
            def test_function(should_fail=False):
                nonlocal call_count
                call_count += 1
                if should_fail:
                    raise Exception("Test failure")
                return "success"
            
            # Successful calls
            result = test_function(False)
            assert result == "success"
            assert cb.failure_count == 0
            
            # Trigger failures
            for i in range(2):
                try:
                    test_function(True)
                except Exception:
                    pass
            
            # Should be open now
            assert cb.state == "open"
            
        except ImportError:
            pytest.skip("CircuitBreaker not available")
    
    def test_retry_service_basic(self):
        """Test retry service basic functionality."""
        try:
            from pynomaly.infrastructure.resilience.retry import RetryService
            
            retry_service = RetryService(
                max_attempts=3,
                base_delay=0.01,  # Very short delay for testing
                max_delay=0.1
            )
            
            # Test successful retry after failures
            attempt_count = 0
            
            def flaky_function():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise Exception("Temporary failure")
                return "success_after_retries"
            
            result = retry_service.execute(flaky_function)
            assert result == "success_after_retries"
            assert attempt_count == 3
            
            # Test immediate success
            def immediate_success():
                return "immediate_success"
            
            result = retry_service.execute(immediate_success)
            assert result == "immediate_success"
            
        except ImportError:
            pytest.skip("RetryService not available")


class TestStreamingComponentsWorking:
    """Working tests for streaming components."""
    
    def test_stream_processor_basic(self):
        """Test stream processor basic functionality."""
        try:
            from pynomaly.infrastructure.streaming.processors import StreamProcessor
            
            processor = StreamProcessor()
            
            # Test basic data point processing
            data_point = {
                "timestamp": datetime.utcnow(),
                "features": [1.0, 2.0, 3.0, 4.0, 5.0],
                "metadata": {"source": "test_sensor"}
            }
            
            result = processor.process_data_point(data_point)
            assert result is not None
            
            # Test batch processing
            batch_data = []
            for i in range(5):
                batch_data.append({
                    "timestamp": datetime.utcnow(),
                    "features": [float(i), float(i*2), float(i*3)],
                    "metadata": {"batch_id": i}
                })
            
            batch_results = processor.process_batch(batch_data)
            assert len(batch_results) == 5
            
        except ImportError:
            pytest.skip("StreamProcessor not available")
    
    def test_redis_connector_mocked(self):
        """Test Redis connector with mocking."""
        try:
            from pynomaly.infrastructure.streaming.redis_connector import RedisConnector
            
            # Mock redis module
            with patch('redis.Redis') as mock_redis_class:
                mock_client = Mock()
                mock_client.ping.return_value = True
                mock_client.set.return_value = True
                mock_client.get.return_value = b'{"test": "data"}'
                mock_client.exists.return_value = 1
                mock_client.delete.return_value = 1
                mock_redis_class.return_value = mock_client
                
                connector = RedisConnector()
                
                # Test connection
                is_connected = connector.is_connected()
                assert is_connected is True
                mock_client.ping.assert_called()
                
                # Test data operations
                connector.set_value("test_key", {"test": "data"})
                mock_client.set.assert_called()
                
                value = connector.get_value("test_key")
                assert value is not None
                mock_client.get.assert_called()
                
        except ImportError:
            pytest.skip("RedisConnector not available")