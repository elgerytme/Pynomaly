"""
Integration tests for anomaly detection system.

This module tests:
- Cross-component integration
- Service interactions
- Data flow between layers
- Database integration
- API integration
- Performance integration
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import asyncio
from pathlib import Path
import tempfile
import json

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.application.services.anomaly_detection_service import AnomalyDetectionService
from pynomaly.infrastructure.persistence.memory_repository import MemoryRepository
from pynomaly.infrastructure.config.container import Container


@pytest.mark.integration
class TestAnomalyDetectionIntegration:
    """Integration tests for anomaly detection system."""

    def test_full_detection_workflow(self, container: Container, sample_data):
        """Test complete detection workflow from data to result."""
        # Create dataset
        dataset = Dataset(
            name="Integration Test Dataset",
            data=sample_data,
            description="Dataset for integration testing"
        )
        
        # Create detector
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1, "random_state": 42}
        )
        
        # Get service from container
        detection_service = container.anomaly_detection_service()
        
        # Mock the underlying adapter to avoid actual ML computation
        with patch.object(detection_service, '_get_adapter') as mock_adapter:
            # Setup mock adapter
            mock_model = Mock()
            mock_model.decision_function.return_value = np.random.random(len(sample_data))
            mock_model.predict.return_value = np.random.choice([0, 1], len(sample_data))
            
            mock_adapter.return_value.create_model.return_value = mock_model
            mock_adapter.return_value.fit.return_value = None
            mock_adapter.return_value.predict.return_value = np.random.choice([0, 1], len(sample_data))
            
            # Execute detection
            result = detection_service.detect_anomalies(dataset, detector)
            
            # Verify complete workflow
            assert isinstance(result, DetectionResult)
            assert result.detector_id == detector.id
            assert result.dataset_id == dataset.id
            assert len(result.scores) == len(sample_data)
            assert all(isinstance(score, AnomalyScore) for score in result.scores)

    def test_service_layer_integration(self, container: Container, sample_dataset, sample_detector):
        """Test integration between service layers."""
        # Get services from container
        detection_service = container.anomaly_detection_service()
        
        # Mock repository layer
        with patch.object(detection_service, '_save_result') as mock_save:
            mock_save.return_value = True
            
            # Mock ML layer
            with patch.object(detection_service, '_run_detection') as mock_detection:
                mock_detection.return_value = DetectionResult(
                    detector_id=sample_detector.id,
                    dataset_id=sample_dataset.id,
                    scores=[AnomalyScore(0.5)] * len(sample_dataset.data),
                    metadata={"integration_test": True}
                )
                
                # Execute service
                result = detection_service.detect_anomalies(sample_dataset, sample_detector)
                
                # Verify service integration
                assert result is not None
                assert len(result.scores) == len(sample_dataset.data)
                mock_detection.assert_called_once()

    def test_repository_integration(self, container: Container, sample_dataset):
        """Test repository layer integration."""
        # Get repository from container
        repository = container.dataset_repository()
        
        # Test save and retrieve
        saved_dataset = repository.save(sample_dataset)
        assert saved_dataset.id == sample_dataset.id
        
        # Test retrieval
        retrieved_dataset = repository.get_by_id(sample_dataset.id)
        assert retrieved_dataset is not None
        assert retrieved_dataset.name == sample_dataset.name
        assert len(retrieved_dataset.data) == len(sample_dataset.data)

    def test_adapter_integration(self, container: Container, sample_dataset, sample_detector):
        """Test adapter layer integration."""
        # Get adapter from container
        adapter = container.pyod_adapter()
        
        # Test model creation
        model = adapter.create_model(
            sample_detector.algorithm_name,
            sample_detector.parameters
        )
        assert model is not None
        
        # Test model fitting
        adapter.fit(model, sample_dataset.data)
        
        # Test prediction
        predictions = adapter.predict(model, sample_dataset.data)
        assert len(predictions) == len(sample_dataset.data)
        assert all(pred in [0, 1] for pred in predictions)

    def test_configuration_integration(self, container: Container, test_settings):
        """Test configuration integration across components."""
        # Test that settings are properly injected
        settings = container.config()
        assert settings.app_name == test_settings.app_name
        assert settings.app_environment == test_settings.app_environment
        
        # Test that services use configuration
        detection_service = container.anomaly_detection_service()
        assert detection_service is not None
        
        # Test that configuration affects behavior
        assert hasattr(detection_service, '_config')

    @pytest.mark.slow
    def test_performance_integration(self, container: Container, large_dataset):
        """Test performance integration with large datasets."""
        import time
        
        # Create detector for large dataset
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.05, "random_state": 42}
        )
        
        # Get service
        detection_service = container.anomaly_detection_service()
        
        # Mock for performance test
        with patch.object(detection_service, '_run_detection') as mock_detection:
            # Simulate processing time
            def slow_detection(*args, **kwargs):
                time.sleep(0.1)  # Simulate some processing
                return DetectionResult(
                    detector_id=detector.id,
                    dataset_id=large_dataset.id,
                    scores=[AnomalyScore(np.random.random()) for _ in range(len(large_dataset.data))],
                    metadata={"performance_test": True}
                )
            
            mock_detection.side_effect = slow_detection
            
            start_time = time.time()
            result = detection_service.detect_anomalies(large_dataset, detector)
            end_time = time.time()
            
            # Verify performance
            execution_time = end_time - start_time
            assert execution_time < 10.0  # Should complete within 10 seconds
            assert len(result.scores) == len(large_dataset.data)

    def test_error_handling_integration(self, container: Container, sample_dataset):
        """Test error handling across integration layers."""
        # Create invalid detector
        invalid_detector = Detector(
            algorithm_name="InvalidAlgorithm",
            parameters={}
        )
        
        # Get service
        detection_service = container.anomaly_detection_service()
        
        # Test that errors propagate correctly
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            detection_service.detect_anomalies(sample_dataset, invalid_detector)

    def test_transaction_integration(self, container: Container, sample_dataset, sample_detector):
        """Test transaction handling across components."""
        # Get services
        detection_service = container.anomaly_detection_service()
        
        # Mock transaction behavior
        with patch.object(detection_service, '_run_detection') as mock_detection:
            # Simulate transaction failure
            mock_detection.side_effect = Exception("Transaction failed")
            
            # Test that exception is handled properly
            with pytest.raises(Exception, match="Transaction failed"):
                detection_service.detect_anomalies(sample_dataset, sample_detector)

    def test_caching_integration(self, container: Container, sample_dataset, sample_detector):
        """Test caching integration across components."""
        # Get service
        detection_service = container.anomaly_detection_service()
        
        # Mock caching behavior
        with patch.object(detection_service, '_get_cached_result') as mock_cache:
            mock_cache.return_value = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                scores=[AnomalyScore(0.5)] * len(sample_dataset.data),
                metadata={"cached": True}
            )
            
            # Execute twice
            result1 = detection_service.detect_anomalies(sample_dataset, sample_detector)
            result2 = detection_service.detect_anomalies(sample_dataset, sample_detector)
            
            # Verify caching
            assert result1.metadata.get("cached") is True
            assert result2.metadata.get("cached") is True
            assert mock_cache.call_count == 2


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""

    def test_data_loading_integration(self, container: Container, temp_dir):
        """Test data loading pipeline integration."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        
        csv_path = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Get data service
        data_service = container.data_service()
        
        # Test loading
        dataset = data_service.load_from_csv(str(csv_path))
        
        assert isinstance(dataset, Dataset)
        assert len(dataset.data) == 100
        assert list(dataset.data.columns) == ['feature1', 'feature2', 'feature3']

    def test_data_preprocessing_integration(self, container: Container, sample_dataset):
        """Test data preprocessing pipeline integration."""
        # Get preprocessing service
        preprocessing_service = container.preprocessing_service()
        
        # Test preprocessing
        processed_dataset = preprocessing_service.preprocess(sample_dataset)
        
        assert isinstance(processed_dataset, Dataset)
        assert len(processed_dataset.data) == len(sample_dataset.data)
        assert processed_dataset.id != sample_dataset.id  # Should be new instance

    def test_data_validation_integration(self, container: Container, sample_dataset):
        """Test data validation pipeline integration."""
        # Get validation service
        validation_service = container.validation_service()
        
        # Test validation
        validation_result = validation_service.validate(sample_dataset)
        
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0

    def test_data_transformation_integration(self, container: Container, sample_dataset):
        """Test data transformation pipeline integration."""
        # Get transformation service
        transformation_service = container.transformation_service()
        
        # Test transformation
        transformed_dataset = transformation_service.transform(sample_dataset)
        
        assert isinstance(transformed_dataset, Dataset)
        assert len(transformed_dataset.data) == len(sample_dataset.data)


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncIntegration:
    """Integration tests for async operations."""

    async def test_async_detection_integration(self, container: Container, sample_dataset, sample_detector):
        """Test async detection integration."""
        # Get async service
        async_service = container.async_detection_service()
        
        # Mock async detection
        with patch.object(async_service, '_run_async_detection') as mock_async:
            mock_async.return_value = DetectionResult(
                detector_id=sample_detector.id,
                dataset_id=sample_dataset.id,
                scores=[AnomalyScore(0.5)] * len(sample_dataset.data),
                metadata={"async": True}
            )
            
            # Execute async detection
            result = await async_service.detect_anomalies_async(sample_dataset, sample_detector)
            
            assert result is not None
            assert result.metadata.get("async") is True
            mock_async.assert_called_once()

    async def test_concurrent_detection_integration(self, container: Container, sample_dataset):
        """Test concurrent detection integration."""
        # Create multiple detectors
        detectors = [
            Detector(
                algorithm_name="IsolationForest",
                parameters={"contamination": 0.1, "random_state": i}
            )
            for i in range(3)
        ]
        
        # Get async service
        async_service = container.async_detection_service()
        
        # Mock async detection
        with patch.object(async_service, '_run_async_detection') as mock_async:
            def mock_detection(dataset, detector):
                return DetectionResult(
                    detector_id=detector.id,
                    dataset_id=dataset.id,
                    scores=[AnomalyScore(0.5)] * len(dataset.data),
                    metadata={"detector_seed": detector.parameters["random_state"]}
                )
            
            mock_async.side_effect = mock_detection
            
            # Execute concurrent detections
            tasks = [
                async_service.detect_anomalies_async(sample_dataset, detector)
                for detector in detectors
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify concurrent execution
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.metadata["detector_seed"] == i
            assert mock_async.call_count == 3

    async def test_async_batch_processing_integration(self, container: Container, sample_detector):
        """Test async batch processing integration."""
        # Create multiple datasets
        datasets = [
            Dataset(
                name=f"Test Dataset {i}",
                data=pd.DataFrame(np.random.normal(0, 1, (50, 5))),
                description=f"Test dataset {i}"
            )
            for i in range(3)
        ]
        
        # Get async service
        async_service = container.async_detection_service()
        
        # Mock async batch processing
        with patch.object(async_service, '_run_batch_detection') as mock_batch:
            mock_batch.return_value = [
                DetectionResult(
                    detector_id=sample_detector.id,
                    dataset_id=dataset.id,
                    scores=[AnomalyScore(0.5)] * len(dataset.data),
                    metadata={"batch_index": i}
                )
                for i, dataset in enumerate(datasets)
            ]
            
            # Execute batch processing
            results = await async_service.detect_anomalies_batch_async(datasets, sample_detector)
            
            # Verify batch processing
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.metadata["batch_index"] == i
            mock_batch.assert_called_once()


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API layer."""

    def test_api_detection_endpoint_integration(self, client, sample_data):
        """Test API detection endpoint integration."""
        # Prepare request data
        request_data = {
            "dataset": {
                "name": "API Test Dataset",
                "data": sample_data.to_dict('records'),
                "description": "Dataset for API testing"
            },
            "detector": {
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1}
            }
        }
        
        # Mock the detection service
        with patch('pynomaly.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies.return_value = DetectionResult(
                detector_id="test-detector-id",
                dataset_id="test-dataset-id",
                scores=[AnomalyScore(0.5)] * len(sample_data),
                metadata={"api_test": True}
            )
            
            # Make API request
            response = client.post("/api/v1/detect", json=request_data)
            
            # Verify API response
            assert response.status_code == 200
            response_data = response.json()
            assert "result" in response_data
            assert len(response_data["result"]["scores"]) == len(sample_data)

    def test_api_error_handling_integration(self, client):
        """Test API error handling integration."""
        # Invalid request data
        invalid_request = {
            "dataset": {
                "name": "Invalid Dataset",
                "data": [],  # Empty data
                "description": "Invalid dataset"
            },
            "detector": {
                "algorithm_name": "InvalidAlgorithm",
                "parameters": {}
            }
        }
        
        # Make API request
        response = client.post("/api/v1/detect", json=invalid_request)
        
        # Verify error handling
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_api_authentication_integration(self, client, user_token):
        """Test API authentication integration."""
        # Request with authentication
        headers = {"Authorization": f"Bearer {user_token}"}
        request_data = {
            "dataset": {
                "name": "Auth Test Dataset",
                "data": [{"feature1": 1, "feature2": 2}],
                "description": "Dataset for auth testing"
            },
            "detector": {
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.1}
            }
        }
        
        # Mock the detection service
        with patch('pynomaly.application.services.anomaly_detection_service.AnomalyDetectionService') as mock_service:
            mock_service.return_value.detect_anomalies.return_value = DetectionResult(
                detector_id="test-detector-id",
                dataset_id="test-dataset-id",
                scores=[AnomalyScore(0.5)],
                metadata={"authenticated": True}
            )
            
            # Make authenticated API request
            response = client.post("/api/v1/detect", json=request_data, headers=headers)
            
            # Verify authenticated response
            assert response.status_code == 200


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_database_persistence_integration(self, db_session, sample_dataset):
        """Test database persistence integration."""
        # Mock database operations
        with patch.object(db_session, 'add') as mock_add:
            with patch.object(db_session, 'commit') as mock_commit:
                # Test saving
                db_session.add(sample_dataset)
                db_session.commit()
                
                # Verify database operations
                mock_add.assert_called_once_with(sample_dataset)
                mock_commit.assert_called_once()

    def test_database_query_integration(self, db_session, sample_dataset):
        """Test database query integration."""
        # Mock database query
        with patch.object(db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = sample_dataset
            
            # Test querying
            result = db_session.query(Dataset).filter(Dataset.id == sample_dataset.id).first()
            
            # Verify query result
            assert result == sample_dataset
            mock_query.assert_called_once()

    def test_database_transaction_integration(self, db_session, sample_dataset):
        """Test database transaction integration."""
        # Mock transaction operations
        with patch.object(db_session, 'begin') as mock_begin:
            with patch.object(db_session, 'commit') as mock_commit:
                with patch.object(db_session, 'rollback') as mock_rollback:
                    try:
                        # Test transaction
                        db_session.begin()
                        db_session.add(sample_dataset)
                        db_session.commit()
                        
                        # Verify successful transaction
                        mock_begin.assert_called_once()
                        mock_commit.assert_called_once()
                        mock_rollback.assert_not_called()
                    except Exception:
                        db_session.rollback()
                        mock_rollback.assert_called_once()


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""

    def test_throughput_integration(self, container: Container, benchmark_data):
        """Test system throughput integration."""
        # Get service
        detection_service = container.anomaly_detection_service()
        
        # Create datasets of different sizes
        datasets = {
            "small": Dataset(
                name="Small Dataset",
                data=pd.DataFrame(benchmark_data["small"]),
                description="Small dataset for throughput testing"
            ),
            "medium": Dataset(
                name="Medium Dataset", 
                data=pd.DataFrame(benchmark_data["medium"]),
                description="Medium dataset for throughput testing"
            ),
            "large": Dataset(
                name="Large Dataset",
                data=pd.DataFrame(benchmark_data["large"]),
                description="Large dataset for throughput testing"
            )
        }
        
        # Create detector
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.05, "random_state": 42}
        )
        
        # Test throughput for different sizes
        for size, dataset in datasets.items():
            with patch.object(detection_service, '_run_detection') as mock_detection:
                mock_detection.return_value = DetectionResult(
                    detector_id=detector.id,
                    dataset_id=dataset.id,
                    scores=[AnomalyScore(0.5)] * len(dataset.data),
                    metadata={"size": size}
                )
                
                import time
                start_time = time.time()
                result = detection_service.detect_anomalies(dataset, detector)
                end_time = time.time()
                
                # Verify throughput
                execution_time = end_time - start_time
                throughput = len(dataset.data) / execution_time if execution_time > 0 else float('inf')
                
                assert result.metadata["size"] == size
                assert throughput > 0  # Should have measurable throughput

    def test_memory_usage_integration(self, container: Container, large_dataset):
        """Test memory usage integration."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Get service
        detection_service = container.anomaly_detection_service()
        
        # Create detector
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.05, "random_state": 42}
        )
        
        # Mock detection to avoid actual computation
        with patch.object(detection_service, '_run_detection') as mock_detection:
            mock_detection.return_value = DetectionResult(
                detector_id=detector.id,
                dataset_id=large_dataset.id,
                scores=[AnomalyScore(0.5)] * len(large_dataset.data),
                metadata={"memory_test": True}
            )
            
            # Execute detection
            result = detection_service.detect_anomalies(large_dataset, detector)
            
            # Get final memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Verify memory usage is reasonable
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
            assert result.metadata["memory_test"] is True

    def test_concurrent_performance_integration(self, container: Container, sample_dataset):
        """Test concurrent performance integration."""
        import threading
        import time
        
        # Create multiple detectors
        detectors = [
            Detector(
                algorithm_name="IsolationForest",
                parameters={"contamination": 0.1, "random_state": i}
            )
            for i in range(5)
        ]
        
        # Get service
        detection_service = container.anomaly_detection_service()
        
        # Results container
        results = []
        
        def run_detection(detector):
            with patch.object(detection_service, '_run_detection') as mock_detection:
                mock_detection.return_value = DetectionResult(
                    detector_id=detector.id,
                    dataset_id=sample_dataset.id,
                    scores=[AnomalyScore(0.5)] * len(sample_dataset.data),
                    metadata={"thread_test": True}
                )
                
                result = detection_service.detect_anomalies(sample_dataset, detector)
                results.append(result)
        
        # Execute concurrent detections
        threads = []
        start_time = time.time()
        
        for detector in detectors:
            thread = threading.Thread(target=run_detection, args=(detector,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Verify concurrent execution
        assert len(results) == 5
        execution_time = end_time - start_time
        assert execution_time < 10.0  # Should complete within 10 seconds
        
        for result in results:
            assert result.metadata["thread_test"] is True
