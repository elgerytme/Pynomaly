"""
SDK Integration Testing

Comprehensive tests for both Python and JavaScript SDKs,
including cross-SDK compatibility and workflow validation.
"""

import pytest
import asyncio
import time
import tempfile
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Python SDK imports
import sys
sys.path.insert(0, '/mnt/c/Users/andre/Pynomaly/src/packages/sdk')

try:
    from pynomaly_sdk import PynomalyClient, Dataset, DetectorConfig
    from pynomaly_sdk.exceptions import PynomalySDKError, AuthenticationError, APIError
    PYTHON_SDK_AVAILABLE = True
except ImportError:
    PYTHON_SDK_AVAILABLE = False
    PynomalyClient = None

from .conftest import (
    assert_api_response_valid,
    assert_detection_quality,
    E2ETestConfig
)


@pytest.mark.asyncio
@pytest.mark.sdk
@pytest.mark.skipif(not PYTHON_SDK_AVAILABLE, reason="Python SDK not available")
class TestPythonSDKIntegration:
    """Test Python SDK integration with the API."""

    async def test_python_sdk_basic_workflow(
        self,
        test_client,
        sample_dataset,
        performance_monitor
    ):
        """Test basic workflow using Python SDK."""
        
        performance_monitor.start_timer("python_sdk_workflow")
        
        # Initialize Python SDK client
        client = PynomalyClient(
            base_url="http://testserver",
            api_key="test-api-key",
            timeout=30.0
        )
        
        try:
            # Test health check
            health = await client.health_check()
            assert health["status"] == "healthy"
            
            # Create detector using SDK
            detector_config = DetectorConfig(
                algorithm_name="IsolationForest",
                hyperparameters={"n_estimators": 50},
                contamination_rate=0.05,
                random_state=42
            )
            
            detector = await client.data_science.create_detector(
                name="python-sdk-detector",
                config=detector_config,
                description="Test detector created via Python SDK"
            )
            
            assert detector["name"] == "python-sdk-detector"
            assert detector["algorithm_name"] == "IsolationForest"
            detector_id = detector["id"]
            
            # Prepare dataset using SDK
            training_data = sample_dataset['features'].iloc[:500]
            dataset = Dataset(
                name="python-sdk-training-data",
                data=training_data.values.tolist(),
                feature_names=training_data.columns.tolist(),
                metadata={"source": "test", "size": len(training_data)}
            )
            
            # Train detector using SDK
            training_job = await client.data_science.train_detector(
                detector_id=detector_id,
                dataset=dataset,
                job_name="python-sdk-training"
            )
            
            assert training_job.name == "python-sdk-training"
            assert training_job.status in ["pending", "running"]
            
            # Monitor training progress
            max_wait = 60
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                job_status = await client.data_science.get_training_job(training_job.job_id)
                
                if job_status.status == "completed":
                    break
                elif job_status.status == "failed":
                    pytest.fail(f"Training failed: {job_status.error_message}")
                
                await asyncio.sleep(2)
            else:
                pytest.fail("Training did not complete within expected time")
            
            # Perform detection using SDK
            test_data = sample_dataset['features'].iloc[500:600]
            test_dataset = Dataset(
                name="python-sdk-test-data",
                data=test_data.values.tolist(),
                feature_names=test_data.columns.tolist()
            )
            
            detection_result = await client.data_science.detect_anomalies(
                detector_id=detector_id,
                dataset=test_dataset,
                options={"return_scores": True, "threshold": None}
            )
            
            # Validate results
            assert detection_result.n_samples == len(test_data)
            assert len(detection_result.anomaly_scores) == len(test_data)
            assert len(detection_result.anomaly_labels) == len(test_data)
            assert_detection_quality(
                {
                    "n_anomalies": detection_result.n_anomalies,
                    "n_samples": detection_result.n_samples
                },
                sample_dataset['anomaly_rate'],
                tolerance=0.2
            )
            
            # Test SDK utility functions
            anomaly_indices = detection_result.anomaly_indices
            normal_indices = detection_result.normal_indices
            
            assert len(anomaly_indices) == detection_result.n_anomalies
            assert len(normal_indices) == detection_result.n_samples - detection_result.n_anomalies
            assert len(set(anomaly_indices) & set(normal_indices)) == 0  # No overlap
            
            # List detectors
            detector_list = await client.data_science.list_detectors(
                options={"page": 1, "page_size": 10}
            )
            
            assert detector_list.total >= 1
            assert any(d["id"] == detector_id for d in detector_list.items)
            
            # Update detector
            updated_detector = await client.data_science.update_detector(
                detector_id=detector_id,
                updates={"description": "Updated via Python SDK"}
            )
            
            assert updated_detector["description"] == "Updated via Python SDK"
            
            # Delete detector
            deletion_result = await client.data_science.delete_detector(detector_id)
            assert deletion_result is True
            
            # Verify deletion
            try:
                await client.data_science.get_detector(detector_id)
                pytest.fail("Detector should have been deleted")
            except Exception as e:
                assert "not found" in str(e).lower() or "404" in str(e)
        
        finally:
            await client.close()
        
        performance_monitor.end_timer("python_sdk_workflow")
        workflow_duration = performance_monitor.get_duration("python_sdk_workflow")
        
        # Should complete within reasonable time
        assert workflow_duration < 120.0, f"Python SDK workflow took {workflow_duration:.2f}s"

    async def test_python_sdk_error_handling(
        self,
        test_client
    ):
        """Test Python SDK error handling."""
        
        # Initialize client with invalid API key
        client = PynomalyClient(
            base_url="http://testserver",
            api_key="invalid-api-key",
            timeout=10.0
        )
        
        try:
            # Test authentication error
            with pytest.raises((AuthenticationError, APIError)):
                await client.data_science.create_detector(
                    name="test-detector",
                    config=DetectorConfig(algorithm_name="IsolationForest")
                )
            
            # Test with valid client
            valid_client = PynomalyClient(
                base_url="http://testserver",
                api_key="test-api-key",
                timeout=10.0
            )
            
            try:
                # Test resource not found error
                with pytest.raises(Exception) as exc_info:
                    await valid_client.data_science.get_detector("non-existent-id")
                
                assert "not found" in str(exc_info.value).lower() or "404" in str(exc_info.value)
                
                # Test validation error with invalid data
                invalid_dataset = Dataset(
                    name="invalid-dataset",
                    data=[],  # Empty data
                    feature_names=["feature1", "feature2"]
                )
                
                # This should either raise an error immediately or during training
                detector_config = DetectorConfig(algorithm_name="IsolationForest")
                detector = await valid_client.data_science.create_detector(
                    name="test-detector-for-error",
                    config=detector_config
                )
                
                try:
                    await valid_client.data_science.train_detector(
                        detector_id=detector["id"],
                        dataset=invalid_dataset
                    )
                    # If training starts, it should eventually fail
                    await asyncio.sleep(5)
                except Exception as e:
                    # Expected to fail with validation or data error
                    assert any(keyword in str(e).lower() for keyword in ["empty", "invalid", "data", "validation"])
                
                # Cleanup
                await valid_client.data_science.delete_detector(detector["id"])
                
            finally:
                await valid_client.close()
        
        finally:
            await client.close()

    async def test_python_sdk_data_utilities(
        self,
        sample_dataset,
        multimodal_dataset,
        time_series_dataset
    ):
        """Test Python SDK data processing utilities."""
        
        from pynomaly_sdk.utils import DatasetConverter, DataValidator, ResultAnalyzer
        
        # Test DatasetConverter
        df = sample_dataset['dataframe']
        
        # Convert from DataFrame
        dataset_from_df = DatasetConverter.from_dataframe(
            name="test-dataset",
            df=df[sample_dataset['feature_names']],
            target_column=None
        )
        
        assert dataset_from_df.name == "test-dataset"
        assert dataset_from_df.feature_names == sample_dataset['feature_names']
        assert len(dataset_from_df.data) == len(df)
        
        # Convert from numpy array
        data_array = df[sample_dataset['feature_names']].values
        dataset_from_numpy = DatasetConverter.from_numpy(
            name="numpy-dataset",
            data=data_array,
            feature_names=sample_dataset['feature_names']
        )
        
        assert dataset_from_numpy.name == "numpy-dataset"
        assert len(dataset_from_numpy.data) == len(data_array)
        
        # Test DataValidator
        validation_result = DataValidator.validate(dataset_from_df)
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
        
        # Test with invalid data
        invalid_dataset = Dataset(
            name="invalid",
            data=[],
            feature_names=["feature1"]
        )
        
        invalid_validation = DataValidator.validate(invalid_dataset)
        assert invalid_validation.is_valid is False
        assert "empty" in invalid_validation.errors[0].lower()
        
        # Test statistics
        stats = DataValidator.get_statistics(dataset_from_df)
        assert stats.rows == len(df)
        assert stats.columns == len(sample_dataset['feature_names'])
        assert stats.numeric_columns == len(sample_dataset['feature_names'])
        
        # Test CSV conversion
        csv_data = df[sample_dataset['feature_names']].to_csv(index=False)
        dataset_from_csv = DatasetConverter.from_csv(
            name="csv-dataset",
            csv_string=csv_data,
            options={"has_header": True}
        )
        
        assert dataset_from_csv.name == "csv-dataset"
        assert len(dataset_from_csv.data) == len(df)


@pytest.mark.asyncio
@pytest.mark.sdk
class TestCrossSDKCompatibility:
    """Test compatibility between different SDK implementations."""

    async def test_data_format_compatibility(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test that data formats are compatible across SDKs."""
        
        # Create detector via REST API
        detector_config = {
            "name": "cross-sdk-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 50},
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Train via REST API
        training_data = sample_dataset['features'].iloc[:300]
        rest_dataset = {
            "name": "cross-sdk-training",
            "data": training_data.values.tolist(),
            "feature_names": training_data.columns.tolist()
        }
        
        training_payload = {
            "detector_id": detector_id,
            "dataset": rest_dataset,
            "job_name": "cross-sdk-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Wait for training completion
        await self._wait_for_job_completion(async_client, training_job["job_id"], api_headers)
        
        # Test detection via REST API
        test_data = sample_dataset['features'].iloc[300:350]
        rest_test_dataset = {
            "name": "cross-sdk-test",
            "data": test_data.values.tolist(),
            "feature_names": test_data.columns.tolist()
        }
        
        detection_payload = {
            "detector_id": detector_id,
            "dataset": rest_test_dataset,
            "return_scores": True
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        rest_result = response.json()
        
        # Verify data format consistency
        assert "anomaly_scores" in rest_result
        assert "anomaly_labels" in rest_result
        assert "n_anomalies" in rest_result
        assert "n_samples" in rest_result
        assert "threshold" in rest_result
        assert "execution_time" in rest_result
        
        # Test that the same data produces consistent results
        # (Run detection again with same data)
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        rest_result_2 = response.json()
        
        # Results should be identical for deterministic algorithms
        assert rest_result_2["n_samples"] == rest_result["n_samples"]
        assert rest_result_2["threshold"] == rest_result["threshold"]
        
        # Scores should be very similar (allowing for minor floating point differences)
        score_differences = [
            abs(s1 - s2) for s1, s2 in zip(rest_result["anomaly_scores"], rest_result_2["anomaly_scores"])
        ]
        max_difference = max(score_differences)
        assert max_difference < 1e-6, f"Scores differ by more than expected: {max_difference}"
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_api_contract_validation(
        self,
        async_client,
        api_headers
    ):
        """Test that API contracts are consistent and well-defined."""
        
        # Test API schema consistency
        response = await async_client.get("/docs", headers=api_headers)
        # OpenAPI documentation should be available
        assert response.status_code in [200, 404]  # 404 if docs not enabled in test
        
        # Test health endpoint schema
        response = await async_client.get("/api/v1/health", headers=api_headers)
        assert_api_response_valid(response)
        health_data = response.json()
        
        # Health response should have expected structure
        required_health_fields = {"status", "version", "timestamp"}
        health_fields = set(health_data.keys())
        assert required_health_fields.issubset(health_fields), f"Missing health fields: {required_health_fields - health_fields}"
        
        # Test detector schema consistency
        detector_config = {
            "name": "schema-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        
        # Detector response should have expected structure
        required_detector_fields = {"id", "name", "algorithm_name", "contamination_rate", "created_at"}
        detector_fields = set(detector.keys())
        assert required_detector_fields.issubset(detector_fields), f"Missing detector fields: {required_detector_fields - detector_fields}"
        
        # Test field types
        assert isinstance(detector["id"], str)
        assert isinstance(detector["name"], str)
        assert isinstance(detector["algorithm_name"], str)
        assert isinstance(detector["contamination_rate"], (int, float))
        assert isinstance(detector["created_at"], str)
        
        # Test list endpoint schema
        response = await async_client.get("/api/v1/detectors", headers=api_headers)
        assert_api_response_valid(response)
        detector_list = response.json()
        
        # List response should have pagination structure
        required_list_fields = {"items", "total", "page", "page_size"}
        list_fields = set(detector_list.keys())
        assert required_list_fields.issubset(list_fields), f"Missing list fields: {required_list_fields - list_fields}"
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector['id']}", headers=api_headers)

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ):
        """Helper method to wait for training job completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] == "completed":
                return job_status
            elif job_status["status"] == "failed":
                pytest.fail(f"Training failed: {job_status.get('error_message', 'Unknown error')}")
                
            await asyncio.sleep(2)
        
        pytest.fail("Training did not complete within expected time")


@pytest.mark.asyncio
@pytest.mark.sdk
class TestSDKPerformance:
    """Test SDK performance and efficiency."""

    @pytest.mark.skipif(not PYTHON_SDK_AVAILABLE, reason="Python SDK not available")
    async def test_python_sdk_performance(
        self,
        test_client,
        sample_dataset,
        performance_monitor
    ):
        """Test Python SDK performance characteristics."""
        
        client = PynomalyClient(
            base_url="http://testserver",
            api_key="test-api-key",
            timeout=30.0
        )
        
        try:
            # Test rapid detector creation
            performance_monitor.start_timer("sdk_rapid_creation")
            
            detector_ids = []
            for i in range(5):
                detector_config = DetectorConfig(
                    algorithm_name="IsolationForest",
                    contamination_rate=0.05
                )
                
                detector = await client.data_science.create_detector(
                    name=f"perf-detector-{i}",
                    config=detector_config
                )
                detector_ids.append(detector["id"])
            
            performance_monitor.end_timer("sdk_rapid_creation")
            creation_time = performance_monitor.get_duration("sdk_rapid_creation")
            
            # Should create 5 detectors in reasonable time
            assert creation_time < 15.0, f"SDK detector creation took {creation_time:.2f}s"
            
            # Test rapid listing
            performance_monitor.start_timer("sdk_listing")
            
            detector_list = await client.data_science.list_detectors(
                options={"page_size": 20}
            )
            
            performance_monitor.end_timer("sdk_listing")
            listing_time = performance_monitor.get_duration("sdk_listing")
            
            assert listing_time < 5.0, f"SDK listing took {listing_time:.2f}s"
            assert detector_list.total >= 5
            
            # Test rapid deletion
            performance_monitor.start_timer("sdk_rapid_deletion")
            
            for detector_id in detector_ids:
                await client.data_science.delete_detector(detector_id)
            
            performance_monitor.end_timer("sdk_rapid_deletion")
            deletion_time = performance_monitor.get_duration("sdk_rapid_deletion")
            
            assert deletion_time < 10.0, f"SDK deletion took {deletion_time:.2f}s"
        
        finally:
            await client.close()

    async def test_sdk_memory_efficiency(
        self,
        sample_dataset
    ):
        """Test SDK memory usage efficiency."""
        
        if not PYTHON_SDK_AVAILABLE:
            pytest.skip("Python SDK not available")
        
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple SDK clients and perform operations
        clients = []
        
        try:
            for i in range(3):
                client = PynomalyClient(
                    base_url="http://testserver",
                    api_key="test-api-key",
                    timeout=30.0
                )
                clients.append(client)
                
                # Test data processing utilities
                from pynomaly_sdk.utils import DataValidator, ResultAnalyzer
                
                # Create datasets for validation
                for j in range(5):
                    dataset = Dataset(
                        name=f"memory-test-{i}-{j}",
                        data=sample_dataset['features'].iloc[:100].values.tolist(),
                        feature_names=sample_dataset['feature_names']
                    )
                    
                    # Validate dataset
                    validation_result = DataValidator.validate(dataset)
                    assert validation_result.is_valid
                    
                    # Get statistics
                    stats = DataValidator.get_statistics(dataset)
                    assert stats.rows == 100
            
            # Check memory after operations
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be reasonable (less than 100MB for this test)
            assert memory_growth < 100, f"SDK memory usage grew by {memory_growth:.1f}MB"
        
        finally:
            # Cleanup clients
            for client in clients:
                await client.close()
            
            # Check final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_growth = final_memory - initial_memory
            
            # After cleanup, memory growth should be minimal
            assert final_growth < 50, f"Final memory growth of {final_growth:.1f}MB indicates possible leak"