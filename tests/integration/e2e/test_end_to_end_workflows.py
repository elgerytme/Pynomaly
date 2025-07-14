"""
End-to-End Workflow Integration Tests

Comprehensive tests covering complete anomaly detection workflows
from data ingestion to result visualization.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .conftest import (
    assert_detection_quality,
    assert_performance_within_limits,
    assert_api_response_valid
)


@pytest.mark.asyncio
@pytest.mark.e2e
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    async def test_basic_anomaly_detection_workflow(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test basic anomaly detection workflow: create detector → train → detect."""
        
        performance_monitor.start_timer("complete_workflow")
        
        # Step 1: Create detector
        detector_config = {
            "name": "test-detector-workflow",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 100, "contamination": 0.05},
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
        
        # Step 2: Prepare training data
        training_data = sample_dataset['features'].iloc[:800]  # 80% for training
        dataset_payload = {
            "name": "training-dataset",
            "data": training_data.values.tolist(),
            "feature_names": training_data.columns.tolist()
        }
        
        # Step 3: Train detector
        training_payload = {
            "detector_id": detector_id,
            "dataset": dataset_payload,
            "job_name": "e2e-training-job"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        job_id = training_job["job_id"]
        
        # Step 4: Monitor training progress
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await async_client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=api_headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] == "completed":
                break
            elif job_status["status"] == "failed":
                pytest.fail(f"Training failed: {job_status.get('error_message', 'Unknown error')}")
                
            await asyncio.sleep(2)
        else:
            pytest.fail("Training did not complete within expected time")
        
        # Step 5: Perform detection
        test_data = sample_dataset['features'].iloc[800:]  # 20% for testing
        detection_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "test-dataset",
                "data": test_data.values.tolist(),
                "feature_names": test_data.columns.tolist()
            },
            "return_scores": True
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        detection_result = response.json()
        
        # Step 6: Validate results
        expected_anomaly_rate = sample_dataset['anomaly_rate']
        assert_detection_quality(detection_result, expected_anomaly_rate, tolerance=0.15)
        
        # Verify result structure
        assert "anomaly_scores" in detection_result
        assert "anomaly_labels" in detection_result
        assert "n_anomalies" in detection_result
        assert "n_samples" in detection_result
        assert "threshold" in detection_result
        assert "execution_time" in detection_result
        
        # Verify data consistency
        assert len(detection_result["anomaly_scores"]) == len(test_data)
        assert len(detection_result["anomaly_labels"]) == len(test_data)
        assert detection_result["n_samples"] == len(test_data)
        
        performance_monitor.end_timer("complete_workflow")
        workflow_duration = performance_monitor.get_duration("complete_workflow")
        
        # Performance assertion (should complete within 2 minutes)
        assert_performance_within_limits(workflow_duration, 120.0)
        
        # Cleanup
        response = await async_client.delete(
            f"/api/v1/detectors/{detector_id}",
            headers=api_headers
        )
        assert_api_response_valid(response, 204)

    async def test_multimodal_detection_workflow(
        self,
        async_client,
        multimodal_dataset,
        api_headers,
        performance_monitor
    ):
        """Test multimodal data detection workflow."""
        
        performance_monitor.start_timer("multimodal_workflow")
        
        # Create detector optimized for multimodal data
        detector_config = {
            "name": "multimodal-detector",
            "algorithm_name": "LocalOutlierFactor",
            "hyperparameters": {"n_neighbors": 20, "contamination": 0.1},
            "contamination_rate": 0.1
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Train and test with multimodal data
        training_data = multimodal_dataset['features']
        dataset_payload = {
            "name": "multimodal-dataset",
            "data": training_data.values.tolist(),
            "feature_names": training_data.columns.tolist()
        }
        
        # Train detector
        training_payload = {
            "detector_id": detector_id,
            "dataset": dataset_payload,
            "job_name": "multimodal-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Wait for training completion
        job_id = training_job["job_id"]
        await self._wait_for_job_completion(async_client, job_id, api_headers)
        
        # Perform detection
        detection_payload = {
            "detector_id": detector_id,
            "dataset": dataset_payload,
            "return_scores": True
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        result = response.json()
        
        # Validate multimodal detection quality
        assert_detection_quality(result, multimodal_dataset['anomaly_rate'], tolerance=0.2)
        
        performance_monitor.end_timer("multimodal_workflow")
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_time_series_detection_workflow(
        self,
        async_client,
        time_series_dataset,
        api_headers,
        performance_monitor
    ):
        """Test time series anomaly detection workflow."""
        
        performance_monitor.start_timer("time_series_workflow")
        
        # Create detector for time series data
        detector_config = {
            "name": "time-series-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 200, "contamination": 0.05},
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
        
        # Prepare time series features (using sliding window approach)
        ts_data = time_series_dataset['features']
        
        # Create windowed features for better temporal context
        window_size = 10
        windowed_features = []
        valid_indices = []
        
        for i in range(window_size, len(ts_data)):
            window = ts_data.iloc[i-window_size:i]['value'].values
            windowed_features.append([
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                window[-1] - window[0]  # Trend
            ])
            valid_indices.append(i)
        
        windowed_features = np.array(windowed_features)
        
        dataset_payload = {
            "name": "time-series-dataset",
            "data": windowed_features.tolist(),
            "feature_names": ["mean", "std", "range", "trend"]
        }
        
        # Train detector
        training_payload = {
            "detector_id": detector_id,
            "dataset": dataset_payload,
            "job_name": "time-series-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Wait for training completion
        job_id = training_job["job_id"]
        await self._wait_for_job_completion(async_client, job_id, api_headers)
        
        # Perform detection
        detection_payload = {
            "detector_id": detector_id,
            "dataset": dataset_payload,
            "return_scores": True
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        result = response.json()
        
        # Validate time series detection
        # Note: Time series detection might be less accurate due to temporal dependencies
        assert result["n_samples"] == len(windowed_features)
        assert len(result["anomaly_scores"]) == len(windowed_features)
        
        performance_monitor.end_timer("time_series_workflow")
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_batch_processing_workflow(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test batch processing workflow with multiple datasets."""
        
        performance_monitor.start_timer("batch_workflow")
        
        # Create detector
        detector_config = {
            "name": "batch-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 100},
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
        
        # Train detector
        training_data = sample_dataset['features'].iloc[:500]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "batch-training-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "batch-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Wait for training completion
        job_id = training_job["job_id"]
        await self._wait_for_job_completion(async_client, job_id, api_headers)
        
        # Prepare multiple datasets for batch processing
        remaining_data = sample_dataset['features'].iloc[500:]
        batch_size = 100
        datasets = []
        
        for i in range(0, len(remaining_data), batch_size):
            batch_data = remaining_data.iloc[i:i+batch_size]
            datasets.append({
                "name": f"batch-dataset-{i//batch_size}",
                "data": batch_data.values.tolist(),
                "feature_names": batch_data.columns.tolist()
            })
        
        # Submit batch detection job
        batch_payload = {
            "detector_id": detector_id,
            "datasets": datasets,
            "job_name": "batch-detection-job"
        }
        
        response = await async_client.post(
            "/api/v1/detection/batch",
            json=batch_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        batch_job = response.json()
        
        # Monitor batch job (if implemented)
        # For now, just verify the job was created successfully
        assert "job_id" in batch_job
        assert batch_job["status"] in ["pending", "running"]
        
        performance_monitor.end_timer("batch_workflow")
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_model_lifecycle_workflow(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test complete model lifecycle: create → train → validate → deploy → monitor."""
        
        performance_monitor.start_timer("lifecycle_workflow")
        
        # Step 1: Create detector
        detector_config = {
            "name": "lifecycle-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 50},
            "contamination_rate": 0.05,
            "description": "Test detector for lifecycle workflow"
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Step 2: Train model
        training_data = sample_dataset['features'].iloc[:700]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "lifecycle-training-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "lifecycle-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        job_id = training_job["job_id"]
        
        # Wait for training completion
        await self._wait_for_job_completion(async_client, job_id, api_headers)
        
        # Step 3: Validate model performance
        validation_data = sample_dataset['features'].iloc[700:850]
        validation_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "validation-data",
                "data": validation_data.values.tolist(),
                "feature_names": validation_data.columns.tolist()
            },
            "return_scores": True
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=validation_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        validation_result = response.json()
        
        # Check validation metrics
        assert validation_result["n_samples"] == len(validation_data)
        validation_anomaly_rate = validation_result["n_anomalies"] / validation_result["n_samples"]
        assert 0.0 <= validation_anomaly_rate <= 0.3  # Reasonable bounds
        
        # Step 4: Test production detection
        test_data = sample_dataset['features'].iloc[850:]
        test_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "production-test-data",
                "data": test_data.values.tolist(),
                "feature_names": test_data.columns.tolist()
            },
            "return_scores": True
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=test_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        production_result = response.json()
        
        # Verify production readiness
        assert production_result["execution_time"] < 10.0  # Should be fast
        assert_detection_quality(production_result, sample_dataset['anomaly_rate'], tolerance=0.2)
        
        # Step 5: Update detector configuration
        update_payload = {
            "description": "Updated detector after lifecycle testing",
            "config": {
                "algorithm_name": "IsolationForest",
                "hyperparameters": {"n_estimators": 100},  # Increased estimators
                "contamination_rate": 0.05
            }
        }
        
        response = await async_client.put(
            f"/api/v1/detectors/{detector_id}",
            json=update_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        updated_detector = response.json()
        
        # Verify update
        assert updated_detector["description"] == update_payload["description"]
        
        performance_monitor.end_timer("lifecycle_workflow")
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

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
@pytest.mark.e2e
class TestWorkflowResilience:
    """Test workflow resilience and error recovery."""

    async def test_training_interruption_recovery(
        self,
        async_client,
        sample_dataset,
        api_headers
    ):
        """Test recovery from training interruption."""
        
        # Create detector
        detector_config = {
            "name": "interruption-test-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 10},
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
        
        # Start training
        training_data = sample_dataset['features'].iloc[:100]  # Small dataset for quick training
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "interruption-test-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "interruption-test-job"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        job_id = training_job["job_id"]
        
        # Wait for completion (should succeed with small dataset)
        start_time = time.time()
        max_wait_time = 30
        
        while time.time() - start_time < max_wait_time:
            response = await async_client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=api_headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
        
        # Verify we can still use the detector (even if training failed)
        response = await async_client.get(
            f"/api/v1/detectors/{detector_id}",
            headers=api_headers
        )
        assert_api_response_valid(response)
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_invalid_data_handling(
        self,
        async_client,
        api_headers
    ):
        """Test handling of invalid data formats."""
        
        # Create detector
        detector_config = {
            "name": "invalid-data-test-detector",
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
        detector_id = detector["id"]
        
        # Test various invalid data scenarios
        invalid_datasets = [
            {
                "name": "empty-data",
                "data": [],
                "feature_names": ["feature1", "feature2"]
            },
            {
                "name": "mismatched-features",
                "data": [[1, 2, 3], [4, 5, 6]],
                "feature_names": ["feature1", "feature2"]  # Missing feature3
            },
            {
                "name": "mixed-types",
                "data": [[1, "invalid"], [3, 4]],
                "feature_names": ["feature1", "feature2"]
            }
        ]
        
        for invalid_dataset in invalid_datasets:
            training_payload = {
                "detector_id": detector_id,
                "dataset": invalid_dataset,
                "job_name": f"invalid-data-test-{invalid_dataset['name']}"
            }
            
            response = await async_client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=api_headers
            )
            
            # Should either reject immediately or fail gracefully
            if response.status_code == 201:
                # If accepted, training should fail
                job = response.json()
                await asyncio.sleep(5)  # Give time to process
                
                response = await async_client.get(
                    f"/api/v1/training/jobs/{job['job_id']}",
                    headers=api_headers
                )
                assert_api_response_valid(response)
                job_status = response.json()
                assert job_status["status"] == "failed"
            else:
                # Should be a 4xx error
                assert 400 <= response.status_code < 500
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)