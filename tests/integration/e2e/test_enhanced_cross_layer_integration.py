"""
Enhanced Cross-Layer Integration Testing
Tests integration across all architectural layers with production-like scenarios.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List, Optional
from unittest.mock import patch, AsyncMock
import pandas as pd
import numpy as np

from .conftest import (
    assert_detection_quality,
    assert_performance_within_limits,
    assert_api_response_valid
)


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.integration
class TestCrossArchitecturalLayerIntegration:
    """Test integration across all architectural layers."""

    async def test_complete_data_flow_integration(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor,
        test_workspace
    ):
        """Test complete data flow from API through all layers to persistence."""
        
        performance_monitor.start_timer("cross_layer_flow")
        
        # Step 1: API Layer - Create detector via REST API
        detector_config = {
            "name": "cross-layer-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {
                "n_estimators": 100,
                "contamination": 0.05,
                "random_state": 42
            },
            "contamination_rate": 0.05,
            "description": "Integration test detector for cross-layer validation"
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Verify API response structure
        assert "id" in detector
        assert detector["name"] == detector_config["name"]
        assert detector["algorithm_name"] == detector_config["algorithm_name"]
        assert detector["status"] == "created"
        
        # Step 2: Application Layer - Train detector through use case
        training_data = sample_dataset['features'].iloc[:600]
        dataset_payload = {
            "name": "cross-layer-training-dataset",
            "data": training_data.values.tolist(),
            "feature_names": training_data.columns.tolist(),
            "metadata": {
                "source": "integration_test",
                "created_by": "test_user",
                "description": "Training dataset for cross-layer testing"
            }
        }
        
        training_payload = {
            "detector_id": detector_id,
            "dataset": dataset_payload,
            "job_name": "cross-layer-training-job",
            "configuration": {
                "validation_split": 0.2,
                "early_stopping": True,
                "metrics_collection": True
            }
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        job_id = training_job["job_id"]
        
        # Step 3: Domain Layer - Validate training progress and business rules
        await self._monitor_training_with_domain_validation(
            async_client, job_id, api_headers, performance_monitor
        )
        
        # Step 4: Infrastructure Layer - Verify persistence and caching
        # Check detector persistence
        response = await async_client.get(
            f"/api/v1/detectors/{detector_id}",
            headers=api_headers
        )
        assert_api_response_valid(response)
        persisted_detector = response.json()
        
        assert persisted_detector["id"] == detector_id
        assert persisted_detector["status"] == "trained"
        assert "training_metrics" in persisted_detector
        assert "model_artifact_path" in persisted_detector
        
        # Step 5: Cross-layer Detection Flow
        test_data = sample_dataset['features'].iloc[600:800]
        detection_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "cross-layer-test-dataset",
                "data": test_data.values.tolist(),
                "feature_names": test_data.columns.tolist()
            },
            "options": {
                "return_scores": True,
                "return_explanations": True,
                "enable_caching": True,
                "batch_size": 50
            }
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        detection_result = response.json()
        
        # Step 6: Validate Cross-Layer Consistency
        await self._validate_cross_layer_consistency(
            async_client, detector_id, detection_result, api_headers
        )
        
        # Step 7: Performance and Resource Validation
        performance_monitor.end_timer("cross_layer_flow")
        total_duration = performance_monitor.get_duration("cross_layer_flow")
        
        # Should complete full workflow within reasonable time
        assert_performance_within_limits(total_duration, 180.0)  # 3 minutes max
        
        # Validate detection quality
        assert_detection_quality(detection_result, sample_dataset['anomaly_rate'], tolerance=0.15)
        
        # Cleanup with proper cascade deletion
        response = await async_client.delete(
            f"/api/v1/detectors/{detector_id}?cascade=true",
            headers=api_headers
        )
        assert_api_response_valid(response, 204)

    async def test_event_driven_workflow_integration(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test event-driven workflows across architectural layers."""
        
        performance_monitor.start_timer("event_driven_flow")
        
        # Create detector with event-driven configuration
        detector_config = {
            "name": "event-driven-detector",
            "algorithm_name": "LocalOutlierFactor",
            "hyperparameters": {"n_neighbors": 20, "contamination": 0.05},
            "contamination_rate": 0.05,
            "event_configuration": {
                "enable_webhooks": True,
                "notification_endpoints": ["http://localhost:8080/webhooks/anomaly"],
                "event_types": ["training_completed", "anomaly_detected", "threshold_exceeded"]
            }
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Setup event monitoring
        captured_events = []
        
        async def mock_event_handler(event_type: str, event_data: Dict[str, Any]):
            captured_events.append({
                "type": event_type,
                "data": event_data,
                "timestamp": time.time()
            })
        
        # Train detector and monitor events
        training_data = sample_dataset['features'].iloc[:400]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "event-test-dataset",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "event-driven-training",
            "event_tracking": True
        }
        
        with patch('pynomaly.infrastructure.events.event_publisher.publish_event', 
                  side_effect=mock_event_handler):
            response = await async_client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=api_headers
            )
            assert_api_response_valid(response, 201)
            training_job = response.json()
            
            # Wait for training completion
            await self._wait_for_job_completion(
                async_client, training_job["job_id"], api_headers
            )
        
        # Verify events were triggered
        assert len(captured_events) > 0, "No events were captured during training"
        
        # Check for expected event types
        event_types = [event["type"] for event in captured_events]
        assert "training_started" in event_types or "training_completed" in event_types
        
        performance_monitor.end_timer("event_driven_flow")
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_resilience_and_recovery_integration(
        self,
        async_client,
        sample_dataset,
        api_headers,
        error_simulator
    ):
        """Test system resilience and recovery across layers."""
        
        # Create detector
        detector_config = {
            "name": "resilience-test-detector",
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
        
        # Test 1: Database connection failure simulation
        training_data = sample_dataset['features'].iloc[:200]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "resilience-test-dataset",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "resilience-training"
        }
        
        # Simulate database failure during training
        with patch('pynomaly.infrastructure.persistence.repositories.detector_repository.DetectorRepository.save') as mock_save:
            mock_save.side_effect = ConnectionError("Database connection failed")
            
            response = await async_client.post(
                "/api/v1/training/jobs",
                json=training_payload,
                headers=api_headers
            )
            
            # Should handle gracefully - either reject or show proper error
            if response.status_code == 201:
                job = response.json()
                # Monitor job status - should fail gracefully
                await asyncio.sleep(5)
                
                response = await async_client.get(
                    f"/api/v1/training/jobs/{job['job_id']}",
                    headers=api_headers
                )
                assert_api_response_valid(response)
                job_status = response.json()
                assert job_status["status"] in ["failed", "error"]
                assert "error_message" in job_status
            else:
                # Should be a proper error response
                assert 400 <= response.status_code < 600
        
        # Test 2: Recovery after temporary failure
        # This time without the mock failure
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Should complete successfully
        await self._wait_for_job_completion(
            async_client, training_job["job_id"], api_headers, max_wait_time=45
        )
        
        # Test 3: Partial network failure during detection
        test_data = sample_dataset['features'].iloc[200:250]
        detection_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "resilience-detection-data",
                "data": test_data.values.tolist(),
                "feature_names": test_data.columns.tolist()
            }
        }
        
        # Should work despite previous failures
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        result = response.json()
        
        assert result["n_samples"] == len(test_data)
        assert "anomaly_scores" in result
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_multi_tenant_isolation_integration(
        self,
        async_client,
        sample_dataset,
        test_config
    ):
        """Test multi-tenant isolation across all layers."""
        
        tenant_detectors = {}
        
        # Create detectors for different tenants
        for tenant_id in test_config.TEST_TENANTS:
            tenant_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer test-api-key-{tenant_id}",
                "X-Tenant-ID": tenant_id
            }
            
            detector_config = {
                "name": f"tenant-{tenant_id}-detector",
                "algorithm_name": "IsolationForest",
                "hyperparameters": {"n_estimators": 30},
                "contamination_rate": 0.05,
                "tenant_id": tenant_id
            }
            
            response = await async_client.post(
                "/api/v1/detectors",
                json=detector_config,
                headers=tenant_headers
            )
            assert_api_response_valid(response, 201)
            detector = response.json()
            tenant_detectors[tenant_id] = {
                "detector": detector,
                "headers": tenant_headers
            }
        
        # Train detectors in parallel for different tenants
        training_tasks = []
        for tenant_id, tenant_data in tenant_detectors.items():
            training_data = sample_dataset['features'].iloc[:100]  # Small dataset for speed
            training_payload = {
                "detector_id": tenant_data["detector"]["id"],
                "dataset": {
                    "name": f"tenant-{tenant_id}-dataset",
                    "data": training_data.values.tolist(),
                    "feature_names": training_data.columns.tolist()
                },
                "job_name": f"tenant-{tenant_id}-training"
            }
            
            task = asyncio.create_task(
                self._train_and_wait(
                    async_client, training_payload, tenant_data["headers"]
                )
            )
            training_tasks.append((tenant_id, task))
        
        # Wait for all training to complete
        for tenant_id, task in training_tasks:
            job_result = await task
            assert job_result["status"] == "completed", f"Training failed for tenant {tenant_id}"
        
        # Test tenant isolation - each tenant should only see their own detectors
        for tenant_id, tenant_data in tenant_detectors.items():
            response = await async_client.get(
                "/api/v1/detectors",
                headers=tenant_data["headers"]
            )
            assert_api_response_valid(response)
            detectors = response.json()
            
            # Should only see detectors for this tenant
            tenant_detector_names = [d["name"] for d in detectors.get("items", [])]
            expected_name = f"tenant-{tenant_id}-detector"
            assert expected_name in tenant_detector_names
            
            # Should not see other tenants' detectors
            for other_tenant in test_config.TEST_TENANTS:
                if other_tenant != tenant_id:
                    other_name = f"tenant-{other_tenant}-detector"
                    assert other_name not in tenant_detector_names
        
        # Cleanup all tenant detectors
        for tenant_id, tenant_data in tenant_detectors.items():
            await async_client.delete(
                f"/api/v1/detectors/{tenant_data['detector']['id']}",
                headers=tenant_data["headers"]
            )

    async def test_distributed_processing_integration(
        self,
        async_client,
        sample_dataset,
        api_headers,
        performance_monitor
    ):
        """Test distributed processing across multiple workers/nodes."""
        
        performance_monitor.start_timer("distributed_processing")
        
        # Create detector optimized for distributed processing
        detector_config = {
            "name": "distributed-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {
                "n_estimators": 200,
                "n_jobs": -1,  # Use all available cores
                "contamination": 0.05
            },
            "contamination_rate": 0.05,
            "processing_config": {
                "enable_distributed": True,
                "batch_size": 1000,
                "parallel_workers": 4
            }
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Use larger dataset to test distributed processing
        large_dataset = sample_dataset['features']  # Use full dataset
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "distributed-training-dataset",
                "data": large_dataset.values.tolist(),
                "feature_names": large_dataset.columns.tolist()
            },
            "job_name": "distributed-training-job",
            "processing_options": {
                "distributed": True,
                "chunk_size": 200,
                "parallel_execution": True
            }
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Monitor training with performance tracking
        job_result = await self._wait_for_job_completion(
            async_client, training_job["job_id"], api_headers, max_wait_time=120
        )
        
        assert job_result["status"] == "completed"
        assert "training_metrics" in job_result
        
        # Test distributed inference
        test_data = large_dataset  # Use same data for inference test
        detection_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "distributed-inference-dataset",
                "data": test_data.values.tolist(),
                "feature_names": test_data.columns.tolist()
            },
            "processing_options": {
                "distributed": True,
                "batch_size": 100,
                "parallel_workers": 4
            }
        }
        
        response = await async_client.post(
            "/api/v1/detection/predict",
            json=detection_payload,
            headers=api_headers
        )
        assert_api_response_valid(response)
        detection_result = response.json()
        
        # Verify distributed processing results
        assert detection_result["n_samples"] == len(test_data)
        assert len(detection_result["anomaly_scores"]) == len(test_data)
        assert "execution_time" in detection_result
        
        # Performance should be reasonable even with large dataset
        execution_time = detection_result["execution_time"]
        assert execution_time < 60.0, f"Distributed inference took too long: {execution_time}s"
        
        performance_monitor.end_timer("distributed_processing")
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    # Helper methods
    async def _monitor_training_with_domain_validation(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        performance_monitor
    ):
        """Monitor training progress with domain-level validation."""
        performance_monitor.start_timer("training_validation")
        
        max_wait_time = 90  # Extended wait time for thorough testing
        start_time = time.time()
        previous_status = None
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            current_status = job_status["status"]
            
            # Validate domain business rules
            assert current_status in ["pending", "running", "completed", "failed"], (
                f"Invalid job status: {current_status}"
            )
            
            # Validate status transitions
            if previous_status:
                assert self._is_valid_status_transition(previous_status, current_status), (
                    f"Invalid status transition: {previous_status} -> {current_status}"
                )
            
            if current_status == "completed":
                # Validate completion requirements
                assert "training_metrics" in job_status
                assert "completion_time" in job_status
                assert "model_artifact_path" in job_status
                break
            elif current_status == "failed":
                assert "error_message" in job_status
                pytest.fail(f"Training failed: {job_status['error_message']}")
            
            previous_status = current_status
            await asyncio.sleep(3)
        else:
            pytest.fail("Training did not complete within expected time")
        
        performance_monitor.end_timer("training_validation")

    async def _validate_cross_layer_consistency(
        self,
        client,
        detector_id: str,
        detection_result: Dict[str, Any],
        headers: Dict[str, str]
    ):
        """Validate consistency across architectural layers."""
        
        # 1. API Layer consistency
        response = await client.get(f"/api/v1/detectors/{detector_id}", headers=headers)
        assert_api_response_valid(response)
        detector_info = response.json()
        
        # 2. Application Layer consistency
        assert detector_info["status"] == "trained"
        assert "training_metrics" in detector_info
        
        # 3. Domain Layer consistency
        # Verify detection results follow domain rules
        assert detection_result["n_samples"] > 0
        assert detection_result["n_anomalies"] >= 0
        assert detection_result["n_anomalies"] <= detection_result["n_samples"]
        
        anomaly_rate = detection_result["n_anomalies"] / detection_result["n_samples"]
        expected_contamination = detector_info.get("contamination_rate", 0.05)
        
        # Anomaly rate should be reasonable compared to expected contamination
        assert 0.0 <= anomaly_rate <= min(0.5, expected_contamination * 10), (
            f"Anomaly rate {anomaly_rate} seems unreasonable for contamination {expected_contamination}"
        )
        
        # 4. Infrastructure Layer consistency
        # Verify response format and required fields
        required_fields = ["anomaly_scores", "anomaly_labels", "n_anomalies", "n_samples", "execution_time"]
        for field in required_fields:
            assert field in detection_result, f"Missing required field: {field}"

    def _is_valid_status_transition(self, from_status: str, to_status: str) -> bool:
        """Validate if status transition is allowed by domain rules."""
        valid_transitions = {
            "pending": ["running", "failed"],
            "running": ["completed", "failed"],
            "completed": [],  # Terminal state
            "failed": []  # Terminal state
        }
        
        return to_status in valid_transitions.get(from_status, [])

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ) -> Dict[str, Any]:
        """Wait for job completion and return final status."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] in ["completed", "failed"]:
                return job_status
                
            await asyncio.sleep(2)
        
        pytest.fail(f"Job {job_id} did not complete within {max_wait_time} seconds")

    async def _train_and_wait(
        self,
        client,
        training_payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Train detector and wait for completion."""
        response = await client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        return await self._wait_for_job_completion(
            client, training_job["job_id"], headers, max_wait_time=60
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])