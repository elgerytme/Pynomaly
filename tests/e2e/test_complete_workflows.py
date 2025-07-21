"""
End-to-End Tests for Complete anomaly_detection Workflows

These tests verify complete user journeys from data input to anomaly detection results,
integrating all system components including API, data processing, and ML models.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

pytestmark = pytest.mark.e2e


class TestCompleteAnomalyDetectionWorkflow:
    """Test complete anomaly detection workflows"""

    def test_api_to_results_workflow(self, api_client, sample_api_data):
        """Test complete workflow from API request to results"""
        
        # Step 1: Health check
        response = api_client.get("/health")
        assert response.status_code == 200
        
        # Step 2: Submit anomaly detection request
        response = api_client.post("/api/v1/detect", json=sample_api_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "predictions" in result
        assert "scores" in result
        assert "metadata" in result
        
        # Step 3: Verify predictions format
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == len(sample_api_data["data"])
        assert all(p in [0, 1] for p in predictions)
        
        # Step 4: Verify scores
        scores = result["scores"]
        assert isinstance(scores, list)
        assert len(scores) == len(sample_api_data["data"])
        assert all(isinstance(s, (int, float)) for s in scores)
        
        # Step 5: Verify metadata
        metadata = result["metadata"]
        assert "algorithm" in metadata
        assert "contamination" in metadata
        assert "processing_time" in metadata
        assert metadata["algorithm"] == sample_api_data["algorithm"]

    def test_batch_processing_workflow(self, api_client):
        """Test batch processing of multiple datasets"""
        
        # Prepare batch data
        batch_data = {
            "datasets": [
                {
                    "id": "dataset_1",
                    "data": np.random.randn(100, 5).tolist(),
                    "contamination": 0.1
                },
                {
                    "id": "dataset_2", 
                    "data": np.random.randn(150, 8).tolist(),
                    "contamination": 0.05
                }
            ],
            "algorithm": "isolation_forest"
        }
        
        # Submit batch request
        response = api_client.post("/api/v1/batch-detect", json=batch_data)
        assert response.status_code == 200
        
        results = response.json()
        assert "results" in results
        assert len(results["results"]) == 2
        
        # Verify each result
        for result in results["results"]:
            assert "id" in result
            assert "predictions" in result
            assert "scores" in result
            assert result["id"] in ["dataset_1", "dataset_2"]

    def test_streaming_workflow(self, api_client):
        """Test streaming data processing workflow"""
        
        # Start streaming session
        session_data = {"algorithm": "isolation_forest", "contamination": 0.1}
        response = api_client.post("/api/v1/stream/start", json=session_data)
        assert response.status_code == 200
        
        session_id = response.json()["session_id"]
        
        # Stream data points
        for i in range(10):
            data_point = {"data": np.random.randn(5).tolist()}
            response = api_client.post(f"/api/v1/stream/{session_id}/data", json=data_point)
            assert response.status_code == 200
            
            result = response.json()
            assert "prediction" in result
            assert "score" in result
        
        # End streaming session
        response = api_client.post(f"/api/v1/stream/{session_id}/stop")
        assert response.status_code == 200

    def test_model_lifecycle_workflow(self, api_client, medium_dataset):
        """Test complete model lifecycle: train -> predict -> update"""
        
        # Step 1: Train new model
        training_data = {
            "data": medium_dataset.tolist(),
            "algorithm": "isolation_forest",
            "parameters": {
                "n_estimators": 50,
                "contamination": 0.1,
                "random_state": 42
            }
        }
        
        response = api_client.post("/api/v1/models", json=training_data)
        assert response.status_code == 201
        
        model_info = response.json()
        model_id = model_info["model_id"]
        
        # Step 2: Use model for prediction
        prediction_data = {
            "model_id": model_id,
            "data": np.random.randn(50, medium_dataset.shape[1]).tolist()
        }
        
        response = api_client.post("/api/v1/predict", json=prediction_data)
        assert response.status_code == 200
        
        predictions = response.json()
        assert "predictions" in predictions
        assert len(predictions["predictions"]) == 50
        
        # Step 3: Get model info
        response = api_client.get(f"/api/v1/models/{model_id}")
        assert response.status_code == 200
        
        # Step 4: Update model with new data
        update_data = {
            "data": np.random.randn(100, medium_dataset.shape[1]).tolist(),
            "retrain": True
        }
        
        response = api_client.put(f"/api/v1/models/{model_id}", json=update_data)
        assert response.status_code == 200
        
        # Step 5: Delete model
        response = api_client.delete(f"/api/v1/models/{model_id}")
        assert response.status_code == 204


class TestDataPipelineWorkflows:
    """Test complete data processing pipeline workflows"""

    def test_csv_upload_and_detection_workflow(self, api_client, temp_directory):
        """Test workflow: CSV upload -> processing -> anomaly detection"""
        
        # Step 1: Create test CSV
        data = np.random.randn(200, 6)
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(6)])
        csv_path = temp_directory / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Step 2: Upload CSV
        with open(csv_path, 'rb') as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            response = api_client.post("/api/v1/upload", files=files)
        
        assert response.status_code == 200
        upload_result = response.json()
        dataset_id = upload_result["dataset_id"]
        
        # Step 3: Configure detection parameters
        config = {
            "dataset_id": dataset_id,
            "algorithm": "local_outlier_factor",
            "contamination": 0.05,
            "features": [f"feature_{i}" for i in range(6)]
        }
        
        response = api_client.post("/api/v1/detect-dataset", json=config)
        assert response.status_code == 200
        
        # Step 4: Get results
        job_id = response.json()["job_id"]
        
        # Poll for completion
        for _ in range(30):  # Max 30 seconds
            response = api_client.get(f"/api/v1/jobs/{job_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(1)
        
        assert response.json()["status"] == "completed"
        
        # Step 5: Download results
        response = api_client.get(f"/api/v1/jobs/{job_id}/results")
        assert response.status_code == 200
        
        results = response.json()
        assert "predictions" in results
        assert len(results["predictions"]) == 200

    def test_data_quality_and_preprocessing_workflow(self, api_client):
        """Test data quality assessment and preprocessing workflow"""
        
        # Step 1: Submit data with quality issues
        problematic_data = np.random.randn(100, 5)
        # Add missing values
        problematic_data[10:20, 2] = np.nan
        # Add extreme outliers
        problematic_data[50:55] *= 100
        
        data_request = {
            "data": problematic_data.tolist(),
            "enable_preprocessing": True,
            "quality_check": True
        }
        
        response = api_client.post("/api/v1/analyze-quality", json=data_request)
        assert response.status_code == 200
        
        quality_report = response.json()
        assert "quality_score" in quality_report
        assert "issues" in quality_report
        assert "preprocessing_recommendations" in quality_report
        
        # Step 2: Apply preprocessing
        preprocessing_config = {
            "data": problematic_data.tolist(),
            "handle_missing": "median",
            "handle_outliers": "clip",
            "normalize": "standard"
        }
        
        response = api_client.post("/api/v1/preprocess", json=preprocessing_config)
        assert response.status_code == 200
        
        processed_data = response.json()
        assert "processed_data" in processed_data
        assert "preprocessing_summary" in processed_data
        
        # Step 3: Run anomaly detection on processed data
        detection_request = {
            "data": processed_data["processed_data"],
            "algorithm": "isolation_forest",
            "contamination": 0.1
        }
        
        response = api_client.post("/api/v1/detect", json=detection_request)
        assert response.status_code == 200


class TestIntegrationWorkflows:
    """Test workflows integrating with external systems"""

    def test_database_integration_workflow(self, api_client, in_memory_db):
        """Test workflow integrating with database"""
        
        # Step 1: Configure database connection
        db_config = {
            "connection_string": "sqlite:///:memory:",
            "table_name": "sensor_data",
            "feature_columns": ["temp", "pressure", "humidity"],
            "timestamp_column": "timestamp"
        }
        
        response = api_client.post("/api/v1/datasources", json=db_config)
        assert response.status_code == 200
        
        datasource_id = response.json()["datasource_id"]
        
        # Step 2: Test connection
        response = api_client.get(f"/api/v1/datasources/{datasource_id}/test")
        assert response.status_code == 200
        
        # Step 3: Schedule periodic anomaly detection
        schedule_config = {
            "datasource_id": datasource_id,
            "algorithm": "isolation_forest",
            "schedule": "0 */6 * * *",  # Every 6 hours
            "contamination": 0.05
        }
        
        response = api_client.post("/api/v1/schedules", json=schedule_config)
        assert response.status_code == 200
        
        schedule_id = response.json()["schedule_id"]
        
        # Step 4: Manually trigger detection
        response = api_client.post(f"/api/v1/schedules/{schedule_id}/trigger")
        assert response.status_code == 200

    def test_alerting_workflow(self, api_client):
        """Test anomaly alerting workflow"""
        
        # Step 1: Configure alerting rules
        alert_config = {
            "name": "High Anomaly Rate Alert",
            "conditions": {
                "anomaly_rate_threshold": 0.2,
                "time_window": "5m"
            },
            "channels": [
                {"type": "email", "config": {"to": "admin@example.com"}},
                {"type": "webhook", "config": {"url": "https://hooks.slack.com/..."}}
            ]
        }
        
        response = api_client.post("/api/v1/alerts", json=alert_config)
        assert response.status_code == 200
        
        alert_id = response.json()["alert_id"]
        
        # Step 2: Generate data that should trigger alert
        high_anomaly_data = np.random.randn(100, 5)
        # Make 30% of data anomalous
        high_anomaly_data[:30] += 5
        
        detection_request = {
            "data": high_anomaly_data.tolist(),
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "check_alerts": True
        }
        
        response = api_client.post("/api/v1/detect", json=detection_request)
        assert response.status_code == 200
        
        # Step 3: Check if alert was triggered
        response = api_client.get(f"/api/v1/alerts/{alert_id}/history")
        assert response.status_code == 200
        
        alert_history = response.json()
        # Should have at least one triggered alert
        assert len(alert_history["triggered_alerts"]) > 0


class TestPerformanceWorkflows:
    """Test performance-critical workflows"""

    @pytest.mark.slow
    def test_large_dataset_processing_workflow(self, api_client, large_dataset):
        """Test processing workflow with large dataset"""
        
        start_time = time.time()
        
        # Submit large dataset for processing
        request_data = {
            "data": large_dataset.tolist(),
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "async": True  # Use async processing for large datasets
        }
        
        response = api_client.post("/api/v1/detect", json=request_data)
        assert response.status_code == 202  # Accepted for async processing
        
        job_id = response.json()["job_id"]
        
        # Poll for completion
        completed = False
        for _ in range(120):  # Max 2 minutes
            response = api_client.get(f"/api/v1/jobs/{job_id}")
            status = response.json()["status"]
            
            if status == "completed":
                completed = True
                break
            elif status == "failed":
                pytest.fail("Large dataset processing failed")
            
            time.sleep(1)
        
        assert completed, "Large dataset processing did not complete in time"
        
        # Verify processing time is reasonable
        processing_time = time.time() - start_time
        assert processing_time < 120, f"Processing took too long: {processing_time:.2f}s"
        
        # Get and verify results
        response = api_client.get(f"/api/v1/jobs/{job_id}/results")
        assert response.status_code == 200
        
        results = response.json()
        assert len(results["predictions"]) == len(large_dataset)

    def test_concurrent_requests_workflow(self, api_client, small_dataset):
        """Test handling multiple concurrent requests"""
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def make_request(thread_id):
            try:
                request_data = {
                    "data": (small_dataset + thread_id * 0.1).tolist(),
                    "algorithm": "isolation_forest",
                    "contamination": 0.1
                }
                
                response = api_client.post("/api/v1/detect", json=request_data)
                results_queue.put((thread_id, response.status_code, response.json()))
            except Exception as e:
                results_queue.put((thread_id, 500, {"error": str(e)}))
        
        # Launch 10 concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests succeeded
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 10
        for thread_id, status_code, response_data in results:
            assert status_code == 200, f"Thread {thread_id} failed with status {status_code}"
            assert "predictions" in response_data


class TestErrorHandlingWorkflows:
    """Test error handling in complete workflows"""

    def test_malformed_data_workflow(self, api_client, security_test_data):
        """Test workflow with malformed/malicious data"""
        
        # Test various malformed inputs
        malformed_inputs = [
            {"data": "not_a_list"},
            {"data": [[1, 2, "not_a_number"]]},
            {"data": [[]]},  # Empty inner list
            {"data": []},    # Empty data
            {"data": None},  # Null data
        ]
        
        for malformed_input in malformed_inputs:
            response = api_client.post("/api/v1/detect", json=malformed_input)
            assert response.status_code in [400, 422], f"Should reject malformed input: {malformed_input}"
            
            error_response = response.json()
            assert "error" in error_response or "detail" in error_response

    def test_resource_exhaustion_workflow(self, api_client):
        """Test workflow behavior under resource constraints"""
        
        # Test with extremely large data request
        oversized_data = {
            "data": [[1.0] * 1000] * 10000,  # 10M elements
            "algorithm": "isolation_forest"
        }
        
        response = api_client.post("/api/v1/detect", json=oversized_data)
        # Should either reject (413) or handle gracefully
        assert response.status_code in [413, 400, 422]

    def test_network_failure_recovery_workflow(self, api_client):
        """Test workflow recovery from network failures"""
        
        # This would typically test with actual network issues
        # For now, test timeout handling
        
        request_data = {
            "data": np.random.randn(100, 5).tolist(),
            "algorithm": "isolation_forest",
            "timeout": 0.001  # Very short timeout
        }
        
        response = api_client.post("/api/v1/detect", json=request_data)
        
        # Should handle timeout gracefully
        if response.status_code != 200:
            assert response.status_code in [408, 504], "Should return timeout error"


# Test workflow execution order
class TestWorkflowOrdering:
    """Test that workflows execute in correct order"""

    def test_dependency_order_workflow(self, api_client):
        """Test that dependent operations execute in correct order"""
        
        # Step 1: Create model (prerequisite)
        model_data = {
            "data": np.random.randn(200, 5).tolist(),
            "algorithm": "isolation_forest"
        }
        response = api_client.post("/api/v1/models", json=model_data)
        assert response.status_code == 201
        model_id = response.json()["model_id"]
        
        # Step 2: Try to use model before it's ready (should handle gracefully)
        prediction_data = {"model_id": model_id, "data": [[1, 2, 3, 4, 5]]}
        response = api_client.post("/api/v1/predict", json=prediction_data)
        
        # Should either work or return appropriate status
        assert response.status_code in [200, 202, 409]
        
        # Step 3: Wait for model to be ready if needed
        if response.status_code != 200:
            # Poll model status
            for _ in range(30):
                response = api_client.get(f"/api/v1/models/{model_id}")
                if response.json().get("status") == "ready":
                    break
                time.sleep(1)
        
        # Step 4: Now prediction should work
        response = api_client.post("/api/v1/predict", json=prediction_data)
        assert response.status_code == 200