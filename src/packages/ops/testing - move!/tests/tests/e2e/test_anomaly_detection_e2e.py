"""
End-to-end tests for anomaly detection system.

This module tests:
- Complete user workflows
- System integration from UI to database
- Real-world scenarios
- Performance under load
- Error recovery
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities import DetectionResult
from monorepo.domain.value_objects import AnomalyScore


@pytest.mark.e2e
class TestEndToEndAnomalyDetection:
    """End-to-end tests for complete anomaly detection workflows."""

    def test_complete_detection_workflow_via_api(self, client, sample_data):
        """Test complete detection workflow through API."""
        # Step 1: Upload dataset
        dataset_data = {
            "name": "E2E Test Dataset",
            "data": sample_data.to_dict("records"),
            "description": "End-to-end test dataset",
        }

        upload_response = client.post("/api/v1/datasets", json=dataset_data)
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # Step 2: Create detector
        detector_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1, "random_state": 42},
            "metadata": {"test": "e2e"},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Step 3: Run detection
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        # Mock the actual ML computation
        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[
                    AnomalyScore(np.random.random()) for _ in range(len(sample_data))
                ],
                metadata={"e2e_test": True},
            )

            detection_response = client.post("/api/v1/detect", json=detection_data)
            assert detection_response.status_code == 200
            result = detection_response.json()

            # Verify results
            assert "scores" in result
            assert len(result["scores"]) == len(sample_data)
            assert result["metadata"]["e2e_test"] is True

    def test_file_upload_and_detection_workflow(self, client, temp_dir):
        """Test workflow with file upload."""
        # Create test file
        test_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )

        csv_path = Path(temp_dir) / "test_upload.csv"
        test_data.to_csv(csv_path, index=False)

        # Upload file
        with open(csv_path, "rb") as f:
            files = {"file": ("test_upload.csv", f, "text/csv")}
            upload_response = client.post("/api/v1/datasets/upload", files=files)

        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # Create detector and run detection
        detector_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.05},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Run detection
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[AnomalyScore(np.random.random()) for _ in range(100)],
                metadata={"file_upload_test": True},
            )

            detection_response = client.post("/api/v1/detect", json=detection_data)
            assert detection_response.status_code == 200

            result = detection_response.json()
            assert len(result["scores"]) == 100
            assert result["metadata"]["file_upload_test"] is True

    def test_batch_processing_workflow(self, client, sample_data):
        """Test batch processing workflow."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset_data = {
                "name": f"Batch Dataset {i}",
                "data": sample_data.to_dict("records"),
                "description": f"Batch dataset {i}",
            }

            response = client.post("/api/v1/datasets", json=dataset_data)
            assert response.status_code == 201
            datasets.append(response.json()["id"])

        # Create detector
        detector_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Run batch detection
        batch_data = {"dataset_ids": datasets, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            # Mock returns different results for each dataset
            mock_detect.side_effect = [
                DetectionResult(
                    detector_id=detector_id,
                    dataset_id=dataset_id,
                    scores=[
                        AnomalyScore(np.random.random())
                        for _ in range(len(sample_data))
                    ],
                    metadata={"batch_index": i},
                )
                for i, dataset_id in enumerate(datasets)
            ]

            batch_response = client.post("/api/v1/detect/batch", json=batch_data)
            assert batch_response.status_code == 200

            results = batch_response.json()
            assert len(results) == 3

            for i, result in enumerate(results):
                assert result["metadata"]["batch_index"] == i

    def test_model_training_and_inference_workflow(self, client, sample_data):
        """Test model training and inference workflow."""
        # Step 1: Upload training data
        training_data = {
            "name": "Training Dataset",
            "data": sample_data.to_dict("records"),
            "description": "Training dataset for model",
        }

        upload_response = client.post("/api/v1/datasets", json=training_data)
        assert upload_response.status_code == 201
        training_dataset_id = upload_response.json()["id"]

        # Step 2: Create and train model
        model_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1, "random_state": 42},
        }

        model_response = client.post("/api/v1/models", json=model_data)
        assert model_response.status_code == 201
        model_id = model_response.json()["id"]

        # Step 3: Train model
        training_request = {"model_id": model_id, "dataset_id": training_dataset_id}

        with patch(
            "monorepo.application.services.model_service.ModelService.train_model"
        ) as mock_train:
            mock_train.return_value = {
                "model_id": model_id,
                "status": "trained",
                "metrics": {"accuracy": 0.95},
            }

            train_response = client.post("/api/v1/models/train", json=training_request)
            assert train_response.status_code == 200

            training_result = train_response.json()
            assert training_result["status"] == "trained"

        # Step 4: Create inference data
        inference_data = {
            "name": "Inference Dataset",
            "data": sample_data.sample(n=50).to_dict("records"),
            "description": "Inference dataset",
        }

        inference_upload = client.post("/api/v1/datasets", json=inference_data)
        assert inference_upload.status_code == 201
        inference_dataset_id = inference_upload.json()["id"]

        # Step 5: Run inference
        inference_request = {"model_id": model_id, "dataset_id": inference_dataset_id}

        with patch(
            "monorepo.application.services.model_service.ModelService.predict"
        ) as mock_predict:
            mock_predict.return_value = DetectionResult(
                detector_id=model_id,
                dataset_id=inference_dataset_id,
                scores=[AnomalyScore(np.random.random()) for _ in range(50)],
                metadata={"inference": True},
            )

            inference_response = client.post(
                "/api/v1/models/predict", json=inference_request
            )
            assert inference_response.status_code == 200

            inference_result = inference_response.json()
            assert len(inference_result["scores"]) == 50
            assert inference_result["metadata"]["inference"] is True

    @pytest.mark.slow
    def test_performance_under_load(self, client, sample_data):
        """Test system performance under load."""
        # Create multiple concurrent requests
        import threading

        results = []
        errors = []

        def make_request(request_id):
            try:
                # Create dataset
                dataset_data = {
                    "name": f"Load Test Dataset {request_id}",
                    "data": sample_data.to_dict("records"),
                    "description": f"Load test dataset {request_id}",
                }

                dataset_response = client.post("/api/v1/datasets", json=dataset_data)
                dataset_id = dataset_response.json()["id"]

                # Create detector
                detector_data = {
                    "algorithm_name": "IsolationForest",
                    "parameters": {"contamination": 0.1, "random_state": request_id},
                }

                detector_response = client.post("/api/v1/detectors", json=detector_data)
                detector_id = detector_response.json()["id"]

                # Run detection
                detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

                with patch(
                    "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
                ) as mock_detect:
                    mock_detect.return_value = DetectionResult(
                        detector_id=detector_id,
                        dataset_id=dataset_id,
                        scores=[
                            AnomalyScore(np.random.random())
                            for _ in range(len(sample_data))
                        ],
                        metadata={"load_test": True, "request_id": request_id},
                    )

                    detection_response = client.post(
                        "/api/v1/detect", json=detection_data
                    )
                    results.append(
                        {
                            "request_id": request_id,
                            "status_code": detection_response.status_code,
                            "response": detection_response.json(),
                        }
                    )

            except Exception as e:
                errors.append({"request_id": request_id, "error": str(e)})

        # Create and start threads
        threads = []
        num_requests = 10

        start_time = time.time()

        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.time()

        # Verify performance
        total_time = end_time - start_time
        assert total_time < 30.0  # Should complete within 30 seconds
        assert len(results) == num_requests
        assert len(errors) == 0

        # Verify all requests succeeded
        for result in results:
            assert result["status_code"] == 200
            assert result["response"]["metadata"]["load_test"] is True

    def test_error_recovery_workflow(self, client, sample_data):
        """Test error recovery in complete workflow."""
        # Step 1: Try to create dataset with invalid data
        invalid_data = {
            "name": "Invalid Dataset",
            "data": [],  # Empty data
            "description": "Invalid dataset",
        }

        invalid_response = client.post("/api/v1/datasets", json=invalid_data)
        assert invalid_response.status_code == 400

        # Step 2: Create valid dataset
        valid_data = {
            "name": "Valid Dataset",
            "data": sample_data.to_dict("records"),
            "description": "Valid dataset",
        }

        valid_response = client.post("/api/v1/datasets", json=valid_data)
        assert valid_response.status_code == 201
        dataset_id = valid_response.json()["id"]

        # Step 3: Try to create detector with invalid algorithm
        invalid_detector = {"algorithm_name": "NonExistentAlgorithm", "parameters": {}}

        invalid_detector_response = client.post(
            "/api/v1/detectors", json=invalid_detector
        )
        assert invalid_detector_response.status_code == 400

        # Step 4: Create valid detector
        valid_detector = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        valid_detector_response = client.post("/api/v1/detectors", json=valid_detector)
        assert valid_detector_response.status_code == 201
        detector_id = valid_detector_response.json()["id"]

        # Step 5: Run detection successfully
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[
                    AnomalyScore(np.random.random()) for _ in range(len(sample_data))
                ],
                metadata={"error_recovery_test": True},
            )

            detection_response = client.post("/api/v1/detect", json=detection_data)
            assert detection_response.status_code == 200

            result = detection_response.json()
            assert result["metadata"]["error_recovery_test"] is True

    def test_data_persistence_workflow(self, client, sample_data):
        """Test data persistence across workflow."""
        # Step 1: Create dataset
        dataset_data = {
            "name": "Persistence Test Dataset",
            "data": sample_data.to_dict("records"),
            "description": "Dataset for persistence testing",
        }

        create_response = client.post("/api/v1/datasets", json=dataset_data)
        assert create_response.status_code == 201
        dataset_id = create_response.json()["id"]

        # Step 2: Retrieve dataset
        get_response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_response.status_code == 200

        retrieved_dataset = get_response.json()
        assert retrieved_dataset["name"] == "Persistence Test Dataset"
        assert len(retrieved_dataset["data"]) == len(sample_data)

        # Step 3: Update dataset
        update_data = {
            "name": "Updated Persistence Test Dataset",
            "description": "Updated dataset for persistence testing",
        }

        update_response = client.put(f"/api/v1/datasets/{dataset_id}", json=update_data)
        assert update_response.status_code == 200

        # Step 4: Verify update
        updated_get_response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert updated_get_response.status_code == 200

        updated_dataset = updated_get_response.json()
        assert updated_dataset["name"] == "Updated Persistence Test Dataset"
        assert (
            updated_dataset["description"] == "Updated dataset for persistence testing"
        )

        # Step 5: Delete dataset
        delete_response = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert delete_response.status_code == 204

        # Step 6: Verify deletion
        deleted_get_response = client.get(f"/api/v1/datasets/{dataset_id}")
        assert deleted_get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_async_workflow(self, client, sample_data):
        """Test asynchronous workflow."""
        # Step 1: Create dataset
        dataset_data = {
            "name": "Async Test Dataset",
            "data": sample_data.to_dict("records"),
            "description": "Dataset for async testing",
        }

        create_response = client.post("/api/v1/datasets", json=dataset_data)
        assert create_response.status_code == 201
        dataset_id = create_response.json()["id"]

        # Step 2: Create detector
        detector_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Step 3: Start async detection
        async_detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies_async"
        ) as mock_async_detect:
            # Mock async detection
            mock_async_detect.return_value = asyncio.Future()
            mock_async_detect.return_value.set_result(
                DetectionResult(
                    detector_id=detector_id,
                    dataset_id=dataset_id,
                    scores=[
                        AnomalyScore(np.random.random())
                        for _ in range(len(sample_data))
                    ],
                    metadata={"async_test": True},
                )
            )

            async_response = client.post(
                "/api/v1/detect/async", json=async_detection_data
            )
            assert async_response.status_code == 202

            task_id = async_response.json()["task_id"]

            # Step 4: Check task status
            status_response = client.get(f"/api/v1/tasks/{task_id}")
            assert status_response.status_code == 200

            task_status = status_response.json()
            assert task_status["status"] in ["pending", "running", "completed"]

            # Step 5: Get results (mock as completed)
            with patch(
                "monorepo.application.services.task_service.TaskService.get_task_result"
            ) as mock_get_result:
                mock_get_result.return_value = {
                    "task_id": task_id,
                    "status": "completed",
                    "result": {
                        "scores": [0.5] * len(sample_data),
                        "metadata": {"async_test": True},
                    },
                }

                result_response = client.get(f"/api/v1/tasks/{task_id}/result")
                assert result_response.status_code == 200

                result = result_response.json()
                assert result["status"] == "completed"
                assert result["result"]["metadata"]["async_test"] is True


@pytest.mark.e2e
class TestRealWorldScenarios:
    """End-to-end tests for real-world scenarios."""

    def test_fraud_detection_scenario(self, client):
        """Test fraud detection scenario."""
        # Create fraud detection dataset
        fraud_data = pd.DataFrame(
            {
                "transaction_amount": np.random.lognormal(3, 1, 1000),
                "account_age_days": np.random.randint(1, 3650, 1000),
                "num_transactions_today": np.random.poisson(5, 1000),
                "time_since_last_transaction": np.random.exponential(24, 1000),
            }
        )

        # Add some fraudulent transactions
        fraud_indices = np.random.choice(1000, 50, replace=False)
        fraud_data.loc[fraud_indices, "transaction_amount"] *= (
            10  # Unusually high amounts
        )
        fraud_data.loc[fraud_indices, "num_transactions_today"] *= (
            5  # Unusually high frequency
        )

        dataset_data = {
            "name": "Fraud Detection Dataset",
            "data": fraud_data.to_dict("records"),
            "description": "Dataset for fraud detection",
        }

        dataset_response = client.post("/api/v1/datasets", json=dataset_data)
        assert dataset_response.status_code == 201
        dataset_id = dataset_response.json()["id"]

        # Create fraud detection model
        detector_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {
                "contamination": 0.05,  # Expect 5% fraud
                "random_state": 42,
            },
            "metadata": {"use_case": "fraud_detection"},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Run fraud detection
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[AnomalyScore(np.random.random()) for _ in range(1000)],
                metadata={"fraud_detection": True},
            )

            detection_response = client.post("/api/v1/detect", json=detection_data)
            assert detection_response.status_code == 200

            result = detection_response.json()
            assert len(result["scores"]) == 1000
            assert result["metadata"]["fraud_detection"] is True

    def test_network_intrusion_scenario(self, client):
        """Test network intrusion detection scenario."""
        # Create network traffic dataset
        network_data = pd.DataFrame(
            {
                "packet_size": np.random.normal(1500, 300, 5000),
                "connection_duration": np.random.exponential(10, 5000),
                "bytes_sent": np.random.lognormal(8, 2, 5000),
                "bytes_received": np.random.lognormal(8, 2, 5000),
                "num_connections": np.random.poisson(3, 5000),
            }
        )

        # Add some intrusion patterns
        intrusion_indices = np.random.choice(5000, 100, replace=False)
        network_data.loc[intrusion_indices, "packet_size"] *= (
            0.1  # Unusually small packets
        )
        network_data.loc[intrusion_indices, "num_connections"] *= 20  # Port scanning

        dataset_data = {
            "name": "Network Intrusion Dataset",
            "data": network_data.to_dict("records"),
            "description": "Dataset for network intrusion detection",
        }

        dataset_response = client.post("/api/v1/datasets", json=dataset_data)
        assert dataset_response.status_code == 201
        dataset_id = dataset_response.json()["id"]

        # Create intrusion detection model
        detector_data = {
            "algorithm_name": "LocalOutlierFactor",
            "parameters": {
                "n_neighbors": 20,
                "contamination": 0.02,  # Expect 2% intrusions
            },
            "metadata": {"use_case": "network_intrusion"},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Run intrusion detection
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[AnomalyScore(np.random.random()) for _ in range(5000)],
                metadata={"network_intrusion": True},
            )

            detection_response = client.post("/api/v1/detect", json=detection_data)
            assert detection_response.status_code == 200

            result = detection_response.json()
            assert len(result["scores"]) == 5000
            assert result["metadata"]["network_intrusion"] is True

    def test_predictive_maintenance_scenario(self, client):
        """Test predictive maintenance scenario."""
        # Create sensor data
        sensor_data = pd.DataFrame(
            {
                "temperature": np.random.normal(25, 5, 2000),
                "vibration": np.random.normal(0.5, 0.1, 2000),
                "pressure": np.random.normal(1013, 10, 2000),
                "humidity": np.random.normal(45, 10, 2000),
                "rotation_speed": np.random.normal(1800, 50, 2000),
            }
        )

        # Add some failure patterns
        failure_indices = np.random.choice(2000, 40, replace=False)
        sensor_data.loc[failure_indices, "temperature"] += 20  # Overheating
        sensor_data.loc[failure_indices, "vibration"] *= 5  # Excessive vibration

        dataset_data = {
            "name": "Predictive Maintenance Dataset",
            "data": sensor_data.to_dict("records"),
            "description": "Dataset for predictive maintenance",
        }

        dataset_response = client.post("/api/v1/datasets", json=dataset_data)
        assert dataset_response.status_code == 201
        dataset_id = dataset_response.json()["id"]

        # Create maintenance prediction model
        detector_data = {
            "algorithm_name": "OneClassSVM",
            "parameters": {
                "nu": 0.02,  # Expect 2% failures
                "gamma": "scale",
            },
            "metadata": {"use_case": "predictive_maintenance"},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Run maintenance prediction
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[AnomalyScore(np.random.random()) for _ in range(2000)],
                metadata={"predictive_maintenance": True},
            )

            detection_response = client.post("/api/v1/detect", json=detection_data)
            assert detection_response.status_code == 200

            result = detection_response.json()
            assert len(result["scores"]) == 2000
            assert result["metadata"]["predictive_maintenance"] is True

    @pytest.mark.slow
    def test_large_scale_scenario(self, client):
        """Test large-scale data processing scenario."""
        # Create large dataset
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 50000) for i in range(20)}
        )

        dataset_data = {
            "name": "Large Scale Dataset",
            "data": large_data.to_dict("records"),
            "description": "Large dataset for scalability testing",
        }

        dataset_response = client.post("/api/v1/datasets", json=dataset_data)
        assert dataset_response.status_code == 201
        dataset_id = dataset_response.json()["id"]

        # Create detector optimized for large data
        detector_data = {
            "algorithm_name": "IsolationForest",
            "parameters": {
                "contamination": 0.01,
                "n_estimators": 50,  # Fewer estimators for speed
                "max_samples": 1000,  # Subsample for speed
                "random_state": 42,
            },
            "metadata": {"use_case": "large_scale"},
        }

        detector_response = client.post("/api/v1/detectors", json=detector_data)
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Run large-scale detection
        detection_data = {"dataset_id": dataset_id, "detector_id": detector_id}

        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                scores=[AnomalyScore(np.random.random()) for _ in range(50000)],
                metadata={"large_scale": True},
            )

            import time

            start_time = time.time()
            detection_response = client.post("/api/v1/detect", json=detection_data)
            end_time = time.time()

            assert detection_response.status_code == 200

            result = detection_response.json()
            assert len(result["scores"]) == 50000
            assert result["metadata"]["large_scale"] is True

            # Verify performance
            processing_time = end_time - start_time
            assert processing_time < 60.0  # Should complete within 60 seconds
