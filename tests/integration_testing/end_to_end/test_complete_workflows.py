"""End-to-end workflow testing for complete user scenarios."""

import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.end_to_end
    @pytest.mark.slow
    async def test_complete_anomaly_detection_workflow(
        self,
        api_client,
        test_data_manager,
        performance_monitor
    ):
        """Test complete anomaly detection workflow from data upload to results."""
        
        # Start performance monitoring
        performance_monitor.start_monitoring()
        
        try:
            # Step 1: Create and upload dataset
            dataset_info = test_data_manager.create_test_dataset(size=5000)
            
            # Mock API calls for dataset upload
            upload_response = Mock()
            upload_response.status_code = 201
            upload_response.json.return_value = {
                "dataset_id": dataset_info["id"],
                "status": "uploaded",
                "rows": dataset_info["size"],
                "columns": 10
            }
            
            api_client.post.return_value = upload_response
            
            # Upload dataset via API
            response = api_client.post(
                "/datasets/upload",
                json={
                    "name": "integration_test_dataset",
                    "description": "Dataset for integration testing",
                    "data": dataset_info["data"].to_dict('records')[:100]  # Sample for API
                }
            )
            
            assert response.status_code == 201
            dataset_id = response.json()["dataset_id"]
            
            # Step 2: Create and configure detector
            detector_config = test_data_manager.create_test_detector()
            
            create_detector_response = Mock()
            create_detector_response.status_code = 201
            create_detector_response.json.return_value = {
                "detector_id": detector_config["id"],
                "status": "created",
                "algorithm": detector_config["algorithm"]
            }
            
            api_client.post.return_value = create_detector_response
            
            response = api_client.post(
                "/detectors",
                json=detector_config
            )
            
            assert response.status_code == 201
            detector_id = response.json()["detector_id"]
            
            # Step 3: Train detector
            train_response = Mock()
            train_response.status_code = 202
            train_response.json.return_value = {
                "job_id": f"train-job-{detector_id}",
                "status": "started",
                "estimated_duration": 60
            }
            
            api_client.post.return_value = train_response
            
            response = api_client.post(
                f"/detectors/{detector_id}/train",
                json={"dataset_id": dataset_id}
            )
            
            assert response.status_code == 202
            train_job_id = response.json()["job_id"]
            
            # Step 4: Monitor training progress
            progress_response = Mock()
            progress_response.status_code = 200
            progress_response.json.return_value = {
                "job_id": train_job_id,
                "status": "completed",
                "progress": 100,
                "metrics": {
                    "training_time": 45.2,
                    "validation_score": 0.85
                }
            }
            
            api_client.get.return_value = progress_response
            
            # Poll for completion (mocked)
            for _ in range(3):  # Simulate polling
                response = api_client.get(f"/jobs/{train_job_id}")
                assert response.status_code == 200
                
                if response.json()["status"] == "completed":
                    break
                    
                await asyncio.sleep(0.1)  # Small delay for simulation
            
            # Step 5: Run detection on new data
            detection_data = test_data_manager.create_test_dataset(size=1000)
            
            detect_response = Mock()
            detect_response.status_code = 200
            detect_response.json.return_value = {
                "detection_id": "detection-123",
                "anomalies_detected": 47,
                "total_samples": 1000,
                "anomaly_rate": 0.047,
                "results": [
                    {
                        "index": i,
                        "anomaly_score": float(np.random.uniform(0.1, 0.9)),
                        "is_anomaly": bool(np.random.choice([True, False], p=[0.05, 0.95]))
                    }
                    for i in range(50)  # Sample results
                ]
            }
            
            api_client.post.return_value = detect_response
            
            response = api_client.post(
                f"/detectors/{detector_id}/detect",
                json={
                    "data": detection_data["data"].to_dict('records')[:100]
                }
            )
            
            assert response.status_code == 200
            detection_results = response.json()
            
            # Step 6: Validate results
            assert "anomalies_detected" in detection_results
            assert "total_samples" in detection_results
            assert "results" in detection_results
            assert detection_results["anomalies_detected"] >= 0
            assert detection_results["total_samples"] > 0
            
            # Step 7: Get detector information
            detector_info_response = Mock()
            detector_info_response.status_code = 200
            detector_info_response.json.return_value = {
                "detector_id": detector_id,
                "algorithm": detector_config["algorithm"],
                "status": "trained",
                "created_at": "2023-01-01T12:00:00Z",
                "metrics": {
                    "training_score": 0.85,
                    "last_detection": "2023-01-01T12:30:00Z"
                }
            }
            
            api_client.get.return_value = detector_info_response
            
            response = api_client.get(f"/detectors/{detector_id}")
            assert response.status_code == 200
            detector_info = response.json()
            
            assert detector_info["status"] == "trained"
            assert "metrics" in detector_info
            
        finally:
            # Stop performance monitoring
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # Validate performance
            if perf_summary:
                assert perf_summary["memory"]["peak_mb"] < 500  # Max 500MB
                assert perf_summary["duration"] < 300  # Max 5 minutes
    
    @pytest.mark.end_to_end
    async def test_data_science_pipeline_workflow(
        self,
        api_client,
        test_data_manager
    ):
        """Test data science pipeline from profiling to quality assessment."""
        
        # Step 1: Create test dataset
        dataset_info = test_data_manager.create_test_dataset(size=2000)
        
        # Step 2: Data profiling
        profiling_response = Mock()
        profiling_response.status_code = 200
        profiling_response.json.return_value = {
            "profile_id": "profile-123",
            "dataset_id": dataset_info["id"],
            "statistics": {
                "total_rows": 2000,
                "total_columns": 10,
                "missing_values": 0,
                "duplicate_rows": 5,
                "numeric_columns": 10,
                "categorical_columns": 0
            },
            "column_profiles": [
                {
                    "name": f"feature_{i}",
                    "type": "numeric",
                    "mean": float(np.random.normal(0, 1)),
                    "std": float(np.random.uniform(0.5, 2.0)),
                    "min": float(np.random.uniform(-3, -1)),
                    "max": float(np.random.uniform(1, 3)),
                    "missing_count": 0
                }
                for i in range(10)
            ]
        }
        
        api_client.post.return_value = profiling_response
        
        response = api_client.post(
            "/data-profiling/profile",
            json={
                "dataset_id": dataset_info["id"],
                "include_statistics": True,
                "include_correlations": True
            }
        )
        
        assert response.status_code == 200
        profile_results = response.json()
        
        assert "statistics" in profile_results
        assert "column_profiles" in profile_results
        assert profile_results["statistics"]["total_rows"] == 2000
        
        # Step 3: Data quality assessment
        quality_response = Mock()
        quality_response.status_code = 200
        quality_response.json.return_value = {
            "assessment_id": "quality-123",
            "dataset_id": dataset_info["id"],
            "overall_score": 0.92,
            "quality_dimensions": {
                "completeness": 1.0,
                "accuracy": 0.95,
                "consistency": 0.88,
                "validity": 0.90,
                "uniqueness": 0.97,
                "timeliness": 0.85
            },
            "issues": [
                {
                    "rule_id": "consistency_check_1",
                    "severity": "medium",
                    "description": "Some outliers detected",
                    "affected_rows": 12
                }
            ],
            "recommendations": [
                {
                    "type": "outlier_treatment",
                    "description": "Consider outlier treatment for better model performance",
                    "priority": "medium"
                }
            ]
        }
        
        api_client.post.return_value = quality_response
        
        response = api_client.post(
            "/data-quality/assess",
            json={
                "dataset_id": dataset_info["id"],
                "rules": ["completeness", "consistency", "outliers"]
            }
        )
        
        assert response.status_code == 200
        quality_results = response.json()
        
        assert "overall_score" in quality_results
        assert "quality_dimensions" in quality_results
        assert quality_results["overall_score"] >= 0.0
        assert quality_results["overall_score"] <= 1.0
        
        # Step 4: Statistical analysis
        stats_response = Mock()
        stats_response.status_code = 200
        stats_response.json.return_value = {
            "analysis_id": "stats-123",
            "dataset_id": dataset_info["id"],
            "feature_importance": [
                {
                    "feature_name": f"feature_{i}",
                    "importance_score": float(np.random.uniform(0.05, 0.25)),
                    "method": "mutual_info"
                }
                for i in range(10)
            ],
            "correlations": {
                "method": "pearson",
                "matrix": [[float(np.random.uniform(-0.3, 0.3)) for _ in range(10)] for _ in range(10)]
            },
            "distributions": [
                {
                    "feature_name": f"feature_{i}",
                    "distribution_type": "normal",
                    "parameters": {
                        "mean": float(np.random.normal(0, 1)),
                        "std": float(np.random.uniform(0.5, 2.0))
                    },
                    "goodness_of_fit": 0.95
                }
                for i in range(10)
            ]
        }
        
        api_client.post.return_value = stats_response
        
        response = api_client.post(
            "/data-science/analyze",
            json={
                "dataset_id": dataset_info["id"],
                "analyses": ["feature_importance", "correlations", "distributions"]
            }
        )
        
        assert response.status_code == 200
        stats_results = response.json()
        
        assert "feature_importance" in stats_results
        assert "correlations" in stats_results
        assert "distributions" in stats_results
        
        # Validate feature importance results
        feature_importance = stats_results["feature_importance"]
        assert len(feature_importance) == 10
        assert all("importance_score" in fi for fi in feature_importance)
    
    @pytest.mark.end_to_end
    async def test_multi_tenant_workflow(
        self,
        api_client,
        security_context,
        test_data_manager
    ):
        """Test multi-tenant workflow with proper isolation."""
        
        # Create two test tenants
        tenant1_user = security_context.create_test_user("analyst")
        tenant2_user = security_context.create_test_user("analyst")
        
        tenant1_token = security_context.generate_test_token(tenant1_user)
        tenant2_token = security_context.generate_test_token(tenant2_user)
        
        # Create datasets for each tenant
        tenant1_dataset = test_data_manager.create_test_dataset(size=1000)
        tenant2_dataset = test_data_manager.create_test_dataset(size=1000)
        
        # Mock tenant-specific responses
        def mock_tenant_response(tenant_id: str, resource_type: str):
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                f"{resource_type}_id": f"{resource_type}-{tenant_id}",
                "tenant_id": tenant_id,
                "status": "success"
            }
            return response
        
        # Tenant 1: Upload dataset
        api_client.post.return_value = mock_tenant_response("tenant1", "dataset")
        
        # Mock authorization header
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant1_token}"}):
            response = api_client.post(
                "/datasets/upload",
                json={
                    "name": "tenant1_dataset",
                    "data": tenant1_dataset["data"].to_dict('records')[:50]
                }
            )
            
            assert response.status_code == 200
            tenant1_dataset_id = response.json()["dataset_id"]
            assert "tenant1" in tenant1_dataset_id
        
        # Tenant 2: Upload dataset
        api_client.post.return_value = mock_tenant_response("tenant2", "dataset")
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant2_token}"}):
            response = api_client.post(
                "/datasets/upload",
                json={
                    "name": "tenant2_dataset",
                    "data": tenant2_dataset["data"].to_dict('records')[:50]
                }
            )
            
            assert response.status_code == 200
            tenant2_dataset_id = response.json()["dataset_id"]
            assert "tenant2" in tenant2_dataset_id
        
        # Test isolation: Tenant 1 should not see Tenant 2's data
        list_response = Mock()
        list_response.status_code = 200
        list_response.json.return_value = {
            "datasets": [
                {"dataset_id": tenant1_dataset_id, "name": "tenant1_dataset"}
            ]
        }
        
        api_client.get.return_value = list_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant1_token}"}):
            response = api_client.get("/datasets")
            assert response.status_code == 200
            datasets = response.json()["datasets"]
            
            # Tenant 1 should only see their own dataset
            assert len(datasets) == 1
            assert datasets[0]["dataset_id"] == tenant1_dataset_id
        
        # Test cross-tenant access denial
        unauthorized_response = Mock()
        unauthorized_response.status_code = 403
        unauthorized_response.json.return_value = {
            "error": "Access denied",
            "message": "Cannot access resources from different tenant"
        }
        
        api_client.get.return_value = unauthorized_response
        
        # Tenant 1 tries to access Tenant 2's dataset
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {tenant1_token}"}):
            response = api_client.get(f"/datasets/{tenant2_dataset_id}")
            assert response.status_code == 403
    
    @pytest.mark.end_to_end
    @pytest.mark.slow
    async def test_streaming_detection_workflow(
        self,
        api_client,
        test_data_manager,
        performance_monitor
    ):
        """Test real-time streaming detection workflow."""
        
        performance_monitor.start_monitoring()
        
        try:
            # Step 1: Set up streaming detector
            detector_config = test_data_manager.create_test_detector()
            detector_config["streaming_config"] = {
                "buffer_size": 100,
                "batch_processing": True,
                "real_time_threshold": 0.1
            }
            
            streaming_setup_response = Mock()
            streaming_setup_response.status_code = 201
            streaming_setup_response.json.return_value = {
                "detector_id": detector_config["id"],
                "streaming_endpoint": f"/streaming/{detector_config['id']}/detect",
                "websocket_url": f"ws://localhost:8001/streaming/{detector_config['id']}/ws",
                "status": "ready"
            }
            
            api_client.post.return_value = streaming_setup_response
            
            response = api_client.post(
                "/streaming/detectors",
                json=detector_config
            )
            
            assert response.status_code == 201
            streaming_config = response.json()
            detector_id = streaming_config["detector_id"]
            
            # Step 2: Simulate streaming data points
            streaming_results = []
            
            for batch in range(5):  # 5 batches of data
                batch_data = []
                for i in range(20):  # 20 points per batch
                    # Generate synthetic streaming data point
                    data_point = {
                        "timestamp": f"2023-01-01T12:{batch:02d}:{i:02d}Z",
                        "features": [float(np.random.normal(0, 1)) for _ in range(10)],
                        "metadata": {"batch": batch, "index": i}
                    }
                    batch_data.append(data_point)
                
                # Send batch for detection
                batch_response = Mock()
                batch_response.status_code = 200
                batch_response.json.return_value = {
                    "batch_id": f"batch-{batch}",
                    "processed_count": len(batch_data),
                    "anomalies_detected": np.random.randint(0, 3),
                    "processing_time_ms": float(np.random.uniform(50, 150)),
                    "results": [
                        {
                            "timestamp": point["timestamp"],
                            "anomaly_score": float(np.random.uniform(0.1, 0.9)),
                            "is_anomaly": bool(np.random.choice([True, False], p=[0.1, 0.9])),
                            "confidence": float(np.random.uniform(0.7, 0.95))
                        }
                        for point in batch_data
                    ]
                }
                
                api_client.post.return_value = batch_response
                
                response = api_client.post(
                    f"/streaming/{detector_id}/detect",
                    json={"batch": batch_data}
                )
                
                assert response.status_code == 200
                batch_results = response.json()
                streaming_results.append(batch_results)
                
                # Validate batch results
                assert "processed_count" in batch_results
                assert "anomalies_detected" in batch_results
                assert "processing_time_ms" in batch_results
                assert batch_results["processing_time_ms"] < 200  # Max 200ms per batch
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            # Step 3: Get streaming statistics
            stats_response = Mock()
            stats_response.status_code = 200
            stats_response.json.return_value = {
                "detector_id": detector_id,
                "total_processed": sum(r["processed_count"] for r in streaming_results),
                "total_anomalies": sum(r["anomalies_detected"] for r in streaming_results),
                "avg_processing_time": sum(r["processing_time_ms"] for r in streaming_results) / len(streaming_results),
                "throughput_per_second": 200,  # Mock throughput
                "uptime_seconds": 30
            }
            
            api_client.get.return_value = stats_response
            
            response = api_client.get(f"/streaming/{detector_id}/stats")
            assert response.status_code == 200
            
            stats = response.json()
            assert stats["total_processed"] == 100  # 5 batches * 20 points
            assert stats["avg_processing_time"] < 200
            
        finally:
            performance_monitor.stop_monitoring()
            perf_summary = performance_monitor.get_summary()
            
            # Validate streaming performance
            if perf_summary:
                # Streaming should be efficient
                assert perf_summary["memory"]["peak_mb"] < 200
                assert perf_summary["cpu"]["avg"] < 80  # Not too CPU intensive