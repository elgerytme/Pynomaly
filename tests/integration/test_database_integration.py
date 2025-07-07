"""Integration tests for database operations and data persistence."""

import pytest
from httpx import AsyncClient

from tests.integration.conftest import IntegrationTestHelper


class TestDatabaseIntegration:
    """Test database integration and data persistence scenarios."""

    @pytest.mark.asyncio
    async def test_dataset_crud_operations(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test complete CRUD operations for datasets."""

        # Step 1: Create (Upload) dataset
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "crud_test_dataset"
        )

        dataset_id = dataset["id"]
        assert dataset["name"] == "crud_test_dataset"
        assert dataset["status"] == "uploaded"

        # Step 2: Read - Get dataset details
        response = await async_test_client.get(f"/api/datasets/{dataset_id}")
        response.raise_for_status()
        retrieved_dataset = response.json()["data"]

        assert retrieved_dataset["id"] == dataset_id
        assert retrieved_dataset["name"] == "crud_test_dataset"
        assert retrieved_dataset["rows"] == 1000
        assert len(retrieved_dataset["columns"]) == 5

        # Step 3: Read - List datasets
        response = await async_test_client.get("/api/datasets")
        response.raise_for_status()
        datasets_list = response.json()["data"]

        dataset_ids = [d["id"] for d in datasets_list["datasets"]]
        assert dataset_id in dataset_ids

        # Step 4: Update - Modify dataset metadata
        update_data = {
            "name": "updated_crud_dataset",
            "description": "Updated description for testing",
            "tags": ["updated", "crud_test"],
        }

        response = await async_test_client.put(
            f"/api/datasets/{dataset_id}", json=update_data
        )
        response.raise_for_status()
        updated_dataset = response.json()["data"]

        assert updated_dataset["name"] == "updated_crud_dataset"
        assert updated_dataset["description"] == "Updated description for testing"
        assert "updated" in updated_dataset["tags"]

        # Step 5: Read - Verify update persisted
        response = await async_test_client.get(f"/api/datasets/{dataset_id}")
        response.raise_for_status()
        verified_dataset = response.json()["data"]

        assert verified_dataset["name"] == "updated_crud_dataset"
        assert verified_dataset["description"] == "Updated description for testing"

        # Step 6: Dataset statistics and validation
        response = await async_test_client.get(f"/api/datasets/{dataset_id}/stats")
        response.raise_for_status()
        stats = response.json()["data"]

        assert "column_stats" in stats
        assert "data_quality" in stats
        assert "missing_values" in stats

        # Step 7: Dataset validation
        response = await async_test_client.post(f"/api/datasets/{dataset_id}/validate")
        response.raise_for_status()
        validation = response.json()["data"]

        assert validation["is_valid"] is True
        assert "validation_results" in validation

        # Step 8: Delete dataset
        response = await async_test_client.delete(f"/api/datasets/{dataset_id}")
        response.raise_for_status()

        # Step 9: Verify deletion
        response = await async_test_client.get(f"/api/datasets/{dataset_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_detector_persistence_and_versioning(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test detector persistence and model versioning."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "versioning_dataset"
        )

        # Step 1: Create initial detector
        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        detector_id = detector["id"]

        # Step 2: Train detector (creates version 1)
        training_result = await integration_helper.train_detector(detector_id)
        model_v1_id = training_result["model_id"]

        # Step 3: Verify model version 1 persistence
        response = await async_test_client.get(f"/api/models/{model_v1_id}")
        response.raise_for_status()
        model_v1 = response.json()["data"]

        assert model_v1["version"] == "1.0.0"
        assert model_v1["detector_id"] == detector_id
        assert model_v1["status"] == "trained"

        # Step 4: Create detector version 2 with different parameters
        detector_v2_config = {
            "name": "versioned_detector_v2",
            "description": "Version 2 with different parameters",
            "algorithm": "isolation_forest",
            "parameters": {
                "contamination": 0.15,  # Different from v1
                "random_state": 123,  # Different from v1
            },
            "feature_columns": ["feature1", "feature2", "feature3"],
        }

        response = await async_test_client.post(
            f"/api/detectors/create?dataset_id={dataset['id']}", json=detector_v2_config
        )
        response.raise_for_status()
        detector_v2 = response.json()["data"]
        detector_v2_id = detector_v2["id"]
        integration_helper.created_resources["detectors"].append(detector_v2_id)

        # Step 5: Train detector v2
        training_result_v2 = await integration_helper.train_detector(detector_v2_id)
        model_v2_id = training_result_v2["model_id"]

        # Step 6: Verify both models persist independently
        response = await async_test_client.get(f"/api/models/{model_v1_id}")
        response.raise_for_status()
        persisted_v1 = response.json()["data"]

        response = await async_test_client.get(f"/api/models/{model_v2_id}")
        response.raise_for_status()
        persisted_v2 = response.json()["data"]

        assert persisted_v1["id"] != persisted_v2["id"]
        assert persisted_v1["version"] != persisted_v2["version"]
        assert persisted_v1["parameters"] != persisted_v2["parameters"]

        # Step 7: Test model lineage tracking
        response = await async_test_client.get(f"/api/model-lineage/{model_v1_id}")
        response.raise_for_status()
        lineage_v1 = response.json()["data"]

        response = await async_test_client.get(f"/api/model-lineage/{model_v2_id}")
        response.raise_for_status()
        lineage_v2 = response.json()["data"]

        assert lineage_v1["model_id"] == model_v1_id
        assert lineage_v2["model_id"] == model_v2_id
        assert (
            lineage_v1["training_dataset"] == lineage_v2["training_dataset"]
        )  # Same dataset

        # Step 8: Test model comparison persistence
        comparison_config = {
            "baseline_model_id": model_v1_id,
            "candidate_model_id": model_v2_id,
            "test_dataset_id": dataset["id"],
            "comparison_metrics": ["accuracy", "precision", "recall"],
        }

        response = await async_test_client.post(
            "/api/models/compare", json=comparison_config
        )
        response.raise_for_status()
        comparison = response.json()["data"]

        assert "comparison_id" in comparison
        assert "baseline_metrics" in comparison
        assert "candidate_metrics" in comparison

        # Step 9: Retrieve stored comparison
        comparison_id = comparison["comparison_id"]
        response = await async_test_client.get(
            f"/api/models/comparisons/{comparison_id}"
        )
        response.raise_for_status()
        stored_comparison = response.json()["data"]

        assert stored_comparison["baseline_model_id"] == model_v1_id
        assert stored_comparison["candidate_model_id"] == model_v2_id

    @pytest.mark.asyncio
    async def test_experiment_result_persistence(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test experiment results persistence and retrieval."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "experiment_persistence_dataset"
        )

        # Step 1: Create comprehensive experiment
        experiment_config = {
            "name": "persistence_test_experiment",
            "description": "Test experiment persistence and retrieval",
            "dataset_id": dataset["id"],
            "algorithms": [
                {
                    "name": "isolation_forest",
                    "parameters": {"contamination": 0.1, "random_state": 42},
                },
                {"name": "one_class_svm", "parameters": {"nu": 0.1, "kernel": "rbf"}},
            ],
            "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "cross_validation": {"enabled": True, "folds": 3, "stratified": True},
            "feature_selection": {
                "enabled": True,
                "method": "mutual_info",
                "n_features": 3,
            },
        }

        response = await async_test_client.post(
            "/api/experiments/create?created_by=test_user", json=experiment_config
        )
        response.raise_for_status()
        experiment = response.json()["data"]
        experiment_id = experiment["id"]
        integration_helper.created_resources["experiments"].append(experiment_id)

        # Step 2: Run experiment
        response = await async_test_client.post(f"/api/experiments/{experiment_id}/run")
        response.raise_for_status()

        # Step 3: Verify experiment state persistence
        response = await async_test_client.get(f"/api/experiments/{experiment_id}")
        response.raise_for_status()
        stored_experiment = response.json()["data"]

        assert stored_experiment["name"] == "persistence_test_experiment"
        assert stored_experiment["status"] in ["running", "completed"]
        assert len(stored_experiment["algorithm_configurations"]) == 2

        # Step 4: Get and verify experiment results
        response = await async_test_client.get(
            f"/api/experiments/{experiment_id}/results"
        )
        response.raise_for_status()
        results = response.json()["data"]

        assert "experiment_id" in results
        assert "algorithm_results" in results
        assert len(results["algorithm_results"]) == 2

        # Verify detailed results structure
        for algo_result in results["algorithm_results"]:
            assert "algorithm_name" in algo_result
            assert "metrics" in algo_result
            assert "cross_validation_scores" in algo_result
            assert "model_id" in algo_result
            assert "training_time" in algo_result

        # Step 5: Test experiment history and audit trail
        response = await async_test_client.get(
            f"/api/experiments/{experiment_id}/history"
        )
        response.raise_for_status()
        history = response.json()["data"]

        assert "experiment_id" in history
        assert "events" in history
        assert len(history["events"]) > 0

        # Should have creation and execution events
        event_types = [event["event_type"] for event in history["events"]]
        assert "experiment_created" in event_types
        assert (
            "experiment_started" in event_types or "experiment_completed" in event_types
        )

        # Step 6: Test experiment metadata persistence
        metadata_update = {
            "tags": ["persistence_test", "completed"],
            "notes": "Experiment completed successfully",
            "custom_metadata": {"test_type": "integration", "validation_score": 0.95},
        }

        response = await async_test_client.put(
            f"/api/experiments/{experiment_id}/metadata", json=metadata_update
        )
        response.raise_for_status()

        # Step 7: Verify metadata update persistence
        response = await async_test_client.get(f"/api/experiments/{experiment_id}")
        response.raise_for_status()
        updated_experiment = response.json()["data"]

        assert "persistence_test" in updated_experiment["tags"]
        assert updated_experiment["notes"] == "Experiment completed successfully"
        assert updated_experiment["custom_metadata"]["test_type"] == "integration"

        # Step 8: Test experiment search and filtering
        search_params = {
            "tags": ["persistence_test"],
            "status": "completed",
            "created_by": "test_user",
        }

        response = await async_test_client.get("/api/experiments", params=search_params)
        response.raise_for_status()
        search_results = response.json()["data"]

        experiment_ids = [exp["id"] for exp in search_results["experiments"]]
        assert experiment_id in experiment_ids

    @pytest.mark.asyncio
    async def test_streaming_session_state_persistence(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test streaming session state and metrics persistence."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "streaming_persistence_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Step 1: Create streaming session with detailed configuration
        streaming_config = {
            "name": "persistence_test_session",
            "description": "Test session for state persistence",
            "detector_id": detector["id"],
            "data_source": {
                "source_type": "mock",
                "connection_config": {"mock_data_rate": 10, "data_pattern": "mixed"},
            },
            "configuration": {
                "processing_mode": "micro_batch",
                "batch_size": 5,
                "max_throughput": 100,
                "schema_validation": True,
                "enable_checkpointing": True,
                "checkpoint_interval_seconds": 30,
            },
            "data_sink": {
                "sink_type": "mock",
                "connection_config": {"output_format": "json"},
            },
            "tags": ["persistence_test", "integration"],
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user", json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        session_id = session["id"]
        integration_helper.created_resources["sessions"].append(session_id)

        # Step 2: Verify initial state persistence
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/summary"
        )
        response.raise_for_status()
        initial_summary = response.json()["data"]

        assert initial_summary["name"] == "persistence_test_session"
        assert initial_summary["status"] == "pending"
        assert initial_summary["detector_id"] == detector["id"]

        # Step 3: Start session and process data
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/start"
        )
        response.raise_for_status()

        # Process multiple data points to generate metrics
        test_data_points = [
            {
                "timestamp": f"2024-12-25T10:{i:02d}:00Z",
                "feature1": i * 0.1,
                "feature2": i * 0.2,
                "feature3": i * 0.05,
            }
            for i in range(10)
        ]

        for data_point in test_data_points:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session_id}/process",
                json={"data": data_point},
            )
            response.raise_for_status()

        # Step 4: Verify metrics persistence
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/metrics"
        )
        response.raise_for_status()
        metrics = response.json()["data"]

        assert metrics["messages_processed"] == 10
        assert metrics["messages_per_second"] > 0
        assert "measurement_time" in metrics

        # Step 5: Pause session and verify state persistence
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/pause"
        )
        response.raise_for_status()

        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/summary"
        )
        response.raise_for_status()
        paused_summary = response.json()["data"]

        assert paused_summary["status"] == "paused"
        assert paused_summary["messages_processed"] == 10

        # Step 6: Resume session and verify state continuity
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/resume"
        )
        response.raise_for_status()

        # Process more data
        for i in range(10, 15):
            data_point = {
                "timestamp": f"2024-12-25T10:{i:02d}:00Z",
                "feature1": i * 0.1,
                "feature2": i * 0.2,
                "feature3": i * 0.05,
            }
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session_id}/process",
                json={"data": data_point},
            )
            response.raise_for_status()

        # Step 7: Verify cumulative metrics persistence
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/metrics"
        )
        response.raise_for_status()
        final_metrics = response.json()["data"]

        assert final_metrics["messages_processed"] == 15  # 10 + 5

        # Step 8: Create and verify alert persistence
        alert_config = {
            "name": "Persistence Test Alert",
            "metric_name": "messages_per_second",
            "threshold_value": 1.0,
            "comparison_operator": ">",
            "severity": "medium",
        }

        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/alerts?created_by=test_user",
            json=alert_config,
        )
        response.raise_for_status()
        alert = response.json()["data"]

        # Step 9: Stop session and verify final state
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/stop"
        )
        response.raise_for_status()

        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/summary"
        )
        response.raise_for_status()
        final_summary = response.json()["data"]

        assert final_summary["status"] in ["stopped", "stopping"]
        assert final_summary["messages_processed"] == 15
        assert final_summary["uptime_seconds"] > 0

        # Step 10: Verify session history persistence
        response = await async_test_client.get("/api/streaming/sessions")
        response.raise_for_status()
        sessions_list = response.json()["data"]

        session_ids = [s["session_id"] for s in sessions_list]
        assert session_id in session_ids

    @pytest.mark.asyncio
    async def test_transaction_integrity_and_rollback(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test database transaction integrity and rollback scenarios."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "transaction_test_dataset"
        )

        # Step 1: Test successful transaction (detector creation + training)
        detector_config = {
            "name": "transaction_test_detector",
            "description": "Test detector for transaction integrity",
            "algorithm": "isolation_forest",
            "parameters": {"contamination": 0.1, "random_state": 42},
            "feature_columns": ["feature1", "feature2", "feature3"],
        }

        response = await async_test_client.post(
            f"/api/detectors/create?dataset_id={dataset['id']}", json=detector_config
        )
        response.raise_for_status()
        detector = response.json()["data"]
        detector_id = detector["id"]
        integration_helper.created_resources["detectors"].append(detector_id)

        # Verify detector was created
        response = await async_test_client.get(f"/api/detectors/{detector_id}")
        response.raise_for_status()
        created_detector = response.json()["data"]
        assert created_detector["status"] == "created"

        # Step 2: Train detector (should update status atomically)
        response = await async_test_client.post(f"/api/detection/train/{detector_id}")
        response.raise_for_status()
        training_result = response.json()["data"]

        # Verify atomic update
        response = await async_test_client.get(f"/api/detectors/{detector_id}")
        response.raise_for_status()
        trained_detector = response.json()["data"]
        assert trained_detector["status"] == "trained"
        assert trained_detector["model_id"] is not None

        # Step 3: Test concurrent access scenarios
        # Simulate concurrent model updates
        update_requests = []
        for i in range(3):
            update_data = {
                "name": f"concurrent_update_{i}",
                "description": f"Concurrent update test {i}",
            }
            update_requests.append(
                async_test_client.put(f"/api/detectors/{detector_id}", json=update_data)
            )

        # Execute concurrent updates
        import asyncio

        responses = await asyncio.gather(*update_requests, return_exceptions=True)

        # At least one should succeed
        successful_updates = [
            r
            for r in responses
            if not isinstance(r, Exception) and r.status_code == 200
        ]
        assert len(successful_updates) >= 1

        # Step 4: Verify final state consistency
        response = await async_test_client.get(f"/api/detectors/{detector_id}")
        response.raise_for_status()
        final_detector = response.json()["data"]

        # Should have one of the concurrent updates applied
        assert "concurrent_update_" in final_detector["name"]
        assert final_detector["status"] == "trained"  # Status should remain consistent

        # Step 5: Test cascade operations
        # Delete dataset should handle dependent detector properly
        try:
            response = await async_test_client.delete(f"/api/datasets/{dataset['id']}")
            # This might fail due to foreign key constraints, which is expected behavior
            if response.status_code == 400:
                error = response.json()
                assert (
                    "dependent" in str(error).lower()
                    or "constraint" in str(error).lower()
                )
        except Exception:
            # Expected if proper foreign key constraints are in place
            pass

        # Verify detector still exists
        response = await async_test_client.get(f"/api/detectors/{detector_id}")
        response.raise_for_status()
        existing_detector = response.json()["data"]
        assert existing_detector["id"] == detector_id
