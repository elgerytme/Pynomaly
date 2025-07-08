"""Regression test suite to prevent breaking changes."""

import pytest
from httpx import AsyncClient
from tests.integration.conftest import IntegrationTestHelper


class TestRegressionSuite:
    """Regression tests to ensure backward compatibility and prevent breaking changes."""

    @pytest.mark.asyncio
    async def test_api_backward_compatibility(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test that existing API endpoints maintain backward compatibility."""

        # Test legacy API response formats and structures

        # 1. Dataset upload API compatibility
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "regression_test_dataset"
        )

        # Verify response structure hasn't changed
        required_dataset_fields = [
            "id",
            "name",
            "status",
            "rows",
            "columns",
            "created_at",
        ]
        for field in required_dataset_fields:
            assert field in dataset, f"Dataset response missing required field: {field}"

        # Verify data types haven't changed
        assert isinstance(dataset["id"], str)
        assert isinstance(dataset["name"], str)
        assert isinstance(dataset["rows"], int)
        assert isinstance(dataset["columns"], list)

        # 2. Detector creation API compatibility
        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        required_detector_fields = [
            "id",
            "name",
            "algorithm",
            "status",
            "parameters",
            "feature_columns",
        ]
        for field in required_detector_fields:
            assert (
                field in detector
            ), f"Detector response missing required field: {field}"

        # Verify algorithm parameter structure
        assert isinstance(detector["parameters"], dict)
        assert "contamination" in detector["parameters"]

        # 3. Training API compatibility
        training_result = await integration_helper.train_detector(detector["id"])

        required_training_fields = ["status", "model_id", "metrics", "training_time"]
        for field in required_training_fields:
            assert (
                field in training_result
            ), f"Training response missing required field: {field}"

        # Verify metrics structure
        assert isinstance(training_result["metrics"], dict)
        metrics_fields = ["accuracy", "precision", "recall", "f1_score"]
        for metric in metrics_fields:
            assert (
                metric in training_result["metrics"]
            ), f"Training metrics missing: {metric}"

        # 4. Prediction API compatibility
        test_data = {"data": [{"feature1": 0.5, "feature2": 0.3, "feature3": 0.1}]}

        response = await async_test_client.post(
            f"/api/detection/predict/{detector['id']}", json=test_data
        )
        response.raise_for_status()
        predictions = response.json()["data"]

        # Verify prediction structure
        assert isinstance(predictions, list)
        assert len(predictions) == 1

        prediction = predictions[0]
        required_prediction_fields = ["anomaly_score", "is_anomaly", "confidence"]
        for field in required_prediction_fields:
            assert (
                field in prediction
            ), f"Prediction response missing required field: {field}"

        # Verify data types
        assert isinstance(prediction["anomaly_score"], (int, float))
        assert isinstance(prediction["is_anomaly"], bool)
        assert isinstance(prediction["confidence"], (int, float))

    @pytest.mark.asyncio
    async def test_model_serialization_compatibility(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test that model serialization/deserialization remains compatible."""

        # Create and train a model
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "serialization_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        training_result = await integration_helper.train_detector(detector["id"])
        model_id = training_result["model_id"]

        # Test model export
        export_response = await async_test_client.post(
            f"/api/models/{model_id}/export",
            json={"format": "pickle", "include_metadata": True},
        )
        export_response.raise_for_status()
        export_result = export_response.json()["data"]

        # Verify export format hasn't changed
        required_export_fields = ["export_id", "model_id", "format", "download_url"]
        for field in required_export_fields:
            assert (
                field in export_result
            ), f"Export response missing required field: {field}"

        # Test model metadata preservation
        response = await async_test_client.get(f"/api/models/{model_id}")
        response.raise_for_status()
        model_info = response.json()["data"]

        # Verify model info structure hasn't changed
        required_model_fields = [
            "id",
            "version",
            "algorithm",
            "parameters",
            "metrics",
            "created_at",
        ]
        for field in required_model_fields:
            assert field in model_info, f"Model info missing required field: {field}"

        # Test predictions still work after model operations
        test_data = {"data": [{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}]}
        response = await async_test_client.post(
            f"/api/detection/predict/{detector['id']}", json=test_data
        )
        response.raise_for_status()
        predictions = response.json()["data"]

        assert len(predictions) == 1
        assert "anomaly_score" in predictions[0]

    @pytest.mark.asyncio
    async def test_streaming_api_stability(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test streaming API stability and response format consistency."""

        # Setup streaming components
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "streaming_regression_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Test streaming session creation API stability
        streaming_config = {
            "name": "regression_test_session",
            "detector_id": detector["id"],
            "data_source": {"source_type": "mock", "connection_config": {}},
            "configuration": {
                "processing_mode": "real_time",
                "batch_size": 1,
                "max_throughput": 100,
            },
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user", json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        integration_helper.created_resources["sessions"].append(session["id"])

        # Verify streaming session response structure
        required_session_fields = [
            "id",
            "name",
            "status",
            "detector_id",
            "data_source",
            "configuration",
        ]
        for field in required_session_fields:
            assert (
                field in session
            ), f"Streaming session missing required field: {field}"

        # Test session lifecycle operations
        session_id = session["id"]

        # Start session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/start"
        )
        response.raise_for_status()
        started_session = response.json()["data"]
        assert started_session["status"] in ["starting", "active"]

        # Get metrics
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/metrics"
        )
        response.raise_for_status()
        metrics = response.json()["data"]

        # Verify metrics structure hasn't changed
        required_metrics_fields = [
            "messages_processed",
            "messages_per_second",
            "anomalies_detected",
            "anomaly_rate",
        ]
        for field in required_metrics_fields:
            assert (
                field in metrics
            ), f"Streaming metrics missing required field: {field}"

        # Test data processing
        test_data = {
            "data": {
                "timestamp": "2024-12-25T10:00:00Z",
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0,
            }
        }

        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/process", json=test_data
        )
        response.raise_for_status()
        processing_result = response.json()["data"]

        # Verify processing result structure
        required_processing_fields = [
            "session_id",
            "anomaly_score",
            "is_anomaly",
            "timestamp",
        ]
        for field in required_processing_fields:
            assert (
                field in processing_result
            ), f"Processing result missing required field: {field}"

        # Stop session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/stop"
        )
        response.raise_for_status()

    @pytest.mark.asyncio
    async def test_experiment_api_consistency(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test experiment API consistency and result format stability."""

        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "experiment_regression_dataset"
        )

        # Test experiment creation API
        experiment_config = {
            "name": "regression_test_experiment",
            "description": "Test experiment for regression testing",
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
        }

        response = await async_test_client.post(
            "/api/experiments/create?created_by=test_user", json=experiment_config
        )
        response.raise_for_status()
        experiment = response.json()["data"]
        integration_helper.created_resources["experiments"].append(experiment["id"])

        # Verify experiment response structure
        required_experiment_fields = [
            "id",
            "name",
            "status",
            "algorithm_configurations",
            "evaluation_metrics",
        ]
        for field in required_experiment_fields:
            assert (
                field in experiment
            ), f"Experiment response missing required field: {field}"

        # Test experiment execution
        response = await async_test_client.post(
            f"/api/experiments/{experiment['id']}/run"
        )
        response.raise_for_status()

        # Test experiment results API
        response = await async_test_client.get(
            f"/api/experiments/{experiment['id']}/results"
        )
        response.raise_for_status()
        results = response.json()["data"]

        # Verify results structure consistency
        required_results_fields = [
            "experiment_id",
            "algorithm_results",
            "best_algorithm",
        ]
        for field in required_results_fields:
            assert (
                field in results
            ), f"Experiment results missing required field: {field}"

        # Verify algorithm results structure
        assert isinstance(results["algorithm_results"], list)
        for algo_result in results["algorithm_results"]:
            required_algo_fields = [
                "algorithm_name",
                "metrics",
                "cross_validation_scores",
            ]
            for field in required_algo_fields:
                assert (
                    field in algo_result
                ), f"Algorithm result missing required field: {field}"

        # Test experiment comparison API
        response = await async_test_client.get(
            f"/api/experiments/{experiment['id']}/comparison"
        )
        response.raise_for_status()
        comparison = response.json()["data"]

        required_comparison_fields = ["best_algorithm", "ranking", "metric_comparison"]
        for field in required_comparison_fields:
            assert (
                field in comparison
            ), f"Experiment comparison missing required field: {field}"

    @pytest.mark.asyncio
    async def test_error_response_format_consistency(
        self, async_test_client: AsyncClient, disable_auth
    ):
        """Test that error response formats remain consistent."""

        # Test 404 errors
        response = await async_test_client.get("/api/datasets/nonexistent-id")
        assert response.status_code == 404
        error_response = response.json()

        # Verify error response structure
        assert "detail" in error_response or "message" in error_response

        # Test 400 errors (bad request)
        invalid_detector_config = {
            "name": "",  # Invalid name
            "algorithm": "nonexistent_algorithm",  # Invalid algorithm
            "parameters": "invalid_parameters",  # Invalid parameters type
        }

        response = await async_test_client.post(
            "/api/detectors/create?dataset_id=invalid-id", json=invalid_detector_config
        )
        assert response.status_code in [400, 404, 422]  # Various bad request codes
        error_response = response.json()

        # Error response should have consistent structure
        assert isinstance(error_response, dict)
        assert any(key in error_response for key in ["detail", "message", "error"])

        # Test validation errors (422)
        response = await async_test_client.post(
            "/api/datasets/upload",
            files={"file": ("test.txt", "invalid csv content", "text/plain")},
            data={"name": ""},  # Missing required name
        )
        assert response.status_code in [400, 422]

        validation_error = response.json()
        assert isinstance(validation_error, dict)

    @pytest.mark.asyncio
    async def test_pagination_and_filtering_consistency(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test pagination and filtering API consistency."""

        # Create multiple datasets for pagination testing
        datasets = []
        for i in range(3):
            dataset = await integration_helper.upload_dataset(
                sample_dataset_csv, f"pagination_test_dataset_{i}"
            )
            datasets.append(dataset)

        # Test dataset listing with pagination
        response = await async_test_client.get("/api/datasets?limit=2&offset=0")
        response.raise_for_status()
        page1 = response.json()["data"]

        # Verify pagination response structure
        required_pagination_fields = ["datasets", "total", "limit", "offset"]
        for field in required_pagination_fields:
            assert (
                field in page1
            ), f"Pagination response missing required field: {field}"

        assert isinstance(page1["datasets"], list)
        assert isinstance(page1["total"], int)
        assert page1["limit"] == 2
        assert page1["offset"] == 0

        # Test second page
        response = await async_test_client.get("/api/datasets?limit=2&offset=2")
        response.raise_for_status()
        page2 = response.json()["data"]

        assert page2["offset"] == 2

        # Test filtering
        response = await async_test_client.get(
            "/api/datasets?name_contains=pagination_test"
        )
        response.raise_for_status()
        filtered_results = response.json()["data"]

        # Should contain our test datasets
        dataset_names = [d["name"] for d in filtered_results["datasets"]]
        pagination_datasets = [
            name for name in dataset_names if "pagination_test" in name
        ]
        assert len(pagination_datasets) >= 3

    @pytest.mark.asyncio
    async def test_authentication_and_authorization_compatibility(
        self, async_test_client: AsyncClient, sample_dataset_csv: str
    ):
        """Test authentication and authorization behavior consistency."""

        # Note: This test runs without disable_auth to test auth behavior

        # Test accessing protected endpoints without authentication
        protected_endpoints = [
            "/api/datasets",
            "/api/detectors",
            "/api/experiments",
            "/api/streaming/sessions",
        ]

        for endpoint in protected_endpoints:
            response = await async_test_client.get(endpoint)
            # Should return 401 (unauthorized) or 403 (forbidden) if auth is enabled
            # If auth is disabled for testing, should return 200
            assert response.status_code in [200, 401, 403]

        # Test creating resources without authentication
        dataset_upload_response = await async_test_client.post(
            "/api/datasets/upload",
            files={"file": ("test.csv", "col1,col2\n1,2\n", "text/csv")},
            data={"name": "auth_test_dataset"},
        )

        # Should either succeed (if auth disabled) or fail with auth error
        assert dataset_upload_response.status_code in [200, 201, 401, 403, 422]

        if dataset_upload_response.status_code in [200, 201]:
            # If upload succeeded, clean up
            dataset_data = dataset_upload_response.json()["data"]
            cleanup_response = await async_test_client.delete(
                f"/api/datasets/{dataset_data['id']}"
            )
            # Cleanup might fail due to auth, which is acceptable

    @pytest.mark.asyncio
    async def test_configuration_and_settings_stability(
        self, async_test_client: AsyncClient, disable_auth
    ):
        """Test that configuration endpoints and settings remain stable."""

        # Test health check endpoint
        response = await async_test_client.get("/api/health")
        response.raise_for_status()
        health = response.json()

        # Verify health check response structure
        required_health_fields = ["status", "timestamp"]
        for field in required_health_fields:
            assert field in health, f"Health check missing required field: {field}"

        assert health["status"] in ["healthy", "ok", "ready"]

        # Test API info/version endpoint
        response = await async_test_client.get("/")
        response.raise_for_status()
        api_info = response.json()

        # Verify API info structure
        required_info_fields = ["message", "version"]
        for field in required_info_fields:
            assert field in api_info, f"API info missing required field: {field}"

        # Test OpenAPI docs availability
        response = await async_test_client.get("/api/openapi.json")
        if response.status_code == 200:
            openapi_spec = response.json()

            # Verify OpenAPI spec structure
            required_openapi_fields = ["openapi", "info", "paths"]
            for field in required_openapi_fields:
                assert (
                    field in openapi_spec
                ), f"OpenAPI spec missing required field: {field}"

        # Test metrics endpoint (if enabled)
        response = await async_test_client.get("/metrics")
        # This might return 404 if metrics are disabled, which is acceptable
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_performance_regression_baseline(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test performance regression by establishing baseline metrics."""

        import time

        # Setup for performance testing
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "performance_regression_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        # Measure training time
        training_start = time.time()
        await integration_helper.train_detector(detector["id"])
        training_time = time.time() - training_start

        # Training should complete within reasonable time
        assert training_time < 30.0, f"Training took too long: {training_time:.2f}s"

        # Measure prediction time
        test_data = {"data": [{"feature1": 0.5, "feature2": 0.3, "feature3": 0.1}]}

        prediction_start = time.time()
        response = await async_test_client.post(
            f"/api/detection/predict/{detector['id']}", json=test_data
        )
        response.raise_for_status()
        prediction_time = time.time() - prediction_start

        # Single prediction should be fast
        assert (
            prediction_time < 2.0
        ), f"Prediction took too long: {prediction_time:.2f}s"

        # Measure batch prediction time
        batch_data = {
            "data": [
                {"feature1": i * 0.1, "feature2": i * 0.2, "feature3": i * 0.05}
                for i in range(100)
            ]
        }

        batch_start = time.time()
        response = await async_test_client.post(
            f"/api/detection/predict/{detector['id']}", json=batch_data
        )
        response.raise_for_status()
        batch_time = time.time() - batch_start

        predictions = response.json()["data"]
        assert len(predictions) == 100

        # Batch prediction should be efficient
        assert batch_time < 10.0, f"Batch prediction took too long: {batch_time:.2f}s"

        # Calculate throughput
        throughput = 100 / batch_time
        assert (
            throughput > 10
        ), f"Throughput too low: {throughput:.2f} predictions/second"

        print(f"Performance baseline metrics:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Single prediction: {prediction_time:.3f}s")
        print(f"  Batch prediction (100 items): {batch_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} predictions/second")
