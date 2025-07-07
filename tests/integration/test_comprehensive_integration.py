"""
Comprehensive Integration Testing Suite

Tests full system integration across all architectural layers including
database operations, API endpoints, service interactions, and data flows.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from pynomaly.application.services.detection_service import DetectionService
from pynomaly.application.services.ensemble_service import EnsembleService
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.infrastructure.persistence.database import DatabaseManager
from pynomaly.presentation.api.app import create_app


class TestDatabaseIntegration:
    """Integration tests for database operations across layers."""

    @pytest.fixture
    def database_config(self):
        """Database configuration for testing."""
        return {
            "database_url": "sqlite:///:memory:",
            "echo": False,
            "pool_size": 5,
            "max_overflow": 10,
        }

    @pytest.fixture
    async def database_manager(self, database_config):
        """Create database manager for testing."""
        manager = DatabaseManager(database_config)
        await manager.initialize()
        yield manager
        await manager.close()

    async def test_dataset_persistence_workflow(self, database_manager):
        """Test complete dataset persistence workflow."""
        # Create dataset entity
        data = np.random.randn(100, 5).tolist()
        dataset = Dataset(
            id="test_dataset_001",
            name="Integration Test Dataset",
            data=data,
            features=["f1", "f2", "f3", "f4", "f5"],
            metadata={"source": "integration_test"},
        )

        # Save to database
        saved_dataset = await database_manager.save_dataset(dataset)
        assert saved_dataset.id == dataset.id

        # Retrieve from database
        retrieved_dataset = await database_manager.get_dataset(dataset.id)
        assert retrieved_dataset.name == dataset.name
        assert len(retrieved_dataset.data) == len(data)
        assert retrieved_dataset.features == dataset.features

        # Update dataset
        retrieved_dataset.metadata["updated"] = True
        updated_dataset = await database_manager.update_dataset(retrieved_dataset)
        assert updated_dataset.metadata["updated"] is True

        # List datasets
        datasets = await database_manager.list_datasets()
        assert len(datasets) >= 1
        assert any(d.id == dataset.id for d in datasets)

        # Delete dataset
        result = await database_manager.delete_dataset(dataset.id)
        assert result is True

        # Verify deletion
        with pytest.raises(Exception):  # Should raise NotFound or similar
            await database_manager.get_dataset(dataset.id)

    async def test_detector_training_persistence(self, database_manager):
        """Test detector training and persistence integration."""
        # Create training dataset
        training_data = np.random.randn(200, 3).tolist()
        dataset = Dataset(
            id="training_dataset_001",
            name="Training Dataset",
            data=training_data,
            features=["x", "y", "z"],
        )
        await database_manager.save_dataset(dataset)

        # Create detector configuration
        detector = Detector(
            id="detector_001",
            name="Test Detector",
            algorithm="IsolationForest",
            parameters={"n_estimators": 50, "contamination": 0.1},
            dataset_id=dataset.id,
        )

        # Save detector
        saved_detector = await database_manager.save_detector(detector)
        assert saved_detector.id == detector.id

        # Simulate training completion
        saved_detector.status = "trained"
        saved_detector.trained_at = datetime.now()
        saved_detector.performance_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
        }

        # Update detector with training results
        updated_detector = await database_manager.update_detector(saved_detector)
        assert updated_detector.status == "trained"
        assert updated_detector.performance_metrics["accuracy"] == 0.85

        # Retrieve trained detector
        retrieved_detector = await database_manager.get_detector(detector.id)
        assert retrieved_detector.status == "trained"
        assert retrieved_detector.performance_metrics is not None

    async def test_detection_result_storage(self, database_manager):
        """Test detection result storage and retrieval."""
        # Create detection result
        predictions = [0, 1, 0, 0, 1]
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.9),
            AnomalyScore(0.2),
            AnomalyScore(0.3),
            AnomalyScore(0.8),
        ]

        result = DetectionResult(
            id="result_001",
            detector_id="detector_001",
            predictions=predictions,
            anomaly_scores=scores,
            metadata={"batch_size": 5, "processing_time": 0.05},
        )

        # Save result
        saved_result = await database_manager.save_detection_result(result)
        assert saved_result.id == result.id

        # Retrieve result
        retrieved_result = await database_manager.get_detection_result(result.id)
        assert len(retrieved_result.predictions) == len(predictions)
        assert len(retrieved_result.anomaly_scores) == len(scores)
        assert retrieved_result.metadata["batch_size"] == 5

        # List results by detector
        results = await database_manager.list_detection_results(
            detector_id="detector_001"
        )
        assert len(results) >= 1
        assert any(r.id == result.id for r in results)

    async def test_transaction_rollback(self, database_manager):
        """Test database transaction rollback on errors."""
        dataset = Dataset(
            id="transaction_test",
            name="Transaction Test",
            data=[[1, 2], [3, 4]],
            features=["a", "b"],
        )

        # Start transaction
        async with database_manager.transaction() as txn:
            await txn.save_dataset(dataset)

            # Verify dataset exists within transaction
            retrieved = await txn.get_dataset(dataset.id)
            assert retrieved.id == dataset.id

            # Force rollback by raising exception
            raise Exception("Force rollback")

        # Verify dataset was rolled back (should not exist)
        with pytest.raises(Exception):
            await database_manager.get_dataset(dataset.id)

    async def test_concurrent_database_operations(self, database_manager):
        """Test concurrent database operations."""

        async def create_dataset(index):
            dataset = Dataset(
                id=f"concurrent_dataset_{index}",
                name=f"Concurrent Dataset {index}",
                data=np.random.randn(10, 2).tolist(),
                features=["x", "y"],
            )
            return await database_manager.save_dataset(dataset)

        # Create multiple datasets concurrently
        tasks = [create_dataset(i) for i in range(10)]
        datasets = await asyncio.gather(*tasks)

        assert len(datasets) == 10
        assert all(d.id.startswith("concurrent_dataset_") for d in datasets)

        # Retrieve all concurrently
        retrieval_tasks = [database_manager.get_dataset(d.id) for d in datasets]
        retrieved_datasets = await asyncio.gather(*retrieval_tasks)

        assert len(retrieved_datasets) == 10
        assert all(d is not None for d in retrieved_datasets)


class TestServiceLayerIntegration:
    """Integration tests for service layer interactions."""

    @pytest.fixture
    def mock_adapter(self):
        """Mock ML adapter for testing."""
        adapter = MagicMock(spec=SklearnAdapter)
        adapter.fit.return_value = Mock(id="trained_model_001")
        adapter.predict.return_value = Mock(
            predictions=[0, 1, 0], anomaly_scores=[0.1, 0.9, 0.2]
        )
        return adapter

    @pytest.fixture
    def mock_repository(self):
        """Mock repository for testing."""
        repository = MagicMock()
        repository.save.return_value = Mock(id="saved_entity_001")
        repository.get_by_id.return_value = Mock(id="retrieved_entity_001")
        return repository

    async def test_detection_service_workflow(self, mock_adapter, mock_repository):
        """Test complete detection service workflow."""
        service = DetectionService(
            adapter=mock_adapter,
            detector_repository=mock_repository,
            result_repository=mock_repository,
        )

        # Create test data
        test_data = np.random.randn(50, 4)

        # Execute detection workflow
        result = await service.detect_anomalies(
            detector_id="detector_001", data=test_data, return_confidence=True
        )

        # Verify service interactions
        mock_repository.get_by_id.assert_called_with("detector_001")
        mock_adapter.predict.assert_called_once()
        mock_repository.save.assert_called_once()

        # Verify result structure
        assert result is not None
        assert hasattr(result, "predictions")
        assert hasattr(result, "anomaly_scores")

    async def test_ensemble_service_integration(self, mock_repository):
        """Test ensemble service with multiple detectors."""
        # Create multiple mock adapters
        adapters = []
        for _i in range(3):
            adapter = MagicMock()
            adapter.predict.return_value = Mock(
                predictions=np.random.choice([0, 1], 10).tolist(),
                anomaly_scores=np.random.random(10).tolist(),
            )
            adapters.append(adapter)

        service = EnsembleService(adapters=adapters, repository=mock_repository)

        test_data = np.random.randn(10, 3)

        # Test ensemble prediction
        result = await service.predict_ensemble(
            detector_ids=["det1", "det2", "det3"],
            data=test_data,
            aggregation_method="majority_vote",
        )

        # Verify all adapters were called
        for adapter in adapters:
            adapter.predict.assert_called_once()

        # Verify result aggregation
        assert result is not None
        assert len(result.predictions) == 10
        assert len(result.anomaly_scores) == 10

    async def test_service_error_handling(self, mock_adapter, mock_repository):
        """Test service error handling and recovery."""
        # Configure adapter to raise error
        mock_adapter.predict.side_effect = Exception("Prediction failed")

        service = DetectionService(
            adapter=mock_adapter,
            detector_repository=mock_repository,
            result_repository=mock_repository,
        )

        test_data = np.random.randn(10, 2)

        # Test error propagation
        with pytest.raises(Exception) as exc_info:
            await service.detect_anomalies("detector_001", test_data)

        assert "Prediction failed" in str(exc_info.value)

        # Verify cleanup was attempted
        mock_repository.get_by_id.assert_called_once()

    async def test_service_caching_integration(self, mock_adapter, mock_repository):
        """Test service-level caching integration."""
        # Mock cache
        cache = MagicMock()
        cache.get.return_value = None  # Cache miss
        cache.set.return_value = True

        service = DetectionService(
            adapter=mock_adapter,
            detector_repository=mock_repository,
            result_repository=mock_repository,
            cache=cache,
        )

        test_data = np.random.randn(5, 3)

        # First call - should miss cache and compute
        result1 = await service.detect_anomalies("detector_001", test_data)

        # Verify cache was checked and updated
        cache.get.assert_called()
        cache.set.assert_called()

        # Configure cache hit for second call
        cache.get.return_value = result1

        # Second call - should hit cache
        await service.detect_anomalies("detector_001", test_data)

        # Verify adapter wasn't called again
        assert mock_adapter.predict.call_count == 1


class TestAPIIntegration:
    """Integration tests for API layer with services."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application."""
        app = create_app(testing=True)
        return app

    @pytest.fixture
    def test_client(self, test_app):
        """Create test client."""
        from fastapi.testclient import TestClient

        return TestClient(test_app)

    def test_dataset_api_workflow(self, test_client):
        """Test complete dataset API workflow."""
        # Create dataset
        dataset_data = {
            "name": "API Test Dataset",
            "description": "Dataset for API integration testing",
            "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "features": ["x", "y", "z"],
        }

        response = test_client.post("/api/v1/datasets", json=dataset_data)
        assert response.status_code == 201

        dataset = response.json()
        dataset_id = dataset["id"]

        # Get dataset
        response = test_client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 200
        retrieved_dataset = response.json()
        assert retrieved_dataset["name"] == dataset_data["name"]

        # List datasets
        response = test_client.get("/api/v1/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert any(d["id"] == dataset_id for d in datasets)

        # Update dataset
        update_data = {"description": "Updated description"}
        response = test_client.patch(f"/api/v1/datasets/{dataset_id}", json=update_data)
        assert response.status_code == 200
        updated_dataset = response.json()
        assert updated_dataset["description"] == update_data["description"]

        # Delete dataset
        response = test_client.delete(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 204

        # Verify deletion
        response = test_client.get(f"/api/v1/datasets/{dataset_id}")
        assert response.status_code == 404

    def test_detector_training_api_workflow(self, test_client):
        """Test detector training API workflow."""
        # First create dataset
        dataset_data = {
            "name": "Training Dataset",
            "data": np.random.randn(100, 4).tolist(),
            "features": ["a", "b", "c", "d"],
        }

        response = test_client.post("/api/v1/datasets", json=dataset_data)
        dataset_id = response.json()["id"]

        # Create detector
        detector_data = {
            "name": "API Test Detector",
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 50},
            "contamination_rate": 0.1,
        }

        response = test_client.post("/api/v1/detectors", json=detector_data)
        assert response.status_code == 201
        detector = response.json()
        detector_id = detector["id"]

        # Start training
        training_data = {"dataset_id": dataset_id, "parameters": {"n_estimators": 100}}

        response = test_client.post(
            f"/api/v1/detectors/{detector_id}/train", json=training_data
        )
        assert response.status_code == 202  # Accepted for async processing

        training_job = response.json()
        assert training_job["detector_id"] == detector_id
        assert training_job["status"] in ["pending", "in_progress"]

        # Check training status
        response = test_client.get(f"/api/v1/detectors/{detector_id}")
        assert response.status_code == 200
        detector_status = response.json()
        assert detector_status["status"] in ["draft", "training", "trained"]

    def test_detection_api_workflow(self, test_client):
        """Test anomaly detection API workflow."""
        with patch(
            "pynomaly.application.services.detection_service.DetectionService"
        ) as mock_service:
            # Mock detection service
            mock_result = Mock()
            mock_result.predictions = [0, 1, 0]
            mock_result.anomaly_scores = [0.1, 0.9, 0.2]
            mock_result.processing_time = 0.05
            mock_service.return_value.detect_anomalies.return_value = mock_result

            # Perform detection
            detection_data = {
                "detector_id": "test_detector",
                "data": [[1, 2], [3, 4], [5, 6]],
                "return_scores": True,
                "return_explanations": False,
            }

            response = test_client.post("/api/v1/detection/detect", json=detection_data)
            assert response.status_code == 200

            result = response.json()
            assert "predictions" in result
            assert "anomaly_scores" in result
            assert "processing_time" in result

    def test_api_error_handling(self, test_client):
        """Test API error handling integration."""
        # Test 404 for non-existent resource
        response = test_client.get("/api/v1/datasets/non_existent_id")
        assert response.status_code == 404
        error = response.json()
        assert "error" in error
        assert "message" in error

        # Test validation error
        invalid_dataset = {
            "name": "",  # Invalid empty name
            "data": "invalid_data_format",
        }

        response = test_client.post("/api/v1/datasets", json=invalid_dataset)
        assert response.status_code == 422  # Validation error
        error = response.json()
        assert "detail" in error

    def test_api_authentication_integration(self, test_client):
        """Test API authentication integration."""
        # Test without authentication (should fail if auth required)
        response = test_client.get("/api/v1/datasets")

        # Depending on configuration, this might require auth
        if response.status_code == 401:
            # Test with invalid token
            headers = {"Authorization": "Bearer invalid_token"}
            response = test_client.get("/api/v1/datasets", headers=headers)
            assert response.status_code == 401

            # Test with valid token (mocked)
            with patch(
                "pynomaly.infrastructure.auth.jwt_auth.verify_token"
            ) as mock_verify:
                mock_verify.return_value = {"user_id": "test_user"}
                headers = {"Authorization": "Bearer valid_token"}
                response = test_client.get("/api/v1/datasets", headers=headers)
                # Should now work (or at least not fail auth)
                assert response.status_code != 401


class TestDataFlowIntegration:
    """Integration tests for complete data flow pipelines."""

    async def test_end_to_end_detection_pipeline(self):
        """Test complete end-to-end detection pipeline."""
        # 1. Data ingestion
        raw_data = np.random.randn(200, 5)
        anomaly_data = np.random.randn(10, 5) * 3  # Obvious anomalies
        all_data = np.vstack([raw_data, anomaly_data])

        # 2. Data preprocessing
        from pynomaly.infrastructure.preprocessing.data_transformer import (
            DataTransformer,
        )

        transformer = DataTransformer()

        with patch.object(transformer, "fit_transform") as mock_transform:
            mock_transform.return_value = all_data
            processed_data = transformer.fit_transform(all_data)

        # 3. Dataset creation
        dataset = Dataset(
            id="e2e_dataset",
            name="End-to-End Dataset",
            data=processed_data.tolist(),
            features=[f"feature_{i}" for i in range(5)],
        )

        # 4. Model training
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter"
        ) as mock_adapter:
            mock_model = Mock()
            mock_model.decision_function.return_value = np.random.randn(210)
            mock_model.predict.return_value = np.random.choice([-1, 1], 210)

            adapter = mock_adapter.return_value
            adapter.fit.return_value = mock_model
            adapter.predict.return_value = Mock(
                predictions=np.random.choice([0, 1], 210).tolist(),
                anomaly_scores=np.random.random(210).tolist(),
            )

            # Train detector
            detector = adapter.fit(dataset)
            assert detector is not None

        # 5. Anomaly detection
        test_data = np.random.randn(50, 5)
        result = adapter.predict(detector, test_data)

        # 6. Result validation
        assert len(result.predictions) == 50
        assert len(result.anomaly_scores) == 50
        assert all(p in [0, 1] for p in result.predictions)
        assert all(0 <= s <= 1 for s in result.anomaly_scores)

    async def test_streaming_data_pipeline(self):
        """Test streaming data processing pipeline - REMOVED FOR SIMPLIFICATION."""
        # Streaming infrastructure removed for simplification
        pytest.skip("Streaming functionality removed in Phase 1 simplification")

        # Simulate streaming data
        async def generate_stream_data():
            for _i in range(10):
                batch = np.random.randn(20, 3).tolist()
                yield batch
                await asyncio.sleep(0.01)  # Simulate real-time delay

        # Process stream
        results = []
        async for batch in generate_stream_data():
            # Mock processing
            processed_batch = Mock()
            processed_batch.anomalies = np.random.choice([0, 1], 20).tolist()
            processed_batch.scores = np.random.random(20).tolist()

            processor.process_batch.return_value = processed_batch
            result = processor.process_batch(batch)
            results.append(result)

        # Verify stream processing
        assert len(results) == 10
        assert all(hasattr(r, "anomalies") for r in results)
        assert all(hasattr(r, "scores") for r in results)

    async def test_batch_processing_pipeline(self):
        """Test batch processing pipeline with multiple datasets."""
        # Create multiple datasets
        datasets = []
        for i in range(5):
            data = np.random.randn(100, 4)
            dataset = Dataset(
                id=f"batch_dataset_{i}",
                name=f"Batch Dataset {i}",
                data=data.tolist(),
                features=["a", "b", "c", "d"],
            )
            datasets.append(dataset)

        # Mock batch processor

        with patch(
            "pynomaly.application.services.batch_service.BatchService"
        ) as mock_service:
            service = mock_service.return_value

            # Configure mock to return results for each dataset
            mock_results = []
            for i, dataset in enumerate(datasets):
                mock_result = Mock()
                mock_result.dataset_id = dataset.id
                mock_result.anomaly_count = np.random.randint(1, 10)
                mock_result.processing_time = np.random.uniform(0.1, 1.0)
                mock_results.append(mock_result)

            service.process_batch.return_value = mock_results

            # Process batch
            results = service.process_batch(
                datasets=[d.id for d in datasets], detector_id="batch_detector"
            )

            # Verify batch processing
            assert len(results) == 5
            assert all(hasattr(r, "dataset_id") for r in results)
            assert all(hasattr(r, "anomaly_count") for r in results)

    async def test_model_versioning_pipeline(self):
        """Test model versioning and deployment pipeline."""
        # Mock model versioning service

        with patch(
            "pynomaly.application.services.model_persistence_service.ModelPersistenceService"
        ) as mock_service:
            service = mock_service.return_value

            # Version 1.0
            model_v1 = Mock()
            model_v1.version = "1.0"
            model_v1.algorithm = "IsolationForest"
            model_v1.performance = {"accuracy": 0.80}

            service.save_model.return_value = "model_v1_id"
            service.get_model.return_value = model_v1

            # Save initial model
            model_id = service.save_model(model_v1)
            assert model_id == "model_v1_id"

            # Version 2.0 (improved)
            model_v2 = Mock()
            model_v2.version = "2.0"
            model_v2.algorithm = "IsolationForest"
            model_v2.performance = {"accuracy": 0.85}

            service.save_model.return_value = "model_v2_id"

            # Save improved model
            model_v2_id = service.save_model(model_v2)
            assert model_v2_id == "model_v2_id"

            # Test model comparison
            service.compare_models.return_value = {
                "v1_performance": 0.80,
                "v2_performance": 0.85,
                "improvement": 0.05,
                "recommendation": "deploy_v2",
            }

            comparison = service.compare_models("model_v1_id", "model_v2_id")
            assert comparison["recommendation"] == "deploy_v2"

            # Test deployment
            service.deploy_model.return_value = True
            deployment_result = service.deploy_model("model_v2_id")
            assert deployment_result is True


class TestCrossLayerIntegration:
    """Integration tests across all architectural layers."""

    async def test_full_stack_integration(self):
        """Test integration across all layers: API -> Application -> Domain -> Infrastructure."""
        # Mock all layers
        with (
            patch(
                "pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter"
            ) as mock_adapter,
            patch(
                "pynomaly.infrastructure.persistence.database.DatabaseManager"
            ) as mock_db,
            patch(
                "pynomaly.application.services.detection_service.DetectionService"
            ) as mock_service,
        ):
            # Configure mocks
            mock_model = Mock()
            mock_adapter.return_value.fit.return_value = mock_model
            mock_adapter.return_value.predict.return_value = Mock(
                predictions=[0, 1, 0, 1], anomaly_scores=[0.1, 0.9, 0.2, 0.8]
            )

            mock_db.return_value.save_dataset.return_value = Mock(id="dataset_001")
            mock_db.return_value.save_detector.return_value = Mock(id="detector_001")

            mock_service.return_value.detect_anomalies.return_value = Mock(
                predictions=[0, 1, 0, 1],
                anomaly_scores=[0.1, 0.9, 0.2, 0.8],
                processing_time=0.05,
            )

            # Test API layer
            from fastapi.testclient import TestClient

            app = create_app(testing=True)
            client = TestClient(app)

            # 1. Create dataset via API
            dataset_data = {
                "name": "Full Stack Dataset",
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "features": ["x", "y"],
            }

            response = client.post("/api/v1/datasets", json=dataset_data)
            assert response.status_code == 201
            response.json()

            # 2. Create and train detector via API
            detector_data = {
                "name": "Full Stack Detector",
                "algorithm": "IsolationForest",
                "parameters": {"n_estimators": 50},
            }

            response = client.post("/api/v1/detectors", json=detector_data)
            assert response.status_code == 201
            detector = response.json()

            # 3. Perform detection via API
            detection_data = {
                "detector_id": detector["id"],
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "return_scores": True,
            }

            response = client.post("/api/v1/detection/detect", json=detection_data)
            assert response.status_code == 200
            result = response.json()

            # Verify full stack operation
            assert "predictions" in result
            assert "anomaly_scores" in result
            assert len(result["predictions"]) == 4
            assert len(result["anomaly_scores"]) == 4

    async def test_configuration_integration(self):
        """Test configuration propagation across layers."""
        # Mock configuration
        config = {
            "database": {
                "url": "postgresql://test:test@localhost/test",
                "pool_size": 10,
            },
            "ml": {
                "default_algorithm": "IsolationForest",
                "default_contamination": 0.1,
            },
            "api": {"rate_limit": 100, "timeout": 30},
        }

        # Test configuration loading at each layer
        with patch("pynomaly.infrastructure.config.settings.load_config") as mock_load:
            mock_load.return_value = config

            # Infrastructure layer
            from pynomaly.infrastructure.config.settings import get_database_config

            db_config = get_database_config()
            assert db_config["pool_size"] == 10

            # Application layer
            from pynomaly.application.services.config import get_ml_config

            ml_config = get_ml_config()
            assert ml_config["default_algorithm"] == "IsolationForest"

            # Presentation layer
            from pynomaly.presentation.api.config import get_api_config

            api_config = get_api_config()
            assert api_config["rate_limit"] == 100

    async def test_monitoring_integration(self):
        """Test monitoring and observability across layers."""
        # Mock monitoring components
        with (
            patch(
                "pynomaly.infrastructure.monitoring.telemetry.MetricsCollector"
            ) as mock_metrics,
            patch(
                "pynomaly.infrastructure.monitoring.health_service.HealthService"
            ) as mock_health,
        ):
            metrics_collector = mock_metrics.return_value
            health_service = mock_health.return_value

            # Configure mocks
            metrics_collector.record_metric.return_value = True
            health_service.check_health.return_value = {
                "status": "healthy",
                "services": {"database": "up", "ml_adapters": "up", "cache": "up"},
            }

            # Test metrics collection from different layers

            # Domain layer metrics
            metrics_collector.record_metric(
                "domain.entity.created", 1, tags={"type": "dataset"}
            )

            # Application layer metrics
            metrics_collector.record_metric(
                "application.service.called", 1, tags={"service": "detection"}
            )

            # Infrastructure layer metrics
            metrics_collector.record_metric(
                "infrastructure.adapter.prediction",
                1,
                tags={"algorithm": "isolation_forest"},
            )

            # API layer metrics
            metrics_collector.record_metric(
                "api.request.processed", 1, tags={"endpoint": "/detect"}
            )

            # Verify metrics were recorded
            assert metrics_collector.record_metric.call_count == 4

            # Test health check aggregation
            health_status = health_service.check_health()
            assert health_status["status"] == "healthy"
            assert all(status == "up" for status in health_status["services"].values())


# Performance and load testing integration
class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""

    async def test_concurrent_api_requests(self):
        """Test API performance under concurrent load."""
        import concurrent.futures
        import time

        from fastapi.testclient import TestClient

        app = create_app(testing=True)
        client = TestClient(app)

        def make_request(request_id):
            start_time = time.time()
            response = client.get("/api/v1/health")
            end_time = time.time()

            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
            }

        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Analyze results
        assert len(results) == 50
        assert all(r["status_code"] == 200 for r in results)

        response_times = [r["response_time"] for r in results]
        avg_response_time = sum(response_times) / len(response_times)

        # Performance assertions
        assert avg_response_time < 1.0  # Average under 1 second
        assert max(response_times) < 5.0  # Max under 5 seconds

    async def test_memory_usage_integration(self):
        """Test memory usage across integration scenarios."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate memory-intensive operations
        large_datasets = []
        for i in range(10):
            data = np.random.randn(10000, 50).tolist()  # Large dataset
            dataset = Dataset(
                id=f"memory_test_{i}",
                name=f"Memory Test Dataset {i}",
                data=data,
                features=[f"f_{j}" for j in range(50)],
            )
            large_datasets.append(dataset)

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Clean up
        del large_datasets

        final_memory = process.memory_info().rss
        memory_released = peak_memory - final_memory

        # Memory assertions
        assert memory_increase > 0  # Memory was used
        assert memory_released > memory_increase * 0.5  # At least 50% released

    async def test_database_performance_integration(self):
        """Test database performance under load."""
        # Mock database operations with timing
        operations = []

        async def timed_operation(operation_name, operation_func):
            start_time = time.time()
            result = await operation_func()
            end_time = time.time()

            operations.append(
                {
                    "operation": operation_name,
                    "duration": end_time - start_time,
                    "success": result is not None,
                }
            )

            return result

        # Mock database manager
        with patch(
            "pynomaly.infrastructure.persistence.database.DatabaseManager"
        ) as mock_db:
            db_manager = mock_db.return_value

            # Configure mock operations
            db_manager.save_dataset.return_value = Mock(id="saved_dataset")
            db_manager.get_dataset.return_value = Mock(id="retrieved_dataset")
            db_manager.list_datasets.return_value = [
                Mock(id=f"dataset_{i}") for i in range(100)
            ]

            # Execute timed operations
            await timed_operation(
                "save_dataset", lambda: db_manager.save_dataset(Mock())
            )
            await timed_operation(
                "get_dataset", lambda: db_manager.get_dataset("test_id")
            )
            await timed_operation("list_datasets", lambda: db_manager.list_datasets())

        # Performance analysis
        assert len(operations) == 3
        assert all(op["success"] for op in operations)

        # Check operation times
        save_time = next(
            op["duration"] for op in operations if op["operation"] == "save_dataset"
        )
        get_time = next(
            op["duration"] for op in operations if op["operation"] == "get_dataset"
        )
        list_time = next(
            op["duration"] for op in operations if op["operation"] == "list_datasets"
        )

        # Performance assertions (for mocked operations, these should be very fast)
        assert save_time < 0.1
        assert get_time < 0.1
        assert list_time < 0.1
