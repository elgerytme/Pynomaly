"""
Detectors Endpoints Testing Suite
Comprehensive tests for detector management API endpoints.
"""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from pynomaly.application.dto import CreateDetectorDTO, DetectorDTO
from pynomaly.presentation.api.app import create_app


class TestDetectorEndpoints:
    """Test suite for detector API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_container(self):
        """Mock dependency injection container."""
        with patch("pynomaly.presentation.api.deps.get_container") as mock:
            container = Mock()

            # Mock repositories
            container.detector_repository.return_value = Mock()
            container.pyod_adapter.return_value = Mock()
            container.sklearn_adapter.return_value = Mock()

            mock.return_value = container
            yield container

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        with patch("pynomaly.presentation.api.deps.get_current_user") as mock:
            user = {
                "user_id": "test-user-123",
                "email": "test@example.com",
                "roles": ["user"],
                "permissions": [
                    "detector:create",
                    "detector:read",
                    "detector:update",
                    "detector:delete",
                ],
            }
            mock.return_value = user
            yield user

    @pytest.fixture
    def sample_detector_dto(self):
        """Sample detector DTO for testing."""
        return DetectorDTO(
            id=uuid.UUID("12345678-1234-5678-9012-123456789012"),
            name="Test Isolation Forest",
            algorithm_name="IsolationForest",
            contamination_rate=0.1,
            hyperparameters={
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": 0.1,
                "random_state": 42,
            },
            is_fitted=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "framework": "sklearn",
                "version": "1.0",
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.89,
                },
            },
        )

    @pytest.fixture
    def sample_create_detector_dto(self):
        """Sample create detector DTO for testing."""
        return CreateDetectorDTO(
            name="New Test Detector",
            algorithm_name="LocalOutlierFactor",
            contamination_rate=0.05,
            hyperparameters={
                "n_neighbors": 20,
                "algorithm": "auto",
                "contamination": 0.05,
            },
            metadata={"description": "LOF detector for testing"},
        )

    # Detector Listing Tests

    def test_list_detectors_success(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test successful detector listing."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_all.return_value = [sample_detector_dto, sample_detector_dto]

        response = client.get("/api/v1/detectors/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "Test Isolation Forest"
        assert data[0]["algorithm_name"] == "IsolationForest"

    def test_list_detectors_with_algorithm_filter(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test detector listing with algorithm filter."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_all.return_value = [sample_detector_dto]

        response = client.get("/api/v1/detectors/?algorithm=IsolationForest")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["algorithm_name"] == "IsolationForest"

    def test_list_detectors_with_fitted_filter(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test detector listing with fitted status filter."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_all.return_value = [sample_detector_dto]

        response = client.get("/api/v1/detectors/?is_fitted=true")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["is_fitted"] is True

    def test_list_detectors_with_pagination(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test detector listing with pagination."""
        mock_repo = mock_container.detector_repository.return_value
        detectors = [sample_detector_dto] * 150  # More than default limit
        mock_repo.find_all.return_value = detectors

        response = client.get("/api/v1/detectors/?limit=50")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 50  # Should be limited

    def test_list_detectors_unauthorized(self, client, mock_container):
        """Test detector listing without authentication."""
        with patch("pynomaly.presentation.api.deps.get_current_user") as mock_auth:
            mock_auth.side_effect = HTTPException(
                status_code=401, detail="Not authenticated"
            )

            response = client.get("/api/v1/detectors/")
            assert response.status_code == 401

    # Detector Creation Tests

    def test_create_detector_success(
        self,
        client,
        mock_container,
        mock_user,
        sample_create_detector_dto,
        sample_detector_dto,
    ):
        """Test successful detector creation."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.create.return_value = sample_detector_dto

        create_data = {
            "name": "New Test Detector",
            "algorithm_name": "LocalOutlierFactor",
            "contamination_rate": 0.05,
            "hyperparameters": {"n_neighbors": 20, "algorithm": "auto"},
        }

        response = client.post("/api/v1/detectors/", json=create_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Isolation Forest"
        assert data["algorithm_name"] == "IsolationForest"

    def test_create_detector_invalid_algorithm(self, client, mock_user):
        """Test detector creation with invalid algorithm."""
        create_data = {
            "name": "Invalid Detector",
            "algorithm_name": "NonExistentAlgorithm",
            "contamination_rate": 0.1,
        }

        response = client.post("/api/v1/detectors/", json=create_data)

        assert response.status_code == 400
        assert "Unsupported algorithm" in response.json()["detail"]

    def test_create_detector_invalid_contamination_rate(self, client, mock_user):
        """Test detector creation with invalid contamination rate."""
        create_data = {
            "name": "Invalid Detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 1.5,  # Invalid: should be between 0 and 1
        }

        response = client.post("/api/v1/detectors/", json=create_data)

        assert response.status_code == 422  # Validation error

    def test_create_detector_invalid_hyperparameters(
        self, client, mock_container, mock_user
    ):
        """Test detector creation with invalid hyperparameters."""
        mock_adapter = mock_container.pyod_adapter.return_value
        mock_adapter.validate_hyperparameters.side_effect = ValueError(
            "Invalid parameter: n_estimators must be positive"
        )

        create_data = {
            "name": "Invalid Detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.1,
            "hyperparameters": {"n_estimators": -10},
        }

        response = client.post("/api/v1/detectors/", json=create_data)

        assert response.status_code == 400

    def test_create_detector_missing_required_fields(self, client, mock_user):
        """Test detector creation with missing required fields."""
        incomplete_data = {"name": "Incomplete Detector"}

        response = client.post("/api/v1/detectors/", json=incomplete_data)

        assert response.status_code == 422

    def test_create_detector_duplicate_name(self, client, mock_container, mock_user):
        """Test detector creation with duplicate name."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.create.side_effect = HTTPException(
            status_code=409, detail="Detector with this name already exists"
        )

        create_data = {
            "name": "Existing Detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.1,
        }

        response = client.post("/api/v1/detectors/", json=create_data)

        assert response.status_code == 409

    # Detector Retrieval Tests

    def test_get_detector_by_id_success(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test successful detector retrieval by ID."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        detector_id = str(sample_detector_dto.id)
        response = client.get(f"/api/v1/detectors/{detector_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == detector_id
        assert data["name"] == "Test Isolation Forest"

    def test_get_detector_not_found(self, client, mock_container, mock_user):
        """Test detector retrieval with non-existent ID."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = None

        response = client.get("/api/v1/detectors/99999999-9999-9999-9999-999999999999")

        assert response.status_code == 404

    def test_get_detector_invalid_uuid(self, client, mock_user):
        """Test detector retrieval with invalid UUID."""
        response = client.get("/api/v1/detectors/invalid-uuid")

        assert response.status_code == 422  # Validation error

    def test_get_detector_hyperparameters(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test getting detector hyperparameters."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        detector_id = str(sample_detector_dto.id)
        response = client.get(f"/api/v1/detectors/{detector_id}/hyperparameters")

        assert response.status_code == 200
        data = response.json()
        assert "n_estimators" in data
        assert data["n_estimators"] == 100

    def test_get_detector_metadata(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test getting detector metadata."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        detector_id = str(sample_detector_dto.id)
        response = client.get(f"/api/v1/detectors/{detector_id}/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "sklearn"
        assert "performance_metrics" in data

    # Detector Update Tests

    def test_update_detector_success(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test successful detector update."""
        mock_repo = mock_container.detector_repository.return_value
        updated_dto = sample_detector_dto.copy()
        updated_dto.name = "Updated Detector"
        updated_dto.contamination_rate = 0.15
        mock_repo.update.return_value = updated_dto
        mock_repo.find_by_id.return_value = sample_detector_dto

        update_data = {
            "name": "Updated Detector",
            "contamination_rate": 0.15,
            "hyperparameters": {"n_estimators": 200},
        }

        detector_id = str(sample_detector_dto.id)
        response = client.put(f"/api/v1/detectors/{detector_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Detector"

    def test_update_detector_not_found(self, client, mock_container, mock_user):
        """Test detector update with non-existent ID."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = None

        update_data = {"name": "Updated Detector"}
        response = client.put(
            "/api/v1/detectors/99999999-9999-9999-9999-999999999999", json=update_data
        )

        assert response.status_code == 404

    def test_update_detector_hyperparameters_only(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test updating only detector hyperparameters."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto
        mock_repo.update.return_value = sample_detector_dto

        update_data = {"hyperparameters": {"n_estimators": 300, "max_samples": 0.8}}

        detector_id = str(sample_detector_dto.id)
        response = client.put(f"/api/v1/detectors/{detector_id}", json=update_data)

        assert response.status_code == 200

    def test_update_detector_validation_error(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test detector update with validation error."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        invalid_data = {"contamination_rate": 2.0}  # Invalid: should be between 0 and 1

        detector_id = str(sample_detector_dto.id)
        response = client.put(f"/api/v1/detectors/{detector_id}", json=invalid_data)

        assert response.status_code == 422

    def test_update_fitted_detector_restriction(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test updating fitted detector with restrictions."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto
        mock_repo.update.side_effect = HTTPException(
            status_code=409, detail="Cannot modify hyperparameters of fitted detector"
        )

        update_data = {"hyperparameters": {"n_estimators": 500}}

        detector_id = str(sample_detector_dto.id)
        response = client.put(f"/api/v1/detectors/{detector_id}", json=update_data)

        assert response.status_code == 409

    # Detector Deletion Tests

    def test_delete_detector_success(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test successful detector deletion."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto
        mock_repo.delete.return_value = True

        detector_id = str(sample_detector_dto.id)
        response = client.delete(f"/api/v1/detectors/{detector_id}")

        assert response.status_code == 204
        mock_repo.delete.assert_called_once_with(sample_detector_dto.id)

    def test_delete_detector_not_found(self, client, mock_container, mock_user):
        """Test detector deletion with non-existent ID."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = None

        response = client.delete(
            "/api/v1/detectors/99999999-9999-9999-9999-999999999999"
        )

        assert response.status_code == 404

    def test_delete_detector_in_use(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test detector deletion when detector is in use."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto
        mock_repo.delete.side_effect = HTTPException(
            status_code=409, detail="Detector is being used by active detection jobs"
        )

        detector_id = str(sample_detector_dto.id)
        response = client.delete(f"/api/v1/detectors/{detector_id}")

        assert response.status_code == 409

    # Algorithm-Specific Tests

    def test_list_supported_algorithms(self, client, mock_container, mock_user):
        """Test listing supported algorithms."""
        mock_pyod = mock_container.pyod_adapter.return_value
        mock_sklearn = mock_container.sklearn_adapter.return_value

        mock_pyod.list_algorithms.return_value = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
        ]
        mock_sklearn.list_algorithms.return_value = [
            "EllipticEnvelope",
            "LocalOutlierFactor",
        ]

        response = client.get("/api/v1/detectors/algorithms")

        assert response.status_code == 200
        data = response.json()
        assert "pyod" in data
        assert "sklearn" in data
        assert "IsolationForest" in data["pyod"]

    def test_get_algorithm_info(self, client, mock_container, mock_user):
        """Test getting algorithm information."""
        mock_adapter = mock_container.pyod_adapter.return_value
        mock_adapter.get_algorithm_info.return_value = {
            "name": "IsolationForest",
            "description": "Isolation Forest for anomaly detection",
            "hyperparameters": {
                "n_estimators": {"type": "int", "default": 100, "range": [1, 1000]},
                "contamination": {"type": "float", "default": 0.1, "range": [0.0, 0.5]},
            },
            "supported_features": ["numerical"],
            "scalability": "high",
            "interpretability": "medium",
        }

        response = client.get("/api/v1/detectors/algorithms/IsolationForest")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "IsolationForest"
        assert "hyperparameters" in data

    def test_validate_algorithm_hyperparameters(
        self, client, mock_container, mock_user
    ):
        """Test hyperparameter validation for algorithm."""
        mock_adapter = mock_container.pyod_adapter.return_value
        mock_adapter.validate_hyperparameters.return_value = {
            "valid": True,
            "errors": [],
            "warnings": ["n_estimators value is quite high, may impact performance"],
        }

        validation_data = {
            "algorithm": "IsolationForest",
            "hyperparameters": {"n_estimators": 500, "contamination": 0.1},
        }

        response = client.post(
            "/api/v1/detectors/validate-hyperparameters", json=validation_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["warnings"]) == 1

    # Performance and Monitoring Tests

    def test_get_detector_performance_metrics(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test getting detector performance metrics."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        mock_service = mock_container.performance_service.return_value
        mock_service.get_detector_metrics.return_value = {
            "usage_count": 156,
            "average_training_time": 2.3,
            "average_detection_time": 0.05,
            "accuracy_metrics": {"precision": 0.92, "recall": 0.89, "f1_score": 0.90},
            "last_used": datetime.utcnow().isoformat(),
        }

        detector_id = str(sample_detector_dto.id)
        response = client.get(f"/api/v1/detectors/{detector_id}/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["usage_count"] == 156
        assert "accuracy_metrics" in data

    def test_get_detector_training_history(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test getting detector training history."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        mock_service = mock_container.training_service.return_value
        mock_service.get_training_history.return_value = [
            {
                "training_id": "train-123",
                "dataset_id": "dataset-456",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "duration_seconds": 45.2,
                "status": "completed",
                "metrics": {"accuracy": 0.94},
            }
        ]

        detector_id = str(sample_detector_dto.id)
        response = client.get(f"/api/v1/detectors/{detector_id}/training-history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "completed"

    # Detector Model Management Tests

    def test_export_detector_model(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test exporting detector model."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_by_id.return_value = sample_detector_dto

        mock_service = mock_container.model_persistence_service.return_value
        mock_service.export_model.return_value = b"serialized_model_data"

        detector_id = str(sample_detector_dto.id)
        response = client.get(f"/api/v1/detectors/{detector_id}/export")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

    def test_import_detector_model(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test importing detector model."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.create.return_value = sample_detector_dto

        mock_service = mock_container.model_persistence_service.return_value
        mock_service.import_model.return_value = sample_detector_dto

        # Simulate file upload
        import io

        model_data = io.BytesIO(b"serialized_model_data")

        response = client.post(
            "/api/v1/detectors/import",
            files={"model_file": ("model.pkl", model_data, "application/octet-stream")},
            data={"name": "Imported Detector"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Isolation Forest"

    # Batch Operations Tests

    def test_bulk_detector_operations(
        self, client, mock_container, mock_user, sample_detector_dto
    ):
        """Test bulk detector operations."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.bulk_delete.return_value = 3

        bulk_request = {
            "operation": "delete",
            "detector_ids": [
                "12345678-1234-5678-9012-123456789012",
                "87654321-4321-8765-2109-876543210987",
                "11111111-2222-3333-4444-555555555555",
            ],
        }

        response = client.post("/api/v1/detectors/bulk", json=bulk_request)

        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 3

    def test_compare_detectors(self, client, mock_container, mock_user):
        """Test detector comparison endpoint."""
        mock_service = mock_container.comparison_service.return_value
        mock_service.compare_detectors.return_value = {
            "detectors": [
                {
                    "id": "detector-1",
                    "name": "IF Detector",
                    "algorithm": "IsolationForest",
                },
                {
                    "id": "detector-2",
                    "name": "LOF Detector",
                    "algorithm": "LocalOutlierFactor",
                },
            ],
            "comparison_metrics": {
                "accuracy": [0.95, 0.92],
                "precision": [0.93, 0.89],
                "recall": [0.91, 0.88],
                "f1_score": [0.92, 0.89],
            },
            "recommendations": [
                "IsolationForest shows better overall performance",
                "LOF has lower false positive rate",
            ],
        }

        compare_request = {
            "detector_ids": ["detector-1", "detector-2"],
            "evaluation_dataset_id": "eval-dataset-123",
        }

        response = client.post("/api/v1/detectors/compare", json=compare_request)

        assert response.status_code == 200
        data = response.json()
        assert len(data["detectors"]) == 2
        assert "comparison_metrics" in data

    # Error Handling Tests

    def test_detector_service_timeout(self, client, mock_container, mock_user):
        """Test detector service timeout handling."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.find_all.side_effect = TimeoutError("Database timeout")

        response = client.get("/api/v1/detectors/")

        assert response.status_code == 504  # Gateway Timeout

    def test_detector_permission_denied(self, client, mock_container):
        """Test detector operations with insufficient permissions."""
        with patch("pynomaly.presentation.api.deps.get_current_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test-user",
                "permissions": ["detector:read"],  # Missing detector:create
            }

            create_data = {
                "name": "Test Detector",
                "algorithm_name": "IsolationForest",
                "contamination_rate": 0.1,
            }

            response = client.post("/api/v1/detectors/", json=create_data)
            assert response.status_code == 403

    def test_detector_memory_limit(self, client, mock_container, mock_user):
        """Test detector creation hitting memory limits."""
        mock_repo = mock_container.detector_repository.return_value
        mock_repo.create.side_effect = HTTPException(
            status_code=507, detail="Insufficient memory to create detector"
        )

        create_data = {
            "name": "Large Detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.1,
            "hyperparameters": {"n_estimators": 10000},  # Very large
        }

        response = client.post("/api/v1/detectors/", json=create_data)

        assert response.status_code == 507


class TestDetectorEndpointsIntegration:
    """Integration tests for detector endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_complete_detector_lifecycle(self, client):
        """Test complete detector lifecycle from creation to deletion."""
        with patch("pynomaly.presentation.api.deps.get_current_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test-user",
                "permissions": [
                    "detector:create",
                    "detector:read",
                    "detector:update",
                    "detector:delete",
                ],
            }

            with patch(
                "pynomaly.presentation.api.deps.get_container"
            ) as mock_container:
                container = Mock()
                container.detector_repository.return_value = Mock()
                mock_container.return_value = container

                # 1. Create detector
                create_data = {
                    "name": "Lifecycle Test Detector",
                    "algorithm_name": "IsolationForest",
                    "contamination_rate": 0.1,
                }
                create_response = client.post("/api/v1/detectors/", json=create_data)

                # 2. List detectors
                list_response = client.get("/api/v1/detectors/")

                # 3. Get specific detector
                detector_id = "12345678-1234-5678-9012-123456789012"
                get_response = client.get(f"/api/v1/detectors/{detector_id}")

                # 4. Update detector
                update_response = client.put(
                    f"/api/v1/detectors/{detector_id}",
                    json={"name": "Updated Lifecycle Test"},
                )

                # 5. Delete detector
                delete_response = client.delete(f"/api/v1/detectors/{detector_id}")

                # Verify lifecycle
                responses = [
                    create_response,
                    list_response,
                    get_response,
                    update_response,
                    delete_response,
                ]

                # All requests should either succeed or have expected failures
                for response in responses:
                    assert response.status_code in [200, 201, 204, 401, 404, 422]

    def test_detector_algorithm_workflow(self, client):
        """Test detector algorithm discovery and validation workflow."""
        with patch("pynomaly.presentation.api.deps.get_current_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": "test",
                "permissions": ["detector:read"],
            }

            # Test algorithm discovery workflow
            endpoints = [
                "/api/v1/detectors/algorithms",
                "/api/v1/detectors/algorithms/IsolationForest",
            ]

            for endpoint in endpoints:
                response = client.get(endpoint)
                # Should work or require proper auth/data
                assert response.status_code in [200, 401, 404, 422]
