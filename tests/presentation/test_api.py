"""Tests for REST API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    container = create_container()
    app = create_app(container)
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client."""
    container = create_container()
    app = create_app(container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client: TestClient):
        """Test health endpoint."""
        response = client.get("/api/health/")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "uptime_seconds" in data

    def test_readiness_check(self, client: TestClient):
        """Test readiness endpoint."""
        response = client.get("/api/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "timestamp" in data


class TestDetectorEndpoints:
    """Test detector management endpoints."""

    def test_create_detector(self, client: TestClient):
        """Test creating a detector."""
        detector_data = {
            "name": "Test Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        response = client.post("/api/detectors/", json=detector_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Detector"
        assert data["algorithm_name"] == "IsolationForest"
        assert "id" in data
        assert not data["is_fitted"]

    def test_list_detectors(self, client: TestClient):
        """Test listing detectors."""
        # Create a detector first
        detector_data = {"name": "List Test", "algorithm_name": "LOF"}
        client.post("/api/detectors/", json=detector_data)

        # List detectors
        response = client.get("/api/detectors/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(d["name"] == "List Test" for d in data)

    def test_get_detector(self, client: TestClient):
        """Test getting specific detector."""
        # Create detector
        create_response = client.post(
            "/api/detectors/", json={"name": "Get Test", "algorithm_name": "OCSVM"}
        )
        detector_id = create_response.json()["id"]

        # Get detector
        response = client.get(f"/api/detectors/{detector_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == detector_id
        assert data["name"] == "Get Test"

    def test_update_detector(self, client: TestClient):
        """Test updating detector."""
        # Create detector
        create_response = client.post(
            "/api/detectors/",
            json={"name": "Update Test", "algorithm_name": "IsolationForest"},
        )
        detector_id = create_response.json()["id"]

        # Update detector
        update_data = {
            "name": "Updated Name",
            "description": "Updated description",
            "parameters": {"contamination": 0.2},
        }
        response = client.put(f"/api/detectors/{detector_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated description"

    def test_delete_detector(self, client: TestClient):
        """Test deleting detector."""
        # Create detector
        create_response = client.post(
            "/api/detectors/", json={"name": "Delete Test", "algorithm_name": "LOF"}
        )
        detector_id = create_response.json()["id"]

        # Delete detector
        response = client.delete(f"/api/detectors/{detector_id}")

        assert response.status_code == 200
        assert response.json()["success"] is True

        # Verify deletion
        get_response = client.get(f"/api/detectors/{detector_id}")
        assert get_response.status_code == 404


class TestDatasetEndpoints:
    """Test dataset management endpoints."""

    def test_upload_dataset_csv(self, client: TestClient):
        """Test uploading CSV dataset."""
        import io

        csv_content = "feature1,feature2,target\n1,2,0\n3,4,0\n5,6,1\n"
        csv_file = io.BytesIO(csv_content.encode())

        files = {"file": ("test.csv", csv_file, "text/csv")}
        data = {"name": "Test CSV Dataset", "target_column": "target"}

        response = client.post("/api/datasets/upload", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "Test CSV Dataset"
        assert result["n_samples"] == 3
        assert result["n_features"] == 2
        assert result["has_target"] is True

    def test_get_dataset_sample(self, client: TestClient):
        """Test getting dataset sample."""
        # Upload dataset first
        import io

        csv_content = "a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(20))
        csv_file = io.BytesIO(csv_content.encode())

        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("sample.csv", csv_file, "text/csv")},
            data={"name": "Sample Dataset"},
        )
        dataset_id = upload_response.json()["id"]

        # Get sample
        response = client.get(f"/api/datasets/{dataset_id}/sample?n=5")

        assert response.status_code == 200
        data = response.json()
        assert data["sample_size"] == 5
        assert len(data["data"]) == 5
        assert data["columns"] == ["a", "b"]

    def test_dataset_quality_check(self, client: TestClient):
        """Test dataset quality check."""
        # Upload dataset with quality issues
        import io

        csv_content = (
            "x,y,z\n1,2,3\n4,,6\n7,8,9\n1,2,3\n"  # Missing value and duplicate
        )
        csv_file = io.BytesIO(csv_content.encode())

        upload_response = client.post(
            "/api/datasets/upload",
            files={"file": ("quality.csv", csv_file, "text/csv")},
            data={"name": "Quality Test"},
        )
        dataset_id = upload_response.json()["id"]

        # Check quality
        response = client.get(f"/api/datasets/{dataset_id}/quality")

        assert response.status_code == 200
        data = response.json()
        assert data["missing_values"]["y"] > 0
        assert data["duplicate_rows"] == 1
        assert len(data["suggestions"]) > 0


class TestDetectionEndpoints:
    """Test detection endpoints."""

    @pytest.mark.asyncio
    async def test_train_detector(self, async_client: AsyncClient):
        """Test training a detector."""
        # Create detector and dataset
        detector_response = await async_client.post(
            "/api/detectors/",
            json={"name": "Train Test", "algorithm_name": "IsolationForest"},
        )
        detector_id = detector_response.json()["id"]

        # Upload dataset
        import io

        csv_content = "x,y\n" + "\n".join(f"{i},{i * 1.5}" for i in range(100))
        csv_file = io.BytesIO(csv_content.encode())

        dataset_response = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("train.csv", csv_file, "text/csv")},
            data={"name": "Training Data"},
        )
        dataset_id = dataset_response.json()["id"]

        # Train detector
        train_response = await async_client.post(
            "/api/detection/train",
            json={
                "detector_id": detector_id,
                "dataset_id": dataset_id,
                "validate_data": True,
                "save_model": True,
            },
        )

        assert train_response.status_code == 200
        data = train_response.json()
        assert data["success"] is True
        assert data["training_time_ms"] > 0
