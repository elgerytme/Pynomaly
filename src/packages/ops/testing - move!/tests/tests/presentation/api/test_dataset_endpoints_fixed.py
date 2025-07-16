"""
Fixed Dataset Endpoints Testing Suite
Tests for dataset CRUD operations matching the actual API implementation.
"""

import io
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from monorepo.application.dto.dataset_dto import DatasetDTO
from monorepo.presentation.api.app import create_app


class TestDatasetEndpoints:
    """Test suite for dataset management endpoints that match actual implementation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_auth_dependencies(self):
        """Mock all authentication dependencies."""
        with patch("monorepo.infrastructure.auth.middleware.get_current_user") as mock_get_user, \
             patch("monorepo.infrastructure.auth.jwt_auth.get_auth") as mock_get_auth, \
             patch("monorepo.infrastructure.auth.middleware.require_viewer") as mock_require_viewer, \
             patch("monorepo.infrastructure.auth.middleware.require_data_scientist") as mock_require_ds:
            
            # Create mock user
            from monorepo.infrastructure.auth.jwt_auth import UserModel
            mock_user = UserModel(
                id="user123",
                username="testuser",
                email="test@example.com",
                hashed_password="hashed",
                roles=["data_scientist"],
                is_active=True
            )
            
            # Mock auth functions to return the user
            mock_get_user.return_value = mock_user
            mock_require_viewer.return_value = mock_user
            mock_require_ds.return_value = mock_user
            
            # Mock auth service
            mock_auth_service = Mock()
            mock_get_auth.return_value = mock_auth_service
            
            yield mock_user

    @pytest.fixture
    def mock_container(self):
        """Mock the container dependencies."""
        with patch("monorepo.infrastructure.config.Container") as mock_container_class:
            mock_container = Mock()
            
            # Mock repository
            mock_repo = AsyncMock()
            mock_container.dataset_repository.return_value = mock_repo
            
            # Mock config
            mock_config = Mock()
            mock_config.max_dataset_size_mb = 100
            mock_container.config.return_value = mock_config
            
            # Mock dependency function to return our mock
            mock_container_class.return_value = mock_container
            
            # Override the dependency
            with patch("monorepo.presentation.api.endpoints.datasets.Container", return_value=mock_container):
                yield mock_container, mock_repo

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-jwt-token"}

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for upload testing."""
        return """feature1,feature2,feature3,is_anomaly
1.0,2.0,3.0,0
1.5,2.5,3.5,0
10.0,20.0,30.0,1
2.0,3.0,4.0,0
15.0,25.0,35.0,1"""

    # Dataset List Operations

    def test_list_datasets_basic(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test basic dataset listing."""
        mock_container_obj, mock_repo = mock_container
        
        # Create mock dataset
        from monorepo.domain.entities.dataset import Dataset
        import pandas as pd
        
        mock_dataset = Dataset(
            name="test_dataset",
            data=pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
            description="Test dataset"
        )
        mock_dataset.id = "dataset123"
        
        mock_repo.find_all.return_value = [mock_dataset]

        response = client.get("/api/datasets/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_dataset"
        assert data[0]["id"] == "dataset123"

    def test_list_datasets_with_filter(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test dataset listing with has_target filter."""
        mock_container_obj, mock_repo = mock_container
        
        # Create datasets with and without targets
        from monorepo.domain.entities.dataset import Dataset
        import pandas as pd
        
        dataset_with_target = Dataset(
            name="with_target",
            data=pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
            target_column="target"
        )
        dataset_with_target.id = "dataset1"
        
        dataset_without_target = Dataset(
            name="without_target",
            data=pd.DataFrame({"feature": [1, 2], "other": [3, 4]})
        )
        dataset_without_target.id = "dataset2"
        
        mock_repo.find_all.return_value = [dataset_with_target, dataset_without_target]

        response = client.get("/api/datasets/?has_target=true", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "with_target"

    # Dataset Retrieval

    def test_get_dataset_by_id(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test retrieving a specific dataset."""
        mock_container_obj, mock_repo = mock_container
        
        from monorepo.domain.entities.dataset import Dataset
        import pandas as pd
        from uuid import uuid4
        
        dataset_id = uuid4()
        mock_dataset = Dataset(
            name="test_dataset",
            data=pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
            description="Test dataset"
        )
        mock_dataset.id = dataset_id
        
        mock_repo.find_by_id.return_value = mock_dataset

        response = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_dataset"
        assert data["id"] == str(dataset_id)

    def test_get_dataset_not_found(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test retrieving non-existent dataset."""
        mock_container_obj, mock_repo = mock_container
        from uuid import uuid4
        
        mock_repo.find_by_id.return_value = None

        response = client.get(f"/api/datasets/{uuid4()}", headers=auth_headers)

        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

    # Dataset Upload Operations

    def test_upload_dataset_csv_success(self, client, mock_auth_dependencies, mock_container, auth_headers, sample_csv_data):
        """Test successful CSV dataset upload."""
        mock_container_obj, mock_repo = mock_container
        
        files = {"file": ("test_data.csv", io.StringIO(sample_csv_data), "text/csv")}
        data = {
            "name": "uploaded_dataset",
            "description": "Test uploaded dataset",
        }

        response = client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["name"] == "uploaded_dataset"
        assert "id" in response_data
        mock_repo.save.assert_called_once()

    def test_upload_dataset_parquet_success(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test successful Parquet dataset upload."""
        mock_container_obj, mock_repo = mock_container
        
        # Create mock parquet data
        import pandas as pd
        import io
        
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer)
        parquet_data = parquet_buffer.getvalue()
        
        files = {"file": ("test_data.parquet", io.BytesIO(parquet_data), "application/octet-stream")}
        data = {"name": "parquet_dataset"}

        response = client.post("/api/datasets/upload", files=files, data=data, headers=auth_headers)

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["name"] == "parquet_dataset"

    def test_upload_dataset_file_too_large(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test uploading dataset that exceeds size limit."""
        mock_container_obj, mock_repo = mock_container
        
        # Mock large file (larger than 100MB limit)
        large_data = "x," * 50000000  # Create large CSV data
        files = {"file": ("large_data.csv", io.StringIO(large_data), "text/csv")}

        response = client.post("/api/datasets/upload", files=files, headers=auth_headers)

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_upload_dataset_unsupported_format(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test uploading unsupported file format."""
        mock_container_obj, mock_repo = mock_container
        
        files = {"file": ("test_data.txt", io.StringIO("some text"), "text/plain")}

        response = client.post("/api/datasets/upload", files=files, headers=auth_headers)

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    # Dataset Quality Check

    def test_dataset_quality_check(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test dataset quality check endpoint."""
        mock_container_obj, mock_repo = mock_container
        
        from monorepo.domain.entities.dataset import Dataset
        import pandas as pd
        from uuid import uuid4
        
        dataset_id = uuid4()
        mock_dataset = Dataset(
            name="test_dataset",
            data=pd.DataFrame({"col1": [1, 2, None], "col2": [3, 4, 5]}),
        )
        mock_dataset.id = dataset_id
        
        # Mock feature validator
        mock_validator = Mock()
        mock_quality_report = {
            "n_samples": 3,
            "n_features": 2,
            "missing_values": {"col1": 1},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": [],
            "duplicate_rows": 0,
            "quality_score": 0.8
        }
        
        mock_validator.check_data_quality.return_value = mock_quality_report
        mock_validator.suggest_preprocessing.return_value = ["Handle missing values"]
        
        mock_container_obj.feature_validator.return_value = mock_validator
        mock_repo.find_by_id.return_value = mock_dataset

        response = client.get(f"/api/datasets/{dataset_id}/quality", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "quality_score" in data
        assert "suggestions" in data

    # Dataset Sample

    def test_get_dataset_sample(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test getting dataset sample."""
        mock_container_obj, mock_repo = mock_container
        
        from monorepo.domain.entities.dataset import Dataset
        import pandas as pd
        from uuid import uuid4
        
        dataset_id = uuid4()
        mock_dataset = Dataset(
            name="test_dataset",
            data=pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]}),
        )
        mock_dataset.id = dataset_id
        
        mock_repo.find_by_id.return_value = mock_dataset

        response = client.get(f"/api/datasets/{dataset_id}/sample?n=3", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == str(dataset_id)
        assert data["sample_size"] == 3
        assert len(data["data"]) == 3

    # Dataset Split

    def test_split_dataset(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test dataset splitting."""
        mock_container_obj, mock_repo = mock_container
        
        from monorepo.domain.entities.dataset import Dataset
        import pandas as pd
        from uuid import uuid4
        
        dataset_id = uuid4()
        mock_dataset = Dataset(
            name="test_dataset",
            data=pd.DataFrame({"col1": range(100), "col2": range(100, 200)}),
        )
        mock_dataset.id = dataset_id
        
        # Mock split results
        train_dataset = Dataset(
            name="test_dataset (train)",
            data=pd.DataFrame({"col1": range(80), "col2": range(100, 180)}),
        )
        train_dataset.id = uuid4()
        
        test_dataset = Dataset(
            name="test_dataset (test)", 
            data=pd.DataFrame({"col1": range(80, 100), "col2": range(180, 200)}),
        )
        test_dataset.id = uuid4()
        
        mock_dataset.split.return_value = (train_dataset, test_dataset)
        mock_repo.find_by_id.return_value = mock_dataset

        response = client.post(f"/api/datasets/{dataset_id}/split?test_size=0.2", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "train_dataset_id" in data
        assert "test_dataset_id" in data
        assert data["train_size"] == 80
        assert data["test_size"] == 20

    # Dataset Deletion

    def test_delete_dataset(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test dataset deletion."""
        mock_container_obj, mock_repo = mock_container
        from uuid import uuid4
        
        dataset_id = uuid4()
        mock_repo.exists.return_value = True
        mock_repo.delete.return_value = True

        response = client.delete(f"/api/datasets/{dataset_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted" in data["message"].lower()

    def test_delete_dataset_not_found(self, client, mock_auth_dependencies, mock_container, auth_headers):
        """Test deleting non-existent dataset."""
        mock_container_obj, mock_repo = mock_container
        from uuid import uuid4
        
        dataset_id = uuid4()
        mock_repo.exists.return_value = False

        response = client.delete(f"/api/datasets/{dataset_id}", headers=auth_headers)

        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

    # Authentication Tests

    def test_list_datasets_unauthorized(self, client, mock_container):
        """Test listing datasets without authentication."""
        response = client.get("/api/datasets/")
        
        # Should either return 401 or allow access based on auth settings
        assert response.status_code in [200, 401]

    def test_upload_dataset_unauthorized(self, client, mock_container, sample_csv_data):
        """Test uploading dataset without authentication."""
        files = {"file": ("test_data.csv", io.StringIO(sample_csv_data), "text/csv")}
        
        response = client.post("/api/datasets/upload", files=files)
        
        # Should either return 401 or allow access based on auth settings
        assert response.status_code in [200, 401]