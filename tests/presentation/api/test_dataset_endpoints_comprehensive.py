"""
Comprehensive tests for dataset endpoints.
Tests dataset upload, management, and validation API endpoints.
"""

import json
import pytest
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi import status

from pynomaly.presentation.web_api.app import app
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.exceptions import DatasetError, ValidationError


class TestDatasetEndpointsComprehensive:
    """Comprehensive test suite for dataset API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_dataset_service(self):
        """Mock dataset service."""
        service = AsyncMock()
        service.upload_dataset.return_value = {
            "dataset_id": str(uuid4()),
            "name": "test-dataset",
            "size": 1024,
            "rows": 1000,
            "columns": 5,
            "file_path": "/tmp/test-dataset.csv",
            "created_at": datetime.utcnow(),
        }
        service.get_dataset.return_value = Dataset(
            id=uuid4(),
            name="test-dataset",
            file_path="/tmp/test-dataset.csv",
            features=["feature1", "feature2", "feature3"],
            feature_types={"feature1": "numeric", "feature2": "numeric", "feature3": "categorical"},
            target_column=None,
            data_shape=(1000, 3),
        )
        service.validate_dataset.return_value = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "rows": 1000,
                "columns": 3,
                "missing_values": 0,
                "duplicates": 0,
                "data_types": {"numeric": 2, "categorical": 1},
            },
        }
        service.get_dataset_preview.return_value = {
            "columns": ["feature1", "feature2", "feature3"],
            "data": [
                [1.0, 2.0, "A"],
                [1.5, 2.5, "B"],
                [2.0, 3.0, "A"],
            ],
            "total_rows": 1000,
            "preview_rows": 3,
        }
        service.get_dataset_statistics.return_value = {
            "shape": (1000, 3),
            "column_types": {"feature1": "float64", "feature2": "float64", "feature3": "object"},
            "missing_values": {"feature1": 0, "feature2": 0, "feature3": 0},
            "summary_stats": {
                "feature1": {"mean": 1.5, "std": 0.5, "min": 1.0, "max": 2.0},
                "feature2": {"mean": 2.5, "std": 0.5, "min": 2.0, "max": 3.0},
            },
        }
        return service

    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        service = AsyncMock()
        service.get_current_user.return_value = {
            "id": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
        }
        return service

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test_token_123"}

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing."""
        return "feature1,feature2,feature3\n1.0,2.0,A\n1.5,2.5,B\n2.0,3.0,A\n"

    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data for testing."""
        return {
            "data": [
                {"feature1": 1.0, "feature2": 2.0, "feature3": "A"},
                {"feature1": 1.5, "feature2": 2.5, "feature3": "B"},
                {"feature1": 2.0, "feature2": 3.0, "feature3": "A"},
            ]
        }

    @pytest.fixture
    def valid_dataset_metadata(self):
        """Valid dataset metadata for testing."""
        return {
            "name": "test-dataset",
            "description": "Test dataset for anomaly detection",
            "features": ["feature1", "feature2", "feature3"],
            "feature_types": {
                "feature1": "numeric",
                "feature2": "numeric",
                "feature3": "categorical",
            },
            "target_column": None,
            "tags": ["test", "anomaly-detection"],
        }

    def test_upload_dataset_csv_success(
        self, client, sample_csv_data, valid_dataset_metadata, mock_dataset_service, auth_headers
    ):
        """Test successful CSV dataset upload."""
        # Create file upload
        files = {
            "file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")
        }
        data = {
            "metadata": json.dumps(valid_dataset_metadata)
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                "/api/v1/datasets/upload",
                files=files,
                data=data,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "dataset_id" in data
        assert "name" in data
        assert "size" in data
        assert "rows" in data
        assert "columns" in data
        assert "file_path" in data

    def test_upload_dataset_json_success(
        self, client, sample_json_data, valid_dataset_metadata, mock_dataset_service, auth_headers
    ):
        """Test successful JSON dataset upload."""
        # Create file upload
        files = {
            "file": ("test.json", BytesIO(json.dumps(sample_json_data).encode()), "application/json")
        }
        data = {
            "metadata": json.dumps(valid_dataset_metadata)
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                "/api/v1/datasets/upload",
                files=files,
                data=data,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        result = response.json()
        assert "dataset_id" in result

    def test_upload_dataset_invalid_format(
        self, client, valid_dataset_metadata, auth_headers
    ):
        """Test dataset upload with invalid file format."""
        # Create invalid file
        files = {
            "file": ("test.txt", BytesIO(b"invalid content"), "text/plain")
        }
        data = {
            "metadata": json.dumps(valid_dataset_metadata)
        }

        response = client.post(
            "/api/v1/datasets/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        result = response.json()
        assert "error" in result
        assert "unsupported format" in result["error"].lower()

    def test_upload_dataset_missing_file(
        self, client, valid_dataset_metadata, auth_headers
    ):
        """Test dataset upload with missing file."""
        data = {
            "metadata": json.dumps(valid_dataset_metadata)
        }

        response = client.post(
            "/api/v1/datasets/upload",
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_dataset_invalid_metadata(
        self, client, sample_csv_data, auth_headers
    ):
        """Test dataset upload with invalid metadata."""
        # Create file upload
        files = {
            "file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")
        }
        data = {
            "metadata": "invalid json"
        }

        response = client.post(
            "/api/v1/datasets/upload",
            files=files,
            data=data,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_dataset_unauthorized(
        self, client, sample_csv_data, valid_dataset_metadata
    ):
        """Test dataset upload without authentication."""
        files = {
            "file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")
        }
        data = {
            "metadata": json.dumps(valid_dataset_metadata)
        }

        response = client.post(
            "/api/v1/datasets/upload",
            files=files,
            data=data,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_upload_dataset_large_file(
        self, client, valid_dataset_metadata, mock_dataset_service, auth_headers
    ):
        """Test upload of large dataset file."""
        # Create large CSV data (simulated)
        large_csv_data = "feature1,feature2\n" + "\n".join(["1.0,2.0"] * 10000)
        
        files = {
            "file": ("large.csv", BytesIO(large_csv_data.encode()), "text/csv")
        }
        data = {
            "metadata": json.dumps(valid_dataset_metadata)
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                "/api/v1/datasets/upload",
                files=files,
                data=data,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED

    def test_get_dataset_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset retrieval."""
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "features" in data
        assert "feature_types" in data
        assert "data_shape" in data

    def test_get_dataset_not_found(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset retrieval with non-existent ID."""
        mock_dataset_service.get_dataset.side_effect = DatasetError("Dataset not found")
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_dataset_invalid_id(self, client, auth_headers):
        """Test dataset retrieval with invalid ID format."""
        invalid_id = "invalid-uuid"
        
        response = client.get(
            f"/api/v1/datasets/{invalid_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_datasets_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset listing."""
        mock_datasets = [
            Dataset(
                id=uuid4(),
                name="dataset-1",
                file_path="/tmp/dataset1.csv",
                features=["feature1", "feature2"],
                feature_types={"feature1": "numeric", "feature2": "numeric"},
                data_shape=(1000, 2),
            ),
            Dataset(
                id=uuid4(),
                name="dataset-2",
                file_path="/tmp/dataset2.csv",
                features=["feature1", "feature2", "feature3"],
                feature_types={"feature1": "numeric", "feature2": "numeric", "feature3": "categorical"},
                data_shape=(2000, 3),
            ),
        ]
        mock_dataset_service.list_datasets.return_value = mock_datasets

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get("/api/v1/datasets", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "datasets" in data
        assert len(data["datasets"]) == 2
        assert data["datasets"][0]["name"] == "dataset-1"
        assert data["datasets"][1]["name"] == "dataset-2"

    def test_list_datasets_with_pagination(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset listing with pagination."""
        mock_dataset_service.list_datasets.return_value = []

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                "/api/v1/datasets?page=1&size=10",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "datasets" in data
        assert "pagination" in data

    def test_list_datasets_with_filters(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset listing with filters."""
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                "/api/v1/datasets?name=test&format=csv&size_min=1000",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "datasets" in data

    def test_validate_dataset_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset validation."""
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                f"/api/v1/datasets/{dataset_id}/validate",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "is_valid" in data
        assert "errors" in data
        assert "warnings" in data
        assert "statistics" in data
        assert data["is_valid"] is True

    def test_validate_dataset_with_errors(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset validation with errors."""
        dataset_id = str(uuid4())
        
        # Mock validation with errors
        mock_dataset_service.validate_dataset.return_value = {
            "is_valid": False,
            "errors": [
                "Missing values in feature1",
                "Invalid data type in feature2",
            ],
            "warnings": [
                "High correlation between feature1 and feature2",
            ],
            "statistics": {
                "rows": 1000,
                "columns": 3,
                "missing_values": 50,
                "duplicates": 10,
            },
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                f"/api/v1/datasets/{dataset_id}/validate",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) == 2
        assert len(data["warnings"]) == 1

    def test_validate_dataset_not_found(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset validation with non-existent ID."""
        mock_dataset_service.validate_dataset.side_effect = DatasetError("Dataset not found")
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                f"/api/v1/datasets/{dataset_id}/validate",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_dataset_preview_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset preview retrieval."""
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}/preview",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "columns" in data
        assert "data" in data
        assert "total_rows" in data
        assert "preview_rows" in data
        assert len(data["data"]) == 3
        assert len(data["columns"]) == 3

    def test_get_dataset_preview_with_limit(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset preview with custom limit."""
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}/preview?limit=10",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "data" in data

    def test_get_dataset_statistics_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset statistics retrieval."""
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}/statistics",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "shape" in data
        assert "column_types" in data
        assert "missing_values" in data
        assert "summary_stats" in data

    def test_update_dataset_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset update."""
        dataset_id = str(uuid4())
        update_payload = {
            "name": "updated-dataset",
            "description": "Updated description",
            "tags": ["updated", "test"],
        }

        mock_updated_dataset = Dataset(
            id=dataset_id,
            name=update_payload["name"],
            file_path="/tmp/updated-dataset.csv",
            features=["feature1", "feature2"],
            feature_types={"feature1": "numeric", "feature2": "numeric"},
            data_shape=(1000, 2),
        )
        mock_dataset_service.update_dataset.return_value = mock_updated_dataset

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.put(
                f"/api/v1/datasets/{dataset_id}",
                json=update_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_payload["name"]

    def test_update_dataset_not_found(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset update with non-existent ID."""
        mock_dataset_service.update_dataset.side_effect = DatasetError("Dataset not found")
        dataset_id = str(uuid4())
        update_payload = {"name": "updated-dataset"}

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.put(
                f"/api/v1/datasets/{dataset_id}",
                json=update_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_dataset_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset deletion."""
        dataset_id = str(uuid4())
        mock_dataset_service.delete_dataset.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.delete(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_dataset_not_found(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test dataset deletion with non-existent ID."""
        mock_dataset_service.delete_dataset.side_effect = DatasetError("Dataset not found")
        dataset_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.delete(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_export_dataset_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset export."""
        dataset_id = str(uuid4())
        export_payload = {
            "format": "csv",
            "include_headers": True,
            "columns": ["feature1", "feature2"],
        }

        mock_dataset_service.export_dataset.return_value = {
            "export_id": str(uuid4()),
            "status": "completed",
            "file_path": "/tmp/exported-dataset.csv",
            "file_size": 1024,
            "download_url": f"/api/v1/datasets/{dataset_id}/download/export_123",
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                f"/api/v1/datasets/{dataset_id}/export",
                json=export_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "export_id" in data
        assert "status" in data
        assert "download_url" in data

    def test_export_dataset_invalid_format(
        self, client, auth_headers
    ):
        """Test dataset export with invalid format."""
        dataset_id = str(uuid4())
        export_payload = {
            "format": "invalid_format",
        }

        response = client.post(
            f"/api/v1/datasets/{dataset_id}/export",
            json=export_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_dataset_search_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset search."""
        search_payload = {
            "query": "test dataset",
            "filters": {
                "format": "csv",
                "size_min": 1000,
                "tags": ["test"],
            },
            "sort_by": "created_at",
            "sort_order": "desc",
        }

        mock_search_results = {
            "results": [
                {
                    "id": str(uuid4()),
                    "name": "test-dataset-1",
                    "description": "Test dataset for anomaly detection",
                    "relevance_score": 0.95,
                },
                {
                    "id": str(uuid4()),
                    "name": "test-dataset-2",
                    "description": "Another test dataset",
                    "relevance_score": 0.87,
                },
            ],
            "total_results": 2,
            "page": 1,
            "size": 10,
        }
        mock_dataset_service.search_datasets.return_value = mock_search_results

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                "/api/v1/datasets/search",
                json=search_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert len(data["results"]) == 2

    def test_dataset_comparison_success(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test successful dataset comparison."""
        comparison_payload = {
            "dataset_ids": [str(uuid4()), str(uuid4())],
            "comparison_type": "statistical",
            "metrics": ["shape", "data_types", "missing_values"],
        }

        mock_comparison_result = {
            "datasets": comparison_payload["dataset_ids"],
            "comparison_type": "statistical",
            "results": {
                "shape": {"dataset_1": (1000, 3), "dataset_2": (2000, 5)},
                "data_types": {
                    "dataset_1": {"numeric": 2, "categorical": 1},
                    "dataset_2": {"numeric": 4, "categorical": 1},
                },
                "missing_values": {"dataset_1": 0, "dataset_2": 50},
            },
            "similarity_score": 0.75,
        }
        mock_dataset_service.compare_datasets.return_value = mock_comparison_result

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.post(
                "/api/v1/datasets/compare",
                json=comparison_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "similarity_score" in data

    def test_dataset_request_validation(self, client, auth_headers):
        """Test comprehensive request validation."""
        # Test invalid JSON
        response = client.post(
            "/api/v1/datasets/search",
            data="invalid json",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid UUID in path
        response = client.get(
            "/api/v1/datasets/invalid-uuid",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_dataset_error_handling(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test error handling in dataset endpoints."""
        # Test service unavailable
        mock_dataset_service.get_dataset.side_effect = Exception("Service unavailable")
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_dataset_concurrent_upload(
        self, client, sample_csv_data, valid_dataset_metadata, mock_dataset_service, auth_headers
    ):
        """Test handling concurrent dataset uploads."""
        import threading
        
        results = []
        
        def upload_dataset():
            files = {
                "file": ("test.csv", BytesIO(sample_csv_data.encode()), "text/csv")
            }
            data = {
                "metadata": json.dumps(valid_dataset_metadata)
            }
            
            with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
                response = client.post(
                    "/api/v1/datasets/upload",
                    files=files,
                    data=data,
                    headers=auth_headers,
                )
                results.append(response.status_code)

        # Create multiple threads for concurrent uploads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=upload_dataset)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All uploads should have completed successfully
        assert len(results) == 3
        assert all(status_code == 201 for status_code in results)

    def test_dataset_security_headers(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test security headers in dataset responses."""
        dataset_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_dataset_cors_handling(
        self, client, mock_dataset_service, auth_headers
    ):
        """Test CORS handling in dataset endpoints."""
        dataset_id = str(uuid4())
        cors_headers = {
            **auth_headers,
            "Origin": "https://example.com",
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_dataset_service", return_value=mock_dataset_service):
            response = client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=cors_headers,
            )

        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200