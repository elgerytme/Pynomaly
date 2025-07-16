"""
Contract tests for API and service interfaces.

This module tests:
- API contract compliance
- Request/response schemas
- Service interface contracts
- Data format validation
- Backward compatibility
"""

import json
from unittest.mock import patch

import jsonschema
import numpy as np
import pandas as pd
import pytest
from jsonschema import ValidationError, validate

from monorepo.domain.entities import Dataset, DetectionResult, Detector
from monorepo.domain.value_objects import AnomalyScore


@pytest.mark.contract
class TestAPIContractCompliance:
    """Tests for API contract compliance."""

    def test_dataset_creation_request_contract(self, client, contract_schemas):
        """Test dataset creation request contract."""
        # Valid request schema
        valid_request = {
            "name": "Contract Test Dataset",
            "data": [
                {"feature1": 1.0, "feature2": 2.0},
                {"feature1": 3.0, "feature2": 4.0},
            ],
            "description": "Dataset for contract testing",
        }

        # Validate request against schema
        try:
            validate(valid_request, contract_schemas["dataset_creation_request"])
        except ValidationError as e:
            pytest.fail(f"Valid request failed schema validation: {e}")

        # Test with API
        with patch(
            "monorepo.application.services.dataset_service.DatasetService.create_dataset"
        ) as mock_create:
            mock_create.return_value = Dataset(
                name=valid_request["name"],
                data=pd.DataFrame(valid_request["data"]),
                description=valid_request["description"],
            )

            response = client.post("/api/v1/datasets", json=valid_request)
            assert response.status_code == 201

            # Validate response schema
            response_data = response.json()
            validate(response_data, contract_schemas["dataset_creation_response"])

    def test_dataset_creation_invalid_request_contract(self, client, contract_schemas):
        """Test dataset creation with invalid request contract."""
        # Invalid requests
        invalid_requests = [
            {
                # Missing required field 'name'
                "data": [{"feature1": 1.0}],
                "description": "Test dataset",
            },
            {
                # Invalid data type for 'data'
                "name": "Test Dataset",
                "data": "invalid_data_type",
                "description": "Test dataset",
            },
            {
                # Empty data array
                "name": "Test Dataset",
                "data": [],
                "description": "Test dataset",
            },
        ]

        for invalid_request in invalid_requests:
            # Should fail schema validation
            with pytest.raises(ValidationError):
                validate(invalid_request, contract_schemas["dataset_creation_request"])

            # Should return 400 from API
            response = client.post("/api/v1/datasets", json=invalid_request)
            assert response.status_code == 400

    def test_detector_creation_request_contract(self, client, contract_schemas):
        """Test detector creation request contract."""
        valid_request = {
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1, "random_state": 42},
            "metadata": {"description": "Contract test detector"},
        }

        # Validate request schema
        validate(valid_request, contract_schemas["detector_creation_request"])

        # Test with API
        with patch(
            "monorepo.application.services.detector_service.DetectorService.create_detector"
        ) as mock_create:
            mock_create.return_value = Detector(
                algorithm_name=valid_request["algorithm_name"],
                parameters=valid_request["parameters"],
                metadata=valid_request["metadata"],
            )

            response = client.post("/api/v1/detectors", json=valid_request)
            assert response.status_code == 201

            # Validate response schema
            response_data = response.json()
            validate(response_data, contract_schemas["detector_creation_response"])

    def test_anomaly_detection_request_contract(self, client, contract_schemas):
        """Test anomaly detection request contract."""
        valid_request = {
            "dataset_id": "test-dataset-id",
            "detector_id": "test-detector-id",
            "parameters": {"return_confidence": True, "batch_size": 1000},
        }

        # Validate request schema
        validate(valid_request, contract_schemas["anomaly_detection_request"])

        # Test with API
        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.return_value = DetectionResult(
                detector_id=valid_request["detector_id"],
                dataset_id=valid_request["dataset_id"],
                scores=[AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)],
                metadata={"contract_test": True},
            )

            response = client.post("/api/v1/detect", json=valid_request)
            assert response.status_code == 200

            # Validate response schema
            response_data = response.json()
            validate(response_data, contract_schemas["anomaly_detection_response"])

    def test_batch_detection_request_contract(self, client, contract_schemas):
        """Test batch detection request contract."""
        valid_request = {
            "dataset_ids": ["dataset-1", "dataset-2", "dataset-3"],
            "detector_id": "test-detector-id",
            "parameters": {"parallel": True, "timeout": 300},
        }

        # Validate request schema
        validate(valid_request, contract_schemas["batch_detection_request"])

        # Test with API
        with patch(
            "monorepo.application.services.anomaly_detection_service.AnomalyDetectionService.detect_anomalies"
        ) as mock_detect:
            mock_detect.side_effect = [
                DetectionResult(
                    detector_id=valid_request["detector_id"],
                    dataset_id=dataset_id,
                    scores=[AnomalyScore(0.5)],
                    metadata={"batch_index": i},
                )
                for i, dataset_id in enumerate(valid_request["dataset_ids"])
            ]

            response = client.post("/api/v1/detect/batch", json=valid_request)
            assert response.status_code == 200

            # Validate response schema
            response_data = response.json()
            validate(response_data, contract_schemas["batch_detection_response"])

    def test_error_response_contract(self, client, contract_schemas):
        """Test error response contract."""
        # Make request that should fail
        invalid_request = {
            "dataset_id": "non-existent-dataset",
            "detector_id": "non-existent-detector",
        }

        response = client.post("/api/v1/detect", json=invalid_request)
        assert response.status_code in [400, 404, 500]

        # Validate error response schema
        response_data = response.json()
        validate(response_data, contract_schemas["error_response"])

    def test_pagination_contract(self, client, contract_schemas):
        """Test pagination contract."""
        # Test datasets list with pagination
        response = client.get("/api/v1/datasets?page=1&limit=10")
        assert response.status_code == 200

        response_data = response.json()
        validate(response_data, contract_schemas["paginated_response"])

        # Verify pagination fields
        assert "items" in response_data
        assert "total" in response_data
        assert "page" in response_data
        assert "limit" in response_data
        assert "has_next" in response_data
        assert "has_previous" in response_data

    def test_authentication_contract(self, client, contract_schemas, user_token):
        """Test authentication contract."""
        # Test with valid token
        headers = {"Authorization": f"Bearer {user_token}"}
        response = client.get("/api/v1/datasets", headers=headers)
        assert response.status_code == 200

        # Test without token
        response = client.get("/api/v1/datasets")
        assert response.status_code == 401

        response_data = response.json()
        validate(response_data, contract_schemas["authentication_error"])

    def test_content_type_contract(self, client):
        """Test content type contract."""
        # Test JSON content type
        response = client.get("/api/v1/datasets")
        assert response.headers["content-type"] == "application/json"

        # Test file upload content type
        test_data = b"feature1,feature2\n1,2\n3,4"
        files = {"file": ("test.csv", test_data, "text/csv")}
        response = client.post("/api/v1/datasets/upload", files=files)
        assert response.headers["content-type"] == "application/json"

    def test_http_methods_contract(self, client):
        """Test HTTP methods contract."""
        # Test GET method
        response = client.get("/api/v1/datasets")
        assert response.status_code in [200, 401]  # 401 if auth required

        # Test POST method
        data = {
            "name": "Test Dataset",
            "data": [{"feature1": 1}],
            "description": "Test",
        }
        response = client.post("/api/v1/datasets", json=data)
        assert response.status_code in [201, 400, 401]

        # Test PUT method (if dataset exists)
        if response.status_code == 201:
            dataset_id = response.json()["id"]
            update_data = {"name": "Updated Test Dataset"}
            response = client.put(f"/api/v1/datasets/{dataset_id}", json=update_data)
            assert response.status_code in [200, 404, 401]

        # Test DELETE method (if dataset exists)
        if response.status_code == 200:
            response = client.delete(f"/api/v1/datasets/{dataset_id}")
            assert response.status_code in [204, 404, 401]

    def test_api_versioning_contract(self, client):
        """Test API versioning contract."""
        # Test v1 API
        response = client.get("/api/v1/datasets")
        assert response.status_code in [200, 401]

        # Test API version in response headers
        if "api-version" in response.headers:
            assert response.headers["api-version"] == "1.0"

    def test_rate_limiting_contract(self, client):
        """Test rate limiting contract."""
        # Make multiple requests rapidly
        responses = []
        for i in range(10):
            response = client.get("/api/v1/datasets")
            responses.append(response)

        # Check for rate limiting headers
        last_response = responses[-1]
        if "x-ratelimit-limit" in last_response.headers:
            assert int(last_response.headers["x-ratelimit-limit"]) > 0

        if "x-ratelimit-remaining" in last_response.headers:
            assert int(last_response.headers["x-ratelimit-remaining"]) >= 0

    def test_cors_headers_contract(self, client):
        """Test CORS headers contract."""
        # Test OPTIONS request
        response = client.options("/api/v1/datasets")

        # Check CORS headers
        expected_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers",
        ]

        for header in expected_headers:
            if header in response.headers:
                assert response.headers[header] is not None


@pytest.mark.contract
class TestServiceInterfaceContracts:
    """Tests for service interface contracts."""

    def test_anomaly_detection_service_contract(self, container):
        """Test anomaly detection service contract."""
        service = container.anomaly_detection_service()

        # Test interface methods exist
        assert hasattr(service, "detect_anomalies")
        assert callable(service.detect_anomalies)

        # Test method signatures
        import inspect

        sig = inspect.signature(service.detect_anomalies)
        params = list(sig.parameters.keys())

        # Should have dataset and detector parameters
        assert "dataset" in params
        assert "detector" in params

    def test_dataset_service_contract(self, container):
        """Test dataset service contract."""
        service = container.dataset_service()

        # Test interface methods
        expected_methods = [
            "create_dataset",
            "get_dataset",
            "update_dataset",
            "delete_dataset",
            "list_datasets",
        ]

        for method in expected_methods:
            assert hasattr(service, method)
            assert callable(getattr(service, method))

    def test_detector_service_contract(self, container):
        """Test detector service contract."""
        service = container.detector_service()

        # Test interface methods
        expected_methods = [
            "create_detector",
            "get_detector",
            "update_detector",
            "delete_detector",
            "list_detectors",
        ]

        for method in expected_methods:
            assert hasattr(service, method)
            assert callable(getattr(service, method))

    def test_repository_interface_contract(self, container):
        """Test repository interface contract."""
        repository = container.dataset_repository()

        # Test interface methods
        expected_methods = ["save", "get_by_id", "get_all", "update", "delete"]

        for method in expected_methods:
            assert hasattr(repository, method)
            assert callable(getattr(repository, method))

    def test_adapter_interface_contract(self, container):
        """Test adapter interface contract."""
        adapter = container.pyod_adapter()

        # Test interface methods
        expected_methods = [
            "create_model",
            "fit",
            "predict",
            "get_supported_algorithms",
        ]

        for method in expected_methods:
            assert hasattr(adapter, method)
            assert callable(getattr(adapter, method))


@pytest.mark.contract
class TestDataFormatContracts:
    """Tests for data format contracts."""

    def test_dataset_entity_contract(self, sample_data):
        """Test Dataset entity contract."""
        dataset = Dataset(
            name="Contract Test Dataset",
            data=sample_data,
            description="Dataset for contract testing",
        )

        # Test required fields
        assert hasattr(dataset, "id")
        assert hasattr(dataset, "name")
        assert hasattr(dataset, "data")
        assert hasattr(dataset, "description")

        # Test field types
        assert isinstance(dataset.name, str)
        assert isinstance(dataset.data, pd.DataFrame)
        assert isinstance(dataset.description, str)

        # Test data integrity
        assert len(dataset.data) > 0
        assert dataset.name == "Contract Test Dataset"

    def test_detector_entity_contract(self):
        """Test Detector entity contract."""
        detector = Detector(
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1},
            metadata={"test": True},
        )

        # Test required fields
        assert hasattr(detector, "id")
        assert hasattr(detector, "algorithm_name")
        assert hasattr(detector, "parameters")
        assert hasattr(detector, "is_fitted")

        # Test field types
        assert isinstance(detector.algorithm_name, str)
        assert isinstance(detector.parameters, dict)
        assert isinstance(detector.is_fitted, bool)

        # Test initial state
        assert detector.is_fitted is False
        assert detector.algorithm_name == "IsolationForest"

    def test_detection_result_contract(self, sample_dataset, sample_detector):
        """Test DetectionResult contract."""
        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]

        result = DetectionResult(
            detector_id=sample_detector.id,
            dataset_id=sample_dataset.id,
            scores=scores,
            metadata={"test": True},
        )

        # Test required fields
        assert hasattr(result, "detector_id")
        assert hasattr(result, "dataset_id")
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")

        # Test field types
        assert isinstance(result.scores, list)
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert isinstance(result.metadata, dict)

        # Test data integrity
        assert len(result.scores) == 3
        assert result.detector_id == sample_detector.id

    def test_anomaly_score_contract(self):
        """Test AnomalyScore contract."""
        score = AnomalyScore(0.7)

        # Test required fields
        assert hasattr(score, "value")

        # Test field types
        assert isinstance(score.value, float)

        # Test value constraints
        assert score.value == 0.7
        assert not np.isnan(score.value)
        assert not np.isinf(score.value)

    def test_json_serialization_contract(self, sample_dataset, sample_detector):
        """Test JSON serialization contract."""
        # Test Dataset serialization
        dataset_dict = {
            "id": sample_dataset.id,
            "name": sample_dataset.name,
            "description": sample_dataset.description,
            "data": sample_dataset.data.to_dict("records"),
        }

        # Should be JSON serializable
        json_str = json.dumps(dataset_dict)
        assert isinstance(json_str, str)

        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized["name"] == sample_dataset.name

        # Test Detector serialization
        detector_dict = {
            "id": sample_detector.id,
            "algorithm_name": sample_detector.algorithm_name,
            "parameters": sample_detector.parameters,
            "is_fitted": sample_detector.is_fitted,
        }

        json_str = json.dumps(detector_dict)
        assert isinstance(json_str, str)

        deserialized = json.loads(json_str)
        assert deserialized["algorithm_name"] == sample_detector.algorithm_name

    def test_pandas_dataframe_contract(self, sample_data):
        """Test pandas DataFrame contract."""
        # Test required DataFrame properties
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) > 0
        assert len(sample_data.columns) > 0

        # Test data types
        for column in sample_data.columns:
            assert sample_data[column].dtype in ["int64", "float64", "object", "bool"]

        # Test no NaN values (unless explicitly allowed)
        assert not sample_data.isnull().all().any()

    def test_numpy_array_contract(self, sample_data):
        """Test numpy array contract."""
        # Convert to numpy array
        np_array = sample_data.values

        # Test array properties
        assert isinstance(np_array, np.ndarray)
        assert np_array.ndim == 2
        assert np_array.shape[0] > 0
        assert np_array.shape[1] > 0

        # Test data types
        assert np_array.dtype in [np.float64, np.int64, np.object_]

        # Test no NaN/inf values
        if np_array.dtype in [np.float64, np.int64]:
            assert not np.isnan(np_array).all()
            assert not np.isinf(np_array).all()


@pytest.mark.contract
class TestBackwardCompatibilityContracts:
    """Tests for backward compatibility contracts."""

    def test_api_v1_backward_compatibility(self, client):
        """Test API v1 backward compatibility."""
        # Test that v1 endpoints still work
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        # Test that response format hasn't changed
        health_data = response.json()
        required_fields = ["status", "timestamp", "version"]
        for field in required_fields:
            assert field in health_data

    def test_dataset_format_backward_compatibility(self, sample_data):
        """Test dataset format backward compatibility."""
        # Test that old dataset format still works
        old_format_dataset = Dataset(
            name="Old Format Dataset",
            data=sample_data,
            description="Dataset in old format",
        )

        # Should still have all required fields
        assert hasattr(old_format_dataset, "id")
        assert hasattr(old_format_dataset, "name")
        assert hasattr(old_format_dataset, "data")
        assert hasattr(old_format_dataset, "description")

    def test_detector_format_backward_compatibility(self):
        """Test detector format backward compatibility."""
        # Test that old detector format still works
        old_format_detector = Detector(
            algorithm_name="IsolationForest", parameters={"contamination": 0.1}
        )

        # Should still have all required fields
        assert hasattr(old_format_detector, "id")
        assert hasattr(old_format_detector, "algorithm_name")
        assert hasattr(old_format_detector, "parameters")
        assert hasattr(old_format_detector, "is_fitted")

    def test_response_format_backward_compatibility(self, client, sample_data):
        """Test response format backward compatibility."""
        # Create dataset using old format
        old_format_request = {
            "name": "Compatibility Test Dataset",
            "data": sample_data.to_dict("records"),
            "description": "Dataset for compatibility testing",
        }

        with patch(
            "monorepo.application.services.dataset_service.DatasetService.create_dataset"
        ) as mock_create:
            mock_create.return_value = Dataset(
                name=old_format_request["name"],
                data=pd.DataFrame(old_format_request["data"]),
                description=old_format_request["description"],
            )

            response = client.post("/api/v1/datasets", json=old_format_request)
            assert response.status_code == 201

            # Response should maintain backward compatibility
            response_data = response.json()
            assert "id" in response_data
            assert "name" in response_data
            assert "description" in response_data


@pytest.mark.contract
class TestContractSchemas:
    """Tests for contract schema definitions."""

    def test_dataset_creation_schema(self, contract_schemas):
        """Test dataset creation schema definition."""
        schema = contract_schemas["dataset_creation_request"]

        # Test schema structure
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Test required fields
        required_fields = schema["required"]
        assert "name" in required_fields
        assert "data" in required_fields

        # Test field definitions
        properties = schema["properties"]
        assert "name" in properties
        assert "data" in properties
        assert "description" in properties

    def test_detector_creation_schema(self, contract_schemas):
        """Test detector creation schema definition."""
        schema = contract_schemas["detector_creation_request"]

        # Test schema structure
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Test required fields
        required_fields = schema["required"]
        assert "algorithm_name" in required_fields
        assert "parameters" in required_fields

    def test_anomaly_detection_schema(self, contract_schemas):
        """Test anomaly detection schema definition."""
        request_schema = contract_schemas["anomaly_detection_request"]
        response_schema = contract_schemas["anomaly_detection_response"]

        # Test request schema
        assert "dataset_id" in request_schema["required"]
        assert "detector_id" in request_schema["required"]

        # Test response schema
        assert "scores" in response_schema["properties"]
        assert "metadata" in response_schema["properties"]

    def test_error_response_schema(self, contract_schemas):
        """Test error response schema definition."""
        schema = contract_schemas["error_response"]

        # Test schema structure
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Test required fields
        required_fields = schema["required"]
        assert "error" in required_fields
        assert "message" in required_fields

    def test_schema_validation_rules(self, contract_schemas):
        """Test schema validation rules."""
        for schema_name, schema in contract_schemas.items():
            # Test that schema is valid JSON Schema
            try:
                jsonschema.Draft7Validator.check_schema(schema)
            except jsonschema.SchemaError as e:
                pytest.fail(f"Schema {schema_name} is invalid: {e}")

            # Test that schema has required structure
            assert "type" in schema
            assert schema["type"] == "object"
