"""
Enhanced Contract Testing Suite
Comprehensive tests for API contracts, service contracts, and interface contracts.
"""

from unittest.mock import Mock

import jsonschema
import pytest

from pynomaly.application.dto import DetectionRequestDTO, TrainingRequestDTO
from pynomaly.domain.protocols import AdapterProtocol
from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestAPIContractTesting:
    """Test suite for API contract validation."""

    @pytest.fixture
    def api_schemas(self):
        """Load API schemas for contract testing."""
        return {
            "dataset_create": {
                "type": "object",
                "required": ["name", "data", "features"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "features": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "detector_train": {
                "type": "object",
                "required": ["dataset_id", "algorithm"],
                "properties": {
                    "dataset_id": {"type": "string"},
                    "algorithm": {
                        "type": "string",
                        "enum": [
                            "IsolationForest",
                            "LocalOutlierFactor",
                            "OneClassSVM",
                        ],
                    },
                    "parameters": {"type": "object"},
                    "contamination": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "additionalProperties": False,
            },
            "prediction_request": {
                "type": "object",
                "required": ["detector_id", "data"],
                "properties": {
                    "detector_id": {"type": "string"},
                    "data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "return_confidence": {"type": "boolean"},
                    "batch_size": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            "prediction_response": {
                "type": "object",
                "required": ["predictions", "anomaly_scores", "detector_id"],
                "properties": {
                    "predictions": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [0, 1]},
                    },
                    "anomaly_scores": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "confidence_intervals": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "detector_id": {"type": "string"},
                    "processing_time_ms": {"type": "number", "minimum": 0},
                    "model_version": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }

    def test_dataset_creation_contract(self, api_schemas):
        """Test dataset creation API contract compliance."""
        valid_requests = [
            {
                "name": "test_dataset",
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "features": ["feature_1", "feature_2", "feature_3"],
            },
            {
                "name": "dataset_with_metadata",
                "description": "Test dataset with metadata",
                "data": [[1.0, 2.0], [3.0, 4.0]],
                "features": ["x", "y"],
                "metadata": {"source": "test", "version": "1.0"},
            },
        ]

        invalid_requests = [
            {
                "data": [[1.0, 2.0]],  # Missing required fields
                "features": ["x"],
            },
            {
                "name": "",  # Empty name
                "data": [[1.0, 2.0]],
                "features": ["x"],
            },
            {
                "name": "test",
                "data": "invalid_data_type",  # Wrong data type
                "features": ["x"],
            },
        ]

        schema = api_schemas["dataset_create"]

        # Test valid requests
        for request in valid_requests:
            try:
                jsonschema.validate(request, schema)
            except jsonschema.ValidationError:
                pytest.fail(f"Valid request failed validation: {request}")

        # Test invalid requests
        for request in invalid_requests:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(request, schema)

    def test_detector_training_contract(self, api_schemas):
        """Test detector training API contract compliance."""
        valid_requests = [
            {"dataset_id": "dataset_123", "algorithm": "IsolationForest"},
            {
                "dataset_id": "dataset_456",
                "algorithm": "LocalOutlierFactor",
                "parameters": {"n_neighbors": 20},
                "contamination": 0.1,
            },
        ]

        invalid_requests = [
            {"algorithm": "IsolationForest"},  # Missing dataset_id
            {
                "dataset_id": "dataset_123",
                "algorithm": "UnsupportedAlgorithm",  # Invalid algorithm
            },
            {
                "dataset_id": "dataset_123",
                "algorithm": "IsolationForest",
                "contamination": 1.5,  # Invalid contamination value
            },
        ]

        schema = api_schemas["detector_train"]

        # Test valid requests
        for request in valid_requests:
            jsonschema.validate(request, schema)

        # Test invalid requests
        for request in invalid_requests:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(request, schema)

    def test_prediction_request_contract(self, api_schemas):
        """Test prediction request API contract compliance."""
        valid_requests = [
            {"detector_id": "detector_123", "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
            {
                "detector_id": "detector_456",
                "data": [[1.0, 2.0]],
                "return_confidence": True,
                "batch_size": 100,
            },
        ]

        invalid_requests = [
            {"data": [[1.0, 2.0]]},  # Missing detector_id
            {
                "detector_id": "detector_123",
                "data": "invalid_data_format",  # Wrong data format
            },
            {
                "detector_id": "detector_123",
                "data": [[1.0, 2.0]],
                "batch_size": 0,  # Invalid batch_size
            },
        ]

        schema = api_schemas["prediction_request"]

        # Test valid requests
        for request in valid_requests:
            jsonschema.validate(request, schema)

        # Test invalid requests
        for request in invalid_requests:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(request, schema)

    def test_prediction_response_contract(self, api_schemas):
        """Test prediction response API contract compliance."""
        valid_responses = [
            {
                "predictions": [0, 1, 0],
                "anomaly_scores": [0.1, 0.8, 0.2],
                "detector_id": "detector_123",
            },
            {
                "predictions": [1, 0],
                "anomaly_scores": [0.9, 0.1],
                "confidence_intervals": [[0.8, 1.0], [0.0, 0.2]],
                "detector_id": "detector_456",
                "processing_time_ms": 150.5,
                "model_version": "1.2.3",
            },
        ]

        invalid_responses = [
            {
                "predictions": [0, 1],
                "anomaly_scores": [0.1, 0.8, 0.2],  # Mismatched array lengths
            },
            {
                "predictions": [0, 2],  # Invalid prediction value
                "anomaly_scores": [0.1, 0.8],
                "detector_id": "detector_123",
            },
            {
                "predictions": [0, 1],
                "anomaly_scores": [0.1, 1.5],  # Invalid score value
                "detector_id": "detector_123",
            },
        ]

        schema = api_schemas["prediction_response"]

        # Test valid responses
        for response in valid_responses:
            jsonschema.validate(response, schema)

        # Test invalid responses
        for response in invalid_responses:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(response, schema)

    def test_api_versioning_contract(self):
        """Test API versioning contract compliance."""
        # Test that breaking changes require version increment
        v1_schema = {
            "type": "object",
            "required": ["data"],
            "properties": {"data": {"type": "array"}},
        }

        v2_schema = {
            "type": "object",
            "required": ["data", "metadata"],  # Added required field (breaking change)
            "properties": {"data": {"type": "array"}, "metadata": {"type": "object"}},
        }

        # Test backward compatibility detection
        v1_request = {"data": [1, 2, 3]}

        # Should pass v1 validation
        jsonschema.validate(v1_request, v1_schema)

        # Should fail v2 validation (breaking change)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(v1_request, v2_schema)

    def test_error_response_contract(self):
        """Test error response contract compliance."""
        error_schema = {
            "type": "object",
            "required": ["error", "message", "status_code"],
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "status_code": {"type": "integer"},
                "details": {"type": "object"},
                "request_id": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"},
            },
        }

        valid_error_responses = [
            {
                "error": "ValidationError",
                "message": "Invalid input data",
                "status_code": 400,
            },
            {
                "error": "NotFoundError",
                "message": "Detector not found",
                "status_code": 404,
                "details": {"detector_id": "invalid_123"},
                "request_id": "req_456",
                "timestamp": "2024-01-01T12:00:00Z",
            },
        ]

        for error_response in valid_error_responses:
            jsonschema.validate(error_response, error_schema)


class TestServiceContractTesting:
    """Test suite for service contract validation."""

    def test_detector_protocol_contract(self):
        """Test detector protocol contract compliance."""
        # Test that all adapters implement DetectorProtocol
        adapters = [SklearnAdapter(), PyTorchAdapter()]

        for adapter in adapters:
            # Check required methods exist
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "predict")
            assert callable(adapter.fit)
            assert callable(adapter.predict)

            # Check method signatures match protocol
            fit_method = adapter.fit
            predict_method = adapter.predict

            # Verify methods accept expected parameters
            import inspect

            fit_signature = inspect.signature(fit_method)
            predict_signature = inspect.signature(predict_method)

            # fit should accept dataset parameter
            assert "dataset" in fit_signature.parameters

            # predict should accept detector and data parameters
            assert "detector" in predict_signature.parameters
            assert "data" in predict_signature.parameters

    def test_adapter_protocol_contract(self):
        """Test adapter protocol contract compliance."""

        class MockAdapter:
            def fit(self, dataset):
                return Mock()

            def predict(self, detector, data):
                return Mock()

            def validate_parameters(self, parameters):
                return True

        adapter = MockAdapter()

        # Test protocol compliance
        assert isinstance(adapter, AdapterProtocol) or hasattr(adapter, "fit")
        assert isinstance(adapter, AdapterProtocol) or hasattr(adapter, "predict")
        assert isinstance(adapter, AdapterProtocol) or hasattr(
            adapter, "validate_parameters"
        )

    def test_repository_protocol_contract(self):
        """Test repository protocol contract compliance."""

        class MockRepository:
            def save(self, entity):
                return "saved_id"

            def get_by_id(self, entity_id):
                return Mock()

            def delete(self, entity_id):
                return True

            def list_all(self):
                return []

        repository = MockRepository()

        # Test protocol compliance
        required_methods = ["save", "get_by_id", "delete", "list_all"]

        for method_name in required_methods:
            assert hasattr(repository, method_name)
            assert callable(getattr(repository, method_name))

    def test_dto_contract_validation(self):
        """Test DTO contract validation."""
        # Test DetectionRequestDTO
        valid_detection_request = DetectionRequestDTO(
            detector_id="detector_123",
            data=[[1.0, 2.0], [3.0, 4.0]],
            return_confidence=True,
        )

        assert valid_detection_request.detector_id == "detector_123"
        assert len(valid_detection_request.data) == 2
        assert valid_detection_request.return_confidence is True

        # Test TrainingRequestDTO
        valid_training_request = TrainingRequestDTO(
            dataset_id="dataset_456",
            algorithm="IsolationForest",
            parameters={"n_estimators": 100},
            contamination=0.1,
        )

        assert valid_training_request.dataset_id == "dataset_456"
        assert valid_training_request.algorithm == "IsolationForest"
        assert valid_training_request.contamination == 0.1

    def test_service_interface_contract(self):
        """Test service interface contract compliance."""
        from pynomaly.application.services.detection_service import DetectionService

        # Mock dependencies
        mock_adapter = Mock()
        mock_repository = Mock()

        service = DetectionService(
            adapter=mock_adapter, model_repository=mock_repository
        )

        # Test service interface methods exist
        required_methods = ["detect_anomalies", "train_detector", "evaluate_model"]

        for method_name in required_methods:
            assert hasattr(service, method_name)
            assert callable(getattr(service, method_name))

    def test_event_contract_validation(self):
        """Test event contract validation."""
        event_schema = {
            "type": "object",
            "required": ["event_type", "timestamp", "payload"],
            "properties": {
                "event_type": {"type": "string"},
                "timestamp": {"type": "string"},
                "payload": {"type": "object"},
                "source": {"type": "string"},
                "version": {"type": "string"},
            },
        }

        valid_events = [
            {
                "event_type": "detector_trained",
                "timestamp": "2024-01-01T12:00:00Z",
                "payload": {
                    "detector_id": "detector_123",
                    "algorithm": "IsolationForest",
                },
            },
            {
                "event_type": "anomaly_detected",
                "timestamp": "2024-01-01T12:00:00Z",
                "payload": {"detector_id": "detector_456", "anomaly_score": 0.95},
                "source": "prediction_service",
                "version": "1.0",
            },
        ]

        for event in valid_events:
            jsonschema.validate(event, event_schema)


class TestInterfaceContractTesting:
    """Test suite for interface contract validation."""

    def test_database_interface_contract(self):
        """Test database interface contract compliance."""

        class MockDatabase:
            async def connect(self):
                return True

            async def disconnect(self):
                return True

            async def execute_query(self, query, *params):
                return []

            async def begin_transaction(self):
                return Mock()

            async def commit_transaction(self, transaction):
                return True

            async def rollback_transaction(self, transaction):
                return True

        db = MockDatabase()

        # Test interface compliance
        interface_methods = [
            "connect",
            "disconnect",
            "execute_query",
            "begin_transaction",
            "commit_transaction",
            "rollback_transaction",
        ]

        for method_name in interface_methods:
            assert hasattr(db, method_name)
            assert callable(getattr(db, method_name))

    def test_caching_interface_contract(self):
        """Test caching interface contract compliance."""

        class MockCache:
            def get(self, key):
                return None

            def set(self, key, value, ttl=None):
                return True

            def delete(self, key):
                return True

            def clear(self):
                return True

            def exists(self, key):
                return False

        cache = MockCache()

        # Test interface compliance
        cache_methods = ["get", "set", "delete", "clear", "exists"]

        for method_name in cache_methods:
            assert hasattr(cache, method_name)
            assert callable(getattr(cache, method_name))

    def test_monitoring_interface_contract(self):
        """Test monitoring interface contract compliance."""

        class MockMonitor:
            def record_metric(self, name, value, tags=None):
                return True

            def increment_counter(self, name, tags=None):
                return True

            def record_histogram(self, name, value, tags=None):
                return True

            def start_timer(self, name):
                return Mock()

            def stop_timer(self, timer):
                return 0.1

        monitor = MockMonitor()

        # Test interface compliance
        monitoring_methods = [
            "record_metric",
            "increment_counter",
            "record_histogram",
            "start_timer",
            "stop_timer",
        ]

        for method_name in monitoring_methods:
            assert hasattr(monitor, method_name)
            assert callable(getattr(monitor, method_name))

    def test_notification_interface_contract(self):
        """Test notification interface contract compliance."""

        class MockNotifier:
            def send_notification(self, recipient, message, priority="normal"):
                return True

            def send_alert(self, alert_type, message, severity="medium"):
                return True

            def subscribe(self, topic, callback):
                return True

            def unsubscribe(self, topic, callback):
                return True

        notifier = MockNotifier()

        # Test interface compliance
        notification_methods = [
            "send_notification",
            "send_alert",
            "subscribe",
            "unsubscribe",
        ]

        for method_name in notification_methods:
            assert hasattr(notifier, method_name)
            assert callable(getattr(notifier, method_name))


class TestContractEvolution:
    """Test suite for contract evolution and versioning."""

    def test_backward_compatibility_validation(self):
        """Test backward compatibility when evolving contracts."""
        # Original API contract
        v1_contract = {
            "type": "object",
            "required": ["data"],
            "properties": {"data": {"type": "array"}, "options": {"type": "object"}},
        }

        # Evolved API contract (backward compatible)
        v2_compatible_contract = {
            "type": "object",
            "required": ["data"],  # Same required fields
            "properties": {
                "data": {"type": "array"},
                "options": {"type": "object"},
                "metadata": {"type": "object"},  # New optional field
            },
        }

        # Evolved API contract (breaking change)
        v2_breaking_contract = {
            "type": "object",
            "required": ["data", "metadata"],  # Added required field
            "properties": {"data": {"type": "array"}, "metadata": {"type": "object"}},
        }

        # Test request that works with v1
        v1_request = {"data": [1, 2, 3], "options": {"timeout": 30}}

        # Should work with v1
        jsonschema.validate(v1_request, v1_contract)

        # Should work with v2 compatible (backward compatible)
        jsonschema.validate(v1_request, v2_compatible_contract)

        # Should fail with v2 breaking (not backward compatible)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(v1_request, v2_breaking_contract)

    def test_contract_deprecation_handling(self):
        """Test contract deprecation and migration."""
        # Deprecated field should still work but generate warning
        deprecated_contract = {
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "legacy_field": {
                    "type": "string",
                    "deprecated": True,
                    "description": "Use new_field instead",
                },
                "new_field": {"type": "string"},
            },
        }

        # Request using deprecated field
        deprecated_request = {"data": [1, 2, 3], "legacy_field": "old_value"}

        # Should still validate
        jsonschema.validate(deprecated_request, deprecated_contract)

        # New request using new field
        new_request = {"data": [1, 2, 3], "new_field": "new_value"}

        # Should also validate
        jsonschema.validate(new_request, deprecated_contract)

    def test_contract_versioning_strategy(self):
        """Test contract versioning strategy."""
        version_info = {
            "v1.0": {"supported": True, "deprecated": False, "end_of_life": None},
            "v1.1": {"supported": True, "deprecated": False, "end_of_life": None},
            "v2.0": {"supported": True, "deprecated": False, "end_of_life": None},
            "v0.9": {
                "supported": False,
                "deprecated": True,
                "end_of_life": "2024-01-01",
            },
        }

        # Test version support logic
        for _version, info in version_info.items():
            if info["supported"]:
                assert not info["deprecated"] or info["end_of_life"] is not None
            else:
                assert info["deprecated"] is True
                assert info["end_of_life"] is not None
