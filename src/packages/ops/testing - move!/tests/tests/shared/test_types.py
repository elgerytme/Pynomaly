"""
Tests for shared types module.

This module tests the shared type definitions to ensure proper type safety,
ID generation, and type distinction functionality.
"""

import uuid

import pytest

from monorepo.shared.types import (  # Infrastructure types; Domain identifier types; Data types; Numeric types; Utility functions
    CacheKey,
    Confidence,
    ConfigKey,
    DatasetId,
    DetectorId,
    FeatureName,
    FeatureValue,
    ModelId,
    RoleId,
    Score,
    SessionId,
    StoragePath,
    TenantId,
    Threshold,
    UserId,
    generate_dataset_id,
    generate_detector_id,
    generate_id,
    generate_model_id,
    generate_role_id,
    generate_session_id,
    generate_tenant_id,
    generate_user_id,
)


class TestDomainIdentifierTypes:
    """Test suite for domain identifier types."""

    def test_dataset_id_creation(self):
        """Test DatasetId type creation."""
        dataset_id = DatasetId("dataset_123")

        assert isinstance(dataset_id, str)
        assert dataset_id == "dataset_123"

    def test_detector_id_creation(self):
        """Test DetectorId type creation."""
        detector_id = DetectorId("detector_456")

        assert isinstance(detector_id, str)
        assert detector_id == "detector_456"

    def test_model_id_creation(self):
        """Test ModelId type creation."""
        model_id = ModelId("model_789")

        assert isinstance(model_id, str)
        assert model_id == "model_789"

    def test_user_id_creation(self):
        """Test UserId type creation."""
        user_id = UserId("user_abc")

        assert isinstance(user_id, str)
        assert user_id == "user_abc"

    def test_tenant_id_creation(self):
        """Test TenantId type creation."""
        tenant_id = TenantId("tenant_xyz")

        assert isinstance(tenant_id, str)
        assert tenant_id == "tenant_xyz"

    def test_role_id_creation(self):
        """Test RoleId type creation."""
        role_id = RoleId("role_admin")

        assert isinstance(role_id, str)
        assert role_id == "role_admin"

    def test_session_id_creation(self):
        """Test SessionId type creation."""
        session_id = SessionId("session_def")

        assert isinstance(session_id, str)
        assert session_id == "session_def"

    def test_id_types_are_strings(self):
        """Test that all ID types are based on strings."""
        id_types = [
            DatasetId("test"),
            DetectorId("test"),
            ModelId("test"),
            UserId("test"),
            TenantId("test"),
            RoleId("test"),
            SessionId("test"),
        ]

        for id_value in id_types:
            assert isinstance(id_value, str)

    def test_id_types_with_uuid_strings(self):
        """Test ID types with UUID strings."""
        uuid_str = str(uuid.uuid4())

        dataset_id = DatasetId(uuid_str)
        detector_id = DetectorId(uuid_str)
        model_id = ModelId(uuid_str)

        assert dataset_id == uuid_str
        assert detector_id == uuid_str
        assert model_id == uuid_str

    def test_id_types_equality(self):
        """Test equality behavior of ID types."""
        id_value = "same_id"

        dataset_id1 = DatasetId(id_value)
        dataset_id2 = DatasetId(id_value)
        detector_id = DetectorId(id_value)

        # Same type with same value should be equal
        assert dataset_id1 == dataset_id2

        # Different types with same value should be equal (since they're just strings)
        assert dataset_id1 == detector_id
        assert dataset_id1 == id_value

    def test_id_types_operations(self):
        """Test that ID types support string operations."""
        dataset_id = DatasetId("test_dataset")

        # Should support string operations
        assert len(dataset_id) == 12
        assert dataset_id.startswith("test")
        assert dataset_id.endswith("dataset")
        assert "test" in dataset_id
        assert dataset_id.upper() == "TEST_DATASET"
        assert dataset_id.replace("test", "prod") == "prod_dataset"


class TestNumericTypes:
    """Test suite for numeric types."""

    def test_score_creation(self):
        """Test Score type creation."""
        score = Score(0.85)

        assert isinstance(score, float)
        assert score == 0.85

    def test_confidence_creation(self):
        """Test Confidence type creation."""
        confidence = Confidence(0.92)

        assert isinstance(confidence, float)
        assert confidence == 0.92

    def test_threshold_creation(self):
        """Test Threshold type creation."""
        threshold = Threshold(0.5)

        assert isinstance(threshold, float)
        assert threshold == 0.5

    def test_numeric_types_with_integers(self):
        """Test numeric types with integer values."""
        score = Score(1)
        confidence = Confidence(0)
        threshold = Threshold(1)

        # Should be converted to float
        assert isinstance(score, float)
        assert isinstance(confidence, float)
        assert isinstance(threshold, float)
        assert score == 1.0
        assert confidence == 0.0
        assert threshold == 1.0

    def test_numeric_types_arithmetic(self):
        """Test arithmetic operations with numeric types."""
        score1 = Score(0.7)
        score2 = Score(0.3)

        # Should support arithmetic operations
        result = score1 + score2
        assert result == 1.0
        assert isinstance(result, float)

        result = score1 - score2
        assert result == 0.4
        assert isinstance(result, float)

        result = score1 * 2
        assert result == 1.4
        assert isinstance(result, float)

        result = score1 / 2
        assert result == 0.35
        assert isinstance(result, float)

    def test_numeric_types_comparison(self):
        """Test comparison operations with numeric types."""
        score1 = Score(0.7)
        score2 = Score(0.3)
        score3 = Score(0.7)

        assert score1 > score2
        assert score2 < score1
        assert score1 == score3
        assert score1 >= score3
        assert score2 <= score1
        assert score1 != score2

    def test_numeric_types_range_validation(self):
        """Test numeric types with various ranges."""
        # Valid score ranges
        valid_scores = [0.0, 0.5, 1.0, 0.999, 0.001]
        for value in valid_scores:
            score = Score(value)
            assert isinstance(score, float)
            assert score == value

        # Edge cases
        negative_score = Score(-0.1)
        high_score = Score(1.1)
        assert isinstance(negative_score, float)
        assert isinstance(high_score, float)


class TestDataTypes:
    """Test suite for data types."""

    def test_feature_name_creation(self):
        """Test FeatureName type creation."""
        feature_name = FeatureName("temperature")

        assert isinstance(feature_name, str)
        assert feature_name == "temperature"

    def test_feature_value_creation(self):
        """Test FeatureValue type creation with various types."""
        # Test with different value types
        numeric_value = FeatureValue(42.5)
        string_value = FeatureValue("categorical")
        boolean_value = FeatureValue(True)
        none_value = FeatureValue(None)
        list_value = FeatureValue([1, 2, 3])
        dict_value = FeatureValue({"nested": "data"})

        assert numeric_value == 42.5
        assert string_value == "categorical"
        assert boolean_value is True
        assert none_value is None
        assert list_value == [1, 2, 3]
        assert dict_value == {"nested": "data"}

    def test_feature_name_string_operations(self):
        """Test FeatureName string operations."""
        feature_name = FeatureName("sensor_temperature_celsius")

        assert len(feature_name) == 25
        assert feature_name.startswith("sensor")
        assert feature_name.endswith("celsius")
        assert "temperature" in feature_name
        assert feature_name.replace("_", "-") == "sensor-temperature-celsius"
        assert feature_name.split("_") == ["sensor", "temperature", "celsius"]

    def test_feature_value_type_preservation(self):
        """Test that FeatureValue preserves the original type."""
        values = [
            42,
            42.5,
            "string",
            True,
            False,
            None,
            [1, 2, 3],
            {"key": "value"},
            (1, 2, 3),
        ]

        for original_value in values:
            feature_value = FeatureValue(original_value)
            assert type(feature_value) == type(original_value)
            assert feature_value == original_value


class TestInfrastructureTypes:
    """Test suite for infrastructure types."""

    def test_cache_key_creation(self):
        """Test CacheKey type creation."""
        cache_key = CacheKey("user:123:profile")

        assert isinstance(cache_key, str)
        assert cache_key == "user:123:profile"

    def test_storage_path_creation(self):
        """Test StoragePath type creation."""
        storage_path = StoragePath("/data/models/anomaly_detector.pkl")

        assert isinstance(storage_path, str)
        assert storage_path == "/data/models/anomaly_detector.pkl"

    def test_config_key_creation(self):
        """Test ConfigKey type creation."""
        config_key = ConfigKey("database.connection_string")

        assert isinstance(config_key, str)
        assert config_key == "database.connection_string"

    def test_infrastructure_types_operations(self):
        """Test infrastructure types support string operations."""
        cache_key = CacheKey("model:detector:isolation_forest")
        storage_path = StoragePath("/app/data/datasets/train.csv")
        config_key = ConfigKey("ml.training.batch_size")

        # Test cache key operations
        assert cache_key.count(":") == 2
        assert cache_key.split(":") == ["model", "detector", "isolation_forest"]

        # Test storage path operations
        assert storage_path.endswith(".csv")
        assert "/data/" in storage_path

        # Test config key operations
        assert config_key.startswith("ml.")
        assert config_key.split(".") == ["ml", "training", "batch_size"]

    def test_infrastructure_types_with_hierarchical_data(self):
        """Test infrastructure types with hierarchical structures."""
        hierarchical_cache_key = CacheKey("tenant:123:user:456:model:789:prediction")
        hierarchical_config_key = ConfigKey("app.security.auth.jwt.secret_key")
        hierarchical_storage_path = StoragePath(
            "/app/storage/tenants/123/models/detector.pkl"
        )

        # Should handle complex hierarchical structures
        assert len(hierarchical_cache_key.split(":")) == 7
        assert len(hierarchical_config_key.split(".")) == 5
        assert "/tenants/" in hierarchical_storage_path


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_generate_id(self):
        """Test basic ID generation."""
        generated_id = generate_id()

        assert isinstance(generated_id, str)
        assert len(generated_id) == 36  # UUID4 string length

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(generated_id)
        assert str(parsed_uuid) == generated_id

    def test_generate_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [generate_id() for _ in range(100)]

        # All IDs should be unique
        assert len(set(ids)) == 100

    def test_generate_dataset_id(self):
        """Test dataset ID generation."""
        dataset_id = generate_dataset_id()

        assert isinstance(dataset_id, str)  # DatasetId is based on str
        assert len(dataset_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(dataset_id)
        assert str(parsed_uuid) == dataset_id

    def test_generate_detector_id(self):
        """Test detector ID generation."""
        detector_id = generate_detector_id()

        assert isinstance(detector_id, str)  # DetectorId is based on str
        assert len(detector_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(detector_id)
        assert str(parsed_uuid) == detector_id

    def test_generate_model_id(self):
        """Test model ID generation."""
        model_id = generate_model_id()

        assert isinstance(model_id, str)  # ModelId is based on str
        assert len(model_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(model_id)
        assert str(parsed_uuid) == model_id

    def test_generate_user_id(self):
        """Test user ID generation."""
        user_id = generate_user_id()

        assert isinstance(user_id, str)  # UserId is based on str
        assert len(user_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(user_id)
        assert str(parsed_uuid) == user_id

    def test_generate_tenant_id(self):
        """Test tenant ID generation."""
        tenant_id = generate_tenant_id()

        assert isinstance(tenant_id, str)  # TenantId is based on str
        assert len(tenant_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(tenant_id)
        assert str(parsed_uuid) == tenant_id

    def test_generate_role_id(self):
        """Test role ID generation."""
        role_id = generate_role_id()

        assert isinstance(role_id, str)  # RoleId is based on str
        assert len(role_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(role_id)
        assert str(parsed_uuid) == role_id

    def test_generate_session_id(self):
        """Test session ID generation."""
        session_id = generate_session_id()

        assert isinstance(session_id, str)  # SessionId is based on str
        assert len(session_id) == 36

        # Should be a valid UUID
        parsed_uuid = uuid.UUID(session_id)
        assert str(parsed_uuid) == session_id

    def test_all_id_generators_uniqueness(self):
        """Test that all ID generators produce unique values."""
        generators = [
            generate_dataset_id,
            generate_detector_id,
            generate_model_id,
            generate_user_id,
            generate_tenant_id,
            generate_role_id,
            generate_session_id,
        ]

        all_generated_ids = []
        for generator in generators:
            for _ in range(10):  # Generate 10 IDs from each generator
                all_generated_ids.append(generator())

        # All IDs should be unique across all generators
        assert len(set(all_generated_ids)) == len(all_generated_ids)

    def test_id_generator_performance(self):
        """Test ID generation performance."""
        import time

        start_time = time.time()

        # Generate a large number of IDs
        ids = [generate_id() for _ in range(1000)]

        end_time = time.time()
        duration = end_time - start_time

        # Should be reasonably fast (less than 1 second for 1000 IDs)
        assert duration < 1.0

        # All IDs should be unique
        assert len(set(ids)) == 1000


class TestTypeInteroperability:
    """Test interoperability between different types."""

    def test_id_types_in_collections(self):
        """Test using ID types in collections."""
        dataset_ids = [generate_dataset_id() for _ in range(5)]
        detector_ids = [generate_detector_id() for _ in range(3)]

        # Should work in sets
        dataset_id_set = set(dataset_ids)
        assert len(dataset_id_set) == 5

        # Should work in dictionaries as keys
        id_mapping = {
            dataset_ids[0]: "training_data",
            dataset_ids[1]: "validation_data",
            detector_ids[0]: "primary_detector",
        }
        assert len(id_mapping) == 3

        # Should work in lists
        all_ids = dataset_ids + detector_ids
        assert len(all_ids) == 8

    def test_numeric_types_interoperability(self):
        """Test numeric types interoperability."""
        score = Score(0.8)
        confidence = Confidence(0.9)
        threshold = Threshold(0.7)

        # Should work in arithmetic operations
        combined = score + confidence
        assert isinstance(combined, float)
        assert combined == 1.7

        # Should work in comparisons
        assert confidence > score
        assert score > threshold
        assert confidence > threshold

        # Should work in collections
        scores = [score, confidence, threshold]
        assert max(scores) == confidence
        assert min(scores) == threshold
        assert sum(scores) == 2.4

    def test_feature_types_in_data_structures(self):
        """Test feature types in data structures."""
        features = {
            FeatureName("temperature"): FeatureValue(23.5),
            FeatureName("humidity"): FeatureValue(65.2),
            FeatureName("status"): FeatureValue("normal"),
            FeatureName("enabled"): FeatureValue(True),
        }

        assert len(features) == 4
        assert features[FeatureName("temperature")] == 23.5
        assert features[FeatureName("status")] == "normal"

        # Should work with iteration
        feature_names = list(features.keys())
        feature_values = list(features.values())

        assert len(feature_names) == 4
        assert len(feature_values) == 4
        assert all(isinstance(name, str) for name in feature_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
