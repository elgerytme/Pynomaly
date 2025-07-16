"""Tests for shared type definitions."""

import re
import uuid

from pynomaly.shared.types import (
    # Infrastructure types
    CacheKey,
    Confidence,
    ConfigKey,
    # Domain identifier types
    DatasetId,
    DetectorId,
    # Data types
    FeatureName,
    FeatureValue,
    ModelId,
    RoleId,
    # Numeric types
    Score,
    SessionId,
    StoragePath,
    TenantId,
    Threshold,
    UserId,
    generate_dataset_id,
    generate_detector_id,
    # Utility functions
    generate_id,
    generate_model_id,
    generate_role_id,
    generate_session_id,
    generate_tenant_id,
    generate_user_id,
)


class TestNewTypeDefinitions:
    """Test suite for NewType definitions."""

    def test_domain_identifier_types(self):
        """Test domain identifier type definitions."""
        # Test DatasetId
        dataset_id = DatasetId("dataset_123")
        assert isinstance(dataset_id, str)
        assert dataset_id == "dataset_123"

        # Test DetectorId
        detector_id = DetectorId("detector_456")
        assert isinstance(detector_id, str)
        assert detector_id == "detector_456"

        # Test ModelId
        model_id = ModelId("model_789")
        assert isinstance(model_id, str)
        assert model_id == "model_789"

        # Test UserId
        user_id = UserId("user_abc")
        assert isinstance(user_id, str)
        assert user_id == "user_abc"

        # Test TenantId
        tenant_id = TenantId("tenant_def")
        assert isinstance(tenant_id, str)
        assert tenant_id == "tenant_def"

        # Test RoleId
        role_id = RoleId("role_ghi")
        assert isinstance(role_id, str)
        assert role_id == "role_ghi"

        # Test SessionId
        session_id = SessionId("session_jkl")
        assert isinstance(session_id, str)
        assert session_id == "session_jkl"

    def test_numeric_types(self):
        """Test numeric type definitions."""
        # Test Score
        score = Score(0.85)
        assert isinstance(score, float)
        assert score == 0.85

        # Test Confidence
        confidence = Confidence(0.92)
        assert isinstance(confidence, float)
        assert confidence == 0.92

        # Test Threshold
        threshold = Threshold(0.75)
        assert isinstance(threshold, float)
        assert threshold == 0.75

    def test_data_types(self):
        """Test data type definitions."""
        # Test FeatureName
        feature_name = FeatureName("temperature")
        assert isinstance(feature_name, str)
        assert feature_name == "temperature"

        # Test FeatureValue with different types
        feature_value_int = FeatureValue(42)
        assert isinstance(feature_value_int, int)
        assert feature_value_int == 42

        feature_value_float = FeatureValue(3.14)
        assert isinstance(feature_value_float, float)
        assert feature_value_float == 3.14

        feature_value_str = FeatureValue("category_a")
        assert isinstance(feature_value_str, str)
        assert feature_value_str == "category_a"

        feature_value_bool = FeatureValue(True)
        assert isinstance(feature_value_bool, bool)
        assert feature_value_bool is True

    def test_infrastructure_types(self):
        """Test infrastructure type definitions."""
        # Test CacheKey
        cache_key = CacheKey("cache:user:123")
        assert isinstance(cache_key, str)
        assert cache_key == "cache:user:123"

        # Test StoragePath
        storage_path = StoragePath("/data/datasets/file.csv")
        assert isinstance(storage_path, str)
        assert storage_path == "/data/datasets/file.csv"

        # Test ConfigKey
        config_key = ConfigKey("database.host")
        assert isinstance(config_key, str)
        assert config_key == "database.host"

    def test_newtype_behavior(self):
        """Test NewType behavior and characteristics."""
        # NewType creates a distinct type for type checking but is the same at runtime
        dataset_id = DatasetId("test_id")
        regular_string = "test_id"

        # At runtime, they are the same
        assert dataset_id == regular_string
        assert type(dataset_id) == type(regular_string)

        # But they are different types for type checking purposes
        assert DatasetId != str  # The NewType is different from the base type


class TestGenerateIdFunction:
    """Test suite for generate_id function."""

    def test_generate_id_returns_string(self):
        """Test generate_id returns a string."""
        result = generate_id()
        assert isinstance(result, str)

    def test_generate_id_returns_uuid_format(self):
        """Test generate_id returns valid UUID format."""
        result = generate_id()
        # Check UUID format with regex
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, result)

    def test_generate_id_is_unique(self):
        """Test generate_id generates unique IDs."""
        ids = [generate_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All IDs should be unique

    def test_generate_id_is_valid_uuid(self):
        """Test generate_id creates valid UUID objects."""
        result = generate_id()
        # Should be able to create UUID object from the string
        uuid_obj = uuid.UUID(result)
        assert str(uuid_obj) == result

    def test_generate_id_multiple_calls(self):
        """Test multiple calls to generate_id."""
        id1 = generate_id()
        id2 = generate_id()
        id3 = generate_id()

        assert id1 != id2
        assert id2 != id3
        assert id1 != id3

        # All should be valid UUIDs
        for test_id in [id1, id2, id3]:
            assert isinstance(test_id, str)
            uuid.UUID(test_id)  # Should not raise exception


class TestSpecificIdGenerators:
    """Test suite for specific ID generator functions."""

    def test_generate_dataset_id(self):
        """Test generate_dataset_id function."""
        dataset_id = generate_dataset_id()
        assert isinstance(dataset_id, str)  # Runtime type is str

        # Should be valid UUID format
        uuid.UUID(dataset_id)

        # Should be unique
        ids = [generate_dataset_id() for _ in range(10)]
        assert len(ids) == len(set(ids))

    def test_generate_detector_id(self):
        """Test generate_detector_id function."""
        detector_id = generate_detector_id()
        assert isinstance(detector_id, str)

        # Should be valid UUID format
        uuid.UUID(detector_id)

        # Should be unique
        ids = [generate_detector_id() for _ in range(10)]
        assert len(ids) == len(set(ids))

    def test_generate_model_id(self):
        """Test generate_model_id function."""
        model_id = generate_model_id()
        assert isinstance(model_id, str)

        # Should be valid UUID format
        uuid.UUID(model_id)

        # Should be unique
        ids = [generate_model_id() for _ in range(10)]
        assert len(ids) == len(set(ids))

    def test_generate_user_id(self):
        """Test generate_user_id function."""
        user_id = generate_user_id()
        assert isinstance(user_id, str)

        # Should be valid UUID format
        uuid.UUID(user_id)

        # Should be unique
        ids = [generate_user_id() for _ in range(10)]
        assert len(ids) == len(set(ids))

    def test_generate_tenant_id(self):
        """Test generate_tenant_id function."""
        tenant_id = generate_tenant_id()
        assert isinstance(tenant_id, str)

        # Should be valid UUID format
        uuid.UUID(tenant_id)

        # Should be unique
        ids = [generate_tenant_id() for _ in range(10)]
        assert len(ids) == len(set(ids))

    def test_generate_role_id(self):
        """Test generate_role_id function."""
        role_id = generate_role_id()
        assert isinstance(role_id, str)

        # Should be valid UUID format
        uuid.UUID(role_id)

        # Should be unique
        ids = [generate_role_id() for _ in range(10)]
        assert len(ids) == len(set(ids))

    def test_generate_session_id(self):
        """Test generate_session_id function."""
        session_id = generate_session_id()
        assert isinstance(session_id, str)

        # Should be valid UUID format
        uuid.UUID(session_id)

        # Should be unique
        ids = [generate_session_id() for _ in range(10)]
        assert len(ids) == len(set(ids))


class TestIdGeneratorConsistency:
    """Test consistency across ID generators."""

    def test_all_generators_use_same_format(self):
        """Test all ID generators use the same UUID format."""
        generators = [
            generate_dataset_id,
            generate_detector_id,
            generate_model_id,
            generate_user_id,
            generate_tenant_id,
            generate_role_id,
            generate_session_id,
        ]

        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

        for generator in generators:
            generated_id = generator()
            assert re.match(uuid_pattern, generated_id)
            assert isinstance(generated_id, str)

    def test_all_generators_produce_unique_ids(self):
        """Test all ID generators produce unique IDs."""
        generators = [
            generate_dataset_id,
            generate_detector_id,
            generate_model_id,
            generate_user_id,
            generate_tenant_id,
            generate_role_id,
            generate_session_id,
        ]

        all_ids = []
        for generator in generators:
            ids = [generator() for _ in range(10)]
            all_ids.extend(ids)

        # All IDs should be unique across all generators
        assert len(all_ids) == len(set(all_ids))

    def test_generators_delegate_to_generate_id(self):
        """Test that specific generators delegate to generate_id."""
        # Mock generate_id to verify it's called
        original_generate_id = generate_id

        def mock_generate_id():
            return "mock-uuid-1234-5678-9012"

        # Patch the generate_id function
        import pynomaly.shared.types as types_module

        types_module.generate_id = mock_generate_id

        try:
            # Test that specific generators use the patched function
            assert generate_dataset_id() == "mock-uuid-1234-5678-9012"
            assert generate_detector_id() == "mock-uuid-1234-5678-9012"
            assert generate_model_id() == "mock-uuid-1234-5678-9012"
            assert generate_user_id() == "mock-uuid-1234-5678-9012"
            assert generate_tenant_id() == "mock-uuid-1234-5678-9012"
            assert generate_role_id() == "mock-uuid-1234-5678-9012"
            assert generate_session_id() == "mock-uuid-1234-5678-9012"
        finally:
            # Restore original function
            types_module.generate_id = original_generate_id


class TestTypeUsageScenarios:
    """Test practical usage scenarios for types."""

    def test_type_usage_in_functions(self):
        """Test using types in function signatures."""

        def process_dataset(dataset_id: DatasetId, name: FeatureName) -> Score:
            # Function that would process dataset
            return Score(0.85)

        dataset_id = generate_dataset_id()
        feature_name = FeatureName("temperature")

        result = process_dataset(dataset_id, feature_name)
        assert isinstance(result, float)
        assert result == 0.85

    def test_type_usage_in_data_structures(self):
        """Test using types in data structures."""
        # Dictionary with typed keys and values
        user_scores = {
            generate_user_id(): Score(0.92),
            generate_user_id(): Score(0.88),
            generate_user_id(): Score(0.76),
        }

        assert len(user_scores) == 3
        for user_id, score in user_scores.items():
            assert isinstance(user_id, str)
            assert isinstance(score, float)

    def test_type_usage_with_collections(self):
        """Test using types with collections."""
        # List of feature names
        feature_names = [
            FeatureName("temperature"),
            FeatureName("humidity"),
            FeatureName("pressure"),
        ]

        assert len(feature_names) == 3
        for name in feature_names:
            assert isinstance(name, str)

        # List of thresholds
        thresholds = [
            Threshold(0.5),
            Threshold(0.7),
            Threshold(0.9),
        ]

        assert len(thresholds) == 3
        for threshold in thresholds:
            assert isinstance(threshold, float)

    def test_feature_value_type_variations(self):
        """Test FeatureValue with different value types."""
        feature_values = [
            FeatureValue(42),  # int
            FeatureValue(3.14),  # float
            FeatureValue("category"),  # str
            FeatureValue(True),  # bool
        ]

        assert isinstance(feature_values[0], int)
        assert isinstance(feature_values[1], float)
        assert isinstance(feature_values[2], str)
        assert isinstance(feature_values[3], bool)

    def test_infrastructure_type_usage(self):
        """Test usage of infrastructure types."""
        # Cache key patterns
        cache_keys = [
            CacheKey("user:123:profile"),
            CacheKey("model:456:predictions"),
            CacheKey("dataset:789:metadata"),
        ]

        for key in cache_keys:
            assert isinstance(key, str)
            assert ":" in key  # Common cache key pattern

        # Storage paths
        storage_paths = [
            StoragePath("/data/models/model.pkl"),
            StoragePath("/data/datasets/train.csv"),
            StoragePath("/data/results/output.json"),
        ]

        for path in storage_paths:
            assert isinstance(path, str)
            assert path.startswith("/data/")  # Common storage pattern

    def test_score_confidence_threshold_relationships(self):
        """Test relationships between Score, Confidence, and Threshold."""
        score = Score(0.85)
        confidence = Confidence(0.92)
        threshold = Threshold(0.75)

        # These should be comparable since they're all floats
        assert score > threshold
        assert confidence > score
        assert confidence > threshold

        # Test arithmetic operations
        assert score + 0.1 == 0.95
        assert confidence - 0.1 == 0.82
        assert threshold * 2 == 1.5


class TestTypeValidation:
    """Test type validation and edge cases."""

    def test_empty_string_ids(self):
        """Test handling of empty string IDs."""
        empty_dataset_id = DatasetId("")
        assert empty_dataset_id == ""
        assert isinstance(empty_dataset_id, str)

    def test_numeric_types_with_edge_values(self):
        """Test numeric types with edge values."""
        # Zero values
        zero_score = Score(0.0)
        assert zero_score == 0.0

        # Negative values
        negative_score = Score(-0.5)
        assert negative_score == -0.5

        # Values > 1
        high_score = Score(1.5)
        assert high_score == 1.5

        # Very small values
        tiny_score = Score(1e-10)
        assert tiny_score == 1e-10

    def test_string_types_with_special_characters(self):
        """Test string types with special characters."""
        special_feature = FeatureName("temperature_°C")
        assert special_feature == "temperature_°C"

        unicode_cache_key = CacheKey("user:测试:profile")
        assert unicode_cache_key == "user:测试:profile"

        path_with_spaces = StoragePath("/data/my data/file.csv")
        assert path_with_spaces == "/data/my data/file.csv"

    def test_type_immutability(self):
        """Test that NewType instances behave like their base types."""
        # String types should support string operations
        dataset_id = DatasetId("test_id")
        assert dataset_id.upper() == "TEST_ID"
        assert dataset_id.replace("test", "prod") == "prod_id"

        # Numeric types should support numeric operations
        score = Score(0.85)
        assert score + 0.1 == 0.95
        assert score * 2 == 1.7
        assert round(score, 1) == 0.8

    def test_type_comparison_operations(self):
        """Test comparison operations on typed values."""
        # Numeric comparisons
        score1 = Score(0.85)
        score2 = Score(0.90)
        threshold = Threshold(0.80)

        assert score2 > score1
        assert score1 > threshold
        assert score2 >= score1
        assert score1 >= threshold

        # String comparisons
        id1 = DatasetId("dataset_a")
        id2 = DatasetId("dataset_b")

        assert id1 < id2
        assert id2 > id1
        assert id1 != id2
        assert id1 == DatasetId("dataset_a")
