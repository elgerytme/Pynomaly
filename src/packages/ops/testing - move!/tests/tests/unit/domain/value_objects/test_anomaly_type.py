"""Tests for anomaly type value object."""

import pytest

from monorepo.domain.value_objects.anomaly_type import AnomalyType


class TestAnomalyType:
    """Test suite for AnomalyType enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert AnomalyType.POINT == "point"
        assert AnomalyType.CONTEXTUAL == "contextual"
        assert AnomalyType.COLLECTIVE == "collective"
        assert AnomalyType.GLOBAL == "global"
        assert AnomalyType.LOCAL == "local"

    def test_enum_inheritance(self):
        """Test enum inherits from str."""
        # Should inherit from str
        assert isinstance(AnomalyType.POINT, str)
        assert isinstance(AnomalyType.CONTEXTUAL, str)
        assert isinstance(AnomalyType.COLLECTIVE, str)
        assert isinstance(AnomalyType.GLOBAL, str)
        assert isinstance(AnomalyType.LOCAL, str)

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        # Test equality
        assert AnomalyType.POINT == AnomalyType.POINT
        assert AnomalyType.POINT != AnomalyType.CONTEXTUAL

        # Test string comparison
        assert AnomalyType.POINT == "point"
        assert AnomalyType.CONTEXTUAL == "contextual"
        assert AnomalyType.COLLECTIVE == "collective"
        assert AnomalyType.GLOBAL == "global"
        assert AnomalyType.LOCAL == "local"

    def test_enum_iteration(self):
        """Test enum iteration."""
        types = list(AnomalyType)
        expected = [
            AnomalyType.POINT,
            AnomalyType.CONTEXTUAL,
            AnomalyType.COLLECTIVE,
            AnomalyType.GLOBAL,
            AnomalyType.LOCAL,
        ]
        assert types == expected

    def test_enum_membership(self):
        """Test enum membership."""
        assert AnomalyType.POINT in AnomalyType
        assert AnomalyType.CONTEXTUAL in AnomalyType
        assert AnomalyType.COLLECTIVE in AnomalyType
        assert AnomalyType.GLOBAL in AnomalyType
        assert AnomalyType.LOCAL in AnomalyType

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(AnomalyType) == 5

    def test_get_default_method(self):
        """Test get_default class method."""
        default = AnomalyType.get_default()
        assert default == AnomalyType.POINT
        assert isinstance(default, AnomalyType)

    def test_get_default_consistency(self):
        """Test get_default returns consistent value."""
        default1 = AnomalyType.get_default()
        default2 = AnomalyType.get_default()
        assert default1 == default2
        assert default1 is default2

    def test_string_representation(self):
        """Test string representation of enum values."""
        assert str(AnomalyType.POINT) == "point"
        assert str(AnomalyType.CONTEXTUAL) == "contextual"
        assert str(AnomalyType.COLLECTIVE) == "collective"
        assert str(AnomalyType.GLOBAL) == "global"
        assert str(AnomalyType.LOCAL) == "local"

    def test_repr_representation(self):
        """Test repr representation of enum values."""
        repr_str = repr(AnomalyType.POINT)
        assert "AnomalyType.POINT" in repr_str

    def test_enum_name_property(self):
        """Test enum name property."""
        assert AnomalyType.POINT.name == "POINT"
        assert AnomalyType.CONTEXTUAL.name == "CONTEXTUAL"
        assert AnomalyType.COLLECTIVE.name == "COLLECTIVE"
        assert AnomalyType.GLOBAL.name == "GLOBAL"
        assert AnomalyType.LOCAL.name == "LOCAL"

    def test_enum_value_property(self):
        """Test enum value property."""
        assert AnomalyType.POINT.value == "point"
        assert AnomalyType.CONTEXTUAL.value == "contextual"
        assert AnomalyType.COLLECTIVE.value == "collective"
        assert AnomalyType.GLOBAL.value == "global"
        assert AnomalyType.LOCAL.value == "local"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert AnomalyType("point") == AnomalyType.POINT
        assert AnomalyType("contextual") == AnomalyType.CONTEXTUAL
        assert AnomalyType("collective") == AnomalyType.COLLECTIVE
        assert AnomalyType("global") == AnomalyType.GLOBAL
        assert AnomalyType("local") == AnomalyType.LOCAL

    def test_enum_from_invalid_string(self):
        """Test creating enum from invalid string raises ValueError."""
        with pytest.raises(ValueError, match="'invalid' is not a valid AnomalyType"):
            AnomalyType("invalid")

    def test_enum_uniqueness(self):
        """Test all enum values are unique."""
        values = [anomaly_type.value for anomaly_type in AnomalyType]
        assert len(values) == len(set(values))

    def test_enum_hashable(self):
        """Test enum values are hashable."""
        types_set = {
            AnomalyType.POINT,
            AnomalyType.CONTEXTUAL,
            AnomalyType.COLLECTIVE,
            AnomalyType.GLOBAL,
            AnomalyType.LOCAL,
        }
        assert len(types_set) == 5

    def test_enum_in_dictionary(self):
        """Test enum values can be used as dictionary keys."""
        type_descriptions = {
            AnomalyType.POINT: "Individual data points that are anomalous",
            AnomalyType.CONTEXTUAL: "Anomalous in specific context",
            AnomalyType.COLLECTIVE: "Group of data points that are anomalous",
            AnomalyType.GLOBAL: "Global anomalies across entire dataset",
            AnomalyType.LOCAL: "Local anomalies within specific regions",
        }

        assert len(type_descriptions) == 5
        assert (
            type_descriptions[AnomalyType.POINT]
            == "Individual data points that are anomalous"
        )
        assert (
            type_descriptions[AnomalyType.CONTEXTUAL] == "Anomalous in specific context"
        )

    def test_enum_sorting(self):
        """Test enum values can be sorted."""
        types = [
            AnomalyType.LOCAL,
            AnomalyType.POINT,
            AnomalyType.GLOBAL,
            AnomalyType.CONTEXTUAL,
        ]

        # Sort by string value
        sorted_types = sorted(types, key=lambda x: x.value)
        expected = [
            AnomalyType.COLLECTIVE,  # "collective" comes first alphabetically
            AnomalyType.CONTEXTUAL,  # "contextual"
            AnomalyType.GLOBAL,  # "global"
            AnomalyType.LOCAL,  # "local"
            AnomalyType.POINT,  # "point"
        ]

        # Adjust expected to match actual input
        expected_from_input = [
            AnomalyType.CONTEXTUAL,  # "contextual"
            AnomalyType.GLOBAL,  # "global"
            AnomalyType.LOCAL,  # "local"
            AnomalyType.POINT,  # "point"
        ]

        assert sorted_types == expected_from_input

    def test_enum_with_conditional_logic(self):
        """Test using enum in conditional logic."""
        anomaly_type = AnomalyType.POINT

        if anomaly_type == AnomalyType.POINT:
            result = "point_detection"
        elif anomaly_type == AnomalyType.CONTEXTUAL:
            result = "contextual_detection"
        else:
            result = "other_detection"

        assert result == "point_detection"

    def test_enum_case_sensitivity(self):
        """Test enum is case sensitive."""
        with pytest.raises(ValueError):
            AnomalyType("POINT")  # Should be lowercase

        with pytest.raises(ValueError):
            AnomalyType("Point")  # Should be lowercase

    def test_enum_comprehensive_coverage(self):
        """Test enum covers expected anomaly types."""
        # Test that we have the main types of anomalies
        expected_types = {
            "point",  # Point anomalies
            "contextual",  # Contextual anomalies
            "collective",  # Collective anomalies
            "global",  # Global anomalies
            "local",  # Local anomalies
        }

        actual_types = {anomaly_type.value for anomaly_type in AnomalyType}
        assert actual_types == expected_types

    def test_enum_method_inheritance(self):
        """Test enum inherits methods from str."""
        anomaly_type = AnomalyType.CONTEXTUAL

        # Should have string methods
        assert anomaly_type.upper() == "CONTEXTUAL"
        assert anomaly_type.startswith("con")
        assert anomaly_type.endswith("ual")
        assert "text" in anomaly_type
        assert len(anomaly_type) == len("contextual")

    def test_enum_default_usage_pattern(self):
        """Test typical usage pattern with default."""

        def get_type_or_default(type_str: str | None = None) -> AnomalyType:
            if type_str:
                try:
                    return AnomalyType(type_str)
                except ValueError:
                    pass
            return AnomalyType.get_default()

        # Test with valid type
        assert get_type_or_default("collective") == AnomalyType.COLLECTIVE

        # Test with invalid type
        assert get_type_or_default("invalid") == AnomalyType.POINT

        # Test with None
        assert get_type_or_default(None) == AnomalyType.POINT

        # Test with empty string
        assert get_type_or_default("") == AnomalyType.POINT

    def test_anomaly_type_semantic_meaning(self):
        """Test semantic meaning of anomaly types."""
        # Test point anomalies
        assert AnomalyType.POINT == "point"
        assert AnomalyType.POINT.name == "POINT"

        # Test contextual anomalies
        assert AnomalyType.CONTEXTUAL == "contextual"
        assert AnomalyType.CONTEXTUAL.name == "CONTEXTUAL"

        # Test collective anomalies
        assert AnomalyType.COLLECTIVE == "collective"
        assert AnomalyType.COLLECTIVE.name == "COLLECTIVE"

        # Test global anomalies
        assert AnomalyType.GLOBAL == "global"
        assert AnomalyType.GLOBAL.name == "GLOBAL"

        # Test local anomalies
        assert AnomalyType.LOCAL == "local"
        assert AnomalyType.LOCAL.name == "LOCAL"

    def test_anomaly_type_classification_logic(self):
        """Test using anomaly types for classification logic."""

        def classify_anomaly_scope(anomaly_type: AnomalyType) -> str:
            if anomaly_type in [AnomalyType.POINT, AnomalyType.LOCAL]:
                return "localized"
            elif anomaly_type in [AnomalyType.GLOBAL, AnomalyType.COLLECTIVE]:
                return "widespread"
            else:
                return "contextual"

        assert classify_anomaly_scope(AnomalyType.POINT) == "localized"
        assert classify_anomaly_scope(AnomalyType.LOCAL) == "localized"
        assert classify_anomaly_scope(AnomalyType.GLOBAL) == "widespread"
        assert classify_anomaly_scope(AnomalyType.COLLECTIVE) == "widespread"
        assert classify_anomaly_scope(AnomalyType.CONTEXTUAL) == "contextual"

    def test_anomaly_type_filtering(self):
        """Test filtering based on anomaly types."""
        all_types = list(AnomalyType)

        # Filter for local scope anomalies
        local_scope = [
            t for t in all_types if t in [AnomalyType.POINT, AnomalyType.LOCAL]
        ]
        assert len(local_scope) == 2
        assert AnomalyType.POINT in local_scope
        assert AnomalyType.LOCAL in local_scope

        # Filter for global scope anomalies
        global_scope = [
            t for t in all_types if t in [AnomalyType.GLOBAL, AnomalyType.COLLECTIVE]
        ]
        assert len(global_scope) == 2
        assert AnomalyType.GLOBAL in global_scope
        assert AnomalyType.COLLECTIVE in global_scope

    def test_anomaly_type_default_behavior(self):
        """Test default behavior matches expected semantics."""
        # Point anomalies are typically the most common, so it's a good default
        default = AnomalyType.get_default()
        assert default == AnomalyType.POINT

        # Verify it's the expected default for general anomaly detection
        assert default.value == "point"
