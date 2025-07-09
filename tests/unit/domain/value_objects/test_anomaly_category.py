"""Tests for anomaly category value object."""

import pytest

from pynomaly.domain.value_objects.anomaly_category import AnomalyCategory


class TestAnomalyCategory:
    """Test suite for AnomalyCategory enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert AnomalyCategory.STATISTICAL == "statistical"
        assert AnomalyCategory.THRESHOLD == "threshold"
        assert AnomalyCategory.CLUSTERING == "clustering"
        assert AnomalyCategory.DISTANCE == "distance"
        assert AnomalyCategory.DENSITY == "density"
        assert AnomalyCategory.NEURAL == "neural"
        assert AnomalyCategory.ENSEMBLE == "ensemble"

    def test_enum_inheritance(self):
        """Test enum inherits from str."""
        # Should inherit from str
        assert isinstance(AnomalyCategory.STATISTICAL, str)
        assert isinstance(AnomalyCategory.THRESHOLD, str)
        assert isinstance(AnomalyCategory.CLUSTERING, str)
        assert isinstance(AnomalyCategory.DISTANCE, str)
        assert isinstance(AnomalyCategory.DENSITY, str)
        assert isinstance(AnomalyCategory.NEURAL, str)
        assert isinstance(AnomalyCategory.ENSEMBLE, str)

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        # Test equality
        assert AnomalyCategory.STATISTICAL == AnomalyCategory.STATISTICAL
        assert AnomalyCategory.STATISTICAL != AnomalyCategory.THRESHOLD
        
        # Test string comparison
        assert AnomalyCategory.STATISTICAL == "statistical"
        assert AnomalyCategory.THRESHOLD == "threshold"
        assert AnomalyCategory.CLUSTERING == "clustering"
        assert AnomalyCategory.DISTANCE == "distance"
        assert AnomalyCategory.DENSITY == "density"
        assert AnomalyCategory.NEURAL == "neural"
        assert AnomalyCategory.ENSEMBLE == "ensemble"

    def test_enum_iteration(self):
        """Test enum iteration."""
        categories = list(AnomalyCategory)
        expected = [
            AnomalyCategory.STATISTICAL,
            AnomalyCategory.THRESHOLD,
            AnomalyCategory.CLUSTERING,
            AnomalyCategory.DISTANCE,
            AnomalyCategory.DENSITY,
            AnomalyCategory.NEURAL,
            AnomalyCategory.ENSEMBLE,
        ]
        assert categories == expected

    def test_enum_membership(self):
        """Test enum membership."""
        assert AnomalyCategory.STATISTICAL in AnomalyCategory
        assert AnomalyCategory.THRESHOLD in AnomalyCategory
        assert AnomalyCategory.CLUSTERING in AnomalyCategory
        assert AnomalyCategory.DISTANCE in AnomalyCategory
        assert AnomalyCategory.DENSITY in AnomalyCategory
        assert AnomalyCategory.NEURAL in AnomalyCategory
        assert AnomalyCategory.ENSEMBLE in AnomalyCategory

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(AnomalyCategory) == 7

    def test_get_default_method(self):
        """Test get_default class method."""
        default = AnomalyCategory.get_default()
        assert default == AnomalyCategory.STATISTICAL
        assert isinstance(default, AnomalyCategory)

    def test_get_default_consistency(self):
        """Test get_default returns consistent value."""
        default1 = AnomalyCategory.get_default()
        default2 = AnomalyCategory.get_default()
        assert default1 == default2
        assert default1 is default2

    def test_string_representation(self):
        """Test string representation of enum values."""
        assert str(AnomalyCategory.STATISTICAL) == "statistical"
        assert str(AnomalyCategory.THRESHOLD) == "threshold"
        assert str(AnomalyCategory.CLUSTERING) == "clustering"
        assert str(AnomalyCategory.DISTANCE) == "distance"
        assert str(AnomalyCategory.DENSITY) == "density"
        assert str(AnomalyCategory.NEURAL) == "neural"
        assert str(AnomalyCategory.ENSEMBLE) == "ensemble"

    def test_repr_representation(self):
        """Test repr representation of enum values."""
        repr_str = repr(AnomalyCategory.STATISTICAL)
        assert "AnomalyCategory.STATISTICAL" in repr_str

    def test_enum_name_property(self):
        """Test enum name property."""
        assert AnomalyCategory.STATISTICAL.name == "STATISTICAL"
        assert AnomalyCategory.THRESHOLD.name == "THRESHOLD"
        assert AnomalyCategory.CLUSTERING.name == "CLUSTERING"
        assert AnomalyCategory.DISTANCE.name == "DISTANCE"
        assert AnomalyCategory.DENSITY.name == "DENSITY"
        assert AnomalyCategory.NEURAL.name == "NEURAL"
        assert AnomalyCategory.ENSEMBLE.name == "ENSEMBLE"

    def test_enum_value_property(self):
        """Test enum value property."""
        assert AnomalyCategory.STATISTICAL.value == "statistical"
        assert AnomalyCategory.THRESHOLD.value == "threshold"
        assert AnomalyCategory.CLUSTERING.value == "clustering"
        assert AnomalyCategory.DISTANCE.value == "distance"
        assert AnomalyCategory.DENSITY.value == "density"
        assert AnomalyCategory.NEURAL.value == "neural"
        assert AnomalyCategory.ENSEMBLE.value == "ensemble"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert AnomalyCategory("statistical") == AnomalyCategory.STATISTICAL
        assert AnomalyCategory("threshold") == AnomalyCategory.THRESHOLD
        assert AnomalyCategory("clustering") == AnomalyCategory.CLUSTERING
        assert AnomalyCategory("distance") == AnomalyCategory.DISTANCE
        assert AnomalyCategory("density") == AnomalyCategory.DENSITY
        assert AnomalyCategory("neural") == AnomalyCategory.NEURAL
        assert AnomalyCategory("ensemble") == AnomalyCategory.ENSEMBLE

    def test_enum_from_invalid_string(self):
        """Test creating enum from invalid string raises ValueError."""
        with pytest.raises(ValueError, match="'invalid' is not a valid AnomalyCategory"):
            AnomalyCategory("invalid")

    def test_enum_uniqueness(self):
        """Test all enum values are unique."""
        values = [category.value for category in AnomalyCategory]
        assert len(values) == len(set(values))

    def test_enum_hashable(self):
        """Test enum values are hashable."""
        categories_set = {
            AnomalyCategory.STATISTICAL,
            AnomalyCategory.THRESHOLD,
            AnomalyCategory.CLUSTERING,
            AnomalyCategory.DISTANCE,
            AnomalyCategory.DENSITY,
            AnomalyCategory.NEURAL,
            AnomalyCategory.ENSEMBLE,
        }
        assert len(categories_set) == 7

    def test_enum_in_dictionary(self):
        """Test enum values can be used as dictionary keys."""
        category_descriptions = {
            AnomalyCategory.STATISTICAL: "Statistical-based anomaly detection",
            AnomalyCategory.THRESHOLD: "Threshold-based anomaly detection",
            AnomalyCategory.CLUSTERING: "Clustering-based anomaly detection",
            AnomalyCategory.DISTANCE: "Distance-based anomaly detection",
            AnomalyCategory.DENSITY: "Density-based anomaly detection",
            AnomalyCategory.NEURAL: "Neural network-based anomaly detection",
            AnomalyCategory.ENSEMBLE: "Ensemble-based anomaly detection",
        }
        
        assert len(category_descriptions) == 7
        assert category_descriptions[AnomalyCategory.STATISTICAL] == "Statistical-based anomaly detection"
        assert category_descriptions[AnomalyCategory.NEURAL] == "Neural network-based anomaly detection"

    def test_enum_sorting(self):
        """Test enum values can be sorted."""
        categories = [
            AnomalyCategory.NEURAL,
            AnomalyCategory.CLUSTERING,
            AnomalyCategory.STATISTICAL,
            AnomalyCategory.ENSEMBLE,
        ]
        
        # Sort by string value
        sorted_categories = sorted(categories, key=lambda x: x.value)
        expected = [
            AnomalyCategory.CLUSTERING,
            AnomalyCategory.ENSEMBLE,
            AnomalyCategory.NEURAL,
            AnomalyCategory.STATISTICAL,
        ]
        
        assert sorted_categories == expected

    def test_enum_with_conditional_logic(self):
        """Test using enum in conditional logic."""
        category = AnomalyCategory.STATISTICAL
        
        if category == AnomalyCategory.STATISTICAL:
            result = "statistical_processing"
        elif category == AnomalyCategory.NEURAL:
            result = "neural_processing"
        else:
            result = "other_processing"
        
        assert result == "statistical_processing"

    def test_enum_case_sensitivity(self):
        """Test enum is case sensitive."""
        with pytest.raises(ValueError):
            AnomalyCategory("STATISTICAL")  # Should be lowercase
            
        with pytest.raises(ValueError):
            AnomalyCategory("Statistical")  # Should be lowercase

    def test_enum_comprehensive_coverage(self):
        """Test enum covers expected anomaly detection categories."""
        # Test that we have the main categories of anomaly detection
        expected_categories = {
            "statistical",    # Statistical methods
            "threshold",      # Threshold-based methods
            "clustering",     # Clustering-based methods
            "distance",       # Distance-based methods
            "density",        # Density-based methods
            "neural",         # Neural network methods
            "ensemble",       # Ensemble methods
        }
        
        actual_categories = {category.value for category in AnomalyCategory}
        assert actual_categories == expected_categories

    def test_enum_method_inheritance(self):
        """Test enum inherits methods from str."""
        category = AnomalyCategory.STATISTICAL
        
        # Should have string methods
        assert category.upper() == "STATISTICAL"
        assert category.startswith("stat")
        assert category.endswith("cal")
        assert "stat" in category
        assert len(category) == len("statistical")

    def test_enum_default_usage_pattern(self):
        """Test typical usage pattern with default."""
        def get_category_or_default(category_str: str | None = None) -> AnomalyCategory:
            if category_str:
                try:
                    return AnomalyCategory(category_str)
                except ValueError:
                    pass
            return AnomalyCategory.get_default()
        
        # Test with valid category
        assert get_category_or_default("neural") == AnomalyCategory.NEURAL
        
        # Test with invalid category
        assert get_category_or_default("invalid") == AnomalyCategory.STATISTICAL
        
        # Test with None
        assert get_category_or_default(None) == AnomalyCategory.STATISTICAL
        
        # Test with empty string
        assert get_category_or_default("") == AnomalyCategory.STATISTICAL