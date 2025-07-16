"""Tests for real domain value object implementations."""

import pytest

from pynomaly.domain.value_objects.anomaly_category import AnomalyCategory
from pynomaly.domain.value_objects.anomaly_type import AnomalyType
from pynomaly.domain.value_objects.severity_score import SeverityLevel, SeverityScore


class TestAnomalyTypeImplementation:
    """Test suite for AnomalyType implementation."""

    def test_anomaly_type_values(self):
        """Test AnomalyType enum values."""
        assert AnomalyType.POINT.value == "point"
        assert AnomalyType.CONTEXTUAL.value == "contextual"
        assert AnomalyType.COLLECTIVE.value == "collective"
        assert AnomalyType.GLOBAL.value == "global"
        assert AnomalyType.LOCAL.value == "local"

    def test_get_default_method(self):
        """Test get_default method."""
        result = AnomalyType.get_default()
        assert result == AnomalyType.POINT
        assert isinstance(result, AnomalyType)

    def test_string_representation(self):
        """Test string representation."""
        assert str(AnomalyType.POINT) == "point"
        assert str(AnomalyType.CONTEXTUAL) == "contextual"


class TestAnomalyCategoryImplementation:
    """Test suite for AnomalyCategory implementation."""

    def test_anomaly_category_values(self):
        """Test AnomalyCategory enum values."""
        assert AnomalyCategory.STATISTICAL.value == "statistical"
        assert AnomalyCategory.THRESHOLD.value == "threshold"
        assert AnomalyCategory.CLUSTERING.value == "clustering"
        assert AnomalyCategory.DISTANCE.value == "distance"
        assert AnomalyCategory.DENSITY.value == "density"
        assert AnomalyCategory.NEURAL.value == "neural"
        assert AnomalyCategory.ENSEMBLE.value == "ensemble"

    def test_get_default_method(self):
        """Test get_default method."""
        result = AnomalyCategory.get_default()
        assert result == AnomalyCategory.STATISTICAL
        assert isinstance(result, AnomalyCategory)

    def test_string_representation(self):
        """Test string representation."""
        assert str(AnomalyCategory.STATISTICAL) == "statistical"
        assert str(AnomalyCategory.THRESHOLD) == "threshold"


class TestSeverityScoreImplementation:
    """Test suite for SeverityScore implementation."""

    def test_severity_score_creation(self):
        """Test SeverityScore creation."""
        score = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        assert score.value == 0.5
        assert score.severity_level == SeverityLevel.MEDIUM

    def test_create_minimal_method(self):
        """Test create_minimal method."""
        result = SeverityScore.create_minimal()
        assert isinstance(result, SeverityScore)
        assert result.value == 0.0
        assert result.severity_level == SeverityLevel.LOW

    def test_from_score_method(self):
        """Test from_score method."""
        # Test critical level
        critical_score = SeverityScore.from_score(0.9)
        assert critical_score.severity_level == SeverityLevel.CRITICAL

        # Test high level
        high_score = SeverityScore.from_score(0.7)
        assert high_score.severity_level == SeverityLevel.HIGH

        # Test medium level
        medium_score = SeverityScore.from_score(0.5)
        assert medium_score.severity_level == SeverityLevel.MEDIUM

        # Test low level
        low_score = SeverityScore.from_score(0.2)
        assert low_score.severity_level == SeverityLevel.LOW

    def test_validation(self):
        """Test validation of SeverityScore."""
        # Valid values
        SeverityScore(value=0.0, severity_level=SeverityLevel.LOW)
        SeverityScore(value=1.0, severity_level=SeverityLevel.CRITICAL)

        # Invalid values
        with pytest.raises(ValueError):
            SeverityScore(value=-0.1, severity_level=SeverityLevel.LOW)

        with pytest.raises(ValueError):
            SeverityScore(value=1.1, severity_level=SeverityLevel.HIGH)

    def test_convenience_methods(self):
        """Test convenience methods."""
        critical_score = SeverityScore(value=0.9, severity_level=SeverityLevel.CRITICAL)
        high_score = SeverityScore(value=0.7, severity_level=SeverityLevel.HIGH)
        medium_score = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        low_score = SeverityScore(value=0.2, severity_level=SeverityLevel.LOW)

        assert critical_score.is_critical() is True
        assert high_score.is_critical() is False

        assert critical_score.is_high() is True
        assert high_score.is_high() is True
        assert medium_score.is_high() is False
        assert low_score.is_high() is False


class TestSeverityLevelImplementation:
    """Test suite for SeverityLevel implementation."""

    def test_severity_level_values(self):
        """Test SeverityLevel enum values."""
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.MEDIUM.value == "medium"
        assert SeverityLevel.HIGH.value == "high"
        assert SeverityLevel.CRITICAL.value == "critical"

    def test_string_representation(self):
        """Test string representation."""
        assert str(SeverityLevel.LOW) == "low"
        assert str(SeverityLevel.MEDIUM) == "medium"
        assert str(SeverityLevel.HIGH) == "high"
        assert str(SeverityLevel.CRITICAL) == "critical"


class TestImplementationIntegration:
    """Test integration between real implementations."""

    def test_all_classes_available(self):
        """Test all classes are available."""
        assert AnomalyType is not None
        assert AnomalyCategory is not None
        assert SeverityScore is not None
        assert SeverityLevel is not None

    def test_enum_comparisons(self):
        """Test enum comparisons work correctly."""
        assert AnomalyType.POINT == AnomalyType.POINT
        assert AnomalyType.POINT != AnomalyType.CONTEXTUAL

        assert AnomalyCategory.STATISTICAL == AnomalyCategory.STATISTICAL
        assert AnomalyCategory.STATISTICAL != AnomalyCategory.THRESHOLD

        assert SeverityLevel.LOW == SeverityLevel.LOW
        assert SeverityLevel.LOW != SeverityLevel.HIGH

    def test_dataclass_equality(self):
        """Test dataclass equality."""
        score1 = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        score2 = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        score3 = SeverityScore(value=0.6, severity_level=SeverityLevel.MEDIUM)

        assert score1 == score2
        assert score1 != score3

    def test_frozen_dataclass(self):
        """Test that SeverityScore is frozen."""
        score = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)

        # Should raise error when trying to modify
        with pytest.raises(AttributeError):
            score.value = 0.6
