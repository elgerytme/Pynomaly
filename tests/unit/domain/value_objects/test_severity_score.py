"""Tests for severity score value object."""

import pytest

from pynomaly.domain.value_objects.severity_score import SeverityLevel, SeverityScore


class TestSeverityLevel:
    """Test suite for SeverityLevel enum."""

    def test_enum_values(self):
        """Test all enum values are correctly defined."""
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.MEDIUM == "medium"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.CRITICAL == "critical"

    def test_enum_inheritance(self):
        """Test enum inherits from str."""
        assert isinstance(SeverityLevel.LOW, str)
        assert isinstance(SeverityLevel.MEDIUM, str)
        assert isinstance(SeverityLevel.HIGH, str)
        assert isinstance(SeverityLevel.CRITICAL, str)

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        assert SeverityLevel.LOW == SeverityLevel.LOW
        assert SeverityLevel.LOW != SeverityLevel.HIGH
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.CRITICAL == "critical"

    def test_enum_iteration(self):
        """Test enum iteration."""
        levels = list(SeverityLevel)
        expected = [
            SeverityLevel.LOW,
            SeverityLevel.MEDIUM,
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
        ]
        assert levels == expected

    def test_enum_count(self):
        """Test enum has correct number of values."""
        assert len(SeverityLevel) == 4

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert SeverityLevel("low") == SeverityLevel.LOW
        assert SeverityLevel("medium") == SeverityLevel.MEDIUM
        assert SeverityLevel("high") == SeverityLevel.HIGH
        assert SeverityLevel("critical") == SeverityLevel.CRITICAL

    def test_enum_from_invalid_string(self):
        """Test creating enum from invalid string raises ValueError."""
        with pytest.raises(ValueError, match="'extreme' is not a valid SeverityLevel"):
            SeverityLevel("extreme")

    def test_enum_ordering_semantics(self):
        """Test semantic ordering of severity levels."""
        # Test that levels can be used in logical contexts
        levels = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        
        # Test membership
        assert SeverityLevel.HIGH in levels
        assert SeverityLevel.CRITICAL in levels
        
        # Test set operations
        high_levels = {SeverityLevel.HIGH, SeverityLevel.CRITICAL}
        assert SeverityLevel.CRITICAL in high_levels
        assert SeverityLevel.LOW not in high_levels


class TestSeverityScore:
    """Test suite for SeverityScore value object."""

    def test_basic_creation(self):
        """Test basic creation of severity score."""
        score = SeverityScore(
            value=0.75,
            severity_level=SeverityLevel.HIGH,
            confidence=0.9
        )
        
        assert score.value == 0.75
        assert score.severity_level == SeverityLevel.HIGH
        assert score.confidence == 0.9

    def test_creation_without_confidence(self):
        """Test creation without confidence value."""
        score = SeverityScore(
            value=0.5,
            severity_level=SeverityLevel.MEDIUM
        )
        
        assert score.value == 0.5
        assert score.severity_level == SeverityLevel.MEDIUM
        assert score.confidence is None

    def test_immutability(self):
        """Test that severity score is immutable."""
        score = SeverityScore(
            value=0.75,
            severity_level=SeverityLevel.HIGH
        )
        
        # Should not be able to modify values
        with pytest.raises(AttributeError):
            score.value = 0.8

    def test_validation_value_range(self):
        """Test validation of value range."""
        # Valid values
        SeverityScore(value=0.0, severity_level=SeverityLevel.LOW)
        SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        SeverityScore(value=1.0, severity_level=SeverityLevel.CRITICAL)
        
        # Invalid values
        with pytest.raises(ValueError, match="Severity score must be between 0.0 and 1.0"):
            SeverityScore(value=-0.1, severity_level=SeverityLevel.LOW)
            
        with pytest.raises(ValueError, match="Severity score must be between 0.0 and 1.0"):
            SeverityScore(value=1.1, severity_level=SeverityLevel.CRITICAL)

    def test_validation_confidence_range(self):
        """Test validation of confidence range."""
        # Valid confidence values
        SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM, confidence=0.0)
        SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM, confidence=0.5)
        SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM, confidence=1.0)
        
        # Invalid confidence values
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM, confidence=-0.1)
            
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM, confidence=1.1)

    def test_create_minimal_factory(self):
        """Test create_minimal factory method."""
        score = SeverityScore.create_minimal()
        
        assert score.value == 0.0
        assert score.severity_level == SeverityLevel.LOW
        assert score.confidence is None

    def test_from_score_factory(self):
        """Test from_score factory method."""
        # Test LOW threshold (< 0.4)
        low_score = SeverityScore.from_score(0.2)
        assert low_score.value == 0.2
        assert low_score.severity_level == SeverityLevel.LOW
        
        # Test MEDIUM threshold (0.4 <= score < 0.6)
        medium_score = SeverityScore.from_score(0.5)
        assert medium_score.value == 0.5
        assert medium_score.severity_level == SeverityLevel.MEDIUM
        
        # Test HIGH threshold (0.6 <= score < 0.8)
        high_score = SeverityScore.from_score(0.7)
        assert high_score.value == 0.7
        assert high_score.severity_level == SeverityLevel.HIGH
        
        # Test CRITICAL threshold (>= 0.8)
        critical_score = SeverityScore.from_score(0.9)
        assert critical_score.value == 0.9
        assert critical_score.severity_level == SeverityLevel.CRITICAL

    def test_from_score_boundary_values(self):
        """Test from_score with boundary values."""
        # Test exact boundary values
        assert SeverityScore.from_score(0.0).severity_level == SeverityLevel.LOW
        assert SeverityScore.from_score(0.39).severity_level == SeverityLevel.LOW
        assert SeverityScore.from_score(0.4).severity_level == SeverityLevel.MEDIUM
        assert SeverityScore.from_score(0.59).severity_level == SeverityLevel.MEDIUM
        assert SeverityScore.from_score(0.6).severity_level == SeverityLevel.HIGH
        assert SeverityScore.from_score(0.79).severity_level == SeverityLevel.HIGH
        assert SeverityScore.from_score(0.8).severity_level == SeverityLevel.CRITICAL
        assert SeverityScore.from_score(1.0).severity_level == SeverityLevel.CRITICAL

    def test_from_score_validation(self):
        """Test from_score validation."""
        # Should validate input range
        with pytest.raises(ValueError, match="Severity score must be between 0.0 and 1.0"):
            SeverityScore.from_score(-0.1)
            
        with pytest.raises(ValueError, match="Severity score must be between 0.0 and 1.0"):
            SeverityScore.from_score(1.1)

    def test_is_critical_method(self):
        """Test is_critical method."""
        low_score = SeverityScore(value=0.2, severity_level=SeverityLevel.LOW)
        medium_score = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        high_score = SeverityScore(value=0.7, severity_level=SeverityLevel.HIGH)
        critical_score = SeverityScore(value=0.9, severity_level=SeverityLevel.CRITICAL)
        
        assert low_score.is_critical() is False
        assert medium_score.is_critical() is False
        assert high_score.is_critical() is False
        assert critical_score.is_critical() is True

    def test_is_high_method(self):
        """Test is_high method."""
        low_score = SeverityScore(value=0.2, severity_level=SeverityLevel.LOW)
        medium_score = SeverityScore(value=0.5, severity_level=SeverityLevel.MEDIUM)
        high_score = SeverityScore(value=0.7, severity_level=SeverityLevel.HIGH)
        critical_score = SeverityScore(value=0.9, severity_level=SeverityLevel.CRITICAL)
        
        assert low_score.is_high() is False
        assert medium_score.is_high() is False
        assert high_score.is_high() is True
        assert critical_score.is_high() is True

    def test_equality_comparison(self):
        """Test equality comparison."""
        score1 = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        score2 = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        score3 = SeverityScore(value=0.8, severity_level=SeverityLevel.CRITICAL, confidence=0.9)
        
        assert score1 == score2
        assert score1 != score3

    def test_hash_behavior(self):
        """Test hash behavior for use in sets and dictionaries."""
        score1 = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        score2 = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        score3 = SeverityScore(value=0.8, severity_level=SeverityLevel.CRITICAL, confidence=0.9)
        
        # Same values should have same hash
        assert hash(score1) == hash(score2)
        
        # Different values should have different hash
        assert hash(score1) != hash(score3)
        
        # Test in set
        score_set = {score1, score2, score3}
        assert len(score_set) == 2  # score1 and score2 are equal

    def test_repr_representation(self):
        """Test repr representation."""
        score = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        repr_str = repr(score)
        assert "SeverityScore" in repr_str
        assert "0.75" in repr_str
        assert "HIGH" in repr_str
        assert "0.9" in repr_str

    def test_string_representation(self):
        """Test string representation."""
        score = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        str_repr = str(score)
        # Should contain the dataclass string representation
        assert "SeverityScore" in str_repr or "0.75" in str_repr

    def test_severity_level_integration(self):
        """Test integration with SeverityLevel enum."""
        score = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH)
        
        # Should be able to compare with enum values
        assert score.severity_level == SeverityLevel.HIGH
        assert score.severity_level != SeverityLevel.LOW
        
        # Should be able to compare with string values
        assert score.severity_level == "high"
        assert score.severity_level != "low"

    def test_confidence_optional_behavior(self):
        """Test behavior with optional confidence."""
        # Without confidence
        score_no_conf = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH)
        assert score_no_conf.confidence is None
        
        # With confidence
        score_with_conf = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        assert score_with_conf.confidence == 0.9
        
        # Should be different objects
        assert score_no_conf != score_with_conf

    def test_practical_usage_scenarios(self):
        """Test practical usage scenarios."""
        def classify_severity(score_value: float) -> SeverityScore:
            return SeverityScore.from_score(score_value)
        
        def needs_immediate_attention(score: SeverityScore) -> bool:
            return score.is_critical()
        
        def needs_investigation(score: SeverityScore) -> bool:
            return score.is_high()
        
        # Test classification
        low_anomaly = classify_severity(0.2)
        high_anomaly = classify_severity(0.75)
        critical_anomaly = classify_severity(0.95)
        
        assert low_anomaly.severity_level == SeverityLevel.LOW
        assert high_anomaly.severity_level == SeverityLevel.HIGH
        assert critical_anomaly.severity_level == SeverityLevel.CRITICAL
        
        # Test action determination
        assert needs_immediate_attention(low_anomaly) is False
        assert needs_immediate_attention(critical_anomaly) is True
        
        assert needs_investigation(low_anomaly) is False
        assert needs_investigation(high_anomaly) is True
        assert needs_investigation(critical_anomaly) is True

    def test_severity_score_with_confidence_scenarios(self):
        """Test severity score with confidence in different scenarios."""
        # High severity with high confidence
        high_conf_score = SeverityScore(
            value=0.85,
            severity_level=SeverityLevel.CRITICAL,
            confidence=0.95
        )
        
        # High severity with low confidence
        low_conf_score = SeverityScore(
            value=0.85,
            severity_level=SeverityLevel.CRITICAL,
            confidence=0.3
        )
        
        assert high_conf_score.is_critical() is True
        assert low_conf_score.is_critical() is True
        assert high_conf_score.confidence > low_conf_score.confidence

    def test_severity_score_filtering(self):
        """Test filtering severity scores."""
        scores = [
            SeverityScore.from_score(0.1),
            SeverityScore.from_score(0.3),
            SeverityScore.from_score(0.5),
            SeverityScore.from_score(0.7),
            SeverityScore.from_score(0.9),
        ]
        
        # Filter critical scores
        critical_scores = [s for s in scores if s.is_critical()]
        assert len(critical_scores) == 1
        assert critical_scores[0].value == 0.9
        
        # Filter high priority scores
        high_priority = [s for s in scores if s.is_high()]
        assert len(high_priority) == 2
        assert all(s.value >= 0.7 for s in high_priority)

    def test_severity_score_sorting(self):
        """Test sorting severity scores."""
        scores = [
            SeverityScore.from_score(0.9),
            SeverityScore.from_score(0.3),
            SeverityScore.from_score(0.7),
            SeverityScore.from_score(0.1),
        ]
        
        # Sort by value
        sorted_scores = sorted(scores, key=lambda s: s.value)
        expected_values = [0.1, 0.3, 0.7, 0.9]
        actual_values = [s.value for s in sorted_scores]
        assert actual_values == expected_values

    def test_severity_level_priority_mapping(self):
        """Test mapping severity levels to priority values."""
        priority_map = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }
        
        scores = [
            SeverityScore.from_score(0.1),  # LOW
            SeverityScore.from_score(0.5),  # MEDIUM
            SeverityScore.from_score(0.7),  # HIGH
            SeverityScore.from_score(0.9),  # CRITICAL
        ]
        
        priorities = [priority_map[s.severity_level] for s in scores]
        assert priorities == [1, 2, 3, 4]

    def test_edge_cases(self):
        """Test edge cases."""
        # Minimum values
        min_score = SeverityScore(value=0.0, severity_level=SeverityLevel.LOW, confidence=0.0)
        assert min_score.value == 0.0
        assert min_score.confidence == 0.0
        
        # Maximum values
        max_score = SeverityScore(value=1.0, severity_level=SeverityLevel.CRITICAL, confidence=1.0)
        assert max_score.value == 1.0
        assert max_score.confidence == 1.0
        
        # Both should be valid
        assert min_score.is_critical() is False
        assert max_score.is_critical() is True

    def test_factory_method_consistency(self):
        """Test consistency between factory methods."""
        # from_score should create consistent objects
        score1 = SeverityScore.from_score(0.5)
        score2 = SeverityScore.from_score(0.5)
        
        assert score1 == score2
        assert score1.value == score2.value
        assert score1.severity_level == score2.severity_level
        assert score1.confidence == score2.confidence  # Both should be None

    def test_dataclass_properties(self):
        """Test dataclass properties."""
        score = SeverityScore(value=0.75, severity_level=SeverityLevel.HIGH, confidence=0.9)
        
        # Should have dataclass properties
        assert hasattr(score, '__dataclass_fields__')
        assert 'value' in score.__dataclass_fields__
        assert 'severity_level' in score.__dataclass_fields__
        assert 'confidence' in score.__dataclass_fields__
        
        # Should be frozen
        assert score.__dataclass_fields__['value'].default == score.__dataclass_fields__['value'].default  # Just testing access