"""Tests for Quality Scores value object."""

import pytest

from ..domain.value_objects.quality_scores import QualityScores, ScoringMethod


class TestQualityScores:
    """Test cases for Quality Scores value object."""
    
    def test_create_quality_scores(self):
        """Test creating quality scores with valid values."""
        scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75
        )
        
        assert scores.overall_score == 0.85
        assert scores.completeness_score == 0.90
        assert scores.get_quality_grade() == "B+"
        assert scores.is_acceptable(threshold=0.80) is True
    
    def test_invalid_score_values(self):
        """Test that invalid score values raise errors."""
        with pytest.raises(ValueError):
            QualityScores(
                overall_score=1.5,  # Invalid: > 1.0
                completeness_score=0.90,
                accuracy_score=0.85,
                consistency_score=0.80,
                validity_score=0.85,
                uniqueness_score=0.95,
                timeliness_score=0.75
            )
        
        with pytest.raises(ValueError):
            QualityScores(
                overall_score=0.85,
                completeness_score=-0.1,  # Invalid: < 0.0
                accuracy_score=0.85,
                consistency_score=0.80,
                validity_score=0.85,
                uniqueness_score=0.95,
                timeliness_score=0.75
            )
    
    def test_default_weights(self):
        """Test default weight configuration."""
        scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75
        )
        
        # Check that weights sum to 1.0
        total_weight = sum(scores.weight_configuration.values())
        assert abs(total_weight - 1.0) < 0.001
        
        # Check specific weights
        assert scores.weight_configuration['completeness'] == 0.20
        assert scores.weight_configuration['accuracy'] == 0.25
    
    def test_custom_weights(self):
        """Test custom weight configuration."""
        custom_weights = {
            'completeness': 0.30,
            'accuracy': 0.30,
            'consistency': 0.15,
            'validity': 0.15,
            'uniqueness': 0.05,
            'timeliness': 0.05
        }
        
        scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75,
            weight_configuration=custom_weights
        )
        
        assert scores.weight_configuration['completeness'] == 0.30
        assert scores.weight_configuration['accuracy'] == 0.30
    
    def test_invalid_weights(self):
        """Test that invalid weights raise errors."""
        invalid_weights = {
            'completeness': 0.50,
            'accuracy': 0.30,
            'consistency': 0.15,
            'validity': 0.15,
            'uniqueness': 0.05,
            'timeliness': 0.05
        }  # Sum = 1.20, invalid
        
        with pytest.raises(ValueError):
            QualityScores(
                overall_score=0.85,
                completeness_score=0.90,
                accuracy_score=0.85,
                consistency_score=0.80,
                validity_score=0.85,
                uniqueness_score=0.95,
                timeliness_score=0.75,
                weight_configuration=invalid_weights
            )
    
    def test_get_dimension_scores(self):
        """Test getting dimension scores as dictionary."""
        scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75
        )
        
        dimension_scores = scores.get_dimension_scores()
        
        assert dimension_scores['completeness'] == 0.90
        assert dimension_scores['accuracy'] == 0.85
        assert dimension_scores['uniqueness'] == 0.95
        assert len(dimension_scores) == 6
    
    def test_get_weighted_score(self):
        """Test weighted score calculation."""
        scores = QualityScores(
            overall_score=0.85,  # This will be ignored in weighted calculation
            completeness_score=0.90,
            accuracy_score=0.80,
            consistency_score=0.85,
            validity_score=0.75,
            uniqueness_score=1.0,
            timeliness_score=0.70
        )
        
        weighted_score = scores.get_weighted_score()
        
        # Manual calculation with default weights:
        # 0.90*0.20 + 0.80*0.25 + 0.85*0.15 + 0.75*0.20 + 1.0*0.10 + 0.70*0.10
        expected = 0.90*0.20 + 0.80*0.25 + 0.85*0.15 + 0.75*0.20 + 1.0*0.10 + 0.70*0.10
        
        assert abs(weighted_score - expected) < 0.001
    
    def test_get_failing_dimensions(self):
        """Test identifying failing dimensions."""
        scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,  # Above threshold
            accuracy_score=0.75,     # Below threshold
            consistency_score=0.85,  # Above threshold
            validity_score=0.70,     # Below threshold
            uniqueness_score=0.95,   # Above threshold
            timeliness_score=0.65    # Below threshold
        )
        
        failing = scores.get_failing_dimensions(threshold=0.80)
        
        assert 'accuracy' in failing
        assert 'validity' in failing
        assert 'timeliness' in failing
        assert 'completeness' not in failing
        assert 'uniqueness' not in failing
        assert len(failing) == 3
    
    def test_get_quality_grade(self):
        """Test quality grade assignment."""
        # Test A+ grade
        excellent_scores = QualityScores(
            overall_score=0.97,
            completeness_score=0.95,
            accuracy_score=0.98,
            consistency_score=0.96,
            validity_score=0.99,
            uniqueness_score=1.0,
            timeliness_score=0.95
        )
        assert excellent_scores.get_quality_grade() == "A+"
        
        # Test B grade
        good_scores = QualityScores(
            overall_score=0.82,
            completeness_score=0.85,
            accuracy_score=0.80,
            consistency_score=0.85,
            validity_score=0.80,
            uniqueness_score=0.85,
            timeliness_score=0.75
        )
        assert good_scores.get_quality_grade() == "B"
        
        # Test F grade
        poor_scores = QualityScores(
            overall_score=0.45,
            completeness_score=0.50,
            accuracy_score=0.40,
            consistency_score=0.45,
            validity_score=0.40,
            uniqueness_score=0.50,
            timeliness_score=0.45
        )
        assert poor_scores.get_quality_grade() == "F"
    
    def test_is_acceptable(self):
        """Test acceptability threshold checking."""
        acceptable_scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75
        )
        
        assert acceptable_scores.is_acceptable(threshold=0.80) is True
        assert acceptable_scores.is_acceptable(threshold=0.90) is False
    
    def test_compare_with(self):
        """Test comparing two quality score instances."""
        scores1 = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75
        )
        
        scores2 = QualityScores(
            overall_score=0.80,
            completeness_score=0.85,
            accuracy_score=0.80,
            consistency_score=0.75,
            validity_score=0.80,
            uniqueness_score=0.90,
            timeliness_score=0.70
        )
        
        comparison = scores1.compare_with(scores2)
        
        assert comparison['overall'] == 0.05  # 0.85 - 0.80
        assert comparison['completeness'] == 0.05  # 0.90 - 0.85
        assert comparison['accuracy'] == 0.05  # 0.85 - 0.80
        assert comparison['uniqueness'] == 0.05  # 0.95 - 0.90
    
    def test_scoring_methods(self):
        """Test different scoring methods."""
        # Test weighted average (default)
        weighted_scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75,
            scoring_method=ScoringMethod.WEIGHTED_AVERAGE
        )
        
        assert weighted_scores.scoring_method == ScoringMethod.WEIGHTED_AVERAGE
        
        # Test simple average
        simple_scores = QualityScores(
            overall_score=0.85,
            completeness_score=0.90,
            accuracy_score=0.85,
            consistency_score=0.80,
            validity_score=0.85,
            uniqueness_score=0.95,
            timeliness_score=0.75,
            scoring_method=ScoringMethod.SIMPLE_AVERAGE
        )
        
        assert simple_scores.scoring_method == ScoringMethod.SIMPLE_AVERAGE