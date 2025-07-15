"""Quality scores value object for data quality assessment."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ScoringMethod(str, Enum):
    """Quality scoring method enumeration."""
    WEIGHTED_AVERAGE = "weighted_average"
    SIMPLE_AVERAGE = "simple_average"
    MINIMUM_SCORE = "minimum_score"
    CUSTOM = "custom"


@dataclass(frozen=True)
class QualityScores:
    """Composite quality scores across multiple dimensions."""
    
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    timeliness_score: float
    scoring_method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE
    weight_configuration: Dict[str, float] = None
    
    def __post_init__(self):
        """Validate score values and weights."""
        # Validate score ranges
        scores = [
            self.overall_score,
            self.completeness_score,
            self.accuracy_score,
            self.consistency_score,
            self.validity_score,
            self.uniqueness_score,
            self.timeliness_score
        ]
        
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score must be between 0.0 and 1.0, got {score}")
        
        # Set default weights if not provided
        if self.weight_configuration is None:
            object.__setattr__(self, 'weight_configuration', {
                'completeness': 0.20,
                'accuracy': 0.25,
                'consistency': 0.15,
                'validity': 0.20,
                'uniqueness': 0.10,
                'timeliness': 0.10
            })
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weight_configuration.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def get_dimension_scores(self) -> Dict[str, float]:
        """Get all dimension scores as a dictionary."""
        return {
            'completeness': self.completeness_score,
            'accuracy': self.accuracy_score,
            'consistency': self.consistency_score,
            'validity': self.validity_score,
            'uniqueness': self.uniqueness_score,
            'timeliness': self.timeliness_score
        }
    
    def get_weighted_score(self) -> float:
        """Calculate weighted average score."""
        dimension_scores = self.get_dimension_scores()
        weighted_sum = sum(
            score * self.weight_configuration.get(dimension, 0.0)
            for dimension, score in dimension_scores.items()
        )
        return round(weighted_sum, 4)
    
    def get_failing_dimensions(self, threshold: float = 0.8) -> Dict[str, float]:
        """Get dimensions that fall below the threshold."""
        dimension_scores = self.get_dimension_scores()
        return {
            dimension: score
            for dimension, score in dimension_scores.items()
            if score < threshold
        }
    
    def get_quality_grade(self) -> str:
        """Get quality grade based on overall score."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "B+"
        elif self.overall_score >= 0.80:
            return "B"
        elif self.overall_score >= 0.75:
            return "C+"
        elif self.overall_score >= 0.70:
            return "C"
        elif self.overall_score >= 0.60:
            return "D"
        else:
            return "F"
    
    def is_acceptable(self, threshold: float = 0.80) -> bool:
        """Check if overall quality is acceptable."""
        return self.overall_score >= threshold
    
    def compare_with(self, other: 'QualityScores') -> Dict[str, float]:
        """Compare scores with another QualityScores instance."""
        dimension_scores = self.get_dimension_scores()
        other_scores = other.get_dimension_scores()
        
        comparison = {}
        for dimension in dimension_scores:
            if dimension in other_scores:
                comparison[dimension] = dimension_scores[dimension] - other_scores[dimension]
        
        comparison['overall'] = self.overall_score - other.overall_score
        return comparison