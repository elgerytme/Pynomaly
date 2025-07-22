"""Quality Scores Value Objects.

Contains value objects for representing quality scores, trends, and related metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from decimal import Decimal


class ScoringMethod(Enum):
    """Methods for calculating quality scores."""
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    MINIMUM = "minimum"
    CUSTOM = "custom"


@dataclass(frozen=True)
class MonetaryAmount:
    """Monetary amount value object."""
    amount: Decimal
    currency: str = "USD"
    
    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"
    
    def __add__(self, other: 'MonetaryAmount') -> 'MonetaryAmount':
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return MonetaryAmount(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'MonetaryAmount') -> 'MonetaryAmount':
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency}")
        return MonetaryAmount(self.amount - other.amount, self.currency)
    
    def __mul__(self, multiplier: float) -> 'MonetaryAmount':
        return MonetaryAmount(self.amount * Decimal(str(multiplier)), self.currency)
    
    def to_float(self) -> float:
        return float(self.amount)


@dataclass(frozen=True)
class QualityScores:
    """Comprehensive quality scores across multiple dimensions."""
    
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    timeliness_score: float
    scoring_method: ScoringMethod
    weight_configuration: Dict[str, float] = field(default_factory=lambda: {
        'completeness': 0.20,
        'accuracy': 0.25,
        'consistency': 0.20,
        'validity': 0.20,
        'uniqueness': 0.10,
        'timeliness': 0.05
    })
    
    # Additional scoring details
    score_confidence: float = 1.0
    score_timestamp: datetime = field(default_factory=datetime.now)
    score_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate quality scores."""
        scores = [
            self.overall_score, self.completeness_score, self.accuracy_score,
            self.consistency_score, self.validity_score, self.uniqueness_score,
            self.timeliness_score
        ]
        
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Quality score must be between 0 and 1, got {score}")
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weight_configuration.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weight configuration must sum to 1.0, got {weight_sum}")
        
        # Validate score confidence
        if not (0.0 <= self.score_confidence <= 1.0):
            raise ValueError(f"Score confidence must be between 0 and 1, got {self.score_confidence}")
    
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
        """Calculate weighted overall score."""
        if self.scoring_method == ScoringMethod.WEIGHTED_AVERAGE:
            return self.overall_score
        
        dimension_scores = self.get_dimension_scores()
        return sum(score * self.weight_configuration.get(dimension, 0.0) 
                  for dimension, score in dimension_scores.items())
    
    def get_lowest_dimension(self) -> tuple[str, float]:
        """Get the dimension with the lowest score."""
        dimension_scores = self.get_dimension_scores()
        return min(dimension_scores.items(), key=lambda x: x[1])
    
    def get_highest_dimension(self) -> tuple[str, float]:
        """Get the dimension with the highest score."""
        dimension_scores = self.get_dimension_scores()
        return max(dimension_scores.items(), key=lambda x: x[1])
    
    def get_quality_grade(self) -> str:
        """Get letter grade for overall quality score."""
        if self.overall_score >= 0.95:
            return 'A+'
        elif self.overall_score >= 0.90:
            return 'A'
        elif self.overall_score >= 0.85:
            return 'B+'
        elif self.overall_score >= 0.80:
            return 'B'
        elif self.overall_score >= 0.75:
            return 'C+'
        elif self.overall_score >= 0.70:
            return 'C'
        elif self.overall_score >= 0.65:
            return 'D+'
        elif self.overall_score >= 0.60:
            return 'D'
        else:
            return 'F'
    
    def get_quality_interpretation(self) -> str:
        """Get interpretation of quality score."""
        if self.overall_score >= 0.95:
            return "Excellent - Data is production-ready with minimal issues"
        elif self.overall_score >= 0.85:
            return "Good - Data has minor issues that should be addressed"
        elif self.overall_score >= 0.75:
            return "Moderate - Data has noticeable issues requiring attention"
        elif self.overall_score >= 0.60:
            return "Poor - Data has significant issues that need resolution"
        else:
            return "Very Poor - Data requires extensive cleanup before use"
    
    def compare_with(self, other: 'QualityScores') -> Dict[str, float]:
        """Compare quality scores with another set of scores."""
        return {
            'overall_delta': self.overall_score - other.overall_score,
            'completeness_delta': self.completeness_score - other.completeness_score,
            'accuracy_delta': self.accuracy_score - other.accuracy_score,
            'consistency_delta': self.consistency_score - other.consistency_score,
            'validity_delta': self.validity_score - other.validity_score,
            'uniqueness_delta': self.uniqueness_score - other.uniqueness_score,
            'timeliness_delta': self.timeliness_score - other.timeliness_score
        }
    
    def with_updated_weights(self, new_weights: Dict[str, float]) -> 'QualityScores':
        """Create new quality scores with updated weights."""
        return QualityScores(
            overall_score=self._calculate_weighted_score(new_weights),
            completeness_score=self.completeness_score,
            accuracy_score=self.accuracy_score,
            consistency_score=self.consistency_score,
            validity_score=self.validity_score,
            uniqueness_score=self.uniqueness_score,
            timeliness_score=self.timeliness_score,
            scoring_method=self.scoring_method,
            weight_configuration=new_weights,
            score_confidence=self.score_confidence,
            score_timestamp=self.score_timestamp,
            score_metadata=self.score_metadata
        )
    
    def _calculate_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted overall score with given weights."""
        dimension_scores = self.get_dimension_scores()
        return sum(score * weights.get(dimension, 0.0) 
                  for dimension, score in dimension_scores.items())


@dataclass(frozen=True)
class QualityTrendPoint:
    """A single point in quality trend analysis."""
    timestamp: datetime
    overall_score: float
    dimension_scores: Dict[str, float]
    record_count: Optional[int] = None
    assessment_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate trend point."""
        if not (0.0 <= self.overall_score <= 1.0):
            raise ValueError(f"Overall score must be between 0 and 1, got {self.overall_score}")
        
        for dimension, score in self.dimension_scores.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Dimension score for {dimension} must be between 0 and 1, got {score}")


@dataclass(frozen=True)
class QualityTrends:
    """Quality trends analysis over time."""
    
    trend_points: List[QualityTrendPoint]
    trend_period_days: int = 30
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    trend_strength: float = 0.0  # -1.0 to 1.0
    
    # Trend statistics
    average_score: float = 0.0
    score_variance: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    # Trend metadata
    trend_analysis_date: datetime = field(default_factory=datetime.now)
    trend_confidence: float = 1.0
    trend_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate trend statistics."""
        if self.trend_points:
            scores = [point.overall_score for point in self.trend_points]
            object.__setattr__(self, 'average_score', sum(scores) / len(scores))
            object.__setattr__(self, 'min_score', min(scores))
            object.__setattr__(self, 'max_score', max(scores))
            
            # Calculate variance
            avg = self.average_score
            variance = sum((score - avg) ** 2 for score in scores) / len(scores)
            object.__setattr__(self, 'score_variance', variance)
            
            # Calculate trend direction and strength
            if len(scores) > 1:
                first_half = scores[:len(scores)//2]
                second_half = scores[len(scores)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                delta = second_avg - first_avg
                object.__setattr__(self, 'trend_strength', delta)
                
                if abs(delta) < 0.01:  # Threshold for stable
                    object.__setattr__(self, 'trend_direction', 'stable')
                elif delta > 0:
                    object.__setattr__(self, 'trend_direction', 'improving')
                else:
                    object.__setattr__(self, 'trend_direction', 'degrading')
    
    def get_latest_score(self) -> Optional[float]:
        """Get the most recent quality score."""
        if self.trend_points:
            return max(self.trend_points, key=lambda p: p.timestamp).overall_score
        return None
    
    def get_score_change(self) -> Optional[float]:
        """Get change in score from first to last point."""
        if len(self.trend_points) < 2:
            return None
        
        sorted_points = sorted(self.trend_points, key=lambda p: p.timestamp)
        return sorted_points[-1].overall_score - sorted_points[0].overall_score
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of trend analysis."""
        return {
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'average_score': self.average_score,
            'score_variance': self.score_variance,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'latest_score': self.get_latest_score(),
            'score_change': self.get_score_change(),
            'data_points': len(self.trend_points),
            'trend_confidence': self.trend_confidence,
            'analysis_date': self.trend_analysis_date.isoformat()
        }
    
    def add_trend_point(self, point: QualityTrendPoint) -> 'QualityTrends':
        """Add a new trend point and recalculate trends."""
        new_points = self.trend_points + [point]
        
        # Keep only points within trend period
        cutoff_date = datetime.now() - timedelta(days=self.trend_period_days)
        filtered_points = [p for p in new_points if p.timestamp >= cutoff_date]
        
        return QualityTrends(
            trend_points=filtered_points,
            trend_period_days=self.trend_period_days,
            trend_analysis_date=datetime.now(),
            trend_confidence=self.trend_confidence,
            trend_metadata=self.trend_metadata
        )
    
    def get_dimension_trends(self) -> Dict[str, List[float]]:
        """Get trends for each quality dimension."""
        dimension_trends = {}
        
        for point in sorted(self.trend_points, key=lambda p: p.timestamp):
            for dimension, score in point.dimension_scores.items():
                if dimension not in dimension_trends:
                    dimension_trends[dimension] = []
                dimension_trends[dimension].append(score)
        
        return dimension_trends
    
    def get_volatility_score(self) -> float:
        """Calculate volatility score (higher = more volatile)."""
        if len(self.trend_points) < 2:
            return 0.0
        
        # Calculate standard deviation of scores
        import math
        return math.sqrt(self.score_variance)
    
    def is_improving(self) -> bool:
        """Check if quality is improving."""
        return self.trend_direction == 'improving'
    
    def is_degrading(self) -> bool:
        """Check if quality is degrading."""
        return self.trend_direction == 'degrading'
    
    def is_stable(self) -> bool:
        """Check if quality is stable."""
        return self.trend_direction == 'stable'


from datetime import timedelta