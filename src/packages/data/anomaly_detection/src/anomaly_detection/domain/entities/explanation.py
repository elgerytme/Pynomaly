"""Explanation entity for anomaly detection results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import uuid


class ExplanationType(str, Enum):
    """Types of explanations."""
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP = "shap"
    LIME = "lime"
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"


class ExplanationMethod(str, Enum):
    """Explanation generation methods."""
    ISOLATION_PATH = "isolation_path"
    FEATURE_DEVIATION = "feature_deviation"
    CLUSTERING_DISTANCE = "clustering_distance"
    ENSEMBLE_VOTING = "ensemble_voting"
    SHAPLEY_VALUES = "shapley_values"


@dataclass
class FeatureContribution:
    """Individual feature contribution to anomaly score."""
    feature_name: str
    value: float
    contribution: float
    rank: int
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if not self.feature_name:
            raise ValueError("Feature name cannot be empty")
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("Rank must be a positive integer")


@dataclass
class Explanation:
    """Explanation for anomaly detection result."""
    
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_id: Optional[str] = None
    explanation_type: ExplanationType = ExplanationType.FEATURE_IMPORTANCE
    method: ExplanationMethod = ExplanationMethod.FEATURE_DEVIATION
    
    # Main explanation content
    feature_contributions: List[FeatureContribution] = field(default_factory=list)
    global_explanation: Optional[Dict[str, Any]] = None
    local_explanation: Optional[Dict[str, Any]] = None
    
    # Metadata
    confidence_score: Optional[float] = None
    algorithm_used: Optional[str] = None
    model_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Human-readable content
    summary: Optional[str] = None
    detailed_description: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    
    # Technical details
    raw_scores: Optional[Dict[str, float]] = None
    threshold_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.confidence_score is not None:
            if not 0.0 <= self.confidence_score <= 1.0:
                raise ValueError("Confidence score must be between 0.0 and 1.0")
    
    @property
    def top_features(self) -> List[FeatureContribution]:
        """Get top contributing features sorted by contribution magnitude."""
        return sorted(
            self.feature_contributions,
            key=lambda x: abs(x.contribution),
            reverse=True
        )
    
    @property
    def is_complete(self) -> bool:
        """Check if explanation has sufficient information."""
        return (
            len(self.feature_contributions) > 0 or
            self.global_explanation is not None or
            self.local_explanation is not None or
            self.summary is not None
        )
    
    def get_top_n_features(self, n: int = 5) -> List[FeatureContribution]:
        """Get top N contributing features."""
        return self.top_features[:n]
    
    def add_feature_contribution(
        self,
        feature_name: str,
        value: float,
        contribution: float,
        confidence: Optional[float] = None
    ) -> None:
        """Add a feature contribution to the explanation."""
        rank = len(self.feature_contributions) + 1
        feature_contrib = FeatureContribution(
            feature_name=feature_name,
            value=value,
            contribution=contribution,
            rank=rank,
            confidence=confidence
        )
        self.feature_contributions.append(feature_contrib)
    
    def set_summary(self, summary: str) -> None:
        """Set human-readable summary."""
        self.summary = summary
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        if recommendation and recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "anomaly_id": self.anomaly_id,
            "explanation_type": self.explanation_type.value,
            "method": self.method.value,
            "feature_contributions": [
                {
                    "feature_name": fc.feature_name,
                    "value": fc.value,
                    "contribution": fc.contribution,
                    "rank": fc.rank,
                    "confidence": fc.confidence
                }
                for fc in self.feature_contributions
            ],
            "global_explanation": self.global_explanation,
            "local_explanation": self.local_explanation,
            "confidence_score": self.confidence_score,
            "algorithm_used": self.algorithm_used,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat(),
            "summary": self.summary,
            "detailed_description": self.detailed_description,
            "recommendations": self.recommendations,
            "raw_scores": self.raw_scores,
            "threshold_info": self.threshold_info,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Explanation":
        """Create explanation from dictionary."""
        feature_contributions = [
            FeatureContribution(
                feature_name=fc["feature_name"],
                value=fc["value"],
                contribution=fc["contribution"],
                rank=fc["rank"],
                confidence=fc.get("confidence")
            )
            for fc in data.get("feature_contributions", [])
        ]
        
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow()
        
        return cls(
            explanation_id=data.get("explanation_id", str(uuid.uuid4())),
            anomaly_id=data.get("anomaly_id"),
            explanation_type=ExplanationType(data.get("explanation_type", ExplanationType.FEATURE_IMPORTANCE)),
            method=ExplanationMethod(data.get("method", ExplanationMethod.FEATURE_DEVIATION)),
            feature_contributions=feature_contributions,
            global_explanation=data.get("global_explanation"),
            local_explanation=data.get("local_explanation"),
            confidence_score=data.get("confidence_score"),
            algorithm_used=data.get("algorithm_used"),
            model_version=data.get("model_version"),
            created_at=created_at,
            summary=data.get("summary"),
            detailed_description=data.get("detailed_description"),
            recommendations=data.get("recommendations", []),
            raw_scores=data.get("raw_scores"),
            threshold_info=data.get("threshold_info"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def create_simple(
        cls,
        anomaly_id: str,
        summary: str,
        algorithm_used: str,
        confidence_score: Optional[float] = None
    ) -> "Explanation":
        """Create a simple explanation with minimal information."""
        return cls(
            anomaly_id=anomaly_id,
            summary=summary,
            algorithm_used=algorithm_used,
            confidence_score=confidence_score
        )