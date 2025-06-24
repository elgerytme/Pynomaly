"""Domain entities for explainable AI (XAI) framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import numpy as np


class ExplanationMethod(Enum):
    """Methods for generating explanations."""
    SHAP_TREE = "shap_tree"
    SHAP_KERNEL = "shap_kernel"
    SHAP_DEEP = "shap_deep"
    SHAP_LINEAR = "shap_linear"
    SHAP_GRADIENT = "shap_gradient"
    LIME = "lime"
    LIME_TABULAR = "lime_tabular"
    LIME_TEXT = "lime_text"
    LIME_IMAGE = "lime_image"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_ABLATION = "feature_ablation"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRAD_CAM = "grad_cam"
    COUNTERFACTUAL = "counterfactual"
    ANCHORS = "anchors"
    PROTOTYPES = "prototypes"


class ExplanationType(Enum):
    """Type of explanation being generated."""
    FEATURE_IMPORTANCE = "feature_importance"
    FEATURE_ATTRIBUTION = "feature_attribution"
    COUNTERFACTUAL = "counterfactual"
    EXEMPLAR = "exemplar"
    RULE_BASED = "rule_based"
    PROTOTYPE = "prototype"
    CONCEPT_ACTIVATION = "concept_activation"
    ATTENTION_WEIGHTS = "attention_weights"
    SALIENCY_MAP = "saliency_map"
    INTERACTION_EFFECTS = "interaction_effects"


class ExplanationScope(Enum):
    """Scope of explanation."""
    LOCAL = "local"              # Single instance explanation
    GLOBAL = "global"            # Entire model explanation
    COHORT = "cohort"           # Group of similar instances
    FEATURE = "feature"         # Single feature analysis
    INTERACTION = "interaction"  # Feature interaction analysis
    TEMPORAL = "temporal"       # Time-based explanation
    COUNTERFACTUAL = "counterfactual"  # What-if analysis


class ExplanationAudience(Enum):
    """Target audience for explanation."""
    TECHNICAL = "technical"      # Data scientists, ML engineers
    BUSINESS = "business"        # Business stakeholders
    REGULATORY = "regulatory"    # Compliance, auditors
    END_USER = "end_user"       # End users of the system
    DOMAIN_EXPERT = "domain_expert"  # Subject matter experts


class TrustLevel(Enum):
    """Level of trust in explanation."""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.6 - 0.8
    LOW = "low"         # 0.4 - 0.6
    VERY_LOW = "very_low"  # < 0.4


class BiasType(Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUALITY_OF_OPPORTUNITY = "equality_of_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"
    TREATMENT_EQUALITY = "treatment_equality"


@dataclass
class FeatureImportance:
    """Importance score for a single feature."""
    feature_name: str
    importance_value: float
    importance_type: str = "shap_value"  # shap_value, lime_coefficient, permutation, etc.
    confidence: float = 1.0
    rank: int = 0
    normalized_importance: Optional[float] = None
    feature_value: Optional[float] = None
    contribution_direction: str = "positive"  # positive, negative, neutral
    statistical_significance: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute derived fields."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.rank < 0:
            raise ValueError("Rank must be non-negative")
        
        # Determine contribution direction
        if self.importance_value > 0:
            self.contribution_direction = "positive"
        elif self.importance_value < 0:
            self.contribution_direction = "negative"
        else:
            self.contribution_direction = "neutral"
    
    def get_absolute_importance(self) -> float:
        """Get absolute importance value."""
        return abs(self.importance_value)
    
    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if feature importance is statistically significant."""
        if self.statistical_significance is None:
            return True  # Assume significant if not tested
        return self.statistical_significance < threshold
    
    def get_importance_category(self) -> str:
        """Categorize importance magnitude."""
        abs_importance = self.get_absolute_importance()
        if abs_importance >= 0.3:
            return "high"
        elif abs_importance >= 0.1:
            return "medium"
        elif abs_importance >= 0.01:
            return "low"
        else:
            return "negligible"


@dataclass
class ExplanationMetadata:
    """Metadata about explanation generation process."""
    generation_timestamp: datetime = field(default_factory=datetime.utcnow)
    generation_time_seconds: float = 0.0
    explanation_confidence: float = 1.0
    model_version: str = "1.0.0"
    explanation_version: str = "1.0.0"
    data_sample_size: int = 0
    feature_coverage: float = 1.0  # Fraction of features explained
    method_parameters: Dict[str, Any] = field(default_factory=dict)
    computational_resources: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metadata."""
        if not (0.0 <= self.explanation_confidence <= 1.0):
            raise ValueError("Explanation confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.feature_coverage <= 1.0):
            raise ValueError("Feature coverage must be between 0.0 and 1.0")
        if self.generation_time_seconds < 0:
            raise ValueError("Generation time must be non-negative")
    
    def is_high_quality(self, confidence_threshold: float = 0.8) -> bool:
        """Check if explanation meets high quality criteria."""
        return (self.explanation_confidence >= confidence_threshold and
                self.feature_coverage >= 0.9)
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score based on time and resources."""
        # Simple efficiency metric (higher is better)
        if self.generation_time_seconds <= 0:
            return 1.0
        
        # Normalize by sample size and feature coverage
        efficiency = (self.feature_coverage * max(1, self.data_sample_size)) / self.generation_time_seconds
        return min(1.0, efficiency / 1000)  # Normalize to 0-1 range


@dataclass
class ModelExplanation:
    """Base class for model explanations."""
    explanation_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP_TREE
    explanation_type: ExplanationType = ExplanationType.FEATURE_IMPORTANCE
    explanation_scope: ExplanationScope = ExplanationScope.LOCAL
    target_audience: ExplanationAudience = ExplanationAudience.TECHNICAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Optional[ExplanationMetadata] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = ExplanationMetadata()
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of explanation."""
        return {
            "explanation_id": str(self.explanation_id),
            "model_id": str(self.model_id),
            "method": self.explanation_method.value,
            "type": self.explanation_type.value,
            "scope": self.explanation_scope.value,
            "audience": self.target_audience.value,
            "created_at": self.created_at.isoformat(),
            "quality": self.metadata.is_high_quality() if self.metadata else False
        }


@dataclass
class InstanceExplanation(ModelExplanation):
    """Explanation for a single instance/prediction."""
    instance_id: str = ""
    input_features: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    prediction_value: Any = None
    prediction_confidence: float = 0.0
    feature_importances: List[FeatureImportance] = field(default_factory=list)
    base_value: float = 0.0  # Base/expected value for additive explanations
    local_fidelity_score: float = 1.0
    counterfactual_examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate instance explanation."""
        super().__post_init__()
        if not (0.0 <= self.prediction_confidence <= 1.0):
            raise ValueError("Prediction confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.local_fidelity_score <= 1.0):
            raise ValueError("Local fidelity score must be between 0.0 and 1.0")
        
        # Sort feature importances by absolute value
        if self.feature_importances:
            self.feature_importances.sort(key=lambda x: abs(x.importance_value), reverse=True)
            # Update ranks
            for i, importance in enumerate(self.feature_importances):
                importance.rank = i + 1
    
    def get_top_features(self, n: int = 5) -> List[FeatureImportance]:
        """Get top N most important features."""
        return self.feature_importances[:n]
    
    def get_positive_contributions(self) -> List[FeatureImportance]:
        """Get features that contribute positively to prediction."""
        return [f for f in self.feature_importances if f.importance_value > 0]
    
    def get_negative_contributions(self) -> List[FeatureImportance]:
        """Get features that contribute negatively to prediction."""
        return [f for f in self.feature_importances if f.importance_value < 0]
    
    def calculate_explanation_completeness(self) -> float:
        """Calculate how complete the explanation is."""
        if not self.feature_importances or self.prediction_value is None:
            return 0.0
        
        # For additive explanations, completeness = sum of attributions + base_value ≈ prediction
        total_attribution = sum(f.importance_value for f in self.feature_importances)
        explained_value = self.base_value + total_attribution
        
        if isinstance(self.prediction_value, (int, float)) and self.prediction_value != 0:
            completeness = min(1.0, abs(explained_value / self.prediction_value))
        else:
            completeness = 1.0 if abs(explained_value) < 0.01 else 0.5
        
        return completeness
    
    def get_feature_contribution_summary(self) -> Dict[str, Any]:
        """Get summary of feature contributions."""
        positive = self.get_positive_contributions()
        negative = self.get_negative_contributions()
        
        return {
            "total_features": len(self.feature_importances),
            "positive_contributors": len(positive),
            "negative_contributors": len(negative),
            "top_positive": positive[:3] if positive else [],
            "top_negative": negative[:3] if negative else [],
            "explanation_completeness": self.calculate_explanation_completeness(),
            "prediction_value": self.prediction_value,
            "base_value": self.base_value
        }


@dataclass
class GlobalExplanation(ModelExplanation):
    """Global explanation for entire model behavior."""
    global_feature_importances: List[FeatureImportance] = field(default_factory=list)
    feature_interactions: Dict[str, Any] = field(default_factory=dict)
    model_behavior_summary: Dict[str, Any] = field(default_factory=dict)
    data_coverage: float = 1.0  # Fraction of data used for explanation
    bias_analysis: Optional[BiasAnalysis] = None
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    decision_boundaries: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate global explanation."""
        super().__post_init__()
        if not (0.0 <= self.data_coverage <= 1.0):
            raise ValueError("Data coverage must be between 0.0 and 1.0")
        
        # Sort global feature importances
        if self.global_feature_importances:
            self.global_feature_importances.sort(key=lambda x: abs(x.importance_value), reverse=True)
            for i, importance in enumerate(self.global_feature_importances):
                importance.rank = i + 1
    
    def get_most_important_features(self, n: int = 10) -> List[FeatureImportance]:
        """Get N most important features globally."""
        return self.global_feature_importances[:n]
    
    def get_feature_stability_score(self) -> float:
        """Calculate stability of feature importances."""
        if not self.global_feature_importances:
            return 0.0
        
        # Calculate coefficient of variation of importances
        importances = [abs(f.importance_value) for f in self.global_feature_importances]
        if not importances:
            return 0.0
        
        mean_importance = np.mean(importances)
        std_importance = np.std(importances)
        
        if mean_importance == 0:
            return 1.0
        
        cv = std_importance / mean_importance
        stability = 1.0 / (1.0 + cv)  # Higher CV = lower stability
        return min(1.0, stability)
    
    def has_bias_issues(self) -> bool:
        """Check if model has bias issues."""
        if self.bias_analysis:
            return self.bias_analysis.bias_detected
        return False
    
    def get_global_summary(self) -> Dict[str, Any]:
        """Get comprehensive global summary."""
        return {
            "total_features_analyzed": len(self.global_feature_importances),
            "top_features": [f.feature_name for f in self.get_most_important_features(5)],
            "feature_stability": self.get_feature_stability_score(),
            "data_coverage": self.data_coverage,
            "has_bias": self.has_bias_issues(),
            "fairness_scores": self.fairness_metrics,
            "model_behavior": self.model_behavior_summary
        }


@dataclass
class TrustScore:
    """Trust and confidence metrics for explanations."""
    overall_trust_score: float = 0.0
    consistency_score: float = 0.0
    stability_score: float = 0.0
    fidelity_score: float = 0.0
    completeness_score: float = 0.0
    robustness_score: float = 0.0
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    trust_level: TrustLevel = TrustLevel.MEDIUM
    trust_factors: Dict[str, float] = field(default_factory=dict)
    uncertainty_quantification: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute trust level."""
        # Validate scores
        scores = [
            self.overall_trust_score, self.consistency_score, self.stability_score,
            self.fidelity_score, self.completeness_score, self.robustness_score
        ]
        
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError("All trust scores must be between 0.0 and 1.0")
        
        # Compute overall trust if not provided
        if self.overall_trust_score == 0.0:
            self.overall_trust_score = self._compute_overall_trust()
        
        # Determine trust level
        self.trust_level = self._determine_trust_level()
    
    def _compute_overall_trust(self) -> float:
        """Compute overall trust score from components."""
        components = [
            self.consistency_score, self.stability_score, self.fidelity_score,
            self.completeness_score, self.robustness_score
        ]
        
        # Weighted average (can be customized)
        weights = [0.25, 0.25, 0.25, 0.15, 0.10]
        weighted_sum = sum(score * weight for score, weight in zip(components, weights))
        
        return min(1.0, weighted_sum)
    
    def _determine_trust_level(self) -> TrustLevel:
        """Determine trust level from overall score."""
        if self.overall_trust_score >= 0.8:
            return TrustLevel.HIGH
        elif self.overall_trust_score >= 0.6:
            return TrustLevel.MEDIUM
        elif self.overall_trust_score >= 0.4:
            return TrustLevel.LOW
        else:
            return TrustLevel.VERY_LOW
    
    def is_trustworthy(self, threshold: float = 0.7) -> bool:
        """Check if explanation is trustworthy."""
        return self.overall_trust_score >= threshold
    
    def get_trust_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of trust components."""
        return {
            "overall_trust": self.overall_trust_score,
            "trust_level": self.trust_level.value,
            "components": {
                "consistency": self.consistency_score,
                "stability": self.stability_score,
                "fidelity": self.fidelity_score,
                "completeness": self.completeness_score,
                "robustness": self.robustness_score
            },
            "confidence_interval": self.confidence_interval,
            "is_trustworthy": self.is_trustworthy(),
            "trust_factors": self.trust_factors,
            "uncertainty": self.uncertainty_quantification
        }


@dataclass
class BiasAnalysis:
    """Analysis of bias in model explanations."""
    analysis_id: UUID = field(default_factory=uuid4)
    overall_bias_score: float = 0.0
    bias_detected: bool = False
    protected_attribute_bias: Dict[str, float] = field(default_factory=dict)
    bias_types_detected: List[BiasType] = field(default_factory=list)
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    bias_sources: List[str] = field(default_factory=list)
    mitigation_recommendations: List[str] = field(default_factory=list)
    intersectional_bias: Dict[str, float] = field(default_factory=dict)
    temporal_bias_trends: Dict[str, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate bias analysis."""
        if not (0.0 <= self.overall_bias_score <= 1.0):
            raise ValueError("Overall bias score must be between 0.0 and 1.0")
        
        # Detect bias if score is high
        if self.overall_bias_score > 0.3:
            self.bias_detected = True
    
    def get_bias_severity(self) -> str:
        """Get qualitative bias severity."""
        if self.overall_bias_score < 0.1:
            return "minimal"
        elif self.overall_bias_score < 0.3:
            return "low"
        elif self.overall_bias_score < 0.6:
            return "moderate"
        else:
            return "high"
    
    def get_most_biased_attributes(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get most biased protected attributes."""
        sorted_bias = sorted(
            self.protected_attribute_bias.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_bias[:n]
    
    def requires_immediate_attention(self) -> bool:
        """Check if bias requires immediate attention."""
        return (self.overall_bias_score > 0.6 or
                any(score > 0.7 for score in self.protected_attribute_bias.values()) or
                BiasType.DEMOGRAPHIC_PARITY in self.bias_types_detected)
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get comprehensive bias summary."""
        return {
            "overall_bias": self.overall_bias_score,
            "bias_severity": self.get_bias_severity(),
            "bias_detected": self.bias_detected,
            "requires_attention": self.requires_immediate_attention(),
            "affected_attributes": list(self.protected_attribute_bias.keys()),
            "most_biased": self.get_most_biased_attributes(),
            "bias_types": [bt.value for bt in self.bias_types_detected],
            "fairness_scores": self.fairness_metrics,
            "mitigation_needed": len(self.mitigation_recommendations) > 0
        }


@dataclass
class ExplanationRequest:
    """Request for generating model explanation."""
    request_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP_TREE
    explanation_scope: ExplanationScope = ExplanationScope.LOCAL
    explanation_type: ExplanationType = ExplanationType.FEATURE_IMPORTANCE
    target_audience: ExplanationAudience = ExplanationAudience.TECHNICAL
    input_data: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    target_class: Optional[str] = None
    num_features: int = 10
    explanation_config: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high, urgent
    requested_at: datetime = field(default_factory=datetime.utcnow)
    requester_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate explanation request."""
        if self.num_features <= 0:
            raise ValueError("Number of features must be positive")
        if self.priority not in ["low", "normal", "high", "urgent"]:
            raise ValueError("Invalid priority level")
    
    def is_high_priority(self) -> bool:
        """Check if request is high priority."""
        return self.priority in ["high", "urgent"]
    
    def get_request_summary(self) -> Dict[str, Any]:
        """Get request summary."""
        return {
            "request_id": str(self.request_id),
            "model_id": str(self.model_id),
            "method": self.explanation_method.value,
            "scope": self.explanation_scope.value,
            "type": self.explanation_type.value,
            "audience": self.target_audience.value,
            "priority": self.priority,
            "requested_at": self.requested_at.isoformat(),
            "num_features": self.num_features,
            "has_input_data": self.input_data is not None
        }


@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    result_id: UUID = field(default_factory=uuid4)
    request_id: UUID = field(default_factory=uuid4)
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP_TREE
    explanation_scope: ExplanationScope = ExplanationScope.LOCAL
    success: bool = True
    instance_explanation: Optional[InstanceExplanation] = None
    global_explanation: Optional[GlobalExplanation] = None
    trust_score: Optional[TrustScore] = None
    bias_analysis: Optional[BiasAnalysis] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[ExplanationMetadata] = None
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = ExplanationMetadata()
    
    def has_instance_explanation(self) -> bool:
        """Check if result contains instance explanation."""
        return self.instance_explanation is not None
    
    def has_global_explanation(self) -> bool:
        """Check if result contains global explanation."""
        return self.global_explanation is not None
    
    def is_high_quality(self) -> bool:
        """Check if explanation is high quality."""
        if not self.success:
            return False
        
        quality_indicators = []
        
        if self.metadata:
            quality_indicators.append(self.metadata.is_high_quality())
        
        if self.trust_score:
            quality_indicators.append(self.trust_score.is_trustworthy())
        
        if self.instance_explanation:
            quality_indicators.append(self.instance_explanation.local_fidelity_score > 0.8)
        
        return all(quality_indicators) if quality_indicators else False
    
    def has_bias_issues(self) -> bool:
        """Check if explanation indicates bias issues."""
        if self.bias_analysis:
            return self.bias_analysis.bias_detected
        return False
    
    def get_result_summary(self) -> Dict[str, Any]:
        """Get comprehensive result summary."""
        summary = {
            "result_id": str(self.result_id),
            "request_id": str(self.request_id),
            "method": self.explanation_method.value,
            "scope": self.explanation_scope.value,
            "success": self.success,
            "has_instance": self.has_instance_explanation(),
            "has_global": self.has_global_explanation(),
            "is_high_quality": self.is_high_quality(),
            "has_bias_issues": self.has_bias_issues(),
            "warnings_count": len(self.warnings)
        }
        
        if self.error_message:
            summary["error"] = self.error_message
        
        if self.trust_score:
            summary["trust_level"] = self.trust_score.trust_level.value
            summary["trust_score"] = self.trust_score.overall_trust_score
        
        if self.metadata:
            summary["generation_time"] = self.metadata.generation_time_seconds
            summary["confidence"] = self.metadata.explanation_confidence
        
        return summary


@dataclass
class ExplanationValidation:
    """Validation metrics for explanation quality."""
    validation_id: UUID = field(default_factory=uuid4)
    explanation_id: UUID = field(default_factory=uuid4)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)
    consistency_tests: Dict[str, float] = field(default_factory=dict)
    stability_tests: Dict[str, float] = field(default_factory=dict)
    fidelity_tests: Dict[str, float] = field(default_factory=dict)
    robustness_tests: Dict[str, float] = field(default_factory=dict)
    ground_truth_comparison: Optional[Dict[str, float]] = None
    human_evaluation_scores: Dict[str, float] = field(default_factory=dict)
    overall_validation_score: float = 0.0
    validation_passed: bool = True
    
    def __post_init__(self):
        """Compute overall validation score."""
        if self.overall_validation_score == 0.0:
            self.overall_validation_score = self._compute_validation_score()
        
        # Determine if validation passed
        self.validation_passed = self.overall_validation_score >= 0.7
    
    def _compute_validation_score(self) -> float:
        """Compute overall validation score."""
        scores = []
        
        if self.consistency_tests:
            scores.append(np.mean(list(self.consistency_tests.values())))
        
        if self.stability_tests:
            scores.append(np.mean(list(self.stability_tests.values())))
        
        if self.fidelity_tests:
            scores.append(np.mean(list(self.fidelity_tests.values())))
        
        if self.robustness_tests:
            scores.append(np.mean(list(self.robustness_tests.values())))
        
        if self.human_evaluation_scores:
            scores.append(np.mean(list(self.human_evaluation_scores.values())))
        
        return float(np.mean(scores)) if scores else 0.0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "validation_id": str(self.validation_id),
            "explanation_id": str(self.explanation_id),
            "overall_score": self.overall_validation_score,
            "validation_passed": self.validation_passed,
            "consistency_score": np.mean(list(self.consistency_tests.values())) if self.consistency_tests else 0.0,
            "stability_score": np.mean(list(self.stability_tests.values())) if self.stability_tests else 0.0,
            "fidelity_score": np.mean(list(self.fidelity_tests.values())) if self.fidelity_tests else 0.0,
            "robustness_score": np.mean(list(self.robustness_tests.values())) if self.robustness_tests else 0.0,
            "has_ground_truth": self.ground_truth_comparison is not None,
            "has_human_evaluation": len(self.human_evaluation_scores) > 0,
            "timestamp": self.validation_timestamp.isoformat()
        }