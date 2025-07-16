"""Model Promotion Service

Domain service for handling model promotion logic and validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import UUID

from pynomaly_mlops.domain.entities.model import Model, ModelStatus
from pynomaly_mlops.domain.value_objects.model_metrics import ModelMetrics


@dataclass
class PromotionCriteria:
    """Criteria for model promotion."""
    
    # Minimum performance thresholds
    min_accuracy: Optional[float] = None
    min_f1_score: Optional[float] = None
    min_precision: Optional[float] = None
    min_recall: Optional[float] = None
    min_auc_roc: Optional[float] = None
    
    # Maximum error thresholds
    max_false_positive_rate: Optional[float] = None
    max_log_loss: Optional[float] = None
    
    # Business criteria
    min_business_value: Optional[float] = None
    max_cost_impact: Optional[float] = None
    
    # Comparison criteria
    require_improvement_over_current: bool = True
    min_improvement_percentage: float = 5.0  # 5% improvement required
    
    # Validation requirements
    require_validation_on_holdout: bool = True
    min_validation_samples: int = 1000
    
    # Custom criteria
    custom_criteria: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize custom criteria if None."""
        if self.custom_criteria is None:
            self.custom_criteria = {}


@dataclass
class PromotionResult:
    """Result of a promotion evaluation."""
    
    approved: bool
    reasons: List[str]
    warnings: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []


class ModelPromotionService(ABC):
    """Abstract base class for model promotion services."""
    
    @abstractmethod
    def evaluate_promotion(
        self,
        candidate_model: Model,
        target_status: ModelStatus,
        current_production_model: Optional[Model] = None,
        criteria: Optional[PromotionCriteria] = None
    ) -> PromotionResult:
        """Evaluate whether a model should be promoted to target status.
        
        Args:
            candidate_model: Model to be promoted
            target_status: Target status for promotion
            current_production_model: Current model in production (if any)
            criteria: Promotion criteria to apply
            
        Returns:
            PromotionResult indicating whether promotion should proceed
        """
        pass
    
    @abstractmethod
    def get_default_criteria(self, target_status: ModelStatus) -> PromotionCriteria:
        """Get default promotion criteria for a target status.
        
        Args:
            target_status: Target status for promotion
            
        Returns:
            Default PromotionCriteria for the status
        """
        pass
    
    @abstractmethod
    def validate_promotion_path(
        self,
        current_status: ModelStatus,
        target_status: ModelStatus
    ) -> bool:
        """Validate that promotion from current to target status is allowed.
        
        Args:
            current_status: Current model status
            target_status: Target status for promotion
            
        Returns:
            True if promotion path is valid
        """
        pass


class DefaultModelPromotionService(ModelPromotionService):
    """Default implementation of model promotion service."""
    
    def __init__(self):
        """Initialize the service with default configurations."""
        self._status_criteria = self._initialize_default_criteria()
    
    def evaluate_promotion(
        self,
        candidate_model: Model,
        target_status: ModelStatus,
        current_production_model: Optional[Model] = None,
        criteria: Optional[PromotionCriteria] = None
    ) -> PromotionResult:
        """Evaluate model promotion."""
        if criteria is None:
            criteria = self.get_default_criteria(target_status)
        
        reasons = []
        warnings = []
        recommendations = []
        
        # Validate promotion path
        if not self.validate_promotion_path(candidate_model.status, target_status):
            return PromotionResult(
                approved=False,
                reasons=[f"Invalid promotion path from {candidate_model.status} to {target_status}"],
                warnings=warnings,
                recommendations=recommendations
            )
        
        # Check if model has metrics
        if not candidate_model.metrics:
            return PromotionResult(
                approved=False,
                reasons=["Model has no performance metrics"],
                warnings=warnings,
                recommendations=["Run model evaluation to generate metrics"]
            )
        
        # Evaluate against criteria
        metrics = candidate_model.metrics
        approved = True
        
        # Check minimum performance thresholds
        if criteria.min_accuracy and (metrics.accuracy is None or metrics.accuracy < criteria.min_accuracy):
            approved = False
            reasons.append(f"Accuracy {metrics.accuracy} below minimum {criteria.min_accuracy}")
        
        if criteria.min_f1_score and (metrics.f1_score is None or metrics.f1_score < criteria.min_f1_score):
            approved = False
            reasons.append(f"F1 score {metrics.f1_score} below minimum {criteria.min_f1_score}")
        
        if criteria.min_precision and (metrics.precision is None or metrics.precision < criteria.min_precision):
            approved = False
            reasons.append(f"Precision {metrics.precision} below minimum {criteria.min_precision}")
        
        if criteria.min_recall and (metrics.recall is None or metrics.recall < criteria.min_recall):
            approved = False
            reasons.append(f"Recall {metrics.recall} below minimum {criteria.min_recall}")
        
        if criteria.min_auc_roc and (metrics.auc_roc is None or metrics.auc_roc < criteria.min_auc_roc):
            approved = False
            reasons.append(f"AUC ROC {metrics.auc_roc} below minimum {criteria.min_auc_roc}")
        
        # Check maximum error thresholds
        if (criteria.max_false_positive_rate and 
            metrics.false_positive_rate is not None and 
            metrics.false_positive_rate > criteria.max_false_positive_rate):
            approved = False
            reasons.append(f"False positive rate {metrics.false_positive_rate} above maximum {criteria.max_false_positive_rate}")
        
        if (criteria.max_log_loss and 
            metrics.log_loss is not None and 
            metrics.log_loss > criteria.max_log_loss):
            approved = False
            reasons.append(f"Log loss {metrics.log_loss} above maximum {criteria.max_log_loss}")
        
        # Check business criteria
        if (criteria.min_business_value and 
            (metrics.business_value is None or metrics.business_value < criteria.min_business_value)):
            approved = False
            reasons.append(f"Business value {metrics.business_value} below minimum {criteria.min_business_value}")
        
        # Compare with current production model
        if (criteria.require_improvement_over_current and 
            current_production_model and 
            current_production_model.metrics):
            
            if not self._shows_improvement(metrics, current_production_model.metrics, criteria.min_improvement_percentage):
                approved = False
                reasons.append(f"Model does not show {criteria.min_improvement_percentage}% improvement over current production model")
        
        # Add success reasons if approved
        if approved and not reasons:
            reasons.append("Model meets all promotion criteria")
            if current_production_model and current_production_model.metrics:
                improvement = self._calculate_improvement_percentage(metrics, current_production_model.metrics)
                if improvement > 0:
                    reasons.append(f"Model shows {improvement:.1f}% improvement over current production model")
        
        # Add warnings for edge cases
        if approved:
            if target_status == ModelStatus.PRODUCTION:
                if candidate_model.deployment_count == 0:
                    warnings.append("Model has never been deployed before")
                    recommendations.append("Consider deploying to staging first")
                
                if not candidate_model.validation_metrics:
                    warnings.append("Model has no validation metrics")
                    recommendations.append("Run validation on holdout dataset")
        
        return PromotionResult(
            approved=approved,
            reasons=reasons,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def get_default_criteria(self, target_status: ModelStatus) -> PromotionCriteria:
        """Get default promotion criteria for target status."""
        return self._status_criteria.get(target_status, PromotionCriteria())
    
    def validate_promotion_path(
        self,
        current_status: ModelStatus,
        target_status: ModelStatus
    ) -> bool:
        """Validate promotion path."""
        valid_transitions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.TESTING, ModelStatus.ARCHIVED],
            ModelStatus.TESTING: [ModelStatus.STAGING, ModelStatus.DEVELOPMENT, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.TESTING, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.DEPRECATED, ModelStatus.ARCHIVED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: [],  # Terminal state
        }
        
        return target_status in valid_transitions.get(current_status, [])
    
    def _initialize_default_criteria(self) -> Dict[ModelStatus, PromotionCriteria]:
        """Initialize default promotion criteria for each status."""
        return {
            ModelStatus.TESTING: PromotionCriteria(
                min_accuracy=0.6,
                require_improvement_over_current=False,
                require_validation_on_holdout=False,
                min_validation_samples=100,
            ),
            ModelStatus.STAGING: PromotionCriteria(
                min_accuracy=0.75,
                min_f1_score=0.7,
                require_improvement_over_current=True,
                min_improvement_percentage=2.0,
                require_validation_on_holdout=True,
                min_validation_samples=500,
            ),
            ModelStatus.PRODUCTION: PromotionCriteria(
                min_accuracy=0.85,
                min_f1_score=0.8,
                min_precision=0.8,
                min_recall=0.75,
                max_false_positive_rate=0.1,
                require_improvement_over_current=True,
                min_improvement_percentage=5.0,
                require_validation_on_holdout=True,
                min_validation_samples=1000,
                min_business_value=0.0,  # Require non-negative business value
            ),
        }
    
    def _shows_improvement(
        self,
        candidate_metrics: ModelMetrics,
        current_metrics: ModelMetrics,
        min_improvement_percentage: float
    ) -> bool:
        """Check if candidate shows minimum improvement over current."""
        improvement = self._calculate_improvement_percentage(candidate_metrics, current_metrics)
        return improvement >= min_improvement_percentage
    
    def _calculate_improvement_percentage(
        self,
        candidate_metrics: ModelMetrics,
        current_metrics: ModelMetrics
    ) -> float:
        """Calculate improvement percentage of candidate over current."""
        # Try to use primary metric first
        candidate_primary = candidate_metrics.get_primary_metric()
        current_primary = current_metrics.get_primary_metric()
        
        if candidate_primary is not None and current_primary is not None and current_primary > 0:
            return ((candidate_primary - current_primary) / current_primary) * 100
        
        # Fall back to accuracy if available
        if (candidate_metrics.accuracy is not None and 
            current_metrics.accuracy is not None and 
            current_metrics.accuracy > 0):
            return ((candidate_metrics.accuracy - current_metrics.accuracy) / current_metrics.accuracy) * 100
        
        # If no comparable metrics, return 0
        return 0.0