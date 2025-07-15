"""Value objects for machine learning model metrics and evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID

import numpy as np


class MetricType(str, Enum):
    """Types of ML metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    LOG_LOSS = "log_loss"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    SILHOUETTE_SCORE = "silhouette_score"
    ADJUSTED_RAND_SCORE = "adjusted_rand_score"
    NORMALIZED_MUTUAL_INFO = "normalized_mutual_info"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Machine learning task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RANKING = "ranking"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


@dataclass(frozen=True)
class ModelMetrics:
    """Comprehensive model performance metrics.
    
    This value object encapsulates all relevant metrics for a trained ML model
    across different evaluation scenarios and data splits.
    
    Attributes:
        model_id: Unique identifier for the model
        task_type: Type of ML task
        metrics: Dictionary of metric name to value
        train_metrics: Metrics on training data
        validation_metrics: Metrics on validation data
        test_metrics: Metrics on test data
        cross_validation_metrics: Cross-validation results
        confusion_matrix: Confusion matrix for classification
        feature_importance: Feature importance scores
        prediction_probabilities: Prediction probability distributions
        evaluation_timestamp: When metrics were calculated
        evaluation_duration_seconds: Time taken for evaluation
        data_size: Size of evaluation datasets
        model_parameters: Model hyperparameters used
        preprocessing_steps: Applied preprocessing steps
        evaluation_notes: Additional evaluation notes
    """
    
    model_id: UUID
    task_type: TaskType
    metrics: Dict[str, float]
    
    # Metrics by data split
    train_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    cross_validation_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    # Detailed results
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_probabilities: Optional[Dict[str, List[float]]] = None
    residuals: Optional[List[float]] = None
    
    # Metadata
    evaluation_timestamp: datetime
    evaluation_duration_seconds: float
    data_size: Dict[str, int]  # train_size, val_size, test_size
    model_parameters: Dict[str, Any]
    preprocessing_steps: List[str]
    evaluation_notes: Optional[str] = None
    
    # Statistical significance
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    statistical_tests: Optional[Dict[str, Dict[str, Any]]] = None
    
    def get_primary_metric(self) -> float:
        """Get the primary metric value for this task type."""
        primary_metrics = {
            TaskType.BINARY_CLASSIFICATION: MetricType.AUC_ROC,
            TaskType.MULTICLASS_CLASSIFICATION: MetricType.F1_SCORE,
            TaskType.REGRESSION: MetricType.RMSE,
            TaskType.CLUSTERING: MetricType.SILHOUETTE_SCORE,
            TaskType.ANOMALY_DETECTION: MetricType.AUC_ROC
        }
        
        primary_metric = primary_metrics.get(self.task_type, MetricType.ACCURACY)
        return self.metrics.get(primary_metric.value, 0.0)
    
    def is_better_than(self, other: ModelMetrics, higher_is_better: Optional[bool] = None) -> bool:
        """Compare this model's performance to another model."""
        if other.task_type != self.task_type:
            raise ValueError("Cannot compare models with different task types")
        
        my_primary = self.get_primary_metric()
        other_primary = other.get_primary_metric()
        
        if higher_is_better is None:
            # Default behavior based on metric type
            higher_better_metrics = {
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1_SCORE, MetricType.AUC_ROC, MetricType.AUC_PR,
                MetricType.R2_SCORE, MetricType.SILHOUETTE_SCORE
            }
            
            primary_metrics = {
                TaskType.BINARY_CLASSIFICATION: MetricType.AUC_ROC,
                TaskType.MULTICLASS_CLASSIFICATION: MetricType.F1_SCORE,
                TaskType.REGRESSION: MetricType.RMSE,
                TaskType.CLUSTERING: MetricType.SILHOUETTE_SCORE,
                TaskType.ANOMALY_DETECTION: MetricType.AUC_ROC
            }
            
            primary_metric = primary_metrics.get(self.task_type, MetricType.ACCURACY)
            higher_is_better = primary_metric in higher_better_metrics
        
        if higher_is_better:
            return my_primary > other_primary
        else:
            return my_primary < other_primary
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        return {
            "model_id": str(self.model_id),
            "task_type": self.task_type.value,
            "primary_metric": self.get_primary_metric(),
            "metric_count": len(self.metrics),
            "evaluation_date": self.evaluation_timestamp.isoformat(),
            "data_splits": list(self.data_size.keys()),
            "total_samples": sum(self.data_size.values()),
            "has_feature_importance": self.feature_importance is not None,
            "has_cross_validation": self.cross_validation_metrics is not None
        }


@dataclass(frozen=True)
class HyperparameterOptimizationResult:
    """Results from hyperparameter optimization."""
    
    optimization_id: UUID
    model_type: str
    optimization_method: str  # grid_search, random_search, bayesian, etc.
    
    # Best configuration
    best_parameters: Dict[str, Any]
    best_score: float
    best_metrics: ModelMetrics
    
    # Optimization history
    parameter_history: List[Dict[str, Any]]
    score_history: List[float]
    optimization_iterations: int
    
    # Search space
    search_space: Dict[str, Any]
    search_strategy: Dict[str, Any]
    
    # Timing and resources
    optimization_start_time: datetime
    optimization_end_time: datetime
    total_duration_seconds: float
    evaluation_count: int
    
    # Convergence analysis
    convergence_metrics: Dict[str, float]
    early_stopping_triggered: bool
    improvement_threshold: float
    
    # Statistical analysis
    parameter_importance: Optional[Dict[str, float]] = None
    parameter_correlations: Optional[Dict[str, Dict[str, float]]] = None
    optimization_insights: Optional[Dict[str, Any]] = None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            "optimization_id": str(self.optimization_id),
            "method": self.optimization_method,
            "best_score": self.best_score,
            "iterations": self.optimization_iterations,
            "duration_minutes": self.total_duration_seconds / 60,
            "evaluations_per_minute": self.evaluation_count / (self.total_duration_seconds / 60),
            "converged": not self.early_stopping_triggered,
            "parameters_tuned": len(self.search_space),
            "improvement_over_baseline": self.convergence_metrics.get("improvement", 0.0)
        }


@dataclass(frozen=True)
class ModelComparison:
    """Comparison results between multiple models."""
    
    comparison_id: UUID
    model_metrics: List[ModelMetrics]
    comparison_timestamp: datetime
    
    # Comparison results
    ranking: List[UUID]  # Model IDs ranked by performance
    statistical_significance: Dict[str, Dict[str, float]]  # p-values between models
    effect_sizes: Dict[str, Dict[str, float]]  # Effect sizes between models
    
    # Ensemble analysis
    ensemble_potential: Dict[str, float]  # Diversity metrics
    complementary_models: List[Tuple[UUID, UUID, float]]  # Model pairs with complementarity scores
    
    # Analysis metadata
    comparison_metric: str
    evaluation_method: str  # cross_validation, holdout, etc.
    confidence_level: float
    
    # Recommendations
    best_model_id: UUID
    ensemble_recommendations: Optional[List[UUID]] = None
    selection_rationale: Optional[str] = None
    
    def get_winner(self) -> ModelMetrics:
        """Get the best performing model."""
        return next(m for m in self.model_metrics if m.model_id == self.best_model_id)
    
    def get_top_k_models(self, k: int) -> List[ModelMetrics]:
        """Get top k performing models."""
        return [next(m for m in self.model_metrics if m.model_id == model_id) 
                for model_id in self.ranking[:k]]
    
    def is_significantly_better(self, model1_id: UUID, model2_id: UUID, 
                              alpha: float = 0.05) -> bool:
        """Check if model1 is significantly better than model2."""
        key = f"{model1_id}_{model2_id}"
        p_value = self.statistical_significance.get(key, {}).get("p_value", 1.0)
        return p_value < alpha
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        winner = self.get_winner()
        return {
            "comparison_id": str(self.comparison_id),
            "models_compared": len(self.model_metrics),
            "winner_id": str(self.best_model_id),
            "winner_score": winner.get_primary_metric(),
            "margin_to_second": self._calculate_margin_to_second(),
            "ensemble_recommended": self.ensemble_recommendations is not None,
            "statistically_significant": self._has_significant_differences(),
            "comparison_date": self.comparison_timestamp.isoformat()
        }
    
    def _calculate_margin_to_second(self) -> float:
        """Calculate margin between first and second place."""
        if len(self.ranking) < 2:
            return 0.0
        
        first = next(m for m in self.model_metrics if m.model_id == self.ranking[0])
        second = next(m for m in self.model_metrics if m.model_id == self.ranking[1])
        
        return abs(first.get_primary_metric() - second.get_primary_metric())
    
    def _has_significant_differences(self) -> bool:
        """Check if there are statistically significant differences."""
        for comparison in self.statistical_significance.values():
            if comparison.get("p_value", 1.0) < 0.05:
                return True
        return False


@dataclass(frozen=True)
class FeatureImportance:
    """Feature importance analysis results."""
    
    model_id: UUID
    importance_method: str  # permutation, shap, built_in, etc.
    feature_scores: Dict[str, float]
    feature_rankings: List[str]  # Features ordered by importance
    
    # Statistical measures
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    stability_scores: Optional[Dict[str, float]] = None  # Across CV folds
    
    # Interaction effects
    feature_interactions: Optional[Dict[str, Dict[str, float]]] = None
    interaction_strength: Optional[float] = None
    
    # Interpretability
    top_features: List[str]
    redundant_features: List[str]
    recommended_feature_subset: List[str]
    
    # Metadata
    calculation_timestamp: datetime
    calculation_method_details: Dict[str, Any]
    
    def get_top_k_features(self, k: int) -> List[str]:
        """Get top k most important features."""
        return self.feature_rankings[:k]
    
    def get_importance_threshold(self, threshold: float) -> List[str]:
        """Get features above importance threshold."""
        return [feature for feature, score in self.feature_scores.items() 
                if score >= threshold]
    
    def get_cumulative_importance(self, target_percentage: float = 0.95) -> List[str]:
        """Get features that explain target percentage of total importance."""
        total_importance = sum(abs(score) for score in self.feature_scores.values())
        target_sum = total_importance * target_percentage
        
        cumulative_sum = 0.0
        selected_features = []
        
        for feature in self.feature_rankings:
            cumulative_sum += abs(self.feature_scores[feature])
            selected_features.append(feature)
            
            if cumulative_sum >= target_sum:
                break
        
        return selected_features