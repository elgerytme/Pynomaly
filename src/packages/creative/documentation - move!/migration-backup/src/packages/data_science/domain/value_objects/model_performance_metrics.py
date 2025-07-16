"""Model Performance Metrics value object for ML model evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class ModelTask(str, Enum):
    """Types of ML tasks."""
    
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RANKING = "ranking"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"


class ModelPerformanceMetrics(BaseValueObject):
    """Value object representing comprehensive ML model performance metrics.
    
    This immutable value object encapsulates performance measurements
    for machine learning models across different task types.
    
    Attributes:
        task_type: Type of ML task (classification, regression, etc.)
        sample_size: Number of samples in evaluation
        
        # Classification metrics
        accuracy: Accuracy score (correct predictions / total)
        precision: Precision score (TP / (TP + FP))
        recall: Recall score (TP / (TP + FN))
        f1_score: F1 score (harmonic mean of precision and recall)
        roc_auc: Area under ROC curve
        pr_auc: Area under Precision-Recall curve
        log_loss: Logarithmic loss
        matthews_correlation: Matthews correlation coefficient
        
        # Multiclass specific
        macro_precision: Macro-averaged precision
        micro_precision: Micro-averaged precision
        weighted_precision: Weighted precision
        macro_recall: Macro-averaged recall
        micro_recall: Micro-averaged recall
        weighted_recall: Weighted recall
        macro_f1: Macro-averaged F1
        micro_f1: Micro-averaged F1
        weighted_f1: Weighted F1
        
        # Regression metrics
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        r2_score: R-squared coefficient of determination
        adjusted_r2: Adjusted R-squared
        explained_variance: Explained variance score
        median_absolute_error: Median absolute error
        
        # Ranking metrics
        ndcg: Normalized Discounted Cumulative Gain
        map_score: Mean Average Precision
        mrr: Mean Reciprocal Rank
        
        # Time series specific
        smape: Symmetric Mean Absolute Percentage Error
        mase: Mean Absolute Scaled Error
        directional_accuracy: Directional accuracy for forecasting
        
        # Cross-validation metrics
        cv_mean: Mean of cross-validation scores
        cv_std: Standard deviation of CV scores
        cv_scores: Individual CV fold scores
        
        # Robustness metrics
        prediction_stability: Stability of predictions
        feature_attribution_stability: Stability of feature importance
        adversarial_robustness: Robustness to adversarial inputs
        
        # Efficiency metrics
        training_time_seconds: Time taken to train model
        prediction_time_seconds: Time for predictions
        memory_usage_mb: Memory usage during inference
        model_size_mb: Size of serialized model
        
        # Business metrics
        business_value_score: Custom business value metric
        cost_per_prediction: Cost efficiency metric
        error_cost: Cost of prediction errors
        
        # Fairness metrics
        demographic_parity: Demographic parity score
        equalized_odds: Equalized odds score
        calibration_score: Calibration quality
        
        # Confidence and uncertainty
        prediction_confidence: Average prediction confidence
        uncertainty_score: Model uncertainty measurement
        coverage_probability: Confidence interval coverage
        
        # Additional metadata
        evaluation_date: When metrics were calculated
        holdout_percentage: Percentage of data held out for testing
        stratified: Whether evaluation used stratified sampling
        confidence_level: Confidence level for intervals
    """
    
    task_type: ModelTask
    sample_size: int = Field(..., gt=0)
    
    # Core classification metrics
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    roc_auc: Optional[float] = Field(None, ge=0, le=1)
    pr_auc: Optional[float] = Field(None, ge=0, le=1)
    log_loss: Optional[float] = Field(None, ge=0)
    matthews_correlation: Optional[float] = Field(None, ge=-1, le=1)
    
    # Multiclass metrics
    macro_precision: Optional[float] = Field(None, ge=0, le=1)
    micro_precision: Optional[float] = Field(None, ge=0, le=1)
    weighted_precision: Optional[float] = Field(None, ge=0, le=1)
    macro_recall: Optional[float] = Field(None, ge=0, le=1)
    micro_recall: Optional[float] = Field(None, ge=0, le=1)
    weighted_recall: Optional[float] = Field(None, ge=0, le=1)
    macro_f1: Optional[float] = Field(None, ge=0, le=1)
    micro_f1: Optional[float] = Field(None, ge=0, le=1)
    weighted_f1: Optional[float] = Field(None, ge=0, le=1)
    
    # Regression metrics
    mse: Optional[float] = Field(None, ge=0)
    rmse: Optional[float] = Field(None, ge=0)
    mae: Optional[float] = Field(None, ge=0)
    mape: Optional[float] = Field(None, ge=0)
    r2_score: Optional[float] = Field(None, le=1)
    adjusted_r2: Optional[float] = Field(None, le=1)
    explained_variance: Optional[float] = Field(None, ge=0, le=1)
    median_absolute_error: Optional[float] = Field(None, ge=0)
    
    # Ranking metrics
    ndcg: Optional[float] = Field(None, ge=0, le=1)
    map_score: Optional[float] = Field(None, ge=0, le=1)
    mrr: Optional[float] = Field(None, ge=0, le=1)
    
    # Time series metrics
    smape: Optional[float] = Field(None, ge=0, le=200)
    mase: Optional[float] = Field(None, ge=0)
    directional_accuracy: Optional[float] = Field(None, ge=0, le=1)
    
    # Cross-validation
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = Field(None, ge=0)
    cv_scores: Optional[list[float]] = None
    
    # Robustness
    prediction_stability: Optional[float] = Field(None, ge=0, le=1)
    feature_attribution_stability: Optional[float] = Field(None, ge=0, le=1)
    adversarial_robustness: Optional[float] = Field(None, ge=0, le=1)
    
    # Efficiency
    training_time_seconds: Optional[float] = Field(None, gt=0)
    prediction_time_seconds: Optional[float] = Field(None, gt=0)
    memory_usage_mb: Optional[float] = Field(None, gt=0)
    model_size_mb: Optional[float] = Field(None, gt=0)
    
    # Business value
    business_value_score: Optional[float] = None
    cost_per_prediction: Optional[float] = Field(None, ge=0)
    error_cost: Optional[float] = Field(None, ge=0)
    
    # Fairness
    demographic_parity: Optional[float] = Field(None, ge=0, le=1)
    equalized_odds: Optional[float] = Field(None, ge=0, le=1)
    calibration_score: Optional[float] = Field(None, ge=0, le=1)
    
    # Uncertainty
    prediction_confidence: Optional[float] = Field(None, ge=0, le=1)
    uncertainty_score: Optional[float] = Field(None, ge=0)
    coverage_probability: Optional[float] = Field(None, ge=0, le=1)
    
    # Metadata
    holdout_percentage: Optional[float] = Field(None, gt=0, le=100)
    stratified: bool = Field(default=False)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    
    @validator('rmse')
    def validate_rmse(cls, v: Optional[float], values: dict[str, Any]) -> Optional[float]:
        """Validate RMSE is consistent with MSE."""
        if v is not None:
            mse = values.get('mse')
            if mse is not None:
                expected_rmse = mse ** 0.5
                if abs(v - expected_rmse) > 1e-10:
                    raise ValueError("RMSE inconsistent with MSE")
        return v
    
    @validator('cv_std')
    def validate_cv_std(cls, v: Optional[float], values: dict[str, Any]) -> Optional[float]:
        """Validate CV std is consistent with scores."""
        if v is not None:
            cv_scores = values.get('cv_scores')
            if cv_scores is not None and len(cv_scores) > 1:
                import numpy as np
                expected_std = float(np.std(cv_scores))
                if abs(v - expected_std) > 1e-6:
                    raise ValueError("CV std inconsistent with CV scores")
        return v
    
    @validator('cv_mean')
    def validate_cv_mean(cls, v: Optional[float], values: dict[str, Any]) -> Optional[float]:
        """Validate CV mean is consistent with scores."""
        if v is not None:
            cv_scores = values.get('cv_scores')
            if cv_scores is not None:
                import numpy as np
                expected_mean = float(np.mean(cv_scores))
                if abs(v - expected_mean) > 1e-6:
                    raise ValueError("CV mean inconsistent with CV scores")
        return v
    
    def get_primary_metric(self) -> Optional[float]:
        """Get the primary metric for this task type."""
        if self.task_type == ModelTask.BINARY_CLASSIFICATION:
            return self.roc_auc or self.f1_score or self.accuracy
        elif self.task_type == ModelTask.MULTICLASS_CLASSIFICATION:
            return self.weighted_f1 or self.macro_f1 or self.accuracy
        elif self.task_type == ModelTask.REGRESSION:
            return self.r2_score or (1 - self.rmse) if self.rmse else None
        elif self.task_type == ModelTask.ANOMALY_DETECTION:
            return self.pr_auc or self.roc_auc
        elif self.task_type == ModelTask.RANKING:
            return self.ndcg or self.map_score
        elif self.task_type == ModelTask.TIME_SERIES:
            return self.r2_score or (1 - self.mape / 100) if self.mape else None
        else:
            return None
    
    def get_performance_grade(self) -> str:
        """Get performance grade based on primary metric."""
        primary = self.get_primary_metric()
        if primary is None:
            return "Unknown"
        
        if primary >= 0.9:
            return "Excellent"
        elif primary >= 0.8:
            return "Good"
        elif primary >= 0.7:
            return "Fair"
        elif primary >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def is_classification_task(self) -> bool:
        """Check if this is a classification task."""
        return self.task_type in [
            ModelTask.BINARY_CLASSIFICATION,
            ModelTask.MULTICLASS_CLASSIFICATION
        ]
    
    def is_regression_task(self) -> bool:
        """Check if this is a regression task."""
        return self.task_type in [
            ModelTask.REGRESSION,
            ModelTask.TIME_SERIES
        ]
    
    def get_classification_summary(self) -> Optional[dict[str, Any]]:
        """Get classification metrics summary."""
        if not self.is_classification_task():
            return None
        
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "primary_metric": self.get_primary_metric(),
            "grade": self.get_performance_grade()
        }
    
    def get_regression_summary(self) -> Optional[dict[str, Any]]:
        """Get regression metrics summary."""
        if not self.is_regression_task():
            return None
        
        return {
            "r2_score": self.r2_score,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "explained_variance": self.explained_variance,
            "primary_metric": self.get_primary_metric(),
            "grade": self.get_performance_grade()
        }
    
    def get_efficiency_summary(self) -> dict[str, Any]:
        """Get efficiency metrics summary."""
        return {
            "training_time": self.training_time_seconds,
            "prediction_time": self.prediction_time_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "model_size_mb": self.model_size_mb,
            "cost_per_prediction": self.cost_per_prediction
        }
    
    def get_robustness_summary(self) -> dict[str, Any]:
        """Get robustness metrics summary."""
        return {
            "prediction_stability": self.prediction_stability,
            "feature_attribution_stability": self.feature_attribution_stability,
            "adversarial_robustness": self.adversarial_robustness,
            "cv_std": self.cv_std,
            "uncertainty_score": self.uncertainty_score
        }
    
    def get_fairness_summary(self) -> dict[str, Any]:
        """Get fairness metrics summary."""
        return {
            "demographic_parity": self.demographic_parity,
            "equalized_odds": self.equalized_odds,
            "calibration_score": self.calibration_score
        }
    
    def compare_with(self, other: ModelPerformanceMetrics) -> dict[str, Any]:
        """Compare performance with another model."""
        if not isinstance(other, ModelPerformanceMetrics):
            raise ValueError("Can only compare with another ModelPerformanceMetrics")
        
        if self.task_type != other.task_type:
            raise ValueError("Can only compare models with same task type")
        
        primary_self = self.get_primary_metric()
        primary_other = other.get_primary_metric()
        
        comparison = {
            "task_type": self.task_type.value,
            "better_model": None,
            "performance_difference": None,
            "efficiency_comparison": {},
            "robustness_comparison": {}
        }
        
        if primary_self is not None and primary_other is not None:
            comparison["performance_difference"] = primary_self - primary_other
            comparison["better_model"] = "self" if primary_self > primary_other else "other"
        
        # Compare efficiency
        if self.training_time_seconds and other.training_time_seconds:
            comparison["efficiency_comparison"]["training_time_ratio"] = (
                self.training_time_seconds / other.training_time_seconds
            )
        
        if self.model_size_mb and other.model_size_mb:
            comparison["efficiency_comparison"]["size_ratio"] = (
                self.model_size_mb / other.model_size_mb
            )
        
        # Compare robustness
        if self.cv_std and other.cv_std:
            comparison["robustness_comparison"]["stability_ratio"] = (
                other.cv_std / self.cv_std  # Lower std is better
            )
        
        return comparison
    
    def is_production_ready(self, min_performance: float = 0.7,
                           max_prediction_time: float = 1.0) -> dict[str, Any]:
        """Check if model meets production readiness criteria."""
        checks = {
            "performance_check": False,
            "efficiency_check": False,
            "robustness_check": False,
            "overall_ready": False,
            "issues": []
        }
        
        # Performance check
        primary_metric = self.get_primary_metric()
        if primary_metric is not None:
            checks["performance_check"] = primary_metric >= min_performance
            if not checks["performance_check"]:
                checks["issues"].append(f"Performance {primary_metric:.3f} below minimum {min_performance}")
        else:
            checks["issues"].append("No primary metric available")
        
        # Efficiency check
        if self.prediction_time_seconds is not None:
            checks["efficiency_check"] = self.prediction_time_seconds <= max_prediction_time
            if not checks["efficiency_check"]:
                checks["issues"].append(
                    f"Prediction time {self.prediction_time_seconds:.3f}s exceeds {max_prediction_time}s"
                )
        else:
            checks["efficiency_check"] = True  # Assume OK if not measured
        
        # Robustness check
        if self.cv_std is not None:
            checks["robustness_check"] = self.cv_std <= 0.1  # Low variance in CV scores
            if not checks["robustness_check"]:
                checks["issues"].append(f"High CV variance: {self.cv_std:.3f}")
        else:
            checks["robustness_check"] = True  # Assume OK if not measured
        
        checks["overall_ready"] = all([
            checks["performance_check"],
            checks["efficiency_check"],
            checks["robustness_check"]
        ])
        
        return checks
    
    @classmethod
    def from_sklearn_metrics(cls, task_type: ModelTask, y_true: Any, y_pred: Any,
                           y_proba: Optional[Any] = None, **kwargs: Any) -> ModelPerformanceMetrics:
        """Create metrics from scikit-learn predictions."""
        from sklearn import metrics
        import numpy as np
        
        sample_size = len(y_true)
        metric_dict = {"task_type": task_type, "sample_size": sample_size}
        
        if task_type == ModelTask.BINARY_CLASSIFICATION:
            metric_dict.update({
                "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
                "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
                "matthews_correlation": float(metrics.matthews_corrcoef(y_true, y_pred))
            })
            
            if y_proba is not None:
                metric_dict.update({
                    "roc_auc": float(metrics.roc_auc_score(y_true, y_proba)),
                    "pr_auc": float(metrics.average_precision_score(y_true, y_proba)),
                    "log_loss": float(metrics.log_loss(y_true, y_proba))
                })
        
        elif task_type == ModelTask.MULTICLASS_CLASSIFICATION:
            metric_dict.update({
                "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
                "macro_precision": float(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)),
                "micro_precision": float(metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)),
                "weighted_precision": float(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                "macro_recall": float(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)),
                "micro_recall": float(metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)),
                "weighted_recall": float(metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                "macro_f1": float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)),
                "micro_f1": float(metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)),
                "weighted_f1": float(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0))
            })
        
        elif task_type == ModelTask.REGRESSION:
            metric_dict.update({
                "mse": float(metrics.mean_squared_error(y_true, y_pred)),
                "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
                "r2_score": float(metrics.r2_score(y_true, y_pred)),
                "explained_variance": float(metrics.explained_variance_score(y_true, y_pred)),
                "median_absolute_error": float(metrics.median_absolute_error(y_true, y_pred))
            })
            
            # Calculate derived metrics
            metric_dict["rmse"] = metric_dict["mse"] ** 0.5
            
            # MAPE calculation
            y_true_array = np.array(y_true)
            y_pred_array = np.array(y_pred)
            non_zero_mask = y_true_array != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true_array[non_zero_mask] - y_pred_array[non_zero_mask]) / 
                                    y_true_array[non_zero_mask])) * 100
                metric_dict["mape"] = float(mape)
        
        return cls(**{**metric_dict, **kwargs})