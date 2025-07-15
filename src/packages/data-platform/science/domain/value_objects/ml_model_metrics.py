"""ML Model Metrics value object for machine learning model evaluation."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class MLModelMetrics(BaseValueObject):
    """Value object representing comprehensive machine learning model metrics.
    
    This immutable value object encapsulates various evaluation metrics
    for different types of machine learning models including classification,
    regression, clustering, and anomaly detection models.
    
    Attributes:
        model_type: Type of ML model (classification, regression, clustering, etc.)
        task_type: Specific task type (binary_classification, multiclass, etc.)
        evaluation_method: Method used for evaluation (holdout, cv, bootstrap)
        sample_size: Number of samples used for evaluation
        
        # Classification metrics
        accuracy: Overall accuracy score
        precision: Precision score(s)
        recall: Recall score(s)
        f1_score: F1 score(s)
        roc_auc: ROC AUC score(s)
        pr_auc: Precision-Recall AUC score(s)
        confusion_matrix: Confusion matrix
        classification_report: Detailed classification metrics
        
        # Regression metrics
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2_score: R-squared coefficient of determination
        adjusted_r2: Adjusted R-squared
        mape: Mean Absolute Percentage Error
        
        # Clustering metrics
        silhouette_score: Silhouette coefficient
        calinski_harabasz_score: Calinski-Harabasz Index
        davies_bouldin_score: Davies-Bouldin Index
        inertia: Within-cluster sum of squares
        
        # Anomaly detection metrics
        anomaly_precision: Precision for anomaly detection
        anomaly_recall: Recall for anomaly detection
        anomaly_f1: F1 score for anomaly detection
        false_positive_rate: False positive rate
        contamination_estimate: Estimated contamination level
        
        # Cross-validation metrics
        cv_scores: Cross-validation scores
        cv_mean: Mean of CV scores
        cv_std: Standard deviation of CV scores
        cv_confidence_interval: Confidence interval for CV scores
        
        # Model complexity metrics
        training_time: Time taken to train the model
        prediction_time: Time taken for predictions
        model_size: Size of the serialized model
        n_parameters: Number of model parameters
        
        # Statistical significance
        p_values: Statistical significance tests
        confidence_intervals: Confidence intervals for metrics
        bootstrap_scores: Bootstrap evaluation scores
        
        # Feature importance and interpretability
        feature_importance: Feature importance scores
        permutation_importance: Permutation-based importance
        shap_values: SHAP explanation values
        
        # Business metrics
        business_impact: Business-relevant metrics
        cost_benefit_analysis: Cost-benefit evaluation
        fairness_metrics: Fairness and bias evaluation
    """
    
    # Model identification
    model_type: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    evaluation_method: str = Field(default="holdout")
    sample_size: int = Field(..., gt=0)
    
    # Classification metrics
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[dict[str, float]] = None
    recall: Optional[dict[str, float]] = None
    f1_score: Optional[dict[str, float]] = None
    roc_auc: Optional[dict[str, float]] = None
    pr_auc: Optional[dict[str, float]] = None
    confusion_matrix: Optional[list[list[int]]] = None
    classification_report: Optional[dict[str, Any]] = None
    
    # Regression metrics
    mse: Optional[float] = Field(None, ge=0)
    rmse: Optional[float] = Field(None, ge=0)
    mae: Optional[float] = Field(None, ge=0)
    r2_score: Optional[float] = None
    adjusted_r2: Optional[float] = None
    mape: Optional[float] = Field(None, ge=0)
    
    # Clustering metrics
    silhouette_score: Optional[float] = Field(None, ge=-1, le=1)
    calinski_harabasz_score: Optional[float] = Field(None, ge=0)
    davies_bouldin_score: Optional[float] = Field(None, ge=0)
    inertia: Optional[float] = Field(None, ge=0)
    
    # Anomaly detection metrics
    anomaly_precision: Optional[float] = Field(None, ge=0, le=1)
    anomaly_recall: Optional[float] = Field(None, ge=0, le=1)
    anomaly_f1: Optional[float] = Field(None, ge=0, le=1)
    false_positive_rate: Optional[float] = Field(None, ge=0, le=1)
    contamination_estimate: Optional[float] = Field(None, ge=0, le=1)
    
    # Cross-validation metrics
    cv_scores: Optional[list[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = Field(None, ge=0)
    cv_confidence_interval: Optional[tuple[float, float]] = None
    
    # Model complexity metrics
    training_time: Optional[float] = Field(None, gt=0)
    prediction_time: Optional[float] = Field(None, gt=0)
    model_size: Optional[int] = Field(None, gt=0)
    n_parameters: Optional[int] = Field(None, ge=0)
    
    # Statistical significance
    p_values: Optional[dict[str, float]] = None
    confidence_intervals: Optional[dict[str, tuple[float, float]]] = None
    bootstrap_scores: Optional[dict[str, list[float]]] = None
    
    # Feature importance and interpretability
    feature_importance: Optional[dict[str, float]] = None
    permutation_importance: Optional[dict[str, float]] = None
    shap_values: Optional[dict[str, Any]] = None
    
    # Business metrics
    business_impact: Optional[dict[str, float]] = None
    cost_benefit_analysis: Optional[dict[str, float]] = None
    fairness_metrics: Optional[dict[str, float]] = None
    
    @validator('model_type')
    def validate_model_type(cls, v: str) -> str:
        """Validate model type."""
        valid_types = {
            'classification', 'regression', 'clustering', 'anomaly_detection',
            'dimensionality_reduction', 'reinforcement_learning', 'neural_network',
            'ensemble', 'time_series', 'nlp', 'computer_vision'
        }
        
        if v.lower() not in valid_types:
            # Allow custom types but validate not empty
            if not v.strip():
                raise ValueError("Model type cannot be empty")
        
        return v.lower()
    
    @validator('task_type')
    def validate_task_type(cls, v: str, values: dict[str, Any]) -> str:
        """Validate task type based on model type."""
        model_type = values.get('model_type', '').lower()
        
        classification_tasks = {
            'binary_classification', 'multiclass_classification', 
            'multilabel_classification', 'imbalanced_classification'
        }
        
        regression_tasks = {
            'linear_regression', 'polynomial_regression', 'logistic_regression',
            'time_series_regression', 'multivariate_regression'
        }
        
        clustering_tasks = {
            'kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture',
            'spectral_clustering', 'density_based'
        }
        
        anomaly_tasks = {
            'outlier_detection', 'novelty_detection', 'change_detection',
            'fraud_detection', 'intrusion_detection'
        }
        
        # Validate task type matches model type
        if model_type == 'classification' and v.lower() not in classification_tasks:
            pass  # Allow custom classification tasks
        elif model_type == 'regression' and v.lower() not in regression_tasks:
            pass  # Allow custom regression tasks
        elif model_type == 'clustering' and v.lower() not in clustering_tasks:
            pass  # Allow custom clustering tasks
        elif model_type == 'anomaly_detection' and v.lower() not in anomaly_tasks:
            pass  # Allow custom anomaly detection tasks
        
        return v.lower()
    
    @validator('cv_scores')
    def validate_cv_scores(cls, v: Optional[list[float]], values: dict[str, Any]) -> Optional[list[float]]:
        """Validate cross-validation scores."""
        if v is not None:
            if len(v) < 2:
                raise ValueError("Cross-validation requires at least 2 folds")
            
            # Auto-calculate CV statistics
            import numpy as np
            values['cv_mean'] = np.mean(v)
            values['cv_std'] = np.std(v)
            
            # Calculate 95% confidence interval
            mean = values['cv_mean']
            std = values['cv_std']
            n = len(v)
            margin = 1.96 * (std / np.sqrt(n))  # 95% CI
            values['cv_confidence_interval'] = (mean - margin, mean + margin)
        
        return v
    
    @validator('confusion_matrix')
    def validate_confusion_matrix(cls, v: Optional[list[list[int]]]) -> Optional[list[list[int]]]:
        """Validate confusion matrix format."""
        if v is not None:
            if not v or not all(len(row) == len(v) for row in v):
                raise ValueError("Confusion matrix must be square")
            
            if any(any(cell < 0 for cell in row) for row in v):
                raise ValueError("Confusion matrix values cannot be negative")
        
        return v
    
    def is_classification_model(self) -> bool:
        """Check if this is a classification model."""
        return self.model_type == 'classification'
    
    def is_regression_model(self) -> bool:
        """Check if this is a regression model."""
        return self.model_type == 'regression'
    
    def is_clustering_model(self) -> bool:
        """Check if this is a clustering model."""
        return self.model_type == 'clustering'
    
    def is_anomaly_detection_model(self) -> bool:
        """Check if this is an anomaly detection model."""
        return self.model_type == 'anomaly_detection'
    
    def get_primary_metric(self) -> Optional[float]:
        """Get the primary evaluation metric for the model type."""
        if self.is_classification_model():
            if self.f1_score and isinstance(self.f1_score, dict):
                if 'weighted' in self.f1_score:
                    return self.f1_score['weighted']
                elif 'macro' in self.f1_score:
                    return self.f1_score['macro']
            return self.accuracy
        
        elif self.is_regression_model():
            return self.r2_score or (1 - self.mse if self.mse else None)
        
        elif self.is_clustering_model():
            return self.silhouette_score
        
        elif self.is_anomaly_detection_model():
            return self.anomaly_f1
        
        return None
    
    def get_metric_by_name(self, metric_name: str) -> Optional[float]:
        """Get a specific metric by name."""
        metric_name = metric_name.lower()
        
        # Direct attribute access
        if hasattr(self, metric_name):
            value = getattr(self, metric_name)
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, dict):
                # For dict metrics, try to get weighted or macro average
                return value.get('weighted') or value.get('macro') or value.get('micro')
        
        # Check in nested dictionaries
        for attr_name in ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
            attr_value = getattr(self, attr_name, None)
            if isinstance(attr_value, dict) and metric_name in attr_value:
                return attr_value[metric_name]
        
        return None
    
    def calculate_model_score(self) -> float:
        """Calculate an overall model quality score (0-1)."""
        scores = []
        
        # Get primary metric
        primary = self.get_primary_metric()
        if primary is not None:
            scores.append(max(0, min(1, primary)))  # Normalize to 0-1
        
        # Add complexity penalty
        complexity_penalty = 0
        if self.training_time and self.prediction_time:
            # Penalize very slow models
            if self.training_time > 3600:  # 1 hour
                complexity_penalty += 0.1
            if self.prediction_time > 1.0:  # 1 second
                complexity_penalty += 0.1
        
        # Add cross-validation stability bonus
        stability_bonus = 0
        if self.cv_std is not None and self.cv_mean is not None:
            if self.cv_mean > 0:
                cv_coefficient = self.cv_std / self.cv_mean
                if cv_coefficient < 0.1:  # Low variability
                    stability_bonus = 0.1
        
        base_score = sum(scores) / len(scores) if scores else 0.5
        final_score = max(0, min(1, base_score - complexity_penalty + stability_bonus))
        
        return final_score
    
    def get_classification_summary(self) -> Optional[dict[str, Any]]:
        """Get classification-specific metrics summary."""
        if not self.is_classification_model():
            return None
        
        summary = {
            "accuracy": self.accuracy,
            "primary_f1": self.get_metric_by_name("f1_score"),
            "overall_precision": self.get_metric_by_name("precision"),
            "overall_recall": self.get_metric_by_name("recall"),
        }
        
        if self.confusion_matrix:
            # Calculate additional metrics from confusion matrix
            total = sum(sum(row) for row in self.confusion_matrix)
            summary["total_predictions"] = total
            
            if len(self.confusion_matrix) == 2:  # Binary classification
                tn, fp, fn, tp = (
                    self.confusion_matrix[0][0], self.confusion_matrix[0][1],
                    self.confusion_matrix[1][0], self.confusion_matrix[1][1]
                )
                summary["true_positives"] = tp
                summary["false_positives"] = fp
                summary["true_negatives"] = tn
                summary["false_negatives"] = fn
                summary["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return summary
    
    def get_regression_summary(self) -> Optional[dict[str, Any]]:
        """Get regression-specific metrics summary."""
        if not self.is_regression_model():
            return None
        
        return {
            "r2_score": self.r2_score,
            "adjusted_r2": self.adjusted_r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "explained_variance": self.r2_score,
        }
    
    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "evaluation_method": self.evaluation_method,
            "sample_size": self.sample_size,
            "primary_metric": self.get_primary_metric(),
            "model_score": self.calculate_model_score(),
        }
        
        # Add type-specific summaries
        if self.is_classification_model():
            summary["classification_metrics"] = self.get_classification_summary()
        elif self.is_regression_model():
            summary["regression_metrics"] = self.get_regression_summary()
        
        # Add cross-validation info
        if self.cv_mean is not None:
            summary["cross_validation"] = {
                "mean_score": self.cv_mean,
                "std_score": self.cv_std,
                "confidence_interval": self.cv_confidence_interval,
                "n_folds": len(self.cv_scores) if self.cv_scores else None
            }
        
        # Add timing information
        if self.training_time or self.prediction_time:
            summary["performance_timing"] = {
                "training_time": self.training_time,
                "prediction_time": self.prediction_time,
                "model_size": self.model_size
            }
        
        return summary
    
    @classmethod
    def from_sklearn_metrics(cls, y_true: Any, y_pred: Any, 
                           model_type: str, **kwargs: Any) -> MLModelMetrics:
        """Create MLModelMetrics from scikit-learn predictions."""
        from sklearn import metrics
        import numpy as np
        
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        
        model_metrics = {}
        
        if model_type.lower() == 'classification':
            # Classification metrics
            model_metrics.update({
                "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
                "precision": {
                    "macro": float(metrics.precision_score(y_true, y_pred, average='macro')),
                    "weighted": float(metrics.precision_score(y_true, y_pred, average='weighted'))
                },
                "recall": {
                    "macro": float(metrics.recall_score(y_true, y_pred, average='macro')),
                    "weighted": float(metrics.recall_score(y_true, y_pred, average='weighted'))
                },
                "f1_score": {
                    "macro": float(metrics.f1_score(y_true, y_pred, average='macro')),
                    "weighted": float(metrics.f1_score(y_true, y_pred, average='weighted'))
                },
                "confusion_matrix": metrics.confusion_matrix(y_true, y_pred).tolist(),
            })
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    roc_auc = metrics.roc_auc_score(y_true, y_pred)
                    model_metrics["roc_auc"] = {"binary": float(roc_auc)}
                except ValueError:
                    pass  # Skip if not applicable
        
        elif model_type.lower() == 'regression':
            # Regression metrics
            model_metrics.update({
                "mse": float(metrics.mean_squared_error(y_true, y_pred)),
                "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
                "r2_score": float(metrics.r2_score(y_true, y_pred)),
            })
            
            # Calculate RMSE
            model_metrics["rmse"] = float(np.sqrt(model_metrics["mse"]))
            
            # Calculate MAPE if no zero values in y_true
            if not np.any(y_true == 0):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                model_metrics["mape"] = float(mape)
        
        return cls(
            model_type=model_type.lower(),
            task_type=kwargs.get('task_type', 'general'),
            sample_size=len(y_true),
            **model_metrics,
            **kwargs
        )