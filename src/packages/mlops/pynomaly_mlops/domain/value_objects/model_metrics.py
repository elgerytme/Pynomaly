"""Model Metrics Value Object

Immutable value object for storing model performance metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ModelMetrics:
    """Model performance metrics value object.
    
    Stores various performance metrics for ML models with validation
    and comparison capabilities.
    """
    
    # Core Metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Regression Metrics
    mae: Optional[float] = None  # Mean Absolute Error
    mse: Optional[float] = None  # Mean Squared Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    r2_score: Optional[float] = None  # R-squared
    
    # Classification Metrics
    auc_roc: Optional[float] = None  # Area Under ROC Curve
    auc_pr: Optional[float] = None   # Area Under Precision-Recall Curve
    log_loss: Optional[float] = None
    
    # Anomaly Detection Metrics
    contamination_rate: Optional[float] = None
    detection_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    
    # Business Metrics
    business_value: Optional[float] = None
    cost_savings: Optional[float] = None
    revenue_impact: Optional[float] = None
    
    # Additional Custom Metrics
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Validate metric ranges
        percentage_metrics = [
            ("accuracy", self.accuracy),
            ("precision", self.precision), 
            ("recall", self.recall),
            ("f1_score", self.f1_score),
            ("auc_roc", self.auc_roc),
            ("auc_pr", self.auc_pr),
            ("r2_score", self.r2_score),
            ("contamination_rate", self.contamination_rate),
            ("detection_rate", self.detection_rate),
            ("false_positive_rate", self.false_positive_rate),
        ]
        
        for name, value in percentage_metrics:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
        
        # Validate positive metrics
        positive_metrics = [
            ("mae", self.mae),
            ("mse", self.mse),
            ("rmse", self.rmse),
        ]
        
        for name, value in positive_metrics:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        
        # Validate log_loss (must be non-negative)
        if self.log_loss is not None and self.log_loss < 0:
            raise ValueError(f"log_loss must be non-negative, got {self.log_loss}")
        
        # Initialize custom_metrics if None
        if self.custom_metrics is None:
            object.__setattr__(self, 'custom_metrics', {})
    
    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, Any]) -> "ModelMetrics":
        """Create ModelMetrics from dictionary.
        
        Args:
            metrics_dict: Dictionary containing metric values
            
        Returns:
            ModelMetrics instance
        """
        # Extract known metrics
        known_metrics = {
            "accuracy", "precision", "recall", "f1_score",
            "mae", "mse", "rmse", "r2_score",
            "auc_roc", "auc_pr", "log_loss",
            "contamination_rate", "detection_rate", "false_positive_rate",
            "business_value", "cost_savings", "revenue_impact"
        }
        
        kwargs = {}
        custom_metrics = {}
        
        for key, value in metrics_dict.items():
            if key in known_metrics:
                kwargs[key] = value
            elif key != "custom_metrics":
                custom_metrics[key] = value
        
        # Add any existing custom_metrics
        if "custom_metrics" in metrics_dict:
            custom_metrics.update(metrics_dict["custom_metrics"])
        
        if custom_metrics:
            kwargs["custom_metrics"] = custom_metrics
        
        return cls(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary containing all metrics
        """
        result = {}
        
        # Add non-None standard metrics
        standard_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "mae", "mse", "rmse", "r2_score", 
            "auc_roc", "auc_pr", "log_loss",
            "contamination_rate", "detection_rate", "false_positive_rate",
            "business_value", "cost_savings", "revenue_impact"
        ]
        
        for metric in standard_metrics:
            value = getattr(self, metric)
            if value is not None:
                result[metric] = value
        
        # Add custom metrics
        if self.custom_metrics:
            result.update(self.custom_metrics)
        
        return result
    
    def get_primary_metric(self) -> Optional[float]:
        """Get the primary metric for this model type.
        
        Returns the most relevant metric based on what's available.
        
        Returns:
            Primary metric value or None
        """
        # Order of preference for primary metric
        preference_order = [
            self.f1_score,
            self.accuracy,
            self.auc_roc,
            self.r2_score,
            self.detection_rate,
            self.precision,
            self.recall,
        ]
        
        for metric in preference_order:
            if metric is not None:
                return metric
        
        return None
    
    def is_better_than(self, other: "ModelMetrics", metric_name: Optional[str] = None) -> bool:
        """Compare this metrics with another set of metrics.
        
        Args:
            other: Other ModelMetrics to compare against
            metric_name: Specific metric to compare (optional)
            
        Returns:
            True if this metrics is better than other
        """
        if metric_name:
            # Compare specific metric
            self_value = getattr(self, metric_name, None)
            other_value = getattr(other, metric_name, None)
            
            if self_value is None or other_value is None:
                return False
            
            # For error metrics, lower is better
            if metric_name in ["mae", "mse", "rmse", "log_loss", "false_positive_rate"]:
                return self_value < other_value
            else:
                return self_value > other_value
        
        # Compare primary metrics
        self_primary = self.get_primary_metric()
        other_primary = other.get_primary_metric()
        
        if self_primary is None or other_primary is None:
            return False
        
        return self_primary > other_primary
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            "primary_metric": self.get_primary_metric(),
            "has_classification_metrics": any([
                self.accuracy, self.precision, self.recall, 
                self.f1_score, self.auc_roc, self.auc_pr
            ]),
            "has_regression_metrics": any([
                self.mae, self.mse, self.rmse, self.r2_score
            ]),
            "has_anomaly_metrics": any([
                self.contamination_rate, self.detection_rate, self.false_positive_rate
            ]),
            "has_business_metrics": any([
                self.business_value, self.cost_savings, self.revenue_impact
            ]),
            "total_metrics": len(self.to_dict()),
        }
        
        return summary
    
    def add_custom_metric(self, name: str, value: float) -> "ModelMetrics":
        """Add a custom metric and return new instance.
        
        Args:
            name: Metric name
            value: Metric value
            
        Returns:
            New ModelMetrics instance with added metric
        """
        new_custom_metrics = dict(self.custom_metrics) if self.custom_metrics else {}
        new_custom_metrics[name] = value
        
        return ModelMetrics(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1_score=self.f1_score,
            mae=self.mae,
            mse=self.mse,
            rmse=self.rmse,
            r2_score=self.r2_score,
            auc_roc=self.auc_roc,
            auc_pr=self.auc_pr,
            log_loss=self.log_loss,
            contamination_rate=self.contamination_rate,
            detection_rate=self.detection_rate,
            false_positive_rate=self.false_positive_rate,
            business_value=self.business_value,
            cost_savings=self.cost_savings,
            revenue_impact=self.revenue_impact,
            custom_metrics=new_custom_metrics,
        )
    
    def filter_metrics(self, metric_type: str) -> Dict[str, float]:
        """Filter metrics by type.
        
        Args:
            metric_type: Type of metrics to filter
                ('classification', 'regression', 'anomaly', 'business', 'custom')
            
        Returns:
            Dictionary of filtered metrics
        """
        if metric_type == "classification":
            metrics = {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "auc_roc": self.auc_roc,
                "auc_pr": self.auc_pr,
                "log_loss": self.log_loss,
            }
        elif metric_type == "regression":
            metrics = {
                "mae": self.mae,
                "mse": self.mse,
                "rmse": self.rmse,
                "r2_score": self.r2_score,
            }
        elif metric_type == "anomaly":
            metrics = {
                "contamination_rate": self.contamination_rate,
                "detection_rate": self.detection_rate,
                "false_positive_rate": self.false_positive_rate,
            }
        elif metric_type == "business":
            metrics = {
                "business_value": self.business_value,
                "cost_savings": self.cost_savings,
                "revenue_impact": self.revenue_impact,
            }
        elif metric_type == "custom":
            return dict(self.custom_metrics) if self.custom_metrics else {}
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        # Filter out None values
        return {k: v for k, v in metrics.items() if v is not None}