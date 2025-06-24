"""Performance metrics value object for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True)
class PerformanceMetrics:
    """Comprehensive performance metrics for anomaly detection models.
    
    This value object captures all relevant performance measurements
    for an anomaly detection model, including detection quality,
    computational efficiency, and resource usage.
    
    Attributes:
        accuracy: Overall accuracy of the model
        precision: Precision score for anomaly detection
        recall: Recall score for anomaly detection  
        f1_score: F1 score (harmonic mean of precision and recall)
        roc_auc: ROC AUC score (if applicable)
        pr_auc: Precision-Recall AUC score
        training_time: Time taken to train the model (seconds)
        inference_time: Average time per prediction (milliseconds)
        model_size: Size of the trained model (bytes)
        memory_usage: Peak memory usage during training (MB)
        cpu_usage: Average CPU usage during training (percentage)
        throughput_rps: Inference throughput (requests per second)
        true_positives: Number of true positive predictions
        true_negatives: Number of true negative predictions
        false_positives: Number of false positive predictions
        false_negatives: Number of false negative predictions
    """
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    model_size: int
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    throughput_rps: Optional[float] = None
    true_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate performance metrics."""
        # Validate score ranges (0.0 to 1.0)
        score_metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }
        
        for name, value in score_metrics.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value)}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
        
        # Validate optional AUC scores
        if self.roc_auc is not None:
            if not isinstance(self.roc_auc, (int, float)):
                raise TypeError(f"ROC AUC must be numeric, got {type(self.roc_auc)}")
            if not 0.0 <= self.roc_auc <= 1.0:
                raise ValueError(f"ROC AUC must be between 0.0 and 1.0, got {self.roc_auc}")
        
        if self.pr_auc is not None:
            if not isinstance(self.pr_auc, (int, float)):
                raise TypeError(f"PR AUC must be numeric, got {type(self.pr_auc)}")
            if not 0.0 <= self.pr_auc <= 1.0:
                raise ValueError(f"PR AUC must be between 0.0 and 1.0, got {self.pr_auc}")
        
        # Validate time metrics (positive values)
        if not isinstance(self.training_time, (int, float)) or self.training_time < 0:
            raise ValueError(f"Training time must be non-negative, got {self.training_time}")
        
        if not isinstance(self.inference_time, (int, float)) or self.inference_time < 0:
            raise ValueError(f"Inference time must be non-negative, got {self.inference_time}")
        
        # Validate model size (positive integer)
        if not isinstance(self.model_size, int) or self.model_size < 0:
            raise ValueError(f"Model size must be non-negative integer, got {self.model_size}")
        
        # Validate optional metrics
        if self.memory_usage is not None:
            if not isinstance(self.memory_usage, (int, float)) or self.memory_usage < 0:
                raise ValueError(f"Memory usage must be non-negative, got {self.memory_usage}")
        
        if self.cpu_usage is not None:
            if not isinstance(self.cpu_usage, (int, float)) or not 0 <= self.cpu_usage <= 100:
                raise ValueError(f"CPU usage must be between 0 and 100, got {self.cpu_usage}")
        
        if self.throughput_rps is not None:
            if not isinstance(self.throughput_rps, (int, float)) or self.throughput_rps < 0:
                raise ValueError(f"Throughput must be non-negative, got {self.throughput_rps}")
        
        # Validate confusion matrix components
        confusion_metrics = [
            self.true_positives, self.true_negatives,
            self.false_positives, self.false_negatives
        ]
        
        for metric in confusion_metrics:
            if metric is not None:
                if not isinstance(metric, int) or metric < 0:
                    raise ValueError(f"Confusion matrix values must be non-negative integers")
    
    @classmethod
    def from_confusion_matrix(
        cls,
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int,
        training_time: float,
        inference_time: float,
        model_size: int,
        **kwargs
    ) -> PerformanceMetrics:
        """Create metrics from confusion matrix values.
        
        Args:
            true_positives: Number of true positive predictions
            true_negatives: Number of true negative predictions
            false_positives: Number of false positive predictions
            false_negatives: Number of false negative predictions
            training_time: Training time in seconds
            inference_time: Inference time in milliseconds
            model_size: Model size in bytes
            **kwargs: Additional optional metrics
            
        Returns:
            PerformanceMetrics instance with calculated scores
        """
        # Calculate derived metrics
        total = true_positives + true_negatives + false_positives + false_negatives
        
        if total == 0:
            raise ValueError("Total predictions cannot be zero")
        
        accuracy = (true_positives + true_negatives) / total
        
        # Handle division by zero for precision and recall
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        
        # Calculate F1 score
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        return cls(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size,
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            **kwargs
        )
    
    @classmethod
    def create_minimal(
        cls,
        accuracy: float,
        training_time: float,
        inference_time: float,
        model_size: int
    ) -> PerformanceMetrics:
        """Create metrics with minimal required information.
        
        Args:
            accuracy: Overall accuracy
            training_time: Training time in seconds
            inference_time: Inference time in milliseconds
            model_size: Model size in bytes
            
        Returns:
            PerformanceMetrics with estimated precision, recall, F1
        """
        # Estimate precision and recall from accuracy for anomaly detection
        # This is a rough approximation assuming balanced performance
        estimated_precision = accuracy
        estimated_recall = accuracy
        estimated_f1 = accuracy
        
        return cls(
            accuracy=accuracy,
            precision=estimated_precision,
            recall=estimated_recall,
            f1_score=estimated_f1,
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size
        )
    
    @property
    def has_confusion_matrix(self) -> bool:
        """Check if confusion matrix values are available."""
        return all(
            metric is not None
            for metric in [
                self.true_positives, self.true_negatives,
                self.false_positives, self.false_negatives
            ]
        )
    
    @property
    def total_predictions(self) -> Optional[int]:
        """Get total number of predictions if confusion matrix available."""
        if self.has_confusion_matrix:
            return (
                self.true_positives + self.true_negatives +
                self.false_positives + self.false_negatives
            )
        return None
    
    @property
    def specificity(self) -> Optional[float]:
        """Calculate specificity (true negative rate)."""
        if self.true_negatives is not None and self.false_positives is not None:
            denominator = self.true_negatives + self.false_positives
            return self.true_negatives / denominator if denominator > 0 else 0.0
        return None
    
    @property
    def false_positive_rate(self) -> Optional[float]:
        """Calculate false positive rate."""
        specificity = self.specificity
        return 1.0 - specificity if specificity is not None else None
    
    @property
    def balanced_accuracy(self) -> Optional[float]:
        """Calculate balanced accuracy (average of sensitivity and specificity)."""
        specificity = self.specificity
        if specificity is not None:
            return (self.recall + specificity) / 2.0
        return None
    
    @property
    def model_size_mb(self) -> float:
        """Get model size in megabytes."""
        return self.model_size / (1024 * 1024)
    
    @property
    def training_time_minutes(self) -> float:
        """Get training time in minutes."""
        return self.training_time / 60.0
    
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score.
        
        Combines multiple metrics into a single score for ranking models.
        Higher is better.
        """
        # Weighted combination of key metrics
        detection_score = (self.precision + self.recall + self.f1_score) / 3.0
        
        # Efficiency bonus for fast inference (< 10ms gets bonus)
        efficiency_bonus = max(0, (10 - self.inference_time) / 10) * 0.1
        
        # Size penalty for very large models (> 100MB gets penalty)
        size_penalty = max(0, (self.model_size_mb - 100) / 1000) * 0.1
        
        return detection_score + efficiency_bonus - size_penalty
    
    def compare_with(self, other: PerformanceMetrics) -> Dict[str, float]:
        """Compare performance with another model.
        
        Args:
            other: Other performance metrics to compare with
            
        Returns:
            Dictionary of metric differences (positive = better)
        """
        if not isinstance(other, PerformanceMetrics):
            raise TypeError("Can only compare with another PerformanceMetrics")
        
        comparison = {
            "accuracy": self.accuracy - other.accuracy,
            "precision": self.precision - other.precision,
            "recall": self.recall - other.recall,
            "f1_score": self.f1_score - other.f1_score,
            "performance_score": self.performance_score - other.performance_score,
            
            # Efficiency metrics (negative is better for time/size)
            "training_time": other.training_time - self.training_time,
            "inference_time": other.inference_time - self.inference_time,
            "model_size": other.model_size - self.model_size,
        }
        
        # Add optional metrics if available in both
        if self.roc_auc is not None and other.roc_auc is not None:
            comparison["roc_auc"] = self.roc_auc - other.roc_auc
        
        if self.pr_auc is not None and other.pr_auc is not None:
            comparison["pr_auc"] = self.pr_auc - other.pr_auc
        
        if self.throughput_rps is not None and other.throughput_rps is not None:
            comparison["throughput_rps"] = self.throughput_rps - other.throughput_rps
        
        return comparison
    
    def is_better_than(self, other: PerformanceMetrics, metric: str = "performance_score") -> bool:
        """Check if this model is better than another on a specific metric.
        
        Args:
            other: Other performance metrics
            metric: Metric to compare on
            
        Returns:
            True if this model is better
        """
        comparison = self.compare_with(other)
        return comparison.get(metric, 0) > 0
    
    def to_dict(self) -> Dict[str, Union[float, int, None]]:
        """Convert to dictionary representation."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "training_time": self.training_time,
            "training_time_minutes": self.training_time_minutes,
            "inference_time": self.inference_time,
            "model_size": self.model_size,
            "model_size_mb": self.model_size_mb,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "throughput_rps": self.throughput_rps,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_predictions": self.total_predictions,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "balanced_accuracy": self.balanced_accuracy,
            "performance_score": self.performance_score,
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Performance(accuracy={self.accuracy:.3f}, "
            f"precision={self.precision:.3f}, "
            f"recall={self.recall:.3f}, "
            f"f1={self.f1_score:.3f})"
        )