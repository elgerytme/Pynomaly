"""Detection metrics value objects."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectionMetrics:
    """Metrics for anomaly detection performance."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    
    def __post_init__(self):
        """Validate metrics are within expected ranges."""
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            value = getattr(self, metric_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} must be between 0.0 and 1.0, got {value}")
        
        if self.auc_roc is not None and not 0.0 <= self.auc_roc <= 1.0:
            raise ValueError(f"AUC-ROC must be between 0.0 and 1.0, got {self.auc_roc}")
    
    @property
    def is_good_performance(self) -> bool:
        """Check if metrics indicate good performance (subjective threshold)."""
        return (
            self.accuracy >= 0.8 and
            self.precision >= 0.7 and
            self.recall >= 0.7 and
            self.f1_score >= 0.7
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate
        }