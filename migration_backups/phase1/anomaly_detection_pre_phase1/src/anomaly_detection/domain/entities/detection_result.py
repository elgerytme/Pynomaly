"""Detection result entity for anomaly detection operations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionResult:
    """Container for anomaly detection results.
    
    Represents the output of an anomaly detection algorithm including
    predictions, confidence scores, metadata, and derived statistics.
    """
    
    predictions: npt.NDArray[np.integer]
    confidence_scores: Optional[npt.NDArray[np.floating]] = None
    algorithm: str = "unknown"
    metadata: Optional[dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Initialize derived fields after object creation."""
        if self.metadata is None:
            self.metadata = {}
        
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
            
        # Calculate statistics
        self.anomaly_count = int(np.sum(self.predictions == -1))
        self.normal_count = len(self.predictions) - self.anomaly_count
        self.total_samples = len(self.predictions)
        self.anomaly_rate = self.anomaly_count / self.total_samples if self.total_samples > 0 else 0.0
        
        # Set success flag
        self.success = True
    
    @property
    def anomalies(self) -> List[int]:
        """Get indices of detected anomalies."""
        return np.where(self.predictions == -1)[0].tolist()
    
    @property 
    def normal_indices(self) -> List[int]:
        """Get indices of normal samples."""
        return np.where(self.predictions == 1)[0].tolist()
    
    def get_anomaly_scores(self) -> Optional[npt.NDArray[np.floating]]:
        """Get confidence scores for anomalies only."""
        if self.confidence_scores is None:
            return None
        return self.confidence_scores[self.predictions == -1]
    
    def get_top_anomalies(self, n: int = 10) -> List[tuple[int, float]]:
        """Get top N anomalies by confidence score.
        
        Args:
            n: Number of top anomalies to return
            
        Returns:
            List of (index, confidence_score) tuples sorted by confidence
        """
        if self.confidence_scores is None:
            # Return first n anomalies if no confidence scores
            anomaly_indices = self.anomalies[:n]
            return [(idx, 0.0) for idx in anomaly_indices]
        
        anomaly_indices = self.anomalies
        anomaly_scores = self.confidence_scores[anomaly_indices]
        
        # Sort by confidence score (higher is more anomalous)
        sorted_pairs = sorted(zip(anomaly_indices, anomaly_scores), 
                            key=lambda x: x[1], reverse=True)
        
        return sorted_pairs[:n]
    
    def summary(self) -> dict[str, Any]:
        """Get summary statistics of detection results."""
        return {
            "algorithm": self.algorithm,
            "total_samples": self.total_samples,
            "anomaly_count": self.anomaly_count,
            "normal_count": self.normal_count,
            "anomaly_rate": self.anomaly_rate,
            "has_confidence_scores": self.confidence_scores is not None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "success": self.success,
            "metadata": self.metadata
        }