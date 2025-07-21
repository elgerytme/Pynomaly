"""
Detection Result entity for machine learning domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np


@dataclass
class DetectionResult:
    """
    Represents the result of an anomaly detection operation.
    
    Attributes:
        result_id: Unique result identifier
        sample_id: ID of the sample that was analyzed
        detector_id: ID of the detector that produced this result
        score: Anomaly score (0.0 to 1.0)
        prediction: Binary prediction (True = anomaly, False = normal)
        confidence: Confidence in the prediction (0.0 to 1.0)
        features: Feature vector used for detection
        model_version: Version of the model used
        created_at: Timestamp when result was created
        metadata: Additional metadata about the detection
        explanation: Optional explanation of the result
    """
    
    result_id: UUID
    sample_id: str
    detector_id: str
    score: float
    prediction: bool
    confidence: float
    features: np.ndarray | None = None
    model_version: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    explanation: str | None = None
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def create(
        cls,
        sample_id: str,
        detector_id: str,
        score: float,
        prediction: bool,
        confidence: float,
        features: np.ndarray | None = None,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
        explanation: str | None = None,
    ) -> DetectionResult:
        """Create a new detection result."""
        return cls(
            result_id=uuid4(),
            sample_id=sample_id,
            detector_id=detector_id,
            score=score,
            prediction=prediction,
            confidence=confidence,
            features=features,
            model_version=model_version,
            created_at=datetime.now(),
            metadata=metadata or {},
            explanation=explanation,
        )
    
    def is_anomaly(self) -> bool:
        """Check if the result indicates an anomaly."""
        return self.prediction
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if the result has high confidence."""
        return self.confidence >= threshold
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance if available in metadata."""
        return self.metadata.get("feature_importance")
    
    def add_explanation(self, explanation: str) -> None:
        """Add an explanation to the result."""
        self.explanation = explanation
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata with a new key-value pair."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value