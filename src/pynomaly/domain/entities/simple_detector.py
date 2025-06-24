"""Simple concrete detector implementation for testing and basic usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from .detector import Detector
from .dataset import Dataset
from .detection_result import DetectionResult


@dataclass
class SimpleDetector(Detector):
    """Simple concrete implementation of Detector for testing purposes."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = "simple_detector"
    algorithm_name: str = "simple"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = None
    is_fitted: bool = False
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the detector (dummy implementation)."""
        self.trained_at = datetime.utcnow()
        self.is_fitted = True
    
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies (dummy implementation)."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before detection")
        
        # Simple mock detection - mark random points as anomalies
        import numpy as np
        n_samples = len(dataset.data)
        scores = np.random.random(n_samples)
        labels = (scores > 0.9).astype(int)
        
        # Create simple mock detection result
        class MockResult:
            def __init__(self, scores, labels):
                self.scores = scores
                self.labels = labels
                self.threshold = 0.9
                self.metadata = {}
        
        return MockResult(scores, labels)
    
    def score(self, dataset: Dataset) -> float:
        """Score the dataset (dummy implementation)."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Return a random score between 0 and 1
        import numpy as np
        return float(np.random.random())