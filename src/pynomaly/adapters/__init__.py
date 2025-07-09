"""
Adapter façades for preventing circular imports.

This module provides thin façades that allow domain and application layers 
to import from adapters without creating circular dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynomaly.domain.entities.anomaly import Anomaly
    from pynomaly.domain.entities.dataset import Dataset
    from pynomaly.domain.entities.detection_result import DetectionResult
    from pynomaly.domain.entities.detector import Detector

# Re-export domain entities through adapters to break circular imports
__all__ = [
    "AnomalyFacade",
    "DatasetFacade", 
    "DetectionResultFacade",
    "DetectorFacade",
]


class AnomalyFacade:
    """Thin façade for Anomaly entity."""
    
    @staticmethod
    def create(*args, **kwargs) -> Anomaly:
        """Create an Anomaly instance."""
        from pynomaly.domain.entities.anomaly import Anomaly
        return Anomaly(*args, **kwargs)
    
    @staticmethod
    def from_dict(data: dict) -> Anomaly:
        """Create Anomaly from dictionary."""
        from pynomaly.domain.entities.anomaly import Anomaly
        return Anomaly(
            score=data["score"],
            data_point=data["data_point"],
            detector_name=data["detector_name"],
            explanation=data.get("explanation"),
        )


class DatasetFacade:
    """Thin façade for Dataset entity."""
    
    @staticmethod
    def create(*args, **kwargs) -> Dataset:
        """Create a Dataset instance."""
        from pynomaly.domain.entities.dataset import Dataset
        return Dataset(*args, **kwargs)
    
    @staticmethod
    def from_pandas(df, name: str) -> Dataset:
        """Create Dataset from pandas DataFrame."""
        from pynomaly.domain.entities.dataset import Dataset
        return Dataset.from_pandas(df, name)


class DetectionResultFacade:
    """Thin façade for DetectionResult entity."""
    
    @staticmethod
    def create(*args, **kwargs) -> DetectionResult:
        """Create a DetectionResult instance."""
        from pynomaly.domain.entities.detection_result import DetectionResult
        return DetectionResult(*args, **kwargs)
    
    @staticmethod
    def empty(dataset_id: str) -> DetectionResult:
        """Create an empty DetectionResult."""
        from pynomaly.domain.entities.detection_result import DetectionResult
        return DetectionResult(dataset_id=dataset_id, anomalies=[], metadata={})


class DetectorFacade:
    """Thin façade for Detector entity."""
    
    @staticmethod
    def create(*args, **kwargs) -> Detector:
        """Create a Detector instance."""
        from pynomaly.domain.entities.detector import Detector
        return Detector(*args, **kwargs)
    
    @staticmethod
    def from_config(config: dict) -> Detector:
        """Create Detector from configuration."""
        from pynomaly.domain.entities.detector import Detector
        return Detector.from_config(config)
