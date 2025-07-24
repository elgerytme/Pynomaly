"""Detection facade for simplified API access."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ...domain.entities.dataset import Dataset
from ...domain.entities.detection_result import DetectionResult
from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository


class DetectionFacade:
    """Simplified facade for anomaly detection operations."""
    
    def __init__(self):
        self._detection_service = DetectionService()
        self._model_repository = ModelRepository()
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        algorithm: str = "isolation_forest",
        parameters: Dict[str, Any] = None
    ) -> DetectionResult:
        """Detect anomalies in tabular data.
        
        Args:
            data: Input DataFrame
            algorithm: Detection algorithm to use
            parameters: Algorithm parameters
            
        Returns:
            Detection results with predictions and scores
        """
        dataset = Dataset.from_dataframe(data)
        return self._detection_service.detect(dataset, algorithm, parameters or {})
    
    def train_model(
        self,
        data: pd.DataFrame,
        algorithm: str = "isolation_forest",
        parameters: Dict[str, Any] = None
    ) -> str:
        """Train a new anomaly detection model.
        
        Args:
            data: Training DataFrame
            algorithm: Algorithm to train
            parameters: Training parameters
            
        Returns:
            Model ID
        """
        dataset = Dataset.from_dataframe(data)
        model = self._detection_service.train(dataset, algorithm, parameters or {})
        return self._model_repository.save(model)
    
    def compare_algorithms(
        self,
        data: pd.DataFrame,
        algorithms: List[str] = None,
        evaluation_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple detection algorithms.
        
        Args:
            data: Evaluation DataFrame with ground truth
            algorithms: List of algorithms to compare
            evaluation_metrics: Metrics to calculate
            
        Returns:
            Comparison results
        """
        if algorithms is None:
            algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        if evaluation_metrics is None:
            evaluation_metrics = ["precision", "recall", "f1_score", "auc_roc"]
            
        dataset = Dataset.from_dataframe(data)
        return self._detection_service.compare_algorithms(
            dataset, algorithms, evaluation_metrics
        )