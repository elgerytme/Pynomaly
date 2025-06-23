"""Use case for evaluating model performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.shared.protocols import DetectorProtocol, DetectorRepositoryProtocol


@dataclass
class EvaluateModelRequest:
    """Request for model evaluation."""
    
    detector_id: UUID
    test_dataset: Dataset
    cross_validate: bool = False
    n_folds: int = 5
    metrics: List[str] = None  # type: ignore


@dataclass
class EvaluateModelResponse:
    """Response from model evaluation."""
    
    detector_id: UUID
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    precision_recall_curve: Optional[Dict[str, np.ndarray]] = None
    cross_validation_scores: Optional[Dict[str, List[float]]] = None


class EvaluateModelUseCase:
    """Use case for evaluating anomaly detection model performance."""
    
    DEFAULT_METRICS = ["auc_roc", "auc_pr", "precision", "recall", "f1"]
    
    def __init__(self, detector_repository: DetectorRepositoryProtocol):
        """Initialize the use case.
        
        Args:
            detector_repository: Repository for detectors
        """
        self.detector_repository = detector_repository
    
    async def execute(self, request: EvaluateModelRequest) -> EvaluateModelResponse:
        """Execute model evaluation.
        
        Args:
            request: Evaluation request
            
        Returns:
            Evaluation response with metrics
            
        Raises:
            ValueError: If detector not found or dataset lacks labels
        """
        # Load detector
        detector = self.detector_repository.find_by_id(request.detector_id)
        if detector is None:
            raise ValueError(f"Detector {request.detector_id} not found")
        
        # Check if dataset has labels
        if not request.test_dataset.has_target:
            raise ValueError(
                "Test dataset must have target labels for evaluation"
            )
        
        # Use default metrics if none specified
        metrics_to_calculate = request.metrics or self.DEFAULT_METRICS
        
        if request.cross_validate:
            return await self._cross_validate(
                detector, request.test_dataset, request.n_folds, metrics_to_calculate
            )
        else:
            return await self._single_evaluation(
                detector, request.test_dataset, metrics_to_calculate
            )
    
    async def _single_evaluation(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        metrics_to_calculate: List[str]
    ) -> EvaluateModelResponse:
        """Perform single evaluation on test set."""
        # Get predictions
        result = detector.detect(dataset)
        true_labels = dataset.target.values  # type: ignore
        pred_labels = result.labels
        pred_scores = np.array([s.value for s in result.scores])
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            true_labels, pred_labels, pred_scores, metrics_to_calculate
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Calculate precision-recall curve if requested
        pr_curve = None
        if "auc_pr" in metrics_to_calculate:
            precision, recall, thresholds = precision_recall_curve(
                true_labels, pred_scores
            )
            pr_curve = {
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds
            }
        
        return EvaluateModelResponse(
            detector_id=detector.id,
            metrics=metrics,
            confusion_matrix=cm,
            precision_recall_curve=pr_curve
        )
    
    async def _cross_validate(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        n_folds: int,
        metrics_to_calculate: List[str]
    ) -> EvaluateModelResponse:
        """Perform cross-validation evaluation."""
        from sklearn.model_selection import StratifiedKFold
        
        # Initialize fold scores
        fold_scores: Dict[str, List[float]] = {
            metric: [] for metric in metrics_to_calculate
        }
        
        # Get features and labels
        X = dataset.features.values
        y = dataset.target.values  # type: ignore
        
        # Perform k-fold cross validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Create train/test datasets for this fold
            train_data = dataset.data.iloc[train_idx]
            test_data = dataset.data.iloc[test_idx]
            
            train_dataset = Dataset(
                name=f"{dataset.name}_train_fold{fold}",
                data=train_data,
                target_column=dataset.target_column
            )
            test_dataset = Dataset(
                name=f"{dataset.name}_test_fold{fold}",
                data=test_data,
                target_column=dataset.target_column
            )
            
            # Train on fold
            detector.fit(train_dataset)
            
            # Evaluate on fold
            result = detector.detect(test_dataset)
            
            true_labels = y[test_idx]
            pred_labels = result.labels
            pred_scores = np.array([s.value for s in result.scores])
            
            # Calculate metrics for this fold
            fold_metrics = self._calculate_metrics(
                true_labels, pred_labels, pred_scores, metrics_to_calculate
            )
            
            # Store fold results
            for metric, value in fold_metrics.items():
                fold_scores[metric].append(value)
        
        # Calculate average metrics
        avg_metrics = {
            metric: float(np.mean(scores))
            for metric, scores in fold_scores.items()
        }
        
        return EvaluateModelResponse(
            detector_id=detector.id,
            metrics=avg_metrics,
            cross_validation_scores=fold_scores
        )
    
    def _calculate_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        metrics_to_calculate: List[str]
    ) -> Dict[str, float]:
        """Calculate requested metrics."""
        metrics = {}
        
        # Basic counts
        tp = np.sum((true_labels == 1) & (pred_labels == 1))
        fp = np.sum((true_labels == 0) & (pred_labels == 1))
        fn = np.sum((true_labels == 1) & (pred_labels == 0))
        tn = np.sum((true_labels == 0) & (pred_labels == 0))
        
        # Calculate each metric
        if "auc_roc" in metrics_to_calculate:
            try:
                metrics["auc_roc"] = float(roc_auc_score(true_labels, pred_scores))
            except ValueError:
                # May fail if only one class present
                metrics["auc_roc"] = 0.5
        
        if "auc_pr" in metrics_to_calculate:
            try:
                metrics["auc_pr"] = float(
                    average_precision_score(true_labels, pred_scores)
                )
            except ValueError:
                metrics["auc_pr"] = 0.0
        
        if "precision" in metrics_to_calculate:
            metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        
        if "recall" in metrics_to_calculate:
            metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        if "f1" in metrics_to_calculate:
            precision = metrics.get("precision", tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recall = metrics.get("recall", tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            if precision + recall > 0:
                metrics["f1"] = float(2 * precision * recall / (precision + recall))
            else:
                metrics["f1"] = 0.0
        
        if "accuracy" in metrics_to_calculate:
            metrics["accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
        
        if "specificity" in metrics_to_calculate:
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        return metrics