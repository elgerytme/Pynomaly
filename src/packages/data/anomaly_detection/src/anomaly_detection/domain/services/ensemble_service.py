"""Consolidated ensemble service for combining multiple anomaly detection algorithms."""

from __future__ import annotations

import logging
from typing import Any, Literal
import numpy as np
import numpy.typing as npt

from .detection_service import DetectionService, DetectionResult

logger = logging.getLogger(__name__)


class EnsembleService:
    """Service for ensemble anomaly detection methods.
    
    Combines multiple algorithms to improve detection accuracy and robustness.
    """
    
    def __init__(self, detection_service: DetectionService | None = None):
        """Initialize ensemble service."""
        self.detection_service = detection_service or DetectionService()
        
    def detect_with_ensemble(
        self,
        data: npt.NDArray[np.floating],
        algorithms: list[str] | None = None,
        combination_method: Literal["majority", "average", "max", "weighted"] = "majority",
        weights: list[float] | None = None,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies using ensemble of algorithms.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            algorithms: List of algorithms to use in ensemble
            combination_method: How to combine results
            weights: Weights for weighted combination
            **kwargs: Parameters passed to individual algorithms
            
        Returns:
            Combined DetectionResult
        """
        if algorithms is None:
            algorithms = ["iforest", "lof"]
            
        if len(algorithms) < 2:
            raise ValueError("Ensemble requires at least 2 algorithms")
            
        # Get predictions from each algorithm
        results = []
        for algorithm in algorithms:
            try:
                result = self.detection_service.detect_anomalies(
                    data, algorithm=algorithm, **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Algorithm {algorithm} failed: {e}")
                continue
                
        if not results:
            raise RuntimeError("All algorithms failed in ensemble")
            
        # Combine results
        combined_predictions = self._combine_predictions(
            [r.predictions for r in results],
            method=combination_method,
            weights=weights
        )
        
        # Combine scores if available
        combined_scores = None
        scores_list = [r.confidence_scores for r in results if r.confidence_scores is not None]
        if scores_list and len(scores_list) == len(results):
            combined_scores = self._combine_scores(
                scores_list,
                method=combination_method,
                weights=weights
            )
            
        return DetectionResult(
            predictions=combined_predictions,
            confidence_scores=combined_scores,
            algorithm=f"ensemble({','.join(algorithms)})",
            metadata={
                "algorithms": algorithms,
                "combination_method": combination_method,
                "weights": weights,
                "individual_results": len(results)
            }
        )
    
    def _combine_predictions(
        self,
        predictions_list: list[npt.NDArray[np.integer]],
        method: str,
        weights: list[float] | None = None
    ) -> npt.NDArray[np.integer]:
        """Combine predictions from multiple algorithms.
        
        Assumes sklearn format: -1 for anomaly, 1 for normal.
        """
        predictions_array = np.array(predictions_list)  # shape: (n_algorithms, n_samples)
        
        # Convert to binary format for easier combination (0=normal, 1=anomaly)
        binary_predictions = (predictions_array == -1).astype(int)
        
        if method == "majority":
            # Majority voting - if more than half predict anomaly
            combined_binary = (np.sum(binary_predictions, axis=0) > len(predictions_list) / 2).astype(int)
            
        elif method == "average":
            # Average and threshold at 0.5
            combined_binary = (np.mean(binary_predictions.astype(float), axis=0) > 0.5).astype(int)
            
        elif method == "max":
            # Any algorithm predicts anomaly
            combined_binary = np.max(binary_predictions, axis=0)
            
        elif method == "weighted":
            if weights is None:
                weights = [1.0] * len(predictions_list)
            if len(weights) != len(predictions_list):
                raise ValueError("Number of weights must match number of algorithms")
                
            weights_array = np.array(weights)
            weights_array = weights_array / np.sum(weights_array)  # normalize
            
            weighted_sum = np.sum(
                binary_predictions * weights_array.reshape(-1, 1), axis=0
            )
            combined_binary = (weighted_sum > 0.5).astype(int)
            
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        # Convert back to sklearn format (-1 for anomaly, 1 for normal)
        result = np.where(combined_binary == 1, -1, 1)
        return result.astype(np.integer)
    
    def _combine_scores(
        self,
        scores_list: list[npt.NDArray[np.floating]],
        method: str,
        weights: list[float] | None = None
    ) -> npt.NDArray[np.floating]:
        """Combine anomaly scores from multiple algorithms."""
        scores_array = np.array(scores_list)  # shape: (n_algorithms, n_samples)
        
        if method in ["majority", "average"]:
            return np.mean(scores_array, axis=0)
            
        elif method == "max":
            return np.max(scores_array, axis=0)
            
        elif method == "weighted":
            if weights is None:
                weights = [1.0] * len(scores_list)
            if len(weights) != len(scores_list):
                raise ValueError("Number of weights must match number of algorithms")
                
            weights_array = np.array(weights)
            weights_array = weights_array / np.sum(weights_array)  # normalize
            
            return np.sum(scores_array * weights_array.reshape(-1, 1), axis=0)
            
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def optimize_ensemble(
        self,
        data: npt.NDArray[np.floating],
        algorithms: list[str] | None = None,
        validation_split: float = 0.2,
        ground_truth: npt.NDArray[np.integer] | None = None
    ) -> dict[str, Any]:
        """Optimize ensemble parameters using validation data.
        
        Args:
            data: Training/validation data
            algorithms: Algorithms to include in ensemble
            validation_split: Fraction of data to use for validation
            ground_truth: True labels for optimization
            
        Returns:
            Dictionary with optimized parameters
        """
        if algorithms is None:
            algorithms = self.detection_service.list_available_algorithms()
            
        # Split data for validation
        n_samples = len(data)
        n_train = int(n_samples * (1 - validation_split))
        
        train_data = data[:n_train]
        val_data = data[n_train:]
        
        if ground_truth is not None:
            val_truth = ground_truth[n_train:]
        else:
            val_truth = None
            
        best_score = -np.inf
        best_params = {}
        
        # Try different combination methods
        for method in ["majority", "average", "weighted"]:
            try:
                if method == "weighted":
                    # Simple weight optimization - try equal weights and algorithm-specific
                    for weights in [None, [0.6, 0.4], [0.4, 0.6]]:
                        result = self.detect_with_ensemble(
                            val_data,
                            algorithms=algorithms[:2],  # Limit for simplicity
                            combination_method=method,
                            weights=weights
                        )
                        
                        score = self._evaluate_result(result, val_truth)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "algorithms": algorithms[:2],
                                "combination_method": method,
                                "weights": weights,
                                "score": score
                            }
                else:
                    result = self.detect_with_ensemble(
                        val_data,
                        algorithms=algorithms[:3],  # Limit for performance
                        combination_method=method
                    )
                    
                    score = self._evaluate_result(result, val_truth)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "algorithms": algorithms[:3],
                            "combination_method": method,
                            "weights": None,
                            "score": score
                        }
                        
            except Exception as e:
                logger.warning(f"Optimization failed for method {method}: {e}")
                continue
                
        return best_params
    
    def _evaluate_result(
        self,
        result: DetectionResult,
        ground_truth: npt.NDArray[np.integer] | None
    ) -> float:
        """Evaluate detection result quality."""
        if ground_truth is not None:
            # Use precision-recall F1 score if ground truth available
            from sklearn.metrics import f1_score
            return f1_score(ground_truth, result.predictions, average='binary')
        else:
            # Use a heuristic based on anomaly ratio (prefer moderate anomaly rates)
            anomaly_rate = result.anomaly_count / len(result.predictions)
            # Penalize extreme anomaly rates (too few or too many anomalies)
            if anomaly_rate < 0.01 or anomaly_rate > 0.5:
                return 0.1
            else:
                # Prefer anomaly rates around 5-15%
                optimal_rate = 0.1
                score = 1.0 - abs(anomaly_rate - optimal_rate) / optimal_rate
                return max(0.1, score)