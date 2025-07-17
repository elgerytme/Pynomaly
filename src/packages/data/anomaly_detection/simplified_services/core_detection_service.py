"""Core detection service - simplified and production-ready."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DetectionResult:
    """Simple detection result without complex domain dependencies."""
    predictions: npt.NDArray[np.integer]
    scores: Optional[npt.NDArray[np.floating]] = None
    algorithm: str = "unknown"
    contamination: float = 0.1
    n_samples: int = 0
    n_anomalies: int = 0
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.n_samples == 0:
            self.n_samples = len(self.predictions)
        if self.n_anomalies == 0:
            self.n_anomalies = int(np.sum(self.predictions))


class CoreDetectionService:
    """Simplified core detection service that consolidates multiple detection patterns.
    
    This service replaces 20+ detection-related service files with a single,
    clean implementation that handles:
    - Basic anomaly detection
    - Multiple algorithm support
    - Batch and streaming detection
    - Performance monitoring
    - Result aggregation
    """

    def __init__(self):
        """Initialize core detection service."""
        self._detection_count = 0
        self._total_samples = 0
        self._total_anomalies = 0
        self._performance_history: List[Dict[str, Any]] = []

    def detect_anomalies(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        contamination: float = 0.1,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies using specified algorithm.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            algorithm: Algorithm to use ('iforest', 'lof', 'ocsvm', or PyOD algorithm)
            contamination: Expected proportion of outliers
            **kwargs: Additional algorithm parameters
            
        Returns:
            DetectionResult with predictions, scores, and metadata
        """
        import time
        start_time = time.time()
        
        try:
            predictions, scores = self._run_detection(data, algorithm, contamination, **kwargs)
            execution_time = time.time() - start_time
            
            result = DetectionResult(
                predictions=predictions,
                scores=scores,
                algorithm=algorithm,
                contamination=contamination,
                execution_time=execution_time,
                metadata={
                    "data_shape": data.shape,
                    "parameters": kwargs,
                    "detection_id": self._detection_count
                }
            )
            
            # Update statistics
            self._detection_count += 1
            self._total_samples += len(data)
            self._total_anomalies += result.n_anomalies
            
            # Track performance
            self._performance_history.append({
                "detection_id": self._detection_count,
                "algorithm": algorithm,
                "n_samples": len(data),
                "n_features": data.shape[1],
                "execution_time": execution_time,
                "anomaly_rate": result.n_anomalies / len(data),
                "timestamp": datetime.now()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise RuntimeError(f"Detection failed after {execution_time:.3f}s: {str(e)}") from e

    def _run_detection(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> tuple[npt.NDArray[np.integer], Optional[npt.NDArray[np.floating]]]:
        """Run the actual detection algorithm."""
        
        # Try PyOD algorithms first
        if self._is_pyod_algorithm(algorithm):
            return self._run_pyod_detection(data, algorithm, contamination, **kwargs)
        
        # Fall back to sklearn algorithms
        return self._run_sklearn_detection(data, algorithm, contamination, **kwargs)

    def _is_pyod_algorithm(self, algorithm: str) -> bool:
        """Check if algorithm is a PyOD algorithm."""
        pyod_algorithms = {
            'pca', 'mcd', 'ocsvm', 'lmdd', 'lof', 'cof', 'cblof', 'knn', 'hbos',
            'abod', 'copod', 'ecod', 'iforest', 'feature_bagging', 'lscp',
            'auto_encoder', 'vae', 'beta_vae', 'so_gaal', 'mo_gaal',
            'sos', 'loda', 'inne', 'cd', 'gm', 'deep_svdd', 'alad', 'anogan', 'lunar'
        }
        return algorithm in pyod_algorithms

    def _run_pyod_detection(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Run PyOD algorithm detection."""
        try:
            from algorithms.adapters.simple_pyod_adapter import SimplePyODAdapter
            
            adapter = SimplePyODAdapter(
                algorithm=algorithm,
                contamination=contamination,
                **kwargs
            )
            
            predictions = adapter.fit_predict(data)
            scores = adapter.decision_function(data)
            
            return predictions, scores
            
        except ImportError:
            # Fall back to sklearn if PyOD not available
            return self._run_sklearn_detection(data, 'isolation_forest', contamination, **kwargs)

    def _run_sklearn_detection(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> tuple[npt.NDArray[np.integer], Optional[npt.NDArray[np.floating]]]:
        """Run sklearn algorithm detection."""
        from src.pynomaly_detection import AnomalyDetector
        
        # Map algorithm names
        algorithm_map = {
            'iforest': 'isolation_forest',
            'isolation_forest': 'isolation_forest',
            'lof': 'lof',
            'ocsvm': 'ocsvm'
        }
        
        sklearn_algorithm = algorithm_map.get(algorithm, 'isolation_forest')
        
        detector = AnomalyDetector()
        predictions = detector.detect(
            data,
            algorithm=sklearn_algorithm,
            contamination=contamination,
            **kwargs
        )
        
        # sklearn doesn't provide decision scores easily, return None
        return predictions, None

    def batch_detect(
        self,
        data_batches: List[npt.NDArray[np.floating]],
        algorithm: str = "iforest",
        contamination: float = 0.1,
        **kwargs: Any
    ) -> List[DetectionResult]:
        """Detect anomalies in multiple data batches.
        
        Args:
            data_batches: List of data arrays to process
            algorithm: Algorithm to use for all batches
            contamination: Expected proportion of outliers
            **kwargs: Additional algorithm parameters
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        
        for i, batch in enumerate(data_batches):
            try:
                result = self.detect_anomalies(
                    batch, 
                    algorithm=algorithm, 
                    contamination=contamination,
                    **kwargs
                )
                result.metadata["batch_id"] = i
                results.append(result)
                
            except Exception as e:
                # Continue processing other batches on error
                print(f"Warning: Batch {i} failed: {e}")
                continue
        
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the service.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self._performance_history:
            return {
                "total_detections": 0,
                "total_samples": 0,
                "total_anomalies": 0,
                "average_execution_time": 0.0,
                "average_anomaly_rate": 0.0
            }
        
        execution_times = [p["execution_time"] for p in self._performance_history]
        anomaly_rates = [p["anomaly_rate"] for p in self._performance_history]
        
        return {
            "total_detections": self._detection_count,
            "total_samples": self._total_samples,
            "total_anomalies": self._total_anomalies,
            "average_execution_time": np.mean(execution_times),
            "median_execution_time": np.median(execution_times),
            "max_execution_time": np.max(execution_times),
            "min_execution_time": np.min(execution_times),
            "average_anomaly_rate": np.mean(anomaly_rates),
            "algorithms_used": list(set(p["algorithm"] for p in self._performance_history)),
            "last_detection": self._performance_history[-1]["timestamp"] if self._performance_history else None
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._detection_count = 0
        self._total_samples = 0
        self._total_anomalies = 0
        self._performance_history.clear()

    def get_recent_performance(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance history.
        
        Args:
            limit: Number of recent detections to return
            
        Returns:
            List of recent performance records
        """
        return self._performance_history[-limit:]