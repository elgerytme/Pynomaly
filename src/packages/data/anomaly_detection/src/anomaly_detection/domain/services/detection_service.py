"""Consolidated detection service for anomaly detection operations."""

from __future__ import annotations

import logging
from typing import Any, Protocol
import numpy as np
import numpy.typing as npt

from ..entities.detection_result import DetectionResult
from ...infrastructure.logging import get_logger, log_decorator, timing_decorator
from ...infrastructure.logging.error_handler import ErrorHandler, InputValidationError, AlgorithmError
from ...infrastructure.monitoring import MetricsCollector, get_metrics_collector

logger = get_logger(__name__)
error_handler = ErrorHandler(logger._logger)
metrics_collector = get_metrics_collector()


class AlgorithmAdapter(Protocol):
    """Protocol for algorithm adapters."""
    
    def fit(self, data: npt.NDArray[np.floating]) -> None:
        """Fit the algorithm on data."""
        ...
    
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies in data."""
        ...
    
    def fit_predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Fit and predict in one step."""
        ...



class DetectionService:
    """Consolidated service for anomaly detection operations.
    
    This service consolidates functionality from multiple detection services
    into a single, clean interface.
    """
    
    def __init__(self):
        """Initialize detection service."""
        self._adapters: dict[str, AlgorithmAdapter] = {}
        self._fitted_models: dict[str, Any] = {}
    
    def register_adapter(self, name: str, adapter: AlgorithmAdapter) -> None:
        """Register an algorithm adapter."""
        self._adapters[name] = adapter
        
    def detect_anomalies(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        contamination: float = 0.1,
        **kwargs: Any
    ) -> DetectionResult:
        """Detect anomalies in data using specified algorithm.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            algorithm: Algorithm name to use
            contamination: Expected proportion of anomalies
            **kwargs: Additional algorithm parameters
            
        Returns:
            DetectionResult with predictions and metadata
        """
        try:
            # Try to use registered adapter first
            if algorithm in self._adapters:
                adapter = self._adapters[algorithm]
                predictions = adapter.fit_predict(data)
            else:
                # Fall back to built-in algorithms
                predictions = self._detect_with_builtin(
                    data, algorithm, contamination, **kwargs
                )
            
            # Get confidence scores if available
            confidence_scores = self._get_confidence_scores(data, algorithm, contamination, **kwargs)
            
            result = DetectionResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                algorithm=algorithm,
                metadata={
                    "contamination": contamination,
                    "data_shape": data.shape,
                    "algorithm_params": kwargs
                }
            )
            
            # Log detection statistics
            logger.info("Anomaly detection completed successfully", 
                       algorithm=algorithm,
                       total_samples=result.total_samples,
                       anomalies_detected=result.anomaly_count,
                       anomaly_rate=result.anomaly_rate)
            
            # Record metrics
            operation_id = metrics_collector.start_operation(f"detection_{algorithm}")
            duration_ms = metrics_collector.end_operation(operation_id, success=True)
            
            metrics_collector.record_model_metrics(
                model_id="realtime",
                algorithm=algorithm,
                operation="detect",
                duration_ms=duration_ms,
                success=True,
                samples_processed=result.total_samples,
                anomalies_detected=result.anomaly_count
            )
            
            # Log data quality metrics
            logger.log_data_quality(
                dataset_name=f"detection_input_{algorithm}",
                quality_metrics={
                    "samples": result.total_samples,
                    "features": data.shape[1] if len(data.shape) > 1 else 1,
                    "anomaly_rate": result.anomaly_rate,
                    "contamination_expected": contamination
                }
            )
            
            return result
            
        except Exception as e:
            return error_handler.handle_error(
                error=e,
                context={
                    "algorithm": algorithm,
                    "data_shape": data.shape,
                    "contamination": contamination
                },
                operation="anomaly_detection",
                reraise=True
            )
    
    @log_decorator(operation="model_fitting", log_args=True, log_duration=True)
    def fit(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest",
        **kwargs: Any
    ) -> DetectionService:
        """Fit a detector on training data.
        
        Args:
            data: Training data
            algorithm: Algorithm name
            **kwargs: Algorithm parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            InputValidationError: If input parameters are invalid
            AlgorithmError: If model fitting fails
        """
        # Input validation
        self._validate_detection_inputs(data, algorithm, 0.1)  # contamination not used in fit
        
        logger.info("Starting model fitting", 
                   algorithm=algorithm, 
                   data_shape=data.shape)
        
        try:
            if algorithm in self._adapters:
                adapter = self._adapters[algorithm]
                adapter.fit(data)
                self._fitted_models[algorithm] = adapter
                logger.debug("Fitted registered adapter", adapter_name=algorithm)
            else:
                # Handle built-in algorithms
                model = self._fit_builtin(data, algorithm, **kwargs)
                self._fitted_models[algorithm] = model
                logger.debug("Fitted built-in algorithm", algorithm=algorithm)
            
            logger.info("Model fitting completed successfully", 
                       algorithm=algorithm,
                       training_samples=data.shape[0])
            
            return self
            
        except Exception as e:
            return error_handler.handle_error(
                error=e,
                context={
                    "algorithm": algorithm,
                    "data_shape": data.shape,
                    "operation": "fit"
                },
                operation="model_fitting",
                reraise=True
            )
    
    @log_decorator(operation="model_prediction", log_args=True, log_duration=True)
    def predict(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest"
    ) -> DetectionResult:
        """Predict anomalies using fitted model.
        
        Args:
            data: Data to predict on
            algorithm: Algorithm name
            
        Returns:
            DetectionResult with predictions
            
        Raises:
            InputValidationError: If input parameters are invalid
            AlgorithmError: If model prediction fails
        """
        # Validate inputs
        if data.size == 0:
            raise InputValidationError("Input data cannot be empty")
        
        if algorithm not in self._fitted_models:
            raise AlgorithmError(
                f"Algorithm {algorithm} not fitted. Call fit() first.",
                details={"available_models": list(self._fitted_models.keys())}
            )
        
        logger.info("Starting model prediction", 
                   algorithm=algorithm, 
                   data_shape=data.shape)
        
        try:
            model = self._fitted_models[algorithm]
            
            if hasattr(model, 'predict'):
                predictions = model.predict(data)
            else:
                # Handle adapter protocol
                predictions = model.predict(data)
            
            result = DetectionResult(predictions=predictions, algorithm=algorithm)
            
            logger.info("Model prediction completed successfully", 
                       algorithm=algorithm,
                       prediction_samples=data.shape[0],
                       anomalies_predicted=result.anomaly_count)
            
            return result
            
        except Exception as e:
            return error_handler.handle_error(
                error=e,
                context={
                    "algorithm": algorithm,
                    "data_shape": data.shape,
                    "operation": "predict"
                },
                operation="model_prediction",
                reraise=True
            )
    
    @timing_decorator(operation="builtin_detection")
    def _detect_with_builtin(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Detect using built-in algorithm implementations."""
        logger.debug("Running built-in detection algorithm", 
                    algorithm=algorithm,
                    contamination=contamination)
        
        if algorithm == "iforest":
            return self._isolation_forest(data, contamination, **kwargs)
        elif algorithm == "lof":
            return self._local_outlier_factor(data, contamination, **kwargs)
        else:
            raise AlgorithmError(
                f"Unknown algorithm: {algorithm}",
                details={
                    "requested_algorithm": algorithm,
                    "available_algorithms": self.list_available_algorithms()
                }
            )
    
    @timing_decorator(operation="fit_builtin_algorithm")
    def _fit_builtin(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        **kwargs: Any
    ) -> Any:
        """Fit built-in algorithm."""
        logger.debug("Fitting built-in algorithm", 
                    algorithm=algorithm,
                    data_samples=data.shape[0])
        
        try:
            if algorithm == "iforest":
                try:
                    from sklearn.ensemble import IsolationForest
                    model = IsolationForest(**kwargs)
                    model.fit(data)
                    logger.debug("IsolationForest model fitted successfully")
                    return model
                except ImportError as e:
                    raise AlgorithmError(
                        "scikit-learn required for IsolationForest",
                        details={"missing_dependency": "scikit-learn"},
                        original_error=e
                    )
            else:
                raise AlgorithmError(
                    f"Unknown algorithm for fitting: {algorithm}",
                    details={"available_algorithms": self.list_available_algorithms()}
                )
        except Exception as e:
            if isinstance(e, AlgorithmError):
                raise
            raise AlgorithmError(
                f"Failed to fit {algorithm} algorithm: {str(e)}",
                details={"algorithm": algorithm, "data_shape": data.shape},
                original_error=e
            )
    
    @timing_decorator(operation="isolation_forest")
    def _isolation_forest(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Run Isolation Forest algorithm."""
        try:
            from sklearn.ensemble import IsolationForest
            
            logger.debug("Running Isolation Forest", 
                        contamination=contamination,
                        data_samples=data.shape[0])
            
            model = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get('random_state', 42),
                **kwargs
            )
            predictions = model.fit_predict(data)
            # Keep sklearn format: -1 for anomaly, 1 for normal
            result = predictions.astype(np.integer)
            
            anomaly_count = np.sum(result == -1)
            logger.debug("Isolation Forest completed", 
                        anomalies_found=anomaly_count,
                        normal_samples=np.sum(result == 1))
            
            return result
            
        except ImportError as e:
            raise AlgorithmError(
                "scikit-learn required for IsolationForest",
                details={"missing_dependency": "scikit-learn"},
                original_error=e
            )
        except Exception as e:
            raise AlgorithmError(
                f"Isolation Forest execution failed: {str(e)}",
                details={"contamination": contamination, "data_shape": data.shape},
                original_error=e
            )
    
    @timing_decorator(operation="local_outlier_factor")
    def _local_outlier_factor(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Run Local Outlier Factor algorithm."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            logger.debug("Running Local Outlier Factor", 
                        contamination=contamination,
                        data_samples=data.shape[0])
            
            model = LocalOutlierFactor(
                contamination=contamination,
                **kwargs
            )
            predictions = model.fit_predict(data)
            # Keep sklearn format: -1 for anomaly, 1 for normal
            result = predictions.astype(np.integer)
            
            anomaly_count = np.sum(result == -1)
            logger.debug("Local Outlier Factor completed", 
                        anomalies_found=anomaly_count,
                        normal_samples=np.sum(result == 1))
            
            return result
            
        except ImportError as e:
            raise AlgorithmError(
                "scikit-learn required for LocalOutlierFactor",
                details={"missing_dependency": "scikit-learn"},
                original_error=e
            )
        except Exception as e:
            raise AlgorithmError(
                f"Local Outlier Factor execution failed: {str(e)}",
                details={"contamination": contamination, "data_shape": data.shape},
                original_error=e
            )
    
    def list_available_algorithms(self) -> list[str]:
        """List all available algorithms."""
        builtin = ["iforest", "lof"]
        registered = list(self._adapters.keys())
        return builtin + registered
    
    def get_algorithm_info(self, algorithm: str) -> dict[str, Any]:
        """Get information about an algorithm."""
        info = {"name": algorithm, "type": "unknown"}
        
        if algorithm in ["iforest", "lof"]:
            info["type"] = "builtin"
            info["requires"] = ["scikit-learn"]
        elif algorithm in self._adapters:
            info["type"] = "registered_adapter"
            
        return info
    
    def _validate_detection_inputs(
        self, 
        data: npt.NDArray[np.floating], 
        algorithm: str, 
        contamination: float
    ) -> None:
        """Validate inputs for detection operations.
        
        Args:
            data: Input data array
            algorithm: Algorithm name
            contamination: Contamination rate
            
        Raises:
            InputValidationError: If any input is invalid
        """
        if data.size == 0:
            raise InputValidationError("Input data cannot be empty")
        
        if len(data.shape) != 2:
            raise InputValidationError(
                f"Input data must be 2-dimensional, got shape {data.shape}"
            )
        
        if data.shape[0] < 2:
            raise InputValidationError(
                f"Need at least 2 samples, got {data.shape[0]}"
            )
        
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise InputValidationError("Algorithm name must be a non-empty string")
        
        if not (0.001 <= contamination <= 0.5):
            raise InputValidationError(
                f"Contamination must be between 0.001 and 0.5, got {contamination}"
            )
        
        # Check for non-finite values
        if not np.isfinite(data).all():
            raise InputValidationError("Input data contains non-finite values (NaN or inf)")
        
        logger.debug("Input validation passed", 
                    data_shape=data.shape,
                    algorithm=algorithm,
                    contamination=contamination)
    
    def _get_confidence_scores(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.floating] | None:
        """Get confidence scores for predictions if available."""
        try:
            if algorithm == "iforest":
                return self._isolation_forest_scores(data, contamination, **kwargs)
            elif algorithm == "lof":
                return self._local_outlier_factor_scores(data, contamination, **kwargs)
            else:
                # Check if registered adapter provides scores
                if algorithm in self._adapters:
                    adapter = self._adapters[algorithm]
                    if hasattr(adapter, 'decision_function') or hasattr(adapter, 'score_samples'):
                        # Try to get scores from adapter
                        return None  # For now, adapters don't implement scoring
                return None
        except Exception as e:
            logger.warning(f"Failed to get confidence scores for {algorithm}: {e}")
            return None
    
    def _isolation_forest_scores(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.floating]:
        """Get confidence scores from Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            model = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get('random_state', 42),
                **kwargs
            )
            model.fit(data)
            # Get anomaly scores (negative values are more anomalous)
            scores = model.decision_function(data)
            # Convert to confidence scores (0-1, higher = more anomalous)
            # Normalize using contamination threshold
            threshold = np.percentile(scores, contamination * 100)
            normalized_scores = np.clip((threshold - scores) / (np.max(scores) - threshold + 1e-10), 0, 1)
            return normalized_scores.astype(np.float64)
            
        except Exception as e:
            logger.warning(f"Failed to get Isolation Forest scores: {e}")
            return None
    
    def _local_outlier_factor_scores(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.floating]:
        """Get confidence scores from Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            model = LocalOutlierFactor(
                contamination=contamination,
                **kwargs
            )
            model.fit(data)
            # Get negative outlier factor scores
            scores = model.negative_outlier_factor_
            # Convert to confidence scores (0-1, higher = more anomalous)
            # LOF scores are negative, more negative = more anomalous
            min_score = np.min(scores)
            max_score = np.max(scores)
            if min_score < max_score:
                normalized_scores = (max_score - scores) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            return normalized_scores.astype(np.float64)
            
        except Exception as e:
            logger.warning(f"Failed to get LOF scores: {e}")
            return None