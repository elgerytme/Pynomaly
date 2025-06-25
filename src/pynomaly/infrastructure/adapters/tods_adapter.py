"""TODS adapter for time-series anomaly detection algorithms.

This module integrates algorithms from the TODS (Time-series Outlier Detection System) library.
TODS provides state-of-the-art algorithms specifically designed for time-series data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from dependency_injector.wiring import Provide, inject

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class TODSAdapter(Detector):
    """Adapter for TODS time-series anomaly detection algorithms."""
    
    # Lazy imports to avoid import errors if TODS not installed
    _algorithm_map: Optional[Dict[str, Type]] = None
    ALGORITHM_MAPPING: Dict[str, Type] = {}
    
    @classmethod
    def _get_algorithm_map(cls) -> Dict[str, Type]:
        """Lazily import and return TODS algorithm mapping."""
        if cls._algorithm_map is None:
            try:
                # Import TODS algorithms
                from tods.detection_algorithms.pyod_ae import AutoEncoderPrimitive
                from tods.detection_algorithms.pyod_lof import LOFPrimitive
                from tods.detection_algorithms.pyod_ocsvm import OCSVMPrimitive
                from tods.detection_algorithms.pyod_iforest import IsolationForestPrimitive
                from tods.detection_algorithms.matrix_profile import MatrixProfilePrimitive
                from tods.detection_algorithms.telemanom import TelemanomPrimitive
                from tods.detection_algorithms.deeplog import DeepLogPrimitive
                from tods.detection_algorithms.lstm import LSTMPrimitive
                
                cls._algorithm_map = {
                    # Statistical methods
                    "MatrixProfile": MatrixProfilePrimitive,
                    
                    # Deep learning methods
                    "LSTM": LSTMPrimitive,
                    "DeepLog": DeepLogPrimitive,
                    "Telemanom": TelemanomPrimitive,
                    "AutoEncoderTS": AutoEncoderPrimitive,
                    
                    # Traditional methods adapted for time-series
                    "LOFTS": LOFPrimitive,
                    "OCSVMTS": OCSVMPrimitive,
                    "IsolationForestTS": IsolationForestPrimitive,
                }
                # Update class attribute for backward compatibility
                cls.ALGORITHM_MAPPING = cls._algorithm_map
            except ImportError as e:
                logger.error(f"Failed to import TODS: {e}")
                cls._algorithm_map = {}
                cls.ALGORITHM_MAPPING = {}
                
        return cls._algorithm_map
    
    def __init__(self, algorithm: str, parameters: Optional[Dict[str, Any]] = None):
        """Initialize TODS adapter with detector configuration.
        
        Args:
            algorithm: Algorithm name
            parameters: Algorithm parameters
        """
        super().__init__(
            name=f"TODS_{algorithm}",
            algorithm_name=algorithm,
            parameters=parameters or {}
        )
        self._model = None
        self._init_algorithm()
    
    def _init_algorithm(self) -> None:
        """Initialize the TODS algorithm instance."""
        algorithm_map = self._get_algorithm_map()
        
        if self.algorithm not in algorithm_map:
            available = ", ".join(algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.algorithm}' not found in TODS. "
                f"Available algorithms: {available}"
            )
        
        try:
            algorithm_class = algorithm_map[self.algorithm]
            
            # TODS uses hyperparameter configuration
            hyperparams = algorithm_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
            
            # Configure hyperparameters
            params = self.parameters.copy()
            
            # Handle common parameter mappings
            if 'contamination' in params:
                params['contamination'] = float(params['contamination'])
            
            # Handle window size for time-series methods
            if self.algorithm in ['MatrixProfile', 'LSTM', 'DeepLog']:
                if 'window_size' not in params:
                    params['window_size'] = 50  # Default window size
            
            # Create hyperparameter instance
            hp = hyperparams.defaults()
            for key, value in params.items():
                if hasattr(hp, key):
                    setattr(hp, key, value)
            
            # Initialize the primitive
            self._model = algorithm_class(hyperparams=hp)
            
        except Exception as e:
            raise AdapterError(
                f"Failed to initialize TODS algorithm '{self.algorithm}': {e}"
            )
    
    def fit(self, dataset: Dataset) -> None:
        """Train the anomaly detector on the dataset.
        
        Args:
            dataset: Training dataset
        """
        if self._model is None:
            raise AdapterError("Model not initialized")
        
        try:
            # Prepare time-series data
            X = self._prepare_timeseries_data(dataset)
            
            # TODS primitives use produce method for training
            self._model.set_training_data(inputs=X)
            self._model.fit()
            
            self.is_fitted = True
            logger.info(f"Successfully trained TODS {self.algorithm}")
            
        except Exception as e:
            raise AdapterError(f"Failed to train TODS model: {e}")
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Detection results with anomaly scores and labels
        """
        if not self.is_fitted:
            raise AdapterError("Model must be fitted before prediction")
        
        try:
            # Prepare time-series data
            X = self._prepare_timeseries_data(dataset)
            
            # Get anomaly scores
            scores_df = self._model.produce(inputs=X)
            scores = scores_df.values.flatten()
            
            # For some algorithms, higher scores indicate anomalies
            # Normalize scores to [0, 1] range
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
            
            # Calculate threshold based on contamination rate
            contamination = self.parameters.get('contamination', 0.1)
            threshold = np.percentile(normalized_scores, (1 - contamination) * 100)
            
            # Create labels (1 for anomaly, 0 for normal)
            labels = (normalized_scores > threshold).astype(int)
            
            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(
                    value=float(score),
                    confidence=self._calculate_confidence(score, threshold)
                )
                for score in normalized_scores
            ]
            
            return DetectionResult(
                detector_id=self.id,
                dataset_id=dataset.id,
                scores=anomaly_scores,
                labels=labels.tolist(),
                metadata={
                    "algorithm": self.algorithm,
                    "threshold": float(threshold),
                    "n_anomalies": int(np.sum(labels)),
                    "contamination_rate": float(np.sum(labels) / len(labels)),
                    "is_timeseries": True
                }
            )
            
        except Exception as e:
            raise AdapterError(f"Failed to predict with TODS model: {e}")
    
    def _prepare_timeseries_data(self, dataset: Dataset) -> pd.DataFrame:
        """Prepare data in TODS format.
        
        Args:
            dataset: Input dataset
            
        Returns:
            DataFrame formatted for TODS
        """
        df = dataset.data.copy()
        
        # Ensure we have a time column
        if 'time' not in df.columns and 'timestamp' not in df.columns:
            # Create synthetic time index if not present
            df['time'] = pd.date_range(
                start='2024-01-01',
                periods=len(df),
                freq='H'  # Hourly frequency
            )
        elif 'timestamp' in df.columns and 'time' not in df.columns:
            df['time'] = df['timestamp']
        
        # TODS expects specific column structure
        # Ensure numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if dataset.target_column and dataset.target_column in numeric_cols:
            numeric_cols.remove(dataset.target_column)
        
        # If no numeric columns, raise error
        if not numeric_cols:
            raise AdapterError("No numeric features found in dataset")
        
        # Create feature matrix
        feature_df = df[['time'] + numeric_cols].copy()
        
        return feature_df
    
    def _calculate_confidence(self, score: float, threshold: float) -> float:
        """Calculate confidence score for anomaly.
        
        Args:
            score: Anomaly score
            threshold: Detection threshold
            
        Returns:
            Confidence value between 0 and 1
        """
        if score <= threshold:
            # Normal point - confidence based on distance from threshold
            return 1.0 - (score / threshold) * 0.5
        else:
            # Anomaly - confidence based on how far above threshold
            return 0.5 + min((score - threshold) / threshold * 0.5, 0.5)
    
    @classmethod
    def get_supported_algorithms(cls) -> List[str]:
        """Get list of supported TODS algorithms.
        
        Returns:
            List of algorithm names
        """
        return list(cls._get_algorithm_map().keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> Dict[str, Any]:
        """Get information about a specific algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Algorithm metadata and parameters
        """
        algorithm_map = cls._get_algorithm_map()
        
        if algorithm not in algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")
        
        # Algorithm-specific information
        info = {
            "MatrixProfile": {
                "name": "Matrix Profile",
                "type": "Statistical",
                "description": "Finds time series discord using matrix profile technique",
                "parameters": {
                    "window_size": {"type": "int", "default": 50, "description": "Subsequence window size"},
                    "threshold": {"type": "float", "default": 3.0, "description": "Anomaly threshold"}
                },
                "suitable_for": ["univariate", "multivariate", "streaming"],
                "pros": ["Fast computation", "Parameter-light", "Interpretable"],
                "cons": ["Requires sufficient data", "Fixed window size"]
            },
            "LSTM": {
                "name": "Long Short-Term Memory",
                "type": "Deep Learning",
                "description": "LSTM-based anomaly detection for complex temporal patterns",
                "parameters": {
                    "window_size": {"type": "int", "default": 50, "description": "Input sequence length"},
                    "hidden_size": {"type": "int", "default": 64, "description": "LSTM hidden units"},
                    "epochs": {"type": "int", "default": 100, "description": "Training epochs"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["multivariate", "complex_patterns", "long_sequences"],
                "pros": ["Captures long-term dependencies", "Handles complex patterns"],
                "cons": ["Computationally expensive", "Requires more data", "Black box"]
            },
            "DeepLog": {
                "name": "DeepLog",
                "type": "Deep Learning",
                "description": "Deep learning approach for log anomaly detection",
                "parameters": {
                    "window_size": {"type": "int", "default": 20, "description": "Log sequence window"},
                    "embedding_dim": {"type": "int", "default": 100, "description": "Embedding dimension"},
                    "hidden_size": {"type": "int", "default": 128, "description": "LSTM hidden size"}
                },
                "suitable_for": ["log_data", "sequential_data", "system_logs"],
                "pros": ["Designed for logs", "Learns patterns automatically"],
                "cons": ["Requires preprocessing", "Domain-specific"]
            },
            "Telemanom": {
                "name": "Telemanom",
                "type": "Deep Learning", 
                "description": "LSTM-based detector designed for spacecraft telemetry",
                "parameters": {
                    "window_size": {"type": "int", "default": 100, "description": "Sequence window"},
                    "smoothing_window": {"type": "int", "default": 30, "description": "Error smoothing"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Anomaly rate"}
                },
                "suitable_for": ["telemetry", "sensor_data", "spacecraft"],
                "pros": ["Proven on real telemetry", "Handles noise well"],
                "cons": ["Domain-specific design", "Computationally intensive"]
            }
        }
        
        return info.get(algorithm, {
            "name": algorithm,
            "type": "Time-series",
            "description": f"TODS implementation of {algorithm}",
            "parameters": {"contamination": {"type": "float", "default": 0.1}},
            "suitable_for": ["time_series"],
            "pros": ["Time-series optimized"],
            "cons": ["Requires sequential data"]
        })