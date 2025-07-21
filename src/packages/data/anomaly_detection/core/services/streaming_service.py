"""Consolidated streaming service for real-time anomaly detection."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Generator
import numpy as np
import numpy.typing as npt
import threading
import time

from .detection_service import DetectionService, DetectionResult

logger = logging.getLogger(__name__)


class StreamingService:
    """Service for real-time anomaly detection on streaming data.
    
    Handles incremental learning, concept drift, and real-time processing.
    """
    
    def __init__(
        self,
        detection_service: DetectionService | None = None,
        window_size: int = 1000,
        update_frequency: int = 100
    ):
        """Initialize streaming service.
        
        Args:
            detection_service: Base detection service
            window_size: Size of sliding window for training
            update_frequency: How often to retrain (number of samples)
        """
        self.detection_service = detection_service or DetectionService()
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        # Streaming state
        self._data_buffer: deque = deque(maxlen=window_size)
        self._sample_count = 0
        self._last_update = 0
        self._model_fitted = False
        self._current_algorithm = "iforest"
        
        # Thread safety
        self._lock = threading.RLock()
        
    def process_sample(
        self,
        sample: npt.NDArray[np.floating],
        algorithm: str = "iforest"
    ) -> DetectionResult:
        """Process a single sample in real-time.
        
        Args:
            sample: Single data point of shape (n_features,)
            algorithm: Algorithm to use
            
        Returns:
            DetectionResult for this sample
        """
        with self._lock:
            # Add to buffer
            self._data_buffer.append(sample)
            self._sample_count += 1
            
            # Check if we need to retrain
            if self._should_retrain():
                self._retrain_model(algorithm)
            
            # Make prediction
            if self._model_fitted:
                # Reshape for prediction
                sample_2d = sample.reshape(1, -1)
                result = self.detection_service.predict(sample_2d, algorithm)
            else:
                # Not enough data yet - use simple threshold
                result = self._simple_threshold_detection(sample)
                
            return result
    
    def process_batch(
        self,
        batch: npt.NDArray[np.floating],
        algorithm: str = "iforest"
    ) -> DetectionResult:
        """Process a batch of samples.
        
        Args:
            batch: Batch of data of shape (n_samples, n_features)
            algorithm: Algorithm to use
            
        Returns:
            DetectionResult for the batch
        """
        with self._lock:
            # Add batch to buffer
            for sample in batch:
                self._data_buffer.append(sample)
                self._sample_count += 1
            
            # Check if we need to retrain
            if self._should_retrain():
                self._retrain_model(algorithm)
            
            # Make predictions
            if self._model_fitted:
                result = self.detection_service.predict(batch, algorithm)
            else:
                # Use detection on available data
                result = self.detection_service.detect_anomalies(batch, algorithm)
                
            return result
    
    def process_stream(
        self,
        data_stream: Generator[npt.NDArray[np.floating], None, None],
        algorithm: str = "iforest",
        callback: Callable[[DetectionResult], None] | None = None
    ) -> Generator[DetectionResult, None, None]:
        """Process a continuous data stream.
        
        Args:
            data_stream: Generator yielding data samples
            algorithm: Algorithm to use
            callback: Optional callback for each result
            
        Yields:
            DetectionResult for each processed sample
        """
        for sample in data_stream:
            result = self.process_sample(sample, algorithm)
            
            if callback:
                callback(result)
                
            yield result
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        return (
            self._sample_count - self._last_update >= self.update_frequency and
            len(self._data_buffer) >= min(100, self.window_size)
        )
    
    def _retrain_model(self, algorithm: str) -> None:
        """Retrain the model with current buffer data."""
        try:
            buffer_data = np.array(list(self._data_buffer))
            
            # Retrain the model
            self.detection_service.fit(buffer_data, algorithm)
            self._model_fitted = True
            self._current_algorithm = algorithm
            self._last_update = self._sample_count
            
            logger.info(f"Model retrained with {len(buffer_data)} samples")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _simple_threshold_detection(
        self, 
        sample: npt.NDArray[np.floating]
    ) -> DetectionResult:
        """Simple threshold-based detection for cold start."""
        if len(self._data_buffer) < 10:
            # Not enough data - assume normal
            predictions = np.array([0])
        else:
            # Use z-score based detection
            buffer_data = np.array(list(self._data_buffer))
            mean = np.mean(buffer_data, axis=0)
            std = np.std(buffer_data, axis=0)
            
            # Avoid division by zero
            std = np.where(std == 0, 1e-8, std)
            
            z_scores = np.abs((sample - mean) / std)
            max_z_score = np.max(z_scores)
            
            # Threshold at 3 standard deviations
            predictions = np.array([1 if max_z_score > 3.0 else 0])
        
        return DetectionResult(
            predictions=predictions,
            scores=np.array([0.5]),  # Neutral score
            algorithm="threshold",
            metadata={"buffer_size": len(self._data_buffer)}
        )
    
    def detect_concept_drift(self, window_size: int = 200) -> dict[str, Any]:
        """Detect concept drift in the data stream.
        
        Args:
            window_size: Window size for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        if len(self._data_buffer) < 2 * window_size:
            return {
                "drift_detected": False,
                "reason": "insufficient_data",
                "buffer_size": len(self._data_buffer)
            }
        
        # Compare recent window with older window
        buffer_data = np.array(list(self._data_buffer))
        recent_window = buffer_data[-window_size:]
        older_window = buffer_data[-2*window_size:-window_size]
        
        # Simple statistical test - compare means
        recent_mean = np.mean(recent_window, axis=0)
        older_mean = np.mean(older_window, axis=0)
        
        # Calculate relative change
        mean_diff = np.abs(recent_mean - older_mean)
        relative_change = mean_diff / (np.abs(older_mean) + 1e-8)
        max_relative_change = np.max(relative_change)
        
        # Drift threshold (20% change)
        drift_threshold = 0.2
        drift_detected = max_relative_change > drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "max_relative_change": float(max_relative_change),
            "drift_threshold": drift_threshold,
            "recent_samples": window_size,
            "buffer_size": len(self._data_buffer)
        }
    
    def get_streaming_stats(self) -> dict[str, Any]:
        """Get statistics about the streaming process."""
        with self._lock:
            return {
                "total_samples": self._sample_count,
                "buffer_size": len(self._data_buffer),
                "buffer_capacity": self.window_size,
                "last_update_at": self._last_update,
                "model_fitted": self._model_fitted,
                "current_algorithm": self._current_algorithm,
                "samples_since_update": self._sample_count - self._last_update,
                "update_frequency": self.update_frequency
            }
    
    def reset_stream(self) -> None:
        """Reset the streaming state."""
        with self._lock:
            self._data_buffer.clear()
            self._sample_count = 0
            self._last_update = 0
            self._model_fitted = False
            logger.info("Stream state reset")
    
    def set_update_frequency(self, frequency: int) -> None:
        """Update the retraining frequency."""
        if frequency <= 0:
            raise ValueError("Update frequency must be positive")
        self.update_frequency = frequency
        logger.info(f"Update frequency set to {frequency}")
    
    def set_window_size(self, size: int) -> None:
        """Update the buffer window size."""
        if size <= 0:
            raise ValueError("Window size must be positive")
        with self._lock:
            # Create new buffer with new size
            old_data = list(self._data_buffer)
            self.window_size = size
            self._data_buffer = deque(old_data[-size:], maxlen=size)
            logger.info(f"Window size updated to {size}")