"""Streaming anomaly detection for real-time applications."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, Iterator
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from collections import deque
import time
import threading
from queue import Queue, Empty

from simplified_services.core_detection_service import CoreDetectionService, DetectionResult


@dataclass
class StreamingConfig:
    """Configuration for streaming detection."""
    window_size: int = 1000  # Size of sliding window
    update_frequency: int = 100  # How often to retrain (samples)
    algorithm: str = "iforest"
    contamination: float = 0.1
    buffer_size: int = 5000  # Maximum buffer size
    min_samples_for_training: int = 100
    max_processing_delay: float = 1.0  # Max delay in seconds
    enable_drift_detection: bool = True


@dataclass 
class StreamingStats:
    """Statistics for streaming detection."""
    total_samples: int = 0
    total_anomalies: int = 0
    current_anomaly_rate: float = 0.0
    average_processing_time: float = 0.0
    retraining_count: int = 0
    buffer_utilization: float = 0.0
    drift_detected: bool = False
    last_update: float = field(default_factory=time.time)


class StreamingDetector:
    """Real-time streaming anomaly detector.
    
    This class provides streaming anomaly detection capabilities that were
    missing from the original package:
    - Real-time processing with sliding windows
    - Automatic model retraining based on new data
    - Drift detection and adaptation
    - Asynchronous processing for high-throughput scenarios
    - Memory-efficient buffering
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize streaming detector.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.detection_service = CoreDetectionService()
        
        # Sliding window buffer
        self._buffer: deque = deque(maxlen=self.config.buffer_size)
        self._window: deque = deque(maxlen=self.config.window_size)
        
        # Model state
        self._trained_model = None
        self._last_training_size = 0
        self._training_data: Optional[npt.NDArray[np.floating]] = None
        
        # Statistics
        self.stats = StreamingStats()
        self._processing_times: deque = deque(maxlen=100)
        
        # Asynchronous processing
        self._async_queue: Queue = Queue()
        self._async_thread: Optional[threading.Thread] = None
        self._stop_async = False
        self._callbacks: List[Callable[[DetectionResult], None]] = []

    def add_sample(self, sample: npt.NDArray[np.floating]) -> Optional[DetectionResult]:
        """Add a single sample for real-time detection.
        
        Args:
            sample: Single sample array of shape (n_features,)
            
        Returns:
            DetectionResult if detection was performed, None otherwise
        """
        start_time = time.time()
        
        # Add to buffers
        self._buffer.append(sample)
        self._window.append(sample)
        
        self.stats.total_samples += 1
        
        # Check if we need to retrain
        if self._should_retrain():
            self._retrain_model()
        
        # Perform detection if model is trained
        result = None
        if self._trained_model is not None:
            result = self._detect_single_sample(sample)
            
            if result and result.predictions[0] == 1:
                self.stats.total_anomalies += 1
                self.stats.current_anomaly_rate = self.stats.total_anomalies / self.stats.total_samples
        
        # Update timing statistics
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        self.stats.average_processing_time = np.mean(self._processing_times)
        self.stats.buffer_utilization = len(self._buffer) / self.config.buffer_size
        self.stats.last_update = time.time()
        
        return result

    def add_batch(self, batch: npt.NDArray[np.floating]) -> List[DetectionResult]:
        """Add a batch of samples for detection.
        
        Args:
            batch: Batch of samples of shape (n_samples, n_features)
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        
        for sample in batch:
            result = self.add_sample(sample)
            if result is not None:
                results.append(result)
        
        return results

    def process_stream(
        self,
        data_stream: Iterator[npt.NDArray[np.floating]],
        max_samples: Optional[int] = None
    ) -> Iterator[DetectionResult]:
        """Process a stream of data samples.
        
        Args:
            data_stream: Iterator yielding individual samples
            max_samples: Maximum number of samples to process
            
        Yields:
            DetectionResult objects for anomalous samples
        """
        sample_count = 0
        
        for sample in data_stream:
            if max_samples and sample_count >= max_samples:
                break
                
            result = self.add_sample(sample)
            
            if result is not None and result.predictions[0] == 1:
                yield result
                
            sample_count += 1

    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        current_size = len(self._buffer)
        
        # Initial training
        if self._trained_model is None and current_size >= self.config.min_samples_for_training:
            return True
        
        # Periodic retraining
        if (current_size - self._last_training_size) >= self.config.update_frequency:
            return True
        
        # Drift detection
        if self.config.enable_drift_detection and self._detect_drift():
            return True
        
        return False

    def _retrain_model(self) -> None:
        """Retrain the detection model with current buffer data."""
        if len(self._buffer) < self.config.min_samples_for_training:
            return
        
        print(f"🔄 Streaming: Retraining model with {len(self._buffer)} samples...")
        
        # Convert buffer to array
        training_data = np.array(list(self._buffer))
        
        try:
            # Train new model
            result = self.detection_service.detect_anomalies(
                training_data,
                algorithm=self.config.algorithm,
                contamination=self.config.contamination
            )
            
            # Store trained model (simplified - in practice would save actual model)
            self._trained_model = {
                "algorithm": self.config.algorithm,
                "contamination": self.config.contamination,
                "training_size": len(training_data),
                "training_time": time.time()
            }
            
            self._training_data = training_data
            self._last_training_size = len(self._buffer)
            self.stats.retraining_count += 1
            
            print(f"✅ Streaming: Model retrained successfully")
            
        except Exception as e:
            print(f"✗ Streaming: Retraining failed: {e}")

    def _detect_single_sample(self, sample: npt.NDArray[np.floating]) -> DetectionResult:
        """Detect anomaly for a single sample using trained model."""
        # In a real implementation, this would use the actual trained model
        # For now, we'll use a simplified detection approach
        
        if self._training_data is None:
            # Fallback: compare with recent window statistics
            return self._simple_statistical_detection(sample)
        
        # Use detection service with current sample
        # Note: This is simplified - real streaming would use incremental detection
        sample_2d = sample.reshape(1, -1)
        
        try:
            result = self.detection_service.detect_anomalies(
                sample_2d,
                algorithm=self.config.algorithm,
                contamination=self.config.contamination
            )
            return result
        except Exception:
            # Fallback to statistical detection
            return self._simple_statistical_detection(sample)

    def _simple_statistical_detection(self, sample: npt.NDArray[np.floating]) -> DetectionResult:
        """Simple statistical anomaly detection for single sample."""
        if len(self._window) < 10:
            # Not enough data for detection
            return DetectionResult(
                predictions=np.array([0]),
                algorithm="statistical_fallback",
                contamination=self.config.contamination,
                n_samples=1,
                n_anomalies=0
            )
        
        # Calculate statistics from window
        window_data = np.array(list(self._window))
        mean = np.mean(window_data, axis=0)
        std = np.std(window_data, axis=0)
        
        # Z-score based detection
        z_scores = np.abs((sample - mean) / (std + 1e-8))
        max_z_score = np.max(z_scores)
        
        # Simple threshold (could be more sophisticated)
        is_anomaly = max_z_score > 3.0
        
        return DetectionResult(
            predictions=np.array([1 if is_anomaly else 0]),
            scores=np.array([max_z_score]),
            algorithm="statistical_fallback",
            contamination=self.config.contamination,
            n_samples=1,
            n_anomalies=1 if is_anomaly else 0,
            metadata={"z_score": max_z_score}
        )

    def _detect_drift(self) -> bool:
        """Detect concept drift in the data stream."""
        if not self.config.enable_drift_detection or len(self._window) < 100:
            return False
        
        # Simple drift detection based on statistical properties
        window_data = np.array(list(self._window))
        
        # Compare recent window with earlier window
        recent_size = min(50, len(window_data) // 2)
        recent_data = window_data[-recent_size:]
        earlier_data = window_data[:-recent_size]
        
        if len(earlier_data) < 10:
            return False
        
        # Compare means (simplified drift detection)
        recent_mean = np.mean(recent_data, axis=0)
        earlier_mean = np.mean(earlier_data, axis=0)
        
        # Calculate relative change
        diff = np.abs(recent_mean - earlier_mean)
        relative_change = np.mean(diff / (np.abs(earlier_mean) + 1e-8))
        
        # Simple threshold for drift detection
        drift_detected = relative_change > 0.5
        
        if drift_detected:
            print("🌊 Streaming: Concept drift detected")
            self.stats.drift_detected = True
        
        return drift_detected

    def start_async_processing(self) -> None:
        """Start asynchronous processing thread."""
        if self._async_thread is not None:
            return
        
        self._stop_async = False
        self._async_thread = threading.Thread(target=self._async_worker, daemon=True)
        self._async_thread.start()
        print("🚀 Streaming: Async processing started")

    def stop_async_processing(self) -> None:
        """Stop asynchronous processing thread."""
        if self._async_thread is None:
            return
        
        self._stop_async = True
        self._async_thread.join(timeout=5.0)
        self._async_thread = None
        print("⏹️  Streaming: Async processing stopped")

    def add_sample_async(self, sample: npt.NDArray[np.floating]) -> None:
        """Add sample for asynchronous processing."""
        if self._async_thread is None:
            raise RuntimeError("Async processing not started")
        
        self._async_queue.put(sample)

    def _async_worker(self) -> None:
        """Worker thread for asynchronous processing."""
        while not self._stop_async:
            try:
                sample = self._async_queue.get(timeout=0.1)
                result = self.add_sample(sample)
                
                # Notify callbacks
                if result is not None:
                    for callback in self._callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"Callback error: {e}")
                
                self._async_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Async worker error: {e}")

    def add_callback(self, callback: Callable[[DetectionResult], None]) -> None:
        """Add callback for anomaly detection events."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[DetectionResult], None]) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_current_stats(self) -> StreamingStats:
        """Get current streaming statistics."""
        return self.stats

    def reset(self) -> None:
        """Reset detector state."""
        self._buffer.clear()
        self._window.clear()
        self._trained_model = None
        self._last_training_size = 0
        self._training_data = None
        self.stats = StreamingStats()
        self._processing_times.clear()
        
        # Clear async queue
        while not self._async_queue.empty():
            try:
                self._async_queue.get_nowait()
            except Empty:
                break

    def save_state(self) -> Dict[str, Any]:
        """Save detector state for persistence."""
        return {
            "config": {
                "window_size": self.config.window_size,
                "update_frequency": self.config.update_frequency,
                "algorithm": self.config.algorithm,
                "contamination": self.config.contamination,
                "buffer_size": self.config.buffer_size
            },
            "stats": {
                "total_samples": self.stats.total_samples,
                "total_anomalies": self.stats.total_anomalies,
                "retraining_count": self.stats.retraining_count
            },
            "trained_model": self._trained_model,
            "buffer_size": len(self._buffer),
            "last_training_size": self._last_training_size
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load detector state from saved data."""
        if "stats" in state:
            stats_data = state["stats"]
            self.stats.total_samples = stats_data.get("total_samples", 0)
            self.stats.total_anomalies = stats_data.get("total_anomalies", 0)
            self.stats.retraining_count = stats_data.get("retraining_count", 0)
        
        if "trained_model" in state:
            self._trained_model = state["trained_model"]
        
        if "last_training_size" in state:
            self._last_training_size = state["last_training_size"]