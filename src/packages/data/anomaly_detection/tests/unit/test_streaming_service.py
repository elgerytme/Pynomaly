"""Unit tests for StreamingService."""

import pytest
import numpy as np
import threading
import time
from collections import deque
from unittest.mock import Mock, MagicMock, patch
from typing import Generator, List

from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestStreamingService:
    """Test suite for StreamingService."""
    
    @pytest.fixture
    def mock_detection_service(self):
        """Create mock detection service."""
        mock_service = Mock(spec=DetectionService)
        
        # Mock predict method
        def mock_predict(data, algorithm):
            predictions = np.array([1 if i % 3 == 0 else 0 for i in range(len(data))])
            return DetectionResult(
                predictions=predictions,
                algorithm=algorithm,
                metadata={"mock": True}
            )
        
        # Mock detect_anomalies method
        def mock_detect_anomalies(data, algorithm, **kwargs):
            predictions = np.array([1 if i % 4 == 0 else 0 for i in range(len(data))])
            return DetectionResult(
                predictions=predictions,
                algorithm=algorithm,
                metadata={"mock": True}
            )
        
        # Mock fit method
        def mock_fit(data, algorithm):
            return mock_service
        
        mock_service.predict.side_effect = mock_predict
        mock_service.detect_anomalies.side_effect = mock_detect_anomalies
        mock_service.fit.side_effect = mock_fit
        
        return mock_service
    
    @pytest.fixture
    def streaming_service(self, mock_detection_service):
        """Create streaming service with mock detection service."""
        return StreamingService(
            detection_service=mock_detection_service,
            window_size=100,
            update_frequency=20
        )
    
    @pytest.fixture
    def sample_data_point(self):
        """Create single sample data point."""
        return np.array([1.0, 2.0, 3.0])
    
    @pytest.fixture
    def sample_batch(self):
        """Create batch of sample data."""
        np.random.seed(42)
        return np.random.randn(10, 3).astype(np.float64)
    
    def test_initialization_with_detection_service(self, mock_detection_service):
        """Test streaming service initialization with provided detection service."""
        service = StreamingService(
            detection_service=mock_detection_service,
            window_size=500,
            update_frequency=50
        )
        
        assert service.detection_service is mock_detection_service
        assert service.window_size == 500
        assert service.update_frequency == 50
        assert len(service._data_buffer) == 0
        assert service._sample_count == 0
        assert service._last_update == 0
        assert service._model_fitted is False
        assert service._current_algorithm == "iforest"
    
    def test_initialization_without_detection_service(self):
        """Test streaming service initialization without detection service."""
        service = StreamingService()
        
        assert isinstance(service.detection_service, DetectionService)
        assert service.window_size == 1000  # default
        assert service.update_frequency == 100  # default
    
    def test_initialization_with_custom_parameters(self, mock_detection_service):
        """Test streaming service initialization with custom parameters."""
        service = StreamingService(
            detection_service=mock_detection_service,
            window_size=200,
            update_frequency=30
        )
        
        assert service.window_size == 200
        assert service.update_frequency == 30
        assert service._data_buffer.maxlen == 200
    
    def test_process_sample_first_samples(self, streaming_service, sample_data_point):
        """Test processing first few samples before model is fitted."""
        result = streaming_service.process_sample(sample_data_point)
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "threshold"
        assert len(result.predictions) == 1
        assert streaming_service._sample_count == 1
        assert len(streaming_service._data_buffer) == 1
        assert streaming_service._model_fitted is False
    
    def test_process_sample_with_model_fitted(self, streaming_service, sample_data_point):
        """Test processing sample when model is already fitted."""
        # Manually set model as fitted
        streaming_service._model_fitted = True
        
        result = streaming_service.process_sample(sample_data_point, algorithm="lof")
        
        assert isinstance(result, DetectionResult)
        assert streaming_service._sample_count == 1
        
        # Verify predict was called on detection service
        streaming_service.detection_service.predict.assert_called_once()
        call_args = streaming_service.detection_service.predict.call_args
        np.testing.assert_array_equal(call_args[0][0], sample_data_point.reshape(1, -1))
        assert call_args[0][1] == "lof"
    
    def test_process_sample_triggers_retrain(self, streaming_service, sample_data_point):
        """Test that processing samples triggers retraining when needed."""
        # Fill buffer to minimum size and trigger retrain
        for i in range(25):  # More than update_frequency (20)
            streaming_service.process_sample(sample_data_point + i * 0.1)
        
        # Should have triggered retrain
        assert streaming_service._model_fitted is True
        assert streaming_service._last_update > 0
        
        # Verify fit was called
        streaming_service.detection_service.fit.assert_called()
    
    def test_process_batch_empty_buffer(self, streaming_service, sample_batch):
        """Test processing batch with empty buffer."""
        result = streaming_service.process_batch(sample_batch, algorithm="iforest")
        
        assert isinstance(result, DetectionResult)
        assert len(result.predictions) == len(sample_batch)
        assert streaming_service._sample_count == len(sample_batch)
        assert len(streaming_service._data_buffer) == len(sample_batch)
        
        # Should call detect_anomalies since model not fitted
        streaming_service.detection_service.detect_anomalies.assert_called_once()
    
    def test_process_batch_with_fitted_model(self, streaming_service, sample_batch):
        """Test processing batch with fitted model."""
        # Set model as fitted
        streaming_service._model_fitted = True
        
        result = streaming_service.process_batch(sample_batch, algorithm="lof")
        
        assert isinstance(result, DetectionResult)
        assert len(result.predictions) == len(sample_batch)
        
        # Should call predict since model is fitted
        streaming_service.detection_service.predict.assert_called_once()
        call_args = streaming_service.detection_service.predict.call_args
        np.testing.assert_array_equal(call_args[0][0], sample_batch)
        assert call_args[0][1] == "lof"
    
    def test_process_batch_triggers_retrain(self, streaming_service):
        """Test that processing batch triggers retraining."""
        # Create batch that will trigger retrain
        large_batch = np.random.randn(25, 3).astype(np.float64)
        
        result = streaming_service.process_batch(large_batch)
        
        # Should have triggered retrain
        assert streaming_service._model_fitted is True
        streaming_service.detection_service.fit.assert_called()
    
    def test_process_stream(self, streaming_service):
        """Test processing continuous data stream."""
        def data_generator():
            for i in range(5):
                yield np.array([i, i+1, i+2], dtype=np.float64)
        
        results = list(streaming_service.process_stream(data_generator()))
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, DetectionResult)
        assert streaming_service._sample_count == 5
    
    def test_process_stream_with_callback(self, streaming_service):
        """Test processing stream with callback function."""
        callback_results = []
        
        def callback(result):
            callback_results.append(result)
        
        def data_generator():
            for i in range(3):
                yield np.array([i, i+1], dtype=np.float64)
        
        list(streaming_service.process_stream(data_generator(), callback=callback))
        
        assert len(callback_results) == 3
        for result in callback_results:
            assert isinstance(result, DetectionResult)
    
    def test_should_retrain_insufficient_samples(self, streaming_service):
        """Test retrain check with insufficient samples."""
        streaming_service._sample_count = 10
        streaming_service._last_update = 0
        
        # Not enough samples since last update
        assert streaming_service._should_retrain() is False
    
    def test_should_retrain_insufficient_buffer(self, streaming_service):
        """Test retrain check with insufficient buffer size."""
        streaming_service._sample_count = 25
        streaming_service._last_update = 0
        # Buffer has fewer than 100 samples (minimum)
        streaming_service._data_buffer = deque([np.array([1, 2])], maxlen=100)
        
        assert streaming_service._should_retrain() is False
    
    def test_should_retrain_ready(self, streaming_service):
        """Test retrain check when ready to retrain."""
        streaming_service._sample_count = 25
        streaming_service._last_update = 0
        # Fill buffer with minimum required samples
        for i in range(100):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        assert streaming_service._should_retrain() is True
    
    def test_retrain_model_success(self, streaming_service):
        """Test successful model retraining."""
        # Fill buffer with data
        for i in range(50):
            streaming_service._data_buffer.append(np.array([i, i+1], dtype=np.float64))
        
        streaming_service._sample_count = 30
        
        streaming_service._retrain_model("iforest")
        
        assert streaming_service._model_fitted is True
        assert streaming_service._current_algorithm == "iforest"
        assert streaming_service._last_update == 30
        streaming_service.detection_service.fit.assert_called_once()
    
    def test_retrain_model_failure(self, streaming_service):
        """Test model retraining failure handling."""
        # Make fit method raise exception
        streaming_service.detection_service.fit.side_effect = Exception("Fit failed")
        
        # Fill buffer with data
        for i in range(10):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        # Should not raise exception, just log error
        streaming_service._retrain_model("iforest")
        
        # Model should still not be fitted
        assert streaming_service._model_fitted is False
    
    def test_simple_threshold_detection_insufficient_data(self, streaming_service):
        """Test simple threshold detection with insufficient data."""
        sample = np.array([1.0, 2.0, 3.0])
        
        result = streaming_service._simple_threshold_detection(sample)
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "threshold"
        assert len(result.predictions) == 1
        assert result.predictions[0] == 0  # Should assume normal
        assert result.metadata["buffer_size"] == 0
    
    def test_simple_threshold_detection_with_buffer(self, streaming_service):
        """Test simple threshold detection with buffer data."""
        # Fill buffer with normal data
        for i in range(15):
            streaming_service._data_buffer.append(np.array([i, i+1, i+2], dtype=np.float64))
        
        # Test normal sample
        normal_sample = np.array([7.0, 8.0, 9.0])
        result = streaming_service._simple_threshold_detection(normal_sample)
        assert result.predictions[0] == 0  # Should be normal
        
        # Test anomalous sample (far from mean)
        anomaly_sample = np.array([100.0, 100.0, 100.0])
        result = streaming_service._simple_threshold_detection(anomaly_sample)
        assert result.predictions[0] == 1  # Should be anomaly
    
    def test_simple_threshold_detection_zero_std(self, streaming_service):
        """Test simple threshold detection with zero standard deviation."""
        # Fill buffer with identical data
        constant_sample = np.array([5.0, 5.0, 5.0])
        for i in range(15):
            streaming_service._data_buffer.append(constant_sample.copy())
        
        # Test with same sample
        result = streaming_service._simple_threshold_detection(constant_sample)
        assert result.predictions[0] == 0  # Should handle zero std gracefully
    
    def test_detect_concept_drift_insufficient_data(self, streaming_service):
        """Test concept drift detection with insufficient data."""
        # Fill buffer with less than 2 * window_size data
        for i in range(100):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        result = streaming_service.detect_concept_drift(window_size=200)
        
        assert result["drift_detected"] is False
        assert result["reason"] == "insufficient_data"
        assert result["buffer_size"] == 100
    
    def test_detect_concept_drift_no_drift(self, streaming_service):
        """Test concept drift detection with no drift."""
        # Fill buffer with consistent data
        for i in range(500):
            streaming_service._data_buffer.append(np.array([i % 10, (i % 10) + 1], dtype=np.float64))
        
        result = streaming_service.detect_concept_drift(window_size=200)
        
        assert result["drift_detected"] is False
        assert result["max_relative_change"] < 0.2  # Below threshold
        assert result["drift_threshold"] == 0.2
        assert result["recent_samples"] == 200
    
    def test_detect_concept_drift_with_drift(self, streaming_service):
        """Test concept drift detection with actual drift."""
        # Fill buffer - first half with one pattern, second half with different pattern
        for i in range(300):
            if i < 200:
                streaming_service._data_buffer.append(np.array([1.0, 2.0], dtype=np.float64))
            else:
                streaming_service._data_buffer.append(np.array([10.0, 20.0], dtype=np.float64))
        
        result = streaming_service.detect_concept_drift(window_size=100)
        
        assert result["drift_detected"] is True
        assert result["max_relative_change"] > 0.2  # Above threshold
    
    def test_get_streaming_stats(self, streaming_service):
        """Test getting streaming statistics."""
        # Add some data and modify state
        streaming_service._sample_count = 50
        streaming_service._last_update = 30
        streaming_service._model_fitted = True
        streaming_service._current_algorithm = "lof"
        
        for i in range(10):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        stats = streaming_service.get_streaming_stats()
        
        assert stats["total_samples"] == 50
        assert stats["buffer_size"] == 10
        assert stats["buffer_capacity"] == 100
        assert stats["last_update_at"] == 30
        assert stats["model_fitted"] is True
        assert stats["current_algorithm"] == "lof"
        assert stats["samples_since_update"] == 20
        assert stats["update_frequency"] == 20
    
    def test_reset_stream(self, streaming_service):
        """Test resetting stream state."""
        # Add some data and modify state
        streaming_service._sample_count = 50
        streaming_service._last_update = 30
        streaming_service._model_fitted = True
        
        for i in range(10):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        streaming_service.reset_stream()
        
        assert streaming_service._sample_count == 0
        assert streaming_service._last_update == 0
        assert streaming_service._model_fitted is False
        assert len(streaming_service._data_buffer) == 0
    
    def test_set_update_frequency_valid(self, streaming_service):
        """Test setting valid update frequency."""
        streaming_service.set_update_frequency(50)
        assert streaming_service.update_frequency == 50
    
    def test_set_update_frequency_invalid(self, streaming_service):
        """Test setting invalid update frequency."""
        with pytest.raises(ValueError) as exc_info:
            streaming_service.set_update_frequency(0)
        
        assert "Update frequency must be positive" in str(exc_info.value)
        
        with pytest.raises(ValueError):
            streaming_service.set_update_frequency(-10)
    
    def test_set_window_size_valid(self, streaming_service):
        """Test setting valid window size."""
        # Add some data first
        for i in range(10):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        streaming_service.set_window_size(50)
        
        assert streaming_service.window_size == 50
        assert streaming_service._data_buffer.maxlen == 50
        assert len(streaming_service._data_buffer) == 10  # Data preserved
    
    def test_set_window_size_smaller_than_buffer(self, streaming_service):
        """Test setting window size smaller than current buffer."""
        # Add more data than new window size
        for i in range(10):
            streaming_service._data_buffer.append(np.array([i, i+1]))
        
        streaming_service.set_window_size(5)
        
        assert streaming_service.window_size == 5
        assert len(streaming_service._data_buffer) == 5  # Truncated to most recent
    
    def test_set_window_size_invalid(self, streaming_service):
        """Test setting invalid window size."""
        with pytest.raises(ValueError) as exc_info:
            streaming_service.set_window_size(0)
        
        assert "Window size must be positive" in str(exc_info.value)
        
        with pytest.raises(ValueError):
            streaming_service.set_window_size(-10)
    
    def test_thread_safety_concurrent_processing(self, streaming_service):
        """Test thread safety with concurrent sample processing."""
        results = []
        
        def process_samples():
            for i in range(10):
                sample = np.array([i, i+1, i+2], dtype=np.float64)
                result = streaming_service.process_sample(sample)
                results.append(result)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_samples)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have processed all samples without errors
        assert len(results) == 30
        assert streaming_service._sample_count == 30
    
    def test_thread_safety_concurrent_stats_access(self, streaming_service):
        """Test thread safety with concurrent stats access."""
        stats_results = []
        
        def get_stats():
            for _ in range(5):
                stats = streaming_service.get_streaming_stats()
                stats_results.append(stats)
                time.sleep(0.001)
        
        def add_samples():
            for i in range(10):
                sample = np.array([i, i+1], dtype=np.float64)
                streaming_service.process_sample(sample)
                time.sleep(0.001)
        
        # Run concurrent operations
        thread1 = threading.Thread(target=get_stats)
        thread2 = threading.Thread(target=add_samples)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Should have collected stats without errors
        assert len(stats_results) == 5
        for stats in stats_results:
            assert isinstance(stats, dict)
            assert "total_samples" in stats
    
    def test_buffer_maxlen_enforcement(self, streaming_service):
        """Test that buffer enforces maximum length."""
        # Process more samples than window size
        for i in range(150):  # More than window_size (100)
            sample = np.array([i, i+1], dtype=np.float64)
            streaming_service.process_sample(sample)
        
        # Buffer should be limited to window size
        assert len(streaming_service._data_buffer) == streaming_service.window_size
        assert streaming_service._sample_count == 150  # But count should be accurate
    
    def test_algorithm_parameter_propagation(self, streaming_service, sample_data_point):
        """Test that algorithm parameter is properly propagated."""
        streaming_service._model_fitted = True
        
        streaming_service.process_sample(sample_data_point, algorithm="custom_algo")
        
        # Verify algorithm was passed to predict
        call_args = streaming_service.detection_service.predict.call_args
        assert call_args[0][1] == "custom_algo"
    
    @pytest.mark.parametrize("window_size,update_freq", [
        (50, 10),
        (200, 25),
        (1000, 100),
    ])
    def test_different_configurations(self, mock_detection_service, window_size, update_freq):
        """Test streaming service with different configurations."""
        service = StreamingService(
            detection_service=mock_detection_service,
            window_size=window_size,
            update_frequency=update_freq
        )
        
        assert service.window_size == window_size
        assert service.update_frequency == update_freq
        assert service._data_buffer.maxlen == window_size
    
    def test_retrain_with_algorithm_change(self, streaming_service):
        """Test retraining when algorithm changes."""
        # Fill buffer to trigger retrain
        for i in range(25):
            streaming_service._data_buffer.append(np.array([i, i+1], dtype=np.float64))
        
        streaming_service._sample_count = 25
        
        # First retrain with iforest
        streaming_service._retrain_model("iforest")
        assert streaming_service._current_algorithm == "iforest"
        
        # Reset for second retrain
        streaming_service._last_update = 0
        streaming_service._sample_count = 50
        
        # Second retrain with different algorithm
        streaming_service._retrain_model("lof")
        assert streaming_service._current_algorithm == "lof"
        
        # Verify fit was called twice with different algorithms
        assert streaming_service.detection_service.fit.call_count == 2
    
    def test_concept_drift_edge_cases(self, streaming_service):
        """Test concept drift detection edge cases."""
        # Test with zero older mean (division by zero protection)
        for i in range(500):
            if i < 250:
                streaming_service._data_buffer.append(np.array([0.0, 0.0], dtype=np.float64))
            else:
                streaming_service._data_buffer.append(np.array([1.0, 1.0], dtype=np.float64))
        
        # Should handle division by zero gracefully
        result = streaming_service.detect_concept_drift(window_size=200)
        assert isinstance(result["drift_detected"], bool)
        assert "max_relative_change" in result