"""Tests for streaming service memory management functionality."""

import gc
import time
import numpy as np
import pytest
from unittest.mock import Mock, patch

from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.domain.services.detection_service_simple import DetectionServiceSimple


class TestStreamingMemoryManagement:
    """Test streaming service memory management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.detection_service = DetectionServiceSimple()
        self.streaming_service = StreamingService(
            detection_service=self.detection_service,
            window_size=100,
            update_frequency=50,
            memory_limit_mb=100,
            cleanup_frequency=25
        )
        
    def test_memory_tracking_initialization(self):
        """Test that memory tracking is properly initialized."""
        stats = self.streaming_service.get_memory_stats()
        
        assert stats["current_memory_mb"] > 0
        assert stats["initial_memory_mb"] > 0
        assert stats["peak_memory_mb"] >= stats["initial_memory_mb"]
        assert stats["memory_limit_mb"] == 100
        assert stats["memory_usage_ratio"] >= 0
        assert stats["memory_warnings"] == 0
        
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring during processing."""
        # Generate test data
        data = np.random.rand(50, 5).astype(np.float64)
        
        initial_stats = self.streaming_service.get_memory_stats()
        
        # Process samples
        with patch.object(self.streaming_service, '_check_memory_usage') as mock_check:
            for sample in data:
                self.streaming_service.process_sample(sample)
            
            # Memory check should be called periodically
            assert mock_check.call_count > 0
            
        final_stats = self.streaming_service.get_memory_stats()
        
        # Memory tracking should be updated
        assert final_stats["peak_memory_mb"] >= initial_stats["peak_memory_mb"]
        
    def test_memory_cleanup_trigger(self):
        """Test that memory cleanup is triggered appropriately."""
        # Set low cleanup frequency for testing
        self.streaming_service.cleanup_frequency = 5
        
        data = np.random.rand(10, 3).astype(np.float64)
        
        with patch.object(self.streaming_service, '_perform_memory_cleanup') as mock_cleanup:
            for sample in data:
                self.streaming_service.process_sample(sample)
            
            # Cleanup should be called at least once
            assert mock_cleanup.call_count >= 1
            
    def test_memory_cleanup_operations(self):
        """Test memory cleanup operations."""
        # Fill processing times buffer
        for i in range(600):
            self.streaming_service._processing_times.append(float(i))
            
        assert len(self.streaming_service._processing_times) == 600
        
        # Trigger cleanup
        self.streaming_service._perform_memory_cleanup()
        
        # Processing times should be reduced
        assert len(self.streaming_service._processing_times) <= 500
        
    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        # Fill buffer with data
        data = np.random.rand(200, 4).astype(np.float64)
        for sample in data:
            self.streaming_service._data_buffer.append(sample)
            
        initial_buffer_size = self.streaming_service.window_size
        
        # Mock high memory usage
        with patch.object(self.streaming_service, '_get_memory_usage', return_value=85.0):
            result = self.streaming_service.optimize_memory_usage()
            
            assert result["optimization_success"] is True
            assert result["initial_memory_mb"] == 85.0
            assert result["final_memory_mb"] <= result["initial_memory_mb"]
            
    def test_buffer_resize_memory_cleanup(self):
        """Test that buffer resizing triggers memory cleanup."""
        initial_size = self.streaming_service.window_size
        
        with patch('gc.collect') as mock_gc:
            self.streaming_service.set_window_size(50)
            
            # Garbage collection should be called
            mock_gc.assert_called_once()
            
        assert self.streaming_service.window_size == 50
        assert self.streaming_service.window_size < initial_size
        
    def test_stream_reset_memory_cleanup(self):
        """Test that stream reset cleans up all memory properly."""
        # Add some data and processing history
        data = np.random.rand(20, 3).astype(np.float64)
        for sample in data:
            self.streaming_service.process_sample(sample)
            
        # Verify data exists
        assert len(self.streaming_service._data_buffer) > 0
        assert len(self.streaming_service._processing_times) > 0
        assert self.streaming_service._sample_count > 0
        
        with patch('gc.collect') as mock_gc:
            self.streaming_service.reset_stream()
            
            # Garbage collection should be called
            mock_gc.assert_called_once()
            
        # All data should be cleared
        assert len(self.streaming_service._data_buffer) == 0
        assert len(self.streaming_service._processing_times) == 0
        assert self.streaming_service._sample_count == 0
        assert self.streaming_service._error_count == 0
        
    def test_comprehensive_streaming_stats(self):
        """Test comprehensive streaming statistics."""
        # Process some samples
        data = np.random.rand(10, 3).astype(np.float64)
        for sample in data:
            self.streaming_service.process_sample(sample)
            
        stats = self.streaming_service.get_streaming_stats()
        
        # Verify all memory stats are present
        required_keys = [
            "current_memory_mb", "initial_memory_mb", "peak_memory_mb",
            "memory_limit_mb", "memory_usage_ratio", "memory_warnings",
            "avg_processing_time_ms", "processing_samples_tracked",
            "cleanup_frequency", "samples_since_cleanup"
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
            
        # Verify reasonable values
        assert stats["current_memory_mb"] > 0
        assert stats["memory_usage_ratio"] >= 0
        assert stats["processing_samples_tracked"] >= 0
        
    def test_memory_warning_threshold(self):
        """Test memory warning threshold detection."""
        # Mock high memory usage (95% of limit)
        with patch.object(self.streaming_service, '_get_memory_usage', return_value=95.0):
            with patch.object(self.streaming_service, '_perform_memory_cleanup') as mock_cleanup:
                self.streaming_service._check_memory_usage()
                
                # Should trigger warning and cleanup
                assert self.streaming_service._memory_warnings > 0
                mock_cleanup.assert_called_with(force=True)
                
    def test_processing_time_tracking(self):
        """Test processing time tracking for performance monitoring."""
        data = np.random.rand(5, 3).astype(np.float64)
        
        for sample in data:
            self.streaming_service.process_sample(sample)
            
        # Processing times should be tracked
        assert len(self.streaming_service._processing_times) == len(data)
        
        # All times should be positive
        for time_ms in self.streaming_service._processing_times:
            assert time_ms >= 0
            
        stats = self.streaming_service.get_streaming_stats()
        assert stats["avg_processing_time_ms"] >= 0
        
    @patch('psutil.Process')
    def test_memory_usage_fallback(self, mock_process):
        """Test fallback when psutil is not available."""
        # Mock psutil failure
        mock_process.side_effect = Exception("psutil not available")
        
        memory_usage = self.streaming_service._get_memory_usage()
        
        # Should return 0.0 as fallback
        assert memory_usage == 0.0
        
    def test_batch_processing_memory_management(self):
        """Test memory management during batch processing."""
        batch_data = np.random.rand(25, 4).astype(np.float64)
        
        with patch.object(self.streaming_service, '_check_memory_usage') as mock_check:
            with patch.object(self.streaming_service, '_perform_memory_cleanup') as mock_cleanup:
                self.streaming_service.process_batch(batch_data)
                
                # Memory checks should occur
                mock_check.assert_called()
                # Cleanup might be triggered depending on cleanup frequency
                
        # Processing times should be tracked for batch
        assert len(self.streaming_service._processing_times) > 0