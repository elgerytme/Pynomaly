"""Tests for streaming profiling service."""

import pytest
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ..application.services.streaming_profiling_service import (
    StreamingProfilingService,
    StreamingConfig,
    StreamingMode,
    UpdateStrategy,
    IncrementalStatisticsAccumulator,
    StreamingPatternDetector,
    StreamingQualityMonitor,
    StreamingMetrics
)
from ..application.services.pattern_discovery_service import PatternDiscoveryService
from ..application.services.quality_assessment_service import QualityAssessmentService
from ..domain.entities.data_profile import ProfilingStatus, DataProfile


class TestIncrementalStatisticsAccumulator:
    """Test incremental statistics accumulation."""
    
    def setup_method(self):
        self.accumulator = IncrementalStatisticsAccumulator()
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.accumulator.total_records == 0
        assert isinstance(self.accumulator.first_seen, datetime)
        assert isinstance(self.accumulator.last_updated, datetime)
    
    def test_update_numeric_data(self):
        """Test updating with numeric data."""
        # First batch
        batch1 = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        self.accumulator.update(batch1)
        
        assert self.accumulator.total_records == 5
        stats = self.accumulator.get_current_statistics()
        
        # Check numeric column stats
        numeric_stats = stats['numeric_col']
        assert numeric_stats['count'] == 5
        assert numeric_stats['mean'] == 3.0
        assert numeric_stats['min'] == 1
        assert numeric_stats['max'] == 5
        assert numeric_stats['null_count'] == 0
        assert numeric_stats['unique_count'] == 5
    
    def test_update_with_nulls(self):
        """Test handling of null values."""
        batch = pd.DataFrame({
            'col_with_nulls': [1, None, 3, None, 5]
        })
        
        self.accumulator.update(batch)
        stats = self.accumulator.get_current_statistics()
        
        col_stats = stats['col_with_nulls']
        assert col_stats['count'] == 3  # Non-null count
        assert col_stats['null_count'] == 2
        assert col_stats['completeness_ratio'] == 0.6  # 3/5
    
    def test_cumulative_updates(self):
        """Test cumulative statistics across multiple batches."""
        # First batch
        batch1 = pd.DataFrame({'values': [1, 2, 3]})
        self.accumulator.update(batch1)
        
        # Second batch
        batch2 = pd.DataFrame({'values': [4, 5, 6]})
        self.accumulator.update(batch2)
        
        stats = self.accumulator.get_current_statistics()
        values_stats = stats['values']
        
        assert self.accumulator.total_records == 6
        assert values_stats['count'] == 6
        assert values_stats['mean'] == 3.5  # (1+2+3+4+5+6)/6
        assert values_stats['min'] == 1
        assert values_stats['max'] == 6
    
    def test_unique_values_memory_limit(self):
        """Test unique values memory management."""
        # Create batch with many unique values
        large_batch = pd.DataFrame({
            'many_unique': list(range(15000))  # More than the 10000 limit
        })
        
        self.accumulator.update(large_batch)
        stats = self.accumulator.get_current_statistics()
        
        # Should be limited to maintain memory
        assert stats['many_unique']['unique_count'] <= 10000
    
    def test_reset(self):
        """Test resetting accumulator."""
        batch = pd.DataFrame({'col': [1, 2, 3]})
        self.accumulator.update(batch)
        
        assert self.accumulator.total_records == 3
        
        self.accumulator.reset()
        
        assert self.accumulator.total_records == 0
        assert len(self.accumulator.stats) == 0


class TestStreamingPatternDetector:
    """Test streaming pattern detection."""
    
    def setup_method(self):
        self.pattern_service = Mock(spec=PatternDiscoveryService)
        self.detector = StreamingPatternDetector(self.pattern_service)
    
    def test_email_pattern_detection(self):
        """Test email pattern detection in streaming data."""
        batch = pd.DataFrame({
            'emails': [
                'user1@example.com',
                'user2@example.com',
                'user3@example.com',
                'not_an_email',
                'user4@example.com'
            ]
        })
        
        patterns = self.detector.analyze_batch_patterns(batch)
        
        assert 'emails' in patterns
        email_patterns = patterns['emails']
        assert len(email_patterns) > 0
        
        # Should detect email pattern with high confidence
        email_pattern = email_patterns[0]
        assert email_pattern.pattern_type == 'email'
        assert email_pattern.confidence > 0.5  # 4/5 = 0.8
    
    def test_phone_pattern_detection(self):
        """Test phone number pattern detection."""
        batch = pd.DataFrame({
            'phones': [
                '555-1234',
                '(555) 123-4567',
                '+1-555-123-4567',
                'not_a_phone',
                '555.123.4567'
            ]
        })
        
        patterns = self.detector.analyze_batch_patterns(batch)
        
        if 'phones' in patterns:
            phone_patterns = patterns['phones']
            assert len(phone_patterns) > 0
            phone_pattern = phone_patterns[0]
            assert phone_pattern.pattern_type == 'phone'
    
    def test_large_unique_values_sampling(self):
        """Test handling of columns with many unique values."""
        # Create batch with many unique string values
        unique_values = [f'value_{i}' for i in range(200)]
        batch = pd.DataFrame({'many_values': unique_values})
        
        patterns = self.detector.analyze_batch_patterns(batch)
        
        # Should handle large datasets without performance issues
        # (sampling should occur internally)
        assert isinstance(patterns, dict)
    
    def test_empty_batch(self):
        """Test handling of empty batches."""
        batch = pd.DataFrame({'col': []})
        patterns = self.detector.analyze_batch_patterns(batch)
        
        assert isinstance(patterns, dict)
        assert len(patterns) == 0


class TestStreamingQualityMonitor:
    """Test streaming quality monitoring."""
    
    def setup_method(self):
        self.quality_service = Mock(spec=QualityAssessmentService)
        self.config = StreamingConfig()
        self.monitor = StreamingQualityMonitor(self.quality_service, self.config)
    
    def test_batch_quality_assessment(self):
        """Test quality assessment of data batches."""
        batch = pd.DataFrame({
            'complete_col': [1, 2, 3, 4, 5],
            'partial_col': [1, None, 3, None, 5],
            'mostly_null': [1, None, None, None, None]
        })
        
        quality_result = self.monitor.assess_batch_quality(batch, [])
        
        assert 'overall_score' in quality_result
        assert 'column_metrics' in quality_result
        assert 'alerts' in quality_result
        
        # Check column metrics
        metrics = quality_result['column_metrics']
        assert 'complete_col' in metrics
        assert 'partial_col' in metrics
        assert 'mostly_null' in metrics
        
        # Complete column should have 100% completeness
        assert metrics['complete_col']['completeness'] == 100.0
        
        # Partial column should have 60% completeness
        assert metrics['partial_col']['completeness'] == 60.0
        
        # Mostly null column should trigger alert
        alerts = quality_result['alerts']
        high_missing_alerts = [a for a in alerts if a['type'] == 'high_missing_data']
        assert len(high_missing_alerts) > 0
    
    def test_quality_history_tracking(self):
        """Test quality history tracking."""
        batch = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        
        # Process multiple batches
        for _ in range(3):
            self.monitor.assess_batch_quality(batch, [])
        
        assert len(self.monitor.quality_history) == 3
        
        # Test quality trends
        trends = self.monitor.get_quality_trends()
        assert 'current_score' in trends
        assert 'average_score' in trends
        assert 'trend' in trends
        assert 'measurements_count' in trends
        assert trends['measurements_count'] == 3
    
    def test_alert_callbacks(self):
        """Test quality alert callbacks."""
        callback_called = []
        
        def test_callback(alerts):
            callback_called.extend(alerts)
        
        self.monitor.add_alert_callback(test_callback)
        
        # Create batch that should trigger alerts
        batch = pd.DataFrame({
            'bad_quality': [1, None, None, None, None]  # 80% nulls
        })
        
        self.monitor.assess_batch_quality(batch, [])
        
        # Callback should have been called with alerts
        assert len(callback_called) > 0
        assert callback_called[0]['type'] == 'high_missing_data'
    
    def test_quality_history_memory_limit(self):
        """Test quality history memory management."""
        batch = pd.DataFrame({'col': [1, 2, 3]})
        
        # Process more than the maxlen (100) batches
        for _ in range(150):
            self.monitor.assess_batch_quality(batch, [])
        
        # Should be limited to 100 entries
        assert len(self.monitor.quality_history) == 100


class TestStreamingProfilingService:
    """Test the main streaming profiling service."""
    
    def setup_method(self):
        self.config = StreamingConfig(
            update_interval_seconds=0.1,  # Fast updates for testing
            batch_size=100
        )
        self.service = StreamingProfilingService(self.config)
    
    def test_service_initialization(self):
        """Test proper service initialization."""
        assert not self.service.is_running
        assert self.service.current_profile is None
        assert isinstance(self.service.stats_accumulator, IncrementalStatisticsAccumulator)
        assert isinstance(self.service.pattern_detector, StreamingPatternDetector)
        assert isinstance(self.service.quality_monitor, StreamingQualityMonitor)
    
    def test_start_stop_streaming(self):
        """Test starting and stopping streaming profiling."""
        dataset_id = "test_dataset"
        source_config = {"type": "stream", "source": "test"}
        
        # Start streaming
        profile = self.service.start_streaming(dataset_id, source_config)
        
        assert self.service.is_running
        assert profile is not None
        assert profile.status == ProfilingStatus.RUNNING
        assert str(profile.dataset_id.value) == dataset_id
        
        # Stop streaming
        final_profile = self.service.stop_streaming()
        
        assert not self.service.is_running
        assert final_profile is not None
        assert final_profile.status == ProfilingStatus.COMPLETED
    
    def test_add_data_batch(self):
        """Test adding data batches to streaming service."""
        dataset_id = "test_dataset"
        source_config = {"type": "stream"}
        
        self.service.start_streaming(dataset_id, source_config)
        
        # Add data batch
        batch = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30],
            'category': ['A', 'B', 'A']
        })
        
        self.service.add_data_batch(batch)
        
        # Give processing time
        time.sleep(0.2)
        
        # Check metrics
        metrics = self.service.get_streaming_metrics()
        assert metrics.records_processed > 0
        assert metrics.batches_processed > 0
        
        self.service.stop_streaming()
    
    def test_current_profile_generation(self):
        """Test current profile snapshot generation."""
        dataset_id = "test_dataset"
        source_config = {"type": "stream"}
        
        self.service.start_streaming(dataset_id, source_config)
        
        # Add some data
        batch = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        self.service.add_data_batch(batch)
        time.sleep(0.2)  # Allow processing
        
        # Get current profile
        current_profile = self.service.get_current_profile()
        
        assert current_profile is not None
        assert current_profile.schema_profile is not None
        assert current_profile.quality_assessment is not None
        assert current_profile.profiling_metadata is not None
        
        # Check schema profile
        schema = current_profile.schema_profile
        assert schema.total_rows == 5
        assert schema.total_columns == 2
        assert len(schema.columns) == 2
        
        self.service.stop_streaming()
    
    def test_error_handling_not_running(self):
        """Test error handling when service is not running."""
        batch = pd.DataFrame({'col': [1, 2, 3]})
        
        with pytest.raises(Exception, match="not running"):
            self.service.add_data_batch(batch)
    
    def test_double_start_error(self):
        """Test error when trying to start already running service."""
        dataset_id = "test_dataset"
        source_config = {"type": "stream"}
        
        self.service.start_streaming(dataset_id, source_config)
        
        with pytest.raises(Exception, match="already running"):
            self.service.start_streaming(dataset_id, source_config)
        
        self.service.stop_streaming()
    
    def test_streaming_metrics(self):
        """Test streaming metrics collection."""
        dataset_id = "test_dataset"
        source_config = {"type": "stream"}
        
        self.service.start_streaming(dataset_id, source_config)
        
        # Process multiple batches
        for i in range(3):
            batch = pd.DataFrame({
                'batch_id': [i] * 10,
                'value': list(range(10))
            })
            self.service.add_data_batch(batch)
        
        time.sleep(0.3)  # Allow processing
        
        metrics = self.service.get_streaming_metrics()
        
        assert metrics.records_processed >= 30  # 3 batches * 10 rows
        assert metrics.batches_processed >= 3
        assert metrics.processing_rate_per_second > 0
        assert metrics.last_update_time is not None
        
        self.service.stop_streaming()
    
    def test_update_callbacks(self):
        """Test profile update callbacks."""
        callback_profiles = []
        
        def update_callback(profile):
            callback_profiles.append(profile)
        
        self.service.add_update_callback(update_callback)
        
        dataset_id = "test_dataset"
        source_config = {"type": "stream"}
        
        self.service.start_streaming(dataset_id, source_config)
        
        # Add data to trigger updates
        batch = pd.DataFrame({'col': [1, 2, 3]})
        self.service.add_data_batch(batch)
        
        time.sleep(0.2)  # Allow processing and updates
        
        self.service.stop_streaming()
        
        # Callback should have been called
        assert len(callback_profiles) > 0
    
    def test_error_callbacks(self):
        """Test error handling callbacks."""
        error_list = []
        
        def error_callback(error):
            error_list.append(error)
        
        self.service.add_error_callback(error_callback)
        
        # Start service
        dataset_id = "test_dataset"
        source_config = {"type": "stream"}
        self.service.start_streaming(dataset_id, source_config)
        
        # This test would require mocking internal errors
        # For now, just verify the callback registration works
        assert len(self.service.error_callbacks) == 1
        
        self.service.stop_streaming()


class TestStreamingConfig:
    """Test streaming configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        
        assert config.mode == StreamingMode.MICRO_BATCH
        assert config.batch_size == 1000
        assert config.window_size == 10000
        assert config.update_interval_seconds == 30.0
        assert config.update_strategy == UpdateStrategy.INCREMENTAL
        assert config.memory_limit_mb == 500.0
        assert config.max_age_hours == 24.0
        assert config.quality_threshold == 0.8
        assert config.enable_anomaly_detection == True
        assert config.persist_intermediate_results == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            mode=StreamingMode.REAL_TIME,
            batch_size=500,
            update_interval_seconds=10.0,
            memory_limit_mb=1000.0
        )
        
        assert config.mode == StreamingMode.REAL_TIME
        assert config.batch_size == 500
        assert config.update_interval_seconds == 10.0
        assert config.memory_limit_mb == 1000.0


class TestStreamingModes:
    """Test different streaming modes."""
    
    def test_real_time_mode(self):
        """Test real-time streaming mode."""
        config = StreamingConfig(mode=StreamingMode.REAL_TIME)
        service = StreamingProfilingService(config)
        
        assert service.config.mode == StreamingMode.REAL_TIME
    
    def test_micro_batch_mode(self):
        """Test micro-batch streaming mode."""
        config = StreamingConfig(mode=StreamingMode.MICRO_BATCH)
        service = StreamingProfilingService(config)
        
        assert service.config.mode == StreamingMode.MICRO_BATCH
    
    def test_sliding_window_mode(self):
        """Test sliding window streaming mode."""
        config = StreamingConfig(mode=StreamingMode.SLIDING_WINDOW)
        service = StreamingProfilingService(config)
        
        assert service.config.mode == StreamingMode.SLIDING_WINDOW
    
    def test_tumbling_window_mode(self):
        """Test tumbling window streaming mode."""
        config = StreamingConfig(mode=StreamingMode.TUMBLING_WINDOW)
        service = StreamingProfilingService(config)
        
        assert service.config.mode == StreamingMode.TUMBLING_WINDOW


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming profiling."""
    
    def test_end_to_end_streaming_profiling(self):
        """Test complete streaming profiling workflow."""
        config = StreamingConfig(
            mode=StreamingMode.MICRO_BATCH,
            batch_size=50,
            update_interval_seconds=0.1
        )
        service = StreamingProfilingService(config)
        
        # Start streaming
        dataset_id = "integration_test"
        source_config = {"type": "simulated_stream"}
        
        profile = service.start_streaming(dataset_id, source_config)
        assert profile.status == ProfilingStatus.RUNNING
        
        # Simulate streaming data with different characteristics
        batches = [
            pd.DataFrame({
                'user_id': [1, 2, 3, 4, 5],
                'email': ['user1@example.com', 'user2@test.com', 'user3@example.com', 'invalid', 'user4@example.com'],
                'age': [25, 30, 35, 28, 22],
                'score': [85.5, 92.3, 78.1, 88.7, 95.2]
            }),
            pd.DataFrame({
                'user_id': [6, 7, 8, 9, 10],
                'email': ['user5@example.com', 'user6@test.com', 'user7@example.com', 'user8@test.com', 'user9@example.com'],
                'age': [27, 33, 24, 31, 29],
                'score': [79.8, 87.4, 91.6, 83.2, 89.9]
            }),
            pd.DataFrame({
                'user_id': [11, 12, 13, 14, 15],
                'email': ['user10@example.com', 'user11@test.com', None, 'user12@example.com', 'user13@test.com'],
                'age': [26, None, 32, 28, 30],
                'score': [88.3, 90.1, 85.7, 92.8, 87.5]
            })
        ]
        
        # Process batches
        for batch in batches:
            service.add_data_batch(batch)
            time.sleep(0.05)  # Small delay between batches
        
        # Allow processing time
        time.sleep(0.5)
        
        # Get final profile
        final_profile = service.stop_streaming()
        
        # Verify comprehensive profiling results
        assert final_profile.status == ProfilingStatus.COMPLETED
        assert final_profile.schema_profile is not None
        assert final_profile.quality_assessment is not None
        
        # Check schema analysis
        schema = final_profile.schema_profile
        assert schema.total_rows == 15  # 3 batches * 5 rows each
        assert schema.total_columns == 4
        
        # Verify column profiles
        column_names = [col.column_name for col in schema.columns]
        assert set(column_names) == {'user_id', 'email', 'age', 'score'}
        
        # Check for email pattern detection
        email_column = next((col for col in schema.columns if col.column_name == 'email'), None)
        assert email_column is not None
        assert len(email_column.patterns) > 0  # Should detect email patterns
        
        # Verify quality assessment
        quality = final_profile.quality_assessment
        assert quality.overall_score > 0
        assert quality.completeness_score > 0
        
        # Check metrics
        metrics = service.get_streaming_metrics()
        assert metrics.records_processed == 15
        assert metrics.batches_processed == 3
        assert metrics.processing_rate_per_second > 0


if __name__ == "__main__":
    pytest.main([__file__])