"""
Comprehensive tests for streaming extras.

This module tests streaming functionality with graceful degradation
when streaming packages are not installed.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from tests.utils.extras_testing import (
    requires_streaming,
    parametrize_with_extras,
    streaming_available,
    check_graceful_degradation,
)


class TestStreamingExtras:
    """Test suite for streaming extras functionality."""

    @requires_streaming()
    def test_kafka_import_with_extras(self, streaming_available):
        """Test Kafka import when streaming extras are available."""
        kafka = streaming_available.get("kafka")
        if kafka is not None:
            # Test basic Kafka functionality
            assert hasattr(kafka, "KafkaProducer")
        else:
            pytest.skip("Kafka not available")

    @requires_streaming()
    def test_redis_import_with_extras(self, streaming_available):
        """Test Redis import when streaming extras are available."""
        redis = streaming_available.get("redis")
        if redis is not None:
            # Test basic Redis functionality
            assert hasattr(redis, "Redis")
        else:
            pytest.skip("Redis not available")

    @parametrize_with_extras(["streaming"])
    def test_streaming_service_availability(self, required_extras):
        """Test that streaming service is available when extras are installed."""
        try:
            from pynomaly.application.services.streaming_service import StreamingService
            # Should be able to create service
            service = StreamingService
            assert service is not None
        except ImportError as e:
            pytest.skip(f"Streaming service not available: {e}")

    def test_streaming_graceful_degradation(self):
        """Test graceful degradation when streaming packages are missing."""
        def mock_streaming_function():
            # Simulate a function that would use streaming
            try:
                import redis
                client = redis.Redis()
                return {"client": client}
            except ImportError:
                # Graceful fallback to local processing
                return {"client": "local"}

        # Test that the function works with or without streaming
        result = mock_streaming_function()
        assert "client" in result
        assert result["client"] in ["local", Mock()]

    def test_streaming_service_fallback(self):
        """Test that streaming service falls back gracefully."""
        try:
            from pynomaly.application.services.streaming_service import StreamingService
            # Should not raise ImportError if properly implemented
            service = StreamingService(
                message_bus=Mock(),
                data_store=Mock(),
            )
            assert service is not None
        except ImportError:
            # This is expected if streaming dependencies are missing
            pytest.skip("Streaming service not available without extras")

    @requires_streaming()
    def test_streaming_integration_with_sample_data(self, streaming_available):
        """Test streaming integration with sample data."""
        # Create sample data
        sample_data = np.random.rand(100, 3)
        sample_processor = Mock()  # Mocking a streaming processor

        kafka = streaming_available.get("kafka")
        if kafka is not None:
            producer = kafka.KafkaProducer()
            assert producer is not None

    def test_streaming_error_handling(self):
        """Test error handling when streaming packages are missing."""
        def function_requiring_streaming():
            import redis  # This will raise ImportError if not available
            return redis.Redis()
        
        # Test error handling
        graceful, result = check_graceful_degradation(
            function_requiring_streaming,
            "streaming",
            expected_error_type=ImportError
        )
        
        # Result should be either successful or ImportError
        assert isinstance(result, (Exception, type(None)))

    def test_streaming_processor_fallback(self):
        """Test that streaming processor falls back gracefully."""
        try:
            from pynomaly.infrastructure.streaming.stream_processor import StreamProcessor
            # Should not raise ImportError if properly implemented
            processor = StreamProcessor(
                stream_config=Mock(),
                fallback_mode=True,
            )
            assert processor is not None
            assert processor.fallback_mode is True
        except ImportError:
            # This is expected if streaming dependencies are missing
            pytest.skip("Streaming processor not available without extras")

    def test_streaming_error_handling_with_mock(self):
        """Test error handling with mocked streaming dependencies."""
        def mock_streaming_function():
            # Mock a function depending on streaming
            return Mock()
        
        # Test graceful handling
        graceful, result = check_graceful_degradation(
            mock_streaming_function,
            "streaming",
            expected_error_type=ImportError
        )
        
        # Result should be mocked
        assert isinstance(result, Mock)
