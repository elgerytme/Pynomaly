"""Memory-efficient data processing infrastructure.

This module provides streaming and memory-optimized data processing capabilities
while maintaining the simplified architecture principles of Phase 1.
"""

from .data_validator import (
    DataValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_file_format,
)
from .memory_efficient_processor import (
    DataChunk,
    LargeDatasetAnalyzer,
    MemoryMetrics,
    MemoryOptimizedDataLoader,
    StreamingDataProcessor,
    get_memory_usage,
    monitor_memory_usage,
)
from .streaming_processor import (
    BackpressureStrategy,
    StreamingConfig,
    StreamingMessage,
    StreamingMetrics,
    StreamingMode,
    StreamingProcessor,
)

__all__ = [
    # Memory-efficient processing
    "MemoryOptimizedDataLoader",
    "StreamingDataProcessor",
    "LargeDatasetAnalyzer",
    "DataChunk",
    "MemoryMetrics",
    "get_memory_usage",
    "monitor_memory_usage",
    # Data validation
    "DataValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationCategory",
    "validate_file_format",
    # Streaming processing
    "StreamingProcessor",
    "StreamingMessage",
    "StreamingConfig",
    "StreamingMetrics",
    "StreamingMode",
    "BackpressureStrategy",
]
