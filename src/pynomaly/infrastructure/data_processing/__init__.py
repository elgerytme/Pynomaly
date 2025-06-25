"""Memory-efficient data processing infrastructure.

This module provides streaming and memory-optimized data processing capabilities
while maintaining the simplified architecture principles of Phase 1.
"""

from .memory_efficient_processor import (
    MemoryOptimizedDataLoader,
    StreamingDataProcessor,
    LargeDatasetAnalyzer,
    DataChunk,
    MemoryMetrics,
    get_memory_usage,
    monitor_memory_usage
)
from .data_validator import (
    DataValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationCategory,
    validate_file_format
)
from .streaming_processor import (
    StreamingProcessor,
    StreamingMessage,
    StreamingConfig,
    StreamingMetrics,
    StreamingMode,
    BackpressureStrategy
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
    "BackpressureStrategy"
]