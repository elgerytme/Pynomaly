"""Memory-efficient data processing infrastructure.

This module provides streaming and memory-optimized data processing capabilities
while maintaining the simplified architecture principles of Phase 1.
"""

from .streaming_processor import (
    StreamingDataProcessor,
    MemoryOptimizedDataLoader,
    LargeDatasetAnalyzer,
    DataChunk,
    get_memory_usage,
    monitor_memory_usage
)

__all__ = [
    "StreamingDataProcessor",
    "MemoryOptimizedDataLoader", 
    "LargeDatasetAnalyzer",
    "DataChunk",
    "get_memory_usage",
    "monitor_memory_usage"
]