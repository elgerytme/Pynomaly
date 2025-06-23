"""Streaming infrastructure package."""

from .processors import (
    ModelBasedStreamProcessor,
    StatisticalStreamProcessor,
    EnsembleStreamProcessor
)

# Optional streaming connectors
try:
    from .kafka_connector import KafkaStreamConnector
    KAFKA_AVAILABLE = True
except ImportError:
    KafkaStreamConnector = None
    KAFKA_AVAILABLE = False

try:
    from .redis_connector import RedisStreamConnector
    REDIS_AVAILABLE = True
except ImportError:
    RedisStreamConnector = None
    REDIS_AVAILABLE = False

__all__ = [
    "ModelBasedStreamProcessor",
    "StatisticalStreamProcessor", 
    "EnsembleStreamProcessor",
    "KafkaStreamConnector",
    "RedisStreamConnector",
    "KAFKA_AVAILABLE",
    "REDIS_AVAILABLE",
]