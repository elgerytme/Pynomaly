"""Enterprise Adapters Package.

Universal adapter patterns for external services and frameworks, providing
consistent interfaces for databases, caches, message queues, cloud services,
and machine learning frameworks.
"""

from .base import AdapterConfiguration, AdapterFactory, AdapterRegistry, BaseAdapter
from .cache import CacheAdapter, MemcachedAdapter, RedisAdapter
from .database import DatabaseAdapter, MongoDBAdapter, SQLAlchemyAdapter
from .messaging import KafkaAdapter, MessageQueueAdapter, RabbitMQAdapter, SQSAdapter
from .ml import (
    JAXAdapter,
    MLFrameworkAdapter,
    PyTorchAdapter,
    SklearnAdapter,
    TensorFlowAdapter,
)
from .storage import AzureBlobAdapter, GCSAdapter, S3Adapter, StorageAdapter

__version__ = "0.1.0"
__all__ = [
    # Base adapter infrastructure
    "BaseAdapter",
    "AdapterFactory",
    "AdapterRegistry",
    "AdapterConfiguration",
    # Database adapters
    "DatabaseAdapter",
    "SQLAlchemyAdapter",
    "MongoDBAdapter",
    # Cache adapters
    "CacheAdapter",
    "RedisAdapter",
    "MemcachedAdapter",
    # Messaging adapters
    "MessageQueueAdapter",
    "RabbitMQAdapter",
    "KafkaAdapter",
    "SQSAdapter",
    # Storage adapters
    "StorageAdapter",
    "S3Adapter",
    "AzureBlobAdapter",
    "GCSAdapter",
    # ML framework adapters
    "MLFrameworkAdapter",
    "SklearnAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "JAXAdapter",
]
