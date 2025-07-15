"""Application layer protocols for dependency injection."""

from .adapter_protocols import (
    ApplicationAlgorithmFactoryProtocol,
    ApplicationDataLoaderProtocol,
    ApplicationModelSerializerProtocol,
)
from .repository_protocols import (
    ApplicationRepositoryProtocol,
    DatasetRepositoryProtocol,
    DetectorRepositoryProtocol,
    ModelRepositoryProtocol,
)
from .service_protocols import (
    ApplicationCacheProtocol,
    ApplicationConfigProtocol,
    ApplicationMetricsProtocol,
    ApplicationSecurityProtocol,
)

__all__ = [
    "ApplicationRepositoryProtocol",
    "DetectorRepositoryProtocol",
    "ModelRepositoryProtocol",
    "DatasetRepositoryProtocol",
    "ApplicationConfigProtocol",
    "ApplicationCacheProtocol",
    "ApplicationMetricsProtocol",
    "ApplicationSecurityProtocol",
    "ApplicationAlgorithmFactoryProtocol",
    "ApplicationDataLoaderProtocol",
    "ApplicationModelSerializerProtocol",
]
