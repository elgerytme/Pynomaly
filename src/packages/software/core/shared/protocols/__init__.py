"""Protocol definitions for clean architecture interfaces."""

from .data_loader_protocol import (
    BatchDataLoaderProtocol,
    DatabaseLoaderProtocol,
    DataLoaderProtocol,
    StreamingDataLoaderProtocol,
)
from .generic_detection_protocol import (
    BatchDetectionProtocol,
    EnsembleDetectionProtocol,
    ExplainableDetectionProtocol,
    GenericDetectionProtocol,
    StreamingDetectionProtocol,
)
from .repository_protocol import (
    AlertNotificationRepositoryProtocol,
    AlertRepositoryProtocol,
    DatasetRepositoryProtocol,
    ModelRepositoryProtocol,
    ModelVersionRepositoryProtocol,
    PipelineRepositoryProtocol,
    PipelineRunRepositoryProtocol,
    RepositoryProtocol,
)

__all__ = [
    "GenericDetectionProtocol",
    "ExplainableDetectionProtocol",
    "EnsembleDetectionProtocol",
    "StreamingDetectionProtocol",
    "BatchDetectionProtocol",
    "RepositoryProtocol",
    "DatasetRepositoryProtocol",
    "ModelRepositoryProtocol",
    "ModelVersionRepositoryProtocol",
    "PipelineRepositoryProtocol",
    "PipelineRunRepositoryProtocol",
    "AlertRepositoryProtocol",
    "AlertNotificationRepositoryProtocol",
    "DataLoaderProtocol",
    "BatchDataLoaderProtocol",
    "StreamingDataLoaderProtocol",
    "DatabaseLoaderProtocol",
]
