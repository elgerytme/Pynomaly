"""Protocol definitions for clean architecture interfaces."""

from .data_loader_protocol import (
    BatchDataLoaderProtocol,
    DatabaseLoaderProtocol,
    DataLoaderProtocol,
    StreamingDataLoaderProtocol,
)
from .detector_protocol import (
    DetectorProtocol,
    EnsembleDetectorProtocol,
    ExplainableDetectorProtocol,
    StreamingDetectorProtocol,
)
from .repository_protocol import (
    AlertNotificationRepositoryProtocol,
    AlertRepositoryProtocol,
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
    ExperimentRepositoryProtocol,
    ExperimentRunRepositoryProtocol,
    ModelRepositoryProtocol,
    ModelVersionRepositoryProtocol,
    PipelineRepositoryProtocol,
    PipelineRunRepositoryProtocol,
    RepositoryProtocol,
)

__all__ = [
    "DetectorProtocol",
    "ExplainableDetectorProtocol",
    "EnsembleDetectorProtocol",
    "StreamingDetectorProtocol",
    "RepositoryProtocol",
    "DetectorRepositoryProtocol",
    "DatasetRepositoryProtocol",
    "DetectionResultRepositoryProtocol",
    "ModelRepositoryProtocol",
    "ModelVersionRepositoryProtocol",
    "ExperimentRepositoryProtocol",
    "ExperimentRunRepositoryProtocol",
    "PipelineRepositoryProtocol",
    "PipelineRunRepositoryProtocol",
    "AlertRepositoryProtocol",
    "AlertNotificationRepositoryProtocol",
    "DataLoaderProtocol",
    "BatchDataLoaderProtocol",
    "StreamingDataLoaderProtocol",
    "DatabaseLoaderProtocol",
]
