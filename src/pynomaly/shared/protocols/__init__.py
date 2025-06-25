"""Protocol definitions for clean architecture interfaces."""

from .detector_protocol import (
    DetectorProtocol,
    ExplainableDetectorProtocol,
    EnsembleDetectorProtocol,
    StreamingDetectorProtocol
)
from .repository_protocol import (
    RepositoryProtocol,
    DetectorRepositoryProtocol,
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    ModelRepositoryProtocol,
    ModelVersionRepositoryProtocol,
    ExperimentRepositoryProtocol,
    ExperimentRunRepositoryProtocol,
    PipelineRepositoryProtocol,
    PipelineRunRepositoryProtocol,
    AlertRepositoryProtocol,
    AlertNotificationRepositoryProtocol
)
from .data_loader_protocol import (
    DataLoaderProtocol,
    BatchDataLoaderProtocol,
    StreamingDataLoaderProtocol,
    DatabaseLoaderProtocol
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
    "DatabaseLoaderProtocol"
]