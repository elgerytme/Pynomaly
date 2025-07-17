"""Domain protocol definitions - alias to shared protocols."""

# Re-export all protocols from shared.protocols for backward compatibility
# Specific aliases for common domain protocols
from monorepo.shared.protocols import DetectorProtocol as AdapterProtocol
from monorepo.shared.protocols import *

__all__ = [
    "AdapterProtocol",
    # Re-export all shared protocols
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
    "ImportProtocol",
    "ExportProtocol",
]
