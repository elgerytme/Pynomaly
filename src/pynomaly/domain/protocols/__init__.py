"""Domain protocol definitions - alias to shared protocols."""

# Re-export all protocols from shared.protocols for backward compatibility
# Specific aliases for common domain protocols
from pynomaly.shared.protocols import *
from pynomaly.shared.protocols import DetectorProtocol as AdapterProtocol

# Domain-specific protocols
from .audit_logger_protocol import AuditLevel, AuditLoggerProtocol, SecurityEventType
from .training_protocols import (
    DatasetRepositoryProtocol as TrainingDatasetRepositoryProtocol,
)
from .training_protocols import (
    ExperimentTrackerProtocol,
    ModelTrainerProtocol,
    NotificationServiceProtocol,
    OptimizationEngineProtocol,
    ResourceManagerProtocol,
    TrainingJobRepositoryProtocol,
)

__all__ = [
    "AdapterProtocol",
    # Domain-specific protocols
    "AuditLevel",
    "AuditLoggerProtocol",
    "SecurityEventType",
    # Training protocols
    "TrainingJobRepositoryProtocol",
    "ModelTrainerProtocol",
    "TrainingDatasetRepositoryProtocol",
    "ExperimentTrackerProtocol",
    "OptimizationEngineProtocol",
    "NotificationServiceProtocol",
    "ResourceManagerProtocol",
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
