"""Application services module.

This module exports all application services including the new batch processing
orchestration system for handling large dataset processing efficiently.
"""

# Core services
from .data_loader_service import DataLoaderService
from .detection_service import DetectionService
from .ensemble_service import EnsembleService
from .experiment_tracking_service import ExperimentTrackingService
from .model_persistence_service import ModelPersistenceService

# Batch processing services
from .batch_processing_service import (
    BatchProcessingService,
    BatchJob,
    BatchStatus,
    BatchPriority,
    BatchConfig,
    BatchMetrics,
    BatchCheckpoint,
    ProgressCallback,
    BatchProcessor
)

from .batch_configuration_manager import (
    BatchConfigurationManager,
    SystemResources,
    DataCharacteristics,
    ProcessingProfile,
    BatchOptimizationResult
)

from .batch_orchestrator import (
    BatchOrchestrator,
    BatchJobRequest,
    BatchJobResult,
    JobDependencyManager
)

from .batch_monitoring_service import (
    BatchMonitoringService,
    SystemMetrics,
    BatchJobMetrics,
    BatchAlert,
    ProgressEvent,
    AlertLevel,
    MetricType
)

from .batch_recovery_service import (
    BatchRecoveryService,
    FailureType,
    RecoveryStrategy,
    FailureRecord,
    RecoveryConfig
)

# Create aliases for backward compatibility
DatasetService = DataLoaderService
DetectorService = DetectionService

__all__ = [
    # Core services
    "DataLoaderService",
    "DatasetService",
    "DetectionService",
    "DetectorService",
    "EnsembleService",
    "ModelPersistenceService",
    "ExperimentTrackingService",
    
    # Batch processing services
    "BatchProcessingService",
    "BatchConfigurationManager",
    "BatchOrchestrator",
    "BatchMonitoringService",
    "BatchRecoveryService",
    
    # Batch processing models
    "BatchJob",
    "BatchStatus",
    "BatchPriority",
    "BatchConfig",
    "BatchMetrics",
    "BatchCheckpoint",
    "BatchJobRequest",
    "BatchJobResult",
    "BatchOptimizationResult",
    "SystemResources",
    "DataCharacteristics",
    "ProcessingProfile",
    "JobDependencyManager",
    "SystemMetrics",
    "BatchJobMetrics",
    "BatchAlert",
    "ProgressEvent",
    "FailureRecord",
    "RecoveryConfig",
    
    # Enums
    "AlertLevel",
    "MetricType",
    "FailureType", 
    "RecoveryStrategy",
    
    # Protocols/Types
    "ProgressCallback",
    "BatchProcessor"
]
