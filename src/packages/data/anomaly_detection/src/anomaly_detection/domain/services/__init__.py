"""Core services for anomaly detection operations."""

# Core detection services (keep as-is)
from .detection_service import DetectionService, DetectionResult
from .detection_service_simple import DetectionServiceSimple
from .ensemble_service import EnsembleService
from .explainability_service import ExplainabilityService

# Consolidated services (new architecture)
from .data_processing_service import DataProcessingService
from .processing_service import ProcessingService
from .enhanced_analytics_service import EnhancedAnalyticsService

# Legacy services (deprecated - use consolidated versions)
from .analytics_service import AnalyticsService
from .batch_processing_service import BatchProcessingService
from .data_conversion_service import DataConversionService
from .data_profiling_service import DataProfilingService
from .data_sampling_service import DataSamplingService
from .data_validation_service import DataValidationService
from .health_monitoring_service import HealthMonitoringService
from .streaming_service import StreamingService

__all__ = [
    # Core services (recommended)
    "DetectionService",
    "DetectionResult", 
    "DetectionServiceSimple",
    "EnsembleService",
    "ExplainabilityService",
    
    # Consolidated services (recommended)
    "DataProcessingService",
    "ProcessingService", 
    "EnhancedAnalyticsService",
    
    # Legacy services (deprecated)
    "AnalyticsService",
    "BatchProcessingService", 
    "DataConversionService",
    "DataProfilingService",
    "DataSamplingService",
    "DataValidationService",
    "HealthMonitoringService",
    "StreamingService",
]