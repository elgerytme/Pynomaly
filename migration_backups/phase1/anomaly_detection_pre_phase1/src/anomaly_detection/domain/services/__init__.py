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

# Legacy services have been removed - use consolidated versions instead

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
    
    # Legacy services have been removed - use consolidated versions
]