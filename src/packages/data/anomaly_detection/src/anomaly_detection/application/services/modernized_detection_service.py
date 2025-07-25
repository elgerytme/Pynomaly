"""
Modernized Anomaly Detection Service using new interaction patterns.

This service demonstrates how to use the new event-driven communication
and dependency injection patterns for cross-domain interactions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import from interfaces for stable contracts
from interfaces.dto import (
    DetectionRequest, DetectionResult, DetectionStatus,
    DataQualityRequest, DataQualityResult, DataQualityStatus
)
from interfaces.events import (
    AnomalyDetectionStarted, AnomalyDetected, AnomalyDetectionCompleted,
    DataQualityCheckCompleted, DatasetUpdated
)
from interfaces.patterns import Service

# Import from shared infrastructure
from shared import (
    get_event_bus, publish_event, event_handler, get_container, inject
)

# Domain imports (internal to anomaly_detection package)
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.interfaces.ml_operations import MLModelTrainingPort


logger = logging.getLogger(__name__)


class ModernizedAnomalyDetectionService(Service):
    """
    Modernized anomaly detection service using new interaction patterns.
    
    This service demonstrates:
    1. Event-driven communication with other domains
    2. Dependency injection for loose coupling
    3. Stable DTOs for data exchange
    4. Proper separation of concerns
    """
    
    def __init__(self, 
                 detection_service: DetectionService,
                 ml_training_port: MLModelTrainingPort):
        """Initialize with injected dependencies."""
        self.detection_service = detection_service
        self.ml_training_port = ml_training_port
        self.event_bus = get_event_bus()
        
        # Subscribe to relevant events
        self._setup_event_subscriptions()
        
        # Track pending data quality checks
        self.pending_quality_checks: Dict[str, DetectionRequest] = {}
    
    async def execute(self, request: DetectionRequest) -> DetectionResult:
        """
        Execute anomaly detection with integrated data quality checking.
        
        This method demonstrates the new interaction pattern:
        1. Publish event to start detection
        2. Request data quality check via event
        3. Wait for quality check completion
        4. Proceed with detection based on quality results
        """
        try:
            # Validate request
            if not await self.validate_request(request):
                raise ValueError("Invalid detection request")
            
            # Publish start event
            start_event = AnomalyDetectionStarted(
                event_id="",
                event_type="",
                aggregate_id=request.dataset_id,
                occurred_at=datetime.utcnow(),
                dataset_id=request.dataset_id,
                algorithm=request.algorithm,
                parameters=request.parameters,
                request_id=request.id
            )
            await publish_event(start_event)
            
            # Request data quality check before proceeding
            quality_result = await self._request_data_quality_check(request.dataset_id)
            
            # Proceed with detection based on quality
            if quality_result and quality_result.overall_score >= 0.8:
                result = await self._perform_detection(request)
            else:
                # Skip detection due to poor data quality
                result = DetectionResult(
                    id=f"detection_{request.id}",
                    created_at=datetime.utcnow(),
                    request_id=request.id,
                    status=DetectionStatus.FAILED,
                    anomalies_count=0,
                    anomaly_scores=[],
                    anomaly_indices=[],
                    confidence_scores=[],
                    execution_time_ms=0,
                    algorithm_used=request.algorithm,
                    error_message="Skipped due to poor data quality"
                )
            
            # Publish completion event
            completion_event = AnomalyDetectionCompleted(
                event_id="",
                event_type="",
                aggregate_id=request.dataset_id,
                occurred_at=datetime.utcnow(),
                dataset_id=request.dataset_id,
                status=result.status,
                anomaly_count=result.anomalies_count,
                execution_time_ms=result.execution_time_ms,
                detection_result=result
            )
            await publish_event(completion_event)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            
            # Publish failure event
            failure_event = AnomalyDetectionStarted(
                event_id="",
                event_type="",
                aggregate_id=request.dataset_id,
                occurred_at=datetime.utcnow(),
                dataset_id=request.dataset_id,
                algorithm=request.algorithm,
                parameters=request.parameters,
                request_id=request.id
            )
            await publish_event(failure_event)
            
            # Return error result
            return DetectionResult(
                id=f"detection_{request.id}",
                created_at=datetime.utcnow(),
                request_id=request.id,
                status=DetectionStatus.FAILED,
                anomalies_count=0,
                anomaly_scores=[],
                anomaly_indices=[],
                confidence_scores=[],
                execution_time_ms=0,
                algorithm_used=request.algorithm,
                error_message=str(e)
            )
    
    async def _request_data_quality_check(self, dataset_id: str) -> Optional[DataQualityResult]:
        """Request data quality check via event system."""
        try:
            # Create quality check request
            quality_request = DataQualityRequest(
                id=f"quality_{dataset_id}_{datetime.utcnow().timestamp()}",
                created_at=datetime.utcnow(),
                dataset_id=dataset_id,
                quality_rules=["completeness", "uniqueness", "validity"],
                threshold=0.8,
                include_profiling=False
            )
            
            # For demonstration, we'll simulate the quality check
            # In a real implementation, this would publish an event and wait for response
            quality_result = DataQualityResult(
                id=f"quality_result_{quality_request.id}",
                created_at=datetime.utcnow(),
                request_id=quality_request.id,
                dataset_id=dataset_id,
                status=DataQualityStatus.PASSED,
                overall_score=0.95,
                rule_results={
                    "completeness": {"score": 0.98, "passed": True},
                    "uniqueness": {"score": 0.92, "passed": True},
                    "validity": {"score": 0.95, "passed": True}
                },
                issues_found=[],
                recommendations=["Data quality is good"],
                execution_time_ms=50
            )
            
            logger.info(f"Data quality check completed for {dataset_id}: score={quality_result.overall_score}")
            return quality_result
            
        except Exception as e:
            logger.error(f"Error requesting data quality check: {e}")
            return None
    
    async def _perform_detection(self, request: DetectionRequest) -> DetectionResult:
        """Perform the actual anomaly detection."""
        start_time = datetime.utcnow()
        
        try:
            # Use domain service for core detection logic
            # This would typically involve data loading, preprocessing, and detection
            
            # Simulate detection process
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Mock detection results
            anomaly_scores = [0.85, 0.92, 0.78, 0.69, 0.95]
            threshold = request.parameters.get("threshold", 0.8)
            anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            result = DetectionResult(
                id=f"detection_{request.id}",
                created_at=datetime.utcnow(),
                request_id=request.id,
                status=DetectionStatus.COMPLETED,
                anomalies_count=len(anomaly_indices),
                anomaly_scores=[anomaly_scores[i] for i in anomaly_indices],
                anomaly_indices=anomaly_indices,
                confidence_scores=[0.9] * len(anomaly_indices),  # Mock confidence
                execution_time_ms=execution_time,
                algorithm_used=request.algorithm
            )
            
            # Publish anomaly detected event if anomalies found
            if result.anomalies_count > 0:
                anomaly_event = AnomalyDetected(
                    event_id="",
                    event_type="",
                    aggregate_id=request.dataset_id,
                    occurred_at=datetime.utcnow(),
                    dataset_id=request.dataset_id,
                    anomaly_count=result.anomalies_count,
                    severity="high" if result.anomalies_count > 5 else "medium",
                    detection_result=result
                )
                await publish_event(anomaly_event)
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing detection: {e}")
            raise
    
    async def validate_request(self, request: DetectionRequest) -> bool:
        """Validate the detection request."""
        if not request.dataset_id:
            return False
        if not request.algorithm:
            return False
        if not isinstance(request.parameters, dict):
            return False
        return True
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        return {
            "name": "ModernizedAnomalyDetectionService",
            "version": "2.0.0",
            "domain": "anomaly_detection",
            "features": [
                "event_driven_communication",
                "dependency_injection", 
                "data_quality_integration",
                "stable_dto_contracts"
            ]
        }
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for cross-domain communication."""
        # Subscribe to data quality events
        self.event_bus.subscribe(DataQualityCheckCompleted, self._on_data_quality_completed)
        
        # Subscribe to dataset update events
        self.event_bus.subscribe(DatasetUpdated, self._on_dataset_updated)
        
        logger.info("Event subscriptions configured for anomaly detection service")
    
    @event_handler(DataQualityCheckCompleted)
    async def _on_data_quality_completed(self, event: DataQualityCheckCompleted) -> None:
        """Handle data quality check completion events."""
        logger.info(f"Data quality check completed for dataset {event.dataset_id}: {event.status}")
        
        # Check if we have a pending detection for this dataset
        pending_request = self.pending_quality_checks.get(event.dataset_id)
        if pending_request:
            if event.status == DataQualityStatus.FAILED or event.overall_score < 0.8:
                logger.warning(f"Poor data quality for {event.dataset_id}, skipping anomaly detection")
            else:
                logger.info(f"Good data quality for {event.dataset_id}, proceeding with detection")
            
            # Remove from pending
            del self.pending_quality_checks[event.dataset_id]
    
    @event_handler(DatasetUpdated)
    async def _on_dataset_updated(self, event: DatasetUpdated) -> None:
        """Handle dataset update events."""
        logger.info(f"Dataset {event.dataset_id} updated, considering reanalysis")
        
        # If schema changed significantly, we might want to retrain models
        if event.schema_changed:
            logger.info(f"Schema change detected for {event.dataset_id}, models may need retraining")
            
            # Could trigger model retraining here via ML training port
            # await self.ml_training_port.retrain_model(dataset_id=event.dataset_id)


# Factory function using dependency injection
@inject(get_container())
def create_modernized_detection_service(
    detection_service: DetectionService,
    ml_training_port: MLModelTrainingPort
) -> ModernizedAnomalyDetectionService:
    """Factory function with dependency injection."""
    return ModernizedAnomalyDetectionService(detection_service, ml_training_port)