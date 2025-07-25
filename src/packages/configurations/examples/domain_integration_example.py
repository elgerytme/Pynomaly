"""
Example: Cross-domain integration using recommended patterns.

This example demonstrates how to properly integrate different domain packages
(ai/, data/) using the interfaces and shared infrastructure while maintaining
proper architectural boundaries.
"""

import asyncio
import logging
from typing import Dict, Any

# Import from interfaces package for stable contracts
from interfaces.dto import (
    DetectionRequest, DetectionResult, 
    DataQualityRequest, DataQualityResult,
    ModelTrainingRequest, ModelTrainingResult
)
from interfaces.events import (
    AnomalyDetected, DataQualityCheckCompleted, 
    ModelTrainingCompleted, DatasetUpdated
)
from interfaces.patterns import Repository, Service

# Import from shared package for infrastructure
from shared import (
    get_event_bus, get_container, configure_container,
    register_service, register_repository, event_handler
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityService(Service):
    """Example data quality service from data domain."""
    
    async def execute(self, request: DataQualityRequest) -> DataQualityResult:
        """Execute data quality check."""
        logger.info(f"Starting data quality check for dataset {request.dataset_id}")
        
        # Simulate data quality analysis
        await asyncio.sleep(0.1)
        
        result = DataQualityResult(
            id=f"dq_{request.id}",
            created_at=request.created_at,
            request_id=request.id,
            dataset_id=request.dataset_id,
            status="passed",
            overall_score=0.95,
            rule_results={
                "completeness": {"score": 0.98, "passed": True},
                "uniqueness": {"score": 0.92, "passed": True},
                "validity": {"score": 0.94, "passed": True}
            },
            issues_found=[],
            recommendations=["Consider adding more validation rules"],
            execution_time_ms=100
        )
        
        # Publish event about completion
        event = DataQualityCheckCompleted(
            event_id="",
            event_type="",
            aggregate_id=request.dataset_id,
            occurred_at=result.created_at,
            dataset_id=request.dataset_id,
            status=result.status,
            overall_score=result.overall_score,
            issues_count=len(result.issues_found),
            quality_result=result
        )
        
        event_bus = get_event_bus()
        await event_bus.publish(event)
        
        return result
    
    async def validate_request(self, request: DataQualityRequest) -> bool:
        """Validate the service request."""
        return bool(request.dataset_id and request.quality_rules)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        return {
            "name": "DataQualityService",
            "version": "1.0.0",
            "domain": "data"
        }


class AnomalyDetectionService(Service):
    """Example anomaly detection service from ai domain."""
    
    async def execute(self, request: DetectionRequest) -> DetectionResult:
        """Execute anomaly detection."""
        logger.info(f"Starting anomaly detection for dataset {request.dataset_id}")
        
        # Simulate anomaly detection
        await asyncio.sleep(0.2)
        
        result = DetectionResult(
            id=f"ad_{request.id}",
            created_at=request.created_at,
            request_id=request.id,
            status="completed",
            anomalies_count=3,
            anomaly_scores=[0.85, 0.92, 0.78],
            anomaly_indices=[42, 128, 256],
            confidence_scores=[0.91, 0.87, 0.83],
            execution_time_ms=200,
            algorithm_used=request.algorithm
        )
        
        # Publish event about detected anomalies
        event = AnomalyDetected(
            event_id="",
            event_type="",
            aggregate_id=request.dataset_id,
            occurred_at=result.created_at,
            dataset_id=request.dataset_id,
            anomaly_count=result.anomalies_count,
            severity="medium" if result.anomalies_count < 5 else "high",
            detection_result=result
        )
        
        event_bus = get_event_bus()
        await event_bus.publish(event)
        
        return result
    
    async def validate_request(self, request: DetectionRequest) -> bool:
        """Validate the service request."""
        return bool(request.dataset_id and request.algorithm)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        return {
            "name": "AnomalyDetectionService", 
            "version": "1.0.0",
            "domain": "ai"
        }


class IntegratedWorkflowService:
    """
    Service that coordinates between domains using events and DI.
    
    This service demonstrates:
    1. Cross-domain communication via events
    2. Dependency injection for loose coupling
    3. Proper use of interfaces for stability
    """
    
    def __init__(self, 
                 data_quality_service: DataQualityService,
                 anomaly_detection_service: AnomalyDetectionService):
        self.data_quality_service = data_quality_service
        self.anomaly_detection_service = anomaly_detection_service
        self.event_bus = get_event_bus()
        
        # Subscribe to relevant events
        self.event_bus.subscribe(DataQualityCheckCompleted, self._on_quality_check_completed)
        self.event_bus.subscribe(DatasetUpdated, self._on_dataset_updated)
    
    async def run_integrated_analysis(self, dataset_id: str) -> Dict[str, Any]:
        """Run integrated data quality and anomaly detection analysis."""
        logger.info(f"Starting integrated analysis for dataset {dataset_id}")
        
        # Step 1: Run data quality check
        quality_request = DataQualityRequest(
            id="",
            created_at=None,
            dataset_id=dataset_id,
            quality_rules=["completeness", "uniqueness", "validity"],
            threshold=0.8
        )
        
        quality_result = await self.data_quality_service.execute(quality_request)
        
        # Step 2: Only run anomaly detection if data quality is acceptable
        detection_result = None
        if quality_result.overall_score >= 0.8:
            detection_request = DetectionRequest(
                id="",
                created_at=None,
                dataset_id=dataset_id,
                algorithm="isolation_forest",
                parameters={"contamination": 0.1}
            )
            
            detection_result = await self.anomaly_detection_service.execute(detection_request)
        else:
            logger.warning(f"Skipping anomaly detection due to poor data quality: {quality_result.overall_score}")
        
        return {
            "dataset_id": dataset_id,
            "data_quality": {
                "score": quality_result.overall_score,
                "status": quality_result.status,
                "issues": len(quality_result.issues_found)
            },
            "anomaly_detection": {
                "anomalies_found": detection_result.anomalies_count if detection_result else 0,
                "status": detection_result.status if detection_result else "skipped"
            }
        }
    
    @event_handler(DataQualityCheckCompleted)
    async def _on_quality_check_completed(self, event: DataQualityCheckCompleted) -> None:
        """React to data quality check completion."""
        logger.info(f"Data quality check completed for dataset {event.dataset_id} with score {event.overall_score}")
        
        # Example: Trigger additional analysis if quality is poor
        if event.overall_score < 0.7:
            logger.warning(f"Poor data quality detected, consider data remediation")
    
    @event_handler(DatasetUpdated) 
    async def _on_dataset_updated(self, event: DatasetUpdated) -> None:
        """React to dataset updates."""
        logger.info(f"Dataset {event.dataset_id} updated, may need reanalysis")


async def configure_services() -> None:
    """Configure dependency injection container."""
    def setup_container(container):
        # Register domain services
        register_service(container, DataQualityService, DataQualityService)
        register_service(container, AnomalyDetectionService, AnomalyDetectionService)
        
        # Register workflow service with dependencies
        container.register_singleton(
            IntegratedWorkflowService,
            factory=lambda dq_service, ad_service: IntegratedWorkflowService(dq_service, ad_service)
        )
    
    configure_container(setup_container)


async def main():
    """Main example function."""
    # Configure services and start event bus
    await configure_services()
    
    event_bus = get_event_bus()
    await event_bus.start()
    
    try:
        # Get the integrated workflow service from DI container
        container = get_container()
        workflow_service = container.resolve(IntegratedWorkflowService)
        
        # Run integrated analysis
        result = await workflow_service.run_integrated_analysis("dataset_123")
        
        logger.info(f"Analysis complete: {result}")
        
        # Wait a bit for events to be processed
        await asyncio.sleep(1)
        
        # Show event bus metrics
        metrics = event_bus.get_metrics()
        logger.info(f"Event bus metrics: {metrics}")
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())