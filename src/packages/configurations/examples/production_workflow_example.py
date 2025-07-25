"""
Production workflow example demonstrating real-world usage of the new interaction patterns.

This example shows a complete MLOps workflow that spans multiple domains:
- Data quality assessment
- Anomaly detection  
- Model training and deployment
- Monitoring and alerting

It demonstrates proper use of:
- Event-driven architecture
- Dependency injection
- Cross-domain communication via stable interfaces
- Configuration composition
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import from interfaces for stable contracts
from interfaces.dto import (
    DetectionRequest, DetectionResult,
    DataQualityRequest, DataQualityResult,
    ModelTrainingRequest, ModelTrainingResult,
    AnalyticsRequest, AnalyticsResult
)
from interfaces.events import (
    AnomalyDetected, DataQualityCheckCompleted, ModelTrainingCompleted,
    DatasetUpdated, SystemHealthChanged
)
from interfaces.patterns import Service, Repository

# Import from shared infrastructure
from shared import (
    get_event_bus, get_container, configure_container,
    publish_event, event_handler, DistributedEventBus
)

# Mock service implementations for demonstration
class MockDataQualityService(Service):
    """Mock data quality service from data domain."""
    
    async def execute(self, request: DataQualityRequest) -> DataQualityResult:
        """Simulate data quality assessment."""
        await asyncio.sleep(0.1)  # Simulate processing
        
        # Simulate varying quality scores
        quality_score = 0.95 if "clean" in request.dataset_id else 0.65
        status = "passed" if quality_score >= 0.8 else "failed"
        
        return DataQualityResult(
            id=f"dq_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            dataset_id=request.dataset_id,
            status=status,
            overall_score=quality_score,
            rule_results={
                "completeness": {"score": quality_score + 0.02, "passed": True},
                "uniqueness": {"score": quality_score - 0.03, "passed": quality_score > 0.7},
                "validity": {"score": quality_score, "passed": quality_score >= 0.8}
            },
            issues_found=[] if status == "passed" else ["Missing values detected", "Duplicate records found"],
            recommendations=["Good quality" if status == "passed" else "Data cleaning required"],
            execution_time_ms=100
        )
    
    async def validate_request(self, request: DataQualityRequest) -> bool:
        return bool(request.dataset_id and request.quality_rules)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "MockDataQualityService", "domain": "data"}


class MockAnomalyDetectionService(Service):
    """Mock anomaly detection service from AI domain."""
    
    async def execute(self, request: DetectionRequest) -> DetectionResult:
        """Simulate anomaly detection."""
        await asyncio.sleep(0.2)  # Simulate processing
        
        # Simulate varying anomaly counts
        anomaly_count = 5 if "anomalous" in request.dataset_id else 1
        anomaly_scores = [0.85, 0.92, 0.78, 0.89, 0.95][:anomaly_count]
        
        return DetectionResult(
            id=f"ad_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            status="completed",
            anomalies_count=anomaly_count,
            anomaly_scores=anomaly_scores,
            anomaly_indices=list(range(anomaly_count)),
            confidence_scores=[0.9] * anomaly_count,
            execution_time_ms=200,
            algorithm_used=request.algorithm
        )
    
    async def validate_request(self, request: DetectionRequest) -> bool:
        return bool(request.dataset_id and request.algorithm)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "MockAnomalyDetectionService", "domain": "ai"}


class MockModelTrainingService(Service):
    """Mock model training service from AI domain."""
    
    async def execute(self, request: ModelTrainingRequest) -> ModelTrainingResult:
        """Simulate model training."""
        await asyncio.sleep(0.3)  # Simulate training time
        
        return ModelTrainingResult(
            id=f"mt_{request.id}",
            created_at=datetime.utcnow(),
            request_id=request.id,
            model_id=f"model_{request.dataset_id}_{datetime.utcnow().timestamp()}",
            status="trained",
            training_metrics={
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.92,
                "f1_score": 0.925
            },
            validation_metrics={
                "val_accuracy": 0.92,
                "val_precision": 0.90,
                "val_recall": 0.89,
                "val_f1_score": 0.895
            },
            model_artifacts={
                "model_path": f"/models/{request.model_type}_{request.dataset_id}.pkl",
                "metadata_path": f"/models/{request.model_type}_{request.dataset_id}_metadata.json"
            },
            training_time_ms=300,
            experiment_id=request.experiment_id
        )
    
    async def validate_request(self, request: ModelTrainingRequest) -> bool:
        return bool(request.model_type and request.dataset_id)
    
    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "MockModelTrainingService", "domain": "ai"}


class ProductionMLOpsWorkflow:
    """
    Production MLOps workflow orchestrator.
    
    This class demonstrates how to build complex workflows using
    the new interaction patterns while maintaining loose coupling
    between domains.
    """
    
    def __init__(self):
        # Get services via dependency injection
        container = get_container()
        self.data_quality_service = container.resolve(MockDataQualityService)
        self.anomaly_detection_service = container.resolve(MockAnomalyDetectionService)
        self.model_training_service = container.resolve(MockModelTrainingService)
        
        # Get event bus for communication
        self.event_bus = get_event_bus()
        
        # Workflow state
        self.workflow_results = {}
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for workflow coordination."""
        self.event_bus.subscribe(DataQualityCheckCompleted, self._on_quality_check_completed)
        self.event_bus.subscribe(AnomalyDetected, self._on_anomaly_detected)
        self.event_bus.subscribe(ModelTrainingCompleted, self._on_model_training_completed)
    
    async def run_complete_workflow(self, dataset_id: str) -> Dict[str, Any]:
        """
        Run a complete MLOps workflow for a dataset.
        
        Workflow steps:
        1. Assess data quality
        2. If quality is good, run anomaly detection
        3. If anomalies found, trigger model retraining
        4. Deploy and monitor new model
        """
        logger.info(f"Starting complete MLOps workflow for dataset: {dataset_id}")
        
        workflow_start = datetime.utcnow()
        workflow_id = f"workflow_{dataset_id}_{workflow_start.timestamp()}"
        
        # Initialize workflow state
        self.workflow_results[workflow_id] = {
            "dataset_id": dataset_id,
            "start_time": workflow_start,
            "status": "running",
            "steps_completed": [],
            "results": {}
        }
        
        try:
            # Step 1: Data Quality Assessment
            logger.info(f"Step 1: Assessing data quality for {dataset_id}")
            quality_result = await self._run_data_quality_assessment(dataset_id)
            self.workflow_results[workflow_id]["results"]["data_quality"] = quality_result
            self.workflow_results[workflow_id]["steps_completed"].append("data_quality")
            
            # Step 2: Anomaly Detection (if quality is good)
            if quality_result.overall_score >= 0.8:
                logger.info(f"Step 2: Running anomaly detection for {dataset_id}")
                detection_result = await self._run_anomaly_detection(dataset_id)
                self.workflow_results[workflow_id]["results"]["anomaly_detection"] = detection_result
                self.workflow_results[workflow_id]["steps_completed"].append("anomaly_detection")
                
                # Step 3: Model Training (if anomalies found)
                if detection_result.anomalies_count > 3:
                    logger.info(f"Step 3: Training new model for {dataset_id} (anomalies detected)")
                    training_result = await self._run_model_training(dataset_id, "anomaly_detector")
                    self.workflow_results[workflow_id]["results"]["model_training"] = training_result
                    self.workflow_results[workflow_id]["steps_completed"].append("model_training")
                    
                    # Step 4: Model Deployment (simulated)
                    logger.info(f"Step 4: Deploying model for {dataset_id}")
                    deployment_result = await self._simulate_model_deployment(training_result.model_id)
                    self.workflow_results[workflow_id]["results"]["deployment"] = deployment_result
                    self.workflow_results[workflow_id]["steps_completed"].append("deployment")
                else:
                    logger.info(f"No significant anomalies found in {dataset_id}, skipping retraining")
            else:
                logger.warning(f"Poor data quality for {dataset_id}, skipping anomaly detection")
            
            # Mark workflow as completed
            self.workflow_results[workflow_id]["status"] = "completed"
            self.workflow_results[workflow_id]["end_time"] = datetime.utcnow()
            self.workflow_results[workflow_id]["duration_ms"] = int(
                (self.workflow_results[workflow_id]["end_time"] - workflow_start).total_seconds() * 1000
            )
            
            logger.info(f"Workflow completed for {dataset_id}")
            return self.workflow_results[workflow_id]
            
        except Exception as e:
            logger.error(f"Workflow failed for {dataset_id}: {e}")
            self.workflow_results[workflow_id]["status"] = "failed"
            self.workflow_results[workflow_id]["error"] = str(e)
            raise
    
    async def _run_data_quality_assessment(self, dataset_id: str) -> DataQualityResult:
        """Run data quality assessment."""
        request = DataQualityRequest(
            id=f"dq_{dataset_id}_{datetime.utcnow().timestamp()}",
            created_at=datetime.utcnow(),
            dataset_id=dataset_id,
            quality_rules=["completeness", "uniqueness", "validity", "consistency"],
            threshold=0.8,
            include_profiling=True
        )
        
        result = await self.data_quality_service.execute(request)
        
        # Publish event
        event = DataQualityCheckCompleted(
            event_id="",
            event_type="",
            aggregate_id=dataset_id,
            occurred_at=datetime.utcnow(),
            dataset_id=dataset_id,
            status=result.status,
            overall_score=result.overall_score,
            issues_count=len(result.issues_found),
            quality_result=result
        )
        await publish_event(event)
        
        return result
    
    async def _run_anomaly_detection(self, dataset_id: str) -> DetectionResult:
        """Run anomaly detection."""
        request = DetectionRequest(
            id=f"ad_{dataset_id}_{datetime.utcnow().timestamp()}",
            created_at=datetime.utcnow(),
            dataset_id=dataset_id,
            algorithm="isolation_forest",
            parameters={
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42
            }
        )
        
        result = await self.anomaly_detection_service.execute(request)
        
        # Publish event if anomalies found
        if result.anomalies_count > 0:
            event = AnomalyDetected(
                event_id="",
                event_type="",
                aggregate_id=dataset_id,
                occurred_at=datetime.utcnow(),
                dataset_id=dataset_id,
                anomaly_count=result.anomalies_count,
                severity="high" if result.anomalies_count > 5 else "medium",
                detection_result=result
            )
            await publish_event(event)
        
        return result
    
    async def _run_model_training(self, dataset_id: str, model_type: str) -> ModelTrainingResult:
        """Run model training."""
        request = ModelTrainingRequest(
            id=f"mt_{dataset_id}_{datetime.utcnow().timestamp()}",
            created_at=datetime.utcnow(),
            model_type=model_type,
            dataset_id=dataset_id,
            training_config={
                "algorithm": "random_forest",
                "max_depth": 10,
                "n_estimators": 100
            },
            validation_split=0.2,
            hyperparameters={
                "learning_rate": 0.1,
                "regularization": 0.01
            },
            experiment_id=f"exp_{dataset_id}_{datetime.utcnow().date()}"
        )
        
        result = await self.model_training_service.execute(request)
        
        # Publish event
        event = ModelTrainingCompleted(
            event_id="",
            event_type="",
            aggregate_id=result.model_id,
            occurred_at=datetime.utcnow(),
            model_id=result.model_id,
            status=result.status,
            training_metrics=result.training_metrics,
            validation_metrics=result.validation_metrics,
            training_result=result
        )
        await publish_event(event)
        
        return result
    
    async def _simulate_model_deployment(self, model_id: str) -> Dict[str, Any]:
        """Simulate model deployment."""
        await asyncio.sleep(0.1)  # Simulate deployment time
        
        deployment_result = {
            "model_id": model_id,
            "endpoint_url": f"https://api.company.com/models/{model_id}/predict",
            "status": "deployed",
            "deployment_time": datetime.utcnow(),
            "health_check_url": f"https://api.company.com/models/{model_id}/health"
        }
        
        logger.info(f"Model {model_id} deployed successfully")
        return deployment_result
    
    # Event handlers for workflow coordination
    @event_handler(DataQualityCheckCompleted)
    async def _on_quality_check_completed(self, event: DataQualityCheckCompleted):
        """Handle data quality check completion."""
        logger.info(f"Quality check completed for {event.dataset_id}: {event.overall_score}")
    
    @event_handler(AnomalyDetected)
    async def _on_anomaly_detected(self, event: AnomalyDetected):
        """Handle anomaly detection events."""
        logger.warning(f"Anomalies detected in {event.dataset_id}: {event.anomaly_count} anomalies")
    
    @event_handler(ModelTrainingCompleted)
    async def _on_model_training_completed(self, event: ModelTrainingCompleted):
        """Handle model training completion."""
        logger.info(f"Model training completed: {event.model_id} with accuracy {event.training_metrics.get('accuracy', 'N/A')}")


async def configure_production_services():
    """Configure services for production workflow."""
    def setup_services(container):
        # Register mock services
        container.register_singleton(MockDataQualityService)
        container.register_singleton(MockAnomalyDetectionService) 
        container.register_singleton(MockModelTrainingService)
        
        # Register workflow orchestrator
        container.register_singleton(ProductionMLOpsWorkflow)
    
    configure_container(setup_services)


async def main():
    """Main function demonstrating the production workflow."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure services
    await configure_production_services()
    
    # Start event bus
    event_bus = get_event_bus()
    await event_bus.start()
    
    try:
        # Get workflow orchestrator
        container = get_container()
        workflow = container.resolve(ProductionMLOpsWorkflow)
        
        # Run workflows for different datasets
        datasets = [
            "dataset_clean_001",      # Good quality data
            "dataset_anomalous_002",  # Data with anomalies
            "dataset_poor_003"        # Poor quality data
        ]
        
        results = []
        for dataset_id in datasets:
            print(f"\n{'='*60}")
            print(f"Running workflow for: {dataset_id}")
            print(f"{'='*60}")
            
            try:
                result = await workflow.run_complete_workflow(dataset_id)
                results.append(result)
                
                print(f"✅ Workflow completed for {dataset_id}")
                print(f"   Steps: {', '.join(result['steps_completed'])}")
                print(f"   Duration: {result['duration_ms']}ms")
                
            except Exception as e:
                print(f"❌ Workflow failed for {dataset_id}: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print("WORKFLOW SUMMARY")
        print(f"{'='*60}")
        
        for result in results:
            status_emoji = "✅" if result["status"] == "completed" else "❌"
            print(f"{status_emoji} {result['dataset_id']}: {result['status']} ({len(result['steps_completed'])} steps)")
        
        # Show event bus metrics
        print(f"\n📊 Event Bus Metrics:")
        metrics = event_bus.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())