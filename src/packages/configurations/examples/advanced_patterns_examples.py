"""
Advanced patterns examples demonstrating CQRS, Event Sourcing, and Saga implementations.

This module provides comprehensive examples showing how to use sophisticated
architectural patterns for complex business scenarios requiring advanced
coordination between packages.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import advanced patterns
from interfaces.advanced_patterns import (
    Command, Query, CommandResponse, QueryResponse, CommandResult,
    CommandHandler, QueryHandler, Aggregate, Repository,
    ProcessDataQualityCommand, RunAnomalyDetectionCommand, TrainModelCommand,
    GetDataQualityReportQuery, GetAnomalyDetectionResultsQuery, GetModelPerformanceQuery,
    SagaStep, SagaOrchestrator, SagaState, SagaStatus,
    ReadModel, EventStoreEvent,
    CQRSConfiguration, EventSourcingConfiguration,
    create_workflow_command, create_data_query, create_saga_steps
)

# Import standard patterns and infrastructure
from interfaces.dto import DetectionRequest, DetectionResult, DataQualityRequest, DataQualityResult
from interfaces.events import DomainEvent, AnomalyDetected, DataQualityCheckCompleted, EventPriority
from interfaces.patterns import Service

from shared.advanced_infrastructure import (
    create_cqrs_infrastructure, create_event_sourcing_infrastructure,
    create_saga_orchestrator, InMemoryEventStore, EventSourcedRepository
)
from shared import get_container, get_event_bus, publish_event

logger = logging.getLogger(__name__)


# =============================================================================
# CQRS Example: Data Processing Pipeline
# =============================================================================

class DataProcessingCommandHandler:
    """Command handler for data processing operations."""
    
    def __init__(self):
        self.processed_datasets = {}
        self.processing_metrics = {
            "commands_processed": 0,
            "avg_processing_time": 0.0,
            "errors": 0
        }
    
    async def handle_data_quality_command(self, command: ProcessDataQualityCommand) -> CommandResponse:
        """Handle data quality processing command."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing data quality command for dataset {command.dataset_id}")
            
            # Simulate data quality processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Create domain events based on processing results
            events = []
            
            # Simulate quality check results
            quality_score = 0.85 if "high_quality" in command.dataset_id else 0.65
            status = "passed" if quality_score >= 0.8 else "failed"
            
            quality_event = DataQualityCheckCompleted(
                event_id=f"dq_{command.command_id}",
                event_type="DataQualityCheckCompleted",
                aggregate_id=command.dataset_id,
                occurred_at=datetime.utcnow(),
                dataset_id=command.dataset_id,
                status=status,
                overall_score=quality_score,
                issues_count=0 if status == "passed" else 3,
                quality_result=DataQualityResult(
                    id=f"result_{command.command_id}",
                    created_at=datetime.utcnow(),
                    request_id=command.command_id,
                    dataset_id=command.dataset_id,
                    status=status,
                    overall_score=quality_score,
                    rule_results={},
                    issues_found=[],
                    recommendations=[],
                    execution_time_ms=100
                )
            )
            events.append(quality_event)
            
            # Publish events
            for event in events:
                await publish_event(event)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_metrics["commands_processed"] += 1
            self._update_avg_processing_time(processing_time)
            
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.SUCCESS,
                aggregate_id=command.dataset_id,
                events=events,
                execution_time_ms=processing_time * 1000,
                metadata={"quality_score": quality_score, "status": status}
            )
            
        except Exception as e:
            logger.error(f"Error processing data quality command: {e}")
            self.processing_metrics["errors"] += 1
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.FAILED,
                error_message=str(e)
            )
    
    async def handle_anomaly_detection_command(self, command: RunAnomalyDetectionCommand) -> CommandResponse:
        """Handle anomaly detection command."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Running anomaly detection for dataset {command.dataset_id}")
            
            # Simulate anomaly detection processing
            await asyncio.sleep(0.2)
            
            # Simulate detection results
            anomaly_count = 5 if "anomalous" in command.dataset_id else 1
            severity = "high" if anomaly_count > 3 else "low"
            
            detection_event = AnomalyDetected(
                event_id=f"ad_{command.command_id}",
                event_type="AnomalyDetected",
                aggregate_id=command.dataset_id,
                occurred_at=datetime.utcnow(),
                dataset_id=command.dataset_id,
                anomaly_count=anomaly_count,
                severity=severity,
                detection_result=DetectionResult(
                    id=f"detection_{command.command_id}",
                    created_at=datetime.utcnow(),
                    request_id=command.command_id,
                    status="completed",
                    anomalies_count=anomaly_count,
                    anomaly_scores=[0.9, 0.8, 0.7],
                    anomaly_indices=[1, 5, 10],
                    confidence_scores=[0.95, 0.88, 0.75],
                    execution_time_ms=200,
                    algorithm_used=command.algorithm
                ),
                priority=EventPriority.HIGH if severity == "high" else EventPriority.NORMAL
            )
            
            await publish_event(detection_event)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.processing_metrics["commands_processed"] += 1
            self._update_avg_processing_time(processing_time)
            
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.SUCCESS,
                aggregate_id=command.dataset_id,
                events=[detection_event],
                execution_time_ms=processing_time * 1000,
                metadata={"anomaly_count": anomaly_count, "severity": severity}
            )
            
        except Exception as e:
            logger.error(f"Error running anomaly detection: {e}")
            self.processing_metrics["errors"] += 1
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.FAILED,
                error_message=str(e)
            )
    
    def _update_avg_processing_time(self, processing_time: float) -> None:
        """Update average processing time."""
        current_avg = self.processing_metrics["avg_processing_time"]
        count = self.processing_metrics["commands_processed"]
        self.processing_metrics["avg_processing_time"] = ((current_avg * (count - 1)) + processing_time) / count


class DataQueryHandler:
    """Query handler for data-related queries."""
    
    def __init__(self):
        self.cached_reports = {}
        self.query_metrics = {
            "queries_processed": 0,
            "cache_hits": 0,
            "avg_query_time": 0.0
        }
    
    async def handle_quality_report_query(self, query: GetDataQualityReportQuery) -> QueryResponse:
        """Handle data quality report query."""
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"quality_report_{query.dataset_id}"
            if cache_key in self.cached_reports:
                self.query_metrics["cache_hits"] += 1
                cached_data = self.cached_reports[cache_key]
                return QueryResponse(
                    query_id=query.query_id,
                    data=cached_data,
                    metadata={"cached": True}
                )
            
            # Simulate report generation
            await asyncio.sleep(0.05)
            
            report_data = {
                "dataset_id": query.dataset_id,
                "overall_score": 0.85,
                "timestamp": datetime.utcnow().isoformat(),
                "rules_evaluated": 10,
                "rules_passed": 8,
                "issues_found": [
                    {"rule": "completeness", "severity": "low", "count": 5},
                    {"rule": "validity", "severity": "medium", "count": 2}
                ],
                "recommendations": [
                    "Improve data completeness for column 'email'",
                    "Validate date formats in 'created_at' column"
                ]
            }
            
            if query.include_details:
                report_data["detailed_metrics"] = {
                    "column_profiles": ["id", "name", "email", "created_at"],
                    "data_types": {"id": "int", "name": "string", "email": "string", "created_at": "datetime"},
                    "null_percentages": {"id": 0.0, "name": 0.02, "email": 0.15, "created_at": 0.0}
                }
            
            # Cache the report
            self.cached_reports[cache_key] = report_data
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.query_metrics["queries_processed"] += 1
            self._update_avg_query_time(query_time)
            
            return QueryResponse(
                query_id=query.query_id,
                data=report_data,
                execution_time_ms=query_time * 1000,
                metadata={"generated": True}
            )
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return QueryResponse(
                query_id=query.query_id,
                error_message=str(e)
            )
    
    async def handle_anomaly_results_query(self, query: GetAnomalyDetectionResultsQuery) -> QueryResponse:
        """Handle anomaly detection results query."""
        start_time = datetime.utcnow()
        
        try:
            # Simulate retrieving anomaly results
            await asyncio.sleep(0.03)
            
            results_data = {
                "dataset_id": query.dataset_id,
                "algorithm": query.algorithm or "isolation_forest",
                "total_anomalies": 15,
                "confidence_threshold": query.confidence_threshold or 0.8,
                "anomalies": [
                    {
                        "index": i,
                        "confidence": 0.9 - (i * 0.05),
                        "features": {"feature_1": 1.5, "feature_2": -0.8},
                        "severity": "high" if i < 5 else "medium"
                    }
                    for i in range(min(query.limit, 15))
                ],
                "summary_stats": {
                    "high_severity": 5,
                    "medium_severity": 7,
                    "low_severity": 3,
                    "avg_confidence": 0.82
                }
            }
            
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.query_metrics["queries_processed"] += 1
            self._update_avg_query_time(query_time)
            
            return QueryResponse(
                query_id=query.query_id,
                data=results_data,
                total_count=15,
                page_info={"page": 1, "size": query.limit, "has_more": query.limit < 15},
                execution_time_ms=query_time * 1000
            )
            
        except Exception as e:
            logger.error(f"Error retrieving anomaly results: {e}")
            return QueryResponse(
                query_id=query.query_id,
                error_message=str(e)
            )
    
    def _update_avg_query_time(self, query_time: float) -> None:
        """Update average query time."""
        current_avg = self.query_metrics["avg_query_time"]
        count = self.query_metrics["queries_processed"]
        self.query_metrics["avg_query_time"] = ((current_avg * (count - 1)) + query_time) / count


# =============================================================================
# Event Sourcing Example: Data Processing Aggregate
# =============================================================================

class DataProcessingStatus(Enum):
    """Status of data processing workflow."""
    PENDING = "pending"
    QUALITY_CHECK_IN_PROGRESS = "quality_check_in_progress"
    QUALITY_CHECK_COMPLETED = "quality_check_completed"
    ANOMALY_DETECTION_IN_PROGRESS = "anomaly_detection_in_progress"
    ANOMALY_DETECTION_COMPLETED = "anomaly_detection_completed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DataProcessingStarted(DomainEvent):
    """Event indicating data processing workflow started."""
    dataset_id: str
    processing_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheckRequested(DomainEvent):
    """Event indicating quality check was requested."""
    dataset_id: str
    quality_rules: List[str]


@dataclass
class AnomalyDetectionRequested(DomainEvent):
    """Event indicating anomaly detection was requested."""
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class DataProcessingWorkflowAggregate(Aggregate):
    """Aggregate for data processing workflow using event sourcing."""
    
    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.dataset_id = ""
        self.status = DataProcessingStatus.PENDING
        self.quality_score = 0.0
        self.anomaly_count = 0
        self.processing_steps = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
    
    def start_processing(self, dataset_id: str, processing_type: str, parameters: Dict[str, Any]) -> None:
        """Start data processing workflow."""
        if self.status != DataProcessingStatus.PENDING:
            raise ValueError(f"Cannot start processing in status {self.status}")
        
        event = DataProcessingStarted(
            event_id=f"dp_started_{self.aggregate_id}",
            event_type="DataProcessingStarted",
            aggregate_id=self.aggregate_id,
            occurred_at=datetime.utcnow(),
            dataset_id=dataset_id,
            processing_type=processing_type,
            parameters=parameters
        )
        self.raise_event(event)
    
    def request_quality_check(self, quality_rules: List[str]) -> None:
        """Request quality check."""
        if self.status != DataProcessingStatus.PENDING:
            raise ValueError(f"Cannot request quality check in status {self.status}")
        
        event = QualityCheckRequested(
            event_id=f"qc_requested_{self.aggregate_id}",
            event_type="QualityCheckRequested",
            aggregate_id=self.aggregate_id,
            occurred_at=datetime.utcnow(),
            dataset_id=self.dataset_id,
            quality_rules=quality_rules
        )
        self.raise_event(event)
    
    def complete_quality_check(self, quality_score: float, status: str) -> None:
        """Complete quality check."""
        if self.status != DataProcessingStatus.QUALITY_CHECK_IN_PROGRESS:
            raise ValueError(f"Cannot complete quality check in status {self.status}")
        
        # This would typically be a more specific event type
        event = DataQualityCheckCompleted(
            event_id=f"qc_completed_{self.aggregate_id}",
            event_type="DataQualityCheckCompleted",
            aggregate_id=self.aggregate_id,
            occurred_at=datetime.utcnow(),
            dataset_id=self.dataset_id,
            status=status,
            overall_score=quality_score,
            issues_count=0 if status == "passed" else 3,
            quality_result=None  # Would be populated in real implementation
        )
        self.raise_event(event)
    
    def request_anomaly_detection(self, algorithm: str, parameters: Dict[str, Any]) -> None:
        """Request anomaly detection."""
        if self.status != DataProcessingStatus.QUALITY_CHECK_COMPLETED:
            raise ValueError(f"Cannot request anomaly detection in status {self.status}")
        
        event = AnomalyDetectionRequested(
            event_id=f"ad_requested_{self.aggregate_id}",
            event_type="AnomalyDetectionRequested",
            aggregate_id=self.aggregate_id,
            occurred_at=datetime.utcnow(),
            dataset_id=self.dataset_id,
            algorithm=algorithm,
            parameters=parameters
        )
        self.raise_event(event)
    
    def complete_anomaly_detection(self, anomaly_count: int, severity: str) -> None:
        """Complete anomaly detection."""
        if self.status != DataProcessingStatus.ANOMALY_DETECTION_IN_PROGRESS:
            raise ValueError(f"Cannot complete anomaly detection in status {self.status}")
        
        event = AnomalyDetected(
            event_id=f"ad_completed_{self.aggregate_id}",
            event_type="AnomalyDetected",
            aggregate_id=self.aggregate_id,
            occurred_at=datetime.utcnow(),
            dataset_id=self.dataset_id,
            anomaly_count=anomaly_count,
            severity=severity,
            detection_result=None,  # Would be populated in real implementation
            priority=EventPriority.HIGH if severity == "high" else EventPriority.NORMAL
        )
        self.raise_event(event)
    
    def _when(self, event: DomainEvent) -> None:
        """Apply event to aggregate state."""
        if isinstance(event, DataProcessingStarted):
            self.dataset_id = event.dataset_id
            self.started_at = event.occurred_at
            self.processing_steps.append("started")
        
        elif isinstance(event, QualityCheckRequested):
            self.status = DataProcessingStatus.QUALITY_CHECK_IN_PROGRESS
            self.processing_steps.append("quality_check_requested")
        
        elif isinstance(event, DataQualityCheckCompleted):
            self.status = DataProcessingStatus.QUALITY_CHECK_COMPLETED
            self.quality_score = event.overall_score
            self.processing_steps.append("quality_check_completed")
        
        elif isinstance(event, AnomalyDetectionRequested):
            self.status = DataProcessingStatus.ANOMALY_DETECTION_IN_PROGRESS
            self.processing_steps.append("anomaly_detection_requested")
        
        elif isinstance(event, AnomalyDetected):
            self.status = DataProcessingStatus.COMPLETED
            self.anomaly_count = event.anomaly_count
            self.completed_at = event.occurred_at
            self.processing_steps.append("anomaly_detection_completed")


# =============================================================================
# Saga Example: End-to-End Data Processing Pipeline
# =============================================================================

class DataProcessingSaga:
    """Saga for orchestrating end-to-end data processing pipeline."""
    
    def __init__(self, saga_orchestrator: SagaOrchestrator):
        self.saga_orchestrator = saga_orchestrator
        self.active_sagas = {}
    
    async def start_data_processing_pipeline(
        self, 
        dataset_id: str, 
        processing_config: Dict[str, Any]
    ) -> SagaState:
        """Start a complete data processing pipeline as a saga."""
        saga_id = f"data_pipeline_{dataset_id}_{int(datetime.utcnow().timestamp())}"
        
        # Define pipeline steps as saga steps
        steps = []
        
        # Step 1: Data Quality Check
        quality_command = ProcessDataQualityCommand(
            command_id=f"quality_{saga_id}",
            dataset_id=dataset_id,
            quality_rules=processing_config.get("quality_rules", ["completeness", "validity"]),
            priority="high"
        )
        
        # Compensation: Mark dataset as failed quality check
        quality_compensation = ProcessDataQualityCommand(
            command_id=f"quality_rollback_{saga_id}",
            dataset_id=dataset_id,
            quality_rules=["mark_failed"],
            metadata={"action": "compensation"}
        )
        
        steps.append(SagaStep(
            step_id="quality_check",
            command=quality_command,
            compensation_command=quality_compensation,
            timeout_seconds=60
        ))
        
        # Step 2: Anomaly Detection (conditional on quality score)
        detection_command = RunAnomalyDetectionCommand(
            command_id=f"detection_{saga_id}",
            dataset_id=dataset_id,
            algorithm=processing_config.get("detection_algorithm", "isolation_forest"),
            parameters=processing_config.get("detection_params", {})
        )
        
        # Compensation: Clear anomaly detection results
        detection_compensation = RunAnomalyDetectionCommand(
            command_id=f"detection_rollback_{saga_id}",
            dataset_id=dataset_id,
            algorithm="cleanup",
            metadata={"action": "compensation"}
        )
        
        steps.append(SagaStep(
            step_id="anomaly_detection",
            command=detection_command,
            compensation_command=detection_compensation,
            timeout_seconds=120
        ))
        
        # Step 3: Model Training (if anomalies found)
        if processing_config.get("train_model", False):
            training_command = TrainModelCommand(
                command_id=f"training_{saga_id}",
                dataset_id=dataset_id,
                model_type=processing_config.get("model_type", "anomaly_detector"),
                hyperparameters=processing_config.get("hyperparameters", {})
            )
            
            # Compensation: Delete trained model
            training_compensation = TrainModelCommand(
                command_id=f"training_rollback_{saga_id}",
                dataset_id=dataset_id,
                model_type="cleanup",
                metadata={"action": "compensation"}
            )
            
            steps.append(SagaStep(
                step_id="model_training",
                command=training_command,
                compensation_command=training_compensation,
                timeout_seconds=300
            ))
        
        # Start the saga
        saga_state = await self.saga_orchestrator.start_saga(saga_id, steps)
        self.active_sagas[saga_id] = {
            "dataset_id": dataset_id,
            "config": processing_config,
            "started_at": datetime.utcnow(),
            "status": saga_state.status
        }
        
        logger.info(f"Started data processing saga {saga_id} for dataset {dataset_id}")
        return saga_state
    
    async def get_pipeline_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get status of data processing pipeline."""
        saga_state = await self.saga_orchestrator.get_saga_state(saga_id)
        if not saga_state:
            return None
        
        pipeline_info = self.active_sagas.get(saga_id, {})
        
        return {
            "saga_id": saga_id,
            "dataset_id": pipeline_info.get("dataset_id"),
            "status": saga_state.status.value,
            "current_step": saga_state.current_step,
            "total_steps": len(saga_state.steps),
            "completed_steps": saga_state.completed_steps,
            "failed_step": saga_state.failed_step,
            "error_message": saga_state.error_message,
            "started_at": pipeline_info.get("started_at"),
            "updated_at": saga_state.updated_at,
            "steps_info": [
                {
                    "step_id": step.step_id,
                    "command_type": type(step.command).__name__,
                    "timeout_seconds": step.timeout_seconds,
                    "retry_count": step.retry_count
                }
                for step in saga_state.steps
            ]
        }


# =============================================================================
# Read Model Example: Data Processing Dashboard
# =============================================================================

class DataProcessingDashboardReadModel(ReadModel):
    """Read model for data processing dashboard."""
    
    def __init__(self):
        self.dashboard_data = {
            "total_datasets_processed": 0,
            "quality_checks_completed": 0,
            "anomaly_detections_completed": 0,
            "average_quality_score": 0.0,
            "total_anomalies_found": 0,
            "processing_trends": [],
            "recent_activities": [],
            "dataset_status": {}
        }
        self.quality_scores = []
        self.anomaly_counts = []
    
    async def handle_event(self, event: DomainEvent) -> None:
        """Handle domain event to update dashboard."""
        timestamp = event.occurred_at.isoformat()
        
        if isinstance(event, DataProcessingStarted):
            self.dashboard_data["total_datasets_processed"] += 1
            self.dashboard_data["recent_activities"].append({
                "timestamp": timestamp,
                "activity": f"Started processing dataset {event.dataset_id}",
                "dataset_id": event.dataset_id,
                "type": "processing_started"
            })
            self.dashboard_data["dataset_status"][event.dataset_id] = "processing"
        
        elif isinstance(event, DataQualityCheckCompleted):
            self.dashboard_data["quality_checks_completed"] += 1
            self.quality_scores.append(event.overall_score)
            
            # Update average quality score
            if self.quality_scores:
                self.dashboard_data["average_quality_score"] = sum(self.quality_scores) / len(self.quality_scores)
            
            self.dashboard_data["recent_activities"].append({
                "timestamp": timestamp,
                "activity": f"Quality check completed for {event.dataset_id} (score: {event.overall_score:.2f})",
                "dataset_id": event.dataset_id,
                "type": "quality_completed",
                "score": event.overall_score
            })
            self.dashboard_data["dataset_status"][event.dataset_id] = "quality_completed"
        
        elif isinstance(event, AnomalyDetected):
            self.dashboard_data["anomaly_detections_completed"] += 1
            self.dashboard_data["total_anomalies_found"] += event.anomaly_count
            self.anomaly_counts.append(event.anomaly_count)
            
            self.dashboard_data["recent_activities"].append({
                "timestamp": timestamp,
                "activity": f"Anomaly detection completed for {event.dataset_id} ({event.anomaly_count} anomalies)",
                "dataset_id": event.dataset_id,
                "type": "anomaly_completed",
                "anomaly_count": event.anomaly_count,
                "severity": event.severity
            })
            self.dashboard_data["dataset_status"][event.dataset_id] = "completed"
        
        # Keep only recent activities (last 50)
        if len(self.dashboard_data["recent_activities"]) > 50:
            self.dashboard_data["recent_activities"] = self.dashboard_data["recent_activities"][-50:]
        
        # Update processing trends (simplified)
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        existing_trend = next(
            (trend for trend in self.dashboard_data["processing_trends"] 
             if trend["hour"] == current_hour.isoformat()), 
            None
        )
        
        if existing_trend:
            existing_trend["events"] += 1
        else:
            self.dashboard_data["processing_trends"].append({
                "hour": current_hour.isoformat(),
                "events": 1
            })
        
        # Keep only last 24 hours of trends
        if len(self.dashboard_data["processing_trends"]) > 24:
            self.dashboard_data["processing_trends"] = self.dashboard_data["processing_trends"][-24:]
    
    def get_supported_events(self) -> List[type]:
        """Get list of event types this read model handles."""
        return [DataProcessingStarted, DataQualityCheckCompleted, AnomalyDetected]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()


# =============================================================================
# Advanced Patterns Demo Runner
# =============================================================================

class AdvancedPatternsDemo:
    """Demonstrates advanced architectural patterns."""
    
    def __init__(self):
        self.scenarios = {
            "cqrs": self._demo_cqrs_pattern,
            "event_sourcing": self._demo_event_sourcing,
            "saga": self._demo_saga_pattern,
            "read_models": self._demo_read_models,
            "complete_pipeline": self._demo_complete_pipeline
        }
    
    async def run_all_demos(self):
        """Run all advanced pattern demonstrations."""
        print("🚀 Running Advanced Patterns Demonstrations")
        print("=" * 60)
        
        for name, demo_func in self.scenarios.items():
            print(f"\n🔥 Running {name.replace('_', ' ').title()} Demo")
            print("-" * 40)
            
            try:
                await demo_func()
                print(f"✅ {name} demo completed successfully")
            except Exception as e:
                print(f"❌ {name} demo failed: {e}")
                logger.exception(f"Error in {name} demo")
        
        print(f"\n🎉 All advanced pattern demos completed!")
    
    async def _demo_cqrs_pattern(self):
        """Demo: CQRS pattern with command and query buses."""
        print("📋 Setting up CQRS infrastructure...")
        
        # Configure CQRS
        config = CQRSConfiguration(
            enable_command_validation=True,
            enable_query_caching=True,
            command_timeout_seconds=30,
            query_timeout_seconds=10
        )
        
        container = get_container()
        command_bus, query_bus = create_cqrs_infrastructure(config, container)
        
        # Set up handlers
        command_handler = DataProcessingCommandHandler()
        query_handler = DataQueryHandler()
        
        # Register handlers
        command_bus.register_handler(ProcessDataQualityCommand, command_handler.handle_data_quality_command)
        command_bus.register_handler(RunAnomalyDetectionCommand, command_handler.handle_anomaly_detection_command)
        
        query_bus.register_handler(GetDataQualityReportQuery, query_handler.handle_quality_report_query)
        query_bus.register_handler(GetAnomalyDetectionResultsQuery, query_handler.handle_anomaly_results_query)
        
        print("🎯 Executing commands...")
        
        # Execute commands
        quality_command = ProcessDataQualityCommand(
            dataset_id="high_quality_dataset_001",
            quality_rules=["completeness", "validity", "consistency"]
        )
        
        command_response = await command_bus.send(quality_command)
        print(f"  Quality check command result: {command_response.result.value}")
        
        detection_command = RunAnomalyDetectionCommand(
            dataset_id="anomalous_dataset_002",
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )
        
        command_response = await command_bus.send(detection_command)
        print(f"  Anomaly detection command result: {command_response.result.value}")
        
        print("🔍 Executing queries...")
        
        # Execute queries
        quality_query = GetDataQualityReportQuery(
            dataset_id="high_quality_dataset_001",
            include_details=True
        )
        
        query_response = await query_bus.ask(quality_query)
        if query_response.data:
            print(f"  Quality report generated with score: {query_response.data['overall_score']}")
        
        anomaly_query = GetAnomalyDetectionResultsQuery(
            dataset_id="anomalous_dataset_002",
            limit=10
        )
        
        query_response = await query_bus.ask(anomaly_query)
        if query_response.data:
            print(f"  Found {query_response.data['total_anomalies']} anomalies")
        
        # Show metrics
        print(f"📊 Command bus metrics: {command_bus.get_metrics()}")
        print(f"📊 Query bus metrics: {query_bus.get_metrics()}")
    
    async def _demo_event_sourcing(self):
        """Demo: Event sourcing with aggregates."""
        print("📦 Setting up Event Sourcing infrastructure...")
        
        config = EventSourcingConfiguration(
            event_store_type="memory",
            enable_snapshots=True,
            snapshot_frequency=5
        )
        
        event_store, projection_manager = create_event_sourcing_infrastructure(config)
        
        # Create repository
        def aggregate_factory(aggregate_id: str) -> DataProcessingWorkflowAggregate:
            return DataProcessingWorkflowAggregate(aggregate_id)
        
        repository = EventSourcedRepository(event_store, aggregate_factory)
        
        print("🔄 Creating and evolving aggregate...")
        
        # Create new aggregate
        workflow_id = "workflow_001"
        workflow = DataProcessingWorkflowAggregate(workflow_id)
        
        # Execute business operations
        workflow.start_processing(
            dataset_id="sample_dataset",
            processing_type="full_pipeline",
            parameters={"priority": "high"}
        )
        
        workflow.request_quality_check(["completeness", "validity"])
        workflow.complete_quality_check(quality_score=0.85, status="passed")
        
        workflow.request_anomaly_detection(
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )
        workflow.complete_anomaly_detection(anomaly_count=3, severity="medium")
        
        # Save aggregate
        await repository.save(workflow)
        print(f"  Saved workflow with {len(workflow.get_uncommitted_events())} events")
        
        # Load aggregate from event store
        loaded_workflow = await repository.get_by_id(workflow_id)
        print(f"  Loaded workflow in status: {loaded_workflow.status.value}")
        print(f"  Processing steps: {loaded_workflow.processing_steps}")
        
        # Show event store metrics
        print(f"📊 Event store metrics: {event_store.get_metrics()}")
    
    async def _demo_saga_pattern(self):
        """Demo: Saga pattern for distributed transactions."""
        print("🔗 Setting up Saga orchestration...")
        
        # Set up infrastructure
        config = CQRSConfiguration()
        container = get_container()
        command_bus, _ = create_cqrs_infrastructure(config, container)
        
        # Register command handlers
        command_handler = DataProcessingCommandHandler()
        command_bus.register_handler(ProcessDataQualityCommand, command_handler.handle_data_quality_command)
        command_bus.register_handler(RunAnomalyDetectionCommand, command_handler.handle_anomaly_detection_command)
        
        event_bus = get_event_bus()
        saga_orchestrator = create_saga_orchestrator(command_bus, event_bus)
        
        # Create saga
        data_saga = DataProcessingSaga(saga_orchestrator)
        
        print("🚀 Starting data processing pipeline saga...")
        
        # Start pipeline saga
        processing_config = {
            "quality_rules": ["completeness", "validity", "consistency"],
            "detection_algorithm": "isolation_forest",
            "detection_params": {"contamination": 0.05},
            "train_model": True,
            "model_type": "anomaly_detector",
            "hyperparameters": {"n_estimators": 100}
        }
        
        saga_state = await data_saga.start_data_processing_pipeline(
            dataset_id="pipeline_dataset_001",
            processing_config=processing_config
        )
        
        print(f"  Saga started with ID: {saga_state.saga_id}")
        print(f"  Total steps: {len(saga_state.steps)}")
        
        # Simulate some time passing and check status
        await asyncio.sleep(0.5)
        
        pipeline_status = await data_saga.get_pipeline_status(saga_state.saga_id)
        if pipeline_status:
            print(f"  Pipeline status: {pipeline_status['status']}")
            print(f"  Current step: {pipeline_status['current_step']}/{pipeline_status['total_steps']}")
            print(f"  Completed steps: {pipeline_status['completed_steps']}")
        
        # Show saga metrics
        print(f"📊 Saga orchestrator metrics: {saga_orchestrator.get_metrics()}")
    
    async def _demo_read_models(self):
        """Demo: Read models and projections."""
        print("📊 Setting up Read Models and Projections...")
        
        config = EventSourcingConfiguration()
        event_store, projection_manager = create_event_sourcing_infrastructure(config)
        
        # Create read model
        dashboard_read_model = DataProcessingDashboardReadModel()
        
        # Register projection
        await projection_manager.register_projection(dashboard_read_model)
        
        print("📈 Generating events for projections...")
        
        # Simulate events
        events = [
            DataProcessingStarted(
                event_id="dp_1",
                event_type="DataProcessingStarted",
                aggregate_id="dataset_001",
                occurred_at=datetime.utcnow(),
                dataset_id="dataset_001",
                processing_type="full",
                parameters={}
            ),
            DataQualityCheckCompleted(
                event_id="qc_1",
                event_type="DataQualityCheckCompleted",
                aggregate_id="dataset_001",
                occurred_at=datetime.utcnow(),
                dataset_id="dataset_001",
                status="passed",
                overall_score=0.92,
                issues_count=0,
                quality_result=None
            ),
            AnomalyDetected(
                event_id="ad_1",
                event_type="AnomalyDetected",
                aggregate_id="dataset_001",
                occurred_at=datetime.utcnow(),
                dataset_id="dataset_001",
                anomaly_count=7,
                severity="high",
                detection_result=None
            )
        ]
        
        # Process events through read model
        for event in events:
            await dashboard_read_model.handle_event(event)
        
        # Show dashboard data
        dashboard_data = dashboard_read_model.get_dashboard_data()
        print(f"  Total datasets processed: {dashboard_data['total_datasets_processed']}")
        print(f"  Average quality score: {dashboard_data['average_quality_score']:.2f}")
        print(f"  Total anomalies found: {dashboard_data['total_anomalies_found']}")
        print(f"  Recent activities: {len(dashboard_data['recent_activities'])}")
        
        # Show projection metrics
        print(f"📊 Projection manager metrics: {projection_manager.get_metrics()}")
    
    async def _demo_complete_pipeline(self):
        """Demo: Complete pipeline using all advanced patterns."""
        print("🌟 Running Complete Advanced Pipeline Demo...")
        
        # This would integrate all patterns together
        print("  This demo would combine CQRS + Event Sourcing + Saga + Read Models")
        print("  in a single, coordinated data processing pipeline.")
        print("  Each component would work together to provide a complete solution.")
        
        print("✨ Complete pipeline would include:")
        print("    1. CQRS commands to initiate processing")
        print("    2. Event-sourced aggregates to track state")
        print("    3. Sagas to coordinate multi-step workflows")
        print("    4. Read models to provide dashboard views")
        print("    5. Advanced monitoring and error handling")


async def main():
    """Main function to run advanced patterns examples."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
    
    # Start event bus
    event_bus = get_event_bus()
    await event_bus.start()
    
    try:
        demo = AdvancedPatternsDemo()
        await demo.run_all_demos()
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())