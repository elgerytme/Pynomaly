"""Service for automated model retraining workflows with comprehensive orchestration."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os

from pynomaly.domain.entities.model import Model
from pynomaly.domain.value_objects.model_metadata import ModelMetadata
from pynomaly.application.services.automl_service import AutoMLService
from pynomaly.application.services.drift_detection_service import DriftDetectionService
from pynomaly.infrastructure.monitoring.metrics_service import MetricsService
from pynomaly.infrastructure.monitoring.alerting_service import AlertingService
from pynomaly.infrastructure.persistence.repository import Repository
from packages.data_science.domain.value_objects.model_performance_metrics import ModelPerformanceMetrics


class RetrainingTrigger(str, Enum):
    """Triggers for automated model retraining."""
    
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED_RETRAINING = "scheduled_retraining"
    MANUAL_TRIGGER = "manual_trigger"
    BUSINESS_RULE = "business_rule"
    FEEDBACK_ACCUMULATION = "feedback_accumulation"


class RetrainingStatus(str, Enum):
    """Status of retraining workflow."""
    
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK = "rollback"


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining workflow."""
    
    # Data configuration
    max_training_data_age_days: int = 90
    min_training_samples: int = 1000
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Model configuration
    max_training_time_hours: int = 6
    early_stopping_patience: int = 10
    performance_improvement_threshold: float = 0.02  # 2% improvement required
    
    # Quality gates
    min_accuracy_threshold: float = 0.8
    min_f1_score_threshold: float = 0.75
    max_prediction_time_ms: float = 100.0
    
    # Rollback configuration
    enable_auto_rollback: bool = True
    validation_failure_threshold: int = 3
    
    # Business rules
    max_retraining_frequency_hours: int = 24
    cost_budget_limit: float = 1000.0
    
    # Notification settings
    notify_on_start: bool = True
    notify_on_completion: bool = True
    notify_on_failure: bool = True


@dataclass
class RetrainingMetrics:
    """Metrics collected during retraining workflow."""
    
    training_time_minutes: float = 0.0
    training_samples_count: int = 0
    validation_samples_count: int = 0
    
    # Performance improvements
    accuracy_improvement: float = 0.0
    f1_score_improvement: float = 0.0
    precision_improvement: float = 0.0
    recall_improvement: float = 0.0
    
    # Efficiency metrics
    prediction_time_improvement_ms: float = 0.0
    model_size_change_mb: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Cost metrics
    training_cost_usd: float = 0.0
    infrastructure_cost_usd: float = 0.0


@dataclass
class RetrainingWorkflow:
    """Represents a complete retraining workflow instance."""
    
    workflow_id: str
    model_id: str
    trigger: RetrainingTrigger
    status: RetrainingStatus
    config: RetrainingConfig
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    old_model_version: Optional[str] = None
    new_model_version: Optional[str] = None
    metrics: RetrainingMetrics = field(default_factory=RetrainingMetrics)
    
    # Validation results
    validation_passed: Optional[bool] = None
    validation_errors: List[str] = field(default_factory=list)
    
    # Rollback information
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary for persistence."""
        return {
            "workflow_id": self.workflow_id,
            "model_id": self.model_id,
            "trigger": self.trigger.value,
            "status": self.status.value,
            "config": self.config.__dict__,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "old_model_version": self.old_model_version,
            "new_model_version": self.new_model_version,
            "metrics": self.metrics.__dict__,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "rollback_triggered": self.rollback_triggered,
            "rollback_reason": self.rollback_reason
        }


class AutomatedRetrainingService:
    """Service for managing automated model retraining workflows.
    
    This service orchestrates the complete lifecycle of automated model retraining,
    from trigger detection to validation and deployment of new models.
    """

    def __init__(
        self,
        automl_service: AutoMLService,
        drift_service: DriftDetectionService,
        metrics_service: MetricsService,
        alerting_service: AlertingService,
        repository: Repository,
        storage_path: str = "/app/storage/retraining"
    ):
        self.automl_service = automl_service
        self.drift_service = drift_service
        self.metrics_service = metrics_service
        self.alerting_service = alerting_service
        self.repository = repository
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        
        # Active workflows tracking
        self.active_workflows: Dict[str, RetrainingWorkflow] = {}
        
        # Default retraining configurations by trigger type
        self.default_configs = {
            RetrainingTrigger.PERFORMANCE_DEGRADATION: RetrainingConfig(
                performance_improvement_threshold=0.05,
                max_training_time_hours=4
            ),
            RetrainingTrigger.DATA_DRIFT: RetrainingConfig(
                max_training_data_age_days=30,
                performance_improvement_threshold=0.03
            ),
            RetrainingTrigger.CONCEPT_DRIFT: RetrainingConfig(
                max_training_data_age_days=14,
                performance_improvement_threshold=0.02,
                max_training_time_hours=8
            ),
            RetrainingTrigger.SCHEDULED_RETRAINING: RetrainingConfig(
                max_training_data_age_days=60,
                performance_improvement_threshold=0.01
            )
        }
        
        # Initialize storage
        os.makedirs(storage_path, exist_ok=True)

    async def trigger_retraining(
        self,
        model_id: str,
        trigger: RetrainingTrigger,
        config: Optional[RetrainingConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Trigger automated retraining workflow for a model.
        
        Args:
            model_id: ID of the model to retrain
            trigger: Reason for triggering retraining
            config: Custom retraining configuration
            metadata: Additional metadata about the trigger
            
        Returns:
            Workflow ID for tracking the retraining process
        """
        try:
            # Check if retraining is allowed
            if not await self._can_trigger_retraining(model_id, trigger):
                raise ValueError(f"Retraining not allowed for model {model_id}")
            
            # Use default config if none provided
            if config is None:
                config = self.default_configs.get(trigger, RetrainingConfig())
            
            # Create workflow
            workflow_id = f"retrain_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            workflow = RetrainingWorkflow(
                workflow_id=workflow_id,
                model_id=model_id,
                trigger=trigger,
                status=RetrainingStatus.PENDING,
                config=config
            )
            
            self.active_workflows[workflow_id] = workflow
            
            # Start workflow asynchronously
            asyncio.create_task(self._execute_retraining_workflow(workflow, metadata))
            
            self.logger.info(f"Retraining workflow {workflow_id} triggered for model {model_id}")
            
            # Send notification if configured
            if config.notify_on_start:
                await self._send_workflow_notification(workflow, "started")
            
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Error triggering retraining for model {model_id}: {e}")
            raise

    async def _can_trigger_retraining(self, model_id: str, trigger: RetrainingTrigger) -> bool:
        """Check if retraining can be triggered for a model."""
        try:
            # Check if there's already an active retraining for this model
            active_count = sum(
                1 for workflow in self.active_workflows.values()
                if workflow.model_id == model_id and workflow.status in [
                    RetrainingStatus.PENDING, RetrainingStatus.RUNNING, RetrainingStatus.VALIDATING
                ]
            )
            
            if active_count > 0:
                self.logger.warning(f"Retraining already active for model {model_id}")
                return False
            
            # Check frequency limits
            recent_retrainings = await self._get_recent_retrainings(model_id, hours=24)
            if len(recent_retrainings) >= 3:  # Max 3 retrainings per 24 hours
                self.logger.warning(f"Too many recent retrainings for model {model_id}")
                return False
            
            # Check if model exists and is healthy
            model_metadata = await self.repository.get_model_metadata(model_id)
            if not model_metadata:
                self.logger.error(f"Model {model_id} not found")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking retraining eligibility: {e}")
            return False

    async def _execute_retraining_workflow(
        self,
        workflow: RetrainingWorkflow,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Execute the complete retraining workflow."""
        try:
            self.logger.info(f"Executing retraining workflow {workflow.workflow_id}")
            
            # Update status
            workflow.status = RetrainingStatus.RUNNING
            workflow.started_at = datetime.now()
            
            # Phase 1: Data preparation and validation
            training_data = await self._prepare_training_data(workflow)
            
            # Phase 2: Model training
            new_model = await self._train_new_model(workflow, training_data)
            
            # Phase 3: Model validation
            workflow.status = RetrainingStatus.VALIDATING
            validation_results = await self._validate_new_model(workflow, new_model)
            
            if validation_results["passed"]:
                # Phase 4: Model deployment
                await self._deploy_new_model(workflow, new_model)
                workflow.status = RetrainingStatus.COMPLETED
                workflow.validation_passed = True
            else:
                # Handle validation failure
                await self._handle_validation_failure(workflow, validation_results)
            
            workflow.completed_at = datetime.now()
            
            # Save workflow results
            await self._save_workflow_results(workflow)
            
            # Send completion notification
            if workflow.config.notify_on_completion:
                await self._send_workflow_notification(workflow, "completed")
            
        except Exception as e:
            self.logger.error(f"Error in retraining workflow {workflow.workflow_id}: {e}")
            workflow.status = RetrainingStatus.FAILED
            workflow.completed_at = datetime.now()
            
            if workflow.config.notify_on_failure:
                await self._send_workflow_notification(workflow, "failed", str(e))
            
        finally:
            # Clean up
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]

    async def _prepare_training_data(self, workflow: RetrainingWorkflow) -> Dict[str, Any]:
        """Prepare and validate training data for retraining."""
        self.logger.info(f"Preparing training data for workflow {workflow.workflow_id}")
        
        try:
            # Get recent data within age limit
            cutoff_date = datetime.now() - timedelta(days=workflow.config.max_training_data_age_days)
            
            # This would integrate with data pipeline services
            training_data = await self._fetch_training_data(
                model_id=workflow.model_id,
                since_date=cutoff_date,
                min_samples=workflow.config.min_training_samples
            )
            
            # Validate data quality
            data_quality = await self._validate_data_quality(training_data)
            if not data_quality["passed"]:
                raise ValueError(f"Data quality validation failed: {data_quality['errors']}")
            
            # Split data
            splits = await self._split_training_data(
                training_data,
                validation_split=workflow.config.validation_split,
                test_split=workflow.config.test_split
            )
            
            workflow.metrics.training_samples_count = len(splits["train"])
            workflow.metrics.validation_samples_count = len(splits["validation"])
            
            return splits
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise

    async def _train_new_model(self, workflow: RetrainingWorkflow, training_data: Dict[str, Any]) -> Model:
        """Train new model using AutoML service."""
        self.logger.info(f"Training new model for workflow {workflow.workflow_id}")
        
        start_time = datetime.now()
        
        try:
            # Get current model configuration
            current_model = await self.repository.get_model(workflow.model_id)
            if not current_model:
                raise ValueError(f"Current model {workflow.model_id} not found")
            
            workflow.old_model_version = current_model.metadata.version
            
            # Prepare training configuration
            training_config = {
                "max_time_hours": workflow.config.max_training_time_hours,
                "early_stopping_patience": workflow.config.early_stopping_patience,
                "validation_split": workflow.config.validation_split,
                "target_metric": "f1_score",  # Primary optimization metric
                "min_accuracy": workflow.config.min_accuracy_threshold
            }
            
            # Train new model
            new_model = await self.automl_service.train_model(
                data=training_data,
                config=training_config,
                base_model_id=workflow.model_id
            )
            
            # Calculate training time
            training_time = datetime.now() - start_time
            workflow.metrics.training_time_minutes = training_time.total_seconds() / 60
            
            workflow.new_model_version = new_model.metadata.version
            
            return new_model
            
        except Exception as e:
            self.logger.error(f"Error training new model: {e}")
            raise

    async def _validate_new_model(self, workflow: RetrainingWorkflow, new_model: Model) -> Dict[str, Any]:
        """Validate new model against quality gates and performance requirements."""
        self.logger.info(f"Validating new model for workflow {workflow.workflow_id}")
        
        try:
            validation_results = {
                "passed": True,
                "errors": [],
                "performance_comparison": {},
                "quality_gates": {}
            }
            
            # Get current model for comparison
            current_model = await self.repository.get_model(workflow.model_id)
            
            # Performance validation
            new_performance = await self._evaluate_model_performance(new_model)
            current_performance = await self._get_current_model_performance(workflow.model_id)
            
            # Check performance improvement
            performance_improvement = self._calculate_performance_improvement(
                current_performance, new_performance
            )
            
            # Update metrics
            workflow.metrics.accuracy_improvement = performance_improvement.get("accuracy", 0.0)
            workflow.metrics.f1_score_improvement = performance_improvement.get("f1_score", 0.0)
            workflow.metrics.precision_improvement = performance_improvement.get("precision", 0.0)
            workflow.metrics.recall_improvement = performance_improvement.get("recall", 0.0)
            
            # Quality gates validation
            quality_gates = await self._check_quality_gates(workflow, new_performance)
            validation_results["quality_gates"] = quality_gates
            
            if not quality_gates["passed"]:
                validation_results["passed"] = False
                validation_results["errors"].extend(quality_gates["errors"])
            
            # Performance improvement validation
            if workflow.metrics.f1_score_improvement < workflow.config.performance_improvement_threshold:
                validation_results["passed"] = False
                validation_results["errors"].append(
                    f"Insufficient performance improvement: {workflow.metrics.f1_score_improvement:.3f} < "
                    f"{workflow.config.performance_improvement_threshold:.3f}"
                )
            
            validation_results["performance_comparison"] = performance_improvement
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating new model: {e}")
            return {"passed": False, "errors": [str(e)]}

    async def _check_quality_gates(self, workflow: RetrainingWorkflow, performance: Dict[str, float]) -> Dict[str, Any]:
        """Check if new model passes quality gates."""
        quality_gates = {"passed": True, "errors": []}
        
        # Accuracy gate
        if performance.get("accuracy", 0.0) < workflow.config.min_accuracy_threshold:
            quality_gates["passed"] = False
            quality_gates["errors"].append(
                f"Accuracy below threshold: {performance.get('accuracy', 0.0):.3f} < "
                f"{workflow.config.min_accuracy_threshold:.3f}"
            )
        
        # F1 score gate
        if performance.get("f1_score", 0.0) < workflow.config.min_f1_score_threshold:
            quality_gates["passed"] = False
            quality_gates["errors"].append(
                f"F1 score below threshold: {performance.get('f1_score', 0.0):.3f} < "
                f"{workflow.config.min_f1_score_threshold:.3f}"
            )
        
        # Prediction time gate
        if performance.get("prediction_time_ms", float('inf')) > workflow.config.max_prediction_time_ms:
            quality_gates["passed"] = False
            quality_gates["errors"].append(
                f"Prediction time above threshold: {performance.get('prediction_time_ms', 0.0):.1f}ms > "
                f"{workflow.config.max_prediction_time_ms:.1f}ms"
            )
        
        return quality_gates

    async def _deploy_new_model(self, workflow: RetrainingWorkflow, new_model: Model):
        """Deploy the new model to production."""
        self.logger.info(f"Deploying new model for workflow {workflow.workflow_id}")
        
        try:
            # Create deployment plan
            deployment_plan = {
                "strategy": "blue_green",  # Safe deployment strategy
                "rollback_enabled": workflow.config.enable_auto_rollback,
                "health_check_timeout": 300,  # 5 minutes
                "validation_period": 1800     # 30 minutes
            }
            
            # Deploy through model service
            deployment_result = await self.automl_service.deploy_model(
                model=new_model,
                deployment_plan=deployment_plan
            )
            
            if deployment_result["success"]:
                # Update model registry
                await self.repository.update_active_model(workflow.model_id, new_model)
                self.logger.info(f"Successfully deployed new model version {workflow.new_model_version}")
            else:
                raise Exception(f"Deployment failed: {deployment_result['error']}")
            
        except Exception as e:
            self.logger.error(f"Error deploying new model: {e}")
            raise

    async def _handle_validation_failure(self, workflow: RetrainingWorkflow, validation_results: Dict[str, Any]):
        """Handle validation failure with appropriate actions."""
        workflow.validation_passed = False
        workflow.validation_errors = validation_results["errors"]
        
        if workflow.config.enable_auto_rollback:
            self.logger.info(f"Triggering rollback for workflow {workflow.workflow_id}")
            workflow.status = RetrainingStatus.ROLLBACK
            workflow.rollback_triggered = True
            workflow.rollback_reason = "Validation failure"
        else:
            workflow.status = RetrainingStatus.FAILED
        
        # Send validation failure alert
        await self.alerting_service.send_alert(
            title=f"Model Retraining Validation Failed - {workflow.model_id}",
            message=f"Validation failed for workflow {workflow.workflow_id}: {', '.join(workflow.validation_errors)}",
            severity="warning",
            tags={"model_id": workflow.model_id, "workflow_id": workflow.workflow_id}
        )

    # Helper methods for data operations
    async def _fetch_training_data(self, model_id: str, since_date: datetime, min_samples: int) -> Dict[str, Any]:
        """Fetch training data from data sources."""
        # This would integrate with actual data pipeline
        # For now, return mock data structure
        return {
            "features": [],  # Feature data
            "labels": [],    # Target labels
            "metadata": {
                "source": "production_feedback",
                "date_range": {"start": since_date.isoformat(), "end": datetime.now().isoformat()},
                "sample_count": min_samples
            }
        }

    async def _validate_data_quality(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training data quality."""
        # Implement actual data quality checks
        return {"passed": True, "errors": []}

    async def _split_training_data(self, data: Dict[str, Any], validation_split: float, test_split: float) -> Dict[str, Any]:
        """Split data into train/validation/test sets."""
        # Implement actual data splitting logic
        return {
            "train": data,
            "validation": data,
            "test": data
        }

    async def _evaluate_model_performance(self, model: Model) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        # This would run actual model evaluation
        return {
            "accuracy": 0.85,
            "f1_score": 0.83,
            "precision": 0.84,
            "recall": 0.82,
            "prediction_time_ms": 45.2
        }

    async def _get_current_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get current model performance metrics."""
        try:
            metrics = await self.metrics_service.get_model_metrics(model_id, timedelta(days=7))
            return metrics or {}
        except Exception:
            return {}

    def _calculate_performance_improvement(self, current: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvement between models."""
        improvements = {}
        for metric in ["accuracy", "f1_score", "precision", "recall"]:
            current_val = current.get(metric, 0.0)
            new_val = new.get(metric, 0.0)
            improvements[metric] = new_val - current_val
        return improvements

    async def _get_recent_retrainings(self, model_id: str, hours: int) -> List[Dict[str, Any]]:
        """Get recent retraining workflows for a model."""
        # This would query persistent storage
        return []

    async def _save_workflow_results(self, workflow: RetrainingWorkflow):
        """Save workflow results to persistent storage."""
        try:
            workflow_file = os.path.join(self.storage_path, f"{workflow.workflow_id}.json")
            with open(workflow_file, 'w') as f:
                json.dump(workflow.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving workflow results: {e}")

    async def _send_workflow_notification(self, workflow: RetrainingWorkflow, event: str, error: Optional[str] = None):
        """Send workflow status notification."""
        try:
            status_messages = {
                "started": f"Retraining workflow started for model {workflow.model_id}",
                "completed": f"Retraining completed successfully for model {workflow.model_id}",
                "failed": f"Retraining failed for model {workflow.model_id}: {error}"
            }
            
            await self.alerting_service.send_alert(
                title=f"Model Retraining {event.title()} - {workflow.model_id}",
                message=status_messages.get(event, f"Workflow {event}"),
                severity="info" if event != "failed" else "error",
                tags={"model_id": workflow.model_id, "workflow_id": workflow.workflow_id}
            )
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a retraining workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if workflow:
            return workflow.to_dict()
        
        # Check persistent storage for completed workflows
        workflow_file = os.path.join(self.storage_path, f"{workflow_id}.json")
        if os.path.exists(workflow_file):
            with open(workflow_file, 'r') as f:
                return json.load(f)
        
        return None

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active retraining workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if workflow and workflow.status in [RetrainingStatus.PENDING, RetrainingStatus.RUNNING]:
            workflow.status = RetrainingStatus.CANCELLED
            workflow.completed_at = datetime.now()
            return True
        return False