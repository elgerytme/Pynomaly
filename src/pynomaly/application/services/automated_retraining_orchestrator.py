"""
Automated Model Retraining Orchestration Service

This service implements Issue #9 (A-001) by creating comprehensive automated model retraining workflows
that coordinate existing retraining services and performance monitoring components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from pynomaly.application.services.automated_retraining_service import (
    AutomatedRetrainingService,
    RetrainingTrigger,
    RetrainingConfig,
    RetrainingRequest,
    RetrainingResult
)
from pynomaly.application.services.model_performance_degradation_service import (
    ModelPerformanceDegradationService,
    DegradationSeverity
)
from pynomaly.application.services.intelligent_retraining_service import (
    IntelligentRetrainingService,
    RetrainingDecision
)
from packages.data_science.application.services.performance_degradation_monitoring_service import (
    PerformanceDegradationMonitoringService
)
from packages.mlops.pynomaly_mlops.application.services.pipeline_orchestration_service import (
    PipelineOrchestrationService
)


class WorkflowType(str, Enum):
    """Types of automated retraining workflows"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"
    DATA_DRIFT_RESPONSE = "data_drift_response"
    CONCEPT_DRIFT_RESPONSE = "concept_drift_response"
    BUSINESS_RULE_TRIGGERED = "business_rule_triggered"
    FEEDBACK_ACCUMULATION = "feedback_accumulation"
    MULTI_MODEL_COORDINATION = "multi_model_coordination"


class WorkflowStatus(str, Enum):
    """Status of workflow execution"""
    IDLE = "idle"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    EXECUTING = "executing"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class WorkflowConfig:
    """Configuration for automated retraining workflows"""
    workflow_type: WorkflowType
    model_ids: List[str]
    monitoring_interval_minutes: int = 15
    
    # Performance degradation thresholds
    accuracy_threshold: float = 0.05
    f1_score_threshold: float = 0.03
    precision_threshold: float = 0.03
    recall_threshold: float = 0.03
    
    # Data drift thresholds
    drift_threshold: float = 0.1
    concept_drift_threshold: float = 0.15
    
    # Scheduling configuration
    schedule_cron: Optional[str] = None
    max_concurrent_retraining: int = 2
    
    # Validation requirements
    validation_dataset_size: int = 1000
    champion_challenger_duration_hours: int = 24
    
    # Rollback configuration
    auto_rollback_enabled: bool = True
    rollback_threshold: float = 0.02
    
    # Business rules
    business_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance"""
    workflow_id: str
    workflow_type: WorkflowType
    model_ids: List[str]
    trigger: RetrainingTrigger
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    retraining_results: List[RetrainingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomatedRetrainingOrchestrator:
    """
    Orchestrates automated model retraining workflows by coordinating existing services
    and implementing comprehensive automation scenarios.
    """
    
    def __init__(
        self,
        retraining_service: AutomatedRetrainingService,
        performance_degradation_service: ModelPerformanceDegradationService,
        intelligent_retraining_service: IntelligentRetrainingService,
        monitoring_service: PerformanceDegradationMonitoringService,
        pipeline_orchestration_service: PipelineOrchestrationService
    ):
        self.retraining_service = retraining_service
        self.performance_degradation_service = performance_degradation_service
        self.intelligent_retraining_service = intelligent_retraining_service
        self.monitoring_service = monitoring_service
        self.pipeline_orchestration_service = pipeline_orchestration_service
        
        self.logger = logging.getLogger(__name__)
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowConfig] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Concurrency control
        self.retraining_semaphore = asyncio.Semaphore(3)
        self.execution_lock = asyncio.Lock()
    
    async def register_workflow(
        self,
        workflow_id: str,
        config: WorkflowConfig
    ) -> bool:
        """Register a new automated retraining workflow"""
        try:
            async with self.execution_lock:
                if workflow_id in self.active_workflows:
                    self.logger.warning(f"Workflow {workflow_id} already registered")
                    return False
                
                self.active_workflows[workflow_id] = config
                
                # Start monitoring task for this workflow
                if config.workflow_type in [
                    WorkflowType.PERFORMANCE_DEGRADATION,
                    WorkflowType.DATA_DRIFT_RESPONSE,
                    WorkflowType.CONCEPT_DRIFT_RESPONSE
                ]:
                    task = asyncio.create_task(
                        self._monitor_workflow(workflow_id, config)
                    )
                    self.monitoring_tasks[workflow_id] = task
                
                # Schedule periodic retraining if configured
                if config.schedule_cron:
                    await self._schedule_workflow(workflow_id, config)
                
                self.logger.info(f"Registered workflow {workflow_id} of type {config.workflow_type}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register workflow {workflow_id}: {str(e)}")
            return False
    
    async def _monitor_workflow(self, workflow_id: str, config: WorkflowConfig):
        """Continuously monitor conditions for workflow triggers"""
        while workflow_id in self.active_workflows:
            try:
                # Check each model in the workflow
                for model_id in config.model_ids:
                    should_trigger = await self._check_trigger_conditions(
                        model_id, config
                    )
                    
                    if should_trigger:
                        await self._trigger_workflow(workflow_id, model_id, config)
                
                # Wait before next check
                await asyncio.sleep(config.monitoring_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error monitoring workflow {workflow_id}: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_trigger_conditions(
        self,
        model_id: str,
        config: WorkflowConfig
    ) -> bool:
        """Check if conditions are met to trigger retraining"""
        try:
            if config.workflow_type == WorkflowType.PERFORMANCE_DEGRADATION:
                # Check performance degradation
                degradation_info = await self.performance_degradation_service.check_degradation(
                    model_id
                )
                
                if degradation_info.severity in [
                    DegradationSeverity.MODERATE,
                    DegradationSeverity.SEVERE,
                    DegradationSeverity.CRITICAL
                ]:
                    return True
            
            elif config.workflow_type == WorkflowType.DATA_DRIFT_RESPONSE:
                # Check data drift
                drift_metrics = await self.monitoring_service.check_data_drift(model_id)
                return drift_metrics.drift_score > config.drift_threshold
            
            elif config.workflow_type == WorkflowType.CONCEPT_DRIFT_RESPONSE:
                # Check concept drift
                concept_drift = await self.monitoring_service.check_concept_drift(model_id)
                return concept_drift.drift_score > config.concept_drift_threshold
            
            elif config.workflow_type == WorkflowType.BUSINESS_RULE_TRIGGERED:
                # Check business rules
                return await self._evaluate_business_rules(model_id, config.business_rules)
            
            elif config.workflow_type == WorkflowType.FEEDBACK_ACCUMULATION:
                # Check feedback accumulation
                feedback_count = await self._get_feedback_count(model_id)
                return feedback_count >= config.business_rules.get("feedback_threshold", 100)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trigger conditions for {model_id}: {str(e)}")
            return False
    
    async def _trigger_workflow(
        self,
        workflow_id: str,
        model_id: str,
        config: WorkflowConfig
    ):
        """Trigger workflow execution for a specific model"""
        try:
            # Check if we're already executing for this workflow
            if workflow_id in self.workflow_executions:
                current_execution = self.workflow_executions[workflow_id]
                if current_execution.status in [
                    WorkflowStatus.EXECUTING,
                    WorkflowStatus.VALIDATING,
                    WorkflowStatus.DEPLOYING
                ]:
                    self.logger.info(f"Workflow {workflow_id} already executing")
                    return
            
            # Create workflow execution
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                workflow_type=config.workflow_type,
                model_ids=[model_id],  # Single model for this trigger
                trigger=self._map_workflow_to_trigger(config.workflow_type),
                status=WorkflowStatus.TRIGGERED,
                started_at=datetime.utcnow()
            )
            
            self.workflow_executions[workflow_id] = execution
            
            # Execute workflow asynchronously
            asyncio.create_task(self._execute_workflow(execution, config))
            
        except Exception as e:
            self.logger.error(f"Error triggering workflow {workflow_id}: {str(e)}")
    
    async def _execute_workflow(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ):
        """Execute the complete workflow"""
        async with self.retraining_semaphore:
            try:
                execution.status = WorkflowStatus.EXECUTING
                self.logger.info(f"Starting workflow execution {execution.workflow_id}")
                
                # Step 1: Intelligent decision making
                decision = await self._make_retraining_decision(execution, config)
                if not decision.should_retrain:
                    execution.status = WorkflowStatus.COMPLETED
                    execution.completed_at = datetime.utcnow()
                    self.logger.info(f"Workflow {execution.workflow_id} completed - no retraining needed")
                    return
                
                # Step 2: Execute retraining for each model
                for model_id in execution.model_ids:
                    retraining_result = await self._execute_model_retraining(
                        model_id, execution, config
                    )
                    execution.retraining_results.append(retraining_result)
                
                # Step 3: Validation phase
                execution.status = WorkflowStatus.VALIDATING
                validation_success = await self._validate_retraining_results(execution, config)
                
                if validation_success:
                    # Step 4: Deployment phase
                    execution.status = WorkflowStatus.DEPLOYING
                    deployment_success = await self._deploy_retrained_models(execution, config)
                    
                    if deployment_success:
                        execution.status = WorkflowStatus.COMPLETED
                        self.logger.info(f"Workflow {execution.workflow_id} completed successfully")
                    else:
                        await self._handle_deployment_failure(execution, config)
                else:
                    await self._handle_validation_failure(execution, config)
                
                execution.completed_at = datetime.utcnow()
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.utcnow()
                self.logger.error(f"Workflow {execution.workflow_id} failed: {str(e)}")
                
                # Attempt rollback if configured
                if config.auto_rollback_enabled:
                    await self._rollback_workflow(execution, config)
    
    async def _make_retraining_decision(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ) -> RetrainingDecision:
        """Use intelligent service to make retraining decisions"""
        try:
            # Aggregate information for decision making
            decision_context = {
                "workflow_type": execution.workflow_type,
                "model_ids": execution.model_ids,
                "trigger": execution.trigger,
                "config": config
            }
            
            decision = await self.intelligent_retraining_service.make_retraining_decision(
                execution.model_ids[0],  # Primary model
                decision_context
            )
            
            execution.metadata["retraining_decision"] = decision
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making retraining decision: {str(e)}")
            # Default to proceeding with retraining
            return RetrainingDecision(
                should_retrain=True,
                confidence=0.5,
                reasoning="Fallback decision due to error"
            )
    
    async def _execute_model_retraining(
        self,
        model_id: str,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ) -> RetrainingResult:
        """Execute retraining for a specific model"""
        try:
            # Create retraining request
            retraining_config = RetrainingConfig(
                max_training_data_age_days=config.business_rules.get("max_data_age", 90),
                min_training_samples=config.business_rules.get("min_samples", 1000),
                max_training_time_hours=config.business_rules.get("max_training_time", 6),
                performance_improvement_threshold=config.business_rules.get("improvement_threshold", 0.02),
                min_accuracy_threshold=config.accuracy_threshold,
                min_f1_score_threshold=config.f1_score_threshold,
                enable_auto_rollback=config.auto_rollback_enabled
            )
            
            request = RetrainingRequest(
                model_id=model_id,
                trigger=execution.trigger,
                config=retraining_config,
                metadata=execution.metadata
            )
            
            # Execute retraining
            result = await self.retraining_service.execute_retraining(request)
            
            self.logger.info(f"Retraining completed for model {model_id} with status {result.status}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error retraining model {model_id}: {str(e)}")
            raise
    
    async def _validate_retraining_results(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ) -> bool:
        """Validate the results of retraining"""
        try:
            all_valid = True
            
            for result in execution.retraining_results:
                if result.status != "completed":
                    all_valid = False
                    continue
                
                # Validate performance improvements
                if result.performance_metrics:
                    baseline_accuracy = result.performance_metrics.get("baseline_accuracy", 0)
                    new_accuracy = result.performance_metrics.get("new_accuracy", 0)
                    
                    if new_accuracy <= baseline_accuracy + config.rollback_threshold:
                        self.logger.warning(
                            f"Model {result.model_id} did not achieve sufficient improvement"
                        )
                        all_valid = False
                
                # Champion/challenger validation if configured
                if config.champion_challenger_duration_hours > 0:
                    challenger_result = await self._run_champion_challenger_test(
                        result, config.champion_challenger_duration_hours
                    )
                    if not challenger_result:
                        all_valid = False
            
            return all_valid
            
        except Exception as e:
            self.logger.error(f"Error validating retraining results: {str(e)}")
            return False
    
    async def _deploy_retrained_models(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ) -> bool:
        """Deploy successfully retrained models"""
        try:
            deployment_success = True
            
            for result in execution.retraining_results:
                if result.status == "completed":
                    # Deploy the model
                    deployment_result = await self.pipeline_orchestration_service.deploy_model(
                        result.model_id,
                        result.model_version,
                        metadata=result.metadata
                    )
                    
                    if not deployment_result.success:
                        deployment_success = False
                        self.logger.error(f"Failed to deploy model {result.model_id}")
            
            return deployment_success
            
        except Exception as e:
            self.logger.error(f"Error deploying retrained models: {str(e)}")
            return False
    
    async def _handle_validation_failure(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ):
        """Handle validation failure"""
        execution.status = WorkflowStatus.FAILED
        self.logger.error(f"Validation failed for workflow {execution.workflow_id}")
        
        # Optionally trigger rollback
        if config.auto_rollback_enabled:
            await self._rollback_workflow(execution, config)
    
    async def _handle_deployment_failure(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ):
        """Handle deployment failure"""
        execution.status = WorkflowStatus.FAILED
        self.logger.error(f"Deployment failed for workflow {execution.workflow_id}")
        
        # Trigger rollback
        if config.auto_rollback_enabled:
            await self._rollback_workflow(execution, config)
    
    async def _rollback_workflow(
        self,
        execution: WorkflowExecution,
        config: WorkflowConfig
    ):
        """Rollback workflow changes"""
        try:
            execution.status = WorkflowStatus.ROLLED_BACK
            
            for result in execution.retraining_results:
                if result.rollback_model_id:
                    await self.pipeline_orchestration_service.rollback_model(
                        result.model_id,
                        result.rollback_model_id
                    )
            
            self.logger.info(f"Workflow {execution.workflow_id} rolled back successfully")
            
        except Exception as e:
            self.logger.error(f"Error rolling back workflow {execution.workflow_id}: {str(e)}")
    
    def _map_workflow_to_trigger(self, workflow_type: WorkflowType) -> RetrainingTrigger:
        """Map workflow type to retraining trigger"""
        mapping = {
            WorkflowType.PERFORMANCE_DEGRADATION: RetrainingTrigger.PERFORMANCE_DEGRADATION,
            WorkflowType.DATA_DRIFT_RESPONSE: RetrainingTrigger.DATA_DRIFT,
            WorkflowType.CONCEPT_DRIFT_RESPONSE: RetrainingTrigger.CONCEPT_DRIFT,
            WorkflowType.SCHEDULED_MAINTENANCE: RetrainingTrigger.SCHEDULED_RETRAINING,
            WorkflowType.BUSINESS_RULE_TRIGGERED: RetrainingTrigger.BUSINESS_RULE,
            WorkflowType.FEEDBACK_ACCUMULATION: RetrainingTrigger.FEEDBACK_ACCUMULATION
        }
        return mapping.get(workflow_type, RetrainingTrigger.MANUAL_TRIGGER)
    
    async def _schedule_workflow(self, workflow_id: str, config: WorkflowConfig):
        """Schedule periodic workflow execution"""
        # Implementation would integrate with scheduling system
        pass
    
    async def _evaluate_business_rules(
        self,
        model_id: str,
        business_rules: Dict[str, Any]
    ) -> bool:
        """Evaluate business rules for triggering"""
        # Implementation would evaluate custom business rules
        return False
    
    async def _get_feedback_count(self, model_id: str) -> int:
        """Get accumulated feedback count for model"""
        # Implementation would check feedback accumulation
        return 0
    
    async def _run_champion_challenger_test(
        self,
        result: RetrainingResult,
        duration_hours: int
    ) -> bool:
        """Run champion/challenger test"""
        # Implementation would run A/B test between models
        return True
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get the status of a workflow execution"""
        return self.workflow_executions.get(workflow_id)
    
    async def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs"""
        return list(self.active_workflows.keys())
    
    async def stop_workflow(self, workflow_id: str) -> bool:
        """Stop a running workflow"""
        try:
            if workflow_id in self.monitoring_tasks:
                self.monitoring_tasks[workflow_id].cancel()
                del self.monitoring_tasks[workflow_id]
            
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping workflow {workflow_id}: {str(e)}")
            return False
    
    @asynccontextmanager
    async def workflow_context(self, workflow_id: str, config: WorkflowConfig):
        """Context manager for workflow lifecycle"""
        try:
            await self.register_workflow(workflow_id, config)
            yield
        finally:
            await self.stop_workflow(workflow_id)


# Factory function for creating orchestrator instances
def create_automated_retraining_orchestrator(
    retraining_service: AutomatedRetrainingService,
    performance_degradation_service: ModelPerformanceDegradationService,
    intelligent_retraining_service: IntelligentRetrainingService,
    monitoring_service: PerformanceDegradationMonitoringService,
    pipeline_orchestration_service: PipelineOrchestrationService
) -> AutomatedRetrainingOrchestrator:
    """Create a configured orchestrator instance"""
    return AutomatedRetrainingOrchestrator(
        retraining_service=retraining_service,
        performance_degradation_service=performance_degradation_service,
        intelligent_retraining_service=intelligent_retraining_service,
        monitoring_service=monitoring_service,
        pipeline_orchestration_service=pipeline_orchestration_service
    )