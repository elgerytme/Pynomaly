"""Machine Learning Pipeline entity for ML workflow management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity


class PipelineType(str, Enum):
    """Types of ML pipelines."""
    
    TRAINING = "training"
    INFERENCE = "inference"
    BATCH_INFERENCE = "batch_inference"
    REAL_TIME_INFERENCE = "real_time_inference"
    RETRAINING = "retraining"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_SELECTION = "model_selection"
    ENSEMBLE = "ensemble"
    AUTO_ML = "auto_ml"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    
    DRAFT = "draft"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class StepType(str, Enum):
    """Types of pipeline steps."""
    
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURE_SELECTION = "feature_selection"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_SPLITTING = "data_splitting"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_SELECTION = "model_selection"
    MODEL_DEPLOYMENT = "model_deployment"
    PREDICTION = "prediction"
    POST_PROCESSING = "post_processing"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class MachineLearningPipeline(BaseEntity):
    """Entity representing a complete ML workflow pipeline.
    
    This entity manages the definition, execution, and monitoring of
    machine learning pipelines from data ingestion to model deployment.
    
    Attributes:
        name: Human-readable name for the pipeline
        pipeline_type: Type of ML pipeline
        description: Detailed description of the pipeline
        status: Current pipeline status
        version_number: Version of the pipeline definition
        steps: Ordered list of pipeline steps
        dependencies: External dependencies required
        parameters: Pipeline-level parameters
        input_schema: Expected input data schema
        output_schema: Expected output data schema
        execution_environment: Environment requirements
        resource_requirements: Compute resource needs
        timeout_seconds: Maximum execution time
        retry_configuration: Retry policy for failed steps
        created_by: User who created the pipeline
        last_executed_at: Last execution timestamp
        execution_count: Number of times pipeline has been executed
        success_rate: Percentage of successful executions
        average_duration_seconds: Average execution time
        artifacts_location: Location of pipeline artifacts
        monitoring_config: Monitoring and alerting configuration
        validation_rules: Pipeline validation rules
        approval_required: Whether execution requires approval
        approved_by: User who approved the pipeline
        approved_at: Approval timestamp
        deployment_config: Deployment configuration
        rollback_config: Rollback strategy configuration
        performance_metrics: Pipeline performance metrics
        cost_tracking: Cost tracking information
        lineage_info: Data and model lineage information
        security_config: Security and access control settings
        compliance_info: Compliance and governance information
    """
    
    name: str = Field(..., min_length=1, max_length=255)
    pipeline_type: PipelineType
    description: Optional[str] = Field(None, max_length=2000)
    status: PipelineStatus = Field(default=PipelineStatus.DRAFT)
    version_number: str = Field(default="1.0.0", regex=r'^\d+\.\d+\.\d+$')
    
    # Pipeline definition
    steps: list[dict[str, Any]] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    
    # Data schemas
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    
    # Execution configuration
    execution_environment: dict[str, Any] = Field(default_factory=dict)
    resource_requirements: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(None, gt=0)
    retry_configuration: dict[str, Any] = Field(default_factory=dict)
    
    # Ownership and approval
    created_by: Optional[str] = None
    approval_required: bool = Field(default=False)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Execution tracking
    last_executed_at: Optional[datetime] = None
    execution_count: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0, le=100)
    average_duration_seconds: Optional[float] = Field(None, ge=0)
    
    # Artifacts and monitoring
    artifacts_location: Optional[str] = None
    monitoring_config: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, float] = Field(default_factory=dict)
    
    # Governance
    validation_rules: list[dict[str, Any]] = Field(default_factory=list)
    deployment_config: dict[str, Any] = Field(default_factory=dict)
    rollback_config: dict[str, Any] = Field(default_factory=dict)
    security_config: dict[str, Any] = Field(default_factory=dict)
    compliance_info: dict[str, Any] = Field(default_factory=dict)
    
    # Cost and lineage
    cost_tracking: dict[str, Any] = Field(default_factory=dict)
    lineage_info: dict[str, Any] = Field(default_factory=dict)
    
    @validator('steps')
    def validate_steps(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate pipeline steps."""
        if not v:
            raise ValueError("Pipeline must have at least one step")
            
        for i, step in enumerate(v):
            if "name" not in step:
                raise ValueError(f"Step {i} must have a name")
            if "type" not in step:
                raise ValueError(f"Step {i} must have a type")
                
            try:
                StepType(step["type"])
            except ValueError:
                raise ValueError(f"Step {i} has invalid type: {step['type']}")
                
        return v
    
    @validator('dependencies')
    def validate_dependencies(cls, v: list[str]) -> list[str]:
        """Validate pipeline dependencies."""
        return [dep.strip() for dep in v if dep.strip()]
    
    def add_step(self, name: str, step_type: StepType, 
                configuration: dict[str, Any], position: Optional[int] = None) -> None:
        """Add a step to the pipeline."""
        step = {
            "name": name,
            "type": step_type.value,
            "configuration": configuration,
            "created_at": datetime.utcnow().isoformat(),
            "id": f"step_{len(self.steps)}"
        }
        
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
            
        self._reindex_steps()
        self.mark_as_updated()
    
    def remove_step(self, step_name: str) -> bool:
        """Remove a step from the pipeline."""
        original_count = len(self.steps)
        self.steps = [step for step in self.steps if step["name"] != step_name]
        
        if len(self.steps) < original_count:
            self._reindex_steps()
            self.mark_as_updated()
            return True
            
        return False
    
    def update_step(self, step_name: str, configuration: dict[str, Any]) -> bool:
        """Update step configuration."""
        for step in self.steps:
            if step["name"] == step_name:
                step["configuration"].update(configuration)
                step["updated_at"] = datetime.utcnow().isoformat()
                self.mark_as_updated()
                return True
                
        return False
    
    def reorder_steps(self, step_order: list[str]) -> None:
        """Reorder pipeline steps."""
        if len(step_order) != len(self.steps):
            raise ValueError("Step order must include all current steps")
            
        step_map = {step["name"]: step for step in self.steps}
        missing_steps = set(step_order) - set(step_map.keys())
        
        if missing_steps:
            raise ValueError(f"Unknown steps in order: {missing_steps}")
            
        self.steps = [step_map[name] for name in step_order]
        self._reindex_steps()
        self.mark_as_updated()
    
    def validate_pipeline(self) -> dict[str, Any]:
        """Validate pipeline configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checked_at": datetime.utcnow().isoformat()
        }
        
        # Check step dependencies
        step_names = {step["name"] for step in self.steps}
        for step in self.steps:
            step_deps = step.get("dependencies", [])
            missing_deps = set(step_deps) - step_names
            if missing_deps:
                validation_result["errors"].append(
                    f"Step '{step['name']}' depends on missing steps: {missing_deps}"
                )
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            validation_result["errors"].append("Pipeline has circular dependencies")
        
        # Check resource requirements
        if not self.resource_requirements:
            validation_result["warnings"].append("No resource requirements specified")
        
        # Apply custom validation rules
        for rule in self.validation_rules:
            rule_result = self._apply_validation_rule(rule)
            if not rule_result["passed"]:
                validation_result["errors"].extend(rule_result["errors"])
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        if validation_result["valid"]:
            self.status = PipelineStatus.VALID
        else:
            self.status = PipelineStatus.INVALID
            
        self.mark_as_updated()
        return validation_result
    
    def submit_for_approval(self, submitted_by: str) -> None:
        """Submit pipeline for approval."""
        if self.status != PipelineStatus.VALID:
            raise ValueError("Pipeline must be valid before submission")
            
        if not self.approval_required:
            raise ValueError("Pipeline does not require approval")
            
        self.metadata["submitted_by"] = submitted_by
        self.metadata["submitted_at"] = datetime.utcnow().isoformat()
        self.mark_as_updated()
    
    def approve_pipeline(self, approved_by: str) -> None:
        """Approve pipeline for execution."""
        if not self.approval_required:
            raise ValueError("Pipeline does not require approval")
            
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()
        self.status = PipelineStatus.VALID
        self.mark_as_updated()
    
    def start_execution(self, execution_id: str) -> None:
        """Start pipeline execution."""
        if self.approval_required and not self.approved_by:
            raise ValueError("Pipeline requires approval before execution")
            
        if self.status != PipelineStatus.VALID:
            raise ValueError("Pipeline must be valid for execution")
            
        self.status = PipelineStatus.RUNNING
        self.last_executed_at = datetime.utcnow()
        self.metadata["current_execution_id"] = execution_id
        self.mark_as_updated()
    
    def complete_execution(self, success: bool, duration_seconds: float,
                          metrics: Optional[dict[str, float]] = None) -> None:
        """Complete pipeline execution."""
        if self.status != PipelineStatus.RUNNING:
            raise ValueError("Can only complete running pipelines")
            
        self.status = PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
        self.execution_count += 1
        
        # Update average duration
        if self.average_duration_seconds is None:
            self.average_duration_seconds = duration_seconds
        else:
            # Running average
            self.average_duration_seconds = (
                (self.average_duration_seconds * (self.execution_count - 1) + duration_seconds) /
                self.execution_count
            )
        
        # Update success rate
        if success:
            success_count = self.execution_count * (self.success_rate / 100) + 1
        else:
            success_count = self.execution_count * (self.success_rate / 100)
            
        self.success_rate = (success_count / self.execution_count) * 100
        
        # Update performance metrics
        if metrics:
            self.performance_metrics.update(metrics)
            
        self.mark_as_updated()
    
    def deploy_pipeline(self, deployment_target: str, 
                       deployment_config: dict[str, Any]) -> None:
        """Deploy pipeline to target environment."""
        if self.status != PipelineStatus.COMPLETED:
            raise ValueError("Pipeline must be completed before deployment")
            
        self.status = PipelineStatus.DEPLOYED
        self.deployment_config.update({
            "target": deployment_target,
            "deployed_at": datetime.utcnow().isoformat(),
            **deployment_config
        })
        self.mark_as_updated()
    
    def _reindex_steps(self) -> None:
        """Reindex step IDs after modifications."""
        for i, step in enumerate(self.steps):
            step["id"] = f"step_{i}"
            step["position"] = i
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in steps."""
        # Simplified circular dependency check
        # In practice, this would use a proper graph algorithm
        step_deps = {}
        for step in self.steps:
            step_deps[step["name"]] = step.get("dependencies", [])
        
        # Basic check for immediate circular refs
        for step_name, deps in step_deps.items():
            if step_name in deps:
                return True
                
        return False
    
    def _apply_validation_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Apply a custom validation rule."""
        # Simplified validation rule application
        # In practice, this would evaluate complex rule expressions
        return {
            "passed": True,
            "errors": [],
            "rule_name": rule.get("name", "unknown")
        }
    
    def get_step_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Get step by name."""
        for step in self.steps:
            if step["name"] == name:
                return step
        return None
    
    def is_executable(self) -> bool:
        """Check if pipeline can be executed."""
        return (
            self.status == PipelineStatus.VALID and
            len(self.steps) > 0 and
            (not self.approval_required or self.approved_by is not None)
        )
    
    def is_deployable(self) -> bool:
        """Check if pipeline can be deployed."""
        return (
            self.status == PipelineStatus.COMPLETED and
            self.success_rate > 0
        )
    
    def get_execution_summary(self) -> dict[str, Any]:
        """Get pipeline execution summary."""
        return {
            "pipeline_id": str(self.id),
            "name": self.name,
            "type": self.pipeline_type.value,
            "status": self.status.value,
            "version": self.version_number,
            "step_count": len(self.steps),
            "execution_count": self.execution_count,
            "success_rate": self.success_rate,
            "average_duration": self.average_duration_seconds,
            "last_executed": self.last_executed_at.isoformat() if self.last_executed_at else None,
            "is_executable": self.is_executable(),
            "is_deployable": self.is_deployable(),
            "requires_approval": self.approval_required,
            "is_approved": self.approved_by is not None,
        }
    
    def validate_invariants(self) -> None:
        """Validate domain invariants."""
        super().validate_invariants()
        
        # Business rule: Deployed pipelines must have deployment config
        if self.status == PipelineStatus.DEPLOYED and not self.deployment_config:
            raise ValueError("Deployed pipelines must have deployment configuration")
        
        # Business rule: Approved pipelines must have approver
        if self.approved_at and not self.approved_by:
            raise ValueError("Approved pipelines must have approver information")
        
        # Business rule: Running pipelines must have current execution ID
        if self.status == PipelineStatus.RUNNING:
            if "current_execution_id" not in self.metadata:
                raise ValueError("Running pipelines must have execution ID")
        
        # Business rule: Success rate must be consistent with execution count
        if self.execution_count > 0 and not 0 <= self.success_rate <= 100:
            raise ValueError("Success rate must be between 0 and 100")