"""
MLOps domain entities for enterprise machine learning operations.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


class ExperimentStatus(str, Enum):
    """Experiment status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"


class DeploymentStatus(str, Enum):
    """Model deployment status."""
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SCALING = "scaling"


class PipelineStatus(str, Enum):
    """ML pipeline status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"


class MLExperiment(BaseModel):
    """
    Machine Learning experiment tracking.
    
    Represents an ML experiment with parameters, metrics,
    artifacts, and lifecycle management.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Experiment identifier")
    
    # Experiment identification
    name: str = Field(..., description="Experiment name")
    project_name: str = Field(..., description="Project name")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Experiment configuration
    description: Optional[str] = Field(None, description="Experiment description")
    tags: Dict[str, str] = Field(default_factory=dict, description="Experiment tags")
    
    # Parameters and hyperparameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Experiment parameters")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    
    # Metrics and results
    metrics: Dict[str, float] = Field(default_factory=dict, description="Experiment metrics")
    best_metrics: Dict[str, float] = Field(default_factory=dict, description="Best metrics achieved")
    
    # Model and artifacts
    model_artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Model artifacts")
    data_artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Data artifacts")
    code_version: Optional[str] = Field(None, description="Code version/commit hash")
    
    # Execution details
    status: ExperimentStatus = Field(default=ExperimentStatus.RUNNING)
    started_at: Optional[datetime] = Field(None, description="Experiment start time")
    ended_at: Optional[datetime] = Field(None, description="Experiment end time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    
    # Environment and infrastructure
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    compute_resources: Dict[str, Any] = Field(default_factory=dict, description="Compute resources used")
    framework_version: Optional[str] = Field(None, description="ML framework version")
    
    # Tracking and lineage
    parent_experiment_id: Optional[UUID] = Field(None, description="Parent experiment ID")
    child_experiments: List[UUID] = Field(default_factory=list, description="Child experiment IDs")
    dataset_versions: List[str] = Field(default_factory=list, description="Dataset versions used")
    
    # External tracking
    external_experiment_id: Optional[str] = Field(None, description="External tracking ID (MLflow, W&B)")
    tracking_uri: Optional[str] = Field(None, description="External tracking URI")
    
    # Metadata
    created_by: Optional[str] = Field(None, description="Experiment creator")
    notes: Optional[str] = Field(None, description="Experiment notes")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return self.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]
    
    def get_duration(self) -> Optional[timedelta]:
        """Get experiment duration."""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        elif self.started_at and not self.ended_at:
            return datetime.utcnow() - self.started_at
        return None
    
    def add_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Add metric to experiment."""
        self.metrics[name] = value
        
        # Update best metrics
        if name not in self.best_metrics or value > self.best_metrics[name]:
            self.best_metrics[name] = value
        
        self.updated_at = datetime.utcnow()
    
    def add_parameter(self, name: str, value: Any) -> None:
        """Add parameter to experiment."""
        self.parameters[name] = value
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, key: str, value: str) -> None:
        """Add tag to experiment."""
        self.tags[key] = value
        self.updated_at = datetime.utcnow()
    
    def start_experiment(self) -> None:
        """Mark experiment as started."""
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def complete_experiment(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """Mark experiment as completed."""
        self.status = status
        self.ended_at = datetime.utcnow()
        
        if self.started_at:
            self.duration_seconds = (self.ended_at - self.started_at).total_seconds()
        
        self.updated_at = datetime.utcnow()


class MLModel(BaseModel):
    """
    Machine Learning model registry.
    
    Represents a trained ML model with versioning,
    metadata, and deployment information.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Model identifier")
    
    # Model identification
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Model metadata
    description: Optional[str] = Field(None, description="Model description")
    algorithm: str = Field(..., description="ML algorithm used")
    framework: str = Field(..., description="ML framework (sklearn, tensorflow, pytorch)")
    framework_version: Optional[str] = Field(None, description="Framework version")
    
    # Training information
    experiment_id: Optional[UUID] = Field(None, description="Associated experiment ID")
    training_dataset: Optional[str] = Field(None, description="Training dataset identifier")
    validation_dataset: Optional[str] = Field(None, description="Validation dataset identifier")
    
    # Model performance
    metrics: Dict[str, float] = Field(default_factory=dict, description="Model performance metrics")
    validation_metrics: Dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    test_metrics: Dict[str, float] = Field(default_factory=dict, description="Test metrics")
    
    # Model artifacts and storage
    model_uri: Optional[str] = Field(None, description="Model artifact URI")
    model_size_bytes: Optional[int] = Field(None, description="Model size in bytes")
    model_format: Optional[str] = Field(None, description="Model format (pickle, onnx, tensorflow)")
    
    # Schema and interface
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input schema definition")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema definition")
    feature_names: List[str] = Field(default_factory=list, description="Feature names")
    
    # Lifecycle and status
    status: ModelStatus = Field(default=ModelStatus.DEVELOPMENT)
    stage: str = Field(default="development", description="Model stage")
    
    # Deployment information
    deployments: List[UUID] = Field(default_factory=list, description="Active deployment IDs")
    serving_endpoints: List[str] = Field(default_factory=list, description="Serving endpoint URLs")
    
    # Lineage and provenance
    parent_model_id: Optional[UUID] = Field(None, description="Parent model ID")
    derived_models: List[UUID] = Field(default_factory=list, description="Derived model IDs")
    data_lineage: List[str] = Field(default_factory=list, description="Data lineage information")
    
    # Monitoring and drift
    drift_detection_enabled: bool = Field(default=False, description="Enable drift detection")
    performance_monitoring_enabled: bool = Field(default=False, description="Enable performance monitoring")
    last_performance_check: Optional[datetime] = Field(None, description="Last performance check")
    
    # Compliance and governance
    approval_status: Optional[str] = Field(None, description="Model approval status")
    approved_by: Optional[str] = Field(None, description="Approved by user")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_by: Optional[str] = Field(None, description="Model creator")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_production_ready(self) -> bool:
        """Check if model is ready for production."""
        return (
            self.status == ModelStatus.PRODUCTION and
            self.approval_status == "approved" and
            self.model_uri is not None
        )
    
    def is_deployed(self) -> bool:
        """Check if model is currently deployed."""
        return len(self.deployments) > 0
    
    def promote_to_stage(self, stage: str) -> None:
        """Promote model to a specific stage."""
        valid_stages = ["development", "staging", "production", "archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")
        
        self.stage = stage
        
        # Update status based on stage
        if stage == "production":
            self.status = ModelStatus.PRODUCTION
        elif stage == "staging":
            self.status = ModelStatus.STAGING
        elif stage == "archived":
            self.status = ModelStatus.ARCHIVED
        
        self.updated_at = datetime.utcnow()
    
    def add_deployment(self, deployment_id: UUID) -> None:
        """Add deployment to model."""
        if deployment_id not in self.deployments:
            self.deployments.append(deployment_id)
            self.updated_at = datetime.utcnow()
    
    def remove_deployment(self, deployment_id: UUID) -> None:
        """Remove deployment from model."""
        if deployment_id in self.deployments:
            self.deployments.remove(deployment_id)
            self.updated_at = datetime.utcnow()
    
    def update_metrics(self, metrics: Dict[str, float], metric_type: str = "metrics") -> None:
        """Update model metrics."""
        if metric_type == "validation":
            self.validation_metrics.update(metrics)
        elif metric_type == "test":
            self.test_metrics.update(metrics)
        else:
            self.metrics.update(metrics)
        
        self.updated_at = datetime.utcnow()


class ModelDeployment(BaseModel):
    """
    Model deployment configuration and status.
    
    Represents a model deployment with infrastructure,
    scaling, and monitoring configuration.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Deployment identifier")
    
    # Deployment identification
    name: str = Field(..., description="Deployment name")
    model_id: UUID = Field(..., description="Associated model ID")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Deployment configuration
    environment: str = Field(..., description="Deployment environment (dev/staging/prod)")
    platform: str = Field(..., description="Deployment platform (kubernetes, aws, azure, gcp)")
    
    # Serving configuration
    endpoint_url: Optional[str] = Field(None, description="Serving endpoint URL")
    endpoint_type: str = Field(default="rest", description="Endpoint type (rest, grpc, batch)")
    authentication_required: bool = Field(default=True, description="Require authentication")
    
    # Infrastructure and scaling
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    scaling_config: Dict[str, Any] = Field(default_factory=dict, description="Auto-scaling configuration")
    instance_count: int = Field(default=1, ge=1, description="Number of instances")
    
    # Performance and SLA
    max_latency_ms: Optional[int] = Field(None, description="Maximum latency SLA")
    target_throughput_rps: Optional[int] = Field(None, description="Target throughput RPS")
    
    # Deployment status and health
    status: DeploymentStatus = Field(default=DeploymentStatus.DEPLOYING)
    health_status: str = Field(default="unknown", description="Health check status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check")
    
    # Monitoring and observability
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    logging_enabled: bool = Field(default=True, description="Enable logging")
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    
    # Traffic and routing
    traffic_percentage: float = Field(default=100.0, ge=0.0, le=100.0, description="Traffic percentage")
    canary_deployment: bool = Field(default=False, description="Is canary deployment")
    
    # Model serving metadata
    model_version: str = Field(..., description="Deployed model version")
    model_uri: str = Field(..., description="Model artifact URI")
    runtime_version: Optional[str] = Field(None, description="Runtime version")
    
    # Deployment history
    deployment_history: List[Dict[str, Any]] = Field(default_factory=list, description="Deployment history")
    rollback_enabled: bool = Field(default=True, description="Enable rollback")
    
    # Performance metrics
    request_count: int = Field(default=0, description="Total request count")
    error_count: int = Field(default=0, description="Total error count")
    average_latency_ms: Optional[float] = Field(None, description="Average latency")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    deployed_by: Optional[str] = Field(None, description="Deployed by user")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_healthy(self) -> bool:
        """Check if deployment is healthy."""
        return self.status == DeploymentStatus.DEPLOYED and self.health_status == "healthy"
    
    def is_production(self) -> bool:
        """Check if deployment is in production."""
        return self.environment.lower() in ["production", "prod"]
    
    def get_error_rate(self) -> float:
        """Get error rate percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    def update_status(self, status: DeploymentStatus) -> None:
        """Update deployment status."""
        self.status = status
        
        if status == DeploymentStatus.DEPLOYED:
            self.deployed_at = datetime.utcnow()
        
        self.updated_at = datetime.utcnow()
    
    def update_health(self, health_status: str) -> None:
        """Update health status."""
        self.health_status = health_status
        self.last_health_check = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def record_request(self, success: bool = True, latency_ms: Optional[float] = None) -> None:
        """Record API request."""
        self.request_count += 1
        
        if not success:
            self.error_count += 1
        
        # Update average latency
        if latency_ms is not None:
            if self.average_latency_ms is None:
                self.average_latency_ms = latency_ms
            else:
                # Simple moving average
                self.average_latency_ms = (self.average_latency_ms + latency_ms) / 2
        
        self.updated_at = datetime.utcnow()
    
    def add_deployment_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add deployment event to history."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "status": self.status.value
        }
        
        self.deployment_history.append(event)
        
        # Keep only last 100 events
        if len(self.deployment_history) > 100:
            self.deployment_history = self.deployment_history[-100:]
        
        self.updated_at = datetime.utcnow()


class MLPipeline(BaseModel):
    """
    Machine Learning pipeline configuration.
    
    Represents an end-to-end ML pipeline with steps,
    dependencies, scheduling, and execution tracking.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Pipeline identifier")
    
    # Pipeline identification
    name: str = Field(..., description="Pipeline name")
    version: str = Field(..., description="Pipeline version")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Pipeline configuration
    description: Optional[str] = Field(None, description="Pipeline description")
    pipeline_type: str = Field(..., description="Pipeline type (training, inference, batch)")
    
    # Pipeline steps and workflow
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline steps")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Step dependencies")
    
    # Execution configuration
    execution_config: Dict[str, Any] = Field(default_factory=dict, description="Execution configuration")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    timeout_minutes: int = Field(default=60, description="Pipeline timeout")
    
    # Scheduling
    schedule_enabled: bool = Field(default=False, description="Enable scheduling")
    schedule_cron: Optional[str] = Field(None, description="Cron schedule expression")
    schedule_timezone: str = Field(default="UTC", description="Schedule timezone")
    
    # Parameters and inputs
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Pipeline parameters")
    input_datasets: List[str] = Field(default_factory=list, description="Input dataset URIs")
    output_artifacts: List[str] = Field(default_factory=list, description="Output artifact URIs")
    
    # Execution tracking
    status: PipelineStatus = Field(default=PipelineStatus.PENDING)
    current_step: Optional[str] = Field(None, description="Currently executing step")
    
    # Execution history
    executions: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline executions")
    last_execution: Optional[datetime] = Field(None, description="Last execution time")
    next_execution: Optional[datetime] = Field(None, description="Next scheduled execution")
    
    # Success and failure tracking
    total_executions: int = Field(default=0, description="Total executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    failed_executions: int = Field(default=0, description="Failed executions")
    
    # Monitoring and alerting
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    alert_on_failure: bool = Field(default=True, description="Alert on failure")
    alert_recipients: List[str] = Field(default_factory=list, description="Alert recipients")
    
    # External pipeline integration
    external_pipeline_id: Optional[str] = Field(None, description="External pipeline ID (Kubeflow, Airflow)")
    pipeline_uri: Optional[str] = Field(None, description="Pipeline definition URI")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    created_by: Optional[str] = Field(None, description="Pipeline creator")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_scheduled(self) -> bool:
        """Check if pipeline is scheduled."""
        return self.schedule_enabled and self.schedule_cron is not None
    
    def is_running(self) -> bool:
        """Check if pipeline is currently running."""
        return self.status == PipelineStatus.RUNNING
    
    def get_success_rate(self) -> float:
        """Get pipeline success rate."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    def add_step(self, step_name: str, step_config: Dict[str, Any], dependencies: Optional[List[str]] = None) -> None:
        """Add step to pipeline."""
        step = {
            "name": step_name,
            "config": step_config,
            "order": len(self.steps)
        }
        
        self.steps.append(step)
        
        if dependencies:
            self.dependencies[step_name] = dependencies
        
        self.updated_at = datetime.utcnow()
    
    def start_execution(self, execution_id: Optional[str] = None) -> str:
        """Start pipeline execution."""
        if not execution_id:
            execution_id = str(uuid4())
        
        execution = {
            "execution_id": execution_id,
            "started_at": datetime.utcnow().isoformat(),
            "status": "running",
            "current_step": self.steps[0]["name"] if self.steps else None,
            "completed_steps": [],
            "failed_steps": []
        }
        
        self.executions.append(execution)
        self.status = PipelineStatus.RUNNING
        self.current_step = execution["current_step"]
        self.last_execution = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        return execution_id
    
    def complete_execution(self, execution_id: str, success: bool = True) -> None:
        """Complete pipeline execution."""
        # Find execution
        execution = None
        for exec_record in self.executions:
            if exec_record["execution_id"] == execution_id:
                execution = exec_record
                break
        
        if execution:
            execution["ended_at"] = datetime.utcnow().isoformat()
            execution["status"] = "completed" if success else "failed"
            
            # Calculate duration
            started = datetime.fromisoformat(execution["started_at"])
            ended = datetime.utcnow()
            execution["duration_seconds"] = (ended - started).total_seconds()
        
        # Update pipeline status
        self.status = PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
        self.current_step = None
        self.total_executions += 1
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        self.updated_at = datetime.utcnow()
    
    def update_step_status(self, execution_id: str, step_name: str, status: str) -> None:
        """Update step status in execution."""
        # Find execution
        execution = None
        for exec_record in self.executions:
            if exec_record["execution_id"] == execution_id:
                execution = exec_record
                break
        
        if execution:
            if status == "completed":
                if step_name not in execution["completed_steps"]:
                    execution["completed_steps"].append(step_name)
            elif status == "failed":
                if step_name not in execution["failed_steps"]:
                    execution["failed_steps"].append(step_name)
            
            # Update current step if this step completed successfully
            if status == "completed":
                # Find next step
                current_order = next((s["order"] for s in self.steps if s["name"] == step_name), -1)
                next_steps = [s for s in self.steps if s["order"] > current_order]
                
                if next_steps:
                    next_step = min(next_steps, key=lambda s: s["order"])
                    execution["current_step"] = next_step["name"]
                    self.current_step = next_step["name"]
                else:
                    execution["current_step"] = None
                    self.current_step = None
        
        self.updated_at = datetime.utcnow()