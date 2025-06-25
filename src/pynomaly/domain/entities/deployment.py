"""Deployment entity for model deployment management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class Environment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStatus(Enum):
    """Status of a deployment."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


class StrategyType(Enum):
    """Deployment strategy types."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    DIRECT = "direct"


@dataclass
class HealthMetrics:
    """Health metrics for a deployment."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def is_healthy(self) -> bool:
        """Check if deployment is healthy based on metrics."""
        return (
            self.cpu_usage < 80.0
            and self.memory_usage < 85.0
            and self.error_rate < 5.0
            and self.response_time_p95 < 1000.0  # 1 second
        )

    def get_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        cpu_score = max(0, 1 - (self.cpu_usage / 100))
        memory_score = max(0, 1 - (self.memory_usage / 100))
        error_score = max(0, 1 - (self.error_rate / 100))
        latency_score = max(0, 1 - (self.response_time_p95 / 5000))  # 5 second max

        return (cpu_score + memory_score + error_score + latency_score) / 4


@dataclass
class RollbackCriteria:
    """Criteria for automatic rollback."""

    max_error_rate: float = 10.0  # Percentage
    max_response_time: float = 5000.0  # Milliseconds
    min_success_rate: float = 90.0  # Percentage
    evaluation_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    min_requests: int = 100  # Minimum requests before evaluation

    def should_rollback(self, metrics: HealthMetrics, request_count: int) -> bool:
        """Determine if rollback should be triggered."""
        if request_count < self.min_requests:
            return False

        return (
            metrics.error_rate > self.max_error_rate
            or metrics.response_time_p95 > self.max_response_time
            or (100 - metrics.error_rate) < self.min_success_rate
        )


@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""

    replicas: int = 3
    cpu_request: str = "250m"
    cpu_limit: str = "1000m"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"
    environment_variables: dict[str, str] = field(default_factory=dict)
    port: int = 8080
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    timeout_seconds: int = 30

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "replicas": self.replicas,
            "resources": {
                "requests": {"cpu": self.cpu_request, "memory": self.memory_request},
                "limits": {"cpu": self.cpu_limit, "memory": self.memory_limit},
            },
            "environment_variables": self.environment_variables.copy(),
            "port": self.port,
            "health_check_path": self.health_check_path,
            "readiness_check_path": self.readiness_check_path,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class DeploymentStrategy:
    """Strategy configuration for deployment."""

    strategy_type: StrategyType
    configuration: dict[str, Any] = field(default_factory=dict)
    rollback_criteria: RollbackCriteria = field(default_factory=RollbackCriteria)

    def __post_init__(self) -> None:
        """Set default configurations based on strategy type."""
        if self.strategy_type == StrategyType.CANARY and not self.configuration:
            self.configuration = {
                "initial_traffic_percentage": 10,
                "increment_percentage": 20,
                "evaluation_interval_minutes": 10,
                "max_traffic_percentage": 100,
            }
        elif self.strategy_type == StrategyType.BLUE_GREEN and not self.configuration:
            self.configuration = {
                "smoke_test_duration_minutes": 5,
                "traffic_switch_delay_seconds": 30,
            }
        elif self.strategy_type == StrategyType.ROLLING and not self.configuration:
            self.configuration = {
                "max_unavailable": 1,
                "max_surge": 1,
                "progress_deadline_seconds": 600,
            }

    def get_canary_traffic_percentage(self, deployment_duration: timedelta) -> int:
        """Get current traffic percentage for canary deployment."""
        if self.strategy_type != StrategyType.CANARY:
            return 100

        initial = self.configuration.get("initial_traffic_percentage", 10)
        increment = self.configuration.get("increment_percentage", 20)
        interval = self.configuration.get("evaluation_interval_minutes", 10)
        max_percentage = self.configuration.get("max_traffic_percentage", 100)

        intervals_passed = int(deployment_duration.total_seconds() / (interval * 60))
        current_percentage = initial + (intervals_passed * increment)

        return min(current_percentage, max_percentage)


@dataclass
class Deployment:
    """Represents a model deployment to a specific environment.

    This entity captures the complete state of a deployed model version
    including its configuration, health metrics, and deployment strategy.

    Attributes:
        id: Unique identifier for this deployment
        model_version_id: ID of the model version being deployed
        environment: Target deployment environment
        deployment_config: Resource and configuration settings
        strategy: Deployment strategy and rollback criteria
        status: Current deployment status
        health_metrics: Real-time health and performance metrics
        created_at: When deployment was initiated
        deployed_at: When deployment completed successfully
        created_by: User who initiated the deployment
        namespace: Kubernetes namespace for deployment
        cluster_name: Target cluster for deployment
        metadata: Additional deployment metadata
        rollback_version_id: Previous version for rollback
        traffic_percentage: Current traffic percentage (for canary)
    """

    model_version_id: UUID
    environment: Environment
    deployment_config: DeploymentConfig
    strategy: DeploymentStrategy
    created_by: str
    id: UUID = field(default_factory=uuid4)
    status: DeploymentStatus = DeploymentStatus.PENDING
    health_metrics: HealthMetrics = field(default_factory=HealthMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: datetime | None = None
    namespace: str = "pynomaly-default"
    cluster_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    rollback_version_id: UUID | None = None
    traffic_percentage: int = 0

    def __post_init__(self) -> None:
        """Validate deployment after initialization."""
        if not isinstance(self.environment, Environment):
            raise TypeError(
                f"Environment must be Environment enum, got {type(self.environment)}"
            )

        if not isinstance(self.deployment_config, DeploymentConfig):
            raise TypeError(
                f"Config must be DeploymentConfig, got {type(self.deployment_config)}"
            )

        if not isinstance(self.strategy, DeploymentStrategy):
            raise TypeError(
                f"Strategy must be DeploymentStrategy, got {type(self.strategy)}"
            )

        if not self.created_by:
            raise ValueError("Created by cannot be empty")

        # Set namespace based on environment if not specified
        if self.namespace == "pynomaly-default":
            self.namespace = f"pynomaly-{self.environment.value}"

    @property
    def is_deployed(self) -> bool:
        """Check if deployment is currently active."""
        return self.status == DeploymentStatus.DEPLOYED

    @property
    def is_healthy(self) -> bool:
        """Check if deployment is healthy."""
        return self.health_metrics.is_healthy()

    @property
    def deployment_duration(self) -> timedelta | None:
        """Get duration since deployment started."""
        if self.deployed_at:
            return datetime.utcnow() - self.deployed_at
        return None

    @property
    def health_score(self) -> float:
        """Get overall health score."""
        return self.health_metrics.get_health_score()

    def mark_deployed(self) -> None:
        """Mark deployment as successfully deployed."""
        self.status = DeploymentStatus.DEPLOYED
        self.deployed_at = datetime.utcnow()
        self.traffic_percentage = (
            100 if self.strategy.strategy_type != StrategyType.CANARY else 10
        )
        self.metadata["deployed_at"] = self.deployed_at.isoformat()

    def mark_failed(self, error_message: str) -> None:
        """Mark deployment as failed."""
        self.status = DeploymentStatus.FAILED
        self.metadata["error_message"] = error_message
        self.metadata["failed_at"] = datetime.utcnow().isoformat()

    def update_health_metrics(self, metrics: HealthMetrics) -> None:
        """Update health metrics."""
        self.health_metrics = metrics
        self.metadata["last_health_update"] = datetime.utcnow().isoformat()

    def should_rollback(self, request_count: int) -> bool:
        """Check if deployment should be rolled back."""
        return self.strategy.rollback_criteria.should_rollback(
            self.health_metrics, request_count
        )

    def start_rollback(self, reason: str) -> None:
        """Initiate rollback process."""
        self.status = DeploymentStatus.ROLLING_BACK
        self.metadata["rollback_reason"] = reason
        self.metadata["rollback_started_at"] = datetime.utcnow().isoformat()

    def complete_rollback(self) -> None:
        """Complete rollback process."""
        self.status = DeploymentStatus.ROLLED_BACK
        self.metadata["rollback_completed_at"] = datetime.utcnow().isoformat()

    def update_traffic_percentage(self) -> None:
        """Update traffic percentage for canary deployments."""
        if (
            self.strategy.strategy_type == StrategyType.CANARY
            and self.deployment_duration
        ):
            new_percentage = self.strategy.get_canary_traffic_percentage(
                self.deployment_duration
            )
            self.traffic_percentage = new_percentage
            self.metadata["traffic_percentage_updated_at"] = (
                datetime.utcnow().isoformat()
            )

    def archive(self) -> None:
        """Archive this deployment."""
        self.status = DeploymentStatus.ARCHIVED
        self.metadata["archived_at"] = datetime.utcnow().isoformat()

    def get_deployment_info(self) -> dict[str, Any]:
        """Get comprehensive deployment information."""
        return {
            "id": str(self.id),
            "model_version_id": str(self.model_version_id),
            "environment": self.environment.value,
            "status": self.status.value,
            "strategy": {
                "type": self.strategy.strategy_type.value,
                "configuration": self.strategy.configuration.copy(),
                "rollback_criteria": {
                    "max_error_rate": self.strategy.rollback_criteria.max_error_rate,
                    "max_response_time": self.strategy.rollback_criteria.max_response_time,
                    "min_success_rate": self.strategy.rollback_criteria.min_success_rate,
                },
            },
            "deployment_config": self.deployment_config.to_dict(),
            "health_metrics": {
                "cpu_usage": self.health_metrics.cpu_usage,
                "memory_usage": self.health_metrics.memory_usage,
                "request_rate": self.health_metrics.request_rate,
                "error_rate": self.health_metrics.error_rate,
                "response_time_p95": self.health_metrics.response_time_p95,
                "health_score": self.health_score,
                "is_healthy": self.is_healthy,
            },
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "created_by": self.created_by,
            "namespace": self.namespace,
            "cluster_name": self.cluster_name,
            "traffic_percentage": self.traffic_percentage,
            "rollback_version_id": str(self.rollback_version_id)
            if self.rollback_version_id
            else None,
            "metadata": self.metadata.copy(),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Deployment({self.environment.value}, "
            f"status={self.status.value}, "
            f"health={self.health_score:.2f})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Deployment(id={self.id}, model_version_id={self.model_version_id}, "
            f"environment={self.environment.value}, status={self.status.value})"
        )
