"""Deployment orchestration service for managing processor deployments."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from monorepo.application.services.model_registry_service import ModelRegistryService
from monorepo.domain.entities.deployment import (
    Deployment,
    DeploymentConfig,
    DeploymentStatus,
    DeploymentStrategy,
    Environment,
    HealthMetrics,
    StrategyType,
)
from monorepo.domain.entities.model_version import ModelVersion


class DeploymentOrchestrationError(Exception):
    """Base exception for deployment orchestration errors."""

    pass


class DeploymentNotFoundError(DeploymentOrchestrationError):
    """Deployment not found error."""

    pass


class EnvironmentNotReadyError(DeploymentOrchestrationError):
    """Environment not ready for deployment."""

    pass


class DeploymentValidationError(DeploymentOrchestrationError):
    """Deployment configuration validation error."""

    pass


class DeploymentOrchestrationService:
    """Service for orchestrating processor deployments across environments.

    This service provides comprehensive deployment lifecycle management including:
    - Processor deployment across multiple environments
    - Blue-green, canary, and rolling deployment strategies
    - Health monitoring and automatic rollback
    - Environment promotion workflows
    - Deployment history and audit trails
    """

    def __init__(
        self,
        processor_registry_service: ModelRegistryService,
        storage_path: Path,
        container_registry_url: str = "software-registry.local",
        kubernetes_config_path: Path | None = None,
    ):
        """Initialize deployment orchestration service.

        Args:
            processor_registry_service: Processor registry service
            storage_path: Path for deployment persistence
            container_registry_url: Container registry URL
            kubernetes_config_path: Kubernetes configuration file path
        """
        self.processor_registry_service = processor_registry_service
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.container_registry_url = container_registry_url
        self.kubernetes_config_path = kubernetes_config_path

        # In-memory deployment tracking
        self.active_deployments: dict[UUID, Deployment] = {}
        self.deployment_history: list[Deployment] = []

        # Background tasks
        self._monitoring_tasks: dict[UUID, asyncio.Task] = {}

        # Load existing deployments
        asyncio.create_task(self._load_deployments())

    async def _load_deployments(self) -> None:
        """Load deployments from storage."""
        deployment_files = self.storage_path.glob("deployment_*.json")

        for deployment_file in deployment_files:
            try:
                deployment = await self._load_deployment_from_file(deployment_file)
                if deployment.is_deployed:
                    self.active_deployments[deployment.id] = deployment
                    # Start monitoring for active deployments
                    await self._start_monitoring(deployment)
                else:
                    self.deployment_history.append(deployment)
            except Exception:
                # Skip corrupted deployment files
                continue

    async def deploy_processor(
        self,
        processor_version_id: UUID,
        target_environment: Environment,
        strategy: DeploymentStrategy,
        deployment_config: DeploymentConfig | None = None,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Deployment:
        """Deploy a processor version to a target environment.

        Args:
            processor_version_id: Processor version to deploy
            target_environment: Target deployment environment
            strategy: Deployment strategy to use
            deployment_config: Deployment configuration
            user: User initiating deployment
            groups: User's groups

        Returns:
            Created deployment

        Raises:
            DeploymentValidationError: If validation fails
            EnvironmentNotReadyError: If environment not ready
        """
        # Validate processor version exists
        processor_version = await self._get_processor_version(processor_version_id, user, groups)

        # Validate environment readiness
        await self._validate_environment_readiness(target_environment)

        # Use default config if none provided
        if deployment_config is None:
            deployment_config = self._get_default_deployment_config(target_environment)

        # Check for existing deployments in environment
        existing_deployment = await self._get_active_deployment_in_environment(
            target_environment
        )

        # Create deployment
        deployment = Deployment(
            processor_version_id=processor_version_id,
            environment=target_environment,
            deployment_config=deployment_config,
            strategy=strategy,
            created_by=user,
            rollback_version_id=(
                existing_deployment.processor_version_id if existing_deployment else None
            ),
        )

        # Start deployment process
        deployment.status = DeploymentStatus.IN_PROGRESS
        await self._save_deployment(deployment)

        try:
            # Execute deployment based on strategy
            await self._execute_deployment_strategy(deployment, processor_version)

            # Mark as deployed and start monitoring
            deployment.mark_deployed()
            self.active_deployments[deployment.id] = deployment
            await self._start_monitoring(deployment)

            # Archive previous deployment if exists
            if existing_deployment:
                await self._archive_deployment(existing_deployment.id)

            await self._save_deployment(deployment)

            return deployment

        except Exception as e:
            deployment.mark_failed(str(e))
            await self._save_deployment(deployment)
            raise DeploymentOrchestrationError(f"Deployment failed: {e}") from e

    async def promote_to_production(
        self,
        deployment_id: UUID,
        approval_metadata: dict[str, Any],
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Deployment:
        """Promote a staging deployment to production.

        Args:
            deployment_id: Deployment to promote
            approval_metadata: Approval information and metadata
            user: User performing promotion
            groups: User's groups

        Returns:
            Production deployment

        Raises:
            DeploymentNotFoundError: If deployment not found
            DeploymentValidationError: If promotion not allowed
        """
        # Get staging deployment
        staging_deployment = await self.get_deployment(deployment_id)

        if staging_deployment.environment != Environment.STAGING:
            raise DeploymentValidationError(
                f"Can only promote from staging, got {staging_deployment.environment.value}"
            )

        if not staging_deployment.is_deployed or not staging_deployment.is_healthy:
            raise DeploymentValidationError("Can only promote healthy, deployed models")

        # Create production deployment with same configuration
        production_deployment = await self.deploy_processor(
            processor_version_id=staging_deployment.processor_version_id,
            target_environment=Environment.PRODUCTION,
            strategy=staging_deployment.strategy,
            deployment_config=staging_deployment.deployment_config,
            user=user,
            groups=groups,
        )

        # Add promotion metadata
        production_deployment.metadata.update(
            {
                "promoted_from": str(staging_deployment.id),
                "approval_metadata": approval_metadata,
                "promoted_at": datetime.utcnow().isoformat(),
                "promoted_by": user,
            }
        )

        await self._save_deployment(production_deployment)
        return production_deployment

    async def rollback_deployment(
        self,
        deployment_id: UUID,
        reason: str,
        user: str = "system",
        groups: list[str] | None = None,
    ) -> Deployment | None:
        """Rollback a deployment to the previous version.

        Args:
            deployment_id: Deployment to rollback
            reason: Reason for rollback
            user: User performing rollback
            groups: User's groups

        Returns:
            New deployment with previous version, or None if no rollback version
        """
        deployment = await self.get_deployment(deployment_id)

        if not deployment.rollback_version_id:
            deployment.start_rollback(f"No rollback version available: {reason}")
            deployment.complete_rollback()
            await self._save_deployment(deployment)
            return None

        # Start rollback process
        deployment.start_rollback(reason)
        await self._save_deployment(deployment)

        try:
            # Deploy previous version
            rollback_deployment = await self.deploy_processor(
                processor_version_id=deployment.rollback_version_id,
                target_environment=deployment.environment,
                strategy=DeploymentStrategy(strategy_type=StrategyType.DIRECT),
                deployment_config=deployment.deployment_config,
                user=user,
                groups=groups,
            )

            # Add rollback metadata
            rollback_deployment.metadata.update(
                {
                    "rolled_back_from": str(deployment.id),
                    "rollback_reason": reason,
                    "rollback_initiated_by": user,
                    "rollback_completed_at": datetime.utcnow().isoformat(),
                }
            )

            # Complete rollback of original deployment
            deployment.complete_rollback()
            await self._stop_monitoring(deployment.id)

            await self._save_deployment(deployment)
            await self._save_deployment(rollback_deployment)

            return rollback_deployment

        except Exception as e:
            deployment.mark_failed(f"Rollback failed: {e}")
            await self._save_deployment(deployment)
            raise DeploymentOrchestrationError(f"Rollback failed: {e}") from e

    async def get_deployment(self, deployment_id: UUID) -> Deployment:
        """Get deployment by ID.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment

        Raises:
            DeploymentNotFoundError: If deployment not found
        """
        # Check active deployments first
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]

        # Check deployment history
        for deployment in self.deployment_history:
            if deployment.id == deployment_id:
                return deployment

        raise DeploymentNotFoundError(f"Deployment {deployment_id} not found")

    async def list_deployments(
        self,
        environment: Environment | None = None,
        status: DeploymentStatus | None = None,
        processor_version_id: UUID | None = None,
        limit: int | None = None,
    ) -> list[Deployment]:
        """List deployments with optional filtering.

        Args:
            environment: Filter by environment
            status: Filter by status
            processor_version_id: Filter by processor version
            limit: Maximum results

        Returns:
            List of matching deployments
        """
        all_deployments = (
            list(self.active_deployments.values()) + self.deployment_history
        )

        # Apply filters
        filtered_deployments = []
        for deployment in all_deployments:
            if environment and deployment.environment != environment:
                continue
            if status and deployment.status != status:
                continue
            if processor_version_id and deployment.processor_version_id != processor_version_id:
                continue

            filtered_deployments.append(deployment)

        # Sort by creation date (newest first)
        filtered_deployments.sort(key=lambda d: d.created_at, reverse=True)

        # Apply limit
        if limit:
            filtered_deployments = filtered_deployments[:limit]

        return filtered_deployments

    async def get_environment_status(self, environment: Environment) -> dict[str, Any]:
        """Get status of a specific environment.

        Args:
            environment: Environment to check

        Returns:
            Environment status information
        """
        active_deployment = await self._get_active_deployment_in_environment(
            environment
        )

        status = {
            "environment": environment.value,
            "has_active_deployment": active_deployment is not None,
            "is_healthy": False,
            "deployment_count": 0,
            "last_deployment": None,
        }

        if active_deployment:
            status.update(
                {
                    "is_healthy": active_deployment.is_healthy,
                    "deployment_id": str(active_deployment.id),
                    "processor_version_id": str(active_deployment.processor_version_id),
                    "status": active_deployment.status.value,
                    "health_score": active_deployment.health_score,
                    "traffic_percentage": active_deployment.traffic_percentage,
                    "deployed_at": (
                        active_deployment.deployed_at.isoformat()
                        if active_deployment.deployed_at
                        else None
                    ),
                }
            )

        # Count deployments in this environment
        env_deployments = await self.list_deployments(environment=environment)
        status["deployment_count"] = len(env_deployments)

        if env_deployments:
            status["last_deployment"] = env_deployments[0].created_at.isoformat()

        return status

    async def update_deployment_health(
        self, deployment_id: UUID, health_measurements: HealthMetrics
    ) -> None:
        """Update health measurements for a deployment.

        Args:
            deployment_id: Deployment ID
            health_measurements: New health measurements
        """
        deployment = await self.get_deployment(deployment_id)
        deployment.update_health_measurements(health_measurements)

        # Check if rollback is needed
        if deployment.is_deployed and deployment.should_rollback(
            100
        ):  # Assuming 100 requests
            await self.rollback_deployment(
                deployment_id,
                f"Automatic rollback due to health degradation: {health_measurements.error_rate:.1f}% error rate",
            )

        await self._save_deployment(deployment)

    async def _execute_deployment_strategy(
        self, deployment: Deployment, processor_version: ModelVersion
    ) -> None:
        """Execute deployment based on strategy."""
        if deployment.strategy.strategy_type == StrategyType.BLUE_GREEN:
            await self._execute_blue_green_deployment(deployment, processor_version)
        elif deployment.strategy.strategy_type == StrategyType.CANARY:
            await self._execute_canary_deployment(deployment, processor_version)
        elif deployment.strategy.strategy_type == StrategyType.ROLLING:
            await self._execute_rolling_deployment(deployment, processor_version)
        else:  # DIRECT
            await self._execute_direct_deployment(deployment, processor_version)

    async def _execute_blue_green_deployment(
        self, deployment: Deployment, processor_version: ModelVersion
    ) -> None:
        """Execute blue-green deployment strategy."""
        # Create new deployment (green)
        await self._create_kubernetes_deployment(
            deployment, processor_version, suffix="green"
        )

        # Wait for green deployment to be ready
        await self._wait_for_deployment_ready(deployment, suffix="green")

        # Run smoke tests
        smoke_test_duration = deployment.strategy.configuration.get(
            "smoke_test_duration_minutes", 5
        )
        await asyncio.sleep(smoke_test_duration * 60)

        # Switch traffic to green
        await self._switch_traffic(deployment, target_suffix="green")

        # Clean up blue deployment after delay
        switch_delay = deployment.strategy.configuration.get(
            "traffic_switch_delay_seconds", 30
        )
        await asyncio.sleep(switch_delay)
        await self._cleanup_deployment(deployment, suffix="blue")

    async def _execute_canary_deployment(
        self, deployment: Deployment, processor_version: ModelVersion
    ) -> None:
        """Execute canary deployment strategy."""
        # Create canary deployment with initial traffic
        await self._create_kubernetes_deployment(
            deployment, processor_version, suffix="canary"
        )

        # Start with initial traffic percentage
        initial_percentage = deployment.strategy.configuration.get(
            "initial_traffic_percentage", 10
        )
        await self._set_traffic_percentage(
            deployment, initial_percentage, suffix="canary"
        )

        deployment.traffic_percentage = initial_percentage

    async def _execute_rolling_deployment(
        self, deployment: Deployment, processor_version: ModelVersion
    ) -> None:
        """Execute rolling deployment strategy."""
        max_unavailable = deployment.strategy.configuration.get("max_unavailable", 1)
        max_surge = deployment.strategy.configuration.get("max_surge", 1)

        # Update existing deployment with rolling update strategy
        await self._update_kubernetes_deployment(
            deployment, processor_version, max_unavailable, max_surge
        )

        # Wait for rollout to complete
        await self._wait_for_rollout_complete(deployment)

    async def _execute_direct_deployment(
        self, deployment: Deployment, processor_version: ModelVersion
    ) -> None:
        """Execute direct deployment strategy."""
        # Replace existing deployment immediately
        await self._create_kubernetes_deployment(deployment, processor_version)
        await self._wait_for_deployment_ready(deployment)

    async def _start_monitoring(self, deployment: Deployment) -> None:
        """Start health monitoring for a deployment."""
        if deployment.id in self._monitoring_tasks:
            return

        monitoring_task = asyncio.create_task(
            self._monitor_deployment_health(deployment)
        )
        self._monitoring_tasks[deployment.id] = monitoring_task

    async def _stop_monitoring(self, deployment_id: UUID) -> None:
        """Stop health monitoring for a deployment."""
        if deployment_id in self._monitoring_tasks:
            task = self._monitoring_tasks.pop(deployment_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _monitor_deployment_health(self, deployment: Deployment) -> None:
        """Monitor deployment health continuously."""
        while deployment.is_deployed:
            try:
                # Simulate health measurements collection
                # In real implementation, this would query Kubernetes/Prometheus
                health_measurements = await self._collect_health_measurements(deployment)
                deployment.update_health_measurements(health_measurements)

                # Update traffic for canary deployments
                if deployment.strategy.strategy_type == StrategyType.CANARY:
                    deployment.update_traffic_percentage()
                    await self._set_traffic_percentage(
                        deployment, deployment.traffic_percentage, suffix="canary"
                    )

                await self._save_deployment(deployment)

                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception:
                # Log error but continue monitoring
                await asyncio.sleep(60)  # Longer sleep on error

    async def _collect_health_measurements(self, deployment: Deployment) -> HealthMetrics:
        """Collect health measurements for a deployment."""
        # Simulate measurements collection
        # In real implementation, this would query monitoring systems
        return HealthMetrics(
            cpu_usage=45.0,
            memory_usage=60.0,
            request_rate=150.0,
            error_rate=1.5,
            response_time_p95=250.0,
            response_time_p99=500.0,
        )

    # Kubernetes integration methods (simplified stubs)
    async def _create_kubernetes_deployment(
        self, deployment: Deployment, processor_version: ModelVersion, suffix: str = ""
    ) -> None:
        """Create Kubernetes deployment."""
        # In real implementation, this would use Kubernetes client
        await asyncio.sleep(2)  # Simulate deployment creation

    async def _wait_for_deployment_ready(
        self, deployment: Deployment, suffix: str = ""
    ) -> None:
        """Wait for Kubernetes deployment to be ready."""
        await asyncio.sleep(5)  # Simulate waiting for readiness

    async def _switch_traffic(self, deployment: Deployment, target_suffix: str) -> None:
        """Switch traffic to target deployment."""
        await asyncio.sleep(1)  # Simulate traffic switching

    async def _set_traffic_percentage(
        self, deployment: Deployment, percentage: int, suffix: str
    ) -> None:
        """Set traffic percentage for deployment."""
        await asyncio.sleep(0.5)  # Simulate traffic adjustment

    async def _cleanup_deployment(self, deployment: Deployment, suffix: str) -> None:
        """Clean up old deployment."""
        await asyncio.sleep(1)  # Simulate cleanup

    async def _update_kubernetes_deployment(
        self,
        deployment: Deployment,
        processor_version: ModelVersion,
        max_unavailable: int,
        max_surge: int,
    ) -> None:
        """Update existing Kubernetes deployment."""
        await asyncio.sleep(3)  # Simulate deployment update

    async def _wait_for_rollout_complete(self, deployment: Deployment) -> None:
        """Wait for rolling update to complete."""
        await asyncio.sleep(10)  # Simulate rollout completion

    # Utility methods
    async def _get_processor_version(
        self, processor_version_id: UUID, user: str, groups: list[str] | None
    ) -> ModelVersion:
        """Get and validate processor version."""
        # This would integrate with processor registry service
        # For now, return a mock version
        from monorepo.domain.value_objects.model_storage_info import (
            ModelStorageInfo,
            SerializationFormat,
            StorageBackend,
        )
        from monorepo.domain.value_objects.performance_metrics import PerformanceMetrics
        from monorepo.domain.value_objects.semantic_version import SemanticVersion

        return ModelVersion(
            processor_id=processor_version_id,
            version=SemanticVersion(1, 0, 0),
            detector_id=processor_version_id,
            created_by=user,
            performance_measurements=PerformanceMetrics(),
            storage_info=ModelStorageInfo(
                storage_backend=StorageBackend.FILESYSTEM,
                serialization_format=SerializationFormat.PICKLE,
                storage_path=f"/models/{processor_version_id}",
                file_size_bytes=1024000,
            ),
        )

    async def _validate_environment_readiness(self, environment: Environment) -> None:
        """Validate that environment is ready for deployment."""
        # Check Kubernetes connectivity, resource availability, etc.
        pass

    def _get_default_deployment_config(
        self, environment: Environment
    ) -> DeploymentConfig:
        """Get default deployment configuration for environment."""
        if environment == Environment.PRODUCTION:
            return DeploymentConfig(
                replicas=5,
                cpu_request="500m",
                cpu_limit="2000m",
                memory_request="1Gi",
                memory_limit="4Gi",
            )
        elif environment == Environment.STAGING:
            return DeploymentConfig(
                replicas=2,
                cpu_request="250m",
                cpu_limit="1000m",
                memory_request="512Mi",
                memory_limit="2Gi",
            )
        else:  # DEVELOPMENT
            return DeploymentConfig(
                replicas=1,
                cpu_request="100m",
                cpu_limit="500m",
                memory_request="256Mi",
                memory_limit="1Gi",
            )

    async def _get_active_deployment_in_environment(
        self, environment: Environment
    ) -> Deployment | None:
        """Get active deployment in environment."""
        for deployment in self.active_deployments.values():
            if deployment.environment == environment and deployment.is_deployed:
                return deployment
        return None

    async def _archive_deployment(self, deployment_id: UUID) -> None:
        """Archive a deployment."""
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments.pop(deployment_id)
            deployment.archive()
            await self._stop_monitoring(deployment_id)
            self.deployment_history.append(deployment)
            await self._save_deployment(deployment)

    async def _save_deployment(self, deployment: Deployment) -> None:
        """Save deployment to storage."""
        deployment_file = self.storage_path / f"deployment_{deployment.id}.json"

        with open(deployment_file, "w") as f:
            json.dump(deployment.get_deployment_info(), f, indent=2, default=str)

    async def _load_deployment_from_file(self, deployment_file: Path) -> Deployment:
        """Load deployment from file."""
        with open(deployment_file) as f:
            data = json.load(f)

        # Simplified deserialization - in practice would be more comprehensive
        # This is a stub implementation
        deployment_config = DeploymentConfig()
        strategy = DeploymentStrategy(strategy_type=StrategyType.DIRECT)

        deployment = Deployment(
            processor_version_id=UUID(data["processor_version_id"]),
            environment=Environment(data["environment"]),
            deployment_config=deployment_config,
            strategy=strategy,
            created_by=data["created_by"],
        )

        return deployment
