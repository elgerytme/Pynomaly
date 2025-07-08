"""Deployment manager for handling application deployments across environments."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pynomaly.domain.models.cicd import (
    Deployment,
    DeploymentEnvironment,
    DeploymentStrategy,
    PipelineStatus,
)


class DeploymentManager:
    """Service for managing application deployments across different environments."""

    def __init__(self, deployment_configs: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)

        # Deployment storage
        self.deployments: Dict[UUID, Deployment] = {}
        self.active_deployments: Set[UUID] = set()

        # Environment configurations
        self.deployment_configs = deployment_configs or self._get_default_configs()

        # Deployment strategies
        self.strategy_handlers = {
            DeploymentStrategy.ROLLING: self._rolling_deployment,
            DeploymentStrategy.BLUE_GREEN: self._blue_green_deployment,
            DeploymentStrategy.CANARY: self._canary_deployment,
            DeploymentStrategy.RECREATE: self._recreate_deployment,
            DeploymentStrategy.A_B_TESTING: self._ab_testing_deployment,
        }

        # Health check configurations
        self.health_check_configs = {
            DeploymentEnvironment.DEVELOPMENT: {"timeout": 30, "retries": 3},
            DeploymentEnvironment.TESTING: {"timeout": 60, "retries": 5},
            DeploymentEnvironment.STAGING: {"timeout": 120, "retries": 10},
            DeploymentEnvironment.PRODUCTION: {"timeout": 300, "retries": 20},
            DeploymentEnvironment.CANARY: {"timeout": 180, "retries": 15},
        }

        # Background tasks
        self.monitoring_tasks: Set[asyncio.Task] = set()
        self.is_running = False

        self.logger.info("Deployment manager initialized")

    def _get_default_configs(self) -> Dict[str, Any]:
        """Get default deployment configurations."""

        return {
            "environments": {
                "development": {
                    "cluster": "dev-cluster",
                    "namespace": "pynomaly-dev",
                    "replicas": 1,
                    "resources": {
                        "cpu_request": "100m",
                        "memory_request": "256Mi",
                        "cpu_limit": "500m",
                        "memory_limit": "512Mi",
                    },
                    "auto_deploy": True,
                    "approval_required": False,
                },
                "testing": {
                    "cluster": "test-cluster",
                    "namespace": "pynomaly-test",
                    "replicas": 2,
                    "resources": {
                        "cpu_request": "200m",
                        "memory_request": "512Mi",
                        "cpu_limit": "1000m",
                        "memory_limit": "1Gi",
                    },
                    "auto_deploy": False,
                    "approval_required": True,
                },
                "staging": {
                    "cluster": "staging-cluster",
                    "namespace": "pynomaly-staging",
                    "replicas": 3,
                    "resources": {
                        "cpu_request": "500m",
                        "memory_request": "1Gi",
                        "cpu_limit": "2000m",
                        "memory_limit": "2Gi",
                    },
                    "auto_deploy": False,
                    "approval_required": True,
                },
                "production": {
                    "cluster": "prod-cluster",
                    "namespace": "pynomaly-prod",
                    "replicas": 5,
                    "resources": {
                        "cpu_request": "1000m",
                        "memory_request": "2Gi",
                        "cpu_limit": "4000m",
                        "memory_limit": "4Gi",
                    },
                    "auto_deploy": False,
                    "approval_required": True,
                },
                "canary": {
                    "cluster": "prod-cluster",
                    "namespace": "pynomaly-canary",
                    "replicas": 1,
                    "resources": {
                        "cpu_request": "500m",
                        "memory_request": "1Gi",
                        "cpu_limit": "2000m",
                        "memory_limit": "2Gi",
                    },
                    "traffic_percentage": 5,
                    "auto_deploy": False,
                    "approval_required": True,
                },
            },
            "strategies": {
                "rolling": {
                    "max_unavailable": "25%",
                    "max_surge": "25%",
                    "timeout": "10m",
                },
                "blue_green": {
                    "switch_traffic_timeout": "5m",
                    "cleanup_delay": "30m",
                },
                "canary": {
                    "initial_traffic": "5%",
                    "step_percentage": "10%",
                    "step_duration": "10m",
                    "success_threshold": "95%",
                },
            },
        }

    async def start_monitoring(self) -> None:
        """Start deployment monitoring."""

        if self.is_running:
            return

        self.is_running = True

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._deployment_monitor_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop()),
        ]

        self.monitoring_tasks.update(tasks)

        self.logger.info("Started deployment monitoring")

    async def stop_monitoring(self) -> None:
        """Stop deployment monitoring."""

        self.is_running = False

        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        self.logger.info("Stopped deployment monitoring")

    async def deploy(
        self,
        environment: DeploymentEnvironment,
        version: str,
        commit_sha: str,
        branch: str,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
        deployed_by: Optional[UUID] = None,
        deployment_notes: str = "",
        rollback_on_failure: bool = True,
    ) -> Deployment:
        """Deploy application to specified environment."""

        deployment = Deployment(
            deployment_id=uuid4(),
            environment=environment,
            strategy=strategy,
            version=version,
            commit_sha=commit_sha,
            branch=branch,
            deployed_by=deployed_by or uuid4(),
            deployment_notes=deployment_notes,
        )

        # Apply environment-specific configuration
        env_config = self.deployment_configs["environments"].get(environment.value, {})
        if env_config:
            deployment.replicas = env_config.get("replicas", 1)
            resources = env_config.get("resources", {})
            deployment.cpu_request = resources.get("cpu_request", "100m")
            deployment.memory_request = resources.get("memory_request", "128Mi")

        # Generate deployment URLs
        base_domain = self._get_base_domain(environment)
        deployment.deployment_url = f"https://{environment.value}.{base_domain}"
        deployment.health_check_url = f"{deployment.deployment_url}/health"

        self.deployments[deployment.deployment_id] = deployment

        # Start deployment
        deployment.start_deployment()
        self.active_deployments.add(deployment.deployment_id)

        try:
            # Execute deployment strategy
            success = await self.strategy_handlers[strategy](deployment)

            if success:
                # Perform health checks
                health_ok = await self._perform_health_checks(deployment)

                if health_ok:
                    deployment.complete_deployment(PipelineStatus.SUCCESS)

                    # Update metrics
                    await self._update_deployment_metrics(deployment)

                    self.logger.info(f"Deployment successful: {version} to {environment.value}")
                else:
                    deployment.complete_deployment(PipelineStatus.FAILED, "Health checks failed")

                    if rollback_on_failure:
                        await self._rollback_deployment(deployment)

                    self.logger.error(f"Deployment failed health checks: {version} to {environment.value}")
            else:
                deployment.complete_deployment(PipelineStatus.FAILED, "Deployment strategy failed")

                if rollback_on_failure:
                    await self._rollback_deployment(deployment)

                self.logger.error(f"Deployment strategy failed: {version} to {environment.value}")

        except Exception as e:
            deployment.complete_deployment(PipelineStatus.FAILED, str(e))

            if rollback_on_failure:
                await self._rollback_deployment(deployment)

            self.logger.error(f"Deployment error: {e}")

        finally:
            self.active_deployments.discard(deployment.deployment_id)

        return deployment

    async def rollback(
        self,
        environment: DeploymentEnvironment,
        target_version: Optional[str] = None,
        rolled_back_by: Optional[UUID] = None,
    ) -> Optional[Deployment]:
        """Rollback to previous or specified version."""

        # Find target deployment
        if target_version:
            target_deployment = self._find_deployment_by_version(environment, target_version)
        else:
            target_deployment = self._find_previous_successful_deployment(environment)

        if not target_deployment:
            self.logger.error(f"No target deployment found for rollback in {environment.value}")
            return None

        # Create rollback deployment
        rollback_deployment = Deployment(
            deployment_id=uuid4(),
            environment=environment,
            strategy=DeploymentStrategy.ROLLING,  # Use safe rollback strategy
            version=target_deployment.version,
            commit_sha=target_deployment.commit_sha,
            branch=target_deployment.branch,
            deployed_by=rolled_back_by or uuid4(),
            deployment_notes=f"Rollback to version {target_deployment.version}",
            previous_version=self._get_current_version(environment),
        )

        self.deployments[rollback_deployment.deployment_id] = rollback_deployment

        # Execute rollback
        rollback_deployment.start_deployment()
        self.active_deployments.add(rollback_deployment.deployment_id)

        try:
            success = await self._rolling_deployment(rollback_deployment)

            if success:
                health_ok = await self._perform_health_checks(rollback_deployment)

                if health_ok:
                    rollback_deployment.complete_deployment(PipelineStatus.SUCCESS)
                    self.logger.info(f"Rollback successful: {target_deployment.version} in {environment.value}")
                else:
                    rollback_deployment.complete_deployment(PipelineStatus.FAILED, "Health checks failed")
                    self.logger.error(f"Rollback health checks failed in {environment.value}")
            else:
                rollback_deployment.complete_deployment(PipelineStatus.FAILED, "Rollback deployment failed")
                self.logger.error(f"Rollback deployment failed in {environment.value}")

        except Exception as e:
            rollback_deployment.complete_deployment(PipelineStatus.FAILED, str(e))
            self.logger.error(f"Rollback error: {e}")

        finally:
            self.active_deployments.discard(rollback_deployment.deployment_id)

        return rollback_deployment

    async def get_deployment_status(
        self,
        deployment_id: Optional[UUID] = None,
        environment: Optional[DeploymentEnvironment] = None,
    ) -> Dict[str, Any]:
        """Get deployment status."""

        if deployment_id:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                return {"error": "Deployment not found"}

            return {
                "deployment": self._get_deployment_summary(deployment),
                "is_active": deployment_id in self.active_deployments,
            }

        elif environment:
            # Get current deployment for environment
            current_deployment = self._get_current_deployment(environment)

            if not current_deployment:
                return {"error": f"No deployment found for {environment.value}"}

            return {
                "current_deployment": self._get_deployment_summary(current_deployment),
                "environment": environment.value,
                "is_active": current_deployment.deployment_id in self.active_deployments,
                "deployment_history": [
                    self._get_deployment_summary(d)
                    for d in self._get_deployment_history(environment, limit=10)
                ],
            }

        else:
            # Get overall status
            return {
                "active_deployments": len(self.active_deployments),
                "total_deployments": len(self.deployments),
                "by_environment": {
                    env.value: len([
                        d for d in self.deployments.values()
                        if d.environment == env
                    ])
                    for env in DeploymentEnvironment
                },
            }

    async def _rolling_deployment(self, deployment: Deployment) -> bool:
        """Execute rolling deployment strategy."""

        try:
            # Simulate rolling deployment
            self.logger.info(f"Starting rolling deployment: {deployment.version}")

            # Update deployment configuration
            await self._update_kubernetes_deployment(deployment)

            # Wait for rollout to complete
            await self._wait_for_rollout(deployment)

            return True

        except Exception as e:
            self.logger.error(f"Rolling deployment failed: {e}")
            return False

    async def _blue_green_deployment(self, deployment: Deployment) -> bool:
        """Execute blue-green deployment strategy."""

        try:
            self.logger.info(f"Starting blue-green deployment: {deployment.version}")

            # Deploy to green environment
            await self._deploy_green_environment(deployment)

            # Validate green environment
            green_healthy = await self._validate_green_environment(deployment)

            if green_healthy:
                # Switch traffic to green
                await self._switch_traffic_to_green(deployment)

                # Schedule blue environment cleanup
                await self._schedule_blue_cleanup(deployment)

                return True
            else:
                # Cleanup failed green deployment
                await self._cleanup_green_environment(deployment)
                return False

        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            return False

    async def _canary_deployment(self, deployment: Deployment) -> bool:
        """Execute canary deployment strategy."""

        try:
            self.logger.info(f"Starting canary deployment: {deployment.version}")

            # Deploy canary version
            await self._deploy_canary_version(deployment)

            # Gradually increase traffic
            success = await self._execute_canary_rollout(deployment)

            if success:
                # Complete canary rollout
                await self._complete_canary_rollout(deployment)
                return True
            else:
                # Rollback canary
                await self._rollback_canary(deployment)
                return False

        except Exception as e:
            self.logger.error(f"Canary deployment failed: {e}")
            return False

    async def _recreate_deployment(self, deployment: Deployment) -> bool:
        """Execute recreate deployment strategy."""

        try:
            self.logger.info(f"Starting recreate deployment: {deployment.version}")

            # Stop existing deployment
            await self._stop_existing_deployment(deployment)

            # Wait for complete shutdown
            await asyncio.sleep(30)

            # Start new deployment
            await self._start_new_deployment(deployment)

            return True

        except Exception as e:
            self.logger.error(f"Recreate deployment failed: {e}")
            return False

    async def _ab_testing_deployment(self, deployment: Deployment) -> bool:
        """Execute A/B testing deployment strategy."""

        try:
            self.logger.info(f"Starting A/B testing deployment: {deployment.version}")

            # Deploy B version
            await self._deploy_b_version(deployment)

            # Configure traffic splitting
            await self._configure_ab_traffic_split(deployment)

            # Monitor A/B test metrics
            success = await self._monitor_ab_test(deployment)

            return success

        except Exception as e:
            self.logger.error(f"A/B testing deployment failed: {e}")
            return False

    async def _perform_health_checks(self, deployment: Deployment) -> bool:
        """Perform health checks on deployment."""

        if not deployment.health_check_url:
            self.logger.warning("No health check URL configured")
            return True

        config = self.health_check_configs.get(
            deployment.environment,
            {"timeout": 60, "retries": 5}
        )

        for attempt in range(config["retries"]):
            try:
                # Simulate health check (would use actual HTTP client in production)
                await asyncio.sleep(2)  # Simulate network delay

                # In production, would make actual HTTP request
                health_status = await self._check_health_endpoint(deployment.health_check_url)

                if health_status:
                    deployment.success_rate = 100.0
                    deployment.error_rate = 0.0
                    deployment.response_time_p95 = 250.0  # Simulated

                    self.logger.info(f"Health check passed for {deployment.version}")
                    return True

            except Exception as e:
                self.logger.warning(f"Health check attempt {attempt + 1} failed: {e}")

            await asyncio.sleep(5)  # Wait before retry

        self.logger.error(f"All health check attempts failed for {deployment.version}")
        return False

    async def _check_health_endpoint(self, url: str) -> bool:
        """Check health endpoint (simulated)."""

        # In production, would make actual HTTP request
        # For now, simulate success
        return True

    async def _update_kubernetes_deployment(self, deployment: Deployment) -> None:
        """Update Kubernetes deployment (simulated)."""

        # In production, would use Kubernetes client
        self.logger.info(f"Updating Kubernetes deployment: {deployment.version}")
        await asyncio.sleep(2)  # Simulate API call

    async def _wait_for_rollout(self, deployment: Deployment) -> None:
        """Wait for deployment rollout to complete."""

        # Simulate rollout wait
        await asyncio.sleep(30)  # Simulate rollout time

    async def _deployment_monitor_loop(self) -> None:
        """Background task for monitoring deployments."""

        while self.is_running:
            try:
                for deployment_id in list(self.active_deployments):
                    deployment = self.deployments.get(deployment_id)
                    if deployment:
                        await self._monitor_deployment(deployment)

            except Exception as e:
                self.logger.error(f"Deployment monitoring error: {e}")

            await asyncio.sleep(30)  # Monitor every 30 seconds

    async def _monitor_deployment(self, deployment: Deployment) -> None:
        """Monitor individual deployment."""

        # Check if deployment is taking too long
        if deployment.start_time:
            elapsed = datetime.utcnow() - deployment.start_time

            # Timeout based on environment
            timeout_minutes = {
                DeploymentEnvironment.DEVELOPMENT: 10,
                DeploymentEnvironment.TESTING: 20,
                DeploymentEnvironment.STAGING: 30,
                DeploymentEnvironment.PRODUCTION: 60,
                DeploymentEnvironment.CANARY: 45,
            }.get(deployment.environment, 30)

            if elapsed.total_seconds() > timeout_minutes * 60:
                self.logger.warning(f"Deployment timeout: {deployment.version} in {deployment.environment.value}")
                deployment.complete_deployment(PipelineStatus.TIMEOUT, "Deployment timeout")
                self.active_deployments.discard(deployment.deployment_id)

    async def _health_check_loop(self) -> None:
        """Background task for health checking active deployments."""

        while self.is_running:
            try:
                current_deployments = [
                    self._get_current_deployment(env)
                    for env in DeploymentEnvironment
                ]

                for deployment in current_deployments:
                    if deployment and deployment.status == PipelineStatus.SUCCESS:
                        await self._perform_health_checks(deployment)

            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")

            await asyncio.sleep(60)  # Health check every minute

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old deployments."""

        while self.is_running:
            try:
                await self._cleanup_old_deployments()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

            await asyncio.sleep(3600)  # Cleanup every hour

    async def _cleanup_old_deployments(self) -> None:
        """Clean up old deployment records."""

        cutoff_time = datetime.utcnow() - timedelta(days=90)

        deployments_to_remove = []
        for deployment_id, deployment in self.deployments.items():
            if (deployment.start_time and
                deployment.start_time < cutoff_time and
                deployment_id not in self.active_deployments):
                deployments_to_remove.append(deployment_id)

        for deployment_id in deployments_to_remove:
            del self.deployments[deployment_id]

        if deployments_to_remove:
            self.logger.info(f"Cleaned up {len(deployments_to_remove)} old deployment records")

    def _get_current_deployment(self, environment: DeploymentEnvironment) -> Optional[Deployment]:
        """Get current successful deployment for environment."""

        env_deployments = [
            d for d in self.deployments.values()
            if d.environment == environment and d.status == PipelineStatus.SUCCESS
        ]

        if not env_deployments:
            return None

        return max(env_deployments, key=lambda d: d.start_time or datetime.min)

    def _find_previous_successful_deployment(self, environment: DeploymentEnvironment) -> Optional[Deployment]:
        """Find previous successful deployment for rollback."""

        env_deployments = [
            d for d in self.deployments.values()
            if (d.environment == environment and
                d.status == PipelineStatus.SUCCESS and
                d.deployment_id not in self.active_deployments)
        ]

        if len(env_deployments) < 2:
            return None

        # Return second most recent
        sorted_deployments = sorted(env_deployments, key=lambda d: d.start_time or datetime.min, reverse=True)
        return sorted_deployments[1] if len(sorted_deployments) > 1 else None

    def _get_deployment_summary(self, deployment: Deployment) -> Dict[str, Any]:
        """Get deployment summary."""

        return {
            "deployment_id": str(deployment.deployment_id),
            "environment": deployment.environment.value,
            "strategy": deployment.strategy.value,
            "version": deployment.version,
            "commit_sha": deployment.commit_sha,
            "branch": deployment.branch,
            "status": deployment.status.value,
            "start_time": deployment.start_time.isoformat() if deployment.start_time else None,
            "end_time": deployment.end_time.isoformat() if deployment.end_time else None,
            "duration_seconds": deployment.duration_seconds,
            "success_rate": deployment.success_rate,
            "error_rate": deployment.error_rate,
            "response_time_p95": deployment.response_time_p95,
            "is_healthy": deployment.is_healthy(),
            "deployment_url": deployment.deployment_url,
        }

    def _get_deployment_history(
        self,
        environment: DeploymentEnvironment,
        limit: int = 10,
    ) -> List[Deployment]:
        """Get deployment history for environment."""

        env_deployments = [
            d for d in self.deployments.values()
            if d.environment == environment
        ]

        # Sort by start time (newest first)
        sorted_deployments = sorted(
            env_deployments,
            key=lambda d: d.start_time or datetime.min,
            reverse=True
        )

        return sorted_deployments[:limit]

    def _get_base_domain(self, environment: DeploymentEnvironment) -> str:
        """Get base domain for environment."""

        domain_map = {
            DeploymentEnvironment.DEVELOPMENT: "dev.pynomaly.ai",
            DeploymentEnvironment.TESTING: "test.pynomaly.ai",
            DeploymentEnvironment.STAGING: "staging.pynomaly.ai",
            DeploymentEnvironment.PRODUCTION: "pynomaly.ai",
            DeploymentEnvironment.CANARY: "canary.pynomaly.ai",
            DeploymentEnvironment.PREVIEW: "preview.pynomaly.ai",
        }

        return domain_map.get(environment, "localhost")

    # Placeholder methods for deployment strategies (would be implemented with actual cloud providers)

    async def _deploy_green_environment(self, deployment: Deployment) -> None:
        await asyncio.sleep(5)  # Simulate deployment

    async def _validate_green_environment(self, deployment: Deployment) -> bool:
        await asyncio.sleep(2)  # Simulate validation
        return True

    async def _switch_traffic_to_green(self, deployment: Deployment) -> None:
        await asyncio.sleep(1)  # Simulate traffic switch

    async def _schedule_blue_cleanup(self, deployment: Deployment) -> None:
        pass  # Would schedule cleanup task

    async def _cleanup_green_environment(self, deployment: Deployment) -> None:
        pass  # Would cleanup failed green deployment

    async def _deploy_canary_version(self, deployment: Deployment) -> None:
        await asyncio.sleep(3)  # Simulate canary deployment

    async def _execute_canary_rollout(self, deployment: Deployment) -> bool:
        await asyncio.sleep(10)  # Simulate gradual rollout
        return True

    async def _complete_canary_rollout(self, deployment: Deployment) -> None:
        pass  # Complete canary rollout

    async def _rollback_canary(self, deployment: Deployment) -> None:
        pass  # Rollback canary deployment

    async def _stop_existing_deployment(self, deployment: Deployment) -> None:
        await asyncio.sleep(5)  # Simulate stopping

    async def _start_new_deployment(self, deployment: Deployment) -> None:
        await asyncio.sleep(10)  # Simulate starting

    async def _deploy_b_version(self, deployment: Deployment) -> None:
        await asyncio.sleep(5)  # Simulate B version deployment

    async def _configure_ab_traffic_split(self, deployment: Deployment) -> None:
        pass  # Configure A/B traffic split

    async def _monitor_ab_test(self, deployment: Deployment) -> bool:
        await asyncio.sleep(20)  # Simulate A/B monitoring
        return True

    async def _rollback_deployment(self, deployment: Deployment) -> None:
        """Perform automatic rollback on failure."""

        self.logger.info(f"Initiating rollback for failed deployment: {deployment.version}")

        # Find previous successful deployment
        previous = self._find_previous_successful_deployment(deployment.environment)

        if previous:
            await self.rollback(
                environment=deployment.environment,
                target_version=previous.version,
                rolled_back_by=deployment.deployed_by,
            )
        else:
            self.logger.warning(f"No previous deployment found for rollback in {deployment.environment.value}")

    async def _update_deployment_metrics(self, deployment: Deployment) -> None:
        """Update deployment metrics."""

        # Would integrate with metrics system
        pass

    def _get_current_version(self, environment: DeploymentEnvironment) -> Optional[str]:
        """Get current deployed version for environment."""

        current = self._get_current_deployment(environment)
        return current.version if current else None

    def _find_deployment_by_version(
        self,
        environment: DeploymentEnvironment,
        version: str,
    ) -> Optional[Deployment]:
        """Find deployment by version in environment."""

        for deployment in self.deployments.values():
            if (deployment.environment == environment and
                deployment.version == version and
                deployment.status == PipelineStatus.SUCCESS):
                return deployment

        return None
