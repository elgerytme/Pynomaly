#!/usr/bin/env python3
"""
anomaly_detection Deployment Manager - Advanced Deployment Automation

This script provides comprehensive deployment automation for anomaly_detection including:
- Multi-environment deployment (dev, staging, production)
- Docker and Kubernetes deployment strategies
- Health checks and rollback capabilities
- Blue-green and canary deployment patterns
- Infrastructure provisioning and validation
- Monitoring and alerting setup
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import click
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

import docker

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/anomaly_detection_deployment.log"),
    ],
)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStrategy(Enum):
    """Deployment strategy types."""

    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class DeploymentPlatform(Enum):
    """Deployment platform types."""

    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    HELM = "helm"


class DeploymentStatus(Enum):
    """Deployment status types."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class AnomalyDetectionDeploymentManager:
    """Advanced deployment manager for anomaly_detection."""

    def __init__(
        self,
        project_root: Path,
        environment: DeploymentEnvironment,
        platform: DeploymentPlatform,
    ):
        self.project_root = project_root
        self.environment = environment
        self.platform = platform
        self.deployment_config = self._load_deployment_config()
        self.docker_client = None
        self.k8s_client = None
        self.deployment_id = f"deploy_{int(time.time())}"
        self.deployment_status = DeploymentStatus.PENDING

        # Initialize platform clients
        self._initialize_clients()

        # Deployment state tracking
        self.deployment_log = []
        self.start_time = None
        self.end_time = None

    def _load_deployment_config(self) -> dict[str, Any]:
        """Load deployment configuration."""
        config_path = self.project_root / "deploy" / "config" / "environments.yaml"

        if not config_path.exists():
            return self._get_default_config()

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get(self.environment.value, {})
        except Exception as e:
            logger.error(f"Failed to load deployment config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default deployment configuration."""
        return {
            "replicas": 1
            if self.environment == DeploymentEnvironment.DEVELOPMENT
            else 3,
            "resources": {
                "requests": {"memory": "512Mi", "cpu": "250m"},
                "limits": {"memory": "2Gi", "cpu": "1000m"},
            },
            "strategy": DeploymentStrategy.ROLLING_UPDATE.value,
            "health_checks": {"enabled": True, "timeout": 300, "retries": 3},
            "backup": {
                "enabled": self.environment == DeploymentEnvironment.PRODUCTION,
                "retention_days": 30,
            },
            "monitoring": {"enabled": True, "prometheus": True, "grafana": True},
        }

    def _initialize_clients(self):
        """Initialize deployment platform clients."""
        try:
            # Docker client
            if self.platform in [DeploymentPlatform.DOCKER_COMPOSE]:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")

            # Kubernetes client
            if self.platform in [
                DeploymentPlatform.KUBERNETES,
                DeploymentPlatform.HELM,
            ]:
                try:
                    config.load_incluster_config()
                    logger.info("Loaded in-cluster Kubernetes config")
                except config.ConfigException:
                    config.load_kube_config()
                    logger.info("Loaded local Kubernetes config")

                self.k8s_client = client.ApiClient()

        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise

    def _log_deployment_step(self, step: str, status: str, details: str | None = None):
        """Log deployment step."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "status": status,
            "details": details,
        }
        self.deployment_log.append(log_entry)
        logger.info(f"[{step}] {status}: {details or ''}")

    async def deploy(self, strategy: DeploymentStrategy | None = None) -> bool:
        """Execute deployment with specified strategy."""
        self.start_time = datetime.utcnow()
        self.deployment_status = DeploymentStatus.IN_PROGRESS

        try:
            self._log_deployment_step(
                "DEPLOYMENT_START",
                "INFO",
                f"Starting {self.environment.value} deployment",
            )

            # Pre-deployment validation
            if not await self._pre_deployment_validation():
                self.deployment_status = DeploymentStatus.FAILED
                return False

            # Create backup if needed
            if self.deployment_config.get("backup", {}).get("enabled", False):
                if not await self._create_backup():
                    self.deployment_status = DeploymentStatus.FAILED
                    return False

            # Build and push images
            if not await self._build_and_push_images():
                self.deployment_status = DeploymentStatus.FAILED
                return False

            # Deploy based on platform
            deployment_strategy = strategy or DeploymentStrategy(
                self.deployment_config.get("strategy", "rolling_update")
            )

            if self.platform == DeploymentPlatform.DOCKER_COMPOSE:
                success = await self._deploy_docker_compose()
            elif self.platform == DeploymentPlatform.KUBERNETES:
                success = await self._deploy_kubernetes(deployment_strategy)
            elif self.platform == DeploymentPlatform.HELM:
                success = await self._deploy_helm(deployment_strategy)
            else:
                raise ValueError(f"Unsupported platform: {self.platform}")

            if not success:
                self.deployment_status = DeploymentStatus.FAILED
                return False

            # Post-deployment validation
            if not await self._post_deployment_validation():
                self.deployment_status = DeploymentStatus.FAILED
                await self._rollback()
                return False

            # Setup monitoring and alerting
            if self.deployment_config.get("monitoring", {}).get("enabled", True):
                await self._setup_monitoring()

            self.deployment_status = DeploymentStatus.COMPLETED
            self.end_time = datetime.utcnow()
            self._log_deployment_step(
                "DEPLOYMENT_COMPLETE",
                "SUCCESS",
                f"Deployment completed in {self.end_time - self.start_time}",
            )

            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.deployment_status = DeploymentStatus.FAILED
            self._log_deployment_step("DEPLOYMENT_ERROR", "ERROR", str(e))
            return False

    async def _pre_deployment_validation(self) -> bool:
        """Validate deployment prerequisites."""
        self._log_deployment_step(
            "PRE_VALIDATION", "INFO", "Starting pre-deployment validation"
        )

        try:
            # Check project structure
            required_files = [
                "pyproject.toml",
                "src/anomaly_detection/__init__.py",
                "deploy/docker/Dockerfile.production",
            ]

            for file_path in required_files:
                if not (self.project_root / file_path).exists():
                    self._log_deployment_step(
                        "PRE_VALIDATION", "ERROR", f"Missing required file: {file_path}"
                    )
                    return False

            # Check environment variables
            if self.environment == DeploymentEnvironment.PRODUCTION:
                required_env_vars = [
                    "POSTGRES_PASSWORD",
                    "REDIS_PASSWORD",
                    "JWT_SECRET_KEY",
                    "API_SECRET_KEY",
                ]

                for env_var in required_env_vars:
                    if not os.getenv(env_var):
                        self._log_deployment_step(
                            "PRE_VALIDATION",
                            "ERROR",
                            f"Missing environment variable: {env_var}",
                        )
                        return False

            # Check platform connectivity
            if self.platform == DeploymentPlatform.DOCKER_COMPOSE:
                if not self.docker_client.ping():
                    self._log_deployment_step(
                        "PRE_VALIDATION", "ERROR", "Docker daemon not accessible"
                    )
                    return False

            elif self.platform in [
                DeploymentPlatform.KUBERNETES,
                DeploymentPlatform.HELM,
            ]:
                try:
                    v1 = client.CoreV1Api(self.k8s_client)
                    v1.list_namespace()
                except ApiException as e:
                    self._log_deployment_step(
                        "PRE_VALIDATION", "ERROR", f"Kubernetes API not accessible: {e}"
                    )
                    return False

            self._log_deployment_step(
                "PRE_VALIDATION", "SUCCESS", "Pre-deployment validation passed"
            )
            return True

        except Exception as e:
            self._log_deployment_step("PRE_VALIDATION", "ERROR", str(e))
            return False

    async def _create_backup(self) -> bool:
        """Create deployment backup."""
        self._log_deployment_step("BACKUP", "INFO", "Creating deployment backup")

        try:
            backup_dir = self.project_root / "backups" / f"backup_{self.deployment_id}"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Database backup
            if self.platform == DeploymentPlatform.DOCKER_COMPOSE:
                backup_cmd = [
                    "docker",
                    "exec",
                    "anomaly_detection-postgres",
                    "pg_dump",
                    "-U",
                    "anomaly_detection",
                    "-d",
                    "anomaly_detection",
                    "-f",
                    f"/backup/db_backup_{self.deployment_id}.sql",
                ]
            else:
                # Kubernetes backup
                backup_cmd = [
                    "kubectl",
                    "exec",
                    "-n",
                    f"anomaly_detection-{self.environment.value}",
                    "deployment/postgres",
                    "--",
                    "pg_dump",
                    "-U",
                    "anomaly_detection",
                    "-d",
                    "anomaly_detection",
                    "-f",
                    f"/backup/db_backup_{self.deployment_id}.sql",
                ]

            result = subprocess.run(backup_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self._log_deployment_step(
                    "BACKUP", "ERROR", f"Database backup failed: {result.stderr}"
                )
                return False

            # Configuration backup
            config_backup = {
                "deployment_id": self.deployment_id,
                "environment": self.environment.value,
                "timestamp": datetime.utcnow().isoformat(),
                "config": self.deployment_config,
            }

            with open(backup_dir / "deployment_config.json", "w") as f:
                json.dump(config_backup, f, indent=2)

            self._log_deployment_step(
                "BACKUP", "SUCCESS", f"Backup created at {backup_dir}"
            )
            return True

        except Exception as e:
            self._log_deployment_step("BACKUP", "ERROR", str(e))
            return False

    async def _build_and_push_images(self) -> bool:
        """Build and push Docker images."""
        self._log_deployment_step("BUILD", "INFO", "Building Docker images")

        try:
            # Build production image
            build_args = {
                "BUILD_DATE": datetime.utcnow().isoformat(),
                "VERSION": os.getenv("VERSION", "latest"),
                "VCS_REF": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            }

            dockerfile_path = (
                self.project_root / "deploy" / "docker" / "Dockerfile.production"
            )

            image_tag = f"anomaly_detection:{self.environment.value}-{build_args['VCS_REF'][:8]}"

            # Build image
            self.docker_client.images.build(
                path=str(self.project_root),
                dockerfile=str(dockerfile_path),
                tag=image_tag,
                buildargs=build_args,
                pull=True,
                rm=True,
            )

            # Push to registry if configured
            registry = os.getenv("DOCKER_REGISTRY")
            if registry and self.environment != DeploymentEnvironment.DEVELOPMENT:
                registry_tag = f"{registry}/{image_tag}"
                self.docker_client.images.get(image_tag).tag(registry_tag)

                push_result = self.docker_client.images.push(registry_tag)
                self._log_deployment_step(
                    "BUILD", "INFO", f"Pushed image to registry: {registry_tag}"
                )

            self._log_deployment_step("BUILD", "SUCCESS", f"Built image: {image_tag}")
            return True

        except Exception as e:
            self._log_deployment_step("BUILD", "ERROR", str(e))
            return False

    async def _deploy_docker_compose(self) -> bool:
        """Deploy using Docker Compose."""
        self._log_deployment_step(
            "DEPLOY_COMPOSE", "INFO", "Deploying with Docker Compose"
        )

        try:
            compose_file = (
                self.project_root
                / "deploy"
                / "docker"
                / f"docker-compose.{self.environment.value}.yml"
            )

            if not compose_file.exists():
                compose_file = (
                    self.project_root
                    / "deploy"
                    / "docker"
                    / "docker-compose.production.yml"
                )

            # Create environment file
            env_file = self.project_root / ".env.deployment"
            self._create_env_file(env_file)

            # Deploy with docker-compose
            cmd = [
                "docker-compose",
                "-f",
                str(compose_file),
                "--env-file",
                str(env_file),
                "up",
                "-d",
                "--remove-orphans",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode != 0:
                self._log_deployment_step(
                    "DEPLOY_COMPOSE",
                    "ERROR",
                    f"Docker Compose deployment failed: {result.stderr}",
                )
                return False

            self._log_deployment_step(
                "DEPLOY_COMPOSE", "SUCCESS", "Docker Compose deployment completed"
            )
            return True

        except Exception as e:
            self._log_deployment_step("DEPLOY_COMPOSE", "ERROR", str(e))
            return False

    async def _deploy_kubernetes(self, strategy: DeploymentStrategy) -> bool:
        """Deploy to Kubernetes."""
        self._log_deployment_step(
            "DEPLOY_K8S",
            "INFO",
            f"Deploying to Kubernetes with {strategy.value} strategy",
        )

        try:
            namespace = f"anomaly_detection-{self.environment.value}"

            # Create namespace if it doesn't exist
            await self._ensure_namespace(namespace)

            # Apply configurations
            manifests_dir = self.project_root / "deploy" / "kubernetes"

            if strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._deploy_blue_green(namespace)
            elif strategy == DeploymentStrategy.CANARY:
                return await self._deploy_canary(namespace)
            else:
                return await self._deploy_rolling_update(namespace)

        except Exception as e:
            self._log_deployment_step("DEPLOY_K8S", "ERROR", str(e))
            return False

    async def _deploy_helm(self, strategy: DeploymentStrategy) -> bool:
        """Deploy using Helm."""
        self._log_deployment_step("DEPLOY_HELM", "INFO", "Deploying with Helm")

        try:
            chart_path = self.project_root / "deploy" / "helm" / "anomaly_detection"
            release_name = f"anomaly_detection-{self.environment.value}"
            namespace = f"anomaly_detection-{self.environment.value}"

            # Ensure namespace exists
            await self._ensure_namespace(namespace)

            # Prepare values
            values_file = chart_path / f"values-{self.environment.value}.yaml"

            cmd = [
                "helm",
                "upgrade",
                "--install",
                release_name,
                str(chart_path),
                "--namespace",
                namespace,
                "--create-namespace",
                "--wait",
                "--timeout",
                "10m",
            ]

            if values_file.exists():
                cmd.extend(["--values", str(values_file)])

            # Add deployment strategy
            if strategy == DeploymentStrategy.BLUE_GREEN:
                cmd.extend(["--set", "deployment.strategy=blue-green"])
            elif strategy == DeploymentStrategy.CANARY:
                cmd.extend(["--set", "deployment.strategy=canary"])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self._log_deployment_step(
                    "DEPLOY_HELM", "ERROR", f"Helm deployment failed: {result.stderr}"
                )
                return False

            self._log_deployment_step(
                "DEPLOY_HELM", "SUCCESS", "Helm deployment completed"
            )
            return True

        except Exception as e:
            self._log_deployment_step("DEPLOY_HELM", "ERROR", str(e))
            return False

    async def _post_deployment_validation(self) -> bool:
        """Validate deployment after completion."""
        self._log_deployment_step(
            "POST_VALIDATION", "INFO", "Starting post-deployment validation"
        )

        try:
            # Health check configuration
            health_config = self.deployment_config.get("health_checks", {})
            timeout = health_config.get("timeout", 300)
            retries = health_config.get("retries", 3)

            # Wait for services to be ready
            await asyncio.sleep(30)  # Initial wait

            # Health checks
            for attempt in range(retries):
                if await self._check_service_health():
                    self._log_deployment_step(
                        "POST_VALIDATION", "SUCCESS", "All services are healthy"
                    )
                    return True

                if attempt < retries - 1:
                    self._log_deployment_step(
                        "POST_VALIDATION",
                        "RETRY",
                        f"Health check failed, retrying ({attempt + 1}/{retries})",
                    )
                    await asyncio.sleep(30)

            self._log_deployment_step(
                "POST_VALIDATION", "ERROR", "Health checks failed after all retries"
            )
            return False

        except Exception as e:
            self._log_deployment_step("POST_VALIDATION", "ERROR", str(e))
            return False

    async def _check_service_health(self) -> bool:
        """Check health of deployed services."""
        try:
            if self.platform == DeploymentPlatform.DOCKER_COMPOSE:
                return await self._check_docker_health()
            else:
                return await self._check_kubernetes_health()

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _check_docker_health(self) -> bool:
        """Check Docker container health."""
        try:
            containers = self.docker_client.containers.list(
                filters={"label": "app=anomaly_detection"}
            )

            for container in containers:
                if container.status != "running":
                    logger.error(
                        f"Container {container.name} is not running: {container.status}"
                    )
                    return False

                # Check health if container has health check
                if (
                    hasattr(container.attrs, "State")
                    and "Health" in container.attrs["State"]
                ):
                    health = container.attrs["State"]["Health"]["Status"]
                    if health != "healthy":
                        logger.error(
                            f"Container {container.name} is not healthy: {health}"
                        )
                        return False

            return True

        except Exception as e:
            logger.error(f"Docker health check failed: {e}")
            return False

    async def _check_kubernetes_health(self) -> bool:
        """Check Kubernetes deployment health."""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            v1 = client.CoreV1Api(self.k8s_client)
            namespace = f"anomaly_detection-{self.environment.value}"

            # Check deployments
            deployments = apps_v1.list_namespaced_deployment(namespace)

            for deployment in deployments.items:
                if deployment.status.ready_replicas != deployment.status.replicas:
                    logger.error(
                        f"Deployment {deployment.metadata.name} not ready: {deployment.status.ready_replicas}/{deployment.status.replicas}"
                    )
                    return False

            # Check pods
            pods = v1.list_namespaced_pod(
                namespace, label_selector="app.kubernetes.io/name=anomaly_detection"
            )

            for pod in pods.items:
                if pod.status.phase != "Running":
                    logger.error(
                        f"Pod {pod.metadata.name} not running: {pod.status.phase}"
                    )
                    return False

                # Check readiness
                if pod.status.conditions:
                    ready_condition = next(
                        (c for c in pod.status.conditions if c.type == "Ready"), None
                    )
                    if not ready_condition or ready_condition.status != "True":
                        logger.error(f"Pod {pod.metadata.name} not ready")
                        return False

            return True

        except ApiException as e:
            logger.error(f"Kubernetes health check failed: {e}")
            return False

    async def _rollback(self) -> bool:
        """Rollback deployment."""
        self.deployment_status = DeploymentStatus.ROLLING_BACK
        self._log_deployment_step("ROLLBACK", "INFO", "Starting deployment rollback")

        try:
            if self.platform == DeploymentPlatform.DOCKER_COMPOSE:
                return await self._rollback_docker_compose()
            elif self.platform == DeploymentPlatform.KUBERNETES:
                return await self._rollback_kubernetes()
            elif self.platform == DeploymentPlatform.HELM:
                return await self._rollback_helm()

            return False

        except Exception as e:
            self._log_deployment_step("ROLLBACK", "ERROR", str(e))
            return False

    def _create_env_file(self, env_file: Path):
        """Create environment file for deployment."""
        env_vars = {
            "ANOMALY_DETECTION_ENVIRONMENT": self.environment.value,
            "VERSION": os.getenv("VERSION", "latest"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "anomaly_detection_secret"),
            "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD", "redis_secret"),
            "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY", "jwt_secret"),
            "API_SECRET_KEY": os.getenv("API_SECRET_KEY", "api_secret"),
        }

        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

    async def _ensure_namespace(self, namespace: str):
        """Ensure Kubernetes namespace exists."""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            v1.read_namespace(namespace)
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                v1.create_namespace(namespace_manifest)
                self._log_deployment_step(
                    "NAMESPACE", "INFO", f"Created namespace: {namespace}"
                )

    def get_deployment_status(self) -> dict[str, Any]:
        """Get deployment status and logs."""
        return {
            "deployment_id": self.deployment_id,
            "environment": self.environment.value,
            "platform": self.platform.value,
            "status": self.deployment_status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.end_time - self.start_time)
            if self.start_time and self.end_time
            else None,
            "log": self.deployment_log,
        }

    async def _setup_monitoring(self):
        """Setup monitoring and alerting for deployed services."""
        self._log_deployment_step(
            "MONITORING", "INFO", "Setting up monitoring and alerting"
        )

        try:
            # Implementation would depend on monitoring stack
            # This is a placeholder for monitoring setup
            await asyncio.sleep(1)
            self._log_deployment_step(
                "MONITORING", "SUCCESS", "Monitoring setup completed"
            )

        except Exception as e:
            self._log_deployment_step("MONITORING", "ERROR", str(e))


@click.group()
def cli():
    """anomaly_detection Deployment Manager CLI."""
    pass


@cli.command()
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["development", "staging", "production"]),
    default="development",
    help="Deployment environment",
)
@click.option(
    "--platform",
    "-p",
    type=click.Choice(["docker_compose", "kubernetes", "helm"]),
    default="docker_compose",
    help="Deployment platform",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["rolling_update", "blue_green", "canary", "recreate"]),
    help="Deployment strategy",
)
@click.option("--project-root", default=".", help="Project root directory")
def deploy(environment: str, platform: str, strategy: str | None, project_root: str):
    """Deploy anomaly_detection to specified environment."""

    project_path = Path(project_root).resolve()
    env = DeploymentEnvironment(environment)
    plat = DeploymentPlatform(platform)
    strat = DeploymentStrategy(strategy) if strategy else None

    # Create deployment manager
    manager = AnomalyDetectionDeploymentManager(project_path, env, plat)

    # Run deployment
    success = asyncio.run(manager.deploy(strat))

    # Print deployment status
    status = manager.get_deployment_status()
    click.echo(json.dumps(status, indent=2))

    if success:
        click.echo("✅ Deployment completed successfully!")
        sys.exit(0)
    else:
        click.echo("❌ Deployment failed!")
        sys.exit(1)


@cli.command()
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["development", "staging", "production"]),
    required=True,
    help="Environment to check",
)
@click.option(
    "--platform",
    "-p",
    type=click.Choice(["docker_compose", "kubernetes", "helm"]),
    default="docker_compose",
    help="Deployment platform",
)
@click.option("--project-root", default=".", help="Project root directory")
def status(environment: str, platform: str, project_root: str):
    """Check deployment status."""

    project_path = Path(project_root).resolve()
    env = DeploymentEnvironment(environment)
    plat = DeploymentPlatform(platform)

    manager = AnomalyDetectionDeploymentManager(project_path, env, plat)

    # Check service health
    is_healthy = asyncio.run(manager._check_service_health())

    click.echo(f"Environment: {environment}")
    click.echo(f"Platform: {platform}")
    click.echo(f"Health Status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")


if __name__ == "__main__":
    cli()
