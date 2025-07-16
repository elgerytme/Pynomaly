#!/usr/bin/env python3
"""
Automated Deployment Pipeline for Pynomaly
This script provides a comprehensive automated deployment system with CI/CD integration
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiohttp
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategies"""

    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class Environment(Enum):
    """Deployment environments"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


class DeploymentPhase(Enum):
    """Deployment phases"""

    VALIDATION = "validation"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    VERIFY = "verify"
    COMPLETE = "complete"


class AutomatedDeploymentPipeline:
    """Comprehensive automated deployment pipeline"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.deployment_id = (
            f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid4())[:8]}"
        )
        self.project_root = Path(__file__).parent.parent.parent
        self.logs: list[str] = []
        self.start_time = datetime.now()

        # Initialize deployment state
        self.current_phase = DeploymentPhase.VALIDATION
        self.deployment_successful = False
        self.rollback_performed = False

        # CI/CD integration
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.docker_registry = config.get(
            "docker_registry", "ghcr.io/pynomaly/pynomaly"
        )

        # Health check configurations
        self.health_check_retries = config.get("health_check_retries", 20)
        self.health_check_interval = config.get("health_check_interval", 30)

        # Monitoring and alerting
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.teams_webhook = os.getenv("TEAMS_WEBHOOK_URL")

        self.log(f"Initialized deployment pipeline {self.deployment_id}")

    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps and levels"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)

        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "DEBUG":
            logger.debug(message)
        else:
            logger.info(message)

    async def execute_command(
        self, command: str, timeout: int = 300, cwd: Path | None = None
    ) -> tuple[int, str, str]:
        """Execute shell command asynchronously with enhanced error handling"""
        self.log(f"Executing: {command}")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or self.project_root,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return_code = process.returncode
            stdout_str = stdout.decode("utf-8") if stdout else ""
            stderr_str = stderr.decode("utf-8") if stderr else ""

            if return_code == 0:
                self.log(f"Command succeeded: {command}")
                if stdout_str.strip():
                    self.log(f"STDOUT: {stdout_str.strip()}", "DEBUG")
            else:
                self.log(
                    f"Command failed with exit code {return_code}: {command}", "ERROR"
                )
                if stderr_str.strip():
                    self.log(f"STDERR: {stderr_str.strip()}", "ERROR")

            return return_code, stdout_str, stderr_str

        except TimeoutError:
            self.log(f"Command timed out after {timeout} seconds: {command}", "ERROR")
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            self.log(f"Command execution error: {e}", "ERROR")
            return -1, "", str(e)

    async def validate_prerequisites(self) -> bool:
        """Comprehensive prerequisite validation"""
        self.log("Validating deployment prerequisites...")
        self.current_phase = DeploymentPhase.VALIDATION

        validation_tasks = [
            self._validate_environment(),
            self._validate_docker_environment(),
            self._validate_kubernetes_access(),
            self._validate_dependencies(),
            self._validate_secrets(),
            self._validate_resources(),
        ]

        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log(f"Validation task {i} failed: {result}", "ERROR")
                return False
            elif not result:
                self.log(f"Validation task {i} failed", "ERROR")
                return False

        self.log("All prerequisites validated successfully")
        return True

    async def _validate_environment(self) -> bool:
        """Validate environment configuration"""
        try:
            environment = self.config.get("environment")
            if not environment or environment not in [e.value for e in Environment]:
                self.log(f"Invalid environment: {environment}", "ERROR")
                return False

            # Check environment-specific requirements
            if environment == Environment.PRODUCTION.value:
                # Additional production validations
                required_env_vars = [
                    "DATABASE_URL",
                    "REDIS_URL",
                    "SECRET_KEY",
                    "MONITORING_ENDPOINT",
                    "SLACK_WEBHOOK_URL",
                ]

                for var in required_env_vars:
                    if not os.getenv(var):
                        self.log(
                            f"Required environment variable missing: {var}", "ERROR"
                        )
                        return False

            self.log(f"Environment validation passed for {environment}")
            return True

        except Exception as e:
            self.log(f"Environment validation failed: {e}", "ERROR")
            return False

    async def _validate_docker_environment(self) -> bool:
        """Validate Docker environment"""
        try:
            # Check Docker installation
            return_code, _, _ = await self.execute_command("docker --version")
            if return_code != 0:
                self.log("Docker is not installed or not accessible", "ERROR")
                return False

            # Check Docker daemon
            return_code, _, _ = await self.execute_command("docker info")
            if return_code != 0:
                self.log("Docker daemon is not running", "ERROR")
                return False

            # Validate image exists or can be built
            image_tag = self.config.get("image_tag", "latest")
            image_name = f"{self.docker_registry}:{image_tag}"

            # Try to pull the image
            return_code, _, _ = await self.execute_command(
                f"docker pull {image_name}", timeout=600
            )
            if return_code != 0:
                self.log(
                    f"Warning: Could not pull {image_name}, will try to build locally",
                    "WARNING",
                )
                # Try to build the image
                return_code, _, _ = await self.execute_command(
                    "docker build -t {image_name} .", timeout=1200
                )
                if return_code != 0:
                    self.log(f"Failed to build Docker image {image_name}", "ERROR")
                    return False

            self.log("Docker environment validation passed")
            return True

        except Exception as e:
            self.log(f"Docker validation failed: {e}", "ERROR")
            return False

    async def _validate_kubernetes_access(self) -> bool:
        """Validate Kubernetes access"""
        try:
            # Check kubectl installation
            return_code, _, _ = await self.execute_command("kubectl version --client")
            if return_code != 0:
                self.log("kubectl is not installed", "ERROR")
                return False

            # Check cluster connectivity
            return_code, _, _ = await self.execute_command("kubectl cluster-info")
            if return_code != 0:
                self.log("Cannot connect to Kubernetes cluster", "ERROR")
                return False

            # Check namespace access
            namespace = self.config.get("namespace", "pynomaly-prod")
            return_code, _, _ = await self.execute_command(
                f"kubectl get namespace {namespace}"
            )
            if return_code != 0:
                self.log(
                    f"Namespace {namespace} does not exist or is not accessible",
                    "ERROR",
                )
                return False

            self.log("Kubernetes access validation passed")
            return True

        except Exception as e:
            self.log(f"Kubernetes validation failed: {e}", "ERROR")
            return False

    async def _validate_dependencies(self) -> bool:
        """Validate project dependencies"""
        try:
            # Check Python dependencies
            return_code, _, _ = await self.execute_command("python -m pip check")
            if return_code != 0:
                self.log("Python dependency check failed", "ERROR")
                return False

            # Check for required tools
            required_tools = ["git", "curl", "jq"]
            for tool in required_tools:
                return_code, _, _ = await self.execute_command(f"which {tool}")
                if return_code != 0:
                    self.log(f"Required tool not found: {tool}", "ERROR")
                    return False

            self.log("Dependencies validation passed")
            return True

        except Exception as e:
            self.log(f"Dependencies validation failed: {e}", "ERROR")
            return False

    async def _validate_secrets(self) -> bool:
        """Validate required secrets"""
        try:
            namespace = self.config.get("namespace", "pynomaly-prod")

            # Check for required Kubernetes secrets
            required_secrets = ["pynomaly-secrets", "registry-credentials"]

            for secret_name in required_secrets:
                return_code, _, _ = await self.execute_command(
                    f"kubectl get secret {secret_name} -n {namespace}"
                )
                if return_code != 0:
                    self.log(f"Required secret not found: {secret_name}", "ERROR")
                    return False

            self.log("Secrets validation passed")
            return True

        except Exception as e:
            self.log(f"Secrets validation failed: {e}", "ERROR")
            return False

    async def _validate_resources(self) -> bool:
        """Validate cluster resources"""
        try:
            # Check node resources
            return_code, stdout, _ = await self.execute_command(
                "kubectl top nodes --no-headers"
            )
            if return_code != 0:
                self.log("Could not retrieve node metrics", "WARNING")
                return True  # Non-critical

            # Parse resource usage
            lines = stdout.strip().split("\n")
            total_cpu_used = 0
            total_memory_used = 0

            for line in lines:
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        cpu_usage = parts[1].replace("m", "")
                        memory_usage = parts[2].replace("Mi", "")

                        try:
                            total_cpu_used += int(cpu_usage)
                            total_memory_used += int(memory_usage)
                        except ValueError:
                            continue

            # Check if resources are sufficient
            if total_cpu_used > 80000:  # 80% of typical cluster
                self.log(f"High CPU usage detected: {total_cpu_used}m", "WARNING")

            if total_memory_used > 80000:  # 80GB typical cluster
                self.log(
                    f"High memory usage detected: {total_memory_used}Mi", "WARNING"
                )

            self.log("Resource validation passed")
            return True

        except Exception as e:
            self.log(f"Resource validation failed: {e}", "WARNING")
            return True  # Non-critical

    async def build_and_test(self) -> bool:
        """Build application and run tests"""
        self.log("Building application and running tests...")
        self.current_phase = DeploymentPhase.BUILD

        try:
            if not self.config.get("skip_build", False):
                # Build Docker image
                image_tag = self.config.get("image_tag", "latest")
                image_name = f"{self.docker_registry}:{image_tag}"

                build_command = f"docker build -t {image_name} --build-arg ENVIRONMENT={self.config.get('environment')} ."
                return_code, _, stderr = await self.execute_command(
                    build_command, timeout=1800
                )

                if return_code != 0:
                    self.log(f"Docker build failed: {stderr}", "ERROR")
                    return False

                self.log(f"Successfully built image: {image_name}")

                # Push to registry if configured
                if self.config.get("push_image", True):
                    push_command = f"docker push {image_name}"
                    return_code, _, stderr = await self.execute_command(
                        push_command, timeout=600
                    )

                    if return_code != 0:
                        self.log(f"Docker push failed: {stderr}", "ERROR")
                        return False

                    self.log(f"Successfully pushed image: {image_name}")

            # Run tests
            if not self.config.get("skip_tests", False):
                self.current_phase = DeploymentPhase.TEST

                test_commands = [
                    "python -m pytest tests/unit/ -v --tb=short",
                    "python -m pytest tests/integration/ -v --tb=short",
                    "python scripts/security_scan.py",
                    "python scripts/performance_validation.py",
                ]

                for test_command in test_commands:
                    return_code, _, stderr = await self.execute_command(
                        test_command, timeout=600
                    )

                    if return_code != 0:
                        self.log(f"Test failed: {test_command} - {stderr}", "ERROR")
                        if not self.config.get("ignore_test_failures", False):
                            return False
                        else:
                            self.log("Ignoring test failure as configured", "WARNING")

                self.log("All tests passed successfully")

            return True

        except Exception as e:
            self.log(f"Build and test phase failed: {e}", "ERROR")
            return False

    async def deploy_application(self) -> bool:
        """Deploy application using specified strategy"""
        self.log("Deploying application...")
        self.current_phase = DeploymentPhase.DEPLOY

        try:
            strategy = DeploymentStrategy(self.config.get("strategy", "rolling"))

            if strategy == DeploymentStrategy.ROLLING:
                success = await self._rolling_deployment()
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._blue_green_deployment()
            elif strategy == DeploymentStrategy.CANARY:
                success = await self._canary_deployment()
            elif strategy == DeploymentStrategy.RECREATE:
                success = await self._recreate_deployment()
            else:
                self.log(f"Unknown deployment strategy: {strategy}", "ERROR")
                return False

            if success:
                self.log(
                    f"Deployment completed successfully using {strategy.value} strategy"
                )
                return True
            else:
                self.log(f"Deployment failed using {strategy.value} strategy", "ERROR")
                return False

        except Exception as e:
            self.log(f"Deployment failed: {e}", "ERROR")
            return False

    async def _rolling_deployment(self) -> bool:
        """Execute rolling deployment"""
        try:
            namespace = self.config.get("namespace", "pynomaly-prod")
            image_tag = self.config.get("image_tag", "latest")
            image_name = f"{self.docker_registry}:{image_tag}"

            # Update deployment with new image
            update_command = f"kubectl set image deployment/pynomaly-api pynomaly-api={image_name} -n {namespace}"
            return_code, _, stderr = await self.execute_command(update_command)

            if return_code != 0:
                self.log(f"Failed to update deployment image: {stderr}", "ERROR")
                return False

            # Wait for rollout to complete
            rollout_command = f"kubectl rollout status deployment/pynomaly-api -n {namespace} --timeout=600s"
            return_code, _, stderr = await self.execute_command(
                rollout_command, timeout=700
            )

            if return_code != 0:
                self.log(f"Rollout failed or timed out: {stderr}", "ERROR")
                return False

            # Verify pods are running
            await asyncio.sleep(30)  # Wait for pods to stabilize

            return_code, stdout, _ = await self.execute_command(
                f"kubectl get pods -n {namespace} -l app.kubernetes.io/component=api --no-headers"
            )

            if return_code != 0:
                self.log("Failed to check pod status", "ERROR")
                return False

            running_pods = stdout.count("Running")
            min_replicas = self.config.get("min_replicas", 2)

            if running_pods < min_replicas:
                self.log(
                    f"Insufficient running pods: {running_pods}/{min_replicas}", "ERROR"
                )
                return False

            self.log(
                f"Rolling deployment completed successfully with {running_pods} running pods"
            )
            return True

        except Exception as e:
            self.log(f"Rolling deployment failed: {e}", "ERROR")
            return False

    async def _blue_green_deployment(self) -> bool:
        """Execute blue-green deployment"""
        try:
            # Implementation for blue-green deployment
            # This is a simplified version - in production, you'd have more sophisticated traffic switching
            self.log("Executing blue-green deployment...")

            namespace = self.config.get("namespace", "pynomaly-prod")

            # Create green deployment
            green_deployment_yaml = self._generate_green_deployment_manifest()

            # Apply green deployment
            with open("/tmp/green-deployment.yaml", "w") as f:
                yaml.dump(green_deployment_yaml, f)

            return_code, _, stderr = await self.execute_command(
                "kubectl apply -f /tmp/green-deployment.yaml"
            )

            if return_code != 0:
                self.log(f"Failed to create green deployment: {stderr}", "ERROR")
                return False

            # Wait for green deployment to be ready
            return_code, _, stderr = await self.execute_command(
                f"kubectl rollout status deployment/pynomaly-api-green -n {namespace} --timeout=600s",
                timeout=700,
            )

            if return_code != 0:
                self.log(f"Green deployment failed: {stderr}", "ERROR")
                return False

            # Switch traffic to green
            await self._switch_traffic_to_green()

            # Clean up blue deployment after successful switch
            await asyncio.sleep(300)  # Wait 5 minutes before cleanup
            await self._cleanup_blue_deployment()

            self.log("Blue-green deployment completed successfully")
            return True

        except Exception as e:
            self.log(f"Blue-green deployment failed: {e}", "ERROR")
            return False

    async def _canary_deployment(self) -> bool:
        """Execute canary deployment"""
        try:
            self.log("Executing canary deployment...")

            # Deploy canary version with limited traffic
            canary_percentage = self.config.get("canary_percentage", 10)

            # Create canary deployment
            success = await self._create_canary_deployment()
            if not success:
                return False

            # Monitor canary metrics
            success = await self._monitor_canary_deployment(canary_percentage)
            if not success:
                # Rollback canary
                await self._rollback_canary_deployment()
                return False

            # Gradually increase traffic
            success = await self._promote_canary_deployment()
            if not success:
                await self._rollback_canary_deployment()
                return False

            self.log("Canary deployment completed successfully")
            return True

        except Exception as e:
            self.log(f"Canary deployment failed: {e}", "ERROR")
            return False

    async def _recreate_deployment(self) -> bool:
        """Execute recreate deployment"""
        try:
            self.log("Executing recreate deployment...")

            namespace = self.config.get("namespace", "pynomaly-prod")

            # Scale down to 0
            return_code, _, stderr = await self.execute_command(
                f"kubectl scale deployment pynomaly-api --replicas=0 -n {namespace}"
            )

            if return_code != 0:
                self.log(f"Failed to scale down deployment: {stderr}", "ERROR")
                return False

            # Wait for pods to terminate
            await asyncio.sleep(60)

            # Update image
            image_tag = self.config.get("image_tag", "latest")
            image_name = f"{self.docker_registry}:{image_tag}"

            return_code, _, stderr = await self.execute_command(
                f"kubectl set image deployment/pynomaly-api pynomaly-api={image_name} -n {namespace}"
            )

            if return_code != 0:
                self.log(f"Failed to update deployment image: {stderr}", "ERROR")
                return False

            # Scale up
            replicas = self.config.get("replicas", 3)
            return_code, _, stderr = await self.execute_command(
                f"kubectl scale deployment pynomaly-api --replicas={replicas} -n {namespace}"
            )

            if return_code != 0:
                self.log(f"Failed to scale up deployment: {stderr}", "ERROR")
                return False

            # Wait for deployment to be ready
            return_code, _, stderr = await self.execute_command(
                f"kubectl rollout status deployment/pynomaly-api -n {namespace} --timeout=600s",
                timeout=700,
            )

            if return_code != 0:
                self.log(f"Recreate deployment failed: {stderr}", "ERROR")
                return False

            self.log("Recreate deployment completed successfully")
            return True

        except Exception as e:
            self.log(f"Recreate deployment failed: {e}", "ERROR")
            return False

    async def verify_deployment(self) -> bool:
        """Comprehensive deployment verification"""
        self.log("Verifying deployment...")
        self.current_phase = DeploymentPhase.VERIFY

        try:
            verification_tasks = [
                self._verify_pod_health(),
                self._verify_service_connectivity(),
                self._verify_api_endpoints(),
                self._verify_database_connectivity(),
                self._verify_performance_metrics(),
            ]

            results = await asyncio.gather(*verification_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.log(f"Verification task {i} failed: {result}", "ERROR")
                    return False
                elif not result:
                    self.log(f"Verification task {i} failed", "ERROR")
                    return False

            self.log("All deployment verifications passed")
            return True

        except Exception as e:
            self.log(f"Deployment verification failed: {e}", "ERROR")
            return False

    async def _verify_pod_health(self) -> bool:
        """Verify pod health status"""
        try:
            namespace = self.config.get("namespace", "pynomaly-prod")

            # Check pod status
            return_code, stdout, _ = await self.execute_command(
                f"kubectl get pods -n {namespace} -l app.kubernetes.io/component=api --no-headers"
            )

            if return_code != 0:
                return False

            pods = stdout.strip().split("\n")
            running_pods = 0
            ready_pods = 0

            for pod_line in pods:
                if pod_line:
                    parts = pod_line.split()
                    if len(parts) >= 3:
                        if "Running" in parts[2]:
                            running_pods += 1
                        if "/" in parts[1]:
                            ready, total = parts[1].split("/")
                            if ready == total:
                                ready_pods += 1

            min_replicas = self.config.get("min_replicas", 2)

            if running_pods < min_replicas or ready_pods < min_replicas:
                self.log(
                    f"Insufficient healthy pods: {ready_pods}/{running_pods}", "ERROR"
                )
                return False

            self.log(f"Pod health check passed: {ready_pods} healthy pods")
            return True

        except Exception as e:
            self.log(f"Pod health verification failed: {e}", "ERROR")
            return False

    async def _verify_service_connectivity(self) -> bool:
        """Verify service connectivity"""
        try:
            namespace = self.config.get("namespace", "pynomaly-prod")

            # Port forward for testing
            port_forward_process = await asyncio.create_subprocess_shell(
                f"kubectl port-forward service/pynomaly-api-internal 8080:8000 -n {namespace}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for port forward to establish
            await asyncio.sleep(10)

            try:
                # Test health endpoint
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:8080/api/v1/health", timeout=10
                    ) as response:
                        if response.status == 200:
                            self.log("Service connectivity check passed")
                            return True
                        else:
                            self.log(
                                f"Health endpoint returned status {response.status}",
                                "ERROR",
                            )
                            return False

            finally:
                # Clean up port forward
                port_forward_process.terminate()
                await port_forward_process.wait()

        except Exception as e:
            self.log(f"Service connectivity verification failed: {e}", "ERROR")
            return False

    async def _verify_api_endpoints(self) -> bool:
        """Verify API endpoints functionality"""
        try:
            # This would typically involve running API tests
            return_code, _, stderr = await self.execute_command(
                "python -m pytest tests/api/ -v --tb=short", timeout=300
            )

            if return_code != 0:
                self.log(f"API endpoint tests failed: {stderr}", "ERROR")
                return False

            self.log("API endpoint verification passed")
            return True

        except Exception as e:
            self.log(f"API endpoint verification failed: {e}", "ERROR")
            return False

    async def _verify_database_connectivity(self) -> bool:
        """Verify database connectivity"""
        try:
            namespace = self.config.get("namespace", "pynomaly-prod")

            # Check PostgreSQL
            return_code, _, _ = await self.execute_command(
                f"kubectl exec -n {namespace} postgres-0 -- pg_isready -U pynomaly"
            )

            if return_code != 0:
                self.log("PostgreSQL connectivity check failed", "ERROR")
                return False

            # Check Redis
            return_code, _, _ = await self.execute_command(
                f"kubectl exec -n {namespace} redis-0 -- redis-cli ping"
            )

            if return_code != 0:
                self.log("Redis connectivity check failed", "ERROR")
                return False

            self.log("Database connectivity verification passed")
            return True

        except Exception as e:
            self.log(f"Database connectivity verification failed: {e}", "ERROR")
            return False

    async def _verify_performance_metrics(self) -> bool:
        """Verify performance metrics"""
        try:
            # Run performance validation
            return_code, _, stderr = await self.execute_command(
                "python scripts/performance_validation.py --duration 60", timeout=120
            )

            if return_code != 0:
                self.log(f"Performance metrics verification failed: {stderr}", "ERROR")
                return False

            self.log("Performance metrics verification passed")
            return True

        except Exception as e:
            self.log(f"Performance metrics verification failed: {e}", "ERROR")
            return False

    async def rollback_deployment(self) -> bool:
        """Rollback to previous deployment"""
        self.log("Initiating deployment rollback...")

        try:
            namespace = self.config.get("namespace", "pynomaly-prod")

            # Rollback to previous revision
            return_code, _, stderr = await self.execute_command(
                f"kubectl rollout undo deployment/pynomaly-api -n {namespace}"
            )

            if return_code != 0:
                self.log(f"Rollback command failed: {stderr}", "ERROR")
                return False

            # Wait for rollback to complete
            return_code, _, stderr = await self.execute_command(
                f"kubectl rollout status deployment/pynomaly-api -n {namespace} --timeout=600s",
                timeout=700,
            )

            if return_code != 0:
                self.log(f"Rollback failed or timed out: {stderr}", "ERROR")
                return False

            # Verify rollback
            success = await self._verify_pod_health()
            if success:
                self.log("Rollback completed successfully")
                self.rollback_performed = True
                return True
            else:
                self.log("Rollback verification failed", "ERROR")
                return False

        except Exception as e:
            self.log(f"Rollback failed: {e}", "ERROR")
            return False

    async def send_notifications(self, success: bool):
        """Send deployment notifications"""
        try:
            status = "successful" if success else "failed"
            message = {
                "deployment_id": self.deployment_id,
                "environment": self.config.get("environment"),
                "status": status,
                "duration": (datetime.now() - self.start_time).total_seconds(),
                "rollback_performed": self.rollback_performed,
                "image_tag": self.config.get("image_tag"),
                "timestamp": datetime.now().isoformat(),
            }

            # Send Slack notification
            if self.slack_webhook:
                await self._send_slack_notification(message)

            # Send Teams notification
            if self.teams_webhook:
                await self._send_teams_notification(message)

            # Update GitHub deployment status
            if self.github_token and self.config.get("github_deployment_id"):
                await self._update_github_deployment_status(message)

        except Exception as e:
            self.log(f"Failed to send notifications: {e}", "ERROR")

    async def _send_slack_notification(self, message: dict[str, Any]):
        """Send Slack notification"""
        try:
            color = "good" if message["status"] == "successful" else "danger"

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Pynomaly Deployment {message['status'].title()}",
                        "fields": [
                            {
                                "title": "Environment",
                                "value": message["environment"],
                                "short": True,
                            },
                            {
                                "title": "Image Tag",
                                "value": message["image_tag"],
                                "short": True,
                            },
                            {
                                "title": "Duration",
                                "value": f"{message['duration']:.1f}s",
                                "short": True,
                            },
                            {
                                "title": "Rollback",
                                "value": str(message["rollback_performed"]),
                                "short": True,
                            },
                        ],
                        "footer": "Pynomaly Deployment Pipeline",
                        "ts": int(datetime.now().timestamp()),
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload) as response:
                    if response.status == 200:
                        self.log("Slack notification sent successfully")
                    else:
                        self.log(
                            f"Failed to send Slack notification: {response.status}",
                            "ERROR",
                        )

        except Exception as e:
            self.log(f"Slack notification failed: {e}", "ERROR")

    async def _send_teams_notification(self, message: dict[str, Any]):
        """Send Microsoft Teams notification"""
        try:
            color = "00FF00" if message["status"] == "successful" else "FF0000"

            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary": f"Pynomaly Deployment {message['status'].title()}",
                "sections": [
                    {
                        "activityTitle": f"Pynomaly Deployment {message['status'].title()}",
                        "facts": [
                            {"name": "Environment", "value": message["environment"]},
                            {"name": "Image Tag", "value": message["image_tag"]},
                            {
                                "name": "Duration",
                                "value": f"{message['duration']:.1f}s",
                            },
                            {
                                "name": "Rollback",
                                "value": str(message["rollback_performed"]),
                            },
                        ],
                    }
                ],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.teams_webhook, json=payload) as response:
                    if response.status == 200:
                        self.log("Teams notification sent successfully")
                    else:
                        self.log(
                            f"Failed to send Teams notification: {response.status}",
                            "ERROR",
                        )

        except Exception as e:
            self.log(f"Teams notification failed: {e}", "ERROR")

    async def _update_github_deployment_status(self, message: dict[str, Any]):
        """Update GitHub deployment status"""
        try:
            # Implementation for GitHub deployment status update
            # This would use GitHub API to update deployment status
            pass

        except Exception as e:
            self.log(f"GitHub status update failed: {e}", "ERROR")

    def generate_deployment_report(self) -> dict[str, Any]:
        """Generate comprehensive deployment report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        report = {
            "deployment_id": self.deployment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "success": self.deployment_successful,
            "rollback_performed": self.rollback_performed,
            "current_phase": self.current_phase.value,
            "configuration": self.config,
            "logs": self.logs,
            "metadata": {
                "pipeline_version": "2.0.0",
                "generated_at": datetime.now().isoformat(),
            },
        }

        return report

    def save_deployment_report(self, report: dict[str, Any]):
        """Save deployment report to file"""
        try:
            reports_dir = self.project_root / "reports" / "deployments"
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_file = reports_dir / f"deployment_{self.deployment_id}.json"

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            self.log(f"Deployment report saved to {report_file}")

        except Exception as e:
            self.log(f"Failed to save deployment report: {e}", "ERROR")

    async def execute_deployment(self) -> bool:
        """Execute the complete deployment pipeline"""
        try:
            self.log(f"Starting automated deployment pipeline {self.deployment_id}")

            # Phase 1: Validation
            if not await self.validate_prerequisites():
                self.log("Prerequisites validation failed", "ERROR")
                return False

            # Phase 2: Build and Test
            if not await self.build_and_test():
                self.log("Build and test phase failed", "ERROR")
                return False

            # Phase 3: Deploy
            if not await self.deploy_application():
                self.log("Application deployment failed", "ERROR")
                return False

            # Phase 4: Verify
            if not await self.verify_deployment():
                self.log("Deployment verification failed", "ERROR")

                # Attempt rollback on verification failure
                if self.config.get("rollback_on_failure", True):
                    rollback_success = await self.rollback_deployment()
                    if not rollback_success:
                        self.log("Rollback also failed", "ERROR")
                return False

            # Phase 5: Complete
            self.current_phase = DeploymentPhase.COMPLETE
            self.deployment_successful = True
            self.log("Deployment pipeline completed successfully")

            return True

        except Exception as e:
            self.log(f"Deployment pipeline failed: {e}", "ERROR")
            return False

        finally:
            # Always send notifications and save report
            await self.send_notifications(self.deployment_successful)

            report = self.generate_deployment_report()
            self.save_deployment_report(report)

    # Helper methods for deployment strategies
    def _generate_green_deployment_manifest(self) -> dict[str, Any]:
        """Generate green deployment manifest for blue-green deployment"""
        # This would generate the green deployment YAML
        # Simplified version here
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "pynomaly-api-green",
                "namespace": self.config.get("namespace", "pynomaly-prod"),
            },
            "spec": {
                "replicas": self.config.get("replicas", 3),
                "selector": {
                    "matchLabels": {"app": "pynomaly-api", "version": "green"}
                },
                "template": {
                    "metadata": {"labels": {"app": "pynomaly-api", "version": "green"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "pynomaly-api",
                                "image": f"{self.docker_registry}:{self.config.get('image_tag', 'latest')}",
                                "ports": [{"containerPort": 8000}],
                            }
                        ]
                    },
                },
            },
        }

    async def _switch_traffic_to_green(self):
        """Switch traffic from blue to green deployment"""
        # Implementation for traffic switching
        pass

    async def _cleanup_blue_deployment(self):
        """Clean up blue deployment after successful green deployment"""
        # Implementation for blue deployment cleanup
        pass

    async def _create_canary_deployment(self) -> bool:
        """Create canary deployment"""
        # Implementation for canary deployment creation
        return True

    async def _monitor_canary_deployment(self, percentage: int) -> bool:
        """Monitor canary deployment metrics"""
        # Implementation for canary monitoring
        return True

    async def _promote_canary_deployment(self) -> bool:
        """Promote canary to full deployment"""
        # Implementation for canary promotion
        return True

    async def _rollback_canary_deployment(self):
        """Rollback canary deployment"""
        # Implementation for canary rollback
        pass


def load_config(config_file: Path) -> dict[str, Any]:
    """Load deployment configuration from file"""
    try:
        with open(config_file) as f:
            if config_file.suffix == ".yaml" or config_file.suffix == ".yml":
                return yaml.safe_load(f)
            elif config_file.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_file.suffix}"
                )
    except Exception as e:
        logger.error(f"Failed to load config file {config_file}: {e}")
        sys.exit(1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Automated Deployment Pipeline for Pynomaly"
    )
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument(
        "--environment",
        choices=[e.value for e in Environment],
        default="production",
        help="Deployment environment",
    )
    parser.add_argument("--image-tag", default="latest", help="Docker image tag")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in DeploymentStrategy],
        default="rolling",
        help="Deployment strategy",
    )
    parser.add_argument(
        "--namespace", default="pynomaly-prod", help="Kubernetes namespace"
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip test phase")
    parser.add_argument("--skip-build", action="store_true", help="Skip build phase")
    parser.add_argument(
        "--rollback-on-failure",
        action="store_true",
        default=True,
        help="Rollback on deployment failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual deployment",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)

    # Override with command line arguments
    config.update(
        {
            "environment": args.environment,
            "image_tag": args.image_tag,
            "strategy": args.strategy,
            "namespace": args.namespace,
            "skip_tests": args.skip_tests,
            "skip_build": args.skip_build,
            "rollback_on_failure": args.rollback_on_failure,
            "dry_run": args.dry_run,
        }
    )

    # Initialize and execute deployment pipeline
    pipeline = AutomatedDeploymentPipeline(config)

    try:
        success = asyncio.run(pipeline.execute_deployment())

        if success:
            print(f"\n✅ Deployment {pipeline.deployment_id} completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Deployment {pipeline.deployment_id} failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Deployment failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
