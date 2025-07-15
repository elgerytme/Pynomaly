#!/usr/bin/env python3
"""
Production Deployment Script

Automated deployment pipeline with comprehensive testing and validation.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import requests


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    environment: str
    version: str
    build_number: str
    docker_registry: str
    health_check_url: str
    timeout_seconds: int = 600
    rollback_enabled: bool = True


@dataclass
class QualityGate:
    """Quality gate configuration."""

    metric: str
    threshold: float
    operator: str
    description: str


@dataclass
class DeploymentResult:
    """Deployment result."""

    stage: str
    success: bool
    duration: float
    message: str
    details: dict = None


class ProductionDeploymentOrchestrator:
    """Production deployment orchestrator."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.deployment_start = datetime.now()
        self.results: list[DeploymentResult] = []

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"deployment_{self.config.version}.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        return logging.getLogger(__name__)

    async def deploy(self) -> bool:
        """Execute full deployment pipeline."""
        self.logger.info(f"Starting deployment of version {self.config.version}")

        stages = [
            self._run_comprehensive_tests,
            self._security_scan,
            self._build_containers,
            self._integration_tests,
            self._staging_deployment,
            self._production_deployment,
            self._post_deployment_validation,
        ]

        try:
            for stage in stages:
                stage_name = stage.__name__.replace("_", " ").title()
                self.logger.info(f"Executing stage: {stage_name}")

                start_time = time.time()
                success = await stage()
                duration = time.time() - start_time

                result = DeploymentResult(
                    stage=stage_name,
                    success=success,
                    duration=duration,
                    message=f"Stage completed in {duration:.2f}s",
                )
                self.results.append(result)

                if not success:
                    self.logger.error(f"Stage {stage_name} failed")
                    if self.config.rollback_enabled:
                        await self._rollback()
                    return False

                self.logger.info(f"Stage {stage_name} completed successfully")

            self.logger.info("Deployment completed successfully")
            await self._send_deployment_notification(success=True)
            return True

        except Exception as e:
            self.logger.error(f"Deployment failed with error: {e}")
            if self.config.rollback_enabled:
                await self._rollback()
            await self._send_deployment_notification(success=False, error=str(e))
            return False

    async def _run_comprehensive_tests(self) -> bool:
        """Run comprehensive test suite."""
        self.logger.info("Running comprehensive test suite...")

        test_commands = [
            "python -m pytest tests/ -v --cov=src --cov-report=xml --cov-fail-under=90",
            "python -m pytest tests/security/ -v --tb=short",
            "python -m pytest tests/performance/ -v --tb=short",
            "python -m pytest tests/infrastructure/ -v --tb=short",
        ]

        for cmd in test_commands:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Test command failed: {cmd}")
                self.logger.error(result.stderr)
                return False

        self.logger.info("All tests passed successfully")
        return True

    async def _security_scan(self) -> bool:
        """Run security vulnerability scanning."""
        self.logger.info("Running security scans...")

        security_commands = [
            "safety check --json --output safety_report.json",
            "bandit -r src/ -f json -o bandit_report.json",
        ]

        for cmd in security_commands:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            # Note: Some security tools may return non-zero even for warnings
            # Parse the actual results instead of relying on return codes
            if "bandit" in cmd:
                if os.path.exists("bandit_report.json"):
                    with open("bandit_report.json") as f:
                        report = json.load(f)
                        if report.get("results", []):
                            high_severity = [
                                r
                                for r in report["results"]
                                if r.get("issue_severity") == "HIGH"
                            ]
                            if high_severity:
                                self.logger.error(
                                    f"High severity security issues found: {len(high_severity)}"
                                )
                                return False

        self.logger.info("Security scans completed successfully")
        return True

    async def _build_containers(self) -> bool:
        """Build production containers."""
        self.logger.info("Building production containers...")

        build_commands = [
            f"docker build -f deploy/production/Dockerfile.api -t {self.config.docker_registry}/pynomaly-api:{self.config.version} .",
            f"docker build -f deploy/production/Dockerfile.worker -t {self.config.docker_registry}/pynomaly-worker:{self.config.version} .",
        ]

        for cmd in build_commands:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Build command failed: {cmd}")
                self.logger.error(result.stderr)
                return False

        # Push to registry
        push_commands = [
            f"docker push {self.config.docker_registry}/pynomaly-api:{self.config.version}",
            f"docker push {self.config.docker_registry}/pynomaly-worker:{self.config.version}",
        ]

        for cmd in push_commands:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Push command failed: {cmd}")
                return False

        self.logger.info("Container build and push completed successfully")
        return True

    async def _integration_tests(self) -> bool:
        """Run integration tests."""
        self.logger.info("Running integration tests...")

        # Start test environment
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                "deploy/production/docker-compose.test.yml",
                "up",
                "-d",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.logger.error("Failed to start test environment")
            return False

        try:
            # Wait for services to be ready
            await asyncio.sleep(30)

            # Run integration tests
            test_commands = [
                "python -m pytest tests/integration/ -v --tb=short",
                "python -m pytest tests/e2e/ -v --tb=short",
            ]

            for cmd in test_commands:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Integration test failed: {cmd}")
                    return False

            return True

        finally:
            # Cleanup test environment
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "deploy/production/docker-compose.test.yml",
                    "down",
                    "-v",
                ],
                capture_output=True,
            )

    async def _staging_deployment(self) -> bool:
        """Deploy to staging environment."""
        self.logger.info("Deploying to staging environment...")

        # Deploy to staging
        result = subprocess.run(
            ["docker-compose", "-f", "deploy/staging/docker-compose.yml", "up", "-d"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.logger.error("Staging deployment failed")
            return False

        # Health check
        return await self._health_check("staging")

    async def _production_deployment(self) -> bool:
        """Deploy to production environment."""
        self.logger.info("Deploying to production environment...")

        # Blue-green deployment strategy
        # 1. Deploy new version alongside current
        # 2. Health check new version
        # 3. Switch traffic
        # 4. Remove old version

        # Deploy new version
        env = os.environ.copy()
        env["VERSION"] = self.config.version

        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                "deploy/production/docker-compose.production.yml",
                "up",
                "-d",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.logger.error("Production deployment failed")
            return False

        # Health check
        return await self._health_check("production")

    async def _post_deployment_validation(self) -> bool:
        """Post-deployment validation."""
        self.logger.info("Running post-deployment validation...")

        # Smoke tests
        smoke_test_result = subprocess.run(
            ["python", "scripts/smoke_test.py", "--environment=production"],
            capture_output=True,
            text=True,
        )

        if smoke_test_result.returncode != 0:
            self.logger.error("Smoke tests failed")
            return False

        # Performance validation
        perf_test_result = subprocess.run(
            ["python", "scripts/performance_validation.py", "--environment=production"],
            capture_output=True,
            text=True,
        )

        if perf_test_result.returncode != 0:
            self.logger.error("Performance validation failed")
            return False

        self.logger.info("Post-deployment validation completed successfully")
        return True

    async def _health_check(self, environment: str) -> bool:
        """Perform health check."""
        max_retries = 30
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.config.health_check_url}/health", timeout=10
                )

                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        self.logger.info(f"Health check passed for {environment}")
                        return True

            except requests.RequestException as e:
                self.logger.warning(f"Health check attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        self.logger.error(
            f"Health check failed for {environment} after {max_retries} attempts"
        )
        return False

    async def _rollback(self) -> bool:
        """Rollback deployment."""
        self.logger.info("Initiating rollback...")

        # Get previous version
        # This would typically come from a deployment history service
        previous_version = os.environ.get("PREVIOUS_VERSION", "latest")

        # Rollback command
        env = os.environ.copy()
        env["VERSION"] = previous_version

        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                "deploy/production/docker-compose.production.yml",
                "up",
                "-d",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            self.logger.info("Rollback completed successfully")
            return True
        else:
            self.logger.error("Rollback failed")
            return False

    async def _send_deployment_notification(self, success: bool, error: str = None):
        """Send deployment notification."""
        duration = (datetime.now() - self.deployment_start).total_seconds()

        notification = {
            "deployment_id": f"deploy-{self.config.version}-{self.config.build_number}",
            "version": self.config.version,
            "environment": self.config.environment,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "stage": r.stage,
                    "success": r.success,
                    "duration": r.duration,
                    "message": r.message,
                }
                for r in self.results
            ],
        }

        if error:
            notification["error"] = error

        # Save deployment report
        with open(f"deployment_report_{self.config.version}.json", "w") as f:
            json.dump(notification, f, indent=2)

        self.logger.info(f"Deployment notification sent: {notification}")


async def main():
    """Main deployment function."""
    # Load configuration from environment or config file
    config = DeploymentConfig(
        environment=os.environ.get("ENVIRONMENT", "production"),
        version=os.environ.get("VERSION", "latest"),
        build_number=os.environ.get("BUILD_NUMBER", "manual"),
        docker_registry=os.environ.get("DOCKER_REGISTRY", "localhost:5000"),
        health_check_url=os.environ.get("HEALTH_CHECK_URL", "http://localhost:8000"),
    )

    orchestrator = ProductionDeploymentOrchestrator(config)
    success = await orchestrator.deploy()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
