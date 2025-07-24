#!/usr/bin/env python3
"""
Automated Deployment Script for anomaly_detection

This script provides fully automated deployment capabilities including:
- Environment-specific deployments
- Blue-green deployment strategies
- Canary releases
- Rollback capabilities
- Health monitoring
- Notification integration
"""

import asyncio
import json
import logging
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import click
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutomatedDeployer:
    """Automated deployment orchestrator for anomaly_detection."""

    def __init__(self, config_path: Path | None = None):
        self.config = self._load_config(config_path)
        self.deployment_history = []
        self.notification_channels = self._setup_notifications()

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load deployment configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            "environments": {
                "development": {
                    "platform": "docker_compose",
                    "strategy": "recreate",
                    "auto_deploy": True,
                    "health_checks": True,
                    "notifications": ["console"],
                },
                "staging": {
                    "platform": "kubernetes",
                    "strategy": "rolling_update",
                    "auto_deploy": True,
                    "health_checks": True,
                    "notifications": ["email", "slack"],
                },
                "production": {
                    "platform": "kubernetes",
                    "strategy": "blue_green",
                    "auto_deploy": False,
                    "health_checks": True,
                    "notifications": ["email", "slack", "pagerduty"],
                    "approval_required": True,
                },
            },
            "notifications": {
                "email": {
                    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                    "username": os.getenv("SMTP_USERNAME"),
                    "password": os.getenv("SMTP_PASSWORD"),
                    "recipients": ["team@anomaly_detection.io"],
                },
                "slack": {
                    "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                    "channel": "#deployments",
                },
            },
            "health_checks": {
                "timeout": 300,
                "retry_interval": 30,
                "endpoints": ["/api/health", "/api/health/ready", "/api/health/live"],
            },
        }

    def _setup_notifications(self) -> dict[str, Any]:
        """Setup notification channels."""
        channels = {}

        # Email notifications
        email_config = self.config.get("notifications", {}).get("email", {})
        if email_config.get("username") and email_config.get("password"):
            channels["email"] = email_config

        # Slack notifications
        slack_config = self.config.get("notifications", {}).get("slack", {})
        if slack_config.get("webhook_url"):
            channels["slack"] = slack_config

        return channels

    async def deploy(
        self,
        environment: str,
        version: str = "latest",
        force: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Execute automated deployment."""

        logger.info(f"Starting automated deployment to {environment}")

        # Get environment configuration
        env_config = self.config["environments"].get(environment)
        if not env_config:
            logger.error(f"Environment {environment} not configured")
            return False

        # Check if approval is required
        if env_config.get("approval_required", False) and not force:
            if not await self._request_approval(environment, version):
                logger.info("Deployment cancelled - approval not granted")
                return False

        # Send deployment start notification
        await self._notify(
            "deployment_start",
            {
                "environment": environment,
                "version": version,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        try:
            # Pre-deployment checks
            if not await self._pre_deployment_checks(environment):
                await self._notify(
                    "deployment_failed",
                    {
                        "environment": environment,
                        "reason": "Pre-deployment checks failed",
                    },
                )
                return False

            # Execute deployment based on strategy
            strategy = env_config.get("strategy", "rolling_update")
            monorepo = env_config.get("platform", "docker_compose")

            if dry_run:
                logger.info(
                    f"DRY RUN: Would deploy to {environment} using {strategy} on {platform}"
                )
                return True

            success = await self._execute_deployment(
                environment, version, strategy, monorepo
            )

            if success:
                # Post-deployment validation
                if env_config.get("health_checks", True):
                    if not await self._post_deployment_validation(environment):
                        logger.error(
                            "Post-deployment validation failed, initiating rollback"
                        )
                        await self._rollback(environment)
                        return False

                # Record successful deployment
                self._record_deployment(environment, version, True)

                await self._notify(
                    "deployment_success",
                    {
                        "environment": environment,
                        "version": version,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

                return True
            else:
                await self._notify(
                    "deployment_failed",
                    {
                        "environment": environment,
                        "reason": "Deployment execution failed",
                    },
                )
                return False

        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            await self._notify(
                "deployment_failed", {"environment": environment, "reason": str(e)}
            )
            return False

    async def _pre_deployment_checks(self, environment: str) -> bool:
        """Execute pre-deployment checks."""
        logger.info("Running pre-deployment checks...")

        try:
            # Check environment health
            if not await self._check_environment_health(environment):
                logger.error("Environment health check failed")
                return False

            # Check resource availability
            if not await self._check_resource_availability(environment):
                logger.error("Resource availability check failed")
                return False

            # Check dependencies
            if not await self._check_dependencies(environment):
                logger.error("Dependency check failed")
                return False

            logger.info("All pre-deployment checks passed")
            return True

        except Exception as e:
            logger.error(f"Pre-deployment checks failed: {e}")
            return False

    async def _execute_deployment(
        self, environment: str, version: str, strategy: str, monorepo: str
    ) -> bool:
        """Execute the actual deployment."""
        logger.info(f"Executing {strategy} deployment on {platform}")

        try:
            if monorepo == "docker_compose":
                return await self._deploy_docker_compose(environment, version)
            elif monorepo == "kubernetes":
                return await self._deploy_kubernetes(environment, version, strategy)
            elif monorepo == "helm":
                return await self._deploy_helm(environment, version, strategy)
            else:
                logger.error(f"Unsupported monorepo: {platform}")
                return False

        except Exception as e:
            logger.error(f"Deployment execution failed: {e}")
            return False

    async def _deploy_docker_compose(self, environment: str, version: str) -> bool:
        """Deploy using Docker Compose."""
        logger.info("Deploying with Docker Compose...")

        try:
            # Set environment variables
            env_vars = {"VERSION": version, "ANOMALY_DETECTION_ENVIRONMENT": environment}

            # Update environment
            env = os.environ.copy()
            env.update(env_vars)

            # Execute docker-compose command
            cmd = [
                "docker-compose",
                "-f",
                f"deploy/docker/docker-compose.{environment}.yml",
                "up",
                "-d",
                "--remove-orphans",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode == 0:
                logger.info("Docker Compose deployment completed successfully")
                return True
            else:
                logger.error(f"Docker Compose deployment failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Docker Compose deployment error: {e}")
            return False

    async def _deploy_kubernetes(
        self, environment: str, version: str, strategy: str
    ) -> bool:
        """Deploy to Kubernetes."""
        logger.info(f"Deploying to Kubernetes with {strategy} strategy...")

        try:
            namespace = f"anomaly_detection-{environment}"

            if strategy == "blue_green":
                return await self._blue_green_deployment(namespace, version)
            elif strategy == "canary":
                return await self._canary_deployment(namespace, version)
            else:
                return await self._rolling_update_deployment(namespace, version)

        except Exception as e:
            logger.error(f"Kubernetes deployment error: {e}")
            return False

    async def _deploy_helm(self, environment: str, version: str, strategy: str) -> bool:
        """Deploy using Helm."""
        logger.info("Deploying with Helm...")

        try:
            release_name = f"anomaly_detection-{environment}"
            namespace = f"anomaly_detection-{environment}"
            chart_path = "deploy/helm/anomaly_detection"

            cmd = [
                "helm",
                "upgrade",
                "--install",
                release_name,
                chart_path,
                "--namespace",
                namespace,
                "--create-namespace",
                "--set",
                f"image.tag={version}",
                "--set",
                f"app.environment={environment}",
                "--wait",
                "--timeout",
                "10m",
            ]

            if strategy == "blue_green":
                cmd.extend(["--set", "deployment.strategy=blue-green"])
            elif strategy == "canary":
                cmd.extend(["--set", "deployment.strategy=canary"])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Helm deployment completed successfully")
                return True
            else:
                logger.error(f"Helm deployment failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Helm deployment error: {e}")
            return False

    async def _blue_green_deployment(self, namespace: str, version: str) -> bool:
        """Execute blue-green deployment."""
        logger.info("Executing blue-green deployment...")

        try:
            # Deploy to green environment
            green_deployment = f"anomaly_detection-green-{int(time.time())}"

            # Apply green deployment
            cmd = ["kubectl", "apply", "-f", "-", "--namespace", namespace]

            # Create green deployment manifest
            green_manifest = self._create_green_deployment_manifest(
                green_deployment, version
            )

            result = subprocess.run(
                cmd, input=green_manifest, text=True, capture_output=True
            )

            if result.returncode != 0:
                logger.error(f"Failed to create green deployment: {result.stderr}")
                return False

            # Wait for green deployment to be ready
            if not await self._wait_for_deployment_ready(namespace, green_deployment):
                logger.error("Green deployment failed to become ready")
                return False

            # Health check green deployment
            if not await self._health_check_deployment(namespace, green_deployment):
                logger.error("Green deployment health check failed")
                return False

            # Switch traffic to green
            await self._switch_traffic_to_green(namespace, green_deployment)

            # Clean up old blue deployment
            await self._cleanup_blue_deployment(namespace)

            logger.info("Blue-green deployment completed successfully")
            return True

        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False

    async def _canary_deployment(self, namespace: str, version: str) -> bool:
        """Execute canary deployment."""
        logger.info("Executing canary deployment...")

        try:
            # Deploy canary version with small percentage of traffic
            canary_percentage = 10

            # Update canary deployment
            await self._update_canary_deployment(namespace, version, canary_percentage)

            # Monitor canary for specified duration
            canary_duration = 300  # 5 minutes
            if not await self._monitor_canary(namespace, canary_duration):
                logger.error("Canary monitoring failed, rolling back")
                await self._rollback_canary(namespace)
                return False

            # Gradually increase traffic to canary
            for percentage in [25, 50, 75, 100]:
                await self._update_canary_traffic(namespace, percentage)
                await asyncio.sleep(120)  # Wait 2 minutes between traffic increases

                if not await self._monitor_canary(namespace, 120):
                    logger.error(
                        f"Canary failed at {percentage}% traffic, rolling back"
                    )
                    await self._rollback_canary(namespace)
                    return False

            # Promote canary to production
            await self._promote_canary(namespace)

            logger.info("Canary deployment completed successfully")
            return True

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return False

    async def _rolling_update_deployment(self, namespace: str, version: str) -> bool:
        """Execute rolling update deployment."""
        logger.info("Executing rolling update deployment...")

        try:
            # Update deployment image
            cmd = [
                "kubectl",
                "set",
                "image",
                "deployment/anomaly_detection-api",
                f"anomaly_detection=anomaly_detection:production-{version}",
                "--namespace",
                namespace,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to update deployment image: {result.stderr}")
                return False

            # Wait for rollout to complete
            cmd = [
                "kubectl",
                "rollout",
                "status",
                "deployment/anomaly_detection-api",
                "--namespace",
                namespace,
                "--timeout",
                "600s",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Rolling update deployment completed successfully")
                return True
            else:
                logger.error(f"Rolling update failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Rolling update deployment failed: {e}")
            return False

    async def _post_deployment_validation(self, environment: str) -> bool:
        """Validate deployment after completion."""
        logger.info("Running post-deployment validation...")

        try:
            health_config = self.config.get("health_checks", {})
            timeout = health_config.get("timeout", 300)
            retry_interval = health_config.get("retry_interval", 30)
            endpoints = health_config.get("endpoints", ["/api/health"])

            # Wait for initial startup
            await asyncio.sleep(30)

            # Check each endpoint
            for endpoint in endpoints:
                if not await self._validate_endpoint(
                    environment, endpoint, timeout, retry_interval
                ):
                    logger.error(f"Endpoint validation failed: {endpoint}")
                    return False

            # Run integration tests
            if not await self._run_integration_tests(environment):
                logger.error("Integration tests failed")
                return False

            logger.info("Post-deployment validation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Post-deployment validation failed: {e}")
            return False

    async def _validate_endpoint(
        self, environment: str, endpoint: str, timeout: int, retry_interval: int
    ) -> bool:
        """Validate a specific endpoint."""
        base_url = self._get_base_url(environment)
        url = f"{base_url}{endpoint}"

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Endpoint {endpoint} is healthy")
                    return True
                else:
                    logger.warning(
                        f"Endpoint {endpoint} returned status {response.status_code}"
                    )

            except Exception as e:
                logger.warning(f"Endpoint {endpoint} check failed: {e}")

            await asyncio.sleep(retry_interval)

        logger.error(f"Endpoint {endpoint} validation timed out")
        return False

    async def _notify(self, event_type: str, data: dict[str, Any]):
        """Send notifications through configured channels."""
        logger.info(f"Sending notification: {event_type}")

        try:
            # Console notification
            print(f"[{event_type.upper()}] {json.dumps(data, indent=2)}")

            # Email notification
            if "email" in self.notification_channels:
                await self._send_email_notification(event_type, data)

            # Slack notification
            if "slack" in self.notification_channels:
                await self._send_slack_notification(event_type, data)

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def _send_email_notification(self, event_type: str, data: dict[str, Any]):
        """Send email notification."""
        try:
            email_config = self.notification_channels["email"]

            subject = f"anomaly_detection Deployment {event_type.replace('_', ' ').title()}"
            body = f"""
            Deployment Event: {event_type}
            Environment: {data.get('environment', 'unknown')}
            Version: {data.get('version', 'unknown')}
            Timestamp: {data.get('timestamp', datetime.utcnow().isoformat())}

            Details: {json.dumps(data, indent=2)}
            """

            msg = MIMEMultipart()
            msg["From"] = email_config["username"]
            msg["To"] = ", ".join(email_config["recipients"])
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(
                email_config["smtp_server"], email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)

            logger.info("Email notification sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _send_slack_notification(self, event_type: str, data: dict[str, Any]):
        """Send Slack notification."""
        try:
            slack_config = self.notification_channels["slack"]

            color = {
                "deployment_start": "#2196F3",
                "deployment_success": "#4CAF50",
                "deployment_failed": "#F44336",
                "rollback_complete": "#FF9800",
            }.get(event_type, "#9E9E9E")

            payload = {
                "channel": slack_config.get("channel", "#deployments"),
                "username": "anomaly_detection Deployer",
                "icon_emoji": ":rocket:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"Deployment {event_type.replace('_', ' ').title()}",
                        "fields": [
                            {
                                "title": "Environment",
                                "value": data.get("environment", "unknown"),
                                "short": True,
                            },
                            {
                                "title": "Version",
                                "value": data.get("version", "unknown"),
                                "short": True,
                            },
                            {
                                "title": "Timestamp",
                                "value": data.get(
                                    "timestamp", datetime.utcnow().isoformat()
                                ),
                                "short": False,
                            },
                        ],
                    }
                ],
            }

            response = requests.post(slack_config["webhook_url"], json=payload)

            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(
                    f"Failed to send Slack notification: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def _get_base_url(self, environment: str) -> str:
        """Get base URL for environment."""
        if environment == "production":
            return "https://api.anomaly_detection.io"
        elif environment == "staging":
            return "https://staging-api.anomaly_detection.io"
        else:
            return "http://localhost:8000"

    def _record_deployment(self, environment: str, version: str, success: bool):
        """Record deployment in history."""
        deployment_record = {
            "environment": environment,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
        }
        self.deployment_history.append(deployment_record)

    async def _rollback(self, environment: str) -> bool:
        """Rollback to previous version."""
        logger.info(f"Initiating rollback for {environment}")

        try:
            # Find last successful deployment
            last_success = None
            for deployment in reversed(self.deployment_history):
                if deployment["environment"] == environment and deployment["success"]:
                    last_success = deployment
                    break

            if not last_success:
                logger.error("No previous successful deployment found for rollback")
                return False

            # Execute rollback
            success = await self.deploy(
                environment, last_success["version"], force=True
            )

            if success:
                await self._notify(
                    "rollback_complete",
                    {
                        "environment": environment,
                        "version": last_success["version"],
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

            return success

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


@click.command()
@click.option(
    "--environment",
    "-e",
    required=True,
    type=click.Choice(["development", "staging", "production"]),
    help="Target environment",
)
@click.option("--version", "-v", default="latest", help="Version to deploy")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option("--force", "-f", is_flag=True, help="Force deployment without approval")
@click.option("--dry-run", "-d", is_flag=True, help="Dry run without actual deployment")
def main(
    environment: str, version: str, config: str | None, force: bool, dry_run: bool
):
    """Automated deployment script for anomaly_detection."""

    config_path = Path(config) if config else None
    deployer = AutomatedDeployer(config_path)

    success = asyncio.run(deployer.deploy(environment, version, force, dry_run))

    if success:
        click.echo(f"✅ Deployment to {environment} completed successfully!")
        sys.exit(0)
    else:
        click.echo(f"❌ Deployment to {environment} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
