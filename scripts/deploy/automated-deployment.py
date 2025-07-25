#!/usr/bin/env python3
"""
Comprehensive Automated Deployment Script
Manages end-to-end deployment automation with advanced rollback capabilities
"""

import os
import sys
import json
import yaml
import time
import argparse
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
import boto3
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'deployment-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Deployment strategy options."""
    BLUE_GREEN = "blue-green"
    CANARY = "canary"
    ROLLING = "rolling"
    FEATURE_FLAG = "feature-flag"


class DeploymentStatus(Enum):
    """Deployment status tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str
    region: str
    cluster_name: str
    namespace: str
    image_tag: str
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    replicas: int = 3
    timeout: int = 600
    rollback_enabled: bool = True
    health_check_url: str = ""
    slack_webhook: str = ""
    feature_flags: List[str] = field(default_factory=list)
    canary_percentage: int = 10
    resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Deployment execution result."""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class KubernetesManager:
    """Kubernetes cluster management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Kubernetes manager."""
        try:
            if config_path:
                config.load_kube_config(config_file=config_path)
            else:
                config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.custom_objects = client.CustomObjectsApi()
    
    def get_deployment(self, name: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Get deployment information."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            return {
                'name': deployment.metadata.name,
                'namespace': deployment.metadata.namespace,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'image': deployment.spec.template.spec.containers[0].image,
                'labels': deployment.metadata.labels or {}
            }
        except ApiException as e:
            if e.status == 404:
                return None
            raise
    
    def update_deployment_image(self, name: str, namespace: str, image: str) -> bool:
        """Update deployment image."""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            
            # Update image
            deployment.spec.template.spec.containers[0].image = image
            
            # Add deployment annotations
            if not deployment.spec.template.metadata.annotations:
                deployment.spec.template.metadata.annotations = {}
            
            deployment.spec.template.metadata.annotations.update({
                'deployment.kubernetes.io/revision': str(int(time.time())),
                'deployment.kubernetes.io/timestamp': datetime.utcnow().isoformat()
            })
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Updated deployment {name} with image {image}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to update deployment: {e}")
            return False
    
    def wait_for_rollout(self, name: str, namespace: str, timeout: int = 600) -> bool:
        """Wait for deployment rollout to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
                
                desired_replicas = deployment.spec.replicas
                ready_replicas = deployment.status.ready_replicas or 0
                
                if ready_replicas == desired_replicas:
                    logger.info(f"Deployment {name} rollout completed successfully")
                    return True
                
                logger.info(f"Waiting for rollout: {ready_replicas}/{desired_replicas} ready")
                time.sleep(10)
                
            except ApiException as e:
                logger.error(f"Error checking deployment status: {e}")
                time.sleep(10)
        
        logger.error(f"Deployment rollout timed out after {timeout} seconds")
        return False
    
    def rollback_deployment(self, name: str, namespace: str) -> bool:
        """Rollback deployment to previous version."""
        try:
            # Get deployment
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            
            # Get revision history
            replica_sets = self.apps_v1.list_namespaced_replica_set(
                namespace=namespace,
                label_selector=f"app={name}"
            )
            
            # Find previous version
            previous_rs = None
            for rs in replica_sets.items:
                if rs.metadata.annotations and rs.spec.replicas == 0:
                    revision = rs.metadata.annotations.get('deployment.kubernetes.io/revision')
                    if revision and int(revision) == int(deployment.metadata.annotations.get('deployment.kubernetes.io/revision', 0)) - 1:
                        previous_rs = rs
                        break
            
            if not previous_rs:
                logger.error("No previous version found for rollback")
                return False
            
            # Update deployment to previous image
            previous_image = previous_rs.spec.template.spec.containers[0].image
            deployment.spec.template.spec.containers[0].image = previous_image
            
            # Add rollback annotations
            if not deployment.spec.template.metadata.annotations:
                deployment.spec.template.metadata.annotations = {}
            
            deployment.spec.template.metadata.annotations.update({
                'deployment.kubernetes.io/rollback': 'true',
                'deployment.kubernetes.io/rollback-timestamp': datetime.utcnow().isoformat()
            })
            
            # Apply rollback
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Initiated rollback for deployment {name} to image {previous_image}")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to rollback deployment: {e}")
            return False
    
    def create_rollout(self, name: str, namespace: str, config: DeploymentConfig) -> bool:
        """Create Argo Rollout for advanced deployment strategies."""
        rollout_manifest = {
            'apiVersion': 'argoproj.io/v1alpha1',
            'kind': 'Rollout',
            'metadata': {
                'name': name,
                'namespace': namespace,
                'labels': {
                    'app': name,
                    'environment': config.environment,
                    'strategy': config.strategy.value
                }
            },
            'spec': {
                'replicas': config.replicas,
                'strategy': self._get_rollout_strategy(config),
                'selector': {
                    'matchLabels': {
                        'app': name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': name,
                            'version': config.image_tag
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': name,
                            'image': f"ghcr.io/elgerytme/monorepo:{config.image_tag}",
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': config.environment},
                                {'name': 'FEATURE_FLAGS', 'value': ','.join(config.feature_flags)}
                            ],
                            'resources': config.resources,
                            'livenessProbe': {
                                'httpGet': {'path': '/api/health/live', 'port': 8000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/api/health/ready', 'port': 8000},
                                'initialDelaySeconds': 15,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        try:
            self.custom_objects.create_namespaced_custom_object(
                group='argoproj.io',
                version='v1alpha1',
                namespace=namespace,
                plural='rollouts',
                body=rollout_manifest
            )
            logger.info(f"Created rollout {name} with strategy {config.strategy.value}")
            return True
        except ApiException as e:
            if e.status == 409:  # Already exists
                # Update existing rollout
                try:
                    self.custom_objects.patch_namespaced_custom_object(
                        group='argoproj.io',
                        version='v1alpha1',
                        namespace=namespace,
                        plural='rollouts',
                        name=name,
                        body=rollout_manifest
                    )
                    logger.info(f"Updated existing rollout {name}")
                    return True
                except ApiException as update_e:
                    logger.error(f"Failed to update rollout: {update_e}")
                    return False
            logger.error(f"Failed to create rollout: {e}")
            return False
    
    def _get_rollout_strategy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Get rollout strategy configuration."""
        if config.strategy == DeploymentStrategy.BLUE_GREEN:
            return {
                'blueGreen': {
                    'activeService': f"{config.namespace}-active",
                    'previewService': f"{config.namespace}-preview",
                    'autoPromotionEnabled': False,
                    'scaleDownDelaySeconds': 30,
                    'prePromotionAnalysis': {
                        'templates': [{'templateName': 'success-rate'}],
                        'args': [{'name': 'service-name', 'value': f"{config.namespace}-preview"}]
                    }
                }
            }
        elif config.strategy == DeploymentStrategy.CANARY:
            return {
                'canary': {
                    'steps': [
                        {'setWeight': config.canary_percentage},
                        {'pause': {'duration': '1m'}},
                        {'setWeight': 50},
                        {'pause': {'duration': '5m'}},
                        {'setWeight': 100}
                    ],
                    'analysis': {
                        'templates': [{'templateName': 'success-rate'}],
                        'startingStep': 2
                    }
                }
            }
        else:  # Rolling update
            return {
                'rollingUpdate': {
                    'maxUnavailable': 1,
                    'maxSurge': 1
                }
            }


class HealthChecker:
    """Application health checking."""
    
    def __init__(self, base_url: str):
        """Initialize health checker."""
        self.base_url = base_url.rstrip('/')
    
    def check_health(self, endpoint: str = '/api/health/ready', timeout: int = 30) -> Tuple[bool, str]:
        """Check application health."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                return True, "Health check passed"
            else:
                return False, f"Health check failed with status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Health check failed: {str(e)}"
    
    def wait_for_health(self, endpoint: str = '/api/health/ready', 
                       timeout: int = 300, interval: int = 10) -> bool:
        """Wait for application to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            healthy, message = self.check_health(endpoint)
            
            if healthy:
                logger.info("Application health check passed")
                return True
            
            logger.info(f"Health check failed: {message}. Retrying in {interval}s...")
            time.sleep(interval)
        
        logger.error(f"Application failed to become healthy within {timeout} seconds")
        return False


class NotificationManager:
    """Deployment notification management."""
    
    def __init__(self, slack_webhook: Optional[str] = None):
        """Initialize notification manager."""
        self.slack_webhook = slack_webhook
    
    def send_deployment_start(self, config: DeploymentConfig, deployment_id: str):
        """Send deployment start notification."""
        message = {
            "text": f"🚀 Deployment Started",
            "attachments": [{
                "color": "warning",
                "fields": [
                    {"title": "Deployment ID", "value": deployment_id, "short": True},
                    {"title": "Environment", "value": config.environment, "short": True},
                    {"title": "Strategy", "value": config.strategy.value, "short": True},
                    {"title": "Image Tag", "value": config.image_tag, "short": True}
                ]
            }]
        }
        self._send_slack_message(message)
    
    def send_deployment_success(self, config: DeploymentConfig, deployment_id: str, duration: float):
        """Send deployment success notification."""
        message = {
            "text": f"✅ Deployment Successful",
            "attachments": [{
                "color": "good",
                "fields": [
                    {"title": "Deployment ID", "value": deployment_id, "short": True},
                    {"title": "Environment", "value": config.environment, "short": True},
                    {"title": "Duration", "value": f"{duration:.1f}s", "short": True},
                    {"title": "Strategy", "value": config.strategy.value, "short": True}
                ]
            }]
        }
        self._send_slack_message(message)
    
    def send_deployment_failure(self, config: DeploymentConfig, deployment_id: str, error: str):
        """Send deployment failure notification."""
        message = {
            "text": f"❌ Deployment Failed",
            "attachments": [{
                "color": "danger",
                "fields": [
                    {"title": "Deployment ID", "value": deployment_id, "short": True},
                    {"title": "Environment", "value": config.environment, "short": True},
                    {"title": "Error", "value": error[:500], "short": False}
                ]
            }]
        }
        self._send_slack_message(message)
    
    def send_rollback_notification(self, config: DeploymentConfig, deployment_id: str):
        """Send rollback notification."""
        message = {
            "text": f"🔄 Automated Rollback Triggered",
            "attachments": [{
                "color": "warning",
                "fields": [
                    {"title": "Deployment ID", "value": deployment_id, "short": True},
                    {"title": "Environment", "value": config.environment, "short": True},
                    {"title": "Action", "value": "Automatic rollback initiated", "short": False}
                ]
            }]
        }
        self._send_slack_message(message)
    
    def _send_slack_message(self, message: Dict[str, Any]):
        """Send message to Slack."""
        if not self.slack_webhook:
            logger.info("No Slack webhook configured, skipping notification")
            return
        
        try:
            response = requests.post(
                self.slack_webhook,
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(f"Failed to send Slack notification: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")


class DeploymentAutomation:
    """Main deployment automation orchestrator."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize deployment automation."""
        self.config = config
        self.deployment_id = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{config.image_tag[:8]}"
        
        # Initialize managers
        self.k8s = KubernetesManager()
        self.notifications = NotificationManager(config.slack_webhook)
        
        if config.health_check_url:
            self.health_checker = HealthChecker(config.health_check_url)
        else:
            self.health_checker = None
        
        logger.info(f"Initialized deployment automation with ID: {self.deployment_id}")
    
    def execute_deployment(self) -> DeploymentResult:
        """Execute complete deployment process."""
        result = DeploymentResult(
            deployment_id=self.deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=datetime.utcnow()
        )
        
        try:
            # Send start notification
            self.notifications.send_deployment_start(self.config, self.deployment_id)
            
            # Update status
            result.status = DeploymentStatus.IN_PROGRESS
            
            # Execute deployment based on strategy
            if self.config.strategy in [DeploymentStrategy.BLUE_GREEN, DeploymentStrategy.CANARY]:
                success = self._execute_advanced_deployment()
            else:
                success = self._execute_rolling_deployment()
            
            if success:
                # Verify deployment health
                if self.health_checker:
                    healthy = self.health_checker.wait_for_health(timeout=self.config.timeout)
                    if not healthy and self.config.rollback_enabled:
                        logger.warning("Health check failed, triggering rollback")
                        self._execute_rollback()
                        result.rollback_performed = True
                        result.status = DeploymentStatus.ROLLED_BACK
                    else:
                        result.status = DeploymentStatus.SUCCESS
                else:
                    result.status = DeploymentStatus.SUCCESS
            else:
                result.status = DeploymentStatus.FAILED
                if self.config.rollback_enabled:
                    logger.info("Deployment failed, triggering rollback")
                    self._execute_rollback()
                    result.rollback_performed = True
            
            result.end_time = datetime.utcnow()
            
            # Send completion notification
            if result.status == DeploymentStatus.SUCCESS:
                duration = (result.end_time - result.start_time).total_seconds()
                self.notifications.send_deployment_success(self.config, self.deployment_id, duration)
            else:
                self.notifications.send_deployment_failure(
                    self.config, self.deployment_id, result.error_message or "Unknown error"
                )
            
            return result
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            
            logger.error(f"Deployment failed with exception: {e}")
            
            # Attempt rollback on exception
            if self.config.rollback_enabled:
                try:
                    self._execute_rollback()
                    result.rollback_performed = True
                except Exception as rollback_e:
                    logger.error(f"Rollback also failed: {rollback_e}")
            
            self.notifications.send_deployment_failure(self.config, self.deployment_id, str(e))
            return result
    
    def _execute_rolling_deployment(self) -> bool:
        """Execute rolling deployment."""
        logger.info("Executing rolling deployment")
        
        deployment_name = f"anomaly-detection"
        
        # Update deployment image
        success = self.k8s.update_deployment_image(
            deployment_name,
            self.config.namespace,
            f"ghcr.io/elgerytme/monorepo:{self.config.image_tag}"
        )
        
        if not success:
            return False
        
        # Wait for rollout to complete
        return self.k8s.wait_for_rollout(
            deployment_name,
            self.config.namespace,
            self.config.timeout
        )
    
    def _execute_advanced_deployment(self) -> bool:
        """Execute advanced deployment using Argo Rollouts."""
        logger.info(f"Executing {self.config.strategy.value} deployment")
        
        rollout_name = "anomaly-detection-rollout"
        
        # Create or update rollout
        success = self.k8s.create_rollout(rollout_name, self.config.namespace, self.config)
        
        if not success:
            return False
        
        # Monitor rollout progress
        return self._monitor_rollout_progress(rollout_name)
    
    def _monitor_rollout_progress(self, rollout_name: str) -> bool:
        """Monitor Argo Rollout progress."""
        start_time = time.time()
        
        while time.time() - start_time < self.config.timeout:
            try:
                rollout = self.k8s.custom_objects.get_namespaced_custom_object(
                    group='argoproj.io',
                    version='v1alpha1',
                    namespace=self.config.namespace,
                    plural='rollouts',
                    name=rollout_name
                )
                
                status = rollout.get('status', {})
                phase = status.get('phase', 'Unknown')
                
                logger.info(f"Rollout status: {phase}")
                
                if phase == 'Healthy':
                    logger.info("Rollout completed successfully")
                    return True
                elif phase in ['Degraded', 'ScaledDown']:
                    logger.error(f"Rollout failed with phase: {phase}")
                    return False
                
                time.sleep(10)
                
            except ApiException as e:
                logger.error(f"Error monitoring rollout: {e}")
                time.sleep(10)
        
        logger.error("Rollout monitoring timed out")
        return False
    
    def _execute_rollback(self) -> bool:
        """Execute deployment rollback."""
        logger.info("Executing deployment rollback")
        
        # Send rollback notification
        self.notifications.send_rollback_notification(self.config, self.deployment_id)
        
        deployment_name = "anomaly-detection"
        
        # Perform rollback
        success = self.k8s.rollback_deployment(deployment_name, self.config.namespace)
        
        if success:
            # Wait for rollback to complete
            return self.k8s.wait_for_rollout(
                deployment_name,
                self.config.namespace,
                self.config.timeout
            )
        
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automated Deployment Script")
    
    parser.add_argument("--environment", required=True, help="Deployment environment")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--cluster-name", required=True, help="EKS cluster name")
    parser.add_argument("--namespace", help="Kubernetes namespace (defaults to environment)")
    parser.add_argument("--image-tag", required=True, help="Docker image tag to deploy")
    parser.add_argument("--strategy", default="rolling", 
                       choices=["blue-green", "canary", "rolling", "feature-flag"],
                       help="Deployment strategy")
    parser.add_argument("--replicas", type=int, default=3, help="Number of replicas")
    parser.add_argument("--timeout", type=int, default=600, help="Deployment timeout in seconds")
    parser.add_argument("--no-rollback", action="store_true", help="Disable automatic rollback")
    parser.add_argument("--health-check-url", help="Base URL for health checks")
    parser.add_argument("--slack-webhook", help="Slack webhook URL for notifications")
    parser.add_argument("--feature-flags", help="Comma-separated list of feature flags")
    parser.add_argument("--canary-percentage", type=int, default=10, help="Canary traffic percentage")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without actual deployment")
    
    return parser.parse_args()


def main():
    """Main deployment execution."""
    args = parse_arguments()
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        region=args.region,
        cluster_name=args.cluster_name,
        namespace=args.namespace or args.environment,
        image_tag=args.image_tag,
        strategy=DeploymentStrategy(args.strategy),
        replicas=args.replicas,
        timeout=args.timeout,
        rollback_enabled=not args.no_rollback,
        health_check_url=args.health_check_url or "",
        slack_webhook=args.slack_webhook or "",
        feature_flags=args.feature_flags.split(",") if args.feature_flags else [],
        canary_percentage=args.canary_percentage,
        resources={
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"}
        }
    )
    
    logger.info(f"Starting deployment with configuration: {config}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE: No actual deployment will be performed")
        logger.info(f"Would deploy image tag: {config.image_tag}")
        logger.info(f"Would use strategy: {config.strategy.value}")
        logger.info(f"Would deploy to: {config.environment}")
        return 0
    
    # Execute deployment
    automation = DeploymentAutomation(config)
    result = automation.execute_deployment()
    
    # Log results
    logger.info(f"Deployment completed with status: {result.status.value}")
    logger.info(f"Deployment ID: {result.deployment_id}")
    
    if result.rollback_performed:
        logger.warning("Automatic rollback was performed")
    
    if result.error_message:
        logger.error(f"Error: {result.error_message}")
    
    # Write deployment report
    report = {
        "deployment_id": result.deployment_id,
        "status": result.status.value,
        "start_time": result.start_time.isoformat(),
        "end_time": result.end_time.isoformat() if result.end_time else None,
        "duration": (result.end_time - result.start_time).total_seconds() if result.end_time else None,
        "config": {
            "environment": config.environment,
            "strategy": config.strategy.value,
            "image_tag": config.image_tag,
            "replicas": config.replicas
        },
        "rollback_performed": result.rollback_performed,
        "error_message": result.error_message
    }
    
    report_file = f"deployment-report-{result.deployment_id}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Deployment report written to: {report_file}")
    
    # Exit with appropriate code
    if result.status == DeploymentStatus.SUCCESS:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())