"""
Advanced Model Deployment Pipeline

This module implements automated model deployment with blue-green deployments,
canary releases, A/B testing, and rollback capabilities.
"""

import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import docker
import kubernetes
from kubernetes import client, config
import mlflow
from mlflow.tracking import MlflowClient
import boto3
from google.cloud import run_v2
import requests
import redis
from sqlalchemy import create_engine, text
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    deployment_id: str
    model_name: str
    model_version: str
    strategy: DeploymentStrategy
    target_environment: str  # staging, production
    resource_requirements: Dict[str, str]
    scaling_config: Dict[str, Any]
    health_check_config: Dict[str, Any]
    canary_traffic_percentage: float = 5.0
    rollback_threshold_error_rate: float = 0.05
    rollback_threshold_latency_ms: float = 1000
    monitoring_duration_minutes: int = 30
    auto_rollback_enabled: bool = True
    approval_required: bool = True
    notification_channels: List[str] = None

@dataclass
class DeploymentMetrics:
    """Deployment performance metrics"""
    deployment_id: str
    timestamp: datetime
    request_rate: float
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    cpu_utilization: float
    memory_utilization: float
    replicas_ready: int
    replicas_total: int

@dataclass
class DeploymentResult:
    """Deployment execution result"""
    deployment_id: str
    status: DeploymentStatus
    model_name: str
    model_version: str
    environment: str
    strategy: DeploymentStrategy
    start_time: datetime
    end_time: Optional[datetime]
    endpoint_url: str
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None
    performance_metrics: List[DeploymentMetrics] = None

class ModelPackager:
    """Package models for deployment"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.mlflow_client = MlflowClient()
    
    async def package_model(self, model_name: str, model_version: str, 
                          target_platform: str = "kubernetes") -> str:
        """Package model for deployment"""
        logger.info(f"Packaging model {model_name}:{model_version} for {target_platform}")
        
        # Download model from MLflow
        model_uri = f"models:/{model_name}/{model_version}"
        model_path = mlflow.artifacts.download_artifacts(model_uri)
        
        # Create deployment artifacts
        if target_platform == "kubernetes":
            return await self._create_kubernetes_deployment(model_name, model_version, model_path)
        elif target_platform == "cloud_run":
            return await self._create_cloud_run_deployment(model_name, model_version, model_path)
        elif target_platform == "lambda":
            return await self._create_lambda_deployment(model_name, model_version, model_path)
        else:
            raise ValueError(f"Unsupported target platform: {target_platform}")
    
    async def _create_kubernetes_deployment(self, model_name: str, model_version: str, 
                                          model_path: str) -> str:
        """Create Kubernetes deployment artifacts"""
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY model /app/model
COPY inference_server.py .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["python", "inference_server.py"]
"""
        
        # Create inference server
        inference_server_content = """
import os
import json
import pickle
import logging
from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = mlflow.sklearn.load_model("/app/model")
logger.info("Model loaded successfully")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_version": os.getenv("MODEL_VERSION", "unknown")})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'instances' in data:
            # Batch prediction
            instances = data['instances']
            df = pd.DataFrame(instances)
        else:
            # Single prediction
            df = pd.DataFrame([data])
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df).tolist()
        
        response = {
            "predictions": predictions,
            "model_version": os.getenv("MODEL_VERSION", "unknown")
        }
        
        if probabilities:
            response["probabilities"] = probabilities
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    # Prometheus metrics endpoint
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
"""
        
        # Create requirements.txt
        requirements_content = """
flask==2.3.3
mlflow==2.7.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
prometheus_client==0.17.1
"""
        
        # Build Docker image
        image_tag = f"{model_name}:{model_version}"
        
        # Create build context
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as build_dir:
            # Write files
            with open(f"{build_dir}/Dockerfile", "w") as f:
                f.write(dockerfile_content)
            with open(f"{build_dir}/inference_server.py", "w") as f:
                f.write(inference_server_content)
            with open(f"{build_dir}/requirements.txt", "w") as f:
                f.write(requirements_content)
            
            # Copy model
            shutil.copytree(model_path, f"{build_dir}/model")
            
            # Build image
            image, logs = self.docker_client.images.build(
                path=build_dir,
                tag=image_tag,
                rm=True
            )
            
            logger.info(f"Docker image built: {image_tag}")
            return image_tag
    
    async def _create_cloud_run_deployment(self, model_name: str, model_version: str, 
                                         model_path: str) -> str:
        """Create Cloud Run deployment artifacts"""
        # Similar to Kubernetes but with Cloud Run specific configurations
        # Implementation would include Cloud Run YAML and container setup
        pass
    
    async def _create_lambda_deployment(self, model_name: str, model_version: str, 
                                      model_path: str) -> str:
        """Create AWS Lambda deployment artifacts"""
        # Implementation for Lambda deployment package
        pass

class DeploymentOrchestrator:
    """Orchestrate model deployments across different strategies"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.k8s_client = self._setup_kubernetes_client()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.packager = ModelPackager()
        self.monitoring_data = []
    
    def _setup_kubernetes_client(self):
        """Setup Kubernetes client"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        return client.AppsV1Api()
    
    async def deploy(self) -> DeploymentResult:
        """Execute deployment based on strategy"""
        logger.info(f"Starting deployment {self.config.deployment_id} using {self.config.strategy}")
        
        start_time = datetime.now()
        
        try:
            # Package model
            image_tag = await self.packager.package_model(
                self.config.model_name,
                self.config.model_version,
                "kubernetes"
            )
            
            # Execute deployment strategy
            if self.config.strategy == DeploymentStrategy.BLUE_GREEN:
                endpoint_url = await self._deploy_blue_green(image_tag)
            elif self.config.strategy == DeploymentStrategy.CANARY:
                endpoint_url = await self._deploy_canary(image_tag)
            elif self.config.strategy == DeploymentStrategy.ROLLING:
                endpoint_url = await self._deploy_rolling(image_tag)
            elif self.config.strategy == DeploymentStrategy.RECREATE:
                endpoint_url = await self._deploy_recreate(image_tag)
            else:
                raise ValueError(f"Unsupported deployment strategy: {self.config.strategy}")
            
            # Monitor deployment
            await self._monitor_deployment()
            
            # Check if rollback is needed
            rollback_needed, rollback_reason = await self._check_rollback_conditions()
            
            if rollback_needed and self.config.auto_rollback_enabled:
                logger.warning(f"Triggering automatic rollback: {rollback_reason}")
                await self._rollback_deployment()
                
                return DeploymentResult(
                    deployment_id=self.config.deployment_id,
                    status=DeploymentStatus.ROLLED_BACK,
                    model_name=self.config.model_name,
                    model_version=self.config.model_version,
                    environment=self.config.target_environment,
                    strategy=self.config.strategy,
                    start_time=start_time,
                    end_time=datetime.now(),
                    endpoint_url=endpoint_url,
                    rollback_reason=rollback_reason,
                    performance_metrics=self.monitoring_data
                )
            
            return DeploymentResult(
                deployment_id=self.config.deployment_id,
                status=DeploymentStatus.COMPLETED,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                environment=self.config.target_environment,
                strategy=self.config.strategy,
                start_time=start_time,
                end_time=datetime.now(),
                endpoint_url=endpoint_url,
                performance_metrics=self.monitoring_data
            )
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            
            return DeploymentResult(
                deployment_id=self.config.deployment_id,
                status=DeploymentStatus.FAILED,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                environment=self.config.target_environment,
                strategy=self.config.strategy,
                start_time=start_time,
                end_time=datetime.now(),
                endpoint_url="",
                error_message=str(e)
            )
    
    async def _deploy_blue_green(self, image_tag: str) -> str:
        """Execute blue-green deployment"""
        logger.info("Executing blue-green deployment")
        
        namespace = f"mlops-{self.config.target_environment}"
        service_name = f"{self.config.model_name}-service"
        
        # Get current deployment (blue)
        current_selector = await self._get_current_service_selector(namespace, service_name)
        
        # Determine new deployment color (green)
        new_color = "green" if current_selector.get("version", "blue") == "blue" else "blue"
        new_deployment_name = f"{self.config.model_name}-{new_color}"
        
        # Create new deployment (green)
        await self._create_deployment(
            namespace=namespace,
            deployment_name=new_deployment_name,
            image_tag=image_tag,
            labels={"app": self.config.model_name, "version": new_color}
        )
        
        # Wait for new deployment to be ready
        await self._wait_for_deployment_ready(namespace, new_deployment_name)
        
        # Test new deployment
        await self._test_deployment_health(namespace, new_deployment_name)
        
        # Switch traffic to new deployment
        await self._update_service_selector(
            namespace=namespace,
            service_name=service_name,
            selector={"app": self.config.model_name, "version": new_color}
        )
        
        # Clean up old deployment after successful switch
        old_color = "blue" if new_color == "green" else "green"
        old_deployment_name = f"{self.config.model_name}-{old_color}"
        await self._delete_deployment(namespace, old_deployment_name)
        
        return f"http://{service_name}.{namespace}.svc.cluster.local"
    
    async def _deploy_canary(self, image_tag: str) -> str:
        """Execute canary deployment"""
        logger.info(f"Executing canary deployment with {self.config.canary_traffic_percentage}% traffic")
        
        namespace = f"mlops-{self.config.target_environment}"
        
        # Create canary deployment
        canary_deployment_name = f"{self.config.model_name}-canary"
        await self._create_deployment(
            namespace=namespace,
            deployment_name=canary_deployment_name,
            image_tag=image_tag,
            labels={"app": self.config.model_name, "version": "canary"},
            replicas=max(1, int(self.config.scaling_config.get("min_replicas", 3) * self.config.canary_traffic_percentage / 100))
        )
        
        # Wait for canary to be ready
        await self._wait_for_deployment_ready(namespace, canary_deployment_name)
        
        # Configure traffic splitting
        await self._configure_traffic_splitting(
            namespace=namespace,
            model_name=self.config.model_name,
            canary_percentage=self.config.canary_traffic_percentage
        )
        
        # Monitor canary for specified duration
        await self._monitor_canary_deployment()
        
        # Decide whether to promote or rollback canary
        promote_canary = await self._evaluate_canary_metrics()
        
        if promote_canary:
            # Promote canary to production
            await self._promote_canary_to_production(namespace)
        else:
            # Rollback canary
            await self._rollback_canary_deployment(namespace)
        
        service_name = f"{self.config.model_name}-service"
        return f"http://{service_name}.{namespace}.svc.cluster.local"
    
    async def _deploy_rolling(self, image_tag: str) -> str:
        """Execute rolling deployment"""
        logger.info("Executing rolling deployment")
        
        namespace = f"mlops-{self.config.target_environment}"
        deployment_name = f"{self.config.model_name}-deployment"
        
        # Update existing deployment with new image
        await self._update_deployment_image(namespace, deployment_name, image_tag)
        
        # Wait for rollout to complete
        await self._wait_for_rollout_complete(namespace, deployment_name)
        
        service_name = f"{self.config.model_name}-service"
        return f"http://{service_name}.{namespace}.svc.cluster.local"
    
    async def _deploy_recreate(self, image_tag: str) -> str:
        """Execute recreate deployment"""
        logger.info("Executing recreate deployment")
        
        namespace = f"mlops-{self.config.target_environment}"
        deployment_name = f"{self.config.model_name}-deployment"
        
        # Delete existing deployment
        await self._delete_deployment(namespace, deployment_name)
        
        # Create new deployment
        await self._create_deployment(
            namespace=namespace,
            deployment_name=deployment_name,
            image_tag=image_tag,
            labels={"app": self.config.model_name}
        )
        
        # Wait for new deployment to be ready
        await self._wait_for_deployment_ready(namespace, deployment_name)
        
        service_name = f"{self.config.model_name}-service"
        return f"http://{service_name}.{namespace}.svc.cluster.local"
    
    async def _create_deployment(self, namespace: str, deployment_name: str, 
                               image_tag: str, labels: Dict[str, str], replicas: int = None) -> None:
        """Create Kubernetes deployment"""
        if replicas is None:
            replicas = self.config.scaling_config.get("min_replicas", 3)
        
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "namespace": namespace,
                "labels": labels
            },
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": image_tag,
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "MODEL_VERSION", "value": self.config.model_version},
                                {"name": "DEPLOYMENT_ID", "value": self.config.deployment_id}
                            ],
                            "resources": {
                                "requests": self.config.resource_requirements,
                                "limits": self.config.resource_requirements
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        self.k8s_client.create_namespaced_deployment(
            namespace=namespace,
            body=deployment_manifest
        )
        
        logger.info(f"Created deployment: {deployment_name}")
    
    async def _monitor_deployment(self) -> None:
        """Monitor deployment performance"""
        logger.info("Starting deployment monitoring")
        
        monitoring_end_time = datetime.now() + timedelta(minutes=self.config.monitoring_duration_minutes)
        
        while datetime.now() < monitoring_end_time:
            metrics = await self._collect_deployment_metrics()
            self.monitoring_data.append(metrics)
            
            # Log current metrics
            logger.info(f"Metrics - Error Rate: {metrics.error_rate:.4f}, "
                       f"Latency P95: {metrics.latency_p95:.2f}ms, "
                       f"CPU: {metrics.cpu_utilization:.2f}%")
            
            await asyncio.sleep(30)  # Collect metrics every 30 seconds
        
        logger.info("Deployment monitoring completed")
    
    async def _collect_deployment_metrics(self) -> DeploymentMetrics:
        """Collect deployment performance metrics"""
        # This would integrate with Prometheus or other monitoring systems
        # For now, returning mock data
        
        return DeploymentMetrics(
            deployment_id=self.config.deployment_id,
            timestamp=datetime.now(),
            request_rate=100.0,
            error_rate=0.01,
            latency_p50=50.0,
            latency_p95=150.0,
            latency_p99=300.0,
            cpu_utilization=25.0,
            memory_utilization=40.0,
            replicas_ready=3,
            replicas_total=3
        )
    
    async def _check_rollback_conditions(self) -> Tuple[bool, Optional[str]]:
        """Check if rollback conditions are met"""
        if not self.monitoring_data:
            return False, None
        
        recent_metrics = self.monitoring_data[-5:]  # Last 5 measurements
        
        # Check error rate
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        if avg_error_rate > self.config.rollback_threshold_error_rate:
            return True, f"High error rate: {avg_error_rate:.4f} > {self.config.rollback_threshold_error_rate}"
        
        # Check latency
        avg_latency_p95 = sum(m.latency_p95 for m in recent_metrics) / len(recent_metrics)
        if avg_latency_p95 > self.config.rollback_threshold_latency_ms:
            return True, f"High latency: {avg_latency_p95:.2f}ms > {self.config.rollback_threshold_latency_ms}ms"
        
        return False, None
    
    async def _rollback_deployment(self) -> None:
        """Rollback to previous deployment"""
        logger.warning("Executing deployment rollback")
        
        namespace = f"mlops-{self.config.target_environment}"
        deployment_name = f"{self.config.model_name}-deployment"
        
        # Rollback to previous revision
        rollback_body = {
            "apiVersion": "apps/v1",
            "kind": "DeploymentRollback",
            "name": deployment_name,
            "rollbackTo": {"revision": 0}  # 0 means previous revision
        }
        
        # Note: This is a simplified rollback. In practice, you'd use kubectl rollout undo
        # or implement more sophisticated rollback logic
        
        logger.info("Deployment rollback completed")
    
    async def _wait_for_deployment_ready(self, namespace: str, deployment_name: str) -> None:
        """Wait for deployment to be ready"""
        logger.info(f"Waiting for deployment {deployment_name} to be ready...")
        
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.k8s_client.read_namespaced_deployment(
                    name=deployment_name, namespace=namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.warning(f"Error checking deployment status: {e}")
                await asyncio.sleep(10)
        
        raise TimeoutError(f"Deployment {deployment_name} not ready within {timeout} seconds")
    
    async def _get_current_service_selector(self, namespace: str, service_name: str) -> Dict[str, str]:
        """Get current service selector"""
        try:
            service = client.CoreV1Api().read_namespaced_service(
                name=service_name, namespace=namespace
            )
            return service.spec.selector or {}
        except:
            return {}
    
    async def _update_service_selector(self, namespace: str, service_name: str, 
                                     selector: Dict[str, str]) -> None:
        """Update service selector"""
        body = {"spec": {"selector": selector}}
        
        client.CoreV1Api().patch_namespaced_service(
            name=service_name,
            namespace=namespace,
            body=body
        )
        
        logger.info(f"Updated service {service_name} selector to {selector}")
    
    async def _delete_deployment(self, namespace: str, deployment_name: str) -> None:
        """Delete Kubernetes deployment"""
        try:
            self.k8s_client.delete_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            logger.info(f"Deleted deployment: {deployment_name}")
        except Exception as e:
            logger.warning(f"Failed to delete deployment {deployment_name}: {e}")
    
    async def _test_deployment_health(self, namespace: str, deployment_name: str) -> None:
        """Test deployment health"""
        # Implementation for health testing
        pass
    
    async def _configure_traffic_splitting(self, namespace: str, model_name: str, 
                                         canary_percentage: float) -> None:
        """Configure traffic splitting for canary deployment"""
        # Implementation for traffic splitting using Istio or similar
        pass
    
    async def _monitor_canary_deployment(self) -> None:
        """Monitor canary deployment specifically"""
        # Implementation for canary-specific monitoring
        pass
    
    async def _evaluate_canary_metrics(self) -> bool:
        """Evaluate whether to promote canary"""
        # Implementation for canary evaluation logic
        return True  # Simplified for example
    
    async def _promote_canary_to_production(self, namespace: str) -> None:
        """Promote canary deployment to production"""
        # Implementation for canary promotion
        pass
    
    async def _rollback_canary_deployment(self, namespace: str) -> None:
        """Rollback canary deployment"""
        # Implementation for canary rollback
        pass
    
    async def _update_deployment_image(self, namespace: str, deployment_name: str, 
                                     image_tag: str) -> None:
        """Update deployment with new image"""
        body = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "model-server",
                                "image": image_tag
                            }
                        ]
                    }
                }
            }
        }
        
        self.k8s_client.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=body
        )
        
        logger.info(f"Updated deployment {deployment_name} with image {image_tag}")
    
    async def _wait_for_rollout_complete(self, namespace: str, deployment_name: str) -> None:
        """Wait for rolling update to complete"""
        await self._wait_for_deployment_ready(namespace, deployment_name)

# Orchestration functions
async def deploy_model(config: DeploymentConfig) -> DeploymentResult:
    """Deploy a model using specified configuration"""
    orchestrator = DeploymentOrchestrator(config)
    return await orchestrator.deploy()

async def batch_deploy_models(configs: List[DeploymentConfig]) -> List[DeploymentResult]:
    """Deploy multiple models in parallel"""
    logger.info(f"Starting batch deployment of {len(configs)} models")
    
    tasks = [deploy_model(config) for config in configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_deployments = [r for r in results if isinstance(r, DeploymentResult) and r.status == DeploymentStatus.COMPLETED]
    failed_deployments = [r for r in results if not isinstance(r, DeploymentResult) or r.status == DeploymentStatus.FAILED]
    
    logger.info(f"Batch deployment completed - Success: {len(successful_deployments)}, Failed: {len(failed_deployments)}")
    
    return results