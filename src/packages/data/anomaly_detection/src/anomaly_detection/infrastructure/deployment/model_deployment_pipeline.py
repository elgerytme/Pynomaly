"""Automated model deployment pipeline for MLOps."""

import logging
import json
import yaml
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import subprocess
import tempfile

try:
    from ai.mlops.domain.services.mlops_service import MLOpsService
    from ai.mlops.domain.value_objects.model_value_objects import ModelVersion
except ImportError:
    from anomaly_detection.domain.services.mlops_service import MLOpsService, ModelVersion

from anomaly_detection.domain.services.detection_service import DetectionService

try:
    from ai.mlops.infrastructure.repositories.model_repository import ModelRepository
except ImportError:
    from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentEnvironment(Enum):
    """Deployment environment enumeration."""
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    environment: DeploymentEnvironment
    replicas: int = 3
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    health_check_path: str = "/health"
    readiness_probe_delay: int = 30
    liveness_probe_delay: int = 60
    auto_scaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    canary_percentage: int = 10  # For canary deployments
    rollback_threshold: float = 0.05  # Error rate threshold for rollback


@dataclass
class DeploymentRecord:
    """Record of a model deployment."""
    deployment_id: str
    model_id: str
    model_version: int
    environment: str
    status: DeploymentStatus
    config: DeploymentConfig
    started_at: datetime
    completed_at: Optional[datetime]
    logs: List[str]
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "environment": self.environment,
            "status": self.status.value,
            "config": asdict(self.config),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "logs": self.logs,
            "metrics": self.metrics,
            "error_message": self.error_message
        }


class ModelDeploymentPipeline:
    """Automated model deployment pipeline."""
    
    def __init__(self, 
                 mlops_service: MLOpsService,
                 model_repository: ModelRepository,
                 kubernetes_namespace: str = "anomaly-detection",
                 registry_url: str = "localhost:5000"):
        """Initialize deployment pipeline.
        
        Args:
            mlops_service: MLOps service for model management
            model_repository: Repository for model storage
            kubernetes_namespace: Kubernetes namespace for deployments
            registry_url: Container registry URL
        """
        self.mlops_service = mlops_service
        self.model_repository = model_repository
        self.kubernetes_namespace = kubernetes_namespace
        self.registry_url = registry_url
        self.logger = logging.getLogger(__name__)
        
        # Deployment records
        self._deployments: Dict[str, DeploymentRecord] = {}
        
        # Validation functions
        self._validators: List[Callable[[ModelVersion], bool]] = []
        
        # Notification callbacks
        self._notification_callbacks: List[Callable[[DeploymentRecord], None]] = []
    
    def add_validator(self, validator: Callable[[ModelVersion], bool]):
        """Add a model validation function.
        
        Args:
            validator: Function that takes a ModelVersion and returns bool
        """
        self._validators.append(validator)
    
    def add_notification_callback(self, callback: Callable[[DeploymentRecord], None]):
        """Add a notification callback for deployment events.
        
        Args:
            callback: Function called when deployment status changes
        """
        self._notification_callbacks.append(callback)
    
    async def deploy_model(self,
                          model_id: str,
                          version: int,
                          environment: DeploymentEnvironment,
                          config: Optional[DeploymentConfig] = None) -> str:
        """Deploy a model version to an environment.
        
        Args:
            model_id: ID of the model to deploy
            version: Version number to deploy
            environment: Target environment
            config: Optional deployment configuration
            
        Returns:
            Deployment ID
        """
        # Get model version
        model_versions = self.mlops_service.get_model_versions(model_id)
        target_version = None
        
        for v in model_versions:
            if v.version == version:
                target_version = v
                break
        
        if not target_version:
            raise ValueError(f"Model version {version} not found for model {model_id}")
        
        # Use default config if not provided
        if not config:
            config = DeploymentConfig(environment=environment)
        
        # Create deployment record
        deployment_id = f"deploy-{model_id}-v{version}-{environment.value}-{int(datetime.now().timestamp())}"
        
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            model_version=version,
            environment=environment.value,
            status=DeploymentStatus.PENDING,
            config=config,
            started_at=datetime.now(),
            completed_at=None,
            logs=[],
            metrics={}
        )
        
        self._deployments[deployment_id] = deployment_record
        
        try:
            # Start deployment process
            await self._execute_deployment(deployment_record, target_version)
            
        except Exception as e:
            deployment_record.status = DeploymentStatus.FAILED
            deployment_record.error_message = str(e)
            deployment_record.completed_at = datetime.now()
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            
            # Notify callbacks
            self._notify_callbacks(deployment_record)
            
            raise
        
        return deployment_id
    
    async def _execute_deployment(self, deployment_record: DeploymentRecord, model_version: ModelVersion):
        """Execute the deployment process.
        
        Args:
            deployment_record: Deployment record to update
            model_version: Model version to deploy
        """
        deployment_record.status = DeploymentStatus.IN_PROGRESS
        deployment_record.logs.append(f"Starting deployment at {datetime.now()}")
        
        try:
            # Step 1: Validate model
            await self._validate_model(deployment_record, model_version)
            
            # Step 2: Build container image
            await self._build_container_image(deployment_record, model_version)
            
            # Step 3: Deploy to Kubernetes
            await self._deploy_to_kubernetes(deployment_record, model_version)
            
            # Step 4: Run health checks
            await self._run_health_checks(deployment_record)
            
            # Step 5: Update deployment status
            deployment_record.status = DeploymentStatus.COMPLETED
            deployment_record.completed_at = datetime.now()
            deployment_record.logs.append(f"Deployment completed successfully at {datetime.now()}")
            
            self.logger.info(f"Successfully deployed {deployment_record.model_id} v{deployment_record.model_version}")
            
        except Exception as e:
            deployment_record.status = DeploymentStatus.FAILED
            deployment_record.error_message = str(e)
            deployment_record.completed_at = datetime.now()
            deployment_record.logs.append(f"Deployment failed: {str(e)}")
            raise
        
        finally:
            # Always notify callbacks
            self._notify_callbacks(deployment_record)
    
    async def _validate_model(self, deployment_record: DeploymentRecord, model_version: ModelVersion):
        """Validate model before deployment.
        
        Args:
            deployment_record: Deployment record
            model_version: Model version to validate
        """
        deployment_record.logs.append("Validating model...")
        
        # Run custom validators
        for validator in self._validators:
            if not validator(model_version):
                raise ValueError("Model validation failed")
        
        # Check minimum performance requirements
        min_accuracy = 0.8  # Configurable threshold
        accuracy = model_version.performance_metrics.get("accuracy", 0.0)
        
        if accuracy < min_accuracy:
            raise ValueError(f"Model accuracy {accuracy} below minimum threshold {min_accuracy}")
        
        deployment_record.logs.append("Model validation passed")
    
    async def _build_container_image(self, deployment_record: DeploymentRecord, model_version: ModelVersion):
        """Build container image for the model.
        
        Args:
            deployment_record: Deployment record
            model_version: Model version to containerize
        """
        deployment_record.logs.append("Building container image...")
        
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile(model_version)
            (temp_path / "Dockerfile").write_text(dockerfile_content)
            
            # Copy model files
            model_dir = temp_path / "model"
            model_dir.mkdir()
            
            # Copy model file (simplified - in practice, might be more complex)
            import shutil
            shutil.copy2(model_version.model_path, model_dir / "model.joblib")
            
            # Create requirements.txt
            requirements = [
                "scikit-learn>=1.0.0",
                "numpy>=1.21.0",
                "fastapi>=0.68.0",
                "uvicorn>=0.15.0",
                "joblib>=1.0.0"
            ]
            (temp_path / "requirements.txt").write_text("\n".join(requirements))
            
            # Create serving script
            serving_script = self._generate_serving_script()
            (temp_path / "serve.py").write_text(serving_script)
            
            # Build Docker image
            image_tag = f"{self.registry_url}/anomaly-detection:{deployment_record.model_id}-v{deployment_record.model_version}"
            
            build_cmd = [
                "docker", "build",
                "-t", image_tag,
                str(temp_path)
            ]
            
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Docker build failed: {result.stderr}")
            
            # Push to registry
            push_cmd = ["docker", "push", image_tag]
            result = subprocess.run(push_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Docker push failed: {result.stderr}")
            
            deployment_record.logs.append(f"Built and pushed image: {image_tag}")
    
    def _generate_dockerfile(self, model_version: ModelVersion) -> str:
        """Generate Dockerfile for model serving.
        
        Args:
            model_version: Model version to containerize
            
        Returns:
            Dockerfile content
        """
        return """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and serving script
COPY model/ ./model/
COPY serve.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the serving script
CMD ["python", "serve.py"]
"""
    
    def _generate_serving_script(self) -> str:
        """Generate model serving script.
        
        Returns:
            Python serving script content
        """
        return '''
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="Anomaly Detection Model API")

# Load model
model = joblib.load("model/model.joblib")

class PredictionRequest(BaseModel):
    data: List[List[float]]

class PredictionResponse(BaseModel):
    anomalies: List[int]
    scores: List[float]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions."""
    try:
        data = np.array(request.data)
        
        # Make predictions
        predictions = model.predict(data)
        scores = model.decision_function(data)
        
        return PredictionResponse(
            anomalies=predictions.tolist(),
            scores=scores.tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    async def _deploy_to_kubernetes(self, deployment_record: DeploymentRecord, model_version: ModelVersion):
        """Deploy to Kubernetes.
        
        Args:
            deployment_record: Deployment record
            model_version: Model version to deploy
        """
        deployment_record.logs.append("Deploying to Kubernetes...")
        
        # Generate Kubernetes manifests
        manifests = self._generate_kubernetes_manifests(deployment_record, model_version)
        
        # Apply manifests
        for manifest_name, manifest_content in manifests.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(manifest_content, f)
                manifest_file = f.name
            
            try:
                # Apply manifest
                cmd = ["kubectl", "apply", "-f", manifest_file, "-n", self.kubernetes_namespace]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"kubectl apply failed for {manifest_name}: {result.stderr}")
                
                deployment_record.logs.append(f"Applied {manifest_name}")
                
            finally:
                Path(manifest_file).unlink()
        
        deployment_record.logs.append("Kubernetes deployment completed")
    
    def _generate_kubernetes_manifests(self, deployment_record: DeploymentRecord, model_version: ModelVersion) -> Dict[str, Dict]:
        """Generate Kubernetes deployment manifests.
        
        Args:
            deployment_record: Deployment record
            model_version: Model version
            
        Returns:
            Dictionary of manifest name to manifest content
        """
        config = deployment_record.config
        image_tag = f"{self.registry_url}/anomaly-detection:{deployment_record.model_id}-v{deployment_record.model_version}"
        app_name = f"{deployment_record.model_id}-{deployment_record.environment}"
        
        # Deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": app_name,
                "labels": {
                    "app": app_name,
                    "model": deployment_record.model_id,
                    "version": str(deployment_record.model_version)
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": app_name,
                            "model": deployment_record.model_id,
                            "version": str(deployment_record.model_version)
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": image_tag,
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {
                                    "cpu": config.cpu_request,
                                    "memory": config.memory_request
                                },
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                }
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": config.readiness_probe_delay,
                                "periodSeconds": 10
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": config.liveness_probe_delay,
                                "periodSeconds": 30
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service_manifest = {
            "apiVersion": "v1",  
            "kind": "Service",
            "metadata": {
                "name": f"{app_name}-service",
                "labels": {
                    "app": app_name
                }
            },
            "spec": {
                "selector": {
                    "app": app_name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
        
        manifests = {
            "deployment": deployment_manifest,
            "service": service_manifest
        }
        
        # Add HPA if auto-scaling is enabled
        if config.auto_scaling_enabled:
            hpa_manifest = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{app_name}-hpa"
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": app_name
                    },
                    "minReplicas": config.min_replicas,
                    "maxReplicas": config.max_replicas,
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.target_cpu_utilization
                            }
                        }
                    }]
                }
            }
            manifests["hpa"] = hpa_manifest
        
        return manifests
    
    async def _run_health_checks(self, deployment_record: DeploymentRecord):
        """Run health checks on deployed model.
        
        Args:
            deployment_record: Deployment record
        """
        deployment_record.logs.append("Running health checks...")
        
        # Wait for deployment to be ready
        app_name = f"{deployment_record.model_id}-{deployment_record.environment}"
        
        # Check deployment status
        cmd = ["kubectl", "rollout", "status", f"deployment/{app_name}", "-n", self.kubernetes_namespace]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"Deployment rollout failed: {result.stderr}")
        
        deployment_record.logs.append("Health checks passed")
    
    def _notify_callbacks(self, deployment_record: DeploymentRecord):
        """Notify all registered callbacks about deployment status.
        
        Args:
            deployment_record: Deployment record
        """
        for callback in self._notification_callbacks:
            try:
                callback(deployment_record)
            except Exception as e:
                self.logger.error(f"Notification callback failed: {e}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get deployment status.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Deployment record or None if not found
        """
        return self._deployments.get(deployment_id)
    
    def get_active_deployments(self) -> List[DeploymentRecord]:
        """Get all active deployments.
        
        Returns:
            List of active deployment records
        """
        return [
            deployment for deployment in self._deployments.values()
            if deployment.status in [DeploymentStatus.PENDING, DeploymentStatus.IN_PROGRESS]
        ]
    
    async def rollback_deployment(self, deployment_id: str) -> str:
        """Rollback a deployment.
        
        Args:
            deployment_id: ID of the deployment to rollback
            
        Returns:
            New deployment ID for rollback
        """
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # Find previous successful deployment
        previous_deployment = None
        for dep in self._deployments.values():
            if (dep.model_id == deployment.model_id and 
                dep.environment == deployment.environment and
                dep.status == DeploymentStatus.COMPLETED and
                dep.started_at < deployment.started_at):
                if not previous_deployment or dep.started_at > previous_deployment.started_at:
                    previous_deployment = dep
        
        if not previous_deployment:
            raise ValueError("No previous successful deployment found for rollback")
        
        # Create rollback deployment
        rollback_id = await self.deploy_model(
            model_id=previous_deployment.model_id,
            version=previous_deployment.model_version,
            environment=DeploymentEnvironment(previous_deployment.environment),
            config=previous_deployment.config
        )
        
        # Mark original deployment as rolled back  
        deployment.status = DeploymentStatus.ROLLED_BACK
        
        return rollback_id