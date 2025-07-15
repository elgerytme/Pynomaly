#!/usr/bin/env python3
"""
MLOps Model Deployment System for Pynomaly.
This module provides comprehensive model deployment, serving, and monitoring capabilities.
"""

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    RETIRED = "retired"

    @classmethod
    def from_string(cls, value: str):
        """Create DeploymentStatus from string."""
        # Handle enum string representations like "DeploymentStatus.PENDING"
        if "." in value:
            value = value.split(".")[-1]

        for item in cls:
            if item.value == value.lower() or item.name == value.upper():
                return item
        raise ValueError(f"Invalid deployment status: {value}")


class DeploymentEnvironment(Enum):
    """Deployment environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

    @classmethod
    def from_string(cls, value: str):
        """Create DeploymentEnvironment from string."""
        # Handle enum string representations like "DeploymentEnvironment.DEVELOPMENT"
        if "." in value:
            value = value.split(".")[-1]

        for item in cls:
            if item.value == value.lower() or item.name == value.upper():
                return item
        raise ValueError(f"Invalid deployment environment: {value}")


@dataclass
class ModelDeployment:
    """Model deployment configuration."""

    deployment_id: str
    model_id: str
    model_version: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    endpoint_url: str
    health_check_url: str
    created_at: datetime
    updated_at: datetime
    deployed_at: datetime | None
    retired_at: datetime | None
    configuration: dict[str, Any]
    resources: dict[str, Any]
    scaling_config: dict[str, Any]
    monitoring_config: dict[str, Any]
    author: str
    notes: str


@dataclass
class ModelPrediction:
    """Model prediction result."""

    prediction_id: str
    model_id: str
    deployment_id: str
    input_data: dict[str, Any]
    prediction: Any
    confidence: float | None
    timestamp: datetime
    processing_time_ms: float
    model_version: str
    metadata: dict[str, Any]


@dataclass
class ModelHealth:
    """Model health status."""

    deployment_id: str
    status: str
    last_check: datetime
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    predictions_count: int
    errors_count: int
    uptime_seconds: float
    version: str


class ModelPredictionRequest(BaseModel):
    """Model prediction request."""

    data: dict[str, Any] = Field(..., description="Input data for prediction")
    model_id: str | None = Field(None, description="Specific model ID to use")
    return_confidence: bool = Field(False, description="Return confidence score")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class ModelPredictionResponse(BaseModel):
    """Model prediction response."""

    prediction_id: str
    prediction: Any
    confidence: float | None
    model_id: str
    model_version: str
    timestamp: datetime
    processing_time_ms: float
    metadata: dict[str, Any]


class ModelHealthResponse(BaseModel):
    """Model health response."""

    deployment_id: str
    status: str
    last_check: datetime
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    predictions_count: int
    errors_count: int
    uptime_seconds: float
    version: str


class ModelServer:
    """Model server for serving ML models."""

    def __init__(self, deployment: ModelDeployment):
        """Initialize model server."""
        self.deployment = deployment
        self.model: BaseEstimator | None = None
        self.model_metadata: dict[str, Any] | None = None
        self.start_time = datetime.now()
        self.predictions_count = 0
        self.errors_count = 0
        self.predictions_queue = queue.Queue()
        self.health_metrics = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "last_prediction": None,
        }

        # Load model
        self._load_model()

        # Start background monitoring
        self._start_monitoring()

        logger.info(
            f"Model server initialized for deployment {deployment.deployment_id}"
        )

    def _load_model(self):
        """Load model from registry."""
        try:
            # Import model registry
            from .model_registry import model_registry

            # Load model
            self.model, metadata = asyncio.run(
                model_registry.get_model(self.deployment.model_id)
            )
            self.model_metadata = asdict(metadata)

            logger.info(f"✅ Model loaded: {self.deployment.model_id}")

        except Exception as e:
            logger.error(f"Failed to load model {self.deployment.model_id}: {e}")
            raise

    def _start_monitoring(self):
        """Start background monitoring thread."""

        def monitor():
            while True:
                try:
                    # Update health metrics
                    self._update_health_metrics()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _update_health_metrics(self):
        """Update health metrics."""
        try:
            import psutil

            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            self.health_metrics["cpu_usage"].append(cpu_percent)
            self.health_metrics["memory_usage"].append(memory_info.percent)

            # Keep only last 100 measurements
            if len(self.health_metrics["cpu_usage"]) > 100:
                self.health_metrics["cpu_usage"] = self.health_metrics["cpu_usage"][
                    -100:
                ]
            if len(self.health_metrics["memory_usage"]) > 100:
                self.health_metrics["memory_usage"] = self.health_metrics[
                    "memory_usage"
                ][-100:]

        except Exception as e:
            logger.warning(f"Failed to update health metrics: {e}")

    async def predict(self, request: ModelPredictionRequest) -> ModelPredictionResponse:
        """Make prediction using the model."""
        start_time = time.time()
        prediction_id = str(uuid.uuid4())

        try:
            # Prepare input data
            input_data = self._prepare_input_data(request.data)

            # Make prediction
            prediction = self.model.predict(input_data)

            # Calculate confidence if requested
            confidence = None
            if request.return_confidence and hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(input_data)
                    confidence = float(np.max(proba))
                except Exception as e:
                    logger.warning(f"Failed to calculate confidence: {e}")

            # Processing time
            processing_time = (time.time() - start_time) * 1000

            # Create prediction result
            prediction_result = ModelPrediction(
                prediction_id=prediction_id,
                model_id=self.deployment.model_id,
                deployment_id=self.deployment.deployment_id,
                input_data=request.data,
                prediction=prediction.tolist()
                if isinstance(prediction, np.ndarray)
                else prediction,
                confidence=confidence,
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                model_version=self.deployment.model_version,
                metadata=request.metadata or {},
            )

            # Update metrics
            self.predictions_count += 1
            self.health_metrics["response_times"].append(processing_time)
            self.health_metrics["last_prediction"] = datetime.now()

            # Keep only last 1000 response times
            if len(self.health_metrics["response_times"]) > 1000:
                self.health_metrics["response_times"] = self.health_metrics[
                    "response_times"
                ][-1000:]

            # Store prediction (async)
            self.predictions_queue.put(prediction_result)

            # Create response
            response = ModelPredictionResponse(
                prediction_id=prediction_id,
                prediction=prediction_result.prediction,
                confidence=confidence,
                model_id=self.deployment.model_id,
                model_version=self.deployment.model_version,
                timestamp=prediction_result.timestamp,
                processing_time_ms=processing_time,
                metadata=prediction_result.metadata,
            )

            logger.debug(f"Prediction made: {prediction_id}")
            return response

        except Exception as e:
            self.errors_count += 1
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def _prepare_input_data(self, data: dict[str, Any]) -> np.ndarray:
        """Prepare input data for model prediction."""
        try:
            # Convert to numpy array based on model requirements
            if isinstance(data, dict):
                if "features" in data:
                    # Standard format with features array
                    return np.array(data["features"]).reshape(1, -1)
                elif "data" in data:
                    # Nested data format
                    return np.array(data["data"]).reshape(1, -1)
                else:
                    # Treat dict values as features
                    features = list(data.values())
                    return np.array(features).reshape(1, -1)
            elif isinstance(data, list):
                # Direct feature array
                return np.array(data).reshape(1, -1)
            else:
                raise ValueError(f"Unsupported input data format: {type(data)}")

        except Exception as e:
            logger.error(f"Failed to prepare input data: {e}")
            raise ValueError(f"Invalid input data format: {str(e)}")

    def get_health(self) -> ModelHealthResponse:
        """Get model health status."""
        try:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            # Calculate average response time
            avg_response_time = (
                np.mean(self.health_metrics["response_times"])
                if self.health_metrics["response_times"]
                else 0
            )

            # Calculate average resource usage
            avg_memory = (
                np.mean(self.health_metrics["memory_usage"])
                if self.health_metrics["memory_usage"]
                else 0
            )
            avg_cpu = (
                np.mean(self.health_metrics["cpu_usage"])
                if self.health_metrics["cpu_usage"]
                else 0
            )

            return ModelHealthResponse(
                deployment_id=self.deployment.deployment_id,
                status="healthy"
                if self.errors_count / max(self.predictions_count, 1) < 0.1
                else "unhealthy",
                last_check=datetime.now(),
                response_time_ms=avg_response_time,
                memory_usage_mb=avg_memory,
                cpu_usage_percent=avg_cpu,
                predictions_count=self.predictions_count,
                errors_count=self.errors_count,
                uptime_seconds=uptime_seconds,
                version=self.deployment.model_version,
            )

        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            raise


class ModelDeploymentManager:
    """Manager for model deployments."""

    def __init__(self, deployments_path: str = "mlops/deployments"):
        """Initialize deployment manager."""
        self.deployments_path = Path(deployments_path)
        self.deployments_path.mkdir(parents=True, exist_ok=True)

        # Active deployments
        self.active_deployments: dict[str, ModelServer] = {}
        self.deployment_apps: dict[str, FastAPI] = {}

        # Deployment index
        self.deployment_index_path = self.deployments_path / "deployment_index.json"
        self.deployment_index = self._load_deployment_index()

        logger.info(f"Deployment manager initialized at {self.deployments_path}")

    def _load_deployment_index(self) -> dict[str, Any]:
        """Load deployment index from file."""
        if self.deployment_index_path.exists():
            try:
                with open(self.deployment_index_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load deployment index: {e}")

        return {
            "deployments": {},
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    def _save_deployment_index(self):
        """Save deployment index to file."""
        try:
            self.deployment_index["updated_at"] = datetime.now().isoformat()
            with open(self.deployment_index_path, "w") as f:
                json.dump(self.deployment_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save deployment index: {e}")

    async def create_deployment(
        self,
        model_id: str,
        model_version: str,
        environment: DeploymentEnvironment,
        configuration: dict[str, Any] = None,
        resources: dict[str, Any] = None,
        scaling_config: dict[str, Any] = None,
        author: str = "system",
        notes: str = "",
    ) -> str:
        """Create a new model deployment."""
        deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"

        try:
            # Create deployment configuration
            deployment = ModelDeployment(
                deployment_id=deployment_id,
                model_id=model_id,
                model_version=model_version,
                environment=environment,
                status=DeploymentStatus.PENDING,
                endpoint_url=f"http://localhost:8000/models/{deployment_id}/predict",
                health_check_url=f"http://localhost:8000/models/{deployment_id}/health",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                deployed_at=None,
                retired_at=None,
                configuration=configuration or {},
                resources=resources or {"cpu": 1, "memory": "1Gi"},
                scaling_config=scaling_config or {"min_replicas": 1, "max_replicas": 3},
                monitoring_config={"enabled": True, "metrics_interval": 30},
                author=author,
                notes=notes,
            )

            # Save deployment
            deployment_path = self.deployments_path / f"{deployment_id}.json"
            with open(deployment_path, "w") as f:
                json.dump(asdict(deployment), f, indent=2, default=str)

            # Update index
            self.deployment_index["deployments"][deployment_id] = {
                "model_id": model_id,
                "model_version": model_version,
                "environment": environment.value,
                "status": DeploymentStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "author": author,
            }
            self._save_deployment_index()

            logger.info(f"✅ Deployment created: {deployment_id}")
            return deployment_id

        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            raise

    async def deploy_model(self, deployment_id: str) -> bool:
        """Deploy model to serve predictions."""
        try:
            # Load deployment configuration
            deployment = await self._load_deployment(deployment_id)
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            # Update status
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.updated_at = datetime.now()
            await self._save_deployment(deployment)

            # Create model server
            model_server = ModelServer(deployment)

            # Create FastAPI app for this deployment
            app = FastAPI(
                title=f"Model Service - {deployment.model_id}",
                description=f"Model serving endpoint for {deployment.model_id}",
                version=deployment.model_version,
            )

            # Add prediction endpoint
            @app.post("/predict", response_model=ModelPredictionResponse)
            async def predict(request: ModelPredictionRequest):
                return await model_server.predict(request)

            # Add health endpoint
            @app.get("/health", response_model=ModelHealthResponse)
            async def health():
                return model_server.get_health()

            # Add metrics endpoint
            @app.get("/metrics")
            async def metrics():
                health = model_server.get_health()
                return {
                    "deployment_id": deployment_id,
                    "model_id": deployment.model_id,
                    "model_version": deployment.model_version,
                    "predictions_count": health.predictions_count,
                    "errors_count": health.errors_count,
                    "avg_response_time_ms": health.response_time_ms,
                    "uptime_seconds": health.uptime_seconds,
                    "status": health.status,
                    "timestamp": datetime.now().isoformat(),
                }

            # Store active deployment
            self.active_deployments[deployment_id] = model_server
            self.deployment_apps[deployment_id] = app

            # Update deployment status
            deployment.status = DeploymentStatus.ACTIVE
            deployment.deployed_at = datetime.now()
            deployment.updated_at = datetime.now()
            await self._save_deployment(deployment)

            # Update index
            self.deployment_index["deployments"][deployment_id]["status"] = (
                DeploymentStatus.ACTIVE.value
            )
            self._save_deployment_index()

            logger.info(f"✅ Model deployed successfully: {deployment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")

            # Update status to failed
            try:
                deployment = await self._load_deployment(deployment_id)
                if deployment:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.updated_at = datetime.now()
                    await self._save_deployment(deployment)
            except:
                pass

            return False

    async def undeploy_model(self, deployment_id: str) -> bool:
        """Undeploy model and stop serving."""
        try:
            # Remove from active deployments
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]

            if deployment_id in self.deployment_apps:
                del self.deployment_apps[deployment_id]

            # Update deployment status
            deployment = await self._load_deployment(deployment_id)
            if deployment:
                deployment.status = DeploymentStatus.INACTIVE
                deployment.updated_at = datetime.now()
                await self._save_deployment(deployment)

                # Update index
                self.deployment_index["deployments"][deployment_id]["status"] = (
                    DeploymentStatus.INACTIVE.value
                )
                self._save_deployment_index()

            logger.info(f"✅ Model undeployed: {deployment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to undeploy model: {e}")
            return False

    async def get_deployment(self, deployment_id: str) -> ModelDeployment | None:
        """Get deployment by ID."""
        return await self._load_deployment(deployment_id)

    async def list_deployments(
        self,
        environment: DeploymentEnvironment | None = None,
        status: DeploymentStatus | None = None,
        model_id: str | None = None,
    ) -> list[ModelDeployment]:
        """List deployments with optional filtering."""
        deployments = []

        try:
            for deployment_id in self.deployment_index["deployments"]:
                deployment = await self._load_deployment(deployment_id)
                if not deployment:
                    continue

                # Apply filters
                if environment and deployment.environment != environment:
                    continue
                if status and deployment.status != status:
                    continue
                if model_id and deployment.model_id != model_id:
                    continue

                deployments.append(deployment)

            # Sort by creation date (newest first)
            deployments.sort(key=lambda d: d.created_at, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")

        return deployments

    async def get_deployment_health(
        self, deployment_id: str
    ) -> ModelHealthResponse | None:
        """Get deployment health status."""
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id].get_health()
        return None

    async def _load_deployment(self, deployment_id: str) -> ModelDeployment | None:
        """Load deployment from file."""
        try:
            deployment_path = self.deployments_path / f"{deployment_id}.json"
            if not deployment_path.exists():
                return None

            with open(deployment_path) as f:
                deployment_data = json.load(f)

            # Convert datetime strings and enums
            deployment_data["environment"] = (
                DeploymentEnvironment.from_string(deployment_data["environment"])
                if isinstance(deployment_data["environment"], str)
                else deployment_data["environment"]
            )
            deployment_data["status"] = (
                DeploymentStatus.from_string(deployment_data["status"])
                if isinstance(deployment_data["status"], str)
                else deployment_data["status"]
            )
            deployment_data["created_at"] = datetime.fromisoformat(
                deployment_data["created_at"]
            )
            deployment_data["updated_at"] = datetime.fromisoformat(
                deployment_data["updated_at"]
            )

            if deployment_data["deployed_at"]:
                deployment_data["deployed_at"] = datetime.fromisoformat(
                    deployment_data["deployed_at"]
                )
            if deployment_data["retired_at"]:
                deployment_data["retired_at"] = datetime.fromisoformat(
                    deployment_data["retired_at"]
                )

            return ModelDeployment(**deployment_data)

        except Exception as e:
            logger.error(f"Failed to load deployment {deployment_id}: {e}")
            return None

    async def _save_deployment(self, deployment: ModelDeployment):
        """Save deployment to file."""
        try:
            deployment_path = self.deployments_path / f"{deployment.deployment_id}.json"
            with open(deployment_path, "w") as f:
                json.dump(asdict(deployment), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save deployment: {e}")

    def get_deployment_app(self, deployment_id: str) -> FastAPI | None:
        """Get FastAPI app for deployment."""
        return self.deployment_apps.get(deployment_id)

    async def get_deployment_metrics(self, deployment_id: str) -> dict[str, Any] | None:
        """Get deployment metrics."""
        if deployment_id in self.active_deployments:
            server = self.active_deployments[deployment_id]
            health = server.get_health()

            return {
                "deployment_id": deployment_id,
                "model_id": server.deployment.model_id,
                "model_version": server.deployment.model_version,
                "predictions_count": health.predictions_count,
                "errors_count": health.errors_count,
                "error_rate": health.errors_count / max(health.predictions_count, 1),
                "avg_response_time_ms": health.response_time_ms,
                "uptime_seconds": health.uptime_seconds,
                "cpu_usage_percent": health.cpu_usage_percent,
                "memory_usage_mb": health.memory_usage_mb,
                "status": health.status,
                "last_prediction": server.health_metrics.get("last_prediction"),
                "timestamp": datetime.now().isoformat(),
            }
        return None


# Global deployment manager instance
deployment_manager = ModelDeploymentManager()

# Make deployment manager available for import
__all__ = [
    "ModelDeploymentManager",
    "ModelDeployment",
    "ModelServer",
    "DeploymentStatus",
    "DeploymentEnvironment",
    "deployment_manager",
]
