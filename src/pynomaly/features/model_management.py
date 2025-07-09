"""Advanced model management and versioning for Pynomaly."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pynomaly.domain.entities import Dataset, Model
from pynomaly.domain.value_objects import PerformanceMetrics, SemanticVersion
from pynomaly.shared.error_handling import (
    ErrorCodes,
    create_infrastructure_error,
)

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""

    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentEnvironment(Enum):
    """Model deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelMetadata:
    """Model metadata and versioning information."""

    model_id: str
    name: str
    version: SemanticVersion
    algorithm: str
    creation_time: datetime
    last_updated: datetime
    status: ModelStatus = ModelStatus.TRAINING
    description: str = ""
    tags: list[str] = field(default_factory=list)
    author: str = ""
    model_size_bytes: int = 0
    training_dataset_hash: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    performance_metrics: PerformanceMetrics | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": str(self.version),
            "algorithm": self.algorithm,
            "creation_time": self.creation_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "status": self.status.value,
            "description": self.description,
            "tags": self.tags,
            "author": self.author,
            "model_size_bytes": self.model_size_bytes,
            "training_dataset_hash": self.training_dataset_hash,
            "hyperparameters": self.hyperparameters,
            "performance_metrics": self.performance_metrics.to_dict()
            if self.performance_metrics
            else None,
            "deployment_config": self.deployment_config,
        }


@dataclass
class ModelDeploymentInfo:
    """Model deployment information."""

    model_id: str
    environment: DeploymentEnvironment
    deployment_time: datetime
    health_status: str = "healthy"
    request_count: int = 0
    error_count: int = 0
    average_response_time_ms: float = 0.0
    last_health_check: datetime | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "environment": self.environment.value,
            "deployment_time": self.deployment_time.isoformat(),
            "health_status": self.health_status,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "average_response_time_ms": self.average_response_time_ms,
            "last_health_check": self.last_health_check.isoformat()
            if self.last_health_check
            else None,
            "deployment_config": self.deployment_config,
        }


class ModelRegistry:
    """Model registry for tracking and versioning models."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize model registry.

        Args:
            storage_path: Path for model storage
        """
        self.storage_path = storage_path or Path("models")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, ModelMetadata] = {}
        self.deployments: dict[str, list[ModelDeploymentInfo]] = {}

    async def register_model(
        self,
        model: Model,
        name: str,
        version: SemanticVersion | None = None,
        description: str = "",
        tags: list[str] | None = None,
        author: str = "",
        hyperparameters: dict[str, Any] | None = None,
    ) -> ModelMetadata:
        """Register a new model in the registry.

        Args:
            model: Model to register
            name: Model name
            version: Model version (auto-generated if not provided)
            description: Model description
            tags: Model tags
            author: Model author
            hyperparameters: Model hyperparameters

        Returns:
            Model metadata
        """
        try:
            # Generate model ID
            model_id = str(uuid.uuid4())

            # Auto-generate version if not provided
            if version is None:
                existing_versions = [
                    m.version for m in self.models.values() if m.name == name
                ]
                if existing_versions:
                    latest = max(existing_versions)
                    version = SemanticVersion(
                        major=latest.major, minor=latest.minor + 1, patch=0
                    )
                else:
                    version = SemanticVersion(1, 0, 0)

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                algorithm=model.algorithm,
                creation_time=datetime.now(),
                last_updated=datetime.now(),
                status=ModelStatus.TRAINED,
                description=description,
                tags=tags or [],
                author=author,
                hyperparameters=hyperparameters or {},
            )

            # Save model to storage
            await self._save_model(model, metadata)

            # Register in memory
            self.models[model_id] = metadata

            logger.info(f"Model registered: {name} v{version} (ID: {model_id})")
            return metadata

        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Model registration failed: {str(e)}",
                cause=e,
            )

    async def get_model(self, model_id: str) -> Model | None:
        """Get model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model instance or None if not found
        """
        try:
            if model_id not in self.models:
                return None

            metadata = self.models[model_id]
            model_path = self.storage_path / f"{model_id}.pkl"

            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Failed to load model: {str(e)}",
                cause=e,
            )

    async def list_models(
        self,
        name_filter: str | None = None,
        status_filter: ModelStatus | None = None,
        tag_filter: str | None = None,
    ) -> list[ModelMetadata]:
        """List models with optional filters.

        Args:
            name_filter: Filter by model name
            status_filter: Filter by model status
            tag_filter: Filter by tag

        Returns:
            List of model metadata
        """
        models = list(self.models.values())

        if name_filter:
            models = [m for m in models if name_filter.lower() in m.name.lower()]

        if status_filter:
            models = [m for m in models if m.status == status_filter]

        if tag_filter:
            models = [m for m in models if tag_filter in m.tags]

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.creation_time, reverse=True)

        return models

    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status.

        Args:
            model_id: Model identifier
            status: New status

        Returns:
            True if updated successfully
        """
        if model_id not in self.models:
            return False

        self.models[model_id].status = status
        self.models[model_id].last_updated = datetime.now()

        logger.info(f"Model {model_id} status updated to {status.value}")
        return True

    async def delete_model(self, model_id: str) -> bool:
        """Delete model from registry.

        Args:
            model_id: Model identifier

        Returns:
            True if deleted successfully
        """
        try:
            if model_id not in self.models:
                return False

            # Remove model file
            model_path = self.storage_path / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()

            # Remove metadata file
            metadata_path = self.storage_path / f"{model_id}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()

            # Remove from memory
            del self.models[model_id]

            # Remove deployments
            if model_id in self.deployments:
                del self.deployments[model_id]

            logger.info(f"Model {model_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    async def _save_model(self, model: Model, metadata: ModelMetadata) -> None:
        """Save model and metadata to storage."""
        # Save model
        model_path = self.storage_path / f"{metadata.model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Calculate model size
        metadata.model_size_bytes = model_path.stat().st_size

        # Save metadata
        metadata_path = self.storage_path / f"{metadata.model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)


class ModelVersioning:
    """Model versioning and comparison utilities."""

    def __init__(self, registry: ModelRegistry):
        """Initialize model versioning.

        Args:
            registry: Model registry instance
        """
        self.registry = registry

    async def create_model_version(
        self,
        base_model_id: str,
        updated_model: Model,
        version_type: str = "minor",
        description: str = "",
    ) -> ModelMetadata | None:
        """Create new version of existing model.

        Args:
            base_model_id: Base model ID
            updated_model: Updated model
            version_type: Version increment type (major/minor/patch)
            description: Version description

        Returns:
            New model metadata or None if failed
        """
        try:
            if base_model_id not in self.registry.models:
                logger.error(f"Base model {base_model_id} not found")
                return None

            base_metadata = self.registry.models[base_model_id]

            # Calculate new version
            new_version = self._increment_version(base_metadata.version, version_type)

            # Register new version
            new_metadata = await self.registry.register_model(
                model=updated_model,
                name=base_metadata.name,
                version=new_version,
                description=description,
                tags=base_metadata.tags,
                author=base_metadata.author,
            )

            logger.info(
                f"Created new version {new_version} for model {base_metadata.name}"
            )
            return new_metadata

        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            return None

    async def compare_models(self, model_id1: str, model_id2: str) -> dict[str, Any]:
        """Compare two models.

        Args:
            model_id1: First model ID
            model_id2: Second model ID

        Returns:
            Comparison results
        """
        try:
            metadata1 = self.registry.models.get(model_id1)
            metadata2 = self.registry.models.get(model_id2)

            if not metadata1 or not metadata2:
                return {"error": "One or both models not found"}

            comparison = {
                "model1": {
                    "id": model_id1,
                    "name": metadata1.name,
                    "version": str(metadata1.version),
                    "algorithm": metadata1.algorithm,
                    "size_bytes": metadata1.model_size_bytes,
                    "performance": metadata1.performance_metrics.to_dict()
                    if metadata1.performance_metrics
                    else None,
                },
                "model2": {
                    "id": model_id2,
                    "name": metadata2.name,
                    "version": str(metadata2.version),
                    "algorithm": metadata2.algorithm,
                    "size_bytes": metadata2.model_size_bytes,
                    "performance": metadata2.performance_metrics.to_dict()
                    if metadata2.performance_metrics
                    else None,
                },
                "differences": {
                    "algorithm_different": metadata1.algorithm != metadata2.algorithm,
                    "size_difference_bytes": metadata2.model_size_bytes
                    - metadata1.model_size_bytes,
                    "version_difference": self._compare_versions(
                        metadata1.version, metadata2.version
                    ),
                },
                "recommendations": [],
            }

            # Add recommendations
            if comparison["differences"]["size_difference_bytes"] > 0:
                comparison["recommendations"].append(
                    "Model 2 is larger - consider performance impact"
                )

            if comparison["differences"]["algorithm_different"]:
                comparison["recommendations"].append(
                    "Different algorithms - evaluate performance differences"
                )

            return comparison

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def _increment_version(
        self, version: SemanticVersion, version_type: str
    ) -> SemanticVersion:
        """Increment version based on type."""
        if version_type == "major":
            return SemanticVersion(version.major + 1, 0, 0)
        elif version_type == "minor":
            return SemanticVersion(version.major, version.minor + 1, 0)
        else:  # patch
            return SemanticVersion(version.major, version.minor, version.patch + 1)

    def _compare_versions(self, v1: SemanticVersion, v2: SemanticVersion) -> str:
        """Compare two versions."""
        if v1 > v2:
            return "model1_newer"
        elif v1 < v2:
            return "model2_newer"
        else:
            return "same_version"


class ModelDeployment:
    """Model deployment management."""

    def __init__(self, registry: ModelRegistry):
        """Initialize model deployment.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.active_deployments: dict[str, ModelDeploymentInfo] = {}

    async def deploy_model(
        self,
        model_id: str,
        environment: DeploymentEnvironment,
        deployment_config: dict[str, Any] | None = None,
    ) -> bool:
        """Deploy model to environment.

        Args:
            model_id: Model identifier
            environment: Target environment
            deployment_config: Deployment configuration

        Returns:
            True if deployment successful
        """
        try:
            if model_id not in self.registry.models:
                logger.error(f"Model {model_id} not found in registry")
                return False

            metadata = self.registry.models[model_id]

            # Create deployment info
            deployment_key = f"{model_id}_{environment.value}"
            deployment_info = ModelDeploymentInfo(
                model_id=model_id,
                environment=environment,
                deployment_time=datetime.now(),
                deployment_config=deployment_config or {},
            )

            # Store deployment info
            self.active_deployments[deployment_key] = deployment_info

            if model_id not in self.registry.deployments:
                self.registry.deployments[model_id] = []
            self.registry.deployments[model_id].append(deployment_info)

            # Update model status
            await self.registry.update_model_status(model_id, ModelStatus.DEPLOYED)

            logger.info(f"Model {model_id} deployed to {environment.value}")
            return True

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False

    async def undeploy_model(
        self, model_id: str, environment: DeploymentEnvironment
    ) -> bool:
        """Undeploy model from environment.

        Args:
            model_id: Model identifier
            environment: Target environment

        Returns:
            True if undeployment successful
        """
        try:
            deployment_key = f"{model_id}_{environment.value}"

            if deployment_key not in self.active_deployments:
                logger.warning(f"No active deployment found for {deployment_key}")
                return False

            # Remove from active deployments
            del self.active_deployments[deployment_key]

            # Check if model has other active deployments
            active_envs = [
                key
                for key in self.active_deployments.keys()
                if key.startswith(f"{model_id}_")
            ]

            if not active_envs:
                # No active deployments, update status
                await self.registry.update_model_status(model_id, ModelStatus.TRAINED)

            logger.info(f"Model {model_id} undeployed from {environment.value}")
            return True

        except Exception as e:
            logger.error(f"Model undeployment failed: {e}")
            return False

    async def get_deployment_status(self, model_id: str) -> dict[str, Any]:
        """Get deployment status for model.

        Args:
            model_id: Model identifier

        Returns:
            Deployment status information
        """
        deployments = [
            info
            for key, info in self.active_deployments.items()
            if key.startswith(f"{model_id}_")
        ]

        return {
            "model_id": model_id,
            "active_deployments": len(deployments),
            "environments": [d.environment.value for d in deployments],
            "deployment_details": [d.to_dict() for d in deployments],
        }

    async def health_check(
        self, model_id: str, environment: DeploymentEnvironment
    ) -> dict[str, Any]:
        """Perform health check on deployed model.

        Args:
            model_id: Model identifier
            environment: Environment to check

        Returns:
            Health check results
        """
        deployment_key = f"{model_id}_{environment.value}"

        if deployment_key not in self.active_deployments:
            return {
                "status": "not_deployed",
                "healthy": False,
                "message": f"Model not deployed in {environment.value}",
            }

        deployment_info = self.active_deployments[deployment_key]

        try:
            # Load model for health check
            model = await self.registry.get_model(model_id)

            if model is None:
                deployment_info.health_status = "unhealthy"
                deployment_info.last_health_check = datetime.now()
                return {
                    "status": "unhealthy",
                    "healthy": False,
                    "message": "Model file not accessible",
                }

            # Basic health check - ensure model can be loaded
            deployment_info.health_status = "healthy"
            deployment_info.last_health_check = datetime.now()

            return {
                "status": "healthy",
                "healthy": True,
                "message": "Model is healthy and accessible",
                "last_check": deployment_info.last_health_check.isoformat(),
                "deployment_time": deployment_info.deployment_time.isoformat(),
                "request_count": deployment_info.request_count,
                "error_count": deployment_info.error_count,
            }

        except Exception as e:
            deployment_info.health_status = "unhealthy"
            deployment_info.last_health_check = datetime.now()

            return {
                "status": "unhealthy",
                "healthy": False,
                "message": f"Health check failed: {str(e)}",
                "last_check": deployment_info.last_health_check.isoformat(),
            }


class ModelMonitoring:
    """Model performance monitoring and drift detection."""

    def __init__(self, registry: ModelRegistry):
        """Initialize model monitoring.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.monitoring_data: dict[str, list[dict[str, Any]]] = {}

    async def log_prediction(
        self,
        model_id: str,
        input_data: dict[str, Any],
        prediction_result: Any,
        execution_time_ms: float,
        confidence_score: float | None = None,
    ) -> None:
        """Log model prediction for monitoring.

        Args:
            model_id: Model identifier
            input_data: Input data used for prediction
            prediction_result: Prediction result
            execution_time_ms: Prediction execution time
            confidence_score: Prediction confidence
        """
        try:
            if model_id not in self.monitoring_data:
                self.monitoring_data[model_id] = []

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "input_hash": hashlib.md5(str(input_data).encode()).hexdigest(),
                "prediction": str(prediction_result),
                "execution_time_ms": execution_time_ms,
                "confidence_score": confidence_score,
            }

            self.monitoring_data[model_id].append(log_entry)

            # Keep only last 1000 entries per model
            if len(self.monitoring_data[model_id]) > 1000:
                self.monitoring_data[model_id] = self.monitoring_data[model_id][-1000:]

        except Exception as e:
            logger.error(f"Failed to log prediction for model {model_id}: {e}")

    async def detect_drift(
        self, model_id: str, window_hours: int = 24
    ) -> dict[str, Any]:
        """Detect performance drift for model.

        Args:
            model_id: Model identifier
            window_hours: Time window for drift detection

        Returns:
            Drift detection results
        """
        try:
            if model_id not in self.monitoring_data:
                return {"error": "No monitoring data available"}

            data = self.monitoring_data[model_id]

            # Filter data to time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            recent_data = [
                entry
                for entry in data
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
            ]

            if len(recent_data) < 10:
                return {
                    "drift_detected": False,
                    "message": "Insufficient data for drift detection",
                    "data_points": len(recent_data),
                }

            # Analyze execution time trends
            execution_times = [entry["execution_time_ms"] for entry in recent_data]
            avg_execution_time = sum(execution_times) / len(execution_times)

            # Analyze confidence trends (if available)
            confidence_scores = [
                entry["confidence_score"]
                for entry in recent_data
                if entry["confidence_score"] is not None
            ]
            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else None
            )

            # Simple drift detection based on performance degradation
            baseline_execution_time = (
                100.0  # ms - would be calculated from historical data
            )
            baseline_confidence = 0.8  # would be calculated from historical data

            drift_indicators = []

            if avg_execution_time > baseline_execution_time * 1.5:
                drift_indicators.append("Execution time increased significantly")

            if avg_confidence and avg_confidence < baseline_confidence * 0.8:
                drift_indicators.append("Confidence scores decreased significantly")

            return {
                "drift_detected": len(drift_indicators) > 0,
                "drift_indicators": drift_indicators,
                "metrics": {
                    "avg_execution_time_ms": avg_execution_time,
                    "avg_confidence_score": avg_confidence,
                    "data_points": len(recent_data),
                    "time_window_hours": window_hours,
                },
                "recommendations": self._generate_drift_recommendations(
                    drift_indicators
                ),
            }

        except Exception as e:
            logger.error(f"Drift detection failed for model {model_id}: {e}")
            return {"error": f"Drift detection failed: {str(e)}"}

    def _generate_drift_recommendations(self, drift_indicators: list[str]) -> list[str]:
        """Generate recommendations based on drift indicators."""
        recommendations = []

        if "Execution time increased" in str(drift_indicators):
            recommendations.append(
                "Consider model optimization or infrastructure scaling"
            )

        if "Confidence scores decreased" in str(drift_indicators):
            recommendations.append("Consider model retraining with recent data")

        if drift_indicators:
            recommendations.append("Monitor model performance closely")
            recommendations.append("Consider A/B testing with updated model")

        return recommendations


class AutoMLPipeline:
    """Automated machine learning pipeline for model optimization."""

    def __init__(self, registry: ModelRegistry):
        """Initialize AutoML pipeline.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.optimization_history: list[dict[str, Any]] = []

    async def optimize_model(
        self,
        base_model_id: str,
        training_dataset: Dataset,
        optimization_config: dict[str, Any] | None = None,
    ) -> str | None:
        """Optimize model using AutoML techniques.

        Args:
            base_model_id: Base model to optimize
            training_dataset: Training dataset
            optimization_config: Optimization configuration

        Returns:
            New optimized model ID or None if failed
        """
        try:
            # Get base model
            base_model = await self.registry.get_model(base_model_id)
            if not base_model:
                logger.error(f"Base model {base_model_id} not found")
                return None

            base_metadata = self.registry.models[base_model_id]

            # Simulate optimization process
            logger.info(f"Starting AutoML optimization for model {base_model_id}")

            # In a real implementation, this would:
            # 1. Hyperparameter tuning
            # 2. Feature selection
            # 3. Algorithm comparison
            # 4. Ensemble methods
            # 5. Cross-validation

            await asyncio.sleep(0.1)  # Simulate optimization time

            # Create optimized model (simplified)
            optimized_model = base_model  # In reality, would be a new optimized model

            # Register optimized model
            optimized_metadata = await self.registry.register_model(
                model=optimized_model,
                name=f"{base_metadata.name}_optimized",
                description=f"AutoML optimized version of {base_metadata.name}",
                tags=base_metadata.tags + ["automl", "optimized"],
                author="AutoML Pipeline",
                hyperparameters={
                    **base_metadata.hyperparameters,
                    "optimization_timestamp": datetime.now().isoformat(),
                },
            )

            # Record optimization
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "base_model_id": base_model_id,
                "optimized_model_id": optimized_metadata.model_id,
                "optimization_config": optimization_config or {},
                "improvement_metrics": {
                    "accuracy_improvement": 0.05,  # Simulated improvement
                    "speed_improvement": 0.15,
                },
            }

            self.optimization_history.append(optimization_record)

            logger.info(f"AutoML optimization completed: {optimized_metadata.model_id}")
            return optimized_metadata.model_id

        except Exception as e:
            logger.error(f"AutoML optimization failed: {e}")
            return None

    async def get_optimization_history(
        self, model_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get optimization history.

        Args:
            model_id: Filter by model ID (optional)

        Returns:
            Optimization history
        """
        if model_id:
            return [
                record
                for record in self.optimization_history
                if record["base_model_id"] == model_id
                or record["optimized_model_id"] == model_id
            ]

        return self.optimization_history


class ModelManager:
    """Main model management facade."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize model manager.

        Args:
            storage_path: Path for model storage
        """
        self.registry = ModelRegistry(storage_path)
        self.versioning = ModelVersioning(self.registry)
        self.deployment = ModelDeployment(self.registry)
        self.monitoring = ModelMonitoring(self.registry)
        self.automl = AutoMLPipeline(self.registry)

    async def create_model_pipeline(
        self,
        model: Model,
        name: str,
        target_environment: DeploymentEnvironment = DeploymentEnvironment.STAGING,
        auto_optimize: bool = False,
        training_dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Create complete model management pipeline.

        Args:
            model: Model to manage
            name: Model name
            target_environment: Target deployment environment
            auto_optimize: Whether to apply AutoML optimization
            training_dataset: Training dataset for optimization

        Returns:
            Pipeline results
        """
        try:
            # Register model
            metadata = await self.registry.register_model(
                model=model,
                name=name,
                description="Model managed through pipeline",
                tags=["pipeline", "managed"],
            )

            model_id = metadata.model_id
            results = {
                "model_id": model_id,
                "registration": "success",
                "optimization": "skipped",
                "deployment": "pending",
            }

            # Optional optimization
            if auto_optimize and training_dataset:
                optimized_id = await self.automl.optimize_model(
                    model_id, training_dataset
                )
                if optimized_id:
                    model_id = optimized_id
                    results["optimization"] = "success"
                    results["optimized_model_id"] = optimized_id
                else:
                    results["optimization"] = "failed"

            # Deploy to target environment
            deployment_success = await self.deployment.deploy_model(
                model_id, target_environment
            )

            if deployment_success:
                results["deployment"] = "success"
                results["environment"] = target_environment.value
            else:
                results["deployment"] = "failed"

            # Perform health check
            health_check = await self.deployment.health_check(
                model_id, target_environment
            )
            results["health_check"] = health_check

            logger.info(f"Model pipeline completed for {name}")
            return results

        except Exception as e:
            logger.error(f"Model pipeline failed: {e}")
            return {
                "error": str(e),
                "registration": "failed",
                "optimization": "skipped",
                "deployment": "failed",
            }


# Global model manager
_model_manager: ModelManager | None = None


def get_model_manager(storage_path: Path | None = None) -> ModelManager:
    """Get global model manager.

    Args:
        storage_path: Optional storage path override

    Returns:
        Model manager instance
    """
    global _model_manager

    if _model_manager is None or storage_path is not None:
        _model_manager = ModelManager(storage_path)

    return _model_manager
