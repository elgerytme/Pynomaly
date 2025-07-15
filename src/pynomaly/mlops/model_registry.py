#!/usr/bin/env python3
"""
MLOps Model Registry for Pynomaly.
This module provides comprehensive model management, versioning, and lifecycle management.
"""

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

    @classmethod
    def from_string(cls, value: str):
        """Create ModelStatus from string."""
        # Handle enum string representations like "ModelStatus.DEVELOPMENT"
        if "." in value:
            value = value.split(".")[-1]

        for item in cls:
            if item.value == value.lower() or item.name == value.upper():
                return item
        raise ValueError(f"Invalid model status: {value}")


class ModelType(Enum):
    """Model type enumeration."""

    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, value: str):
        """Create ModelType from string."""
        # Handle enum string representations like "ModelType.ISOLATION_FOREST"
        if "." in value:
            value = value.split(".")[-1]

        for item in cls:
            if item.value == value.lower() or item.name == value.upper():
                return item
        raise ValueError(f"Invalid model type: {value}")


@dataclass
class ModelMetadata:
    """Model metadata structure."""

    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    tags: list[str]
    performance_metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    training_data_info: dict[str, Any]
    artifact_path: str
    checksum: str
    size_bytes: int
    dependencies: list[str]
    environment_info: dict[str, str]


@dataclass
class ModelExperiment:
    """Model experiment structure."""

    experiment_id: str
    name: str
    description: str
    created_at: datetime
    models: list[str]  # List of model IDs
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    status: str
    author: str


@dataclass
class ModelDeployment:
    """Model deployment structure."""

    deployment_id: str
    model_id: str
    environment: str
    status: str
    deployed_at: datetime
    endpoint_url: str
    health_check_url: str
    metrics: dict[str, Any]
    configuration: dict[str, Any]


class ModelRegistry:
    """MLOps Model Registry for managing ML models."""

    def __init__(self, registry_path: str = "mlops/model_registry"):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "models"
        self.experiments_path = self.registry_path / "experiments"
        self.deployments_path = self.registry_path / "deployments"
        self.metadata_path = self.registry_path / "metadata"

        # Create directory structure
        for path in [
            self.models_path,
            self.experiments_path,
            self.deployments_path,
            self.metadata_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize registry index
        self.registry_index_path = self.registry_path / "registry_index.json"
        self.registry_index = self._load_registry_index()

        logger.info(f"Model registry initialized at {self.registry_path}")

    def _load_registry_index(self) -> dict[str, Any]:
        """Load registry index from file."""
        if self.registry_index_path.exists():
            try:
                with open(self.registry_index_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry index: {e}")

        return {
            "models": {},
            "experiments": {},
            "deployments": {},
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    def _save_registry_index(self):
        """Save registry index to file."""
        try:
            self.registry_index["updated_at"] = datetime.now().isoformat()
            with open(self.registry_index_path, "w") as f:
                json.dump(self.registry_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry index: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_model_size(self, artifact_path: Path) -> int:
        """Get size of model artifact in bytes."""
        if artifact_path.is_file():
            return artifact_path.stat().st_size
        elif artifact_path.is_dir():
            return sum(
                f.stat().st_size for f in artifact_path.rglob("*") if f.is_file()
            )
        return 0

    async def register_model(
        self,
        model: BaseEstimator,
        name: str,
        version: str,
        model_type: ModelType,
        author: str,
        description: str = "",
        tags: list[str] = None,
        performance_metrics: dict[str, float] = None,
        hyperparameters: dict[str, Any] = None,
        training_data_info: dict[str, Any] = None,
        dependencies: list[str] = None,
    ) -> str:
        """Register a new model in the registry."""
        logger.info(f"Registering model: {name} v{version}")

        try:
            # Generate unique model ID
            model_id = f"{name}_{version}_{uuid.uuid4().hex[:8]}"

            # Create model directory
            model_dir = self.models_path / model_id
            model_dir.mkdir(exist_ok=True)

            # Save model artifact
            artifact_path = model_dir / "model.pkl"
            joblib.dump(model, artifact_path)

            # Calculate checksum and size
            checksum = self._calculate_checksum(artifact_path)
            size_bytes = self._get_model_size(artifact_path)

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                model_type=model_type,
                status=ModelStatus.DEVELOPMENT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                author=author,
                description=description,
                tags=tags or [],
                performance_metrics=performance_metrics or {},
                hyperparameters=hyperparameters or {},
                training_data_info=training_data_info or {},
                artifact_path=str(artifact_path),
                checksum=checksum,
                size_bytes=size_bytes,
                dependencies=dependencies or [],
                environment_info=self._get_environment_info(),
            )

            # Save metadata
            metadata_path = self.metadata_path / f"{model_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            # Update registry index
            self.registry_index["models"][model_id] = {
                "name": name,
                "version": version,
                "status": ModelStatus.DEVELOPMENT.value,
                "created_at": datetime.now().isoformat(),
                "author": author,
                "metadata_path": str(metadata_path),
            }
            self._save_registry_index()

            logger.info(f"✅ Model registered successfully: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    async def get_model(self, model_id: str) -> tuple[BaseEstimator, ModelMetadata]:
        """Get model and metadata by ID."""
        logger.info(f"Loading model: {model_id}")

        try:
            # Load metadata
            metadata_path = self.metadata_path / f"{model_id}.json"
            if not metadata_path.exists():
                raise ValueError(f"Model not found: {model_id}")

            with open(metadata_path) as f:
                metadata_dict = json.load(f)

            # Convert to ModelMetadata object
            metadata_dict["model_type"] = (
                ModelType.from_string(metadata_dict["model_type"])
                if isinstance(metadata_dict["model_type"], str)
                else metadata_dict["model_type"]
            )
            metadata_dict["status"] = (
                ModelStatus.from_string(metadata_dict["status"])
                if isinstance(metadata_dict["status"], str)
                else metadata_dict["status"]
            )
            metadata_dict["created_at"] = datetime.fromisoformat(
                metadata_dict["created_at"]
            )
            metadata_dict["updated_at"] = datetime.fromisoformat(
                metadata_dict["updated_at"]
            )

            metadata = ModelMetadata(**metadata_dict)

            # Load model artifact
            artifact_path = Path(metadata.artifact_path)
            if not artifact_path.exists():
                raise ValueError(f"Model artifact not found: {artifact_path}")

            model = joblib.load(artifact_path)

            # Verify checksum
            current_checksum = self._calculate_checksum(artifact_path)
            if current_checksum != metadata.checksum:
                logger.warning(f"Model checksum mismatch for {model_id}")

            logger.info(f"✅ Model loaded successfully: {model_id}")
            return model, metadata

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def list_models(
        self,
        status: ModelStatus | None = None,
        model_type: ModelType | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ModelMetadata]:
        """List models with optional filtering."""
        logger.info("Listing models")

        models = []

        try:
            for model_id in self.registry_index["models"]:
                try:
                    _, metadata = await self.get_model(model_id)

                    # Apply filters
                    if status and metadata.status != status:
                        continue
                    if model_type and metadata.model_type != model_type:
                        continue
                    if author and metadata.author != author:
                        continue
                    if tags and not any(tag in metadata.tags for tag in tags):
                        continue

                    models.append(metadata)

                except Exception as e:
                    logger.warning(f"Failed to load model {model_id}: {e}")
                    continue

            # Sort by creation date (newest first)
            models.sort(key=lambda m: m.created_at, reverse=True)

            logger.info(f"✅ Found {len(models)} models")
            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status."""
        logger.info(f"Updating model status: {model_id} -> {status.value}")

        try:
            # Load current metadata
            _, metadata = await self.get_model(model_id)

            # Update status
            metadata.status = status
            metadata.updated_at = datetime.now()

            # Save updated metadata
            metadata_path = self.metadata_path / f"{model_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2, default=str)

            # Update registry index
            self.registry_index["models"][model_id]["status"] = status.value
            self._save_registry_index()

            logger.info(f"✅ Model status updated: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update model status: {e}")
            return False

    async def delete_model(self, model_id: str) -> bool:
        """Delete model from registry."""
        logger.info(f"Deleting model: {model_id}")

        try:
            # Remove model directory
            model_dir = self.models_path / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)

            # Remove metadata
            metadata_path = self.metadata_path / f"{model_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()

            # Update registry index
            if model_id in self.registry_index["models"]:
                del self.registry_index["models"][model_id]
                self._save_registry_index()

            logger.info(f"✅ Model deleted: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    async def create_experiment(
        self,
        name: str,
        description: str,
        author: str,
        parameters: dict[str, Any] = None,
    ) -> str:
        """Create a new experiment."""
        logger.info(f"Creating experiment: {name}")

        try:
            # Generate unique experiment ID
            experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

            # Create experiment
            experiment = ModelExperiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                created_at=datetime.now(),
                models=[],
                metrics={},
                parameters=parameters or {},
                status="active",
                author=author,
            )

            # Save experiment
            experiment_path = self.experiments_path / f"{experiment_id}.json"
            with open(experiment_path, "w") as f:
                json.dump(asdict(experiment), f, indent=2, default=str)

            # Update registry index
            self.registry_index["experiments"][experiment_id] = {
                "name": name,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "author": author,
            }
            self._save_registry_index()

            logger.info(f"✅ Experiment created: {experiment_id}")
            return experiment_id

        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise

    async def add_model_to_experiment(self, experiment_id: str, model_id: str) -> bool:
        """Add model to experiment."""
        logger.info(f"Adding model {model_id} to experiment {experiment_id}")

        try:
            # Load experiment
            experiment_path = self.experiments_path / f"{experiment_id}.json"
            if not experiment_path.exists():
                raise ValueError(f"Experiment not found: {experiment_id}")

            with open(experiment_path) as f:
                experiment_dict = json.load(f)

            # Add model to experiment
            if model_id not in experiment_dict["models"]:
                experiment_dict["models"].append(model_id)
                experiment_dict["updated_at"] = datetime.now().isoformat()

            # Save experiment
            with open(experiment_path, "w") as f:
                json.dump(experiment_dict, f, indent=2, default=str)

            logger.info(f"✅ Model added to experiment: {model_id} -> {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add model to experiment: {e}")
            return False

    async def compare_models(self, model_ids: list[str]) -> dict[str, Any]:
        """Compare multiple models."""
        logger.info(f"Comparing models: {model_ids}")

        try:
            comparison_results = {
                "models": [],
                "comparison_metrics": {},
                "best_model": None,
                "timestamp": datetime.now().isoformat(),
            }

            models_data = []

            # Load all models
            for model_id in model_ids:
                try:
                    _, metadata = await self.get_model(model_id)
                    models_data.append(metadata)

                    comparison_results["models"].append(
                        {
                            "model_id": model_id,
                            "name": metadata.name,
                            "version": metadata.version,
                            "type": metadata.model_type.value,
                            "status": metadata.status.value,
                            "performance_metrics": metadata.performance_metrics,
                            "created_at": metadata.created_at.isoformat(),
                            "author": metadata.author,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to load model {model_id}: {e}")
                    continue

            # Compare performance metrics
            if models_data:
                metric_names = set()
                for model in models_data:
                    metric_names.update(model.performance_metrics.keys())

                for metric_name in metric_names:
                    metric_values = []
                    for model in models_data:
                        if metric_name in model.performance_metrics:
                            metric_values.append(
                                {
                                    "model_id": model.model_id,
                                    "value": model.performance_metrics[metric_name],
                                }
                            )

                    if metric_values:
                        # Find best model for this metric
                        best_value = max(metric_values, key=lambda x: x["value"])
                        comparison_results["comparison_metrics"][metric_name] = {
                            "values": metric_values,
                            "best_model": best_value["model_id"],
                            "best_value": best_value["value"],
                        }

                # Determine overall best model (based on first metric)
                if comparison_results["comparison_metrics"]:
                    first_metric = list(
                        comparison_results["comparison_metrics"].keys()
                    )[0]
                    comparison_results["best_model"] = comparison_results[
                        "comparison_metrics"
                    ][first_metric]["best_model"]

            logger.info(f"✅ Model comparison completed for {len(models_data)} models")
            return comparison_results

        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise

    def _get_environment_info(self) -> dict[str, str]:
        """Get environment information."""
        try:
            import platform
            import sys

            return {
                "python_version": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "hostname": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
            }
        except Exception as e:
            logger.warning(f"Failed to get environment info: {e}")
            return {}

    async def get_model_lineage(self, model_id: str) -> dict[str, Any]:
        """Get model lineage and dependencies."""
        logger.info(f"Getting model lineage: {model_id}")

        try:
            # Load model metadata
            _, metadata = await self.get_model(model_id)

            # Build lineage information
            lineage = {
                "model_id": model_id,
                "model_info": {
                    "name": metadata.name,
                    "version": metadata.version,
                    "type": metadata.model_type.value,
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "author": metadata.author,
                },
                "dependencies": metadata.dependencies,
                "training_data": metadata.training_data_info,
                "environment": metadata.environment_info,
                "parent_models": [],  # Would be populated if we track model inheritance
                "child_models": [],  # Would be populated if we track model derivation
                "experiments": [],  # Experiments this model was part of
                "deployments": [],  # Deployments using this model
            }

            # Find experiments containing this model
            for exp_id, exp_info in self.registry_index["experiments"].items():
                try:
                    experiment_path = self.experiments_path / f"{exp_id}.json"
                    if experiment_path.exists():
                        with open(experiment_path) as f:
                            exp_data = json.load(f)

                        if model_id in exp_data.get("models", []):
                            lineage["experiments"].append(
                                {
                                    "experiment_id": exp_id,
                                    "name": exp_info["name"],
                                    "created_at": exp_info["created_at"],
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to check experiment {exp_id}: {e}")

            # Find deployments using this model
            for dep_id, dep_info in self.registry_index["deployments"].items():
                if dep_info.get("model_id") == model_id:
                    lineage["deployments"].append(
                        {
                            "deployment_id": dep_id,
                            "environment": dep_info["environment"],
                            "deployed_at": dep_info["deployed_at"],
                        }
                    )

            logger.info(f"✅ Model lineage retrieved: {model_id}")
            return lineage

        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            raise

    async def archive_old_models(self, retention_days: int = 90) -> list[str]:
        """Archive models older than retention period."""
        logger.info(f"Archiving models older than {retention_days} days")

        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            archived_models = []

            models = await self.list_models()

            for model in models:
                if (
                    model.status == ModelStatus.DEVELOPMENT
                    and model.created_at < cutoff_date
                ):
                    # Archive the model
                    success = await self.update_model_status(
                        model.model_id, ModelStatus.ARCHIVED
                    )
                    if success:
                        archived_models.append(model.model_id)
                        logger.info(f"Archived model: {model.model_id}")

            logger.info(f"✅ Archived {len(archived_models)} models")
            return archived_models

        except Exception as e:
            logger.error(f"Failed to archive old models: {e}")
            raise

    async def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        logger.info("Getting registry statistics")

        try:
            stats = {
                "total_models": len(self.registry_index["models"]),
                "total_experiments": len(self.registry_index["experiments"]),
                "total_deployments": len(self.registry_index["deployments"]),
                "models_by_status": {},
                "models_by_type": {},
                "models_by_author": {},
                "storage_usage": {},
                "timestamp": datetime.now().isoformat(),
            }

            # Count models by status and type
            models = await self.list_models()

            for model in models:
                # Count by status
                status_key = model.status.value
                stats["models_by_status"][status_key] = (
                    stats["models_by_status"].get(status_key, 0) + 1
                )

                # Count by type
                type_key = model.model_type.value
                stats["models_by_type"][type_key] = (
                    stats["models_by_type"].get(type_key, 0) + 1
                )

                # Count by author
                author_key = model.author
                stats["models_by_author"][author_key] = (
                    stats["models_by_author"].get(author_key, 0) + 1
                )

            # Calculate storage usage
            total_size = sum(
                f.stat().st_size for f in self.models_path.rglob("*") if f.is_file()
            )
            stats["storage_usage"] = {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "average_model_size_mb": round(
                    total_size / len(models) / 1024 / 1024, 2
                )
                if models
                else 0,
            }

            logger.info("✅ Registry statistics retrieved")
            return stats

        except Exception as e:
            logger.error(f"Failed to get registry statistics: {e}")
            raise


# Global model registry instance
model_registry = ModelRegistry()

# Make registry available for import
__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "ModelStatus",
    "ModelType",
    "model_registry",
]
