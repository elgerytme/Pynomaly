"""File-based implementations for model registry operations."""

import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mlops.domain.interfaces.model_registry_operations import (
    ModelRegistryPort,
    ModelLifecyclePort,
    ModelDeploymentPort,
    ModelStoragePort,
    ModelVersioningPort,
    ModelSearchPort,
    ModelRegistrationRequest,
    ModelInfo,
    ModelStatus,
    ModelValidationResult,
    DeploymentConfig,
    DeploymentInfo,
    DeploymentStage,
    ModelSearchQuery
)

logger = logging.getLogger(__name__)


class FileBasedModelRegistry(ModelRegistryPort):
    """File-based implementation for model registry."""
    
    def __init__(self, storage_path: str = "./mlops_data/models"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self._storage_path / "registry.json"
        self._models = self._load_registry()
        logger.info(f"FileBasedModelRegistry initialized with storage at {storage_path}")
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load model registry from file."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for model_data in data.values():
                        model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                        model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    return data
            except Exception as e:
                logger.warning(f"Failed to load registry file: {e}")
        return {}
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for model_key, model_data in self._models.items():
                serializable_data[model_key] = model_data.copy()
                serializable_data[model_key]['created_at'] = model_data['created_at'].isoformat()
                serializable_data[model_key]['updated_at'] = model_data['updated_at'].isoformat()
            
            with open(self._registry_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry file: {e}")
    
    async def register_model(self, request: ModelRegistrationRequest) -> str:
        """Register a new model in the registry."""
        model_key = f"{request.metadata.model_id}:{request.metadata.version}"
        
        # Convert metadata and metrics to serializable format
        metadata_dict = {
            "model_id": request.metadata.model_id,
            "version": request.metadata.version,
            "algorithm": request.metadata.algorithm,
            "framework": request.metadata.framework.value,
            "created_by": request.metadata.created_by,
            "description": request.metadata.description,
            "tags": request.metadata.tags,
            "hyperparameters": request.metadata.hyperparameters,
            "training_data_info": request.metadata.training_data_info,
            "feature_schema": request.metadata.feature_schema,
            "model_signature": request.metadata.model_signature,
            "dependencies": request.metadata.dependencies,
            "custom_metadata": request.metadata.custom_metadata
        }
        
        metrics_dict = {
            "accuracy": request.metrics.accuracy,
            "precision": request.metrics.precision,
            "recall": request.metrics.recall,
            "f1_score": request.metrics.f1_score,
            "auc_roc": request.metrics.auc_roc,
            "custom_metrics": request.metrics.custom_metrics
        }
        
        model_data = {
            "model_id": request.metadata.model_id,
            "version": request.metadata.version,
            "status": ModelStatus.REGISTERED.value,
            "metadata": metadata_dict,
            "metrics": metrics_dict,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "model_path": request.model_path,
            "model_checksum": f"checksum_{str(uuid4())[:16]}",
            "experiment_id": request.experiment_id
        }
        
        self._models[model_key] = model_data
        self._save_registry()
        
        logger.info(f"Registered model {model_key}")
        return model_key
    
    async def get_model(self, model_id: str, version: str = "latest") -> Optional[ModelInfo]:
        """Get model information from registry."""
        if version == "latest":
            # Find latest version
            model_versions = [
                (key, data) for key, data in self._models.items()
                if key.startswith(f"{model_id}:")
            ]
            if not model_versions:
                return None
            # Sort by created_at and get the latest
            latest_key, model_data = max(model_versions, key=lambda x: x[1]['created_at'])
        else:
            model_key = f"{model_id}:{version}"
            model_data = self._models.get(model_key)
            if not model_data:
                return None
        
        return self._convert_to_model_info(model_data)
    
    def _convert_to_model_info(self, model_data: Dict[str, Any]) -> ModelInfo:
        """Convert stored model data to ModelInfo object."""
        from mlops.domain.interfaces.model_registry_operations import (
            ModelMetadata, ModelMetrics, ModelFramework
        )
        
        # Reconstruct metadata
        metadata_dict = model_data["metadata"]
        metadata = ModelMetadata(
            model_id=metadata_dict["model_id"],
            version=metadata_dict["version"],
            algorithm=metadata_dict["algorithm"],
            framework=ModelFramework(metadata_dict["framework"]),
            created_by=metadata_dict["created_by"],
            description=metadata_dict["description"],
            tags=metadata_dict["tags"],
            hyperparameters=metadata_dict["hyperparameters"],
            training_data_info=metadata_dict["training_data_info"],
            feature_schema=metadata_dict["feature_schema"],
            model_signature=metadata_dict["model_signature"],
            dependencies=metadata_dict["dependencies"],
            custom_metadata=metadata_dict["custom_metadata"]
        )
        
        # Reconstruct metrics
        metrics_dict = model_data["metrics"]
        metrics = ModelMetrics(
            accuracy=metrics_dict["accuracy"],
            precision=metrics_dict["precision"],
            recall=metrics_dict["recall"],
            f1_score=metrics_dict["f1_score"],
            auc_roc=metrics_dict["auc_roc"],
            custom_metrics=metrics_dict["custom_metrics"]
        )
        
        return ModelInfo(
            model_id=model_data["model_id"],
            version=model_data["version"],
            status=ModelStatus(model_data["status"]),
            metadata=metadata,
            metrics=metrics,
            created_at=model_data["created_at"],
            updated_at=model_data["updated_at"],
            model_path=model_data["model_path"],
            model_checksum=model_data["model_checksum"],
            experiment_id=model_data.get("experiment_id")
        )
    
    async def list_models(
        self,
        query: Optional[ModelSearchQuery] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ModelInfo]:
        """List models with optional filters."""
        models = []
        
        for model_data in self._models.values():
            # Apply filters if query provided
            if query:
                if query.model_ids and model_data["model_id"] not in query.model_ids:
                    continue
                if query.status and model_data["status"] != query.status.value:
                    continue
                if query.algorithm and model_data["metadata"]["algorithm"] != query.algorithm:
                    continue
                if query.framework and model_data["metadata"]["framework"] != query.framework.value:
                    continue
                if query.tags:
                    model_tags = model_data["metadata"]["tags"]
                    if not any(tag in model_tags for tag in query.tags):
                        continue
                if query.created_by and model_data["metadata"]["created_by"] != query.created_by:
                    continue
                if query.min_accuracy:
                    accuracy = model_data["metrics"]["accuracy"]
                    if not accuracy or accuracy < query.min_accuracy:
                        continue
            
            models.append(self._convert_to_model_info(model_data))
        
        # Sort by created_at descending
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return models[offset:offset + limit]
    
    async def update_model_metadata(
        self,
        model_id: str,
        version: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update model metadata."""
        model_key = f"{model_id}:{version}"
        if model_key not in self._models:
            return False
        
        # Update metadata fields
        model_data = self._models[model_key]
        for key, value in metadata_updates.items():
            if key in model_data["metadata"]:
                model_data["metadata"][key] = value
        
        model_data["updated_at"] = datetime.utcnow()
        self._save_registry()
        
        logger.info(f"Updated metadata for model {model_key}")
        return True
    
    async def delete_model(self, model_id: str, version: str) -> bool:
        """Delete a model from the registry."""
        model_key = f"{model_id}:{version}"
        if model_key not in self._models:
            return False
        
        del self._models[model_key]
        self._save_registry()
        
        # Also delete model files if they exist
        model_dir = self._storage_path / model_id / version
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        logger.info(f"Deleted model {model_key}")
        return True


class FileBasedModelLifecycle(ModelLifecyclePort):
    """File-based implementation for model lifecycle management."""
    
    def __init__(self, registry: FileBasedModelRegistry):
        self._registry = registry
        logger.info("FileBasedModelLifecycle initialized")
    
    async def promote_model(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStatus
    ) -> bool:
        """Promote model to different lifecycle stage."""
        model_key = f"{model_id}:{version}"
        if model_key not in self._registry._models:
            return False
        
        self._registry._models[model_key]["status"] = target_stage.value
        self._registry._models[model_key]["updated_at"] = datetime.utcnow()
        self._registry._save_registry()
        
        logger.info(f"Promoted model {model_key} to {target_stage.value}")
        return True
    
    async def validate_model(
        self,
        model_id: str,
        version: str,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ModelValidationResult:
        """Validate a model for promotion."""
        model_key = f"{model_id}:{version}"
        model_data = self._registry._models.get(model_key)
        
        if not model_data:
            return ModelValidationResult(
                is_valid=False,
                validation_checks={},
                errors=[f"Model {model_key} not found"],
                warnings=[]
            )
        
        # Perform basic validation checks
        validation_checks = {
            "model_exists": True,
            "has_metrics": bool(model_data["metrics"]["accuracy"]),
            "has_description": bool(model_data["metadata"]["description"]),
            "has_tags": len(model_data["metadata"]["tags"]) > 0,
            "performance_threshold": True  # Assume passes for file-based impl
        }
        
        errors = []
        warnings = []
        
        # Check minimum performance requirements
        accuracy = model_data["metrics"]["accuracy"]
        if accuracy and accuracy < 0.7:
            validation_checks["performance_threshold"] = False
            errors.append(f"Model accuracy {accuracy} below minimum threshold 0.7")
        
        # Add warnings for missing optional fields
        if not model_data["metadata"]["description"]:
            warnings.append("Model description is empty")
        
        if not model_data["metadata"]["tags"]:
            warnings.append("Model has no tags")
        
        is_valid = len(errors) == 0
        
        result = ModelValidationResult(
            is_valid=is_valid,
            validation_checks=validation_checks,
            errors=errors,
            warnings=warnings
        )
        
        logger.info(f"Validated model {model_key}: valid={is_valid}")
        return result
    
    async def check_promotion_eligibility(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStatus
    ) -> Dict[str, Any]:
        """Check if model is eligible for promotion."""
        model_key = f"{model_id}:{version}"
        model_data = self._registry._models.get(model_key)
        
        if not model_data:
            return {
                "eligible": False,
                "requirements": {},
                "blocking_issues": [f"Model {model_key} not found"],
                "recommendations": []
            }
        
        current_status = ModelStatus(model_data["status"])
        blocking_issues = []
        requirements = {}
        
        # Check stage progression rules
        if target_stage == ModelStatus.STAGING and current_status != ModelStatus.REGISTERED:
            blocking_issues.append(f"Can only promote to staging from registered status, current: {current_status.value}")
        
        if target_stage == ModelStatus.PRODUCTION and current_status not in [ModelStatus.STAGING, ModelStatus.REGISTERED]:
            blocking_issues.append(f"Can only promote to production from staging or registered status, current: {current_status.value}")
        
        # Check performance requirements
        accuracy = model_data["metrics"]["accuracy"]
        requirements["min_accuracy"] = {
            "required": 0.8 if target_stage == ModelStatus.PRODUCTION else 0.7,
            "current": accuracy or 0.0,
            "passed": accuracy is not None and accuracy >= (0.8 if target_stage == ModelStatus.PRODUCTION else 0.7)
        }
        
        if not requirements["min_accuracy"]["passed"]:
            blocking_issues.append(f"Model accuracy {accuracy} below required threshold")
        
        return {
            "eligible": len(blocking_issues) == 0,
            "requirements": requirements,
            "blocking_issues": blocking_issues,
            "recommendations": ["Ensure model meets all requirements before promotion"] if blocking_issues else []
        }
    
    async def deprecate_model(
        self,
        model_id: str,
        version: str,
        reason: str
    ) -> bool:
        """Deprecate a model."""
        model_key = f"{model_id}:{version}"
        if model_key not in self._registry._models:
            return False
        
        self._registry._models[model_key]["status"] = ModelStatus.DEPRECATED.value
        self._registry._models[model_key]["updated_at"] = datetime.utcnow()
        self._registry._models[model_key]["deprecation_reason"] = reason
        self._registry._save_registry()
        
        logger.info(f"Deprecated model {model_key}: {reason}")
        return True
    
    async def archive_model(
        self,
        model_id: str,
        version: str,
        retain_metadata: bool = True
    ) -> bool:
        """Archive a model."""
        model_key = f"{model_id}:{version}"
        if model_key not in self._registry._models:
            return False
        
        if retain_metadata:
            self._registry._models[model_key]["status"] = ModelStatus.ARCHIVED.value
            self._registry._models[model_key]["updated_at"] = datetime.utcnow()
            self._registry._save_registry()
        else:
            # Remove from registry
            del self._registry._models[model_key]
            self._registry._save_registry()
        
        # Move model files to archive directory
        model_dir = self._registry._storage_path / model_id / version
        archive_dir = self._registry._storage_path / "archived" / model_id / version
        
        if model_dir.exists():
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(model_dir), str(archive_dir))
        
        logger.info(f"Archived model {model_key} (retain_metadata={retain_metadata})")
        return True


class FileBasedModelDeployment(ModelDeploymentPort):
    """File-based implementation for model deployment."""
    
    def __init__(self, storage_path: str = "./mlops_data/deployments"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._deployments_file = self._storage_path / "deployments.json"
        self._deployments = self._load_deployments()
        logger.info(f"FileBasedModelDeployment initialized with storage at {storage_path}")
    
    def _load_deployments(self) -> Dict[str, Dict[str, Any]]:
        """Load deployments from file."""
        if self._deployments_file.exists():
            try:
                with open(self._deployments_file, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for deployment_data in data.values():
                        deployment_data['deployment_time'] = datetime.fromisoformat(deployment_data['deployment_time'])
                    return data
            except Exception as e:
                logger.warning(f"Failed to load deployments file: {e}")
        return {}
    
    def _save_deployments(self) -> None:
        """Save deployments to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for deployment_id, deployment_data in self._deployments.items():
                serializable_data[deployment_id] = deployment_data.copy()
                serializable_data[deployment_id]['deployment_time'] = deployment_data['deployment_time'].isoformat()
            
            with open(self._deployments_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save deployments file: {e}")
    
    async def deploy_model(
        self,
        model_id: str,
        version: str,
        config: DeploymentConfig
    ) -> Optional[str]:
        """Deploy model to specified environment."""
        deployment_id = f"deploy_{str(uuid4())[:8]}"
        
        deployment_data = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "model_version": version,
            "stage": config.stage.value,
            "endpoint_url": f"http://localhost:8000/models/{model_id}/{version}/predict",
            "replicas": config.replicas,
            "resources": config.resources,
            "environment_variables": config.environment_variables,
            "auto_scaling": config.auto_scaling or {},
            "traffic_percentage": config.traffic_percentage,
            "deployment_time": datetime.utcnow(),
            "health_status": "healthy",
            "metrics": {"requests_per_second": 0.0, "average_latency_ms": 0.0}
        }
        
        self._deployments[deployment_id] = deployment_data
        self._save_deployments()
        
        logger.info(f"Deployed model {model_id}:{version} as {deployment_id}")
        return deployment_id
    
    async def get_deployment(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Get deployment information."""
        deployment_data = self._deployments.get(deployment_id)
        if not deployment_data:
            return None
        
        return DeploymentInfo(
            deployment_id=deployment_data["deployment_id"],
            model_id=deployment_data["model_id"],
            model_version=deployment_data["model_version"],
            stage=DeploymentStage(deployment_data["stage"]),
            endpoint_url=deployment_data["endpoint_url"],
            replicas=deployment_data["replicas"],
            resources=deployment_data["resources"],
            auto_scaling=deployment_data["auto_scaling"],
            traffic_percentage=deployment_data["traffic_percentage"],
            deployment_time=deployment_data["deployment_time"],
            health_status=deployment_data["health_status"],
            metrics=deployment_data["metrics"]
        )
    
    async def list_deployments(
        self,
        model_id: Optional[str] = None,
        stage: Optional[DeploymentStage] = None,
        status: Optional[str] = None
    ) -> List[DeploymentInfo]:
        """List deployments with optional filters."""
        deployments = []
        
        for deployment_data in self._deployments.values():
            # Apply filters
            if model_id and deployment_data["model_id"] != model_id:
                continue
            if stage and deployment_data["stage"] != stage.value:
                continue
            if status and deployment_data["health_status"] != status:
                continue
            
            deployments.append(DeploymentInfo(
                deployment_id=deployment_data["deployment_id"],
                model_id=deployment_data["model_id"],
                model_version=deployment_data["model_version"],
                stage=DeploymentStage(deployment_data["stage"]),
                endpoint_url=deployment_data["endpoint_url"],
                replicas=deployment_data["replicas"],
                resources=deployment_data["resources"],
                auto_scaling=deployment_data["auto_scaling"],
                traffic_percentage=deployment_data["traffic_percentage"],
                deployment_time=deployment_data["deployment_time"],
                health_status=deployment_data["health_status"],
                metrics=deployment_data["metrics"]
            ))
        
        # Sort by deployment time descending
        deployments.sort(key=lambda x: x.deployment_time, reverse=True)
        return deployments
    
    async def update_deployment(
        self,
        deployment_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update deployment configuration."""
        if deployment_id not in self._deployments:
            return False
        
        deployment_data = self._deployments[deployment_id]
        
        # Update allowed fields
        allowed_updates = ["replicas", "resources", "environment_variables", "traffic_percentage", "health_status", "metrics"]
        for key, value in updates.items():
            if key in allowed_updates:
                deployment_data[key] = value
        
        self._save_deployments()
        logger.info(f"Updated deployment {deployment_id}")
        return True
    
    async def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment to specified number of replicas."""
        return await self.update_deployment(deployment_id, {"replicas": replicas})
    
    async def undeploy_model(self, deployment_id: str) -> bool:
        """Remove model deployment."""
        if deployment_id not in self._deployments:
            return False
        
        del self._deployments[deployment_id]
        self._save_deployments()
        
        logger.info(f"Undeployed {deployment_id}")
        return True


class FileBasedModelStorage(ModelStoragePort):
    """File-based implementation for model storage."""
    
    def __init__(self, storage_path: str = "./mlops_data/model_storage"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedModelStorage initialized with storage at {storage_path}")
    
    async def store_model(
        self,
        model_path: str,
        model_id: str,
        version: str,
        metadata: Any
    ) -> str:
        """Store model artifacts."""
        # Create directory for this model version
        model_dir = self._storage_path / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file to storage
        source_path = Path(model_path)
        storage_path = model_dir / source_path.name
        
        if source_path.exists():
            shutil.copy2(source_path, storage_path)
        else:
            # Create a placeholder file for testing
            storage_path.write_text(f"Model placeholder for {model_id}:{version}")
        
        # Store metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Stored model {model_id}:{version} to {storage_path}")
        return str(storage_path)
    
    async def retrieve_model(
        self,
        model_id: str,
        version: str,
        download_path: str
    ) -> bool:
        """Retrieve model from storage."""
        model_dir = self._storage_path / model_id / version
        
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return False
        
        # Find model file (assume first non-metadata file)
        model_files = [f for f in model_dir.iterdir() if f.name != "metadata.json"]
        
        if not model_files:
            logger.error(f"No model files found in {model_dir}")
            return False
        
        source_file = model_files[0]
        destination_path = Path(download_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_file, destination_path)
        logger.info(f"Retrieved model {model_id}:{version} to {download_path}")
        return True
    
    async def verify_model_integrity(
        self,
        model_id: str,
        version: str
    ) -> bool:
        """Verify model file integrity."""
        model_dir = self._storage_path / model_id / version
        
        if not model_dir.exists():
            return False
        
        # Check if model files exist
        model_files = [f for f in model_dir.iterdir() if f.name != "metadata.json"]
        
        if not model_files:
            return False
        
        # For file-based implementation, just check file exists and is readable
        try:
            for model_file in model_files:
                model_file.read_bytes()
            logger.info(f"Verified integrity for model {model_id}:{version}")
            return True
        except Exception as e:
            logger.error(f"Integrity check failed for model {model_id}:{version}: {e}")
            return False
    
    async def get_model_info(
        self,
        model_id: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Get storage information about a model."""
        model_dir = self._storage_path / model_id / version
        
        if not model_dir.exists():
            return None
        
        # Get model files
        model_files = [f for f in model_dir.iterdir() if f.name != "metadata.json"]
        
        if not model_files:
            return None
        
        main_model_file = model_files[0]
        
        model_info = {
            "model_id": model_id,
            "version": version,
            "storage_path": str(main_model_file),
            "size_bytes": main_model_file.stat().st_size,
            "checksum": f"checksum_{str(uuid4())[:16]}",  # Simple stub checksum
            "stored_at": datetime.fromtimestamp(main_model_file.stat().st_ctime).isoformat(),
            "storage_backend": "file_based"
        }
        
        logger.info(f"Retrieved storage info for model {model_id}:{version}")
        return model_info
    
    async def delete_model_artifacts(
        self,
        model_id: str,
        version: str
    ) -> bool:
        """Delete model artifacts from storage."""
        model_dir = self._storage_path / model_id / version
        
        if not model_dir.exists():
            return False
        
        try:
            shutil.rmtree(model_dir)
            
            # Clean up empty parent directory
            parent_dir = model_dir.parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
            
            logger.info(f"Deleted artifacts for model {model_id}:{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete artifacts for model {model_id}:{version}: {e}")
            return False


class FileBasedModelVersioning(ModelVersioningPort):
    """File-based implementation for model versioning."""
    
    def __init__(self, registry: FileBasedModelRegistry):
        self._registry = registry
        self._versions = self._extract_versions()
        logger.info("FileBasedModelVersioning initialized")
    
    def _extract_versions(self) -> Dict[str, List[str]]:
        """Extract version information from registry."""
        versions = {}
        for model_key in self._registry._models.keys():
            model_id, version = model_key.split(':', 1)
            if model_id not in versions:
                versions[model_id] = []
            versions[model_id].append(version)
        
        # Sort versions for each model
        for model_id in versions:
            versions[model_id].sort()
        
        return versions
    
    async def create_version(
        self,
        model_id: str,
        parent_version: Optional[str] = None,
        version_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new model version."""
        # Refresh versions from registry
        self._versions = self._extract_versions()
        
        if model_id not in self._versions:
            self._versions[model_id] = []
        
        current_versions = self._versions[model_id]
        
        if not current_versions:
            new_version = "1.0.0"
        else:
            # Simple version increment
            latest_version = max(current_versions)
            try:
                major, minor, patch = latest_version.split('.')
                new_version = f"{major}.{minor}.{int(patch) + 1}"
            except ValueError:
                # Fallback for non-semantic versions
                new_version = f"{len(current_versions) + 1}.0.0"
        
        self._versions[model_id].append(new_version)
        logger.info(f"Created version {new_version} for model {model_id}")
        return new_version
    
    async def list_versions(
        self,
        model_id: str,
        include_archived: bool = False
    ) -> List[str]:
        """List all versions of a model."""
        # Refresh versions from registry
        self._versions = self._extract_versions()
        
        versions = self._versions.get(model_id, [])
        
        if not include_archived:
            # Filter out archived versions
            active_versions = []
            for version in versions:
                model_key = f"{model_id}:{version}"
                model_data = self._registry._models.get(model_key)
                if model_data and model_data["status"] != ModelStatus.ARCHIVED.value:
                    active_versions.append(version)
            versions = active_versions
        
        logger.info(f"Listed {len(versions)} versions for model {model_id}")
        return versions
    
    async def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        model1_key = f"{model_id}:{version1}"
        model2_key = f"{model_id}:{version2}"
        
        model1_data = self._registry._models.get(model1_key)
        model2_data = self._registry._models.get(model2_key)
        
        if not model1_data or not model2_data:
            return {
                "error": f"One or both versions not found: {version1}, {version2}"
            }
        
        # Compare performance metrics
        metrics1 = model1_data["metrics"]
        metrics2 = model2_data["metrics"]
        
        performance_diff = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            if val1 is not None and val2 is not None:
                performance_diff[metric] = {
                    "v1": val1,
                    "v2": val2,
                    "improvement": val2 - val1,
                    "percent_change": ((val2 - val1) / val1) * 100 if val1 != 0 else 0
                }
        
        # Compare hyperparameters
        params1 = model1_data["metadata"]["hyperparameters"]
        params2 = model2_data["metadata"]["hyperparameters"]
        
        param_changes = {
            "added": {k: v for k, v in params2.items() if k not in params1},
            "removed": {k: v for k, v in params1.items() if k not in params2},
            "modified": {
                k: {"from": params1[k], "to": params2[k]}
                for k in params1.keys() & params2.keys()
                if params1[k] != params2[k]
            }
        }
        
        comparison_result = {
            "model_id": model_id,
            "version1": version1,
            "version2": version2,
            "differences": {
                "performance": performance_diff,
                "parameters": param_changes,
                "status": {
                    "v1": model1_data["status"],
                    "v2": model2_data["status"]
                }
            },
            "recommendation": self._generate_recommendation(performance_diff, version1, version2)
        }
        
        logger.info(f"Compared versions {version1} and {version2} for model {model_id}")
        return comparison_result
    
    def _generate_recommendation(
        self,
        performance_diff: Dict[str, Any],
        version1: str,
        version2: str
    ) -> str:
        """Generate a recommendation based on performance differences."""
        if not performance_diff:
            return "Insufficient performance data for recommendation"
        
        improvements = 0
        degradations = 0
        
        for metric, diff in performance_diff.items():
            if diff["improvement"] > 0:
                improvements += 1
            elif diff["improvement"] < 0:
                degradations += 1
        
        if improvements > degradations:
            return f"Version {version2} shows overall improvement over {version1}"
        elif degradations > improvements:
            return f"Version {version1} performs better than {version2}"
        else:
            return f"Versions {version1} and {version2} have similar performance"
    
    async def get_version_lineage(
        self,
        model_id: str,
        version: str
    ) -> Dict[str, Any]:
        """Get version lineage and history."""
        # For file-based implementation, create simple lineage
        versions = await self.list_versions(model_id, include_archived=True)
        
        try:
            version_index = versions.index(version)
            parent_version = versions[version_index - 1] if version_index > 0 else None
            child_versions = versions[version_index + 1:version_index + 2]  # Next version only
        except ValueError:
            parent_version = None
            child_versions = []
        
        model_key = f"{model_id}:{version}"
        model_data = self._registry._models.get(model_key)
        
        lineage = {
            "model_id": model_id,
            "version": version,
            "parent_version": parent_version,
            "child_versions": child_versions,
            "created_at": model_data["created_at"].isoformat() if model_data else None,
            "created_by": model_data["metadata"]["created_by"] if model_data else None,
            "branch_info": {
                "branch": "main",
                "commit_hash": f"commit_{str(uuid4())[:8]}"
            },
            "changelog": [
                "Initial version" if not parent_version else "Version update"
            ]
        }
        
        logger.info(f"Retrieved lineage for model {model_id}:{version}")
        return lineage
    
    async def tag_version(
        self,
        model_id: str,
        version: str,
        tag: str
    ) -> bool:
        """Tag a model version."""
        model_key = f"{model_id}:{version}"
        model_data = self._registry._models.get(model_key)
        
        if not model_data:
            return False
        
        # Add tag to metadata
        if "version_tags" not in model_data["metadata"]:
            model_data["metadata"]["version_tags"] = []
        
        if tag not in model_data["metadata"]["version_tags"]:
            model_data["metadata"]["version_tags"].append(tag)
            model_data["updated_at"] = datetime.utcnow()
            self._registry._save_registry()
        
        logger.info(f"Tagged version {version} of model {model_id} with '{tag}'")
        return True


class FileBasedModelSearch(ModelSearchPort):
    """File-based implementation for model search and discovery."""
    
    def __init__(self, registry: FileBasedModelRegistry):
        self._registry = registry
        logger.info("FileBasedModelSearch initialized")
    
    async def search_models(
        self,
        query: ModelSearchQuery,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0
    ) -> List[ModelInfo]:
        """Search models with advanced query."""
        # Use registry's list_models method with the query
        models = await self._registry.list_models(query, limit=1000)  # Get all matching
        
        # Apply text search if query has text_query
        if hasattr(query, 'text_query') and query.text_query:
            text_query = query.text_query.lower()
            filtered_models = []
            
            for model in models:
                score = 0
                
                # Search in model_id
                if text_query in model.model_id.lower():
                    score += 10
                
                # Search in algorithm
                if text_query in model.metadata.algorithm.lower():
                    score += 5
                
                # Search in description
                if model.metadata.description and text_query in model.metadata.description.lower():
                    score += 3
                
                # Search in tags
                for tag in model.metadata.tags:
                    if text_query in tag.lower():
                        score += 2
                
                if score > 0:
                    filtered_models.append((model, score))
            
            # Sort by relevance score
            filtered_models.sort(key=lambda x: x[1], reverse=True)
            models = [model for model, _ in filtered_models]
        
        # Apply sorting
        if sort_by == "created_at":
            models.sort(key=lambda x: x.created_at, reverse=(sort_order == "desc"))
        elif sort_by == "updated_at":
            models.sort(key=lambda x: x.updated_at, reverse=(sort_order == "desc"))
        elif sort_by in ["accuracy", "f1_score", "precision", "recall"]:
            models.sort(
                key=lambda x: getattr(x.metrics, sort_by) or 0,
                reverse=(sort_order == "desc")
            )
        
        # Apply pagination
        paginated_models = models[offset:offset + limit]
        
        logger.info(f"Search found {len(paginated_models)} models")
        return paginated_models
    
    async def find_similar_models(
        self,
        model_id: str,
        version: str,
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[ModelInfo]:
        """Find models similar to the given model."""
        target_model = await self._registry.get_model(model_id, version)
        if not target_model:
            return []
        
        # Get all models
        all_models = await self._registry.list_models(limit=1000)
        
        # Exclude the target model itself
        all_models = [m for m in all_models if not (m.model_id == model_id and m.version == version)]
        
        similar_models = []
        
        for model in all_models:
            similarity_score = self._calculate_similarity(target_model, model)
            
            if similarity_score >= similarity_threshold:
                similar_models.append((model, similarity_score))
        
        # Sort by similarity score descending
        similar_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return top models
        result = [model for model, _ in similar_models[:limit]]
        
        logger.info(f"Found {len(result)} similar models to {model_id}:{version}")
        return result
    
    def _calculate_similarity(self, model1: ModelInfo, model2: ModelInfo) -> float:
        """Calculate similarity score between two models."""
        score = 0.0
        
        # Algorithm similarity (high weight)
        if model1.metadata.algorithm == model2.metadata.algorithm:
            score += 0.4
        
        # Framework similarity
        if model1.metadata.framework == model2.metadata.framework:
            score += 0.2
        
        # Tag similarity
        common_tags = set(model1.metadata.tags) & set(model2.metadata.tags)
        total_tags = set(model1.metadata.tags) | set(model2.metadata.tags)
        if total_tags:
            score += 0.2 * (len(common_tags) / len(total_tags))
        
        # Performance similarity (based on accuracy)
        if model1.metrics.accuracy and model2.metrics.accuracy:
            accuracy_diff = abs(model1.metrics.accuracy - model2.metrics.accuracy)
            accuracy_similarity = max(0, 1 - accuracy_diff)  # Closer to 1 means more similar
            score += 0.2 * accuracy_similarity
        
        return score
    
    async def recommend_models(
        self,
        use_case: str,
        data_characteristics: Dict[str, Any],
        performance_requirements: Dict[str, float],
        limit: int = 5
    ) -> List[ModelInfo]:
        """Recommend models for a use case."""
        # Get all models
        all_models = await self._registry.list_models(limit=1000)
        
        # Filter to production and staging models only
        candidate_models = [
            m for m in all_models 
            if m.status in [ModelStatus.PRODUCTION, ModelStatus.STAGING]
        ]
        
        recommendations = []
        
        for model in candidate_models:
            recommendation_score = self._calculate_recommendation_score(
                model, use_case, data_characteristics, performance_requirements
            )
            
            if recommendation_score > 0:
                recommendations.append((model, recommendation_score))
        
        # Sort by recommendation score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        result = [model for model, _ in recommendations[:limit]]
        
        logger.info(f"Generated {len(result)} model recommendations for use case '{use_case}'")
        return result
    
    def _calculate_recommendation_score(
        self,
        model: ModelInfo,
        use_case: str,
        data_characteristics: Dict[str, Any],
        performance_requirements: Dict[str, float]
    ) -> float:
        """Calculate recommendation score for a model."""
        score = 0.0
        
        # Use case relevance (simple keyword matching)
        use_case_lower = use_case.lower()
        if any(keyword in model.metadata.algorithm.lower() for keyword in use_case_lower.split()):
            score += 20
        
        if model.metadata.description:
            if any(keyword in model.metadata.description.lower() for keyword in use_case_lower.split()):
                score += 10
        
        # Performance requirements matching
        performance_score = 0
        performance_count = 0
        
        for metric, required_value in performance_requirements.items():
            model_value = getattr(model.metrics, metric, None)
            if model_value is not None:
                performance_count += 1
                if model_value >= required_value:
                    performance_score += 10
                else:
                    # Partial score based on how close it is
                    performance_score += max(0, (model_value / required_value) * 5)
        
        if performance_count > 0:
            score += performance_score / performance_count
        
        # Status bonus (production models preferred)
        if model.status == ModelStatus.PRODUCTION:
            score += 5
        elif model.status == ModelStatus.STAGING:
            score += 3
        
        # Recency bonus
        days_old = (datetime.utcnow() - model.created_at).days
        if days_old < 30:
            score += 2
        elif days_old < 90:
            score += 1
        
        return score
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        all_models = await self._registry.list_models(limit=10000)
        
        if not all_models:
            return {
                "total_models": 0,
                "models_by_status": {},
                "models_by_framework": {},
                "models_by_algorithm": {},
                "average_performance": {},
                "recent_activity": {}
            }
        
        # Count by status
        status_counts = {}
        for model in all_models:
            status = model.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by framework
        framework_counts = {}
        for model in all_models:
            framework = model.metadata.framework.value
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
        
        # Count by algorithm
        algorithm_counts = {}
        for model in all_models:
            algorithm = model.metadata.algorithm
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        # Calculate average performance
        performance_metrics = ["accuracy", "precision", "recall", "f1_score"]
        avg_performance = {}
        
        for metric in performance_metrics:
            values = [getattr(model.metrics, metric) for model in all_models 
                     if getattr(model.metrics, metric) is not None]
            if values:
                avg_performance[metric] = sum(values) / len(values)
        
        # Recent activity (last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_models = [m for m in all_models if m.created_at >= recent_cutoff]
        
        statistics = {
            "total_models": len(all_models),
            "models_by_status": status_counts,
            "models_by_framework": framework_counts,
            "models_by_algorithm": algorithm_counts,
            "average_performance": avg_performance,
            "recent_activity": {
                "models_registered_last_week": len(recent_models)
            }
        }
        
        logger.info("Generated model registry statistics")
        return statistics