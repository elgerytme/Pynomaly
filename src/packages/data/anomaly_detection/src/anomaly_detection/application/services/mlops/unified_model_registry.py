"""Unified Model Registry Integration Service.

This service integrates the anomaly detection package with the MLOps package's
model management capabilities, providing a unified interface for model lifecycle
management across both systems.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# Anomaly detection imports
try:
    from data.processing.domain.entities.model import Model as AnomalyModel
except ImportError:
    from anomaly_detection.domain.entities.model import Model as AnomalyModel

try:
    from data.processing.domain.entities.detection_result import DetectionResult
except ImportError:
    from anomaly_detection.domain.entities.detection_result import DetectionResult

try:
    from ai.mlops.domain.services.mlops_service import MLOpsService
    from ai.mlops.domain.value_objects.model_value_objects import ModelVersion as AnomalyModelVersion
except ImportError:
    from anomaly_detection.domain.services.mlops_service import MLOpsService, ModelVersion as AnomalyModelVersion

try:
    from ai.mlops.infrastructure.repositories.model_repository import ModelRepository
except ImportError:
    from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository

# Type stub for MLOps integration (to avoid hard dependency)
try:
    from ai.mlops.domain.entities.model import Model as MLOpsModel, ModelStage
    from ai.mlops.domain.value_objects.model_value_objects import ModelType, PerformanceMetrics, ModelStorageInfo
    from ai.mlops.domain.services.model_management_service import ModelManagementService
    MLOPS_AVAILABLE = True
except ImportError:
    # Create type stubs when MLOps is not available
    MLOPS_AVAILABLE = False
    
    class ModelStage:
        DEVELOPMENT = "development"
        STAGING = "staging"
        PRODUCTION = "production"
        ARCHIVED = "archived"
    
    class ModelType:
        ANOMALY_DETECTION = "anomaly_detection"
    
    MLOpsModel = Any
    PerformanceMetrics = Any
    ModelStorageInfo = Any
    ModelManagementService = Any


@dataclass
class UnifiedModelMetadata:
    """Unified metadata that bridges both systems."""
    model_id: str
    name: str
    description: str
    algorithm: str
    framework: str
    version: str
    performance_metrics: Dict[str, float]
    training_data_info: Dict[str, Any]
    deployment_config: Dict[str, Any]
    tags: List[str]
    created_by: str
    created_at: datetime
    anomaly_model_id: Optional[str] = None
    mlops_model_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "algorithm": self.algorithm,
            "framework": self.framework,
            "version": self.version,
            "performance_metrics": self.performance_metrics,
            "training_data_info": self.training_data_info,
            "deployment_config": self.deployment_config,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "anomaly_model_id": self.anomaly_model_id,
            "mlops_model_id": self.mlops_model_id
        }


@dataclass
class ModelRegistrationRequest:
    """Request for registering a model in the unified registry."""
    name: str
    description: str
    algorithm: str
    model_object: Any
    performance_metrics: Dict[str, float]
    training_data_info: Dict[str, Any]
    deployment_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    created_by: str = "system"
    framework: str = "scikit-learn"
    use_cases: Optional[List[str]] = None
    data_requirements: Optional[Dict[str, Any]] = None


class ModelRegistryProtocol(Protocol):
    """Protocol for model registry operations."""
    
    async def register_model(self, request: ModelRegistrationRequest) -> UnifiedModelMetadata:
        """Register a new model."""
        ...
    
    async def get_model(self, model_id: str) -> Optional[UnifiedModelMetadata]:
        """Get model by ID."""
        ...
    
    async def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[UnifiedModelMetadata]:
        """List models with optional filters."""
        ...
    
    async def promote_model(self, model_id: str, stage: str, promoted_by: str) -> UnifiedModelMetadata:
        """Promote model to a different stage."""
        ...


class UnifiedModelRegistry:
    """Unified model registry that integrates anomaly detection with MLOps."""
    
    def __init__(
        self,
        anomaly_mlops_service: MLOpsService,
        anomaly_model_repository: ModelRepository,
        mlops_model_service: Optional[ModelManagementService] = None
    ):
        """Initialize unified model registry.
        
        Args:
            anomaly_mlops_service: Anomaly detection MLOps service
            anomaly_model_repository: Anomaly detection model repository
            mlops_model_service: Optional MLOps model management service
        """
        self.anomaly_mlops_service = anomaly_mlops_service
        self.anomaly_model_repository = anomaly_model_repository
        self.mlops_model_service = mlops_model_service
        self.logger = logging.getLogger(__name__)
        
        # Registry for mapping between systems
        self._model_mappings: Dict[str, Dict[str, str]] = {}
        
        # Check MLOps availability
        self.mlops_integration_enabled = MLOPS_AVAILABLE and mlops_model_service is not None
        
        self.logger.info(
            f"Unified Model Registry initialized. MLOps integration: {self.mlops_integration_enabled}"
        )
    
    async def register_model(self, request: ModelRegistrationRequest) -> UnifiedModelMetadata:
        """Register a model in both anomaly detection and MLOps systems.
        
        Args:
            request: Model registration request
            
        Returns:
            Unified model metadata
        """
        unified_id = str(uuid.uuid4())
        
        try:
            # 1. Register in anomaly detection system
            anomaly_model_metadata = await self._register_in_anomaly_system(
                unified_id, request
            )
            
            # 2. Register in MLOps system (if available)
            mlops_model_metadata = None
            if self.mlops_integration_enabled:
                mlops_model_metadata = await self._register_in_mlops_system(
                    unified_id, request
                )
            
            # 3. Create unified metadata
            unified_metadata = UnifiedModelMetadata(
                model_id=unified_id,
                name=request.name,
                description=request.description,
                algorithm=request.algorithm,
                framework=request.framework,
                version="1.0.0",
                performance_metrics=request.performance_metrics,
                training_data_info=request.training_data_info,
                deployment_config=request.deployment_config or {},
                tags=request.tags or [],
                created_by=request.created_by,
                created_at=datetime.now(),
                anomaly_model_id=anomaly_model_metadata.get("model_id"),
                mlops_model_id=mlops_model_metadata.get("model_id") if mlops_model_metadata else None
            )
            
            # 4. Store mapping
            self._model_mappings[unified_id] = {
                "anomaly_model_id": anomaly_model_metadata.get("model_id", ""),
                "mlops_model_id": mlops_model_metadata.get("model_id", "") if mlops_model_metadata else "",
                "created_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully registered model '{request.name}' with unified ID: {unified_id}")
            return unified_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to register model '{request.name}': {e}")
            raise
    
    async def _register_in_anomaly_system(
        self, 
        unified_id: str, 
        request: ModelRegistrationRequest
    ) -> Dict[str, Any]:
        """Register model in anomaly detection system."""
        # Create experiment for this model
        experiment_id = self.anomaly_mlops_service.create_experiment(
            experiment_name=f"model_registration_{request.name}",
            description=f"Model registration for {request.name}",
            tags={"unified_registry": "true", "model_id": unified_id},
            created_by=request.created_by
        )
        
        # Start run for model registration
        run_id = self.anomaly_mlops_service.start_run(
            experiment_id=experiment_id,
            parameters={
                "algorithm": request.algorithm,
                "framework": request.framework,
                "unified_model_id": unified_id
            },
            tags={"registration": "true"}
        )
        
        # Log model metrics
        self.anomaly_mlops_service.log_metrics(run_id, request.performance_metrics)
        
        # Log model object
        model_path = self.anomaly_mlops_service.log_model(
            run_id=run_id,
            model=request.model_object,
            model_name=request.name
        )
        
        # End run
        self.anomaly_mlops_service.end_run(run_id, "completed")
        
        # Register model version
        model_version = self.anomaly_mlops_service.register_model_version(
            model_id=request.name,
            run_id=run_id,
            performance_metrics=request.performance_metrics,
            deployment_config=request.deployment_config
        )
        
        return {
            "model_id": request.name,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "model_path": model_path,
            "version": model_version.version
        }
    
    async def _register_in_mlops_system(
        self, 
        unified_id: str, 
        request: ModelRegistrationRequest
    ) -> Optional[Dict[str, Any]]:
        """Register model in MLOps system."""
        if not self.mlops_integration_enabled:
            return None
        
        try:
            # Create MLOps model
            mlops_model = await self.mlops_model_service.create_model(
                name=f"{request.name}_unified_{unified_id[:8]}",
                description=request.description,
                model_type=ModelType.ANOMALY_DETECTION,
                algorithm_family=request.algorithm,
                created_by=request.created_by,
                use_cases=request.use_cases or ["anomaly_detection"],
                data_requirements=request.data_requirements or {}
            )
            
            # Create performance metrics object
            performance_metrics = PerformanceMetrics(
                accuracy=request.performance_metrics.get("accuracy", 0.0),
                precision=request.performance_metrics.get("precision", 0.0),
                recall=request.performance_metrics.get("recall", 0.0),
                f1_score=request.performance_metrics.get("f1_score", 0.0),
                training_time=request.performance_metrics.get("training_time", 0.0),
                inference_time=request.performance_metrics.get("inference_time", 0.0),
                model_size_mb=request.performance_metrics.get("model_size_mb", 0.0),
                additional_metrics=request.performance_metrics
            )
            
            # Create storage info
            storage_info = ModelStorageInfo(
                model_path=f"/models/{unified_id}/model.joblib",
                model_format="joblib",
                model_size_mb=request.performance_metrics.get("model_size_mb", 0.0),
                storage_backend="local",
                checksum="",
                metadata={"unified_model_id": unified_id}
            )
            
            # Create model version
            try:
                from ai.mlops.domain.value_objects.model_value_objects import ModelVersion as SemanticVersion
            except ImportError:
                from anomaly_detection.domain.value_objects.model_value_objects import ModelVersion as SemanticVersion
            semantic_version = SemanticVersion(major=1, minor=0, patch=0)
            
            model_version = await self.mlops_model_service.create_model_version(
                model_id=mlops_model.id,
                detector_id=uuid.uuid4(),  # Create a detector ID for this version
                version=semantic_version,
                performance_metrics=performance_metrics,
                storage_info=storage_info,
                created_by=request.created_by,
                description=f"Initial version registered via unified registry",
                tags=request.tags or []
            )
            
            return {
                "model_id": str(mlops_model.id),
                "version_id": str(model_version.id),
                "name": mlops_model.name
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to register in MLOps system: {e}")
            return None
    
    async def get_model(self, model_id: str) -> Optional[UnifiedModelMetadata]:
        """Get unified model metadata by ID.
        
        Args:
            model_id: Unified model ID
            
        Returns:
            Unified model metadata or None
        """
        if model_id not in self._model_mappings:
            self.logger.warning(f"Model {model_id} not found in registry")
            return None
        
        mapping = self._model_mappings[model_id]
        
        try:
            # Get anomaly detection model data
            anomaly_data = await self._get_anomaly_model_data(mapping["anomaly_model_id"])
            
            # Get MLOps model data (if available)
            mlops_data = None
            if self.mlops_integration_enabled and mapping["mlops_model_id"]:
                mlops_data = await self._get_mlops_model_data(mapping["mlops_model_id"])
            
            # Merge data into unified metadata
            unified_metadata = self._merge_model_data(model_id, anomaly_data, mlops_data)
            
            return unified_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    async def _get_anomaly_model_data(self, model_id: str) -> Dict[str, Any]:
        """Get model data from anomaly detection system."""
        # Get model versions
        versions = self.anomaly_mlops_service.get_model_versions(model_id)
        
        if not versions:
            raise ValueError(f"No versions found for anomaly model {model_id}")
        
        latest_version = versions[-1]  # Get latest version
        
        return {
            "model_id": model_id,
            "versions": versions,
            "latest_version": latest_version,
            "performance_metrics": latest_version.performance_metrics,
            "deployment_config": latest_version.deployment_config
        }
    
    async def _get_mlops_model_data(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model data from MLOps system."""
        if not self.mlops_integration_enabled:
            return None
        
        try:
            model_uuid = uuid.UUID(model_id)
            model_data = await self.mlops_model_service.get_model_with_versions(model_uuid)
            return model_data
        except Exception as e:
            self.logger.warning(f"Failed to get MLOps model data: {e}")
            return None
    
    def _merge_model_data(
        self, 
        unified_id: str, 
        anomaly_data: Dict[str, Any], 
        mlops_data: Optional[Dict[str, Any]]
    ) -> UnifiedModelMetadata:
        """Merge data from both systems into unified metadata."""
        latest_version = anomaly_data["latest_version"]
        
        # Use MLOps data if available, otherwise fall back to anomaly data
        if mlops_data and mlops_data.get("model"):
            mlops_model = mlops_data["model"]
            name = mlops_model.get("name", "Unknown Model")
            description = mlops_model.get("description", "")
            algorithm = mlops_model.get("algorithm_family", "unknown")
            tags = mlops_model.get("tags", [])
            created_by = mlops_model.get("created_by", "system")
        else:
            name = anomaly_data["model_id"]
            description = f"Anomaly detection model: {anomaly_data['model_id']}"
            algorithm = latest_version.metadata.get("algorithm", "unknown")
            tags = []
            created_by = "system"
        
        return UnifiedModelMetadata(
            model_id=unified_id,
            name=name,
            description=description,
            algorithm=algorithm,
            framework="scikit-learn",
            version=str(latest_version.version),
            performance_metrics=latest_version.performance_metrics,
            training_data_info=latest_version.metadata.get("training_data", {}),
            deployment_config=latest_version.deployment_config or {},
            tags=tags,
            created_by=created_by,
            created_at=latest_version.created_at,
            anomaly_model_id=anomaly_data["model_id"],
            mlops_model_id=mlops_data.get("model", {}).get("id") if mlops_data else None
        )
    
    async def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[UnifiedModelMetadata]:
        """List all models in the unified registry.
        
        Args:
            filters: Optional filters (algorithm, framework, tags, etc.)
            
        Returns:
            List of unified model metadata
        """
        models = []
        
        for unified_id in self._model_mappings.keys():
            try:
                model = await self.get_model(unified_id)
                if model and self._matches_filters(model, filters):
                    models.append(model)
            except Exception as e:
                self.logger.warning(f"Failed to get model {unified_id}: {e}")
                continue
        
        return models
    
    def _matches_filters(self, model: UnifiedModelMetadata, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if model matches the provided filters."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == "algorithm" and model.algorithm != value:
                return False
            elif key == "framework" and model.framework != value:
                return False
            elif key == "tags" and not any(tag in model.tags for tag in value):
                return False
            elif key == "created_by" and model.created_by != value:
                return False
        
        return True
    
    async def promote_model(self, model_id: str, stage: str, promoted_by: str) -> UnifiedModelMetadata:
        """Promote model to a different stage in both systems.
        
        Args:
            model_id: Unified model ID
            stage: Target stage (development, staging, production, archived)
            promoted_by: User promoting the model
            
        Returns:
            Updated unified model metadata
        """
        if model_id not in self._model_mappings:
            raise ValueError(f"Model {model_id} not found in registry")
        
        mapping = self._model_mappings[model_id]
        
        try:
            # Promote in anomaly detection system
            anomaly_model_id = mapping["anomaly_model_id"]
            if stage == "production":
                # Get latest version and promote to production
                versions = self.anomaly_mlops_service.get_model_versions(anomaly_model_id)
                if versions:
                    latest_version = versions[-1]
                    self.anomaly_mlops_service.promote_model_version(
                        anomaly_model_id, 
                        latest_version.version, 
                        "production"
                    )
            
            # Promote in MLOps system (if available)
            if self.mlops_integration_enabled and mapping["mlops_model_id"]:
                mlops_model_id = uuid.UUID(mapping["mlops_model_id"])
                mlops_model_data = await self.mlops_model_service.get_model_with_versions(mlops_model_id)
                
                if mlops_model_data and mlops_model_data.get("versions"):
                    latest_version = mlops_model_data["versions"][-1]
                    await self.mlops_model_service.promote_to_production(
                        mlops_model_id,
                        uuid.UUID(latest_version["id"]),
                        promoted_by
                    )
            
            # Get updated model metadata
            updated_model = await self.get_model(model_id)
            
            self.logger.info(f"Successfully promoted model {model_id} to {stage}")
            return updated_model
            
        except Exception as e:
            self.logger.error(f"Failed to promote model {model_id} to {stage}: {e}")
            raise
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the unified registry.
        
        Returns:
            Registry statistics
        """
        total_models = len(self._model_mappings)
        anomaly_integrated = sum(1 for m in self._model_mappings.values() if m["anomaly_model_id"])
        mlops_integrated = sum(1 for m in self._model_mappings.values() if m["mlops_model_id"])
        
        return {
            "total_models": total_models,
            "anomaly_detection_integrated": anomaly_integrated,
            "mlops_integrated": mlops_integrated,
            "mlops_integration_enabled": self.mlops_integration_enabled,
            "registry_mappings": len(self._model_mappings)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the registry.
        
        Returns:
            Health status information
        """
        status = {
            "unified_registry": "healthy",
            "anomaly_detection_service": "unknown",
            "mlops_service": "unknown",
            "model_mappings": len(self._model_mappings),
            "last_check": datetime.now().isoformat()
        }
        
        try:
            # Check anomaly detection service
            if hasattr(self.anomaly_mlops_service, '_experiments'):
                status["anomaly_detection_service"] = "healthy"
        except Exception:
            status["anomaly_detection_service"] = "unhealthy"
        
        try:
            # Check MLOps service
            if self.mlops_integration_enabled and self.mlops_model_service:
                status["mlops_service"] = "healthy"
            else:
                status["mlops_service"] = "disabled"
        except Exception:
            status["mlops_service"] = "unhealthy"
        
        return status


# Global registry instance
_unified_model_registry: Optional[UnifiedModelRegistry] = None


def initialize_unified_model_registry(
    anomaly_mlops_service: MLOpsService,
    anomaly_model_repository: ModelRepository,
    mlops_model_service: Optional[ModelManagementService] = None
) -> UnifiedModelRegistry:
    """Initialize global unified model registry.
    
    Args:
        anomaly_mlops_service: Anomaly detection MLOps service
        anomaly_model_repository: Anomaly detection model repository
        mlops_model_service: Optional MLOps model management service
        
    Returns:
        Initialized unified model registry
    """
    global _unified_model_registry
    _unified_model_registry = UnifiedModelRegistry(
        anomaly_mlops_service=anomaly_mlops_service,
        anomaly_model_repository=anomaly_model_repository,
        mlops_model_service=mlops_model_service
    )
    return _unified_model_registry


def get_unified_model_registry() -> Optional[UnifiedModelRegistry]:
    """Get global unified model registry instance.
    
    Returns:
        Unified model registry instance or None
    """
    return _unified_model_registry