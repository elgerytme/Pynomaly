"""MLOps Model Registry Adapter.

This adapter implements the MLOpsModelRegistryPort interface by integrating
with the mlops package. It translates between anomaly detection domain concepts
and the MLOps package's model registry APIs.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from anomaly_detection.domain.interfaces.mlops_operations import (
    MLOpsModelRegistryPort,
    ModelVersionMetadata,
    ModelStage,
    ModelRegistrationError,
    VersionCreationError,
    VersionRetrievalError,
    VersionQueryError,
    StageTransitionError,
)

# MLOps package imports
try:
    from mlops.domain.services.model_management_service import ModelManagementService
    from mlops.domain.entities.model import Model as MLOpsModel
    from mlops.domain.entities.model_version import ModelVersion as MLOpsModelVersion
    from mlops.domain.value_objects.model_value_objects import ModelType, PerformanceMetrics, ModelStorageInfo
    from mlops.application.use_cases.create_model_use_case import CreateModelUseCase
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    # Create type stubs when MLOps package is not available
    ModelManagementService = Any
    MLOpsModel = Any
    MLOpsModelVersion = Any
    ModelType = Any
    PerformanceMetrics = Any
    ModelStorageInfo = Any
    CreateModelUseCase = Any


class MLOpsModelRegistryAdapter(MLOpsModelRegistryPort):
    """Adapter for MLOps model registry operations.
    
    This adapter integrates the anomaly detection domain with the
    MLOps package, providing model registration and versioning capabilities.
    """

    def __init__(
        self,
        model_management_service: ModelManagementService,
        create_model_use_case: CreateModelUseCase,
    ):
        """Initialize the MLOps model registry adapter.
        
        Args:
            model_management_service: Model management service from MLOps package
            create_model_use_case: Create model use case implementation
        """
        if not MLOPS_AVAILABLE:
            raise ImportError(
                "mlops package is not available. "
                "Please install it to use this adapter."
            )
        
        self._model_service = model_management_service
        self._create_model_use_case = create_model_use_case
        self._logger = logging.getLogger(__name__)
        
        # Stage mapping from domain to MLOps format
        self._stage_mapping = {
            ModelStage.DEVELOPMENT: "development",
            ModelStage.STAGING: "staging",
            ModelStage.PRODUCTION: "production",
            ModelStage.ARCHIVED: "archived",
        }
        
        self._reverse_stage_mapping = {
            v: k for k, v in self._stage_mapping.items()
        }
        
        self._logger.info("MLOpsModelRegistryAdapter initialized successfully")

    async def register_model(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Register a new model in the registry.
        
        Args:
            name: Model name
            description: Model description
            tags: Optional tags
            created_by: User registering the model
            
        Returns:
            Unique model identifier
            
        Raises:
            ModelRegistrationError: If registration fails
        """
        try:
            # Create model request for MLOps package
            model_request = {
                "name": name,
                "description": description,
                "model_type": ModelType.ANOMALY_DETECTION,
                "algorithm_family": "anomaly_detection",
                "created_by": created_by,
                "use_cases": ["anomaly_detection", "outlier_detection"],
                "data_requirements": {
                    "input_format": "tabular",
                    "feature_types": ["numerical", "categorical"],
                    "data_preprocessing": "standardization"
                },
                "tags": tags or {},
                "metadata": {
                    "source": "anomaly_detection_package",
                    "registered_at": datetime.now().isoformat(),
                }
            }
            
            self._logger.info(f"Registering model: {name}")
            
            # Register model through MLOps package
            mlops_model = await self._create_model_use_case.execute(model_request)
            
            model_id = str(mlops_model.id)
            
            self._logger.info(f"Model registered successfully with ID: {model_id}")
            
            return model_id
            
        except Exception as e:
            self._logger.error(f"Failed to register model '{name}': {str(e)}")
            raise ModelRegistrationError(f"Model registration failed: {str(e)}") from e

    async def create_model_version(
        self,
        model_id: str,
        version: str,
        run_id: str,
        source_path: str,
        description: str = "",
        performance_metrics: Optional[Dict[str, float]] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new version of a model.
        
        Args:
            model_id: Model identifier
            version: Version string
            run_id: Associated run identifier
            source_path: Path to model artifacts
            description: Version description
            performance_metrics: Model performance metrics
            deployment_config: Deployment configuration
            tags: Optional tags
            created_by: User creating the version
            
        Returns:
            Unique version identifier
            
        Raises:
            VersionCreationError: If version creation fails
        """
        try:
            # Create performance metrics object
            perf_metrics = None
            if performance_metrics:
                perf_metrics = PerformanceMetrics(
                    accuracy=performance_metrics.get("accuracy", 0.0),
                    precision=performance_metrics.get("precision", 0.0),
                    recall=performance_metrics.get("recall", 0.0),
                    f1_score=performance_metrics.get("f1_score", 0.0),
                    training_time=performance_metrics.get("training_time", 0.0),
                    inference_time=performance_metrics.get("inference_time", 0.0),
                    model_size_mb=performance_metrics.get("model_size_mb", 0.0),
                    additional_metrics=performance_metrics
                )
            
            # Create storage info object
            storage_info = ModelStorageInfo(
                model_path=source_path,
                model_format="joblib",  # Default format for anomaly detection models
                model_size_mb=performance_metrics.get("model_size_mb", 0.0) if performance_metrics else 0.0,
                storage_backend="local",
                checksum="",  # Would be calculated in production
                metadata={
                    "created_by": created_by,
                    "source": "anomaly_detection_package",
                }
            )
            
            # Parse version string to semantic version
            from mlops.domain.value_objects.model_value_objects import ModelVersion as SemanticVersion
            try:
                version_parts = version.split(".")
                major = int(version_parts[0]) if len(version_parts) > 0 else 1
                minor = int(version_parts[1]) if len(version_parts) > 1 else 0
                patch = int(version_parts[2]) if len(version_parts) > 2 else 0
                semantic_version = SemanticVersion(major=major, minor=minor, patch=patch)
            except (ValueError, IndexError):
                # Fallback to default versioning
                semantic_version = SemanticVersion(major=1, minor=0, patch=0)
            
            self._logger.info(f"Creating version {version} for model {model_id}")
            
            # Create model version through MLOps package
            model_version = await self._model_service.create_model_version(
                model_id=uuid.UUID(model_id),
                detector_id=uuid.UUID(run_id),
                version=semantic_version,
                performance_metrics=perf_metrics,
                storage_info=storage_info,
                created_by=created_by,
                description=description,
                tags=tags or []
            )
            
            version_id = str(model_version.id)
            
            self._logger.info(f"Model version created successfully with ID: {version_id}")
            
            return version_id
            
        except Exception as e:
            self._logger.error(f"Failed to create version {version} for model {model_id}: {str(e)}")
            raise VersionCreationError(f"Version creation failed: {str(e)}") from e

    async def get_model_version(
        self, 
        model_id: str, 
        version: str
    ) -> Optional[ModelVersionMetadata]:
        """Retrieve model version metadata.
        
        Args:
            model_id: Model identifier
            version: Version string
            
        Returns:
            Model version metadata if found, None otherwise
            
        Raises:
            VersionRetrievalError: If retrieval fails
        """
        try:
            # Get model version from MLOps package
            mlops_version = await self._model_service.get_model_version_by_name(
                uuid.UUID(model_id), version
            )
            
            if not mlops_version:
                return None
            
            # Convert to domain format
            version_metadata = self._convert_version_to_domain_format(mlops_version)
            
            return version_metadata
            
        except Exception as e:
            self._logger.error(f"Failed to retrieve version {version} for model {model_id}: {str(e)}")
            raise VersionRetrievalError(f"Version retrieval failed: {str(e)}") from e

    async def list_model_versions(
        self, 
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersionMetadata]:
        """List versions of a model.
        
        Args:
            model_id: Model identifier
            stage: Optional stage filter
            
        Returns:
            List of model version metadata
            
        Raises:
            VersionQueryError: If listing fails
        """
        try:
            # Convert stage filter to MLOps format
            mlops_stage = None
            if stage:
                mlops_stage = self._stage_mapping.get(stage)
            
            # List model versions through MLOps package
            mlops_versions = await self._model_service.list_model_versions(
                uuid.UUID(model_id), stage_filter=mlops_stage
            )
            
            # Convert to domain format
            versions = [
                self._convert_version_to_domain_format(version)
                for version in mlops_versions
            ]
            
            self._logger.info(f"Retrieved {len(versions)} versions for model {model_id}")
            
            return versions
            
        except Exception as e:
            self._logger.error(f"Failed to list versions for model {model_id}: {str(e)}")
            raise VersionQueryError(f"Version listing failed: {str(e)}") from e

    async def transition_model_stage(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
        comment: str = "",
        archive_existing: bool = True
    ) -> None:
        """Transition model version to a different stage.
        
        Args:
            model_id: Model identifier
            version: Version string
            stage: Target stage
            comment: Optional comment
            archive_existing: Whether to archive existing model in target stage
            
        Raises:
            StageTransitionError: If transition fails
        """
        try:
            # Convert stage to MLOps format
            mlops_stage = self._stage_mapping.get(stage)
            if not mlops_stage:
                raise ValueError(f"Unsupported stage: {stage}")
            
            # Get the model version first
            model_version = await self.get_model_version(model_id, version)
            if not model_version:
                raise ValueError(f"Model version {version} not found for model {model_id}")
            
            self._logger.info(
                f"Transitioning model {model_id} version {version} to stage {stage.value}"
            )
            
            # Transition stage through MLOps package
            if stage == ModelStage.PRODUCTION:
                await self._model_service.promote_to_production(
                    uuid.UUID(model_id),
                    uuid.UUID(model_version.version_id),
                    promoted_by=comment or "system"
                )
            else:
                # For other stages, use generic stage transition
                await self._model_service.transition_model_stage(
                    uuid.UUID(model_id),
                    uuid.UUID(model_version.version_id),
                    mlops_stage,
                    comment,
                    archive_existing
                )
            
            self._logger.info(f"Stage transition completed successfully")
            
        except Exception as e:
            self._logger.error(
                f"Failed to transition model {model_id} version {version} to stage {stage.value}: {str(e)}"
            )
            raise StageTransitionError(f"Stage transition failed: {str(e)}") from e

    async def get_latest_model_version(
        self, 
        model_id: str, 
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersionMetadata]:
        """Get the latest version of a model in a specific stage.
        
        Args:
            model_id: Model identifier
            stage: Optional stage filter
            
        Returns:
            Latest model version metadata if found, None otherwise
            
        Raises:
            VersionRetrievalError: If retrieval fails
        """
        try:
            # List all versions and find the latest
            versions = await self.list_model_versions(model_id, stage)
            
            if not versions:
                return None
            
            # Sort by creation date and return the latest
            latest_version = max(versions, key=lambda v: v.created_at)
            
            return latest_version
            
        except Exception as e:
            self._logger.error(f"Failed to get latest version for model {model_id}: {str(e)}")
            raise VersionRetrievalError(f"Latest version retrieval failed: {str(e)}") from e

    def _convert_version_to_domain_format(self, mlops_version: MLOpsModelVersion) -> ModelVersionMetadata:
        """Convert MLOps model version to domain format."""
        # Convert performance metrics
        performance_metrics = {}
        if hasattr(mlops_version, 'performance_metrics') and mlops_version.performance_metrics:
            perf = mlops_version.performance_metrics
            performance_metrics = {
                "accuracy": getattr(perf, 'accuracy', 0.0),
                "precision": getattr(perf, 'precision', 0.0),
                "recall": getattr(perf, 'recall', 0.0),
                "f1_score": getattr(perf, 'f1_score', 0.0),
                "training_time": getattr(perf, 'training_time', 0.0),
                "inference_time": getattr(perf, 'inference_time', 0.0),
                "model_size_mb": getattr(perf, 'model_size_mb', 0.0),
            }
            if hasattr(perf, 'additional_metrics'):
                performance_metrics.update(perf.additional_metrics)
        
        # Convert deployment configuration
        deployment_config = {}
        if hasattr(mlops_version, 'deployment_config'):
            deployment_config = mlops_version.deployment_config or {}
        
        # Convert stage
        stage = ModelStage.DEVELOPMENT
        if hasattr(mlops_version, 'stage'):
            stage = self._reverse_stage_mapping.get(
                mlops_version.stage, ModelStage.DEVELOPMENT
            )
        
        # Get source path
        source_path = ""
        if hasattr(mlops_version, 'storage_info') and mlops_version.storage_info:
            source_path = mlops_version.storage_info.model_path
        
        return ModelVersionMetadata(
            version_id=str(mlops_version.id),
            model_id=str(mlops_version.model_id),
            version=str(mlops_version.version),
            stage=stage,
            created_at=mlops_version.created_at,
            created_by=mlops_version.created_by,
            description=getattr(mlops_version, 'description', ''),
            run_id=str(getattr(mlops_version, 'detector_id', '')),
            source_path=source_path,
            performance_metrics=performance_metrics,
            deployment_config=deployment_config,
            tags=getattr(mlops_version, 'tags', {}),
        )