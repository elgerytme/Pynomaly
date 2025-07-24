"""Domain interfaces for model registry operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path


class ModelStatus(Enum):
    """Model lifecycle status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentStage(Enum):
    """Model deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    SHADOW = "shadow"


class ModelFramework(Enum):
    """Supported ML frameworks."""
    SCIKIT_LEARN = "scikit-learn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    OTHER = "other"


@dataclass
class ModelMetadata:
    """Complete model metadata."""
    model_id: str
    version: str
    algorithm: str
    framework: ModelFramework
    created_by: str
    description: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    training_data_info: Dict[str, Any]
    feature_schema: Dict[str, Any]
    model_signature: Dict[str, Any]
    dependencies: List[str]
    custom_metadata: Dict[str, Any]


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    training_time: Optional[float] = None
    inference_latency: Optional[float] = None
    memory_usage: Optional[float] = None
    model_size_bytes: Optional[int] = None
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class ModelInfo:
    """Complete model information."""
    model_id: str
    version: str
    status: ModelStatus
    metadata: ModelMetadata
    metrics: ModelMetrics
    created_at: datetime
    updated_at: datetime
    model_path: str
    model_checksum: str
    experiment_id: Optional[str] = None
    deployment_info: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentConfig:
    """Model deployment configuration."""
    stage: DeploymentStage
    replicas: int = 1
    resources: Dict[str, str] = None
    auto_scaling: Dict[str, Any] = None
    environment_variables: Dict[str, str] = None
    traffic_percentage: float = 100.0
    health_check_config: Dict[str, Any] = None
    monitoring_config: Dict[str, Any] = None


@dataclass
class DeploymentInfo:
    """Model deployment information."""
    deployment_id: str
    model_id: str
    model_version: str
    stage: DeploymentStage
    endpoint_url: Optional[str]
    replicas: int
    resources: Dict[str, str]
    auto_scaling: Dict[str, Any]
    traffic_percentage: float
    deployment_time: datetime
    health_status: str
    metrics: Dict[str, Any]


@dataclass
class ModelRegistrationRequest:
    """Request for registering a new model."""
    model_path: str
    metadata: ModelMetadata
    metrics: ModelMetrics
    experiment_id: Optional[str] = None
    validate_model: bool = True


@dataclass
class ModelValidationResult:
    """Result of model validation."""
    is_valid: bool
    validation_checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    performance_metrics: Optional[ModelMetrics] = None


@dataclass
class ModelSearchQuery:
    """Query for searching models."""
    model_ids: Optional[List[str]] = None
    status: Optional[ModelStatus] = None
    algorithm: Optional[str] = None
    framework: Optional[ModelFramework] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_accuracy: Optional[float] = None
    text_query: Optional[str] = None


class ModelRegistryPort(ABC):
    """Port for model registry operations."""

    @abstractmethod
    async def register_model(self, request: ModelRegistrationRequest) -> str:
        """Register a new model in the registry.
        
        Args:
            request: Model registration request
            
        Returns:
            Model identifier (model_id:version)
        """
        pass

    @abstractmethod
    async def get_model(
        self, 
        model_id: str, 
        version: str = "latest"
    ) -> Optional[ModelInfo]:
        """Get model information from registry.
        
        Args:
            model_id: ID of the model
            version: Version of the model (default: latest)
            
        Returns:
            Model information or None if not found
        """
        pass

    @abstractmethod
    async def list_models(
        self,
        query: Optional[ModelSearchQuery] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ModelInfo]:
        """List models with optional filters.
        
        Args:
            query: Search query and filters
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of model information
        """
        pass

    @abstractmethod
    async def update_model_metadata(
        self,
        model_id: str,
        version: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update model metadata.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            metadata_updates: Fields to update
            
        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: str, version: str) -> bool:
        """Delete a model from the registry.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            
        Returns:
            True if deletion successful
        """
        pass


class ModelLifecyclePort(ABC):
    """Port for model lifecycle management operations."""

    @abstractmethod
    async def promote_model(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStatus
    ) -> bool:
        """Promote model to different lifecycle stage.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            target_stage: Target lifecycle stage
            
        Returns:
            True if promotion successful
        """
        pass

    @abstractmethod
    async def validate_model(
        self,
        model_id: str,
        version: str,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ModelValidationResult:
        """Validate a model for promotion.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            validation_config: Optional validation configuration
            
        Returns:
            Validation result
        """
        pass

    @abstractmethod
    async def check_promotion_eligibility(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStatus
    ) -> Dict[str, Any]:
        """Check if model is eligible for promotion.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            target_stage: Target stage to check
            
        Returns:
            Eligibility result with requirements status
        """
        pass

    @abstractmethod
    async def deprecate_model(
        self,
        model_id: str,
        version: str,
        reason: str
    ) -> bool:
        """Deprecate a model.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            reason: Reason for deprecation
            
        Returns:
            True if deprecation successful
        """
        pass

    @abstractmethod
    async def archive_model(
        self,
        model_id: str,
        version: str,
        retain_metadata: bool = True
    ) -> bool:
        """Archive a model.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            retain_metadata: Whether to retain metadata
            
        Returns:
            True if archival successful
        """
        pass


class ModelDeploymentPort(ABC):
    """Port for model deployment operations."""

    @abstractmethod
    async def deploy_model(
        self,
        model_id: str,
        version: str,
        config: DeploymentConfig
    ) -> Optional[str]:
        """Deploy model to specified environment.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            config: Deployment configuration
            
        Returns:
            Deployment ID if successful, None otherwise
        """
        pass

    @abstractmethod
    async def get_deployment(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Get deployment information.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Deployment information or None if not found
        """
        pass

    @abstractmethod
    async def list_deployments(
        self,
        model_id: Optional[str] = None,
        stage: Optional[DeploymentStage] = None,
        status: Optional[str] = None
    ) -> List[DeploymentInfo]:
        """List deployments with optional filters.
        
        Args:
            model_id: Filter by model ID
            stage: Filter by deployment stage
            status: Filter by health status
            
        Returns:
            List of deployment information
        """
        pass

    @abstractmethod
    async def update_deployment(
        self,
        deployment_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update deployment configuration.
        
        Args:
            deployment_id: ID of the deployment
            updates: Configuration updates
            
        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def scale_deployment(
        self,
        deployment_id: str,
        replicas: int
    ) -> bool:
        """Scale deployment to specified number of replicas.
        
        Args:
            deployment_id: ID of the deployment
            replicas: Target number of replicas
            
        Returns:
            True if scaling successful
        """
        pass

    @abstractmethod
    async def undeploy_model(self, deployment_id: str) -> bool:
        """Remove model deployment.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            True if undeployment successful
        """
        pass


class ModelStoragePort(ABC):
    """Port for model storage operations."""

    @abstractmethod
    async def store_model(
        self,
        model_path: str,
        model_id: str,
        version: str,
        metadata: ModelMetadata
    ) -> str:
        """Store model artifacts.
        
        Args:
            model_path: Local path to model file
            model_id: ID of the model
            version: Version of the model
            metadata: Model metadata
            
        Returns:
            Storage path where model was stored
        """
        pass

    @abstractmethod
    async def retrieve_model(
        self,
        model_id: str,
        version: str,
        download_path: str
    ) -> bool:
        """Retrieve model from storage.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            download_path: Local path to download to
            
        Returns:
            True if retrieval successful
        """
        pass

    @abstractmethod
    async def verify_model_integrity(
        self,
        model_id: str,
        version: str
    ) -> bool:
        """Verify model file integrity.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            
        Returns:
            True if model integrity is valid
        """
        pass

    @abstractmethod
    async def get_model_info(
        self,
        model_id: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Get storage information about a model.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            
        Returns:
            Storage information or None if not found
        """
        pass

    @abstractmethod
    async def delete_model_artifacts(
        self,
        model_id: str,
        version: str
    ) -> bool:
        """Delete model artifacts from storage.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            
        Returns:
            True if deletion successful
        """
        pass


class ModelVersioningPort(ABC):
    """Port for model versioning operations."""

    @abstractmethod
    async def create_version(
        self,
        model_id: str,
        parent_version: Optional[str] = None,
        version_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new model version.
        
        Args:
            model_id: ID of the model
            parent_version: Parent version (for branching)
            version_metadata: Optional version metadata
            
        Returns:
            New version identifier
        """
        pass

    @abstractmethod
    async def list_versions(
        self,
        model_id: str,
        include_archived: bool = False
    ) -> List[str]:
        """List all versions of a model.
        
        Args:
            model_id: ID of the model
            include_archived: Whether to include archived versions
            
        Returns:
            List of version identifiers
        """
        pass

    @abstractmethod
    async def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            model_id: ID of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        pass

    @abstractmethod
    async def get_version_lineage(
        self,
        model_id: str,
        version: str
    ) -> Dict[str, Any]:
        """Get version lineage and history.
        
        Args:
            model_id: ID of the model
            version: Version to get lineage for
            
        Returns:
            Version lineage information
        """
        pass

    @abstractmethod
    async def tag_version(
        self,
        model_id: str,
        version: str,
        tag: str
    ) -> bool:
        """Tag a model version.
        
        Args:
            model_id: ID of the model
            version: Version to tag
            tag: Tag name
            
        Returns:
            True if tagging successful
        """
        pass


class ModelSearchPort(ABC):
    """Port for model search and discovery operations."""

    @abstractmethod
    async def search_models(
        self,
        query: ModelSearchQuery,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0
    ) -> List[ModelInfo]:
        """Search models with advanced query.
        
        Args:
            query: Search query with filters
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of matching models
        """
        pass

    @abstractmethod
    async def find_similar_models(
        self,
        model_id: str,
        version: str,
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[ModelInfo]:
        """Find models similar to the given model.
        
        Args:
            model_id: Reference model ID
            version: Reference model version
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            
        Returns:
            List of similar models
        """
        pass

    @abstractmethod
    async def recommend_models(
        self,
        use_case: str,
        data_characteristics: Dict[str, Any],
        performance_requirements: Dict[str, float],
        limit: int = 5
    ) -> List[ModelInfo]:
        """Recommend models for a use case.
        
        Args:
            use_case: Description of the use case
            data_characteristics: Data characteristics
            performance_requirements: Performance requirements
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended models
        """
        pass

    @abstractmethod
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Registry statistics
        """
        pass