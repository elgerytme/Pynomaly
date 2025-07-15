"""Model Registry Service

High-level service for managing model lifecycle, storage, and promotion.
"""

import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from pynomaly_mlops.domain.entities.model import Model, ModelStatus, ModelType
from pynomaly_mlops.domain.repositories.model_repository import ModelRepository
from pynomaly_mlops.domain.services.model_promotion_service import (
    ModelPromotionService, PromotionCriteria, PromotionResult
)
from pynomaly_mlops.domain.value_objects.semantic_version import SemanticVersion
from pynomaly_mlops.domain.value_objects.model_metrics import ModelMetrics
from pynomaly_mlops.infrastructure.storage.artifact_storage import ArtifactStorageService


class ModelRegistryService:
    """Application service for model registry operations."""
    
    def __init__(
        self,
        model_repository: ModelRepository,
        artifact_storage: ArtifactStorageService,
        promotion_service: ModelPromotionService
    ):
        """Initialize model registry service.
        
        Args:
            model_repository: Repository for model persistence
            artifact_storage: Service for storing model artifacts
            promotion_service: Service for model promotion logic
        """
        self.model_repository = model_repository
        self.artifact_storage = artifact_storage
        self.promotion_service = promotion_service
    
    async def register_model(
        self,
        name: str,
        version: SemanticVersion,
        model_type: ModelType,
        model_artifact: Any,
        description: Optional[str] = None,
        created_by: str = "system",
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        parent_model_id: Optional[UUID] = None,
        experiment_id: Optional[UUID] = None
    ) -> Model:
        """Register a new model with the registry.
        
        Args:
            name: Model name
            version: Model version
            model_type: Type of the model
            model_artifact: Trained model object
            description: Model description
            created_by: Creator of the model
            hyperparameters: Model hyperparameters
            training_config: Training configuration
            parent_model_id: Parent model for lineage tracking
            experiment_id: Associated experiment
            
        Returns:
            Registered model entity
            
        Raises:
            ValueError: If model with same name/version already exists
        """
        # Check if model already exists
        existing_model = await self.model_repository.get_by_name_and_version(name, version)
        if existing_model:
            raise ValueError(f"Model {name} version {version} already exists")
        
        # Create model entity
        model = Model(
            name=name,
            version=version,
            model_type=model_type,
            description=description,
            created_by=created_by,
            hyperparameters=hyperparameters or {},
            training_config=training_config,
            parent_model_id=parent_model_id,
            experiment_id=experiment_id
        )
        
        # Store model artifact
        artifact_uri = await self.artifact_storage.store_model(model, model_artifact)
        
        # Get artifact info and update model
        artifact_info = await self.artifact_storage.get_model_info(artifact_uri)
        if artifact_info:
            model.artifact_uri = artifact_uri
            model.size_bytes = artifact_info.get('size_bytes')
            model.checksum = artifact_info.get('checksum')
        
        # Save model to repository
        return await self.model_repository.save(model)
    
    async def update_model_metrics(
        self,
        model_id: UUID,
        metrics: ModelMetrics,
        validation_metrics: Optional[ModelMetrics] = None
    ) -> Model:
        """Update model performance metrics.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics
            validation_metrics: Validation metrics
            
        Returns:
            Updated model entity
            
        Raises:
            ValueError: If model not found
        """
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Update metrics
        model.metrics = metrics
        if validation_metrics:
            model.validation_metrics = validation_metrics
        
        # Update timestamp
        model.updated_at = datetime.utcnow()
        
        return await self.model_repository.save(model)
    
    async def promote_model(
        self,
        model_id: UUID,
        target_status: ModelStatus,
        criteria: Optional[PromotionCriteria] = None,
        promoted_by: str = "system"
    ) -> Tuple[bool, PromotionResult]:
        """Promote model to a new status.
        
        Args:
            model_id: Model identifier
            target_status: Target status for promotion
            criteria: Promotion criteria (optional)
            promoted_by: User promoting the model
            
        Returns:
            Tuple of (success, promotion_result)
            
        Raises:
            ValueError: If model not found or promotion invalid
        """
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Get current production model for comparison
        current_production_model = None
        if target_status == ModelStatus.PRODUCTION:
            production_models = await self.model_repository.get_production_models()
            current_production_model = next(
                (m for m in production_models if m.name == model.name),
                None
            )
        
        # Evaluate promotion
        promotion_result = self.promotion_service.evaluate_promotion(
            candidate_model=model,
            target_status=target_status,
            current_production_model=current_production_model,
            criteria=criteria
        )
        
        if promotion_result.approved:
            # Update model status
            model.promote_to_status(target_status, promoted_by)
            await self.model_repository.save(model)
            
            # If promoting to production, demote current production model
            if target_status == ModelStatus.PRODUCTION and current_production_model:
                current_production_model.promote_to_status(ModelStatus.DEPRECATED, promoted_by)
                await self.model_repository.save(current_production_model)
        
        return promotion_result.approved, promotion_result
    
    async def get_model(self, model_id: UUID) -> Optional[Model]:
        """Get model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model entity if found
        """
        return await self.model_repository.get_by_id(model_id)
    
    async def get_model_by_name_version(
        self,
        name: str,
        version: SemanticVersion
    ) -> Optional[Model]:
        """Get model by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Model entity if found
        """
        return await self.model_repository.get_by_name_and_version(name, version)
    
    async def get_latest_model(self, name: str) -> Optional[Model]:
        """Get latest version of a model.
        
        Args:
            name: Model name
            
        Returns:
            Latest model version if found
        """
        return await self.model_repository.get_latest_by_name(name)
    
    async def list_models(
        self,
        name_pattern: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        model_type: Optional[ModelType] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Model]:
        """List models with filtering.
        
        Args:
            name_pattern: Pattern to match model names
            status: Filter by status
            model_type: Filter by model type
            created_by: Filter by creator
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching models
        """
        return await self.model_repository.search(
            name_pattern=name_pattern,
            status=status,
            model_type=model_type,
            created_by=created_by,
            limit=limit,
            offset=offset
        )
    
    async def get_production_models(self) -> List[Model]:
        """Get all models currently in production.
        
        Returns:
            List of production models
        """
        return await self.model_repository.get_production_models()
    
    async def get_model_versions(self, name: str) -> List[Model]:
        """Get all versions of a model.
        
        Args:
            name: Model name
            
        Returns:
            List of model versions, ordered by version descending
        """
        return await self.model_repository.list_by_name(name)
    
    async def get_model_lineage(self, model_id: UUID) -> List[Model]:
        """Get model lineage (ancestors and descendants).
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of models in lineage chain
        """
        return await self.model_repository.get_model_lineage(model_id)
    
    async def load_model_artifact(self, model_id: UUID) -> Any:
        """Load model artifact for inference.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model not found or no artifact
            FileNotFoundError: If artifact not found in storage
        """
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if not model.artifact_uri:
            raise ValueError(f"Model {model_id} has no stored artifact")
        
        return await self.artifact_storage.load_model(model.artifact_uri)
    
    async def delete_model(self, model_id: UUID) -> bool:
        """Delete a model and its artifact.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False if not found
        """
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            return False
        
        # Delete artifact if exists
        if model.artifact_uri:
            try:
                await self.artifact_storage.delete_model(model.artifact_uri)
            except Exception:
                # Continue with model deletion even if artifact deletion fails
                pass
        
        # Delete model from repository
        return await self.model_repository.delete(model_id)
    
    async def compare_models(
        self,
        model_ids: List[UUID],
        metrics_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models by their metrics.
        
        Args:
            model_ids: List of model identifiers to compare
            metrics_keys: Specific metrics to compare (optional)
            
        Returns:
            Comparison results with model performance data
        """
        models = []
        for model_id in model_ids:
            model = await self.model_repository.get_by_id(model_id)
            if model:
                models.append(model)
        
        if not models:
            return {}
        
        comparison = {
            'models': [],
            'metrics_comparison': {},
            'best_model': None
        }
        
        # Collect model data
        for model in models:
            model_data = {
                'id': str(model.id),
                'name': model.name,
                'version': str(model.version),
                'status': model.status.value,
                'metrics': {}
            }
            
            if model.metrics:
                if metrics_keys:
                    # Only include specified metrics
                    for key in metrics_keys:
                        value = getattr(model.metrics, key, None)
                        if value is not None:
                            model_data['metrics'][key] = value
                else:
                    # Include all available metrics
                    model_data['metrics'] = model.metrics.to_dict()
            
            comparison['models'].append(model_data)
        
        # Calculate metrics comparison
        if metrics_keys:
            for key in metrics_keys:
                values = []
                for model_data in comparison['models']:
                    if key in model_data['metrics']:
                        values.append(model_data['metrics'][key])
                
                if values:
                    comparison['metrics_comparison'][key] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'range': max(values) - min(values)
                    }
        
        # Determine best model (highest accuracy or F1 score)
        best_score = -1
        best_model_data = None
        
        for model_data in comparison['models']:
            metrics = model_data['metrics']
            # Try accuracy first, then F1 score
            score = metrics.get('accuracy') or metrics.get('f1_score') or 0
            if score > best_score:
                best_score = score
                best_model_data = model_data
        
        if best_model_data:
            comparison['best_model'] = {
                'id': best_model_data['id'],
                'name': best_model_data['name'],
                'version': best_model_data['version'],
                'score': best_score
            }
        
        return comparison
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with model statistics
        """
        # Get counts by status
        status_counts = {}
        for status in ModelStatus:
            count = await self.model_repository.count({'status': status.value})
            status_counts[status.value] = count
        
        # Get counts by type
        type_counts = {}
        for model_type in ModelType:
            count = await self.model_repository.count({'model_type': model_type.value})
            type_counts[model_type.value] = count
        
        # Get total count
        total_count = await self.model_repository.count()
        
        return {
            'total_models': total_count,
            'by_status': status_counts,
            'by_type': type_counts,
            'production_models': status_counts.get('production', 0)
        }