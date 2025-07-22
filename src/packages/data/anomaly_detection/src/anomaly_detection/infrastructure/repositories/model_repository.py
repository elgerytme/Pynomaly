"""Repository for managing anomaly detection model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...domain.entities.model import Model, ModelMetadata, ModelStatus, SerializationFormat


class ModelRepository:
    """Repository for storing and retrieving trained models."""
    
    def __init__(self, storage_path: str | Path = "models"):
        """Initialize model repository.
        
        Args:
            storage_path: Base directory for storing models
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "models").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        
        # Registry file to track all models
        self.registry_file = self.storage_path / "model_registry.json"
        self._ensure_registry()
    
    def _ensure_registry(self) -> None:
        """Ensure model registry file exists."""
        if not self.registry_file.exists():
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump({'models': {}}, f, indent=2)
    
    def save(self, model: Model, format: SerializationFormat = SerializationFormat.PICKLE) -> str:
        """Save a model to the repository.
        
        Args:
            model: Model to save
            format: Serialization format
            
        Returns:
            Model ID of the saved model
        """
        model_id = model.metadata.model_id
        
        # Create model-specific directory
        model_dir = self.storage_path / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        if format == SerializationFormat.PICKLE:
            model_file = model_dir / f"{model_id}.pkl"
        elif format == SerializationFormat.JOBLIB:
            model_file = model_dir / f"{model_id}.joblib"
        else:
            model_file = model_dir / f"{model_id}.json"
        
        model.save(model_file, format)
        
        # Save metadata separately for quick access
        metadata_file = self.storage_path / "metadata" / f"{model_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self._metadata_to_dict(model.metadata), f, indent=2)
        
        # Update registry
        self._update_registry(model_id, {
            'name': model.metadata.name,
            'algorithm': model.metadata.algorithm,
            'status': model.metadata.status.value,
            'created_at': model.metadata.created_at.isoformat(),
            'updated_at': model.metadata.updated_at.isoformat(),
            'model_file': str(model_file),
            'metadata_file': str(metadata_file),
            'format': format.value
        })
        
        return model_id
    
    def load(self, model_id: str) -> Model:
        """Load a model from the repository.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If model not found
        """
        registry = self._load_registry()
        
        if model_id not in registry['models']:
            raise FileNotFoundError(f"Model with ID '{model_id}' not found in repository")
        
        model_info = registry['models'][model_id]
        model_file = Path(model_info['model_file'])
        
        return Model.load(model_file)
    
    def delete(self, model_id: str) -> bool:
        """Delete a model from the repository.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if deleted, False if not found
        """
        registry = self._load_registry()
        
        if model_id not in registry['models']:
            return False
        
        model_info = registry['models'][model_id]
        
        # Delete model file
        model_file = Path(model_info['model_file'])
        if model_file.exists():
            model_file.unlink()
        
        # Delete metadata file
        metadata_file = Path(model_info['metadata_file'])
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Delete model directory if empty
        model_dir = model_file.parent
        try:
            model_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue
        
        # Update registry
        del registry['models'][model_id]
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        
        return True
    
    def list_models(self, status: Optional[ModelStatus] = None, 
                   algorithm: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List models in the repository with optional filtering.
        
        Args:
            status: Filter by model status
            algorithm: Filter by algorithm
            tags: Filter by tags (model must have all specified tags)
            
        Returns:
            List of model summaries
        """
        registry = self._load_registry()
        models = []
        
        for model_id, model_info in registry['models'].items():
            # Load metadata for detailed filtering
            try:
                metadata = self._load_model_metadata(model_id)
                
                # Apply filters
                if status and ModelStatus(model_info['status']) != status:
                    continue
                
                if algorithm and model_info['algorithm'] != algorithm:
                    continue
                
                if tags:
                    model_tags = metadata.get('tags', [])
                    if not all(tag in model_tags for tag in tags):
                        continue
                
                # Create model summary
                summary = {
                    'model_id': model_id,
                    'name': model_info['name'],
                    'algorithm': model_info['algorithm'],
                    'status': model_info['status'],
                    'created_at': model_info['created_at'],
                    'updated_at': model_info['updated_at'],
                    'accuracy': metadata.get('accuracy'),
                    'tags': metadata.get('tags', []),
                    'description': metadata.get('description', ''),
                    'training_samples': metadata.get('training_samples')
                }
                
                models.append(summary)
                
            except Exception:
                # Skip models with corrupted metadata
                continue
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get metadata for a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model metadata dictionary
            
        Raises:
            FileNotFoundError: If model not found
        """
        registry = self._load_registry()
        
        if model_id not in registry['models']:
            raise FileNotFoundError(f"Model with ID '{model_id}' not found in repository")
        
        return self._load_model_metadata(model_id)
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update the status of a model.
        
        Args:
            model_id: ID of the model
            status: New status
            
        Returns:
            True if updated, False if model not found
        """
        try:
            model = self.load(model_id)
            model.metadata.status = status
            model.metadata.updated_at = datetime.utcnow()
            
            # Determine format from registry
            registry = self._load_registry()
            format_str = registry['models'][model_id]['format']
            format = SerializationFormat(format_str)
            
            self.save(model, format)
            return True
            
        except FileNotFoundError:
            return False
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search models by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching model summaries
        """
        all_models = self.list_models()
        matching_models = []
        
        query_lower = query.lower()
        
        for model in all_models:
            # Search in name
            if query_lower in model['name'].lower():
                matching_models.append(model)
                continue
            
            # Search in description
            if query_lower in model['description'].lower():
                matching_models.append(model)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in model['tags']):
                matching_models.append(model)
                continue
        
        return matching_models
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository statistics.
        
        Returns:
            Dictionary with repository statistics
        """
        registry = self._load_registry()
        models = registry['models']
        
        if not models:
            return {
                'total_models': 0,
                'by_status': {},
                'by_algorithm': {},
                'storage_size_mb': 0
            }
        
        # Count by status
        status_counts = {}
        for model_info in models.values():
            status = model_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by algorithm
        algorithm_counts = {}
        for model_info in models.values():
            algorithm = model_info['algorithm']
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        # Calculate storage size
        total_size = 0
        for path in self.storage_path.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        
        return {
            'total_models': len(models),
            'by_status': status_counts,
            'by_algorithm': algorithm_counts,
            'storage_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_path': str(self.storage_path)
        }
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _update_registry(self, model_id: str, model_info: Dict[str, Any]) -> None:
        """Update the model registry."""
        registry = self._load_registry()
        registry['models'][model_id] = model_info
        
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
    
    def _load_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Load metadata for a specific model."""
        metadata_file = self.storage_path / "metadata" / f"{model_id}.json"
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _metadata_to_dict(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Convert ModelMetadata to dictionary."""
        return {
            'model_id': metadata.model_id,
            'name': metadata.name,
            'algorithm': metadata.algorithm,
            'version': metadata.version,
            'created_at': metadata.created_at.isoformat(),
            'updated_at': metadata.updated_at.isoformat(),
            'status': metadata.status.value,
            'training_samples': metadata.training_samples,
            'training_features': metadata.training_features,
            'contamination_rate': metadata.contamination_rate,
            'training_duration_seconds': metadata.training_duration_seconds,
            'accuracy': metadata.accuracy,
            'precision': metadata.precision,
            'recall': metadata.recall,
            'f1_score': metadata.f1_score,
            'hyperparameters': metadata.hyperparameters,
            'feature_names': metadata.feature_names,
            'deployment_environment': metadata.deployment_environment,
            'api_endpoint': metadata.api_endpoint,
            'tags': metadata.tags,
            'description': metadata.description,
            'author': metadata.author
        }
    
    def cleanup_old_models(self, days: int = 30, keep_deployed: bool = True) -> int:
        """Clean up old models based on age.
        
        Args:
            days: Delete models older than this many days
            keep_deployed: Whether to keep deployed models regardless of age
            
        Returns:
            Number of models deleted
        """
        cutoff_date = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        models = self.list_models()
        
        for model in models:
            created_at = datetime.fromisoformat(model['created_at']).timestamp()
            
            # Skip if too recent
            if created_at > cutoff_date:
                continue
            
            # Skip deployed models if keep_deployed is True
            if keep_deployed and model['status'] == ModelStatus.DEPLOYED.value:
                continue
            
            # Delete the model
            if self.delete(model['model_id']):
                deleted_count += 1
        
        return deleted_count