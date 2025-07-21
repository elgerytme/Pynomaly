"""Model persistence and versioning system for anomaly detection."""

from __future__ import annotations

import json
import pickle
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
import numpy.typing as npt

from simplified_services.core_detection_service import DetectionResult


@dataclass
class ModelMetadata:
    """Metadata for persisted models."""
    model_id: str
    algorithm: str
    contamination: float
    creation_time: str
    training_samples: int
    feature_count: int
    version: str = "1.0"
    tags: List[str] = None
    performance_metrics: Dict[str, float] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class ModelPersistence:
    """Model persistence and versioning system.
    
    This system provides advanced model management capabilities:
    - Save and load trained models with metadata
    - Version control and model lineage tracking
    - Performance metrics tracking
    - Model comparison and selection
    - Automatic cleanup of old models
    """
    
    def __init__(self, storage_path: str = "models"):
        """Initialize model persistence system.
        
        Args:
            storage_path: Base path for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "models").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "versions").mkdir(exist_ok=True)
        
        self._model_registry: Dict[str, ModelMetadata] = {}
        self._load_registry()
    
    def save_model(
        self,
        model_data: Dict[str, Any],
        training_data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Save a trained model with metadata.
        
        Args:
            model_data: Serializable model data
            training_data: Training data used for the model
            algorithm: Algorithm name
            contamination: Contamination parameter
            model_id: Optional custom model ID
            tags: Optional tags for categorization
            description: Model description
            performance_metrics: Performance metrics
            
        Returns:
            Model ID for the saved model
        """
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(algorithm, training_data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            algorithm=algorithm,
            contamination=contamination,
            creation_time=datetime.now().isoformat(),
            training_samples=len(training_data),
            feature_count=training_data.shape[1] if len(training_data.shape) > 1 else 1,
            tags=tags or [],
            performance_metrics=performance_metrics or {},
            description=description
        )
        
        # Save model data
        model_path = self.storage_path / "models" / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model_data': model_data,
                'algorithm': algorithm,
                'contamination': contamination,
                'training_shape': training_data.shape,
                'training_stats': {
                    'mean': np.mean(training_data, axis=0).tolist(),
                    'std': np.std(training_data, axis=0).tolist(),
                    'min': np.min(training_data, axis=0).tolist(),
                    'max': np.max(training_data, axis=0).tolist()
                }
            }, f)
        
        # Save metadata
        metadata_path = self.storage_path / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Update registry
        self._model_registry[model_id] = metadata
        self._save_registry()
        
        print(f"💾 Model saved: {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a saved model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model data
        """
        model_path = self.storage_path / "models" / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        print(f"📂 Model loaded: {model_id}")
        return model_info
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata
        """
        if model_id not in self._model_registry:
            raise KeyError(f"Model not found in registry: {model_id}")
        
        return self._model_registry[model_id]
    
    def list_models(
        self,
        algorithm: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """List available models with optional filtering.
        
        Args:
            algorithm: Filter by algorithm
            tags: Filter by tags (any tag match)
            
        Returns:
            List of model metadata
        """
        models = list(self._model_registry.values())
        
        if algorithm:
            models = [m for m in models if m.algorithm == algorithm]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.creation_time, reverse=True)
        
        return models
    
    def delete_model(self, model_id: str) -> None:
        """Delete a model and its metadata.
        
        Args:
            model_id: Model identifier
        """
        # Remove files
        model_path = self.storage_path / "models" / f"{model_id}.pkl"
        metadata_path = self.storage_path / "metadata" / f"{model_id}.json"
        
        if model_path.exists():
            model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from registry
        if model_id in self._model_registry:
            del self._model_registry[model_id]
            self._save_registry()
        
        print(f"🗑️  Model deleted: {model_id}")
    
    def update_performance_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics to add/update
        """
        if model_id not in self._model_registry:
            raise KeyError(f"Model not found: {model_id}")
        
        # Update metrics
        self._model_registry[model_id].performance_metrics.update(metrics)
        
        # Save updated metadata
        metadata_path = self.storage_path / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self._model_registry[model_id]), f, indent=2)
        
        self._save_registry()
        
        print(f"📊 Performance metrics updated for: {model_id}")
    
    def compare_models(
        self,
        model_ids: List[str],
        metric: str = "accuracy"
    ) -> List[Dict[str, Any]]:
        """Compare models by a specific metric.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare by
            
        Returns:
            List of model comparisons sorted by metric
        """
        comparisons = []
        
        for model_id in model_ids:
            if model_id in self._model_registry:
                metadata = self._model_registry[model_id]
                metric_value = metadata.performance_metrics.get(metric, 0.0)
                
                comparisons.append({
                    'model_id': model_id,
                    'algorithm': metadata.algorithm,
                    'contamination': metadata.contamination,
                    metric: metric_value,
                    'creation_time': metadata.creation_time,
                    'training_samples': metadata.training_samples
                })
        
        # Sort by metric (higher is better)
        comparisons.sort(key=lambda x: x[metric], reverse=True)
        
        return comparisons
    
    def get_best_model(
        self,
        algorithm: Optional[str] = None,
        metric: str = "accuracy",
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Get the best performing model.
        
        Args:
            algorithm: Filter by algorithm
            metric: Metric to optimize for
            tags: Filter by tags
            
        Returns:
            Model ID of best model, or None if no models found
        """
        models = self.list_models(algorithm=algorithm, tags=tags)
        
        if not models:
            return None
        
        # Find best model by metric
        best_model = None
        best_score = -float('inf')
        
        for model in models:
            score = model.performance_metrics.get(metric, 0.0)
            if score > best_score:
                best_score = score
                best_model = model.model_id
        
        return best_model
    
    def cleanup_old_models(
        self,
        keep_best_n: int = 5,
        algorithm: Optional[str] = None
    ) -> List[str]:
        """Clean up old models, keeping only the best N.
        
        Args:
            keep_best_n: Number of best models to keep
            algorithm: Filter by algorithm
            
        Returns:
            List of deleted model IDs
        """
        models = self.list_models(algorithm=algorithm)
        
        if len(models) <= keep_best_n:
            return []
        
        # Sort by performance (assume accuracy metric)
        models_with_scores = [
            (model, model.performance_metrics.get('accuracy', 0.0))
            for model in models
        ]
        models_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Delete models beyond keep_best_n
        deleted_models = []
        for model, _ in models_with_scores[keep_best_n:]:
            self.delete_model(model.model_id)
            deleted_models.append(model.model_id)
        
        return deleted_models
    
    def export_model_info(self, output_path: str) -> None:
        """Export model registry information to JSON.
        
        Args:
            output_path: Path to save the export
        """
        export_data = {
            'registry': {
                model_id: asdict(metadata)
                for model_id, metadata in self._model_registry.items()
            },
            'export_time': datetime.now().isoformat(),
            'total_models': len(self._model_registry)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📤 Model info exported to: {output_path}")
    
    def _generate_model_id(
        self,
        algorithm: str,
        training_data: npt.NDArray[np.floating]
    ) -> str:
        """Generate a unique model ID based on algorithm and data."""
        # Create hash from algorithm and data characteristics
        data_hash = hashlib.md5(
            f"{algorithm}_{training_data.shape}_{np.mean(training_data):.6f}".encode()
        ).hexdigest()[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{algorithm}_{timestamp}_{data_hash}"
    
    def _load_registry(self) -> None:
        """Load model registry from disk."""
        registry_path = self.storage_path / "registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, metadata_dict in registry_data.items():
                    self._model_registry[model_id] = ModelMetadata(**metadata_dict)
                
            except Exception as e:
                print(f"Warning: Could not load model registry: {e}")
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        registry_path = self.storage_path / "registry.json"
        
        registry_data = {
            model_id: asdict(metadata)
            for model_id, metadata in self._model_registry.items()
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        models_dir = self.storage_path / "models"
        metadata_dir = self.storage_path / "metadata"
        
        total_size = 0
        model_count = 0
        
        if models_dir.exists():
            for model_file in models_dir.glob("*.pkl"):
                total_size += model_file.stat().st_size
                model_count += 1
        
        return {
            'total_models': model_count,
            'total_size_mb': total_size / (1024 * 1024),
            'storage_path': str(self.storage_path),
            'models_directory': str(models_dir),
            'metadata_directory': str(metadata_dir)
        }