"""Model entity for managing trained anomaly detection models."""

from __future__ import annotations

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union, Protocol
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from enum import Enum


class ModelStatus(Enum):
    """Status of a trained model."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class SerializationFormat(Enum):
    """Supported serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    JSON = "json"


class ModelType(Enum):
    """Types of anomaly detection models."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    
    model_id: str
    name: str
    algorithm: str
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: ModelStatus = ModelStatus.TRAINING
    
    # Training metadata
    training_samples: Optional[int] = None
    training_features: Optional[int] = None
    contamination_rate: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Model parameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: Optional[list[str]] = None
    
    # Deployment info
    deployment_environment: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # Additional metadata
    tags: list[str] = field(default_factory=list)
    description: str = ""
    author: str = "system"


class SerializableModel(Protocol):
    """Protocol for models that can be serialized."""
    
    def fit(self, X: npt.NDArray[np.floating]) -> None:
        """Fit the model to training data."""
        ...
    
    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies in data."""
        ...
    
    def decision_function(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Compute anomaly scores."""
        ...


@dataclass
class Model:
    """Container for trained anomaly detection models with persistence."""
    
    metadata: ModelMetadata
    model_object: Optional[Any] = None
    preprocessing_pipeline: Optional[Any] = None
    
    def __post_init__(self) -> None:
        """Initialize model after creation."""
        if self.metadata.updated_at < self.metadata.created_at:
            self.metadata.updated_at = self.metadata.created_at
    
    def save(self, file_path: Union[str, Path], format: SerializationFormat = SerializationFormat.PICKLE) -> None:
        """Save model to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata.updated_at = datetime.utcnow()
        
        if format == SerializationFormat.PICKLE:
            self._save_pickle(file_path)
        elif format == SerializationFormat.JOBLIB:
            self._save_joblib(file_path)
        elif format == SerializationFormat.JSON:
            self._save_json(file_path)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    def _save_pickle(self, file_path: Path) -> None:
        """Save model using pickle format."""
        if not file_path.suffix:
            file_path = file_path.with_suffix('.pkl')
        
        model_data = {
            'metadata': self.metadata,
            'model_object': self.model_object,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'serialization_format': 'pickle',
            'saved_at': datetime.utcnow().isoformat()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_joblib(self, file_path: Path) -> None:
        """Save model using joblib format."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for joblib serialization format")
        
        if not file_path.suffix:
            file_path = file_path.with_suffix('.joblib')
        
        model_data = {
            'metadata': self.metadata,
            'model_object': self.model_object,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'serialization_format': 'joblib',
            'saved_at': datetime.utcnow().isoformat()
        }
        
        joblib.dump(model_data, file_path, compress=3)
    
    def _save_json(self, file_path: Path) -> None:
        """Save model metadata as JSON (model object not serialized)."""
        if not file_path.suffix:
            file_path = file_path.with_suffix('.json')
        
        # Convert metadata to dict (model object cannot be JSON serialized)
        metadata_dict = {
            'model_id': self.metadata.model_id,
            'name': self.metadata.name,
            'algorithm': self.metadata.algorithm,
            'version': self.metadata.version,
            'created_at': self.metadata.created_at.isoformat(),
            'updated_at': self.metadata.updated_at.isoformat(),
            'status': self.metadata.status.value,
            'training_samples': self.metadata.training_samples,
            'training_features': self.metadata.training_features,
            'contamination_rate': self.metadata.contamination_rate,
            'training_duration_seconds': self.metadata.training_duration_seconds,
            'accuracy': self.metadata.accuracy,
            'precision': self.metadata.precision,
            'recall': self.metadata.recall,
            'f1_score': self.metadata.f1_score,
            'hyperparameters': self.metadata.hyperparameters,
            'feature_names': self.metadata.feature_names,
            'deployment_environment': self.metadata.deployment_environment,
            'api_endpoint': self.metadata.api_endpoint,
            'tags': self.metadata.tags,
            'description': self.metadata.description,
            'author': self.metadata.author,
            'serialization_format': 'json',
            'saved_at': datetime.utcnow().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Model:
        """Load model from disk."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pkl':
            return cls._load_pickle(file_path)
        elif suffix == '.joblib':
            return cls._load_joblib(file_path)
        elif suffix == '.json':
            return cls._load_json(file_path)
        else:
            # Try to detect format by attempting to load
            try:
                return cls._load_pickle(file_path)
            except:
                try:
                    return cls._load_joblib(file_path)
                except:
                    return cls._load_json(file_path)
    
    @classmethod
    def _load_pickle(cls, file_path: Path) -> Model:
        """Load model from pickle format."""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return cls(
            metadata=model_data['metadata'],
            model_object=model_data.get('model_object'),
            preprocessing_pipeline=model_data.get('preprocessing_pipeline')
        )
    
    @classmethod
    def _load_joblib(cls, file_path: Path) -> Model:
        """Load model from joblib format."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required to load joblib format models")
        
        model_data = joblib.load(file_path)
        
        return cls(
            metadata=model_data['metadata'],
            model_object=model_data.get('model_object'),
            preprocessing_pipeline=model_data.get('preprocessing_pipeline')
        )
    
    @classmethod
    def _load_json(cls, file_path: Path) -> Model:
        """Load model metadata from JSON format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct metadata
        metadata = ModelMetadata(
            model_id=data['model_id'],
            name=data['name'],
            algorithm=data['algorithm'],
            version=data.get('version', '1.0.0'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            status=ModelStatus(data['status']),
            training_samples=data.get('training_samples'),
            training_features=data.get('training_features'),
            contamination_rate=data.get('contamination_rate'),
            training_duration_seconds=data.get('training_duration_seconds'),
            accuracy=data.get('accuracy'),
            precision=data.get('precision'),
            recall=data.get('recall'),
            f1_score=data.get('f1_score'),
            hyperparameters=data.get('hyperparameters', {}),
            feature_names=data.get('feature_names'),
            deployment_environment=data.get('deployment_environment'),
            api_endpoint=data.get('api_endpoint'),
            tags=data.get('tags', []),
            description=data.get('description', ''),
            author=data.get('author', 'system')
        )
        
        return cls(
            metadata=metadata,
            model_object=None,  # JSON format doesn't store model object
            preprocessing_pipeline=None
        )
    
    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Make predictions using the loaded model."""
        if self.model_object is None:
            raise ValueError("No model object loaded. Cannot make predictions.")
        
        # Apply preprocessing if available
        processed_X = X
        if self.preprocessing_pipeline is not None:
            processed_X = self.preprocessing_pipeline.transform(X)
        
        # Make predictions
        if hasattr(self.model_object, 'predict'):
            return self.model_object.predict(processed_X)
        else:
            raise ValueError("Model object does not support prediction")
    
    def get_anomaly_scores(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get anomaly scores from the model."""
        if self.model_object is None:
            raise ValueError("No model object loaded. Cannot compute scores.")
        
        # Apply preprocessing if available
        processed_X = X
        if self.preprocessing_pipeline is not None:
            processed_X = self.preprocessing_pipeline.transform(X)
        
        # Get scores
        if hasattr(self.model_object, 'decision_function'):
            return self.model_object.decision_function(processed_X)
        elif hasattr(self.model_object, 'score_samples'):
            return self.model_object.score_samples(processed_X)
        else:
            raise ValueError("Model object does not support anomaly scoring")
    
    def update_performance_metrics(self, accuracy: float, precision: float, 
                                 recall: float, f1_score: float) -> None:
        """Update model performance metrics."""
        self.metadata.accuracy = accuracy
        self.metadata.precision = precision
        self.metadata.recall = recall
        self.metadata.f1_score = f1_score
        self.metadata.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)
            self.metadata.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.metadata.tags:
            self.metadata.tags.remove(tag)
            self.metadata.updated_at = datetime.utcnow()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model."""
        return {
            'model_id': self.metadata.model_id,
            'name': self.metadata.name,
            'algorithm': self.metadata.algorithm,
            'version': self.metadata.version,
            'status': self.metadata.status.value,
            'created_at': self.metadata.created_at.isoformat(),
            'training_samples': self.metadata.training_samples,
            'accuracy': self.metadata.accuracy,
            'tags': self.metadata.tags,
            'has_model_object': self.model_object is not None,
            'has_preprocessing': self.preprocessing_pipeline is not None
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return (f"Model(id={self.metadata.model_id}, name={self.metadata.name}, "
               f"algorithm={self.metadata.algorithm}, status={self.metadata.status.value})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()