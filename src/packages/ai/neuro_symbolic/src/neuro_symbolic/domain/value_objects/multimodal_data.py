"""Value objects for multi-modal data handling in neuro-symbolic AI."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
import json


class ModalityType(Enum):
    """Types of data modalities."""
    NUMERICAL = "numerical"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TIME_SERIES = "time_series"
    CATEGORICAL = "categorical"
    GRAPH = "graph"
    SENSOR = "sensor"
    LOG = "log"
    METADATA = "metadata"


class FusionLevel(Enum):
    """Levels at which modalities can be fused."""
    EARLY = "early"          # Raw feature level
    INTERMEDIATE = "intermediate"  # Hidden representation level
    LATE = "late"            # Decision level
    HIERARCHICAL = "hierarchical"  # Multiple levels


@dataclass(frozen=True)
class ModalityInfo:
    """Information about a specific data modality."""
    name: str
    modality_type: ModalityType
    shape: Tuple[int, ...]
    dtype: str
    description: Optional[str] = None
    preprocessing_info: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Modality name cannot be empty")
        
        if len(self.shape) == 0:
            raise ValueError("Modality shape cannot be empty")
        
        if any(dim <= 0 for dim in self.shape):
            raise ValueError("All shape dimensions must be positive")


@dataclass
class MultiModalSample:
    """A single multi-modal data sample."""
    sample_id: str
    modalities: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("Sample ID cannot be empty")
        
        if not self.modalities:
            raise ValueError("At least one modality must be provided")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def get_modality(self, name: str) -> Any:
        """Get data for a specific modality."""
        if name not in self.modalities:
            raise KeyError(f"Modality '{name}' not found in sample")
        return self.modalities[name]
    
    def has_modality(self, name: str) -> bool:
        """Check if sample contains a specific modality."""
        return name in self.modalities
    
    def get_modality_names(self) -> List[str]:
        """Get names of all modalities in this sample."""
        return list(self.modalities.keys())


@dataclass
class MultiModalBatch:
    """A batch of multi-modal data samples."""
    samples: List[MultiModalSample]
    modality_info: Dict[str, ModalityInfo]
    batch_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.samples:
            raise ValueError("Batch cannot be empty")
        
        # Validate that all samples have consistent modalities
        first_sample_modalities = set(self.samples[0].get_modality_names())
        for i, sample in enumerate(self.samples[1:], 1):
            sample_modalities = set(sample.get_modality_names())
            if sample_modalities != first_sample_modalities:
                raise ValueError(
                    f"Sample {i} has inconsistent modalities. "
                    f"Expected: {first_sample_modalities}, Got: {sample_modalities}"
                )
        
        # Validate modality info matches actual modalities
        for modality_name in first_sample_modalities:
            if modality_name not in self.modality_info:
                raise ValueError(f"Missing modality info for '{modality_name}'")
    
    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return len(self.samples)
    
    @property
    def modality_names(self) -> List[str]:
        """Names of all modalities in the batch."""
        return list(self.modality_info.keys())
    
    def get_modality_data(self, modality_name: str) -> List[Any]:
        """Get all data for a specific modality across the batch."""
        if modality_name not in self.modality_info:
            raise KeyError(f"Modality '{modality_name}' not found in batch")
        
        return [sample.get_modality(modality_name) for sample in self.samples]
    
    def get_modality_array(self, modality_name: str) -> NDArray[np.floating]:
        """Get modality data as a numpy array."""
        modality_data = self.get_modality_data(modality_name)
        
        try:
            # Try to stack into a single array
            return np.array(modality_data)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert modality '{modality_name}' to array: {e}. "
                f"Data might have inconsistent shapes or types."
            )
    
    def filter_by_modalities(self, modality_names: List[str]) -> 'MultiModalBatch':
        """Create a new batch with only specified modalities."""
        # Validate requested modalities exist
        missing = set(modality_names) - set(self.modality_names)
        if missing:
            raise ValueError(f"Requested modalities not found: {missing}")
        
        # Filter samples
        filtered_samples = []
        for sample in self.samples:
            filtered_modalities = {
                name: sample.get_modality(name) 
                for name in modality_names
            }
            filtered_sample = MultiModalSample(
                sample_id=sample.sample_id,
                modalities=filtered_modalities,
                metadata=sample.metadata.copy(),
                timestamp=sample.timestamp
            )
            filtered_samples.append(filtered_sample)
        
        # Filter modality info
        filtered_info = {
            name: self.modality_info[name] 
            for name in modality_names
        }
        
        return MultiModalBatch(
            samples=filtered_samples,
            modality_info=filtered_info,
            batch_metadata=self.batch_metadata.copy()
        )


@dataclass
class ModalityWeight:
    """Weight configuration for modality fusion."""
    modality_name: str
    weight: float
    confidence: float = 1.0
    adaptive: bool = False
    
    def __post_init__(self):
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class FusionConfiguration:
    """Configuration for multi-modal fusion."""
    fusion_level: FusionLevel
    modality_weights: List[ModalityWeight]
    fusion_method: str = "weighted_average"
    normalization: str = "l2"
    temperature: float = 1.0
    adaptive_weighting: bool = False
    
    def __post_init__(self):
        if not self.modality_weights:
            raise ValueError("At least one modality weight must be specified")
        
        # Validate weights sum to approximately 1.0
        total_weight = sum(w.weight for w in self.modality_weights)
        if not (0.9 <= total_weight <= 1.1):
            raise ValueError(f"Modality weights should sum to ~1.0, got {total_weight}")
        
        # Validate fusion method
        valid_methods = [
            "weighted_average", "attention", "concatenation", 
            "product", "maximum", "gating", "transformer"
        ]
        if self.fusion_method not in valid_methods:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")
        
        # Validate normalization
        valid_norms = ["l1", "l2", "max", "none"]
        if self.normalization not in valid_norms:
            raise ValueError(f"Invalid normalization: {self.normalization}")
        
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
    
    def get_weight(self, modality_name: str) -> Optional[float]:
        """Get weight for a specific modality."""
        for weight_config in self.modality_weights:
            if weight_config.modality_name == modality_name:
                return weight_config.weight
        return None
    
    def update_weights(self, new_weights: Dict[str, float]) -> 'FusionConfiguration':
        """Create a new configuration with updated weights."""
        updated_weights = []
        
        for weight_config in self.modality_weights:
            if weight_config.modality_name in new_weights:
                updated_weight = ModalityWeight(
                    modality_name=weight_config.modality_name,
                    weight=new_weights[weight_config.modality_name],
                    confidence=weight_config.confidence,
                    adaptive=weight_config.adaptive
                )
            else:
                updated_weight = weight_config
            
            updated_weights.append(updated_weight)
        
        return FusionConfiguration(
            fusion_level=self.fusion_level,
            modality_weights=updated_weights,
            fusion_method=self.fusion_method,
            normalization=self.normalization,
            temperature=self.temperature,
            adaptive_weighting=self.adaptive_weighting
        )


@dataclass
class MultiModalResult:
    """Result of multi-modal processing."""
    predictions: NDArray[np.integer]
    confidence_scores: NDArray[np.floating]
    modality_contributions: Dict[str, NDArray[np.floating]]
    fusion_weights: Optional[Dict[str, float]] = None
    attention_scores: Optional[Dict[str, NDArray[np.floating]]] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.predictions) != len(self.confidence_scores):
            raise ValueError("Predictions and confidence scores must have same length")
        
        # Validate modality contributions
        expected_length = len(self.predictions)
        for modality, contributions in self.modality_contributions.items():
            if len(contributions) != expected_length:
                raise ValueError(
                    f"Modality '{modality}' contributions length ({len(contributions)}) "
                    f"doesn't match predictions length ({expected_length})"
                )
    
    @property
    def num_samples(self) -> int:
        """Number of samples in the result."""
        return len(self.predictions)
    
    def get_top_contributing_modality(self, sample_idx: int) -> str:
        """Get the modality that contributed most to a specific sample's prediction."""
        if sample_idx >= self.num_samples:
            raise IndexError(f"Sample index {sample_idx} out of range")
        
        max_contribution = -1
        top_modality = None
        
        for modality, contributions in self.modality_contributions.items():
            if contributions[sample_idx] > max_contribution:
                max_contribution = contributions[sample_idx]
                top_modality = modality
        
        return top_modality
    
    def get_modality_ranking(self, sample_idx: int) -> List[Tuple[str, float]]:
        """Get modalities ranked by their contribution to a specific sample."""
        if sample_idx >= self.num_samples:
            raise IndexError(f"Sample index {sample_idx} out of range")
        
        contributions = [
            (modality, contributions[sample_idx])
            for modality, contributions in self.modality_contributions.items()
        ]
        
        return sorted(contributions, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "predictions": self.predictions.tolist(),
            "confidence_scores": self.confidence_scores.tolist(),
            "modality_contributions": {
                modality: contributions.tolist()
                for modality, contributions in self.modality_contributions.items()
            },
            "fusion_weights": self.fusion_weights,
            "attention_scores": {
                modality: scores.tolist()
                for modality, scores in (self.attention_scores or {}).items()
            },
            "processing_metadata": self.processing_metadata,
            "num_samples": self.num_samples
        }