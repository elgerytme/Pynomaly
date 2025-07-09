"""Multi-modal anomaly detection domain models for handling diverse data types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.value_objects import PerformanceMetrics

# Type alias for backward compatibility
ModelMetrics = PerformanceMetrics


class ModalityType(Enum):
    """Types of data modalities supported."""
    
    # Structured data
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    
    # Unstructured data
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    
    # Graph data
    GRAPH = "graph"
    NETWORK = "network"
    
    # Sensor data
    IOT_SENSOR = "iot_sensor"
    ACCELEROMETER = "accelerometer"
    GPS = "gps"
    
    # Domain-specific
    LOG_DATA = "log_data"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    CYBERSECURITY = "cybersecurity"


class FusionStrategy(Enum):
    """Strategies for fusing multiple modalities."""
    
    EARLY_FUSION = "early_fusion"          # Concatenate features before processing
    LATE_FUSION = "late_fusion"            # Combine predictions from individual models
    INTERMEDIATE_FUSION = "intermediate_fusion"  # Fuse at intermediate layers
    ATTENTION_FUSION = "attention_fusion"   # Use attention mechanisms
    ENSEMBLE_FUSION = "ensemble_fusion"     # Ensemble of modality-specific models
    HIERARCHICAL_FUSION = "hierarchical_fusion"  # Multi-level fusion
    ADAPTIVE_FUSION = "adaptive_fusion"     # Learn fusion weights dynamically


class EncodingType(Enum):
    """Types of encoding for different modalities."""
    
    # Numerical encodings
    STANDARDIZATION = "standardization"
    NORMALIZATION = "normalization"
    ROBUST_SCALING = "robust_scaling"
    
    # Categorical encodings
    ONE_HOT = "one_hot"
    LABEL_ENCODING = "label_encoding"
    TARGET_ENCODING = "target_encoding"
    EMBEDDING = "embedding"
    
    # Text encodings
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    BERT = "bert"
    TRANSFORMER = "transformer"
    
    # Image encodings
    CNN = "cnn"
    RESNET = "resnet"
    VIT = "vision_transformer"
    
    # Audio encodings
    MFCC = "mfcc"
    SPECTROGRAM = "spectrogram"
    MEL_SPECTROGRAM = "mel_spectrogram"
    
    # Graph encodings
    NODE2VEC = "node2vec"
    GRAPH_NEURAL_NETWORK = "gnn"
    GRAPH_ATTENTION = "graph_attention"


@dataclass
class ModalityConfig:
    """Configuration for a specific data modality."""
    
    modality_type: ModalityType
    encoding_type: EncodingType
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    feature_extraction_params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Importance weight for this modality
    is_required: bool = True  # Whether this modality is required for prediction
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError("Modality weight must be between 0 and 1")
        
        # Set default preprocessing parameters based on modality type
        if self.modality_type == ModalityType.TEXT and not self.preprocessing_params:
            self.preprocessing_params = {
                "lowercase": True,
                "remove_punctuation": True,
                "remove_stopwords": True,
                "max_length": 512,
            }
        elif self.modality_type == ModalityType.IMAGE and not self.preprocessing_params:
            self.preprocessing_params = {
                "resize": (224, 224),
                "normalize": True,
                "augmentation": False,
            }
        elif self.modality_type == ModalityType.TIME_SERIES and not self.preprocessing_params:
            self.preprocessing_params = {
                "window_size": 100,
                "overlap": 0.5,
                "normalization": "z_score",
            }


@dataclass
class MultiModalData:
    """Container for multi-modal data samples."""
    
    sample_id: str
    modalities: Dict[ModalityType, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_modality_data(self, modality_type: ModalityType) -> Optional[Any]:
        """Get data for specific modality."""
        return self.modalities.get(modality_type)
    
    def has_modality(self, modality_type: ModalityType) -> bool:
        """Check if sample has specific modality."""
        return modality_type in self.modalities
    
    def get_available_modalities(self) -> Set[ModalityType]:
        """Get set of available modalities in this sample."""
        return set(self.modalities.keys())
    
    def is_complete(self, required_modalities: Set[ModalityType]) -> bool:
        """Check if sample has all required modalities."""
        return required_modalities.issubset(self.get_available_modalities())


@dataclass
class ModalityEncoder:
    """Encoder for transforming raw modality data to features."""
    
    encoder_id: UUID
    modality_type: ModalityType
    encoding_type: EncodingType
    config: ModalityConfig
    feature_dimension: int
    is_trained: bool = False
    encoder_state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.feature_dimension <= 0:
            raise ValueError("Feature dimension must be positive")
    
    def encode(self, data: Any) -> np.ndarray:
        """Encode raw data to feature vector."""
        if not self.is_trained and self.encoding_type not in [
            EncodingType.STANDARDIZATION, 
            EncodingType.NORMALIZATION
        ]:
            raise ValueError("Encoder must be trained before encoding")
        
        return self._encode_data(data)
    
    def _encode_data(self, data: Any) -> np.ndarray:
        """Internal encoding implementation."""
        if self.modality_type == ModalityType.TABULAR:
            return self._encode_tabular(data)
        elif self.modality_type == ModalityType.TEXT:
            return self._encode_text(data)
        elif self.modality_type == ModalityType.IMAGE:
            return self._encode_image(data)
        elif self.modality_type == ModalityType.TIME_SERIES:
            return self._encode_time_series(data)
        else:
            # Default to simple flattening
            if isinstance(data, np.ndarray):
                return data.flatten()[:self.feature_dimension]
            else:
                return np.array([float(data)] * self.feature_dimension)
    
    def _encode_tabular(self, data: Union[np.ndarray, List]) -> np.ndarray:
        """Encode tabular data."""
        if isinstance(data, list):
            data = np.array(data)
        
        if self.encoding_type == EncodingType.STANDARDIZATION:
            if "mean" in self.encoder_state and "std" in self.encoder_state:
                return (data - self.encoder_state["mean"]) / (self.encoder_state["std"] + 1e-8)
            else:
                # First time - compute and store statistics
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                self.encoder_state["mean"] = mean
                self.encoder_state["std"] = std
                return (data - mean) / (std + 1e-8)
        
        return data.astype(np.float32)
    
    def _encode_text(self, data: str) -> np.ndarray:
        """Encode text data."""
        if self.encoding_type == EncodingType.TFIDF:
            # Simplified TF-IDF encoding
            words = data.lower().split()
            # Create simple bag-of-words representation
            vocab_size = min(self.feature_dimension, 1000)
            features = np.zeros(vocab_size)
            
            for word in words[:vocab_size]:
                # Simple hash-based word mapping
                word_idx = hash(word) % vocab_size
                features[word_idx] += 1
            
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
        
        # Default: character-level encoding
        char_features = np.zeros(self.feature_dimension)
        for i, char in enumerate(data[:self.feature_dimension]):
            char_features[i] = ord(char) / 255.0  # Normalize to [0, 1]
        
        return char_features
    
    def _encode_image(self, data: np.ndarray) -> np.ndarray:
        """Encode image data."""
        if len(data.shape) == 3:  # Color image
            # Simple feature extraction: flatten and downsample
            flattened = data.flatten()
            if len(flattened) > self.feature_dimension:
                # Downsample
                step = len(flattened) // self.feature_dimension
                features = flattened[::step][:self.feature_dimension]
            else:
                # Pad if necessary
                features = np.pad(flattened, (0, self.feature_dimension - len(flattened)))
        else:
            # Grayscale or already flattened
            features = data.flatten()[:self.feature_dimension]
        
        return features.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    def _encode_time_series(self, data: np.ndarray) -> np.ndarray:
        """Encode time series data."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Extract statistical features
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            np.min(data, axis=0),
            np.max(data, axis=0),
        ])
        
        # Trend features
        if len(data) > 1:
            diff = np.diff(data, axis=0)
            features.extend([
                np.mean(diff, axis=0),
                np.std(diff, axis=0),
            ])
        
        # Flatten and ensure correct dimension
        features = np.concatenate([f.flatten() if hasattr(f, 'flatten') else [f] for f in features])
        
        if len(features) > self.feature_dimension:
            features = features[:self.feature_dimension]
        elif len(features) < self.feature_dimension:
            features = np.pad(features, (0, self.feature_dimension - len(features)))
        
        return features.astype(np.float32)


@dataclass
class FusionLayer:
    """Layer for fusing multiple modality features."""
    
    fusion_id: UUID
    fusion_strategy: FusionStrategy
    input_modalities: List[ModalityType]
    output_dimension: int
    fusion_weights: Dict[ModalityType, float] = field(default_factory=dict)
    fusion_params: Dict[str, Any] = field(default_factory=dict)
    is_trainable: bool = True
    
    def __post_init__(self):
        # Initialize equal weights if not provided
        if not self.fusion_weights:
            weight = 1.0 / len(self.input_modalities)
            self.fusion_weights = {mod: weight for mod in self.input_modalities}
        
        # Normalize weights
        total_weight = sum(self.fusion_weights.values())
        if total_weight > 0:
            self.fusion_weights = {
                mod: weight / total_weight 
                for mod, weight in self.fusion_weights.items()
            }
    
    def fuse(self, modality_features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Fuse features from multiple modalities."""
        if self.fusion_strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(modality_features)
        elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(modality_features)
        elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            return self._attention_fusion(modality_features)
        elif self.fusion_strategy == FusionStrategy.ADAPTIVE_FUSION:
            return self._adaptive_fusion(modality_features)
        else:
            # Default to early fusion
            return self._early_fusion(modality_features)
    
    def _early_fusion(self, modality_features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Concatenate features from different modalities."""
        features_list = []
        
        for modality in self.input_modalities:
            if modality in modality_features:
                features = modality_features[modality]
                weight = self.fusion_weights.get(modality, 1.0)
                features_list.append(features * weight)
        
        if not features_list:
            return np.zeros(self.output_dimension)
        
        # Concatenate
        concatenated = np.concatenate(features_list)
        
        # Adjust to output dimension
        if len(concatenated) > self.output_dimension:
            concatenated = concatenated[:self.output_dimension]
        elif len(concatenated) < self.output_dimension:
            concatenated = np.pad(concatenated, (0, self.output_dimension - len(concatenated)))
        
        return concatenated
    
    def _late_fusion(self, modality_features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Weighted average of modality features."""
        weighted_sum = np.zeros(self.output_dimension)
        total_weight = 0.0
        
        for modality in self.input_modalities:
            if modality in modality_features:
                features = modality_features[modality]
                weight = self.fusion_weights.get(modality, 1.0)
                
                # Adjust feature dimension to output dimension
                if len(features) != self.output_dimension:
                    if len(features) > self.output_dimension:
                        features = features[:self.output_dimension]
                    else:
                        features = np.pad(features, (0, self.output_dimension - len(features)))
                
                weighted_sum += features * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.zeros(self.output_dimension)
    
    def _attention_fusion(self, modality_features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Attention-based fusion of modality features."""
        # Simplified attention mechanism
        
        # Compute attention weights based on feature magnitudes
        attention_weights = {}
        total_magnitude = 0.0
        
        for modality in self.input_modalities:
            if modality in modality_features:
                features = modality_features[modality]
                magnitude = np.linalg.norm(features)
                attention_weights[modality] = magnitude
                total_magnitude += magnitude
        
        # Normalize attention weights
        if total_magnitude > 0:
            for modality in attention_weights:
                attention_weights[modality] /= total_magnitude
        
        # Apply attention weights
        weighted_sum = np.zeros(self.output_dimension)
        
        for modality in self.input_modalities:
            if modality in modality_features:
                features = modality_features[modality]
                attention_weight = attention_weights.get(modality, 0.0)
                fusion_weight = self.fusion_weights.get(modality, 1.0)
                
                # Adjust feature dimension
                if len(features) != self.output_dimension:
                    if len(features) > self.output_dimension:
                        features = features[:self.output_dimension]
                    else:
                        features = np.pad(features, (0, self.output_dimension - len(features)))
                
                weighted_sum += features * attention_weight * fusion_weight
        
        return weighted_sum
    
    def _adaptive_fusion(self, modality_features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Adaptive fusion that learns optimal weights."""
        # For now, use quality-based weighting
        quality_weights = {}
        
        for modality in self.input_modalities:
            if modality in modality_features:
                features = modality_features[modality]
                
                # Simple quality metric: inverse of feature variance
                variance = np.var(features)
                quality = 1.0 / (1.0 + variance)
                quality_weights[modality] = quality
        
        # Normalize quality weights
        total_quality = sum(quality_weights.values())
        if total_quality > 0:
            for modality in quality_weights:
                quality_weights[modality] /= total_quality
        
        # Combine with configured weights
        final_weights = {}
        for modality in self.input_modalities:
            if modality in modality_features:
                quality_weight = quality_weights.get(modality, 0.0)
                fusion_weight = self.fusion_weights.get(modality, 1.0)
                final_weights[modality] = quality_weight * fusion_weight
        
        # Apply final weights
        weighted_sum = np.zeros(self.output_dimension)
        total_weight = sum(final_weights.values())
        
        for modality in self.input_modalities:
            if modality in modality_features:
                features = modality_features[modality]
                weight = final_weights.get(modality, 0.0)
                
                if total_weight > 0:
                    weight /= total_weight
                
                # Adjust feature dimension
                if len(features) != self.output_dimension:
                    if len(features) > self.output_dimension:
                        features = features[:self.output_dimension]
                    else:
                        features = np.pad(features, (0, self.output_dimension - len(features)))
                
                weighted_sum += features * weight
        
        return weighted_sum


@dataclass
class MultiModalDetector:
    """Multi-modal anomaly detector for handling diverse data types."""
    
    detector_id: UUID
    name: str
    modality_configs: Dict[ModalityType, ModalityConfig]
    encoders: Dict[ModalityType, ModalityEncoder]
    fusion_layers: List[FusionLayer]
    output_dimension: int
    
    # Detection thresholds
    anomaly_threshold: float = 0.5
    confidence_threshold: float = 0.8
    
    # Training state
    is_trained: bool = False
    training_samples: int = 0
    validation_metrics: Optional[ModelMetrics] = None
    
    # Model state
    model_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_importance: Dict[ModalityType, float] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.output_dimension <= 0:
            raise ValueError("Output dimension must be positive")
        
        # Validate modality configs match encoders
        for modality in self.modality_configs.keys():
            if modality not in self.encoders:
                raise ValueError(f"Missing encoder for modality {modality}")
        
        # Initialize feature importance
        if not self.feature_importance:
            num_modalities = len(self.modality_configs)
            if num_modalities > 0:
                importance = 1.0 / num_modalities
                self.feature_importance = {
                    modality: importance for modality in self.modality_configs.keys()
                }
    
    def get_required_modalities(self) -> Set[ModalityType]:
        """Get set of required modalities for this detector."""
        return {
            modality for modality, config in self.modality_configs.items()
            if config.is_required
        }
    
    def get_optional_modalities(self) -> Set[ModalityType]:
        """Get set of optional modalities for this detector."""
        return {
            modality for modality, config in self.modality_configs.items()
            if not config.is_required
        }
    
    def can_process_sample(self, sample: MultiModalData) -> bool:
        """Check if detector can process given sample."""
        required_modalities = self.get_required_modalities()
        return sample.is_complete(required_modalities)
    
    def extract_features(self, sample: MultiModalData) -> np.ndarray:
        """Extract fused features from multi-modal sample."""
        if not self.can_process_sample(sample):
            raise ValueError("Sample missing required modalities")
        
        # Extract features from each modality
        modality_features = {}
        
        for modality_type, encoder in self.encoders.items():
            if sample.has_modality(modality_type):
                raw_data = sample.get_modality_data(modality_type)
                features = encoder.encode(raw_data)
                modality_features[modality_type] = features
        
        # Apply fusion layers
        fused_features = modality_features
        
        for fusion_layer in self.fusion_layers:
            # Apply fusion to relevant modalities
            relevant_features = {
                mod: features for mod, features in fused_features.items()
                if mod in fusion_layer.input_modalities
            }
            
            if relevant_features:
                fused_output = fusion_layer.fuse(relevant_features)
                
                # Update fused features (simplified - in practice would be more complex)
                # For now, replace all input modalities with fused output
                for mod in fusion_layer.input_modalities:
                    if mod in fused_features:
                        del fused_features[mod]
                
                # Add fused features (using first modality as key)
                if fusion_layer.input_modalities:
                    fusion_key = fusion_layer.input_modalities[0]
                    fused_features[fusion_key] = fused_output
        
        # Final feature vector
        if len(fused_features) == 1:
            final_features = next(iter(fused_features.values()))
        else:
            # Concatenate remaining features
            feature_list = list(fused_features.values())
            final_features = np.concatenate(feature_list)
        
        # Ensure correct output dimension
        if len(final_features) > self.output_dimension:
            final_features = final_features[:self.output_dimension]
        elif len(final_features) < self.output_dimension:
            final_features = np.pad(
                final_features, 
                (0, self.output_dimension - len(final_features))
            )
        
        return final_features
    
    def detect_anomaly(self, sample: MultiModalData) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomaly in multi-modal sample."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        # Extract features
        features = self.extract_features(sample)
        
        # Simple anomaly detection (placeholder for actual ML model)
        # In practice, this would use a trained model
        anomaly_score = self._calculate_anomaly_score(features)
        
        # Determine if anomalous
        is_anomaly = anomaly_score > self.anomaly_threshold
        
        # Calculate confidence
        confidence = min(1.0, abs(anomaly_score - 0.5) * 2)
        
        # Additional details
        details = {
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "feature_dimension": len(features),
            "modalities_used": list(sample.get_available_modalities()),
            "modality_contributions": self._calculate_modality_contributions(sample),
        }
        
        return is_anomaly, anomaly_score, details
    
    def _calculate_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score from features."""
        # Simple scoring based on feature magnitudes and patterns
        # In practice, this would use a trained ML model
        
        # L2 norm of features
        l2_norm = np.linalg.norm(features)
        
        # Statistical measures
        mean_val = np.mean(features)
        std_val = np.std(features)
        
        # Combine measures
        score = (l2_norm * 0.4 + abs(mean_val) * 0.3 + std_val * 0.3)
        
        # Normalize to [0, 1]
        score = 1.0 / (1.0 + np.exp(-score + 5))  # Sigmoid with offset
        
        return float(score)
    
    def _calculate_modality_contributions(self, sample: MultiModalData) -> Dict[str, float]:
        """Calculate contribution of each modality to anomaly score."""
        contributions = {}
        
        for modality_type in sample.get_available_modalities():
            # Get modality importance
            importance = self.feature_importance.get(modality_type, 0.0)
            
            # Get modality config weight
            config = self.modality_configs.get(modality_type)
            weight = config.weight if config else 1.0
            
            # Combined contribution
            contributions[modality_type.value] = importance * weight
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {
                mod: contrib / total_contribution
                for mod, contrib in contributions.items()
            }
        
        return contributions
    
    def get_detector_summary(self) -> Dict[str, Any]:
        """Get comprehensive detector summary."""
        return {
            "detector_id": str(self.detector_id),
            "name": self.name,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "modalities": {
                modality.value: {
                    "encoding_type": config.encoding_type.value,
                    "weight": config.weight,
                    "is_required": config.is_required,
                    "feature_importance": self.feature_importance.get(modality, 0.0),
                }
                for modality, config in self.modality_configs.items()
            },
            "fusion_layers": len(self.fusion_layers),
            "output_dimension": self.output_dimension,
            "anomaly_threshold": self.anomaly_threshold,
            "confidence_threshold": self.confidence_threshold,
            "validation_metrics": {
                "accuracy": self.validation_metrics.accuracy,
                "precision": self.validation_metrics.precision,
                "recall": self.validation_metrics.recall,
                "f1_score": self.validation_metrics.f1_score,
            } if self.validation_metrics else None,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }