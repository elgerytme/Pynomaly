"""Multi-modal data fusion for anomaly detection with transformer architectures."""

from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Types of data modalities."""
    
    IMAGE = "image"
    TIME_SERIES = "time_series"
    TEXT = "text"
    NUMERICAL = "numerical"
    AUDIO = "audio"
    VIDEO = "video"
    GRAPH = "graph"
    TABULAR = "tabular"


class FusionStrategy(str, Enum):
    """Multi-modal fusion strategies."""
    
    EARLY_FUSION = "early_fusion"  # Feature-level fusion
    LATE_FUSION = "late_fusion"    # Decision-level fusion
    HYBRID_FUSION = "hybrid_fusion"  # Combination of early and late
    ATTENTION_FUSION = "attention_fusion"  # Attention-based fusion
    TRANSFORMER_FUSION = "transformer_fusion"  # Transformer-based fusion
    GRAPH_FUSION = "graph_fusion"  # Graph neural network fusion


class AttentionType(str, Enum):
    """Types of attention mechanisms."""
    
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    ADDITIVE_ATTENTION = "additive_attention"


@dataclass
class ModalityConfig:
    """Configuration for a data modality."""
    
    modality_type: ModalityType
    input_shape: Tuple[int, ...]
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_extractor: str = "default"
    normalization: str = "standard"  # standard, minmax, robust, none
    augmentation_enabled: bool = False
    missing_value_strategy: str = "interpolation"  # interpolation, masking, imputation
    weight: float = 1.0  # Importance weight for this modality


@dataclass
class MultiModalSample:
    """Multi-modal data sample."""
    
    sample_id: str
    modalities: Dict[ModalityType, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    labels: Optional[Dict[str, Any]] = None


@dataclass
class FusionResult:
    """Result of multi-modal fusion."""
    
    fused_features: np.ndarray
    modality_weights: Dict[ModalityType, float]
    attention_scores: Optional[np.ndarray] = None
    intermediate_features: Dict[ModalityType, np.ndarray] = field(default_factory=dict)
    fusion_confidence: float = 0.0
    processing_time: float = 0.0


class FeatureExtractor(ABC):
    """Base class for modality-specific feature extractors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    async def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from modality data."""
        pass
    
    @abstractmethod
    async def fit(self, data: List[np.ndarray]) -> None:
        """Fit feature extractor on training data."""
        pass


class ImageFeatureExtractor(FeatureExtractor):
    """CNN-based image feature extractor."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feature_maps = None
        self.pooling_type = config.get("pooling", "global_average")
        self.pretrained_model = config.get("pretrained_model", "resnet50")
    
    async def fit(self, data: List[np.ndarray]) -> None:
        """Fit image feature extractor."""
        try:
            logger.info("Fitting image feature extractor")
            
            # In real implementation, this would train/fine-tune a CNN
            # For now, we'll simulate the fitting process
            
            sample_image = data[0] if data else np.random.random((224, 224, 3))
            self.input_shape = sample_image.shape
            
            # Simulate learned feature maps
            self.feature_maps = {
                "conv1": np.random.random((64, 112, 112)),
                "conv2": np.random.random((128, 56, 56)),
                "conv3": np.random.random((256, 28, 28)),
                "conv4": np.random.random((512, 14, 14)),
                "conv5": np.random.random((1024, 7, 7))
            }
            
            self.is_fitted = True
            logger.info("Image feature extractor fitted")
            
        except Exception as e:
            logger.error(f"Image feature extractor fitting failed: {e}")
            raise
    
    async def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract CNN features from image data."""
        try:
            if not self.is_fitted:
                raise ValueError("Feature extractor must be fitted first")
            
            # Simulate CNN feature extraction
            batch_size = len(data) if len(data.shape) == 4 else 1
            
            if len(data.shape) == 3:
                data = data.reshape(1, *data.shape)
            
            # Simulate conv layers
            features = []
            for img in data:
                # Simulate feature extraction through conv layers
                img_features = []
                
                # Global average pooling simulation
                for layer_name, feature_map in self.feature_maps.items():
                    if self.pooling_type == "global_average":
                        pooled = np.mean(feature_map, axis=(1, 2))  # Global average pooling
                    elif self.pooling_type == "global_max":
                        pooled = np.max(feature_map, axis=(1, 2))  # Global max pooling
                    else:
                        pooled = feature_map.flatten()[:512]  # Flatten and truncate
                    
                    img_features.extend(pooled)
                
                features.append(img_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}")
            return np.random.random((len(data), 512))  # Fallback
    
    async def extract_attention_maps(self, data: np.ndarray) -> np.ndarray:
        """Extract attention maps for interpretability."""
        try:
            # Simulate attention map extraction
            batch_size = len(data) if len(data.shape) == 4 else 1
            attention_maps = np.random.random((batch_size, 7, 7))  # 7x7 attention maps
            
            return attention_maps
            
        except Exception as e:
            logger.error(f"Attention map extraction failed: {e}")
            return np.random.random((1, 7, 7))


class TimeSeriesFeatureExtractor(FeatureExtractor):
    """Time series feature extractor with LSTM/Transformer support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config.get("sequence_length", 100)
        self.model_type = config.get("model_type", "lstm")  # lstm, gru, transformer
        self.feature_dim = config.get("feature_dim", 128)
        self.statistical_features = config.get("statistical_features", True)
    
    async def fit(self, data: List[np.ndarray]) -> None:
        """Fit time series feature extractor."""
        try:
            logger.info("Fitting time series feature extractor")
            
            # Analyze time series characteristics
            sample_ts = data[0] if data else np.random.random((100, 1))
            self.input_shape = sample_ts.shape
            
            # Compute statistical properties
            all_data = np.concatenate(data, axis=0) if data else sample_ts
            self.ts_stats = {
                "mean": np.mean(all_data, axis=0),
                "std": np.std(all_data, axis=0),
                "min": np.min(all_data, axis=0),
                "max": np.max(all_data, axis=0)
            }
            
            self.is_fitted = True
            logger.info("Time series feature extractor fitted")
            
        except Exception as e:
            logger.error(f"Time series feature extractor fitting failed: {e}")
            raise
    
    async def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from time series data."""
        try:
            if not self.is_fitted:
                raise ValueError("Feature extractor must be fitted first")
            
            features = []
            
            for ts in data:
                ts_features = []
                
                # Statistical features
                if self.statistical_features:
                    stat_features = await self._extract_statistical_features(ts)
                    ts_features.extend(stat_features)
                
                # Sequence modeling features
                if self.model_type == "lstm":
                    seq_features = await self._extract_lstm_features(ts)
                elif self.model_type == "transformer":
                    seq_features = await self._extract_transformer_features(ts)
                else:
                    seq_features = await self._extract_basic_sequence_features(ts)
                
                ts_features.extend(seq_features)
                features.append(ts_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Time series feature extraction failed: {e}")
            return np.random.random((len(data), self.feature_dim))
    
    async def _extract_statistical_features(self, ts: np.ndarray) -> List[float]:
        """Extract statistical features from time series."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(ts),
            np.std(ts),
            np.min(ts),
            np.max(ts),
            np.median(ts),
            np.percentile(ts, 25),
            np.percentile(ts, 75)
        ])
        
        # Trend and seasonality
        if len(ts) > 1:
            # Linear trend
            x = np.arange(len(ts))
            trend_coef = np.polyfit(x, ts.flatten(), 1)[0]
            features.append(trend_coef)
            
            # Autocorrelation
            if len(ts) > 10:
                autocorr = np.corrcoef(ts[:-1].flatten(), ts[1:].flatten())[0, 1]
                features.append(autocorr if not np.isnan(autocorr) else 0.0)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Spectral features (simplified)
        if len(ts) > 4:
            try:
                fft_vals = np.fft.fft(ts.flatten())
                power_spectrum = np.abs(fft_vals) ** 2
                # Dominant frequency
                dominant_freq = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                features.append(dominant_freq / len(ts))
                
                # Spectral centroid
                freqs = np.fft.fftfreq(len(ts))
                spectral_centroid = np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2])
                features.append(spectral_centroid if not np.isnan(spectral_centroid) else 0.0)
            except:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    async def _extract_lstm_features(self, ts: np.ndarray) -> List[float]:
        """Extract LSTM-based features."""
        # Simulate LSTM feature extraction
        # In real implementation, this would use a trained LSTM
        
        hidden_state = np.random.random(64)  # LSTM hidden state
        cell_state = np.random.random(64)    # LSTM cell state
        
        # Combine hidden and cell states
        lstm_features = np.concatenate([hidden_state, cell_state])
        
        return lstm_features.tolist()
    
    async def _extract_transformer_features(self, ts: np.ndarray) -> List[float]:
        """Extract Transformer-based features."""
        # Simulate Transformer feature extraction
        # In real implementation, this would use a trained Transformer
        
        # Self-attention simulation
        seq_len = len(ts)
        attention_weights = np.random.random((seq_len, seq_len))
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Weighted average of sequence
        attended_features = np.mean(attention_weights, axis=0)
        
        # Positional encoding simulation
        pos_encoding = np.sin(np.arange(seq_len) / 10000) + np.cos(np.arange(seq_len) / 10000)
        
        # Combine features
        transformer_features = np.concatenate([
            attended_features[:32] if len(attended_features) >= 32 else np.pad(attended_features, (0, 32-len(attended_features))),
            pos_encoding[:32] if len(pos_encoding) >= 32 else np.pad(pos_encoding, (0, 32-len(pos_encoding)))
        ])
        
        return transformer_features.tolist()
    
    async def _extract_basic_sequence_features(self, ts: np.ndarray) -> List[float]:
        """Extract basic sequence features."""
        # Moving averages
        if len(ts) >= 5:
            ma_5 = np.mean(ts[-5:])
            ma_10 = np.mean(ts[-10:]) if len(ts) >= 10 else ma_5
        else:
            ma_5 = ma_10 = np.mean(ts)
        
        # Volatility
        if len(ts) > 1:
            volatility = np.std(np.diff(ts.flatten()))
        else:
            volatility = 0.0
        
        # Recent trend
        if len(ts) >= 3:
            recent_trend = (ts[-1] - ts[-3]) / 3
        else:
            recent_trend = 0.0
        
        return [ma_5, ma_10, volatility, recent_trend]


class TextFeatureExtractor(FeatureExtractor):
    """Text feature extractor with transformer support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_length = config.get("max_length", 512)
        self.model_type = config.get("model_type", "tfidf")  # tfidf, bert, roberta
        self.vocab_size = config.get("vocab_size", 10000)
        self.embedding_dim = config.get("embedding_dim", 300)
    
    async def fit(self, data: List[str]) -> None:
        """Fit text feature extractor."""
        try:
            logger.info("Fitting text feature extractor")
            
            # Build vocabulary
            all_text = " ".join(data) if data else "sample text for fitting"
            words = all_text.lower().split()
            unique_words = list(set(words))
            
            self.vocab = {word: idx for idx, word in enumerate(unique_words[:self.vocab_size])}
            self.word_counts = {word: words.count(word) for word in unique_words}
            
            # Compute IDF for TF-IDF
            self.idf = {}
            for word in self.vocab:
                doc_count = sum(1 for text in data if word in text.lower())
                self.idf[word] = math.log(len(data) / (doc_count + 1))
            
            self.is_fitted = True
            logger.info("Text feature extractor fitted")
            
        except Exception as e:
            logger.error(f"Text feature extractor fitting failed: {e}")
            raise
    
    async def extract_features(self, data: List[str]) -> np.ndarray:
        """Extract features from text data."""
        try:
            if not self.is_fitted:
                raise ValueError("Feature extractor must be fitted first")
            
            features = []
            
            for text in data:
                if self.model_type == "tfidf":
                    text_features = await self._extract_tfidf_features(text)
                elif self.model_type in ["bert", "roberta"]:
                    text_features = await self._extract_transformer_features(text)
                else:
                    text_features = await self._extract_bow_features(text)
                
                features.append(text_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Text feature extraction failed: {e}")
            return np.random.random((len(data), self.embedding_dim))
    
    async def _extract_tfidf_features(self, text: str) -> List[float]:
        """Extract TF-IDF features."""
        words = text.lower().split()
        word_count = len(words)
        
        tfidf_vector = np.zeros(len(self.vocab))
        
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
                tf = words.count(word) / word_count
                idf = self.idf[word]
                tfidf_vector[idx] = tf * idf
        
        return tfidf_vector.tolist()
    
    async def _extract_transformer_features(self, text: str) -> List[float]:
        """Extract transformer-based features."""
        # Simulate transformer feature extraction
        # In real implementation, this would use BERT/RoBERTa
        
        # Tokenization simulation
        tokens = text.lower().split()[:self.max_length]
        
        # Simulate transformer embeddings
        transformer_features = np.random.random(self.embedding_dim)
        
        # Add positional information
        pos_weight = len(tokens) / self.max_length
        transformer_features = transformer_features * (1 + pos_weight)
        
        return transformer_features.tolist()
    
    async def _extract_bow_features(self, text: str) -> List[float]:
        """Extract bag-of-words features."""
        words = text.lower().split()
        bow_vector = np.zeros(len(self.vocab))
        
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
                bow_vector[idx] += 1
        
        # Normalize
        if np.sum(bow_vector) > 0:
            bow_vector = bow_vector / np.sum(bow_vector)
        
        return bow_vector.tolist()


class NumericalFeatureExtractor(FeatureExtractor):
    """Numerical data feature extractor."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.normalization = config.get("normalization", "standard")
        self.feature_engineering = config.get("feature_engineering", True)
        self.polynomial_degree = config.get("polynomial_degree", 2)
    
    async def fit(self, data: List[np.ndarray]) -> None:
        """Fit numerical feature extractor."""
        try:
            logger.info("Fitting numerical feature extractor")
            
            # Concatenate all data for statistics
            all_data = np.concatenate(data, axis=0) if data else np.random.random((100, 5))
            
            # Compute normalization parameters
            self.mean = np.mean(all_data, axis=0)
            self.std = np.std(all_data, axis=0)
            self.min = np.min(all_data, axis=0)
            self.max = np.max(all_data, axis=0)
            
            # Robust statistics
            self.median = np.median(all_data, axis=0)
            self.q25 = np.percentile(all_data, 25, axis=0)
            self.q75 = np.percentile(all_data, 75, axis=0)
            self.iqr = self.q75 - self.q25
            
            self.input_dim = all_data.shape[1]
            self.is_fitted = True
            
            logger.info("Numerical feature extractor fitted")
            
        except Exception as e:
            logger.error(f"Numerical feature extractor fitting failed: {e}")
            raise
    
    async def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from numerical data."""
        try:
            if not self.is_fitted:
                raise ValueError("Feature extractor must be fitted first")
            
            features = []
            
            for sample in data:
                sample_features = []
                
                # Normalized features
                normalized = await self._normalize_features(sample)
                sample_features.extend(normalized)
                
                # Engineered features
                if self.feature_engineering:
                    engineered = await self._engineer_features(sample)
                    sample_features.extend(engineered)
                
                features.append(sample_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Numerical feature extraction failed: {e}")
            return data  # Return original data as fallback
    
    async def _normalize_features(self, sample: np.ndarray) -> List[float]:
        """Normalize numerical features."""
        if self.normalization == "standard":
            # Z-score normalization
            normalized = (sample - self.mean) / (self.std + 1e-8)
        elif self.normalization == "minmax":
            # Min-max normalization
            normalized = (sample - self.min) / (self.max - self.min + 1e-8)
        elif self.normalization == "robust":
            # Robust normalization using IQR
            normalized = (sample - self.median) / (self.iqr + 1e-8)
        else:
            # No normalization
            normalized = sample
        
        return normalized.tolist()
    
    async def _engineer_features(self, sample: np.ndarray) -> List[float]:
        """Engineer additional features."""
        engineered = []
        
        # Polynomial features (degree 2)
        if self.polynomial_degree >= 2:
            for i in range(len(sample)):
                for j in range(i, len(sample)):
                    engineered.append(sample[i] * sample[j])
        
        # Statistical features
        engineered.extend([
            np.sum(sample),
            np.mean(sample),
            np.std(sample),
            np.min(sample),
            np.max(sample)
        ])
        
        # Ratios and differences
        if len(sample) >= 2:
            for i in range(len(sample) - 1):
                # Ratios
                if abs(sample[i+1]) > 1e-8:
                    engineered.append(sample[i] / sample[i+1])
                else:
                    engineered.append(0.0)
                
                # Differences
                engineered.append(sample[i] - sample[i+1])
        
        return engineered


class MultiModalFusionTransformer:
    """Transformer-based multi-modal fusion."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_heads = config.get("num_heads", 8)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_layers = config.get("num_layers", 6)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        self.max_seq_length = config.get("max_seq_length", 1000)
        
        self.modality_embeddings = {}
        self.positional_encodings = None
        self.is_fitted = False
    
    async def fit(self, multimodal_samples: List[MultiModalSample]) -> None:
        """Fit the transformer fusion model."""
        try:
            logger.info("Fitting multi-modal transformer")
            
            # Analyze modalities
            all_modalities = set()
            for sample in multimodal_samples:
                all_modalities.update(sample.modalities.keys())
            
            # Create modality embeddings
            for i, modality in enumerate(all_modalities):
                self.modality_embeddings[modality] = {
                    "embedding_id": i,
                    "projection_matrix": np.random.random((self.hidden_dim, self.hidden_dim)),
                    "layer_norm_weight": np.random.random(self.hidden_dim),
                    "layer_norm_bias": np.random.random(self.hidden_dim)
                }
            
            # Create positional encodings
            self.positional_encodings = self._create_positional_encoding(
                self.max_seq_length, self.hidden_dim
            )
            
            self.is_fitted = True
            logger.info("Multi-modal transformer fitted")
            
        except Exception as e:
            logger.error(f"Transformer fitting failed: {e}")
            raise
    
    async def fuse_modalities(self, sample: MultiModalSample) -> FusionResult:
        """Fuse modalities using transformer architecture."""
        try:
            if not self.is_fitted:
                raise ValueError("Transformer must be fitted first")
            
            start_time = datetime.now()
            
            # Prepare input sequences
            sequences = []
            modality_types = []
            
            for modality_type, data in sample.modalities.items():
                if modality_type in self.modality_embeddings:
                    # Project to hidden dimension
                    projected = await self._project_to_hidden_dim(data, modality_type)
                    sequences.append(projected)
                    modality_types.append(modality_type)
            
            if not sequences:
                raise ValueError("No valid modalities found")
            
            # Concatenate sequences
            combined_sequence = np.concatenate(sequences, axis=0)
            seq_length = len(combined_sequence)
            
            # Add positional encoding
            if seq_length <= self.max_seq_length:
                pos_encoded = combined_sequence + self.positional_encodings[:seq_length]
            else:
                pos_encoded = combined_sequence[:self.max_seq_length] + self.positional_encodings
                seq_length = self.max_seq_length
            
            # Multi-head self-attention
            attention_output, attention_weights = await self._multi_head_attention(
                pos_encoded, pos_encoded, pos_encoded
            )
            
            # Feed-forward network
            ffn_output = await self._feed_forward_network(attention_output)
            
            # Global pooling
            fused_features = np.mean(ffn_output, axis=0)
            
            # Compute modality weights
            modality_weights = await self._compute_modality_weights(
                attention_weights, sequences, modality_types
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FusionResult(
                fused_features=fused_features,
                modality_weights=modality_weights,
                attention_scores=attention_weights,
                intermediate_features={mt: seq for mt, seq in zip(modality_types, sequences)},
                fusion_confidence=np.mean(attention_weights),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transformer fusion failed: {e}")
            # Return fallback result
            return FusionResult(
                fused_features=np.random.random(self.hidden_dim),
                modality_weights={modality: 1.0/len(sample.modalities) for modality in sample.modalities},
                fusion_confidence=0.5
            )
    
    async def _project_to_hidden_dim(self, data: np.ndarray, modality_type: ModalityType) -> np.ndarray:
        """Project modality data to hidden dimension."""
        try:
            modality_config = self.modality_embeddings[modality_type]
            projection_matrix = modality_config["projection_matrix"]
            
            # Flatten data if needed
            if len(data.shape) > 1:
                flattened = data.flatten()
            else:
                flattened = data
            
            # Pad or truncate to match projection matrix input size
            if len(flattened) > projection_matrix.shape[1]:
                flattened = flattened[:projection_matrix.shape[1]]
            elif len(flattened) < projection_matrix.shape[1]:
                flattened = np.pad(flattened, (0, projection_matrix.shape[1] - len(flattened)))
            
            # Project to hidden dimension
            projected = np.dot(flattened, projection_matrix.T)
            
            # Layer normalization
            layer_norm_weight = modality_config["layer_norm_weight"]
            layer_norm_bias = modality_config["layer_norm_bias"]
            
            mean = np.mean(projected)
            std = np.std(projected) + 1e-8
            normalized = (projected - mean) / std
            normalized = normalized * layer_norm_weight + layer_norm_bias
            
            return normalized.reshape(1, -1)  # Shape: (1, hidden_dim)
            
        except Exception as e:
            logger.error(f"Projection failed for {modality_type}: {e}")
            return np.random.random((1, self.hidden_dim))
    
    async def _multi_head_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute multi-head self-attention."""
        try:
            seq_length, hidden_dim = query.shape
            head_dim = hidden_dim // self.num_heads
            
            # Split into multiple heads
            query_heads = query.reshape(seq_length, self.num_heads, head_dim)
            key_heads = key.reshape(seq_length, self.num_heads, head_dim)
            value_heads = value.reshape(seq_length, self.num_heads, head_dim)
            
            # Compute attention for each head
            attention_outputs = []
            attention_weights_all = []
            
            for h in range(self.num_heads):
                q_h = query_heads[:, h, :]  # (seq_length, head_dim)
                k_h = key_heads[:, h, :]    # (seq_length, head_dim)
                v_h = value_heads[:, h, :]  # (seq_length, head_dim)
                
                # Scaled dot-product attention
                scores = np.dot(q_h, k_h.T) / math.sqrt(head_dim)  # (seq_length, seq_length)
                attention_weights = self._softmax(scores)
                attention_output = np.dot(attention_weights, v_h)  # (seq_length, head_dim)
                
                attention_outputs.append(attention_output)
                attention_weights_all.append(attention_weights)
            
            # Concatenate heads
            concatenated = np.concatenate(attention_outputs, axis=1)  # (seq_length, hidden_dim)
            
            # Average attention weights across heads
            avg_attention_weights = np.mean(attention_weights_all, axis=0)
            
            return concatenated, avg_attention_weights
            
        except Exception as e:
            logger.error(f"Multi-head attention failed: {e}")
            return query, np.eye(len(query))
    
    async def _feed_forward_network(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward network."""
        try:
            # Two-layer MLP with ReLU activation
            intermediate_dim = self.hidden_dim * 4
            
            # First layer
            W1 = np.random.random((self.hidden_dim, intermediate_dim))
            b1 = np.random.random(intermediate_dim)
            hidden = np.maximum(0, np.dot(x, W1) + b1)  # ReLU activation
            
            # Second layer
            W2 = np.random.random((intermediate_dim, self.hidden_dim))
            b2 = np.random.random(self.hidden_dim)
            output = np.dot(hidden, W2) + b2
            
            # Residual connection and layer norm
            output = x + output  # Residual connection
            
            # Layer normalization
            mean = np.mean(output, axis=1, keepdims=True)
            std = np.std(output, axis=1, keepdims=True) + 1e-8
            normalized = (output - mean) / std
            
            return normalized
            
        except Exception as e:
            logger.error(f"Feed-forward network failed: {e}")
            return x
    
    def _create_positional_encoding(self, max_length: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    async def _compute_modality_weights(
        self,
        attention_weights: np.ndarray,
        sequences: List[np.ndarray],
        modality_types: List[ModalityType]
    ) -> Dict[ModalityType, float]:
        """Compute importance weights for each modality."""
        try:
            weights = {}
            current_pos = 0
            
            for i, (seq, modality_type) in enumerate(zip(sequences, modality_types)):
                seq_length = len(seq)
                
                # Extract attention weights for this modality
                modality_attention = attention_weights[current_pos:current_pos + seq_length, :]
                modality_weight = np.mean(modality_attention)
                
                weights[modality_type] = float(modality_weight)
                current_pos += seq_length
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"Modality weight computation failed: {e}")
            return {modality: 1.0 / len(modality_types) for modality in modality_types}


class MultiModalAnomalyDetector:
    """Multi-modal anomaly detection system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modality_configs: Dict[ModalityType, ModalityConfig] = {}
        self.feature_extractors: Dict[ModalityType, FeatureExtractor] = {}
        self.fusion_strategy = FusionStrategy(config.get("fusion_strategy", "transformer_fusion"))
        self.transformer_fusion = None
        self.is_fitted = False
        
        # Anomaly detection parameters
        self.anomaly_threshold = config.get("anomaly_threshold", 0.8)
        self.baseline_features = None
        self.feature_statistics = {}
    
    async def add_modality(self, modality_config: ModalityConfig) -> None:
        """Add a data modality to the detector."""
        try:
            self.modality_configs[modality_config.modality_type] = modality_config
            
            # Initialize appropriate feature extractor
            if modality_config.modality_type == ModalityType.IMAGE:
                self.feature_extractors[modality_config.modality_type] = ImageFeatureExtractor(
                    self.config.get("image_config", {})
                )
            elif modality_config.modality_type == ModalityType.TIME_SERIES:
                self.feature_extractors[modality_config.modality_type] = TimeSeriesFeatureExtractor(
                    self.config.get("timeseries_config", {})
                )
            elif modality_config.modality_type == ModalityType.TEXT:
                self.feature_extractors[modality_config.modality_type] = TextFeatureExtractor(
                    self.config.get("text_config", {})
                )
            elif modality_config.modality_type == ModalityType.NUMERICAL:
                self.feature_extractors[modality_config.modality_type] = NumericalFeatureExtractor(
                    self.config.get("numerical_config", {})
                )
            
            logger.info(f"Added modality: {modality_config.modality_type}")
            
        except Exception as e:
            logger.error(f"Failed to add modality: {e}")
            raise
    
    async def fit(self, training_samples: List[MultiModalSample]) -> None:
        """Fit the multi-modal anomaly detector."""
        try:
            logger.info("Fitting multi-modal anomaly detector")
            
            # Fit feature extractors for each modality
            for modality_type, extractor in self.feature_extractors.items():
                modality_data = []
                for sample in training_samples:
                    if modality_type in sample.modalities:
                        modality_data.append(sample.modalities[modality_type])
                
                if modality_data:
                    await extractor.fit(modality_data)
                    logger.info(f"Fitted feature extractor for {modality_type}")
            
            # Fit fusion model
            if self.fusion_strategy == FusionStrategy.TRANSFORMER_FUSION:
                self.transformer_fusion = MultiModalFusionTransformer(
                    self.config.get("transformer_config", {})
                )
                await self.transformer_fusion.fit(training_samples)
            
            # Extract baseline features for anomaly detection
            await self._extract_baseline_features(training_samples)
            
            self.is_fitted = True
            logger.info("Multi-modal anomaly detector fitted successfully")
            
        except Exception as e:
            logger.error(f"Multi-modal detector fitting failed: {e}")
            raise
    
    async def detect_anomalies(self, samples: List[MultiModalSample]) -> List[Dict[str, Any]]:
        """Detect anomalies in multi-modal samples."""
        try:
            if not self.is_fitted:
                raise ValueError("Detector must be fitted before anomaly detection")
            
            anomaly_results = []
            
            for sample in samples:
                result = await self._detect_sample_anomaly(sample)
                anomaly_results.append(result)
            
            logger.info(f"Processed {len(samples)} samples for anomaly detection")
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _detect_sample_anomaly(self, sample: MultiModalSample) -> Dict[str, Any]:
        """Detect anomaly in a single multi-modal sample."""
        try:
            # Extract features from each modality
            modality_features = {}
            for modality_type, data in sample.modalities.items():
                if modality_type in self.feature_extractors:
                    features = await self.feature_extractors[modality_type].extract_features(
                        np.array([data])
                    )
                    modality_features[modality_type] = features[0]
            
            if not modality_features:
                return {
                    "sample_id": sample.sample_id,
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "explanation": "No valid modalities found"
                }
            
            # Fuse modalities
            if self.fusion_strategy == FusionStrategy.TRANSFORMER_FUSION and self.transformer_fusion:
                fusion_result = await self.transformer_fusion.fuse_modalities(sample)
                fused_features = fusion_result.fused_features
                modality_weights = fusion_result.modality_weights
            else:
                # Simple concatenation fusion
                fused_features = np.concatenate(list(modality_features.values()))
                modality_weights = {mt: 1.0/len(modality_features) for mt in modality_features.keys()}
            
            # Compute anomaly score
            anomaly_score = await self._compute_anomaly_score(fused_features)
            
            # Determine if anomaly
            is_anomaly = anomaly_score > self.anomaly_threshold
            
            # Generate explanation
            explanation = await self._generate_explanation(
                sample, modality_features, modality_weights, anomaly_score
            )
            
            return {
                "sample_id": sample.sample_id,
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "modality_weights": {str(k): float(v) for k, v in modality_weights.items()},
                "modality_features": {str(k): v.tolist() for k, v in modality_features.items()},
                "fused_features": fused_features.tolist(),
                "explanation": explanation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sample anomaly detection failed: {e}")
            return {
                "sample_id": sample.sample_id,
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "error": str(e)
            }
    
    async def _extract_baseline_features(self, training_samples: List[MultiModalSample]) -> None:
        """Extract baseline features for anomaly detection."""
        try:
            all_fused_features = []
            
            for sample in training_samples:
                # Extract and fuse features
                modality_features = {}
                for modality_type, data in sample.modalities.items():
                    if modality_type in self.feature_extractors:
                        features = await self.feature_extractors[modality_type].extract_features(
                            np.array([data])
                        )
                        modality_features[modality_type] = features[0]
                
                if modality_features:
                    if self.fusion_strategy == FusionStrategy.TRANSFORMER_FUSION and self.transformer_fusion:
                        fusion_result = await self.transformer_fusion.fuse_modalities(sample)
                        fused_features = fusion_result.fused_features
                    else:
                        fused_features = np.concatenate(list(modality_features.values()))
                    
                    all_fused_features.append(fused_features)
            
            if all_fused_features:
                all_fused_features = np.array(all_fused_features)
                
                # Compute statistics
                self.feature_statistics = {
                    "mean": np.mean(all_fused_features, axis=0),
                    "std": np.std(all_fused_features, axis=0),
                    "min": np.min(all_fused_features, axis=0),
                    "max": np.max(all_fused_features, axis=0),
                    "median": np.median(all_fused_features, axis=0),
                    "q25": np.percentile(all_fused_features, 25, axis=0),
                    "q75": np.percentile(all_fused_features, 75, axis=0)
                }
                
                self.baseline_features = all_fused_features
                logger.info("Extracted baseline features for anomaly detection")
            
        except Exception as e:
            logger.error(f"Baseline feature extraction failed: {e}")
    
    async def _compute_anomaly_score(self, features: np.ndarray) -> float:
        """Compute anomaly score for fused features."""
        try:
            if self.baseline_features is None:
                return 0.5  # Default score
            
            # Mahalanobis distance-based scoring
            mean = self.feature_statistics["mean"]
            std = self.feature_statistics["std"]
            
            # Normalize features
            normalized_features = (features - mean) / (std + 1e-8)
            
            # Compute distance from baseline
            distances = []
            for baseline_sample in self.baseline_features:
                normalized_baseline = (baseline_sample - mean) / (std + 1e-8)
                distance = np.linalg.norm(normalized_features - normalized_baseline)
                distances.append(distance)
            
            # Use minimum distance as anomaly score
            min_distance = min(distances)
            
            # Normalize to [0, 1] range
            max_distance = np.percentile(distances, 95)  # 95th percentile as max
            anomaly_score = min(min_distance / (max_distance + 1e-8), 1.0)
            
            return anomaly_score
            
        except Exception as e:
            logger.error(f"Anomaly score computation failed: {e}")
            return 0.5
    
    async def _generate_explanation(
        self,
        sample: MultiModalSample,
        modality_features: Dict[ModalityType, np.ndarray],
        modality_weights: Dict[ModalityType, float],
        anomaly_score: float
    ) -> str:
        """Generate explanation for anomaly detection result."""
        try:
            explanations = []
            
            # Sort modalities by importance weight
            sorted_modalities = sorted(modality_weights.items(), key=lambda x: x[1], reverse=True)
            
            if anomaly_score > self.anomaly_threshold:
                explanations.append(f"ANOMALY DETECTED (score: {anomaly_score:.3f})")
                explanations.append("Contributing factors:")
                
                for modality, weight in sorted_modalities:
                    explanations.append(f"  - {modality.value}: {weight:.3f} importance")
            else:
                explanations.append(f"Normal sample (score: {anomaly_score:.3f})")
                explanations.append("Modality contributions:")
                
                for modality, weight in sorted_modalities:
                    explanations.append(f"  - {modality.value}: {weight:.3f}")
            
            return " | ".join(explanations)
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return f"Anomaly score: {anomaly_score:.3f}"
    
    async def get_modality_statistics(self) -> Dict[str, Any]:
        """Get statistics about modalities and features."""
        try:
            stats = {
                "configured_modalities": [mt.value for mt in self.modality_configs.keys()],
                "feature_extractors": [mt.value for mt in self.feature_extractors.keys()],
                "fusion_strategy": self.fusion_strategy.value,
                "is_fitted": self.is_fitted
            }
            
            if self.feature_statistics:
                stats["feature_dimensions"] = len(self.feature_statistics["mean"])
                stats["baseline_samples"] = len(self.baseline_features) if self.baseline_features is not None else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {}