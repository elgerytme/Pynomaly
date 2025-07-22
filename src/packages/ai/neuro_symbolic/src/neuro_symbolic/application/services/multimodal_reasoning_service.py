"""Multi-modal reasoning service for neuro-symbolic AI."""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from ...domain.value_objects.multimodal_data import (
    MultiModalBatch, MultiModalSample, MultiModalResult,
    ModalityInfo, ModalityType, ModalityWeight,
    FusionConfiguration, FusionLevel
)
from ...domain.value_objects.neuro_symbolic_result import NeuroSymbolicResult
from ...infrastructure.error_handling import (
    NeuroSymbolicError, ValidationError, InferenceError, DataError,
    InputValidator, error_handler, setup_error_logging
)
from .neuro_symbolic_reasoning_service import NeuroSymbolicReasoningService


class MultiModalReasoningService:
    """
    Service for multi-modal reasoning combining multiple data types
    with neuro-symbolic analysis.
    """
    
    def __init__(self, base_service: Optional[NeuroSymbolicReasoningService] = None):
        """Initialize multi-modal reasoning service."""
        self.logger = setup_error_logging()
        self.base_service = base_service or NeuroSymbolicReasoningService()
        self.validator = InputValidator()
        
        # Multi-modal fusion configurations
        self.fusion_configs: Dict[str, FusionConfiguration] = {}
        
        # Modality-specific processors
        self.modality_processors = {
            ModalityType.NUMERICAL: self._process_numerical_modality,
            ModalityType.TEXT: self._process_text_modality,
            ModalityType.IMAGE: self._process_image_modality,
            ModalityType.TIME_SERIES: self._process_time_series_modality,
            ModalityType.CATEGORICAL: self._process_categorical_modality,
            ModalityType.SENSOR: self._process_sensor_modality,
            ModalityType.LOG: self._process_log_modality,
            ModalityType.METADATA: self._process_metadata_modality
        }
        
        self.logger.info("MultiModalReasoningService initialized successfully")
    
    @error_handler(reraise=True)
    def analyze_multimodal_data(
        self,
        model_id: str,
        multimodal_batch: MultiModalBatch,
        fusion_config: Optional[FusionConfiguration] = None,
        include_modality_explanations: bool = True,
        task_type: str = "classification",
        **kwargs
    ) -> MultiModalResult:
        """Analyze multi-modal data for insights and patterns."""
        
        # Validate inputs
        model_id = self.validator.validate_model_id(model_id)
        self._validate_multimodal_batch(multimodal_batch)
        
        # Use provided fusion config or default
        if fusion_config is None:
            fusion_config = self._create_default_fusion_config(multimodal_batch)
        
        self._validate_fusion_config(fusion_config, multimodal_batch)
        
        self.logger.info(
            f"Starting multi-modal analysis for model '{model_id}' "
            f"with {multimodal_batch.batch_size} samples and "
            f"{len(multimodal_batch.modality_names)} modalities"
        )
        
        start_time = datetime.now()
        
        try:
            # Process each modality
            modality_features = {}
            modality_scores = {}
            
            for modality_name in multimodal_batch.modality_names:
                modality_info = multimodal_batch.modality_info[modality_name]
                modality_data = multimodal_batch.get_modality_data(modality_name)
                
                # Process modality-specific features
                processed_features = self._process_modality(
                    modality_data, modality_info, model_id
                )
                modality_features[modality_name] = processed_features
                
                # Get modality-specific analysis scores
                scores = self._compute_modality_analysis(
                    processed_features, modality_info, model_id
                )
                modality_scores[modality_name] = scores
            
            # Fuse modality scores
            fused_scores, fusion_weights, attention_scores = self._fuse_modality_scores(
                modality_scores, fusion_config
            )
            
            # Convert to predictions  
            predictions = self._scores_to_classifications(fused_scores, task_type)
            
            # Calculate modality contributions
            modality_contributions = self._calculate_modality_contributions(
                modality_scores, fusion_weights, predictions
            )
            
            # Generate multi-modal explanations if requested
            if include_modality_explanations:
                explanation_metadata = self._generate_modality_explanations(
                    multimodal_batch, modality_scores, modality_contributions,
                    fusion_config, model_id
                )
            else:
                explanation_metadata = {}
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = MultiModalResult(
                predictions=predictions,
                confidence_scores=fused_scores,
                modality_contributions=modality_contributions,
                fusion_weights=fusion_weights,
                attention_scores=attention_scores,
                processing_metadata={
                    'model_id': model_id,
                    'batch_size': multimodal_batch.batch_size,
                    'modalities': multimodal_batch.modality_names,
                    'fusion_method': fusion_config.fusion_method,
                    'processing_time': processing_time,
                    **explanation_metadata
                }
            )
            
            self.logger.info(
                f"Multi-modal analysis completed: processed {len(predictions)} samples"
            )
            
            return result
            
        except Exception as e:
            raise InferenceError(
                f"Multi-modal analysis failed for model '{model_id}': {str(e)}",
                cause=e,
                remediation="Check modality data compatibility and fusion configuration"
            )
    
    def create_fusion_configuration(
        self,
        modality_names: List[str],
        fusion_level: FusionLevel = FusionLevel.LATE,
        fusion_method: str = "attention",
        adaptive_weighting: bool = True,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> FusionConfiguration:
        """Create a fusion configuration for multi-modal processing."""
        
        if not modality_names:
            raise ValidationError("At least one modality must be specified")
        
        # Create modality weights
        modality_weights = []
        
        if custom_weights:
            # Use provided weights
            for name in modality_names:
                weight = custom_weights.get(name, 1.0 / len(modality_names))
                modality_weights.append(ModalityWeight(
                    modality_name=name,
                    weight=weight,
                    adaptive=adaptive_weighting
                ))
        else:
            # Equal weights by default
            equal_weight = 1.0 / len(modality_names)
            for name in modality_names:
                modality_weights.append(ModalityWeight(
                    modality_name=name,
                    weight=equal_weight,
                    adaptive=adaptive_weighting
                ))
        
        return FusionConfiguration(
            fusion_level=fusion_level,
            modality_weights=modality_weights,
            fusion_method=fusion_method,
            adaptive_weighting=adaptive_weighting
        )
    
    def _validate_multimodal_batch(self, batch: MultiModalBatch) -> None:
        """Validate multi-modal batch."""
        if batch.batch_size == 0:
            raise ValidationError("Batch cannot be empty")
        
        if len(batch.modality_names) == 0:
            raise ValidationError("At least one modality must be present")
        
        # Check for consistent sample IDs
        sample_ids = [sample.sample_id for sample in batch.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValidationError("Sample IDs must be unique within batch")
    
    def _validate_fusion_config(
        self, 
        config: FusionConfiguration, 
        batch: MultiModalBatch
    ) -> None:
        """Validate fusion configuration against batch."""
        config_modalities = set(w.modality_name for w in config.modality_weights)
        batch_modalities = set(batch.modality_names)
        
        missing = config_modalities - batch_modalities
        if missing:
            raise ValidationError(
                f"Fusion config references missing modalities: {missing}",
                remediation="Update fusion config to match available modalities"
            )
    
    def _create_default_fusion_config(self, batch: MultiModalBatch) -> FusionConfiguration:
        """Create default fusion configuration for a batch."""
        return self.create_fusion_configuration(
            modality_names=batch.modality_names,
            fusion_level=FusionLevel.LATE,
            fusion_method="attention",
            adaptive_weighting=True
        )
    
    def _process_modality(
        self,
        modality_data: List[Any],
        modality_info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process data for a specific modality."""
        
        processor = self.modality_processors.get(modality_info.modality_type)
        if processor is None:
            raise ValidationError(
                f"No processor available for modality type: {modality_info.modality_type}",
                remediation="Implement processor for this modality type"
            )
        
        try:
            return processor(modality_data, modality_info, model_id)
        except Exception as e:
            raise DataError(
                f"Failed to process modality '{modality_info.name}': {str(e)}",
                cause=e
            )
    
    def _process_numerical_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process numerical modality data."""
        # Convert to numpy array
        array_data = np.array(data, dtype=np.float32)
        
        # Validate shape
        if len(array_data.shape) < 2:
            array_data = array_data.reshape(-1, 1)
        
        # Apply normalization if specified in preprocessing info
        if info.preprocessing_info and 'normalization' in info.preprocessing_info:
            norm_type = info.preprocessing_info['normalization']
            if norm_type == 'standard':
                mean = np.mean(array_data, axis=0)
                std = np.std(array_data, axis=0) + 1e-8
                array_data = (array_data - mean) / std
            elif norm_type == 'minmax':
                min_val = np.min(array_data, axis=0)
                max_val = np.max(array_data, axis=0)
                array_data = (array_data - min_val) / (max_val - min_val + 1e-8)
        
        return array_data
    
    def _process_text_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process text modality data."""
        # Simple text processing - convert to numerical features
        features = []
        
        for text_item in data:
            if isinstance(text_item, str):
                # Basic text features
                text_features = [
                    len(text_item),  # Length
                    len(text_item.split()),  # Word count
                    len(set(text_item.lower())),  # Unique characters
                    text_item.count('.'),  # Sentence count (approx)
                    sum(1 for c in text_item if c.isupper()),  # Uppercase count
                ]
            else:
                # Assume already processed (e.g., embeddings)
                text_features = list(text_item)
            
            features.append(text_features)
        
        return np.array(features, dtype=np.float32)
    
    def _process_image_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process image modality data."""
        # Assume images are already processed into feature vectors
        # In practice, this would involve CNN feature extraction
        try:
            image_features = np.array(data, dtype=np.float32)
            if len(image_features.shape) > 2:
                # Flatten if needed
                image_features = image_features.reshape(image_features.shape[0], -1)
            return image_features
        except Exception:
            # Fallback: create dummy features based on shape info
            return np.random.randn(len(data), np.prod(info.shape[1:])).astype(np.float32)
    
    def _process_time_series_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process time series modality data."""
        time_series_data = np.array(data, dtype=np.float32)
        
        # Extract time series features
        features = []
        for series in time_series_data:
            ts_features = [
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.median(series),
                len(series),
                np.sum(np.diff(series) > 0),  # Increasing points
                np.sum(np.diff(series) < 0),  # Decreasing points
            ]
            features.append(ts_features)
        
        return np.array(features, dtype=np.float32)
    
    def _process_categorical_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process categorical modality data."""
        # Simple one-hot encoding
        unique_values = list(set(data))
        features = []
        
        for item in data:
            one_hot = [1.0 if item == val else 0.0 for val in unique_values]
            features.append(one_hot)
        
        return np.array(features, dtype=np.float32)
    
    def _process_sensor_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process sensor modality data."""
        # Similar to numerical but with sensor-specific preprocessing
        return self._process_numerical_modality(data, info, model_id)
    
    def _process_log_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process log modality data."""
        # Extract log-specific features
        features = []
        
        for log_entry in data:
            if isinstance(log_entry, str):
                log_features = [
                    len(log_entry),
                    log_entry.count('ERROR'),
                    log_entry.count('WARNING'),
                    log_entry.count('INFO'),
                    log_entry.count('DEBUG'),
                    1.0 if 'exception' in log_entry.lower() else 0.0,
                ]
            else:
                log_features = list(log_entry)
            
            features.append(log_features)
        
        return np.array(features, dtype=np.float32)
    
    def _process_metadata_modality(
        self,
        data: List[Any],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Process metadata modality data."""
        # Convert metadata to numerical features
        features = []
        
        for metadata in data:
            if isinstance(metadata, dict):
                meta_features = [
                    len(metadata),  # Number of fields
                    sum(1 for v in metadata.values() if isinstance(v, str)),
                    sum(1 for v in metadata.values() if isinstance(v, (int, float))),
                ]
            else:
                meta_features = [0, 0, 0]
            
            features.append(meta_features)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_modality_analysis(
        self,
        features: NDArray[np.floating],
        info: ModalityInfo,
        model_id: str
    ) -> NDArray[np.floating]:
        """Compute analysis scores for a single modality."""
        try:
            # Use the base reasoning service for individual modality analysis
            result = self.base_service.perform_reasoning(
                model_id=model_id,
                data=features,
                include_causal_explanations=False,
                include_counterfactuals=False
            )
            return result.confidence_scores
            
        except Exception:
            # Fallback: simple statistical scoring
            z_scores = np.abs((features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8))
            return np.mean(z_scores, axis=1)
    
    def _fuse_modality_scores(
        self,
        modality_scores: Dict[str, NDArray[np.floating]],
        fusion_config: FusionConfiguration
    ) -> Tuple[NDArray[np.floating], Dict[str, float], Optional[Dict[str, NDArray[np.floating]]]]:
        """Fuse scores from multiple modalities."""
        
        num_samples = len(next(iter(modality_scores.values())))
        fusion_method = fusion_config.fusion_method
        
        # Get weights
        weights = {}
        for weight_config in fusion_config.modality_weights:
            weights[weight_config.modality_name] = weight_config.weight
        
        if fusion_method == "weighted_average":
            fused_scores = np.zeros(num_samples)
            for modality, scores in modality_scores.items():
                weight = weights.get(modality, 0.0)
                fused_scores += weight * scores
            return fused_scores, weights, None
            
        elif fusion_method == "attention":
            # Simplified attention mechanism
            attention_scores = {}
            fused_scores = np.zeros(num_samples)
            
            for i in range(num_samples):
                sample_scores = [modality_scores[mod][i] for mod in modality_scores.keys()]
                # Softmax attention
                attention_weights = np.exp(np.array(sample_scores) / fusion_config.temperature)
                attention_weights /= np.sum(attention_weights)
                
                # Store attention scores
                for j, modality in enumerate(modality_scores.keys()):
                    if modality not in attention_scores:
                        attention_scores[modality] = np.zeros(num_samples)
                    attention_scores[modality][i] = attention_weights[j]
                
                # Compute weighted score
                fused_scores[i] = np.sum(attention_weights * sample_scores)
            
            return fused_scores, weights, attention_scores
            
        elif fusion_method == "maximum":
            fused_scores = np.zeros(num_samples)
            for i in range(num_samples):
                sample_scores = [modality_scores[mod][i] for mod in modality_scores.keys()]
                fused_scores[i] = np.max(sample_scores)
            return fused_scores, weights, None
            
        else:
            # Default to weighted average
            return self._fuse_modality_scores(
                modality_scores,
                fusion_config._replace(fusion_method="weighted_average")
            )
    
    def _scores_to_classifications(self, scores: NDArray[np.floating], task_type: str = "classification") -> NDArray[np.floating]:
        """Convert analysis scores to classifications."""
        if task_type == "binary_classification":
            threshold = np.percentile(scores, 90)  # Top 10% as positive class
            return np.where(scores > threshold, 1, 0).astype(np.floating)
        elif task_type == "regression":
            return scores  # Return continuous scores
        else:  # general classification
            return scores  # Return continuous scores
    
    def _calculate_modality_contributions(
        self,
        modality_scores: Dict[str, NDArray[np.floating]],
        fusion_weights: Dict[str, float],
        predictions: NDArray[np.integer]
    ) -> Dict[str, NDArray[np.floating]]:
        """Calculate how much each modality contributed to final predictions."""
        contributions = {}
        
        for modality, scores in modality_scores.items():
            weight = fusion_weights.get(modality, 0.0)
            # Contribution is weighted score normalized by total possible contribution
            contributions[modality] = scores * weight
        
        return contributions
    
    def _generate_modality_explanations(
        self,
        batch: MultiModalBatch,
        modality_scores: Dict[str, NDArray[np.floating]],
        contributions: Dict[str, NDArray[np.floating]],
        fusion_config: FusionConfiguration,
        model_id: str
    ) -> Dict[str, Any]:
        """Generate explanations for multi-modal predictions."""
        
        explanations = {
            'modality_rankings': [],
            'fusion_explanation': {
                'method': fusion_config.fusion_method,
                'level': fusion_config.fusion_level.value,
                'adaptive': fusion_config.adaptive_weighting
            },
            'modality_summaries': {}
        }
        
        # Per-sample modality rankings
        num_samples = len(next(iter(contributions.values())))
        for i in range(num_samples):
            sample_ranking = []
            for modality in contributions.keys():
                sample_ranking.append({
                    'modality': modality,
                    'contribution': float(contributions[modality][i]),
                    'raw_score': float(modality_scores[modality][i])
                })
            
            sample_ranking.sort(key=lambda x: x['contribution'], reverse=True)
            explanations['modality_rankings'].append(sample_ranking)
        
        # Modality summaries
        for modality in contributions.keys():
            modality_info = batch.modality_info[modality]
            explanations['modality_summaries'][modality] = {
                'type': modality_info.modality_type.value,
                'shape': modality_info.shape,
                'avg_contribution': float(np.mean(contributions[modality])),
                'max_contribution': float(np.max(contributions[modality])),
                'description': modality_info.description
            }
        
        return explanations