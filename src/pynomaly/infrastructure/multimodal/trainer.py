"""Multi-modal detector training and optimization service."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.multimodal import (
    FusionLayer,
    FusionStrategy,
    ModalityConfig,
    ModalityEncoder,
    ModalityType,
    MultiModalData,
    MultiModalDetector,
)
from pynomaly.domain.value_objects import ModelMetrics
from pynomaly.infrastructure.multimodal.processor import MultiModalProcessor


class MultiModalTrainer:
    """Service for training multi-modal anomaly detectors."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processor = MultiModalProcessor()
        
        # Training state
        self.training_history: Dict[UUID, List[Dict[str, Any]]] = {}
        self.validation_history: Dict[UUID, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.training_stats: Dict[str, Any] = {
            "total_training_sessions": 0,
            "total_training_time": 0.0,
            "average_training_time": 0.0,
        }

    async def train_detector(
        self,
        detector: MultiModalDetector,
        training_data: List[MultiModalData],
        validation_data: Optional[List[MultiModalData]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> MultiModalDetector:
        """Train multi-modal anomaly detector."""
        
        training_config = training_config or {}
        
        self.logger.info(
            f"Starting training for detector '{detector.name}' with "
            f"{len(training_data)} samples"
        )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate training data
            await self._validate_training_data(detector, training_data)
            
            # Initialize training history
            self.training_history[detector.detector_id] = []
            if validation_data:
                self.validation_history[detector.detector_id] = []
            
            # Train encoders for each modality
            await self._train_modality_encoders(detector, training_data)
            
            # Extract features from training data
            training_features = await self._extract_batch_features(detector, training_data)
            
            # Train fusion layers
            await self._train_fusion_layers(detector, training_features)
            
            # Train anomaly detection model
            await self._train_anomaly_model(detector, training_features, training_config)
            
            # Validation
            if validation_data:
                validation_metrics = await self._validate_detector(detector, validation_data)
                detector.validation_metrics = validation_metrics
            
            # Finalize training
            detector.is_trained = True
            detector.training_samples = len(training_data)
            detector.last_updated = asyncio.get_event_loop().time()
            
            # Update feature importance
            await self._calculate_feature_importance(detector, training_features)
            
            training_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.info(
                f"Training completed for detector '{detector.name}' in {training_time:.2f}s"
            )
            
            # Update stats
            self._update_training_stats(training_time)
            
            return detector
            
        except Exception as e:
            self.logger.error(f"Training failed for detector '{detector.name}': {e}")
            raise

    async def _validate_training_data(
        self, 
        detector: MultiModalDetector, 
        training_data: List[MultiModalData]
    ) -> None:
        """Validate training data compatibility with detector."""
        
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        required_modalities = detector.get_required_modalities()
        
        # Check that at least some samples have required modalities
        valid_samples = 0
        for sample in training_data:
            if sample.is_complete(required_modalities):
                valid_samples += 1
        
        if valid_samples == 0:
            raise ValueError("No training samples contain all required modalities")
        
        if valid_samples < len(training_data) * 0.5:
            self.logger.warning(
                f"Only {valid_samples}/{len(training_data)} samples have all required modalities"
            )

    async def _train_modality_encoders(
        self, 
        detector: MultiModalDetector, 
        training_data: List[MultiModalData]
    ) -> None:
        """Train encoders for each modality."""
        
        self.logger.info("Training modality encoders...")
        
        for modality_type, encoder in detector.encoders.items():
            if encoder.is_trained:
                continue
            
            self.logger.debug(f"Training encoder for {modality_type.value}")
            
            # Collect data for this modality
            modality_data = []
            for sample in training_data:
                if sample.has_modality(modality_type):
                    data = sample.get_modality_data(modality_type)
                    modality_data.append(data)
            
            if not modality_data:
                self.logger.warning(f"No training data for modality {modality_type.value}")
                continue
            
            # Train encoder based on modality type
            await self._train_single_encoder(encoder, modality_data, modality_type)
            
            encoder.is_trained = True
            
            self.logger.debug(f"Completed training encoder for {modality_type.value}")

    async def _train_single_encoder(
        self,
        encoder: ModalityEncoder,
        data: List[Any],
        modality_type: ModalityType,
    ) -> None:
        """Train a single modality encoder."""
        
        # Convert data to appropriate format
        if modality_type in [ModalityType.TABULAR, ModalityType.TIME_SERIES]:
            # Numerical data
            if all(isinstance(d, np.ndarray) for d in data):
                combined_data = np.vstack(data)
            else:
                combined_data = np.array(data)
            
            # Train statistical parameters
            if encoder.encoding_type.value in ["standardization", "normalization"]:
                encoder.encoder_state["mean"] = np.mean(combined_data, axis=0)
                encoder.encoder_state["std"] = np.std(combined_data, axis=0)
                encoder.encoder_state["min"] = np.min(combined_data, axis=0)
                encoder.encoder_state["max"] = np.max(combined_data, axis=0)
        
        elif modality_type == ModalityType.TEXT:
            # Text data
            all_text = " ".join(data) if isinstance(data[0], str) else ""
            
            # Build vocabulary
            words = all_text.split()
            vocab = list(set(words))
            
            encoder.encoder_state["vocabulary"] = vocab[:1000]  # Limit vocab size
            encoder.encoder_state["word_counts"] = {
                word: words.count(word) for word in encoder.encoder_state["vocabulary"]
            }
        
        elif modality_type == ModalityType.IMAGE:
            # Image data
            if data and isinstance(data[0], np.ndarray):
                # Calculate image statistics
                all_images = np.stack(data)
                encoder.encoder_state["mean_pixel"] = np.mean(all_images)
                encoder.encoder_state["std_pixel"] = np.std(all_images)
        
        # Store training parameters
        encoder.encoder_state["training_samples"] = len(data)
        encoder.encoder_state["feature_stats"] = {
            "data_type": str(type(data[0])),
            "sample_count": len(data),
        }

    async def _extract_batch_features(
        self, 
        detector: MultiModalDetector, 
        data: List[MultiModalData]
    ) -> List[Dict[ModalityType, np.ndarray]]:
        """Extract features from batch of multi-modal data."""
        
        self.logger.info(f"Extracting features from {len(data)} samples...")
        
        batch_features = []
        
        # Process samples concurrently
        tasks = []
        for sample in data:
            if detector.can_process_sample(sample):
                task = self.processor.process_multimodal_sample(sample, detector)
                tasks.append(task)
        
        # Wait for all processing to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error processing sample: {result}")
                continue
            
            batch_features.append(result)
        
        self.logger.info(f"Successfully extracted features from {len(batch_features)} samples")
        
        return batch_features

    async def _train_fusion_layers(
        self, 
        detector: MultiModalDetector, 
        training_features: List[Dict[ModalityType, np.ndarray]]
    ) -> None:
        """Train fusion layers."""
        
        if not detector.fusion_layers:
            self.logger.info("No fusion layers to train")
            return
        
        self.logger.info(f"Training {len(detector.fusion_layers)} fusion layers...")
        
        for i, fusion_layer in enumerate(detector.fusion_layers):
            self.logger.debug(f"Training fusion layer {i}: {fusion_layer.fusion_strategy.value}")
            
            # Collect relevant features for this layer
            layer_features = []
            
            for sample_features in training_features:
                relevant_features = {
                    mod: features for mod, features in sample_features.items()
                    if mod in fusion_layer.input_modalities
                }
                
                if relevant_features:
                    layer_features.append(relevant_features)
            
            if not layer_features:
                self.logger.warning(f"No features for fusion layer {i}")
                continue
            
            # Train fusion parameters
            await self._train_fusion_layer(fusion_layer, layer_features)
            
            self.logger.debug(f"Completed training fusion layer {i}")

    async def _train_fusion_layer(
        self,
        fusion_layer: FusionLayer,
        features: List[Dict[ModalityType, np.ndarray]],
    ) -> None:
        """Train a single fusion layer."""
        
        if fusion_layer.fusion_strategy == FusionStrategy.ADAPTIVE_FUSION:
            # Learn adaptive weights based on feature quality
            modality_qualities = {mod: [] for mod in fusion_layer.input_modalities}
            
            for sample_features in features:
                for modality, modality_features in sample_features.items():
                    if modality in fusion_layer.input_modalities:
                        # Simple quality metric: inverse variance
                        quality = 1.0 / (1.0 + np.var(modality_features))
                        modality_qualities[modality].append(quality)
            
            # Update fusion weights based on average quality
            total_quality = 0.0
            for modality in fusion_layer.input_modalities:
                if modality_qualities[modality]:
                    avg_quality = np.mean(modality_qualities[modality])
                    fusion_layer.fusion_weights[modality] = avg_quality
                    total_quality += avg_quality
            
            # Normalize weights
            if total_quality > 0:
                for modality in fusion_layer.input_modalities:
                    fusion_layer.fusion_weights[modality] /= total_quality
        
        elif fusion_layer.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            # Learn attention parameters (simplified)
            attention_scores = {mod: [] for mod in fusion_layer.input_modalities}
            
            for sample_features in features:
                # Calculate attention scores based on feature magnitudes
                total_magnitude = 0.0
                
                for modality, modality_features in sample_features.items():
                    if modality in fusion_layer.input_modalities:
                        magnitude = np.linalg.norm(modality_features)
                        attention_scores[modality].append(magnitude)
                        total_magnitude += magnitude
                
                # Normalize attention scores for this sample
                if total_magnitude > 0:
                    for modality in attention_scores:
                        if attention_scores[modality]:
                            attention_scores[modality][-1] /= total_magnitude
            
            # Average attention scores
            for modality in fusion_layer.input_modalities:
                if attention_scores[modality]:
                    avg_attention = np.mean(attention_scores[modality])
                    fusion_layer.fusion_params[f"attention_{modality.value}"] = avg_attention
        
        # Store training information
        fusion_layer.fusion_params["training_samples"] = len(features)
        fusion_layer.fusion_params["trained"] = True

    async def _train_anomaly_model(
        self,
        detector: MultiModalDetector,
        training_features: List[Dict[ModalityType, np.ndarray]],
        training_config: Dict[str, Any],
    ) -> None:
        """Train the anomaly detection model."""
        
        self.logger.info("Training anomaly detection model...")
        
        # Extract final fused features
        fused_features = []
        
        for sample_features in training_features:
            # Create temporary sample for feature extraction
            temp_sample = MultiModalData(
                sample_id="temp",
                modalities={mod: features for mod, features in sample_features.items()},
            )
            
            # This is a simplified approach - in practice, we'd use the actual fusion pipeline
            try:
                final_features = self._apply_fusion_layers(sample_features, detector.fusion_layers)
                fused_features.append(final_features)
            except Exception as e:
                self.logger.error(f"Error fusing features: {e}")
                continue
        
        if not fused_features:
            raise ValueError("No valid fused features for training")
        
        fused_features = np.array(fused_features)
        
        # Simple anomaly detection model training (placeholder)
        # In practice, this would train a proper ML model
        
        # Calculate statistics for anomaly detection
        feature_mean = np.mean(fused_features, axis=0)
        feature_std = np.std(fused_features, axis=0)
        feature_median = np.median(fused_features, axis=0)
        
        # Store model parameters
        detector.model_weights["feature_mean"] = feature_mean
        detector.model_weights["feature_std"] = feature_std
        detector.model_weights["feature_median"] = feature_median
        
        # Calculate threshold based on training data distribution
        # Use 95th percentile of distances as threshold
        distances = []
        for features in fused_features:
            distance = np.linalg.norm(features - feature_mean)
            distances.append(distance)
        
        threshold = np.percentile(distances, 95)
        detector.anomaly_threshold = threshold
        
        # Training configuration
        learning_rate = training_config.get("learning_rate", 0.001)
        epochs = training_config.get("epochs", 100)
        
        # Simple iterative refinement (placeholder for actual training)
        for epoch in range(min(epochs, 10)):  # Limit for testing
            # Calculate loss (simplified)
            total_loss = 0.0
            
            for features in fused_features:
                # Reconstruction loss (simplified)
                reconstructed = feature_mean  # Placeholder
                loss = np.mean((features - reconstructed) ** 2)
                total_loss += loss
            
            avg_loss = total_loss / len(fused_features)
            
            # Store training history
            epoch_history = {
                "epoch": epoch,
                "loss": avg_loss,
                "threshold": detector.anomaly_threshold,
            }
            
            self.training_history[detector.detector_id].append(epoch_history)
            
            if epoch % 10 == 0:
                self.logger.debug(f"Epoch {epoch}: loss = {avg_loss:.4f}")
        
        self.logger.info("Completed anomaly detection model training")

    def _apply_fusion_layers(
        self,
        sample_features: Dict[ModalityType, np.ndarray],
        fusion_layers: List[FusionLayer],
    ) -> np.ndarray:
        """Apply fusion layers to sample features."""
        
        current_features = sample_features.copy()
        
        for fusion_layer in fusion_layers:
            # Get relevant features for this layer
            relevant_features = {
                mod: features for mod, features in current_features.items()
                if mod in fusion_layer.input_modalities
            }
            
            if not relevant_features:
                continue
            
            # Apply fusion
            fused_output = fusion_layer.fuse(relevant_features)
            
            # Update current features (simplified)
            # Remove input modalities and add fused output
            for mod in fusion_layer.input_modalities:
                if mod in current_features:
                    del current_features[mod]
            
            # Add fused output with a fusion key
            fusion_key = f"fused_{fusion_layer.fusion_id}"
            current_features[fusion_key] = fused_output
        
        # Concatenate remaining features
        if len(current_features) == 1:
            return next(iter(current_features.values()))
        else:
            feature_list = list(current_features.values())
            return np.concatenate(feature_list)

    async def _validate_detector(
        self, 
        detector: MultiModalDetector, 
        validation_data: List[MultiModalData]
    ) -> ModelMetrics:
        """Validate trained detector."""
        
        self.logger.info(f"Validating detector with {len(validation_data)} samples...")
        
        true_labels = []
        predicted_labels = []
        anomaly_scores = []
        
        for sample in validation_data:
            if not detector.can_process_sample(sample):
                continue
            
            try:
                # Get true label (if available)
                true_label = sample.metadata.get("is_anomaly", False)
                true_labels.append(true_label)
                
                # Predict
                is_anomaly, score, _ = detector.detect_anomaly(sample)
                predicted_labels.append(is_anomaly)
                anomaly_scores.append(score)
                
            except Exception as e:
                self.logger.error(f"Error validating sample {sample.sample_id}: {e}")
                continue
        
        if not true_labels:
            self.logger.warning("No validation samples with labels")
            return ModelMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0)
        
        # Calculate metrics
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        anomaly_scores = np.array(anomaly_scores)
        
        # Basic metrics
        accuracy = np.mean(true_labels == predicted_labels)
        
        # Precision, Recall, F1
        true_positives = np.sum((true_labels == True) & (predicted_labels == True))
        false_positives = np.sum((true_labels == False) & (predicted_labels == True))
        false_negatives = np.sum((true_labels == True) & (predicted_labels == False))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC (simplified)
        if len(np.unique(true_labels)) > 1:
            # Sort by anomaly score
            sorted_indices = np.argsort(anomaly_scores)
            sorted_labels = true_labels[sorted_indices]
            
            # Calculate AUC using trapezoidal rule (simplified)
            auc_roc = self._calculate_simple_auc(sorted_labels)
        else:
            auc_roc = 0.5
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            training_time=0.0,  # Will be set by calling function
            inference_time=0.0,  # Placeholder
            model_size=1024,  # Placeholder
            roc_auc=auc_roc,
        )
        
        # Store validation history
        validation_history = {
            "validation_samples": len(true_labels),
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auc_roc": auc_roc,
            },
        }
        
        self.validation_history[detector.detector_id].append(validation_history)
        
        self.logger.info(
            f"Validation completed: accuracy={accuracy:.3f}, "
            f"precision={precision:.3f}, recall={recall:.3f}, f1={f1_score:.3f}"
        )
        
        return metrics

    def _calculate_simple_auc(self, sorted_labels: np.ndarray) -> float:
        """Calculate simple AUC approximation."""
        positive_count = np.sum(sorted_labels)
        negative_count = len(sorted_labels) - positive_count
        
        if positive_count == 0 or negative_count == 0:
            return 0.5
        
        # Count correctly ranked pairs
        correct_pairs = 0
        
        for i, label in enumerate(sorted_labels):
            if label:  # Positive sample
                # Count negatives ranked lower
                correct_pairs += np.sum(sorted_labels[:i] == False)
        
        total_pairs = positive_count * negative_count
        auc = correct_pairs / total_pairs if total_pairs > 0 else 0.5
        
        return auc

    async def _calculate_feature_importance(
        self,
        detector: MultiModalDetector,
        training_features: List[Dict[ModalityType, np.ndarray]],
    ) -> None:
        """Calculate feature importance for each modality."""
        
        self.logger.info("Calculating feature importance...")
        
        modality_contributions = {mod: [] for mod in detector.modality_configs.keys()}
        
        for sample_features in training_features:
            total_magnitude = 0.0
            sample_magnitudes = {}
            
            # Calculate magnitude for each modality
            for modality, features in sample_features.items():
                magnitude = np.linalg.norm(features)
                sample_magnitudes[modality] = magnitude
                total_magnitude += magnitude
            
            # Normalize contributions
            if total_magnitude > 0:
                for modality, magnitude in sample_magnitudes.items():
                    contribution = magnitude / total_magnitude
                    modality_contributions[modality].append(contribution)
        
        # Calculate average importance
        for modality in detector.modality_configs.keys():
            if modality_contributions[modality]:
                avg_importance = np.mean(modality_contributions[modality])
                detector.feature_importance[modality] = avg_importance
            else:
                detector.feature_importance[modality] = 0.0
        
        self.logger.info(f"Feature importance: {detector.feature_importance}")

    def _update_training_stats(self, training_time: float) -> None:
        """Update training statistics."""
        self.training_stats["total_training_sessions"] += 1
        self.training_stats["total_training_time"] += training_time
        self.training_stats["average_training_time"] = (
            self.training_stats["total_training_time"] / 
            self.training_stats["total_training_sessions"]
        )

    def get_training_history(self, detector_id: UUID) -> Optional[List[Dict[str, Any]]]:
        """Get training history for detector."""
        return self.training_history.get(detector_id)

    def get_validation_history(self, detector_id: UUID) -> Optional[List[Dict[str, Any]]]:
        """Get validation history for detector."""
        return self.validation_history.get(detector_id)

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            "training_stats": self.training_stats,
            "processor_stats": self.processor.get_processing_stats(),
            "active_training_sessions": len(self.training_history),
        }