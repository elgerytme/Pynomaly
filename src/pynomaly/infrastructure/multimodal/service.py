"""Multi-modal anomaly detection service for coordinating diverse data types."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.multimodal import (
    EncodingType,
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
from pynomaly.infrastructure.multimodal.trainer import MultiModalTrainer


class MultiModalDetectionService:
    """Main service for multi-modal anomaly detection operations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Service components
        self.processor = MultiModalProcessor()
        self.trainer = MultiModalTrainer()
        
        # Detector registry
        self.detectors: Dict[UUID, MultiModalDetector] = {}
        self.detector_configs: Dict[UUID, Dict[str, Any]] = {}
        
        # Performance tracking
        self.detection_stats: Dict[str, Any] = {
            "total_detections": 0,
            "total_samples_processed": 0,
            "average_detection_time": 0.0,
            "anomaly_detection_rate": 0.0,
        }
        
        self.logger.info("Multi-modal detection service initialized")

    async def create_detector(
        self,
        name: str,
        modality_configs: Dict[ModalityType, ModalityConfig],
        fusion_strategy: FusionStrategy = FusionStrategy.EARLY_FUSION,
        output_dimension: int = 128,
    ) -> MultiModalDetector:
        """Create new multi-modal anomaly detector."""
        
        detector_id = uuid4()
        
        # Create encoders for each modality
        encoders = {}
        for modality_type, config in modality_configs.items():
            encoder = ModalityEncoder(
                encoder_id=uuid4(),
                modality_type=modality_type,
                encoding_type=config.encoding_type,
                config=config,
                feature_dimension=output_dimension // len(modality_configs),
            )
            encoders[modality_type] = encoder
        
        # Create fusion layers
        fusion_layers = []
        if fusion_strategy != FusionStrategy.EARLY_FUSION:
            fusion_layer = FusionLayer(
                fusion_id=uuid4(),
                fusion_strategy=fusion_strategy,
                input_modalities=list(modality_configs.keys()),
                output_dimension=output_dimension,
            )
            fusion_layers.append(fusion_layer)
        
        # Create detector
        detector = MultiModalDetector(
            detector_id=detector_id,
            name=name,
            modality_configs=modality_configs,
            encoders=encoders,
            fusion_layers=fusion_layers,
            output_dimension=output_dimension,
        )
        
        # Store detector
        self.detectors[detector_id] = detector
        self.detector_configs[detector_id] = {
            "fusion_strategy": fusion_strategy.value,
            "modality_count": len(modality_configs),
            "modality_types": [mod.value for mod in modality_configs.keys()],
        }
        
        self.logger.info(
            f"Created multi-modal detector '{name}' with "
            f"{len(modality_configs)} modalities: {list(modality_configs.keys())}"
        )
        
        return detector

    async def train_detector(
        self,
        detector_id: UUID,
        training_data: List[MultiModalData],
        validation_data: Optional[List[MultiModalData]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> MultiModalDetector:
        """Train multi-modal detector."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        
        # Train detector using trainer service
        trained_detector = await self.trainer.train_detector(
            detector, training_data, validation_data, training_config
        )
        
        # Update detector in registry
        self.detectors[detector_id] = trained_detector
        
        return trained_detector

    async def detect_anomaly(
        self,
        detector_id: UUID,
        sample: MultiModalData,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomaly in multi-modal sample."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Detect anomaly
            is_anomaly, score, details = detector.detect_anomaly(sample)
            
            # Update statistics
            detection_time = asyncio.get_event_loop().time() - start_time
            self._update_detection_stats(detection_time, is_anomaly)
            
            # Add performance info to details
            details["detection_time_ms"] = detection_time * 1000
            details["detector_id"] = str(detector_id)
            details["detector_name"] = detector.name
            
            return is_anomaly, score, details
            
        except Exception as e:
            self.logger.error(f"Error detecting anomaly: {e}")
            raise

    async def batch_detect_anomalies(
        self,
        detector_id: UUID,
        samples: List[MultiModalData],
        max_concurrent: int = 10,
    ) -> List[Tuple[str, bool, float, Dict[str, Any]]]:
        """Detect anomalies in batch of samples."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        self.logger.info(f"Processing batch of {len(samples)} samples")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def detect_sample(sample: MultiModalData):
            async with semaphore:
                try:
                    is_anomaly, score, details = await self.detect_anomaly(detector_id, sample)
                    return sample.sample_id, is_anomaly, score, details
                except Exception as e:
                    self.logger.error(f"Error processing sample {sample.sample_id}: {e}")
                    return sample.sample_id, False, 0.0, {"error": str(e)}
        
        # Process all samples concurrently
        tasks = [detect_sample(sample) for sample in samples]
        results = await asyncio.gather(*tasks)
        
        self.logger.info(f"Completed batch processing of {len(samples)} samples")
        
        return results

    async def create_text_image_detector(
        self,
        name: str,
        text_encoding: EncodingType = EncodingType.TFIDF,
        image_encoding: EncodingType = EncodingType.CNN,
        fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION,
    ) -> MultiModalDetector:
        """Create detector for text and image data."""
        
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=text_encoding,
                weight=0.6,  # Text slightly more important
                preprocessing_params={
                    "lowercase": True,
                    "remove_punctuation": True,
                    "max_length": 512,
                },
            ),
            ModalityType.IMAGE: ModalityConfig(
                modality_type=ModalityType.IMAGE,
                encoding_type=image_encoding,
                weight=0.4,
                preprocessing_params={
                    "resize": (224, 224),
                    "normalize": True,
                },
            ),
        }
        
        return await self.create_detector(
            name=name,
            modality_configs=modality_configs,
            fusion_strategy=fusion_strategy,
            output_dimension=256,
        )

    async def create_tabular_timeseries_detector(
        self,
        name: str,
        fusion_strategy: FusionStrategy = FusionStrategy.LATE_FUSION,
    ) -> MultiModalDetector:
        """Create detector for tabular and time series data."""
        
        modality_configs = {
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.5,
                preprocessing_params={
                    "normalize": True,
                    "handle_missing": True,
                },
            ),
            ModalityType.TIME_SERIES: ModalityConfig(
                modality_type=ModalityType.TIME_SERIES,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.5,
                preprocessing_params={
                    "window_size": 100,
                    "overlap": 0.5,
                    "normalization": "z_score",
                    "extract_features": True,
                },
            ),
        }
        
        return await self.create_detector(
            name=name,
            modality_configs=modality_configs,
            fusion_strategy=fusion_strategy,
            output_dimension=128,
        )

    async def create_multimodal_iot_detector(
        self,
        name: str,
        include_audio: bool = False,
        fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL_FUSION,
    ) -> MultiModalDetector:
        """Create detector for IoT multi-modal data."""
        
        modality_configs = {
            ModalityType.IOT_SENSOR: ModalityConfig(
                modality_type=ModalityType.IOT_SENSOR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.4,
                is_required=True,
            ),
            ModalityType.TIME_SERIES: ModalityConfig(
                modality_type=ModalityType.TIME_SERIES,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.3,
                is_required=True,
                preprocessing_params={
                    "window_size": 50,
                    "overlap": 0.3,
                    "normalization": "min_max",
                },
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.3,
                is_required=False,
            ),
        }
        
        if include_audio:
            modality_configs[ModalityType.AUDIO] = ModalityConfig(
                modality_type=ModalityType.AUDIO,
                encoding_type=EncodingType.MFCC,
                weight=0.2,
                is_required=False,
                preprocessing_params={
                    "sample_rate": 16000,
                    "normalize": True,
                },
            )
        
        return await self.create_detector(
            name=name,
            modality_configs=modality_configs,
            fusion_strategy=fusion_strategy,
            output_dimension=256,
        )

    async def evaluate_detector_performance(
        self,
        detector_id: UUID,
        test_data: List[MultiModalData],
    ) -> Dict[str, Any]:
        """Evaluate detector performance on test data."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        
        self.logger.info(f"Evaluating detector '{detector.name}' on {len(test_data)} samples")
        
        # Process test samples
        results = await self.batch_detect_anomalies(detector_id, test_data)
        
        # Calculate metrics
        true_labels = []
        predicted_labels = []
        anomaly_scores = []
        detection_times = []
        
        for sample_id, is_anomaly, score, details in results:
            # Find corresponding sample
            sample = next((s for s in test_data if s.sample_id == sample_id), None)
            if sample:
                true_label = sample.metadata.get("is_anomaly", False)
                true_labels.append(true_label)
                predicted_labels.append(is_anomaly)
                anomaly_scores.append(score)
                detection_times.append(details.get("detection_time_ms", 0))
        
        if not true_labels:
            return {"error": "No samples with ground truth labels"}
        
        # Calculate performance metrics
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        anomaly_scores = np.array(anomaly_scores)
        
        accuracy = np.mean(true_labels == predicted_labels)
        
        # Precision, Recall, F1
        tp = np.sum((true_labels == True) & (predicted_labels == True))
        fp = np.sum((true_labels == False) & (predicted_labels == True))
        fn = np.sum((true_labels == True) & (predicted_labels == False))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Performance metrics
        avg_detection_time = np.mean(detection_times)
        
        # Modality analysis
        modality_analysis = self._analyze_modality_performance(results, test_data)
        
        evaluation_result = {
            "detector_id": str(detector_id),
            "detector_name": detector.name,
            "test_samples": len(test_data),
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            },
            "timing_metrics": {
                "average_detection_time_ms": avg_detection_time,
                "total_evaluation_time_ms": sum(detection_times),
            },
            "anomaly_statistics": {
                "true_anomaly_rate": np.mean(true_labels),
                "predicted_anomaly_rate": np.mean(predicted_labels),
                "average_anomaly_score": np.mean(anomaly_scores),
            },
            "modality_analysis": modality_analysis,
        }
        
        self.logger.info(
            f"Evaluation completed: accuracy={accuracy:.3f}, "
            f"precision={precision:.3f}, recall={recall:.3f}, f1={f1_score:.3f}"
        )
        
        return evaluation_result

    def _analyze_modality_performance(
        self,
        results: List[Tuple[str, bool, float, Dict[str, Any]]],
        test_data: List[MultiModalData],
    ) -> Dict[str, Any]:
        """Analyze performance by modality combinations."""
        
        modality_combinations = {}
        
        for sample in test_data:
            available_modalities = tuple(sorted(sample.get_available_modalities()))
            
            if available_modalities not in modality_combinations:
                modality_combinations[available_modalities] = {
                    "count": 0,
                    "correct_predictions": 0,
                    "total_score": 0.0,
                }
            
            modality_combinations[available_modalities]["count"] += 1
            
            # Find corresponding result
            result = next((r for r in results if r[0] == sample.sample_id), None)
            if result:
                _, is_anomaly, score, _ = result
                true_label = sample.metadata.get("is_anomaly", False)
                
                if is_anomaly == true_label:
                    modality_combinations[available_modalities]["correct_predictions"] += 1
                
                modality_combinations[available_modalities]["total_score"] += score
        
        # Calculate accuracy for each combination
        analysis = {}
        for modalities, stats in modality_combinations.items():
            if stats["count"] > 0:
                accuracy = stats["correct_predictions"] / stats["count"]
                avg_score = stats["total_score"] / stats["count"]
                
                modality_names = [mod.value for mod in modalities]
                analysis[str(modality_names)] = {
                    "sample_count": stats["count"],
                    "accuracy": accuracy,
                    "average_score": avg_score,
                }
        
        return analysis

    async def get_detector_summary(self, detector_id: UUID) -> Dict[str, Any]:
        """Get comprehensive detector summary."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        config = self.detector_configs[detector_id]
        
        summary = detector.get_detector_summary()
        summary.update({
            "configuration": config,
            "training_history": self.trainer.get_training_history(detector_id),
            "validation_history": self.trainer.get_validation_history(detector_id),
        })
        
        return summary

    def get_available_modalities(self) -> List[str]:
        """Get list of supported modality types."""
        return [modality.value for modality in ModalityType]

    def get_available_encodings(self) -> List[str]:
        """Get list of supported encoding types."""
        return [encoding.value for encoding in EncodingType]

    def get_available_fusion_strategies(self) -> List[str]:
        """Get list of supported fusion strategies."""
        return [strategy.value for strategy in FusionStrategy]

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        
        detector_stats = {
            "total_detectors": len(self.detectors),
            "trained_detectors": sum(1 for d in self.detectors.values() if d.is_trained),
            "modality_distribution": {},
        }
        
        # Analyze modality distribution
        for detector in self.detectors.values():
            for modality in detector.modality_configs.keys():
                modality_name = modality.value
                detector_stats["modality_distribution"][modality_name] = (
                    detector_stats["modality_distribution"].get(modality_name, 0) + 1
                )
        
        return {
            "detector_statistics": detector_stats,
            "detection_statistics": self.detection_stats,
            "processor_statistics": self.processor.get_processing_stats(),
            "trainer_statistics": self.trainer.get_training_stats(),
            "supported_capabilities": {
                "modalities": self.get_available_modalities(),
                "encodings": self.get_available_encodings(),
                "fusion_strategies": self.get_available_fusion_strategies(),
            },
        }

    def _update_detection_stats(self, detection_time: float, is_anomaly: bool) -> None:
        """Update detection statistics."""
        
        self.detection_stats["total_detections"] += 1
        self.detection_stats["total_samples_processed"] += 1
        
        # Update average detection time
        total_time = (
            self.detection_stats["average_detection_time"] * 
            (self.detection_stats["total_detections"] - 1) + 
            detection_time
        )
        self.detection_stats["average_detection_time"] = total_time / self.detection_stats["total_detections"]
        
        # Update anomaly detection rate
        if is_anomaly:
            anomaly_count = (
                self.detection_stats["anomaly_detection_rate"] * 
                (self.detection_stats["total_detections"] - 1) + 1
            )
            self.detection_stats["anomaly_detection_rate"] = anomaly_count / self.detection_stats["total_detections"]
        else:
            self.detection_stats["anomaly_detection_rate"] = (
                self.detection_stats["anomaly_detection_rate"] * 
                (self.detection_stats["total_detections"] - 1) / 
                self.detection_stats["total_detections"]
            )

    async def cleanup_detectors(self, keep_trained: bool = True) -> int:
        """Clean up detectors to free memory."""
        
        cleanup_count = 0
        detectors_to_remove = []
        
        for detector_id, detector in self.detectors.items():
            if not keep_trained or not detector.is_trained:
                detectors_to_remove.append(detector_id)
        
        for detector_id in detectors_to_remove:
            del self.detectors[detector_id]
            if detector_id in self.detector_configs:
                del self.detector_configs[detector_id]
            cleanup_count += 1
        
        self.logger.info(f"Cleaned up {cleanup_count} detectors")
        
        return cleanup_count