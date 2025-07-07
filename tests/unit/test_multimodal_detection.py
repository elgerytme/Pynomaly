"""Unit tests for multi-modal anomaly detection components."""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

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
from pynomaly.infrastructure.multimodal.service import MultiModalDetectionService
from pynomaly.infrastructure.multimodal.trainer import MultiModalTrainer


class TestModalityConfig:
    """Test modality configuration."""

    def test_modality_config_creation(self):
        """Test creating modality configuration."""
        config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
            weight=0.8,
            is_required=True,
        )
        
        assert config.modality_type == ModalityType.TEXT
        assert config.encoding_type == EncodingType.TFIDF
        assert config.weight == 0.8
        assert config.is_required is True
        assert config.preprocessing_params["lowercase"] is True

    def test_modality_config_invalid_weight(self):
        """Test invalid weight raises error."""
        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=1.5,
            )

    def test_modality_config_default_preprocessing(self):
        """Test default preprocessing parameters."""
        text_config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
        )
        assert "lowercase" in text_config.preprocessing_params
        
        image_config = ModalityConfig(
            modality_type=ModalityType.IMAGE,
            encoding_type=EncodingType.CNN,
        )
        assert "resize" in image_config.preprocessing_params
        
        ts_config = ModalityConfig(
            modality_type=ModalityType.TIME_SERIES,
            encoding_type=EncodingType.STANDARDIZATION,
        )
        assert "window_size" in ts_config.preprocessing_params


class TestMultiModalData:
    """Test multi-modal data container."""

    def test_multimodal_data_creation(self):
        """Test creating multi-modal data."""
        modalities = {
            ModalityType.TEXT: "This is test text",
            ModalityType.TABULAR: np.array([1, 2, 3, 4, 5]),
        }
        
        data = MultiModalData(
            sample_id="test_sample",
            modalities=modalities,
            metadata={"is_anomaly": False},
        )
        
        assert data.sample_id == "test_sample"
        assert data.has_modality(ModalityType.TEXT)
        assert data.has_modality(ModalityType.TABULAR)
        assert not data.has_modality(ModalityType.IMAGE)
        assert data.get_modality_data(ModalityType.TEXT) == "This is test text"

    def test_available_modalities(self):
        """Test getting available modalities."""
        modalities = {
            ModalityType.TEXT: "text data",
            ModalityType.IMAGE: np.random.rand(64, 64, 3),
        }
        
        data = MultiModalData(sample_id="test", modalities=modalities)
        available = data.get_available_modalities()
        
        assert ModalityType.TEXT in available
        assert ModalityType.IMAGE in available
        assert len(available) == 2

    def test_is_complete(self):
        """Test checking if sample is complete."""
        modalities = {
            ModalityType.TEXT: "text",
            ModalityType.TABULAR: np.array([1, 2, 3]),
        }
        
        data = MultiModalData(sample_id="test", modalities=modalities)
        
        # Complete with required modalities
        required = {ModalityType.TEXT, ModalityType.TABULAR}
        assert data.is_complete(required)
        
        # Incomplete with additional required modality
        required_with_extra = {ModalityType.TEXT, ModalityType.TABULAR, ModalityType.IMAGE}
        assert not data.is_complete(required_with_extra)


class TestModalityEncoder:
    """Test modality encoder."""

    def test_encoder_creation(self):
        """Test creating modality encoder."""
        config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
        )
        
        encoder = ModalityEncoder(
            encoder_id=uuid4(),
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
            config=config,
            feature_dimension=100,
        )
        
        assert encoder.modality_type == ModalityType.TEXT
        assert encoder.encoding_type == EncodingType.TFIDF
        assert encoder.feature_dimension == 100
        assert not encoder.is_trained

    def test_encoder_invalid_dimension(self):
        """Test invalid feature dimension."""
        config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
        )
        
        with pytest.raises(ValueError, match="Feature dimension must be positive"):
            ModalityEncoder(
                encoder_id=uuid4(),
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                config=config,
                feature_dimension=0,
            )

    def test_text_encoding(self):
        """Test text encoding."""
        config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
        )
        
        encoder = ModalityEncoder(
            encoder_id=uuid4(),
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
            config=config,
            feature_dimension=100,
            is_trained=True,  # Skip training requirement for unit test
        )
        
        features = encoder.encode("This is test text for encoding")
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 100
        assert features.dtype == np.float32

    def test_tabular_encoding(self):
        """Test tabular encoding."""
        config = ModalityConfig(
            modality_type=ModalityType.TABULAR,
            encoding_type=EncodingType.STANDARDIZATION,
        )
        
        encoder = ModalityEncoder(
            encoder_id=uuid4(),
            modality_type=ModalityType.TABULAR,
            encoding_type=EncodingType.STANDARDIZATION,
            config=config,
            feature_dimension=50,
        )
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = encoder.encode(data)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32

    def test_image_encoding(self):
        """Test image encoding."""
        config = ModalityConfig(
            modality_type=ModalityType.IMAGE,
            encoding_type=EncodingType.CNN,
        )
        
        encoder = ModalityEncoder(
            encoder_id=uuid4(),
            modality_type=ModalityType.IMAGE,
            encoding_type=EncodingType.CNN,
            config=config,
            feature_dimension=512,
            is_trained=True,
        )
        
        # Create test image
        image = np.random.rand(64, 64, 3) * 255
        features = encoder.encode(image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 512
        assert 0 <= features.max() <= 1  # Should be normalized


class TestFusionLayer:
    """Test fusion layer."""

    def test_fusion_layer_creation(self):
        """Test creating fusion layer."""
        fusion_layer = FusionLayer(
            fusion_id=uuid4(),
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            input_modalities=[ModalityType.TEXT, ModalityType.TABULAR],
            output_dimension=128,
        )
        
        assert fusion_layer.fusion_strategy == FusionStrategy.EARLY_FUSION
        assert len(fusion_layer.input_modalities) == 2
        assert fusion_layer.output_dimension == 128
        assert len(fusion_layer.fusion_weights) == 2
        
        # Check equal weights initialization
        for weight in fusion_layer.fusion_weights.values():
            assert abs(weight - 0.5) < 1e-6

    def test_early_fusion(self):
        """Test early fusion strategy."""
        fusion_layer = FusionLayer(
            fusion_id=uuid4(),
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            input_modalities=[ModalityType.TEXT, ModalityType.TABULAR],
            output_dimension=100,
        )
        
        modality_features = {
            ModalityType.TEXT: np.random.rand(50),
            ModalityType.TABULAR: np.random.rand(50),
        }
        
        fused = fusion_layer.fuse(modality_features)
        
        assert isinstance(fused, np.ndarray)
        assert len(fused) == 100

    def test_late_fusion(self):
        """Test late fusion strategy."""
        fusion_layer = FusionLayer(
            fusion_id=uuid4(),
            fusion_strategy=FusionStrategy.LATE_FUSION,
            input_modalities=[ModalityType.TEXT, ModalityType.TABULAR],
            output_dimension=64,
        )
        
        modality_features = {
            ModalityType.TEXT: np.random.rand(64),
            ModalityType.TABULAR: np.random.rand(64),
        }
        
        fused = fusion_layer.fuse(modality_features)
        
        assert isinstance(fused, np.ndarray)
        assert len(fused) == 64

    def test_attention_fusion(self):
        """Test attention fusion strategy."""
        fusion_layer = FusionLayer(
            fusion_id=uuid4(),
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            input_modalities=[ModalityType.TEXT, ModalityType.TABULAR],
            output_dimension=64,
        )
        
        modality_features = {
            ModalityType.TEXT: np.random.rand(64),
            ModalityType.TABULAR: np.random.rand(64),
        }
        
        fused = fusion_layer.fuse(modality_features)
        
        assert isinstance(fused, np.ndarray)
        assert len(fused) == 64


class TestMultiModalDetector:
    """Test multi-modal detector."""

    def create_test_detector(self) -> MultiModalDetector:
        """Create test detector for testing."""
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=0.6,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.4,
            ),
        }
        
        encoders = {}
        for modality, config in modality_configs.items():
            encoder = ModalityEncoder(
                encoder_id=uuid4(),
                modality_type=modality,
                encoding_type=config.encoding_type,
                config=config,
                feature_dimension=64,
                is_trained=True,
            )
            encoders[modality] = encoder
        
        fusion_layer = FusionLayer(
            fusion_id=uuid4(),
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            input_modalities=list(modality_configs.keys()),
            output_dimension=128,
        )
        
        return MultiModalDetector(
            detector_id=uuid4(),
            name="test_detector",
            modality_configs=modality_configs,
            encoders=encoders,
            fusion_layers=[fusion_layer],
            output_dimension=128,
            is_trained=True,
        )

    def test_detector_creation(self):
        """Test creating multi-modal detector."""
        detector = self.create_test_detector()
        
        assert detector.name == "test_detector"
        assert len(detector.modality_configs) == 2
        assert len(detector.encoders) == 2
        assert len(detector.fusion_layers) == 1
        assert detector.output_dimension == 128
        assert detector.is_trained

    def test_detector_validation(self):
        """Test detector validation."""
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
            ),
        }
        
        # Missing encoder should raise error
        with pytest.raises(ValueError, match="Missing encoder"):
            MultiModalDetector(
                detector_id=uuid4(),
                name="invalid_detector",
                modality_configs=modality_configs,
                encoders={},  # Missing encoders
                fusion_layers=[],
                output_dimension=128,
            )

    def test_required_modalities(self):
        """Test getting required modalities."""
        detector = self.create_test_detector()
        
        required = detector.get_required_modalities()
        optional = detector.get_optional_modalities()
        
        assert ModalityType.TEXT in required
        assert ModalityType.TABULAR in required
        assert len(optional) == 0  # All are required by default

    def test_can_process_sample(self):
        """Test checking if detector can process sample."""
        detector = self.create_test_detector()
        
        # Complete sample
        complete_sample = MultiModalData(
            sample_id="complete",
            modalities={
                ModalityType.TEXT: "test text",
                ModalityType.TABULAR: np.array([1, 2, 3]),
            },
        )
        assert detector.can_process_sample(complete_sample)
        
        # Incomplete sample
        incomplete_sample = MultiModalData(
            sample_id="incomplete",
            modalities={
                ModalityType.TEXT: "test text",
            },
        )
        assert not detector.can_process_sample(incomplete_sample)

    def test_feature_extraction(self):
        """Test feature extraction."""
        detector = self.create_test_detector()
        
        sample = MultiModalData(
            sample_id="test",
            modalities={
                ModalityType.TEXT: "test text for extraction",
                ModalityType.TABULAR: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            },
        )
        
        features = detector.extract_features(sample)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == detector.output_dimension

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        detector = self.create_test_detector()
        
        sample = MultiModalData(
            sample_id="test",
            modalities={
                ModalityType.TEXT: "this is normal text",
                ModalityType.TABULAR: np.array([1.0, 2.0, 3.0]),
            },
        )
        
        is_anomaly, score, details = detector.detect_anomaly(sample)
        
        assert isinstance(is_anomaly, bool)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(details, dict)
        assert "anomaly_score" in details
        assert "confidence" in details
        assert "modalities_used" in details

    def test_detector_summary(self):
        """Test getting detector summary."""
        detector = self.create_test_detector()
        
        summary = detector.get_detector_summary()
        
        assert "detector_id" in summary
        assert "name" in summary
        assert "is_trained" in summary
        assert "modalities" in summary
        assert "output_dimension" in summary


@pytest.mark.asyncio
class TestMultiModalProcessor:
    """Test multi-modal processor."""

    def test_processor_creation(self):
        """Test creating processor."""
        processor = MultiModalProcessor()
        
        assert isinstance(processor.processing_stats, dict)
        assert isinstance(processor.processing_cache, dict)

    async def test_tabular_processing(self):
        """Test tabular data processing."""
        processor = MultiModalProcessor()
        
        config = ModalityConfig(
            modality_type=ModalityType.TABULAR,
            encoding_type=EncodingType.STANDARDIZATION,
            preprocessing_params={"normalize": True},
        )
        
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        processed = await processor.process_tabular_data(data, config)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape == data.shape

    async def test_text_processing(self):
        """Test text data processing."""
        processor = MultiModalProcessor()
        
        config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            encoding_type=EncodingType.TFIDF,
            preprocessing_params={"lowercase": True, "remove_punctuation": True},
        )
        
        text = "This is TEST text with PUNCTUATION!"
        processed = await processor.process_text_data(text, config)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0

    async def test_image_processing(self):
        """Test image data processing."""
        processor = MultiModalProcessor()
        
        config = ModalityConfig(
            modality_type=ModalityType.IMAGE,
            encoding_type=EncodingType.CNN,
            preprocessing_params={"normalize": True, "resize": (64, 64)},
        )
        
        image = np.random.rand(128, 128, 3) * 255
        processed = await processor.process_image_data(image, config)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0

    async def test_time_series_processing(self):
        """Test time series data processing."""
        processor = MultiModalProcessor()
        
        config = ModalityConfig(
            modality_type=ModalityType.TIME_SERIES,
            encoding_type=EncodingType.STANDARDIZATION,
            preprocessing_params={
                "window_size": 10,
                "overlap": 0.5,
                "normalization": "z_score",
                "extract_features": True,
            },
        )
        
        data = np.random.rand(100, 3)
        processed = await processor.process_time_series_data(data, config)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0

    async def test_audio_processing(self):
        """Test audio data processing."""
        processor = MultiModalProcessor()
        
        config = ModalityConfig(
            modality_type=ModalityType.AUDIO,
            encoding_type=EncodingType.MFCC,
            preprocessing_params={"sample_rate": 16000, "normalize": True},
        )
        
        audio = np.random.rand(16000)  # 1 second of audio
        processed = await processor.process_audio_data(audio, config)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0


@pytest.mark.asyncio
class TestMultiModalTrainer:
    """Test multi-modal trainer."""

    def create_test_detector(self) -> MultiModalDetector:
        """Create test detector for training."""
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=0.6,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.4,
            ),
        }
        
        encoders = {}
        for modality, config in modality_configs.items():
            encoder = ModalityEncoder(
                encoder_id=uuid4(),
                modality_type=modality,
                encoding_type=config.encoding_type,
                config=config,
                feature_dimension=32,
            )
            encoders[modality] = encoder
        
        fusion_layer = FusionLayer(
            fusion_id=uuid4(),
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            input_modalities=list(modality_configs.keys()),
            output_dimension=64,
        )
        
        return MultiModalDetector(
            detector_id=uuid4(),
            name="test_training_detector",
            modality_configs=modality_configs,
            encoders=encoders,
            fusion_layers=[fusion_layer],
            output_dimension=64,
        )

    def create_training_data(self, n_samples: int = 10) -> list[MultiModalData]:
        """Create sample training data."""
        training_data = []
        
        for i in range(n_samples):
            sample = MultiModalData(
                sample_id=f"train_sample_{i}",
                modalities={
                    ModalityType.TEXT: f"training text sample {i}",
                    ModalityType.TABULAR: np.random.rand(5),
                },
                metadata={"is_anomaly": i % 5 == 0},  # 20% anomalies
            )
            training_data.append(sample)
        
        return training_data

    async def test_trainer_creation(self):
        """Test creating trainer."""
        trainer = MultiModalTrainer()
        
        assert isinstance(trainer.training_history, dict)
        assert isinstance(trainer.validation_history, dict)
        assert isinstance(trainer.training_stats, dict)

    async def test_detector_training(self):
        """Test training detector."""
        trainer = MultiModalTrainer()
        detector = self.create_test_detector()
        training_data = self.create_training_data()
        validation_data = self.create_training_data(5)
        
        # Train detector
        trained_detector = await trainer.train_detector(
            detector, training_data, validation_data
        )
        
        assert trained_detector.is_trained
        assert trained_detector.training_samples == len(training_data)
        assert trained_detector.validation_metrics is not None
        assert isinstance(trained_detector.validation_metrics, ModelMetrics)
        
        # Check training history
        assert detector.detector_id in trainer.training_history
        assert len(trainer.training_history[detector.detector_id]) > 0

    async def test_training_validation_failure(self):
        """Test training with invalid data."""
        trainer = MultiModalTrainer()
        detector = self.create_test_detector()
        
        # Empty training data should fail
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            await trainer.train_detector(detector, [])

    async def test_get_training_history(self):
        """Test getting training history."""
        trainer = MultiModalTrainer()
        detector = self.create_test_detector()
        training_data = self.create_training_data()
        
        # Train detector
        await trainer.train_detector(detector, training_data)
        
        # Get history
        history = trainer.get_training_history(detector.detector_id)
        assert history is not None
        assert len(history) > 0
        
        validation_history = trainer.get_validation_history(detector.detector_id)
        assert validation_history is None  # No validation data provided

    async def test_training_stats(self):
        """Test training statistics."""
        trainer = MultiModalTrainer()
        detector = self.create_test_detector()
        training_data = self.create_training_data()
        
        # Get initial stats
        initial_stats = trainer.get_training_stats()
        initial_sessions = initial_stats["training_stats"]["total_training_sessions"]
        
        # Train detector
        await trainer.train_detector(detector, training_data)
        
        # Check updated stats
        updated_stats = trainer.get_training_stats()
        assert updated_stats["training_stats"]["total_training_sessions"] == initial_sessions + 1
        assert updated_stats["training_stats"]["total_training_time"] > 0


@pytest.mark.asyncio
class TestMultiModalDetectionService:
    """Test multi-modal detection service."""

    async def test_service_creation(self):
        """Test creating detection service."""
        service = MultiModalDetectionService()
        
        assert isinstance(service.detectors, dict)
        assert isinstance(service.detection_stats, dict)
        assert len(service.detectors) == 0

    async def test_create_detector(self):
        """Test creating detector through service."""
        service = MultiModalDetectionService()
        
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=0.6,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.4,
            ),
        }
        
        detector = await service.create_detector(
            name="test_service_detector",
            modality_configs=modality_configs,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            output_dimension=128,
        )
        
        assert detector.name == "test_service_detector"
        assert len(detector.modality_configs) == 2
        assert detector.detector_id in service.detectors

    async def test_specialized_detector_creation(self):
        """Test creating specialized detectors."""
        service = MultiModalDetectionService()
        
        # Text-Image detector
        text_image_detector = await service.create_text_image_detector(
            name="text_image_detector"
        )
        assert ModalityType.TEXT in text_image_detector.modality_configs
        assert ModalityType.IMAGE in text_image_detector.modality_configs
        
        # Tabular-TimeSeries detector
        tabular_ts_detector = await service.create_tabular_timeseries_detector(
            name="tabular_ts_detector"
        )
        assert ModalityType.TABULAR in tabular_ts_detector.modality_configs
        assert ModalityType.TIME_SERIES in tabular_ts_detector.modality_configs
        
        # IoT detector
        iot_detector = await service.create_multimodal_iot_detector(
            name="iot_detector",
            include_audio=True,
        )
        assert ModalityType.IOT_SENSOR in iot_detector.modality_configs
        assert ModalityType.AUDIO in iot_detector.modality_configs

    async def test_detector_training_via_service(self):
        """Test training detector through service."""
        service = MultiModalDetectionService()
        
        # Create detector
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=0.6,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.4,
            ),
        }
        
        detector = await service.create_detector(
            name="training_test_detector",
            modality_configs=modality_configs,
        )
        
        # Create training data
        training_data = []
        for i in range(5):
            sample = MultiModalData(
                sample_id=f"service_train_{i}",
                modalities={
                    ModalityType.TEXT: f"training text {i}",
                    ModalityType.TABULAR: np.random.rand(5),
                },
            )
            training_data.append(sample)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id, training_data
        )
        
        assert trained_detector.is_trained
        assert trained_detector.training_samples == len(training_data)

    async def test_anomaly_detection_via_service(self):
        """Test anomaly detection through service."""
        service = MultiModalDetectionService()
        
        # Create and train detector
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
            ),
        }
        
        detector = await service.create_detector(
            name="detection_test_detector",
            modality_configs=modality_configs,
        )
        
        # Create minimal training data
        training_data = [
            MultiModalData(
                sample_id="train_1",
                modalities={
                    ModalityType.TEXT: "normal text",
                    ModalityType.TABULAR: np.array([1.0, 2.0, 3.0]),
                },
            )
        ]
        
        await service.train_detector(detector.detector_id, training_data)
        
        # Test detection
        test_sample = MultiModalData(
            sample_id="test_sample",
            modalities={
                ModalityType.TEXT: "test detection text",
                ModalityType.TABULAR: np.array([4.0, 5.0, 6.0]),
            },
        )
        
        is_anomaly, score, details = await service.detect_anomaly(
            detector.detector_id, test_sample
        )
        
        assert isinstance(is_anomaly, bool)
        assert isinstance(score, float)
        assert isinstance(details, dict)
        assert "detection_time_ms" in details

    async def test_batch_detection(self):
        """Test batch anomaly detection."""
        service = MultiModalDetectionService()
        
        # Create and train detector
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
            ),
        }
        
        detector = await service.create_detector(
            name="batch_test_detector",
            modality_configs=modality_configs,
        )
        
        # Train with minimal data
        training_data = [
            MultiModalData(
                sample_id="train",
                modalities={ModalityType.TEXT: "normal text"},
            )
        ]
        await service.train_detector(detector.detector_id, training_data)
        
        # Create batch test data
        test_samples = []
        for i in range(3):
            sample = MultiModalData(
                sample_id=f"batch_test_{i}",
                modalities={ModalityType.TEXT: f"test text {i}"},
            )
            test_samples.append(sample)
        
        # Batch detection
        results = await service.batch_detect_anomalies(
            detector.detector_id, test_samples, max_concurrent=2
        )
        
        assert len(results) == len(test_samples)
        for sample_id, is_anomaly, score, details in results:
            assert isinstance(sample_id, str)
            assert isinstance(is_anomaly, bool)
            assert isinstance(score, float)
            assert isinstance(details, dict)

    async def test_service_statistics(self):
        """Test service statistics."""
        service = MultiModalDetectionService()
        
        # Get initial stats
        stats = service.get_service_statistics()
        
        assert "detector_statistics" in stats
        assert "detection_statistics" in stats
        assert "supported_capabilities" in stats
        
        # Check capabilities
        capabilities = stats["supported_capabilities"]
        assert "modalities" in capabilities
        assert "encodings" in capabilities
        assert "fusion_strategies" in capabilities
        
        assert len(capabilities["modalities"]) > 0
        assert len(capabilities["encodings"]) > 0
        assert len(capabilities["fusion_strategies"]) > 0

    async def test_detector_cleanup(self):
        """Test detector cleanup."""
        service = MultiModalDetectionService()
        
        # Create multiple detectors
        for i in range(3):
            modality_configs = {
                ModalityType.TEXT: ModalityConfig(
                    modality_type=ModalityType.TEXT,
                    encoding_type=EncodingType.TFIDF,
                ),
            }
            
            await service.create_detector(
                name=f"cleanup_detector_{i}",
                modality_configs=modality_configs,
            )
        
        assert len(service.detectors) == 3
        
        # Cleanup untrained detectors
        cleanup_count = await service.cleanup_detectors(keep_trained=True)
        
        assert cleanup_count == 3
        assert len(service.detectors) == 0

    async def test_invalid_detector_operations(self):
        """Test operations with invalid detector IDs."""
        service = MultiModalDetectionService()
        invalid_id = uuid4()
        
        # Training non-existent detector should fail
        with pytest.raises(ValueError, match="Detector .* not found"):
            await service.train_detector(invalid_id, [])
        
        # Detection with non-existent detector should fail
        test_sample = MultiModalData(
            sample_id="test",
            modalities={ModalityType.TEXT: "test"},
        )
        
        with pytest.raises(ValueError, match="Detector .* not found"):
            await service.detect_anomaly(invalid_id, test_sample)
