"""Integration tests for multi-modal anomaly detection system."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.domain.models.multimodal import (
    EncodingType,
    FusionStrategy,
    ModalityConfig,
    ModalityType,
    MultiModalData,
)
from pynomaly.infrastructure.multimodal.processor import MultiModalProcessor
from pynomaly.infrastructure.multimodal.service import MultiModalDetectionService
from pynomaly.infrastructure.multimodal.trainer import MultiModalTrainer


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiModalIntegration:
    """Integration tests for multi-modal detection system."""

    async def test_end_to_end_text_tabular_detection(self):
        """Test complete text + tabular detection workflow."""
        service = MultiModalDetectionService()
        
        # Create detector for text and tabular data
        detector = await service.create_text_image_detector(
            name="text_tabular_integration_test",
            text_encoding=EncodingType.TFIDF,
            image_encoding=EncodingType.CNN,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
        )
        
        # Create diverse training dataset
        training_data = []
        
        # Normal samples
        for i in range(20):
            sample = MultiModalData(
                sample_id=f"normal_{i}",
                modalities={
                    ModalityType.TEXT: f"normal business transaction report {i}",
                    ModalityType.IMAGE: np.random.normal(0.5, 0.1, (64, 64, 3)),  # Normal images
                },
                metadata={"is_anomaly": False, "category": "normal"},
            )
            training_data.append(sample)
        
        # Anomalous samples
        for i in range(5):
            sample = MultiModalData(
                sample_id=f"anomaly_{i}",
                modalities={
                    ModalityType.TEXT: f"suspicious unusual activity detected {i}",
                    ModalityType.IMAGE: np.random.normal(0.8, 0.3, (64, 64, 3)),  # Unusual images
                },
                metadata={"is_anomaly": True, "category": "anomaly"},
            )
            training_data.append(sample)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id,
            training_data,
            validation_data=training_data[:10],  # Use subset for validation
        )
        
        assert trained_detector.is_trained
        assert trained_detector.training_samples == len(training_data)
        assert trained_detector.validation_metrics is not None
        
        # Test detection on new samples
        test_samples = [
            MultiModalData(
                sample_id="test_normal",
                modalities={
                    ModalityType.TEXT: "regular business transaction",
                    ModalityType.IMAGE: np.random.normal(0.5, 0.1, (64, 64, 3)),
                },
            ),
            MultiModalData(
                sample_id="test_anomaly",
                modalities={
                    ModalityType.TEXT: "suspicious fraudulent activity",
                    ModalityType.IMAGE: np.random.normal(0.9, 0.4, (64, 64, 3)),
                },
            ),
        ]
        
        # Individual detection
        for sample in test_samples:
            is_anomaly, score, details = await service.detect_anomaly(
                detector.detector_id, sample
            )
            
            assert isinstance(is_anomaly, bool)
            assert 0 <= score <= 1
            assert "modalities_used" in details
            assert "modality_contributions" in details
            assert "confidence" in details
        
        # Batch detection
        batch_results = await service.batch_detect_anomalies(
            detector.detector_id, test_samples
        )
        
        assert len(batch_results) == len(test_samples)
        
        for sample_id, is_anomaly, score, details in batch_results:
            assert sample_id in ["test_normal", "test_anomaly"]
            assert isinstance(is_anomaly, bool)
            assert 0 <= score <= 1

    async def test_multimodal_iot_sensor_detection(self):
        """Test IoT multi-modal sensor detection."""
        service = MultiModalDetectionService()
        
        # Create IoT detector with multiple sensor types
        detector = await service.create_multimodal_iot_detector(
            name="iot_sensor_integration_test",
            include_audio=True,
            fusion_strategy=FusionStrategy.HIERARCHICAL_FUSION,
        )
        
        # Generate synthetic IoT sensor data
        training_data = []
        
        # Normal IoT readings
        for i in range(30):
            sample = MultiModalData(
                sample_id=f"iot_normal_{i}",
                modalities={
                    ModalityType.IOT_SENSOR: np.random.normal(25, 2, 10),  # Temperature, humidity, etc.
                    ModalityType.TIME_SERIES: np.random.normal(50, 5, 100),  # Vibration data
                    ModalityType.TABULAR: np.array([
                        np.random.normal(20, 1),  # Power consumption
                        np.random.normal(60, 5),  # Speed
                        np.random.normal(0.1, 0.02),  # Error rate
                    ]),
                    ModalityType.AUDIO: np.random.normal(0, 0.1, 8000),  # Machine sounds
                },
                metadata={"is_anomaly": False, "machine_id": f"machine_{i % 5}"},
            )
            training_data.append(sample)
        
        # Anomalous IoT readings
        for i in range(8):
            sample = MultiModalData(
                sample_id=f"iot_anomaly_{i}",
                modalities={
                    ModalityType.IOT_SENSOR: np.random.normal(35, 8, 10),  # High temperature
                    ModalityType.TIME_SERIES: np.random.normal(100, 20, 100),  # High vibration
                    ModalityType.TABULAR: np.array([
                        np.random.normal(40, 5),  # High power consumption
                        np.random.normal(30, 10),  # Low speed
                        np.random.normal(0.3, 0.1),  # High error rate
                    ]),
                    ModalityType.AUDIO: np.random.normal(0, 0.3, 8000),  # Unusual sounds
                },
                metadata={"is_anomaly": True, "machine_id": f"machine_{i % 5}"},
            )
            training_data.append(sample)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id, training_data
        )
        
        assert trained_detector.is_trained
        
        # Test real-time detection simulation
        test_samples = []
        for i in range(5):
            # Mix of normal and edge-case samples
            if i < 3:
                # Normal samples
                sample = MultiModalData(
                    sample_id=f"runtime_normal_{i}",
                    modalities={
                        ModalityType.IOT_SENSOR: np.random.normal(25, 2, 10),
                        ModalityType.TIME_SERIES: np.random.normal(50, 5, 100),
                        ModalityType.TABULAR: np.array([22, 58, 0.08]),
                        ModalityType.AUDIO: np.random.normal(0, 0.1, 8000),
                    },
                )
            else:
                # Potential anomalies
                sample = MultiModalData(
                    sample_id=f"runtime_anomaly_{i}",
                    modalities={
                        ModalityType.IOT_SENSOR: np.random.normal(40, 10, 10),
                        ModalityType.TIME_SERIES: np.random.normal(120, 30, 100),
                        ModalityType.TABULAR: np.array([45, 25, 0.4]),
                        ModalityType.AUDIO: np.random.normal(0, 0.5, 8000),
                    },
                )
            test_samples.append(sample)
        
        # Process samples concurrently (simulating real-time)
        detection_tasks = [
            service.detect_anomaly(detector.detector_id, sample)
            for sample in test_samples
        ]
        
        results = await asyncio.gather(*detection_tasks)
        
        assert len(results) == len(test_samples)
        
        # Analyze results
        normal_scores = []
        anomaly_scores = []
        
        for i, (is_anomaly, score, details) in enumerate(results):
            if i < 3:  # Normal samples
                normal_scores.append(score)
            else:  # Anomalous samples
                anomaly_scores.append(score)
            
            # Validate detection details
            assert "modality_contributions" in details
            assert len(details["modality_contributions"]) >= 3  # At least IoT, TS, Tabular
            
            # Check that all modalities contribute
            total_contribution = sum(details["modality_contributions"].values())
            assert abs(total_contribution - 1.0) < 0.1  # Should sum to ~1

    async def test_performance_under_load(self):
        """Test system performance under concurrent load."""
        service = MultiModalDetectionService()
        
        # Create lightweight detector
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=0.7,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.3,
            ),
        }
        
        detector = await service.create_detector(
            name="performance_test_detector",
            modality_configs=modality_configs,
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            output_dimension=64,
        )
        
        # Quick training with minimal data
        training_data = [
            MultiModalData(
                sample_id="quick_train",
                modalities={
                    ModalityType.TEXT: "normal text",
                    ModalityType.TABULAR: np.array([1, 2, 3]),
                },
            )
        ]
        
        await service.train_detector(detector.detector_id, training_data)
        
        # Generate large batch of test samples
        test_samples = []
        for i in range(100):
            sample = MultiModalData(
                sample_id=f"load_test_{i}",
                modalities={
                    ModalityType.TEXT: f"test text sample {i}",
                    ModalityType.TABULAR: np.random.rand(5),
                },
            )
            test_samples.append(sample)
        
        # Test concurrent processing
        import time
        start_time = time.time()
        
        # Process in batches to test concurrent handling
        batch_size = 20
        batch_results = []
        
        for i in range(0, len(test_samples), batch_size):
            batch = test_samples[i:i+batch_size]
            batch_result = await service.batch_detect_anomalies(
                detector.detector_id, batch, max_concurrent=10
            )
            batch_results.extend(batch_result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Validate performance
        assert len(batch_results) == len(test_samples)
        assert processing_time < 30  # Should complete within 30 seconds
        
        # Calculate throughput
        throughput = len(test_samples) / processing_time
        assert throughput > 3  # At least 3 samples per second
        
        print(f"Performance test: {throughput:.2f} samples/sec")

    async def test_data_quality_validation(self):
        """Test handling of various data quality issues."""
        service = MultiModalDetectionService()
        
        # Create detector with optional modalities
        modality_configs = {
            ModalityType.TEXT: ModalityConfig(
                modality_type=ModalityType.TEXT,
                encoding_type=EncodingType.TFIDF,
                weight=0.5,
                is_required=True,
            ),
            ModalityType.TABULAR: ModalityConfig(
                modality_type=ModalityType.TABULAR,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.3,
                is_required=False,  # Optional
            ),
            ModalityType.TIME_SERIES: ModalityConfig(
                modality_type=ModalityType.TIME_SERIES,
                encoding_type=EncodingType.STANDARDIZATION,
                weight=0.2,
                is_required=False,  # Optional
            ),
        }
        
        detector = await service.create_detector(
            name="data_quality_test_detector",
            modality_configs=modality_configs,
            fusion_strategy=FusionStrategy.ADAPTIVE_FUSION,
        )
        
        # Training data with varying completeness
        training_data = []
        
        # Complete samples
        for i in range(10):
            sample = MultiModalData(
                sample_id=f"complete_{i}",
                modalities={
                    ModalityType.TEXT: f"complete text sample {i}",
                    ModalityType.TABULAR: np.random.rand(5),
                    ModalityType.TIME_SERIES: np.random.rand(50),
                },
            )
            training_data.append(sample)
        
        # Samples missing optional modalities
        for i in range(5):
            sample = MultiModalData(
                sample_id=f"partial_{i}",
                modalities={
                    ModalityType.TEXT: f"partial text sample {i}",
                    ModalityType.TABULAR: np.random.rand(5),
                    # Missing TIME_SERIES
                },
            )
            training_data.append(sample)
        
        # Samples with only required modality
        for i in range(3):
            sample = MultiModalData(
                sample_id=f"minimal_{i}",
                modalities={
                    ModalityType.TEXT: f"minimal text sample {i}",
                    # Missing optional modalities
                },
            )
            training_data.append(sample)
        
        # Train detector
        trained_detector = await service.train_detector(
            detector.detector_id, training_data
        )
        
        assert trained_detector.is_trained
        
        # Test detection with various data quality scenarios
        test_scenarios = [
            # Complete sample
            MultiModalData(
                sample_id="test_complete",
                modalities={
                    ModalityType.TEXT: "complete test sample",
                    ModalityType.TABULAR: np.array([1, 2, 3, 4, 5]),
                    ModalityType.TIME_SERIES: np.random.rand(50),
                },
            ),
            # Missing optional modality
            MultiModalData(
                sample_id="test_partial",
                modalities={
                    ModalityType.TEXT: "partial test sample",
                    ModalityType.TABULAR: np.array([1, 2, 3, 4, 5]),
                },
            ),
            # Only required modality
            MultiModalData(
                sample_id="test_minimal",
                modalities={
                    ModalityType.TEXT: "minimal test sample",
                },
            ),
            # Edge case: empty text
            MultiModalData(
                sample_id="test_empty_text",
                modalities={
                    ModalityType.TEXT: "",
                    ModalityType.TABULAR: np.array([1, 2, 3]),
                },
            ),
        ]
        
        # Process all scenarios
        for sample in test_scenarios:
            try:
                is_anomaly, score, details = await service.detect_anomaly(
                    detector.detector_id, sample
                )
                
                # Validate results
                assert isinstance(is_anomaly, bool)
                assert 0 <= score <= 1
                assert "modalities_used" in details
                
                # Check that detector adapts to available modalities
                used_modalities = details["modalities_used"]
                available_modalities = [mod.value for mod in sample.get_available_modalities()]
                
                for used_mod in used_modalities:
                    assert used_mod in available_modalities
                
                print(f"Sample {sample.sample_id}: score={score:.3f}, "
                      f"modalities={len(used_modalities)}")
                
            except Exception as e:
                # Should not fail on valid samples with required modalities
                if sample.has_modality(ModalityType.TEXT):
                    pytest.fail(f"Detection failed for valid sample {sample.sample_id}: {e}")

    async def test_detector_comparison_and_evaluation(self):
        """Test comparing multiple detectors and evaluation metrics."""
        service = MultiModalDetectionService()
        
        # Create multiple detectors with different configurations
        detectors_config = [
            {
                "name": "early_fusion_detector",
                "fusion_strategy": FusionStrategy.EARLY_FUSION,
            },
            {
                "name": "late_fusion_detector", 
                "fusion_strategy": FusionStrategy.LATE_FUSION,
            },
            {
                "name": "attention_fusion_detector",
                "fusion_strategy": FusionStrategy.ATTENTION_FUSION,
            },
        ]
        
        detectors = []
        
        # Create detectors
        for config in detectors_config:
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
                name=config["name"],
                modality_configs=modality_configs,
                fusion_strategy=config["fusion_strategy"],
                output_dimension=64,
            )
            detectors.append(detector)
        
        # Create evaluation dataset with known labels
        evaluation_data = []
        
        # Normal samples
        for i in range(20):
            sample = MultiModalData(
                sample_id=f"eval_normal_{i}",
                modalities={
                    ModalityType.TEXT: f"normal evaluation text {i}",
                    ModalityType.TABULAR: np.random.normal(0, 1, 5),
                },
                metadata={"is_anomaly": False},
            )
            evaluation_data.append(sample)
        
        # Anomalous samples
        for i in range(5):
            sample = MultiModalData(
                sample_id=f"eval_anomaly_{i}",
                modalities={
                    ModalityType.TEXT: f"anomalous suspicious text {i}",
                    ModalityType.TABULAR: np.random.normal(3, 2, 5),  # Different distribution
                },
                metadata={"is_anomaly": True},
            )
            evaluation_data.append(sample)
        
        # Train all detectors
        detector_results = {}
        
        for detector in detectors:
            # Train detector
            await service.train_detector(detector.detector_id, evaluation_data)
            
            # Evaluate performance
            evaluation_result = await service.evaluate_detector_performance(
                detector.detector_id, evaluation_data
            )
            
            detector_results[detector.name] = evaluation_result
            
            # Validate evaluation results
            assert "performance_metrics" in evaluation_result
            assert "timing_metrics" in evaluation_result
            assert "anomaly_statistics" in evaluation_result
            
            metrics = evaluation_result["performance_metrics"]
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics
            
            # All metrics should be between 0 and 1
            for metric_name, metric_value in metrics.items():
                assert 0 <= metric_value <= 1, f"{metric_name} = {metric_value}"
        
        # Compare detector performance
        print("\nDetector Performance Comparison:")
        for name, result in detector_results.items():
            metrics = result["performance_metrics"]
            timing = result["timing_metrics"]
            print(f"{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            print(f"  Avg Detection Time: {timing['average_detection_time_ms']:.2f}ms")
        
        # At least one detector should have reasonable performance
        best_f1 = max(result["performance_metrics"]["f1_score"] 
                     for result in detector_results.values())
        assert best_f1 > 0.1  # Should detect some patterns

    async def test_service_resource_management(self):
        """Test service resource management and cleanup."""
        service = MultiModalDetectionService()
        
        # Create multiple detectors
        detector_ids = []
        for i in range(5):
            modality_configs = {
                ModalityType.TEXT: ModalityConfig(
                    modality_type=ModalityType.TEXT,
                    encoding_type=EncodingType.TFIDF,
                ),
            }
            
            detector = await service.create_detector(
                name=f"resource_test_detector_{i}",
                modality_configs=modality_configs,
            )
            detector_ids.append(detector.detector_id)
        
        # Train some detectors
        training_data = [
            MultiModalData(
                sample_id="resource_train",
                modalities={ModalityType.TEXT: "resource training text"},
            )
        ]
        
        # Train first 3 detectors
        for detector_id in detector_ids[:3]:
            await service.train_detector(detector_id, training_data)
        
        # Check service statistics
        stats = service.get_service_statistics()
        
        assert stats["detector_statistics"]["total_detectors"] == 5
        assert stats["detector_statistics"]["trained_detectors"] == 3
        
        # Test detector cleanup - keep trained
        cleanup_count = await service.cleanup_detectors(keep_trained=True)
        assert cleanup_count == 2  # Should remove 2 untrained detectors
        
        updated_stats = service.get_service_statistics()
        assert updated_stats["detector_statistics"]["total_detectors"] == 3
        assert updated_stats["detector_statistics"]["trained_detectors"] == 3
        
        # Test detector cleanup - remove all
        cleanup_count = await service.cleanup_detectors(keep_trained=False)
        assert cleanup_count == 3  # Should remove all remaining detectors
        
        final_stats = service.get_service_statistics()
        assert final_stats["detector_statistics"]["total_detectors"] == 0
        assert final_stats["detector_statistics"]["trained_detectors"] == 0