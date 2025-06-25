"""
Comprehensive Domain Layer Testing Suite

In-depth testing of all domain entities, value objects, domain services,
business rules, invariants, and domain events.
"""

import pytest
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
from decimal import Decimal

from pynomaly.domain.entities import (
    Dataset, Detector, DetectionResult, Anomaly, ExperimentRun
)
from pynomaly.domain.value_objects import (
    AnomalyScore, ContaminationRate, ConfidenceInterval,
    ThresholdConfig, PerformanceMetrics
)
# from pynomaly.domain.services import (
#     DetectionService, EnsembleService, ModelValidationService,
#     DataQualityService, PerformanceAnalysisService
# )
from pynomaly.domain.exceptions import (
    DomainError, DetectorNotFittedError, InsufficientDataError, ValidationError
)
# InvalidAnomalyScoreError, InvalidContaminationRateError don't exist
# from pynomaly.domain.events import (
#     DetectorCreated, DetectorTrained, AnomalyDetected,
#     DatasetCreated, ExperimentCompleted
# )
# from pynomaly.domain.specifications import (
#     TrainedDetectorSpecification, HighQualityDatasetSpecification,
#     AnomalyThresholdSpecification
# )


class TestDatasetEntity:
    """Comprehensive testing of Dataset entity."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample dataset for testing."""
        return {
            "id": DatasetId("dataset_001"),
            "name": "Test Dataset",
            "description": "Dataset for comprehensive testing",
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            "features": ["feature_1", "feature_2", "feature_3"],
            "target": [0, 0, 1],  # Binary target for supervised learning
            "metadata": {
                "source": "test_generator",
                "created_by": "test_user",
                "version": "1.0"
            },
            "quality_metrics": {
                "completeness": 1.0,
                "consistency": 0.95,
                "validity": 0.98
            }
        }
    
    def test_dataset_creation_valid(self, sample_data):
        """Test valid dataset creation."""
        dataset = Dataset(**sample_data)
        
        assert dataset.id == sample_data["id"]
        assert dataset.name == sample_data["name"]
        assert len(dataset.data) == 3
        assert len(dataset.features) == 3
        assert dataset.is_valid()
        
    def test_dataset_creation_minimal(self):
        """Test dataset creation with minimal required fields."""
        minimal_data = {
            "name": "Minimal Dataset",
            "data": [[1, 2], [3, 4]],
            "features": ["x", "y"]
        }
        
        dataset = Dataset(**minimal_data)
        
        assert dataset.name == "Minimal Dataset"
        assert len(dataset.data) == 2
        assert dataset.id is not None  # Should auto-generate
        assert dataset.created_at is not None
        
    def test_dataset_validation_empty_data(self):
        """Test validation fails for empty data."""
        with pytest.raises(InsufficientDataError):
            Dataset(
                name="Empty Dataset",
                data=[],
                features=["x"]
            )
            
    def test_dataset_validation_mismatched_features(self):
        """Test validation fails for mismatched features."""
        with pytest.raises(ValidationError):
            Dataset(
                name="Mismatched Dataset",
                data=[[1, 2, 3], [4, 5, 6]],
                features=["x", "y"]  # Only 2 features for 3 columns
            )
            
    def test_dataset_add_sample(self, sample_data):
        """Test adding samples to dataset."""
        dataset = Dataset(**sample_data)
        initial_count = len(dataset.data)
        
        new_sample = [10.0, 11.0, 12.0]
        dataset.add_sample(new_sample)
        
        assert len(dataset.data) == initial_count + 1
        assert dataset.data[-1] == new_sample
        
    def test_dataset_add_sample_invalid_dimensions(self, sample_data):
        """Test adding sample with wrong dimensions fails."""
        dataset = Dataset(**sample_data)
        
        with pytest.raises(ValidationError):
            dataset.add_sample([1, 2])  # Missing one feature
            
    def test_dataset_remove_sample(self, sample_data):
        """Test removing samples from dataset."""
        dataset = Dataset(**sample_data)
        initial_count = len(dataset.data)
        
        dataset.remove_sample(1)  # Remove second sample
        
        assert len(dataset.data) == initial_count - 1
        assert dataset.data[1] == [7.0, 8.0, 9.0]  # Third sample moved up
        
    def test_dataset_get_subset(self, sample_data):
        """Test getting dataset subset."""
        dataset = Dataset(**sample_data)
        
        subset = dataset.get_subset(indices=[0, 2])
        
        assert len(subset.data) == 2
        assert subset.data[0] == [1.0, 2.0, 3.0]
        assert subset.data[1] == [7.0, 8.0, 9.0]
        assert subset.features == dataset.features
        
    def test_dataset_statistics(self, sample_data):
        """Test dataset statistics calculation."""
        dataset = Dataset(**sample_data)
        
        stats = dataset.get_statistics()
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert len(stats["mean"]) == len(dataset.features)
        
    def test_dataset_quality_assessment(self, sample_data):
        """Test dataset quality assessment."""
        dataset = Dataset(**sample_data)
        
        quality = dataset.assess_quality()
        
        assert "completeness" in quality
        assert "consistency" in quality
        assert "duplicates" in quality
        assert "outliers" in quality
        assert 0 <= quality["completeness"] <= 1
        
    def test_dataset_transformation_log(self, sample_data):
        """Test dataset transformation logging."""
        dataset = Dataset(**sample_data)
        
        # Apply transformation
        dataset.apply_transformation("normalize", {"method": "min_max"})
        
        assert len(dataset.transformation_history) == 1
        assert dataset.transformation_history[0]["type"] == "normalize"
        assert dataset.transformation_history[0]["parameters"]["method"] == "min_max"
        
    def test_dataset_versioning(self, sample_data):
        """Test dataset versioning."""
        dataset = Dataset(**sample_data)
        original_version = dataset.version
        
        # Modify dataset
        dataset.add_sample([13.0, 14.0, 15.0])
        
        assert dataset.version > original_version
        assert dataset.updated_at > dataset.created_at
        
    def test_dataset_equality(self, sample_data):
        """Test dataset equality comparison."""
        dataset1 = Dataset(**sample_data)
        dataset2 = Dataset(**sample_data)
        
        # Same content, different instances
        assert dataset1 == dataset2
        
        # Different data
        dataset2.add_sample([99.0, 99.0, 99.0])
        assert dataset1 != dataset2
        
    def test_dataset_serialization(self, sample_data):
        """Test dataset serialization/deserialization."""
        dataset = Dataset(**sample_data)
        
        # Serialize
        serialized = dataset.to_dict()
        
        # Deserialize
        restored = Dataset.from_dict(serialized)
        
        assert restored == dataset
        assert restored.data == dataset.data
        assert restored.features == dataset.features


class TestDetectorEntity:
    """Comprehensive testing of Detector entity."""
    
    @pytest.fixture
    def sample_detector_data(self):
        """Sample detector for testing."""
        return {
            "id": DetectorId("detector_001"),
            "name": "Test Detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": ContaminationRate(0.1),
            "parameters": {
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": 0.1,
                "random_state": 42
            },
            "hyperparameters": {
                "n_estimators": {"min": 10, "max": 500, "default": 100},
                "contamination": {"min": 0.001, "max": 0.5, "default": 0.1}
            },
            "metadata": {
                "category": "unsupervised",
                "complexity": "medium",
                "scalability": "high"
            }
        }
    
    def test_detector_creation_valid(self, sample_detector_data):
        """Test valid detector creation."""
        detector = Detector(**sample_detector_data)
        
        assert detector.id == sample_detector_data["id"]
        assert detector.name == sample_detector_data["name"]
        assert detector.algorithm_name == "IsolationForest"
        assert not detector.is_fitted
        assert detector.training_time is None
        
    def test_detector_creation_minimal(self):
        """Test detector creation with minimal fields."""
        minimal_data = {
            "name": "Minimal Detector",
            "algorithm_name": "LocalOutlierFactor",
            "contamination_rate": ContaminationRate(0.05)
        }
        
        detector = Detector(**minimal_data)
        
        assert detector.name == "Minimal Detector"
        assert detector.id is not None
        assert detector.created_at is not None
        assert not detector.is_fitted
        
    def test_detector_fit_operation(self, sample_detector_data):
        """Test detector fitting operation."""
        detector = Detector(**sample_detector_data)
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.data = [[1, 2], [3, 4], [5, 6]]
        mock_dataset.features = ["x", "y"]
        
        # Fit detector
        start_time = datetime.utcnow()
        detector.fit(mock_dataset)
        
        assert detector.is_fitted
        assert detector.training_time is not None
        assert detector.fitted_at >= start_time
        assert detector.training_dataset_id == mock_dataset.id
        
    def test_detector_fit_already_fitted(self, sample_detector_data):
        """Test fitting already fitted detector."""
        detector = Detector(**sample_detector_data)
        mock_dataset = Mock()
        
        # Fit first time
        detector.fit(mock_dataset)
        original_fitted_at = detector.fitted_at
        
        # Fit again (should update)
        detector.fit(mock_dataset)
        
        assert detector.fitted_at > original_fitted_at
        
    def test_detector_predict_unfitted(self, sample_detector_data):
        """Test prediction with unfitted detector fails."""
        detector = Detector(**sample_detector_data)
        mock_dataset = Mock()
        
        with pytest.raises(DetectorNotFittedError):
            detector.predict(mock_dataset)
            
    def test_detector_predict_fitted(self, sample_detector_data):
        """Test prediction with fitted detector."""
        detector = Detector(**sample_detector_data)
        
        # Mock fitting
        detector._is_fitted = True
        detector._model = Mock()
        detector._model.predict.return_value = [0, 1, 0]
        detector._model.decision_function.return_value = [0.1, 0.9, 0.2]
        
        mock_dataset = Mock()
        mock_dataset.data = [[1, 2], [3, 4], [5, 6]]
        
        result = detector.predict(mock_dataset)
        
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'anomaly_scores')
        
    def test_detector_parameter_validation(self):
        """Test detector parameter validation."""
        # Valid parameters
        valid_params = {
            "n_estimators": 100,
            "contamination": 0.1
        }
        
        detector = Detector(
            name="Valid Detector",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters=valid_params
        )
        
        assert detector.validate_parameters() is True
        
        # Invalid parameters
        detector.parameters["n_estimators"] = -1
        assert detector.validate_parameters() is False
        
    def test_detector_hyperparameter_tuning(self, sample_detector_data):
        """Test detector hyperparameter tuning."""
        detector = Detector(**sample_detector_data)
        
        # Mock tuning results
        tuned_params = {
            "n_estimators": 150,
            "contamination": 0.08
        }
        
        detector.update_parameters(tuned_params)
        
        assert detector.parameters["n_estimators"] == 150
        assert detector.parameters["contamination"] == 0.08
        assert len(detector.parameter_history) > 0
        
    def test_detector_performance_tracking(self, sample_detector_data):
        """Test detector performance tracking."""
        detector = Detector(**sample_detector_data)
        
        # Add performance metrics
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_roc=0.91
        )
        
        detector.add_performance_metrics(metrics)
        
        assert detector.performance_history[0] == metrics
        assert detector.best_performance.accuracy == 0.85
        
        # Add better performance
        better_metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            auc_roc=0.95
        )
        
        detector.add_performance_metrics(better_metrics)
        
        assert detector.best_performance == better_metrics
        
    def test_detector_model_persistence(self, sample_detector_data):
        """Test detector model persistence."""
        detector = Detector(**sample_detector_data)
        
        # Mock fitted model
        mock_model = Mock()
        detector._model = mock_model
        detector._is_fitted = True
        
        # Test model serialization
        model_data = detector.serialize_model()
        assert model_data is not None
        
        # Test model loading
        new_detector = Detector(**sample_detector_data)
        new_detector.load_model(model_data)
        
        assert new_detector.is_fitted
        
    def test_detector_clone(self, sample_detector_data):
        """Test detector cloning."""
        detector = Detector(**sample_detector_data)
        
        cloned = detector.clone()
        
        assert cloned.id != detector.id
        assert cloned.name == detector.name
        assert cloned.algorithm_name == detector.algorithm_name
        assert cloned.parameters == detector.parameters
        assert not cloned.is_fitted  # Clone should not be fitted


class TestAnomalyScoreValueObject:
    """Comprehensive testing of AnomalyScore value object."""
    
    def test_anomaly_score_creation_valid(self):
        """Test valid anomaly score creation."""
        score = AnomalyScore(0.75)
        
        assert score.value == 0.75
        assert 0 <= score.value <= 1
        
    def test_anomaly_score_creation_boundary_values(self):
        """Test anomaly score creation with boundary values."""
        # Minimum value
        min_score = AnomalyScore(0.0)
        assert min_score.value == 0.0
        
        # Maximum value
        max_score = AnomalyScore(1.0)
        assert max_score.value == 1.0
        
    def test_anomaly_score_creation_invalid(self):
        """Test invalid anomaly score creation."""
        # Below minimum
        with pytest.raises(InvalidAnomalyScoreError):
            AnomalyScore(-0.1)
            
        # Above maximum
        with pytest.raises(InvalidAnomalyScoreError):
            AnomalyScore(1.1)
            
        # NaN
        with pytest.raises(InvalidAnomalyScoreError):
            AnomalyScore(float('nan'))
            
        # Infinity
        with pytest.raises(InvalidAnomalyScoreError):
            AnomalyScore(float('inf'))
            
    def test_anomaly_score_comparison(self):
        """Test anomaly score comparison operations."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.3)
        
        # Equality
        assert score1 == score3
        assert score1 != score2
        
        # Ordering
        assert score1 < score2
        assert score2 > score1
        assert score1 <= score3
        assert score2 >= score1
        
    def test_anomaly_score_arithmetic(self):
        """Test anomaly score arithmetic operations."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.4)
        
        # Addition (should clamp to 1.0)
        result_add = score1 + score2
        assert result_add.value == 0.7
        
        # Addition with overflow
        high_score = AnomalyScore(0.8)
        result_overflow = high_score + score2
        assert result_overflow.value == 1.0
        
        # Subtraction (should clamp to 0.0)
        result_sub = score2 - score1
        assert result_sub.value == 0.1
        
        # Subtraction with underflow
        result_underflow = score1 - score2
        assert result_underflow.value == 0.0
        
    def test_anomaly_score_classification(self):
        """Test anomaly score classification methods."""
        low_score = AnomalyScore(0.2)
        medium_score = AnomalyScore(0.5)
        high_score = AnomalyScore(0.9)
        
        threshold = 0.6
        
        assert not low_score.exceeds_threshold(threshold)
        assert not medium_score.exceeds_threshold(threshold)
        assert high_score.exceeds_threshold(threshold)
        
    def test_anomaly_score_confidence_level(self):
        """Test anomaly score confidence levels."""
        uncertain_score = AnomalyScore(0.51)  # Just above threshold
        confident_score = AnomalyScore(0.95)  # High confidence
        
        threshold = 0.5
        
        # Test that uncertain score just exceeds threshold
        assert uncertain_score.exceeds_threshold(threshold)
        # Test that confident score significantly exceeds threshold  
        assert confident_score.exceeds_threshold(threshold)
        
    def test_anomaly_score_serialization(self):
        """Test anomaly score serialization."""
        score = AnomalyScore(0.678)
        
        # To dict
        score_dict = score.to_dict()
        assert score_dict["value"] == 0.678
        
        # From dict
        restored_score = AnomalyScore.from_dict(score_dict)
        assert restored_score == score
        
        # To JSON
        score_json = score.to_json()
        assert "0.678" in score_json
        
    def test_anomaly_score_statistical_operations(self):
        """Test statistical operations on anomaly scores."""
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.3),
            AnomalyScore(0.7),
            AnomalyScore(0.9)
        ]
        
        # Mean
        mean_score = AnomalyScore.mean(scores)
        assert abs(mean_score.value - 0.5) < 0.01
        
        # Median
        median_score = AnomalyScore.median(scores)
        assert median_score.value == 0.5
        
        # Standard deviation
        std_score = AnomalyScore.std(scores)
        assert std_score > 0


class TestContaminationRateValueObject:
    """Comprehensive testing of ContaminationRate value object."""
    
    def test_contamination_rate_creation_valid(self):
        """Test valid contamination rate creation."""
        rate = ContaminationRate(0.1)
        
        assert rate.value == 0.1
        assert 0 < rate.value <= 0.5
        
    def test_contamination_rate_creation_boundary(self):
        """Test contamination rate creation with boundary values."""
        # Minimum valid value
        min_rate = ContaminationRate(0.001)
        assert min_rate.value == 0.001
        
        # Maximum valid value
        max_rate = ContaminationRate(0.5)
        assert max_rate.value == 0.5
        
    def test_contamination_rate_creation_invalid(self):
        """Test invalid contamination rate creation."""
        # Zero (invalid)
        with pytest.raises(InvalidContaminationRateError):
            ContaminationRate(0.0)
            
        # Negative (invalid)
        with pytest.raises(InvalidContaminationRateError):
            ContaminationRate(-0.1)
            
        # Above maximum (invalid)
        with pytest.raises(InvalidContaminationRateError):
            ContaminationRate(0.6)
            
    def test_contamination_rate_to_sample_count(self):
        """Test conversion to sample count."""
        rate = ContaminationRate(0.1)
        
        # For 1000 samples
        anomaly_count = rate.to_sample_count(1000)
        assert anomaly_count == 100
        
        # For 50 samples
        anomaly_count_small = rate.to_sample_count(50)
        assert anomaly_count_small == 5
        
    def test_contamination_rate_adjustment(self):
        """Test contamination rate adjustment based on data."""
        rate = ContaminationRate(0.1)
        
        # Adjust based on detected anomalies
        adjusted_rate = rate.adjust_based_on_detection(detected=15, total=100)
        assert adjusted_rate.value == 0.15
        
        # Ensure adjustment stays within bounds
        extreme_adjustment = rate.adjust_based_on_detection(detected=80, total=100)
        assert extreme_adjustment.value <= 0.5


class TestDetectionResultEntity:
    """Comprehensive testing of DetectionResult entity."""
    
    @pytest.fixture
    def sample_detection_result(self):
        """Sample detection result for testing."""
        return {
            "dataset_id": DatasetId("dataset_001"),
            "detector_id": DetectorId("detector_001"),
            "predictions": [0, 1, 0, 1, 0],
            "anomaly_scores": [
                AnomalyScore(0.1), AnomalyScore(0.9),
                AnomalyScore(0.2), AnomalyScore(0.8),
                AnomalyScore(0.3)
            ],
            "confidence_intervals": [
                ConfidenceInterval(0.05, 0.15, 0.95),
                ConfidenceInterval(0.85, 0.95, 0.95),
                ConfidenceInterval(0.15, 0.25, 0.95),
                ConfidenceInterval(0.75, 0.85, 0.95),
                ConfidenceInterval(0.25, 0.35, 0.95)
            ],
            "processing_time": 0.045,
            "threshold": 0.5,
            "metadata": {
                "algorithm": "IsolationForest",
                "parameters": {"n_estimators": 100}
            }
        }
    
    def test_detection_result_creation_valid(self, sample_detection_result):
        """Test valid detection result creation."""
        result = DetectionResult(**sample_detection_result)
        
        assert len(result.predictions) == 5
        assert len(result.anomaly_scores) == 5
        assert result.anomaly_count == 2  # Two anomalies (predictions = 1)
        assert result.anomaly_rate == 0.4  # 2/5
        
    def test_detection_result_validation_mismatched_lengths(self):
        """Test detection result validation with mismatched lengths."""
        with pytest.raises(ValidationError):
            DetectionResult(
                dataset_id=DatasetId("test"),
                detector_id=DetectorId("test"),
                predictions=[0, 1, 0],
                anomaly_scores=[AnomalyScore(0.1), AnomalyScore(0.9)]  # Wrong length
            )
            
    def test_detection_result_anomaly_extraction(self, sample_detection_result):
        """Test anomaly extraction from result."""
        result = DetectionResult(**sample_detection_result)
        
        anomalies = result.get_anomalies()
        
        assert len(anomalies) == 2  # Indices 1 and 3
        assert anomalies[0].index == 1
        assert anomalies[0].score == AnomalyScore(0.9)
        assert anomalies[1].index == 3
        assert anomalies[1].score == AnomalyScore(0.8)
        
    def test_detection_result_threshold_adjustment(self, sample_detection_result):
        """Test threshold adjustment in result."""
        result = DetectionResult(**sample_detection_result)
        
        # Adjust threshold
        new_threshold = 0.25
        adjusted_result = result.adjust_threshold(new_threshold)
        
        # Should have more anomalies with lower threshold
        assert adjusted_result.anomaly_count > result.anomaly_count
        
    def test_detection_result_statistical_summary(self, sample_detection_result):
        """Test statistical summary of detection result."""
        result = DetectionResult(**sample_detection_result)
        
        summary = result.get_statistical_summary()
        
        assert "mean_score" in summary
        assert "std_score" in summary
        assert "min_score" in summary
        assert "max_score" in summary
        assert "anomaly_rate" in summary
        
    def test_detection_result_performance_metrics(self, sample_detection_result):
        """Test performance metrics calculation."""
        result = DetectionResult(**sample_detection_result)
        
        # Mock ground truth
        ground_truth = [0, 1, 0, 1, 1]  # Last one is false negative
        
        metrics = result.calculate_performance_metrics(ground_truth)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
    def test_detection_result_export_formats(self, sample_detection_result):
        """Test detection result export to different formats."""
        result = DetectionResult(**sample_detection_result)
        
        # Export to DataFrame
        df = result.to_dataframe()
        assert len(df) == 5
        assert "prediction" in df.columns
        assert "anomaly_score" in df.columns
        
        # Export to CSV string
        csv_data = result.to_csv()
        assert "prediction,anomaly_score" in csv_data
        
        # Export to JSON
        json_data = result.to_json()
        assert "predictions" in json_data
        assert "anomaly_scores" in json_data


class TestDomainServices:
    """Comprehensive testing of domain services."""
    
    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        repo = Mock()
        repo.find_by_id.return_value = Mock()
        repo.save.return_value = Mock()
        return repo
    
    @pytest.fixture
    def mock_dataset_repository(self):
        """Mock dataset repository."""
        repo = Mock()
        repo.find_by_id.return_value = Mock()
        repo.save.return_value = Mock()
        return repo
    
    def test_detection_service_single_detection(self, mock_detector_repository, mock_dataset_repository):
        """Test single anomaly detection via service."""
        service = DetectionService(
            detector_repository=mock_detector_repository,
            dataset_repository=mock_dataset_repository
        )
        
        # Mock detector
        mock_detector = Mock()
        mock_detector.is_fitted = True
        mock_detector.predict.return_value = Mock(
            predictions=[0, 1, 0],
            anomaly_scores=[AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.2)]
        )
        mock_detector_repository.find_by_id.return_value = mock_detector
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.data = [[1, 2], [3, 4], [5, 6]]
        mock_dataset_repository.find_by_id.return_value = mock_dataset
        
        # Execute detection
        result = service.detect_anomalies(
            detector_id=DetectorId("det_001"),
            dataset_id=DatasetId("data_001")
        )
        
        assert result is not None
        mock_detector.predict.assert_called_once()
        
    def test_ensemble_service_multiple_detectors(self):
        """Test ensemble detection with multiple detectors."""
        service = EnsembleService()
        
        # Mock multiple detectors
        detectors = []
        for i in range(3):
            detector = Mock()
            detector.is_fitted = True
            detector.predict.return_value = Mock(
                predictions=[0, 1] if i % 2 == 0 else [1, 0],
                anomaly_scores=[AnomalyScore(0.3), AnomalyScore(0.7)]
            )
            detectors.append(detector)
        
        mock_dataset = Mock()
        mock_dataset.data = [[1, 2], [3, 4]]
        
        # Execute ensemble detection
        result = service.ensemble_detect(
            detectors=detectors,
            dataset=mock_dataset,
            aggregation_method="majority_vote"
        )
        
        assert result is not None
        assert len(result.predictions) == 2
        
    def test_model_validation_service_cross_validation(self):
        """Test model validation with cross-validation."""
        service = ModelValidationService()
        
        mock_detector = Mock()
        mock_dataset = Mock()
        mock_dataset.data = [[i, i+1] for i in range(100)]
        
        # Mock cross-validation results
        with patch.object(service, '_perform_cv_fold') as mock_cv:
            mock_cv.return_value = {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.90
            }
            
            results = service.cross_validate(
                detector=mock_detector,
                dataset=mock_dataset,
                cv_folds=5
            )
            
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert mock_cv.call_count == 5
        
    def test_data_quality_service_assessment(self):
        """Test data quality assessment service."""
        service = DataQualityService()
        
        # Mock dataset with quality issues
        mock_dataset = Mock()
        mock_dataset.data = [
            [1.0, 2.0, 3.0],
            [4.0, None, 6.0],  # Missing value
            [7.0, 8.0, 9.0],
            [1.0, 2.0, 3.0],  # Duplicate
            [100.0, 200.0, 300.0]  # Potential outlier
        ]
        mock_dataset.features = ["x", "y", "z"]
        
        quality_report = service.assess_quality(mock_dataset)
        
        assert "completeness" in quality_report
        assert "consistency" in quality_report
        assert "uniqueness" in quality_report
        assert "validity" in quality_report
        assert "outliers" in quality_report
        
    def test_performance_analysis_service_comparison(self):
        """Test performance analysis service for detector comparison."""
        service = PerformanceAnalysisService()
        
        # Mock performance data for multiple detectors
        detector_performances = {
            "detector_1": {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.90,
                "f1_score": 0.85
            },
            "detector_2": {
                "accuracy": 0.82,
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81
            },
            "detector_3": {
                "accuracy": 0.88,
                "precision": 0.82,
                "recall": 0.95,
                "f1_score": 0.88
            }
        }
        
        comparison = service.compare_detectors(detector_performances)
        
        assert "ranking" in comparison
        assert "statistical_significance" in comparison
        assert "best_detector" in comparison
        assert comparison["best_detector"] == "detector_3"


class TestDomainEvents:
    """Comprehensive testing of domain events."""
    
    def test_detector_created_event(self):
        """Test DetectorCreated event."""
        detector_id = DetectorId("detector_001")
        event = DetectorCreated(
            detector_id=detector_id,
            algorithm_name="IsolationForest",
            created_by="test_user",
            timestamp=datetime.utcnow()
        )
        
        assert event.detector_id == detector_id
        assert event.algorithm_name == "IsolationForest"
        assert event.event_type == "DetectorCreated"
        
    def test_anomaly_detected_event(self):
        """Test AnomalyDetected event."""
        event = AnomalyDetected(
            detector_id=DetectorId("detector_001"),
            dataset_id=DatasetId("dataset_001"),
            anomaly_count=5,
            anomaly_rate=0.05,
            severity="high",
            timestamp=datetime.utcnow()
        )
        
        assert event.anomaly_count == 5
        assert event.anomaly_rate == 0.05
        assert event.severity == "high"
        
    def test_experiment_completed_event(self):
        """Test ExperimentCompleted event."""
        event = ExperimentCompleted(
            experiment_id="exp_001",
            detector_count=3,
            best_detector_id=DetectorId("detector_002"),
            best_performance=0.92,
            duration=timedelta(minutes=45),
            timestamp=datetime.utcnow()
        )
        
        assert event.detector_count == 3
        assert event.best_performance == 0.92
        assert event.duration.total_seconds() == 45 * 60
        
    def test_event_serialization(self):
        """Test event serialization and deserialization."""
        event = DetectorCreated(
            detector_id=DetectorId("detector_001"),
            algorithm_name="IsolationForest",
            created_by="test_user",
            timestamp=datetime.utcnow()
        )
        
        # Serialize
        event_dict = event.to_dict()
        assert "detector_id" in event_dict
        assert "algorithm_name" in event_dict
        assert "timestamp" in event_dict
        
        # Deserialize
        restored_event = DetectorCreated.from_dict(event_dict)
        assert restored_event.detector_id == event.detector_id
        assert restored_event.algorithm_name == event.algorithm_name


class TestDomainSpecifications:
    """Comprehensive testing of domain specifications."""
    
    def test_trained_detector_specification(self):
        """Test TrainedDetectorSpecification."""
        spec = TrainedDetectorSpecification()
        
        # Fitted detector should satisfy specification
        fitted_detector = Mock()
        fitted_detector.is_fitted = True
        fitted_detector.performance_metrics = {"accuracy": 0.85}
        
        assert spec.is_satisfied_by(fitted_detector)
        
        # Unfitted detector should not satisfy specification
        unfitted_detector = Mock()
        unfitted_detector.is_fitted = False
        
        assert not spec.is_satisfied_by(unfitted_detector)
        
    def test_high_quality_dataset_specification(self):
        """Test HighQualityDatasetSpecification."""
        spec = HighQualityDatasetSpecification(
            min_completeness=0.95,
            min_consistency=0.90,
            max_duplicates_rate=0.05
        )
        
        # High quality dataset
        good_dataset = Mock()
        good_dataset.quality_metrics = {
            "completeness": 0.98,
            "consistency": 0.95,
            "duplicates_rate": 0.02
        }
        
        assert spec.is_satisfied_by(good_dataset)
        
        # Low quality dataset
        bad_dataset = Mock()
        bad_dataset.quality_metrics = {
            "completeness": 0.80,
            "consistency": 0.85,
            "duplicates_rate": 0.10
        }
        
        assert not spec.is_satisfied_by(bad_dataset)
        
    # def test_anomaly_threshold_specification(self):
    #     """Test AnomalyThresholdSpecification."""
    #     threshold = 0.7
    #     spec = AnomalyThresholdSpecification(threshold)
    #     
    #     # Score above threshold
    #     high_score = AnomalyScore(0.85)
    #     assert spec.is_satisfied_by(high_score)
    #     
    #     # Score below threshold
    #     low_score = AnomalyScore(0.45)
    #     assert not spec.is_satisfied_by(low_score)
        
    def test_specification_composition(self):
        """Test specification composition (AND, OR, NOT)."""
        spec1 = TrainedDetectorSpecification()
        spec2 = Mock()
        spec2.is_satisfied_by.return_value = True
        
        # AND composition
        and_spec = spec1.and_specification(spec2)
        
        fitted_detector = Mock()
        fitted_detector.is_fitted = True
        fitted_detector.performance_metrics = {"accuracy": 0.85}
        
        assert and_spec.is_satisfied_by(fitted_detector)
        
        # OR composition
        or_spec = spec1.or_specification(spec2)
        
        unfitted_detector = Mock()
        unfitted_detector.is_fitted = False
        
        assert or_spec.is_satisfied_by(unfitted_detector)  # spec2 returns True
        
        # NOT composition
        not_spec = spec1.not_specification()
        
        assert not not_spec.is_satisfied_by(fitted_detector)
        assert not_spec.is_satisfied_by(unfitted_detector)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
