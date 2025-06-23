"""Enhanced tests for application use cases - Phase 1 Coverage Enhancement."""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime
from typing import Optional, Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4

from pynomaly.application.use_cases.detect_anomalies import (
    DetectAnomaliesUseCase,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse
)
from pynomaly.application.use_cases.train_detector import (
    TrainDetectorUseCase,
    TrainDetectorRequest,
    TrainDetectorResponse
)
from pynomaly.application.use_cases.evaluate_model import (
    EvaluateModelUseCase,
    EvaluateModelRequest,
    EvaluateModelResponse
)
from pynomaly.application.use_cases.explain_anomaly import (
    ExplainAnomalyUseCase,
    ExplainAnomalyRequest,
    ExplainAnomalyResponse
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore, ConfidenceInterval
from pynomaly.domain.exceptions import DetectorNotFittedError, DatasetError, InsufficientDataError, FittingError
from pynomaly.domain.services import FeatureValidator
from pynomaly.infrastructure.repositories import InMemoryDetectorRepository


class TestDetectAnomaliesUseCaseEnhanced:
    """Enhanced tests for DetectAnomaliesUseCase covering edge cases."""
    
    @pytest.fixture
    def feature_validator(self):
        """Create a mock FeatureValidator."""
        validator = Mock(spec=FeatureValidator)
        validator.check_data_quality.return_value = {"quality_score": 0.9}
        validator.suggest_preprocessing.return_value = []
        return validator
    
    @pytest.fixture
    def use_case(self, feature_validator):
        """Create DetectAnomaliesUseCase."""
        detector_repo = InMemoryDetectorRepository()
        return DetectAnomaliesUseCase(
            detector_repository=detector_repo,
            feature_validator=feature_validator
        )
    
    @pytest.mark.asyncio
    async def test_execute_with_low_quality_data(self, use_case, feature_validator):
        """Test detection with low quality data that generates warnings."""
        # Create detector
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1)
        )
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        # Create dataset
        features = np.random.random((100, 5))
        dataset = Dataset(name="low_quality_dataset", features=features)
        
        # Mock low quality data
        feature_validator.check_data_quality.return_value = {
            "quality_score": 0.6,  # Low quality
            "missing_values": ["feature_0", "feature_2"],
            "constant_features": ["feature_4"]
        }
        feature_validator.suggest_preprocessing.return_value = [
            "Consider imputing missing values",
            "Remove constant features"
        ]
        
        # Mock detector behavior
        mock_result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=[],
            scores=[AnomalyScore(0.5)] * 100,
            labels=np.array([0] * 100),
            threshold=0.7
        )
        
        with patch.object(detector, 'detect', return_value=mock_result):
            request = DetectAnomaliesRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_features=True,
                save_results=True
            )
            
            response = await use_case.execute(request)
            
            assert isinstance(response, DetectAnomaliesResponse)
            assert response.quality_report is not None
            assert response.quality_report["quality_score"] == 0.6
            assert response.warnings is not None
            assert len(response.warnings) == 3  # 1 quality warning + 2 suggestions
            assert "Data quality score is low: 0.60" in response.warnings[0]
    
    @pytest.mark.asyncio
    async def test_execute_without_validation(self, use_case):
        """Test detection without feature validation."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        
        mock_result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=[],
            scores=[AnomalyScore(0.5)] * 50,
            labels=np.array([0] * 50),
            threshold=0.7
        )
        
        with patch.object(detector, 'detect', return_value=mock_result):
            request = DetectAnomaliesRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_features=False,  # No validation
                save_results=False
            )
            
            response = await use_case.execute(request)
            
            assert response.quality_report is None
            assert response.warnings is None
    
    @pytest.mark.asyncio
    async def test_execute_detector_not_fitted(self, use_case):
        """Test detection with unfitted detector."""
        detector = Detector(name="unfitted", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = False  # Not fitted
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        
        request = DetectAnomaliesRequest(detector_id=detector.id, dataset=dataset)
        
        with pytest.raises(DetectorNotFittedError) as exc_info:
            await use_case.execute(request)
        
        assert exc_info.value.detector_name == "unfitted"
        assert exc_info.value.operation == "detect"
    
    @pytest.mark.asyncio
    async def test_execute_detector_not_found(self, use_case):
        """Test detection with non-existent detector."""
        non_existent_id = uuid4()
        features = np.random.random((20, 2))
        dataset = Dataset(name="test", features=features)
        
        request = DetectAnomaliesRequest(detector_id=non_existent_id, dataset=dataset)
        
        with pytest.raises(ValueError, match="Detector .* not found"):
            await use_case.execute(request)


class TestTrainDetectorUseCaseEnhanced:
    """Enhanced tests for TrainDetectorUseCase covering edge cases."""
    
    @pytest.fixture
    def feature_validator(self):
        """Create a mock FeatureValidator."""
        validator = Mock(spec=FeatureValidator)
        validator.validate_numeric_features.return_value = ["feature_0", "feature_1", "feature_2"]
        validator.check_data_quality.return_value = {
            "quality_score": 0.85,
            "missing_values": [],
            "constant_features": []
        }
        return validator
    
    @pytest.fixture
    def use_case(self, feature_validator):
        """Create TrainDetectorUseCase."""
        detector_repo = InMemoryDetectorRepository()
        return TrainDetectorUseCase(
            detector_repository=detector_repo,
            feature_validator=feature_validator,
            min_samples=10
        )
    
    @pytest.mark.asyncio
    async def test_execute_successful_training(self, use_case, feature_validator):
        """Test successful detector training."""
        detector = Detector(
            name="trainable_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1)
        )
        await use_case.detector_repository.save(detector)
        
        # Create sufficient training data
        features = np.random.RandomState(42).normal(0, 1, (100, 5))
        dataset = Dataset(name="training_data", features=features)
        
        # Mock successful fitting
        with patch.object(detector, 'fit') as mock_fit:
            with patch.object(detector, 'update_parameters') as mock_update:
                request = TrainDetectorRequest(
                    detector_id=detector.id,
                    dataset=dataset,
                    contamination_rate=ContaminationRate(0.15),
                    parameters={"n_estimators": 200, "random_state": 42},
                    validate_data=True,
                    save_model=True
                )
                
                response = await use_case.execute(request)
                
                assert isinstance(response, TrainDetectorResponse)
                assert response.detector_id == detector.id
                assert response.training_time_ms > 0
                assert response.validation_results is not None
                assert response.validation_results["numeric_features"] == 3
                assert response.validation_results["quality_score"] == 0.85
                
                # Verify interactions
                mock_update.assert_called_once_with(n_estimators=200, random_state=42)
                mock_fit.assert_called_once_with(dataset)
                assert detector.contamination_rate.value == 0.15
    
    @pytest.mark.asyncio
    async def test_execute_insufficient_data(self, use_case):
        """Test training with insufficient data."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await use_case.detector_repository.save(detector)
        
        # Create insufficient data (less than min_samples=10)
        features = np.random.random((5, 3))
        dataset = Dataset(name="small_dataset", features=features)
        
        request = TrainDetectorRequest(detector_id=detector.id, dataset=dataset)
        
        with pytest.raises(InsufficientDataError) as exc_info:
            await use_case.execute(request)
        
        assert exc_info.value.dataset_name == "small_dataset"
        assert exc_info.value.n_samples == 5
        assert exc_info.value.min_required == 10
        assert exc_info.value.operation == "training"
    
    @pytest.mark.asyncio
    async def test_execute_no_numeric_features(self, use_case, feature_validator):
        """Test training with no numeric features."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="non_numeric_dataset", features=features)
        
        # Mock no numeric features found
        feature_validator.validate_numeric_features.return_value = []
        
        request = TrainDetectorRequest(
            detector_id=detector.id,
            dataset=dataset,
            validate_data=True
        )
        
        with pytest.raises(FittingError) as exc_info:
            await use_case.execute(request)
        
        assert exc_info.value.detector_name == "test"
        assert "No numeric features found" in exc_info.value.reason
        assert exc_info.value.dataset_name == "non_numeric_dataset"
    
    @pytest.mark.asyncio
    async def test_execute_fitting_failure(self, use_case):
        """Test training when fitting fails."""
        detector = Detector(name="failing_detector", algorithm="test", contamination=ContaminationRate(0.1))
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="test_dataset", features=features)
        
        # Mock fitting failure
        with patch.object(detector, 'fit', side_effect=RuntimeError("Fitting failed")):
            request = TrainDetectorRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_data=False  # Skip validation to reach fitting
            )
            
            with pytest.raises(FittingError) as exc_info:
                await use_case.execute(request)
            
            assert exc_info.value.detector_name == "failing_detector"
            assert "Fitting failed" in exc_info.value.reason
            assert exc_info.value.dataset_name == "test_dataset"
    
    @pytest.mark.asyncio
    async def test_execute_with_data_quality_issues(self, use_case, feature_validator):
        """Test training with data quality issues that generate warnings."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 5))
        dataset = Dataset(name="problematic_dataset", features=features)
        
        # Mock data quality issues
        feature_validator.check_data_quality.return_value = {
            "quality_score": 0.7,
            "missing_values": ["feature_0", "feature_3"],
            "constant_features": ["feature_4"]
        }
        
        with patch.object(detector, 'fit'):
            request = TrainDetectorRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_data=True
            )
            
            response = await use_case.execute(request)
            
            assert response.validation_results is not None
            assert len(response.validation_results["issues"]) == 2
            assert "Missing values in 2 features" in response.validation_results["issues"][0]
            assert "1 constant features" in response.validation_results["issues"][1]
    
    @pytest.mark.asyncio
    async def test_execute_without_validation(self, use_case):
        """Test training without data validation."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((30, 3))
        dataset = Dataset(name="test", features=features)
        
        with patch.object(detector, 'fit'):
            request = TrainDetectorRequest(
                detector_id=detector.id,
                dataset=dataset,
                validate_data=False  # No validation
            )
            
            response = await use_case.execute(request)
            
            assert response.validation_results is None


class TestEvaluateModelUseCaseEnhanced:
    """Enhanced tests for EvaluateModelUseCase covering various scenarios."""
    
    @pytest.fixture
    def use_case(self):
        """Create EvaluateModelUseCase."""
        detector_repo = InMemoryDetectorRepository()
        return EvaluateModelUseCase(detector_repository=detector_repo)
    
    @pytest.mark.asyncio
    async def test_execute_with_labeled_data(self, use_case):
        """Test model evaluation with labeled data."""
        detector = Detector(name="eval_detector", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        # Create dataset with labels
        features = np.random.RandomState(42).normal(0, 1, (100, 5))
        targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
        dataset = Dataset(name="labeled_test_data", features=features, targets=targets)
        
        # Mock detector predictions
        mock_scores = [AnomalyScore(v) for v in np.random.RandomState(42).beta(2, 8, 100)]
        mock_labels = np.random.RandomState(42).choice([0, 1], size=100, p=[0.92, 0.08])
        
        with patch.object(detector, 'score', return_value=mock_scores):
            with patch.object(detector, 'predict', return_value=mock_labels):
                request = EvaluateModelRequest(
                    detector_id=detector.id,
                    test_data=dataset,
                    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"]
                )
                
                response = await use_case.execute(request)
                
                assert isinstance(response, EvaluateModelResponse)
                assert response.evaluation_metrics is not None
                assert "accuracy" in response.evaluation_metrics
                assert "precision" in response.evaluation_metrics
                assert "recall" in response.evaluation_metrics
                assert "f1" in response.evaluation_metrics
                assert "roc_auc" in response.evaluation_metrics
    
    @pytest.mark.asyncio
    async def test_execute_without_labels(self, use_case):
        """Test model evaluation without ground truth labels."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        # Create dataset without labels
        features = np.random.random((80, 4))
        dataset = Dataset(name="unlabeled_test_data", features=features)
        
        mock_scores = [AnomalyScore(v) for v in np.random.random(80)]
        
        with patch.object(detector, 'score', return_value=mock_scores):
            request = EvaluateModelRequest(
                detector_id=detector.id,
                test_data=dataset,
                metrics=["score_statistics"]  # Unsupervised metrics
            )
            
            response = await use_case.execute(request)
            
            assert response.evaluation_metrics is not None
            # Should contain unsupervised metrics like score statistics
            assert len(response.evaluation_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_execute_with_cross_validation(self, use_case):
        """Test model evaluation with cross-validation."""
        detector = Detector(name="cv_detector", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        features = np.random.RandomState(42).normal(0, 1, (150, 6))
        targets = np.random.RandomState(42).choice([0, 1], size=150, p=[0.88, 0.12])
        dataset = Dataset(name="cv_test_data", features=features, targets=targets)
        
        # Mock CV predictions
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.85, 0.87, 0.83, 0.89, 0.86])
            
            mock_scores = [AnomalyScore(v) for v in np.random.random(150)]
            with patch.object(detector, 'score', return_value=mock_scores):
                request = EvaluateModelRequest(
                    detector_id=detector.id,
                    test_data=dataset,
                    cross_validation=True,
                    cv_folds=5,
                    metrics=["f1"]
                )
                
                response = await use_case.execute(request)
                
                assert response.cross_validation_scores is not None
                assert len(response.cross_validation_scores) > 0
    
    @pytest.mark.asyncio
    async def test_execute_detector_not_fitted(self, use_case):
        """Test evaluation with unfitted detector."""
        detector = Detector(name="unfitted", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = False
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        
        request = EvaluateModelRequest(detector_id=detector.id, test_data=dataset)
        
        with pytest.raises(DetectorNotFittedError):
            await use_case.execute(request)


class TestExplainAnomalyUseCaseEnhanced:
    """Enhanced tests for ExplainAnomalyUseCase covering different explanation methods."""
    
    @pytest.fixture
    def use_case(self):
        """Create ExplainAnomalyUseCase."""
        detector_repo = InMemoryDetectorRepository()
        return ExplainAnomalyUseCase(detector_repository=detector_repo)
    
    @pytest.mark.asyncio
    async def test_execute_feature_importance_explanation(self, use_case):
        """Test anomaly explanation using feature importance."""
        detector = Detector(name="explainable_detector", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        features = np.random.RandomState(42).normal(0, 1, (100, 5))
        dataset = Dataset(name="test_data", features=features)
        
        # Mock explanation methods
        with patch.object(detector, 'explain_instance') as mock_explain:
            mock_explain.return_value = {
                "feature_importance": {
                    "feature_0": 0.3,
                    "feature_1": 0.25,
                    "feature_2": 0.2,
                    "feature_3": 0.15,
                    "feature_4": 0.1
                },
                "anomaly_score": 0.85,
                "explanation_method": "feature_importance"
            }
            
            request = ExplainAnomalyRequest(
                detector_id=detector.id,
                dataset=dataset,
                instance_index=42,
                explanation_method="feature_importance"
            )
            
            response = await use_case.execute(request)
            
            assert isinstance(response, ExplainAnomalyResponse)
            assert response.explanation is not None
            assert "feature_importance" in response.explanation
            assert response.explanation["anomaly_score"] == 0.85
            assert response.explanation["explanation_method"] == "feature_importance"
    
    @pytest.mark.asyncio
    async def test_execute_shap_explanation(self, use_case):
        """Test anomaly explanation using SHAP values."""
        detector = Detector(name="shap_detector", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((80, 4))
        dataset = Dataset(name="shap_data", features=features)
        
        with patch.object(detector, 'explain_instance') as mock_explain:
            mock_explain.return_value = {
                "shap_values": [0.1, -0.2, 0.3, -0.15],
                "base_value": 0.5,
                "anomaly_score": 0.73,
                "explanation_method": "shap"
            }
            
            request = ExplainAnomalyRequest(
                detector_id=detector.id,
                dataset=dataset,
                instance_index=25,
                explanation_method="shap"
            )
            
            response = await use_case.execute(request)
            
            assert response.explanation["shap_values"] == [0.1, -0.2, 0.3, -0.15]
            assert response.explanation["base_value"] == 0.5
            assert response.explanation["explanation_method"] == "shap"
    
    @pytest.mark.asyncio
    async def test_execute_invalid_instance_index(self, use_case):
        """Test explanation with invalid instance index."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((50, 3))
        dataset = Dataset(name="test", features=features)
        
        request = ExplainAnomalyRequest(
            detector_id=detector.id,
            dataset=dataset,
            instance_index=100,  # Out of bounds
            explanation_method="feature_importance"
        )
        
        with pytest.raises(ValueError, match="Instance index .* out of bounds"):
            await use_case.execute(request)
    
    @pytest.mark.asyncio
    async def test_execute_unsupported_explanation_method(self, use_case):
        """Test explanation with unsupported method."""
        detector = Detector(name="test", algorithm="test", contamination=ContaminationRate(0.1))
        detector._is_fitted = True
        await use_case.detector_repository.save(detector)
        
        features = np.random.random((30, 2))
        dataset = Dataset(name="test", features=features)
        
        with patch.object(detector, 'explain_instance', side_effect=NotImplementedError("Method not supported")):
            request = ExplainAnomalyRequest(
                detector_id=detector.id,
                dataset=dataset,
                instance_index=10,
                explanation_method="unsupported_method"
            )
            
            with pytest.raises(NotImplementedError, match="Method not supported"):
                await use_case.execute(request)