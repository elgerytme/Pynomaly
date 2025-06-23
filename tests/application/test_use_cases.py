"""Comprehensive tests for application layer use cases."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, AsyncMock, patch

from pynomaly.application.use_cases import (
    DetectAnomaliesUseCase,
    TrainDetectorUseCase,
    EvaluateModelUseCase,
    ExplainAnomalyUseCase,
)
from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore
from pynomaly.domain.services import FeatureValidator
from pynomaly.infrastructure.repositories import InMemoryDetectorRepository
from pynomaly.domain.exceptions import InvalidValueError


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
    return Dataset(name="test_dataset", features=features, targets=targets)


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return Detector(
        name="test_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1),
        hyperparameters={"n_estimators": 100, "random_state": 42}
    )


@pytest.fixture
def feature_validator():
    """Create a feature validator."""
    return FeatureValidator()


@pytest.fixture
def detector_repository():
    """Create an in-memory detector repository."""
    return InMemoryDetectorRepository()


class TestDetectAnomaliesUseCase:
    """Test DetectAnomaliesUseCase functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, detector_repository, feature_validator, sample_detector, sample_dataset):
        """Test successful anomaly detection execution."""
        # Save detector
        await detector_repository.save(sample_detector)
        
        # Create use case
        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator
        )
        
        # Mock the actual detection algorithm
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_detector_impl.fit.return_value = None
            mock_detector_impl.decision_function.return_value = np.random.random(len(sample_dataset.features))
            mock_detector_impl.predict.return_value = np.random.choice([-1, 1], len(sample_dataset.features))
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            # Execute use case
            result = await use_case.execute(
                detector_id=sample_detector.id,
                dataset=sample_dataset
            )
            
            # Verify result
            assert isinstance(result, DetectionResult)
            assert result.detector.id == sample_detector.id
            assert result.dataset.name == sample_dataset.name
            assert len(result.scores) == len(sample_dataset.features)
            mock_detector_impl.fit.assert_called_once()
            mock_detector_impl.decision_function.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_detector_not_found(self, detector_repository, feature_validator, sample_dataset):
        """Test detection with non-existent detector."""
        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator
        )
        
        with pytest.raises(ValueError, match="Detector.*not found"):
            await use_case.execute(
                detector_id="non_existent",
                dataset=sample_dataset
            )
    
    @pytest.mark.asyncio
    async def test_execute_invalid_features(self, detector_repository, feature_validator, sample_detector):
        """Test detection with invalid features."""
        await detector_repository.save(sample_detector)
        
        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator
        )
        
        # Create dataset with invalid features
        invalid_dataset = Dataset(
            name="invalid",
            features=np.array([[np.inf, 1, 2], [1, np.nan, 3]]),  # Contains inf and nan
            targets=None
        )
        
        with pytest.raises(ValueError, match="Invalid features"):
            await use_case.execute(
                detector_id=sample_detector.id,
                dataset=invalid_dataset
            )
    
    @pytest.mark.asyncio
    async def test_execute_with_preprocessing(self, detector_repository, feature_validator, sample_detector, sample_dataset):
        """Test detection with feature preprocessing."""
        await detector_repository.save(sample_detector)
        
        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator
        )
        
        # Mock preprocessing
        with patch.object(use_case, '_preprocess_features') as mock_preprocess:
            preprocessed_features = sample_dataset.features * 2  # Simple transformation
            mock_preprocess.return_value = preprocessed_features
            
            with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_detector_impl = Mock()
                mock_detector_impl.fit.return_value = None
                mock_detector_impl.decision_function.return_value = np.random.random(len(sample_dataset.features))
                mock_adapter.create_detector.return_value = mock_detector_impl
                mock_adapter_class.return_value = mock_adapter
                
                result = await use_case.execute(
                    detector_id=sample_detector.id,
                    dataset=sample_dataset,
                    preprocess=True
                )
                
                assert isinstance(result, DetectionResult)
                mock_preprocess.assert_called_once_with(sample_dataset.features)
    
    @pytest.mark.asyncio
    async def test_execute_batch_processing(self, detector_repository, feature_validator, sample_detector):
        """Test batch processing of multiple datasets."""
        await detector_repository.save(sample_detector)
        
        use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator
        )
        
        # Create multiple datasets
        datasets = []
        for i in range(3):
            features = np.random.RandomState(42 + i).normal(0, 1, (50, 5))
            dataset = Dataset(name=f"dataset_{i}", features=features)
            datasets.append(dataset)
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_detector_impl.fit.return_value = None
            mock_detector_impl.decision_function.return_value = np.random.random(50)
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            results = await use_case.execute_batch(
                detector_id=sample_detector.id,
                datasets=datasets
            )
            
            assert len(results) == 3
            assert all(isinstance(r, DetectionResult) for r in results)
            # Detector should be fitted once and used for all datasets
            assert mock_detector_impl.fit.call_count == 1
            assert mock_detector_impl.decision_function.call_count == 3


class TestTrainDetectorUseCase:
    """Test TrainDetectorUseCase functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, detector_repository, feature_validator, sample_detector, sample_dataset):
        """Test successful detector training."""
        await detector_repository.save(sample_detector)
        
        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10
        )
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_detector_impl.fit.return_value = None
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            trained_detector = await use_case.execute(
                detector_id=sample_detector.id,
                training_data=sample_dataset
            )
            
            assert isinstance(trained_detector, Detector)
            assert trained_detector.id == sample_detector.id
            mock_detector_impl.fit.assert_called_once_with(sample_dataset.features)
    
    @pytest.mark.asyncio
    async def test_execute_insufficient_samples(self, detector_repository, feature_validator, sample_detector):
        """Test training with insufficient samples."""
        await detector_repository.save(sample_detector)
        
        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=100
        )
        
        # Create small dataset
        small_dataset = Dataset(
            name="small",
            features=np.random.random((5, 3)),  # Only 5 samples
            targets=None
        )
        
        with pytest.raises(ValueError, match="Insufficient training samples"):
            await use_case.execute(
                detector_id=sample_detector.id,
                training_data=small_dataset
            )
    
    @pytest.mark.asyncio
    async def test_execute_with_validation_split(self, detector_repository, feature_validator, sample_detector, sample_dataset):
        """Test training with validation split."""
        await detector_repository.save(sample_detector)
        
        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10
        )
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_detector_impl.fit.return_value = None
            mock_detector_impl.decision_function.return_value = np.random.random(80)  # 80% of 100
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            trained_detector = await use_case.execute(
                detector_id=sample_detector.id,
                training_data=sample_dataset,
                validation_split=0.2
            )
            
            assert isinstance(trained_detector, Detector)
            # Should fit on 80% of data (validation split = 0.2)
            fit_call_args = mock_detector_impl.fit.call_args[0]
            assert len(fit_call_args[0]) == 80  # 80% of 100 samples
    
    @pytest.mark.asyncio
    async def test_execute_hyperparameter_tuning(self, detector_repository, feature_validator, sample_detector, sample_dataset):
        """Test training with hyperparameter tuning."""
        await detector_repository.save(sample_detector)
        
        use_case = TrainDetectorUseCase(
            detector_repository=detector_repository,
            feature_validator=feature_validator,
            min_samples=10
        )
        
        # Define hyperparameter grid
        param_grid = {
            "n_estimators": [50, 100],
            "contamination": [0.1, 0.2]
        }
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_detector_impl.fit.return_value = None
            mock_detector_impl.decision_function.return_value = np.random.random(len(sample_dataset.features))
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            trained_detector = await use_case.execute(
                detector_id=sample_detector.id,
                training_data=sample_dataset,
                hyperparameter_grid=param_grid
            )
            
            assert isinstance(trained_detector, Detector)
            # Should try multiple parameter combinations
            assert mock_adapter.create_detector.call_count >= 2


class TestEvaluateModelUseCase:
    """Test EvaluateModelUseCase functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, detector_repository, sample_detector, sample_dataset):
        """Test successful model evaluation."""
        await detector_repository.save(sample_detector)
        
        use_case = EvaluateModelUseCase(detector_repository=detector_repository)
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_scores = np.random.random(len(sample_dataset.features))
            mock_predictions = np.random.choice([-1, 1], len(sample_dataset.features))
            mock_detector_impl.decision_function.return_value = mock_scores
            mock_detector_impl.predict.return_value = mock_predictions
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            evaluation = await use_case.execute(
                detector_id=sample_detector.id,
                test_data=sample_dataset
            )
            
            assert isinstance(evaluation, dict)
            assert "metrics" in evaluation
            assert "predictions" in evaluation
            assert "scores" in evaluation
    
    @pytest.mark.asyncio
    async def test_execute_with_ground_truth(self, detector_repository, sample_detector, sample_dataset):
        """Test evaluation with ground truth labels."""
        await detector_repository.save(sample_detector)
        
        use_case = EvaluateModelUseCase(detector_repository=detector_repository)
        
        # Dataset already has targets for evaluation
        assert sample_dataset.targets is not None
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_scores = np.random.random(len(sample_dataset.features))
            mock_predictions = np.random.choice([-1, 1], len(sample_dataset.features))
            mock_detector_impl.decision_function.return_value = mock_scores
            mock_detector_impl.predict.return_value = mock_predictions
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            evaluation = await use_case.execute(
                detector_id=sample_detector.id,
                test_data=sample_dataset
            )
            
            assert "metrics" in evaluation
            # Should include classification metrics when ground truth is available
            metrics = evaluation["metrics"]
            assert "accuracy" in metrics or "precision" in metrics or "recall" in metrics
    
    @pytest.mark.asyncio
    async def test_execute_cross_validation(self, detector_repository, sample_detector, sample_dataset):
        """Test evaluation with cross-validation."""
        await detector_repository.save(sample_detector)
        
        use_case = EvaluateModelUseCase(detector_repository=detector_repository)
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_scores = np.random.random(len(sample_dataset.features))
            mock_detector_impl.decision_function.return_value = mock_scores
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            evaluation = await use_case.execute(
                detector_id=sample_detector.id,
                test_data=sample_dataset,
                cross_validation=True,
                cv_folds=5
            )
            
            assert "cv_scores" in evaluation
            assert "cv_mean" in evaluation
            assert "cv_std" in evaluation


class TestExplainAnomalyUseCase:
    """Test ExplainAnomalyUseCase functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, detector_repository, sample_detector, sample_dataset):
        """Test successful anomaly explanation."""
        await detector_repository.save(sample_detector)
        
        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)
        
        # Create sample anomaly
        anomaly = Anomaly(
            score=AnomalyScore(0.95),
            index=0,
            features=sample_dataset.features[0],
            confidence=None,
            explanation=None
        )
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            # Mock explanation (feature importance)
            mock_explanation = {"feature_0": 0.3, "feature_1": 0.5, "feature_2": 0.2}
            mock_detector_impl.explain_anomaly = Mock(return_value=mock_explanation)
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            explanation = await use_case.execute(
                detector_id=sample_detector.id,
                anomaly=anomaly,
                method="feature_importance"
            )
            
            assert isinstance(explanation, dict)
            assert len(explanation) > 0
    
    @pytest.mark.asyncio
    async def test_execute_shap_explanation(self, detector_repository, sample_detector, sample_dataset):
        """Test SHAP-based explanation."""
        await detector_repository.save(sample_detector)
        
        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)
        
        anomaly = Anomaly(
            score=AnomalyScore(0.95),
            index=0,
            features=sample_dataset.features[0],
            confidence=None,
            explanation=None
        )
        
        with patch('shap.Explainer') as mock_shap_explainer:
            mock_explainer = Mock()
            mock_shap_values = np.array([0.1, -0.2, 0.3, 0.0, -0.1])
            mock_explainer.shap_values.return_value = mock_shap_values
            mock_shap_explainer.return_value = mock_explainer
            
            with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_detector_impl = Mock()
                mock_adapter.create_detector.return_value = mock_detector_impl
                mock_adapter_class.return_value = mock_adapter
                
                explanation = await use_case.execute(
                    detector_id=sample_detector.id,
                    anomaly=anomaly,
                    method="shap"
                )
                
                assert isinstance(explanation, dict)
                assert "shap_values" in explanation
    
    @pytest.mark.asyncio
    async def test_execute_lime_explanation(self, detector_repository, sample_detector, sample_dataset):
        """Test LIME-based explanation."""
        await detector_repository.save(sample_detector)
        
        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)
        
        anomaly = Anomaly(
            score=AnomalyScore(0.95),
            index=0,
            features=sample_dataset.features[0],
            confidence=None,
            explanation=None
        )
        
        with patch('lime.lime_tabular.LimeTabularExplainer') as mock_lime:
            mock_explainer = Mock()
            mock_explanation = Mock()
            mock_explanation.as_list.return_value = [("feature_0", 0.3), ("feature_1", -0.2)]
            mock_explainer.explain_instance.return_value = mock_explanation
            mock_lime.return_value = mock_explainer
            
            with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_detector_impl = Mock()
                mock_adapter.create_detector.return_value = mock_detector_impl
                mock_adapter_class.return_value = mock_adapter
                
                explanation = await use_case.execute(
                    detector_id=sample_detector.id,
                    anomaly=anomaly,
                    method="lime"
                )
                
                assert isinstance(explanation, dict)
                assert "lime_explanation" in explanation
    
    @pytest.mark.asyncio
    async def test_execute_batch_explanation(self, detector_repository, sample_detector, sample_dataset):
        """Test batch explanation of multiple anomalies."""
        await detector_repository.save(sample_detector)
        
        use_case = ExplainAnomalyUseCase(detector_repository=detector_repository)
        
        # Create multiple anomalies
        anomalies = []
        for i in range(3):
            anomaly = Anomaly(
                score=AnomalyScore(0.9 + i * 0.02),
                index=i,
                features=sample_dataset.features[i],
                confidence=None,
                explanation=None
            )
            anomalies.append(anomaly)
        
        with patch('pynomaly.infrastructure.adapters.SklearnAdapter') as mock_adapter_class:
            mock_adapter = Mock()
            mock_detector_impl = Mock()
            mock_explanation = {"feature_0": 0.3, "feature_1": 0.5}
            mock_detector_impl.explain_anomaly = Mock(return_value=mock_explanation)
            mock_adapter.create_detector.return_value = mock_detector_impl
            mock_adapter_class.return_value = mock_adapter
            
            explanations = await use_case.execute_batch(
                detector_id=sample_detector.id,
                anomalies=anomalies,
                method="feature_importance"
            )
            
            assert len(explanations) == 3
            assert all(isinstance(exp, dict) for exp in explanations)