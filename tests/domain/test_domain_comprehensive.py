"""Comprehensive tests for domain layer without external dependencies."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite

from tests.conftest_dependencies import requires_dependency

from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import (
    AnomalyScore, ContaminationRate, ConfidenceInterval
)
from pynomaly.domain.exceptions import (
    ValidationError, InvalidParameterError, ProcessingError
)
from pynomaly.domain.services.anomaly_scorer import AnomalyScorer
from pynomaly.domain.services.feature_validator import FeatureValidator
from pynomaly.domain.services.threshold_calculator import ThresholdCalculator


@composite
def valid_anomaly_scores(draw):
    """Generate valid anomaly scores between 0.0 and 1.0."""
    return AnomalyScore(draw(st.floats(min_value=0.0, max_value=1.0)))


@composite
def valid_contamination_rates(draw):
    """Generate valid contamination rates."""
    return ContaminationRate(draw(st.floats(min_value=0.001, max_value=0.5)))


@composite
def valid_confidence_intervals(draw):
    """Generate valid confidence intervals."""
    lower = draw(st.floats(min_value=0.0, max_value=0.8))
    upper = draw(st.floats(min_value=lower + 0.1, max_value=1.0))
    return ConfidenceInterval(lower=lower, upper=upper)


@composite
def sample_datasets(draw):
    """Generate sample datasets with realistic properties."""
    n_samples = draw(st.integers(min_value=10, max_value=1000))
    n_features = draw(st.integers(min_value=1, max_value=20))
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(n_samples, n_features)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    
    return Dataset(
        name=draw(st.text(min_size=1, max_size=50)),
        data=df,
        metadata={"generated": True, "n_samples": n_samples, "n_features": n_features}
    )


class TestDomainValueObjects:
    """Test domain value objects."""
    
    def test_anomaly_score_creation(self):
        """Test AnomalyScore creation and validation."""
        score = AnomalyScore(0.75)
        assert score.value == 0.75
        assert 0.0 <= score.value <= 1.0
    
    def test_anomaly_score_invalid_values(self):
        """Test AnomalyScore validation with invalid values."""
        with pytest.raises((ValueError, ValidationError)):
            AnomalyScore(-0.1)
        
        with pytest.raises((ValueError, ValidationError)):
            AnomalyScore(1.1)
    
    def test_contamination_rate_creation(self):
        """Test ContaminationRate creation and validation."""
        rate = ContaminationRate(0.05)
        assert rate.value == 0.05
        assert 0.0 < rate.value < 1.0
    
    def test_contamination_rate_invalid_values(self):
        """Test ContaminationRate validation with invalid values."""
        with pytest.raises((ValueError, ValidationError)):
            ContaminationRate(0.0)  # Must be > 0
            
        with pytest.raises((ValueError, ValidationError)):
            ContaminationRate(1.0)  # Must be < 1
    
    def test_confidence_interval_creation(self):
        """Test ConfidenceInterval creation and validation."""
        ci = ConfidenceInterval(lower=0.6, upper=0.8)
        assert ci.lower == 0.6
        assert ci.upper == 0.8
        assert ci.width == 0.2
        assert ci.midpoint == 0.7
    
    def test_confidence_interval_invalid_bounds(self):
        """Test ConfidenceInterval validation with invalid bounds."""
        with pytest.raises((ValueError, ValidationError)):
            ConfidenceInterval(lower=0.8, upper=0.6)  # Lower > upper
    
    @given(valid_anomaly_scores())
    @settings(max_examples=50)
    def test_anomaly_score_properties(self, score):
        """Property test: anomaly scores have valid properties."""
        assert 0.0 <= score.value <= 1.0
        assert isinstance(score.value, float)
        assert score.is_anomaly == (score.value > 0.5)  # Assuming 0.5 threshold
    
    @given(valid_contamination_rates())
    @settings(max_examples=50)
    def test_contamination_rate_properties(self, rate):
        """Property test: contamination rates have valid properties."""
        assert 0.0 < rate.value < 1.0
        assert isinstance(rate.value, float)
    
    @given(valid_confidence_intervals())
    @settings(max_examples=50)
    def test_confidence_interval_properties(self, ci):
        """Property test: confidence intervals have valid properties."""
        assert ci.lower <= ci.upper
        assert ci.width >= 0
        assert ci.lower <= ci.midpoint <= ci.upper


class TestDomainEntities:
    """Test domain entities."""
    
    def test_dataset_creation(self):
        """Test Dataset entity creation."""
        data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10]
        })
        
        dataset = Dataset(
            name="test_dataset",
            data=data,
            metadata={"source": "test"}
        )
        
        assert dataset.name == "test_dataset"
        assert dataset.data.shape == (5, 2)
        assert dataset.metadata["source"] == "test"
        assert dataset.feature_names == ['feature_1', 'feature_2']
    
    def test_detector_creation(self):
        """Test Detector entity creation."""
        detector = Detector(
            name="test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100}
        )
        
        assert detector.name == "test_detector"
        assert detector.algorithm == "IsolationForest"
        assert detector.parameters["contamination"] == 0.1
        assert detector.parameters["n_estimators"] == 100
        assert not detector.is_fitted
    
    def test_anomaly_creation(self):
        """Test Anomaly entity creation."""
        anomaly = Anomaly(
            index=42,
            score=AnomalyScore(0.85),
            features={"feature_1": 1.5, "feature_2": -2.3},
            timestamp=datetime.now()
        )
        
        assert anomaly.index == 42
        assert anomaly.score.value == 0.85
        assert anomaly.features["feature_1"] == 1.5
        assert isinstance(anomaly.timestamp, datetime)
    
    def test_detection_result_creation(self):
        """Test DetectionResult entity creation."""
        scores = [AnomalyScore(0.1), AnomalyScore(0.8), AnomalyScore(0.3)]
        anomalies = [
            Anomaly(index=1, score=AnomalyScore(0.8), features={"f1": 1.0})
        ]
        
        result = DetectionResult(
            detector_name="test_detector",
            dataset_name="test_dataset",
            scores=scores,
            anomalies=anomalies,
            threshold=0.5,
            execution_time=0.123
        )
        
        assert result.detector_name == "test_detector"
        assert result.dataset_name == "test_dataset"
        assert len(result.scores) == 3
        assert len(result.anomalies) == 1
        assert result.threshold == 0.5
        assert result.execution_time == 0.123
    
    @given(sample_datasets())
    @settings(max_examples=20)
    def test_dataset_properties(self, dataset):
        """Property test: datasets have consistent properties."""
        assert dataset.name is not None
        assert len(dataset.name) > 0
        assert dataset.data is not None
        assert dataset.data.shape[0] > 0  # At least one row
        assert dataset.data.shape[1] > 0  # At least one column
        assert len(dataset.feature_names) == dataset.data.shape[1]


class TestDomainServices:
    """Test domain services."""
    
    def test_anomaly_scorer_basic_functionality(self):
        """Test AnomalyScorer basic functionality."""
        scorer = AnomalyScorer()
        
        # Mock raw scores from an algorithm
        raw_scores = np.array([-0.5, -0.1, 0.8, 0.2, -0.3])
        
        # Test score normalization
        normalized = scorer.normalize_scores(raw_scores)
        assert len(normalized) == len(raw_scores)
        assert all(isinstance(score, AnomalyScore) for score in normalized)
        assert all(0.0 <= score.value <= 1.0 for score in normalized)
    
    def test_feature_validator_basic_functionality(self):
        """Test FeatureValidator basic functionality."""
        validator = FeatureValidator()
        
        # Valid data
        valid_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        # Should not raise for valid data
        try:
            validator.validate_features(valid_data)
        except Exception as e:
            pytest.fail(f"Validation failed for valid data: {e}")
        
        # Invalid data - all NaN column
        invalid_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        # Should handle NaN appropriately
        validator.validate_features(invalid_data)  # Should not crash
    
    def test_threshold_calculator_basic_functionality(self):
        """Test ThresholdCalculator basic functionality."""
        calculator = ThresholdCalculator()
        
        # Generate test scores
        scores = [AnomalyScore(x) for x in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]]
        contamination = ContaminationRate(0.3)  # 30% contamination
        
        # Calculate threshold
        threshold = calculator.calculate_threshold(scores, contamination)
        
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0
        
        # With 30% contamination, threshold should be around the 70th percentile
        # Since we have 6 scores, 30% would be ~2 scores as anomalies
        # So threshold should be between 0.6 and 0.8
        assert 0.5 <= threshold <= 0.9


class TestDomainExceptions:
    """Test domain exceptions."""
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Invalid input parameter")
        assert str(error) == "Invalid input parameter"
        assert isinstance(error, Exception)
    
    def test_invalid_parameter_error(self):
        """Test InvalidParameterError exception."""
        error = InvalidParameterError("Parameter out of range")
        assert str(error) == "Parameter out of range"
        assert isinstance(error, Exception)
    
    def test_processing_error(self):
        """Test ProcessingError exception."""
        error = ProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, Exception)


class TestDomainBusinessLogic:
    """Test domain business logic."""
    
    def test_anomaly_detection_workflow(self):
        """Test end-to-end anomaly detection workflow in domain layer."""
        # Create test data
        data = pd.DataFrame({
            'feature_1': [1, 2, 3, 10, 2],  # 10 is an outlier
            'feature_2': [0.1, 0.2, 0.3, 5.0, 0.2]  # 5.0 is an outlier
        })
        
        dataset = Dataset(name="test", data=data)
        detector = Detector(
            name="test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.2}
        )
        
        # Mock algorithm scores (would come from infrastructure layer)
        mock_scores = [0.1, 0.2, 0.15, 0.9, 0.18]  # High score for outlier
        anomaly_scores = [AnomalyScore(score) for score in mock_scores]
        
        # Calculate threshold
        calculator = ThresholdCalculator()
        contamination = ContaminationRate(0.2)
        threshold = calculator.calculate_threshold(anomaly_scores, contamination)
        
        # Identify anomalies
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score.value > threshold:
                anomaly = Anomaly(
                    index=i,
                    score=score,
                    features=data.iloc[i].to_dict(),
                    timestamp=datetime.now()
                )
                anomalies.append(anomaly)
        
        # Create result
        result = DetectionResult(
            detector_name=detector.name,
            dataset_name=dataset.name,
            scores=anomaly_scores,
            anomalies=anomalies,
            threshold=threshold,
            execution_time=0.05
        )
        
        # Verify results
        assert len(result.scores) == 5
        assert len(result.anomalies) >= 1  # Should detect at least the obvious outlier
        assert result.threshold > 0
        assert all(anomaly.score.value > threshold for anomaly in result.anomalies)
    
    def test_detector_lifecycle(self):
        """Test detector lifecycle management."""
        detector = Detector(
            name="lifecycle_test",
            algorithm="TestAlgorithm",
            parameters={"param1": "value1"}
        )
        
        # Initial state
        assert not detector.is_fitted
        assert detector.fitted_model is None
        
        # Simulate training
        detector.is_fitted = True
        detector.fitted_model = Mock()  # Mock trained model
        detector.metadata["training_samples"] = 1000
        detector.metadata["training_time"] = 2.5
        
        # Verify trained state
        assert detector.is_fitted
        assert detector.fitted_model is not None
        assert detector.metadata["training_samples"] == 1000
        assert detector.metadata["training_time"] == 2.5
    
    def test_dataset_validation_workflow(self):
        """Test dataset validation workflow."""
        # Valid dataset
        valid_data = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0, 4.0],
            'another_feature': [0.5, 1.5, 2.5, 3.5]
        })
        
        valid_dataset = Dataset(name="valid", data=valid_data)
        
        validator = FeatureValidator()
        
        # Should not raise for valid dataset
        try:
            validator.validate_features(valid_dataset.data)
            validation_passed = True
        except Exception:
            validation_passed = False
        
        assert validation_passed
        
        # Check dataset properties
        assert valid_dataset.n_samples == 4
        assert valid_dataset.n_features == 2
        assert set(valid_dataset.feature_names) == {'numeric_feature', 'another_feature'}


@requires_dependency('hypothesis')
class TestPropertyBasedDomainTests:
    """Property-based tests for domain layer."""
    
    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_anomaly_score_list_properties(self, score_values):
        """Property test: lists of anomaly scores maintain invariants."""
        scores = [AnomalyScore(value) for value in score_values]
        
        # All scores should be valid
        assert all(0.0 <= score.value <= 1.0 for score in scores)
        assert len(scores) == len(score_values)
        
        # Score statistics should be reasonable
        if len(scores) > 1:
            score_values_list = [s.value for s in scores]
            assert min(score_values_list) >= 0.0
            assert max(score_values_list) <= 1.0
    
    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20)
    def test_dataset_size_properties(self, n_samples, n_features):
        """Property test: datasets with different sizes work correctly."""
        # Generate random data
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(data, columns=[f'f_{i}' for i in range(n_features)])
        
        dataset = Dataset(name="property_test", data=df)
        
        assert dataset.n_samples == n_samples
        assert dataset.n_features == n_features
        assert len(dataset.feature_names) == n_features
        assert dataset.data.shape == (n_samples, n_features)
    
    @given(
        st.floats(min_value=0.001, max_value=0.499),
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=10, max_size=100)
    )
    @settings(max_examples=20)
    def test_threshold_calculation_properties(self, contamination_value, score_values):
        """Property test: threshold calculation maintains invariants."""
        contamination = ContaminationRate(contamination_value)
        scores = [AnomalyScore(value) for value in score_values]
        
        calculator = ThresholdCalculator()
        threshold = calculator.calculate_threshold(scores, contamination)
        
        # Threshold should be within score range
        assert 0.0 <= threshold <= 1.0
        
        # Number of scores above threshold should approximately match contamination rate
        scores_above_threshold = sum(1 for score in scores if score.value > threshold)
        expected_anomalies = int(len(scores) * contamination_value)
        
        # Allow some tolerance due to discrete nature of threshold calculation
        tolerance = max(1, len(scores) * 0.1)  # 10% tolerance or at least 1
        assert abs(scores_above_threshold - expected_anomalies) <= tolerance