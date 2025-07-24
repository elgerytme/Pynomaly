"""Unit tests for ExplainabilityService."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from anomaly_detection.domain.services.explainability_service import (
    ExplainabilityService,
    ExplainerType,
    ExplanationResult
)
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.entities.detection_result import DetectionResult


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, has_decision_function=True, fail_predict=False):
        self.has_decision_function = has_decision_function
        self.fail_predict = fail_predict
    
    def predict(self, X):
        if self.fail_predict:
            raise ValueError("Predict failed")
        return np.array([1 if i % 2 == 0 else -1 for i in range(len(X))])
    
    def decision_function(self, X):
        if not self.has_decision_function:
            raise AttributeError("No decision_function")
        return np.array([0.5 if i % 2 == 0 else -0.5 for i in range(len(X))])


class TestExplainabilityService:
    """Test suite for ExplainabilityService."""
    
    @pytest.fixture
    def mock_detection_service(self):
        """Create mock detection service."""
        mock_service = Mock(spec=DetectionService)
        
        # Mock predict method
        def mock_predict(data, algorithm):
            predictions = np.array([-1])  # Anomaly
            confidence_scores = np.array([0.8])
            return DetectionResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                algorithm=algorithm
            )
        
        mock_service.predict.side_effect = mock_predict
        mock_service._fitted_models = {}
        
        return mock_service
    
    @pytest.fixture
    def explainability_service(self, mock_detection_service):
        """Create explainability service with mock detection service."""
        return ExplainabilityService(detection_service=mock_detection_service)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    @pytest.fixture
    def sample_batch(self):
        """Create batch of sample data."""
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    @pytest.fixture
    def feature_names(self):
        """Create feature names."""
        return ["temp", "pressure", "flow", "voltage", "current"]
    
    def test_initialization_with_detection_service(self, mock_detection_service):
        """Test explainability service initialization with provided detection service."""
        service = ExplainabilityService(detection_service=mock_detection_service)
        assert service.detection_service is mock_detection_service
        assert len(service.available_explainers) >= 2  # At least feature_importance and permutation
        assert ExplainerType.FEATURE_IMPORTANCE in service.available_explainers
        assert ExplainerType.PERMUTATION in service.available_explainers
    
    def test_initialization_without_detection_service(self):
        """Test explainability service initialization without detection service."""
        service = ExplainabilityService()
        assert isinstance(service.detection_service, DetectionService)
        assert len(service.available_explainers) >= 2
    
    def test_explain_prediction_feature_importance(self, explainability_service, sample_data, feature_names):
        """Test prediction explanation using feature importance."""
        # Add mock model
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            explainer_type=ExplainerType.FEATURE_IMPORTANCE,
            feature_names=feature_names
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.explainer_type == "feature_importance"
        assert result.feature_names == feature_names
        assert len(result.feature_importance) == len(feature_names)
        assert result.is_anomaly is True  # Mock returns -1 (anomaly)
        assert result.prediction_confidence == 0.8
        assert result.data_sample == sample_data.tolist()
        assert result.top_features is not None
        assert len(result.top_features) <= 5
    
    def test_explain_prediction_with_list_input(self, explainability_service, feature_names):
        """Test prediction explanation with list input."""
        sample_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_list,
            algorithm="iforest",
            feature_names=feature_names
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.data_sample == sample_list
    
    def test_explain_prediction_without_feature_names(self, explainability_service, sample_data):
        """Test prediction explanation without provided feature names."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest"
        )
        
        expected_names = [f"feature_{i}" for i in range(len(sample_data))]
        assert result.feature_names == expected_names
    
    def test_explain_prediction_normal_sample(self, explainability_service, sample_data, feature_names):
        """Test prediction explanation for normal sample."""
        # Mock detection service to return normal prediction
        def mock_predict_normal(data, algorithm):
            predictions = np.array([1])  # Normal
            confidence_scores = np.array([0.6])
            return DetectionResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                algorithm=algorithm
            )
        
        explainability_service.detection_service.predict.side_effect = mock_predict_normal
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            feature_names=feature_names
        )
        
        assert result.is_anomaly is False
        assert result.prediction_confidence == 0.6
    
    def test_explain_prediction_permutation(self, explainability_service, sample_data, feature_names):
        """Test prediction explanation using permutation importance."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        with patch('numpy.random.normal', return_value=0.0):  # Make permutation predictable
            result = explainability_service.explain_prediction(
                sample=sample_data,
                algorithm="iforest",
                explainer_type=ExplainerType.PERMUTATION,
                feature_names=feature_names
            )
        
        assert isinstance(result, ExplanationResult)
        assert result.explainer_type == "permutation"
        assert result.feature_names == feature_names
        assert len(result.feature_importance) == len(feature_names)
        assert result.base_value is not None
        assert result.metadata["method"] == "permutation_importance"
    
    def test_explain_prediction_model_not_fitted(self, explainability_service, sample_data, feature_names):
        """Test prediction explanation when model is not fitted."""
        # No model in _fitted_models
        
        with pytest.raises(ValueError) as exc_info:
            explainability_service.explain_prediction(
                sample=sample_data,
                algorithm="unfitted_algo",
                explainer_type=ExplainerType.PERMUTATION,
                feature_names=feature_names
            )
        
        assert "Model for algorithm 'unfitted_algo' not fitted" in str(exc_info.value)
    
    @patch('anomaly_detection.domain.services.explainability_service.SHAP_AVAILABLE', True)
    def test_explain_prediction_shap_not_available(self, explainability_service, sample_data, feature_names):
        """Test SHAP explanation when SHAP is not available."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        with patch('anomaly_detection.domain.services.explainability_service.shap', None):
            # Should fallback to feature importance
            result = explainability_service.explain_prediction(
                sample=sample_data,
                algorithm="iforest",
                explainer_type=ExplainerType.SHAP,
                feature_names=feature_names
            )
        
        assert result.explainer_type == "feature_importance"  # Fallback
    
    @patch('anomaly_detection.domain.services.explainability_service.LIME_AVAILABLE', True)
    def test_explain_prediction_lime_without_training_data(self, explainability_service, sample_data, feature_names):
        """Test LIME explanation without training data."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        with pytest.raises(ValueError) as exc_info:
            explainability_service.explain_prediction(
                sample=sample_data,
                algorithm="iforest",
                explainer_type=ExplainerType.LIME,
                feature_names=feature_names
            )
        
        assert "Training data required for LIME explainer" in str(exc_info.value)
    
    def test_explain_prediction_model_without_decision_function(self, explainability_service, sample_data, feature_names):
        """Test explanation with model that doesn't have decision_function."""
        mock_model = MockModel(has_decision_function=False)
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            explainer_type=ExplainerType.PERMUTATION,
            feature_names=feature_names
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.explainer_type == "permutation"
    
    def test_explain_prediction_model_predict_failure(self, explainability_service, sample_data, feature_names):
        """Test handling of model prediction failure."""
        mock_model = MockModel(fail_predict=True)
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        # Should fallback to feature importance when permutation fails
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            explainer_type=ExplainerType.PERMUTATION,
            feature_names=feature_names
        )
        
        assert result.explainer_type == "feature_importance"  # Fallback
    
    def test_feature_importance_zero_values(self, explainability_service, feature_names):
        """Test feature importance with all zero values."""
        zero_sample = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=zero_sample,
            algorithm="iforest",
            feature_names=feature_names
        )
        
        # All importance should be 0
        for importance in result.feature_importance.values():
            assert importance == 0.0
    
    def test_get_top_features(self, explainability_service):
        """Test getting top contributing features."""
        feature_importance = {
            "temp": 0.8,
            "pressure": 0.2,
            "flow": 0.9,
            "voltage": 0.1,
            "current": 0.5
        }
        sample_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        top_features = explainability_service._get_top_features(
            feature_importance, sample_values, limit=3
        )
        
        assert len(top_features) == 3
        assert top_features[0]["feature_name"] == "flow"  # Highest importance
        assert top_features[0]["importance"] == 0.9
        assert top_features[0]["rank"] == 1
        assert top_features[1]["feature_name"] == "temp"  # Second highest
        assert top_features[1]["importance"] == 0.8
        assert top_features[1]["rank"] == 2
    
    def test_explain_batch(self, explainability_service, sample_batch, feature_names):
        """Test explaining batch of predictions."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        results = explainability_service.explain_batch(
            samples=sample_batch,
            algorithm="iforest",
            feature_names=feature_names[:3]  # Match sample dimensions
        )
        
        assert len(results) == len(sample_batch)
        for result in results:
            assert isinstance(result, ExplanationResult)
            assert result.feature_names == feature_names[:3]
            assert len(result.feature_importance) == 3
    
    def test_explain_batch_different_explainer_types(self, explainability_service, sample_batch, feature_names):
        """Test explaining batch with different explainer types."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        # Test with permutation
        with patch('numpy.random.normal', return_value=0.0):
            results = explainability_service.explain_batch(
                samples=sample_batch,
                algorithm="iforest",
                explainer_type=ExplainerType.PERMUTATION,
                feature_names=feature_names[:3]
            )
        
        assert len(results) == len(sample_batch)
        for result in results:
            assert result.explainer_type == "permutation"
    
    def test_get_global_feature_importance(self, explainability_service, sample_batch, feature_names):
        """Test getting global feature importance across multiple samples."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        # Create larger training data
        training_data = np.random.randn(50, 3)
        
        global_importance = explainability_service.get_global_feature_importance(
            algorithm="iforest",
            training_data=training_data,
            feature_names=feature_names[:3],
            n_samples=10
        )
        
        assert len(global_importance) == 3
        for feature_name in feature_names[:3]:
            assert feature_name in global_importance
            assert isinstance(global_importance[feature_name], float)
            assert global_importance[feature_name] >= 0.0
    
    def test_get_global_feature_importance_without_feature_names(self, explainability_service):
        """Test global feature importance without provided feature names."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        training_data = np.random.randn(20, 4)
        
        global_importance = explainability_service.get_global_feature_importance(
            algorithm="iforest",
            training_data=training_data,
            n_samples=5
        )
        
        expected_names = [f"feature_{i}" for i in range(4)]
        assert list(global_importance.keys()) == expected_names
    
    def test_get_global_feature_importance_more_samples_than_data(self, explainability_service):
        """Test global feature importance when requesting more samples than available."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        training_data = np.random.randn(5, 3)  # Only 5 samples
        
        global_importance = explainability_service.get_global_feature_importance(
            algorithm="iforest",
            training_data=training_data,
            n_samples=10  # Request more than available
        )
        
        # Should use all available samples
        assert len(global_importance) == 3
    
    def test_get_available_explainers(self, explainability_service):
        """Test getting available explainer types."""
        available = explainability_service.get_available_explainers()
        
        assert isinstance(available, list)
        assert "feature_importance" in available
        assert "permutation" in available
        # SHAP and LIME may or may not be available depending on imports
    
    def test_explainer_caching(self, explainability_service, sample_data, feature_names):
        """Test that explainers are cached for reuse."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        # First explanation
        result1 = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            explainer_type=ExplainerType.PERMUTATION,
            feature_names=feature_names
        )
        
        # Second explanation with same algorithm
        result2 = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest", 
            explainer_type=ExplainerType.PERMUTATION,
            feature_names=feature_names
        )
        
        # Both should succeed (tests that explainer caching works)
        assert isinstance(result1, ExplanationResult)
        assert isinstance(result2, ExplanationResult)
    
    def test_explanation_with_no_confidence_scores(self, explainability_service, sample_data, feature_names):
        """Test explanation when detection result has no confidence scores."""
        # Mock detection service to return result without confidence scores
        def mock_predict_no_confidence(data, algorithm):
            predictions = np.array([-1])  # Anomaly
            return DetectionResult(
                predictions=predictions,
                confidence_scores=None,  # No confidence scores
                algorithm=algorithm
            )
        
        explainability_service.detection_service.predict.side_effect = mock_predict_no_confidence
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            feature_names=feature_names
        )
        
        assert result.prediction_confidence is None
    
    def test_feature_importance_normalization(self, explainability_service, feature_names):
        """Test feature importance normalization with different value ranges."""
        # Sample with widely different feature magnitudes
        sample_data = np.array([1000.0, 0.1, 50.0, 0.001, 500.0])
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            feature_names=feature_names
        )
        
        # The feature with highest absolute value should have importance 1.0
        max_importance = max(result.feature_importance.values())
        assert max_importance == 1.0
        
        # All importance values should be between 0 and 1
        for importance in result.feature_importance.values():
            assert 0.0 <= importance <= 1.0
    
    @pytest.mark.parametrize("explainer_type", [
        ExplainerType.FEATURE_IMPORTANCE,
        ExplainerType.PERMUTATION,
    ])
    def test_different_explainer_types(self, explainability_service, sample_data, 
                                     feature_names, explainer_type):
        """Test different explainer types produce valid results."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        if explainer_type == ExplainerType.PERMUTATION:
            with patch('numpy.random.normal', return_value=0.0):
                result = explainability_service.explain_prediction(
                    sample=sample_data,
                    algorithm="iforest",
                    explainer_type=explainer_type,
                    feature_names=feature_names
                )
        else:
            result = explainability_service.explain_prediction(
                sample=sample_data,
                algorithm="iforest",
                explainer_type=explainer_type,
                feature_names=feature_names
            )
        
        assert isinstance(result, ExplanationResult)
        assert result.explainer_type == explainer_type.value
        assert len(result.feature_importance) == len(feature_names)
        assert result.top_features is not None
    
    def test_explanation_result_structure(self, explainability_service, sample_data, feature_names):
        """Test that ExplanationResult has all expected fields."""
        mock_model = MockModel()
        explainability_service.detection_service._fitted_models["iforest"] = mock_model
        
        result = explainability_service.explain_prediction(
            sample=sample_data,
            algorithm="iforest",
            feature_names=feature_names
        )
        
        # Check all required fields are present
        assert hasattr(result, 'explainer_type')
        assert hasattr(result, 'feature_names')
        assert hasattr(result, 'feature_importance')
        assert hasattr(result, 'explanation_values')
        assert hasattr(result, 'base_value')
        assert hasattr(result, 'data_sample')
        assert hasattr(result, 'is_anomaly')
        assert hasattr(result, 'prediction_confidence')
        assert hasattr(result, 'top_features')
        assert hasattr(result, 'metadata')
        
        # Check field types
        assert isinstance(result.explainer_type, str)
        assert isinstance(result.feature_names, list)
        assert isinstance(result.feature_importance, dict)
        assert isinstance(result.data_sample, list)
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.top_features, list)
        assert isinstance(result.metadata, dict)