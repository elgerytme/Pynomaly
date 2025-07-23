"""Comprehensive unit tests for PredictionResult entity."""

import pytest
import numpy as np
from datetime import datetime
from uuid import UUID, uuid4
from unittest.mock import patch

from machine_learning.domain.entities.prediction_result import PredictionResult


@pytest.fixture
def sample_features():
    """Sample feature vector."""
    return np.array([1.2, -0.5, 2.1, 0.8, -1.1])


@pytest.fixture
def sample_metadata():
    """Sample metadata dictionary."""
    return {
        'feature_importance': {'feature_1': 0.6, 'feature_2': 0.4},
        'processing_time': 0.045,
        'model_type': 'random_forest'
    }


@pytest.fixture
def sample_prediction_result(sample_features, sample_metadata):
    """Sample prediction result instance."""
    return PredictionResult(
        result_id=uuid4(),
        sample_id="sample_123",
        model_id="model_456",
        score=0.85,
        prediction=True,
        confidence=0.92,
        features=sample_features,
        model_version="v1.2.0",
        metadata=sample_metadata,
        explanation="High anomaly score due to unusual feature pattern"
    )


class TestPredictionResult:
    """Test cases for PredictionResult entity."""

    def test_initialization_all_fields(self, sample_features, sample_metadata):
        """Test initialization with all fields provided."""
        result_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        
        result = PredictionResult(
            result_id=result_id,
            sample_id="sample_123",
            model_id="model_456",
            score=0.85,
            prediction=True,
            confidence=0.92,
            features=sample_features,
            model_version="v1.2.0",
            created_at=created_at,
            metadata=sample_metadata,
            explanation="Test explanation"
        )
        
        assert result.result_id == result_id
        assert result.sample_id == "sample_123"
        assert result.model_id == "model_456"
        assert result.score == 0.85
        assert result.prediction is True
        assert result.confidence == 0.92
        assert np.array_equal(result.features, sample_features)
        assert result.model_version == "v1.2.0"
        assert result.created_at == created_at
        assert result.metadata == sample_metadata
        assert result.explanation == "Test explanation"

    def test_initialization_minimal_fields(self):
        """Test initialization with only required fields."""
        result_id = uuid4()
        
        result = PredictionResult(
            result_id=result_id,
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=False,
            confidence=0.68
        )
        
        assert result.result_id == result_id
        assert result.sample_id == "sample_123"
        assert result.model_id == "model_456"
        assert result.score == 0.75
        assert result.prediction is False
        assert result.confidence == 0.68
        assert result.features is None
        assert result.model_version is None
        assert result.explanation is None

    def test_post_init_sets_created_at(self):
        """Test that __post_init__ sets created_at if None."""
        with patch('machine_learning.domain.entities.prediction_result.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            result = PredictionResult(
                result_id=uuid4(),
                sample_id="sample_123",
                model_id="model_456",
                score=0.75,
                prediction=True,
                confidence=0.8,
                created_at=None
            )
            
            assert result.created_at == mock_now

    def test_post_init_preserves_created_at(self):
        """Test that __post_init__ preserves existing created_at."""
        existing_time = datetime(2024, 1, 10, 9, 0, 0)
        
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            created_at=existing_time
        )
        
        assert result.created_at == existing_time

    def test_post_init_sets_metadata(self):
        """Test that __post_init__ sets metadata if None."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            metadata=None
        )
        
        assert result.metadata == {}

    def test_post_init_preserves_metadata(self, sample_metadata):
        """Test that __post_init__ preserves existing metadata."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            metadata=sample_metadata
        )
        
        assert result.metadata == sample_metadata

    def test_create_classmethod(self, sample_features, sample_metadata):
        """Test the create classmethod."""
        with patch('machine_learning.domain.entities.prediction_result.datetime') as mock_datetime, \
             patch('machine_learning.domain.entities.prediction_result.uuid4') as mock_uuid:
            
            mock_now = datetime(2024, 1, 15, 14, 30, 0)
            mock_datetime.now.return_value = mock_now
            
            mock_id = uuid4()
            mock_uuid.return_value = mock_id
            
            result = PredictionResult.create(
                sample_id="sample_789",
                model_id="model_101",
                score=0.92,
                prediction=True,
                confidence=0.88,
                features=sample_features,
                model_version="v2.0.0",
                metadata=sample_metadata,
                explanation="Created via classmethod"
            )
            
            assert result.result_id == mock_id
            assert result.sample_id == "sample_789"
            assert result.model_id == "model_101"
            assert result.score == 0.92
            assert result.prediction is True
            assert result.confidence == 0.88
            assert np.array_equal(result.features, sample_features)
            assert result.model_version == "v2.0.0"
            assert result.created_at == mock_now
            assert result.metadata == sample_metadata
            assert result.explanation == "Created via classmethod"

    def test_create_with_none_metadata(self):
        """Test create method with None metadata."""
        result = PredictionResult.create(
            sample_id="sample_789",
            model_id="model_101",
            score=0.92,
            prediction=True,
            confidence=0.88,
            metadata=None
        )
        
        assert result.metadata == {}

    def test_create_minimal_params(self):
        """Test create method with minimal parameters."""
        result = PredictionResult.create(
            sample_id="sample_minimal",
            model_id="model_minimal",
            score=0.55,
            prediction=False,
            confidence=0.45
        )
        
        assert result.sample_id == "sample_minimal"
        assert result.model_id == "model_minimal"
        assert result.score == 0.55
        assert result.prediction is False
        assert result.confidence == 0.45
        assert result.features is None
        assert result.model_version is None
        assert result.metadata == {}
        assert result.explanation is None

    def test_is_positive_true(self, sample_prediction_result):
        """Test is_positive returns True for positive prediction."""
        sample_prediction_result.prediction = True
        assert sample_prediction_result.is_positive() is True

    def test_is_positive_false(self, sample_prediction_result):
        """Test is_positive returns False for negative prediction."""
        sample_prediction_result.prediction = False
        assert sample_prediction_result.is_positive() is False

    def test_is_high_confidence_default_threshold(self, sample_prediction_result):
        """Test is_high_confidence with default threshold (0.8)."""
        sample_prediction_result.confidence = 0.85
        assert sample_prediction_result.is_high_confidence() is True
        
        sample_prediction_result.confidence = 0.75
        assert sample_prediction_result.is_high_confidence() is False
        
        sample_prediction_result.confidence = 0.8
        assert sample_prediction_result.is_high_confidence() is True

    def test_is_high_confidence_custom_threshold(self, sample_prediction_result):
        """Test is_high_confidence with custom threshold."""
        sample_prediction_result.confidence = 0.75
        
        assert sample_prediction_result.is_high_confidence(threshold=0.7) is True
        assert sample_prediction_result.is_high_confidence(threshold=0.8) is False
        assert sample_prediction_result.is_high_confidence(threshold=0.75) is True

    def test_is_high_confidence_edge_cases(self, sample_prediction_result):
        """Test is_high_confidence edge cases."""
        sample_prediction_result.confidence = 0.0
        assert sample_prediction_result.is_high_confidence(threshold=0.0) is True
        assert sample_prediction_result.is_high_confidence(threshold=0.1) is False
        
        sample_prediction_result.confidence = 1.0
        assert sample_prediction_result.is_high_confidence(threshold=1.0) is True
        assert sample_prediction_result.is_high_confidence(threshold=0.99) is True

    def test_get_feature_importance_exists(self, sample_prediction_result):
        """Test get_feature_importance when feature importance exists."""
        expected_importance = {'feature_1': 0.6, 'feature_2': 0.4}
        sample_prediction_result.metadata['feature_importance'] = expected_importance
        
        assert sample_prediction_result.get_feature_importance() == expected_importance

    def test_get_feature_importance_not_exists(self, sample_prediction_result):
        """Test get_feature_importance when feature importance doesn't exist."""
        sample_prediction_result.metadata.pop('feature_importance', None)
        
        assert sample_prediction_result.get_feature_importance() is None

    def test_get_feature_importance_empty_metadata(self):
        """Test get_feature_importance with empty metadata."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            metadata={}
        )
        
        assert result.get_feature_importance() is None

    def test_add_explanation(self, sample_prediction_result):
        """Test adding explanation to result."""
        new_explanation = "Updated explanation with more details"
        sample_prediction_result.add_explanation(new_explanation)
        
        assert sample_prediction_result.explanation == new_explanation

    def test_add_explanation_overwrites(self, sample_prediction_result):
        """Test that add_explanation overwrites existing explanation."""
        original_explanation = sample_prediction_result.explanation
        new_explanation = "Completely new explanation"
        
        sample_prediction_result.add_explanation(new_explanation)
        
        assert sample_prediction_result.explanation == new_explanation
        assert sample_prediction_result.explanation != original_explanation

    def test_update_metadata_existing_key(self, sample_prediction_result):
        """Test updating existing metadata key."""
        sample_prediction_result.update_metadata('processing_time', 0.089)
        
        assert sample_prediction_result.metadata['processing_time'] == 0.089

    def test_update_metadata_new_key(self, sample_prediction_result):
        """Test adding new metadata key."""
        sample_prediction_result.update_metadata('new_metric', 'new_value')
        
        assert sample_prediction_result.metadata['new_metric'] == 'new_value'

    def test_update_metadata_none_metadata(self):
        """Test update_metadata when metadata is None."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            metadata=None
        )
        
        result.update_metadata('test_key', 'test_value')
        
        assert result.metadata == {'test_key': 'test_value'}

    def test_update_metadata_complex_values(self, sample_prediction_result):
        """Test updating metadata with complex values."""
        complex_value = {'nested': {'key': [1, 2, 3]}, 'list': ['a', 'b']}
        sample_prediction_result.update_metadata('complex_data', complex_value)
        
        assert sample_prediction_result.metadata['complex_data'] == complex_value

    def test_score_validation_edge_cases(self):
        """Test score with edge case values."""
        # Test with boundary values
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.0,
            prediction=False,
            confidence=0.0
        )
        assert result.score == 0.0
        
        result.score = 1.0
        assert result.score == 1.0
        
        # Test with values outside normal range (should be allowed)
        result.score = -0.5
        assert result.score == -0.5
        
        result.score = 1.5
        assert result.score == 1.5

    def test_confidence_validation_edge_cases(self):
        """Test confidence with edge case values."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.5,
            prediction=True,
            confidence=0.0
        )
        assert result.confidence == 0.0
        
        result.confidence = 1.0
        assert result.confidence == 1.0

    def test_features_array_operations(self, sample_features):
        """Test operations with features array."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            features=sample_features
        )
        
        # Test array operations
        assert len(result.features) == len(sample_features)
        assert np.array_equal(result.features, sample_features)
        assert result.features.dtype == sample_features.dtype

    def test_features_none_handling(self):
        """Test handling of None features."""
        result = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.75,
            prediction=True,
            confidence=0.8,
            features=None
        )
        
        assert result.features is None

    def test_dataclass_equality(self, sample_features, sample_metadata):
        """Test equality between PredictionResult instances."""
        result_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        
        result1 = PredictionResult(
            result_id=result_id,
            sample_id="sample_123",
            model_id="model_456",
            score=0.85,
            prediction=True,
            confidence=0.92,
            features=sample_features,
            model_version="v1.0.0",
            created_at=created_at,
            metadata=sample_metadata,
            explanation="Same explanation"
        )
        
        result2 = PredictionResult(
            result_id=result_id,
            sample_id="sample_123",
            model_id="model_456",
            score=0.85,
            prediction=True,
            confidence=0.92,
            features=sample_features,
            model_version="v1.0.0",
            created_at=created_at,
            metadata=sample_metadata,
            explanation="Same explanation"
        )
        
        assert result1 == result2

    def test_dataclass_inequality(self):
        """Test inequality between PredictionResult instances."""
        result1 = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_123",
            model_id="model_456",
            score=0.85,
            prediction=True,
            confidence=0.92
        )
        
        result2 = PredictionResult(
            result_id=uuid4(),
            sample_id="sample_456",
            model_id="model_789",
            score=0.75,
            prediction=False,
            confidence=0.68
        )
        
        assert result1 != result2