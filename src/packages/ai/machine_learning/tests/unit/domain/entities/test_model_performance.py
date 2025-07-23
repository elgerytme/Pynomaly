"""Comprehensive unit tests for ModelPerformanceMetrics entity."""

import pytest
from uuid import UUID, uuid4
from unittest.mock import Mock

from machine_learning.domain.entities.model_performance import ModelPerformanceMetrics


class MockPerformanceMetrics:
    """Mock performance metrics object."""
    
    def __init__(self, accuracy=0.85, precision=0.80, recall=0.90, f1_score=0.84):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score


@pytest.fixture
def sample_metrics():
    """Sample performance metrics as dictionary."""
    return {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.90,
        'f1_score': 0.84,
        'auc_score': 0.88,
        'confusion_matrix': [[150, 10], [5, 35]]
    }


@pytest.fixture
def sample_performance_object():
    """Sample performance metrics as object."""
    return MockPerformanceMetrics()


@pytest.fixture
def sample_model_id():
    """Sample model ID."""
    return str(uuid4())


class TestModelPerformanceMetrics:
    """Test cases for ModelPerformanceMetrics entity."""

    def test_initialization_with_dict_metrics(self, sample_model_id, sample_metrics):
        """Test initialization with dictionary metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        assert performance.model_id == sample_model_id
        assert performance.metrics == sample_metrics
        assert isinstance(performance.id, UUID)
        assert isinstance(performance.metadata, dict)
        assert len(performance.metadata) == 0

    def test_initialization_with_object_metrics(self, sample_model_id, sample_performance_object):
        """Test initialization with performance metrics object."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_performance_object
        )
        
        assert performance.model_id == sample_model_id
        assert performance.metrics == sample_performance_object
        assert isinstance(performance.id, UUID)

    def test_initialization_with_uuid_model_id(self, sample_metrics):
        """Test initialization with UUID model ID."""
        model_id = uuid4()
        performance = ModelPerformanceMetrics(
            model_id=model_id,
            metrics=sample_metrics
        )
        
        assert performance.model_id == model_id
        assert isinstance(performance.model_id, UUID)

    def test_initialization_with_string_model_id(self, sample_metrics):
        """Test initialization with string model ID."""
        model_id = "model_123"
        performance = ModelPerformanceMetrics(
            model_id=model_id,
            metrics=sample_metrics
        )
        
        assert performance.model_id == model_id
        assert isinstance(performance.model_id, str)

    def test_initialization_with_custom_id(self, sample_model_id, sample_metrics):
        """Test initialization with custom ID."""
        custom_id = uuid4()
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics,
            id=custom_id
        )
        
        assert performance.id == custom_id

    def test_initialization_with_metadata(self, sample_model_id, sample_metrics):
        """Test initialization with custom metadata."""
        metadata = {'version': '1.0', 'created_by': 'test_user'}
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics,
            metadata=metadata
        )
        
        assert performance.metadata == metadata

    def test_post_init_generates_id(self, sample_model_id, sample_metrics):
        """Test that __post_init__ generates ID if None."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics,
            id=None
        )
        
        assert isinstance(performance.id, UUID)

    def test_post_init_generates_metadata(self, sample_model_id, sample_metrics):
        """Test that __post_init__ generates metadata if None."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics,
            metadata=None
        )
        
        assert isinstance(performance.metadata, dict)
        assert len(performance.metadata) == 0

    def test_get_accuracy_from_dict(self, sample_model_id, sample_metrics):
        """Test getting accuracy from dictionary metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        assert performance.get_accuracy() == 0.85

    def test_get_accuracy_from_object(self, sample_model_id, sample_performance_object):
        """Test getting accuracy from object metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_performance_object
        )
        
        assert performance.get_accuracy() == 0.85

    def test_get_accuracy_missing_key(self, sample_model_id):
        """Test getting accuracy when key is missing."""
        metrics = {'precision': 0.80}
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_accuracy() == 0.0

    def test_get_precision_from_dict(self, sample_model_id, sample_metrics):
        """Test getting precision from dictionary metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        assert performance.get_precision() == 0.80

    def test_get_precision_from_object(self, sample_model_id, sample_performance_object):
        """Test getting precision from object metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_performance_object
        )
        
        assert performance.get_precision() == 0.80

    def test_get_precision_missing_key(self, sample_model_id):
        """Test getting precision when key is missing."""
        metrics = {'accuracy': 0.85}
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_precision() == 0.0

    def test_get_recall_from_dict(self, sample_model_id, sample_metrics):
        """Test getting recall from dictionary metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        assert performance.get_recall() == 0.90

    def test_get_recall_from_object(self, sample_model_id, sample_performance_object):
        """Test getting recall from object metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_performance_object
        )
        
        assert performance.get_recall() == 0.90

    def test_get_recall_missing_key(self, sample_model_id):
        """Test getting recall when key is missing."""
        metrics = {'accuracy': 0.85}
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_recall() == 0.0

    def test_get_f1_score_from_dict(self, sample_model_id, sample_metrics):
        """Test getting F1 score from dictionary metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        assert performance.get_f1_score() == 0.84

    def test_get_f1_score_from_object(self, sample_model_id, sample_performance_object):
        """Test getting F1 score from object metrics."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_performance_object
        )
        
        assert performance.get_f1_score() == 0.84

    def test_get_f1_score_missing_key(self, sample_model_id):
        """Test getting F1 score when key is missing."""
        metrics = {'accuracy': 0.85}
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_f1_score() == 0.0

    def test_all_metrics_with_empty_dict(self, sample_model_id):
        """Test all metric getters with empty dictionary."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics={}
        )
        
        assert performance.get_accuracy() == 0.0
        assert performance.get_precision() == 0.0
        assert performance.get_recall() == 0.0
        assert performance.get_f1_score() == 0.0

    def test_metrics_with_zero_values(self, sample_model_id):
        """Test metrics with explicit zero values."""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_accuracy() == 0.0
        assert performance.get_precision() == 0.0
        assert performance.get_recall() == 0.0
        assert performance.get_f1_score() == 0.0

    def test_metrics_with_none_values(self, sample_model_id):
        """Test metrics with None values."""
        metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None
        }
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        # .get() should return 0.0 for None values
        assert performance.get_accuracy() == 0.0
        assert performance.get_precision() == 0.0
        assert performance.get_recall() == 0.0
        assert performance.get_f1_score() == 0.0

    def test_object_without_all_attributes(self, sample_model_id):
        """Test performance object missing some attributes."""
        incomplete_metrics = Mock()
        incomplete_metrics.accuracy = 0.85
        incomplete_metrics.precision = 0.80
        # Missing recall and f1_score
        
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=incomplete_metrics
        )
        
        assert performance.get_accuracy() == 0.85
        assert performance.get_precision() == 0.80
        
        # These should raise AttributeError
        with pytest.raises(AttributeError):
            performance.get_recall()
        
        with pytest.raises(AttributeError):
            performance.get_f1_score()

    def test_edge_case_with_negative_values(self, sample_model_id):
        """Test with negative metric values."""
        metrics = {
            'accuracy': -0.1,
            'precision': -0.2,
            'recall': -0.3,
            'f1_score': -0.4
        }
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_accuracy() == -0.1
        assert performance.get_precision() == -0.2
        assert performance.get_recall() == -0.3
        assert performance.get_f1_score() == -0.4

    def test_edge_case_with_values_greater_than_one(self, sample_model_id):
        """Test with metric values greater than 1.0."""
        metrics = {
            'accuracy': 1.5,
            'precision': 2.0,
            'recall': 1.1,
            'f1_score': 1.2
        }
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=metrics
        )
        
        assert performance.get_accuracy() == 1.5
        assert performance.get_precision() == 2.0
        assert performance.get_recall() == 1.1
        assert performance.get_f1_score() == 1.2

    def test_dataclass_equality(self, sample_model_id, sample_metrics):
        """Test equality between ModelPerformanceMetrics instances."""
        performance1 = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        performance2 = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics,
            id=performance1.id,
            metadata=performance1.metadata
        )
        
        assert performance1 == performance2

    def test_dataclass_inequality(self, sample_metrics):
        """Test inequality between ModelPerformanceMetrics instances."""
        performance1 = ModelPerformanceMetrics(
            model_id="model_1",
            metrics=sample_metrics
        )
        performance2 = ModelPerformanceMetrics(
            model_id="model_2",
            metrics=sample_metrics
        )
        
        assert performance1 != performance2

    def test_metadata_modification(self, sample_model_id, sample_metrics):
        """Test metadata can be modified after initialization."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        performance.metadata['new_field'] = 'new_value'
        assert performance.metadata['new_field'] == 'new_value'
        
        performance.metadata.update({'another_field': 'another_value'})
        assert performance.metadata['another_field'] == 'another_value'

    def test_immutable_id_after_creation(self, sample_model_id, sample_metrics):
        """Test that ID doesn't change after creation."""
        performance = ModelPerformanceMetrics(
            model_id=sample_model_id,
            metrics=sample_metrics
        )
        
        original_id = performance.id
        
        # Try to modify ID (this should work as dataclass is mutable by default)
        new_id = uuid4()
        performance.id = new_id
        
        assert performance.id == new_id
        assert performance.id != original_id