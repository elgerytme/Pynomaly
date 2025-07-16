"""
Fixed unit tests for domain value objects.
Tests that demonstrate improved test coverage for domain components.
"""

from dataclasses import FrozenInstanceError

import pytest

from monorepo.domain.value_objects.model_metrics import ModelMetrics


class TestModelMetricsFixed:
    """Tests for the fixed ModelMetrics value object."""

    def test_model_metrics_creation_valid(self):
        """Test creating ModelMetrics with valid values."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            auc_score=0.93
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.90
        assert metrics.auc_score == 0.93
        assert metrics.metadata == {}

    def test_model_metrics_with_custom_metadata(self):
        """Test creating ModelMetrics with custom metadata."""
        metadata = {"model_type": "isolation_forest", "dataset_size": 1000}
        metrics = ModelMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.82,
            f1_score=0.81,
            metadata=metadata
        )

        assert metrics.metadata == metadata

    def test_model_metrics_default_auc_score(self):
        """Test that auc_score defaults to 0.0."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )

        assert metrics.auc_score == 0.0

    def test_model_metrics_validation_accuracy_too_high(self):
        """Test validation fails when accuracy is too high."""
        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=1.5,  # Invalid: > 1.0
                precision=0.92,
                recall=0.88,
                f1_score=0.90
            )

    def test_model_metrics_validation_precision_too_low(self):
        """Test validation fails when precision is too low."""
        with pytest.raises(ValueError, match="precision must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=0.95,
                precision=-0.1,  # Invalid: < 0.0
                recall=0.88,
                f1_score=0.90
            )

    def test_model_metrics_validation_recall_boundary_values(self):
        """Test validation with boundary values for recall."""
        # Test minimum boundary
        metrics_min = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.0,  # Valid boundary
            f1_score=0.90
        )
        assert metrics_min.recall == 0.0

        # Test maximum boundary
        metrics_max = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=1.0,  # Valid boundary
            f1_score=0.90
        )
        assert metrics_max.recall == 1.0

    def test_model_metrics_validation_f1_score_invalid(self):
        """Test validation fails when f1_score is invalid."""
        with pytest.raises(ValueError, match="f1_score must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=0.95,
                precision=0.92,
                recall=0.88,
                f1_score=1.1  # Invalid: > 1.0
            )

    def test_model_metrics_validation_auc_score_invalid(self):
        """Test validation fails when auc_score is invalid."""
        with pytest.raises(ValueError, match="auc_score must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=0.95,
                precision=0.92,
                recall=0.88,
                f1_score=0.90,
                auc_score=-0.5  # Invalid: < 0.0
            )

    def test_model_metrics_to_dict(self):
        """Test converting ModelMetrics to dictionary."""
        metadata = {"test": "value"}
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            auc_score=0.93,
            metadata=metadata
        )

        result = metrics.to_dict()
        expected = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
            "auc_score": 0.93,
            "metadata": metadata
        }

        assert result == expected

    def test_model_metrics_from_dict(self):
        """Test creating ModelMetrics from dictionary."""
        data = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
            "auc_score": 0.93,
            "metadata": {"test": "value"}
        }

        metrics = ModelMetrics.from_dict(data)

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.90
        assert metrics.auc_score == 0.93
        assert metrics.metadata == {"test": "value"}

    def test_model_metrics_from_dict_missing_optional_fields(self):
        """Test creating ModelMetrics from dictionary with missing optional fields."""
        data = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90
            # Missing auc_score and metadata
        }

        metrics = ModelMetrics.from_dict(data)

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.90
        assert metrics.auc_score == 0.0  # Default value
        assert metrics.metadata == {}  # Default value

    def test_model_metrics_immutability(self):
        """Test that ModelMetrics is immutable (frozen dataclass)."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )

        # Should not be able to modify frozen dataclass
        with pytest.raises(FrozenInstanceError):
            metrics.accuracy = 0.90  # type: ignore

    def test_model_metrics_equality(self):
        """Test equality comparison of ModelMetrics instances."""
        metrics1 = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )

        metrics2 = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )

        metrics3 = ModelMetrics(
            accuracy=0.90,  # Different value
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )

        assert metrics1 == metrics2
        assert metrics1 != metrics3

    def test_model_metrics_str_representation(self):
        """Test string representation of ModelMetrics."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )

        str_repr = str(metrics)
        assert "ModelMetrics" in str_repr
        assert "accuracy=0.95" in str_repr
        assert "precision=0.92" in str_repr
