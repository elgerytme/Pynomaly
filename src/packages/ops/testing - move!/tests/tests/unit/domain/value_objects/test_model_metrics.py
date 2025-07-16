"""Tests for ModelMetrics value object."""

import pytest

from monorepo.domain.value_objects.model_metrics import ModelMetrics


class TestModelMetrics:
    """Test suite for ModelMetrics value object."""

    def test_basic_creation(self):
        """Test basic ModelMetrics creation."""
        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.auc_score == 0.0  # Default value
        assert metrics.metadata == {}  # Default empty dict

    def test_creation_with_all_fields(self):
        """Test ModelMetrics creation with all fields."""
        metadata = {"model_type": "IsolationForest", "training_time": 120}

        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            auc_score=0.92,
            metadata=metadata,
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.auc_score == 0.92
        assert metrics.metadata == metadata

    def test_frozen_dataclass(self):
        """Test that ModelMetrics is immutable."""
        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        with pytest.raises(AttributeError):
            metrics.accuracy = 0.80

    def test_validation_accuracy_range(self):
        """Test accuracy validation."""
        # Valid accuracy
        ModelMetrics(accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87)

        # Invalid accuracy - below 0
        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=-0.1, precision=0.90, recall=0.85, f1_score=0.87)

        # Invalid accuracy - above 1
        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=1.1, precision=0.90, recall=0.85, f1_score=0.87)

    def test_validation_precision_range(self):
        """Test precision validation."""
        # Valid precision
        ModelMetrics(accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87)

        # Invalid precision - below 0
        with pytest.raises(ValueError, match="precision must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=0.95, precision=-0.1, recall=0.85, f1_score=0.87)

        # Invalid precision - above 1
        with pytest.raises(ValueError, match="precision must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=0.95, precision=1.5, recall=0.85, f1_score=0.87)

    def test_validation_recall_range(self):
        """Test recall validation."""
        # Valid recall
        ModelMetrics(accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87)

        # Invalid recall - below 0
        with pytest.raises(ValueError, match="recall must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=0.95, precision=0.90, recall=-0.1, f1_score=0.87)

        # Invalid recall - above 1
        with pytest.raises(ValueError, match="recall must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=0.95, precision=0.90, recall=1.2, f1_score=0.87)

    def test_validation_f1_score_range(self):
        """Test f1_score validation."""
        # Valid f1_score
        ModelMetrics(accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87)

        # Invalid f1_score - below 0
        with pytest.raises(ValueError, match="f1_score must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=0.95, precision=0.90, recall=0.85, f1_score=-0.1)

        # Invalid f1_score - above 1
        with pytest.raises(ValueError, match="f1_score must be between 0.0 and 1.0"):
            ModelMetrics(accuracy=0.95, precision=0.90, recall=0.85, f1_score=1.3)

    def test_validation_auc_score_range(self):
        """Test auc_score validation."""
        # Valid auc_score
        ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87, auc_score=0.92
        )

        # Invalid auc_score - below 0
        with pytest.raises(ValueError, match="auc_score must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=0.95,
                precision=0.90,
                recall=0.85,
                f1_score=0.87,
                auc_score=-0.1,
            )

        # Invalid auc_score - above 1
        with pytest.raises(ValueError, match="auc_score must be between 0.0 and 1.0"):
            ModelMetrics(
                accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87, auc_score=1.1
            )

    def test_boundary_values(self):
        """Test boundary values for all metrics."""
        # All zeros
        metrics_zero = ModelMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0, auc_score=0.0
        )
        assert metrics_zero.accuracy == 0.0

        # All ones
        metrics_one = ModelMetrics(
            accuracy=1.0, precision=1.0, recall=1.0, f1_score=1.0, auc_score=1.0
        )
        assert metrics_one.accuracy == 1.0

    def test_to_dict(self):
        """Test to_dict method."""
        metadata = {"model_type": "IsolationForest"}
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            auc_score=0.92,
            metadata=metadata,
        )

        result = metrics.to_dict()
        expected = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1_score": 0.87,
            "auc_score": 0.92,
            "metadata": metadata,
        }

        assert result == expected

    def test_to_dict_with_defaults(self):
        """Test to_dict with default values."""
        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        result = metrics.to_dict()
        expected = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1_score": 0.87,
            "auc_score": 0.0,
            "metadata": {},
        }

        assert result == expected

    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1_score": 0.87,
            "auc_score": 0.92,
            "metadata": {"model_type": "IsolationForest"},
        }

        metrics = ModelMetrics.from_dict(data)

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.auc_score == 0.92
        assert metrics.metadata == {"model_type": "IsolationForest"}

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict with missing optional fields."""
        data = {"accuracy": 0.95, "precision": 0.90, "recall": 0.85, "f1_score": 0.87}

        metrics = ModelMetrics.from_dict(data)

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.auc_score == 0.0  # Default value
        assert metrics.metadata == {}  # Default value

    def test_from_dict_with_partial_optional_fields(self):
        """Test from_dict with some optional fields."""
        data = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.85,
            "f1_score": 0.87,
            "auc_score": 0.92,
            # metadata missing
        }

        metrics = ModelMetrics.from_dict(data)

        assert metrics.accuracy == 0.95
        assert metrics.auc_score == 0.92
        assert metrics.metadata == {}  # Default value

    def test_roundtrip_serialization(self):
        """Test roundtrip to_dict -> from_dict."""
        original = ModelMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            auc_score=0.92,
            metadata={"model_type": "IsolationForest", "version": "1.0"},
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = ModelMetrics.from_dict(data)

        assert restored.accuracy == original.accuracy
        assert restored.precision == original.precision
        assert restored.recall == original.recall
        assert restored.f1_score == original.f1_score
        assert restored.auc_score == original.auc_score
        assert restored.metadata == original.metadata

    def test_metadata_none_initialization(self):
        """Test that None metadata is converted to empty dict."""
        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87, metadata=None
        )

        assert metrics.metadata == {}

    def test_equality(self):
        """Test equality comparison."""
        metrics1 = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        metrics2 = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        metrics3 = ModelMetrics(
            accuracy=0.90,  # Different value
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
        )

        assert metrics1 == metrics2
        assert metrics1 != metrics3

    def test_hash(self):
        """Test that ModelMetrics is NOT hashable due to mutable metadata dict."""
        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        # Should NOT be hashable due to mutable metadata dict
        with pytest.raises(TypeError, match="unhashable type"):
            hash(metrics)

        with pytest.raises(TypeError, match="unhashable type"):
            {metrics}

        with pytest.raises(TypeError, match="unhashable type"):
            {metrics: "test"}

    def test_complex_metadata(self):
        """Test with complex metadata structure."""
        metadata = {
            "model_info": {
                "algorithm": "IsolationForest",
                "parameters": {"n_estimators": 100, "contamination": 0.1},
            },
            "training": {"duration_seconds": 45.2, "samples": 10000},
            "validation": {"cross_validation_folds": 5, "stratified": True},
        }

        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87, metadata=metadata
        )

        assert metrics.metadata == metadata
        assert metrics.metadata["model_info"]["algorithm"] == "IsolationForest"

    def test_repr_and_str(self):
        """Test string representations."""
        metrics = ModelMetrics(
            accuracy=0.95, precision=0.90, recall=0.85, f1_score=0.87
        )

        repr_str = repr(metrics)
        str_str = str(metrics)

        # Both should contain the class name and key values
        assert "ModelMetrics" in repr_str
        assert "0.95" in repr_str
        assert "ModelMetrics" in str_str
        assert "0.95" in str_str

    def test_perfect_scores(self):
        """Test with perfect scores (all 1.0)."""
        metrics = ModelMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            auc_score=1.0,
            metadata={"note": "perfect_model"},
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_score == 1.0

    def test_worst_scores(self):
        """Test with worst scores (all 0.0)."""
        metrics = ModelMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_score=0.0,
            metadata={"note": "worst_model"},
        )

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.auc_score == 0.0

    def test_realistic_scores(self):
        """Test with realistic model scores."""
        metrics = ModelMetrics(
            accuracy=0.847,
            precision=0.823,
            recall=0.901,
            f1_score=0.860,
            auc_score=0.876,
            metadata={"model": "RandomForest", "features": 15, "test_size": 0.2},
        )

        assert 0.8 < metrics.accuracy < 0.9
        assert 0.8 < metrics.precision < 0.9
        assert 0.8 < metrics.f1_score < 0.9
        assert 0.8 < metrics.auc_score < 0.9
