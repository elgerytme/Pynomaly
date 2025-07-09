"""Tests for performance metrics value object."""

import pytest

from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics value object."""

    def test_basic_creation(self):
        """Test basic creation of performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            training_time=120.5,
            inference_time=5.2,
            model_size=1024000,
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.90
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.87
        assert metrics.training_time == 120.5
        assert metrics.inference_time == 5.2
        assert metrics.model_size == 1024000

    def test_immutability(self):
        """Test that performance metrics are immutable."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            training_time=120.5,
            inference_time=5.2,
            model_size=1024000,
        )
        
        # Should not be able to modify values
        with pytest.raises(AttributeError):
            metrics.accuracy = 0.96

    def test_validation_score_ranges(self):
        """Test validation of score ranges (0.0 to 1.0)."""
        # Valid scores
        PerformanceMetrics(
            accuracy=0.0,
            precision=1.0,
            recall=0.5,
            f1_score=0.33,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        
        # Invalid accuracy
        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=1.5,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
            )
        
        # Invalid precision
        with pytest.raises(ValueError, match="precision must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=-0.1,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
            )

    def test_validation_type_errors(self):
        """Test validation of type errors."""
        # Invalid accuracy type
        with pytest.raises(TypeError, match="accuracy must be numeric"):
            PerformanceMetrics(
                accuracy="0.95",
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
            )

    def test_validation_time_metrics(self):
        """Test validation of time metrics."""
        # Valid times
        PerformanceMetrics(
            accuracy=0.95,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            training_time=0.0,
            inference_time=0.0,
            model_size=1000,
        )
        
        # Invalid training time
        with pytest.raises(ValueError, match="Training time must be non-negative"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=-10.0,
                inference_time=10.0,
                model_size=1000,
            )
        
        # Invalid inference time
        with pytest.raises(ValueError, match="Inference time must be non-negative"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=-5.0,
                model_size=1000,
            )

    def test_validation_model_size(self):
        """Test validation of model size."""
        # Valid model size
        PerformanceMetrics(
            accuracy=0.95,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            training_time=100.0,
            inference_time=10.0,
            model_size=0,
        )
        
        # Invalid model size (negative)
        with pytest.raises(ValueError, match="Model size must be non-negative integer"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=-1000,
            )
        
        # Invalid model size (float)
        with pytest.raises(ValueError, match="Model size must be non-negative integer"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000.5,
            )

    def test_validation_optional_auc_scores(self):
        """Test validation of optional AUC scores."""
        # Valid optional AUC scores
        PerformanceMetrics(
            accuracy=0.95,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
            roc_auc=0.92,
            pr_auc=0.88,
        )
        
        # Invalid ROC AUC
        with pytest.raises(ValueError, match="ROC AUC must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
                roc_auc=1.2,
            )
        
        # Invalid PR AUC type
        with pytest.raises(TypeError, match="PR AUC must be numeric"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
                pr_auc="0.88",
            )

    def test_validation_optional_metrics(self):
        """Test validation of optional metrics."""
        # Valid optional metrics
        PerformanceMetrics(
            accuracy=0.95,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
            memory_usage=512.0,
            cpu_usage=75.5,
            throughput_rps=1000.0,
        )
        
        # Invalid memory usage
        with pytest.raises(ValueError, match="Memory usage must be non-negative"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
                memory_usage=-100.0,
            )
        
        # Invalid CPU usage
        with pytest.raises(ValueError, match="CPU usage must be between 0 and 100"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
                cpu_usage=150.0,
            )

    def test_validation_confusion_matrix(self):
        """Test validation of confusion matrix components."""
        # Valid confusion matrix
        PerformanceMetrics(
            accuracy=0.95,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )
        
        # Invalid confusion matrix (negative)
        with pytest.raises(ValueError, match="Confusion matrix values must be non-negative integers"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
                true_positives=-5,
            )
        
        # Invalid confusion matrix (float)
        with pytest.raises(ValueError, match="Confusion matrix values must be non-negative integers"):
            PerformanceMetrics(
                accuracy=0.95,
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
                true_positives=80.5,
            )

    def test_from_confusion_matrix_factory(self):
        """Test creating metrics from confusion matrix."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        
        assert metrics.true_positives == 80
        assert metrics.true_negatives == 90
        assert metrics.false_positives == 10
        assert metrics.false_negatives == 20
        
        # Check calculated metrics
        assert metrics.accuracy == (80 + 90) / (80 + 90 + 10 + 20)  # 0.85
        assert metrics.precision == 80 / (80 + 10)  # 0.888...
        assert metrics.recall == 80 / (80 + 20)  # 0.8
        
        # Check F1 score calculation
        precision = 80 / (80 + 10)
        recall = 80 / (80 + 20)
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert abs(metrics.f1_score - expected_f1) < 1e-10

    def test_from_confusion_matrix_edge_cases(self):
        """Test edge cases for confusion matrix factory."""
        # Zero total predictions
        with pytest.raises(ValueError, match="Total predictions cannot be zero"):
            PerformanceMetrics.from_confusion_matrix(
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                training_time=100.0,
                inference_time=10.0,
                model_size=1000,
            )
        
        # Zero precision denominator
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=0,
            true_negatives=90,
            false_positives=0,
            false_negatives=10,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        assert metrics.precision == 0.0
        
        # Zero recall denominator
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=0,
            true_negatives=90,
            false_positives=10,
            false_negatives=0,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        assert metrics.recall == 0.0

    def test_create_minimal_factory(self):
        """Test creating minimal metrics."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.85
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.85
        assert metrics.training_time == 120.0
        assert metrics.inference_time == 5.0
        assert metrics.model_size == 2048000

    def test_has_confusion_matrix_property(self):
        """Test has_confusion_matrix property."""
        # Without confusion matrix
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,
        )
        assert metrics1.has_confusion_matrix is False
        
        # With complete confusion matrix
        metrics2 = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        assert metrics2.has_confusion_matrix is True
        
        # With partial confusion matrix
        metrics3 = PerformanceMetrics(
            accuracy=0.95,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
            true_positives=80,
            # Missing other confusion matrix components
        )
        assert metrics3.has_confusion_matrix is False

    def test_total_predictions_property(self):
        """Test total_predictions property."""
        # Without confusion matrix
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,
        )
        assert metrics1.total_predictions is None
        
        # With confusion matrix
        metrics2 = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        assert metrics2.total_predictions == 200

    def test_specificity_property(self):
        """Test specificity property."""
        # Without confusion matrix
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,
        )
        assert metrics1.specificity is None
        
        # With confusion matrix
        metrics2 = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        expected_specificity = 90 / (90 + 10)  # 0.9
        assert metrics2.specificity == expected_specificity

    def test_false_positive_rate_property(self):
        """Test false_positive_rate property."""
        # Without confusion matrix
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,
        )
        assert metrics1.false_positive_rate is None
        
        # With confusion matrix
        metrics2 = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        specificity = 90 / (90 + 10)  # 0.9
        expected_fpr = 1.0 - specificity  # 0.1
        assert metrics2.false_positive_rate == expected_fpr

    def test_balanced_accuracy_property(self):
        """Test balanced_accuracy property."""
        # Without confusion matrix
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,
        )
        assert metrics1.balanced_accuracy is None
        
        # With confusion matrix
        metrics2 = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1000,
        )
        recall = 80 / (80 + 20)  # 0.8
        specificity = 90 / (90 + 10)  # 0.9
        expected_balanced = (recall + specificity) / 2.0  # 0.85
        assert metrics2.balanced_accuracy == expected_balanced

    def test_model_size_mb_property(self):
        """Test model_size_mb property."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=2048000,  # 2MB in bytes
        )
        expected_mb = 2048000 / (1024 * 1024)  # ~1.95MB
        assert abs(metrics.model_size_mb - expected_mb) < 1e-6

    def test_training_time_minutes_property(self):
        """Test training_time_minutes property."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,  # 2 minutes
            inference_time=5.0,
            model_size=2048000,
        )
        assert metrics.training_time_minutes == 2.0

    def test_performance_score_property(self):
        """Test performance_score property."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,  # Fast inference
            model_size=10 * 1024 * 1024,  # 10MB - small model
        )
        
        # Should be primarily the detection score
        expected_detection = (0.85 + 0.85 + 0.85) / 3.0  # 0.85
        
        # Efficiency bonus for fast inference (< 10ms)
        expected_efficiency = max(0, (10 - 5.0) / 10) * 0.1  # 0.05
        
        # Size penalty for large models (> 100MB) - none in this case
        expected_size_penalty = 0.0
        
        expected_score = expected_detection + expected_efficiency - expected_size_penalty
        assert abs(metrics.performance_score - expected_score) < 1e-6

    def test_compare_with_method(self):
        """Test compare_with method."""
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        metrics2 = PerformanceMetrics.create_minimal(
            accuracy=0.90,
            training_time=100.0,
            inference_time=8.0,
            model_size=2048000,
        )
        
        comparison = metrics1.compare_with(metrics2)
        
        # metrics1 vs metrics2
        assert comparison["accuracy"] == 0.85 - 0.90  # -0.05
        assert comparison["training_time"] == 100.0 - 120.0  # -20.0 (negative is better)
        assert comparison["inference_time"] == 8.0 - 5.0  # 3.0 (negative is better)
        assert comparison["model_size"] == 2048000 - 1024000  # 1024000 (negative is better)

    def test_compare_with_invalid_type(self):
        """Test compare_with with invalid type."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        with pytest.raises(TypeError, match="Can only compare with another PerformanceMetrics"):
            metrics.compare_with("not_metrics")

    def test_is_better_than_method(self):
        """Test is_better_than method."""
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.90,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        metrics2 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=100.0,
            inference_time=8.0,
            model_size=2048000,
        )
        
        # metrics1 should be better on accuracy
        assert metrics1.is_better_than(metrics2, "accuracy") is True
        assert metrics2.is_better_than(metrics1, "accuracy") is False
        
        # metrics2 should be better on training time
        assert metrics2.is_better_than(metrics1, "training_time") is True
        assert metrics1.is_better_than(metrics2, "training_time") is False

    def test_to_dict_method(self):
        """Test to_dict method."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=100.0,
            inference_time=10.0,
            model_size=1024000,
            roc_auc=0.92,
        )
        
        result = metrics.to_dict()
        
        # Check required fields
        assert result["accuracy"] == metrics.accuracy
        assert result["precision"] == metrics.precision
        assert result["recall"] == metrics.recall
        assert result["f1_score"] == metrics.f1_score
        assert result["roc_auc"] == 0.92
        assert result["training_time"] == 100.0
        assert result["inference_time"] == 10.0
        assert result["model_size"] == 1024000
        
        # Check derived fields
        assert result["model_size_mb"] == metrics.model_size_mb
        assert result["training_time_minutes"] == metrics.training_time_minutes
        assert result["total_predictions"] == 200
        assert result["specificity"] == metrics.specificity
        assert result["performance_score"] == metrics.performance_score

    def test_string_representation(self):
        """Test string representation."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        result = str(metrics)
        assert "Performance" in result
        assert "accuracy=0.850" in result
        assert "precision=0.850" in result
        assert "recall=0.850" in result
        assert "f1=0.850" in result

    def test_equality_comparison(self):
        """Test equality comparison."""
        metrics1 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        metrics2 = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        metrics3 = PerformanceMetrics.create_minimal(
            accuracy=0.90,
            training_time=120.0,
            inference_time=5.0,
            model_size=1024000,
        )
        
        assert metrics1 == metrics2
        assert metrics1 != metrics3

    def test_comprehensive_metrics_example(self):
        """Test comprehensive metrics with all fields."""
        metrics = PerformanceMetrics(
            accuracy=0.92,
            precision=0.88,
            recall=0.85,
            f1_score=0.865,
            training_time=240.5,
            inference_time=3.2,
            model_size=5242880,  # 5MB
            roc_auc=0.94,
            pr_auc=0.89,
            memory_usage=1024.0,
            cpu_usage=68.5,
            throughput_rps=500.0,
            true_positives=170,
            true_negatives=730,
            false_positives=30,
            false_negatives=70,
        )
        
        # Verify all fields are accessible
        assert metrics.accuracy == 0.92
        assert metrics.roc_auc == 0.94
        assert metrics.memory_usage == 1024.0
        assert metrics.cpu_usage == 68.5
        assert metrics.throughput_rps == 500.0
        assert metrics.has_confusion_matrix is True
        assert metrics.total_predictions == 1000
        
        # Verify calculated properties
        assert metrics.model_size_mb == 5.0
        assert abs(metrics.training_time_minutes - 4.0075) < 0.01
        assert metrics.specificity == 730 / (730 + 30)  # 0.9605...
        assert metrics.balanced_accuracy is not None

    def test_edge_case_zero_values(self):
        """Test edge cases with zero values."""
        # All zeros where allowed
        metrics = PerformanceMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            training_time=0.0,
            inference_time=0.0,
            model_size=0,
            memory_usage=0.0,
            cpu_usage=0.0,
            throughput_rps=0.0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
        )
        
        # Should not raise validation errors
        assert metrics.accuracy == 0.0
        assert metrics.model_size_mb == 0.0
        assert metrics.training_time_minutes == 0.0

    def test_performance_score_edge_cases(self):
        """Test performance score with edge cases."""
        # Very slow inference (> 10ms)
        slow_metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=50.0,  # Very slow
            model_size=1024000,
        )
        
        # Should get no efficiency bonus
        detection_score = 0.85
        efficiency_bonus = 0.0  # No bonus for slow inference
        size_penalty = 0.0  # Small model
        expected_score = detection_score + efficiency_bonus - size_penalty
        assert abs(slow_metrics.performance_score - expected_score) < 1e-6
        
        # Very large model (> 100MB)
        large_metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85,
            training_time=120.0,
            inference_time=5.0,
            model_size=500 * 1024 * 1024,  # 500MB
        )
        
        # Should get size penalty
        detection_score = 0.85
        efficiency_bonus = 0.05  # Fast inference
        size_penalty = max(0, (500 - 100) / 1000) * 0.1  # 0.04
        expected_score = detection_score + efficiency_bonus - size_penalty
        assert abs(large_metrics.performance_score - expected_score) < 1e-6