"""Comprehensive tests for PerformanceMetrics value object."""

import pytest

from monorepo.domain.value_objects.performance_metrics import PerformanceMetrics


class TestPerformanceMetricsInitialization:
    """Test performance metrics initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.5,
            inference_time=15.2,
            model_size=1024000,
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.80
        assert metrics.recall == 0.75
        assert metrics.f1_score == 0.77
        assert metrics.training_time == 120.5
        assert metrics.inference_time == 15.2
        assert metrics.model_size == 1024000

        # Check defaults
        assert metrics.roc_auc is None
        assert metrics.pr_auc is None
        assert metrics.memory_usage is None
        assert metrics.cpu_usage is None
        assert metrics.throughput_rps is None
        assert metrics.true_positives is None
        assert metrics.true_negatives is None
        assert metrics.false_positives is None
        assert metrics.false_negatives is None

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters."""
        metrics = PerformanceMetrics(
            accuracy=0.90,
            precision=0.88,
            recall=0.85,
            f1_score=0.86,
            training_time=180.0,
            inference_time=10.5,
            model_size=2048000,
            roc_auc=0.92,
            pr_auc=0.89,
            memory_usage=512.0,
            cpu_usage=75.5,
            throughput_rps=1000.0,
            true_positives=850,
            true_negatives=900,
            false_positives=120,
            false_negatives=30,
        )

        assert metrics.accuracy == 0.90
        assert metrics.precision == 0.88
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.86
        assert metrics.roc_auc == 0.92
        assert metrics.pr_auc == 0.89
        assert metrics.memory_usage == 512.0
        assert metrics.cpu_usage == 75.5
        assert metrics.throughput_rps == 1000.0
        assert metrics.true_positives == 850
        assert metrics.true_negatives == 900
        assert metrics.false_positives == 120
        assert metrics.false_negatives == 30

    def test_validation_accuracy_out_of_range(self):
        """Test validation of accuracy outside valid range."""
        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=1.5,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
            )

        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=-0.1,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
            )

    def test_validation_precision_out_of_range(self):
        """Test validation of precision outside valid range."""
        with pytest.raises(ValueError, match="precision must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=1.5,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
            )

    def test_validation_recall_out_of_range(self):
        """Test validation of recall outside valid range."""
        with pytest.raises(ValueError, match="recall must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=1.2,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
            )

    def test_validation_f1_score_out_of_range(self):
        """Test validation of f1_score outside valid range."""
        with pytest.raises(ValueError, match="f1_score must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=-0.1,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
            )

    def test_validation_roc_auc_out_of_range(self):
        """Test validation of roc_auc outside valid range."""
        with pytest.raises(ValueError, match="ROC AUC must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                roc_auc=1.1,
            )

    def test_validation_pr_auc_out_of_range(self):
        """Test validation of pr_auc outside valid range."""
        with pytest.raises(ValueError, match="PR AUC must be between 0.0 and 1.0"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                pr_auc=-0.1,
            )

    def test_validation_negative_training_time(self):
        """Test validation of negative training time."""
        with pytest.raises(ValueError, match="Training time must be non-negative"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=-100.0,
                inference_time=10.0,
                model_size=1024,
            )

    def test_validation_negative_inference_time(self):
        """Test validation of negative inference time."""
        with pytest.raises(ValueError, match="Inference time must be non-negative"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=-10.0,
                model_size=1024,
            )

    def test_validation_negative_model_size(self):
        """Test validation of negative model size."""
        with pytest.raises(ValueError, match="Model size must be non-negative integer"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=-1024,
            )

    def test_validation_non_integer_model_size(self):
        """Test validation of non-integer model size."""
        with pytest.raises(ValueError, match="Model size must be non-negative integer"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024.5,
            )

    def test_validation_negative_memory_usage(self):
        """Test validation of negative memory usage."""
        with pytest.raises(ValueError, match="Memory usage must be non-negative"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                memory_usage=-512.0,
            )

    def test_validation_cpu_usage_out_of_range(self):
        """Test validation of CPU usage outside valid range."""
        with pytest.raises(ValueError, match="CPU usage must be between 0 and 100"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                cpu_usage=150.0,
            )

    def test_validation_negative_throughput(self):
        """Test validation of negative throughput."""
        with pytest.raises(ValueError, match="Throughput must be non-negative"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                throughput_rps=-100.0,
            )

    def test_validation_negative_confusion_matrix_values(self):
        """Test validation of negative confusion matrix values."""
        with pytest.raises(
            ValueError, match="Confusion matrix values must be non-negative integers"
        ):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                true_positives=-10,
            )

    def test_validation_non_integer_confusion_matrix_values(self):
        """Test validation of non-integer confusion matrix values."""
        with pytest.raises(
            ValueError, match="Confusion matrix values must be non-negative integers"
        ):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                true_positives=10.5,
            )

    def test_validation_invalid_types(self):
        """Test validation of invalid types."""
        with pytest.raises(TypeError, match="accuracy must be numeric"):
            PerformanceMetrics(
                accuracy="0.8",
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
            )

        with pytest.raises(TypeError, match="ROC AUC must be numeric"):
            PerformanceMetrics(
                accuracy=0.8,
                precision=0.8,
                recall=0.7,
                f1_score=0.75,
                training_time=100.0,
                inference_time=10.0,
                model_size=1024,
                roc_auc="0.9",
            )


class TestPerformanceMetricsFromConfusionMatrix:
    """Test creation from confusion matrix."""

    def test_from_confusion_matrix_basic(self):
        """Test creation from confusion matrix with basic parameters."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        # Check confusion matrix values
        assert metrics.true_positives == 80
        assert metrics.true_negatives == 90
        assert metrics.false_positives == 10
        assert metrics.false_negatives == 20

        # Check calculated metrics
        total = 80 + 90 + 10 + 20  # 200
        expected_accuracy = (80 + 90) / 200  # 0.85
        expected_precision = 80 / (80 + 10)  # 0.888...
        expected_recall = 80 / (80 + 20)  # 0.8
        expected_f1 = (
            2
            * (expected_precision * expected_recall)
            / (expected_precision + expected_recall)
        )

        assert metrics.accuracy == expected_accuracy
        assert abs(metrics.precision - expected_precision) < 0.001
        assert metrics.recall == expected_recall
        assert abs(metrics.f1_score - expected_f1) < 0.001

    def test_from_confusion_matrix_with_additional_kwargs(self):
        """Test creation from confusion matrix with additional parameters."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            roc_auc=0.92,
            pr_auc=0.89,
            memory_usage=512.0,
        )

        assert metrics.roc_auc == 0.92
        assert metrics.pr_auc == 0.89
        assert metrics.memory_usage == 512.0

    def test_from_confusion_matrix_zero_precision_denominator(self):
        """Test creation when precision denominator is zero."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=0,
            true_negatives=90,
            false_positives=0,
            false_negatives=10,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_from_confusion_matrix_zero_recall_denominator(self):
        """Test creation when recall denominator is zero."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=0,
            true_negatives=90,
            false_positives=10,
            false_negatives=0,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_from_confusion_matrix_zero_total(self):
        """Test creation with zero total predictions."""
        with pytest.raises(ValueError, match="Total predictions cannot be zero"):
            PerformanceMetrics.from_confusion_matrix(
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                training_time=120.0,
                inference_time=15.0,
                model_size=1024000,
            )

    def test_from_confusion_matrix_perfect_performance(self):
        """Test creation with perfect performance (no false predictions)."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=50,
            true_negatives=50,
            false_positives=0,
            false_negatives=0,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0


class TestPerformanceMetricsCreateMinimal:
    """Test creation of minimal performance metrics."""

    def test_create_minimal_basic(self):
        """Test creation of minimal metrics."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.85, training_time=120.0, inference_time=15.0, model_size=1024000
        )

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.85  # Estimated from accuracy
        assert metrics.recall == 0.85  # Estimated from accuracy
        assert metrics.f1_score == 0.85  # Estimated from accuracy
        assert metrics.training_time == 120.0
        assert metrics.inference_time == 15.0
        assert metrics.model_size == 1024000

    def test_create_minimal_boundary_values(self):
        """Test creation of minimal metrics with boundary values."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=0.0, training_time=0.0, inference_time=0.0, model_size=0
        )

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_create_minimal_perfect_accuracy(self):
        """Test creation of minimal metrics with perfect accuracy."""
        metrics = PerformanceMetrics.create_minimal(
            accuracy=1.0, training_time=60.0, inference_time=5.0, model_size=512000
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0


class TestPerformanceMetricsProperties:
    """Test performance metrics properties."""

    def test_has_confusion_matrix_true(self):
        """Test has_confusion_matrix property when all values are present."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )

        assert metrics.has_confusion_matrix is True

    def test_has_confusion_matrix_false(self):
        """Test has_confusion_matrix property when values are missing."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            # false_negatives is missing
        )

        assert metrics.has_confusion_matrix is False

    def test_total_predictions_with_confusion_matrix(self):
        """Test total_predictions property when confusion matrix is available."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )

        assert metrics.total_predictions == 200

    def test_total_predictions_without_confusion_matrix(self):
        """Test total_predictions property when confusion matrix is not available."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.total_predictions is None

    def test_specificity_calculation(self):
        """Test specificity calculation."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )

        expected_specificity = 90 / (90 + 10)  # 0.9
        assert metrics.specificity == expected_specificity

    def test_specificity_with_missing_values(self):
        """Test specificity when values are missing."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            false_negatives=20,
        )

        assert metrics.specificity is None

    def test_false_positive_rate_calculation(self):
        """Test false positive rate calculation."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )

        expected_fpr = 1.0 - (90 / (90 + 10))  # 1.0 - 0.9 = 0.1
        assert metrics.false_positive_rate == expected_fpr

    def test_false_positive_rate_with_missing_values(self):
        """Test false positive rate when values are missing."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.false_positive_rate is None

    def test_balanced_accuracy_calculation(self):
        """Test balanced accuracy calculation."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )

        specificity = 90 / (90 + 10)  # 0.9
        expected_balanced_accuracy = (0.75 + specificity) / 2.0  # (0.75 + 0.9) / 2
        assert metrics.balanced_accuracy == expected_balanced_accuracy

    def test_balanced_accuracy_with_missing_values(self):
        """Test balanced accuracy when values are missing."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.balanced_accuracy is None

    def test_model_size_mb_conversion(self):
        """Test model size conversion to MB."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024 * 1024 * 2,  # 2 MB in bytes
        )

        assert metrics.model_size_mb == 2.0

    def test_training_time_minutes_conversion(self):
        """Test training time conversion to minutes."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,  # 2 minutes
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.training_time_minutes == 2.0

    def test_performance_score_calculation(self):
        """Test performance score calculation."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=5.0,  # Fast inference
            model_size=1024000,  # Small model
        )

        # Detection score: (0.80 + 0.75 + 0.77) / 3 = 0.7733...
        detection_score = (0.80 + 0.75 + 0.77) / 3.0
        # Efficiency bonus: (10 - 5) / 10 * 0.1 = 0.05
        efficiency_bonus = (10 - 5) / 10 * 0.1
        # Size penalty: 0 (model is small)
        size_penalty = 0

        expected_score = detection_score + efficiency_bonus - size_penalty
        assert abs(metrics.performance_score - expected_score) < 0.001

    def test_performance_score_with_large_model(self):
        """Test performance score with large model (penalty)."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=200 * 1024 * 1024,  # 200 MB
        )

        # Should have size penalty
        detection_score = (0.80 + 0.75 + 0.77) / 3.0
        efficiency_bonus = 0  # No bonus for 15ms
        size_penalty = (200 - 100) / 1000 * 0.1  # 0.01

        expected_score = detection_score + efficiency_bonus - size_penalty
        assert abs(metrics.performance_score - expected_score) < 0.001

    def test_performance_score_with_slow_inference(self):
        """Test performance score with slow inference (no bonus)."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=20.0,  # Slow inference
            model_size=1024000,
        )

        # Should have no efficiency bonus
        detection_score = (0.80 + 0.75 + 0.77) / 3.0
        efficiency_bonus = 0
        size_penalty = 0

        expected_score = detection_score + efficiency_bonus - size_penalty
        assert abs(metrics.performance_score - expected_score) < 0.001


class TestPerformanceMetricsComparison:
    """Test performance metrics comparison methods."""

    def test_compare_with_basic(self):
        """Test basic comparison between two metrics."""
        metrics1 = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            training_time=100.0,
            inference_time=20.0,
            model_size=2048000,
        )

        comparison = metrics1.compare_with(metrics2)

        assert comparison["accuracy"] == 0.05  # 0.85 - 0.80
        assert comparison["precision"] == 0.05  # 0.80 - 0.75
        assert comparison["recall"] == 0.05  # 0.75 - 0.70
        assert comparison["f1_score"] == 0.05  # 0.77 - 0.72
        assert comparison["training_time"] == 20.0  # 100.0 - 120.0 (negative is better)
        assert comparison["inference_time"] == 5.0  # 20.0 - 15.0 (negative is better)
        assert (
            comparison["model_size"] == 1024000
        )  # 2048000 - 1024000 (negative is better)

    def test_compare_with_optional_metrics(self):
        """Test comparison with optional metrics."""
        metrics1 = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            roc_auc=0.90,
            pr_auc=0.85,
            throughput_rps=500.0,
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            training_time=100.0,
            inference_time=20.0,
            model_size=2048000,
            roc_auc=0.85,
            pr_auc=0.80,
            throughput_rps=400.0,
        )

        comparison = metrics1.compare_with(metrics2)

        assert comparison["roc_auc"] == 0.05  # 0.90 - 0.85
        assert comparison["pr_auc"] == 0.05  # 0.85 - 0.80
        assert comparison["throughput_rps"] == 100.0  # 500.0 - 400.0

    def test_compare_with_missing_optional_metrics(self):
        """Test comparison when optional metrics are missing."""
        metrics1 = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            roc_auc=0.90,
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            training_time=100.0,
            inference_time=20.0,
            model_size=2048000,
            # roc_auc is missing
        )

        comparison = metrics1.compare_with(metrics2)

        # Should not include roc_auc in comparison
        assert "roc_auc" not in comparison

    def test_compare_with_invalid_type(self):
        """Test comparison with invalid type."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        with pytest.raises(
            TypeError, match="Can only compare with another PerformanceMetrics"
        ):
            metrics.compare_with("not a metrics object")

    def test_is_better_than_default_metric(self):
        """Test is_better_than with default metric (performance_score)."""
        metrics1 = PerformanceMetrics(
            accuracy=0.90,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            training_time=120.0,
            inference_time=5.0,  # Fast
            model_size=1024000,  # Small
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            training_time=100.0,
            inference_time=20.0,  # Slow
            model_size=2048000,  # Large
        )

        assert metrics1.is_better_than(metrics2) is True
        assert metrics2.is_better_than(metrics1) is False

    def test_is_better_than_specific_metric(self):
        """Test is_better_than with specific metric."""
        metrics1 = PerformanceMetrics(
            accuracy=0.90,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.80,
            precision=0.95,  # Better precision
            recall=0.70,
            f1_score=0.72,
            training_time=100.0,
            inference_time=20.0,
            model_size=2048000,
        )

        assert metrics1.is_better_than(metrics2, "accuracy") is True
        assert metrics1.is_better_than(metrics2, "precision") is False
        assert metrics2.is_better_than(metrics1, "precision") is True

    def test_is_better_than_nonexistent_metric(self):
        """Test is_better_than with non-existent metric."""
        metrics1 = PerformanceMetrics(
            accuracy=0.90,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.80,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            training_time=100.0,
            inference_time=20.0,
            model_size=2048000,
        )

        # Should return False for non-existent metric
        assert metrics1.is_better_than(metrics2, "non_existent_metric") is False


class TestPerformanceMetricsConversion:
    """Test performance metrics conversion methods."""

    def test_to_dict_basic(self):
        """Test to_dict conversion with basic metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        result = metrics.to_dict()

        assert result["accuracy"] == 0.85
        assert result["precision"] == 0.80
        assert result["recall"] == 0.75
        assert result["f1_score"] == 0.77
        assert result["training_time"] == 120.0
        assert result["training_time_minutes"] == 2.0
        assert result["inference_time"] == 15.0
        assert result["model_size"] == 1024000
        assert result["model_size_mb"] == 1.0
        assert result["roc_auc"] is None
        assert result["pr_auc"] is None
        assert result["memory_usage"] is None
        assert result["cpu_usage"] is None
        assert result["throughput_rps"] is None
        assert result["true_positives"] is None
        assert result["true_negatives"] is None
        assert result["false_positives"] is None
        assert result["false_negatives"] is None
        assert result["total_predictions"] is None
        assert result["specificity"] is None
        assert result["false_positive_rate"] is None
        assert result["balanced_accuracy"] is None
        assert "performance_score" in result

    def test_to_dict_with_all_metrics(self):
        """Test to_dict conversion with all metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
            roc_auc=0.90,
            pr_auc=0.85,
            memory_usage=512.0,
            cpu_usage=75.0,
            throughput_rps=500.0,
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )

        result = metrics.to_dict()

        assert result["roc_auc"] == 0.90
        assert result["pr_auc"] == 0.85
        assert result["memory_usage"] == 512.0
        assert result["cpu_usage"] == 75.0
        assert result["throughput_rps"] == 500.0
        assert result["true_positives"] == 80
        assert result["true_negatives"] == 90
        assert result["false_positives"] == 10
        assert result["false_negatives"] == 20
        assert result["total_predictions"] == 200
        assert result["specificity"] == 0.9
        assert result["false_positive_rate"] == 0.1
        assert result["balanced_accuracy"] == 0.825  # (0.75 + 0.9) / 2

    def test_str_representation(self):
        """Test string representation."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        str_repr = str(metrics)

        assert "Performance(" in str_repr
        assert "accuracy=0.850" in str_repr
        assert "precision=0.800" in str_repr
        assert "recall=0.750" in str_repr
        assert "f1=0.770" in str_repr


class TestPerformanceMetricsImmutability:
    """Test performance metrics immutability."""

    def test_immutable_attributes(self):
        """Test that attributes cannot be modified."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        with pytest.raises(AttributeError):
            metrics.accuracy = 0.90

        with pytest.raises(AttributeError):
            metrics.precision = 0.85

        with pytest.raises(AttributeError):
            metrics.training_time = 100.0

    def test_frozen_dataclass(self):
        """Test that the dataclass is frozen."""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        # This should raise an error due to frozen=True
        with pytest.raises(AttributeError):
            metrics.__dict__["accuracy"] = 0.90


class TestPerformanceMetricsEdgeCases:
    """Test performance metrics edge cases."""

    def test_perfect_performance(self):
        """Test perfect performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            training_time=0.0,
            inference_time=0.0,
            model_size=0,
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.model_size_mb == 0.0
        assert metrics.training_time_minutes == 0.0

    def test_zero_performance(self):
        """Test zero performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            training_time=0.0,
            inference_time=0.0,
            model_size=0,
        )

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_large_model_size(self):
        """Test with very large model size."""
        large_size = 1024 * 1024 * 1024  # 1 GB
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=120.0,
            inference_time=15.0,
            model_size=large_size,
        )

        assert metrics.model_size_mb == 1024.0  # 1 GB in MB

    def test_very_slow_training(self):
        """Test with very slow training time."""
        slow_time = 86400.0  # 24 hours in seconds
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            training_time=slow_time,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.training_time_minutes == 1440.0  # 24 hours in minutes

    def test_confusion_matrix_with_zeros(self):
        """Test confusion matrix with some zero values."""
        metrics = PerformanceMetrics.from_confusion_matrix(
            true_positives=100,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            training_time=120.0,
            inference_time=15.0,
            model_size=1024000,
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.specificity == 0.0  # 0 / (0 + 0) = 0
        assert metrics.false_positive_rate == 1.0  # 1 - 0 = 1

    def test_integer_vs_float_values(self):
        """Test with integer values for float fields."""
        metrics = PerformanceMetrics(
            accuracy=1,  # Integer instead of float
            precision=1,  # Integer instead of float
            recall=1,  # Integer instead of float
            f1_score=1,  # Integer instead of float
            training_time=120,  # Integer instead of float
            inference_time=15,  # Integer instead of float
            model_size=1024000,
        )

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.training_time == 120.0
        assert metrics.inference_time == 15.0
