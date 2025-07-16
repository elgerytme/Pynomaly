"""
Enhanced domain value objects tests.
Tests core value objects with comprehensive validation and edge cases.
"""

from unittest.mock import Mock

import pytest

from monorepo.domain.exceptions import ValidationError
from monorepo.domain.value_objects import (
    AnomalyCategory,
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
    PerformanceMetrics,
    SeverityScore,
)


class TestAnomalyScore:
    """Test suite for AnomalyScore value object."""

    def test_anomaly_score_creation(self):
        """Test basic anomaly score creation."""
        score = AnomalyScore(0.75)

        assert score.value == 0.75
        assert score.confidence_lower is None
        assert score.confidence_upper is None
        assert not score.is_confident

    def test_anomaly_score_with_confidence(self):
        """Test anomaly score with confidence interval."""
        score = AnomalyScore(0.75, confidence_lower=0.7, confidence_upper=0.8)

        assert score.value == 0.75
        assert score.confidence_lower == 0.7
        assert score.confidence_upper == 0.8
        assert score.is_confident

    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        # Valid scores
        AnomalyScore(0.0)
        AnomalyScore(1.0)
        AnomalyScore(0.5)

        # Invalid scores
        with pytest.raises(ValidationError):
            AnomalyScore(-0.1)

        with pytest.raises(ValidationError):
            AnomalyScore(1.1)

        with pytest.raises(ValidationError):
            AnomalyScore(float("nan"))

        with pytest.raises(ValidationError):
            AnomalyScore(float("inf"))

    def test_anomaly_score_confidence_validation(self):
        """Test confidence interval validation."""
        # Valid confidence intervals
        AnomalyScore(0.5, confidence_lower=0.4, confidence_upper=0.6)

        # Invalid confidence intervals
        with pytest.raises(ValidationError):
            AnomalyScore(
                0.5, confidence_lower=0.6, confidence_upper=0.4
            )  # Lower > upper

        with pytest.raises(ValidationError):
            AnomalyScore(
                0.5, confidence_lower=0.3, confidence_upper=0.4
            )  # Value outside interval

        with pytest.raises(ValidationError):
            AnomalyScore(0.5, confidence_lower=-0.1, confidence_upper=0.6)  # Lower < 0

        with pytest.raises(ValidationError):
            AnomalyScore(0.5, confidence_lower=0.4, confidence_upper=1.1)  # Upper > 1

    def test_anomaly_score_comparison(self):
        """Test anomaly score comparison operations."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.7)

        # Test equality
        assert score2 == score3
        assert score1 != score2

        # Test ordering
        assert score1 < score2
        assert score2 > score1
        assert score2 <= score3
        assert score2 >= score3

    def test_anomaly_score_arithmetic(self):
        """Test anomaly score arithmetic operations."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.4)

        # Test addition
        result = score1 + score2
        assert result.value == 0.7

        # Test subtraction
        result = score2 - score1
        assert result.value == 0.1

        # Test multiplication
        result = score1 * 2
        assert result.value == 0.6

        # Test division
        result = score2 / 2
        assert result.value == 0.2

    def test_anomaly_score_normalization(self):
        """Test anomaly score normalization."""
        score = AnomalyScore(0.75)

        # Test different normalization methods
        normalized_minmax = score.normalize(method="minmax", min_val=0.0, max_val=1.0)
        assert normalized_minmax.value == 0.75

        normalized_zscore = score.normalize(method="zscore", mean=0.5, std=0.2)
        assert abs(normalized_zscore.value - 1.25) < 0.001

    def test_anomaly_score_discretization(self):
        """Test anomaly score discretization."""
        score = AnomalyScore(0.75)

        # Test threshold-based discretization
        binary = score.discretize(threshold=0.5)
        assert binary == 1

        binary_low = score.discretize(threshold=0.8)
        assert binary_low == 0

        # Test multi-level discretization
        levels = score.discretize(thresholds=[0.3, 0.7, 0.9])
        assert levels == 2  # Falls in second threshold range

    def test_anomaly_score_serialization(self):
        """Test anomaly score serialization."""
        score = AnomalyScore(0.75, confidence_lower=0.7, confidence_upper=0.8)

        # Test to dict
        score_dict = score.to_dict()
        assert score_dict["value"] == 0.75
        assert score_dict["confidence_lower"] == 0.7
        assert score_dict["confidence_upper"] == 0.8
        assert score_dict["is_confident"] is True

        # Test from dict
        reconstructed = AnomalyScore.from_dict(score_dict)
        assert reconstructed == score


class TestConfidenceInterval:
    """Test suite for ConfidenceInterval value object."""

    def test_confidence_interval_creation(self):
        """Test basic confidence interval creation."""
        ci = ConfidenceInterval(lower=0.3, upper=0.7, confidence_level=0.95)

        assert ci.lower == 0.3
        assert ci.upper == 0.7
        assert ci.confidence_level == 0.95
        assert ci.width == 0.4

    def test_confidence_interval_validation(self):
        """Test confidence interval validation."""
        # Valid intervals
        ConfidenceInterval(0.2, 0.8, 0.95)
        ConfidenceInterval(0.0, 1.0, 0.99)

        # Invalid intervals
        with pytest.raises(ValidationError):
            ConfidenceInterval(0.8, 0.2, 0.95)  # Lower > upper

        with pytest.raises(ValidationError):
            ConfidenceInterval(0.2, 0.8, 1.5)  # Invalid confidence level

        with pytest.raises(ValidationError):
            ConfidenceInterval(0.2, 0.8, -0.1)  # Invalid confidence level

    def test_confidence_interval_contains(self):
        """Test confidence interval contains method."""
        ci = ConfidenceInterval(0.3, 0.7, 0.95)

        assert ci.contains(0.5)
        assert ci.contains(0.3)
        assert ci.contains(0.7)
        assert not ci.contains(0.2)
        assert not ci.contains(0.8)

    def test_confidence_interval_overlap(self):
        """Test confidence interval overlap detection."""
        ci1 = ConfidenceInterval(0.2, 0.6, 0.95)
        ci2 = ConfidenceInterval(0.4, 0.8, 0.95)
        ci3 = ConfidenceInterval(0.7, 0.9, 0.95)

        assert ci1.overlaps(ci2)
        assert ci2.overlaps(ci1)
        assert not ci1.overlaps(ci3)
        assert not ci3.overlaps(ci1)

    def test_confidence_interval_intersection(self):
        """Test confidence interval intersection."""
        ci1 = ConfidenceInterval(0.2, 0.6, 0.95)
        ci2 = ConfidenceInterval(0.4, 0.8, 0.95)

        intersection = ci1.intersection(ci2)
        assert intersection.lower == 0.4
        assert intersection.upper == 0.6

        # Non-overlapping intervals
        ci3 = ConfidenceInterval(0.7, 0.9, 0.95)
        assert ci1.intersection(ci3) is None

    def test_confidence_interval_union(self):
        """Test confidence interval union."""
        ci1 = ConfidenceInterval(0.2, 0.6, 0.95)
        ci2 = ConfidenceInterval(0.4, 0.8, 0.95)

        union = ci1.union(ci2)
        assert union.lower == 0.2
        assert union.upper == 0.8

    def test_confidence_interval_adjust(self):
        """Test confidence interval adjustment."""
        ci = ConfidenceInterval(0.3, 0.7, 0.95)

        # Test expansion
        expanded = ci.adjust(factor=1.2)
        assert expanded.width > ci.width

        # Test contraction
        contracted = ci.adjust(factor=0.8)
        assert contracted.width < ci.width


class TestContaminationRate:
    """Test suite for ContaminationRate value object."""

    def test_contamination_rate_creation(self):
        """Test basic contamination rate creation."""
        rate = ContaminationRate(0.1)

        assert rate.value == 0.1
        assert not rate.is_auto

    def test_contamination_rate_auto(self):
        """Test auto contamination rate."""
        rate = ContaminationRate.auto()

        assert rate.is_auto
        assert rate.value is None

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        # Valid rates
        ContaminationRate(0.01)
        ContaminationRate(0.5)
        ContaminationRate(0.49)

        # Invalid rates
        with pytest.raises(ValidationError):
            ContaminationRate(0.0)

        with pytest.raises(ValidationError):
            ContaminationRate(0.5)

        with pytest.raises(ValidationError):
            ContaminationRate(-0.1)

        with pytest.raises(ValidationError):
            ContaminationRate(1.0)

    def test_contamination_rate_estimation(self):
        """Test contamination rate estimation."""
        # Mock data
        mock_data = Mock()
        mock_data.shape = (1000, 5)

        rate = ContaminationRate.estimate_from_data(mock_data, method="iqr")

        assert isinstance(rate, ContaminationRate)
        assert 0.0 < rate.value < 0.5

    def test_contamination_rate_adaptation(self):
        """Test contamination rate adaptation."""
        rate = ContaminationRate(0.1)

        # Test adaptation based on performance
        performance_metrics = {"precision": 0.8, "recall": 0.6, "f1_score": 0.69}

        adapted = rate.adapt(
            performance_metrics, target_metric="f1_score", target_value=0.8
        )

        assert isinstance(adapted, ContaminationRate)
        assert adapted.value != rate.value

    def test_contamination_rate_comparison(self):
        """Test contamination rate comparison."""
        rate1 = ContaminationRate(0.1)
        rate2 = ContaminationRate(0.2)
        rate3 = ContaminationRate(0.1)

        assert rate1 == rate3
        assert rate1 != rate2
        assert rate1 < rate2
        assert rate2 > rate1

    def test_contamination_rate_serialization(self):
        """Test contamination rate serialization."""
        rate = ContaminationRate(0.15)

        # Test to dict
        rate_dict = rate.to_dict()
        assert rate_dict["value"] == 0.15
        assert rate_dict["is_auto"] is False

        # Test from dict
        reconstructed = ContaminationRate.from_dict(rate_dict)
        assert reconstructed == rate


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics value object."""

    def test_performance_metrics_creation(self):
        """Test basic performance metrics creation."""
        metrics = PerformanceMetrics(
            precision=0.85, recall=0.78, f1_score=0.814, accuracy=0.92, roc_auc=0.88
        )

        assert metrics.precision == 0.85
        assert metrics.recall == 0.78
        assert metrics.f1_score == 0.814
        assert metrics.accuracy == 0.92
        assert metrics.roc_auc == 0.88

    def test_performance_metrics_validation(self):
        """Test performance metrics validation."""
        # Valid metrics
        PerformanceMetrics(precision=0.85, recall=0.78, f1_score=0.814, accuracy=0.92)

        # Invalid metrics
        with pytest.raises(ValidationError):
            PerformanceMetrics(precision=-0.1, recall=0.5, f1_score=0.5, accuracy=0.5)

        with pytest.raises(ValidationError):
            PerformanceMetrics(precision=0.5, recall=1.1, f1_score=0.5, accuracy=0.5)

    def test_performance_metrics_comparison(self):
        """Test performance metrics comparison."""
        metrics1 = PerformanceMetrics(
            precision=0.8, recall=0.7, f1_score=0.75, accuracy=0.85
        )
        metrics2 = PerformanceMetrics(
            precision=0.85, recall=0.75, f1_score=0.8, accuracy=0.88
        )

        assert metrics2.is_better_than(metrics1, primary_metric="f1_score")
        assert metrics2.is_better_than(metrics1, primary_metric="accuracy")
        assert not metrics1.is_better_than(metrics2, primary_metric="f1_score")

    def test_performance_metrics_aggregation(self):
        """Test performance metrics aggregation."""
        metrics_list = [
            PerformanceMetrics(precision=0.8, recall=0.7, f1_score=0.75, accuracy=0.85),
            PerformanceMetrics(
                precision=0.85, recall=0.75, f1_score=0.8, accuracy=0.88
            ),
            PerformanceMetrics(
                precision=0.82, recall=0.72, f1_score=0.77, accuracy=0.86
            ),
        ]

        averaged = PerformanceMetrics.average(metrics_list)

        assert abs(averaged.precision - 0.823) < 0.001
        assert abs(averaged.recall - 0.723) < 0.001
        assert abs(averaged.f1_score - 0.773) < 0.001
        assert abs(averaged.accuracy - 0.863) < 0.001

    def test_performance_metrics_confidence_intervals(self):
        """Test performance metrics with confidence intervals."""
        metrics = PerformanceMetrics(
            precision=0.85,
            recall=0.78,
            f1_score=0.814,
            accuracy=0.92,
            precision_ci=ConfidenceInterval(0.82, 0.88, 0.95),
            recall_ci=ConfidenceInterval(0.75, 0.81, 0.95),
            f1_score_ci=ConfidenceInterval(0.80, 0.83, 0.95),
        )

        assert metrics.has_confidence_intervals
        assert metrics.precision_ci.contains(0.85)
        assert metrics.recall_ci.contains(0.78)
        assert metrics.f1_score_ci.contains(0.814)

    def test_performance_metrics_bootstrap(self):
        """Test performance metrics bootstrap statistics."""
        metrics = PerformanceMetrics(
            precision=0.85, recall=0.78, f1_score=0.814, accuracy=0.92
        )

        # Add bootstrap statistics
        metrics.add_bootstrap_stats(
            precision_bootstrap=[0.83, 0.84, 0.85, 0.86, 0.87],
            recall_bootstrap=[0.76, 0.77, 0.78, 0.79, 0.80],
            f1_score_bootstrap=[0.80, 0.81, 0.814, 0.82, 0.83],
        )

        assert metrics.has_bootstrap_stats
        assert "precision_mean" in metrics.bootstrap_stats
        assert "precision_std" in metrics.bootstrap_stats
        assert "recall_mean" in metrics.bootstrap_stats
        assert "recall_std" in metrics.bootstrap_stats


class TestAnomalyCategory:
    """Test suite for AnomalyCategory value object."""

    def test_anomaly_category_creation(self):
        """Test basic anomaly category creation."""
        category = AnomalyCategory("outlier", description="Statistical outlier")

        assert category.name == "outlier"
        assert category.description == "Statistical outlier"
        assert category.severity is None

    def test_anomaly_category_with_severity(self):
        """Test anomaly category with severity."""
        category = AnomalyCategory(
            "critical_outlier",
            description="Critical statistical outlier",
            severity=SeverityScore(0.9),
        )

        assert category.name == "critical_outlier"
        assert category.severity.value == 0.9

    def test_anomaly_category_predefined(self):
        """Test predefined anomaly categories."""
        outlier = AnomalyCategory.outlier()
        novelty = AnomalyCategory.novelty()
        drift = AnomalyCategory.drift()

        assert outlier.name == "outlier"
        assert novelty.name == "novelty"
        assert drift.name == "drift"

    def test_anomaly_category_validation(self):
        """Test anomaly category validation."""
        # Valid categories
        AnomalyCategory("valid_name", "Valid description")

        # Invalid categories
        with pytest.raises(ValidationError):
            AnomalyCategory("", "Empty name")

        with pytest.raises(ValidationError):
            AnomalyCategory("valid_name", "")  # Empty description

    def test_anomaly_category_hierarchy(self):
        """Test anomaly category hierarchy."""
        parent = AnomalyCategory("outlier", "Statistical outlier")
        child = AnomalyCategory(
            "extreme_outlier", "Extreme statistical outlier", parent=parent
        )

        assert child.parent == parent
        assert child.is_child_of(parent)
        assert parent.is_parent_of(child)

    def test_anomaly_category_matching(self):
        """Test anomaly category matching."""
        category = AnomalyCategory("outlier", "Statistical outlier")

        assert category.matches("outlier")
        assert category.matches("OUTLIER")  # Case insensitive
        assert not category.matches("novelty")


class TestSeverityScore:
    """Test suite for SeverityScore value object."""

    def test_severity_score_creation(self):
        """Test basic severity score creation."""
        score = SeverityScore(0.75)

        assert score.value == 0.75
        assert score.level == "high"

    def test_severity_score_levels(self):
        """Test severity score level classification."""
        low = SeverityScore(0.2)
        medium = SeverityScore(0.5)
        high = SeverityScore(0.8)
        critical = SeverityScore(0.95)

        assert low.level == "low"
        assert medium.level == "medium"
        assert high.level == "high"
        assert critical.level == "critical"

    def test_severity_score_validation(self):
        """Test severity score validation."""
        # Valid scores
        SeverityScore(0.0)
        SeverityScore(1.0)
        SeverityScore(0.5)

        # Invalid scores
        with pytest.raises(ValidationError):
            SeverityScore(-0.1)

        with pytest.raises(ValidationError):
            SeverityScore(1.1)

    def test_severity_score_comparison(self):
        """Test severity score comparison."""
        low = SeverityScore(0.2)
        medium = SeverityScore(0.5)
        high = SeverityScore(0.8)

        assert low < medium < high
        assert high > medium > low
        assert low != high

    def test_severity_score_escalation(self):
        """Test severity score escalation."""
        score = SeverityScore(0.6)

        escalated = score.escalate(factor=1.5)
        assert escalated.value > score.value
        assert escalated.level == "critical"

        de_escalated = score.escalate(factor=0.5)
        assert de_escalated.value < score.value
        assert de_escalated.level == "medium"

    def test_severity_score_threshold_adjustment(self):
        """Test severity score threshold adjustment."""
        score = SeverityScore(0.65)

        # Test with custom thresholds
        score_custom = SeverityScore(
            0.65, thresholds={"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 0.95}
        )

        assert score_custom.level == "high"  # With custom threshold


class TestValueObjectIntegration:
    """Integration tests for value objects."""

    def test_score_with_confidence_and_category(self):
        """Test anomaly score with confidence interval and category."""
        score = AnomalyScore(0.85, confidence_lower=0.8, confidence_upper=0.9)
        category = AnomalyCategory("outlier", "Statistical outlier")
        severity = SeverityScore(0.85)

        # Test integration
        assert score.is_confident
        assert score.confidence_lower < score.value < score.confidence_upper
        assert category.name == "outlier"
        assert severity.level == "high"

    def test_performance_metrics_with_confidence(self):
        """Test performance metrics with confidence intervals."""
        metrics = PerformanceMetrics(
            precision=0.85,
            recall=0.78,
            f1_score=0.814,
            accuracy=0.92,
            precision_ci=ConfidenceInterval(0.82, 0.88, 0.95),
            recall_ci=ConfidenceInterval(0.75, 0.81, 0.95),
        )

        assert metrics.has_confidence_intervals
        assert metrics.precision_ci.contains(metrics.precision)
        assert metrics.recall_ci.contains(metrics.recall)

    def test_contamination_rate_with_performance(self):
        """Test contamination rate adaptation with performance metrics."""
        rate = ContaminationRate(0.1)
        metrics = PerformanceMetrics(
            precision=0.85, recall=0.78, f1_score=0.814, accuracy=0.92
        )

        # Test adaptation
        adapted_rate = rate.adapt(
            metrics.to_dict(), target_metric="f1_score", target_value=0.85
        )

        assert isinstance(adapted_rate, ContaminationRate)
        assert adapted_rate.value != rate.value

    def test_value_object_serialization_roundtrip(self):
        """Test complete serialization roundtrip for all value objects."""
        # Create complex objects
        score = AnomalyScore(0.85, confidence_lower=0.8, confidence_upper=0.9)
        ci = ConfidenceInterval(0.8, 0.9, 0.95)
        rate = ContaminationRate(0.1)
        metrics = PerformanceMetrics(
            precision=0.85, recall=0.78, f1_score=0.814, accuracy=0.92
        )
        category = AnomalyCategory("outlier", "Statistical outlier")
        severity = SeverityScore(0.85)

        # Test serialization roundtrip
        objects = [score, ci, rate, metrics, category, severity]

        for obj in objects:
            obj_dict = obj.to_dict()
            reconstructed = obj.__class__.from_dict(obj_dict)
            assert reconstructed == obj
