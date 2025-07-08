"""Tests for performance degradation domain entities."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from pynomaly.domain.entities.performance_degradation import (
    PerformanceMetricType,
    DegradationSeverity,
    DegradationStatus,
    PerformanceThreshold,
    PerformanceMetric,
    PerformanceDegradationEvent,
    PerformanceMonitoringConfiguration,
    PerformanceBaseline,
)


class TestPerformanceThreshold:
    """Test PerformanceThreshold value object."""
    
    def test_create_valid_threshold(self):
        """Test creating a valid threshold."""
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE,
            description="Accuracy must be at least 80%"
        )
        
        assert threshold.metric_type == PerformanceMetricType.ACCURACY
        assert threshold.threshold_value == 0.8
        assert threshold.comparison_operator == ">="
        assert threshold.severity == DegradationSeverity.MODERATE
        assert threshold.description == "Accuracy must be at least 80%"
    
    def test_invalid_comparison_operator(self):
        """Test invalid comparison operator raises error."""
        with pytest.raises(ValueError, match="Invalid comparison operator"):
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value=0.8,
                comparison_operator="invalid",
                severity=DegradationSeverity.MODERATE
            )
    
    def test_invalid_threshold_value_type(self):
        """Test non-numeric threshold value raises error."""
        with pytest.raises(ValueError, match="Threshold value must be numeric"):
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value="invalid",
                comparison_operator=">=",
                severity=DegradationSeverity.MODERATE
            )
    
    def test_invalid_range_for_percentage_metric(self):
        """Test invalid range for percentage-based metrics."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value=1.5,
                comparison_operator=">=",
                severity=DegradationSeverity.MODERATE
            )
    
    def test_evaluate_greater_than_equal(self):
        """Test threshold evaluation with >= operator."""
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        assert threshold.evaluate(0.9) is True
        assert threshold.evaluate(0.8) is True
        assert threshold.evaluate(0.7) is False
    
    def test_evaluate_less_than(self):
        """Test threshold evaluation with < operator."""
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.FPR,
            threshold_value=0.1,
            comparison_operator="<",
            severity=DegradationSeverity.MAJOR
        )
        
        assert threshold.evaluate(0.05) is True
        assert threshold.evaluate(0.1) is False
        assert threshold.evaluate(0.15) is False
    
    def test_evaluate_equals(self):
        """Test threshold evaluation with == operator."""
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.CUSTOM,
            threshold_value=100.0,
            comparison_operator="==",
            severity=DegradationSeverity.MINOR
        )
        
        assert threshold.evaluate(100.0) is True
        assert threshold.evaluate(99.9) is False
        assert threshold.evaluate(100.1) is False


class TestPerformanceMetric:
    """Test PerformanceMetric entity."""
    
    def test_create_valid_metric(self):
        """Test creating a valid performance metric."""
        detector_id = uuid4()
        timestamp = datetime.utcnow()
        
        metric = PerformanceMetric(
            detector_id=detector_id,
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.85,
            timestamp=timestamp,
            dataset_id="test_dataset",
            sample_count=1000,
            confidence_interval=(0.82, 0.88),
            true_positives=425,
            false_positives=75,
            true_negatives=425,
            false_negatives=75
        )
        
        assert metric.detector_id == detector_id
        assert metric.metric_type == PerformanceMetricType.ACCURACY
        assert metric.value == 0.85
        assert metric.timestamp == timestamp
        assert metric.dataset_id == "test_dataset"
        assert metric.sample_count == 1000
        assert metric.confidence_interval == (0.82, 0.88)
        assert metric.true_positives == 425
        assert metric.false_positives == 75
        assert metric.true_negatives == 425
        assert metric.false_negatives == 75
    
    def test_missing_detector_id(self):
        """Test metric creation fails without detector ID."""
        with pytest.raises(ValueError, match="Detector ID is required"):
            PerformanceMetric(
                metric_type=PerformanceMetricType.ACCURACY,
                value=0.85
            )
    
    def test_missing_metric_type(self):
        """Test metric creation fails without metric type."""
        with pytest.raises(ValueError, match="Metric type is required"):
            PerformanceMetric(
                detector_id=uuid4(),
                value=0.85
            )
    
    def test_invalid_metric_value_type(self):
        """Test metric creation fails with non-numeric value."""
        with pytest.raises(ValueError, match="Metric value must be numeric"):
            PerformanceMetric(
                detector_id=uuid4(),
                metric_type=PerformanceMetricType.ACCURACY,
                value="invalid"
            )
    
    def test_invalid_accuracy_range(self):
        """Test accuracy metric with invalid range."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            PerformanceMetric(
                detector_id=uuid4(),
                metric_type=PerformanceMetricType.ACCURACY,
                value=1.5
            )
    
    def test_is_recent(self):
        """Test checking if metric is recent."""
        recent_metric = PerformanceMetric(
            detector_id=uuid4(),
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.85,
            timestamp=datetime.utcnow() - timedelta(minutes=30)
        )
        
        old_metric = PerformanceMetric(
            detector_id=uuid4(),
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.85,
            timestamp=datetime.utcnow() - timedelta(days=2)
        )
        
        assert recent_metric.is_recent(timedelta(hours=1)) is True
        assert old_metric.is_recent(timedelta(hours=1)) is False
    
    def test_get_confusion_matrix_metrics(self):
        """Test getting confusion matrix metrics."""
        metric = PerformanceMetric(
            detector_id=uuid4(),
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.85,
            true_positives=425,
            false_positives=75,
            true_negatives=425,
            false_negatives=75
        )
        
        cm_metrics = metric.get_confusion_matrix_metrics()
        assert cm_metrics is not None
        assert cm_metrics["true_positives"] == 425
        assert cm_metrics["false_positives"] == 75
        assert cm_metrics["true_negatives"] == 425
        assert cm_metrics["false_negatives"] == 75
    
    def test_get_confusion_matrix_metrics_incomplete(self):
        """Test getting confusion matrix metrics with incomplete data."""
        metric = PerformanceMetric(
            detector_id=uuid4(),
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.85,
            true_positives=425,
            false_positives=75
            # Missing true_negatives and false_negatives
        )
        
        cm_metrics = metric.get_confusion_matrix_metrics()
        assert cm_metrics is None


class TestPerformanceDegradationEvent:
    """Test PerformanceDegradationEvent entity."""
    
    def test_create_valid_event(self):
        """Test creating a valid degradation event."""
        detector_id = uuid4()
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        trigger_metric = PerformanceMetric(
            detector_id=detector_id,
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.75
        )
        
        event = PerformanceDegradationEvent(
            detector_id=detector_id,
            severity=DegradationSeverity.MODERATE,
            status=DegradationStatus.DEGRADED,
            violated_threshold=threshold,
            trigger_metric=trigger_metric,
            baseline_value=0.85,
            current_value=0.75
        )
        
        assert event.detector_id == detector_id
        assert event.severity == DegradationSeverity.MODERATE
        assert event.status == DegradationStatus.DEGRADED
        assert event.violated_threshold == threshold
        assert event.trigger_metric == trigger_metric
        assert event.baseline_value == 0.85
        assert event.current_value == 0.75
        assert event.degradation_percentage is not None
        assert abs(event.degradation_percentage - (-11.76)) < 0.01  # (0.75 - 0.85) / 0.85 * 100
    
    def test_missing_detector_id(self):
        """Test event creation fails without detector ID."""
        with pytest.raises(ValueError, match="Detector ID is required"):
            PerformanceDegradationEvent(
                violated_threshold=PerformanceThreshold(
                    metric_type=PerformanceMetricType.ACCURACY,
                    threshold_value=0.8,
                    comparison_operator=">=",
                    severity=DegradationSeverity.MODERATE
                ),
                trigger_metric=PerformanceMetric(
                    detector_id=uuid4(),
                    metric_type=PerformanceMetricType.ACCURACY,
                    value=0.75
                )
            )
    
    def test_missing_threshold(self):
        """Test event creation fails without threshold."""
        with pytest.raises(ValueError, match="Violated threshold is required"):
            PerformanceDegradationEvent(
                detector_id=uuid4(),
                trigger_metric=PerformanceMetric(
                    detector_id=uuid4(),
                    metric_type=PerformanceMetricType.ACCURACY,
                    value=0.75
                )
            )
    
    def test_missing_trigger_metric(self):
        """Test event creation fails without trigger metric."""
        with pytest.raises(ValueError, match="Trigger metric is required"):
            PerformanceDegradationEvent(
                detector_id=uuid4(),
                violated_threshold=PerformanceThreshold(
                    metric_type=PerformanceMetricType.ACCURACY,
                    threshold_value=0.8,
                    comparison_operator=">=",
                    severity=DegradationSeverity.MODERATE
                )
            )
    
    def test_is_resolved(self):
        """Test checking if event is resolved."""
        detector_id = uuid4()
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        trigger_metric = PerformanceMetric(
            detector_id=detector_id,
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.75
        )
        
        event = PerformanceDegradationEvent(
            detector_id=detector_id,
            violated_threshold=threshold,
            trigger_metric=trigger_metric
        )
        
        assert event.is_resolved() is False
        
        event.resolve("Performance improved after retraining")
        assert event.is_resolved() is True
        assert event.resolution_notes == "Performance improved after retraining"
        assert event.status == DegradationStatus.NORMAL
    
    def test_add_action(self):
        """Test adding actions to event."""
        detector_id = uuid4()
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        trigger_metric = PerformanceMetric(
            detector_id=detector_id,
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.75
        )
        
        event = PerformanceDegradationEvent(
            detector_id=detector_id,
            violated_threshold=threshold,
            trigger_metric=trigger_metric
        )
        
        event.add_action("alert_sent")
        event.add_action("retraining_triggered")
        event.add_action("alert_sent")  # Duplicate should not be added
        
        assert len(event.actions_triggered) == 2
        assert "alert_sent" in event.actions_triggered
        assert "retraining_triggered" in event.actions_triggered
    
    def test_get_duration(self):
        """Test getting event duration."""
        detector_id = uuid4()
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        trigger_metric = PerformanceMetric(
            detector_id=detector_id,
            metric_type=PerformanceMetricType.ACCURACY,
            value=0.75
        )
        
        detected_at = datetime.utcnow() - timedelta(hours=2)
        event = PerformanceDegradationEvent(
            detector_id=detector_id,
            violated_threshold=threshold,
            trigger_metric=trigger_metric,
            detected_at=detected_at
        )
        
        duration = event.get_duration()
        assert duration >= timedelta(hours=1, minutes=59)
        assert duration <= timedelta(hours=2, minutes=1)
        
        # Test with resolved event
        event.resolve()
        resolved_duration = event.get_duration()
        assert resolved_duration >= timedelta(hours=1, minutes=59)
        assert resolved_duration <= timedelta(hours=2, minutes=1)


class TestPerformanceMonitoringConfiguration:
    """Test PerformanceMonitoringConfiguration entity."""
    
    def test_create_valid_configuration(self):
        """Test creating a valid monitoring configuration."""
        detector_id = uuid4()
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        config = PerformanceMonitoringConfiguration(
            detector_id=detector_id,
            enabled=True,
            monitoring_interval=timedelta(hours=1),
            evaluation_window=timedelta(days=7),
            baseline_window=timedelta(days=30),
            performance_thresholds=[threshold],
            min_samples_required=100,
            confidence_level=0.95,
            alert_on_degradation=True,
            auto_trigger_retraining=True,
            retraining_threshold_severity=DegradationSeverity.MODERATE
        )
        
        assert config.detector_id == detector_id
        assert config.enabled is True
        assert config.monitoring_interval == timedelta(hours=1)
        assert config.evaluation_window == timedelta(days=7)
        assert config.baseline_window == timedelta(days=30)
        assert len(config.performance_thresholds) == 1
        assert config.min_samples_required == 100
        assert config.confidence_level == 0.95
        assert config.alert_on_degradation is True
        assert config.auto_trigger_retraining is True
        assert config.retraining_threshold_severity == DegradationSeverity.MODERATE
    
    def test_missing_detector_id(self):
        """Test configuration creation fails without detector ID."""
        with pytest.raises(ValueError, match="Detector ID is required"):
            PerformanceMonitoringConfiguration()
    
    def test_invalid_min_samples(self):
        """Test configuration creation fails with invalid min samples."""
        with pytest.raises(ValueError, match="Minimum samples required must be at least 1"):
            PerformanceMonitoringConfiguration(
                detector_id=uuid4(),
                min_samples_required=0
            )
    
    def test_invalid_confidence_level(self):
        """Test configuration creation fails with invalid confidence level."""
        with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
            PerformanceMonitoringConfiguration(
                detector_id=uuid4(),
                confidence_level=1.5
            )
    
    def test_invalid_monitoring_interval(self):
        """Test configuration creation fails with invalid monitoring interval."""
        with pytest.raises(ValueError, match="Monitoring interval must be positive"):
            PerformanceMonitoringConfiguration(
                detector_id=uuid4(),
                monitoring_interval=timedelta(seconds=-1)
            )
    
    def test_add_remove_threshold(self):
        """Test adding and removing thresholds."""
        config = PerformanceMonitoringConfiguration(
            detector_id=uuid4()
        )
        
        threshold1 = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        threshold2 = PerformanceThreshold(
            metric_type=PerformanceMetricType.PRECISION,
            threshold_value=0.7,
            comparison_operator=">=",
            severity=DegradationSeverity.MAJOR
        )
        
        # Add thresholds
        config.add_threshold(threshold1)
        config.add_threshold(threshold2)
        assert len(config.performance_thresholds) == 2
        
        # Adding same threshold again should not duplicate
        config.add_threshold(threshold1)
        assert len(config.performance_thresholds) == 2
        
        # Remove threshold
        removed = config.remove_threshold(threshold1)
        assert removed is True
        assert len(config.performance_thresholds) == 1
        assert threshold1 not in config.performance_thresholds
        
        # Remove non-existent threshold
        removed = config.remove_threshold(threshold1)
        assert removed is False
    
    def test_get_thresholds_by_severity(self):
        """Test getting thresholds by severity."""
        config = PerformanceMonitoringConfiguration(
            detector_id=uuid4()
        )
        
        moderate_threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        major_threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.PRECISION,
            threshold_value=0.7,
            comparison_operator=">=",
            severity=DegradationSeverity.MAJOR
        )
        
        config.add_threshold(moderate_threshold)
        config.add_threshold(major_threshold)
        
        moderate_thresholds = config.get_thresholds_by_severity(DegradationSeverity.MODERATE)
        major_thresholds = config.get_thresholds_by_severity(DegradationSeverity.MAJOR)
        minor_thresholds = config.get_thresholds_by_severity(DegradationSeverity.MINOR)
        
        assert len(moderate_thresholds) == 1
        assert moderate_thresholds[0] == moderate_threshold
        assert len(major_thresholds) == 1
        assert major_thresholds[0] == major_threshold
        assert len(minor_thresholds) == 0
    
    def test_should_trigger_retraining(self):
        """Test retraining trigger logic."""
        config = PerformanceMonitoringConfiguration(
            detector_id=uuid4(),
            auto_trigger_retraining=True,
            retraining_threshold_severity=DegradationSeverity.MODERATE
        )
        
        # Should trigger for moderate and higher severity
        assert config.should_trigger_retraining(DegradationSeverity.MODERATE) is True
        assert config.should_trigger_retraining(DegradationSeverity.MAJOR) is True
        assert config.should_trigger_retraining(DegradationSeverity.CRITICAL) is True
        
        # Should not trigger for minor severity
        assert config.should_trigger_retraining(DegradationSeverity.MINOR) is False
        assert config.should_trigger_retraining(DegradationSeverity.NONE) is False
        
        # Should not trigger if auto retraining is disabled
        config.auto_trigger_retraining = False
        assert config.should_trigger_retraining(DegradationSeverity.MAJOR) is False


class TestPerformanceBaseline:
    """Test PerformanceBaseline entity."""
    
    def test_create_valid_baseline(self):
        """Test creating a valid baseline."""
        detector_id = uuid4()
        baseline_start = datetime.utcnow() - timedelta(days=30)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        
        baseline = PerformanceBaseline(
            detector_id=detector_id,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            baseline_metrics={
                PerformanceMetricType.ACCURACY: 0.85,
                PerformanceMetricType.PRECISION: 0.82,
                PerformanceMetricType.RECALL: 0.88
            },
            sample_count=1000,
            confidence_intervals={
                PerformanceMetricType.ACCURACY: (0.83, 0.87),
                PerformanceMetricType.PRECISION: (0.80, 0.84),
                PerformanceMetricType.RECALL: (0.86, 0.90)
            }
        )
        
        assert baseline.detector_id == detector_id
        assert baseline.baseline_start == baseline_start
        assert baseline.baseline_end == baseline_end
        assert baseline.sample_count == 1000
        assert baseline.is_valid is True
        assert len(baseline.baseline_metrics) == 3
        assert len(baseline.confidence_intervals) == 3
    
    def test_missing_detector_id(self):
        """Test baseline creation fails without detector ID."""
        with pytest.raises(ValueError, match="Detector ID is required"):
            PerformanceBaseline(
                baseline_start=datetime.utcnow() - timedelta(days=30),
                baseline_end=datetime.utcnow() - timedelta(days=1),
                sample_count=1000
            )
    
    def test_missing_baseline_times(self):
        """Test baseline creation fails without baseline times."""
        with pytest.raises(ValueError, match="Baseline start and end times are required"):
            PerformanceBaseline(
                detector_id=uuid4(),
                sample_count=1000
            )
    
    def test_invalid_baseline_times(self):
        """Test baseline creation fails with invalid time range."""
        with pytest.raises(ValueError, match="Baseline start must be before baseline end"):
            PerformanceBaseline(
                detector_id=uuid4(),
                baseline_start=datetime.utcnow(),
                baseline_end=datetime.utcnow() - timedelta(days=1),
                sample_count=1000
            )
    
    def test_invalid_sample_count(self):
        """Test baseline creation fails with invalid sample count."""
        with pytest.raises(ValueError, match="Sample count must be at least 1"):
            PerformanceBaseline(
                detector_id=uuid4(),
                baseline_start=datetime.utcnow() - timedelta(days=30),
                baseline_end=datetime.utcnow() - timedelta(days=1),
                sample_count=0
            )
    
    def test_get_baseline_value(self):
        """Test getting baseline value for metric type."""
        baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            baseline_metrics={
                PerformanceMetricType.ACCURACY: 0.85,
                PerformanceMetricType.PRECISION: 0.82
            },
            sample_count=1000
        )
        
        assert baseline.get_baseline_value(PerformanceMetricType.ACCURACY) == 0.85
        assert baseline.get_baseline_value(PerformanceMetricType.PRECISION) == 0.82
        assert baseline.get_baseline_value(PerformanceMetricType.RECALL) is None
    
    def test_get_confidence_interval(self):
        """Test getting confidence interval for metric type."""
        baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            confidence_intervals={
                PerformanceMetricType.ACCURACY: (0.83, 0.87),
                PerformanceMetricType.PRECISION: (0.80, 0.84)
            },
            sample_count=1000
        )
        
        assert baseline.get_confidence_interval(PerformanceMetricType.ACCURACY) == (0.83, 0.87)
        assert baseline.get_confidence_interval(PerformanceMetricType.PRECISION) == (0.80, 0.84)
        assert baseline.get_confidence_interval(PerformanceMetricType.RECALL) is None
    
    def test_is_within_confidence_interval(self):
        """Test checking if value is within confidence interval."""
        baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            confidence_intervals={
                PerformanceMetricType.ACCURACY: (0.83, 0.87)
            },
            sample_count=1000
        )
        
        assert baseline.is_within_confidence_interval(PerformanceMetricType.ACCURACY, 0.85) is True
        assert baseline.is_within_confidence_interval(PerformanceMetricType.ACCURACY, 0.83) is True
        assert baseline.is_within_confidence_interval(PerformanceMetricType.ACCURACY, 0.87) is True
        assert baseline.is_within_confidence_interval(PerformanceMetricType.ACCURACY, 0.82) is False
        assert baseline.is_within_confidence_interval(PerformanceMetricType.ACCURACY, 0.88) is False
        
        # No interval available, should return True
        assert baseline.is_within_confidence_interval(PerformanceMetricType.PRECISION, 0.75) is True
    
    def test_get_duration(self):
        """Test getting baseline duration."""
        baseline_start = datetime.utcnow() - timedelta(days=30)
        baseline_end = datetime.utcnow() - timedelta(days=1)
        
        baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            sample_count=1000
        )
        
        duration = baseline.get_duration()
        assert duration == timedelta(days=29)
    
    def test_is_recent(self):
        """Test checking if baseline is recent."""
        recent_baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            sample_count=1000
        )
        
        old_baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=datetime.utcnow() - timedelta(days=60),
            baseline_end=datetime.utcnow() - timedelta(days=30),
            sample_count=1000
        )
        
        assert recent_baseline.is_recent(timedelta(days=7)) is True
        assert old_baseline.is_recent(timedelta(days=7)) is False
    
    def test_invalidate(self):
        """Test invalidating baseline."""
        baseline = PerformanceBaseline(
            detector_id=uuid4(),
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            sample_count=1000
        )
        
        assert baseline.is_valid is True
        assert baseline.validation_notes is None
        
        baseline.invalidate("Data quality issues detected")
        assert baseline.is_valid is False
        assert baseline.validation_notes == "Data quality issues detected"
