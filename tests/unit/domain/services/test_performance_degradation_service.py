"""Tests for performance degradation domain service."""

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
from pynomaly.domain.services.performance_degradation_service import (
    PerformanceDegradationService,
)
from pynomaly.domain.exceptions import ValidationError


class TestPerformanceDegradationService:
    """Test PerformanceDegradationService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = PerformanceDegradationService()
        self.detector_id = uuid4()
        
        # Create sample metrics
        self.sample_metrics = [
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.ACCURACY,
                value=0.75,
                timestamp=datetime.utcnow() - timedelta(minutes=10),
                sample_count=1000
            ),
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.PRECISION,
                value=0.73,
                timestamp=datetime.utcnow() - timedelta(minutes=5),
                sample_count=1000
            ),
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.RECALL,
                value=0.78,
                timestamp=datetime.utcnow() - timedelta(minutes=1),
                sample_count=1000
            )
        ]
        
        # Create sample baseline
        self.sample_baseline = PerformanceBaseline(
            detector_id=self.detector_id,
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            baseline_metrics={
                PerformanceMetricType.ACCURACY: 0.85,
                PerformanceMetricType.PRECISION: 0.82,
                PerformanceMetricType.RECALL: 0.88
            },
            confidence_intervals={
                PerformanceMetricType.ACCURACY: (0.83, 0.87),
                PerformanceMetricType.PRECISION: (0.80, 0.84),
                PerformanceMetricType.RECALL: (0.86, 0.90)
            },
            sample_count=1000
        )
        
        # Create sample configuration
        self.sample_config = PerformanceMonitoringConfiguration(
            detector_id=self.detector_id,
            enabled=True,
            monitoring_interval=timedelta(hours=1),
            evaluation_window=timedelta(days=7),
            baseline_window=timedelta(days=30),
            performance_thresholds=[
                PerformanceThreshold(
                    metric_type=PerformanceMetricType.ACCURACY,
                    threshold_value=0.8,
                    comparison_operator=">=",
                    severity=DegradationSeverity.MODERATE
                ),
                PerformanceThreshold(
                    metric_type=PerformanceMetricType.PRECISION,
                    threshold_value=0.75,
                    comparison_operator=">=",
                    severity=DegradationSeverity.MAJOR
                )
            ],
            min_samples_required=1,
            confidence_level=0.95
        )
    
    def test_evaluate_performance_degradation_with_violations(self):
        """Test evaluating performance degradation with threshold violations."""
        events = self.service.evaluate_performance_degradation(
            current_metrics=self.sample_metrics,
            baseline=self.sample_baseline,
            configuration=self.sample_config
        )
        
        # Should detect degradation for accuracy (0.75 < 0.8 threshold)
        # and precision (0.73 < 0.75 threshold)
        assert len(events) == 2
        
        # Check accuracy degradation event
        accuracy_event = next(
            (e for e in events if e.trigger_metric.metric_type == PerformanceMetricType.ACCURACY),
            None
        )
        assert accuracy_event is not None
        assert accuracy_event.detector_id == self.detector_id
        assert accuracy_event.severity == DegradationSeverity.MODERATE
        assert accuracy_event.status == DegradationStatus.DEGRADED
        assert accuracy_event.baseline_value == 0.85
        assert accuracy_event.current_value == 0.75
        assert accuracy_event.degradation_percentage is not None
        
        # Check precision degradation event
        precision_event = next(
            (e for e in events if e.trigger_metric.metric_type == PerformanceMetricType.PRECISION),
            None
        )
        assert precision_event is not None
        assert precision_event.detector_id == self.detector_id
        assert precision_event.severity == DegradationSeverity.MAJOR
        assert precision_event.status == DegradationStatus.DEGRADED
        assert precision_event.baseline_value == 0.82
        assert precision_event.current_value == 0.73
    
    def test_evaluate_performance_degradation_no_violations(self):
        """Test evaluating performance degradation with no threshold violations."""
        # Create metrics that don't violate thresholds
        good_metrics = [
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.ACCURACY,
                value=0.85,  # Above threshold
                timestamp=datetime.utcnow() - timedelta(minutes=10),
                sample_count=1000
            ),
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.PRECISION,
                value=0.80,  # Above threshold
                timestamp=datetime.utcnow() - timedelta(minutes=5),
                sample_count=1000
            )
        ]
        
        events = self.service.evaluate_performance_degradation(
            current_metrics=good_metrics,
            baseline=self.sample_baseline,
            configuration=self.sample_config
        )
        
        assert len(events) == 0
    
    def test_evaluate_performance_degradation_validation_errors(self):
        """Test validation errors in performance degradation evaluation."""
        # Test with empty metrics
        with pytest.raises(ValidationError, match="Current metrics are required"):
            self.service.evaluate_performance_degradation(
                current_metrics=[],
                baseline=self.sample_baseline,
                configuration=self.sample_config
            )
        
        # Test with invalid baseline
        invalid_baseline = PerformanceBaseline(
            detector_id=self.detector_id,
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            sample_count=1000
        )
        invalid_baseline.invalidate("Test invalidation")
        
        with pytest.raises(ValidationError, match="Baseline is not valid"):
            self.service.evaluate_performance_degradation(
                current_metrics=self.sample_metrics,
                baseline=invalid_baseline,
                configuration=self.sample_config
            )
        
        # Test with disabled configuration
        disabled_config = PerformanceMonitoringConfiguration(
            detector_id=self.detector_id,
            enabled=False
        )
        
        with pytest.raises(ValidationError, match="Monitoring configuration is disabled"):
            self.service.evaluate_performance_degradation(
                current_metrics=self.sample_metrics,
                baseline=self.sample_baseline,
                configuration=disabled_config
            )
        
        # Test with no thresholds
        no_threshold_config = PerformanceMonitoringConfiguration(
            detector_id=self.detector_id,
            enabled=True,
            performance_thresholds=[]
        )
        
        with pytest.raises(ValidationError, match="No performance thresholds configured"):
            self.service.evaluate_performance_degradation(
                current_metrics=self.sample_metrics,
                baseline=self.sample_baseline,
                configuration=no_threshold_config
            )
    
    def test_calculate_baseline_statistics(self):
        """Test calculating baseline statistics from metrics."""
        # Create historical metrics
        historical_metrics = []
        for i in range(50):  # Enough samples for reliable statistics
            historical_metrics.extend([
                PerformanceMetric(
                    detector_id=self.detector_id,
                    metric_type=PerformanceMetricType.ACCURACY,
                    value=0.85 + (i % 10) * 0.01,  # Values between 0.85 and 0.94
                    timestamp=datetime.utcnow() - timedelta(days=i),
                    sample_count=1000
                ),
                PerformanceMetric(
                    detector_id=self.detector_id,
                    metric_type=PerformanceMetricType.PRECISION,
                    value=0.80 + (i % 10) * 0.01,  # Values between 0.80 and 0.89
                    timestamp=datetime.utcnow() - timedelta(days=i),
                    sample_count=1000
                )
            ])
        
        statistics = self.service.calculate_baseline_statistics(
            metrics=historical_metrics,
            confidence_level=0.95
        )
        
        assert len(statistics) == 2
        assert PerformanceMetricType.ACCURACY in statistics
        assert PerformanceMetricType.PRECISION in statistics
        
        # Check accuracy statistics
        accuracy_stats = statistics[PerformanceMetricType.ACCURACY]
        assert 'mean' in accuracy_stats
        assert 'std_dev' in accuracy_stats
        assert 'median' in accuracy_stats
        assert 'min' in accuracy_stats
        assert 'max' in accuracy_stats
        assert 'confidence_interval_lower' in accuracy_stats
        assert 'confidence_interval_upper' in accuracy_stats
        assert 'sample_count' in accuracy_stats
        assert accuracy_stats['sample_count'] == 50
        
        # Check that mean is reasonable
        assert 0.85 <= accuracy_stats['mean'] <= 0.95
        assert accuracy_stats['min'] >= 0.85
        assert accuracy_stats['max'] <= 0.94
    
    def test_calculate_baseline_statistics_insufficient_samples(self):
        """Test calculating baseline statistics with insufficient samples."""
        # Create metrics with insufficient samples
        few_metrics = [
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.ACCURACY,
                value=0.85,
                timestamp=datetime.utcnow() - timedelta(days=1),
                sample_count=1000
            )
        ]
        
        statistics = self.service.calculate_baseline_statistics(
            metrics=few_metrics,
            confidence_level=0.95
        )
        
        # Should return empty statistics due to insufficient samples
        assert len(statistics) == 0
    
    def test_analyze_performance_trend(self):
        """Test analyzing performance trend."""
        # Create metrics with declining trend
        declining_metrics = []
        for i in range(10):
            declining_metrics.append(
                PerformanceMetric(
                    detector_id=self.detector_id,
                    metric_type=PerformanceMetricType.ACCURACY,
                    value=0.90 - (9 - i) * 0.02,  # Declining from 0.90 to 0.72
                    timestamp=datetime.utcnow() - timedelta(days=i),
                    sample_count=1000
                )
            )
        
        trend_analysis = self.service.analyze_performance_trend(
            metrics=declining_metrics,
            metric_type=PerformanceMetricType.ACCURACY,
            window=timedelta(days=15)
        )
        
        assert trend_analysis['trend'] == 'declining'
        assert trend_analysis['slope'] < 0  # Negative slope for declining trend
        assert trend_analysis['sample_count'] == 10
        assert trend_analysis['recent_value'] == 0.72  # Most recent value
        assert trend_analysis['initial_value'] == 0.90  # Oldest value in sorted order
        assert abs(trend_analysis['total_change'] - (-0.18)) < 0.001  # 0.72 - 0.90
        assert abs(trend_analysis['percentage_change'] - (-20.0)) < 0.001  # (0.72 - 0.90) / 0.90 * 100
    
    def test_analyze_performance_trend_insufficient_data(self):
        """Test analyzing performance trend with insufficient data."""
        single_metric = [
            PerformanceMetric(
                detector_id=self.detector_id,
                metric_type=PerformanceMetricType.ACCURACY,
                value=0.85,
                timestamp=datetime.utcnow() - timedelta(days=1),
                sample_count=1000
            )
        ]
        
        trend_analysis = self.service.analyze_performance_trend(
            metrics=single_metric,
            metric_type=PerformanceMetricType.ACCURACY
        )
        
        assert trend_analysis['trend'] == 'insufficient_data'
        assert trend_analysis['slope'] == 0.0
        assert trend_analysis['r_squared'] == 0.0
    
    def test_determine_degradation_severity(self):
        """Test determining degradation severity."""
        thresholds = [
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value=0.9,
                comparison_operator=">=",
                severity=DegradationSeverity.MINOR
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value=0.8,
                comparison_operator=">=",
                severity=DegradationSeverity.MODERATE
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value=0.7,
                comparison_operator=">=",
                severity=DegradationSeverity.MAJOR
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ACCURACY,
                threshold_value=0.6,
                comparison_operator=">=",
                severity=DegradationSeverity.CRITICAL
            )
        ]
        
        # Test different severity levels
        severity = self.service.determine_degradation_severity(
            current_value=0.85,
            baseline_value=0.90,
            metric_type=PerformanceMetricType.ACCURACY,
            thresholds=thresholds
        )
        assert severity == DegradationSeverity.MINOR  # 0.85 < 0.9 but >= 0.8
        
        severity = self.service.determine_degradation_severity(
            current_value=0.75,
            baseline_value=0.90,
            metric_type=PerformanceMetricType.ACCURACY,
            thresholds=thresholds
        )
        assert severity == DegradationSeverity.MODERATE  # 0.75 < 0.8 but >= 0.7
        
        severity = self.service.determine_degradation_severity(
            current_value=0.65,
            baseline_value=0.90,
            metric_type=PerformanceMetricType.ACCURACY,
            thresholds=thresholds
        )
        assert severity == DegradationSeverity.MAJOR  # 0.65 < 0.7 but >= 0.6
        
        severity = self.service.determine_degradation_severity(
            current_value=0.55,
            baseline_value=0.90,
            metric_type=PerformanceMetricType.ACCURACY,
            thresholds=thresholds
        )
        assert severity == DegradationSeverity.CRITICAL  # 0.55 < 0.6
        
        severity = self.service.determine_degradation_severity(
            current_value=0.95,
            baseline_value=0.90,
            metric_type=PerformanceMetricType.ACCURACY,
            thresholds=thresholds
        )
        assert severity == DegradationSeverity.NONE  # 0.95 >= 0.9 (no violation)
    
    def test_create_monitoring_configuration(self):
        """Test creating monitoring configuration."""
        metric_thresholds = {
            PerformanceMetricType.ACCURACY: [
                (0.8, ">=", DegradationSeverity.MODERATE),
                (0.7, ">=", DegradationSeverity.MAJOR)
            ],
            PerformanceMetricType.PRECISION: [
                (0.75, ">=", DegradationSeverity.MODERATE)
            ]
        }
        
        config = self.service.create_monitoring_configuration(
            detector_id=str(self.detector_id),
            metric_thresholds=metric_thresholds,
            monitoring_interval=timedelta(hours=2),
            evaluation_window=timedelta(days=14),
            baseline_window=timedelta(days=60)
        )
        
        assert config.detector_id == str(self.detector_id)
        assert config.monitoring_interval == timedelta(hours=2)
        assert config.evaluation_window == timedelta(days=14)
        assert config.baseline_window == timedelta(days=60)
        assert config.enabled is True
        assert config.alert_on_degradation is True
        assert config.auto_trigger_retraining is True
        
        # Check thresholds
        assert len(config.performance_thresholds) == 3
        
        accuracy_thresholds = [
            t for t in config.performance_thresholds 
            if t.metric_type == PerformanceMetricType.ACCURACY
        ]
        assert len(accuracy_thresholds) == 2
        
        precision_thresholds = [
            t for t in config.performance_thresholds 
            if t.metric_type == PerformanceMetricType.PRECISION
        ]
        assert len(precision_thresholds) == 1
    
    def test_group_metrics_by_type(self):
        """Test grouping metrics by type."""
        grouped = self.service._group_metrics_by_type(self.sample_metrics)
        
        assert len(grouped) == 3
        assert PerformanceMetricType.ACCURACY in grouped
        assert PerformanceMetricType.PRECISION in grouped
        assert PerformanceMetricType.RECALL in grouped
        
        assert len(grouped[PerformanceMetricType.ACCURACY]) == 1
        assert len(grouped[PerformanceMetricType.PRECISION]) == 1
        assert len(grouped[PerformanceMetricType.RECALL]) == 1
    
    def test_is_threshold_violated(self):
        """Test checking if threshold is violated."""
        threshold = PerformanceThreshold(
            metric_type=PerformanceMetricType.ACCURACY,
            threshold_value=0.8,
            comparison_operator=">=",
            severity=DegradationSeverity.MODERATE
        )
        
        # Value below threshold should be violation
        assert self.service._is_threshold_violated(0.7, threshold) is True
        
        # Value meeting threshold should not be violation
        assert self.service._is_threshold_violated(0.8, threshold) is False
        
        # Value above threshold should not be violation
        assert self.service._is_threshold_violated(0.9, threshold) is False
    
    def test_is_statistically_significant(self):
        """Test checking statistical significance."""
        # Test with confidence interval
        result = self.service._is_statistically_significant(
            current_value=0.75,  # Below confidence interval
            baseline_value=0.85,
            baseline=self.sample_baseline,
            metric_type=PerformanceMetricType.ACCURACY
        )
        assert result is True  # 0.75 is outside (0.83, 0.87)
        
        result = self.service._is_statistically_significant(
            current_value=0.85,  # Within confidence interval
            baseline_value=0.85,
            baseline=self.sample_baseline,
            metric_type=PerformanceMetricType.ACCURACY
        )
        assert result is False  # 0.85 is within (0.83, 0.87)
        
        # Test without confidence interval
        baseline_no_interval = PerformanceBaseline(
            detector_id=self.detector_id,
            baseline_start=datetime.utcnow() - timedelta(days=30),
            baseline_end=datetime.utcnow() - timedelta(days=1),
            baseline_metrics={PerformanceMetricType.ACCURACY: 0.85},
            sample_count=1000
        )
        
        result = self.service._is_statistically_significant(
            current_value=0.75,  # 10 percentage points below baseline
            baseline_value=0.85,
            baseline=baseline_no_interval,
            metric_type=PerformanceMetricType.ACCURACY
        )
        assert result is True  # Should be significant due to large difference
    
    def test_calculate_confidence_interval(self):
        """Test calculating confidence interval."""
        values = [0.8, 0.82, 0.85, 0.88, 0.9, 0.87, 0.83, 0.86, 0.84, 0.89]
        
        lower, upper = self.service._calculate_confidence_interval(values, 0.95)
        
        mean_val = sum(values) / len(values)
        assert lower < mean_val < upper
        assert lower >= min(values)
        assert upper <= max(values)
    
    def test_calculate_confidence_interval_single_value(self):
        """Test calculating confidence interval with single value."""
        values = [0.85]
        
        lower, upper = self.service._calculate_confidence_interval(values, 0.95)
        
        assert lower == 0.85
        assert upper == 0.85
    
    def test_percentile(self):
        """Test calculating percentile."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        assert self.service._percentile(values, 0) == 1
        assert self.service._percentile(values, 50) == 5.5
        assert self.service._percentile(values, 100) == 10
        assert self.service._percentile(values, 25) == 3.25
        assert self.service._percentile(values, 75) == 7.75
    
    def test_percentile_empty_list(self):
        """Test calculating percentile with empty list."""
        assert self.service._percentile([], 50) == 0.0
    
    def test_linear_regression(self):
        """Test linear regression calculation."""
        # Test with perfect linear relationship
        x_values = [1, 2, 3, 4, 5]
        y_values = [2, 4, 6, 8, 10]  # y = 2x
        
        slope, intercept, r_squared = self.service._linear_regression(x_values, y_values)
        
        assert abs(slope - 2.0) < 0.001
        assert abs(intercept - 0.0) < 0.001
        assert abs(r_squared - 1.0) < 0.001
    
    def test_linear_regression_no_relationship(self):
        """Test linear regression with no relationship."""
        x_values = [1, 2, 3, 4, 5]
        y_values = [5, 5, 5, 5, 5]  # Constant y
        
        slope, intercept, r_squared = self.service._linear_regression(x_values, y_values)
        
        assert abs(slope - 0.0) < 0.001
        assert abs(intercept - 5.0) < 0.001
        assert abs(r_squared - 0.0) < 0.001
    
    def test_linear_regression_insufficient_data(self):
        """Test linear regression with insufficient data."""
        x_values = [1]
        y_values = [2]
        
        slope, intercept, r_squared = self.service._linear_regression(x_values, y_values)
        
        assert slope == 0.0
        assert intercept == 0.0
        assert r_squared == 0.0
