"""Tests for PerformanceBaselineService."""

import pytest
from datetime import datetime, timedelta
from statistics import mean

from packages.data_science.domain.services.performance_baseline_service import (
    PerformanceBaselineService,
)
from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    DegradationMetricType,
    DegradationSeverity,
)


class TestPerformanceBaselineService:
    """Test suite for PerformanceBaselineService."""
    
    @pytest.fixture
    def baseline_service(self):
        """Create a baseline service instance."""
        return PerformanceBaselineService(confidence_level=0.95, min_samples=5)
    
    @pytest.fixture
    def classification_metrics(self):
        """Create sample classification metrics."""
        base_date = datetime.utcnow() - timedelta(days=30)
        metrics = []
        
        for i in range(10):
            metrics.append(ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.90 + (i * 0.001),  # Slightly increasing
                precision=0.88 + (i * 0.002),
                recall=0.92 - (i * 0.001),
                f1_score=0.90,
                roc_auc=0.95,
                prediction_time_seconds=0.05 + (i * 0.001),
                evaluation_date=base_date + timedelta(days=i),
            ))
        
        return metrics
    
    @pytest.fixture
    def regression_metrics(self):
        """Create sample regression metrics."""
        base_date = datetime.utcnow() - timedelta(days=30)
        metrics = []
        
        for i in range(10):
            metrics.append(ModelPerformanceMetrics(
                task_type=ModelTask.REGRESSION,
                sample_size=1000,
                mse=0.10 - (i * 0.002),  # Decreasing (improving)
                rmse=0.316 - (i * 0.003),
                mae=0.20 - (i * 0.001),
                r2_score=0.80 + (i * 0.005),
                prediction_time_seconds=0.08 + (i * 0.002),
                evaluation_date=base_date + timedelta(days=i),
            ))
        
        return metrics
    
    def test_establish_baseline_recent_average(self, baseline_service, classification_metrics):
        """Test establishing baseline using recent average method."""
        baseline = baseline_service.establish_baseline(
            classification_metrics,
            baseline_method="recent_average",
            lookback_days=10
        )
        
        assert baseline.task_type == ModelTask.BINARY_CLASSIFICATION
        assert baseline.accuracy is not None
        assert baseline.precision is not None
        assert baseline.recall is not None
        assert baseline.f1_score is not None
        assert baseline.roc_auc is not None
        
        # Should be close to the mean of input values
        expected_accuracy = mean([m.accuracy for m in classification_metrics])
        assert abs(baseline.accuracy - expected_accuracy) < 0.001
    
    def test_establish_baseline_best_performance(self, baseline_service, classification_metrics):
        """Test establishing baseline using best performance method."""
        baseline = baseline_service.establish_baseline(
            classification_metrics,
            baseline_method="best_performance"
        )
        
        # Should return the metric with highest primary metric
        best_metric = max(classification_metrics, key=lambda m: m.get_primary_metric())
        assert baseline.accuracy == best_metric.accuracy
        assert baseline.precision == best_metric.precision
    
    def test_establish_baseline_median_performance(self, baseline_service, classification_metrics):
        """Test establishing baseline using median method."""
        baseline = baseline_service.establish_baseline(
            classification_metrics,
            baseline_method="median_performance"
        )
        
        assert baseline.task_type == ModelTask.BINARY_CLASSIFICATION
        assert baseline.accuracy is not None
        
        # Median should be different from mean for this dataset
        mean_accuracy = mean([m.accuracy for m in classification_metrics])
        assert baseline.accuracy != mean_accuracy
    
    def test_establish_baseline_weighted_average(self, baseline_service, classification_metrics):
        """Test establishing baseline using weighted average method."""
        baseline = baseline_service.establish_baseline(
            classification_metrics,
            baseline_method="weighted_average"
        )
        
        assert baseline.task_type == ModelTask.BINARY_CLASSIFICATION
        assert baseline.accuracy is not None
        
        # Weighted average should give more weight to recent metrics
        # So it should be closer to the last values than simple average
        simple_mean = mean([m.accuracy for m in classification_metrics])
        recent_mean = mean([m.accuracy for m in classification_metrics[-3:]])
        
        # Weighted average should be closer to recent mean than simple mean
        diff_to_simple = abs(baseline.accuracy - simple_mean)
        diff_to_recent = abs(baseline.accuracy - recent_mean)
        assert diff_to_recent <= diff_to_simple
    
    def test_establish_baseline_insufficient_data(self, baseline_service):
        """Test baseline establishment with insufficient data."""
        insufficient_metrics = [
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.90,
            )
        ]
        
        with pytest.raises(ValueError, match="Need at least .* samples"):
            baseline_service.establish_baseline(insufficient_metrics)
    
    def test_establish_baseline_empty_data(self, baseline_service):
        """Test baseline establishment with empty data."""
        with pytest.raises(ValueError, match="Historical metrics cannot be empty"):
            baseline_service.establish_baseline([])
    
    def test_establish_baseline_invalid_method(self, baseline_service, classification_metrics):
        """Test baseline establishment with invalid method."""
        with pytest.raises(ValueError, match="Unknown baseline method"):
            baseline_service.establish_baseline(
                classification_metrics,
                baseline_method="invalid_method"
            )
    
    def test_compare_against_baseline(self, baseline_service, classification_metrics):
        """Test comparing current metrics against baseline."""
        baseline = baseline_service.establish_baseline(classification_metrics)
        
        # Create degraded current metrics
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,  # Significantly lower than baseline
            precision=0.75,
            recall=0.85,
            f1_score=0.80,
            roc_auc=0.85,
            prediction_time_seconds=0.10,  # Higher than baseline
        )
        
        thresholds = {
            DegradationMetricType.ACCURACY_DROP: 5.0,  # 5% threshold
            DegradationMetricType.PRECISION_DROP: 5.0,
            DegradationMetricType.PREDICTION_TIME_INCREASE: 20.0,  # 20% threshold
        }
        
        degradations = baseline_service.compare_against_baseline(
            current_metrics, baseline, thresholds
        )
        
        assert len(degradations) > 0
        
        # Check accuracy degradation
        accuracy_degradation = next(
            (d for d in degradations if d["metric_type"] == DegradationMetricType.ACCURACY_DROP),
            None
        )
        assert accuracy_degradation is not None
        assert accuracy_degradation["degradation_percentage"] > 5.0
        assert accuracy_degradation["severity"] in [s.value for s in DegradationSeverity]
    
    def test_compare_against_baseline_incompatible_tasks(self, baseline_service, classification_metrics):
        """Test comparison with incompatible task types."""
        baseline = baseline_service.establish_baseline(classification_metrics)
        
        # Create regression metrics (incompatible)
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.REGRESSION,
            sample_size=1000,
            mse=0.15,
            rmse=0.387,
            mae=0.25,
            r2_score=0.75,
        )
        
        with pytest.raises(ValueError, match="same task type"):
            baseline_service.compare_against_baseline(
                current_metrics, baseline, {}
            )
    
    def test_calculate_baseline_stability(self, baseline_service, classification_metrics):
        """Test baseline stability calculation."""
        stability = baseline_service.calculate_baseline_stability(
            classification_metrics,
            DegradationMetricType.ACCURACY_DROP
        )
        
        assert "stability_score" in stability
        assert "coefficient_of_variation" in stability
        assert "mean_value" in stability
        assert "std_value" in stability
        assert "sample_count" in stability
        
        assert 0 <= stability["stability_score"] <= 1
        assert stability["coefficient_of_variation"] >= 0
        assert stability["sample_count"] == len(classification_metrics)
    
    def test_calculate_baseline_stability_insufficient_data(self, baseline_service):
        """Test stability calculation with insufficient data."""
        single_metric = [ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.90,
        )]
        
        stability = baseline_service.calculate_baseline_stability(
            single_metric,
            DegradationMetricType.ACCURACY_DROP
        )
        
        assert stability["stability_score"] == 0.0
        assert stability["coefficient_of_variation"] == 0.0
    
    def test_suggest_degradation_thresholds(self, baseline_service, classification_metrics):
        """Test threshold suggestion based on historical variability."""
        thresholds = baseline_service.suggest_degradation_thresholds(
            classification_metrics,
            ModelTask.BINARY_CLASSIFICATION
        )
        
        assert isinstance(thresholds, dict)
        assert DegradationMetricType.ACCURACY_DROP in thresholds
        assert DegradationMetricType.PRECISION_DROP in thresholds
        assert DegradationMetricType.PREDICTION_TIME_INCREASE in thresholds
        
        # All thresholds should be positive
        for threshold in thresholds.values():
            assert threshold > 0
            assert threshold <= 25.0  # Maximum suggested threshold
    
    def test_suggest_degradation_thresholds_regression(self, baseline_service, regression_metrics):
        """Test threshold suggestion for regression task."""
        thresholds = baseline_service.suggest_degradation_thresholds(
            regression_metrics,
            ModelTask.REGRESSION
        )
        
        assert DegradationMetricType.MSE_INCREASE in thresholds
        assert DegradationMetricType.RMSE_INCREASE in thresholds
        assert DegradationMetricType.R2_SCORE_DROP in thresholds
        
        # Should not include classification metrics
        assert DegradationMetricType.ACCURACY_DROP not in thresholds
    
    def test_extract_metric_value(self, baseline_service):
        """Test metric value extraction."""
        metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.90,
            precision=0.88,
            prediction_time_seconds=0.05,
        )
        
        # Test classification metrics
        assert baseline_service._extract_metric_value(
            metrics, DegradationMetricType.ACCURACY_DROP
        ) == 0.90
        
        assert baseline_service._extract_metric_value(
            metrics, DegradationMetricType.PRECISION_DROP
        ) == 0.88
        
        assert baseline_service._extract_metric_value(
            metrics, DegradationMetricType.PREDICTION_TIME_INCREASE
        ) == 0.05
        
        # Test non-existent metric
        assert baseline_service._extract_metric_value(
            metrics, DegradationMetricType.MSE_INCREASE
        ) is None
    
    def test_calculate_degradation_percentage(self, baseline_service):
        """Test degradation percentage calculation."""
        # Test accuracy drop (higher is better)
        degradation_pct = baseline_service._calculate_degradation_percentage(
            current=0.80,
            baseline=0.90,
            metric_type=DegradationMetricType.ACCURACY_DROP
        )
        assert abs(degradation_pct - 11.11) < 0.01  # (0.90-0.80)/0.90 * 100
        
        # Test MSE increase (lower is better)
        degradation_pct = baseline_service._calculate_degradation_percentage(
            current=0.15,
            baseline=0.10,
            metric_type=DegradationMetricType.MSE_INCREASE
        )
        assert degradation_pct == 50.0  # (0.15-0.10)/0.10 * 100
        
        # Test no degradation (current better than baseline)
        degradation_pct = baseline_service._calculate_degradation_percentage(
            current=0.95,
            baseline=0.90,
            metric_type=DegradationMetricType.ACCURACY_DROP
        )
        assert degradation_pct == 0.0
        
        # Test zero baseline
        degradation_pct = baseline_service._calculate_degradation_percentage(
            current=0.10,
            baseline=0.0,
            metric_type=DegradationMetricType.ACCURACY_DROP
        )
        assert degradation_pct == 0.0
    
    def test_determine_severity(self, baseline_service):
        """Test severity determination."""
        assert baseline_service._determine_severity(60.0) == DegradationSeverity.CRITICAL
        assert baseline_service._determine_severity(40.0) == DegradationSeverity.MAJOR
        assert baseline_service._determine_severity(20.0) == DegradationSeverity.MODERATE
        assert baseline_service._determine_severity(5.0) == DegradationSeverity.MINOR
    
    def test_is_statistically_significant(self, baseline_service):
        """Test statistical significance check."""
        # Large relative difference should be significant
        assert baseline_service._is_statistically_significant(
            current=0.70,
            baseline=0.90,
            metric_type=DegradationMetricType.ACCURACY_DROP
        ) is True
        
        # Small relative difference should not be significant
        assert baseline_service._is_statistically_significant(
            current=0.89,
            baseline=0.90,
            metric_type=DegradationMetricType.ACCURACY_DROP
        ) is False
        
        # Zero baseline should not be significant
        assert baseline_service._is_statistically_significant(
            current=0.10,
            baseline=0.0,
            metric_type=DegradationMetricType.ACCURACY_DROP
        ) is False
    
    def test_get_relevant_metrics_for_task(self, baseline_service):
        """Test getting relevant metrics for different task types."""
        # Test classification
        classification_metrics = baseline_service._get_relevant_metrics_for_task(
            ModelTask.BINARY_CLASSIFICATION
        )
        assert DegradationMetricType.ACCURACY_DROP in classification_metrics
        assert DegradationMetricType.PRECISION_DROP in classification_metrics
        assert DegradationMetricType.MSE_INCREASE not in classification_metrics
        
        # Test regression
        regression_metrics = baseline_service._get_relevant_metrics_for_task(
            ModelTask.REGRESSION
        )
        assert DegradationMetricType.MSE_INCREASE in regression_metrics
        assert DegradationMetricType.R2_SCORE_DROP in regression_metrics
        assert DegradationMetricType.ACCURACY_DROP not in regression_metrics
        
        # Test other tasks (should include only general metrics)
        other_metrics = baseline_service._get_relevant_metrics_for_task(
            ModelTask.CLUSTERING
        )
        assert DegradationMetricType.CONFIDENCE_DROP in other_metrics
        assert DegradationMetricType.PREDICTION_TIME_INCREASE in other_metrics
        assert DegradationMetricType.ACCURACY_DROP not in other_metrics
        assert DegradationMetricType.MSE_INCREASE not in other_metrics
    
    def test_safe_average(self, baseline_service):
        """Test safe average calculation with None values."""
        # Test with valid values
        assert baseline_service._safe_average([1.0, 2.0, 3.0]) == 2.0
        
        # Test with None values
        assert baseline_service._safe_average([1.0, None, 3.0]) == 2.0
        
        # Test with all None values
        assert baseline_service._safe_average([None, None, None]) is None
        
        # Test with empty list
        assert baseline_service._safe_average([]) is None
    
    def test_safe_median(self, baseline_service):
        """Test safe median calculation with None values."""
        # Test with odd number of values
        assert baseline_service._safe_median([1.0, 2.0, 3.0]) == 2.0
        
        # Test with even number of values
        assert baseline_service._safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5
        
        # Test with None values
        assert baseline_service._safe_median([1.0, None, 3.0]) == 2.0
        
        # Test with all None values
        assert baseline_service._safe_median([None, None, None]) is None
    
    def test_weighted_average(self, baseline_service):
        """Test weighted average calculation."""
        # Test basic weighted average
        result = baseline_service._weighted_average([1.0, 2.0, 3.0], [1, 2, 3])
        expected = (1*1 + 2*2 + 3*3) / (1 + 2 + 3)  # = 14/6 = 2.33
        assert abs(result - expected) < 0.01
        
        # Test with None values
        result = baseline_service._weighted_average([1.0, None, 3.0], [1, 2, 3])
        expected = (1*1 + 3*3) / (1 + 3)  # = 10/4 = 2.5
        assert result == expected
        
        # Test with all None values
        assert baseline_service._weighted_average([None, None], [1, 2]) is None
        
        # Test with zero weights
        assert baseline_service._weighted_average([1.0, 2.0], [0, 0]) is None
    
    def test_baseline_with_lookback_filtering(self, baseline_service):
        """Test baseline calculation with lookback period filtering."""
        # Create metrics with different dates
        old_date = datetime.utcnow() - timedelta(days=50)
        recent_date = datetime.utcnow() - timedelta(days=5)
        
        metrics = [
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.80,  # Lower accuracy (older)
                evaluation_date=old_date,
            ),
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.90,  # Higher accuracy (recent)
                evaluation_date=recent_date,
            ),
        ]
        
        # Use short lookback period (should only include recent metric)
        baseline = baseline_service.establish_baseline(
            metrics,
            baseline_method="recent_average",
            lookback_days=10
        )
        
        # Should be closer to recent accuracy
        assert abs(baseline.accuracy - 0.90) < abs(baseline.accuracy - 0.80)