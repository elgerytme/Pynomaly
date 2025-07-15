"""Tests for performance degradation detection service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np

from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask
)
from pynomaly.domain.services.performance_degradation_service import PerformanceDegradationService
from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationSeverity,
    DegradationType,
    MetricThreshold,
    PerformanceBaseline,
    PerformanceDegradation,
)


class TestPerformanceDegradationService:
    """Test suite for PerformanceDegradationService."""
    
    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        model_repo = MagicMock()
        performance_repo = MagicMock()
        alert_service = MagicMock()
        
        return model_repo, performance_repo, alert_service
    
    @pytest.fixture
    def degradation_service(self, mock_repositories):
        """Create degradation service with mocked dependencies."""
        model_repo, performance_repo, alert_service = mock_repositories
        return PerformanceDegradationService(model_repo, performance_repo, alert_service)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        return ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.92
        )
    
    @pytest.fixture
    def degraded_metrics(self):
        """Create degraded performance metrics."""
        return ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.75,  # 10% degradation
            precision=0.70,  # 12% degradation
            recall=0.78,    # 11% degradation
            f1_score=0.74,  # 13% degradation
            roc_auc=0.82    # 11% degradation
        )
    
    @pytest.fixture
    def custom_threshold(self):
        """Create custom threshold for testing."""
        return MetricThreshold(
            metric_name="accuracy",
            warning_threshold=5.0,
            critical_threshold=15.0,
            threshold_type="percentage",
            direction="decrease",
            min_samples=5
        )
    
    def test_default_thresholds_creation(self, degradation_service):
        """Test that default thresholds are created correctly."""
        thresholds = degradation_service._default_thresholds
        
        assert "accuracy" in thresholds
        assert "precision" in thresholds
        assert "recall" in thresholds
        assert "f1_score" in thresholds
        assert "roc_auc" in thresholds
        assert "rmse" in thresholds
        
        # Check accuracy threshold
        accuracy_threshold = thresholds["accuracy"]
        assert accuracy_threshold.metric_name == "accuracy"
        assert accuracy_threshold.direction == "decrease"
        assert accuracy_threshold.threshold_type == "percentage"
    
    @pytest.mark.asyncio
    async def test_detect_degradation_no_history(self, degradation_service, sample_metrics):
        """Test degradation detection when no performance history exists."""
        model_id = uuid4()
        
        # Mock empty performance history
        degradation_service.performance_repository.get_model_performance_history = AsyncMock(return_value=[])
        
        degradations = await degradation_service.detect_degradation(
            model_id=model_id,
            current_metrics=sample_metrics
        )
        
        assert degradations == []
    
    @pytest.mark.asyncio
    async def test_detect_degradation_with_history(self, degradation_service, degraded_metrics):
        """Test degradation detection with performance history."""
        model_id = uuid4()
        
        # Create mock performance history
        baseline_metrics = [
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.90,
                precision=0.88,
                recall=0.92,
                f1_score=0.90,
                roc_auc=0.95
            ) for _ in range(20)
        ]
        
        degradation_service.performance_repository.get_model_performance_history = AsyncMock(
            return_value=baseline_metrics
        )
        
        degradations = await degradation_service.detect_degradation(
            model_id=model_id,
            current_metrics=degraded_metrics
        )
        
        # Should detect degradations in multiple metrics
        assert len(degradations) > 0
        
        # Check that accuracy degradation is detected
        accuracy_degradations = [d for d in degradations if d.metric_name == "accuracy"]
        assert len(accuracy_degradations) > 0
        
        accuracy_degradation = accuracy_degradations[0]
        assert accuracy_degradation.degradation_type == DegradationType.ACCURACY_DROP
        assert accuracy_degradation.current_value == 0.75
        assert accuracy_degradation.baseline_value == 0.90
    
    @pytest.mark.asyncio
    async def test_monitor_continuous_degradation(self, degradation_service):
        """Test continuous degradation monitoring."""
        model_id = uuid4()
        
        # Create trend of degrading metrics
        recent_metrics = []
        for i in range(15):
            accuracy = 0.90 - (i * 0.01)  # Gradual degradation
            recent_metrics.append(
                ModelPerformanceMetrics(
                    task_type=ModelTask.BINARY_CLASSIFICATION,
                    sample_size=1000,
                    accuracy=accuracy,
                    precision=0.85,
                    recall=0.88,
                    f1_score=0.86
                )
            )
        
        # Create stable baseline
        baseline_metrics = [
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.90,
                precision=0.85,
                recall=0.88,
                f1_score=0.86
            ) for _ in range(30)
        ]
        
        # Mock repository calls
        degradation_service.performance_repository.get_model_performance_history = AsyncMock(
            side_effect=[recent_metrics, baseline_metrics]
        )
        
        degradations = await degradation_service.monitor_continuous_degradation(
            model_id=model_id,
            monitoring_window_hours=24,
            min_samples=10
        )
        
        # Should detect degradation trend
        assert len(degradations) > 0
    
    def test_create_baseline_from_values(self, degradation_service):
        """Test baseline creation from metric values."""
        values = [0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.89, 0.84, 0.87, 0.86]
        
        baseline = degradation_service._create_baseline("accuracy", values)
        
        assert baseline.metric_name == "accuracy"
        assert baseline.sample_count == len(values)
        assert 0.84 <= baseline.baseline_value <= 0.89
        assert baseline.standard_deviation > 0
        assert baseline.min_value is not None
        assert baseline.max_value is not None
    
    def test_create_baseline_with_outliers(self, degradation_service):
        """Test baseline creation with outlier removal."""
        # Include some outliers
        values = [0.85, 0.87, 0.84, 0.86, 0.88, 0.50, 0.89, 0.84, 0.95, 0.86]  # 0.50 and 0.95 are outliers
        
        baseline = degradation_service._create_baseline("accuracy", values)
        
        # Outliers should be removed, so baseline should be around normal values
        assert 0.83 <= baseline.baseline_value <= 0.90
        assert baseline.sample_count <= len(values)  # May be fewer due to outlier removal
    
    def test_determine_degradation_type(self, degradation_service):
        """Test degradation type determination."""
        # Test classification metrics
        assert degradation_service._determine_degradation_type("accuracy", "decrease") == DegradationType.ACCURACY_DROP
        assert degradation_service._determine_degradation_type("precision", "decrease") == DegradationType.PRECISION_DROP
        assert degradation_service._determine_degradation_type("recall", "decrease") == DegradationType.RECALL_DROP
        assert degradation_service._determine_degradation_type("f1_score", "decrease") == DegradationType.F1_DROP
        assert degradation_service._determine_degradation_type("roc_auc", "decrease") == DegradationType.AUC_DROP
        
        # Test regression metrics
        assert degradation_service._determine_degradation_type("rmse", "increase") == DegradationType.RMSE_INCREASE
        assert degradation_service._determine_degradation_type("mae", "increase") == DegradationType.MAE_INCREASE
        assert degradation_service._determine_degradation_type("r2_score", "decrease") == DegradationType.R2_DROP
        
        # Test efficiency metrics
        assert degradation_service._determine_degradation_type("prediction_time_seconds", "increase") == DegradationType.LATENCY_INCREASE
        assert degradation_service._determine_degradation_type("memory_usage_mb", "increase") == DegradationType.MEMORY_INCREASE
    
    def test_calculate_confidence(self, degradation_service):
        """Test confidence calculation."""
        baseline = PerformanceBaseline(
            metric_name="accuracy",
            baseline_value=0.85,
            standard_deviation=0.02,
            sample_count=30
        )
        
        threshold = MetricThreshold(
            metric_name="accuracy",
            warning_threshold=5.0,
            critical_threshold=10.0,
            threshold_type="percentage",
            direction="decrease"
        )
        
        # Test high confidence (many standard deviations away)
        high_confidence = degradation_service._calculate_confidence(0.79, baseline, threshold)  # 3 std away
        assert 0.8 <= high_confidence <= 1.0
        
        # Test low confidence (close to baseline)
        low_confidence = degradation_service._calculate_confidence(0.84, baseline, threshold)   # 0.5 std away
        assert 0.0 <= low_confidence <= 0.3
    
    @pytest.mark.asyncio
    async def test_generate_degradation_report(self, degradation_service):
        """Test degradation report generation."""
        model_id = uuid4()
        
        # Create sample degradations
        degradations = [
            PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            PerformanceDegradation(
                degradation_type=DegradationType.PRECISION_DROP,
                severity=DegradationSeverity.CRITICAL,
                metric_name="precision",
                current_value=0.65,
                baseline_value=0.85,
                degradation_amount=-0.20,
                degradation_percentage=-23.53,
                threshold_violated="critical",
                confidence_level=0.92,
                detection_method="baseline_comparison",
                samples_used=25
            )
        ]
        
        time_start = datetime.utcnow() - timedelta(days=1)
        time_end = datetime.utcnow()
        
        report = await degradation_service.generate_degradation_report(
            model_id=model_id,
            degradations=degradations,
            time_period_start=time_start,
            time_period_end=time_end
        )
        
        assert report.model_id == str(model_id)
        assert len(report.degradations) == 2
        assert report.has_critical_degradations
        assert report.has_any_degradations
        assert report.overall_health_score < 1.0
        assert len(report.recommendations) > 0
        
        # Check severity breakdown
        severity_counts = report.degradation_count_by_severity
        assert severity_counts['high'] == 1
        assert severity_counts['critical'] == 1
    
    def test_calculate_health_score_no_degradations(self, degradation_service):
        """Test health score calculation with no degradations."""
        health_score = degradation_service._calculate_health_score([])
        assert health_score == 1.0
    
    def test_calculate_health_score_with_degradations(self, degradation_service):
        """Test health score calculation with various degradations."""
        degradations = [
            PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.CRITICAL,
                metric_name="accuracy",
                current_value=0.70,
                baseline_value=0.85,
                degradation_amount=-0.15,
                degradation_percentage=-17.65,
                threshold_violated="critical",
                confidence_level=0.90,
                detection_method="baseline_comparison",
                samples_used=20
            ),
            PerformanceDegradation(
                degradation_type=DegradationType.PRECISION_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="precision",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=20
            )
        ]
        
        health_score = degradation_service._calculate_health_score(degradations)
        
        # Should be significantly reduced due to critical and high severity degradations
        expected_penalty = 0.5 + 0.3  # critical + high
        expected_score = max(0.0, 1.0 - expected_penalty)
        assert health_score == expected_score
    
    def test_generate_recommendations_no_degradations(self, degradation_service):
        """Test recommendation generation with no degradations."""
        recommendations = degradation_service._generate_recommendations([])
        
        assert len(recommendations) == 1
        assert "stable" in recommendations[0].lower()
    
    def test_generate_recommendations_with_degradations(self, degradation_service):
        """Test recommendation generation with degradations."""
        degradations = [
            PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.CRITICAL,
                metric_name="accuracy",
                current_value=0.70,
                baseline_value=0.85,
                degradation_amount=-0.15,
                degradation_percentage=-17.65,
                threshold_violated="critical",
                confidence_level=0.90,
                detection_method="baseline_comparison",
                samples_used=20
            ),
            PerformanceDegradation(
                degradation_type=DegradationType.PRECISION_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="precision",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=20
            )
        ]
        
        recommendations = degradation_service._generate_recommendations(degradations)
        
        # Should include urgent recommendations for critical degradation
        assert any("URGENT" in rec for rec in recommendations)
        assert any("accuracy" in rec for rec in recommendations)
        assert any("precision" in rec for rec in recommendations)
        assert len(recommendations) >= 3  # Should have multiple recommendations
    
    @pytest.mark.asyncio
    async def test_extract_metric_series(self, degradation_service):
        """Test metric series extraction."""
        metrics_list = [
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.85,
                precision=0.82
            ),
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.87,
                precision=0.84
            ),
            ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.83,
                precision=None  # Missing value
            )
        ]
        
        series = degradation_service._extract_metric_series(metrics_list)
        
        assert "accuracy" in series
        assert "precision" in series
        assert len(series["accuracy"]) == 3
        assert len(series["precision"]) == 2  # One missing value
        assert series["accuracy"] == [0.85, 0.87, 0.83]
        assert series["precision"] == [0.82, 0.84]
    
    @pytest.mark.asyncio
    async def test_perform_trend_analysis_no_degradation(self, degradation_service):
        """Test trend analysis with no significant degradation."""
        recent_values = [0.85, 0.86, 0.84, 0.87, 0.85]  # Stable values
        baseline_values = [0.85, 0.86, 0.85, 0.84, 0.86, 0.85]  # Similar stable values
        
        degradation = await degradation_service._perform_trend_analysis(
            metric_name="accuracy",
            recent_values=recent_values,
            baseline_values=baseline_values
        )
        
        # Should not detect degradation
        assert degradation is None
    
    @pytest.mark.asyncio
    async def test_perform_trend_analysis_with_degradation(self, degradation_service):
        """Test trend analysis with significant degradation."""
        recent_values = [0.75, 0.74, 0.73, 0.72, 0.71]  # Degrading values
        baseline_values = [0.85, 0.86, 0.85, 0.84, 0.86, 0.85]  # Good baseline values
        
        degradation = await degradation_service._perform_trend_analysis(
            metric_name="accuracy",
            recent_values=recent_values,
            baseline_values=baseline_values
        )
        
        # Should detect degradation
        assert degradation is not None
        assert degradation.metric_name == "accuracy"
        assert degradation.degradation_type == DegradationType.ACCURACY_DROP
        assert degradation.detection_method == "trend_analysis"
        assert degradation.severity in [DegradationSeverity.HIGH, DegradationSeverity.CRITICAL]


class TestMetricThreshold:
    """Test suite for MetricThreshold value object."""
    
    def test_valid_threshold_creation(self):
        """Test creating valid metric thresholds."""
        threshold = MetricThreshold(
            metric_name="accuracy",
            warning_threshold=5.0,
            critical_threshold=10.0,
            threshold_type="percentage",
            direction="decrease"
        )
        
        assert threshold.metric_name == "accuracy"
        assert threshold.warning_threshold == 5.0
        assert threshold.critical_threshold == 10.0
        assert threshold.threshold_type == "percentage"
        assert threshold.direction == "decrease"
    
    def test_invalid_threshold_type(self):
        """Test validation of threshold type."""
        with pytest.raises(ValueError, match="threshold_type must be one of"):
            MetricThreshold(
                metric_name="accuracy",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="invalid",
                direction="decrease"
            )
    
    def test_invalid_direction(self):
        """Test validation of direction."""
        with pytest.raises(ValueError, match="direction must be one of"):
            MetricThreshold(
                metric_name="accuracy",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="invalid"
            )
    
    def test_critical_threshold_validation_decrease(self):
        """Test critical threshold validation for decrease direction."""
        with pytest.raises(ValueError, match="Critical threshold must be lower than warning"):
            MetricThreshold(
                metric_name="accuracy",
                warning_threshold=10.0,
                critical_threshold=5.0,  # Should be higher for decrease
                threshold_type="percentage",
                direction="decrease"
            )
    
    def test_critical_threshold_validation_increase(self):
        """Test critical threshold validation for increase direction."""
        with pytest.raises(ValueError, match="Critical threshold must be higher than warning"):
            MetricThreshold(
                metric_name="latency",
                warning_threshold=10.0,
                critical_threshold=5.0,  # Should be higher for increase
                threshold_type="percentage",
                direction="increase"
            )


class TestPerformanceBaseline:
    """Test suite for PerformanceBaseline value object."""
    
    def test_baseline_creation(self):
        """Test creating a performance baseline."""
        baseline = PerformanceBaseline(
            metric_name="accuracy",
            baseline_value=0.85,
            standard_deviation=0.02,
            sample_count=100
        )
        
        assert baseline.metric_name == "accuracy"
        assert baseline.baseline_value == 0.85
        assert baseline.standard_deviation == 0.02
        assert baseline.sample_count == 100
        assert baseline.is_stable is True
    
    def test_is_degraded_percentage_decrease(self):
        """Test degradation detection with percentage threshold (decrease)."""
        baseline = PerformanceBaseline(
            metric_name="accuracy",
            baseline_value=0.80,
            standard_deviation=0.02,
            sample_count=50
        )
        
        threshold = MetricThreshold(
            metric_name="accuracy",
            warning_threshold=5.0,    # 5% decrease
            critical_threshold=10.0,  # 10% decrease
            threshold_type="percentage",
            direction="decrease"
        )
        
        # Test no degradation
        is_degraded, severity = baseline.is_degraded(0.78, threshold)  # 2.5% decrease
        assert not is_degraded
        assert severity == DegradationSeverity.NONE
        
        # Test warning level degradation
        is_degraded, severity = baseline.is_degraded(0.74, threshold)  # 7.5% decrease
        assert is_degraded
        assert severity == DegradationSeverity.HIGH
        
        # Test critical degradation
        is_degraded, severity = baseline.is_degraded(0.70, threshold)  # 12.5% decrease
        assert is_degraded
        assert severity == DegradationSeverity.CRITICAL
    
    def test_is_degraded_standard_deviation(self):
        """Test degradation detection with standard deviation threshold."""
        baseline = PerformanceBaseline(
            metric_name="accuracy",
            baseline_value=0.80,
            standard_deviation=0.02,
            sample_count=50
        )
        
        threshold = MetricThreshold(
            metric_name="accuracy",
            warning_threshold=2.0,    # 2 std deviations
            critical_threshold=3.0,   # 3 std deviations
            threshold_type="standard_deviation",
            direction="decrease"
        )
        
        # Test no degradation (within 2 std)
        is_degraded, severity = baseline.is_degraded(0.78, threshold)  # 1 std away
        assert not is_degraded
        assert severity == DegradationSeverity.NONE
        
        # Test warning level degradation (2-3 std)
        is_degraded, severity = baseline.is_degraded(0.755, threshold)  # 2.25 std away
        assert is_degraded
        assert severity == DegradationSeverity.HIGH
        
        # Test critical degradation (>3 std)
        is_degraded, severity = baseline.is_degraded(0.74, threshold)  # 3 std away
        assert is_degraded
        assert severity == DegradationSeverity.CRITICAL
    
    def test_update_baseline(self):
        """Test baseline updating with new values."""
        original_baseline = PerformanceBaseline(
            metric_name="accuracy",
            baseline_value=0.80,
            standard_deviation=0.02,
            sample_count=30,
            established_at=datetime.utcnow() - timedelta(days=30)
        )
        
        new_values = [0.82, 0.83, 0.81, 0.84, 0.82, 0.83, 0.81, 0.82, 0.83, 0.82]
        
        updated_baseline = original_baseline.update_baseline(new_values)
        
        assert updated_baseline.metric_name == "accuracy"
        assert updated_baseline.sample_count == len(new_values)
        assert 0.81 <= updated_baseline.baseline_value <= 0.84
        assert updated_baseline.established_at == original_baseline.established_at
        assert updated_baseline.last_updated > original_baseline.last_updated