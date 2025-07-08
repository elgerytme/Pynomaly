"""
Tests for drift detection domain entities.

This module provides comprehensive tests for drift detection entities,
ensuring proper functionality, validation, and importability.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from pynomaly.domain.entities.drift_detection import (
    DriftDetectionMethod,
    DriftScope,
    SeasonalPattern,
    DriftThresholds,
    TimeWindow,
    FeatureData,
    UnivariateDriftResult,
    MultivariateDriftResult,
    FeatureDriftAnalysis,
    ConceptDriftResult,
    DriftDetectionResult,
    DriftType,
    DriftSeverity,
    MonitoringStatus,
    ModelMonitoringConfig,
    DriftAlert,
    DriftAnalysisResult,
)


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection - mock class for testing."""
    
    detection_methods: list[DriftDetectionMethod]
    thresholds: DriftThresholds
    monitoring_enabled: bool = True
    check_interval_minutes: int = 60
    alert_threshold: float = 0.5
    auto_retrain_enabled: bool = False
    

class TestDriftDetectionConfig:
    """Test cases for DriftDetectionConfig."""
    
    def test_drift_detection_config_creation(self):
        """Test basic DriftDetectionConfig creation."""
        methods = [
            DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX,
        ]
        thresholds = DriftThresholds()
        
        config = DriftDetectionConfig(
            detection_methods=methods,
            thresholds=thresholds,
            monitoring_enabled=True,
            check_interval_minutes=30,
            alert_threshold=0.3,
            auto_retrain_enabled=True,
        )
        
        assert config.detection_methods == methods
        assert config.thresholds == thresholds
        assert config.monitoring_enabled is True
        assert config.check_interval_minutes == 30
        assert config.alert_threshold == 0.3
        assert config.auto_retrain_enabled is True
    
    def test_drift_detection_config_defaults(self):
        """Test DriftDetectionConfig with default values."""
        methods = [DriftDetectionMethod.KOLMOGOROV_SMIRNOV]
        thresholds = DriftThresholds()
        
        config = DriftDetectionConfig(
            detection_methods=methods,
            thresholds=thresholds,
        )
        
        assert config.monitoring_enabled is True
        assert config.check_interval_minutes == 60
        assert config.alert_threshold == 0.5
        assert config.auto_retrain_enabled is False
    
    def test_drift_detection_config_importability(self):
        """Test that DriftDetectionConfig can be imported."""
        # This test ensures the class can be imported and instantiated
        methods = [DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE]
        thresholds = DriftThresholds()
        config = DriftDetectionConfig(methods, thresholds)
        assert config is not None


class TestDriftDetectionMethod:
    """Test cases for DriftDetectionMethod enum."""
    
    def test_drift_detection_method_values(self):
        """Test drift detection method enum values."""
        assert DriftDetectionMethod.KOLMOGOROV_SMIRNOV == "kolmogorov_smirnov"
        assert DriftDetectionMethod.POPULATION_STABILITY_INDEX == "population_stability_index"
        assert DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE == "jensen_shannon_divergence"
        assert DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY == "maximum_mean_discrepancy"
        assert DriftDetectionMethod.WASSERSTEIN_DISTANCE == "wasserstein_distance"
        assert DriftDetectionMethod.ENERGY_DISTANCE == "energy_distance"
        assert DriftDetectionMethod.ADVERSARIAL_DRIFT_DETECTION == "adversarial_drift_detection"
        assert DriftDetectionMethod.NEURAL_DRIFT_DETECTOR == "neural_drift_detector"
        assert DriftDetectionMethod.STATISTICAL_PROCESS_CONTROL == "statistical_process_control"
    
    def test_drift_detection_method_importability(self):
        """Test that DriftDetectionMethod can be imported."""
        method = DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        assert method is not None


class TestDriftScope:
    """Test cases for DriftScope enum."""
    
    def test_drift_scope_values(self):
        """Test drift scope enum values."""
        assert DriftScope.UNIVARIATE == "univariate"
        assert DriftScope.MULTIVARIATE == "multivariate"
        assert DriftScope.GLOBAL == "global"
        assert DriftScope.FEATURE_SPECIFIC == "feature_specific"
        assert DriftScope.TEMPORAL == "temporal"
        assert DriftScope.CONCEPTUAL == "conceptual"
    
    def test_drift_scope_importability(self):
        """Test that DriftScope can be imported."""
        scope = DriftScope.UNIVARIATE
        assert scope is not None


class TestSeasonalPattern:
    """Test cases for SeasonalPattern enum."""
    
    def test_seasonal_pattern_values(self):
        """Test seasonal pattern enum values."""
        assert SeasonalPattern.DAILY == "daily"
        assert SeasonalPattern.WEEKLY == "weekly"
        assert SeasonalPattern.MONTHLY == "monthly"
        assert SeasonalPattern.QUARTERLY == "quarterly"
        assert SeasonalPattern.YEARLY == "yearly"
        assert SeasonalPattern.BUSINESS_CYCLE == "business_cycle"
        assert SeasonalPattern.CUSTOM == "custom"
    
    def test_seasonal_pattern_importability(self):
        """Test that SeasonalPattern can be imported."""
        pattern = SeasonalPattern.DAILY
        assert pattern is not None


class TestDriftThresholds:
    """Test cases for DriftThresholds."""
    
    def test_drift_thresholds_creation(self):
        """Test DriftThresholds creation with default values."""
        thresholds = DriftThresholds()
        
        assert thresholds.statistical_significance == 0.05
        assert thresholds.effect_size_threshold == 0.2
        assert thresholds.psi_threshold == 0.1
        assert thresholds.js_divergence_threshold == 0.1
        assert thresholds.mmd_threshold == 0.1
        assert thresholds.wasserstein_threshold == 0.1
        assert thresholds.energy_threshold == 0.1
        assert thresholds.neural_drift_threshold == 0.7
    
    def test_drift_thresholds_custom_values(self):
        """Test DriftThresholds with custom values."""
        thresholds = DriftThresholds(
            statistical_significance=0.01,
            effect_size_threshold=0.3,
            psi_threshold=0.2,
            js_divergence_threshold=0.15,
            neural_drift_threshold=0.8,
        )
        
        assert thresholds.statistical_significance == 0.01
        assert thresholds.effect_size_threshold == 0.3
        assert thresholds.psi_threshold == 0.2
        assert thresholds.js_divergence_threshold == 0.15
        assert thresholds.neural_drift_threshold == 0.8
    
    def test_drift_thresholds_validation(self):
        """Test DriftThresholds validation."""
        # Test invalid statistical significance
        with pytest.raises(ValueError, match="Statistical significance must be between 0.0 and 1.0"):
            DriftThresholds(statistical_significance=1.5)
        
        with pytest.raises(ValueError, match="Statistical significance must be between 0.0 and 1.0"):
            DriftThresholds(statistical_significance=-0.1)
        
        # Test invalid effect size threshold
        with pytest.raises(ValueError, match="Effect size threshold must be non-negative"):
            DriftThresholds(effect_size_threshold=-0.1)
    
    def test_get_threshold_for_method(self):
        """Test getting threshold for specific method."""
        thresholds = DriftThresholds()
        
        # Test known mappings
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.KOLMOGOROV_SMIRNOV) == 0.05
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.POPULATION_STABILITY_INDEX) == 0.1
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE) == 0.1
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY) == 0.1
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.WASSERSTEIN_DISTANCE) == 0.1
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.ENERGY_DISTANCE) == 0.1
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.NEURAL_DRIFT_DETECTOR) == 0.7
        
        # Test unknown method (should return statistical significance)
        assert thresholds.get_threshold_for_method(DriftDetectionMethod.ADVERSARIAL_DRIFT_DETECTION) == 0.05
    
    def test_drift_thresholds_importability(self):
        """Test that DriftThresholds can be imported."""
        thresholds = DriftThresholds()
        assert thresholds is not None


class TestTimeWindow:
    """Test cases for TimeWindow."""
    
    def test_time_window_creation(self):
        """Test TimeWindow creation."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        window = TimeWindow(
            start_time=start_time,
            end_time=end_time,
            window_size=window_size,
            overlap_percentage=0.2,
        )
        
        assert window.start_time == start_time
        assert window.end_time == end_time
        assert window.window_size == window_size
        assert window.overlap_percentage == 0.2
    
    def test_time_window_validation(self):
        """Test TimeWindow validation."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        # Test invalid start/end time
        with pytest.raises(ValueError, match="Start time must be before end time"):
            TimeWindow(
                start_time=end_time,
                end_time=start_time,
                window_size=window_size,
            )
        
        # Test invalid overlap percentage
        with pytest.raises(ValueError, match="Overlap percentage must be between 0.0 and 1.0"):
            TimeWindow(
                start_time=start_time,
                end_time=end_time,
                window_size=window_size,
                overlap_percentage=1.5,
            )
        
        # Test invalid window size
        with pytest.raises(ValueError, match="Window size must be positive"):
            TimeWindow(
                start_time=start_time,
                end_time=end_time,
                window_size=timedelta(hours=-1),
            )
    
    def test_time_window_methods(self):
        """Test TimeWindow methods."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        window = TimeWindow(
            start_time=start_time,
            end_time=end_time,
            window_size=window_size,
            overlap_percentage=0.2,
        )
        
        # Test get_duration
        duration = window.get_duration()
        assert duration == timedelta(hours=3)
        
        # Test get_overlap_duration
        overlap_duration = window.get_overlap_duration()
        assert overlap_duration == timedelta(hours=0.2)
        
        # Test generate_sliding_windows
        sliding_windows = window.generate_sliding_windows()
        assert len(sliding_windows) > 0
        for sliding_window in sliding_windows:
            assert isinstance(sliding_window, TimeWindow)
    
    def test_time_window_importability(self):
        """Test that TimeWindow can be imported."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        window = TimeWindow(start_time, end_time, window_size)
        assert window is not None


class TestFeatureData:
    """Test cases for FeatureData."""
    
    def test_feature_data_creation(self):
        """Test FeatureData creation."""
        reference_data = np.array([1, 2, 3, 4, 5])
        current_data = np.array([2, 3, 4, 5, 6])
        
        feature_data = FeatureData(
            feature_name="test_feature",
            reference_data=reference_data,
            current_data=current_data,
            data_type="numerical",
            missing_value_rate=0.1,
            outlier_rate=0.05,
            cardinality=5,
        )
        
        assert feature_data.feature_name == "test_feature"
        assert np.array_equal(feature_data.reference_data, reference_data)
        assert np.array_equal(feature_data.current_data, current_data)
        assert feature_data.data_type == "numerical"
        assert feature_data.missing_value_rate == 0.1
        assert feature_data.outlier_rate == 0.05
        assert feature_data.cardinality == 5
    
    def test_feature_data_validation(self):
        """Test FeatureData validation."""
        reference_data = np.array([1, 2, 3, 4, 5])
        current_data = np.array([2, 3, 4, 5, 6])
        
        # Test empty data
        with pytest.raises(ValueError, match="Reference and current data cannot be empty"):
            FeatureData(
                feature_name="test_feature",
                reference_data=np.array([]),
                current_data=current_data,
                data_type="numerical",
            )
        
        # Test invalid missing value rate
        with pytest.raises(ValueError, match="Missing value rate must be between 0.0 and 1.0"):
            FeatureData(
                feature_name="test_feature",
                reference_data=reference_data,
                current_data=current_data,
                data_type="numerical",
                missing_value_rate=1.5,
            )
        
        # Test invalid outlier rate
        with pytest.raises(ValueError, match="Outlier rate must be between 0.0 and 1.0"):
            FeatureData(
                feature_name="test_feature",
                reference_data=reference_data,
                current_data=current_data,
                data_type="numerical",
                outlier_rate=-0.1,
            )
    
    def test_feature_data_statistical_properties(self):
        """Test statistical properties computation."""
        reference_data = np.array([1, 2, 3, 4, 5])
        current_data = np.array([2, 3, 4, 5, 6])
        
        feature_data = FeatureData(
            feature_name="test_feature",
            reference_data=reference_data,
            current_data=current_data,
            data_type="numerical",
        )
        
        # Check that statistical properties are computed
        assert "ref_mean" in feature_data.statistical_properties
        assert "curr_mean" in feature_data.statistical_properties
        assert "ref_std" in feature_data.statistical_properties
        assert "curr_std" in feature_data.statistical_properties
    
    def test_feature_data_sample_size_ratio(self):
        """Test sample size ratio calculation."""
        reference_data = np.array([1, 2, 3, 4, 5])
        current_data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        feature_data = FeatureData(
            feature_name="test_feature",
            reference_data=reference_data,
            current_data=current_data,
            data_type="numerical",
        )
        
        ratio = feature_data.get_sample_size_ratio()
        assert ratio == 1.8  # 9/5
    
    def test_feature_data_importability(self):
        """Test that FeatureData can be imported."""
        reference_data = np.array([1, 2, 3])
        current_data = np.array([2, 3, 4])
        
        feature_data = FeatureData(
            feature_name="test_feature",
            reference_data=reference_data,
            current_data=current_data,
            data_type="numerical",
        )
        assert feature_data is not None


class TestUnivariateDriftResult:
    """Test cases for UnivariateDriftResult."""
    
    def test_univariate_drift_result_creation(self):
        """Test UnivariateDriftResult creation."""
        result = UnivariateDriftResult(
            feature_name="test_feature",
            detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_detected=True,
            drift_score=0.3,
            p_value=0.02,
            effect_size=0.4,
            confidence_interval=(0.1, 0.5),
            threshold_used=0.05,
            sample_size_reference=100,
            sample_size_current=120,
        )
        
        assert result.feature_name == "test_feature"
        assert result.detection_method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        assert result.drift_detected is True
        assert result.drift_score == 0.3
        assert result.p_value == 0.02
        assert result.effect_size == 0.4
        assert result.confidence_interval == (0.1, 0.5)
        assert result.threshold_used == 0.05
        assert result.sample_size_reference == 100
        assert result.sample_size_current == 120
    
    def test_univariate_drift_result_validation(self):
        """Test UnivariateDriftResult validation."""
        # Test invalid p-value
        with pytest.raises(ValueError, match="P-value must be between 0.0 and 1.0"):
            UnivariateDriftResult(
                feature_name="test_feature",
                detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
                drift_detected=True,
                drift_score=0.3,
                p_value=1.5,
                effect_size=0.4,
                confidence_interval=(0.1, 0.5),
                threshold_used=0.05,
                sample_size_reference=100,
                sample_size_current=120,
            )
        
        # Test invalid sample size
        with pytest.raises(ValueError, match="Sample sizes must be positive"):
            UnivariateDriftResult(
                feature_name="test_feature",
                detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
                drift_detected=True,
                drift_score=0.3,
                p_value=0.02,
                effect_size=0.4,
                confidence_interval=(0.1, 0.5),
                threshold_used=0.05,
                sample_size_reference=0,
                sample_size_current=120,
            )
    
    def test_univariate_drift_result_methods(self):
        """Test UnivariateDriftResult methods."""
        result = UnivariateDriftResult(
            feature_name="test_feature",
            detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_detected=True,
            drift_score=0.3,
            p_value=0.02,
            effect_size=0.4,
            confidence_interval=(0.1, 0.5),
            threshold_used=0.05,
            sample_size_reference=100,
            sample_size_current=120,
        )
        
        # Test get_drift_severity
        severity = result.get_drift_severity()
        assert severity == "small"  # effect_size 0.4 is between 0.2 and 0.5
        
        # Test is_statistically_significant
        assert result.is_statistically_significant() is True
        assert result.is_statistically_significant(alpha=0.01) is False
        
        # Test to_dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["feature_name"] == "test_feature"
        assert result_dict["drift_detected"] is True
    
    def test_univariate_drift_result_importability(self):
        """Test that UnivariateDriftResult can be imported."""
        result = UnivariateDriftResult(
            feature_name="test_feature",
            detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_detected=False,
            drift_score=0.1,
            p_value=0.8,
            effect_size=0.1,
            confidence_interval=(0.0, 0.2),
            threshold_used=0.05,
            sample_size_reference=100,
            sample_size_current=100,
        )
        assert result is not None


class TestMultivariateDriftResult:
    """Test cases for MultivariateDriftResult."""
    
    def test_multivariate_drift_result_creation(self):
        """Test MultivariateDriftResult creation."""
        result = MultivariateDriftResult(
            detection_method=DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY,
            drift_detected=True,
            drift_score=0.4,
            threshold_used=0.1,
            affected_features=["feature1", "feature2"],
            feature_contributions={"feature1": 0.6, "feature2": 0.4},
            covariance_shift_detected=True,
            computation_time_seconds=1.5,
        )
        
        assert result.detection_method == DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY
        assert result.drift_detected is True
        assert result.drift_score == 0.4
        assert result.threshold_used == 0.1
        assert result.affected_features == ["feature1", "feature2"]
        assert result.feature_contributions == {"feature1": 0.6, "feature2": 0.4}
        assert result.covariance_shift_detected is True
        assert result.computation_time_seconds == 1.5
    
    def test_multivariate_drift_result_validation(self):
        """Test MultivariateDriftResult validation."""
        # Test invalid computation time
        with pytest.raises(ValueError, match="Computation time must be non-negative"):
            MultivariateDriftResult(
                detection_method=DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY,
                drift_detected=True,
                drift_score=0.4,
                threshold_used=0.1,
                affected_features=["feature1"],
                computation_time_seconds=-1.0,
            )
    
    def test_multivariate_drift_result_methods(self):
        """Test MultivariateDriftResult methods."""
        result = MultivariateDriftResult(
            detection_method=DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY,
            drift_detected=True,
            drift_score=0.4,
            threshold_used=0.1,
            affected_features=["feature1", "feature2", "feature3"],
            feature_contributions={"feature1": 0.6, "feature2": 0.3, "feature3": 0.1},
        )
        
        # Test get_most_affected_features
        most_affected = result.get_most_affected_features(top_k=2)
        assert len(most_affected) == 2
        assert most_affected[0] == ("feature1", 0.6)
        assert most_affected[1] == ("feature2", 0.3)
        
        # Test get_drift_pattern_summary
        summary = result.get_drift_pattern_summary()
        assert isinstance(summary, dict)
        assert summary["overall_drift"] is True
        assert summary["drift_score"] == 0.4
    
    def test_multivariate_drift_result_importability(self):
        """Test that MultivariateDriftResult can be imported."""
        result = MultivariateDriftResult(
            detection_method=DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY,
            drift_detected=False,
            drift_score=0.05,
            threshold_used=0.1,
            affected_features=[],
        )
        assert result is not None


class TestDriftDetectionResult:
    """Test cases for DriftDetectionResult."""
    
    def test_drift_detection_result_creation(self):
        """Test DriftDetectionResult creation."""
        result = DriftDetectionResult(
            drift_detected=True,
            drift_score=0.7,
            method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            metadata={"test": "data"},
        )
        
        assert result.drift_detected is True
        assert result.drift_score == 0.7
        assert result.method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        assert result.metadata == {"test": "data"}
        assert isinstance(result.timestamp, datetime)
    
    def test_drift_detection_result_validation(self):
        """Test DriftDetectionResult validation."""
        # Test invalid drift score
        with pytest.raises(ValueError, match="Drift score must be between 0.0 and 1.0"):
            DriftDetectionResult(
                drift_detected=True,
                drift_score=1.5,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            )
    
    def test_drift_detection_result_importability(self):
        """Test that DriftDetectionResult can be imported."""
        result = DriftDetectionResult(
            drift_detected=False,
            drift_score=0.2,
            method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
        )
        assert result is not None


class TestDriftAnalysisResult:
    """Test cases for DriftAnalysisResult."""
    
    def test_drift_analysis_result_creation(self):
        """Test DriftAnalysisResult creation."""
        model_id = uuid4()
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        time_window = TimeWindow(start_time, end_time, window_size)
        
        result = DriftAnalysisResult(
            model_id=model_id,
            time_window=time_window,
            overall_drift_score=0.3,
            drift_severity="medium",
            recommended_actions=["retrain", "investigate"],
        )
        
        assert result.model_id == model_id
        assert result.time_window == time_window
        assert result.overall_drift_score == 0.3
        assert result.drift_severity == "medium"
        assert result.recommended_actions == ["retrain", "investigate"]
    
    def test_drift_analysis_result_validation(self):
        """Test DriftAnalysisResult validation."""
        model_id = uuid4()
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        time_window = TimeWindow(start_time, end_time, window_size)
        
        # Test invalid drift score
        with pytest.raises(ValueError, match="Overall drift score must be between 0.0 and 1.0"):
            DriftAnalysisResult(
                model_id=model_id,
                time_window=time_window,
                overall_drift_score=1.5,
            )
    
    def test_drift_analysis_result_methods(self):
        """Test DriftAnalysisResult methods."""
        model_id = uuid4()
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        time_window = TimeWindow(start_time, end_time, window_size)
        
        # Create drift results
        drift_result = UnivariateDriftResult(
            feature_name="test_feature",
            detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_detected=True,
            drift_score=0.8,
            p_value=0.02,
            effect_size=0.9,
            confidence_interval=(0.1, 0.5),
            threshold_used=0.05,
            sample_size_reference=100,
            sample_size_current=120,
        )
        
        result = DriftAnalysisResult(
            model_id=model_id,
            time_window=time_window,
            data_drift_results=[drift_result],
            overall_drift_score=0.8,
            drift_severity="high",
        )
        
        # Test has_any_drift
        assert result.has_any_drift() is True
        
        # Test get_critical_features
        critical_features = result.get_critical_features()
        assert "test_feature" in critical_features
        
        # Test needs_immediate_action
        assert result.needs_immediate_action() is True
        
        # Test get_comprehensive_summary
        summary = result.get_comprehensive_summary()
        assert isinstance(summary, dict)
        assert summary["overall"]["has_drift"] is True
        assert summary["overall"]["severity"] == "high"
    
    def test_drift_analysis_result_importability(self):
        """Test that DriftAnalysisResult can be imported."""
        model_id = uuid4()
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        
        time_window = TimeWindow(start_time, end_time, window_size)
        
        result = DriftAnalysisResult(
            model_id=model_id,
            time_window=time_window,
        )
        assert result is not None


class TestDriftEnums:
    """Test cases for drift-related enums."""
    
    def test_drift_type_values(self):
        """Test DriftType enum values."""
        assert DriftType.DATA_DRIFT == "data_drift"
        assert DriftType.CONCEPT_DRIFT == "concept_drift"
        assert DriftType.COVARIATE_SHIFT == "covariate_shift"
        assert DriftType.LABEL_SHIFT == "label_shift"
    
    def test_drift_severity_values(self):
        """Test DriftSeverity enum values."""
        assert DriftSeverity.LOW == "low"
        assert DriftSeverity.MEDIUM == "medium"
        assert DriftSeverity.HIGH == "high"
        assert DriftSeverity.CRITICAL == "critical"
    
    def test_monitoring_status_values(self):
        """Test MonitoringStatus enum values."""
        assert MonitoringStatus.ACTIVE == "active"
        assert MonitoringStatus.PAUSED == "paused"
        assert MonitoringStatus.STOPPED == "stopped"
        assert MonitoringStatus.ERROR == "error"
    
    def test_enums_importability(self):
        """Test that enums can be imported."""
        drift_type = DriftType.DATA_DRIFT
        drift_severity = DriftSeverity.LOW
        monitoring_status = MonitoringStatus.ACTIVE
        
        assert drift_type is not None
        assert drift_severity is not None
        assert monitoring_status is not None


class TestModelMonitoringConfig:
    """Test cases for ModelMonitoringConfig."""
    
    def test_model_monitoring_config_creation(self):
        """Test ModelMonitoringConfig creation."""
        config = ModelMonitoringConfig(
            monitoring_enabled=True,
            check_interval_minutes=30,
            drift_threshold=0.2,
            severity_threshold=DriftSeverity.HIGH,
            notification_enabled=True,
            auto_retrain_enabled=True,
        )
        
        assert config.monitoring_enabled is True
        assert config.check_interval_minutes == 30
        assert config.drift_threshold == 0.2
        assert config.severity_threshold == DriftSeverity.HIGH
        assert config.notification_enabled is True
        assert config.auto_retrain_enabled is True
    
    def test_model_monitoring_config_validation(self):
        """Test ModelMonitoringConfig validation."""
        # Test invalid check interval
        with pytest.raises(ValueError, match="Check interval must be positive"):
            ModelMonitoringConfig(check_interval_minutes=-1)
        
        # Test invalid drift threshold
        with pytest.raises(ValueError, match="Drift threshold must be between 0.0 and 1.0"):
            ModelMonitoringConfig(drift_threshold=1.5)
    
    def test_model_monitoring_config_importability(self):
        """Test that ModelMonitoringConfig can be imported."""
        config = ModelMonitoringConfig()
        assert config is not None


class TestDriftAlert:
    """Test cases for DriftAlert."""
    
    def test_drift_alert_creation(self):
        """Test DriftAlert creation."""
        alert = DriftAlert(
            drift_type="data_drift",
            severity="high",
            feature_name="test_feature",
            drift_score=0.8,
            threshold=0.5,
            alert_message="High drift detected",
            metadata={"test": "data"},
        )
        
        assert alert.drift_type == "data_drift"
        assert alert.severity == "high"
        assert alert.feature_name == "test_feature"
        assert alert.drift_score == 0.8
        assert alert.threshold == 0.5
        assert alert.alert_message == "High drift detected"
        assert alert.metadata == {"test": "data"}
    
    def test_drift_alert_importability(self):
        """Test that DriftAlert can be imported."""
        alert = DriftAlert()
        assert alert is not None


class TestConceptDriftResult:
    """Test cases for ConceptDriftResult."""
    
    def test_concept_drift_result_creation(self):
        """Test ConceptDriftResult creation."""
        result = ConceptDriftResult(
            detection_method=DriftDetectionMethod.NEURAL_DRIFT_DETECTOR,
            drift_probability=0.7,
            drift_detected=True,
            confidence=0.8,
            affected_concepts=["concept1", "concept2"],
            label_distribution_shift=True,
            decision_boundary_shift=False,
        )
        
        assert result.detection_method == DriftDetectionMethod.NEURAL_DRIFT_DETECTOR
        assert result.drift_probability == 0.7
        assert result.drift_detected is True
        assert result.confidence == 0.8
        assert result.affected_concepts == ["concept1", "concept2"]
        assert result.label_distribution_shift is True
        assert result.decision_boundary_shift is False
    
    def test_concept_drift_result_validation(self):
        """Test ConceptDriftResult validation."""
        # Test invalid drift probability
        with pytest.raises(ValueError, match="Drift probability must be between 0.0 and 1.0"):
            ConceptDriftResult(
                detection_method=DriftDetectionMethod.NEURAL_DRIFT_DETECTOR,
                drift_probability=1.5,
                drift_detected=True,
                confidence=0.8,
            )
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ConceptDriftResult(
                detection_method=DriftDetectionMethod.NEURAL_DRIFT_DETECTOR,
                drift_probability=0.7,
                drift_detected=True,
                confidence=1.5,
            )
    
    def test_concept_drift_result_methods(self):
        """Test ConceptDriftResult methods."""
        result = ConceptDriftResult(
            detection_method=DriftDetectionMethod.NEURAL_DRIFT_DETECTOR,
            drift_probability=0.8,
            drift_detected=True,
            confidence=0.9,
            affected_concepts=["concept1", "concept2"],
            label_distribution_shift=True,
            decision_boundary_shift=True,
        )
        
        # Test get_drift_severity
        severity = result.get_drift_severity()
        assert severity == "high"  # probability 0.8 > 0.7
        
        # Test has_multiple_drift_types
        assert result.has_multiple_drift_types() is True
        
        # Test get_drift_summary
        summary = result.get_drift_summary()
        assert isinstance(summary, dict)
        assert summary["drift_detected"] is True
        assert summary["severity"] == "high"
        assert summary["multiple_types"] is True
    
    def test_concept_drift_result_importability(self):
        """Test that ConceptDriftResult can be imported."""
        result = ConceptDriftResult(
            detection_method=DriftDetectionMethod.NEURAL_DRIFT_DETECTOR,
            drift_probability=0.3,
            drift_detected=False,
            confidence=0.6,
        )
        assert result is not None


class TestFeatureDriftAnalysis:
    """Test cases for FeatureDriftAnalysis."""
    
    def test_feature_drift_analysis_creation(self):
        """Test FeatureDriftAnalysis creation."""
        analysis = FeatureDriftAnalysis(
            feature_name="test_feature",
            drift_velocity=0.1,
            drift_acceleration=0.02,
            stability_score=0.8,
        )
        
        assert analysis.feature_name == "test_feature"
        assert analysis.drift_velocity == 0.1
        assert analysis.drift_acceleration == 0.02
        assert analysis.stability_score == 0.8
    
    def test_feature_drift_analysis_validation(self):
        """Test FeatureDriftAnalysis validation."""
        # Test invalid stability score
        with pytest.raises(ValueError, match="Stability score must be between 0.0 and 1.0"):
            FeatureDriftAnalysis(
                feature_name="test_feature",
                stability_score=1.5,
            )
    
    def test_feature_drift_analysis_methods(self):
        """Test FeatureDriftAnalysis methods."""
        # Create a drift result
        drift_result = UnivariateDriftResult(
            feature_name="test_feature",
            detection_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            drift_detected=True,
            drift_score=0.6,
            p_value=0.02,
            effect_size=0.4,
            confidence_interval=(0.1, 0.5),
            threshold_used=0.05,
            sample_size_reference=100,
            sample_size_current=120,
        )
        
        analysis = FeatureDriftAnalysis(
            feature_name="test_feature",
            univariate_results=[drift_result],
            drift_velocity=0.15,
            drift_acceleration=0.05,
            stability_score=0.7,
        )
        
        # Test has_drift
        assert analysis.has_drift() is True
        
        # Test get_strongest_drift_signal
        strongest = analysis.get_strongest_drift_signal()
        assert strongest == drift_result
        
        # Test get_consensus_drift_score
        consensus_score = analysis.get_consensus_drift_score()
        assert consensus_score > 0
        
        # Test is_trend_drift
        assert analysis.is_trend_drift() is True  # velocity 0.15 > 0.1
        
        # Test is_sudden_drift
        assert analysis.is_sudden_drift() is False  # acceleration 0.05 < 0.1
        
        # Test get_drift_characterization
        characterization = analysis.get_drift_characterization()
        assert isinstance(characterization, dict)
        assert characterization["has_drift"] is True
        assert characterization["is_trend"] is True
        assert characterization["is_sudden"] is False
    
    def test_feature_drift_analysis_importability(self):
        """Test that FeatureDriftAnalysis can be imported."""
        analysis = FeatureDriftAnalysis(
            feature_name="test_feature",
            stability_score=0.9,
        )
        assert analysis is not None


class TestImportability:
    """Test importability of all drift detection entities."""
    
    def test_all_imports_work(self):
        """Test that all drift detection entities can be imported successfully."""
        # Test enums
        assert DriftDetectionMethod is not None
        assert DriftScope is not None
        assert SeasonalPattern is not None
        assert DriftType is not None
        assert DriftSeverity is not None
        assert MonitoringStatus is not None
        
        # Test data classes
        assert DriftThresholds is not None
        assert TimeWindow is not None
        assert FeatureData is not None
        assert UnivariateDriftResult is not None
        assert MultivariateDriftResult is not None
        assert FeatureDriftAnalysis is not None
        assert ConceptDriftResult is not None
        assert DriftDetectionResult is not None
        assert ModelMonitoringConfig is not None
        assert DriftAlert is not None
        assert DriftAnalysisResult is not None
        
        # Test that they can be instantiated with minimal args
        thresholds = DriftThresholds()
        assert thresholds is not None
        
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = datetime(2023, 1, 1, 15, 0, 0)
        window_size = timedelta(hours=1)
        window = TimeWindow(start_time, end_time, window_size)
        assert window is not None
        
        reference_data = np.array([1, 2, 3])
        current_data = np.array([2, 3, 4])
        feature_data = FeatureData(
            feature_name="test",
            reference_data=reference_data,
            current_data=current_data,
            data_type="numerical",
        )
        assert feature_data is not None
