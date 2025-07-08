#!/usr/bin/env python3
"""
Simple test runner for drift detection tests without pytest dependencies.
"""

import sys
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

# Add src to path
sys.path.insert(0, 'src')

import numpy as np

# Import directly from the module to avoid initialization issues
sys.path.insert(0, 'src/pynomaly/domain/entities')
from drift_detection import (
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


def test_drift_detection_config():
    """Test basic DriftDetectionConfig creation."""
    print("Testing DriftDetectionConfig...")
    
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
    print("✓ DriftDetectionConfig test passed")


def test_drift_detection_method():
    """Test drift detection method enum values."""
    print("Testing DriftDetectionMethod...")
    
    assert DriftDetectionMethod.KOLMOGOROV_SMIRNOV.value == "kolmogorov_smirnov"
    assert DriftDetectionMethod.POPULATION_STABILITY_INDEX.value == "population_stability_index"
    assert DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE.value == "jensen_shannon_divergence"
    assert DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY.value == "maximum_mean_discrepancy"
    assert DriftDetectionMethod.WASSERSTEIN_DISTANCE.value == "wasserstein_distance"
    assert DriftDetectionMethod.ENERGY_DISTANCE.value == "energy_distance"
    assert DriftDetectionMethod.ADVERSARIAL_DRIFT_DETECTION.value == "adversarial_drift_detection"
    assert DriftDetectionMethod.NEURAL_DRIFT_DETECTOR.value == "neural_drift_detector"
    assert DriftDetectionMethod.STATISTICAL_PROCESS_CONTROL.value == "statistical_process_control"
    
    method = DriftDetectionMethod.KOLMOGOROV_SMIRNOV
    assert method is not None
    print("✓ DriftDetectionMethod test passed")


def test_drift_thresholds():
    """Test DriftThresholds."""
    print("Testing DriftThresholds...")
    
    # Test default values
    thresholds = DriftThresholds()
    assert thresholds.statistical_significance == 0.05
    assert thresholds.effect_size_threshold == 0.2
    assert thresholds.psi_threshold == 0.1
    assert thresholds.js_divergence_threshold == 0.1
    assert thresholds.mmd_threshold == 0.1
    assert thresholds.wasserstein_threshold == 0.1
    assert thresholds.energy_threshold == 0.1
    assert thresholds.neural_drift_threshold == 0.7
    
    # Test custom values
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
    
    # Test validation
    try:
        DriftThresholds(statistical_significance=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        DriftThresholds(effect_size_threshold=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test get_threshold_for_method
    assert thresholds.get_threshold_for_method(DriftDetectionMethod.KOLMOGOROV_SMIRNOV) == 0.01
    assert thresholds.get_threshold_for_method(DriftDetectionMethod.POPULATION_STABILITY_INDEX) == 0.2
    
    print("✓ DriftThresholds test passed")


def test_time_window():
    """Test TimeWindow."""
    print("Testing TimeWindow...")
    
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
    
    # Test methods
    duration = window.get_duration()
    assert duration == timedelta(hours=3)
    
    overlap_duration = window.get_overlap_duration()
    assert overlap_duration == timedelta(hours=0.2)
    
    sliding_windows = window.generate_sliding_windows()
    assert len(sliding_windows) > 0
    
    print("✓ TimeWindow test passed")


def test_feature_data():
    """Test FeatureData."""
    print("Testing FeatureData...")
    
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
    
    # Test statistical properties
    assert "ref_mean" in feature_data.statistical_properties
    assert "curr_mean" in feature_data.statistical_properties
    
    # Test sample size ratio
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
    
    print("✓ FeatureData test passed")


def test_univariate_drift_result():
    """Test UnivariateDriftResult."""
    print("Testing UnivariateDriftResult...")
    
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
    
    # Test methods
    severity = result.get_drift_severity()
    assert severity == "small"  # effect_size 0.4 is between 0.2 and 0.5
    
    assert result.is_statistically_significant() is True
    assert result.is_statistically_significant(alpha=0.01) is False
    
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict["feature_name"] == "test_feature"
    assert result_dict["drift_detected"] is True
    
    print("✓ UnivariateDriftResult test passed")


def test_multivariate_drift_result():
    """Test MultivariateDriftResult."""
    print("Testing MultivariateDriftResult...")
    
    result = MultivariateDriftResult(
        detection_method=DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY,
        drift_detected=True,
        drift_score=0.4,
        threshold_used=0.1,
        affected_features=["feature1", "feature2", "feature3"],
        feature_contributions={"feature1": 0.6, "feature2": 0.3, "feature3": 0.1},
        covariance_shift_detected=True,
        computation_time_seconds=1.5,
    )
    
    assert result.detection_method == DriftDetectionMethod.MAXIMUM_MEAN_DISCREPANCY
    assert result.drift_detected is True
    assert result.drift_score == 0.4
    assert result.threshold_used == 0.1
    assert result.affected_features == ["feature1", "feature2", "feature3"]
    assert result.feature_contributions == {"feature1": 0.6, "feature2": 0.3, "feature3": 0.1}
    assert result.covariance_shift_detected is True
    assert result.computation_time_seconds == 1.5
    
    # Test methods
    most_affected = result.get_most_affected_features(top_k=2)
    assert len(most_affected) == 2
    assert most_affected[0] == ("feature1", 0.6)
    assert most_affected[1] == ("feature2", 0.3)
    
    summary = result.get_drift_pattern_summary()
    assert isinstance(summary, dict)
    assert summary["overall_drift"] is True
    assert summary["drift_score"] == 0.4
    
    print("✓ MultivariateDriftResult test passed")


def test_drift_analysis_result():
    """Test DriftAnalysisResult."""
    print("Testing DriftAnalysisResult...")
    
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
        recommended_actions=["retrain", "investigate"],
    )
    
    assert result.model_id == model_id
    assert result.time_window == time_window
    assert result.overall_drift_score == 0.8
    assert result.drift_severity == "high"
    assert result.recommended_actions == ["retrain", "investigate"]
    
    # Test methods
    assert result.has_any_drift() is True
    
    critical_features = result.get_critical_features()
    assert "test_feature" in critical_features
    
    assert result.needs_immediate_action() is True
    
    summary = result.get_comprehensive_summary()
    assert isinstance(summary, dict)
    assert summary["overall"]["has_drift"] is True
    assert summary["overall"]["severity"] == "high"
    
    print("✓ DriftAnalysisResult test passed")


def test_all_enums():
    """Test all enum values and importability."""
    print("Testing all enums...")
    
    # Test DriftScope
    assert DriftScope.UNIVARIATE.value == "univariate"
    assert DriftScope.MULTIVARIATE.value == "multivariate"
    assert DriftScope.GLOBAL.value == "global"
    assert DriftScope.FEATURE_SPECIFIC.value == "feature_specific"
    assert DriftScope.TEMPORAL.value == "temporal"
    assert DriftScope.CONCEPTUAL.value == "conceptual"
    
    # Test SeasonalPattern
    assert SeasonalPattern.DAILY.value == "daily"
    assert SeasonalPattern.WEEKLY.value == "weekly"
    assert SeasonalPattern.MONTHLY.value == "monthly"
    assert SeasonalPattern.QUARTERLY.value == "quarterly"
    assert SeasonalPattern.YEARLY.value == "yearly"
    assert SeasonalPattern.BUSINESS_CYCLE.value == "business_cycle"
    assert SeasonalPattern.CUSTOM.value == "custom"
    
    # Test DriftType
    assert DriftType.DATA_DRIFT.value == "data_drift"
    assert DriftType.CONCEPT_DRIFT.value == "concept_drift"
    assert DriftType.COVARIATE_SHIFT.value == "covariate_shift"
    assert DriftType.LABEL_SHIFT.value == "label_shift"
    
    # Test DriftSeverity
    assert DriftSeverity.LOW.value == "low"
    assert DriftSeverity.MEDIUM.value == "medium"
    assert DriftSeverity.HIGH.value == "high"
    assert DriftSeverity.CRITICAL.value == "critical"
    
    # Test MonitoringStatus
    assert MonitoringStatus.ACTIVE.value == "active"
    assert MonitoringStatus.PAUSED.value == "paused"
    assert MonitoringStatus.STOPPED.value == "stopped"
    assert MonitoringStatus.ERROR.value == "error"
    
    print("✓ All enums test passed")


def test_all_imports():
    """Test that all drift detection entities can be imported successfully."""
    print("Testing all imports...")
    
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
    
    print("✓ All imports test passed")


def main():
    """Run all tests."""
    print("Running drift detection entity tests...")
    print("=" * 50)
    
    tests = [
        test_drift_detection_config,
        test_drift_detection_method,
        test_drift_thresholds,
        test_time_window,
        test_feature_data,
        test_univariate_drift_result,
        test_multivariate_drift_result,
        test_drift_analysis_result,
        test_all_enums,
        test_all_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {failed} tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
