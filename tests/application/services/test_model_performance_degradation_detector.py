"""Test suite for ModelPerformanceDegradationDetector service."""

from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np
import pytest

from pynomaly.application.services.model_performance_degradation_detector import (
    ModelPerformanceDegradationDetector,
    DetectionAlgorithm,
    DegradationDetectorConfig,
    DegradationDetails,
    DegradationResult
)
from pynomaly.domain.entities.model_performance import (
    ModelPerformanceMetrics,
    ModelPerformanceBaseline
)


class TestModelPerformanceDegradationDetector:
    """Test suite for ModelPerformanceDegradationDetector functionality."""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Create baseline metrics for testing."""
        return ModelPerformanceBaseline(
            model_id="test_model_1",
            version="1.0",
            mean=0.85,
            std=0.05
        )
    
    @pytest.fixture
    def good_current_metrics(self):
        """Create current metrics that should not trigger degradation."""
        return ModelPerformanceMetrics(
            accuracy=0.82,
            precision=0.83,
            recall=0.81,
            f1=0.82,
            timestamp=datetime.now(),
            model_id="test_model_1",
            dataset_id="test_dataset_1"
        )
    
    @pytest.fixture
    def degraded_current_metrics(self):
        """Create current metrics that should trigger degradation."""
        return ModelPerformanceMetrics(
            accuracy=0.70,
            precision=0.68,
            recall=0.65,
            f1=0.66,
            timestamp=datetime.now(),
            model_id="test_model_1",
            dataset_id="test_dataset_1"
        )
    
    @pytest.fixture
    def zero_std_baseline(self):
        """Create baseline with zero standard deviation for edge case testing."""
        return ModelPerformanceBaseline(
            model_id="test_model_1",
            version="1.0",
            mean=0.85,
            std=0.0
        )

    def test_config_validation_valid(self):
        """Test configuration validation with valid parameters."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            delta=0.1,
            confidence=0.95
        )
        assert config.algorithm == DetectionAlgorithm.SIMPLE_THRESHOLD
        assert config.delta == 0.1
        assert config.confidence == 0.95
    
    def test_config_validation_invalid_delta(self):
        """Test configuration validation with invalid delta."""
        with pytest.raises(ValueError, match="Delta must be between 0.0 and 1.0"):
            DegradationDetectorConfig(
                algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
                delta=1.5
            )
    
    def test_config_validation_invalid_confidence(self):
        """Test configuration validation with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            DegradationDetectorConfig(
                algorithm=DetectionAlgorithm.STATISTICAL,
                confidence=1.5
            )
    
    def test_config_validation_invalid_statistical_method(self):
        """Test configuration validation with invalid statistical method."""
        with pytest.raises(ValueError, match="Statistical method must be"):
            DegradationDetectorConfig(
                algorithm=DetectionAlgorithm.STATISTICAL,
                statistical_method="invalid_method"
            )
    
    def test_config_validation_metric_weights(self):
        """Test configuration validation with metric weights."""
        # Valid weights
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            metric_weights={"accuracy": 0.4, "precision": 0.3, "recall": 0.2, "f1": 0.1}
        )
        assert config.metric_weights is not None
        
        # Invalid weights (don't sum to 1)
        with pytest.raises(ValueError, match="Metric weights must sum to 1.0"):
            DegradationDetectorConfig(
                algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
                metric_weights={"accuracy": 0.6, "precision": 0.3}
            )
    
    def test_simple_threshold_no_degradation(self, baseline_metrics, good_current_metrics):
        """Test simple threshold detection with no degradation."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            delta=0.1
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(good_current_metrics, baseline_metrics)
        
        assert isinstance(result, DegradationResult)
        assert result.degrade_flag is False
        assert len(result.affected_metrics) == 0
        assert result.detection_algorithm == DetectionAlgorithm.SIMPLE_THRESHOLD
        assert result.overall_severity == 0.0
    
    def test_simple_threshold_with_degradation(self, baseline_metrics, degraded_current_metrics):
        """Test simple threshold detection with degradation."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            delta=0.15
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        assert result.degrade_flag is True
        assert len(result.affected_metrics) > 0
        assert result.detection_algorithm == DetectionAlgorithm.SIMPLE_THRESHOLD
        assert result.overall_severity > 0.0
        
        # Check that affected metrics have correct structure
        for detail in result.affected_metrics:
            assert isinstance(detail, DegradationDetails)
            assert detail.metric_name in ['accuracy', 'precision', 'recall', 'f1']
            assert detail.current_value < baseline_metrics.mean * (1 - config.delta)
            assert detail.deviation > 0  # Degradation means positive deviation
    
    def test_statistical_z_score_no_degradation(self, baseline_metrics, good_current_metrics):
        """Test statistical z-score detection with no degradation."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.STATISTICAL,
            statistical_method="z_score",
            confidence=0.95
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(good_current_metrics, baseline_metrics)
        
        assert result.degrade_flag is False
        assert len(result.affected_metrics) == 0
        assert result.detection_algorithm == DetectionAlgorithm.STATISTICAL
    
    def test_statistical_z_score_with_degradation(self, baseline_metrics, degraded_current_metrics):
        """Test statistical z-score detection with degradation."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.STATISTICAL,
            statistical_method="z_score",
            confidence=0.95
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        assert result.degrade_flag is True
        assert len(result.affected_metrics) > 0
        assert result.detection_algorithm == DetectionAlgorithm.STATISTICAL
        
        # Check statistical significance data
        for detail in result.affected_metrics:
            assert detail.statistical_significance is not None
            assert "z_score" in detail.statistical_significance
            assert "p_value" in detail.statistical_significance
            assert detail.statistical_significance["z_score"] < 0  # Negative z-score indicates degradation
    
    def test_statistical_t_test_with_degradation(self, baseline_metrics, degraded_current_metrics):
        """Test statistical t-test detection with degradation."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.STATISTICAL,
            statistical_method="t_test",
            confidence=0.95
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        assert result.degrade_flag is True
        assert len(result.affected_metrics) > 0
        
        # Check t-test specific data
        for detail in result.affected_metrics:
            assert detail.statistical_significance is not None
            assert "t_stat" in detail.statistical_significance
            assert "p_value" in detail.statistical_significance
            assert "df" in detail.statistical_significance
    
    def test_statistical_zero_std_fallback(self, zero_std_baseline, degraded_current_metrics):
        """Test statistical detection fallback when std is zero."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.STATISTICAL,
            statistical_method="z_score",
            confidence=0.95
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, zero_std_baseline)
        
        # Should fallback to threshold method
        assert isinstance(result, DegradationResult)
        if result.degrade_flag:
            for detail in result.affected_metrics:
                assert "threshold_fallback" in detail.statistical_significance["method"]
    
    def test_ml_based_no_model(self, baseline_metrics, degraded_current_metrics):
        """Test ML-based detection without providing a model."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.ML_BASED,
            ml_model=None
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        # Should fallback to simple threshold
        assert isinstance(result, DegradationResult)
        assert result.detection_algorithm == DetectionAlgorithm.ML_BASED
    
    def test_ml_based_with_mock_model(self, baseline_metrics, degraded_current_metrics):
        """Test ML-based detection with a mock model."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% degradation probability
        
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.ML_BASED,
            ml_model=mock_model
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        assert result.degrade_flag is True
        assert len(result.affected_metrics) > 0
        assert result.detection_algorithm == DetectionAlgorithm.ML_BASED
        
        # Check that mock model was called
        mock_model.predict_proba.assert_called_once()
        
        # Check ML-specific metadata
        assert "ml_model_type" in result.metadata
        assert "degradation_probability" in result.metadata
        assert result.metadata["degradation_probability"] == 0.7
    
    def test_ml_based_model_exception_fallback(self, baseline_metrics, degraded_current_metrics):
        """Test ML-based detection fallback when model raises exception."""
        # Create a mock model that raises exception
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Model error")
        
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.ML_BASED,
            ml_model=mock_model
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        # Should fallback to simple threshold
        assert isinstance(result, DegradationResult)
        # Note: The algorithm stays ML_BASED but uses simple threshold logic internally
        assert result.detection_algorithm == DetectionAlgorithm.ML_BASED
    
    def test_degradation_details_calculation(self, baseline_metrics):
        """Test DegradationDetails calculation of derived values."""
        detail = DegradationDetails(
            metric_name="accuracy",
            current_value=0.70,
            baseline_value=0.85,
            deviation=0.0,  # Will be calculated
            relative_deviation=0.0  # Will be calculated
        )
        
        assert abs(detail.deviation - 0.15) < 0.001  # 0.85 - 0.70
        assert abs(detail.relative_deviation - 17.647) < 0.001  # (0.15 / 0.85) * 100
    
    def test_degradation_result_to_dict(self, baseline_metrics, degraded_current_metrics):
        """Test DegradationResult to_dict conversion."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            delta=0.15
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "degrade_flag" in result_dict
        assert "affected_metrics" in result_dict
        assert "detection_algorithm" in result_dict
        assert "overall_severity" in result_dict
        assert "metadata" in result_dict
        
        # Check affected metrics structure
        if result_dict["affected_metrics"]:
            for metric_dict in result_dict["affected_metrics"]:
                assert "metric_name" in metric_dict
                assert "current_value" in metric_dict
                assert "baseline_value" in metric_dict
                assert "deviation" in metric_dict
                assert "relative_deviation" in metric_dict
    
    def test_unsupported_algorithm_error(self, baseline_metrics, degraded_current_metrics):
        """Test error handling for unsupported algorithm."""
        # This should not be possible with enum, but test the error handling
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD
        )
        detector = ModelPerformanceDegradationDetector(config)
        
        # Manually change algorithm to invalid value
        detector.config.algorithm = "invalid_algorithm"
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            detector.detect(degraded_current_metrics, baseline_metrics)
    
    def test_overall_severity_calculation(self, baseline_metrics, degraded_current_metrics):
        """Test overall severity calculation."""
        config = DegradationDetectorConfig(
            algorithm=DetectionAlgorithm.SIMPLE_THRESHOLD,
            delta=0.15
        )
        detector = ModelPerformanceDegradationDetector(config)
        result = detector.detect(degraded_current_metrics, baseline_metrics)
        
        if result.degrade_flag:
            # Severity should be calculated as average relative deviation / 100
            expected_severity = np.mean([
                detail.relative_deviation for detail in result.affected_metrics
            ]) / 100.0
            
            assert abs(result.overall_severity - expected_severity) < 0.001
        else:
            assert result.overall_severity == 0.0
