"""Comprehensive test suite for ConceptDriftDetectionService."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from anomaly_detection.domain.services.concept_drift_detection_service import (
    ConceptDriftDetectionService,
    DriftDetectionMethod,
    DriftSeverity,
    DriftDetectionResult,
    DriftAnalysisReport
)


class TestConceptDriftDetectionService:
    """Test suite for ConceptDriftDetectionService."""
    
    @pytest.fixture
    def drift_service(self):
        """Create ConceptDriftDetectionService instance."""
        return ConceptDriftDetectionService(
            window_size=100,
            reference_window_size=200,
            drift_threshold=0.05,
            min_samples=50
        )
    
    @pytest.fixture
    def reference_data(self):
        """Create stable reference data."""
        np.random.seed(42)
        return np.random.normal(0, 1, (200, 5))
    
    @pytest.fixture
    def drifted_data(self):
        """Create drifted data with shifted distribution."""
        np.random.seed(43)
        # Shifted mean and variance
        return np.random.normal(2, 1.5, (100, 5))
    
    @pytest.fixture
    def stable_data(self):
        """Create stable current data (no drift)."""
        np.random.seed(44)
        return np.random.normal(0, 1, (100, 5))
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        service = ConceptDriftDetectionService()
        
        assert service.window_size == 1000
        assert service.reference_window_size == 2000
        assert service.drift_threshold == 0.05
        assert service.min_samples == 100
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        service = ConceptDriftDetectionService(
            window_size=500,
            reference_window_size=1000,
            drift_threshold=0.1,
            min_samples=75
        )
        
        assert service.window_size == 500
        assert service.reference_window_size == 1000
        assert service.drift_threshold == 0.1
        assert service.min_samples == 75
    
    def test_add_reference_data(self, drift_service, reference_data):
        """Test adding reference data."""
        model_id = "test_model"
        predictions = np.array([-1, 1, 1, -1, 1] * 40)
        performance_metrics = {"accuracy": 0.95, "precision": 0.90}
        
        drift_service.add_reference_data(
            model_id=model_id,
            data=reference_data,
            predictions=predictions,
            performance_metrics=performance_metrics
        )
        
        assert model_id in drift_service._reference_data
        assert len(drift_service._reference_data[model_id]) == len(reference_data)
        assert len(drift_service._prediction_history[model_id]) == len(predictions)
        assert len(drift_service._performance_history[model_id]) == 1
    
    def test_add_current_data(self, drift_service, drifted_data):
        """Test adding current data."""
        model_id = "test_model"
        
        drift_service.add_current_data(model_id=model_id, data=drifted_data)
        
        assert model_id in drift_service._current_data
        assert len(drift_service._current_data[model_id]) == len(drifted_data)
    
    def test_detect_drift_insufficient_data(self, drift_service):
        """Test drift detection with insufficient data."""
        model_id = "test_model"
        
        # Add minimal data (below min_samples)
        small_data = np.random.normal(0, 1, (10, 3))
        drift_service.add_reference_data(model_id, small_data)
        drift_service.add_current_data(model_id, small_data)
        
        report = drift_service.detect_drift(model_id)
        
        assert report.model_id == model_id
        assert not report.overall_drift_detected
        assert report.overall_severity == DriftSeverity.NO_DRIFT
        assert len(report.detection_results) == 0
        assert "Insufficient data" in report.recommendations[0]
    
    def test_detect_drift_no_drift_case(self, drift_service, reference_data, stable_data):
        """Test drift detection when no drift is present."""
        model_id = "test_model"
        
        # Add data without drift
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, stable_data)
        
        # Test specific methods that don't require external dependencies
        methods = [
            DriftDetectionMethod.STATISTICAL_DISTANCE,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
            DriftDetectionMethod.DISTRIBUTION_SHIFT
        ]
        
        report = drift_service.detect_drift(model_id, methods=methods)
        
        assert report.model_id == model_id
        assert len(report.detection_results) == len(methods)
        # With stable data, most methods should not detect drift
        assert report.consensus_score < 0.5  # Less than half methods detect drift
    
    def test_detect_drift_with_drift(self, drift_service, reference_data, drifted_data):
        """Test drift detection when drift is present."""
        model_id = "test_model"
        
        # Add data with clear drift
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        # Test methods that don't require external dependencies
        methods = [
            DriftDetectionMethod.STATISTICAL_DISTANCE,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
            DriftDetectionMethod.DISTRIBUTION_SHIFT
        ]
        
        report = drift_service.detect_drift(model_id, methods=methods)
        
        assert report.model_id == model_id
        assert len(report.detection_results) == len(methods)
        # With drifted data, should detect drift
        detected_methods = [r for r in report.detection_results if r.drift_detected]
        assert len(detected_methods) > 0
        
        # Check that some drift was detected
        if report.overall_drift_detected:
            assert report.overall_severity != DriftSeverity.NO_DRIFT
            assert "drift detected" in " ".join(report.recommendations).lower()
    
    def test_statistical_distance_drift(self, drift_service, reference_data, drifted_data):
        """Test statistical distance drift detection method."""
        model_id = "test_model"
        threshold = 0.1
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        result = drift_service._detect_statistical_distance_drift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.STATISTICAL_DISTANCE
        assert result.threshold == threshold
        assert result.timestamp is not None
        assert "feature_distances" in result.metadata
        assert result.confidence > 0
    
    def test_psi_drift_detection(self, drift_service, reference_data, drifted_data):
        """Test Population Stability Index drift detection."""
        model_id = "test_model"
        threshold = 0.05
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        result = drift_service._detect_psi_drift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.POPULATION_STABILITY_INDEX
        assert "psi_scores" in result.metadata
        assert "interpretation" in result.metadata
        assert result.drift_score >= 0
    
    def test_js_divergence_drift(self, drift_service, reference_data, drifted_data):
        """Test Jensen-Shannon divergence drift detection."""
        model_id = "test_model"
        threshold = 0.05
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        result = drift_service._detect_js_divergence_drift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE
        assert result.drift_score >= 0
        assert result.drift_score <= 1  # JS divergence is bounded
        assert "js_divergences" in result.metadata
    
    def test_ks_drift_with_scipy(self, drift_service, reference_data, drifted_data):
        """Test Kolmogorov-Smirnov drift detection with scipy."""
        model_id = "test_model"
        threshold = 0.05
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        with patch('scipy.stats.ks_2samp') as mock_ks:
            # Mock KS test results
            mock_ks.return_value = (0.3, 0.01)  # High statistic, low p-value
            
            result = drift_service._detect_ks_drift(model_id, threshold)
            
            assert isinstance(result, DriftDetectionResult)
            assert result.method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV
            assert result.p_value is not None
            assert "p_values" in result.metadata
            assert "ks_statistics" in result.metadata
    
    def test_ks_drift_without_scipy(self, drift_service, reference_data, drifted_data):
        """Test KS drift detection fallback when scipy is not available."""
        model_id = "test_model"
        threshold = 0.05
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        with patch('scipy.stats.ks_2samp', side_effect=ImportError):
            result = drift_service._detect_ks_drift(model_id, threshold)
            
            # Should fall back to statistical distance method
            assert isinstance(result, DriftDetectionResult)
            assert result.method == DriftDetectionMethod.STATISTICAL_DISTANCE
    
    def test_distribution_shift_drift(self, drift_service, reference_data, drifted_data):
        """Test distribution shift drift detection."""
        model_id = "test_model"
        threshold = 0.1
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        result = drift_service._detect_distribution_shift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.DISTRIBUTION_SHIFT
        assert "mean_shifts" in result.metadata
        assert "std_shifts" in result.metadata
        assert "ref_means" in result.metadata
        assert "curr_means" in result.metadata
    
    def test_performance_degradation_drift(self, drift_service):
        """Test performance degradation drift detection."""
        model_id = "test_model"
        threshold = 0.1
        
        # Add performance history showing degradation
        performance_metrics = []
        # Good performance initially
        for _ in range(10):
            performance_metrics.append({"accuracy": 0.95, "precision": 0.92, "recall": 0.90})
        # Degraded performance recently
        for _ in range(10):
            performance_metrics.append({"accuracy": 0.80, "precision": 0.75, "recall": 0.78})
        
        # Add to service
        for metrics in performance_metrics:
            drift_service._performance_history[model_id] = drift_service._performance_history.get(model_id, [])
            drift_service._performance_history[model_id].append(metrics)
        
        result = drift_service._detect_performance_degradation(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.PERFORMANCE_DEGRADATION
        # Should detect degradation
        assert result.drift_detected
        assert len(result.affected_features) > 0
    
    def test_performance_degradation_insufficient_data(self, drift_service):
        """Test performance degradation with insufficient data."""
        model_id = "test_model"
        threshold = 0.1
        
        result = drift_service._detect_performance_degradation(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert not result.drift_detected
        assert result.severity == DriftSeverity.NO_DRIFT
        assert "Insufficient performance history" in result.metadata["error"]
    
    def test_prediction_drift_detection(self, drift_service):
        """Test prediction drift detection."""
        model_id = "test_model"
        threshold = 0.2
        
        # Add prediction history with shift in anomaly rate
        predictions = []
        # Initially low anomaly rate
        predictions.extend([1] * 80 + [-1] * 20)  # 20% anomaly rate
        # Then high anomaly rate  
        predictions.extend([1] * 40 + [-1] * 60)  # 60% anomaly rate
        
        drift_service._prediction_history[model_id] = predictions
        
        result = drift_service._detect_prediction_drift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.PREDICTION_DRIFT
        # Should detect significant change in anomaly rate
        assert result.drift_detected
        assert "anomaly_rate" in result.affected_features
    
    def test_prediction_drift_insufficient_data(self, drift_service):
        """Test prediction drift with insufficient data."""
        model_id = "test_model"
        threshold = 0.1
        
        result = drift_service._detect_prediction_drift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert not result.drift_detected
        assert "Insufficient prediction history" in result.metadata["error"]
    
    def test_feature_importance_drift(self, drift_service):
        """Test feature importance drift detection (placeholder)."""
        model_id = "test_model"
        threshold = 0.1
        
        result = drift_service._detect_feature_importance_drift(model_id, threshold)
        
        assert isinstance(result, DriftDetectionResult)
        assert result.method == DriftDetectionMethod.FEATURE_IMPORTANCE_DRIFT
        # Currently a placeholder
        assert not result.drift_detected
        assert "placeholder" in result.metadata["implementation"]
    
    def test_custom_thresholds(self, drift_service, reference_data, drifted_data):
        """Test drift detection with custom thresholds."""
        model_id = "test_model"
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        custom_thresholds = {
            DriftDetectionMethod.STATISTICAL_DISTANCE: 0.01,  # Very sensitive
            DriftDetectionMethod.DISTRIBUTION_SHIFT: 0.5      # Very tolerant
        }
        
        methods = list(custom_thresholds.keys())
        report = drift_service.detect_drift(
            model_id, 
            methods=methods, 
            custom_thresholds=custom_thresholds
        )
        
        for result in report.detection_results:
            expected_threshold = custom_thresholds[result.method]
            assert result.threshold == expected_threshold
    
    def test_severity_calculation(self, drift_service):
        """Test drift severity calculation."""
        threshold = 0.1
        
        # Test different severity levels
        test_cases = [
            (0.05, DriftSeverity.NO_DRIFT),
            (0.15, DriftSeverity.LOW),      # 1.5 * threshold
            (0.3, DriftSeverity.MEDIUM),    # 3 * threshold
            (0.6, DriftSeverity.HIGH),      # 6 * threshold
            (1.0, DriftSeverity.CRITICAL)   # 10 * threshold
        ]
        
        for score, expected_severity in test_cases:
            severity = drift_service._calculate_severity(score, threshold)
            assert severity == expected_severity
    
    def test_wasserstein_distance_calculation(self, drift_service):
        """Test Wasserstein distance approximation."""
        # Create two different distributions
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(2, 1, 100)  # Shifted distribution
        
        distance = drift_service._wasserstein_distance(x, y)
        
        assert distance > 0
        assert isinstance(distance, float)
        
        # Distance to itself should be small
        self_distance = drift_service._wasserstein_distance(x, x)
        assert self_distance < distance
    
    def test_psi_calculation(self, drift_service):
        """Test Population Stability Index calculation."""
        # Create reference and current data
        ref = np.random.normal(0, 1, 1000)
        curr_stable = np.random.normal(0, 1, 1000)  # Stable
        curr_drifted = np.random.normal(2, 1, 1000)  # Drifted
        
        psi_stable = drift_service._calculate_psi(ref, curr_stable)
        psi_drifted = drift_service._calculate_psi(ref, curr_drifted)
        
        assert psi_stable >= 0
        assert psi_drifted >= 0
        assert psi_drifted > psi_stable  # Drifted should have higher PSI
    
    def test_jensen_shannon_divergence_calculation(self, drift_service):
        """Test Jensen-Shannon divergence calculation."""
        # Create different distributions
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)  # Same distribution
        z = np.random.normal(3, 1, 100)  # Different distribution
        
        js_same = drift_service._jensen_shannon_divergence(x, y)
        js_diff = drift_service._jensen_shannon_divergence(x, z)
        
        assert 0 <= js_same <= 1
        assert 0 <= js_diff <= 1
        assert js_diff > js_same  # Different distributions should have higher JS divergence
    
    def test_drift_analysis_report_methods(self):
        """Test DriftAnalysisReport utility methods."""
        # Create sample results
        result1 = DriftDetectionResult(
            method=DriftDetectionMethod.STATISTICAL_DISTANCE,
            drift_detected=True,
            drift_score=0.15,
            severity=DriftSeverity.LOW,
            p_value=None,
            affected_features=["feature_1"],
            threshold=0.1,
            confidence=0.9,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        result2 = DriftDetectionResult(
            method=DriftDetectionMethod.DISTRIBUTION_SHIFT,
            drift_detected=False,
            drift_score=0.05,
            severity=DriftSeverity.NO_DRIFT,
            p_value=None,
            affected_features=[],
            threshold=0.1,
            confidence=0.8,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        report = DriftAnalysisReport(
            model_id="test_model",
            timestamp=datetime.utcnow(),
            reference_period=(datetime.utcnow() - timedelta(days=7), datetime.utcnow() - timedelta(days=3)),
            current_period=(datetime.utcnow() - timedelta(days=3), datetime.utcnow()),
            detection_results=[result1, result2],
            overall_drift_detected=True,
            overall_severity=DriftSeverity.LOW,
            consensus_score=0.5,
            recommendations=["Test recommendation"]
        )
        
        # Test get_results_by_method
        stat_result = report.get_results_by_method(DriftDetectionMethod.STATISTICAL_DISTANCE)
        assert stat_result == result1
        
        missing_result = report.get_results_by_method(DriftDetectionMethod.KOLMOGOROV_SMIRNOV)
        assert missing_result is None
        
        # Test get_detected_drifts
        detected = report.get_detected_drifts()
        assert len(detected) == 1
        assert detected[0] == result1
    
    def test_drift_detection_result_summary(self):
        """Test DriftDetectionResult summary property."""
        result = DriftDetectionResult(
            method=DriftDetectionMethod.STATISTICAL_DISTANCE,
            drift_detected=True,
            drift_score=0.15,
            severity=DriftSeverity.MEDIUM,
            p_value=0.02,
            affected_features=["feature_1", "feature_2"],
            threshold=0.1,
            confidence=0.85,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        summary = result.summary
        assert "DETECTED" in summary
        assert "statistical_distance" in summary
        assert "0.1500" in summary
        assert "medium" in summary
    
    def test_clear_data(self, drift_service, reference_data):
        """Test clearing stored data for a model."""
        model_id = "test_model"
        
        # Add some data
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, reference_data)
        
        # Verify data exists
        assert model_id in drift_service._reference_data
        assert model_id in drift_service._current_data
        
        # Clear data
        drift_service.clear_data(model_id)
        
        # Verify data is cleared
        assert model_id not in drift_service._reference_data
        assert model_id not in drift_service._current_data
    
    def test_get_drift_history(self, drift_service):
        """Test getting drift history (placeholder)."""
        model_id = "test_model"
        history = drift_service.get_drift_history(model_id, hours=24)
        
        # Currently returns empty list as placeholder
        assert isinstance(history, list)
        assert len(history) == 0
    
    @patch('anomaly_detection.domain.services.concept_drift_detection_service.get_model_performance_monitor')
    def test_record_drift_metrics_integration(self, mock_get_monitor, drift_service):
        """Test integration with model performance monitor."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        
        # Create a drift report
        result = DriftDetectionResult(
            method=DriftDetectionMethod.STATISTICAL_DISTANCE,
            drift_detected=True,
            drift_score=0.15,
            severity=DriftSeverity.MEDIUM,
            p_value=None,
            affected_features=["feature_1"],
            threshold=0.1,
            confidence=0.9,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        report = DriftAnalysisReport(
            model_id="test_model",
            timestamp=datetime.utcnow(),
            reference_period=(datetime.utcnow() - timedelta(days=7), datetime.utcnow() - timedelta(days=3)),
            current_period=(datetime.utcnow() - timedelta(days=3), datetime.utcnow()),
            detection_results=[result],
            overall_drift_detected=True,
            overall_severity=DriftSeverity.MEDIUM,
            consensus_score=0.8,
            recommendations=["Test recommendation"]
        )
        
        # Test recording metrics
        drift_service._record_drift_metrics("test_model", report)
        
        # Verify monitor was called
        mock_monitor.record_drift_metrics.assert_called_once()
        call_args = mock_monitor.record_drift_metrics.call_args[1]
        assert call_args["model_id"] == "test_model"
        assert call_args["drift_detected"] is True
        assert call_args["severity"] == "medium"
    
    def test_generate_recommendations(self, drift_service):
        """Test recommendation generation."""
        # Test no drift case
        recommendations = drift_service._generate_recommendations([], DriftSeverity.NO_DRIFT)
        assert "No significant drift detected" in recommendations[0]
        
        # Test critical drift case
        critical_result = DriftDetectionResult(
            method=DriftDetectionMethod.PERFORMANCE_DEGRADATION,
            drift_detected=True,
            drift_score=0.8,
            severity=DriftSeverity.CRITICAL,
            p_value=None,
            affected_features=["accuracy"],
            threshold=0.1,
            confidence=0.95,
            timestamp=datetime.utcnow(),
            metadata={}
        )
        
        recommendations = drift_service._generate_recommendations(
            [critical_result], 
            DriftSeverity.CRITICAL
        )
        
        assert any("URGENT" in rec for rec in recommendations)
        assert any("backup model" in rec for rec in recommendations)
        assert any("Performance degradation" in rec for rec in recommendations)
    
    def test_error_handling_in_methods(self, drift_service):
        """Test error handling in drift detection methods."""
        model_id = "test_model"
        
        # Add minimal data that might cause errors
        small_data = np.array([[1, 2], [3, 4]])
        drift_service.add_reference_data(model_id, small_data)
        drift_service.add_current_data(model_id, small_data)
        
        # Test that methods handle errors gracefully
        try:
            result = drift_service._detect_statistical_distance_drift(model_id, 0.1)
            assert isinstance(result, DriftDetectionResult)
        except Exception as e:
            pytest.fail(f"Method should handle errors gracefully: {e}")
    
    def test_drift_detection_with_method_failures(self, drift_service, reference_data, drifted_data):
        """Test drift detection when some methods fail."""
        model_id = "test_model"
        
        drift_service.add_reference_data(model_id, reference_data)
        drift_service.add_current_data(model_id, drifted_data)
        
        # Mock one method to raise an exception
        with patch.object(drift_service, '_detect_statistical_distance_drift', side_effect=Exception("Test error")):
            methods = [
                DriftDetectionMethod.STATISTICAL_DISTANCE,
                DriftDetectionMethod.DISTRIBUTION_SHIFT
            ]
            
            report = drift_service.detect_drift(model_id, methods=methods)
            
            # Should continue with remaining methods
            assert len(report.detection_results) < len(methods)  # Some methods failed
            assert len(report.detection_results) > 0  # But some succeeded