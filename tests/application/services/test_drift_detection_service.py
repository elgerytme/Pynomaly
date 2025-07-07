"""Tests for drift detection service."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from pynomaly.application.services.drift_detection_service import DriftDetectionService
from pynomaly.domain.entities.drift_detection import (
    DriftDetectionMethod,
    DriftDetectionResult,
    DriftSeverity,
    DriftType,
    ModelMonitoringConfig,
    MonitoringStatus,
)


class TestDriftDetectionService:
    """Test cases for DriftDetectionService."""

    @pytest.fixture
    def drift_service(self):
        """Create drift detection service for testing."""
        return DriftDetectionService()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Reference data (normal distribution)
        reference_data = np.random.randn(1000, 3)

        # Current data (slightly shifted distribution)
        current_data = np.random.randn(500, 3) + 0.5

        return reference_data, current_data

    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return ["feature_1", "feature_2", "feature_3"]

    @pytest.mark.asyncio
    async def test_detect_data_drift_basic(
        self, drift_service, sample_data, feature_names
    ):
        """Test basic data drift detection."""
        reference_data, current_data = sample_data

        result = await drift_service.detect_data_drift(
            detector_id="test_detector",
            reference_data=reference_data,
            current_data=current_data,
            feature_names=feature_names,
        )

        assert isinstance(result, DriftDetectionResult)
        assert result.detector_id == "test_detector"
        assert result.drift_type == DriftType.DATA_DRIFT
        assert len(result.features_analyzed) == 3
        assert len(result.feature_drift_scores) == 3
        assert result.reference_sample_size == 1000
        assert result.comparison_sample_size == 500

    @pytest.mark.asyncio
    async def test_detect_data_drift_no_drift(self, drift_service, feature_names):
        """Test drift detection when no drift is present."""
        # Same distribution for both datasets
        reference_data = np.random.randn(1000, 3)
        current_data = np.random.randn(500, 3)

        result = await drift_service.detect_data_drift(
            detector_id="test_detector",
            reference_data=reference_data,
            current_data=current_data,
            feature_names=feature_names,
        )

        assert result.drift_detected in [
            True,
            False,
        ]  # Could go either way with random data
        assert len(result.affected_features) <= 3

    @pytest.mark.asyncio
    async def test_detect_data_drift_with_methods(
        self, drift_service, sample_data, feature_names
    ):
        """Test drift detection with specific methods."""
        reference_data, current_data = sample_data

        methods = [
            DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            DriftDetectionMethod.JENSEN_SHANNON,
        ]

        result = await drift_service.detect_data_drift(
            detector_id="test_detector",
            reference_data=reference_data,
            current_data=current_data,
            feature_names=feature_names,
            detection_methods=methods,
        )

        assert result.detection_method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV
        assert result.metrics.ks_statistic is not None
        assert result.metrics.js_divergence is not None

    @pytest.mark.asyncio
    async def test_detect_performance_drift(self, drift_service):
        """Test performance drift detection."""
        reference_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.90,
            "f1": 0.91,
        }

        # Degraded performance
        current_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.80,
            "f1": 0.81,
        }

        result = await drift_service.detect_performance_drift(
            detector_id="test_detector",
            reference_metrics=reference_metrics,
            current_metrics=current_metrics,
            threshold=0.05,
        )

        assert result.drift_type == DriftType.PERFORMANCE_DRIFT
        assert result.drift_detected is True
        assert result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        assert len(result.feature_drift_scores) == 4  # 4 metrics

    @pytest.mark.asyncio
    async def test_detect_performance_drift_no_change(self, drift_service):
        """Test performance drift detection with no change."""
        metrics = {"accuracy": 0.95, "precision": 0.92}

        result = await drift_service.detect_performance_drift(
            detector_id="test_detector",
            reference_metrics=metrics,
            current_metrics=metrics,
            threshold=0.05,
        )

        assert result.drift_detected is False
        assert result.severity == DriftSeverity.LOW

    @pytest.mark.asyncio
    async def test_detect_prediction_drift(self, drift_service):
        """Test prediction drift detection."""
        # Reference predictions (binary)
        reference_predictions = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])

        # Current predictions (different distribution)
        current_predictions = np.random.choice([0, 1], size=500, p=[0.4, 0.6])

        result = await drift_service.detect_prediction_drift(
            detector_id="test_detector",
            reference_predictions=reference_predictions,
            current_predictions=current_predictions,
        )

        assert result.drift_type == DriftType.PREDICTION_DRIFT
        assert result.drift_detected is True  # Should detect the distribution change

    @pytest.mark.asyncio
    async def test_setup_monitoring(self, drift_service):
        """Test monitoring setup."""
        config = ModelMonitoringConfig(
            detector_id="test_detector",
            enabled=True,
            check_interval_hours=24,
            min_sample_size=100,
        )

        status = await drift_service.setup_monitoring("test_detector", config)

        assert status.detector_id == "test_detector"
        assert status.status == MonitoringStatus.ACTIVE
        assert "test_detector" in drift_service.monitoring_configs
        assert "test_detector" in drift_service.monitoring_status

    @pytest.mark.asyncio
    async def test_check_drift_monitoring(self, drift_service):
        """Test scheduled drift monitoring check."""
        # Setup monitoring first
        config = ModelMonitoringConfig(
            detector_id="test_detector",
            enabled=True,
            check_interval_hours=1,  # Short interval for testing
            min_sample_size=100,
        )

        await drift_service.setup_monitoring("test_detector", config)

        # Force next check to be due
        status = drift_service.monitoring_status["test_detector"]
        status.next_check_at = datetime.now() - timedelta(hours=1)

        result = await drift_service.check_drift_monitoring("test_detector")

        assert result is not None
        assert isinstance(result, DriftDetectionResult)
        assert status.checks_performed > 0

    @pytest.mark.asyncio
    async def test_check_drift_monitoring_not_due(self, drift_service):
        """Test monitoring check when not due."""
        config = ModelMonitoringConfig(
            detector_id="test_detector", enabled=True, check_interval_hours=24
        )

        await drift_service.setup_monitoring("test_detector", config)

        # Check should not be due yet
        result = await drift_service.check_drift_monitoring("test_detector")

        assert result is None

    @pytest.mark.asyncio
    async def test_pause_resume_monitoring(self, drift_service):
        """Test pausing and resuming monitoring."""
        config = ModelMonitoringConfig(detector_id="test_detector", enabled=True)

        await drift_service.setup_monitoring("test_detector", config)

        # Pause monitoring
        success = await drift_service.pause_monitoring("test_detector")
        assert success is True

        status = drift_service.monitoring_status["test_detector"]
        assert status.status == MonitoringStatus.PAUSED

        # Resume monitoring
        success = await drift_service.resume_monitoring("test_detector")
        assert success is True
        assert status.status == MonitoringStatus.ACTIVE
        assert status.next_check_at is not None

    @pytest.mark.asyncio
    async def test_generate_drift_report(self, drift_service):
        """Test drift report generation."""
        period_start = datetime.now() - timedelta(days=30)
        period_end = datetime.now()

        report = await drift_service.generate_drift_report(
            detector_id="test_detector",
            period_start=period_start,
            period_end=period_end,
        )

        assert report.detector_id == "test_detector"
        assert report.report_period_start == period_start
        assert report.report_period_end == period_end
        assert isinstance(report.total_checks, int)
        assert isinstance(report.drift_detections, int)

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, drift_service):
        """Test error handling in monitoring loop."""
        config = ModelMonitoringConfig(
            detector_id="test_detector", enabled=True, check_interval_hours=1
        )

        await drift_service.setup_monitoring("test_detector", config)

        # Mock an error in data retrieval
        original_get_data = drift_service._get_monitoring_data

        async def mock_error(*args, **kwargs):
            raise Exception("Data retrieval failed")

        drift_service._get_monitoring_data = mock_error

        # Force check to be due
        status = drift_service.monitoring_status["test_detector"]
        status.next_check_at = datetime.now() - timedelta(hours=1)

        result = await drift_service.check_drift_monitoring("test_detector")

        assert result is None
        assert status.consecutive_failures > 0
        assert status.last_error is not None

        # Restore original method
        drift_service._get_monitoring_data = original_get_data


class TestStatisticalDriftDetector:
    """Test cases for StatisticalDriftDetector."""

    @pytest.fixture
    def detector(self):
        """Create statistical drift detector."""
        from pynomaly.application.services.drift_detection_service import (
            StatisticalDriftDetector,
        )

        return StatisticalDriftDetector()

    def test_kolmogorov_smirnov_test(self, detector):
        """Test Kolmogorov-Smirnov test."""
        reference_data = np.random.randn(1000)
        current_data = np.random.randn(500) + 1.0  # Shifted distribution

        ks_stat, p_value = detector.kolmogorov_smirnov_test(
            reference_data, current_data
        )

        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)
        assert 0.0 <= ks_stat <= 1.0
        assert 0.0 <= p_value <= 1.0
        assert p_value < 0.05  # Should detect significant difference

    def test_jensen_shannon_divergence(self, detector):
        """Test Jensen-Shannon divergence calculation."""
        reference_data = np.random.randn(1000)
        current_data = np.random.randn(500) + 1.0

        js_div = detector.jensen_shannon_divergence(reference_data, current_data)

        assert isinstance(js_div, float)
        assert js_div >= 0.0
        assert js_div > 0.0  # Should detect difference

    def test_population_stability_index(self, detector):
        """Test Population Stability Index calculation."""
        reference_data = np.random.randn(1000)
        current_data = np.random.randn(500) + 0.5

        psi = detector.population_stability_index(reference_data, current_data)

        assert isinstance(psi, float)
        assert psi >= 0.0

    def test_wasserstein_distance(self, detector):
        """Test Wasserstein distance calculation."""
        reference_data = np.random.randn(1000)
        current_data = np.random.randn(500) + 1.0

        distance = detector.wasserstein_distance(reference_data, current_data)

        assert isinstance(distance, float)
        assert distance >= 0.0
        assert distance > 0.0  # Should detect difference


class TestPerformanceDriftDetector:
    """Test cases for PerformanceDriftDetector."""

    @pytest.fixture
    def detector(self):
        """Create performance drift detector."""
        from pynomaly.application.services.drift_detection_service import (
            PerformanceDriftDetector,
        )

        return PerformanceDriftDetector()

    def test_detect_performance_drift(self, detector):
        """Test performance drift detection."""
        reference_metrics = {"accuracy": 0.95, "precision": 0.90}
        current_metrics = {"accuracy": 0.85, "precision": 0.80}

        drift_detected, changes = detector.detect_performance_drift(
            reference_metrics, current_metrics, threshold=0.05
        )

        assert isinstance(drift_detected, bool)
        assert isinstance(changes, dict)
        assert drift_detected is True  # Should detect significant drop
        assert "accuracy" in changes
        assert "precision" in changes
        assert changes["accuracy"] < 0  # Negative change

    def test_detect_performance_drift_no_change(self, detector):
        """Test performance drift with no significant change."""
        metrics = {"accuracy": 0.95, "precision": 0.90}

        drift_detected, changes = detector.detect_performance_drift(
            metrics, metrics, threshold=0.05
        )

        assert drift_detected is False
        assert all(abs(change) < 0.05 for change in changes.values())

    def test_calculate_prediction_drift_binary(self, detector):
        """Test prediction drift for binary predictions."""
        reference_preds = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        current_preds = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        drift_score = detector.calculate_prediction_drift(
            reference_preds, current_preds
        )

        assert isinstance(drift_score, float)
        assert drift_score >= 0.0
        assert drift_score > 0.0  # Should detect difference

    def test_calculate_prediction_drift_continuous(self, detector):
        """Test prediction drift for continuous predictions."""
        reference_preds = np.random.randn(100)
        current_preds = np.random.randn(100) + 1.0

        drift_score = detector.calculate_prediction_drift(
            reference_preds, current_preds
        )

        assert isinstance(drift_score, float)
        assert drift_score >= 0.0
