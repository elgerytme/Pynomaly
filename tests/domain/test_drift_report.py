"""Tests for drift report domain entities."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pynomaly.domain.entities.drift_report import (
    DriftConfiguration,
    DriftDetectionMethod,
    DriftMonitor,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDrift,
)


class TestFeatureDrift:
    """Test cases for FeatureDrift dataclass."""

    def test_create_feature_drift(self):
        """Test creating feature drift."""
        drift = FeatureDrift(
            feature_name="temperature",
            drift_score=0.75,
            threshold=0.05,
            is_drifted=True,
            severity=DriftSeverity.HIGH,
            method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            p_value=0.001,
            reference_mean=20.5,
            current_mean=25.3,
        )

        assert drift.feature_name == "temperature"
        assert drift.drift_score == 0.75
        assert drift.is_drifted is True
        assert drift.severity == DriftSeverity.HIGH


class TestDriftConfiguration:
    """Test cases for DriftConfiguration dataclass."""

    def test_default_configuration(self):
        """Test creating default drift configuration."""
        config = DriftConfiguration()

        assert config.drift_threshold == 0.05
        assert config.min_sample_size == 100
        assert config.detection_window_size == 1000
        assert DriftDetectionMethod.KOLMOGOROV_SMIRNOV in config.enabled_methods
        assert config.enable_multivariate_detection is True

    def test_custom_configuration(self):
        """Test creating custom drift configuration."""
        config = DriftConfiguration(
            drift_threshold=0.1,
            min_sample_size=500,
            detection_window_size=2000,
            enabled_methods=[DriftDetectionMethod.WASSERSTEIN_DISTANCE],
            alert_severity_threshold=DriftSeverity.HIGH,
        )

        assert config.drift_threshold == 0.1
        assert config.min_sample_size == 500
        assert config.alert_severity_threshold == DriftSeverity.HIGH

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Invalid drift threshold
        with pytest.raises(
            ValueError, match="Drift threshold must be between 0.0 and 1.0"
        ):
            DriftConfiguration(drift_threshold=1.5)

        with pytest.raises(
            ValueError, match="Drift threshold must be between 0.0 and 1.0"
        ):
            DriftConfiguration(drift_threshold=-0.1)

        # Invalid sample size
        with pytest.raises(ValueError, match="Minimum sample size must be positive"):
            DriftConfiguration(min_sample_size=0)

        # Invalid window size
        with pytest.raises(ValueError, match="Detection window size must be positive"):
            DriftConfiguration(detection_window_size=-100)

    def test_severity_thresholds(self):
        """Test default severity thresholds."""
        config = DriftConfiguration()

        assert config.severity_thresholds["low"] == 0.1
        assert config.severity_thresholds["medium"] == 0.3
        assert config.severity_thresholds["high"] == 0.6
        assert config.severity_thresholds["critical"] == 0.8


class TestDriftReport:
    """Test cases for DriftReport dataclass."""

    def test_create_drift_report(self):
        """Test creating drift report."""
        model_id = uuid4()
        config = DriftConfiguration()

        feature_drift = {
            "feature1": FeatureDrift(
                feature_name="feature1",
                drift_score=0.8,
                threshold=0.05,
                is_drifted=True,
                severity=DriftSeverity.HIGH,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
            "feature2": FeatureDrift(
                feature_name="feature2",
                drift_score=0.02,
                threshold=0.05,
                is_drifted=False,
                severity=DriftSeverity.NONE,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
        }

        report = DriftReport(
            model_id=model_id,
            reference_sample_size=1000,
            current_sample_size=950,
            overall_drift_detected=True,
            overall_drift_severity=DriftSeverity.HIGH,
            drift_types_detected=[DriftType.DATA_DRIFT],
            feature_drift=feature_drift,
            drifted_features=["feature1"],
            configuration=config,
            detection_start_time=datetime.utcnow() - timedelta(minutes=30),
            detection_end_time=datetime.utcnow(),
        )

        assert report.model_id == model_id
        assert report.overall_drift_detected is True
        assert report.overall_drift_severity == DriftSeverity.HIGH
        assert len(report.drifted_features) == 1
        assert "feature1" in report.drifted_features

    def test_get_high_priority_features(self):
        """Test getting high priority features."""
        feature_drift = {
            "low_drift": FeatureDrift(
                feature_name="low_drift",
                drift_score=0.1,
                threshold=0.05,
                is_drifted=True,
                severity=DriftSeverity.LOW,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
            "high_drift": FeatureDrift(
                feature_name="high_drift",
                drift_score=0.9,
                threshold=0.05,
                is_drifted=True,
                severity=DriftSeverity.HIGH,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
            "critical_drift": FeatureDrift(
                feature_name="critical_drift",
                drift_score=0.95,
                threshold=0.05,
                is_drifted=True,
                severity=DriftSeverity.CRITICAL,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
        }

        report = DriftReport(
            model_id=uuid4(),
            reference_sample_size=1000,
            current_sample_size=1000,
            overall_drift_detected=True,
            overall_drift_severity=DriftSeverity.HIGH,
            drift_types_detected=[DriftType.DATA_DRIFT],
            feature_drift=feature_drift,
            drifted_features=["low_drift", "high_drift", "critical_drift"],
            configuration=DriftConfiguration(),
            detection_start_time=datetime.utcnow(),
            detection_end_time=datetime.utcnow(),
        )

        high_priority = report.get_high_priority_features()
        assert len(high_priority) == 2
        assert "high_drift" in high_priority
        assert "critical_drift" in high_priority
        assert "low_drift" not in high_priority

    def test_get_drift_summary(self):
        """Test getting drift summary."""
        feature_drift = {
            "feature1": FeatureDrift(
                feature_name="feature1",
                drift_score=0.8,
                threshold=0.05,
                is_drifted=True,
                severity=DriftSeverity.HIGH,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
            "feature2": FeatureDrift(
                feature_name="feature2",
                drift_score=0.02,
                threshold=0.05,
                is_drifted=False,
                severity=DriftSeverity.NONE,
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            ),
        }

        report = DriftReport(
            model_id=uuid4(),
            reference_sample_size=1000,
            current_sample_size=1000,
            overall_drift_detected=True,
            overall_drift_severity=DriftSeverity.HIGH,
            drift_types_detected=[DriftType.DATA_DRIFT],
            feature_drift=feature_drift,
            drifted_features=["feature1"],
            configuration=DriftConfiguration(),
            detection_start_time=datetime.utcnow(),
            detection_end_time=datetime.utcnow(),
            multivariate_drift_detected=True,
            concept_drift_detected=False,
        )

        summary = report.get_drift_summary()

        assert summary["total_features"] == 2
        assert summary["drifted_features"] == 1
        assert summary["drift_percentage"] == 50.0
        assert summary["multivariate_drift"] is True
        assert summary["concept_drift"] is False
        assert summary["overall_severity"] == "high"

    def test_requires_immediate_attention(self):
        """Test checking if drift requires immediate attention."""
        # High severity report
        report = DriftReport(
            model_id=uuid4(),
            reference_sample_size=1000,
            current_sample_size=1000,
            overall_drift_detected=True,
            overall_drift_severity=DriftSeverity.HIGH,
            drift_types_detected=[DriftType.DATA_DRIFT],
            feature_drift={},
            drifted_features=[],
            configuration=DriftConfiguration(),
            detection_start_time=datetime.utcnow(),
            detection_end_time=datetime.utcnow(),
        )
        assert report.requires_immediate_attention() is True

        # Low severity report
        report.overall_drift_severity = DriftSeverity.LOW
        assert report.requires_immediate_attention() is False

        # Concept drift detected
        report.concept_drift_detected = True
        assert report.requires_immediate_attention() is True

    def test_get_recommended_actions(self):
        """Test getting recommended actions."""
        report = DriftReport(
            model_id=uuid4(),
            reference_sample_size=1000,
            current_sample_size=1000,
            overall_drift_detected=True,
            overall_drift_severity=DriftSeverity.CRITICAL,
            drift_types_detected=[DriftType.DATA_DRIFT],
            feature_drift={},
            drifted_features=[],
            configuration=DriftConfiguration(),
            detection_start_time=datetime.utcnow(),
            detection_end_time=datetime.utcnow(),
            concept_drift_detected=True,
            multivariate_drift_detected=True,
        )

        actions = report.get_recommended_actions()

        assert any("URGENT" in action for action in actions)
        assert any("target variable" in action.lower() for action in actions)
        assert any("feature interactions" in action.lower() for action in actions)


class TestDriftMonitor:
    """Test cases for DriftMonitor dataclass."""

    def test_create_drift_monitor(self):
        """Test creating drift monitor."""
        model_id = uuid4()
        config = DriftConfiguration()

        monitor = DriftMonitor(
            model_id=model_id,
            name="Production Model Monitor",
            configuration=config,
            created_by="admin",
            description="Monitor for production model drift",
            monitoring_frequency="hourly",
        )

        assert monitor.model_id == model_id
        assert monitor.name == "Production Model Monitor"
        assert monitor.monitoring_enabled is True
        assert monitor.monitoring_frequency == "hourly"
        assert monitor.consecutive_drift_detections == 0

    def test_monitor_frequency_validation(self):
        """Test monitor frequency validation."""
        model_id = uuid4()
        config = DriftConfiguration()

        # Valid frequencies
        for freq in ["hourly", "daily", "weekly"]:
            DriftMonitor(
                model_id=model_id,
                name="Test Monitor",
                configuration=config,
                created_by="user",
                monitoring_frequency=freq,
            )

        # Invalid frequency
        with pytest.raises(ValueError, match="Monitoring frequency must be one of"):
            DriftMonitor(
                model_id=model_id,
                name="Test Monitor",
                configuration=config,
                created_by="user",
                monitoring_frequency="invalid",
            )

    def test_should_check_now(self):
        """Test checking if drift detection should be performed."""
        monitor = DriftMonitor(
            model_id=uuid4(),
            name="Test Monitor",
            configuration=DriftConfiguration(),
            created_by="user",
        )

        # Monitoring disabled
        monitor.monitoring_enabled = False
        assert monitor.should_check_now() is False

        # Monitoring enabled, no next check time
        monitor.monitoring_enabled = True
        monitor.next_check_time = None
        assert monitor.should_check_now() is True

        # Next check time in future
        monitor.next_check_time = datetime.utcnow() + timedelta(hours=1)
        assert monitor.should_check_now() is False

        # Next check time in past
        monitor.next_check_time = datetime.utcnow() - timedelta(hours=1)
        assert monitor.should_check_now() is True

    def test_record_drift_detection(self):
        """Test recording drift detection results."""
        monitor = DriftMonitor(
            model_id=uuid4(),
            name="Test Monitor",
            configuration=DriftConfiguration(),
            created_by="user",
        )

        report_id = uuid4()

        # Record drift detection
        monitor.record_drift_detection(DriftSeverity.HIGH, report_id)

        assert monitor.consecutive_drift_detections == 1
        assert monitor.last_drift_detection is not None
        assert monitor.current_drift_severity == DriftSeverity.HIGH
        assert report_id in monitor.recent_reports

        # Record no drift
        monitor.record_drift_detection(DriftSeverity.NONE, uuid4())

        assert monitor.consecutive_drift_detections == 0
        assert monitor.current_drift_severity == DriftSeverity.NONE

    def test_recent_reports_limit(self):
        """Test recent reports list limit."""
        monitor = DriftMonitor(
            model_id=uuid4(),
            name="Test Monitor",
            configuration=DriftConfiguration(),
            created_by="user",
        )

        # Add 15 reports (more than limit of 10)
        for i in range(15):
            monitor.record_drift_detection(DriftSeverity.LOW, uuid4())

        # Should only keep last 10
        assert len(monitor.recent_reports) == 10

    def test_needs_alert(self):
        """Test checking if alert should be sent."""
        config = DriftConfiguration()
        config.alert_severity_threshold = DriftSeverity.MEDIUM

        monitor = DriftMonitor(
            model_id=uuid4(),
            name="Test Monitor",
            configuration=config,
            created_by="user",
        )

        # Alert disabled
        monitor.alert_enabled = False
        assert monitor.needs_alert(DriftSeverity.HIGH) is False

        # Alert enabled, severity below threshold
        monitor.alert_enabled = True
        assert monitor.needs_alert(DriftSeverity.LOW) is False

        # Alert enabled, severity meets threshold
        assert monitor.needs_alert(DriftSeverity.MEDIUM) is True
        assert monitor.needs_alert(DriftSeverity.HIGH) is True
        assert monitor.needs_alert(DriftSeverity.CRITICAL) is True
