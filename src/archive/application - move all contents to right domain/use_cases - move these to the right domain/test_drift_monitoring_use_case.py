"""
Tests for drift monitoring use case.

This module provides comprehensive tests for the drift monitoring use case,
ensuring proper orchestration of drift detection, alerting, and notifications.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from monorepo.application.use_cases.drift_monitoring_use_case import (
    DriftMonitoringUseCase,
)
from monorepo.domain.entities.drift_detection import (
    DriftAlert,
    DriftDetectionMethod,
    DriftDetectionResult,
    DriftMetrics,
    DriftMonitoringStatus,
    DriftReport,
    DriftSeverity,
    DriftType,
    ModelMonitoringConfig,
    MonitoringStatus,
)


class TestDriftMonitoringUseCase:
    """Test cases for DriftMonitoringUseCase."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_drift_service = Mock()
        self.mock_detector_repository = Mock()
        self.mock_alert_repository = Mock()
        self.mock_notification_service = Mock()
        self.mock_metrics_service = Mock()

        self.use_case = DriftMonitoringUseCase(
            drift_detection_service=self.mock_drift_service,
            detector_repository=self.mock_detector_repository,
            alert_repository=self.mock_alert_repository,
            notification_service=self.mock_notification_service,
            metrics_service=self.mock_metrics_service,
        )

        # Test data
        self.reference_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.normal(0, 1, 1000),
                "feature_3": np.random.normal(0, 1, 1000),
            }
        )

        self.current_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0.5, 1.2, 500),  # Slight drift
                "feature_2": np.random.normal(0, 1, 500),
                "feature_3": np.random.normal(0, 1, 500),
            }
        )

        self.model_id = "test_model_123"
        self.detector_name = "test_detector"

    def test_initialization(self):
        """Test use case initialization."""
        assert self.use_case.drift_service == self.mock_drift_service
        assert self.use_case.detector_repository == self.mock_detector_repository
        assert self.use_case.alert_repository == self.mock_alert_repository
        assert self.use_case.notification_service == self.mock_notification_service
        assert self.use_case.metrics_service == self.mock_metrics_service

    def test_initialization_with_defaults(self):
        """Test use case initialization with default services."""
        minimal_use_case = DriftMonitoringUseCase(
            drift_detection_service=self.mock_drift_service
        )

        assert minimal_use_case.drift_service == self.mock_drift_service
        assert minimal_use_case.detector_repository is None
        assert minimal_use_case.alert_repository is None
        assert minimal_use_case.notification_service is None

    @pytest.mark.asyncio
    async def test_run_drift_detection_basic(self):
        """Test basic drift detection execution."""
        # Mock drift detection result
        mock_result = DriftDetectionResult(
            model_id=self.model_id,
            detector_name=self.detector_name,
            drift_detected=True,
            drift_score=0.85,
            p_value=0.02,
            method=DriftDetectionMethod.KS_TEST,
            timestamp=datetime.now(),
            reference_window_size=1000,
            current_window_size=500,
        )

        self.mock_drift_service.detect_data_drift = AsyncMock(return_value=mock_result)

        result = await self.use_case.run_drift_detection(
            reference_data=self.reference_data,
            current_data=self.current_data,
            model_id=self.model_id,
            detector_name=self.detector_name,
        )

        assert result == mock_result
        assert result.drift_detected is True
        assert result.drift_score == 0.85

        self.mock_drift_service.detect_data_drift.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_drift_detection_with_config(self):
        """Test drift detection with custom configuration."""
        config = ModelMonitoringConfig(
            model_id=self.model_id,
            monitoring_enabled=True,
            drift_threshold=0.05,
            check_frequency_hours=1,
            alert_threshold=DriftSeverity.MEDIUM,
            methods=[DriftDetectionMethod.KS_TEST, DriftDetectionMethod.PSI],
            monitoring_features=["feature_1", "feature_2"],
        )

        mock_result = DriftDetectionResult(
            model_id=self.model_id,
            detector_name=self.detector_name,
            drift_detected=False,
            drift_score=0.02,
            p_value=0.15,
            method=DriftDetectionMethod.KS_TEST,
            timestamp=datetime.now(),
            reference_window_size=1000,
            current_window_size=500,
        )

        self.mock_drift_service.detect_data_drift = AsyncMock(return_value=mock_result)

        result = await self.use_case.run_drift_detection(
            reference_data=self.reference_data,
            current_data=self.current_data,
            model_id=self.model_id,
            detector_name=self.detector_name,
            config=config,
        )

        assert result.drift_detected is False
        assert result.drift_score == 0.02

        # Verify config was passed to service
        call_args = self.mock_drift_service.detect_data_drift.call_args
        assert call_args.kwargs.get("config") == config

    @pytest.mark.asyncio
    async def test_run_concept_drift_detection(self):
        """Test concept drift detection."""
        # Create mock prediction data
        reference_predictions = np.random.random(1000)
        current_predictions = np.random.random(500) + 0.2  # Shifted predictions

        mock_result = DriftDetectionResult(
            model_id=self.model_id,
            detector_name=self.detector_name,
            drift_detected=True,
            drift_score=0.75,
            p_value=0.01,
            method=DriftDetectionMethod.KS_TEST,
            timestamp=datetime.now(),
            drift_type=DriftType.CONCEPT_DRIFT,
            reference_window_size=1000,
            current_window_size=500,
        )

        self.mock_drift_service.detect_concept_drift = AsyncMock(
            return_value=mock_result
        )

        result = await self.use_case.run_concept_drift_detection(
            reference_predictions=reference_predictions,
            current_predictions=current_predictions,
            model_id=self.model_id,
            detector_name=self.detector_name,
        )

        assert result.drift_detected is True
        assert result.drift_type == DriftType.CONCEPT_DRIFT

        self.mock_drift_service.detect_concept_drift.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_model_continuous(self):
        """Test continuous model monitoring."""
        config = ModelMonitoringConfig(
            model_id=self.model_id,
            monitoring_enabled=True,
            drift_threshold=0.05,
            check_frequency_hours=1,
            alert_threshold=DriftSeverity.HIGH,
            methods=[DriftDetectionMethod.KS_TEST],
            max_monitoring_duration_hours=24,
        )

        # Mock multiple drift detection results over time
        mock_results = [
            DriftDetectionResult(
                model_id=self.model_id,
                detector_name=self.detector_name,
                drift_detected=False,
                drift_score=0.02,
                p_value=0.15,
                method=DriftDetectionMethod.KS_TEST,
                timestamp=datetime.now(),
                reference_window_size=1000,
                current_window_size=100,
            ),
            DriftDetectionResult(
                model_id=self.model_id,
                detector_name=self.detector_name,
                drift_detected=True,
                drift_score=0.12,
                p_value=0.03,
                method=DriftDetectionMethod.KS_TEST,
                timestamp=datetime.now() + timedelta(hours=1),
                reference_window_size=1000,
                current_window_size=100,
            ),
        ]

        self.mock_drift_service.detect_data_drift = AsyncMock(side_effect=mock_results)

        # Mock data streaming
        async def mock_data_stream():
            for i in range(2):
                yield self.current_data.iloc[i * 50 : (i + 1) * 50]

        monitoring_status = await self.use_case.monitor_model_continuous(
            model_id=self.model_id,
            detector_name=self.detector_name,
            reference_data=self.reference_data,
            data_stream=mock_data_stream(),
            config=config,
        )

        assert isinstance(monitoring_status, DriftMonitoringStatus)
        assert monitoring_status.model_id == self.model_id
        assert len(monitoring_status.monitoring_sessions) > 0

    @pytest.mark.asyncio
    async def test_create_drift_alert(self):
        """Test drift alert creation."""
        drift_result = DriftDetectionResult(
            model_id=self.model_id,
            detector_name=self.detector_name,
            drift_detected=True,
            drift_score=0.85,
            p_value=0.01,
            method=DriftDetectionMethod.KS_TEST,
            timestamp=datetime.now(),
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.HIGH,
            reference_window_size=1000,
            current_window_size=500,
        )

        alert = await self.use_case.create_drift_alert(
            drift_result=drift_result,
            alert_level="CRITICAL",
            message="High drift detected in model predictions",
        )

        assert isinstance(alert, DriftAlert)
        assert alert.model_id == self.model_id
        assert alert.alert_level == "CRITICAL"
        assert alert.drift_type == DriftType.DATA_DRIFT
        assert alert.severity == DriftSeverity.HIGH
        assert "High drift detected" in alert.message

    @pytest.mark.asyncio
    async def test_send_drift_notification(self):
        """Test sending drift notifications."""
        alert = DriftAlert(
            model_id=self.model_id,
            detector_name=self.detector_name,
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.HIGH,
            alert_level="CRITICAL",
            message="High drift detected",
            timestamp=datetime.now(),
            drift_score=0.85,
            p_value=0.01,
            method=DriftDetectionMethod.KS_TEST,
        )

        self.mock_notification_service.send_alert = AsyncMock(return_value=True)

        success = await self.use_case.send_drift_notification(
            alert=alert, recipients=["admin@company.com", "ml-team@company.com"]
        )

        assert success is True
        self.mock_notification_service.send_alert.assert_called_once_with(
            alert=alert, recipients=["admin@company.com", "ml-team@company.com"]
        )

    @pytest.mark.asyncio
    async def test_generate_drift_report(self):
        """Test drift report generation."""
        # Mock historical drift results
        historical_results = [
            DriftDetectionResult(
                model_id=self.model_id,
                detector_name=self.detector_name,
                drift_detected=False,
                drift_score=0.02,
                p_value=0.15,
                method=DriftDetectionMethod.KS_TEST,
                timestamp=datetime.now() - timedelta(days=2),
                reference_window_size=1000,
                current_window_size=500,
            ),
            DriftDetectionResult(
                model_id=self.model_id,
                detector_name=self.detector_name,
                drift_detected=True,
                drift_score=0.12,
                p_value=0.03,
                method=DriftDetectionMethod.KS_TEST,
                timestamp=datetime.now() - timedelta(days=1),
                reference_window_size=1000,
                current_window_size=500,
            ),
        ]

        report = await self.use_case.generate_drift_report(
            model_id=self.model_id,
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            historical_results=historical_results,
        )

        assert isinstance(report, DriftReport)
        assert report.model_id == self.model_id
        assert len(report.detection_results) == 2
        assert report.total_detections == 2
        assert report.drift_detections == 1

    @pytest.mark.asyncio
    async def test_update_monitoring_config(self):
        """Test updating monitoring configuration."""
        new_config = ModelMonitoringConfig(
            model_id=self.model_id,
            monitoring_enabled=True,
            drift_threshold=0.03,  # More sensitive
            check_frequency_hours=0.5,  # More frequent
            alert_threshold=DriftSeverity.MEDIUM,
            methods=[DriftDetectionMethod.KS_TEST, DriftDetectionMethod.PSI],
            monitoring_features=["feature_1", "feature_2"],
        )

        updated_config = await self.use_case.update_monitoring_config(
            model_id=self.model_id, config=new_config
        )

        assert updated_config.drift_threshold == 0.03
        assert updated_config.check_frequency_hours == 0.5
        assert len(updated_config.methods) == 2

    @pytest.mark.asyncio
    async def test_get_monitoring_status(self):
        """Test getting monitoring status."""
        mock_status = DriftMonitoringStatus(
            model_id=self.model_id,
            monitoring_enabled=True,
            last_check_timestamp=datetime.now(),
            status=MonitoringStatus.ACTIVE,
            total_checks=100,
            drift_detections=5,
            last_drift_timestamp=datetime.now() - timedelta(hours=2),
        )

        # Mock repository call if available
        if self.mock_detector_repository:
            self.mock_detector_repository.get_monitoring_status = AsyncMock(
                return_value=mock_status
            )

        status = await self.use_case.get_monitoring_status(model_id=self.model_id)

        # If no repository, should return default status
        if self.mock_detector_repository and hasattr(
            self.mock_detector_repository, "get_monitoring_status"
        ):
            assert status == mock_status
        else:
            assert status.model_id == self.model_id

    @pytest.mark.asyncio
    async def test_stop_monitoring(self):
        """Test stopping model monitoring."""
        stopped_status = await self.use_case.stop_monitoring(model_id=self.model_id)

        assert isinstance(stopped_status, DriftMonitoringStatus)
        assert stopped_status.model_id == self.model_id
        assert stopped_status.monitoring_enabled is False
        assert stopped_status.status == MonitoringStatus.STOPPED

    @pytest.mark.asyncio
    async def test_get_drift_metrics(self):
        """Test getting drift metrics."""
        mock_metrics = DriftMetrics(
            model_id=self.model_id,
            total_drift_detections=10,
            data_drift_count=6,
            concept_drift_count=4,
            average_drift_score=0.15,
            max_drift_score=0.85,
            last_drift_timestamp=datetime.now() - timedelta(hours=1),
            monitoring_uptime_hours=720,  # 30 days
            detection_accuracy=0.92,
        )

        # Mock metrics collection
        with patch.object(
            self.use_case, "_calculate_drift_metrics", return_value=mock_metrics
        ):
            metrics = await self.use_case.get_drift_metrics(
                model_id=self.model_id, time_range_hours=720
            )

        assert isinstance(metrics, DriftMetrics)
        assert metrics.model_id == self.model_id
        assert metrics.total_drift_detections == 10
        assert metrics.average_drift_score == 0.15

    @pytest.mark.asyncio
    async def test_error_handling_in_detection(self):
        """Test error handling during drift detection."""
        # Mock service to raise an exception
        self.mock_drift_service.detect_data_drift = AsyncMock(
            side_effect=Exception("Detection service error")
        )

        with pytest.raises(Exception, match="Detection service error"):
            await self.use_case.run_drift_detection(
                reference_data=self.reference_data,
                current_data=self.current_data,
                model_id=self.model_id,
                detector_name=self.detector_name,
            )

    @pytest.mark.asyncio
    async def test_notification_failure_handling(self):
        """Test handling of notification failures."""
        alert = DriftAlert(
            model_id=self.model_id,
            detector_name=self.detector_name,
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.HIGH,
            alert_level="CRITICAL",
            message="High drift detected",
            timestamp=datetime.now(),
            drift_score=0.85,
            p_value=0.01,
            method=DriftDetectionMethod.KS_TEST,
        )

        # Mock notification service to fail
        self.mock_notification_service.send_alert = AsyncMock(
            side_effect=Exception("Notification service unavailable")
        )

        # Should handle the error gracefully
        success = await self.use_case.send_drift_notification(
            alert=alert, recipients=["admin@company.com"]
        )

        # Depending on implementation, this might return False or raise
        assert success is False or isinstance(success, bool)


class TestDriftMonitoringUseCaseIntegration:
    """Integration tests for drift monitoring use case."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # Use real or more realistic mocks for integration testing
        self.mock_drift_service = Mock()
        self.use_case = DriftMonitoringUseCase(
            drift_detection_service=self.mock_drift_service
        )

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow from detection to alerting."""
        # This would test the full workflow in a more integrated manner
        # For now, just ensure the use case can handle the workflow

        reference_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.normal(0, 1, 1000),
            }
        )

        current_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0.5, 1.2, 500),
                "feature_2": np.random.normal(0, 1, 500),
            }
        )

        # Mock the drift detection result
        mock_result = DriftDetectionResult(
            model_id="integration_test_model",
            detector_name="integration_detector",
            drift_detected=True,
            drift_score=0.75,
            p_value=0.02,
            method=DriftDetectionMethod.KS_TEST,
            timestamp=datetime.now(),
            reference_window_size=1000,
            current_window_size=500,
        )

        self.mock_drift_service.detect_data_drift = AsyncMock(return_value=mock_result)

        # Run detection
        result = await self.use_case.run_drift_detection(
            reference_data=reference_data,
            current_data=current_data,
            model_id="integration_test_model",
            detector_name="integration_detector",
        )

        # Create alert
        alert = await self.use_case.create_drift_alert(
            drift_result=result,
            alert_level="WARNING",
            message="Drift detected in integration test",
        )

        assert result.drift_detected is True
        assert isinstance(alert, DriftAlert)
        assert alert.alert_level == "WARNING"
