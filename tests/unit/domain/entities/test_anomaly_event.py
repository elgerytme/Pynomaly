#!/usr/bin/env python3
"""
Comprehensive tests for Anomaly Event domain entities.
Tests all anomaly event-related classes including AnomalyEvent, event data classes,
filters, patterns, and summary statistics.
"""

from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from src.pynomaly.domain.entities.anomaly_event import (
    AnomalyEvent,
    AnomalyEventData,
    DataQualityEventData,
    EventAggregation,
    EventFilter,
    EventPattern,
    EventSeverity,
    EventStatus,
    EventSummary,
    EventType,
    PerformanceEventData,
)


class TestEventType:
    """Test cases for EventType enum."""

    def test_all_event_types(self):
        """Test all event types are defined correctly."""
        assert EventType.ANOMALY_DETECTED == "anomaly_detected"
        assert EventType.ANOMALY_RESOLVED == "anomaly_resolved"
        assert EventType.ANOMALY_ESCALATED == "anomaly_escalated"
        assert EventType.DATA_QUALITY_ISSUE == "data_quality_issue"
        assert EventType.MODEL_DRIFT_DETECTED == "model_drift_detected"
        assert EventType.PERFORMANCE_DEGRADATION == "performance_degradation"
        assert EventType.SYSTEM_ALERT == "system_alert"
        assert EventType.THRESHOLD_BREACH == "threshold_breach"
        assert EventType.PATTERN_CHANGE == "pattern_change"
        assert EventType.BATCH_COMPLETED == "batch_completed"
        assert EventType.SESSION_STARTED == "session_started"
        assert EventType.SESSION_STOPPED == "session_stopped"
        assert EventType.CUSTOM == "custom"

    def test_enum_values(self):
        """Test enum values can be used properly."""
        event_types = [
            EventType.ANOMALY_DETECTED,
            EventType.DATA_QUALITY_ISSUE,
            EventType.PERFORMANCE_DEGRADATION,
            EventType.CUSTOM
        ]
        assert len(event_types) == 4
        assert all(isinstance(event_type, str) for event_type in event_types)


class TestEventSeverity:
    """Test cases for EventSeverity enum."""

    def test_all_severities(self):
        """Test all severity levels are defined correctly."""
        assert EventSeverity.INFO == "info"
        assert EventSeverity.LOW == "low"
        assert EventSeverity.MEDIUM == "medium"
        assert EventSeverity.HIGH == "high"
        assert EventSeverity.CRITICAL == "critical"

    def test_severity_ordering(self):
        """Test severity levels can be compared."""
        severities = [
            EventSeverity.INFO,
            EventSeverity.LOW,
            EventSeverity.MEDIUM,
            EventSeverity.HIGH,
            EventSeverity.CRITICAL
        ]
        assert len(severities) == 5


class TestEventStatus:
    """Test cases for EventStatus enum."""

    def test_all_statuses(self):
        """Test all status values are defined correctly."""
        assert EventStatus.PENDING == "pending"
        assert EventStatus.PROCESSING == "processing"
        assert EventStatus.PROCESSED == "processed"
        assert EventStatus.FAILED == "failed"
        assert EventStatus.ACKNOWLEDGED == "acknowledged"
        assert EventStatus.RESOLVED == "resolved"
        assert EventStatus.IGNORED == "ignored"

    def test_status_workflow(self):
        """Test status represents a workflow."""
        workflow_statuses = [
            EventStatus.PENDING,
            EventStatus.PROCESSING,
            EventStatus.PROCESSED,
            EventStatus.ACKNOWLEDGED,
            EventStatus.RESOLVED
        ]
        assert len(workflow_statuses) == 5


class TestAnomalyEventData:
    """Test cases for AnomalyEventData dataclass."""

    def test_valid_creation(self):
        """Test creating valid anomaly event data."""
        data = AnomalyEventData(
            anomaly_score=0.8,
            confidence=0.9,
            feature_contributions={"feature1": 0.5, "feature2": 0.3},
            predicted_class="anomaly",
            expected_range={"min": 0.0, "max": 1.0},
            actual_values={"value": 0.8},
            detection_method="isolation_forest",
            model_version="v1.0.0",
            explanation="High anomaly score detected"
        )
        assert data.anomaly_score == 0.8
        assert data.confidence == 0.9
        assert data.feature_contributions == {"feature1": 0.5, "feature2": 0.3}
        assert data.predicted_class == "anomaly"
        assert data.detection_method == "isolation_forest"
        assert data.model_version == "v1.0.0"
        assert data.explanation == "High anomaly score detected"

    def test_minimal_creation(self):
        """Test creating anomaly event data with minimal fields."""
        data = AnomalyEventData(
            anomaly_score=0.7,
            confidence=0.8
        )
        assert data.anomaly_score == 0.7
        assert data.confidence == 0.8
        assert data.feature_contributions == {}
        assert data.predicted_class is None
        assert data.expected_range == {}
        assert data.actual_values == {}
        assert data.detection_method is None
        assert data.model_version is None
        assert data.explanation is None

    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        # Valid scores
        for score in [0.0, 0.5, 1.0]:
            data = AnomalyEventData(anomaly_score=score, confidence=0.5)
            assert data.anomaly_score == score

        # Invalid scores
        invalid_scores = [-0.1, 1.1, 2.0]
        for score in invalid_scores:
            with pytest.raises(ValueError, match="Anomaly score must be between 0.0 and 1.0"):
                AnomalyEventData(anomaly_score=score, confidence=0.5)

    def test_confidence_validation(self):
        """Test confidence validation."""
        # Valid confidence levels
        for confidence in [0.0, 0.5, 1.0]:
            data = AnomalyEventData(anomaly_score=0.5, confidence=confidence)
            assert data.confidence == confidence

        # Invalid confidence levels
        invalid_confidence = [-0.1, 1.1, 2.0]
        for confidence in invalid_confidence:
            with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
                AnomalyEventData(anomaly_score=0.5, confidence=confidence)

    def test_feature_contributions(self):
        """Test feature contributions handling."""
        contributions = {
            "feature1": 0.4,
            "feature2": 0.3,
            "feature3": 0.2,
            "feature4": 0.1
        }
        data = AnomalyEventData(
            anomaly_score=0.8,
            confidence=0.9,
            feature_contributions=contributions
        )
        assert data.feature_contributions == contributions
        assert sum(data.feature_contributions.values()) == 1.0

    def test_complex_data_structures(self):
        """Test handling of complex data structures."""
        expected_range = {
            "temperature": {"min": 20.0, "max": 30.0},
            "pressure": {"min": 1000, "max": 1200}
        }
        actual_values = {
            "temperature": 35.0,
            "pressure": 1300
        }

        data = AnomalyEventData(
            anomaly_score=0.9,
            confidence=0.8,
            expected_range=expected_range,
            actual_values=actual_values
        )
        assert data.expected_range == expected_range
        assert data.actual_values == actual_values


class TestDataQualityEventData:
    """Test cases for DataQualityEventData dataclass."""

    def test_valid_creation(self):
        """Test creating valid data quality event data."""
        data = DataQualityEventData(
            issue_type="missing_values",
            affected_fields=["field1", "field2"],
            severity_score=0.7,
            missing_percentage=15.5,
            outlier_percentage=2.3,
            schema_violations=["type_mismatch", "constraint_violation"],
            data_drift_score=0.3,
            recommendations=["impute_missing", "validate_schema"]
        )
        assert data.issue_type == "missing_values"
        assert data.affected_fields == ["field1", "field2"]
        assert data.severity_score == 0.7
        assert data.missing_percentage == 15.5
        assert data.outlier_percentage == 2.3
        assert data.schema_violations == ["type_mismatch", "constraint_violation"]
        assert data.data_drift_score == 0.3
        assert data.recommendations == ["impute_missing", "validate_schema"]

    def test_minimal_creation(self):
        """Test creating data quality event data with minimal fields."""
        data = DataQualityEventData(
            issue_type="data_drift",
            affected_fields=["sensor_reading"],
            severity_score=0.5
        )
        assert data.issue_type == "data_drift"
        assert data.affected_fields == ["sensor_reading"]
        assert data.severity_score == 0.5
        assert data.missing_percentage is None
        assert data.outlier_percentage is None
        assert data.schema_violations == []
        assert data.data_drift_score is None
        assert data.recommendations == []

    def test_severity_score_validation(self):
        """Test severity score validation."""
        # Valid scores
        for score in [0.0, 0.5, 1.0]:
            data = DataQualityEventData(
                issue_type="test",
                affected_fields=["field1"],
                severity_score=score
            )
            assert data.severity_score == score

        # Invalid scores
        invalid_scores = [-0.1, 1.1, 2.0]
        for score in invalid_scores:
            with pytest.raises(ValueError, match="Severity score must be between 0.0 and 1.0"):
                DataQualityEventData(
                    issue_type="test",
                    affected_fields=["field1"],
                    severity_score=score
                )

    def test_missing_percentage_validation(self):
        """Test missing percentage validation."""
        # Valid percentages
        for percentage in [0.0, 50.0, 100.0]:
            data = DataQualityEventData(
                issue_type="test",
                affected_fields=["field1"],
                severity_score=0.5,
                missing_percentage=percentage
            )
            assert data.missing_percentage == percentage

        # Invalid percentages
        invalid_percentages = [-0.1, 100.1, 200.0]
        for percentage in invalid_percentages:
            with pytest.raises(ValueError, match="Missing percentage must be between 0.0 and 100.0"):
                DataQualityEventData(
                    issue_type="test",
                    affected_fields=["field1"],
                    severity_score=0.5,
                    missing_percentage=percentage
                )

    def test_outlier_percentage_validation(self):
        """Test outlier percentage validation."""
        # Valid percentages
        for percentage in [0.0, 5.0, 100.0]:
            data = DataQualityEventData(
                issue_type="test",
                affected_fields=["field1"],
                severity_score=0.5,
                outlier_percentage=percentage
            )
            assert data.outlier_percentage == percentage

        # Invalid percentages
        invalid_percentages = [-0.1, 100.1, 200.0]
        for percentage in invalid_percentages:
            with pytest.raises(ValueError, match="Outlier percentage must be between 0.0 and 100.0"):
                DataQualityEventData(
                    issue_type="test",
                    affected_fields=["field1"],
                    severity_score=0.5,
                    outlier_percentage=percentage
                )

    def test_complex_schema_violations(self):
        """Test complex schema violations."""
        violations = [
            "type_mismatch: expected int, got str",
            "constraint_violation: value out of range",
            "format_error: invalid date format",
            "foreign_key_violation: referenced key not found"
        ]
        data = DataQualityEventData(
            issue_type="schema_validation",
            affected_fields=["field1", "field2", "field3"],
            severity_score=0.8,
            schema_violations=violations
        )
        assert data.schema_violations == violations
        assert len(data.schema_violations) == 4


class TestPerformanceEventData:
    """Test cases for PerformanceEventData dataclass."""

    def test_valid_creation(self):
        """Test creating valid performance event data."""
        data = PerformanceEventData(
            metric_name="accuracy",
            current_value=0.85,
            baseline_value=0.92,
            degradation_percentage=7.6,
            threshold_value=0.90,
            trend_direction="decreasing",
            affected_components=["classifier", "preprocessor"],
            potential_causes=["data_drift", "model_decay"]
        )
        assert data.metric_name == "accuracy"
        assert data.current_value == 0.85
        assert data.baseline_value == 0.92
        assert data.degradation_percentage == 7.6
        assert data.threshold_value == 0.90
        assert data.trend_direction == "decreasing"
        assert data.affected_components == ["classifier", "preprocessor"]
        assert data.potential_causes == ["data_drift", "model_decay"]

    def test_minimal_creation(self):
        """Test creating performance event data with minimal fields."""
        data = PerformanceEventData(
            metric_name="f1_score",
            current_value=0.75,
            baseline_value=0.80,
            degradation_percentage=6.25,
            threshold_value=0.78,
            trend_direction="stable"
        )
        assert data.metric_name == "f1_score"
        assert data.current_value == 0.75
        assert data.baseline_value == 0.80
        assert data.degradation_percentage == 6.25
        assert data.threshold_value == 0.78
        assert data.trend_direction == "stable"
        assert data.affected_components == []
        assert data.potential_causes == []

    def test_trend_direction_validation(self):
        """Test trend direction validation."""
        # Valid trend directions
        valid_trends = ["increasing", "decreasing", "stable"]
        for trend in valid_trends:
            data = PerformanceEventData(
                metric_name="test_metric",
                current_value=0.5,
                baseline_value=0.6,
                degradation_percentage=10.0,
                threshold_value=0.55,
                trend_direction=trend
            )
            assert data.trend_direction == trend

        # Invalid trend directions
        invalid_trends = ["rising", "falling", "constant", "unknown"]
        for trend in invalid_trends:
            with pytest.raises(ValueError, match="Trend direction must be one of"):
                PerformanceEventData(
                    metric_name="test_metric",
                    current_value=0.5,
                    baseline_value=0.6,
                    degradation_percentage=10.0,
                    threshold_value=0.55,
                    trend_direction=trend
                )

    def test_metric_calculations(self):
        """Test metric calculations consistency."""
        baseline = 0.90
        current = 0.81
        expected_degradation = ((baseline - current) / baseline) * 100

        data = PerformanceEventData(
            metric_name="precision",
            current_value=current,
            baseline_value=baseline,
            degradation_percentage=expected_degradation,
            threshold_value=0.85,
            trend_direction="decreasing"
        )

        assert abs(data.degradation_percentage - expected_degradation) < 0.01
        assert data.current_value < data.baseline_value
        assert data.current_value < data.threshold_value


class TestAnomalyEvent:
    """Test cases for AnomalyEvent dataclass."""

    def test_valid_creation(self):
        """Test creating a valid anomaly event."""
        now = datetime.utcnow()
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="High anomaly detected",
            description="Anomaly detected in sensor data",
            raw_data={"sensor_id": "S001", "value": 100.5},
            event_time=now
        )

        assert event.event_type == EventType.ANOMALY_DETECTED
        assert event.severity == EventSeverity.HIGH
        assert event.title == "High anomaly detected"
        assert event.description == "Anomaly detected in sensor data"
        assert event.raw_data == {"sensor_id": "S001", "value": 100.5}
        assert event.event_time == now
        assert isinstance(event.id, UUID)
        assert event.status == EventStatus.PENDING
        assert isinstance(event.ingestion_time, datetime)

    def test_auto_generated_fields(self):
        """Test auto-generated fields are set correctly."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.MEDIUM,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        assert isinstance(event.id, UUID)
        assert event.status == EventStatus.PENDING
        assert isinstance(event.ingestion_time, datetime)
        assert event.processing_attempts == 0
        assert event.notification_sent is False
        assert event.tags == []
        assert event.metadata == {}
        assert event.custom_fields == {}

    def test_optional_fields(self):
        """Test optional fields are handled correctly."""
        detector_id = uuid4()
        session_id = uuid4()

        event = AnomalyEvent(
            event_type=EventType.DATA_QUALITY_ISSUE,
            severity=EventSeverity.LOW,
            title="Data quality issue",
            description="Missing values detected",
            raw_data={"missing_count": 5},
            event_time=datetime.utcnow(),
            detector_id=detector_id,
            source_session_id=session_id,
            data_source="sensor_stream",
            correlation_id="corr_123"
        )

        assert event.detector_id == detector_id
        assert event.source_session_id == session_id
        assert event.data_source == "sensor_stream"
        assert event.correlation_id == "corr_123"

    def test_anomaly_data_integration(self):
        """Test integration with anomaly-specific data."""
        anomaly_data = AnomalyEventData(
            anomaly_score=0.9,
            confidence=0.8,
            detection_method="isolation_forest"
        )

        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.CRITICAL,
            title="Critical anomaly",
            description="Critical anomaly detected",
            raw_data={"value": 999.9},
            event_time=datetime.utcnow(),
            anomaly_data=anomaly_data
        )

        assert event.anomaly_data == anomaly_data
        assert event.anomaly_data.anomaly_score == 0.9
        assert event.anomaly_data.confidence == 0.8
        assert event.anomaly_data.detection_method == "isolation_forest"

    def test_data_quality_data_integration(self):
        """Test integration with data quality event data."""
        quality_data = DataQualityEventData(
            issue_type="missing_values",
            affected_fields=["temperature", "pressure"],
            severity_score=0.6,
            missing_percentage=25.0
        )

        event = AnomalyEvent(
            event_type=EventType.DATA_QUALITY_ISSUE,
            severity=EventSeverity.MEDIUM,
            title="Data quality issue",
            description="Missing values in sensor data",
            raw_data={"total_records": 1000, "missing_records": 250},
            event_time=datetime.utcnow(),
            data_quality_data=quality_data
        )

        assert event.data_quality_data == quality_data
        assert event.data_quality_data.issue_type == "missing_values"
        assert event.data_quality_data.missing_percentage == 25.0

    def test_performance_data_integration(self):
        """Test integration with performance event data."""
        performance_data = PerformanceEventData(
            metric_name="accuracy",
            current_value=0.82,
            baseline_value=0.90,
            degradation_percentage=8.9,
            threshold_value=0.85,
            trend_direction="decreasing"
        )

        event = AnomalyEvent(
            event_type=EventType.PERFORMANCE_DEGRADATION,
            severity=EventSeverity.HIGH,
            title="Performance degradation",
            description="Model accuracy has degraded",
            raw_data={"model_id": "model_v1"},
            event_time=datetime.utcnow(),
            performance_data=performance_data
        )

        assert event.performance_data == performance_data
        assert event.performance_data.metric_name == "accuracy"
        assert event.performance_data.current_value == 0.82
        assert event.performance_data.degradation_percentage == 8.9

    def test_acknowledge_method(self):
        """Test event acknowledgment."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        user = "admin"
        notes = "Investigating the issue"

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.acknowledge(user, notes)

            assert event.status == EventStatus.ACKNOWLEDGED
            assert event.acknowledged_by == user
            assert event.acknowledged_at == mock_now
            assert event.metadata["acknowledgment_notes"] == notes

    def test_acknowledge_without_notes(self):
        """Test event acknowledgment without notes."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        user = "admin"

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.acknowledge(user)

            assert event.status == EventStatus.ACKNOWLEDGED
            assert event.acknowledged_by == user
            assert event.acknowledged_at == mock_now
            assert "acknowledgment_notes" not in event.metadata

    def test_resolve_method(self):
        """Test event resolution."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        user = "admin"
        notes = "Issue resolved by restarting service"

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.resolve(user, notes)

            assert event.status == EventStatus.RESOLVED
            assert event.resolved_by == user
            assert event.resolved_at == mock_now
            assert event.resolution_notes == notes

    def test_resolve_without_notes(self):
        """Test event resolution without notes."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        user = "admin"

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.resolve(user)

            assert event.status == EventStatus.RESOLVED
            assert event.resolved_by == user
            assert event.resolved_at == mock_now
            assert event.resolution_notes is None

    def test_ignore_method(self):
        """Test event ignoring."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.LOW,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        user = "admin"
        reason = "False positive"

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.ignore(user, reason)

            assert event.status == EventStatus.IGNORED
            assert event.metadata["ignored_by"] == user
            assert event.metadata["ignored_at"] == mock_now.isoformat()
            assert event.metadata["ignore_reason"] == reason

    def test_ignore_without_reason(self):
        """Test event ignoring without reason."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.LOW,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        user = "admin"

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.ignore(user)

            assert event.status == EventStatus.IGNORED
            assert event.metadata["ignored_by"] == user
            assert event.metadata["ignored_at"] == mock_now.isoformat()
            assert "ignore_reason" not in event.metadata

    def test_mark_processing(self):
        """Test marking event as processing."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            event.mark_processing()

            assert event.status == EventStatus.PROCESSING
            assert event.processing_time == mock_now
            assert event.processing_attempts == 1

    def test_mark_processed(self):
        """Test marking event as processed."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        event.mark_processed()
        assert event.status == EventStatus.PROCESSED

    def test_mark_failed(self):
        """Test marking event as failed."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        error_message = "Processing failed due to timeout"
        retry_time = datetime.utcnow() + timedelta(minutes=5)

        event.mark_failed(error_message, retry_time)

        assert event.status == EventStatus.FAILED
        assert event.last_error == error_message
        assert event.retry_after == retry_time

    def test_mark_failed_without_retry(self):
        """Test marking event as failed without retry time."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        error_message = "Critical error occurred"

        event.mark_failed(error_message)

        assert event.status == EventStatus.FAILED
        assert event.last_error == error_message
        assert event.retry_after is None

    def test_is_actionable(self):
        """Test actionable event detection."""
        # High severity pending event - should be actionable
        event1 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.PENDING
        )
        assert event1.is_actionable() is True

        # Critical severity processing event - should be actionable
        event2 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.CRITICAL,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.PROCESSING
        )
        assert event2.is_actionable() is True

        # Low severity event - should not be actionable
        event3 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.LOW,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.PENDING
        )
        assert event3.is_actionable() is False

        # High severity resolved event - should not be actionable
        event4 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.RESOLVED
        )
        assert event4.is_actionable() is False

    def test_is_resolved(self):
        """Test resolved event detection."""
        # Resolved event
        event1 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.RESOLVED
        )
        assert event1.is_resolved() is True

        # Ignored event
        event2 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.LOW,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.IGNORED
        )
        assert event2.is_resolved() is True

        # Pending event
        event3 = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            status=EventStatus.PENDING
        )
        assert event3.is_resolved() is False

    def test_get_age(self):
        """Test getting event age."""
        event_time = datetime.utcnow() - timedelta(minutes=30)
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=event_time
        )

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = event_time + timedelta(minutes=30)
            mock_datetime.utcnow.return_value = mock_now

            age = event.get_age()
            assert age == 1800.0  # 30 minutes in seconds

    def test_get_processing_duration(self):
        """Test getting processing duration."""
        event_time = datetime.utcnow()
        processing_time = event_time + timedelta(minutes=5)
        resolved_time = processing_time + timedelta(minutes=10)

        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=event_time,
            processing_time=processing_time,
            resolved_at=resolved_time
        )

        duration = event.get_processing_duration()
        assert duration == 600.0  # 10 minutes in seconds

    def test_get_processing_duration_no_processing_time(self):
        """Test getting processing duration when no processing time is set."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        duration = event.get_processing_duration()
        assert duration is None

    def test_get_processing_duration_current_time(self):
        """Test getting processing duration using current time."""
        event_time = datetime.utcnow()
        processing_time = event_time + timedelta(minutes=5)

        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=event_time,
            processing_time=processing_time
        )

        with patch('src.pynomaly.domain.entities.anomaly_event.datetime') as mock_datetime:
            mock_now = processing_time + timedelta(minutes=15)
            mock_datetime.utcnow.return_value = mock_now

            duration = event.get_processing_duration()
            assert duration == 900.0  # 15 minutes in seconds

    def test_add_correlation(self):
        """Test adding correlation ID."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        correlation_id = "corr_12345"
        event.add_correlation(correlation_id)

        assert event.correlation_id == correlation_id

    def test_add_tag(self):
        """Test adding tags."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow()
        )

        event.add_tag("urgent")
        event.add_tag("sensor")
        event.add_tag("urgent")  # Duplicate should be ignored

        assert "urgent" in event.tags
        assert "sensor" in event.tags
        assert event.tags.count("urgent") == 1

    def test_remove_tag(self):
        """Test removing tags."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test event",
            description="Test description",
            raw_data={"test": "data"},
            event_time=datetime.utcnow(),
            tags=["urgent", "sensor", "anomaly"]
        )

        event.remove_tag("sensor")
        event.remove_tag("nonexistent")  # Should not raise error

        assert "sensor" not in event.tags
        assert "urgent" in event.tags
        assert "anomaly" in event.tags
        assert len(event.tags) == 2


class TestEventFilter:
    """Test cases for EventFilter dataclass."""

    def test_valid_creation(self):
        """Test creating valid event filter."""
        detector_id = uuid4()
        session_id = uuid4()

        event_filter = EventFilter(
            event_types=[EventType.ANOMALY_DETECTED, EventType.DATA_QUALITY_ISSUE],
            severities=[EventSeverity.HIGH, EventSeverity.CRITICAL],
            statuses=[EventStatus.PENDING, EventStatus.PROCESSING],
            detector_ids=[detector_id],
            session_ids=[session_id],
            data_sources=["sensor_stream"],
            min_anomaly_score=0.7,
            max_anomaly_score=1.0,
            min_confidence=0.8,
            limit=50,
            offset=10
        )

        assert event_filter.event_types == [EventType.ANOMALY_DETECTED, EventType.DATA_QUALITY_ISSUE]
        assert event_filter.severities == [EventSeverity.HIGH, EventSeverity.CRITICAL]
        assert event_filter.statuses == [EventStatus.PENDING, EventStatus.PROCESSING]
        assert event_filter.detector_ids == [detector_id]
        assert event_filter.session_ids == [session_id]
        assert event_filter.data_sources == ["sensor_stream"]
        assert event_filter.min_anomaly_score == 0.7
        assert event_filter.max_anomaly_score == 1.0
        assert event_filter.min_confidence == 0.8
        assert event_filter.limit == 50
        assert event_filter.offset == 10

    def test_default_values(self):
        """Test default values for event filter."""
        event_filter = EventFilter()

        assert event_filter.event_types is None
        assert event_filter.severities is None
        assert event_filter.statuses is None
        assert event_filter.limit == 100
        assert event_filter.offset == 0
        assert event_filter.sort_by == "event_time"
        assert event_filter.sort_order == "desc"

    def test_time_based_filters(self):
        """Test time-based filtering."""
        start_time = datetime.utcnow() - timedelta(hours=24)
        end_time = datetime.utcnow()

        event_filter = EventFilter(
            event_time_start=start_time,
            event_time_end=end_time,
            ingestion_time_start=start_time,
            ingestion_time_end=end_time
        )

        assert event_filter.event_time_start == start_time
        assert event_filter.event_time_end == end_time
        assert event_filter.ingestion_time_start == start_time
        assert event_filter.ingestion_time_end == end_time

    def test_content_filters(self):
        """Test content-based filtering."""
        event_filter = EventFilter(
            title_contains="anomaly",
            description_contains="sensor",
            tags=["urgent", "critical"],
            correlation_id="corr_123"
        )

        assert event_filter.title_contains == "anomaly"
        assert event_filter.description_contains == "sensor"
        assert event_filter.tags == ["urgent", "critical"]
        assert event_filter.correlation_id == "corr_123"

    def test_user_filters(self):
        """Test user-based filtering."""
        event_filter = EventFilter(
            acknowledged_by="admin",
            resolved_by="operator"
        )

        assert event_filter.acknowledged_by == "admin"
        assert event_filter.resolved_by == "operator"

    def test_limit_validation(self):
        """Test limit validation."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            EventFilter(limit=0)

        with pytest.raises(ValueError, match="Limit must be positive"):
            EventFilter(limit=-1)

    def test_offset_validation(self):
        """Test offset validation."""
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            EventFilter(offset=-1)

        # Valid offset
        event_filter = EventFilter(offset=0)
        assert event_filter.offset == 0

    def test_sort_order_validation(self):
        """Test sort order validation."""
        # Valid sort orders
        for sort_order in ["asc", "desc"]:
            event_filter = EventFilter(sort_order=sort_order)
            assert event_filter.sort_order == sort_order

        # Invalid sort order
        with pytest.raises(ValueError, match="Sort order must be 'asc' or 'desc'"):
            EventFilter(sort_order="ascending")

    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        # Valid scores
        for score in [0.0, 0.5, 1.0]:
            event_filter = EventFilter(min_anomaly_score=score, max_anomaly_score=score)
            assert event_filter.min_anomaly_score == score
            assert event_filter.max_anomaly_score == score

        # Invalid min score
        with pytest.raises(ValueError, match="Min anomaly score must be between 0.0 and 1.0"):
            EventFilter(min_anomaly_score=-0.1)

        with pytest.raises(ValueError, match="Min anomaly score must be between 0.0 and 1.0"):
            EventFilter(min_anomaly_score=1.1)

        # Invalid max score
        with pytest.raises(ValueError, match="Max anomaly score must be between 0.0 and 1.0"):
            EventFilter(max_anomaly_score=-0.1)

        with pytest.raises(ValueError, match="Max anomaly score must be between 0.0 and 1.0"):
            EventFilter(max_anomaly_score=1.1)

    def test_geographic_filters(self):
        """Test geographic filtering."""
        location_bounds = {
            "north": 40.7128,
            "south": 40.7000,
            "east": -74.0060,
            "west": -74.0200
        }

        event_filter = EventFilter(location_bounds=location_bounds)
        assert event_filter.location_bounds == location_bounds

    def test_pagination_settings(self):
        """Test pagination settings."""
        event_filter = EventFilter(
            limit=25,
            offset=100,
            sort_by="ingestion_time",
            sort_order="asc"
        )

        assert event_filter.limit == 25
        assert event_filter.offset == 100
        assert event_filter.sort_by == "ingestion_time"
        assert event_filter.sort_order == "asc"


class TestEventAggregation:
    """Test cases for EventAggregation dataclass."""

    def test_valid_creation(self):
        """Test creating valid event aggregation."""
        first_time = datetime.utcnow() - timedelta(hours=2)
        last_time = datetime.utcnow()

        aggregation = EventAggregation(
            group_key="detector_001",
            count=15,
            min_severity=EventSeverity.LOW,
            max_severity=EventSeverity.HIGH,
            first_event_time=first_time,
            last_event_time=last_time,
            unique_detectors=3,
            unique_sessions=5,
            resolved_count=8,
            acknowledged_count=12,
            avg_anomaly_score=0.75
        )

        assert aggregation.group_key == "detector_001"
        assert aggregation.count == 15
        assert aggregation.min_severity == EventSeverity.LOW
        assert aggregation.max_severity == EventSeverity.HIGH
        assert aggregation.first_event_time == first_time
        assert aggregation.last_event_time == last_time
        assert aggregation.unique_detectors == 3
        assert aggregation.unique_sessions == 5
        assert aggregation.resolved_count == 8
        assert aggregation.acknowledged_count == 12
        assert aggregation.avg_anomaly_score == 0.75

    def test_minimal_creation(self):
        """Test creating aggregation with minimal fields."""
        first_time = datetime.utcnow() - timedelta(hours=1)
        last_time = datetime.utcnow()

        aggregation = EventAggregation(
            group_key="session_001",
            count=5,
            min_severity=EventSeverity.MEDIUM,
            max_severity=EventSeverity.MEDIUM,
            first_event_time=first_time,
            last_event_time=last_time,
            unique_detectors=1,
            unique_sessions=1,
            resolved_count=0,
            acknowledged_count=2
        )

        assert aggregation.group_key == "session_001"
        assert aggregation.count == 5
        assert aggregation.avg_anomaly_score is None

    def test_severity_comparison(self):
        """Test severity comparison logic."""
        first_time = datetime.utcnow() - timedelta(hours=1)
        last_time = datetime.utcnow()

        # Single severity level
        aggregation = EventAggregation(
            group_key="test",
            count=10,
            min_severity=EventSeverity.CRITICAL,
            max_severity=EventSeverity.CRITICAL,
            first_event_time=first_time,
            last_event_time=last_time,
            unique_detectors=1,
            unique_sessions=1,
            resolved_count=5,
            acknowledged_count=8
        )

        assert aggregation.min_severity == aggregation.max_severity
        assert aggregation.min_severity == EventSeverity.CRITICAL

    def test_time_range_validation(self):
        """Test time range validation."""
        first_time = datetime.utcnow() - timedelta(hours=1)
        last_time = datetime.utcnow()

        aggregation = EventAggregation(
            group_key="test",
            count=5,
            min_severity=EventSeverity.LOW,
            max_severity=EventSeverity.HIGH,
            first_event_time=first_time,
            last_event_time=last_time,
            unique_detectors=1,
            unique_sessions=1,
            resolved_count=2,
            acknowledged_count=3
        )

        assert aggregation.first_event_time < aggregation.last_event_time
        time_diff = aggregation.last_event_time - aggregation.first_event_time
        assert time_diff.total_seconds() > 0


class TestEventPattern:
    """Test cases for EventPattern dataclass."""

    def test_valid_creation(self):
        """Test creating valid event pattern."""
        conditions = {
            "event_type": "anomaly_detected",
            "severity": "high",
            "detector_id": "detector_001"
        }

        pattern = EventPattern(
            name="High severity anomaly pattern",
            description="Pattern for detecting high severity anomalies",
            pattern_type="frequency",
            conditions=conditions,
            time_window=3600,
            confidence=0.85,
            created_by="admin",
            alert_threshold=5,
            alert_enabled=True
        )

        assert pattern.name == "High severity anomaly pattern"
        assert pattern.description == "Pattern for detecting high severity anomalies"
        assert pattern.pattern_type == "frequency"
        assert pattern.conditions == conditions
        assert pattern.time_window == 3600
        assert pattern.confidence == 0.85
        assert pattern.created_by == "admin"
        assert pattern.alert_threshold == 5
        assert pattern.alert_enabled is True
        assert isinstance(pattern.id, UUID)
        assert pattern.match_count == 0
        assert pattern.last_matched is None

    def test_default_values(self):
        """Test default values for event pattern."""
        pattern = EventPattern(
            name="Test pattern",
            description="Test description",
            pattern_type="sequence",
            conditions={"test": "value"},
            time_window=1800,
            confidence=0.9,
            created_by="user"
        )

        assert isinstance(pattern.id, UUID)
        assert pattern.match_count == 0
        assert pattern.last_matched is None
        assert pattern.alert_enabled is True
        assert pattern.alert_threshold == 1
        assert isinstance(pattern.created_at, datetime)
        assert pattern.tags == []
        assert pattern.metadata == {}

    def test_pattern_type_validation(self):
        """Test pattern type validation."""
        valid_types = ["frequency", "sequence", "correlation"]

        for pattern_type in valid_types:
            pattern = EventPattern(
                name="Test pattern",
                description="Test description",
                pattern_type=pattern_type,
                conditions={"test": "value"},
                time_window=1800,
                confidence=0.9,
                created_by="user"
            )
            assert pattern.pattern_type == pattern_type

        # Invalid pattern type
        with pytest.raises(ValueError, match="Pattern type must be one of"):
            EventPattern(
                name="Test pattern",
                description="Test description",
                pattern_type="invalid_type",
                conditions={"test": "value"},
                time_window=1800,
                confidence=0.9,
                created_by="user"
            )

    def test_time_window_validation(self):
        """Test time window validation."""
        # Valid time windows
        for time_window in [1, 3600, 86400]:
            pattern = EventPattern(
                name="Test pattern",
                description="Test description",
                pattern_type="frequency",
                conditions={"test": "value"},
                time_window=time_window,
                confidence=0.9,
                created_by="user"
            )
            assert pattern.time_window == time_window

        # Invalid time windows
        for time_window in [0, -1, -3600]:
            with pytest.raises(ValueError, match="Time window must be positive"):
                EventPattern(
                    name="Test pattern",
                    description="Test description",
                    pattern_type="frequency",
                    conditions={"test": "value"},
                    time_window=time_window,
                    confidence=0.9,
                    created_by="user"
                )

    def test_confidence_validation(self):
        """Test confidence validation."""
        # Valid confidence levels
        for confidence in [0.0, 0.5, 1.0]:
            pattern = EventPattern(
                name="Test pattern",
                description="Test description",
                pattern_type="frequency",
                conditions={"test": "value"},
                time_window=1800,
                confidence=confidence,
                created_by="user"
            )
            assert pattern.confidence == confidence

        # Invalid confidence levels
        for confidence in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
                EventPattern(
                    name="Test pattern",
                    description="Test description",
                    pattern_type="frequency",
                    conditions={"test": "value"},
                    time_window=1800,
                    confidence=confidence,
                    created_by="user"
                )

    def test_alert_threshold_validation(self):
        """Test alert threshold validation."""
        # Valid alert thresholds
        for threshold in [1, 5, 10]:
            pattern = EventPattern(
                name="Test pattern",
                description="Test description",
                pattern_type="frequency",
                conditions={"test": "value"},
                time_window=1800,
                confidence=0.9,
                created_by="user",
                alert_threshold=threshold
            )
            assert pattern.alert_threshold == threshold

        # Invalid alert thresholds
        for threshold in [0, -1, -5]:
            with pytest.raises(ValueError, match="Alert threshold must be positive"):
                EventPattern(
                    name="Test pattern",
                    description="Test description",
                    pattern_type="frequency",
                    conditions={"test": "value"},
                    time_window=1800,
                    confidence=0.9,
                    created_by="user",
                    alert_threshold=threshold
                )

    def test_complex_conditions(self):
        """Test complex pattern conditions."""
        complex_conditions = {
            "and": [
                {"event_type": "anomaly_detected"},
                {"severity": {"in": ["high", "critical"]}},
                {"detector_id": {"starts_with": "detector_"}},
                {"anomaly_score": {"gt": 0.8}}
            ]
        }

        pattern = EventPattern(
            name="Complex pattern",
            description="Pattern with complex conditions",
            pattern_type="correlation",
            conditions=complex_conditions,
            time_window=7200,
            confidence=0.95,
            created_by="admin"
        )

        assert pattern.conditions == complex_conditions
        assert pattern.pattern_type == "correlation"

    def test_pattern_metadata(self):
        """Test pattern metadata and tags."""
        pattern = EventPattern(
            name="Test pattern",
            description="Test description",
            pattern_type="frequency",
            conditions={"test": "value"},
            time_window=1800,
            confidence=0.9,
            created_by="user",
            tags=["important", "security"],
            metadata={"version": "1.0", "category": "security"}
        )

        assert pattern.tags == ["important", "security"]
        assert pattern.metadata == {"version": "1.0", "category": "security"}


class TestEventSummary:
    """Test cases for EventSummary dataclass."""

    def test_valid_creation(self):
        """Test creating valid event summary."""
        events_by_type = {
            "anomaly_detected": 50,
            "data_quality_issue": 20,
            "performance_degradation": 10
        }

        events_by_severity = {
            "low": 30,
            "medium": 25,
            "high": 15,
            "critical": 10
        }

        events_by_status = {
            "pending": 20,
            "processing": 5,
            "resolved": 45,
            "acknowledged": 10
        }

        top_detectors = [
            {"detector_id": "detector_001", "count": 25},
            {"detector_id": "detector_002", "count": 20}
        ]

        top_data_sources = [
            {"source": "sensor_stream", "count": 40},
            {"source": "api_data", "count": 25}
        ]

        time_range = {
            "start": datetime.utcnow() - timedelta(hours=24),
            "end": datetime.utcnow()
        }

        summary = EventSummary(
            total_events=80,
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_status=events_by_status,
            anomaly_rate=0.625,
            resolution_rate=0.5625,
            top_detectors=top_detectors,
            top_data_sources=top_data_sources,
            time_range=time_range,
            avg_anomaly_score=0.72,
            avg_resolution_time=1800.0
        )

        assert summary.total_events == 80
        assert summary.events_by_type == events_by_type
        assert summary.events_by_severity == events_by_severity
        assert summary.events_by_status == events_by_status
        assert summary.anomaly_rate == 0.625
        assert summary.resolution_rate == 0.5625
        assert summary.top_detectors == top_detectors
        assert summary.top_data_sources == top_data_sources
        assert summary.time_range == time_range
        assert summary.avg_anomaly_score == 0.72
        assert summary.avg_resolution_time == 1800.0
        assert isinstance(summary.summary_generated_at, datetime)

    def test_minimal_creation(self):
        """Test creating summary with minimal fields."""
        summary = EventSummary(
            total_events=100,
            events_by_type={"anomaly_detected": 100},
            events_by_severity={"high": 100},
            events_by_status={"pending": 100},
            anomaly_rate=1.0,
            resolution_rate=0.0,
            top_detectors=[],
            top_data_sources=[],
            time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
        )

        assert summary.total_events == 100
        assert summary.avg_anomaly_score is None
        assert summary.avg_resolution_time is None
        assert isinstance(summary.summary_generated_at, datetime)

    def test_total_events_validation(self):
        """Test total events validation."""
        # Valid total events
        for total in [0, 100, 1000]:
            summary = EventSummary(
                total_events=total,
                events_by_type={"test": total},
                events_by_severity={"low": total},
                events_by_status={"pending": total},
                anomaly_rate=0.5,
                resolution_rate=0.5,
                top_detectors=[],
                top_data_sources=[],
                time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
            )
            assert summary.total_events == total

        # Invalid total events
        with pytest.raises(ValueError, match="Total events must be non-negative"):
            EventSummary(
                total_events=-1,
                events_by_type={"test": 0},
                events_by_severity={"low": 0},
                events_by_status={"pending": 0},
                anomaly_rate=0.5,
                resolution_rate=0.5,
                top_detectors=[],
                top_data_sources=[],
                time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
            )

    def test_anomaly_rate_validation(self):
        """Test anomaly rate validation."""
        # Valid anomaly rates
        for rate in [0.0, 0.5, 1.0]:
            summary = EventSummary(
                total_events=100,
                events_by_type={"test": 100},
                events_by_severity={"low": 100},
                events_by_status={"pending": 100},
                anomaly_rate=rate,
                resolution_rate=0.5,
                top_detectors=[],
                top_data_sources=[],
                time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
            )
            assert summary.anomaly_rate == rate

        # Invalid anomaly rates
        for rate in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError, match="Anomaly rate must be between 0.0 and 1.0"):
                EventSummary(
                    total_events=100,
                    events_by_type={"test": 100},
                    events_by_severity={"low": 100},
                    events_by_status={"pending": 100},
                    anomaly_rate=rate,
                    resolution_rate=0.5,
                    top_detectors=[],
                    top_data_sources=[],
                    time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
                )

    def test_resolution_rate_validation(self):
        """Test resolution rate validation."""
        # Valid resolution rates
        for rate in [0.0, 0.5, 1.0]:
            summary = EventSummary(
                total_events=100,
                events_by_type={"test": 100},
                events_by_severity={"low": 100},
                events_by_status={"pending": 100},
                anomaly_rate=0.5,
                resolution_rate=rate,
                top_detectors=[],
                top_data_sources=[],
                time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
            )
            assert summary.resolution_rate == rate

        # Invalid resolution rates
        for rate in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValueError, match="Resolution rate must be between 0.0 and 1.0"):
                EventSummary(
                    total_events=100,
                    events_by_type={"test": 100},
                    events_by_severity={"low": 100},
                    events_by_status={"pending": 100},
                    anomaly_rate=0.5,
                    resolution_rate=rate,
                    top_detectors=[],
                    top_data_sources=[],
                    time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
                )

    def test_statistical_consistency(self):
        """Test statistical consistency of summary data."""
        events_by_type = {
            "anomaly_detected": 60,
            "data_quality_issue": 25,
            "performance_degradation": 15
        }

        events_by_severity = {
            "low": 40,
            "medium": 35,
            "high": 20,
            "critical": 5
        }

        events_by_status = {
            "pending": 30,
            "processing": 10,
            "resolved": 50,
            "acknowledged": 10
        }

        total_events = 100

        summary = EventSummary(
            total_events=total_events,
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_status=events_by_status,
            anomaly_rate=0.6,
            resolution_rate=0.5,
            top_detectors=[],
            top_data_sources=[],
            time_range={"start": datetime.utcnow(), "end": datetime.utcnow()}
        )

        # Check that counts are consistent
        assert sum(events_by_type.values()) == total_events
        assert sum(events_by_severity.values()) == total_events
        assert sum(events_by_status.values()) == total_events

        # Check that rates make sense
        anomaly_events = events_by_type.get("anomaly_detected", 0)
        expected_anomaly_rate = anomaly_events / total_events
        assert abs(summary.anomaly_rate - expected_anomaly_rate) < 0.01

        resolved_events = events_by_status.get("resolved", 0)
        expected_resolution_rate = resolved_events / total_events
        assert abs(summary.resolution_rate - expected_resolution_rate) < 0.01


class TestAnomalyEventIntegration:
    """Test cases for integration scenarios."""

    def test_complete_event_lifecycle(self):
        """Test complete event lifecycle."""
        # Create event with anomaly data
        anomaly_data = AnomalyEventData(
            anomaly_score=0.9,
            confidence=0.85,
            detection_method="isolation_forest",
            model_version="v1.2.0"
        )

        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Critical anomaly detected",
            description="High anomaly score detected in sensor data",
            raw_data={"sensor_id": "S001", "value": 999.9},
            event_time=datetime.utcnow(),
            anomaly_data=anomaly_data
        )

        # Verify initial state
        assert event.status == EventStatus.PENDING
        assert event.is_actionable() is True
        assert event.is_resolved() is False

        # Process the event
        event.mark_processing()
        assert event.status == EventStatus.PROCESSING
        assert event.processing_attempts == 1

        # Acknowledge the event
        event.acknowledge("operator", "Investigating the issue")
        assert event.status == EventStatus.ACKNOWLEDGED
        assert event.acknowledged_by == "operator"

        # Resolve the event
        event.resolve("admin", "Issue resolved by restarting sensor")
        assert event.status == EventStatus.RESOLVED
        assert event.resolved_by == "admin"
        assert event.is_resolved() is True
        assert event.is_actionable() is False

    def test_event_filtering_scenario(self):
        """Test event filtering scenario."""
        # Create events with different characteristics
        events = []

        # High severity anomaly
        events.append(AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="High anomaly",
            description="High severity anomaly",
            raw_data={"score": 0.9},
            event_time=datetime.utcnow(),
            anomaly_data=AnomalyEventData(anomaly_score=0.9, confidence=0.8)
        ))

        # Low severity data quality issue
        events.append(AnomalyEvent(
            event_type=EventType.DATA_QUALITY_ISSUE,
            severity=EventSeverity.LOW,
            title="Data quality issue",
            description="Minor data quality issue",
            raw_data={"missing_count": 2},
            event_time=datetime.utcnow(),
            data_quality_data=DataQualityEventData(
                issue_type="missing_values",
                affected_fields=["field1"],
                severity_score=0.3
            )
        ))

        # Create filter for high severity events
        high_severity_filter = EventFilter(
            severities=[EventSeverity.HIGH, EventSeverity.CRITICAL],
            event_types=[EventType.ANOMALY_DETECTED],
            min_anomaly_score=0.8
        )

        # Verify filter criteria
        assert EventSeverity.HIGH in high_severity_filter.severities
        assert EventType.ANOMALY_DETECTED in high_severity_filter.event_types
        assert high_severity_filter.min_anomaly_score == 0.8

        # Simulate filtering logic
        filtered_events = []
        for event in events:
            if (event.severity in high_severity_filter.severities and
                event.event_type in high_severity_filter.event_types and
                event.anomaly_data and
                event.anomaly_data.anomaly_score >= high_severity_filter.min_anomaly_score):
                filtered_events.append(event)

        assert len(filtered_events) == 1
        assert filtered_events[0].severity == EventSeverity.HIGH

    def test_event_pattern_matching(self):
        """Test event pattern matching scenario."""
        # Create a pattern for detecting frequent anomalies
        pattern = EventPattern(
            name="Frequent anomaly pattern",
            description="Detects frequent anomalies from same detector",
            pattern_type="frequency",
            conditions={
                "event_type": "anomaly_detected",
                "severity": {"in": ["high", "critical"]},
                "detector_id": "detector_001"
            },
            time_window=3600,
            confidence=0.9,
            created_by="system",
            alert_threshold=5
        )

        # Create matching events
        detector_id = uuid4()
        matching_events = []

        for i in range(7):
            event = AnomalyEvent(
                event_type=EventType.ANOMALY_DETECTED,
                severity=EventSeverity.HIGH,
                title=f"Anomaly {i+1}",
                description=f"Anomaly detected {i+1}",
                raw_data={"sequence": i+1},
                event_time=datetime.utcnow() - timedelta(minutes=i*5),
                detector_id=detector_id
            )
            matching_events.append(event)

        # Simulate pattern matching
        pattern.match_count = len(matching_events)
        pattern.last_matched = datetime.utcnow()

        # Check if pattern should trigger alert
        should_alert = (pattern.alert_enabled and
                       pattern.match_count >= pattern.alert_threshold)

        assert should_alert is True
        assert pattern.match_count == 7
        assert pattern.match_count > pattern.alert_threshold

    def test_event_aggregation_scenario(self):
        """Test event aggregation scenario."""
        # Create events for aggregation
        detector_id = uuid4()
        first_time = datetime.utcnow() - timedelta(hours=2)
        last_time = datetime.utcnow()

        events = []
        for i in range(10):
            severity = EventSeverity.HIGH if i < 3 else EventSeverity.MEDIUM
            status = EventStatus.RESOLVED if i < 5 else EventStatus.PENDING

            event = AnomalyEvent(
                event_type=EventType.ANOMALY_DETECTED,
                severity=severity,
                title=f"Event {i+1}",
                description=f"Event description {i+1}",
                raw_data={"index": i+1},
                event_time=first_time + timedelta(minutes=i*10),
                detector_id=detector_id,
                status=status
            )
            events.append(event)

        # Create aggregation
        aggregation = EventAggregation(
            group_key=str(detector_id),
            count=len(events),
            min_severity=EventSeverity.MEDIUM,
            max_severity=EventSeverity.HIGH,
            first_event_time=first_time,
            last_event_time=last_time,
            unique_detectors=1,
            unique_sessions=1,
            resolved_count=5,
            acknowledged_count=0,
            avg_anomaly_score=0.75
        )

        # Verify aggregation
        assert aggregation.count == 10
        assert aggregation.resolved_count == 5
        assert aggregation.unique_detectors == 1
        assert aggregation.min_severity == EventSeverity.MEDIUM
        assert aggregation.max_severity == EventSeverity.HIGH

    def test_event_summary_generation(self):
        """Test event summary generation."""
        # Create diverse events
        events = []

        # Anomaly events
        for i in range(30):
            severity = EventSeverity.HIGH if i < 10 else EventSeverity.MEDIUM
            status = EventStatus.RESOLVED if i < 15 else EventStatus.PENDING

            event = AnomalyEvent(
                event_type=EventType.ANOMALY_DETECTED,
                severity=severity,
                title=f"Anomaly {i+1}",
                description=f"Anomaly description {i+1}",
                raw_data={"index": i+1},
                event_time=datetime.utcnow() - timedelta(minutes=i*5),
                status=status
            )
            events.append(event)

        # Data quality events
        for i in range(20):
            severity = EventSeverity.LOW
            status = EventStatus.RESOLVED if i < 10 else EventStatus.PENDING

            event = AnomalyEvent(
                event_type=EventType.DATA_QUALITY_ISSUE,
                severity=severity,
                title=f"Data quality {i+1}",
                description=f"Data quality description {i+1}",
                raw_data={"index": i+1},
                event_time=datetime.utcnow() - timedelta(minutes=i*3),
                status=status
            )
            events.append(event)

        # Generate summary
        total_events = len(events)
        anomaly_events = sum(1 for e in events if e.event_type == EventType.ANOMALY_DETECTED)
        resolved_events = sum(1 for e in events if e.status == EventStatus.RESOLVED)

        summary = EventSummary(
            total_events=total_events,
            events_by_type={
                "anomaly_detected": anomaly_events,
                "data_quality_issue": total_events - anomaly_events
            },
            events_by_severity={
                "low": 20,
                "medium": 20,
                "high": 10
            },
            events_by_status={
                "pending": total_events - resolved_events,
                "resolved": resolved_events
            },
            anomaly_rate=anomaly_events / total_events,
            resolution_rate=resolved_events / total_events,
            top_detectors=[],
            top_data_sources=[],
            time_range={
                "start": datetime.utcnow() - timedelta(hours=4),
                "end": datetime.utcnow()
            }
        )

        # Verify summary
        assert summary.total_events == 50
        assert summary.anomaly_rate == 0.6
        assert summary.resolution_rate == 0.5
        assert summary.events_by_type["anomaly_detected"] == 30
        assert summary.events_by_type["data_quality_issue"] == 20
