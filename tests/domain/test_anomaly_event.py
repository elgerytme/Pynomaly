"""Tests for anomaly event domain entities."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from pynomaly.domain.entities.anomaly_event import (
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


class TestAnomalyEventData:
    """Test cases for AnomalyEventData dataclass."""

    def test_valid_anomaly_event_data(self):
        """Test creating valid anomaly event data."""
        data = AnomalyEventData(
            anomaly_score=0.8,
            confidence=0.95,
            feature_contributions={"feature1": 0.6, "feature2": 0.4},
            predicted_class="outlier",
            detection_method="IsolationForest",
            model_version="v1.2.3",
        )

        assert data.anomaly_score == 0.8
        assert data.confidence == 0.95
        assert data.predicted_class == "outlier"
        assert len(data.feature_contributions) == 2

    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        # Valid scores
        AnomalyEventData(anomaly_score=0.0, confidence=0.5)
        AnomalyEventData(anomaly_score=1.0, confidence=0.5)

        # Invalid scores
        with pytest.raises(
            ValueError, match="Anomaly score must be between 0.0 and 1.0"
        ):
            AnomalyEventData(anomaly_score=-0.1, confidence=0.5)

        with pytest.raises(
            ValueError, match="Anomaly score must be between 0.0 and 1.0"
        ):
            AnomalyEventData(anomaly_score=1.1, confidence=0.5)

    def test_confidence_validation(self):
        """Test confidence validation."""
        # Valid confidence
        AnomalyEventData(anomaly_score=0.5, confidence=0.0)
        AnomalyEventData(anomaly_score=0.5, confidence=1.0)

        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            AnomalyEventData(anomaly_score=0.5, confidence=-0.1)

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            AnomalyEventData(anomaly_score=0.5, confidence=1.1)


class TestDataQualityEventData:
    """Test cases for DataQualityEventData dataclass."""

    def test_valid_data_quality_event(self):
        """Test creating valid data quality event data."""
        data = DataQualityEventData(
            issue_type="missing_values",
            affected_fields=["field1", "field2"],
            severity_score=0.7,
            missing_percentage=15.5,
            outlier_percentage=2.3,
        )

        assert data.issue_type == "missing_values"
        assert len(data.affected_fields) == 2
        assert data.severity_score == 0.7

    def test_severity_score_validation(self):
        """Test severity score validation."""
        with pytest.raises(
            ValueError, match="Severity score must be between 0.0 and 1.0"
        ):
            DataQualityEventData(
                issue_type="test",
                affected_fields=["field1"],
                severity_score=1.5,
            )

    def test_percentage_validation(self):
        """Test percentage validation."""
        # Valid percentages
        DataQualityEventData(
            issue_type="test",
            affected_fields=["field1"],
            severity_score=0.5,
            missing_percentage=0.0,
            outlier_percentage=100.0,
        )

        # Invalid percentages
        with pytest.raises(
            ValueError, match="Missing percentage must be between 0.0 and 100.0"
        ):
            DataQualityEventData(
                issue_type="test",
                affected_fields=["field1"],
                severity_score=0.5,
                missing_percentage=101.0,
            )


class TestPerformanceEventData:
    """Test cases for PerformanceEventData dataclass."""

    def test_valid_performance_event(self):
        """Test creating valid performance event data."""
        data = PerformanceEventData(
            metric_name="accuracy",
            current_value=0.75,
            baseline_value=0.85,
            degradation_percentage=11.76,
            threshold_value=0.80,
            trend_direction="decreasing",
        )

        assert data.metric_name == "accuracy"
        assert data.current_value == 0.75
        assert data.trend_direction == "decreasing"

    def test_trend_direction_validation(self):
        """Test trend direction validation."""
        # Valid directions
        for direction in ["increasing", "decreasing", "stable"]:
            PerformanceEventData(
                metric_name="test",
                current_value=1.0,
                baseline_value=1.0,
                degradation_percentage=0.0,
                threshold_value=1.0,
                trend_direction=direction,
            )

        # Invalid direction
        with pytest.raises(ValueError, match="Trend direction must be one of"):
            PerformanceEventData(
                metric_name="test",
                current_value=1.0,
                baseline_value=1.0,
                degradation_percentage=0.0,
                threshold_value=1.0,
                trend_direction="invalid",
            )


class TestAnomalyEvent:
    """Test cases for AnomalyEvent dataclass."""

    def test_create_minimal_event(self):
        """Test creating minimal anomaly event."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test Anomaly",
            description="Test description",
            raw_data={"value": 100},
            event_time=datetime.utcnow(),
        )

        assert event.event_type == EventType.ANOMALY_DETECTED
        assert event.severity == EventSeverity.HIGH
        assert event.status == EventStatus.PENDING
        assert event.id is not None
        assert isinstance(event.ingestion_time, datetime)

    def test_acknowledge_event(self):
        """Test acknowledging an event."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.MEDIUM,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )

        event.acknowledge("user1", "Acknowledged for investigation")

        assert event.status == EventStatus.ACKNOWLEDGED
        assert event.acknowledged_by == "user1"
        assert event.acknowledged_at is not None
        assert "acknowledgment_notes" in event.metadata

    def test_resolve_event(self):
        """Test resolving an event."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.LOW,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )

        event.resolve("user2", "False positive")

        assert event.status == EventStatus.RESOLVED
        assert event.resolved_by == "user2"
        assert event.resolved_at is not None
        assert event.resolution_notes == "False positive"

    def test_ignore_event(self):
        """Test ignoring an event."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.LOW,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )

        event.ignore("user3", "Not relevant")

        assert event.status == EventStatus.IGNORED
        assert event.metadata["ignored_by"] == "user3"
        assert event.metadata["ignore_reason"] == "Not relevant"

    def test_mark_processing(self):
        """Test marking event as processing."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )

        event.mark_processing()

        assert event.status == EventStatus.PROCESSING
        assert event.processing_time is not None
        assert event.processing_attempts == 1

    def test_mark_failed(self):
        """Test marking event as failed."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.CRITICAL,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )

        retry_time = datetime.utcnow() + timedelta(hours=1)
        event.mark_failed("Processing error", retry_time)

        assert event.status == EventStatus.FAILED
        assert event.last_error == "Processing error"
        assert event.retry_after == retry_time

    def test_is_actionable(self):
        """Test checking if event is actionable."""
        # High severity, pending status
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.HIGH,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )
        assert event.is_actionable()

        # Low severity
        event.severity = EventSeverity.LOW
        assert not event.is_actionable()

        # High severity but resolved
        event.severity = EventSeverity.HIGH
        event.status = EventStatus.RESOLVED
        assert not event.is_actionable()

    def test_get_age(self):
        """Test getting event age."""
        past_time = datetime.utcnow() - timedelta(hours=2)
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.MEDIUM,
            title="Test",
            description="Test",
            raw_data={},
            event_time=past_time,
        )

        age = event.get_age()
        assert age > 7000  # More than 2 hours in seconds

    def test_add_remove_tags(self):
        """Test adding and removing tags."""
        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=EventSeverity.MEDIUM,
            title="Test",
            description="Test",
            raw_data={},
            event_time=datetime.utcnow(),
        )

        # Add tags
        event.add_tag("critical")
        event.add_tag("financial")
        assert "critical" in event.tags
        assert "financial" in event.tags

        # Add duplicate tag (should not duplicate)
        event.add_tag("critical")
        assert event.tags.count("critical") == 1

        # Remove tag
        event.remove_tag("financial")
        assert "financial" not in event.tags
        assert "critical" in event.tags


class TestEventFilter:
    """Test cases for EventFilter dataclass."""

    def test_default_event_filter(self):
        """Test creating event filter with defaults."""
        filter_obj = EventFilter()

        assert filter_obj.limit == 100
        assert filter_obj.offset == 0
        assert filter_obj.sort_by == "event_time"
        assert filter_obj.sort_order == "desc"

    def test_event_filter_validation(self):
        """Test event filter validation."""
        # Valid filter
        EventFilter(limit=50, offset=10, sort_order="asc")

        # Invalid limit
        with pytest.raises(ValueError, match="Limit must be positive"):
            EventFilter(limit=0)

        # Invalid offset
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            EventFilter(offset=-1)

        # Invalid sort order
        with pytest.raises(ValueError, match="Sort order must be 'asc' or 'desc'"):
            EventFilter(sort_order="invalid")

    def test_anomaly_score_filters(self):
        """Test anomaly score filter validation."""
        # Valid scores
        EventFilter(min_anomaly_score=0.0, max_anomaly_score=1.0)

        # Invalid scores
        with pytest.raises(
            ValueError, match="Min anomaly score must be between 0.0 and 1.0"
        ):
            EventFilter(min_anomaly_score=-0.1)

        with pytest.raises(
            ValueError, match="Max anomaly score must be between 0.0 and 1.0"
        ):
            EventFilter(max_anomaly_score=1.1)


class TestEventPattern:
    """Test cases for EventPattern dataclass."""

    def test_create_event_pattern(self):
        """Test creating event pattern."""
        pattern = EventPattern(
            name="High Frequency Anomalies",
            description="Pattern for detecting frequent anomalies",
            pattern_type="frequency",
            conditions={"count": 5, "window": 300},
            time_window=600,
            confidence=0.85,
            created_by="admin",
        )

        assert pattern.name == "High Frequency Anomalies"
        assert pattern.pattern_type == "frequency"
        assert pattern.match_count == 0
        assert pattern.alert_enabled is True

    def test_pattern_validation(self):
        """Test pattern validation."""
        # Invalid pattern type
        with pytest.raises(ValueError, match="Pattern type must be one of"):
            EventPattern(
                name="Test",
                description="Test",
                pattern_type="invalid",
                conditions={},
                time_window=600,
                confidence=0.5,
                created_by="user",
            )

        # Invalid time window
        with pytest.raises(ValueError, match="Time window must be positive"):
            EventPattern(
                name="Test",
                description="Test",
                pattern_type="frequency",
                conditions={},
                time_window=0,
                confidence=0.5,
                created_by="user",
            )

        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            EventPattern(
                name="Test",
                description="Test",
                pattern_type="sequence",
                conditions={},
                time_window=600,
                confidence=1.5,
                created_by="user",
            )


class TestEventSummary:
    """Test cases for EventSummary dataclass."""

    def test_create_event_summary(self):
        """Test creating event summary."""
        summary = EventSummary(
            total_events=1000,
            events_by_type={"anomaly_detected": 800, "data_quality_issue": 200},
            events_by_severity={"high": 100, "medium": 400, "low": 500},
            events_by_status={"resolved": 750, "pending": 200, "acknowledged": 50},
            anomaly_rate=0.15,
            resolution_rate=0.75,
            top_detectors=[{"name": "detector1", "count": 500}],
            top_data_sources=[{"name": "source1", "count": 600}],
            time_range={"start": datetime.utcnow(), "end": datetime.utcnow()},
        )

        assert summary.total_events == 1000
        assert summary.anomaly_rate == 0.15
        assert summary.resolution_rate == 0.75

    def test_summary_validation(self):
        """Test summary validation."""
        base_data = {
            "total_events": 100,
            "events_by_type": {},
            "events_by_severity": {},
            "events_by_status": {},
            "anomaly_rate": 0.1,
            "resolution_rate": 0.8,
            "top_detectors": [],
            "top_data_sources": [],
            "time_range": {},
        }

        # Valid data
        EventSummary(**base_data)

        # Invalid total events
        with pytest.raises(ValueError, match="Total events must be non-negative"):
            EventSummary(**{**base_data, "total_events": -1})

        # Invalid anomaly rate
        with pytest.raises(
            ValueError, match="Anomaly rate must be between 0.0 and 1.0"
        ):
            EventSummary(**{**base_data, "anomaly_rate": 1.5})

        # Invalid resolution rate
        with pytest.raises(
            ValueError, match="Resolution rate must be between 0.0 and 1.0"
        ):
            EventSummary(**{**base_data, "resolution_rate": -0.1})


class TestEventAggregation:
    """Test cases for EventAggregation dataclass."""

    def test_create_event_aggregation(self):
        """Test creating event aggregation."""
        now = datetime.utcnow()
        aggregation = EventAggregation(
            group_key="detector_id:123",
            count=50,
            min_severity=EventSeverity.LOW,
            max_severity=EventSeverity.HIGH,
            first_event_time=now - timedelta(hours=24),
            last_event_time=now,
            unique_detectors=5,
            unique_sessions=10,
            resolved_count=30,
            acknowledged_count=15,
            avg_anomaly_score=0.72,
        )

        assert aggregation.group_key == "detector_id:123"
        assert aggregation.count == 50
        assert aggregation.avg_anomaly_score == 0.72
        assert aggregation.resolved_count == 30
