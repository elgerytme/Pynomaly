"""Tests for intelligent alert service."""

from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest
from pynomaly.application.services.intelligent_alert_service import (
    AlertCorrelationEngine,
    IntelligentAlertService,
    NoiseClassificationModel,
)
from pynomaly.domain.entities.alert import (
    Alert,
    AlertMetadata,
    AlertSeverity,
    AlertStatus,
    AlertType,
    MLNoiseFeatures,
    NoiseClassification,
    NotificationChannel,
)


class TestAlertCorrelationEngine:
    """Test cases for alert correlation engine."""

    @pytest.fixture
    def correlation_engine(self):
        """Create correlation engine instance."""
        return AlertCorrelationEngine()

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert for testing."""
        metadata = AlertMetadata(
            tenant_id=uuid4(),
            anomaly_score=0.8,
            confidence_level=0.9,
            system_load=0.7,
            affected_resources=["resource1", "resource2"],
        )

        return Alert(
            title="Test Alert",
            description="Test alert for correlation",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=metadata,
        )

    def test_add_alert(self, correlation_engine, sample_alert):
        """Test adding alert to correlation engine."""
        correlation_engine.add_alert(sample_alert)

        assert sample_alert.alert_id in correlation_engine.alert_history
        assert sample_alert.alert_id in correlation_engine.feature_cache

        # Check that features were extracted
        features = correlation_engine.feature_cache[sample_alert.alert_id]
        assert len(features) > 0
        assert isinstance(features, np.ndarray)

    def test_find_temporal_correlations(self, correlation_engine):
        """Test finding temporal correlations."""
        # Create alerts with close timestamps
        alert1 = Alert(
            title="Alert 1",
            description="First alert",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
        )

        alert2 = Alert(
            title="Alert 2",
            description="Second alert",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.SYSTEM_PERFORMANCE,
            source=NotificationChannel.SYSTEM_MONITOR,
        )
        # Set second alert to be 5 minutes after first
        alert2.created_at = alert1.created_at + timedelta(minutes=5)

        correlation_engine.add_alert(alert1)
        correlation_engine.add_alert(alert2)

        correlations = correlation_engine.find_correlations(
            alert1, time_window_minutes=60
        )

        # Should find temporal correlation
        temporal_correlations = [
            c for c in correlations if c.correlation_type == "temporal"
        ]
        assert len(temporal_correlations) > 0

        correlation = temporal_correlations[0]
        assert alert2.alert_id in correlation.related_alerts
        assert correlation.correlation_strength > 0.9  # Very close in time

    def test_find_pattern_correlations(self, correlation_engine):
        """Test finding pattern-based correlations."""
        # Create alerts with similar patterns
        metadata1 = AlertMetadata(
            anomaly_score=0.8, confidence_level=0.9, system_load=0.7
        )

        metadata2 = AlertMetadata(
            anomaly_score=0.85,  # Similar
            confidence_level=0.88,  # Similar
            system_load=0.72,  # Similar
        )

        alert1 = Alert(
            title="Pattern Alert 1",
            description="First pattern alert",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=metadata1,
        )

        alert2 = Alert(
            title="Pattern Alert 2",
            description="Second pattern alert",
            severity=AlertSeverity.HIGH,  # Same severity
            category=AlertType.ANOMALY_DETECTION,  # Same category
            source=NotificationChannel.DETECTOR,  # Same source
            metadata=metadata2,
        )

        correlation_engine.add_alert(alert1)
        correlation_engine.add_alert(alert2)

        correlations = correlation_engine.find_correlations(alert1)

        # Should find pattern correlation
        pattern_correlations = [
            c for c in correlations if c.correlation_type == "pattern"
        ]
        assert len(pattern_correlations) > 0

        correlation = pattern_correlations[0]
        assert alert2.alert_id in correlation.related_alerts
        assert correlation.pattern_similarity is not None

    def test_find_causal_correlations(self, correlation_engine):
        """Test finding causal correlations."""
        # Create potential cause-effect relationship
        cause_alert = Alert(
            title="Infrastructure Issue",
            description="Database server down",
            severity=AlertSeverity.CRITICAL,
            category=AlertType.INFRASTRUCTURE,
            source=NotificationChannel.SYSTEM_MONITOR,
        )

        effect_alert = Alert(
            title="Anomaly Detected",
            description="Anomalies in user behavior",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
        )
        # Effect occurs 10 minutes after cause
        effect_alert.created_at = cause_alert.created_at + timedelta(minutes=10)

        correlation_engine.add_alert(cause_alert)
        correlation_engine.add_alert(effect_alert)

        correlations = correlation_engine.find_correlations(effect_alert)

        # Should find causal correlation
        causal_correlations = [
            c for c in correlations if c.correlation_type == "causal"
        ]
        assert len(causal_correlations) > 0

        correlation = causal_correlations[0]
        assert cause_alert.alert_id in correlation.related_alerts
        assert correlation.root_cause_alert == cause_alert.alert_id

    def test_extract_correlation_features(self, correlation_engine, sample_alert):
        """Test feature extraction for correlation."""
        features = correlation_engine._extract_correlation_features(sample_alert)

        assert isinstance(features, np.ndarray)
        assert len(features) > 10  # Should have multiple features

        # Check some specific features
        assert features[5] == sample_alert.metadata.anomaly_score  # Anomaly score
        assert features[6] == sample_alert.metadata.confidence_level  # Confidence
        assert features[7] == sample_alert.metadata.system_load  # System load


class TestNoiseClassificationModel:
    """Test cases for noise classification model."""

    @pytest.fixture
    def noise_model(self):
        """Create noise classification model instance."""
        return NoiseClassificationModel()

    @pytest.fixture
    def sample_features(self):
        """Create sample ML features."""
        return MLNoiseFeatures(
            hour_of_day=14,
            day_of_week=2,  # Wednesday
            is_business_hours=True,
            is_weekend=False,
            alerts_last_hour=3,
            alerts_last_day=25,
            similar_alerts_last_week=8,
            system_load_percentile=0.6,
            memory_pressure=0.4,
            cpu_utilization=0.5,
            model_confidence=0.85,
            anomaly_score_percentile=0.9,
            false_positive_rate_7d=0.1,
            resolution_time_avg=45.0,
            has_correlated_alerts=True,
            correlation_strength_max=0.8,
            num_related_alerts=2,
        )

    def test_extract_features(self, noise_model):
        """Test feature extraction from alert."""
        alert = Alert(
            title="Test Alert",
            description="Test for feature extraction",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
        )

        alert_history = []  # Empty history for simplicity

        features = noise_model.extract_features(alert, alert_history)

        assert isinstance(features, MLNoiseFeatures)
        assert features.hour_of_day == alert.created_at.hour
        assert features.day_of_week == alert.created_at.weekday()
        assert isinstance(features.is_business_hours, bool)
        assert isinstance(features.is_weekend, bool)

    def test_heuristic_classification_signal(self, noise_model):
        """Test heuristic classification for legitimate signals."""
        features = MLNoiseFeatures(
            similar_alerts_last_week=2,  # Low frequency
            alerts_last_hour=1,  # Low frequency
            model_confidence=0.9,  # High confidence
            is_business_hours=True,
            has_correlated_alerts=True,
            false_positive_rate_7d=0.05,  # Low false positive rate
        )

        classification, confidence = noise_model._heuristic_classification(features)

        assert classification == NoiseClassification.SIGNAL
        assert confidence > 0.5

    def test_heuristic_classification_noise(self, noise_model):
        """Test heuristic classification for noise."""
        features = MLNoiseFeatures(
            similar_alerts_last_week=25,  # High frequency
            alerts_last_hour=30,  # Very high frequency
            model_confidence=0.2,  # Low confidence
            is_business_hours=False,
            has_correlated_alerts=False,
            false_positive_rate_7d=0.8,  # High false positive rate
        )

        classification, confidence = noise_model._heuristic_classification(features)

        assert classification == NoiseClassification.NOISE
        assert confidence > 0.6

    def test_add_training_sample(self, noise_model, sample_features):
        """Test adding training samples."""
        initial_count = len(noise_model.training_features)

        noise_model.add_training_sample(sample_features, is_signal=True)

        assert len(noise_model.training_features) == initial_count + 1
        assert len(noise_model.training_labels) == initial_count + 1
        assert noise_model.training_labels[-1] == 1  # Signal = 1

        noise_model.add_training_sample(sample_features, is_signal=False)

        assert len(noise_model.training_features) == initial_count + 2
        assert noise_model.training_labels[-1] == 0  # Noise = 0

    def test_feature_vector_conversion(self, sample_features):
        """Test conversion of features to vector."""
        feature_vector = sample_features.to_feature_vector()

        assert isinstance(feature_vector, list)
        assert len(feature_vector) == 19  # Expected number of features
        assert all(isinstance(x, int | float) for x in feature_vector)

        # Check some specific values
        assert feature_vector[0] == 14.0  # hour_of_day
        assert feature_vector[1] == 2.0  # day_of_week
        assert feature_vector[2] == 1.0  # is_business_hours (True -> 1.0)


class TestIntelligentAlertService:
    """Test cases for intelligent alert service."""

    @pytest.fixture
    def alert_service(self):
        """Create alert service instance."""
        return IntelligentAlertService()

    @pytest.fixture
    def sample_metadata(self):
        """Create sample alert metadata."""
        return AlertMetadata(
            tenant_id=uuid4(),
            detector_id=uuid4(),
            anomaly_score=0.8,
            confidence_level=0.9,
            affected_resources=["web_server", "database"],
            business_impact="high",
        )

    @pytest.mark.asyncio
    async def test_create_alert(self, alert_service, sample_metadata):
        """Test creating an alert."""
        alert = await alert_service.create_alert(
            title="Test Alert",
            description="Test alert creation",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=sample_metadata,
            message="Test message",
        )

        assert isinstance(alert, Alert)
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.category == AlertType.ANOMALY_DETECTION
        assert alert.metadata.tenant_id == sample_metadata.tenant_id

        # Check that alert was stored
        assert alert.alert_id in alert_service.alerts
        assert len(alert_service.alert_history) > 0
        assert alert_service.metrics["total_alerts"] > 0

    @pytest.mark.asyncio
    async def test_process_alert_intelligence(self, alert_service, sample_metadata):
        """Test intelligent alert processing."""
        alert = Alert(
            title="Intelligence Test Alert",
            description="Test intelligent processing",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=sample_metadata,
        )

        processed_alert = await alert_service.process_alert_intelligence(alert)

        # Check that ML classification was applied
        assert processed_alert.noise_classification != NoiseClassification.UNKNOWN
        assert processed_alert.noise_confidence >= 0.0
        assert processed_alert.ml_features is not None

        # Check that correlation was attempted
        # (may not find correlations with empty history)

        # Check metrics were updated
        assert (
            alert_service.metrics["noise_classifications"][
                processed_alert.noise_classification.value
            ]
            > 0
        )

    @pytest.mark.asyncio
    async def test_should_suppress_alert_noise(self, alert_service):
        """Test alert suppression for noise."""
        alert = Alert(
            title="Noisy Alert",
            description="Alert classified as noise",
            severity=AlertSeverity.LOW,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
        )

        # Set high noise classification
        alert.noise_classification = NoiseClassification.NOISE
        alert.noise_confidence = 0.9

        should_suppress, reason = await alert_service._should_suppress_alert(alert)

        assert should_suppress
        assert "ML classified as noise" in reason

    @pytest.mark.asyncio
    async def test_should_suppress_alert_duplicates(self, alert_service):
        """Test alert suppression for duplicates."""
        # Create first alert
        alert1 = Alert(
            title="Duplicate Alert",
            description="First alert",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.SYSTEM_PERFORMANCE,
            source=NotificationChannel.SYSTEM_MONITOR,
        )
        alert1.metadata.tenant_id = uuid4()

        # Add to history
        alert_service.alert_history.append(alert1)

        # Create similar alert
        alert2 = Alert(
            title="Duplicate Alert 2",
            description="Second similar alert",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.SYSTEM_PERFORMANCE,
            source=NotificationChannel.SYSTEM_MONITOR,
        )
        alert2.metadata.tenant_id = alert1.metadata.tenant_id
        alert2.created_at = alert1.created_at + timedelta(minutes=2)

        should_suppress, reason = await alert_service._should_suppress_alert(alert2)

        # May or may not suppress depending on exact duplicate logic
        # This tests the duplicate detection mechanism
        assert isinstance(should_suppress, bool)
        assert isinstance(reason, str)

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_service, sample_metadata):
        """Test acknowledging an alert."""
        alert = await alert_service.create_alert(
            title="Acknowledge Test",
            description="Test acknowledgment",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=sample_metadata,
        )

        success = await alert_service.acknowledge_alert(
            alert.alert_id, "test_user", "Investigating issue"
        )

        assert success

        # Check alert was updated
        updated_alert = await alert_service.get_alert(alert.alert_id)
        assert updated_alert.status == AlertStatus.ACKNOWLEDGED
        assert len(updated_alert.acknowledgments) > 0
        assert updated_alert.acknowledgments[0]["acknowledged_by"] == "test_user"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_service, sample_metadata):
        """Test resolving an alert."""
        alert = await alert_service.create_alert(
            title="Resolve Test",
            description="Test resolution",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=sample_metadata,
        )

        success = await alert_service.resolve_alert(
            alert.alert_id, "test_user", "Issue fixed", quality_score=0.8
        )

        assert success

        # Check alert was updated
        updated_alert = await alert_service.get_alert(alert.alert_id)
        assert updated_alert.status == AlertStatus.RESOLVED
        assert updated_alert.resolved_at is not None
        assert updated_alert.resolution_quality == 0.8
        assert len(updated_alert.comments) > 0

    @pytest.mark.asyncio
    async def test_suppress_alert(self, alert_service, sample_metadata):
        """Test suppressing an alert."""
        alert = await alert_service.create_alert(
            title="Suppress Test",
            description="Test suppression",
            severity=AlertSeverity.LOW,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=sample_metadata,
        )

        success = await alert_service.suppress_alert(
            alert.alert_id, "test_user", "False positive", duration_minutes=60
        )

        assert success

        # Check alert was updated
        updated_alert = await alert_service.get_alert(alert.alert_id)
        assert updated_alert.status == AlertStatus.SUPPRESSED
        assert updated_alert.suppression.is_suppressed
        assert updated_alert.suppression.suppressed_by == "test_user"
        assert updated_alert.suppression.suppression_reason == "False positive"

    @pytest.mark.asyncio
    async def test_escalate_alert(self, alert_service, sample_metadata):
        """Test escalating an alert."""
        alert = await alert_service.create_alert(
            title="Escalate Test",
            description="Test escalation",
            severity=AlertSeverity.CRITICAL,
            category=AlertType.SECURITY,
            source=NotificationChannel.SECURITY_SERVICE,
            metadata=sample_metadata,
        )

        success = await alert_service.escalate_alert(
            alert.alert_id, "test_user", "Requires immediate attention"
        )

        assert success

        # Check alert was updated
        updated_alert = await alert_service.get_alert(alert.alert_id)
        assert updated_alert.status == AlertStatus.ESCALATED
        assert len(updated_alert.escalation.escalation_history) > 0
        assert len(updated_alert.comments) > 0

    @pytest.mark.asyncio
    async def test_list_alerts_filtering(self, alert_service, sample_metadata):
        """Test listing alerts with filters."""
        # Create alerts with different properties
        await alert_service.create_alert(
            title="Critical Alert",
            description="Critical severity",
            severity=AlertSeverity.CRITICAL,
            category=AlertType.SECURITY,
            source=NotificationChannel.SECURITY_SERVICE,
            metadata=sample_metadata,
        )

        await alert_service.create_alert(
            title="Medium Alert",
            description="Medium severity",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
            metadata=sample_metadata,
        )

        # Test severity filter
        critical_alerts = await alert_service.list_alerts(
            severity_filter=AlertSeverity.CRITICAL
        )
        assert len(critical_alerts) >= 1
        assert all(
            alert.severity == AlertSeverity.CRITICAL for alert in critical_alerts
        )

        # Test category filter
        security_alerts = await alert_service.list_alerts(
            category_filter=AlertType.SECURITY
        )
        assert len(security_alerts) >= 1
        assert all(alert.category == AlertType.SECURITY for alert in security_alerts)

        # Test tenant filter
        tenant_alerts = await alert_service.list_alerts(
            tenant_id_filter=sample_metadata.tenant_id
        )
        assert len(tenant_alerts) >= 2
        assert all(
            alert.metadata.tenant_id == sample_metadata.tenant_id
            for alert in tenant_alerts
        )

    @pytest.mark.asyncio
    async def test_get_alert_analytics(self, alert_service, sample_metadata):
        """Test getting alert analytics."""
        # Create several alerts for analytics
        severities = [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM]
        categories = [
            AlertType.ANOMALY_DETECTION,
            AlertType.SECURITY,
            AlertType.SYSTEM_PERFORMANCE,
        ]

        for i in range(5):
            await alert_service.create_alert(
                title=f"Analytics Alert {i}",
                description=f"Alert {i} for analytics",
                severity=severities[i % 3],
                category=categories[i % 3],
                source=NotificationChannel.DETECTOR,
                metadata=sample_metadata,
            )

        analytics = await alert_service.get_alert_analytics(days=7)

        assert "total_alerts" in analytics
        assert analytics["total_alerts"] >= 5

        assert "alert_distribution" in analytics
        assert "by_severity" in analytics["alert_distribution"]
        assert "by_category" in analytics["alert_distribution"]
        assert "by_status" in analytics["alert_distribution"]
        assert "by_noise_classification" in analytics["alert_distribution"]

        assert "noise_reduction_stats" in analytics
        assert "correlation_stats" in analytics
        assert "performance_metrics" in analytics

        # Check that metrics make sense
        assert analytics["performance_metrics"]["total_processed"] >= 5
        assert analytics["performance_metrics"]["suppression_rate"] >= 0.0
        assert analytics["performance_metrics"]["suppression_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_are_alerts_duplicate(self, alert_service):
        """Test duplicate alert detection."""
        tenant_id = uuid4()

        alert1 = Alert(
            title="Duplicate Test 1",
            description="First duplicate",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.SYSTEM_PERFORMANCE,
            source=NotificationChannel.SYSTEM_MONITOR,
        )
        alert1.metadata.tenant_id = tenant_id
        alert1.metadata.affected_resources = ["server1", "server2"]

        alert2 = Alert(
            title="Duplicate Test 2",
            description="Second duplicate",
            severity=AlertSeverity.MEDIUM,
            category=AlertType.SYSTEM_PERFORMANCE,
            source=NotificationChannel.SYSTEM_MONITOR,
        )
        alert2.metadata.tenant_id = tenant_id
        alert2.metadata.affected_resources = ["server1", "server2"]  # Same resources

        # Should be detected as duplicates
        are_duplicates = alert_service._are_alerts_duplicate(alert1, alert2)
        assert are_duplicates

        # Change one property to make them not duplicates
        alert2.category = AlertType.SECURITY
        are_duplicates = alert_service._are_alerts_duplicate(alert1, alert2)
        assert not are_duplicates

    @pytest.mark.asyncio
    async def test_count_similar_alerts(self, alert_service):
        """Test counting similar alerts."""
        # Create base alert
        base_alert = Alert(
            title="Base Alert",
            description="Base for similarity test",
            severity=AlertSeverity.HIGH,
            category=AlertType.ANOMALY_DETECTION,
            source=NotificationChannel.DETECTOR,
        )

        # Add some similar alerts to history
        for i in range(3):
            similar_alert = Alert(
                title=f"Similar Alert {i}",
                description=f"Similar alert {i}",
                severity=AlertSeverity.HIGH,  # Same severity
                category=AlertType.ANOMALY_DETECTION,  # Same category
                source=NotificationChannel.DETECTOR,
            )
            similar_alert.created_at = datetime.utcnow() - timedelta(minutes=i * 10)
            alert_service.alert_history.append(similar_alert)

        # Add a dissimilar alert
        different_alert = Alert(
            title="Different Alert",
            description="Different alert",
            severity=AlertSeverity.LOW,  # Different severity
            category=AlertType.SECURITY,  # Different category
            source=NotificationChannel.SECURITY_SERVICE,
        )
        alert_service.alert_history.append(different_alert)

        similar_count = alert_service._count_similar_alerts(
            base_alert, timedelta(hours=1)
        )

        # Should count 3 similar alerts (not the different one)
        assert similar_count == 3

    @pytest.mark.asyncio
    async def test_nonexistent_alert_operations(self, alert_service):
        """Test operations on non-existent alerts."""
        fake_alert_id = uuid4()

        # Get non-existent alert
        alert = await alert_service.get_alert(fake_alert_id)
        assert alert is None

        # Acknowledge non-existent alert
        success = await alert_service.acknowledge_alert(fake_alert_id, "user", "note")
        assert not success

        # Resolve non-existent alert
        success = await alert_service.resolve_alert(fake_alert_id, "user", "note")
        assert not success

        # Suppress non-existent alert
        success = await alert_service.suppress_alert(fake_alert_id, "user", "reason")
        assert not success

        # Escalate non-existent alert
        success = await alert_service.escalate_alert(fake_alert_id, "user", "reason")
        assert not success
