"""Test suite for advanced threat detection capabilities."""

import time
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from pynomaly.infrastructure.security.advanced_threat_detection import (
    AdvancedBehaviorAnalyzer,
    BehaviorProfile,
    DataExfiltrationDetector,
    ThreatIntelligence,
    ThreatIntelligenceDetector,
    ThreatIntelligenceSource,
    create_advanced_threat_detectors,
)
from pynomaly.infrastructure.security.audit_logger import SecurityEventType
from pynomaly.infrastructure.security.security_monitor import AlertType, ThreatLevel


class TestAdvancedBehaviorAnalyzer:
    """Test cases for AdvancedBehaviorAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            return AdvancedBehaviorAnalyzer()

    @pytest.mark.asyncio
    async def test_new_user_no_alert(self, analyzer):
        """Test that new users don't trigger alerts."""
        event_data = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": "new_user",
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
        }

        alert = await analyzer.analyze(event_data)
        assert alert is None

    @pytest.mark.asyncio
    async def test_insufficient_samples_no_alert(self, analyzer):
        """Test that users with insufficient samples don't trigger alerts."""
        user_id = "test_user"

        # Add some login events but not enough for analysis
        for i in range(20):  # Less than min_samples_for_profile (50)
            event_data = {
                "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
                "user_id": user_id,
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
            }
            await analyzer.analyze(event_data)

        # Now test with anomalous event
        anomalous_event = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": user_id,
            "ip_address": "10.0.0.50",  # Different IP
            "user_agent": "Chrome/90.0",
        }

        alert = await analyzer.analyze(anomalous_event)
        assert alert is None  # Should not alert due to insufficient samples

    @pytest.mark.asyncio
    async def test_behavioral_anomaly_detection(self, analyzer):
        """Test detection of behavioral anomalies."""
        user_id = "established_user"

        # Build a behavior profile with enough samples
        for i in range(60):  # More than min_samples_for_profile
            event_data = {
                "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
                "user_id": user_id,
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
            }
            await analyzer.analyze(event_data)

        # Now test with anomalous event (different IP)
        anomalous_event = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": user_id,
            "ip_address": "10.0.0.50",  # Completely different IP
            "user_agent": "Mozilla/5.0",
        }

        alert = await analyzer.analyze(anomalous_event)
        assert alert is not None
        assert alert.alert_type == AlertType.UNUSUAL_BEHAVIOR
        assert alert.user_id == user_id
        assert "10.0.0.50" in str(alert.evidence)

    @pytest.mark.asyncio
    async def test_api_usage_anomaly(self, analyzer):
        """Test detection of API usage anomalies."""
        user_id = "api_user"

        # Build profile with typical API usage
        for i in range(60):
            event_data = {
                "event_type": SecurityEventType.API_REQUEST,
                "user_id": user_id,
                "endpoint": "/api/detect",
            }
            await analyzer.analyze(event_data)

        # Test unusual endpoint access
        unusual_event = {
            "event_type": SecurityEventType.API_REQUEST,
            "user_id": user_id,
            "endpoint": "/api/admin/delete",  # Unusual endpoint
        }

        alert = await analyzer.analyze(unusual_event)
        assert alert is not None
        assert "unusual endpoint" in str(alert.evidence).lower()

    def test_configuration_management(self, analyzer):
        """Test configuration getter and setter methods."""
        config = analyzer.get_configuration()
        assert "learning_period_days" in config
        assert "anomaly_threshold" in config
        assert "min_samples_for_profile" in config

        # Update configuration
        new_config = {
            "learning_period_days": 21,
            "anomaly_threshold": 0.8,
            "min_samples_for_profile": 75,
        }
        analyzer.update_configuration(new_config)

        assert analyzer.learning_period_days == 21
        assert analyzer.anomaly_threshold == 0.8
        assert analyzer.min_samples_for_profile == 75


class TestThreatIntelligenceDetector:
    """Test cases for ThreatIntelligenceDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            return ThreatIntelligenceDetector()

    @pytest.mark.asyncio
    async def test_malicious_ip_detection(self, detector):
        """Test detection of known malicious IPs."""
        event_data = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": "test_user",
            "ip_address": "192.168.1.100",  # This should be in the initial malicious IPs
        }

        alert = await detector.analyze(event_data)
        assert alert is not None
        assert alert.alert_type == AlertType.SYSTEM_COMPROMISE
        assert alert.threat_level == ThreatLevel.CRITICAL
        assert "known malicious ip" in alert.title.lower()

    @pytest.mark.asyncio
    async def test_suspicious_user_agent_detection(self, detector):
        """Test detection of suspicious user agents."""
        event_data = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": "test_user",
            "ip_address": "10.0.0.1",
            "user_agent": "sqlmap/1.0",  # Suspicious tool
        }

        alert = await detector.analyze(event_data)
        assert alert is not None
        assert alert.alert_type == AlertType.MALWARE_DETECTED
        assert alert.threat_level == ThreatLevel.HIGH
        assert "malicious tool" in alert.title.lower()

    @pytest.mark.asyncio
    async def test_safe_request_no_alert(self, detector):
        """Test that safe requests don't trigger alerts."""
        event_data = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": "test_user",
            "ip_address": "10.0.0.1",  # Clean IP
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",  # Normal browser
        }

        alert = await detector.analyze(event_data)
        assert alert is None

    def test_add_threat_intelligence(self, detector):
        """Test adding new threat intelligence."""
        threat_intel = ThreatIntelligence(
            source=ThreatIntelligenceSource.KNOWN_BAD_IPS,
            indicator="123.45.67.89",
            threat_type="botnet",
            confidence=0.95,
            first_seen=datetime.now(UTC),
            last_updated=datetime.now(UTC),
            metadata={"source": "test"},
        )

        initial_count = len(
            detector.threat_feeds[ThreatIntelligenceSource.KNOWN_BAD_IPS]
        )
        detector.add_threat_intelligence(threat_intel)

        assert (
            len(detector.threat_feeds[ThreatIntelligenceSource.KNOWN_BAD_IPS])
            == initial_count + 1
        )
        assert (
            threat_intel
            in detector.threat_feeds[ThreatIntelligenceSource.KNOWN_BAD_IPS]
        )

    def test_configuration_management(self, detector):
        """Test configuration methods."""
        config = detector.get_configuration()
        assert "update_interval" in config
        assert "confidence_threshold" in config
        assert "feed_counts" in config

        # Update configuration
        new_config = {"update_interval": 7200, "confidence_threshold": 0.8}
        detector.update_configuration(new_config)

        assert detector.update_interval == 7200
        assert detector.confidence_threshold == 0.8


class TestDataExfiltrationDetector:
    """Test cases for DataExfiltrationDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            return DataExfiltrationDetector()

    @pytest.mark.asyncio
    async def test_large_data_access_detection(self, detector):
        """Test detection of large data access."""
        user_id = "data_user"
        current_time = time.time()

        # Simulate large data access
        event_data = {
            "event_type": SecurityEventType.DATA_ACCESS,
            "user_id": user_id,
            "ip_address": "192.168.1.100",
            "endpoint": "/api/data/export",
            "details": {
                "data_size_bytes": 150 * 1024 * 1024  # 150MB, above threshold
            },
        }

        # First access shouldn't trigger (need at least 2 accesses)
        alert = await detector.analyze(event_data)
        assert alert is None

        # Second access should trigger
        alert = await detector.analyze(event_data)
        assert alert is not None
        assert alert.alert_type == AlertType.DATA_EXFILTRATION
        assert alert.threat_level == ThreatLevel.HIGH
        assert alert.user_id == user_id

    @pytest.mark.asyncio
    async def test_high_request_count_detection(self, detector):
        """Test detection of high request count."""
        user_id = "busy_user"

        # Simulate many small requests
        for i in range(60):  # Above threshold of 50
            event_data = {
                "event_type": SecurityEventType.DATA_ACCESS,
                "user_id": user_id,
                "ip_address": "192.168.1.100",
                "endpoint": f"/api/data/item_{i}",
                "details": {
                    "data_size_bytes": 1024  # 1KB each
                },
            }

            alert = await detector.analyze(event_data)
            if alert:
                # Should trigger somewhere in the sequence
                assert alert.alert_type == AlertType.DATA_EXFILTRATION
                assert alert.user_id == user_id
                break
        else:
            pytest.fail("Expected data exfiltration alert was not triggered")

    @pytest.mark.asyncio
    async def test_normal_usage_no_alert(self, detector):
        """Test that normal usage doesn't trigger alerts."""
        event_data = {
            "event_type": SecurityEventType.DATA_ACCESS,
            "user_id": "normal_user",
            "ip_address": "192.168.1.100",
            "endpoint": "/api/data/read",
            "details": {
                "data_size_bytes": 1024  # 1KB, well below threshold
            },
        }

        # Multiple normal accesses
        for _ in range(10):
            alert = await detector.analyze(event_data)
            assert alert is None

    def test_configuration_management(self, detector):
        """Test configuration methods."""
        config = detector.get_configuration()
        assert "size_threshold_mb" in config
        assert "time_window_seconds" in config
        assert "request_count_threshold" in config

        # Update configuration
        new_config = {
            "size_threshold_mb": 200,
            "time_window_seconds": 600,
            "request_count_threshold": 100,
        }
        detector.update_configuration(new_config)

        assert detector.size_threshold_mb == 200
        assert detector.time_window == 600
        assert detector.request_count_threshold == 100


class TestThreatIntelligence:
    """Test cases for ThreatIntelligence data class."""

    def test_threat_intelligence_creation(self):
        """Test creation of ThreatIntelligence objects."""
        current_time = datetime.now(UTC)

        threat_intel = ThreatIntelligence(
            source=ThreatIntelligenceSource.MALWARE_DOMAINS,
            indicator="evil.com",
            threat_type="malware_domain",
            confidence=0.9,
            first_seen=current_time,
            last_updated=current_time,
            metadata={"category": "trojan"},
        )

        assert threat_intel.source == ThreatIntelligenceSource.MALWARE_DOMAINS
        assert threat_intel.indicator == "evil.com"
        assert threat_intel.threat_type == "malware_domain"
        assert threat_intel.confidence == 0.9
        assert threat_intel.metadata["category"] == "trojan"


class TestBehaviorProfile:
    """Test cases for BehaviorProfile data class."""

    def test_behavior_profile_creation(self):
        """Test creation of BehaviorProfile objects."""
        profile = BehaviorProfile(user_id="test_user")

        assert profile.user_id == "test_user"
        assert isinstance(profile.typical_login_hours, set)
        assert isinstance(profile.typical_ips, set)
        assert isinstance(profile.typical_user_agents, set)
        assert isinstance(profile.typical_endpoints, dict)
        assert profile.avg_requests_per_hour == 0.0
        assert profile.confidence_score == 0.0
        assert profile.sample_count == 0

    def test_behavior_profile_with_data(self):
        """Test BehaviorProfile with initial data."""
        profile = BehaviorProfile(
            user_id="active_user",
            typical_login_hours={9, 10, 11, 14, 15, 16},
            typical_ips={"192.168.1.100", "192.168.1.101"},
            typical_endpoints={"/api/detect": 150, "/api/train": 30},
        )

        assert len(profile.typical_login_hours) == 6
        assert len(profile.typical_ips) == 2
        assert profile.typical_endpoints["/api/detect"] == 150


class TestFactoryFunction:
    """Test cases for the factory function."""

    def test_create_advanced_threat_detectors(self):
        """Test factory function creates all detectors."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            detectors = create_advanced_threat_detectors()

        assert len(detectors) == 3
        detector_names = [d.name for d in detectors]

        assert "advanced_behavior" in detector_names
        assert "threat_intelligence" in detector_names
        assert "data_exfiltration" in detector_names

        # Verify all are ThreatDetector instances
        for detector in detectors:
            assert hasattr(detector, "analyze")
            assert hasattr(detector, "get_configuration")
            assert hasattr(detector, "update_configuration")


class TestIntegration:
    """Integration tests for advanced threat detection."""

    @pytest.mark.asyncio
    async def test_multiple_detector_analysis(self):
        """Test that multiple detectors can analyze the same event."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            detectors = create_advanced_threat_detectors()

        # Event that might trigger multiple detectors
        event_data = {
            "event_type": SecurityEventType.AUTH_LOGIN_SUCCESS,
            "user_id": "suspicious_user",
            "ip_address": "192.168.1.100",  # Known bad IP
            "user_agent": "sqlmap/1.0",  # Suspicious tool
        }

        alerts = []
        for detector in detectors:
            alert = await detector.analyze(event_data)
            if alert:
                alerts.append(alert)

        # Should have at least one alert (from threat intelligence)
        assert len(alerts) >= 1

        # Check that alert IDs are unique
        alert_ids = [alert.alert_id for alert in alerts]
        assert len(alert_ids) == len(set(alert_ids))

    @pytest.mark.asyncio
    async def test_detector_configuration_isolation(self):
        """Test that detector configurations are isolated."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            detector1 = AdvancedBehaviorAnalyzer()
            detector2 = AdvancedBehaviorAnalyzer()

        # Modify configuration of one detector
        detector1.update_configuration({"anomaly_threshold": 0.9})

        # Other detector should be unaffected
        assert detector1.anomaly_threshold == 0.9
        assert detector2.anomaly_threshold == 0.7  # Default value

    def test_detector_enable_disable(self):
        """Test enabling and disabling detectors."""
        with patch(
            "pynomaly.infrastructure.security.advanced_threat_detection.get_threat_detection_manager"
        ) as mock_manager:
            # Mock the configuration manager to return None (use defaults)
            mock_manager.return_value.get_detector_config.return_value = None
            detector = AdvancedBehaviorAnalyzer()

        assert detector.enabled is True  # Default state

        detector.enabled = False
        assert detector.enabled is False

        detector.enabled = True
        assert detector.enabled is True
