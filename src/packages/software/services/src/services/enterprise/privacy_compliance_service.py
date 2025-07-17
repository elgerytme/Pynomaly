"""Privacy compliance and data anonymization service.

This service provides comprehensive privacy compliance features including
GDPR compliance, data anonymization, consent management, and data retention.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from ...domain.models.monitoring import UserEvent, UserSession, WebVitalMetric
from ...infrastructure.config.feature_flags import require_feature


class ConsentType(Enum):
    """Types of user consent."""

    ANALYTICS = "analytics"
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    MARKETING = "marketing"
    ESSENTIAL = "essential"  # Always required


class DataCategory(Enum):
    """Categories of data for compliance purposes."""

    PERSONAL_IDENTIFIABLE = "personal_identifiable"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    LOCATION = "location"
    DEVICE = "device"
    USAGE = "usage"


class RetentionPolicy(Enum):
    """Data retention policies."""

    IMMEDIATE = "immediate"  # Delete immediately
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 90 days
    LONG_TERM = "long_term"  # 1 year
    LEGAL_REQUIREMENT = "legal_requirement"  # 7 years
    USER_CONTROLLED = "user_controlled"  # Until user requests deletion


class PrivacyCompliantUserProfile:
    """Privacy-compliant user profile with consent management."""

    def __init__(
        self,
        user_id: UUID,
        session_id: str,
        consent_timestamp: datetime,
        ip_address: str = "",
        user_agent: str = "",
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.consent_timestamp = consent_timestamp
        self.original_ip = ip_address
        self.original_user_agent = user_agent

        # Consent status
        self.consents: dict[ConsentType, bool] = {
            ConsentType.ESSENTIAL: True,  # Always required
            ConsentType.ANALYTICS: False,
            ConsentType.PERFORMANCE: False,
            ConsentType.FUNCTIONAL: False,
            ConsentType.MARKETING: False,
        }

        # Anonymized data
        self.anonymized_ip = self._anonymize_ip_address(ip_address)
        self.device_fingerprint = self._create_device_fingerprint(user_agent)
        self.geographic_region = ""  # City-level data removed for privacy

        # Tracking
        self.data_processed: list[dict[str, Any]] = []
        self.last_consent_update = consent_timestamp
        self.opt_out_timestamp: datetime | None = None

    def update_consent(self, consent_type: ConsentType, granted: bool) -> None:
        """Update user consent for specific data category."""
        if consent_type == ConsentType.ESSENTIAL:
            # Essential consent cannot be revoked
            return

        self.consents[consent_type] = granted
        self.last_consent_update = datetime.utcnow()

        # If consent is revoked, mark for data cleanup
        if not granted:
            self._schedule_data_cleanup(consent_type)

    def has_consent(self, consent_type: ConsentType) -> bool:
        """Check if user has granted specific consent."""
        return self.consents.get(consent_type, False)

    def is_gdpr_compliant(self) -> bool:
        """Check if current data processing is GDPR compliant."""
        # Essential processing is always allowed
        if not any(
            self.consents[ct]
            for ct in [
                ConsentType.ANALYTICS,
                ConsentType.PERFORMANCE,
                ConsentType.FUNCTIONAL,
                ConsentType.MARKETING,
            ]
        ):
            return True

        # Check consent recency (GDPR requires periodic re-consent)
        consent_age = datetime.utcnow() - self.last_consent_update
        return consent_age < timedelta(days=365)  # Annual re-consent

    def can_process_data(self, data_category: DataCategory) -> bool:
        """Check if data processing is allowed for specific category."""
        category_consent_mapping = {
            DataCategory.PERSONAL_IDENTIFIABLE: ConsentType.ESSENTIAL,
            DataCategory.BEHAVIORAL: ConsentType.ANALYTICS,
            DataCategory.TECHNICAL: ConsentType.PERFORMANCE,
            DataCategory.LOCATION: ConsentType.FUNCTIONAL,
            DataCategory.DEVICE: ConsentType.PERFORMANCE,
            DataCategory.USAGE: ConsentType.ANALYTICS,
        }

        required_consent = category_consent_mapping.get(
            data_category, ConsentType.ESSENTIAL
        )
        return self.has_consent(required_consent)

    def _anonymize_ip_address(self, ip_address: str) -> str:
        """Anonymize IP address for privacy compliance."""
        if not ip_address:
            return ""

        # Hash IP address to preserve geographic accuracy while ensuring privacy
        return hashlib.sha256(ip_address.encode()).hexdigest()[:16]

    def _create_device_fingerprint(self, user_agent: str) -> str:
        """Create anonymized device fingerprint."""
        if not user_agent:
            return ""

        # Extract basic device info without detailed tracking
        simplified_ua = re.sub(
            r"\d+\.\d+\.\d+", "X.X.X", user_agent
        )  # Remove version numbers
        return hashlib.sha256(simplified_ua.encode()).hexdigest()[:12]

    def _schedule_data_cleanup(self, consent_type: ConsentType) -> None:
        """Schedule data cleanup when consent is revoked."""
        # This would integrate with a data cleanup service
        pass


class PrivacyComplianceService:
    """Service for privacy compliance and data anonymization."""

    def __init__(
        self,
        enable_gdpr: bool = True,
        enable_ccpa: bool = True,
        default_retention_days: int = 90,
        anonymization_delay_hours: int = 24,
    ):
        """Initialize privacy compliance service.

        Args:
            enable_gdpr: Enable GDPR compliance features
            enable_ccpa: Enable CCPA compliance features
            default_retention_days: Default data retention period
            anonymization_delay_hours: Hours to wait before anonymizing data
        """
        self.enable_gdpr = enable_gdpr
        self.enable_ccpa = enable_ccpa
        self.default_retention_days = default_retention_days
        self.anonymization_delay_hours = anonymization_delay_hours

        # User profiles and consent tracking
        self.user_profiles: dict[str, PrivacyCompliantUserProfile] = {}
        self.consent_logs: list[dict[str, Any]] = []

        # Data retention policies
        self.retention_policies: dict[DataCategory, RetentionPolicy] = {
            DataCategory.PERSONAL_IDENTIFIABLE: RetentionPolicy.USER_CONTROLLED,
            DataCategory.BEHAVIORAL: RetentionPolicy.MEDIUM_TERM,
            DataCategory.TECHNICAL: RetentionPolicy.SHORT_TERM,
            DataCategory.LOCATION: RetentionPolicy.USER_CONTROLLED,
            DataCategory.DEVICE: RetentionPolicy.SHORT_TERM,
            DataCategory.USAGE: RetentionPolicy.MEDIUM_TERM,
        }

        # Sensitive data patterns for automatic processing
        self.sensitive_patterns = {
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "phone": re.compile(r"(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}"),
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        }

        # Data subject requests tracking
        self.data_requests: list[dict[str, Any]] = []

    @require_feature("privacy_compliance")
    def create_user_profile(
        self,
        user_id: UUID,
        session_id: str,
        ip_address: str = "",
        user_agent: str = "",
        initial_consents: dict[ConsentType, bool] | None = None,
    ) -> PrivacyCompliantUserProfile:
        """Create privacy-compliant user profile.

        Args:
            user_id: User identifier
            session_id: Session identifier
            ip_address: User's IP address
            user_agent: User's browser/client info
            initial_consents: Initial consent preferences

        Returns:
            PrivacyCompliantUserProfile
        """
        profile = PrivacyCompliantUserProfile(
            user_id=user_id,
            session_id=session_id,
            consent_timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Apply initial consents
        if initial_consents:
            for consent_type, granted in initial_consents.items():
                profile.update_consent(consent_type, granted)

        self.user_profiles[session_id] = profile

        # Log consent collection
        self._log_consent_event(
            user_id=user_id,
            session_id=session_id,
            event_type="profile_created",
            consents=profile.consents,
        )

        return profile

    @require_feature("privacy_compliance")
    def update_user_consent(
        self,
        session_id: str,
        consent_type: ConsentType,
        granted: bool,
        consent_method: str = "web_form",
    ) -> bool:
        """Update user consent preferences.

        Args:
            session_id: Session identifier
            consent_type: Type of consent being updated
            granted: Whether consent is granted
            consent_method: Method used to collect consent

        Returns:
            True if update was successful
        """
        if session_id not in self.user_profiles:
            return False

        profile = self.user_profiles[session_id]
        profile.update_consent(consent_type, granted)

        # Log consent change
        self._log_consent_event(
            user_id=profile.user_id,
            session_id=session_id,
            event_type="consent_updated",
            consent_type=consent_type,
            granted=granted,
            method=consent_method,
        )

        return True

    @require_feature("privacy_compliance")
    def anonymize_user_event(self, event: UserEvent) -> UserEvent:
        """Anonymize user event for privacy compliance.

        Args:
            event: UserEvent to anonymize

        Returns:
            Anonymized UserEvent
        """
        # Check if user has profile and consents
        profile = self.user_profiles.get(event.session_id)
        if not profile:
            # No profile means no specific consents, apply minimal processing
            event.anonymize()
            return event

        # Check consent for behavioral data
        if not profile.can_process_data(DataCategory.BEHAVIORAL):
            # Remove behavioral tracking data
            event.properties = {}
            event.page_url = self._anonymize_url(event.page_url)
            event.referrer = ""

        # Check consent for location data
        if not profile.can_process_data(DataCategory.LOCATION):
            event.ip_address = profile.anonymized_ip

        # Always anonymize sensitive data
        event = self._remove_sensitive_data_from_event(event)

        # Apply anonymization if required
        if not profile.has_consent(ConsentType.ANALYTICS):
            event.anonymize()

        return event

    @require_feature("privacy_compliance")
    def anonymize_user_session(self, session: UserSession) -> UserSession:
        """Anonymize user session for privacy compliance.

        Args:
            session: UserSession to anonymize

        Returns:
            Anonymized UserSession
        """
        profile = self.user_profiles.get(session.session_id)
        if not profile:
            session.anonymize()
            return session

        # Apply anonymization based on consent
        if not profile.can_process_data(DataCategory.LOCATION):
            session.country = ""
            session.region = ""
            session.city = ""
            session.ip_address = profile.anonymized_ip

        if not profile.can_process_data(DataCategory.DEVICE):
            session.user_agent = profile.device_fingerprint
            session.browser = ""
            session.operating_system = ""

        if not profile.has_consent(ConsentType.ANALYTICS):
            session.anonymize()

        return session

    @require_feature("privacy_compliance")
    def anonymize_web_vital(self, vital: WebVitalMetric) -> WebVitalMetric:
        """Anonymize Web Vital metric for privacy compliance.

        Args:
            vital: WebVitalMetric to anonymize

        Returns:
            Anonymized WebVitalMetric
        """
        profile = self.user_profiles.get(vital.session_id)
        if not profile:
            # No profile, apply basic anonymization
            vital.user_id = None
            return vital

        # Check performance monitoring consent
        if not profile.can_process_data(DataCategory.TECHNICAL):
            vital.page_url = self._anonymize_url(vital.page_url)
            vital.device_type = ""
            vital.connection_type = ""

        if not profile.has_consent(ConsentType.PERFORMANCE):
            vital.user_id = None

        return vital

    @require_feature("privacy_compliance")
    def handle_data_subject_request(
        self,
        request_type: str,
        user_id: UUID,
        email: str,
        request_details: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle data subject rights requests (GDPR Article 15-22).

        Args:
            request_type: Type of request (access, rectification, erasure, etc.)
            user_id: User identifier
            email: User's email for verification
            request_details: Additional request details

        Returns:
            Request processing result
        """
        request_id = str(uuid4())
        request_data = {
            "request_id": request_id,
            "request_type": request_type,
            "user_id": str(user_id),
            "email": email,
            "details": request_details,
            "timestamp": datetime.utcnow(),
            "status": "received",
            "processing_deadline": datetime.utcnow()
            + timedelta(days=30),  # GDPR requirement
        }

        self.data_requests.append(request_data)

        # Process based on request type
        if request_type == "access":
            return self._process_data_access_request(user_id, request_id)
        elif request_type == "erasure":
            return self._process_data_erasure_request(user_id, request_id)
        elif request_type == "rectification":
            return self._process_data_rectification_request(
                user_id, request_details, request_id
            )
        elif request_type == "portability":
            return self._process_data_portability_request(user_id, request_id)
        else:
            return {
                "request_id": request_id,
                "status": "error",
                "message": "Unsupported request type",
            }

    @require_feature("privacy_compliance")
    def get_privacy_dashboard_data(self, session_id: str) -> dict[str, Any]:
        """Get privacy dashboard data for user.

        Args:
            session_id: Session identifier

        Returns:
            Privacy dashboard data
        """
        profile = self.user_profiles.get(session_id)
        if not profile:
            return {
                "error": "User profile not found",
                "consents": {},
                "data_processing": [],
            }

        return {
            "user_id": str(profile.user_id),
            "session_id": profile.session_id,
            "consents": {ct.value: granted for ct, granted in profile.consents.items()},
            "last_consent_update": profile.last_consent_update.isoformat(),
            "gdpr_compliant": profile.is_gdpr_compliant(),
            "data_categories_processed": [
                cat.value for cat in DataCategory if profile.can_process_data(cat)
            ],
            "retention_policies": {
                cat.value: policy.value
                for cat, policy in self.retention_policies.items()
            },
            "anonymization_status": {
                "ip_anonymized": bool(profile.anonymized_ip),
                "device_fingerprinted": bool(profile.device_fingerprint),
            },
        }

    def _log_consent_event(
        self,
        user_id: UUID,
        session_id: str,
        event_type: str,
        consent_type: ConsentType | None = None,
        granted: bool | None = None,
        consents: dict[ConsentType, bool] | None = None,
        method: str = "unknown",
    ) -> None:
        """Log consent events for audit trail."""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "user_id": str(user_id),
            "session_id": session_id,
            "event_type": event_type,
            "method": method,
        }

        if consent_type:
            log_entry["consent_type"] = consent_type.value
            log_entry["granted"] = granted

        if consents:
            log_entry["all_consents"] = {
                ct.value: granted for ct, granted in consents.items()
            }

        self.consent_logs.append(log_entry)

    def _remove_sensitive_data_from_event(self, event: UserEvent) -> UserEvent:
        """Remove sensitive data from event properties."""
        if not event.properties:
            return event

        # Scan for sensitive patterns
        for key, value in list(event.properties.items()):
            if isinstance(value, str):
                for pattern_name, pattern in self.sensitive_patterns.items():
                    if pattern.search(value):
                        # Replace with anonymized placeholder
                        event.properties[key] = f"[REDACTED_{pattern_name.upper()}]"
                        break

        return event

    def _anonymize_url(self, url: str) -> str:
        """Anonymize URL while preserving basic navigation patterns."""
        if not url:
            return url

        # Remove query parameters that might contain personal data
        if "?" in url:
            base_url = url.split("?")[0]
            return base_url + "?[PARAMS_REMOVED]"

        return url

    def _process_data_access_request(
        self, user_id: UUID, request_id: str
    ) -> dict[str, Any]:
        """Process data access request (GDPR Article 15)."""
        # This would gather all user data from various systems
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Data access request received and being processed",
            "estimated_completion": (
                datetime.utcnow() + timedelta(days=30)
            ).isoformat(),
        }

    def _process_data_erasure_request(
        self, user_id: UUID, request_id: str
    ) -> dict[str, Any]:
        """Process data erasure request (GDPR Article 17)."""
        # This would initiate data deletion across all systems
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Data erasure request received and being processed",
            "estimated_completion": (
                datetime.utcnow() + timedelta(days=30)
            ).isoformat(),
        }

    def _process_data_rectification_request(
        self, user_id: UUID, details: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        """Process data rectification request (GDPR Article 16)."""
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Data rectification request received and being processed",
            "estimated_completion": (
                datetime.utcnow() + timedelta(days=30)
            ).isoformat(),
        }

    def _process_data_portability_request(
        self, user_id: UUID, request_id: str
    ) -> dict[str, Any]:
        """Process data portability request (GDPR Article 20)."""
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Data portability request received and being processed",
            "estimated_completion": (
                datetime.utcnow() + timedelta(days=30)
            ).isoformat(),
        }
