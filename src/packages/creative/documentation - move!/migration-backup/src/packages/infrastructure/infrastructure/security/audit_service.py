"""Audit service for comprehensive security and compliance logging."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from monorepo.domain.models.security import (
    ActionType,
    AuditEvent,
    ComplianceFramework,
    ComplianceReport,
    SecurityIncident,
    SecurityPolicy,
)


class AuditService:
    """Comprehensive audit service for security and compliance tracking."""

    def __init__(self, security_policy: SecurityPolicy):
        self.security_policy = security_policy
        self.logger = logging.getLogger(__name__)

        # Audit storage (would be replaced with persistent storage in production)
        self.audit_events: dict[UUID, AuditEvent] = {}
        self.security_incidents: dict[UUID, SecurityIncident] = {}
        self.compliance_reports: dict[UUID, ComplianceReport] = {}

        # Event indexes for efficient querying
        self.events_by_user: dict[UUID, list[UUID]] = {}
        self.events_by_type: dict[ActionType, list[UUID]] = {}
        self.events_by_timestamp: list[
            tuple[datetime, UUID]
        ] = []  # (timestamp, event_id)

        # Real-time monitoring
        self.security_alerts: list[dict[str, Any]] = []
        self.compliance_violations: list[dict[str, Any]] = []

        # Background tasks
        self.monitoring_tasks: set[asyncio.Task] = set()

        self.logger.info("Audit service initialized")

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._security_monitoring_loop()),
            asyncio.create_task(self._compliance_monitoring_loop()),
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self._alert_processing_loop()),
        ]

        self.monitoring_tasks.update(tasks)

        self.logger.info("Started audit monitoring tasks")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""

        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        self.logger.info("Stopped audit monitoring tasks")

    async def log_event(
        self,
        user_id: UUID | None,
        action: ActionType,
        resource_type: str,
        success: bool,
        username: str | None = None,
        resource_id: str | None = None,
        resource_name: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        additional_data: dict[str, Any] | None = None,
        security_level: str = "INFO",
        compliance_relevant: bool = None,
    ) -> AuditEvent:
        """Log audit event with comprehensive metadata."""

        # Determine compliance relevance if not specified
        if compliance_relevant is None:
            compliance_relevant = self._is_compliance_relevant(action, security_level)

        event = AuditEvent(
            event_id=uuid4(),
            user_id=user_id,
            username=username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            success=success,
            additional_data=additional_data or {},
            security_level=security_level,
            compliance_relevant=compliance_relevant,
            server_timestamp=datetime.utcnow(),
        )

        # Store event
        self.audit_events[event.event_id] = event

        # Update indexes
        if user_id:
            if user_id not in self.events_by_user:
                self.events_by_user[user_id] = []
            self.events_by_user[user_id].append(event.event_id)

        if action not in self.events_by_type:
            self.events_by_type[action] = []
        self.events_by_type[action].append(event.event_id)

        self.events_by_timestamp.append((event.timestamp, event.event_id))

        # Real-time security analysis
        await self._analyze_security_event(event)

        # Compliance analysis
        if compliance_relevant:
            await self._analyze_compliance_event(event)

        self.logger.debug(
            f"Logged audit event: {action.value} by {username or 'system'}"
        )
        return event

    async def create_security_incident(
        self,
        incident_type: str,
        title: str,
        description: str,
        severity: str = "medium",
        affected_systems: list[str] | None = None,
        affected_users: list[UUID] | None = None,
        detected_by: UUID | None = None,
    ) -> SecurityIncident:
        """Create security incident record."""

        incident = SecurityIncident(
            incident_id=uuid4(),
            incident_type=incident_type,
            title=title,
            description=description,
            severity=severity,
            affected_systems=affected_systems or [],
            affected_users=affected_users or [],
            detected_by=detected_by,
            detection_method="manual" if detected_by else "automated",
        )

        self.security_incidents[incident.incident_id] = incident

        # Log incident creation
        await self.log_event(
            user_id=detected_by,
            action=ActionType.SYSTEM_MAINTENANCE,  # Using as closest match
            resource_type="security_incident",
            resource_id=str(incident.incident_id),
            resource_name=title,
            success=True,
            additional_data={
                "incident_type": incident_type,
                "severity": severity,
                "affected_systems_count": len(incident.affected_systems),
                "affected_users_count": len(incident.affected_users),
            },
            security_level="CRITICAL"
            if severity in ["high", "critical"]
            else "WARNING",
            compliance_relevant=True,
        )

        # Generate alert for high/critical incidents
        if severity in ["high", "critical"]:
            await self._generate_security_alert(incident)

        self.logger.warning(
            f"Security incident created: {title} (Severity: {severity})"
        )
        return incident

    async def query_events(
        self,
        user_id: UUID | None = None,
        action_types: list[ActionType] | None = None,
        resource_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        success_only: bool | None = None,
        security_levels: list[str] | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query audit events with filters."""

        events = []

        # Get candidate events based on most selective filter
        if user_id:
            candidate_event_ids = self.events_by_user.get(user_id, [])
        elif action_types and len(action_types) == 1:
            candidate_event_ids = self.events_by_type.get(action_types[0], [])
        else:
            # Use all events (would be optimized with better indexing in production)
            candidate_event_ids = list(self.audit_events.keys())

        for event_id in candidate_event_ids:
            event = self.audit_events.get(event_id)
            if not event:
                continue

            # Apply filters
            if action_types and event.action not in action_types:
                continue

            if resource_type and event.resource_type != resource_type:
                continue

            if start_time and event.timestamp < start_time:
                continue

            if end_time and event.timestamp > end_time:
                continue

            if success_only is not None and event.success != success_only:
                continue

            if security_levels and event.security_level not in security_levels:
                continue

            events.append(event)

            if len(events) >= limit:
                break

        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events[:limit]

    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime,
        generated_by: UUID,
    ) -> ComplianceReport:
        """Generate compliance assessment report."""

        report = ComplianceReport(
            report_id=uuid4(),
            framework=framework,
            assessment_period_start=start_date,
            assessment_period_end=end_date,
            generated_by=generated_by,
        )

        # Query relevant events for the period
        events = await self.query_events(
            start_time=start_date,
            end_time=end_date,
        )

        # Framework-specific assessments
        if framework == ComplianceFramework.GDPR:
            await self._assess_gdpr_compliance(report, events)
        elif framework == ComplianceFramework.HIPAA:
            await self._assess_hipaa_compliance(report, events)
        elif framework == ComplianceFramework.SOX:
            await self._assess_sox_compliance(report, events)
        elif framework == ComplianceFramework.SOC2:
            await self._assess_soc2_compliance(report, events)

        # Calculate overall compliance score
        if report.total_controls > 0:
            report.overall_compliance_score = (
                report.compliant_controls / report.total_controls
            ) * 100

        # Determine overall risk level
        critical_failures = len(report.critical_findings)
        high_failures = len(report.high_findings)

        if critical_failures > 0:
            report.overall_risk_level = "critical"
        elif high_failures > 5:
            report.overall_risk_level = "high"
        elif high_failures > 0 or len(report.medium_findings) > 10:
            report.overall_risk_level = "medium"
        else:
            report.overall_risk_level = "low"

        self.compliance_reports[report.report_id] = report

        # Log report generation
        await self.log_event(
            user_id=generated_by,
            action=ActionType.SYSTEM_BACKUP,  # Using as closest match for reporting
            resource_type="compliance_report",
            resource_id=str(report.report_id),
            resource_name=f"{framework.value}_report",
            success=True,
            additional_data={
                "framework": framework.value,
                "compliance_score": report.overall_compliance_score,
                "risk_level": report.overall_risk_level,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
            },
            compliance_relevant=True,
        )

        self.logger.info(
            f"Generated {framework.value} compliance report: {report.overall_compliance_score:.1f}% compliant"
        )
        return report

    async def get_security_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get security metrics for monitoring dashboard."""

        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=30)
        if not end_time:
            end_time = datetime.utcnow()

        events = await self.query_events(start_time=start_time, end_time=end_time)

        # Calculate metrics
        total_events = len(events)
        security_events = sum(1 for event in events if event.is_security_event())
        failed_events = sum(1 for event in events if not event.success)

        # Authentication metrics
        login_attempts = sum(1 for event in events if event.action == ActionType.LOGIN)
        failed_logins = sum(
            1 for event in events if event.action == ActionType.LOGIN_FAILED
        )

        # Administrative actions
        admin_actions = sum(
            1
            for event in events
            if event.action
            in [
                ActionType.USER_CREATED,
                ActionType.USER_UPDATED,
                ActionType.USER_DELETED,
                ActionType.ROLE_ASSIGNED,
                ActionType.PERMISSION_GRANTED,
                ActionType.PERMISSION_REVOKED,
            ]
        )

        # Data access patterns
        data_access_events = sum(
            1
            for event in events
            if event.action
            in [
                ActionType.DATA_READ,
                ActionType.DATA_EXPORTED,
                ActionType.SENSITIVE_DATA_ACCESSED,
            ]
        )

        # Security incidents
        recent_incidents = [
            incident
            for incident in self.security_incidents.values()
            if start_time <= incident.detected_at <= end_time
        ]

        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": (end_time - start_time).days,
            },
            "events": {
                "total": total_events,
                "security_related": security_events,
                "failed": failed_events,
                "success_rate": ((total_events - failed_events) / total_events * 100)
                if total_events > 0
                else 0,
            },
            "authentication": {
                "login_attempts": login_attempts,
                "failed_logins": failed_logins,
                "success_rate": (
                    (login_attempts - failed_logins) / login_attempts * 100
                )
                if login_attempts > 0
                else 0,
            },
            "administration": {
                "admin_actions": admin_actions,
                "daily_average": admin_actions / max(1, (end_time - start_time).days),
            },
            "data_access": {
                "total_access_events": data_access_events,
                "daily_average": data_access_events
                / max(1, (end_time - start_time).days),
            },
            "incidents": {
                "total": len(recent_incidents),
                "by_severity": {
                    "critical": sum(
                        1 for i in recent_incidents if i.severity == "critical"
                    ),
                    "high": sum(1 for i in recent_incidents if i.severity == "high"),
                    "medium": sum(
                        1 for i in recent_incidents if i.severity == "medium"
                    ),
                    "low": sum(1 for i in recent_incidents if i.severity == "low"),
                },
                "open": sum(
                    1
                    for i in recent_incidents
                    if i.response_status in ["open", "investigating"]
                ),
                "resolved": sum(
                    1 for i in recent_incidents if i.response_status == "resolved"
                ),
            },
            "alerts": {
                "active_security_alerts": len(self.security_alerts),
                "compliance_violations": len(self.compliance_violations),
            },
        }

    async def _security_monitoring_loop(self) -> None:
        """Background task for security monitoring."""

        while True:
            try:
                # Check for suspicious patterns
                await self._detect_suspicious_patterns()

                # Check for security policy violations
                await self._detect_policy_violations()

                # Update security metrics
                await self._update_security_metrics()

            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _compliance_monitoring_loop(self) -> None:
        """Background task for compliance monitoring."""

        while True:
            try:
                # Check compliance requirements
                await self._check_compliance_requirements()

                # Validate data retention policies
                await self._validate_retention_policies()

            except Exception as e:
                self.logger.error(f"Compliance monitoring error: {e}")

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _cleanup_loop(self) -> None:
        """Background task for data cleanup based on retention policies."""

        while True:
            try:
                # Clean up old audit events
                await self._cleanup_old_events()

                # Clean up resolved incidents
                await self._cleanup_old_incidents()

                # Clean up old alerts
                await self._cleanup_old_alerts()

            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

            await asyncio.sleep(3600)  # Clean up every hour

    async def _alert_processing_loop(self) -> None:
        """Background task for processing security alerts."""

        while True:
            try:
                # Process pending alerts
                await self._process_security_alerts()

                # Process compliance violations
                await self._process_compliance_violations()

            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")

            await asyncio.sleep(30)  # Process alerts every 30 seconds

    async def _analyze_security_event(self, event: AuditEvent) -> None:
        """Analyze event for security implications."""

        # Check for suspicious patterns
        if event.action == ActionType.LOGIN_FAILED:
            await self._check_brute_force_attempts(event)

        elif event.action == ActionType.SENSITIVE_DATA_ACCESSED:
            await self._check_unusual_data_access(event)

        elif event.action in [ActionType.PERMISSION_GRANTED, ActionType.ROLE_ASSIGNED]:
            await self._check_privilege_escalation(event)

        elif event.security_level == "CRITICAL":
            await self._generate_critical_alert(event)

    async def _analyze_compliance_event(self, event: AuditEvent) -> None:
        """Analyze event for compliance implications."""

        # Check data protection requirements
        if event.action in [ActionType.DATA_EXPORTED, ActionType.DATA_DELETED]:
            await self._check_data_protection_compliance(event)

        # Check access control requirements
        if event.action in [
            ActionType.PERMISSION_GRANTED,
            ActionType.PERMISSION_REVOKED,
        ]:
            await self._check_access_control_compliance(event)

        # Check audit trail completeness
        await self._check_audit_trail_compliance(event)

    async def _check_brute_force_attempts(self, event: AuditEvent) -> None:
        """Check for brute force attack patterns."""

        if not event.ip_address:
            return

        # Get recent failed login attempts from same IP
        recent_events = await self.query_events(
            action_types=[ActionType.LOGIN_FAILED],
            start_time=datetime.utcnow() - timedelta(minutes=15),
            limit=20,
        )

        same_ip_failures = [
            e for e in recent_events if e.ip_address == event.ip_address
        ]

        if len(same_ip_failures) >= 5:  # 5 failures in 15 minutes
            alert = {
                "alert_id": str(uuid4()),
                "alert_type": "brute_force_attack",
                "severity": "high",
                "description": f"Potential brute force attack from {event.ip_address}",
                "ip_address": event.ip_address,
                "failure_count": len(same_ip_failures),
                "timestamp": datetime.utcnow(),
                "related_events": [str(e.event_id) for e in same_ip_failures],
            }

            self.security_alerts.append(alert)

            # Create security incident for severe cases
            if len(same_ip_failures) >= 10:
                await self.create_security_incident(
                    incident_type="brute_force_attack",
                    title=f"Brute force attack from {event.ip_address}",
                    description=f"Detected {len(same_ip_failures)} failed login attempts in 15 minutes",
                    severity="high",
                    affected_systems=["authentication"],
                )

    async def _assess_gdpr_compliance(
        self, report: ComplianceReport, events: list[AuditEvent]
    ) -> None:
        """Assess GDPR compliance requirements."""

        # Article 30: Records of processing activities
        data_processing_events = [
            e
            for e in events
            if e.action
            in [
                ActionType.DATA_CREATED,
                ActionType.DATA_READ,
                ActionType.DATA_UPDATED,
                ActionType.DATA_DELETED,
                ActionType.DATA_EXPORTED,
            ]
        ]

        if data_processing_events:
            report.control_results["gdpr_article_30"] = {
                "status": "compliant",
                "description": "Data processing activities are logged",
                "evidence_count": len(data_processing_events),
                "criticality": "high",
            }
            report.compliant_controls += 1
        else:
            report.control_results["gdpr_article_30"] = {
                "status": "failed",
                "description": "No data processing activities logged",
                "criticality": "high",
            }
            report.non_compliant_controls += 1
            report.critical_findings.append(
                "GDPR Article 30: No records of processing activities"
            )

        # Article 32: Security of processing
        security_events = [e for e in events if e.is_security_event()]

        if security_events:
            report.control_results["gdpr_article_32"] = {
                "status": "compliant",
                "description": "Security measures are implemented and monitored",
                "evidence_count": len(security_events),
                "criticality": "high",
            }
            report.compliant_controls += 1
        else:
            report.control_results["gdpr_article_32"] = {
                "status": "failed",
                "description": "Insufficient security monitoring",
                "criticality": "high",
            }
            report.non_compliant_controls += 1
            report.critical_findings.append(
                "GDPR Article 32: Insufficient security of processing"
            )

        report.total_controls += 2

    async def _assess_soc2_compliance(
        self, report: ComplianceReport, events: list[AuditEvent]
    ) -> None:
        """Assess SOC 2 compliance requirements."""

        # CC6.1: Logical and physical access controls
        access_control_events = [
            e
            for e in events
            if e.action
            in [
                ActionType.LOGIN,
                ActionType.LOGIN_FAILED,
                ActionType.PERMISSION_GRANTED,
                ActionType.PERMISSION_REVOKED,
                ActionType.ROLE_ASSIGNED,
            ]
        ]

        if access_control_events:
            report.control_results["soc2_cc6_1"] = {
                "status": "compliant",
                "description": "Access control activities are logged",
                "evidence_count": len(access_control_events),
                "criticality": "high",
            }
            report.compliant_controls += 1
        else:
            report.control_results["soc2_cc6_1"] = {
                "status": "failed",
                "description": "No access control activities logged",
                "criticality": "high",
            }
            report.non_compliant_controls += 1
            report.critical_findings.append(
                "SOC 2 CC6.1: No logical access control monitoring"
            )

        # CC7.1: System monitoring
        system_events = [
            e
            for e in events
            if e.action
            in [
                ActionType.SYSTEM_BACKUP,
                ActionType.SYSTEM_RESTORE,
                ActionType.SYSTEM_MAINTENANCE,
            ]
        ]

        if system_events:
            report.control_results["soc2_cc7_1"] = {
                "status": "compliant",
                "description": "System activities are monitored",
                "evidence_count": len(system_events),
                "criticality": "medium",
            }
            report.compliant_controls += 1
        else:
            report.control_results["soc2_cc7_1"] = {
                "status": "failed",
                "description": "Insufficient system monitoring",
                "criticality": "medium",
            }
            report.non_compliant_controls += 1
            report.medium_findings.append("SOC 2 CC7.1: Insufficient system monitoring")

        report.total_controls += 2

    def _is_compliance_relevant(self, action: ActionType, security_level: str) -> bool:
        """Determine if event is compliance-relevant."""

        compliance_actions = {
            ActionType.DATA_CREATED,
            ActionType.DATA_READ,
            ActionType.DATA_UPDATED,
            ActionType.DATA_DELETED,
            ActionType.DATA_EXPORTED,
            ActionType.SENSITIVE_DATA_ACCESSED,
            ActionType.USER_CREATED,
            ActionType.USER_UPDATED,
            ActionType.USER_DELETED,
            ActionType.ROLE_ASSIGNED,
            ActionType.PERMISSION_GRANTED,
            ActionType.PERMISSION_REVOKED,
            ActionType.SECURITY_POLICY_UPDATED,
        }

        return action in compliance_actions or security_level in ["WARNING", "CRITICAL"]

    async def _generate_security_alert(self, incident: SecurityIncident) -> None:
        """Generate security alert for incident."""

        alert = {
            "alert_id": str(uuid4()),
            "alert_type": "security_incident",
            "severity": incident.severity,
            "incident_id": str(incident.incident_id),
            "title": incident.title,
            "description": incident.description,
            "timestamp": incident.detected_at,
            "requires_immediate_attention": incident.severity in ["high", "critical"],
        }

        self.security_alerts.append(alert)

    async def _cleanup_old_events(self) -> None:
        """Clean up old audit events based on retention policy."""

        cutoff_date = datetime.utcnow() - timedelta(
            days=self.security_policy.retention_period_days
        )

        events_to_remove = []
        for event_id, event in self.audit_events.items():
            if event.timestamp < cutoff_date and event.requires_retention:
                # Check if event has custom retention requirements
                retention_days = event.get_retention_period_days()
                event_cutoff = datetime.utcnow() - timedelta(days=retention_days)

                if event.timestamp < event_cutoff:
                    events_to_remove.append(event_id)

        # Remove old events
        for event_id in events_to_remove:
            if event_id in self.audit_events:
                del self.audit_events[event_id]

        if events_to_remove:
            self.logger.info(f"Cleaned up {len(events_to_remove)} old audit events")

    async def _detect_suspicious_patterns(self) -> None:
        """Detect suspicious activity patterns."""

        # This would contain sophisticated anomaly detection algorithms
        # For now, implementing basic pattern detection

        recent_time = datetime.utcnow() - timedelta(hours=1)
        recent_events = await self.query_events(start_time=recent_time)

        # Check for unusual user activity patterns
        user_activity = {}
        for event in recent_events:
            if event.user_id:
                if event.user_id not in user_activity:
                    user_activity[event.user_id] = []
                user_activity[event.user_id].append(event)

        # Flag users with unusually high activity
        for user_id, events in user_activity.items():
            if len(events) > 100:  # More than 100 events in an hour
                alert = {
                    "alert_id": str(uuid4()),
                    "alert_type": "unusual_user_activity",
                    "severity": "medium",
                    "user_id": str(user_id),
                    "event_count": len(events),
                    "timestamp": datetime.utcnow(),
                    "description": f"User generated {len(events)} events in the last hour",
                }
                self.security_alerts.append(alert)

    async def _process_security_alerts(self) -> None:
        """Process pending security alerts."""

        # Remove alerts older than 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.security_alerts = [
            alert for alert in self.security_alerts if alert["timestamp"] > cutoff
        ]

        # In production, this would send alerts to SIEM, notification systems, etc.
        critical_alerts = [
            alert for alert in self.security_alerts if alert["severity"] == "critical"
        ]

        if critical_alerts:
            self.logger.critical(
                f"Processing {len(critical_alerts)} critical security alerts"
            )

    async def _check_data_protection_compliance(self, event: AuditEvent) -> None:
        """Check data protection compliance for the event."""

        # Ensure data exports are properly authorized
        if event.action == ActionType.DATA_EXPORTED:
            if not event.additional_data.get("authorized_by"):
                violation = {
                    "violation_id": str(uuid4()),
                    "type": "unauthorized_data_export",
                    "event_id": str(event.event_id),
                    "description": "Data export without proper authorization",
                    "timestamp": datetime.utcnow(),
                    "severity": "high",
                }
                self.compliance_violations.append(violation)
