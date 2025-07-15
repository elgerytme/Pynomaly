"""Real-time compliance monitoring and alerting system."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pynomaly.domain.entities.audit import AuditEvent, AuditEventType, EventSeverity
from pynomaly.domain.entities.compliance import ComplianceFramework, ComplianceRule
from pynomaly.infrastructure.security.audit_logging import AuditLogger
from pynomaly.infrastructure.security.database_audit_storage import DatabaseAuditStorage


class ComplianceViolation:
    """Represents a compliance violation detected by the monitoring system."""
    
    def __init__(
        self,
        violation_id: UUID,
        framework: ComplianceFramework,
        rule_name: str,
        severity: EventSeverity,
        description: str,
        detected_at: datetime,
        related_events: List[AuditEvent],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.violation_id = violation_id
        self.framework = framework
        self.rule_name = rule_name
        self.severity = severity
        self.description = description
        self.detected_at = detected_at
        self.related_events = related_events
        self.metadata = metadata or {}


class ComplianceRule:
    """Defines a compliance rule for monitoring."""
    
    def __init__(
        self,
        rule_id: str,
        framework: ComplianceFramework,
        name: str,
        description: str,
        severity: EventSeverity,
        condition: callable,
        time_window_minutes: int = 60
    ):
        self.rule_id = rule_id
        self.framework = framework
        self.name = name
        self.description = description
        self.severity = severity
        self.condition = condition
        self.time_window_minutes = time_window_minutes


class ComplianceMonitor:
    """Real-time compliance monitoring system."""
    
    def __init__(self, audit_storage: DatabaseAuditStorage, audit_logger: AuditLogger):
        """Initialize compliance monitor.
        
        Args:
            audit_storage: Database audit storage instance
            audit_logger: Audit logger for compliance events
        """
        self.audit_storage = audit_storage
        self.audit_logger = audit_logger
        self.compliance_rules = self._initialize_compliance_rules()
        self.violation_callbacks = []
        self.monitoring_active = False
    
    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """Initialize default compliance rules."""
        return [
            # GDPR Rules
            ComplianceRule(
                rule_id="GDPR_001",
                framework=ComplianceFramework.GDPR,
                name="Excessive Data Access",
                description="User accessed more than 100 records in 1 hour",
                severity=EventSeverity.MEDIUM,
                condition=self._check_excessive_data_access,
                time_window_minutes=60
            ),
            ComplianceRule(
                rule_id="GDPR_002",
                framework=ComplianceFramework.GDPR,
                name="Unauthorized Data Export",
                description="Data export without proper authorization",
                severity=EventSeverity.HIGH,
                condition=self._check_unauthorized_data_export,
                time_window_minutes=5
            ),
            ComplianceRule(
                rule_id="GDPR_003",
                framework=ComplianceFramework.GDPR,
                name="Failed Data Subject Request",
                description="Failed to process data subject request within timeframe",
                severity=EventSeverity.HIGH,
                condition=self._check_failed_data_subject_request,
                time_window_minutes=1440  # 24 hours
            ),
            
            # SOX Rules
            ComplianceRule(
                rule_id="SOX_001",
                framework=ComplianceFramework.SOX,
                name="Financial Data Access Without Approval",
                description="Financial data accessed without proper approval workflow",
                severity=EventSeverity.HIGH,
                condition=self._check_financial_data_access,
                time_window_minutes=1
            ),
            ComplianceRule(
                rule_id="SOX_002",
                framework=ComplianceFramework.SOX,
                name="Missing Audit Trail",
                description="Financial transaction without complete audit trail",
                severity=EventSeverity.CRITICAL,
                condition=self._check_missing_audit_trail,
                time_window_minutes=5
            ),
            
            # HIPAA Rules
            ComplianceRule(
                rule_id="HIPAA_001",
                framework=ComplianceFramework.HIPAA,
                name="Unauthorized PHI Access",
                description="Protected health information accessed without authorization",
                severity=EventSeverity.CRITICAL,
                condition=self._check_unauthorized_phi_access,
                time_window_minutes=1
            ),
            ComplianceRule(
                rule_id="HIPAA_002",
                framework=ComplianceFramework.HIPAA,
                name="PHI Disclosure Without Consent",
                description="PHI disclosed without proper patient consent",
                severity=EventSeverity.CRITICAL,
                condition=self._check_phi_disclosure,
                time_window_minutes=1
            ),
            
            # General Security Rules
            ComplianceRule(
                rule_id="SEC_001",
                framework=ComplianceFramework.SOC2,
                name="Multiple Failed Login Attempts",
                description="Multiple failed login attempts indicating brute force attack",
                severity=EventSeverity.HIGH,
                condition=self._check_failed_logins,
                time_window_minutes=15
            ),
            ComplianceRule(
                rule_id="SEC_002",
                framework=ComplianceFramework.SOC2,
                name="Privilege Escalation",
                description="Unauthorized privilege escalation detected",
                severity=EventSeverity.CRITICAL,
                condition=self._check_privilege_escalation,
                time_window_minutes=5
            ),
        ]
    
    async def start_monitoring(self) -> None:
        """Start real-time compliance monitoring."""
        self.monitoring_active = True
        
        # Start monitoring tasks
        await asyncio.gather(
            self._monitor_compliance_rules(),
            self._monitor_data_retention(),
            self._monitor_access_patterns()
        )
    
    async def stop_monitoring(self) -> None:
        """Stop compliance monitoring."""
        self.monitoring_active = False
    
    async def _monitor_compliance_rules(self) -> None:
        """Monitor compliance rules continuously."""
        while self.monitoring_active:
            try:
                for rule in self.compliance_rules:
                    violations = await self._check_rule_violations(rule)
                    for violation in violations:
                        await self._handle_compliance_violation(violation)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                await self.audit_logger.log_event(
                    event_type=AuditEventType.SYSTEM,
                    action="compliance_monitoring_error",
                    outcome="FAILURE",
                    details={"error": str(e)}
                )
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_rule_violations(self, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Check for violations of a specific compliance rule."""
        try:
            # Get recent events within the rule's time window
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=rule.time_window_minutes)
            
            recent_events = await self.audit_storage.query_events(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            # Apply rule condition
            violations = await rule.condition(recent_events, rule)
            return violations
            
        except Exception as e:
            print(f"Failed to check rule violations for {rule.rule_id}: {e}")
            return []
    
    async def _handle_compliance_violation(self, violation: ComplianceViolation) -> None:
        """Handle detected compliance violation."""
        # Log the violation as an audit event
        await self.audit_logger.log_event(
            event_type=AuditEventType.SECURITY,
            action="compliance_violation",
            outcome="FAILURE",
            severity=violation.severity,
            details={
                "violation_id": str(violation.violation_id),
                "framework": violation.framework.value,
                "rule_name": violation.rule_name,
                "description": violation.description,
                "related_events": [str(event.id) for event in violation.related_events]
            }
        )
        
        # Notify violation callbacks
        for callback in self.violation_callbacks:
            try:
                await callback(violation)
            except Exception as e:
                print(f"Violation callback failed: {e}")
    
    def add_violation_callback(self, callback: callable) -> None:
        """Add callback function for compliance violations."""
        self.violation_callbacks.append(callback)
    
    # Compliance Rule Implementations
    
    async def _check_excessive_data_access(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for excessive data access patterns."""
        violations = []
        
        # Group events by user
        user_events = {}
        for event in events:
            if event.user_id and "data_access" in event.action:
                if event.user_id not in user_events:
                    user_events[event.user_id] = []
                user_events[event.user_id].append(event)
        
        # Check for excessive access
        for user_id, user_event_list in user_events.items():
            if len(user_event_list) > 100:  # Configurable threshold
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"User {user_id} accessed {len(user_event_list)} records in {rule.time_window_minutes} minutes",
                    detected_at=datetime.now(timezone.utc),
                    related_events=user_event_list,
                    metadata={"user_id": str(user_id), "access_count": len(user_event_list)}
                )
                violations.append(violation)
        
        return violations
    
    async def _check_unauthorized_data_export(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for unauthorized data exports."""
        violations = []
        
        export_events = [e for e in events if "export" in e.action and e.outcome == "SUCCESS"]
        
        for event in export_events:
            # Check if export was properly authorized
            if not event.metadata.get("authorized", False):
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"Unauthorized data export by user {event.user_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=[event],
                    metadata={"user_id": str(event.user_id) if event.user_id else None}
                )
                violations.append(violation)
        
        return violations
    
    async def _check_failed_data_subject_request(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for failed data subject requests."""
        violations = []
        
        failed_requests = [
            e for e in events 
            if "data_subject_request" in e.action and e.outcome == "FAILURE"
        ]
        
        for event in failed_requests:
            violation = ComplianceViolation(
                violation_id=UUID(),
                framework=rule.framework,
                rule_name=rule.name,
                severity=rule.severity,
                description=f"Failed to process data subject request: {event.details.get('request_type', 'unknown')}",
                detected_at=datetime.now(timezone.utc),
                related_events=[event],
                metadata={"request_type": event.details.get("request_type")}
            )
            violations.append(violation)
        
        return violations
    
    async def _check_financial_data_access(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for unauthorized financial data access."""
        violations = []
        
        financial_events = [
            e for e in events 
            if e.resource_type == "financial_data" and e.action == "read"
        ]
        
        for event in financial_events:
            # Check if access was approved
            if not event.metadata.get("approval_id"):
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"Financial data accessed without approval by user {event.user_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=[event]
                )
                violations.append(violation)
        
        return violations
    
    async def _check_missing_audit_trail(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for missing audit trails in financial transactions."""
        violations = []
        
        # This would be more complex in practice, checking for complete audit chains
        financial_transactions = [
            e for e in events 
            if e.resource_type == "financial_transaction"
        ]
        
        for event in financial_transactions:
            if not event.correlation_id:
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"Financial transaction without correlation ID: {event.resource_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=[event]
                )
                violations.append(violation)
        
        return violations
    
    async def _check_unauthorized_phi_access(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for unauthorized PHI access."""
        violations = []
        
        phi_events = [
            e for e in events 
            if e.resource_type == "phi_data" and e.action in ["read", "export"]
        ]
        
        for event in phi_events:
            # Check if user has proper authorization for PHI access
            if not event.metadata.get("hipaa_authorized", False):
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"Unauthorized PHI access by user {event.user_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=[event]
                )
                violations.append(violation)
        
        return violations
    
    async def _check_phi_disclosure(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for PHI disclosure without consent."""
        violations = []
        
        disclosure_events = [
            e for e in events 
            if e.resource_type == "phi_data" and "share" in e.action
        ]
        
        for event in disclosure_events:
            if not event.metadata.get("patient_consent", False):
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"PHI disclosed without patient consent: {event.resource_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=[event]
                )
                violations.append(violation)
        
        return violations
    
    async def _check_failed_logins(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for multiple failed login attempts."""
        violations = []
        
        # Group failed logins by IP and user
        failed_logins = {}
        for event in events:
            if event.event_type == AuditEventType.AUTHENTICATION and event.outcome == "FAILURE":
                key = (event.ip_address, event.user_id)
                if key not in failed_logins:
                    failed_logins[key] = []
                failed_logins[key].append(event)
        
        # Check for excessive failed attempts
        for (ip_address, user_id), login_events in failed_logins.items():
            if len(login_events) >= 5:  # Configurable threshold
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"Multiple failed login attempts from {ip_address} for user {user_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=login_events,
                    metadata={"ip_address": ip_address, "failed_attempts": len(login_events)}
                )
                violations.append(violation)
        
        return violations
    
    async def _check_privilege_escalation(
        self, events: List[AuditEvent], rule: ComplianceRule
    ) -> List[ComplianceViolation]:
        """Check for unauthorized privilege escalation."""
        violations = []
        
        privilege_events = [
            e for e in events 
            if e.action == "privilege_escalation" and e.outcome == "SUCCESS"
        ]
        
        for event in privilege_events:
            # Check if escalation was properly authorized
            if not event.metadata.get("authorized_by"):
                violation = ComplianceViolation(
                    violation_id=UUID(),
                    framework=rule.framework,
                    rule_name=rule.name,
                    severity=rule.severity,
                    description=f"Unauthorized privilege escalation by user {event.user_id}",
                    detected_at=datetime.now(timezone.utc),
                    related_events=[event]
                )
                violations.append(violation)
        
        return violations
    
    async def _monitor_data_retention(self) -> None:
        """Monitor data retention policy compliance."""
        while self.monitoring_active:
            try:
                # Check for data that should be deleted based on retention policies
                await self._check_retention_compliance()
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                print(f"Data retention monitoring error: {e}")
                await asyncio.sleep(3600)
    
    async def _monitor_access_patterns(self) -> None:
        """Monitor unusual access patterns."""
        while self.monitoring_active:
            try:
                # Analyze access patterns for anomalies
                await self._analyze_access_patterns()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"Access pattern monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _check_retention_compliance(self) -> None:
        """Check compliance with data retention policies."""
        # Implementation would check various data types against retention policies
        pass
    
    async def _analyze_access_patterns(self) -> None:
        """Analyze access patterns for anomalies."""
        # Implementation would use ML/statistical analysis to detect unusual patterns
        pass
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        try:
            # Get compliance summary from audit storage
            summary = await self.audit_storage.get_compliance_summary(
                start_time=start_time,
                end_time=end_time,
                compliance_framework=framework.value
            )
            
            # Get violations for the period
            violations = []
            for rule in self.compliance_rules:
                if rule.framework == framework:
                    events = await self.audit_storage.query_events(
                        start_time=start_time,
                        end_time=end_time,
                        filters={"action": "compliance_violation"}
                    )
                    violations.extend([
                        e for e in events 
                        if e.details.get("rule_name") == rule.name
                    ])
            
            report = {
                "framework": framework.value,
                "report_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "compliance_summary": summary,
                "violations": [
                    {
                        "timestamp": v.timestamp.isoformat(),
                        "rule_name": v.details.get("rule_name"),
                        "description": v.details.get("description"),
                        "severity": v.severity.value
                    }
                    for v in violations
                ],
                "recommendations": await self._generate_compliance_recommendations(violations),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Failed to generate compliance report: {e}")
            return {}
    
    async def _generate_compliance_recommendations(
        self, violations: List[AuditEvent]
    ) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        # Analyze violation patterns and generate recommendations
        violation_types = {}
        for violation in violations:
            rule_name = violation.details.get("rule_name", "unknown")
            violation_types[rule_name] = violation_types.get(rule_name, 0) + 1
        
        # Generate recommendations based on most common violations
        for rule_name, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True):
            if count > 5:
                recommendations.append(
                    f"Address recurring violations in {rule_name} ({count} occurrences)"
                )
        
        if not recommendations:
            recommendations.append("No major compliance issues detected during this period")
        
        return recommendations