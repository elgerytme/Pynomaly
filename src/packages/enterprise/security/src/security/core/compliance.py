"""Enterprise compliance and audit management."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import csv

import structlog

from ..config.security_config import SecurityConfig
from ...shared.infrastructure.exceptions.base_exceptions import (
    BaseApplicationError,
    ErrorCategory,
    ErrorSeverity
)
from ...shared.infrastructure.logging.structured_logging import StructuredLogger


logger = structlog.get_logger()


class ComplianceError(BaseApplicationError):
    """Compliance-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"


class AuditEventType(Enum):
    """Types of audit events."""
    USER_AUTHENTICATION = "user_authentication"
    USER_AUTHORIZATION = "user_authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    SYSTEM_CONFIGURATION = "system_configuration"
    PRIVILEGE_CHANGE = "privilege_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_VIOLATION = "compliance_violation"
    DATA_EXPORT = "data_export"
    DATA_RETENTION_ACTION = "data_retention_action"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class AuditEvent:
    """Audit trail event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = ""  # success, failure, denied
    data_classification: Optional[DataClassification] = None
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    details: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Generate checksum for audit integrity."""
        self.checksum = self._generate_checksum()
    
    def _generate_checksum(self) -> str:
        """Generate checksum for audit event integrity."""
        # Create a stable string representation
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "data_classification": self.data_classification.value if self.data_classification else None,
            "details": json.dumps(self.details, sort_keys=True)
        }
        
        # Generate SHA-256 hash
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify audit event integrity."""
        expected_checksum = self._generate_checksum()
        return self.checksum == expected_checksum


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    name: str
    description: str
    framework: ComplianceFramework
    category: str
    requirement: str
    validation_function: Callable[[Dict[str, Any]], bool]
    severity: str = "medium"
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_name: str
    framework: ComplianceFramework
    severity: str
    description: str
    timestamp: datetime
    user_id: Optional[str] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""
    name: str
    description: str
    data_classification: DataClassification
    retention_days: int
    auto_delete: bool = False
    framework_requirements: Set[ComplianceFramework] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ComplianceManager:
    """Enterprise compliance and audit management system.
    
    Provides comprehensive compliance capabilities including:
    - Audit trail management
    - Compliance rule enforcement
    - Data retention policies
    - Regulatory reporting
    - Violation tracking and remediation
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = StructuredLogger(config.logging)
        
        # Audit storage (in production, use secure external storage)
        self._audit_events: List[AuditEvent] = []
        self._audit_index: Dict[str, AuditEvent] = {}
        
        # Compliance rules and violations
        self._compliance_rules: Dict[str, ComplianceRule] = {}
        self._violations: List[ComplianceViolation] = []
        
        # Data retention policies
        self._retention_policies: Dict[DataClassification, DataRetentionPolicy] = {}
        
        # Initialize default rules and policies
        self._initialize_compliance_rules()
        self._initialize_retention_policies()
    
    def log_audit_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str = "success",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        data_classification: Optional[DataClassification] = None,
        details: Dict[str, Any] = None
    ) -> AuditEvent:
        """Log an audit event."""
        try:
            event_id = self._generate_event_id()
            
            # Determine applicable compliance frameworks
            frameworks = self._determine_applicable_frameworks(
                event_type, data_classification, details or {}
            )
            
            audit_event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                result=result,
                data_classification=data_classification,
                compliance_frameworks=frameworks,
                details=details or {}
            )
            
            # Store audit event
            self._audit_events.append(audit_event)
            self._audit_index[event_id] = audit_event
            
            # Check compliance rules
            self._check_compliance_rules(audit_event)
            
            # Log to structured logger
            self.logger.info(
                "Audit event logged",
                event_id=event_id,
                event_type=event_type.value,
                action=action,
                result=result,
                user_id=user_id,
                resource=resource,
                frameworks=[f.value for f in frameworks]
            )
            
            return audit_event
            
        except Exception as e:
            self.logger.error("Failed to log audit event", error=str(e))
            raise ComplianceError("Audit logging failed") from e
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a compliance rule."""
        self._compliance_rules[rule.name] = rule
        self.logger.info(
            "Compliance rule added",
            rule=rule.name,
            framework=rule.framework.value
        )
    
    def check_data_retention_compliance(self) -> List[Dict[str, Any]]:
        """Check data retention compliance and identify expired data."""
        try:
            expired_data = []
            current_time = datetime.now(timezone.utc)
            
            # Group audit events by data classification
            classified_events = {}
            for event in self._audit_events:
                if event.data_classification:
                    classification = event.data_classification
                    if classification not in classified_events:
                        classified_events[classification] = []
                    classified_events[classification].append(event)
            
            # Check each classification against retention policies
            for classification, events in classified_events.items():
                policy = self._retention_policies.get(classification)
                if not policy:
                    continue
                
                retention_cutoff = current_time - timedelta(days=policy.retention_days)
                
                for event in events:
                    if event.timestamp < retention_cutoff:
                        expired_data.append({
                            "event_id": event.event_id,
                            "event_type": event.event_type.value,
                            "timestamp": event.timestamp.isoformat(),
                            "classification": classification.value,
                            "policy": policy.name,
                            "days_expired": (current_time - event.timestamp).days - policy.retention_days,
                            "auto_delete": policy.auto_delete
                        })
            
            return expired_data
            
        except Exception as e:
            self.logger.error("Data retention compliance check failed", error=str(e))
            return []
    
    def generate_compliance_report(
        self, 
        framework: ComplianceFramework,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report for specific framework."""
        try:
            if start_date is None:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now(timezone.utc)
            
            # Filter events by framework and date range
            relevant_events = [
                event for event in self._audit_events
                if framework in event.compliance_frameworks
                and start_date <= event.timestamp <= end_date
            ]
            
            # Group events by type
            event_counts = {}
            for event in relevant_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Get violations for this framework
            framework_violations = [
                v for v in self._violations
                if v.framework == framework
                and start_date <= v.timestamp <= end_date
            ]
            
            # Calculate compliance metrics
            total_events = len(relevant_events)
            violation_count = len(framework_violations)
            compliance_rate = (total_events - violation_count) / total_events * 100 if total_events > 0 else 100
            
            # Group violations by severity
            violation_severity = {}
            for violation in framework_violations:
                severity = violation.severity
                violation_severity[severity] = violation_severity.get(severity, 0) + 1
            
            return {
                "framework": framework.value,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_events": total_events,
                    "violation_count": violation_count,
                    "compliance_rate_percent": round(compliance_rate, 2)
                },
                "event_counts": event_counts,
                "violations": {
                    "total": violation_count,
                    "by_severity": violation_severity,
                    "unresolved": len([v for v in framework_violations if not v.resolved])
                },
                "data_retention": self._get_retention_status(framework),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error("Compliance report generation failed", error=str(e))
            return {}
    
    def export_audit_trail(
        self, 
        format_type: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None
    ) -> str:
        """Export audit trail in specified format."""
        try:
            # Filter events
            filtered_events = self._audit_events
            
            if start_date:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
            
            if end_date:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
            
            if event_types:
                filtered_events = [e for e in filtered_events if e.event_type in event_types]
            
            # Export in requested format
            if format_type.lower() == "json":
                return self._export_as_json(filtered_events)
            elif format_type.lower() == "csv":
                return self._export_as_csv(filtered_events)
            else:
                raise ComplianceError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            self.logger.error("Audit trail export failed", error=str(e))
            raise
    
    def get_violations(
        self,
        framework: Optional[ComplianceFramework] = None,
        resolved: Optional[bool] = None,
        severity: Optional[str] = None
    ) -> List[ComplianceViolation]:
        """Get compliance violations with optional filtering."""
        violations = self._violations
        
        if framework:
            violations = [v for v in violations if v.framework == framework]
        
        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]
        
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        return violations
    
    def resolve_violation(
        self, 
        violation_id: str, 
        resolution_notes: str,
        resolved_by: str
    ) -> bool:
        """Mark a compliance violation as resolved."""
        try:
            for violation in self._violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    violation.resolution_notes = resolution_notes
                    violation.resolved_at = datetime.now(timezone.utc)
                    
                    # Log resolution
                    self.log_audit_event(
                        AuditEventType.COMPLIANCE_VIOLATION,
                        action="resolve_violation",
                        result="success",
                        user_id=resolved_by,
                        details={
                            "violation_id": violation_id,
                            "resolution_notes": resolution_notes
                        }
                    )
                    
                    self.logger.info(
                        "Compliance violation resolved",
                        violation_id=violation_id,
                        resolved_by=resolved_by
                    )
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error("Violation resolution failed", error=str(e))
            return False
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit trail."""
        try:
            total_events = len(self._audit_events)
            corrupted_events = []
            
            for event in self._audit_events:
                if not event.verify_integrity():
                    corrupted_events.append({
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "expected_checksum": event._generate_checksum(),
                        "actual_checksum": event.checksum
                    })
            
            integrity_percentage = (total_events - len(corrupted_events)) / total_events * 100 if total_events > 0 else 100
            
            return {
                "total_events": total_events,
                "corrupted_events": len(corrupted_events),
                "integrity_percentage": round(integrity_percentage, 2),
                "corrupted_event_details": corrupted_events,
                "verification_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error("Audit integrity verification failed", error=str(e))
            return {"error": str(e)}
    
    # Private helper methods
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _determine_applicable_frameworks(
        self,
        event_type: AuditEventType,
        data_classification: Optional[DataClassification],
        details: Dict[str, Any]
    ) -> Set[ComplianceFramework]:
        """Determine which compliance frameworks apply to this event."""
        frameworks = set()
        
        # GDPR applies to personal data events
        if (data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED] or
            event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION, AuditEventType.DATA_DELETION] or
            details.get("contains_pii", False)):
            if self.config.compliance.enable_gdpr_compliance:
                frameworks.add(ComplianceFramework.GDPR)
        
        # HIPAA applies to healthcare data
        if (details.get("contains_phi", False) or details.get("healthcare_related", False)):
            if self.config.compliance.enable_hipaa_compliance:
                frameworks.add(ComplianceFramework.HIPAA)
        
        # SOX applies to financial data and system changes
        if (event_type in [AuditEventType.SYSTEM_CONFIGURATION, AuditEventType.PRIVILEGE_CHANGE] or
            details.get("financial_data", False)):
            if self.config.compliance.enable_sox_compliance:
                frameworks.add(ComplianceFramework.SOX)
        
        return frameworks
    
    def _check_compliance_rules(self, event: AuditEvent) -> None:
        """Check event against compliance rules."""
        try:
            for rule_name, rule in self._compliance_rules.items():
                if not rule.enabled:
                    continue
                
                # Check if rule applies to this event's frameworks
                if rule.framework not in event.compliance_frameworks:
                    continue
                
                # Evaluate rule
                event_data = asdict(event)
                if not rule.validation_function(event_data):
                    # Rule violation detected
                    violation = ComplianceViolation(
                        violation_id=self._generate_event_id(),
                        rule_name=rule_name,
                        framework=rule.framework,
                        severity=rule.severity,
                        description=f"Violation of rule: {rule.description}",
                        timestamp=datetime.now(timezone.utc),
                        user_id=event.user_id,
                        resource=event.resource,
                        details={"original_event_id": event.event_id}
                    )
                    
                    self._violations.append(violation)
                    
                    self.logger.warning(
                        "Compliance violation detected",
                        violation_id=violation.violation_id,
                        rule=rule_name,
                        framework=rule.framework.value,
                        severity=rule.severity
                    )
                    
        except Exception as e:
            self.logger.error("Compliance rule check failed", error=str(e))
    
    def _get_retention_status(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Get data retention status for framework."""
        applicable_policies = [
            p for p in self._retention_policies.values()
            if framework in p.framework_requirements
        ]
        
        return {
            "policies_count": len(applicable_policies),
            "policies": [
                {
                    "name": p.name,
                    "data_classification": p.data_classification.value,
                    "retention_days": p.retention_days,
                    "auto_delete": p.auto_delete
                }
                for p in applicable_policies
            ]
        }
    
    def _export_as_json(self, events: List[AuditEvent]) -> str:
        """Export events as JSON."""
        export_data = []
        for event in events:
            event_dict = asdict(event)
            # Convert datetime objects to ISO strings
            event_dict["timestamp"] = event.timestamp.isoformat()
            event_dict["compliance_frameworks"] = [f.value for f in event.compliance_frameworks]
            if event.data_classification:
                event_dict["data_classification"] = event.data_classification.value
            event_dict["event_type"] = event.event_type.value
            export_data.append(event_dict)
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_as_csv(self, events: List[AuditEvent]) -> str:
        """Export events as CSV."""
        if not events:
            return ""
        
        from io import StringIO
        output = StringIO()
        
        fieldnames = [
            "event_id", "event_type", "timestamp", "user_id", "session_id",
            "ip_address", "user_agent", "resource", "action", "result",
            "data_classification", "compliance_frameworks", "checksum"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for event in events:
            row = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id or "",
                "session_id": event.session_id or "",
                "ip_address": event.ip_address or "",
                "user_agent": event.user_agent or "",
                "resource": event.resource or "",
                "action": event.action,
                "result": event.result,
                "data_classification": event.data_classification.value if event.data_classification else "",
                "compliance_frameworks": ",".join([f.value for f in event.compliance_frameworks]),
                "checksum": event.checksum or ""
            }
            writer.writerow(row)
        
        return output.getvalue()
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize default compliance rules."""
        # GDPR rules
        if self.config.compliance.enable_gdpr_compliance:
            self.add_compliance_rule(ComplianceRule(
                name="gdpr_data_access_logging",
                description="All personal data access must be logged",
                framework=ComplianceFramework.GDPR,
                category="data_protection",
                requirement="Article 30 - Records of processing activities",
                validation_function=lambda event: event.get("event_type") in [
                    "data_access", "data_modification", "data_deletion"
                ] and event.get("user_id") is not None,
                severity="high"
            ))
        
        # HIPAA rules
        if self.config.compliance.enable_hipaa_compliance:
            self.add_compliance_rule(ComplianceRule(
                name="hipaa_phi_access_control",
                description="PHI access must be authorized and logged",
                framework=ComplianceFramework.HIPAA,
                category="access_control",
                requirement="164.308(a)(4) - Information security management process",
                validation_function=lambda event: (
                    not event.get("details", {}).get("contains_phi", False) or
                    event.get("result") == "success"
                ),
                severity="critical"
            ))
        
        # SOX rules
        if self.config.compliance.enable_sox_compliance:
            self.add_compliance_rule(ComplianceRule(
                name="sox_privileged_access_monitoring",
                description="All privileged access must be monitored",
                framework=ComplianceFramework.SOX,
                category="access_control",
                requirement="Section 404 - Management assessment of internal controls",
                validation_function=lambda event: (
                    event.get("event_type") != "privilege_change" or
                    event.get("user_id") is not None
                ),
                severity="high"
            ))
    
    def _initialize_retention_policies(self) -> None:
        """Initialize default data retention policies."""
        retention_policies = [
            DataRetentionPolicy(
                name="Public Data Retention",
                description="Retention policy for public data",
                data_classification=DataClassification.PUBLIC,
                retention_days=2555,  # 7 years
                framework_requirements={ComplianceFramework.SOX}
            ),
            DataRetentionPolicy(
                name="Internal Data Retention",
                description="Retention policy for internal data",
                data_classification=DataClassification.INTERNAL,
                retention_days=1825,  # 5 years
                framework_requirements={ComplianceFramework.SOX}
            ),
            DataRetentionPolicy(
                name="Confidential Data Retention",
                description="Retention policy for confidential data including PII",
                data_classification=DataClassification.CONFIDENTIAL,
                retention_days=2555,  # 7 years for GDPR
                framework_requirements={ComplianceFramework.GDPR, ComplianceFramework.HIPAA}
            ),
            DataRetentionPolicy(
                name="Restricted Data Retention",
                description="Retention policy for highly sensitive data",
                data_classification=DataClassification.RESTRICTED,
                retention_days=3650,  # 10 years
                auto_delete=False,  # Manual review required
                framework_requirements={ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.SOX}
            )
        ]
        
        for policy in retention_policies:
            self._retention_policies[policy.data_classification] = policy