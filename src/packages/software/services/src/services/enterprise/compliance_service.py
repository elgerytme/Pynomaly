"""
Compliance and audit logging service.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any

from monorepo.domain.entities.compliance import (
    DEFAULT_COMPLIANCE_RULES,
    AuditAction,
    AuditEvent,
    AuditSeverity,
    BackupRecord,
    ComplianceCheck,
    ComplianceFramework,
    ComplianceReport,
    ComplianceRule,
    DataRetentionPolicy,
    EncryptionKey,
    GDPRRequest,
)
from monorepo.shared.exceptions import ValidationError
from monorepo.shared.types import TenantId, UserId


class ComplianceService:
    """Service for managing compliance and audit logging."""

    def __init__(self, audit_repository, compliance_repository, encryption_service):
        self._audit_repo = audit_repository
        self._compliance_repo = compliance_repository
        self._encryption_service = encryption_service

    # Audit Logging
    async def log_audit_event(
        self,
        action: AuditAction,
        tenant_id: TenantId,
        user_id: UserId | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        session_id: str | None = None,
        severity: AuditSeverity | None = None,
    ) -> AuditEvent:
        """Log an audit event."""
        # Determine severity if not provided
        if severity is None:
            severity = self._determine_severity(action)

        # Calculate risk score
        risk_score = self._calculate_risk_score(action, details or {})

        # Determine applicable compliance frameworks
        frameworks = self._get_applicable_frameworks(action, tenant_id)

        # Create audit event
        event = AuditEvent(
            id=str(uuid.uuid4()),
            action=action,
            severity=severity,
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            risk_score=risk_score,
            compliance_frameworks=frameworks,
        )

        # Store audit event
        await self._audit_repo.create_audit_event(event)

        # Check for immediate compliance violations
        await self._check_immediate_compliance_violations(event)

        return event

    async def get_audit_trail(
        self,
        tenant_id: TenantId,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        actions: list[AuditAction] | None = None,
        user_id: UserId | None = None,
        resource_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Get audit trail with filtering options."""
        return await self._audit_repo.get_audit_events(
            tenant_id=tenant_id,
            start_date=start_date or datetime.utcnow() - timedelta(days=30),
            end_date=end_date or datetime.utcnow(),
            actions=actions,
            user_id=user_id,
            resource_type=resource_type,
            limit=limit,
            offset=offset,
        )

    async def get_high_risk_events(
        self, tenant_id: TenantId, days: int = 7
    ) -> list[AuditEvent]:
        """Get high-risk audit events."""
        start_date = datetime.utcnow() - timedelta(days=days)
        events = await self._audit_repo.get_audit_events(
            tenant_id=tenant_id, start_date=start_date, min_risk_score=70
        )
        return [event for event in events if event.is_high_risk]

    # Data Retention Management
    async def create_retention_policy(
        self, policy: DataRetentionPolicy, user_id: UserId
    ) -> DataRetentionPolicy:
        """Create a new data retention policy."""
        policy.id = str(uuid.uuid4())
        policy.created_by = user_id
        policy.created_at = datetime.utcnow()
        policy.updated_at = datetime.utcnow()

        # Validate policy
        await self._validate_retention_policy(policy)

        # Store policy
        created_policy = await self._compliance_repo.create_retention_policy(policy)

        # Log audit event
        await self.log_audit_event(
            action=AuditAction.DATA_RETENTION_POLICY_APPLIED,
            tenant_id=policy.tenant_id,
            user_id=user_id,
            resource_type="retention_policy",
            resource_id=policy.id,
            details={
                "data_type": policy.data_type,
                "retention_days": policy.retention_period_days,
                "auto_delete": policy.auto_delete,
            },
            severity=AuditSeverity.MEDIUM,
        )

        return created_policy

    async def apply_retention_policies(self, tenant_id: TenantId) -> dict[str, int]:
        """Apply all active retention policies for a tenant."""
        policies = await self._compliance_repo.get_active_retention_policies(tenant_id)
        results = {"deleted_records": 0, "archived_records": 0, "policies_applied": 0}

        for policy in policies:
            policy_results = await self._apply_single_retention_policy(policy)
            results["deleted_records"] += policy_results["deleted"]
            results["archived_records"] += policy_results["archived"]
            results["policies_applied"] += 1

            # Log policy application
            await self.log_audit_event(
                action=AuditAction.DATA_RETENTION_POLICY_APPLIED,
                tenant_id=tenant_id,
                resource_type="retention_policy",
                resource_id=policy.id,
                details={
                    "deleted_records": policy_results["deleted"],
                    "archived_records": policy_results["archived"],
                },
                severity=AuditSeverity.LOW,
            )

        return results

    async def _apply_single_retention_policy(
        self, policy: DataRetentionPolicy
    ) -> dict[str, int]:
        """Apply a single retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=policy.retention_period_days)

        # Get expired data
        expired_data = await self._compliance_repo.get_expired_data(
            tenant_id=policy.tenant_id,
            data_type=policy.data_type,
            cutoff_date=cutoff_date,
        )

        archived_count = 0
        deleted_count = 0

        for data_record in expired_data:
            if policy.archive_before_delete:
                # Archive first
                await self._archive_data_record(data_record)
                archived_count += 1

            if policy.auto_delete:
                # Then delete
                await self._delete_data_record(data_record)
                deleted_count += 1

        return {"deleted": deleted_count, "archived": archived_count}

    # GDPR Compliance
    async def create_gdpr_request(
        self,
        request_type: str,
        tenant_id: TenantId,
        data_subject_id: str,
        data_subject_email: str,
        request_details: str,
        submitted_by: UserId,
    ) -> GDPRRequest:
        """Create a new GDPR data subject request."""
        # Calculate completion deadline (30 days for GDPR)
        deadline = datetime.utcnow() + timedelta(days=30)

        gdpr_request = GDPRRequest(
            id=str(uuid.uuid4()),
            request_type=request_type,
            tenant_id=tenant_id,
            data_subject_id=data_subject_id,
            data_subject_email=data_subject_email,
            request_details=request_details,
            submitted_at=datetime.utcnow(),
            completion_deadline=deadline,
        )

        # Store request
        created_request = await self._compliance_repo.create_gdpr_request(gdpr_request)

        # Log audit event
        await self.log_audit_event(
            action=AuditAction.GDPR_REQUEST,
            tenant_id=tenant_id,
            user_id=submitted_by,
            resource_type="gdpr_request",
            resource_id=gdpr_request.id,
            details={
                "request_type": request_type,
                "data_subject_email": data_subject_email,
                "deadline": deadline.isoformat(),
            },
            severity=AuditSeverity.HIGH,
        )

        return created_request

    async def process_gdpr_request(
        self,
        request_id: str,
        processor_id: UserId,
        response_data: dict[str, Any] | None = None,
    ) -> GDPRRequest:
        """Process a GDPR request."""
        gdpr_request = await self._compliance_repo.get_gdpr_request(request_id)
        if not gdpr_request:
            raise ValidationError("GDPR request not found")

        # Update request
        gdpr_request.status = "completed"
        gdpr_request.assigned_to = processor_id
        gdpr_request.response_data = response_data
        gdpr_request.processed_at = datetime.utcnow()

        # Process based on request type
        if gdpr_request.request_type == "erasure":
            await self._process_data_erasure(gdpr_request)
        elif gdpr_request.request_type == "access":
            await self._process_data_access(gdpr_request)
        elif gdpr_request.request_type == "portability":
            await self._process_data_portability(gdpr_request)

        # Update in repository
        updated_request = await self._compliance_repo.update_gdpr_request(gdpr_request)

        # Log completion
        await self.log_audit_event(
            action=AuditAction.GDPR_REQUEST,
            tenant_id=gdpr_request.tenant_id,
            user_id=processor_id,
            resource_type="gdpr_request",
            resource_id=request_id,
            details={
                "request_type": gdpr_request.request_type,
                "status": "completed",
                "processing_time_days": (
                    datetime.utcnow() - gdpr_request.submitted_at
                ).days,
            },
            severity=AuditSeverity.MEDIUM,
        )

        return updated_request

    async def get_overdue_gdpr_requests(self, tenant_id: TenantId) -> list[GDPRRequest]:
        """Get GDPR requests that are overdue."""
        requests = await self._compliance_repo.get_gdpr_requests_by_tenant(tenant_id)
        return [req for req in requests if req.is_overdue]

    # Compliance Checking
    async def run_compliance_check(
        self, tenant_id: TenantId, framework: ComplianceFramework, user_id: UserId
    ) -> ComplianceReport:
        """Run a comprehensive compliance check."""
        # Get applicable rules for framework
        rules = await self._compliance_repo.get_compliance_rules(framework)
        if not rules:
            # Use default rules
            rules = DEFAULT_COMPLIANCE_RULES.get(framework, [])

        # Run checks for each rule
        checks = []
        for rule in rules:
            check = await self._run_single_compliance_check(tenant_id, rule)
            checks.append(check)

        # Generate report
        report = self._generate_compliance_report(
            tenant_id=tenant_id, framework=framework, checks=checks, user_id=user_id
        )

        # Store report
        await self._compliance_repo.create_compliance_report(report)

        # Log audit event
        await self.log_audit_event(
            action=AuditAction.SYSTEM_CONFIGURATION_CHANGED,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="compliance_report",
            resource_id=report.id,
            details={
                "framework": framework.value,
                "compliance_score": report.compliance_score,
                "risk_level": report.risk_level,
            },
            severity=AuditSeverity.MEDIUM,
        )

        return report

    async def _run_single_compliance_check(
        self, tenant_id: TenantId, rule: ComplianceRule
    ) -> ComplianceCheck:
        """Run a compliance check for a single rule."""
        check = ComplianceCheck(
            id=str(uuid.uuid4()),
            rule_id=rule.id,
            tenant_id=tenant_id,
            check_timestamp=datetime.utcnow(),
        )

        # Run validation based on rule type
        if rule.rule_type == "data_protection":
            await self._check_data_protection_compliance(check, rule)
        elif rule.rule_type == "access_control":
            await self._check_access_control_compliance(check, rule)
        elif rule.rule_type == "audit_log":
            await self._check_audit_log_compliance(check, rule)
        elif rule.rule_type == "financial_control":
            await self._check_financial_control_compliance(check, rule)
        else:
            check.status = "not_applicable"

        # Set next check date
        check.next_check_due = datetime.utcnow() + timedelta(days=30)

        return check

    async def _check_data_protection_compliance(
        self, check: ComplianceCheck, rule: ComplianceRule
    ) -> None:
        """Check data protection compliance."""
        criteria = rule.validation_criteria
        compliant = True
        issues = []

        # Check encryption
        if criteria.get("encryption_enabled"):
            if not await self._is_encryption_enabled(check.tenant_id):
                compliant = False
                issues.append("Data encryption is not enabled")

        # Check retention policies
        if criteria.get("max_retention_days"):
            max_days = criteria["max_retention_days"]
            violations = await self._check_retention_compliance(
                check.tenant_id, max_days
            )
            if violations:
                compliant = False
                issues.extend(violations)

        # Check auto-deletion
        if criteria.get("auto_delete_enabled"):
            if not await self._is_auto_delete_enabled(check.tenant_id):
                compliant = False
                issues.append("Automatic data deletion is not enabled")

        # Set check status
        if compliant:
            check.status = "compliant"
        else:
            check.status = "non_compliant"
            check.details["issues"] = issues
            check.recommendations = [
                "Enable data encryption for all personal data",
                "Configure appropriate retention policies",
                "Enable automatic deletion of expired data",
            ]

    async def _check_audit_log_compliance(
        self, check: ComplianceCheck, rule: ComplianceRule
    ) -> None:
        """Check audit log compliance."""
        criteria = rule.validation_criteria
        compliant = True
        issues = []

        # Check if audit logging is enabled
        if criteria.get("audit_logging_enabled"):
            recent_events = await self._audit_repo.get_recent_audit_events(
                check.tenant_id, days=7
            )
            if not recent_events:
                compliant = False
                issues.append("No recent audit events found - logging may be disabled")

        # Check log retention
        if criteria.get("log_retention_days"):
            retention_days = criteria["log_retention_days"]
            oldest_event = await self._audit_repo.get_oldest_audit_event(
                check.tenant_id
            )
            if oldest_event:
                age_days = (datetime.utcnow() - oldest_event.timestamp).days
                if age_days < retention_days:
                    # Not enough history yet, this is okay
                    pass
                else:
                    # Check if old logs are being retained properly
                    cutoff = datetime.utcnow() - timedelta(days=retention_days)
                    old_events = await self._audit_repo.count_events_before(
                        check.tenant_id, cutoff
                    )
                    if old_events == 0:
                        compliant = False
                        issues.append(
                            f"Audit logs are not being retained for required {retention_days} days"
                        )

        # Set check status
        if compliant:
            check.status = "compliant"
        else:
            check.status = "non_compliant"
            check.details["issues"] = issues

    # Encryption Key Management
    async def create_encryption_key(
        self,
        key_name: str,
        algorithm: str,
        key_size: int,
        tenant_id: TenantId,
        purpose: str,
        user_id: UserId,
    ) -> EncryptionKey:
        """Create and register a new encryption key."""
        key = EncryptionKey(
            id=str(uuid.uuid4()),
            key_name=key_name,
            algorithm=algorithm,
            key_size=key_size,
            tenant_id=tenant_id,
            purpose=purpose,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),  # 1 year default
        )

        # Store key metadata
        await self._compliance_repo.create_encryption_key(key)

        # Log key creation
        await self.log_audit_event(
            action=AuditAction.ENCRYPTION_KEY_ROTATED,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="encryption_key",
            resource_id=key.id,
            details={"algorithm": algorithm, "key_size": key_size, "purpose": purpose},
            severity=AuditSeverity.HIGH,
        )

        return key

    async def rotate_encryption_keys(self, tenant_id: TenantId, user_id: UserId) -> int:
        """Rotate encryption keys that need rotation."""
        keys = await self._compliance_repo.get_encryption_keys(tenant_id)
        rotated_count = 0

        for key in keys:
            if key.needs_rotation:
                # Create new key
                new_key = await self.create_encryption_key(
                    key_name=f"{key.key_name}_rotated",
                    algorithm=key.algorithm,
                    key_size=key.key_size,
                    tenant_id=tenant_id,
                    purpose=key.purpose,
                    user_id=user_id,
                )

                # Mark old key as retired
                key.status = "retired"
                key.rotated_at = datetime.utcnow()
                await self._compliance_repo.update_encryption_key(key)

                rotated_count += 1

        return rotated_count

    # Backup Management
    async def create_backup_record(
        self,
        backup_type: str,
        tenant_id: TenantId,
        data_types: list[str],
        backup_location: str,
        encryption_key_id: str,
        user_id: UserId,
    ) -> BackupRecord:
        """Create a backup operation record."""
        backup = BackupRecord(
            id=str(uuid.uuid4()),
            backup_type=backup_type,
            tenant_id=tenant_id,
            data_types=data_types,
            backup_location=backup_location,
            encryption_key_id=encryption_key_id,
            started_at=datetime.utcnow(),
            retention_until=datetime.utcnow() + timedelta(days=2555),  # 7 years
        )

        # Store backup record
        await self._compliance_repo.create_backup_record(backup)

        # Log backup creation
        await self.log_audit_event(
            action=AuditAction.BACKUP_CREATED,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="backup",
            resource_id=backup.id,
            details={
                "backup_type": backup_type,
                "data_types": data_types,
                "encrypted": True,
            },
            severity=AuditSeverity.MEDIUM,
        )

        return backup

    # Utility Methods
    def _determine_severity(self, action: AuditAction) -> AuditSeverity:
        """Determine severity level for an audit action."""
        high_severity_actions = [
            AuditAction.USER_DELETED,
            AuditAction.DATASET_DELETED,
            AuditAction.MODEL_DELETED,
            AuditAction.PERMISSIONS_CHANGED,
            AuditAction.SYSTEM_CONFIGURATION_CHANGED,
            AuditAction.GDPR_REQUEST,
        ]

        critical_severity_actions = [
            AuditAction.TENANT_SUSPENDED,
            AuditAction.ENCRYPTION_KEY_ROTATED,
            AuditAction.BACKUP_RESTORED,
        ]

        if action in critical_severity_actions:
            return AuditSeverity.CRITICAL
        elif action in high_severity_actions:
            return AuditSeverity.HIGH
        elif "LOGIN" in action.value or "LOGOUT" in action.value:
            return AuditSeverity.LOW
        else:
            return AuditSeverity.MEDIUM

    def _calculate_risk_score(
        self, action: AuditAction, details: dict[str, Any]
    ) -> int:
        """Calculate risk score (0-100) for an audit event."""
        base_scores = {
            AuditAction.USER_DELETED: 90,
            AuditAction.DATASET_DELETED: 85,
            AuditAction.MODEL_DELETED: 80,
            AuditAction.PERMISSIONS_CHANGED: 75,
            AuditAction.SYSTEM_CONFIGURATION_CHANGED: 70,
            AuditAction.GDPR_REQUEST: 60,
            AuditAction.USER_LOGIN: 10,
            AuditAction.USER_LOGOUT: 5,
        }

        score = base_scores.get(action, 30)  # Default medium risk

        # Adjust based on details
        if details.get("failed_attempt"):
            score += 20
        if details.get("admin_action"):
            score += 15
        if details.get("bulk_operation"):
            score += 10

        return min(score, 100)

    async def _get_applicable_frameworks(
        self, action: AuditAction, tenant_id: TenantId
    ) -> list[ComplianceFramework]:
        """Determine which compliance frameworks apply to an action."""
        # Get tenant's active compliance frameworks
        tenant_frameworks = await self._compliance_repo.get_tenant_frameworks(tenant_id)

        # All frameworks care about these actions
        universal_actions = [
            AuditAction.USER_DELETED,
            AuditAction.DATASET_DELETED,
            AuditAction.PERMISSIONS_CHANGED,
            AuditAction.SYSTEM_CONFIGURATION_CHANGED,
        ]

        if action in universal_actions:
            return tenant_frameworks

        # GDPR-specific actions
        gdpr_actions = [
            AuditAction.GDPR_REQUEST,
            AuditAction.DATA_RETENTION_POLICY_APPLIED,
        ]
        if action in gdpr_actions and ComplianceFramework.GDPR in tenant_frameworks:
            return [ComplianceFramework.GDPR]

        return tenant_frameworks

    async def _check_immediate_compliance_violations(self, event: AuditEvent) -> None:
        """Check for immediate compliance violations after an audit event."""
        # Check for suspicious patterns
        if event.is_high_risk:
            # Alert on high-risk events
            await self._create_compliance_alert(event, "High-risk audit event detected")

        # Check for failed access attempts
        if "failed" in event.outcome and event.action == AuditAction.USER_LOGIN:
            recent_failures = await self._audit_repo.count_recent_failed_logins(
                event.tenant_id, event.user_id, hours=1
            )
            if recent_failures >= 5:
                await self._create_compliance_alert(
                    event, f"Multiple failed login attempts: {recent_failures}"
                )

    async def _create_compliance_alert(self, event: AuditEvent, message: str) -> None:
        """Create a compliance alert for immediate attention."""
        # This would integrate with the notification system
        alert_details = {
            "alert_type": "compliance_violation",
            "message": message,
            "event_id": event.id,
            "severity": event.severity.value,
            "timestamp": event.timestamp.isoformat(),
        }

        # Log the alert as an audit event
        await self.log_audit_event(
            action=AuditAction.ALERT_TRIGGERED,
            tenant_id=event.tenant_id,
            details=alert_details,
            severity=AuditSeverity.HIGH,
        )

    def _generate_compliance_report(
        self,
        tenant_id: TenantId,
        framework: ComplianceFramework,
        checks: list[ComplianceCheck],
        user_id: UserId,
    ) -> ComplianceReport:
        """Generate a comprehensive compliance report."""
        now = datetime.utcnow()

        # Calculate statistics
        total_checks = len(checks)
        compliant_checks = len([c for c in checks if c.status == "compliant"])
        non_compliant_checks = len([c for c in checks if c.status == "non_compliant"])
        warning_checks = len([c for c in checks if c.status == "warning"])

        # Generate recommendations
        recommendations = []
        for check in checks:
            if check.needs_attention:
                recommendations.extend(check.recommendations)

        # Create report
        report = ComplianceReport(
            id=str(uuid.uuid4()),
            report_type="periodic",
            framework=framework,
            tenant_id=tenant_id,
            reporting_period_start=now - timedelta(days=30),
            reporting_period_end=now,
            generated_at=now,
            generated_by=user_id,
            total_checks=total_checks,
            compliant_checks=compliant_checks,
            non_compliant_checks=non_compliant_checks,
            warning_checks=warning_checks,
            findings=checks,
            recommendations=list(set(recommendations)),  # Remove duplicates
        )

        return report

    # Placeholder methods for data operations
    async def _validate_retention_policy(self, policy: DataRetentionPolicy) -> None:
        """Validate a retention policy."""
        if policy.retention_period_days < 1:
            raise ValidationError("Retention period must be at least 1 day")
        if policy.retention_period_days > 10000:  # ~27 years
            raise ValidationError("Retention period cannot exceed 10,000 days")

    async def _archive_data_record(self, data_record: Any) -> None:
        """Archive a data record."""
        # TODO: Implement data archiving
        pass

    async def _delete_data_record(self, data_record: Any) -> None:
        """Delete a data record."""
        # TODO: Implement secure data deletion
        pass

    async def _process_data_erasure(self, gdpr_request: GDPRRequest) -> None:
        """Process GDPR data erasure request."""
        # TODO: Implement data erasure
        pass

    async def _process_data_access(self, gdpr_request: GDPRRequest) -> None:
        """Process GDPR data access request."""
        # TODO: Implement data access
        pass

    async def _process_data_portability(self, gdpr_request: GDPRRequest) -> None:
        """Process GDPR data portability request."""
        # TODO: Implement data portability
        pass

    async def _is_encryption_enabled(self, tenant_id: TenantId) -> bool:
        """Check if encryption is enabled for tenant."""
        # TODO: Implement encryption check
        return True

    async def _check_retention_compliance(
        self, tenant_id: TenantId, max_days: int
    ) -> list[str]:
        """Check retention policy compliance."""
        # TODO: Implement retention compliance check
        return []

    async def _is_auto_delete_enabled(self, tenant_id: TenantId) -> bool:
        """Check if auto-delete is enabled for tenant."""
        # TODO: Implement auto-delete check
        return True

    async def _check_access_control_compliance(
        self, check: ComplianceCheck, rule: ComplianceRule
    ) -> None:
        """Check access control compliance."""
        # TODO: Implement access control compliance check
        check.status = "compliant"

    async def _check_financial_control_compliance(
        self, check: ComplianceCheck, rule: ComplianceRule
    ) -> None:
        """Check financial control compliance."""
        # TODO: Implement financial control compliance check
        check.status = "compliant"
