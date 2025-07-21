"""
Enterprise Governance Service

This service orchestrates governance, audit, compliance, and SLA management
operations for enterprise governance requirements.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from structlog import get_logger

from ...domain.entities.audit_log import AuditLog, AuditQuery, AuditStatistics, AuditRetentionPolicy
from ...domain.entities.compliance import (
    ComplianceControl, ComplianceAssessment, ComplianceReport, DataPrivacyRecord,
    ComplianceFramework, ComplianceStatus, ControlStatus
)
from ...domain.entities.sla import (
    ServiceLevelAgreement, SLAMetric, SLAViolation,
    SLAStatus, SLAViolationSeverity
)

logger = get_logger(__name__)


class GovernanceService:
    """
    Enterprise Governance Service
    
    Provides comprehensive governance capabilities including:
    - Audit logging and trail management
    - Compliance framework implementation
    - SLA monitoring and violation management
    - Regulatory reporting and documentation
    - Data privacy and protection compliance
    """
    
    def __init__(
        self,
        audit_repository,
        compliance_repository,
        sla_repository,
        notification_service,
        report_generator
    ):
        self.audit_repo = audit_repository
        self.compliance_repo = compliance_repository
        self.sla_repo = sla_repository
        self.notification_service = notification_service
        self.report_generator = report_generator
        
        logger.info("GovernanceService initialized")
    
    # Audit Management Methods
    
    async def create_audit_log(
        self,
        tenant_id: UUID,
        event_type: str,
        user_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> AuditLog:
        """Create a new audit log entry."""
        logger.info("Creating audit log entry", event_type=event_type, tenant_id=tenant_id)
        
        try:
            audit_log = AuditLog(
                event_type=event_type,
                category=self._categorize_event(event_type),
                severity=self._determine_severity(event_type),
                tenant_id=tenant_id,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                message=self._generate_message(event_type, details or {}),
                details=details or {},
                ip_address=ip_address,
                source_system="anomaly_detection-enterprise",
                environment=self._get_environment(),
                compliance_tags=self._get_compliance_tags(event_type)
            )
            
            # Calculate integrity checksum
            audit_log.checksum = audit_log.calculate_checksum()
            
            # Save audit log
            saved_log = await self.audit_repo.create(audit_log)
            
            # Check if this is a security event that needs immediate attention
            if saved_log.requires_immediate_attention():
                await self._handle_security_event(saved_log)
            
            logger.info("Audit log created", audit_log_id=saved_log.id)
            return saved_log
            
        except Exception as e:
            logger.error("Failed to create audit log", error=str(e), event_type=event_type)
            raise
    
    async def search_audit_logs(self, query: AuditQuery) -> Dict[str, Any]:
        """Search audit logs with filtering and pagination."""
        logger.info("Searching audit logs", tenant_id=query.tenant_id)
        
        try:
            # Execute search query
            results = await self.audit_repo.search(query)
            
            # Generate audit statistics if requested
            stats = await self._generate_audit_statistics(query) if query.include_stats else None
            
            return {
                "logs": results,
                "total_count": len(results),
                "query": query.dict(),
                "statistics": stats
            }
            
        except Exception as e:
            logger.error("Audit log search failed", error=str(e))
            raise
    
    async def generate_audit_report(
        self,
        tenant_id: UUID,
        start_time: datetime,
        end_time: datetime,
        report_format: str = "pdf"
    ) -> str:
        """Generate comprehensive audit report."""
        logger.info("Generating audit report", tenant_id=tenant_id, format=report_format)
        
        try:
            # Create query for report period
            query = AuditQuery(
                tenant_id=tenant_id,
                start_time=start_time,
                end_time=end_time,
                page_size=10000  # Large page size for report
            )
            
            # Get audit logs and statistics
            audit_data = await self.search_audit_logs(query)
            statistics = await self._generate_audit_statistics(query)
            
            # Generate report
            report_path = await self.report_generator.generate_audit_report(
                audit_data=audit_data,
                statistics=statistics,
                format=report_format,
                tenant_id=tenant_id
            )
            
            logger.info("Audit report generated", report_path=report_path)
            return report_path
            
        except Exception as e:
            logger.error("Audit report generation failed", error=str(e))
            raise
    
    # Compliance Management Methods
    
    async def create_compliance_assessment(
        self,
        tenant_id: UUID,
        framework: ComplianceFramework,
        assessment_name: str,
        scope: str,
        lead_assessor: str
    ) -> ComplianceAssessment:
        """Create a new compliance assessment."""
        logger.info("Creating compliance assessment", framework=framework, tenant_id=tenant_id)
        
        try:
            assessment = ComplianceAssessment(
                tenant_id=tenant_id,
                framework=framework,
                assessment_name=assessment_name,
                description=f"Compliance assessment for {framework.value.upper()} framework",
                scope=scope,
                start_date=date.today(),
                lead_assessor=lead_assessor
            )
            
            # Load framework controls
            controls = await self._load_framework_controls(framework, tenant_id)
            assessment.total_controls = len(controls)
            
            # Save assessment
            saved_assessment = await self.compliance_repo.create_assessment(assessment)
            
            # Create audit log
            await self.create_audit_log(
                tenant_id=tenant_id,
                event_type="compliance.assessment_created",
                details={
                    "assessment_id": str(saved_assessment.id),
                    "framework": framework,
                    "controls_count": len(controls)
                }
            )
            
            logger.info("Compliance assessment created", assessment_id=saved_assessment.id)
            return saved_assessment
            
        except Exception as e:
            logger.error("Failed to create compliance assessment", error=str(e))
            raise
    
    async def update_control_status(
        self,
        control_id: UUID,
        status: ControlStatus,
        evidence_refs: Optional[List[str]] = None,
        notes: str = ""
    ) -> ComplianceControl:
        """Update compliance control status and evidence."""
        logger.info("Updating control status", control_id=control_id, status=status)
        
        try:
            control = await self.compliance_repo.get_control(control_id)
            if not control:
                raise ValueError(f"Control {control_id} not found")
            
            # Update control status
            control.update_status(status, notes)
            
            # Add evidence if provided
            if evidence_refs:
                for evidence_ref in evidence_refs:
                    control.add_evidence(evidence_ref, "document")  # Could be more specific
            
            # Save updated control
            updated_control = await self.compliance_repo.update_control(control)
            
            # Update associated assessments
            await self._update_assessment_compliance(control.tenant_id, control.framework)
            
            # Create audit log
            await self.create_audit_log(
                tenant_id=control.tenant_id,
                event_type="compliance.control_updated",
                details={
                    "control_id": str(control_id),
                    "status": status,
                    "framework": control.framework
                }
            )
            
            logger.info("Control status updated", control_id=control_id)
            return updated_control
            
        except Exception as e:
            logger.error("Failed to update control status", error=str(e), control_id=control_id)
            raise
    
    async def generate_compliance_report(
        self,
        assessment_id: UUID,
        report_type: str = "executive_summary"
    ) -> ComplianceReport:
        """Generate compliance report for assessment."""
        logger.info("Generating compliance report", assessment_id=assessment_id)
        
        try:
            # Get assessment and controls
            assessment = await self.compliance_repo.get_assessment(assessment_id)
            if not assessment:
                raise ValueError(f"Assessment {assessment_id} not found")
            
            controls = await self.compliance_repo.get_controls_by_framework(
                assessment.tenant_id,
                assessment.framework
            )
            
            # Create compliance report
            report = ComplianceReport(
                tenant_id=assessment.tenant_id,
                assessment_id=assessment_id,
                report_type=report_type,
                title=f"{assessment.framework.value.upper()} Compliance Report",
                framework=assessment.framework,
                report_period_start=assessment.start_date,
                report_period_end=assessment.end_date or date.today(),
                generated_by="system",
                executive_summary=self._generate_executive_summary(assessment, controls),
                compliance_score=assessment.compliance_percentage,
                risk_rating=self._calculate_risk_rating(assessment),
                control_effectiveness=self._assess_control_effectiveness(controls)
            )
            
            # Add findings and recommendations
            await self._populate_report_findings(report, controls)
            
            # Save report
            saved_report = await self.compliance_repo.create_report(report)
            
            logger.info("Compliance report generated", report_id=saved_report.id)
            return saved_report
            
        except Exception as e:
            logger.error("Failed to generate compliance report", error=str(e))
            raise
    
    # SLA Management Methods
    
    async def create_sla(
        self,
        tenant_id: UUID,
        name: str,
        sla_type: str,
        service_provider: str,
        service_consumer: str,
        services_covered: List[str],
        overall_target: float,
        effective_date: datetime,
        expiry_date: Optional[datetime] = None
    ) -> ServiceLevelAgreement:
        """Create a new Service Level Agreement."""
        logger.info("Creating SLA", name=name, tenant_id=tenant_id)
        
        try:
            sla = ServiceLevelAgreement(
                name=name,
                sla_type=sla_type,
                service_provider=service_provider,
                service_consumer=service_consumer,
                tenant_id=tenant_id,
                services_covered=services_covered,
                metrics=[],  # Will be added separately
                overall_target=overall_target,
                effective_date=effective_date,
                expiry_date=expiry_date,
                measurement_period="monthly",
                reporting_frequency="monthly",
                review_schedule="quarterly"
            )
            
            # Schedule first review
            sla.schedule_next_review(3)  # 3 months
            
            # Save SLA
            saved_sla = await self.sla_repo.create_sla(sla)
            
            # Create audit log
            await self.create_audit_log(
                tenant_id=tenant_id,
                event_type="sla.created",
                details={
                    "sla_id": str(saved_sla.id),
                    "sla_name": name,
                    "target": overall_target
                }
            )
            
            logger.info("SLA created", sla_id=saved_sla.id)
            return saved_sla
            
        except Exception as e:
            logger.error("Failed to create SLA", error=str(e))
            raise
    
    async def add_sla_metric(
        self,
        sla_id: UUID,
        name: str,
        metric_type: str,
        target_value: float,
        minimum_acceptable: float,
        measurement_unit: str
    ) -> SLAMetric:
        """Add a metric to an SLA."""
        logger.info("Adding SLA metric", sla_id=sla_id, metric_name=name)
        
        try:
            metric = SLAMetric(
                name=name,
                description=f"SLA metric for {name}",
                metric_type=metric_type,
                target_value=target_value,
                minimum_acceptable=minimum_acceptable,
                measurement_unit=measurement_unit,
                measurement_frequency="hourly",
                calculation_method="average",
                data_source="system_metrics",
                aggregation_period="1h"
            )
            
            # Save metric
            saved_metric = await self.sla_repo.create_metric(metric)
            
            # Add metric to SLA
            sla = await self.sla_repo.get_sla(sla_id)
            sla.metrics.append(saved_metric.id)
            await self.sla_repo.update_sla(sla)
            
            logger.info("SLA metric added", metric_id=saved_metric.id)
            return saved_metric
            
        except Exception as e:
            logger.error("Failed to add SLA metric", error=str(e))
            raise
    
    async def record_metric_measurement(
        self,
        metric_id: UUID,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a measurement for an SLA metric."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        try:
            metric = await self.sla_repo.get_metric(metric_id)
            if not metric:
                raise ValueError(f"Metric {metric_id} not found")
            
            # Record measurement
            metric.record_measurement(value, timestamp)
            
            # Save updated metric
            await self.sla_repo.update_metric(metric)
            
            # Check for violations
            if metric.is_in_violation():
                await self._handle_sla_violation(metric, value)
            
            logger.debug("Metric measurement recorded", metric_id=metric_id, value=value)
            
        except Exception as e:
            logger.error("Failed to record metric measurement", error=str(e))
            raise
    
    async def check_sla_compliance(self, tenant_id: UUID) -> Dict[str, Any]:
        """Check SLA compliance for a tenant."""
        logger.info("Checking SLA compliance", tenant_id=tenant_id)
        
        try:
            slas = await self.sla_repo.get_slas_by_tenant(tenant_id)
            compliance_summary = {
                "tenant_id": str(tenant_id),
                "total_slas": len(slas),
                "active_slas": 0,
                "compliant_slas": 0,
                "violations_today": 0,
                "sla_details": []
            }
            
            for sla in slas:
                if sla.is_active():
                    compliance_summary["active_slas"] += 1
                    
                    # Check compliance
                    metrics = await self.sla_repo.get_metrics_by_sla(sla.id)
                    sla_compliance = await self._calculate_sla_compliance(sla, metrics)
                    
                    if sla_compliance >= sla.overall_target:
                        compliance_summary["compliant_slas"] += 1
                    
                    # Update SLA compliance
                    sla.update_compliance(sla_compliance)
                    await self.sla_repo.update_sla(sla)
                    
                    compliance_summary["sla_details"].append({
                        "sla_id": str(sla.id),
                        "name": sla.name,
                        "compliance": sla_compliance,
                        "target": sla.overall_target,
                        "status": sla.status
                    })
            
            # Count today's violations
            today_violations = await self.sla_repo.get_violations_by_date(
                tenant_id,
                date.today()
            )
            compliance_summary["violations_today"] = len(today_violations)
            
            return compliance_summary
            
        except Exception as e:
            logger.error("Failed to check SLA compliance", error=str(e))
            raise
    
    # Private helper methods
    
    def _categorize_event(self, event_type: str) -> str:
        """Categorize audit event type."""
        if "user" in event_type or "login" in event_type:
            return "authentication"
        elif "data" in event_type:
            return "data_access"
        elif "security" in event_type:
            return "security"
        elif "system" in event_type:
            return "system"
        elif "compliance" in event_type:
            return "compliance"
        else:
            return "general"
    
    def _determine_severity(self, event_type: str) -> str:
        """Determine audit event severity."""
        if "breach" in event_type or "violation" in event_type:
            return "critical"
        elif "failed" in event_type or "locked" in event_type:
            return "high"
        elif "accessed" in event_type or "updated" in event_type:
            return "medium"
        else:
            return "low"
    
    def _generate_message(self, event_type: str, details: Dict[str, Any]) -> str:
        """Generate human-readable audit message."""
        # Simple message generation - could be enhanced with templates
        return f"Event: {event_type.replace('_', ' ').title()}"
    
    def _get_environment(self) -> str:
        """Get current environment."""
        import os
        return os.getenv("ANOMALY_DETECTION_ENVIRONMENT", "development")
    
    def _get_compliance_tags(self, event_type: str) -> List[str]:
        """Get compliance tags for event type."""
        tags = []
        if "data" in event_type:
            tags.extend(["gdpr", "ccpa"])
        if "security" in event_type:
            tags.extend(["soc2", "iso27001"])
        if "user" in event_type:
            tags.append("identity_management")
        return tags
    
    async def _handle_security_event(self, audit_log: AuditLog) -> None:
        """Handle security events that need immediate attention."""
        if audit_log.is_security_event():
            await self.notification_service.send_security_alert(
                tenant_id=audit_log.tenant_id,
                event=audit_log,
                severity=audit_log.severity
            )
    
    async def _generate_audit_statistics(self, query: AuditQuery) -> AuditStatistics:
        """Generate audit statistics for query period."""
        # Implementation would aggregate audit data
        return AuditStatistics(
            start_time=query.start_time or datetime.utcnow() - timedelta(days=30),
            end_time=query.end_time or datetime.utcnow()
        )
    
    async def _load_framework_controls(
        self,
        framework: ComplianceFramework,
        tenant_id: UUID
    ) -> List[ComplianceControl]:
        """Load predefined controls for compliance framework."""
        # This would load framework-specific controls from a knowledge base
        return []
    
    async def _update_assessment_compliance(
        self,
        tenant_id: UUID,
        framework: ComplianceFramework
    ) -> None:
        """Update assessment compliance based on control statuses."""
        assessments = await self.compliance_repo.get_assessments_by_framework(
            tenant_id,
            framework
        )
        
        for assessment in assessments:
            if assessment.overall_status == ComplianceStatus.UNDER_REVIEW:
                controls = await self.compliance_repo.get_controls_by_framework(
                    tenant_id,
                    framework
                )
                assessment.update_control_counts(controls)
                await self.compliance_repo.update_assessment(assessment)
    
    def _generate_executive_summary(
        self,
        assessment: ComplianceAssessment,
        controls: List[ComplianceControl]
    ) -> str:
        """Generate executive summary for compliance report."""
        return f"Compliance assessment for {assessment.framework.value.upper()} framework showing {assessment.compliance_percentage:.1f}% compliance across {len(controls)} controls."
    
    def _calculate_risk_rating(self, assessment: ComplianceAssessment) -> str:
        """Calculate overall risk rating."""
        if assessment.compliance_percentage >= 95:
            return "Low"
        elif assessment.compliance_percentage >= 80:
            return "Medium"
        else:
            return "High"
    
    def _assess_control_effectiveness(self, controls: List[ComplianceControl]) -> str:
        """Assess overall control effectiveness."""
        if not controls:
            return "Not Assessed"
        
        implemented = sum(1 for c in controls if c.is_compliant())
        percentage = (implemented / len(controls)) * 100
        
        if percentage >= 95:
            return "Highly Effective"
        elif percentage >= 80:
            return "Effective"
        elif percentage >= 60:
            return "Partially Effective"
        else:
            return "Ineffective"
    
    async def _populate_report_findings(
        self,
        report: ComplianceReport,
        controls: List[ComplianceControl]
    ) -> None:
        """Populate report with findings and recommendations."""
        for control in controls:
            if not control.is_compliant():
                report.add_finding(
                    control_id=control.control_id,
                    title=f"Control {control.control_number} - {control.title}",
                    description=f"Control not implemented: {control.description}",
                    risk_level=control.risk_level,
                    recommendation=f"Implement control: {control.remediation_plan}"
                )
    
    async def _calculate_sla_compliance(
        self,
        sla: ServiceLevelAgreement,
        metrics: List[SLAMetric]
    ) -> float:
        """Calculate overall SLA compliance percentage."""
        if not metrics:
            return 100.0
        
        total_compliance = sum(m.get_compliance_percentage() for m in metrics)
        return total_compliance / len(metrics)
    
    async def _handle_sla_violation(self, metric: SLAMetric, actual_value: float) -> None:
        """Handle SLA metric violation."""
        try:
            violation = SLAViolation(
                sla_id=UUID("00000000-0000-0000-0000-000000000000"),  # Would get from metric
                metric_id=metric.id,
                tenant_id=UUID("00000000-0000-0000-0000-000000000000"),  # Would get from context
                violation_type=metric.name,
                severity=metric.get_violation_severity() or SLAViolationSeverity.MEDIUM,
                description=f"Metric {metric.name} violated: {actual_value} {metric.measurement_unit}",
                start_time=datetime.utcnow(),
                target_value=metric.target_value,
                actual_value=actual_value,
                deviation_percentage=abs((actual_value - metric.target_value) / metric.target_value) * 100
            )
            
            # Save violation
            await self.sla_repo.create_violation(violation)
            
            # Send notification
            await self.notification_service.send_sla_violation_alert(violation)
            
            logger.warning("SLA violation detected", metric_id=metric.id, violation_id=violation.id)
            
        except Exception as e:
            logger.error("Failed to handle SLA violation", error=str(e))