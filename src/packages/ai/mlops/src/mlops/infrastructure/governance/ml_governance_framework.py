"""
ML Governance and Compliance Framework

Comprehensive framework for ML model governance, compliance tracking,
audit trails, and regulatory compliance management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
import structlog

from mlops.domain.entities.model import Model, ModelVersion


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CUSTOM = "custom"


class GovernanceRisk(Enum):
    """Risk levels for governance issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """Types of audit events."""
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_PREDICTION = "model_prediction"
    DATA_ACCESS = "data_access"
    FEATURE_ENGINEERING = "feature_engineering"
    A_B_TEST = "a_b_test"
    DRIFT_DETECTION = "drift_detection"
    COMPLIANCE_CHECK = "compliance_check"
    POLICY_VIOLATION = "policy_violation"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


@dataclass
class GovernancePolicy:
    """ML governance policy definition."""
    policy_id: str
    name: str
    description: str
    compliance_frameworks: List[ComplianceFramework]
    policy_type: str  # "data_privacy", "model_fairness", "security", etc.
    rules: List[Dict[str, Any]] = field(default_factory=list)
    required_approvals: List[str] = field(default_factory=list)  # Role names
    auto_enforcement: bool = True
    severity: GovernanceRisk = GovernanceRisk.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    is_active: bool = True
    
    def add_rule(self, rule_type: str, conditions: Dict[str, Any], actions: List[str]) -> None:
        """Add a governance rule to the policy."""
        rule = {
            "rule_id": str(uuid.uuid4()),
            "rule_type": rule_type,
            "conditions": conditions,
            "actions": actions,
            "created_at": datetime.utcnow().isoformat()
        }
        self.rules.append(rule)
        self.updated_at = datetime.utcnow()


@dataclass
class AuditEvent:
    """Audit trail event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.MODEL_TRAINING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    session_id: str = ""
    model_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    # Event details
    action: str = ""
    resource: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance and governance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    risk_level: GovernanceRisk = GovernanceRisk.LOW
    
    # Technical details
    ip_address: str = ""
    user_agent: str = ""
    request_id: str = ""
    
    # Outcome
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "pipeline_id": self.pipeline_id,
            "experiment_id": self.experiment_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "compliance_frameworks": [cf.value for cf in self.compliance_frameworks],
            "policy_violations": self.policy_violations,
            "risk_level": self.risk_level.value,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "success": self.success,
            "error_message": self.error_message
        }


@dataclass
class ComplianceCheck:
    """Compliance validation result."""
    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    model_id: str = ""
    compliance_framework: ComplianceFramework = ComplianceFramework.GDPR
    check_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Results
    is_compliant: bool = False
    compliance_score: float = 0.0  # 0-1 scale
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Evidence and documentation
    evidence: Dict[str, Any] = field(default_factory=dict)
    documentation_links: List[str] = field(default_factory=list)
    
    # Follow-up actions
    required_actions: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None


@dataclass
class ApprovalRequest:
    """Model deployment/change approval request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: str = ""  # "deployment", "model_change", "policy_exception"
    model_id: str = ""
    requester_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Request details
    title: str = ""
    description: str = ""
    justification: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Approval workflow
    required_approvers: List[str] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    status: str = "pending"  # pending, approved, rejected, cancelled
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    # Compliance
    compliance_requirements: List[ComplianceFramework] = field(default_factory=list)
    
    def is_approved(self) -> bool:
        """Check if request is fully approved."""
        return self.status == "approved"
    
    def add_approval(self, approver_id: str, comments: str = "") -> None:
        """Add an approval."""
        approval = {
            "approver_id": approver_id,
            "timestamp": datetime.utcnow().isoformat(),
            "comments": comments
        }
        self.approvals.append(approval)
        
        # Check if all required approvals are received
        approved_by = {a["approver_id"] for a in self.approvals}
        if all(approver in approved_by for approver in self.required_approvers):
            self.status = "approved"
            self.approved_at = datetime.utcnow()
            self.approved_by = approver_id


class MLGovernanceFramework:
    """Comprehensive ML governance and compliance framework."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.logger = structlog.get_logger(__name__)
        
        # Policy management
        self.policies: Dict[str, GovernancePolicy] = {}
        self.policy_engine = PolicyEngine()
        
        # Audit and compliance
        self.audit_trail: List[AuditEvent] = []
        self.compliance_checker = ComplianceChecker()
        
        # Approval workflow
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.approval_engine = ApprovalEngine()
        
        # Risk management
        self.risk_assessor = RiskAssessor()
        
        # Data retention and archival
        self.data_retention_days = self.config.get("data_retention_days", 2555)  # 7 years default
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load governance configuration."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yml', '.yaml')):
                    import yaml
                    return yaml.safe_load(f)
                else:
                    return {}
        except Exception as e:
            self.logger.warning("Failed to load config", path=config_path, error=str(e))
            return {}
    
    async def create_policy(self, 
                           name: str,
                           description: str,
                           compliance_frameworks: List[ComplianceFramework],
                           policy_type: str,
                           created_by: str) -> str:
        """Create a new governance policy."""
        
        policy_id = str(uuid.uuid4())
        
        policy = GovernancePolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            compliance_frameworks=compliance_frameworks,
            policy_type=policy_type,
            created_by=created_by
        )
        
        self.policies[policy_id] = policy
        
        # Log policy creation
        await self.log_audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user_id=created_by,
            action="create_policy",
            resource=f"policy:{policy_id}",
            details={
                "policy_name": name,
                "policy_type": policy_type,
                "compliance_frameworks": [cf.value for cf in compliance_frameworks]
            }
        )
        
        self.logger.info(
            "Governance policy created",
            policy_id=policy_id,
            name=name,
            type=policy_type,
            frameworks=[cf.value for cf in compliance_frameworks]
        )
        
        return policy_id
    
    async def add_policy_rule(self,
                            policy_id: str,
                            rule_type: str,
                            conditions: Dict[str, Any],
                            actions: List[str]) -> None:
        """Add a rule to an existing policy."""
        
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")
        
        policy = self.policies[policy_id]
        policy.add_rule(rule_type, conditions, actions)
        
        self.logger.info(
            "Policy rule added",
            policy_id=policy_id,
            rule_type=rule_type,
            conditions=conditions,
            actions=actions
        )
    
    async def evaluate_policy_compliance(self,
                                       model_id: str,
                                       action: str,
                                       context: Dict[str, Any]) -> List[ComplianceCheck]:
        """Evaluate model action against governance policies."""
        
        compliance_checks = []
        
        for policy_id, policy in self.policies.items():
            if not policy.is_active:
                continue
            
            # Check if policy applies to this context
            if await self.policy_engine.applies_to_context(policy, context):
                
                for framework in policy.compliance_frameworks:
                    check = await self.compliance_checker.check_compliance(
                        policy, model_id, action, context, framework
                    )
                    compliance_checks.append(check)
                    
                    # Log compliance check
                    await self.log_audit_event(
                        event_type=AuditEventType.COMPLIANCE_CHECK,
                        user_id=context.get("user_id", "system"),
                        model_id=model_id,
                        action=f"compliance_check:{action}",
                        resource=f"policy:{policy_id}",
                        details={
                            "compliance_framework": framework.value,
                            "is_compliant": check.is_compliant,
                            "compliance_score": check.compliance_score,
                            "violations": len(check.violations)
                        },
                        risk_level=GovernanceRisk.HIGH if not check.is_compliant else GovernanceRisk.LOW
                    )
        
        return compliance_checks
    
    async def request_approval(self,
                             request_type: str,
                             model_id: str,
                             requester_id: str,
                             title: str,
                             description: str,
                             justification: str) -> str:
        """Create an approval request."""
        
        request_id = str(uuid.uuid4())
        
        # Determine required approvers based on context
        required_approvers = await self.approval_engine.get_required_approvers(
            request_type, model_id, requester_id
        )
        
        # Perform risk assessment
        risk_assessment = await self.risk_assessor.assess_request_risk(
            request_type, model_id, description
        )
        
        request = ApprovalRequest(
            request_id=request_id,
            request_type=request_type,
            model_id=model_id,
            requester_id=requester_id,
            title=title,
            description=description,
            justification=justification,
            required_approvers=required_approvers,
            risk_assessment=risk_assessment
        )
        
        self.approval_requests[request_id] = request
        
        # Log approval request
        await self.log_audit_event(
            event_type=AuditEventType.APPROVAL_REQUEST,
            user_id=requester_id,
            model_id=model_id,
            action=f"request_approval:{request_type}",
            resource=f"approval_request:{request_id}",
            details={
                "title": title,
                "required_approvers": required_approvers,
                "risk_level": risk_assessment.get("risk_level", "medium")
            }
        )
        
        # Notify approvers
        await self._notify_approvers(request)
        
        self.logger.info(
            "Approval request created",
            request_id=request_id,
            request_type=request_type,
            model_id=model_id,
            required_approvers=required_approvers
        )
        
        return request_id
    
    async def approve_request(self,
                            request_id: str,
                            approver_id: str,
                            comments: str = "") -> bool:
        """Approve an approval request."""
        
        if request_id not in self.approval_requests:
            raise ValueError(f"Approval request {request_id} not found")
        
        request = self.approval_requests[request_id]
        
        if approver_id not in request.required_approvers:
            raise ValueError(f"User {approver_id} is not authorized to approve this request")
        
        # Add approval
        request.add_approval(approver_id, comments)
        
        # Log approval
        await self.log_audit_event(
            event_type=AuditEventType.APPROVAL_GRANTED,
            user_id=approver_id,
            model_id=request.model_id,
            action=f"approve_request:{request.request_type}",
            resource=f"approval_request:{request_id}",
            details={
                "comments": comments,
                "is_fully_approved": request.is_approved()
            }
        )
        
        self.logger.info(
            "Approval granted",
            request_id=request_id,
            approver_id=approver_id,
            fully_approved=request.is_approved()
        )
        
        return request.is_approved()
    
    async def log_audit_event(self,
                            event_type: AuditEventType,
                            user_id: str,
                            action: str,
                            resource: str,
                            details: Dict[str, Any] = None,
                            model_id: str = None,
                            pipeline_id: str = None,
                            experiment_id: str = None,
                            session_id: str = None,
                            ip_address: str = "",
                            user_agent: str = "",
                            request_id: str = "",
                            success: bool = True,
                            error_message: str = None,
                            risk_level: GovernanceRisk = GovernanceRisk.LOW) -> str:
        """Log an audit event."""
        
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            model_id=model_id,
            pipeline_id=pipeline_id,
            experiment_id=experiment_id,
            action=action,
            resource=resource,
            details=details or {},
            risk_level=risk_level,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            success=success,
            error_message=error_message
        )
        
        # Add to audit trail
        self.audit_trail.append(event)
        
        # In production, would persist to secure audit database
        self._persist_audit_event(event)
        
        return event.event_id
    
    async def generate_compliance_report(self,
                                       compliance_framework: ComplianceFramework,
                                       start_date: datetime = None,
                                       end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Filter audit events by timeframe
        relevant_events = [
            event for event in self.audit_trail
            if start_date <= event.timestamp <= end_date
        ]
        
        # Filter by compliance framework
        framework_events = [
            event for event in relevant_events
            if compliance_framework in event.compliance_frameworks
        ]
        
        # Analyze compliance violations
        violations = [
            event for event in framework_events
            if event.policy_violations
        ]
        
        # Group by risk level
        risk_distribution = {
            "low": len([e for e in framework_events if e.risk_level == GovernanceRisk.LOW]),
            "medium": len([e for e in framework_events if e.risk_level == GovernanceRisk.MEDIUM]),
            "high": len([e for e in framework_events if e.risk_level == GovernanceRisk.HIGH]),
            "critical": len([e for e in framework_events if e.risk_level == GovernanceRisk.CRITICAL])
        }
        
        # Model compliance status
        model_compliance = {}
        for event in framework_events:
            if event.model_id:
                if event.model_id not in model_compliance:
                    model_compliance[event.model_id] = {
                        "total_events": 0,
                        "violations": 0,
                        "risk_score": 0
                    }
                
                model_compliance[event.model_id]["total_events"] += 1
                if event.policy_violations:
                    model_compliance[event.model_id]["violations"] += 1
                
                # Add to risk score
                risk_scores = {
                    GovernanceRisk.LOW: 1,
                    GovernanceRisk.MEDIUM: 3,
                    GovernanceRisk.HIGH: 7,
                    GovernanceRisk.CRITICAL: 10
                }
                model_compliance[event.model_id]["risk_score"] += risk_scores.get(event.risk_level, 0)
        
        # Calculate compliance metrics
        total_events = len(framework_events)
        violation_rate = len(violations) / total_events if total_events > 0 else 0
        
        report = {
            "compliance_framework": compliance_framework.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": total_events,
                "violations": len(violations),
                "violation_rate": violation_rate,
                "compliance_rate": 1.0 - violation_rate
            },
            "risk_distribution": risk_distribution,
            "model_compliance": model_compliance,
            "top_violations": self._get_top_violations(violations),
            "recommendations": self._generate_compliance_recommendations(
                framework_events, compliance_framework
            ),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report
    
    async def get_audit_trail(self,
                            user_id: str = None,
                            model_id: str = None,
                            event_type: AuditEventType = None,
                            start_date: datetime = None,
                            end_date: datetime = None,
                            limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filtering."""
        
        events = self.audit_trail
        
        # Apply filters
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if model_id:
            events = [e for e in events if e.model_id == model_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        events = events[:limit]
        
        return [event.to_dict() for event in events]
    
    async def start_monitoring(self) -> None:
        """Start background governance monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._periodic_compliance_check()),
            asyncio.create_task(self._audit_trail_maintenance()),
            asyncio.create_task(self._approval_request_monitoring())
        ]
        
        self.logger.info("ML governance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background governance monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("ML governance monitoring stopped")
    
    async def _periodic_compliance_check(self) -> None:
        """Periodic compliance verification."""
        while self.is_running:
            try:
                # Check for policy violations in recent activities
                recent_events = [
                    e for e in self.audit_trail
                    if e.timestamp >= datetime.utcnow() - timedelta(hours=1)
                ]
                
                # Analyze for patterns that might indicate compliance issues
                risk_patterns = await self._analyze_risk_patterns(recent_events)
                
                for pattern in risk_patterns:
                    await self.log_audit_event(
                        event_type=AuditEventType.COMPLIANCE_CHECK,
                        user_id="system",
                        action="risk_pattern_detected",
                        resource="governance_system",
                        details=pattern,
                        risk_level=GovernanceRisk.HIGH
                    )
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error("Error in periodic compliance check", error=str(e))
    
    async def _audit_trail_maintenance(self) -> None:
        """Maintain audit trail and archive old events."""
        while self.is_running:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=self.data_retention_days)
                
                # Archive old events (in production, move to long-term storage)
                old_events = [
                    e for e in self.audit_trail
                    if e.timestamp < cutoff_date
                ]
                
                if old_events:
                    await self._archive_audit_events(old_events)
                    
                    # Remove from active trail
                    self.audit_trail = [
                        e for e in self.audit_trail
                        if e.timestamp >= cutoff_date
                    ]
                    
                    self.logger.info(
                        "Archived old audit events",
                        archived_count=len(old_events),
                        active_count=len(self.audit_trail)
                    )
                
                await asyncio.sleep(86400)  # Check daily
                
            except Exception as e:
                self.logger.error("Error in audit trail maintenance", error=str(e))
    
    async def _approval_request_monitoring(self) -> None:
        """Monitor approval requests for timeouts and escalation."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for request_id, request in self.approval_requests.items():
                    if request.status != "pending":
                        continue
                    
                    # Check for timeout (e.g., 72 hours)
                    if (current_time - request.created_at).total_seconds() > 259200:  # 72 hours
                        # Escalate request
                        await self._escalate_approval_request(request)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error("Error in approval request monitoring", error=str(e))
    
    def _persist_audit_event(self, event: AuditEvent) -> None:
        """Persist audit event to secure storage."""
        # In production, this would write to a secure, immutable audit database
        # For now, just log it
        self.logger.info(
            "Audit event",
            event_id=event.event_id,
            event_type=event.event_type.value,
            user_id=event.user_id,
            action=event.action,
            resource=event.resource,
            success=event.success,
            risk_level=event.risk_level.value
        )
    
    async def _notify_approvers(self, request: ApprovalRequest) -> None:
        """Notify required approvers of pending request."""
        # In production, would send actual notifications (email, Slack, etc.)
        self.logger.info(
            "Approval notification sent",
            request_id=request.request_id,
            approvers=request.required_approvers,
            title=request.title
        )
    
    async def _escalate_approval_request(self, request: ApprovalRequest) -> None:
        """Escalate overdue approval request."""
        await self.log_audit_event(
            event_type=AuditEventType.APPROVAL_REQUEST,
            user_id="system",
            model_id=request.model_id,
            action="escalate_approval_request",
            resource=f"approval_request:{request.request_id}",
            details={
                "original_request_time": request.created_at.isoformat(),
                "overdue_hours": (datetime.utcnow() - request.created_at).total_seconds() / 3600
            },
            risk_level=GovernanceRisk.HIGH
        )
    
    def _get_top_violations(self, violations: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Get most common policy violations."""
        violation_counts = {}
        
        for event in violations:
            for violation in event.policy_violations:
                if violation not in violation_counts:
                    violation_counts[violation] = 0
                violation_counts[violation] += 1
        
        # Sort by frequency
        sorted_violations = sorted(
            violation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"violation": violation, "count": count}
            for violation, count in sorted_violations[:10]
        ]
    
    def _generate_compliance_recommendations(self,
                                           events: List[AuditEvent],
                                           framework: ComplianceFramework) -> List[str]:
        """Generate compliance improvement recommendations."""
        recommendations = []
        
        # Analyze violation patterns
        violation_types = {}
        for event in events:
            if event.policy_violations:
                for violation in event.policy_violations:
                    if violation not in violation_types:
                        violation_types[violation] = 0
                    violation_types[violation] += 1
        
        # Generate recommendations based on common violations
        if violation_types:
            most_common = max(violation_types.items(), key=lambda x: x[1])
            recommendations.append(
                f"Address most frequent violation: {most_common[0]} ({most_common[1]} occurrences)"
            )
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Implement data minimization principles in feature selection",
                "Ensure explicit consent tracking for all data processing",
                "Establish automated data deletion workflows"
            ])
        elif framework == ComplianceFramework.HIPAA:
            recommendations.extend([
                "Implement additional encryption for PHI data",
                "Establish audit log monitoring for all PHI access",
                "Review and update business associate agreements"
            ])
        
        return recommendations
    
    async def _analyze_risk_patterns(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Analyze events for risk patterns."""
        patterns = []
        
        # Check for unusual access patterns
        user_activity = {}
        for event in events:
            if event.user_id not in user_activity:
                user_activity[event.user_id] = []
            user_activity[event.user_id].append(event)
        
        # Detect users with unusual activity levels
        for user_id, user_events in user_activity.items():
            if len(user_events) > 100:  # Threshold for unusual activity
                patterns.append({
                    "pattern_type": "unusual_user_activity",
                    "user_id": user_id,
                    "event_count": len(user_events),
                    "risk_level": "medium"
                })
        
        return patterns
    
    async def _archive_audit_events(self, events: List[AuditEvent]) -> None:
        """Archive old audit events to long-term storage."""
        # In production, would move to secure archival storage
        self.logger.info(
            "Archiving audit events",
            event_count=len(events),
            oldest_event=min(e.timestamp for e in events).isoformat(),
            newest_event=max(e.timestamp for e in events).isoformat()
        )


class PolicyEngine:
    """Engine for evaluating governance policies."""
    
    async def applies_to_context(self, policy: GovernancePolicy, context: Dict[str, Any]) -> bool:
        """Check if policy applies to the given context."""
        # Simple implementation - in production would be more sophisticated
        return True  # For now, assume all policies apply to all contexts


class ComplianceChecker:
    """Checks compliance against various frameworks."""
    
    async def check_compliance(self,
                             policy: GovernancePolicy,
                             model_id: str,
                             action: str,
                             context: Dict[str, Any],
                             framework: ComplianceFramework) -> ComplianceCheck:
        """Check compliance for a specific action."""
        
        check = ComplianceCheck(
            policy_id=policy.policy_id,
            model_id=model_id,
            compliance_framework=framework,
            check_type=action
        )
        
        # Perform framework-specific compliance checks
        if framework == ComplianceFramework.GDPR:
            check = await self._check_gdpr_compliance(check, policy, context)
        elif framework == ComplianceFramework.HIPAA:
            check = await self._check_hipaa_compliance(check, policy, context)
        # Add more frameworks as needed
        
        return check
    
    async def _check_gdpr_compliance(self,
                                   check: ComplianceCheck,
                                   policy: GovernancePolicy,
                                   context: Dict[str, Any]) -> ComplianceCheck:
        """Check GDPR compliance requirements."""
        
        compliance_score = 1.0
        violations = []
        recommendations = []
        
        # Check for data subject consent
        if not context.get("consent_provided", False):
            violations.append({
                "violation_type": "missing_consent",
                "description": "No explicit consent recorded for data processing",
                "severity": "high"
            })
            compliance_score -= 0.3
        
        # Check for data minimization
        feature_count = context.get("feature_count", 0)
        if feature_count > 100:  # Arbitrary threshold
            violations.append({
                "violation_type": "data_minimization",
                "description": f"High number of features ({feature_count}) may violate data minimization",
                "severity": "medium"
            })
            compliance_score -= 0.2
        
        # Check for purpose limitation
        if not context.get("processing_purpose"):
            violations.append({
                "violation_type": "purpose_limitation",
                "description": "No clear processing purpose documented",
                "severity": "medium"
            })
            compliance_score -= 0.2
        
        check.compliance_score = max(0.0, compliance_score)
        check.is_compliant = compliance_score >= 0.8
        check.violations = violations
        
        if not check.is_compliant:
            recommendations.extend([
                "Obtain explicit consent for data processing",
                "Document clear purpose for data processing",
                "Review feature selection for data minimization"
            ])
        
        check.recommendations = recommendations
        
        return check
    
    async def _check_hipaa_compliance(self,
                                    check: ComplianceCheck,
                                    policy: GovernancePolicy,
                                    context: Dict[str, Any]) -> ComplianceCheck:
        """Check HIPAA compliance requirements."""
        
        compliance_score = 1.0
        violations = []
        recommendations = []
        
        # Check for PHI handling
        contains_phi = context.get("contains_phi", False)
        if contains_phi:
            # Check encryption
            if not context.get("data_encrypted", False):
                violations.append({
                    "violation_type": "phi_encryption",
                    "description": "PHI data not properly encrypted",
                    "severity": "critical"
                })
                compliance_score -= 0.5
            
            # Check access controls
            if not context.get("access_controls_verified", False):
                violations.append({
                    "violation_type": "access_controls",
                    "description": "Access controls not verified for PHI access",
                    "severity": "high"
                })
                compliance_score -= 0.3
        
        check.compliance_score = max(0.0, compliance_score)
        check.is_compliant = compliance_score >= 0.8
        check.violations = violations
        
        if not check.is_compliant:
            recommendations.extend([
                "Implement end-to-end encryption for PHI data",
                "Establish role-based access controls",
                "Conduct regular access audits"
            ])
        
        check.recommendations = recommendations
        
        return check


class ApprovalEngine:
    """Manages approval workflows."""
    
    async def get_required_approvers(self,
                                   request_type: str,
                                   model_id: str,
                                   requester_id: str) -> List[str]:
        """Determine required approvers for a request."""
        
        approvers = []
        
        # Basic approval matrix
        if request_type == "deployment":
            approvers.extend(["ml_manager", "security_officer"])
            
        elif request_type == "model_change":
            approvers.extend(["ml_manager"])
            
        elif request_type == "policy_exception":
            approvers.extend(["compliance_officer", "ml_manager", "security_officer"])
        
        # Add data owner if applicable
        approvers.append("data_owner")
        
        return list(set(approvers))  # Remove duplicates


class RiskAssessor:
    """Assesses risks for governance decisions."""
    
    async def assess_request_risk(self,
                                request_type: str,
                                model_id: str,
                                description: str) -> Dict[str, Any]:
        """Assess risk level for an approval request."""
        
        risk_score = 0
        risk_factors = []
        
        # Risk factors based on request type
        if request_type == "deployment":
            risk_score += 5
            risk_factors.append("Production deployment")
            
        elif request_type == "policy_exception":
            risk_score += 8
            risk_factors.append("Policy exception requested")
        
        # Risk factors based on description keywords
        high_risk_keywords = ["production", "customer", "financial", "medical", "personal"]
        for keyword in high_risk_keywords:
            if keyword.lower() in description.lower():
                risk_score += 2
                risk_factors.append(f"Contains high-risk keyword: {keyword}")
        
        # Determine risk level
        if risk_score >= 10:
            risk_level = "critical"
        elif risk_score >= 7:
            risk_level = "high"
        elif risk_score >= 4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }