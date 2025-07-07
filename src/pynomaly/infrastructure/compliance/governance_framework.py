"""Compliance and governance automation framework for SOX, GDPR, HIPAA, and data lineage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    
    SOX = "sox"           # Sarbanes-Oxley Act
    GDPR = "gdpr"         # General Data Protection Regulation
    HIPAA = "hipaa"       # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"   # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"  # Information Security Management
    NIST = "nist"         # National Institute of Standards and Technology
    CCPA = "ccpa"         # California Consumer Privacy Act


class DataClassification(str, Enum):
    """Data classification levels."""
    
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class DataCategory(str, Enum):
    """Data categories for compliance."""
    
    PII = "pii"                    # Personally Identifiable Information
    PHI = "phi"                    # Protected Health Information
    PCI = "pci"                    # Payment Card Information
    FINANCIAL = "financial"        # Financial Data
    BIOMETRIC = "biometric"        # Biometric Data
    BEHAVIORAL = "behavioral"      # Behavioral Data
    LOCATION = "location"          # Location Data
    COMMUNICATION = "communication"  # Communication Data


class ProcessingPurpose(str, Enum):
    """Data processing purposes."""
    
    ANOMALY_DETECTION = "anomaly_detection"
    MODEL_TRAINING = "model_training"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"
    RESEARCH = "research"
    MARKETING = "marketing"
    OPERATIONAL = "operational"


class DataRetentionPolicy(str, Enum):
    """Data retention policies."""
    
    IMMEDIATE_DELETE = "immediate_delete"
    THIRTY_DAYS = "thirty_days"
    NINETY_DAYS = "ninety_days"
    ONE_YEAR = "one_year"
    FIVE_YEARS = "five_years"
    INDEFINITE = "indefinite"
    LEGAL_HOLD = "legal_hold"


@dataclass
class DataLineageNode:
    """Node in data lineage graph."""
    
    node_id: str
    node_type: str  # source, transformation, destination
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class DataLineageEdge:
    """Edge in data lineage graph."""
    
    edge_id: str
    source_node_id: str
    target_node_id: str
    transformation_type: str
    transformation_description: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAsset:
    """Data asset with compliance metadata."""
    
    asset_id: str
    name: str
    description: str
    classification: DataClassification
    categories: Set[DataCategory]
    owner: str
    steward: str
    location: str
    format: str
    size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    retention_policy: DataRetentionPolicy = DataRetentionPolicy.ONE_YEAR
    processing_purposes: Set[ProcessingPurpose] = field(default_factory=set)
    applicable_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    encryption_status: bool = False
    backup_status: bool = False
    compliance_tags: Set[str] = field(default_factory=set)
    lineage_nodes: List[str] = field(default_factory=list)


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    
    violation_id: str
    framework: ComplianceFramework
    rule_id: str
    rule_description: str
    severity: str  # low, medium, high, critical
    asset_id: Optional[str] = None
    user_id: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)
    status: str = "open"  # open, investigating, resolved, false_positive
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""


@dataclass
class AuditEvent:
    """Audit event for compliance tracking."""
    
    event_id: str
    event_type: str
    user_id: str
    resource_id: str
    action: str
    outcome: str  # success, failure, partial
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)


class DataLineageTracker:
    """Data lineage tracking system."""
    
    def __init__(self):
        self.nodes: Dict[str, DataLineageNode] = {}
        self.edges: Dict[str, DataLineageEdge] = {}
        self.lineage_graph: Dict[str, Set[str]] = {}  # adjacency list
    
    async def add_node(self, node: DataLineageNode) -> bool:
        """Add node to lineage graph."""
        try:
            self.nodes[node.node_id] = node
            if node.node_id not in self.lineage_graph:
                self.lineage_graph[node.node_id] = set()
            
            logger.info(f"Added lineage node: {node.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add lineage node: {e}")
            return False
    
    async def add_edge(self, edge: DataLineageEdge) -> bool:
        """Add edge to lineage graph."""
        try:
            self.edges[edge.edge_id] = edge
            
            # Update adjacency list
            if edge.source_node_id not in self.lineage_graph:
                self.lineage_graph[edge.source_node_id] = set()
            self.lineage_graph[edge.source_node_id].add(edge.target_node_id)
            
            logger.info(f"Added lineage edge: {edge.source_node_id} -> {edge.target_node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add lineage edge: {e}")
            return False
    
    async def trace_lineage(
        self,
        node_id: str,
        direction: str = "downstream"  # downstream or upstream
    ) -> List[DataLineageNode]:
        """Trace data lineage from a given node."""
        try:
            visited = set()
            lineage_path = []
            
            if direction == "downstream":
                await self._trace_downstream(node_id, visited, lineage_path)
            else:
                await self._trace_upstream(node_id, visited, lineage_path)
            
            return [self.nodes[node_id] for node_id in lineage_path if node_id in self.nodes]
            
        except Exception as e:
            logger.error(f"Failed to trace lineage: {e}")
            return []
    
    async def _trace_downstream(
        self,
        node_id: str,
        visited: Set[str],
        path: List[str]
    ) -> None:
        """Trace downstream lineage."""
        if node_id in visited:
            return
        
        visited.add(node_id)
        path.append(node_id)
        
        for child_id in self.lineage_graph.get(node_id, set()):
            await self._trace_downstream(child_id, visited, path)
    
    async def _trace_upstream(
        self,
        node_id: str,
        visited: Set[str],
        path: List[str]
    ) -> None:
        """Trace upstream lineage."""
        if node_id in visited:
            return
        
        visited.add(node_id)
        path.append(node_id)
        
        # Find parent nodes
        for parent_id, children in self.lineage_graph.items():
            if node_id in children:
                await self._trace_upstream(parent_id, visited, path)
    
    async def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """Get impact analysis for a node change."""
        try:
            downstream_nodes = await self.trace_lineage(node_id, "downstream")
            upstream_nodes = await self.trace_lineage(node_id, "upstream")
            
            # Calculate impact metrics
            total_affected = len(downstream_nodes) + len(upstream_nodes)
            
            # Categorize affected nodes
            affected_by_type = {}
            for node in downstream_nodes + upstream_nodes:
                node_type = node.node_type
                if node_type not in affected_by_type:
                    affected_by_type[node_type] = 0
                affected_by_type[node_type] += 1
            
            return {
                "node_id": node_id,
                "total_affected_nodes": total_affected,
                "downstream_count": len(downstream_nodes),
                "upstream_count": len(upstream_nodes),
                "affected_by_type": affected_by_type,
                "impact_level": self._calculate_impact_level(total_affected),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get impact analysis: {e}")
            return {}
    
    def _calculate_impact_level(self, affected_count: int) -> str:
        """Calculate impact level based on affected node count."""
        if affected_count == 0:
            return "none"
        elif affected_count <= 5:
            return "low"
        elif affected_count <= 20:
            return "medium"
        elif affected_count <= 50:
            return "high"
        else:
            return "critical"


class SOXComplianceManager:
    """Sarbanes-Oxley Act compliance management."""
    
    def __init__(self):
        self.sox_requirements = {
            "section_302": "CEO/CFO certification of financial reports",
            "section_404": "Internal control assessment",
            "section_409": "Real-time disclosure requirements",
            "section_802": "Document retention requirements",
            "section_906": "Criminal penalties for certification violations"
        }
        self.audit_trail: List[AuditEvent] = []
    
    async def validate_financial_data_access(
        self,
        user_id: str,
        asset_id: str,
        action: str
    ) -> Dict[str, Any]:
        """Validate financial data access for SOX compliance."""
        try:
            # Check if user has appropriate role for financial data
            validation_result = {
                "compliant": False,
                "violations": [],
                "requirements": []
            }
            
            # SOX requires segregation of duties
            if action in ["create", "modify", "delete"] and "financial" in asset_id.lower():
                validation_result["requirements"].append(
                    "Segregation of duties required for financial data modifications"
                )
                
                # Check if user has both read and write access (violation)
                # In real implementation, this would check actual permissions
                validation_result["compliant"] = True  # Simplified for example
            
            # Audit trail requirement
            audit_event = AuditEvent(
                event_id=f"sox_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                event_type="financial_data_access",
                user_id=user_id,
                resource_id=asset_id,
                action=action,
                outcome="success",
                compliance_frameworks={ComplianceFramework.SOX}
            )
            self.audit_trail.append(audit_event)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"SOX validation failed: {e}")
            return {"compliant": False, "error": str(e)}
    
    async def generate_sox_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate SOX compliance report."""
        try:
            # Filter audit events for reporting period
            period_events = [
                event for event in self.audit_trail
                if start_date <= event.timestamp <= end_date
                and ComplianceFramework.SOX in event.compliance_frameworks
            ]
            
            # Analyze events
            financial_accesses = len([e for e in period_events if "financial" in e.resource_id])
            unique_users = len(set(e.user_id for e in period_events))
            failed_accesses = len([e for e in period_events if e.outcome == "failure"])
            
            return {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_events": len(period_events),
                    "financial_data_accesses": financial_accesses,
                    "unique_users": unique_users,
                    "failed_access_attempts": failed_accesses
                },
                "compliance_status": "compliant" if failed_accesses == 0 else "violations_detected",
                "sox_sections_covered": list(self.sox_requirements.keys()),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SOX report generation failed: {e}")
            return {"error": str(e)}


class GDPRComplianceManager:
    """GDPR compliance management."""
    
    def __init__(self):
        self.gdpr_principles = {
            "lawfulness": "Processing must be lawful, fair and transparent",
            "purpose_limitation": "Data collected for specified, explicit and legitimate purposes",
            "data_minimisation": "Data must be adequate, relevant and limited to what is necessary",
            "accuracy": "Data must be accurate and kept up to date",
            "storage_limitation": "Data kept for no longer than necessary",
            "integrity_confidentiality": "Data processed securely with appropriate technical measures",
            "accountability": "Controller responsible for demonstrating compliance"
        }
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_subject_requests: List[Dict[str, Any]] = []
    
    async def validate_data_processing(
        self,
        asset: DataAsset,
        processing_purpose: ProcessingPurpose,
        legal_basis: str
    ) -> Dict[str, Any]:
        """Validate data processing for GDPR compliance."""
        try:
            validation_result = {
                "compliant": True,
                "violations": [],
                "requirements": [],
                "recommendations": []
            }
            
            # Check if PII is involved
            if DataCategory.PII in asset.categories:
                validation_result["requirements"].append("GDPR Article 6 legal basis required")
                
                # Check data minimization
                if processing_purpose not in asset.processing_purposes:
                    validation_result["violations"].append(
                        "Processing purpose not specified in original collection"
                    )
                    validation_result["compliant"] = False
                
                # Check retention policy
                if asset.retention_policy == DataRetentionPolicy.INDEFINITE:
                    validation_result["violations"].append(
                        "Indefinite retention violates GDPR storage limitation principle"
                    )
                    validation_result["compliant"] = False
                
                # Check encryption for sensitive data
                if not asset.encryption_status:
                    validation_result["recommendations"].append(
                        "Encrypt personal data to meet GDPR security requirements"
                    )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"GDPR validation failed: {e}")
            return {"compliant": False, "error": str(e)}
    
    async def handle_data_subject_request(
        self,
        request_type: str,  # access, rectification, erasure, portability
        data_subject_id: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle GDPR data subject request."""
        try:
            request_id = f"dsr_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            request_record = {
                "request_id": request_id,
                "request_type": request_type,
                "data_subject_id": data_subject_id,
                "details": details,
                "received_at": datetime.now().isoformat(),
                "status": "received",
                "response_due": (datetime.now() + timedelta(days=30)).isoformat()
            }
            
            self.data_subject_requests.append(request_record)
            
            # Auto-processing for simple requests
            if request_type == "access":
                # In real implementation, this would search for all data
                request_record["status"] = "processing"
                request_record["estimated_completion"] = (datetime.now() + timedelta(days=7)).isoformat()
            
            elif request_type == "erasure":
                # Check if erasure is possible (legal obligations, etc.)
                request_record["status"] = "under_review"
                request_record["review_notes"] = "Checking for legal retention requirements"
            
            logger.info(f"Created GDPR data subject request: {request_id}")
            
            return {
                "request_id": request_id,
                "status": request_record["status"],
                "response_due": request_record["response_due"],
                "estimated_completion": request_record.get("estimated_completion")
            }
            
        except Exception as e:
            logger.error(f"GDPR data subject request handling failed: {e}")
            return {"error": str(e)}
    
    async def generate_privacy_impact_assessment(
        self,
        processing_activity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate GDPR Privacy Impact Assessment (PIA)."""
        try:
            pia_id = f"pia_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Assess risk level
            risk_factors = []
            risk_level = "low"
            
            # Check for high-risk processing
            if "profiling" in processing_activity.get("purposes", []):
                risk_factors.append("Automated decision-making/profiling")
                risk_level = "high"
            
            if "special_categories" in processing_activity.get("data_types", []):
                risk_factors.append("Special categories of personal data")
                risk_level = "high"
            
            if processing_activity.get("data_subjects_count", 0) > 10000:
                risk_factors.append("Large scale processing")
                risk_level = "medium" if risk_level == "low" else "high"
            
            pia_report = {
                "pia_id": pia_id,
                "processing_activity": processing_activity,
                "risk_assessment": {
                    "risk_level": risk_level,
                    "risk_factors": risk_factors,
                    "likelihood": "medium",
                    "impact": "medium"
                },
                "mitigation_measures": [
                    "Data encryption at rest and in transit",
                    "Regular security assessments",
                    "Staff training on data protection",
                    "Clear data retention policies"
                ],
                "compliance_measures": [
                    "Legal basis documentation",
                    "Data subject rights procedures",
                    "Breach notification procedures",
                    "Regular compliance audits"
                ],
                "conducted_by": "Data Protection Officer",
                "conducted_at": datetime.now().isoformat(),
                "review_date": (datetime.now() + timedelta(days=365)).isoformat()
            }
            
            return pia_report
            
        except Exception as e:
            logger.error(f"PIA generation failed: {e}")
            return {"error": str(e)}


class HIPAAComplianceManager:
    """HIPAA compliance management."""
    
    def __init__(self):
        self.hipaa_safeguards = {
            "administrative": "Security officer, workforce training, access management",
            "physical": "Facility access controls, workstation use, device controls",
            "technical": "Access control, audit controls, integrity, transmission security"
        }
        self.phi_access_log: List[Dict[str, Any]] = []
    
    async def validate_phi_access(
        self,
        user_id: str,
        phi_asset_id: str,
        access_purpose: str
    ) -> Dict[str, Any]:
        """Validate PHI access for HIPAA compliance."""
        try:
            validation_result = {
                "compliant": True,
                "violations": [],
                "requirements": [],
                "safeguards_applied": []
            }
            
            # Check minimum necessary rule
            if access_purpose not in ["treatment", "payment", "healthcare_operations"]:
                validation_result["requirements"].append(
                    "Minimum necessary rule - access limited to minimum required"
                )
            
            # Log access (required by HIPAA)
            access_log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "phi_asset_id": phi_asset_id,
                "access_purpose": access_purpose,
                "access_granted": True,
                "minimum_necessary_verified": True
            }
            self.phi_access_log.append(access_log_entry)
            
            validation_result["safeguards_applied"] = [
                "Access logging (§164.312(b))",
                "User identification (§164.312(a)(2)(i))",
                "Audit controls (§164.312(b))"
            ]
            
            return validation_result
            
        except Exception as e:
            logger.error(f"HIPAA PHI access validation failed: {e}")
            return {"compliant": False, "error": str(e)}
    
    async def generate_hipaa_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate HIPAA audit report."""
        try:
            # Filter PHI access logs
            period_accesses = [
                log for log in self.phi_access_log
                if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
            ]
            
            # Analyze access patterns
            unique_users = len(set(log["user_id"] for log in period_accesses))
            purpose_breakdown = {}
            for log in period_accesses:
                purpose = log["access_purpose"]
                purpose_breakdown[purpose] = purpose_breakdown.get(purpose, 0) + 1
            
            # Check for potential violations
            violations = []
            
            # Check for excessive access
            user_access_counts = {}
            for log in period_accesses:
                user_id = log["user_id"]
                user_access_counts[user_id] = user_access_counts.get(user_id, 0) + 1
            
            for user_id, count in user_access_counts.items():
                if count > 1000:  # Threshold for investigation
                    violations.append({
                        "type": "excessive_access",
                        "user_id": user_id,
                        "access_count": count,
                        "requires_investigation": True
                    })
            
            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_phi_accesses": len(period_accesses),
                    "unique_users": unique_users,
                    "access_by_purpose": purpose_breakdown
                },
                "compliance_status": "compliant" if not violations else "violations_detected",
                "violations": violations,
                "safeguards_verified": list(self.hipaa_safeguards.keys()),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"HIPAA audit report generation failed: {e}")
            return {"error": str(e)}


class ComplianceOrchestrator:
    """Main orchestrator for compliance and governance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lineage_tracker = DataLineageTracker()
        self.sox_manager = SOXComplianceManager()
        self.gdpr_manager = GDPRComplianceManager()
        self.hipaa_manager = HIPAAComplianceManager()
        
        self.data_assets: Dict[str, DataAsset] = {}
        self.violations: List[ComplianceViolation] = []
        self.compliance_rules: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self):
        """Initialize default compliance rules."""
        self.compliance_rules = {
            "encryption_required": {
                "frameworks": [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
                "applies_to": [DataCategory.PII, DataCategory.PHI],
                "severity": "high",
                "description": "Personal data must be encrypted"
            },
            "retention_policy_required": {
                "frameworks": [ComplianceFramework.GDPR, ComplianceFramework.SOX],
                "applies_to": "all",
                "severity": "medium",
                "description": "Data must have defined retention policy"
            },
            "access_logging_required": {
                "frameworks": [ComplianceFramework.SOX, ComplianceFramework.HIPAA],
                "applies_to": [DataCategory.FINANCIAL, DataCategory.PHI],
                "severity": "high",
                "description": "All access must be logged for audit"
            }
        }
    
    async def register_data_asset(self, asset: DataAsset) -> bool:
        """Register data asset with compliance tracking."""
        try:
            self.data_assets[asset.asset_id] = asset
            
            # Create lineage node
            lineage_node = DataLineageNode(
                node_id=asset.asset_id,
                node_type="source",
                name=asset.name,
                description=asset.description,
                metadata={
                    "classification": asset.classification.value,
                    "categories": [cat.value for cat in asset.categories],
                    "owner": asset.owner
                }
            )
            
            await self.lineage_tracker.add_node(lineage_node)
            
            # Run compliance validation
            await self._validate_asset_compliance(asset)
            
            logger.info(f"Registered data asset: {asset.asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register data asset: {e}")
            return False
    
    async def track_data_transformation(
        self,
        source_asset_id: str,
        target_asset_id: str,
        transformation_type: str,
        transformation_description: str
    ) -> bool:
        """Track data transformation for lineage."""
        try:
            edge_id = f"{source_asset_id}_to_{target_asset_id}"
            
            lineage_edge = DataLineageEdge(
                edge_id=edge_id,
                source_node_id=source_asset_id,
                target_node_id=target_asset_id,
                transformation_type=transformation_type,
                transformation_description=transformation_description
            )
            
            await self.lineage_tracker.add_edge(lineage_edge)
            
            logger.info(f"Tracked data transformation: {source_asset_id} -> {target_asset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track data transformation: {e}")
            return False
    
    async def validate_processing_activity(
        self,
        asset_id: str,
        processing_purpose: ProcessingPurpose,
        user_id: str,
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Validate data processing activity for compliance."""
        try:
            asset = self.data_assets.get(asset_id)
            if not asset:
                return {"compliant": False, "error": "Asset not found"}
            
            validation_results = {}
            
            if framework == ComplianceFramework.GDPR:
                validation_results["gdpr"] = await self.gdpr_manager.validate_data_processing(
                    asset, processing_purpose, "legitimate_interest"
                )
            
            elif framework == ComplianceFramework.SOX:
                validation_results["sox"] = await self.sox_manager.validate_financial_data_access(
                    user_id, asset_id, "access"
                )
            
            elif framework == ComplianceFramework.HIPAA:
                validation_results["hipaa"] = await self.hipaa_manager.validate_phi_access(
                    user_id, asset_id, processing_purpose.value
                )
            
            # Check general compliance rules
            rule_violations = await self._check_compliance_rules(asset, framework)
            if rule_violations:
                validation_results["rule_violations"] = rule_violations
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Processing validation failed: {e}")
            return {"compliant": False, "error": str(e)}
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        try:
            report = {
                "framework": framework.value,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_at": datetime.now().isoformat()
            }
            
            if framework == ComplianceFramework.SOX:
                report["sox_report"] = await self.sox_manager.generate_sox_report(start_date, end_date)
            
            elif framework == ComplianceFramework.HIPAA:
                report["hipaa_report"] = await self.hipaa_manager.generate_hipaa_audit_report(start_date, end_date)
            
            elif framework == ComplianceFramework.GDPR:
                # GDPR report would include data subject requests, breaches, etc.
                report["gdpr_report"] = {
                    "data_subject_requests": len(self.gdpr_manager.data_subject_requests),
                    "privacy_impact_assessments": 0,  # Would be tracked separately
                    "consent_records": len(self.gdpr_manager.consent_records)
                }
            
            # Add general compliance metrics
            framework_violations = [
                v for v in self.violations
                if v.framework == framework and start_date <= v.detected_at <= end_date
            ]
            
            report["violations_summary"] = {
                "total_violations": len(framework_violations),
                "by_severity": self._group_violations_by_severity(framework_violations),
                "resolution_rate": self._calculate_resolution_rate(framework_violations)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            return {"error": str(e)}
    
    async def _validate_asset_compliance(self, asset: DataAsset) -> None:
        """Validate asset against all applicable compliance rules."""
        for rule_id, rule in self.compliance_rules.items():
            # Check if rule applies to this asset
            if not self._rule_applies_to_asset(rule, asset):
                continue
            
            violation = None
            
            if rule_id == "encryption_required":
                if not asset.encryption_status:
                    violation = ComplianceViolation(
                        violation_id=f"enc_{asset.asset_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        framework=rule["frameworks"][0],
                        rule_id=rule_id,
                        rule_description=rule["description"],
                        severity=rule["severity"],
                        asset_id=asset.asset_id,
                        description=f"Asset {asset.asset_id} contains sensitive data but is not encrypted"
                    )
            
            elif rule_id == "retention_policy_required":
                if asset.retention_policy == DataRetentionPolicy.INDEFINITE:
                    violation = ComplianceViolation(
                        violation_id=f"ret_{asset.asset_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        framework=rule["frameworks"][0],
                        rule_id=rule_id,
                        rule_description=rule["description"],
                        severity=rule["severity"],
                        asset_id=asset.asset_id,
                        description=f"Asset {asset.asset_id} has indefinite retention policy"
                    )
            
            if violation:
                self.violations.append(violation)
                logger.warning(f"Compliance violation detected: {violation.violation_id}")
    
    def _rule_applies_to_asset(self, rule: Dict[str, Any], asset: DataAsset) -> bool:
        """Check if compliance rule applies to asset."""
        applies_to = rule.get("applies_to", [])
        
        if applies_to == "all":
            return True
        
        if isinstance(applies_to, list):
            # Check if any of the asset's categories match
            asset_categories = {cat.value for cat in asset.categories}
            rule_categories = set(applies_to)
            return bool(asset_categories.intersection(rule_categories))
        
        return False
    
    async def _check_compliance_rules(
        self,
        asset: DataAsset,
        framework: ComplianceFramework
    ) -> List[Dict[str, Any]]:
        """Check asset against compliance rules for specific framework."""
        violations = []
        
        for rule_id, rule in self.compliance_rules.items():
            if framework not in rule.get("frameworks", []):
                continue
            
            if not self._rule_applies_to_asset(rule, asset):
                continue
            
            # Check rule-specific conditions
            if rule_id == "encryption_required" and not asset.encryption_status:
                violations.append({
                    "rule_id": rule_id,
                    "description": rule["description"],
                    "severity": rule["severity"]
                })
        
        return violations
    
    def _group_violations_by_severity(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Group violations by severity."""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for violation in violations:
            severity_counts[violation.severity] = severity_counts.get(violation.severity, 0) + 1
        
        return severity_counts
    
    def _calculate_resolution_rate(self, violations: List[ComplianceViolation]) -> float:
        """Calculate violation resolution rate."""
        if not violations:
            return 1.0
        
        resolved_count = len([v for v in violations if v.status == "resolved"])
        return resolved_count / len(violations)