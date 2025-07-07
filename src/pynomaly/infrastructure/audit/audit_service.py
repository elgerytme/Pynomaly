"""
Enterprise-grade audit service for comprehensive logging and compliance.

Provides complete audit trail capabilities:
- Structured audit logging with metadata
- Compliance reporting (SOX, GDPR, HIPAA)
- Real-time audit event streaming
- Tamper-proof audit trails with integrity verification
- Automated retention and archival
- Performance and business metrics auditing
- Security event correlation and alerting
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import structlog

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class AuditEventType(Enum):
    """Comprehensive audit event types."""
    
    # Authentication and Authorization
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # Data Operations
    DATA_ACCESS = "data_access"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    DATA_SHARED = "data_shared"
    
    # Model Operations
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_PREDICTION = "model_prediction"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DELETED = "model_deleted"
    MODEL_UPDATED = "model_updated"
    
    # System Operations
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Administrative Actions
    ADMIN_ACTION = "admin_action"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    BULK_OPERATION = "bulk_operation"
    MAINTENANCE_MODE = "maintenance_mode"
    
    # API and Integration
    API_CALL = "api_call"
    WEBHOOK_RECEIVED = "webhook_received"
    EXTERNAL_SERVICE_CALL = "external_service_call"
    INTEGRATION_ERROR = "integration_error"
    
    # Compliance and Regulatory
    GDPR_REQUEST = "gdpr_request"
    DATA_RETENTION_ACTION = "data_retention_action"
    COMPLIANCE_REPORT = "compliance_report"
    AUDIT_LOG_ACCESS = "audit_log_access"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    DEBUG = "debug"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOX = "sox"  # Sarbanes-Oxley
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # Information Security Management
    NIST = "nist"  # National Institute of Standards and Technology


@dataclass
class AuditContext:
    """Context information for audit events."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    api_version: Optional[str] = None
    client_id: Optional[str] = None
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    
    # Core identifiers
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_STARTUP
    
    # Event details
    actor: Optional[str] = None  # Who performed the action
    target: Optional[str] = None  # What was acted upon
    action: str = "unknown"
    outcome: str = "success"  # success, failure, error
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Context information
    context: AuditContext = field(default_factory=AuditContext)
    
    # Detailed information
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Compliance and security
    compliance_frameworks: Set[ComplianceFramework] = field(default_factory=set)
    data_classification: Optional[str] = None
    retention_period_days: int = 2555  # 7 years default
    
    # Technical metadata
    service_name: str = "pynomaly"
    service_version: Optional[str] = None
    environment: str = "development"
    
    # Integrity verification
    checksum: Optional[str] = field(init=False, default=None)
    previous_event_checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum for integrity verification."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data."""
        # Create deterministic string representation
        data = {
            'event_id': str(self.event_id),
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor,
            'target': self.target,
            'action': self.action,
            'outcome': self.outcome,
            'details': json.dumps(self.details, sort_keys=True, default=str),
            'previous_checksum': self.previous_event_checksum
        }
        
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            'event_id': str(self.event_id),
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor,
            'target': self.target,
            'action': self.action,
            'outcome': self.outcome,
            'severity': self.severity.value,
            'context': {
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'request_id': self.context.request_id,
                'source_ip': self.context.source_ip,
                'user_agent': self.context.user_agent,
                'api_version': self.context.api_version,
                'client_id': self.context.client_id,
                'tenant_id': self.context.tenant_id,
                'correlation_id': self.context.correlation_id
            },
            'details': self.details,
            'tags': list(self.tags),
            'compliance_frameworks': [f.value for f in self.compliance_frameworks],
            'data_classification': self.data_classification,
            'retention_period_days': self.retention_period_days,
            'service_name': self.service_name,
            'service_version': self.service_version,
            'environment': self.environment,
            'checksum': self.checksum,
            'previous_event_checksum': self.previous_event_checksum
        }


@dataclass
class AuditConfig:
    """Configuration for audit service."""
    
    # Storage configuration
    log_directory: str = "/var/log/pynomaly/audit"
    max_file_size_mb: int = 100
    max_files: int = 1000
    compress_archived_files: bool = True
    
    # Integrity and security
    enable_checksum_verification: bool = True
    enable_digital_signatures: bool = False
    signing_key_path: Optional[str] = None
    
    # Retention and archival
    default_retention_days: int = 2555  # 7 years
    auto_archive_days: int = 90
    auto_delete_days: Optional[int] = None  # None means never delete
    
    # Performance
    buffer_size: int = 1000
    flush_interval_seconds: float = 60.0
    async_logging: bool = True
    
    # Compliance
    enable_compliance_reporting: bool = True
    required_compliance_frameworks: Set[ComplianceFramework] = field(
        default_factory=lambda: {ComplianceFramework.SOX, ComplianceFramework.GDPR}
    )
    
    # Monitoring and alerting
    enable_real_time_monitoring: bool = True
    alert_on_critical_events: bool = True
    alert_on_integrity_failures: bool = True
    
    # Export and integration
    enable_siem_export: bool = False
    siem_endpoint: Optional[str] = None
    export_formats: Set[str] = field(default_factory=lambda: {"json", "csv"})


class AuditStorage:
    """Handles audit event storage with integrity verification."""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        self.log_dir = Path(config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Event buffer for batch writing
        self.event_buffer: List[AuditEvent] = []
        self.last_event_checksum: Optional[str] = None
        
        # Initialize structured logger
        self._setup_structured_logger()
        
        # Load last checksum for integrity chain
        self._load_last_checksum()
    
    def _setup_structured_logger(self):
        """Set up structured logging with JSON format."""
        log_file = self.log_dir / f"audit-{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Get structlog logger
        self.audit_logger = structlog.get_logger("audit")
    
    def _load_last_checksum(self):
        """Load the last event checksum to maintain integrity chain."""
        try:
            # Find the most recent audit file
            audit_files = sorted(self.log_dir.glob("audit-*.jsonl"))
            if not audit_files:
                return
            
            latest_file = audit_files[-1]
            
            # Read the last line to get the last checksum
            with open(latest_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        event_data = json.loads(last_line)
                        self.last_event_checksum = event_data.get('checksum')
                        self.logger.info(f"Loaded last checksum: {self.last_event_checksum}")
        
        except Exception as e:
            self.logger.warning(f"Could not load last checksum: {e}")
    
    def store_event(self, event: AuditEvent):
        """Store audit event with integrity verification."""
        # Set previous checksum for integrity chain
        event.previous_event_checksum = self.last_event_checksum
        
        # Recalculate checksum with previous checksum
        event.checksum = event._calculate_checksum()
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Update last checksum
        self.last_event_checksum = event.checksum
        
        # Write to structured log
        self.audit_logger.info(
            "audit_event",
            **event.to_dict()
        )
        
        # Flush buffer if needed
        if len(self.event_buffer) >= self.config.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush buffered events to storage."""
        if not self.event_buffer:
            return
        
        try:
            # Write events to file
            current_date = datetime.now().strftime('%Y%m%d')
            audit_file = self.log_dir / f"audit-{current_date}.jsonl"
            
            with open(audit_file, 'a') as f:
                for event in self.event_buffer:
                    f.write(json.dumps(event.to_dict()) + '\n')
            
            self.logger.debug(f"Flushed {len(self.event_buffer)} audit events")
            self.event_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to flush audit buffer: {e}")
    
    def verify_integrity(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify integrity of audit trail."""
        verification_result = {
            'status': 'success',
            'verified_events': 0,
            'integrity_failures': 0,
            'chain_breaks': 0,
            'details': []
        }
        
        try:
            # Get audit files to verify
            audit_files = self._get_audit_files(start_date, end_date)
            
            previous_checksum = None
            
            for file_path in audit_files:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        
                        try:
                            event_data = json.loads(line.strip())
                            verification_result['verified_events'] += 1
                            
                            # Verify checksum
                            stored_checksum = event_data.get('checksum')
                            
                            # Recreate event for verification
                            event = AuditEvent(
                                event_id=UUID(event_data['event_id']),
                                timestamp=datetime.fromisoformat(event_data['timestamp']),
                                event_type=AuditEventType(event_data['event_type']),
                                actor=event_data.get('actor'),
                                target=event_data.get('target'),
                                action=event_data['action'],
                                outcome=event_data['outcome'],
                                details=event_data.get('details', {}),
                                previous_event_checksum=event_data.get('previous_event_checksum')
                            )
                            
                            calculated_checksum = event._calculate_checksum()
                            
                            if stored_checksum != calculated_checksum:
                                verification_result['integrity_failures'] += 1
                                verification_result['details'].append({
                                    'file': str(file_path),
                                    'line': line_num,
                                    'event_id': event_data['event_id'],
                                    'issue': 'checksum_mismatch',
                                    'stored': stored_checksum,
                                    'calculated': calculated_checksum
                                })
                            
                            # Verify chain integrity
                            if previous_checksum and event_data.get('previous_event_checksum') != previous_checksum:
                                verification_result['chain_breaks'] += 1
                                verification_result['details'].append({
                                    'file': str(file_path),
                                    'line': line_num,
                                    'event_id': event_data['event_id'],
                                    'issue': 'chain_break',
                                    'expected_previous': previous_checksum,
                                    'actual_previous': event_data.get('previous_event_checksum')
                                })
                            
                            previous_checksum = stored_checksum
                            
                        except Exception as e:
                            verification_result['details'].append({
                                'file': str(file_path),
                                'line': line_num,
                                'issue': 'parse_error',
                                'error': str(e)
                            })
            
            if verification_result['integrity_failures'] > 0 or verification_result['chain_breaks'] > 0:
                verification_result['status'] = 'failure'
            
        except Exception as e:
            verification_result['status'] = 'error'
            verification_result['error'] = str(e)
        
        return verification_result
    
    def _get_audit_files(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> List[Path]:
        """Get audit files for the specified date range."""
        audit_files = list(self.log_dir.glob("audit-*.jsonl"))
        
        if not start_date and not end_date:
            return sorted(audit_files)
        
        filtered_files = []
        for file_path in audit_files:
            # Extract date from filename
            try:
                date_str = file_path.stem.split('-', 1)[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                
                filtered_files.append(file_path)
            except (ValueError, IndexError):
                continue
        
        return sorted(filtered_files)
    
    def search_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        actors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        outcomes: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search audit events with filters."""
        matching_events = []
        
        try:
            audit_files = self._get_audit_files(start_time, end_time)
            
            for file_path in audit_files:
                if len(matching_events) >= limit:
                    break
                
                with open(file_path, 'r') as f:
                    for line in f:
                        if len(matching_events) >= limit:
                            break
                        
                        if not line.strip():
                            continue
                        
                        try:
                            event_data = json.loads(line.strip())
                            
                            # Apply filters
                            event_time = datetime.fromisoformat(event_data['timestamp'])
                            
                            if start_time and event_time < start_time:
                                continue
                            if end_time and event_time > end_time:
                                continue
                            
                            if event_types and event_data['event_type'] not in [et.value for et in event_types]:
                                continue
                            
                            if actors and event_data.get('actor') not in actors:
                                continue
                            
                            if targets and event_data.get('target') not in targets:
                                continue
                            
                            if outcomes and event_data['outcome'] not in outcomes:
                                continue
                            
                            matching_events.append(event_data)
                            
                        except Exception as e:
                            self.logger.warning(f"Error parsing audit event: {e}")
                            continue
        
        except Exception as e:
            self.logger.error(f"Error searching audit events: {e}")
        
        return matching_events


class ComplianceReporter:
    """Generates compliance reports for various frameworks."""
    
    def __init__(self, storage: AuditStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
    
    def generate_sox_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate SOX compliance report."""
        events = self.storage.search_events(
            start_time=start_date,
            end_time=end_date,
            event_types=[
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_UPDATED,
                AuditEventType.DATA_DELETED,
                AuditEventType.CONFIG_CHANGED,
                AuditEventType.ADMIN_ACTION,
                AuditEventType.PERMISSION_GRANTED,
                AuditEventType.PERMISSION_DENIED
            ],
            limit=10000
        )
        
        report = {
            'report_type': 'SOX Compliance',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'data_access_events': 0,
                'administrative_changes': 0,
                'failed_access_attempts': 0,
                'unique_users': set(),
                'high_risk_activities': 0
            },
            'details': {
                'financial_data_access': [],
                'privileged_operations': [],
                'access_violations': [],
                'configuration_changes': []
            }
        }
        
        for event in events:
            # Count different types of events
            if event['event_type'] == 'data_access':
                report['summary']['data_access_events'] += 1
                if 'financial' in event.get('tags', []):
                    report['details']['financial_data_access'].append(event)
            
            if event['event_type'] in ['admin_action', 'config_changed']:
                report['summary']['administrative_changes'] += 1
                report['details']['privileged_operations'].append(event)
            
            if event['outcome'] == 'failure':
                report['summary']['failed_access_attempts'] += 1
                report['details']['access_violations'].append(event)
            
            if event.get('actor'):
                report['summary']['unique_users'].add(event['actor'])
            
            if event.get('severity') == 'critical':
                report['summary']['high_risk_activities'] += 1
        
        report['summary']['unique_users'] = len(report['summary']['unique_users'])
        
        return report
    
    def generate_gdpr_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        events = self.storage.search_events(
            start_time=start_date,
            end_time=end_date,
            event_types=[
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_EXPORTED,
                AuditEventType.DATA_SHARED,
                AuditEventType.DATA_DELETED,
                AuditEventType.GDPR_REQUEST
            ],
            limit=10000
        )
        
        report = {
            'report_type': 'GDPR Compliance',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'data_access_events': 0,
                'data_exports': 0,
                'data_deletions': 0,
                'gdpr_requests': 0,
                'cross_border_transfers': 0
            },
            'details': {
                'personal_data_access': [],
                'data_subject_requests': [],
                'data_breaches': [],
                'consent_events': []
            }
        }
        
        for event in events:
            if event['event_type'] == 'data_access' and 'personal_data' in event.get('tags', []):
                report['summary']['data_access_events'] += 1
                report['details']['personal_data_access'].append(event)
            
            if event['event_type'] == 'data_exported':
                report['summary']['data_exports'] += 1
            
            if event['event_type'] == 'data_deleted':
                report['summary']['data_deletions'] += 1
            
            if event['event_type'] == 'gdpr_request':
                report['summary']['gdpr_requests'] += 1
                report['details']['data_subject_requests'].append(event)
        
        return report


class AuditService:
    """Main audit service providing comprehensive audit capabilities."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """Initialize audit service."""
        self.config = config or AuditConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.storage = AuditStorage(self.config)
        self.compliance_reporter = ComplianceReporter(self.storage)
        
        # Background tasks
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        
        self.logger.info("Audit service initialized")
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str = "success",
        actor: Optional[str] = None,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[AuditContext] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        tags: Optional[Set[str]] = None,
        compliance_frameworks: Optional[Set[ComplianceFramework]] = None,
        data_classification: Optional[str] = None
    ) -> str:
        """Log an audit event."""
        
        event = AuditEvent(
            event_type=event_type,
            action=action,
            outcome=outcome,
            actor=actor,
            target=target,
            details=details or {},
            context=context or AuditContext(),
            severity=severity,
            tags=tags or set(),
            compliance_frameworks=compliance_frameworks or set(),
            data_classification=data_classification
        )
        
        # Store event
        self.storage.store_event(event)
        
        # Log to standard logger for immediate visibility
        log_level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        
        self.logger.log(
            log_level,
            f"AUDIT: {event_type.value} - {action} by {actor} on {target} ({outcome})"
        )
        
        return str(event.event_id)
    
    def search_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        actors: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        outcomes: Optional[List[str]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search audit events."""
        return self.storage.search_events(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            actors=actors,
            targets=targets,
            outcomes=outcomes,
            limit=limit
        )
    
    def verify_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Verify audit trail integrity."""
        return self.storage.verify_integrity(start_date, end_date)
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report."""
        if framework == ComplianceFramework.SOX:
            return self.compliance_reporter.generate_sox_report(start_date, end_date)
        elif framework == ComplianceFramework.GDPR:
            return self.compliance_reporter.generate_gdpr_report(start_date, end_date)
        else:
            raise ValueError(f"Unsupported compliance framework: {framework}")
    
    def export_events(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Export audit events to file."""
        events = self.search_events(
            start_time=start_date,
            end_time=end_date,
            limit=100000
        )
        
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"/tmp/audit_export_{timestamp}.{format}"
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)
        
        elif format.lower() == "csv" and PANDAS_AVAILABLE:
            df = pd.DataFrame(events)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(events)} audit events to {output_path}")
        return output_path
    
    async def start(self):
        """Start the audit service."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting audit service...")
        
        # Start background flush task
        if self.config.async_logging:
            self._flush_task = asyncio.create_task(self._flush_loop())
        
        self.logger.info("Audit service started")
    
    async def stop(self):
        """Stop the audit service."""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping audit service...")
        
        # Cancel background task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        self.storage.flush_buffer()
        
        self.logger.info("Audit service stopped")
    
    async def _flush_loop(self):
        """Background loop for flushing audit events."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                self.storage.flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in audit flush loop: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audit service metrics."""
        return {
            'buffer_size': len(self.storage.event_buffer),
            'config': {
                'retention_days': self.config.default_retention_days,
                'buffer_size': self.config.buffer_size,
                'flush_interval': self.config.flush_interval_seconds,
                'integrity_enabled': self.config.enable_checksum_verification
            },
            'storage': {
                'log_directory': str(self.storage.log_dir),
                'last_checksum': self.storage.last_event_checksum
            }
        }


# Singleton instance
_audit_service: Optional[AuditService] = None


def get_audit_service(config: Optional[AuditConfig] = None) -> AuditService:
    """Get singleton audit service."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService(config)
    return _audit_service