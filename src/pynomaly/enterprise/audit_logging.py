#!/usr/bin/env python3
"""
Enterprise Audit Logging System for Pynomaly.
This module provides comprehensive audit logging capabilities for compliance and security.
"""

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

import elasticsearch
import redis
import structlog
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure structured logging
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
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Database setup
Base = declarative_base()


class AuditAction(Enum):
    """Audit action types."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    IMPORT = "import"
    DEPLOY = "deploy"
    UNDEPLOY = "undeploy"
    EXECUTE = "execute"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    APPROVE = "approve"
    REJECT = "reject"
    CONFIGURE = "configure"
    BACKUP = "backup"
    RESTORE = "restore"


class AuditStatus(Enum):
    """Audit status types."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"
    ERROR = "error"


class ComplianceLevel(Enum):
    """Compliance level types."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CUSTOM = "custom"


class SensitivityLevel(Enum):
    """Data sensitivity levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


# Enhanced Database Models
class AuditEvent(Base):
    """Audit event database model."""

    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False)
    user_id = Column(String(36), nullable=True)
    session_id = Column(String(255), nullable=True)
    request_id = Column(String(255), nullable=True)

    # Event details
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    resource_name = Column(String(255), nullable=True)

    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    duration_ms = Column(Integer, nullable=True)

    # Context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    geo_location = Column(JSON, nullable=True)

    # Status and results
    status = Column(String(50), nullable=False)
    status_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Data
    before_data = Column(JSON, nullable=True)
    after_data = Column(JSON, nullable=True)
    event_metadata = Column(JSON, nullable=True)

    # Compliance
    compliance_level = Column(String(50), nullable=True)
    sensitivity_level = Column(String(50), nullable=True)
    retention_until = Column(DateTime, nullable=True)

    # Integrity
    checksum = Column(String(64), nullable=True)
    signature = Column(Text, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_audit_tenant_timestamp", "tenant_id", "timestamp"),
        Index("idx_audit_user_timestamp", "user_id", "timestamp"),
        Index("idx_audit_action_timestamp", "action", "timestamp"),
        Index("idx_audit_resource_type_timestamp", "resource_type", "timestamp"),
        Index("idx_audit_status_timestamp", "status", "timestamp"),
    )


class AuditConfiguration(Base):
    """Audit configuration database model."""

    __tablename__ = "audit_configurations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False)

    # Configuration
    enabled = Column(Boolean, default=True)
    compliance_level = Column(String(50), nullable=False)
    retention_days = Column(Integer, default=365)

    # Rules
    audit_rules = Column(JSON, nullable=False)
    exclusion_rules = Column(JSON, nullable=False)

    # Alerting
    alert_rules = Column(JSON, nullable=False)
    notification_settings = Column(JSON, nullable=False)

    # Created/Updated
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditAlert(Base):
    """Audit alert database model."""

    __tablename__ = "audit_alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(36), nullable=False)

    # Alert details
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    # Trigger
    trigger_event_id = Column(String(36), nullable=True)
    trigger_conditions = Column(JSON, nullable=False)

    # Status
    status = Column(String(50), default="open")
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    alert_metadata = Column(JSON, nullable=True)


# Pydantic Models
@dataclass
class AuditEventInfo:
    """Audit event information structure."""

    id: str
    tenant_id: str
    user_id: str | None
    session_id: str | None
    request_id: str | None
    action: AuditAction
    resource_type: str
    resource_id: str | None
    resource_name: str | None
    timestamp: datetime
    duration_ms: int | None
    ip_address: str | None
    user_agent: str | None
    geo_location: dict[str, Any] | None
    status: AuditStatus
    status_code: int | None
    error_message: str | None
    before_data: dict[str, Any] | None
    after_data: dict[str, Any] | None
    event_metadata: dict[str, Any] | None
    compliance_level: ComplianceLevel | None
    sensitivity_level: SensitivityLevel | None
    retention_until: datetime | None
    checksum: str | None
    signature: str | None


class AuditEventCreate(BaseModel):
    """Audit event creation request."""

    action: AuditAction
    resource_type: str
    resource_id: str | None = None
    resource_name: str | None = None
    status: AuditStatus
    status_code: int | None = None
    error_message: str | None = None
    before_data: dict[str, Any] | None = None
    after_data: dict[str, Any] | None = None
    event_metadata: dict[str, Any] | None = None
    compliance_level: ComplianceLevel | None = None
    sensitivity_level: SensitivityLevel | None = None
    duration_ms: int | None = None


class AuditQuery(BaseModel):
    """Audit query parameters."""

    tenant_id: str | None = None
    user_id: str | None = None
    action: AuditAction | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    status: AuditStatus | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    ip_address: str | None = None
    compliance_level: ComplianceLevel | None = None
    sensitivity_level: SensitivityLevel | None = None
    skip: int = 0
    limit: int = 100


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(
        self,
        database_url: str,
        elasticsearch_url: str | None = None,
        redis_url: str | None = None,
        enable_encryption: bool = True,
        enable_signing: bool = True,
    ):
        """Initialize audit logger."""
        self.database_url = database_url
        self.elasticsearch_url = elasticsearch_url
        self.redis_url = redis_url
        self.enable_encryption = enable_encryption
        self.enable_signing = enable_signing

        # Database setup
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        Base.metadata.create_all(bind=self.engine)

        # Elasticsearch setup
        self.es_client = None
        if elasticsearch_url:
            try:
                self.es_client = elasticsearch.Elasticsearch([elasticsearch_url])
                self._setup_elasticsearch_index()
            except Exception as e:
                logger.warning(f"Failed to connect to Elasticsearch: {e}")

        # Redis setup
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

        # Default configurations
        self.default_audit_config = {
            "enabled": True,
            "compliance_level": ComplianceLevel.GDPR.value,
            "retention_days": 365,
            "audit_rules": {
                "log_all_actions": True,
                "log_data_changes": True,
                "log_failed_attempts": True,
                "log_privileged_actions": True,
                "exclude_read_operations": False,
            },
            "exclusion_rules": {
                "health_checks": True,
                "system_monitoring": True,
                "automated_backups": False,
            },
            "alert_rules": {
                "failed_login_threshold": 5,
                "privilege_escalation": True,
                "data_export_large": True,
                "unusual_activity": True,
            },
            "notification_settings": {
                "email_enabled": True,
                "slack_enabled": False,
                "webhook_enabled": False,
            },
        }

        logger.info("Audit logger initialized")

    def get_db(self):
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def _setup_elasticsearch_index(self):
        """Set up Elasticsearch index for audit logs."""
        try:
            index_name = "audit_logs"
            mapping = {
                "mappings": {
                    "properties": {
                        "tenant_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "action": {"type": "keyword"},
                        "resource_type": {"type": "keyword"},
                        "resource_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "status": {"type": "keyword"},
                        "ip_address": {"type": "ip"},
                        "user_agent": {"type": "text", "analyzer": "keyword"},
                        "geo_location": {"type": "geo_point"},
                        "compliance_level": {"type": "keyword"},
                        "sensitivity_level": {"type": "keyword"},
                        "event_metadata": {"type": "object"},
                        "before_data": {"type": "object"},
                        "after_data": {"type": "object"},
                    }
                }
            }

            if not self.es_client.indices.exists(index=index_name):
                self.es_client.indices.create(index=index_name, body=mapping)
                logger.info(f"✅ Elasticsearch index created: {index_name}")

        except Exception as e:
            logger.error(f"Failed to set up Elasticsearch index: {e}")

    def _calculate_checksum(self, data: dict[str, Any]) -> str:
        """Calculate checksum for audit event."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _sign_event(self, event_data: dict[str, Any]) -> str:
        """Sign audit event for integrity."""
        # In production, use proper cryptographic signing
        # This is a simplified example
        checksum = self._calculate_checksum(event_data)
        signature = hashlib.sha256(
            f"{checksum}:{os.getenv('AUDIT_SECRET', 'default_secret')}".encode()
        ).hexdigest()
        return signature

    async def log_event(
        self,
        tenant_id: str,
        user_id: str | None,
        request: AuditEventCreate,
        session_id: str | None = None,
        request_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        geo_location: dict[str, Any] | None = None,
    ) -> str:
        """Log audit event."""
        db = next(self.get_db())

        try:
            # Calculate retention
            retention_days = 365  # Default, should be configurable per tenant
            retention_until = datetime.utcnow() + timedelta(days=retention_days)

            # Prepare event data
            event_data = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "action": request.action.value,
                "resource_type": request.resource_type,
                "resource_id": request.resource_id,
                "resource_name": request.resource_name,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_ms": request.duration_ms,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "geo_location": geo_location,
                "status": request.status.value,
                "status_code": request.status_code,
                "error_message": request.error_message,
                "before_data": request.before_data,
                "after_data": request.after_data,
                "event_metadata": request.event_metadata,
                "compliance_level": request.compliance_level.value
                if request.compliance_level
                else None,
                "sensitivity_level": request.sensitivity_level.value
                if request.sensitivity_level
                else None,
                "retention_until": retention_until.isoformat(),
            }

            # Calculate checksum and signature
            checksum = self._calculate_checksum(event_data)
            signature = self._sign_event(event_data) if self.enable_signing else None

            # Create audit event
            audit_event = AuditEvent(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                action=request.action.value,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                resource_name=request.resource_name,
                timestamp=datetime.utcnow(),
                duration_ms=request.duration_ms,
                ip_address=ip_address,
                user_agent=user_agent,
                geo_location=geo_location,
                status=request.status.value,
                status_code=request.status_code,
                error_message=request.error_message,
                before_data=request.before_data,
                after_data=request.after_data,
                event_metadata=request.event_metadata,
                compliance_level=request.compliance_level.value
                if request.compliance_level
                else None,
                sensitivity_level=request.sensitivity_level.value
                if request.sensitivity_level
                else None,
                retention_until=retention_until,
                checksum=checksum,
                signature=signature,
            )

            db.add(audit_event)
            db.commit()
            db.refresh(audit_event)

            event_id = str(audit_event.id)

            # Index in Elasticsearch
            await self._index_in_elasticsearch(event_id, event_data)

            # Cache in Redis
            await self._cache_event(event_id, event_data)

            # Check for alerts
            await self._check_alert_conditions(tenant_id, audit_event)

            logger.info(f"✅ Audit event logged: {event_id}")
            return event_id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to log audit event: {e}")
            raise
        finally:
            db.close()

    async def _index_in_elasticsearch(self, event_id: str, event_data: dict[str, Any]):
        """Index audit event in Elasticsearch."""
        if not self.es_client:
            return

        try:
            doc = {"event_id": event_id, **event_data}

            self.es_client.index(index="audit_logs", id=event_id, body=doc)

        except Exception as e:
            logger.error(f"Failed to index in Elasticsearch: {e}")

    async def _cache_event(self, event_id: str, event_data: dict[str, Any]):
        """Cache audit event in Redis."""
        if not self.redis_client:
            return

        try:
            # Cache recent events for fast access
            key = f"audit:recent:{event_id}"
            self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(event_data, default=str),
            )

            # Add to tenant's recent events list
            tenant_key = f"audit:tenant:{event_data['tenant_id']}:recent"
            self.redis_client.lpush(tenant_key, event_id)
            self.redis_client.ltrim(tenant_key, 0, 1000)  # Keep last 1000 events

        except Exception as e:
            logger.error(f"Failed to cache audit event: {e}")

    async def _check_alert_conditions(self, tenant_id: str, audit_event: AuditEvent):
        """Check if audit event triggers any alerts."""
        # This is a simplified example - implement based on your alert rules
        try:
            # Check for failed login attempts
            if (
                audit_event.action == AuditAction.LOGIN.value
                and audit_event.status == AuditStatus.FAILURE.value
            ):
                await self._check_failed_login_threshold(tenant_id, audit_event)

            # Check for privilege escalation
            if audit_event.action in [
                AuditAction.UPDATE.value,
                AuditAction.CONFIGURE.value,
            ]:
                await self._check_privilege_escalation(tenant_id, audit_event)

            # Check for data export
            if audit_event.action == AuditAction.EXPORT.value:
                await self._check_data_export_alert(tenant_id, audit_event)

        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")

    async def _check_failed_login_threshold(
        self, tenant_id: str, audit_event: AuditEvent
    ):
        """Check failed login threshold."""
        db = next(self.get_db())

        try:
            # Count failed logins in last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)

            failed_count = (
                db.query(AuditEvent)
                .filter(
                    AuditEvent.tenant_id == tenant_id,
                    AuditEvent.action == AuditAction.LOGIN.value,
                    AuditEvent.status == AuditStatus.FAILURE.value,
                    AuditEvent.timestamp >= one_hour_ago,
                    AuditEvent.ip_address == audit_event.ip_address,
                )
                .count()
            )

            if failed_count >= 5:  # Threshold
                await self._create_alert(
                    tenant_id=tenant_id,
                    alert_type="failed_login_threshold",
                    severity="high",
                    title="Multiple Failed Login Attempts",
                    description=f"IP {audit_event.ip_address} has {failed_count} failed login attempts in the last hour",
                    trigger_event_id=str(audit_event.id),
                    trigger_conditions={"threshold": 5, "window": "1h"},
                )

        finally:
            db.close()

    async def _check_privilege_escalation(
        self, tenant_id: str, audit_event: AuditEvent
    ):
        """Check for privilege escalation."""
        # Implementation depends on your specific privilege model
        pass

    async def _check_data_export_alert(self, tenant_id: str, audit_event: AuditEvent):
        """Check for large data export."""
        # Check if export is larger than threshold
        event_metadata = audit_event.event_metadata or {}
        export_size = event_metadata.get("export_size", 0)

        if export_size > 1000000:  # 1MB threshold
            await self._create_alert(
                tenant_id=tenant_id,
                alert_type="large_data_export",
                severity="medium",
                title="Large Data Export",
                description=f"User exported {export_size} bytes of data",
                trigger_event_id=str(audit_event.id),
                trigger_conditions={"size_threshold": 1000000},
            )

    async def _create_alert(
        self,
        tenant_id: str,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        trigger_event_id: str,
        trigger_conditions: dict[str, Any],
    ):
        """Create audit alert."""
        db = next(self.get_db())

        try:
            alert = AuditAlert(
                tenant_id=tenant_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                description=description,
                trigger_event_id=trigger_event_id,
                trigger_conditions=trigger_conditions,
            )

            db.add(alert)
            db.commit()

            logger.warning(
                f"🚨 Audit alert created: {alert_type} for tenant {tenant_id}"
            )

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create audit alert: {e}")
        finally:
            db.close()

    async def query_events(self, query: AuditQuery) -> list[AuditEventInfo]:
        """Query audit events."""
        db = next(self.get_db())

        try:
            # Build query
            db_query = db.query(AuditEvent)

            if query.tenant_id:
                db_query = db_query.filter(AuditEvent.tenant_id == query.tenant_id)
            if query.user_id:
                db_query = db_query.filter(AuditEvent.user_id == query.user_id)
            if query.action:
                db_query = db_query.filter(AuditEvent.action == query.action.value)
            if query.resource_type:
                db_query = db_query.filter(
                    AuditEvent.resource_type == query.resource_type
                )
            if query.resource_id:
                db_query = db_query.filter(AuditEvent.resource_id == query.resource_id)
            if query.status:
                db_query = db_query.filter(AuditEvent.status == query.status.value)
            if query.start_time:
                db_query = db_query.filter(AuditEvent.timestamp >= query.start_time)
            if query.end_time:
                db_query = db_query.filter(AuditEvent.timestamp <= query.end_time)
            if query.ip_address:
                db_query = db_query.filter(AuditEvent.ip_address == query.ip_address)
            if query.compliance_level:
                db_query = db_query.filter(
                    AuditEvent.compliance_level == query.compliance_level.value
                )
            if query.sensitivity_level:
                db_query = db_query.filter(
                    AuditEvent.sensitivity_level == query.sensitivity_level.value
                )

            # Order by timestamp descending
            db_query = db_query.order_by(AuditEvent.timestamp.desc())

            # Apply pagination
            events = db_query.offset(query.skip).limit(query.limit).all()

            # Convert to AuditEventInfo
            result = []
            for event in events:
                result.append(
                    AuditEventInfo(
                        id=str(event.id),
                        tenant_id=str(event.tenant_id),
                        user_id=str(event.user_id) if event.user_id else None,
                        session_id=event.session_id,
                        request_id=event.request_id,
                        action=AuditAction(event.action),
                        resource_type=event.resource_type,
                        resource_id=event.resource_id,
                        resource_name=event.resource_name,
                        timestamp=event.timestamp,
                        duration_ms=event.duration_ms,
                        ip_address=event.ip_address,
                        user_agent=event.user_agent,
                        geo_location=event.geo_location,
                        status=AuditStatus(event.status),
                        status_code=event.status_code,
                        error_message=event.error_message,
                        before_data=event.before_data,
                        after_data=event.after_data,
                        event_metadata=event.event_metadata,
                        compliance_level=ComplianceLevel(event.compliance_level)
                        if event.compliance_level
                        else None,
                        sensitivity_level=SensitivityLevel(event.sensitivity_level)
                        if event.sensitivity_level
                        else None,
                        retention_until=event.retention_until,
                        checksum=event.checksum,
                        signature=event.signature,
                    )
                )

            return result

        finally:
            db.close()

    async def get_event(self, event_id: str) -> AuditEventInfo | None:
        """Get audit event by ID."""
        db = next(self.get_db())

        try:
            event = db.query(AuditEvent).filter(AuditEvent.id == event_id).first()
            if not event:
                return None

            return AuditEventInfo(
                id=str(event.id),
                tenant_id=str(event.tenant_id),
                user_id=str(event.user_id) if event.user_id else None,
                session_id=event.session_id,
                request_id=event.request_id,
                action=AuditAction(event.action),
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                resource_name=event.resource_name,
                timestamp=event.timestamp,
                duration_ms=event.duration_ms,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                geo_location=event.geo_location,
                status=AuditStatus(event.status),
                status_code=event.status_code,
                error_message=event.error_message,
                before_data=event.before_data,
                after_data=event.after_data,
                event_metadata=event.event_metadata,
                compliance_level=ComplianceLevel(event.compliance_level)
                if event.compliance_level
                else None,
                sensitivity_level=SensitivityLevel(event.sensitivity_level)
                if event.sensitivity_level
                else None,
                retention_until=event.retention_until,
                checksum=event.checksum,
                signature=event.signature,
            )

        finally:
            db.close()

    async def cleanup_expired_events(self):
        """Clean up expired audit events."""
        db = next(self.get_db())

        try:
            # Delete events past retention period
            expired_events = (
                db.query(AuditEvent)
                .filter(AuditEvent.retention_until < datetime.utcnow())
                .all()
            )

            for event in expired_events:
                db.delete(event)

            db.commit()

            logger.info(f"✅ Cleaned up {len(expired_events)} expired audit events")

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cleanup expired events: {e}")
        finally:
            db.close()


# Decorators for automatic audit logging
def audit_log(
    action: AuditAction,
    resource_type: str,
    compliance_level: ComplianceLevel = None,
    sensitivity_level: SensitivityLevel = None,
):
    """Decorator for automatic audit logging."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()

            # Extract context (this would be more sophisticated in practice)
            tenant_id = kwargs.get("tenant_id") or getattr(
                kwargs.get("current_tenant"), "id", None
            )
            user_id = kwargs.get("user_id") or getattr(
                kwargs.get("current_user"), "id", None
            )

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Log success
                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                audit_request = AuditEventCreate(
                    action=action,
                    resource_type=resource_type,
                    status=AuditStatus.SUCCESS,
                    duration_ms=duration_ms,
                    compliance_level=compliance_level,
                    sensitivity_level=sensitivity_level,
                    event_metadata={
                        "function": func.__name__,
                        "result_type": type(result).__name__,
                    },
                )

                if tenant_id:
                    audit_logger = get_audit_logger()
                    await audit_logger.log_event(
                        tenant_id=tenant_id, user_id=user_id, request=audit_request
                    )

                return result

            except Exception as e:
                # Log failure
                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                audit_request = AuditEventCreate(
                    action=action,
                    resource_type=resource_type,
                    status=AuditStatus.FAILURE,
                    duration_ms=duration_ms,
                    error_message=str(e),
                    compliance_level=compliance_level,
                    sensitivity_level=sensitivity_level,
                    event_metadata={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                )

                if tenant_id:
                    audit_logger = get_audit_logger()
                    await audit_logger.log_event(
                        tenant_id=tenant_id, user_id=user_id, request=audit_request
                    )

                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio to run the audit logging
            return asyncio.run(async_wrapper(*args, **kwargs))

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Global audit logger instance
audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    global audit_logger
    if audit_logger is None:
        database_url = os.getenv(
            "DATABASE_URL", "sqlite:///test_audit.db"
        )
        elasticsearch_url = os.getenv("ELASTICSEARCH_URL")
        redis_url = os.getenv("REDIS_URL")

        audit_logger = AuditLogger(
            database_url=database_url,
            elasticsearch_url=elasticsearch_url,
            redis_url=redis_url,
        )
    return audit_logger


# Make components available for import
__all__ = [
    "AuditLogger",
    "AuditEventInfo",
    "AuditEventCreate",
    "AuditQuery",
    "AuditAction",
    "AuditStatus",
    "ComplianceLevel",
    "SensitivityLevel",
    "audit_log",
    "get_audit_logger",
]
