"""Database-backed audit logging storage implementation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from pynomaly.domain.entities.audit import AuditEvent, AuditEventType, EventSeverity
from pynomaly.infrastructure.security.audit_logging import AuditStorage

Base = declarative_base()


class AuditEventModel(Base):
    """Database model for audit events."""
    
    __tablename__ = "audit_events"
    
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(PostgreSQLUUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    tenant_id = Column(PostgreSQLUUID(as_uuid=True), nullable=True, index=True)
    resource_type = Column(String(100), nullable=True, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    outcome = Column(String(20), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    risk_score = Column(Integer, nullable=False, default=0, index=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    correlation_id = Column(String(255), nullable=True, index=True)
    details = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class DatabaseAuditStorage(AuditStorage):
    """Database-backed audit event storage with compliance features."""
    
    def __init__(self, database_url: str, **kwargs):
        """Initialize database audit storage.
        
        Args:
            database_url: Database connection URL
            **kwargs: Additional database configuration
        """
        self.engine = create_engine(database_url, **kwargs)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
    async def store_event(self, event: AuditEvent) -> bool:
        """Store audit event in database.
        
        Args:
            event: Audit event to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Convert AuditEvent to database model
            db_event = AuditEventModel(
                id=event.id,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                user_id=event.user_id,
                session_id=event.session_id,
                tenant_id=event.tenant_id,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                action=event.action,
                outcome=event.outcome,
                severity=event.severity.value,
                risk_score=event.risk_score,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                correlation_id=event.correlation_id,
                details=event.details,
                metadata=event.metadata
            )
            
            self.session.add(db_event)
            self.session.commit()
            return True
            
        except Exception as e:
            self.session.rollback()
            # Log error but don't fail - audit logging should be resilient
            print(f"Failed to store audit event: {e}")
            return False
    
    async def query_events(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[AuditEvent]:
        """Query audit events from database.
        
        Args:
            filters: Filter criteria
            start_time: Start time for query
            end_time: End time for query
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of audit events
        """
        try:
            query = self.session.query(AuditEventModel)
            
            # Apply time range filters
            if start_time:
                query = query.filter(AuditEventModel.timestamp >= start_time)
            if end_time:
                query = query.filter(AuditEventModel.timestamp <= end_time)
            
            # Apply additional filters
            if filters:
                for field, value in filters.items():
                    if hasattr(AuditEventModel, field):
                        if isinstance(value, list):
                            query = query.filter(getattr(AuditEventModel, field).in_(value))
                        else:
                            query = query.filter(getattr(AuditEventModel, field) == value)
            
            # Apply pagination and ordering
            query = query.order_by(AuditEventModel.timestamp.desc())
            query = query.offset(offset).limit(limit)
            
            # Execute query and convert to domain objects
            db_events = query.all()
            events = []
            
            for db_event in db_events:
                event = AuditEvent(
                    id=db_event.id,
                    timestamp=db_event.timestamp,
                    event_type=AuditEventType(db_event.event_type),
                    user_id=db_event.user_id,
                    session_id=db_event.session_id,
                    tenant_id=db_event.tenant_id,
                    resource_type=db_event.resource_type,
                    resource_id=db_event.resource_id,
                    action=db_event.action,
                    outcome=db_event.outcome,
                    severity=EventSeverity(db_event.severity),
                    risk_score=db_event.risk_score,
                    ip_address=db_event.ip_address,
                    user_agent=db_event.user_agent,
                    correlation_id=db_event.correlation_id,
                    details=db_event.details or {},
                    metadata=db_event.metadata or {}
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            print(f"Failed to query audit events: {e}")
            return []
    
    async def get_event_count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """Get count of audit events matching criteria.
        
        Args:
            filters: Filter criteria
            start_time: Start time for query
            end_time: End time for query
            
        Returns:
            Count of matching events
        """
        try:
            query = self.session.query(AuditEventModel)
            
            # Apply time range filters
            if start_time:
                query = query.filter(AuditEventModel.timestamp >= start_time)
            if end_time:
                query = query.filter(AuditEventModel.timestamp <= end_time)
            
            # Apply additional filters
            if filters:
                for field, value in filters.items():
                    if hasattr(AuditEventModel, field):
                        if isinstance(value, list):
                            query = query.filter(getattr(AuditEventModel, field).in_(value))
                        else:
                            query = query.filter(getattr(AuditEventModel, field) == value)
            
            return query.count()
            
        except Exception as e:
            print(f"Failed to count audit events: {e}")
            return 0
    
    async def cleanup_old_events(self, retention_days: int = 365) -> int:
        """Clean up old audit events based on retention policy.
        
        Args:
            retention_days: Number of days to retain events
            
        Returns:
            Number of events deleted
        """
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=retention_days)
            
            deleted = self.session.query(AuditEventModel).filter(
                AuditEventModel.timestamp < cutoff_date
            ).delete()
            
            self.session.commit()
            return deleted
            
        except Exception as e:
            self.session.rollback()
            print(f"Failed to cleanup old audit events: {e}")
            return 0
    
    async def export_events(
        self,
        start_time: datetime,
        end_time: datetime,
        format_type: str = "json",
        filters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Export audit events for compliance reporting.
        
        Args:
            start_time: Start time for export
            end_time: End time for export
            format_type: Export format (json, csv)
            filters: Additional filters
            
        Returns:
            Exported data as bytes
        """
        try:
            events = await self.query_events(
                filters=filters,
                start_time=start_time,
                end_time=end_time,
                limit=10000  # Adjust for large exports
            )
            
            if format_type.lower() == "json":
                export_data = []
                for event in events:
                    export_data.append({
                        "id": str(event.id),
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "user_id": str(event.user_id) if event.user_id else None,
                        "action": event.action,
                        "outcome": event.outcome,
                        "severity": event.severity.value,
                        "risk_score": event.risk_score,
                        "details": event.details,
                        "metadata": event.metadata
                    })
                
                return json.dumps(export_data, indent=2).encode('utf-8')
            
            elif format_type.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "ID", "Timestamp", "Event Type", "User ID", "Action", 
                    "Outcome", "Severity", "Risk Score", "IP Address"
                ])
                
                # Write data
                for event in events:
                    writer.writerow([
                        str(event.id),
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        str(event.user_id) if event.user_id else "",
                        event.action,
                        event.outcome,
                        event.severity.value,
                        event.risk_score,
                        event.ip_address or ""
                    ])
                
                return output.getvalue().encode('utf-8')
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            print(f"Failed to export audit events: {e}")
            return b""
    
    async def get_compliance_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        compliance_framework: str = "GDPR"
    ) -> Dict[str, Any]:
        """Generate compliance summary report.
        
        Args:
            start_time: Start time for report
            end_time: End time for report
            compliance_framework: Compliance framework to report on
            
        Returns:
            Compliance summary data
        """
        try:
            # Get relevant events for compliance framework
            compliance_events = await self.query_events(
                start_time=start_time,
                end_time=end_time,
                limit=100000
            )
            
            # Calculate compliance metrics
            total_events = len(compliance_events)
            security_events = len([e for e in compliance_events if e.event_type == AuditEventType.SECURITY])
            failed_events = len([e for e in compliance_events if e.outcome == "FAILURE"])
            high_risk_events = len([e for e in compliance_events if e.risk_score >= 80])
            
            # Framework-specific metrics
            framework_metrics = {}
            if compliance_framework == "GDPR":
                data_access_events = len([e for e in compliance_events if "data_access" in e.action])
                data_export_events = len([e for e in compliance_events if "export" in e.action])
                framework_metrics = {
                    "data_access_events": data_access_events,
                    "data_export_events": data_export_events,
                    "consent_events": 0  # Would need to implement consent tracking
                }
            
            return {
                "compliance_framework": compliance_framework,
                "report_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "summary": {
                    "total_events": total_events,
                    "security_events": security_events,
                    "failed_events": failed_events,
                    "high_risk_events": high_risk_events,
                    "compliance_score": max(0, 100 - (failed_events / max(total_events, 1)) * 100)
                },
                "framework_metrics": framework_metrics,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            print(f"Failed to generate compliance summary: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.session:
            self.session.close()