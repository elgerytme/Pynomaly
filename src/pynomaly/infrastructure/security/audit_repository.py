"""Abstract audit repository interface and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from pynomaly.domain.entities.audit import AuditEvent
from pynomaly.domain.models.security import AuditEventType, SecurityFramework


class QueryCriteria(BaseModel):
    """Criteria for querying audit events."""
    
    user_id: Optional[UUID] = None
    event_type: Optional[AuditEventType] = None
    severity_min: Optional[int] = None
    risk_score_min: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tenant_id: Optional[UUID] = None
    compliance_framework: Optional[SecurityFramework] = None
    limit: int = 100
    offset: int = 0


class AuditRepository(ABC):
    """Abstract repository interface for audit data persistence."""
    
    @abstractmethod
    async def store_event(self, event: AuditEvent) -> UUID:
        """Store a single audit event."""
        pass
    
    @abstractmethod
    async def store_events_batch(self, events: List[AuditEvent]) -> List[UUID]:
        """Store multiple audit events in a batch."""
        pass
    
    @abstractmethod
    async def query_events(self, criteria: QueryCriteria) -> List[AuditEvent]:
        """Query audit events based on criteria."""
        pass
    
    @abstractmethod
    async def get_event_by_id(self, event_id: UUID) -> Optional[AuditEvent]:
        """Get a specific audit event by ID."""
        pass
    
    @abstractmethod
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[AuditEvent]:
        """Get all events with the same correlation ID."""
        pass
    
    @abstractmethod
    async def delete_expired_events(self, before_date: datetime) -> int:
        """Delete expired events and return count deleted."""
        pass
    
    @abstractmethod
    async def get_event_statistics(self, timeframe: timedelta) -> dict:
        """Get audit event statistics for the given timeframe."""
        pass
    
    @abstractmethod
    async def verify_event_integrity(self, event_id: UUID) -> bool:
        """Verify the integrity of a specific audit event."""
        pass


class DatabaseAuditRepository(AuditRepository):
    """Database-backed audit repository implementation."""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
    
    async def store_event(self, event: AuditEvent) -> UUID:
        """Store a single audit event in database."""
        query = """
        INSERT INTO audit_events (
            id, event_type, user_id, tenant_id, timestamp, severity, 
            risk_score, details, correlation_id, compliance_frameworks,
            ip_address, user_agent, session_id, integrity_hash
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
        )
        """
        
        async with self.connection_manager.get_connection() as conn:
            await conn.execute(
                query,
                event.id,
                event.event_type.value,
                event.user_id,
                event.tenant_id,
                event.timestamp,
                event.severity.value,
                event.risk_score,
                event.details,
                event.correlation_id,
                [f.value for f in event.compliance_frameworks],
                event.context.ip_address if event.context else None,
                event.context.user_agent if event.context else None,
                event.context.session_id if event.context else None,
                event.integrity_hash
            )
        
        return event.id
    
    async def store_events_batch(self, events: List[AuditEvent]) -> List[UUID]:
        """Store multiple audit events in a batch."""
        if not events:
            return []
        
        query = """
        INSERT INTO audit_events (
            id, event_type, user_id, tenant_id, timestamp, severity, 
            risk_score, details, correlation_id, compliance_frameworks,
            ip_address, user_agent, session_id, integrity_hash
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """
        
        async with self.connection_manager.get_connection() as conn:
            async with conn.transaction():
                for event in events:
                    await conn.execute(
                        query,
                        event.id,
                        event.event_type.value,
                        event.user_id,
                        event.tenant_id,
                        event.timestamp,
                        event.severity.value,
                        event.risk_score,
                        event.details,
                        event.correlation_id,
                        [f.value for f in event.compliance_frameworks],
                        event.context.ip_address if event.context else None,
                        event.context.user_agent if event.context else None,
                        event.context.session_id if event.context else None,
                        event.integrity_hash
                    )
        
        return [event.id for event in events]
    
    async def query_events(self, criteria: QueryCriteria) -> List[AuditEvent]:
        """Query audit events based on criteria."""
        query_parts = ["SELECT * FROM audit_events WHERE 1=1"]
        params = []
        param_counter = 1
        
        if criteria.user_id:
            query_parts.append(f"AND user_id = ${param_counter}")
            params.append(criteria.user_id)
            param_counter += 1
        
        if criteria.event_type:
            query_parts.append(f"AND event_type = ${param_counter}")
            params.append(criteria.event_type.value)
            param_counter += 1
        
        if criteria.severity_min:
            query_parts.append(f"AND severity >= ${param_counter}")
            params.append(criteria.severity_min)
            param_counter += 1
        
        if criteria.risk_score_min:
            query_parts.append(f"AND risk_score >= ${param_counter}")
            params.append(criteria.risk_score_min)
            param_counter += 1
        
        if criteria.start_time:
            query_parts.append(f"AND timestamp >= ${param_counter}")
            params.append(criteria.start_time)
            param_counter += 1
        
        if criteria.end_time:
            query_parts.append(f"AND timestamp <= ${param_counter}")
            params.append(criteria.end_time)
            param_counter += 1
        
        if criteria.tenant_id:
            query_parts.append(f"AND tenant_id = ${param_counter}")
            params.append(criteria.tenant_id)
            param_counter += 1
        
        if criteria.compliance_framework:
            query_parts.append(f"AND ${param_counter} = ANY(compliance_frameworks)")
            params.append(criteria.compliance_framework.value)
            param_counter += 1
        
        query_parts.append("ORDER BY timestamp DESC")
        query_parts.append(f"LIMIT ${param_counter} OFFSET ${param_counter + 1}")
        params.extend([criteria.limit, criteria.offset])
        
        query = " ".join(query_parts)
        
        async with self.connection_manager.get_connection() as conn:
            rows = await conn.fetch(query, *params)
        
        return [self._row_to_audit_event(row) for row in rows]
    
    async def get_event_by_id(self, event_id: UUID) -> Optional[AuditEvent]:
        """Get a specific audit event by ID."""
        query = "SELECT * FROM audit_events WHERE id = $1"
        
        async with self.connection_manager.get_connection() as conn:
            row = await conn.fetchrow(query, event_id)
        
        return self._row_to_audit_event(row) if row else None
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[AuditEvent]:
        """Get all events with the same correlation ID."""
        query = "SELECT * FROM audit_events WHERE correlation_id = $1 ORDER BY timestamp"
        
        async with self.connection_manager.get_connection() as conn:
            rows = await conn.fetch(query, correlation_id)
        
        return [self._row_to_audit_event(row) for row in rows]
    
    async def delete_expired_events(self, before_date: datetime) -> int:
        """Delete expired events and return count deleted."""
        query = "DELETE FROM audit_events WHERE timestamp < $1"
        
        async with self.connection_manager.get_connection() as conn:
            result = await conn.execute(query, before_date)
        
        # Extract count from result string like "DELETE 42"
        return int(result.split()[-1]) if result.split()[-1].isdigit() else 0
    
    async def get_event_statistics(self, timeframe: timedelta) -> dict:
        """Get audit event statistics for the given timeframe."""
        start_time = datetime.utcnow() - timeframe
        
        query = """
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT tenant_id) as unique_tenants,
            AVG(risk_score) as avg_risk_score,
            MAX(risk_score) as max_risk_score,
            COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical_events,
            COUNT(CASE WHEN severity = 'HIGH' THEN 1 END) as high_events,
            COUNT(CASE WHEN severity = 'MEDIUM' THEN 1 END) as medium_events,
            COUNT(CASE WHEN severity = 'LOW' THEN 1 END) as low_events
        FROM audit_events 
        WHERE timestamp >= $1
        """
        
        async with self.connection_manager.get_connection() as conn:
            row = await conn.fetchrow(query, start_time)
        
        return dict(row) if row else {}
    
    async def verify_event_integrity(self, event_id: UUID) -> bool:
        """Verify the integrity of a specific audit event."""
        event = await self.get_event_by_id(event_id)
        if not event:
            return False
        
        # Recalculate integrity hash and compare
        expected_hash = event.calculate_integrity_hash()
        return event.integrity_hash == expected_hash
    
    def _row_to_audit_event(self, row) -> AuditEvent:
        """Convert database row to AuditEvent object."""
        # This would need to be implemented based on your specific AuditEvent model
        # and database schema
        pass


class InMemoryAuditRepository(AuditRepository):
    """In-memory audit repository implementation for testing."""
    
    def __init__(self):
        self._events = {}
        self._events_by_correlation = {}
    
    async def store_event(self, event: AuditEvent) -> UUID:
        """Store a single audit event in memory."""
        self._events[event.id] = event
        
        # Index by correlation ID
        if event.correlation_id:
            if event.correlation_id not in self._events_by_correlation:
                self._events_by_correlation[event.correlation_id] = []
            self._events_by_correlation[event.correlation_id].append(event)
        
        return event.id
    
    async def store_events_batch(self, events: List[AuditEvent]) -> List[UUID]:
        """Store multiple audit events in memory."""
        event_ids = []
        for event in events:
            event_ids.append(await self.store_event(event))
        return event_ids
    
    async def query_events(self, criteria: QueryCriteria) -> List[AuditEvent]:
        """Query audit events based on criteria."""
        filtered_events = []
        
        for event in self._events.values():
            if criteria.user_id and event.user_id != criteria.user_id:
                continue
            if criteria.event_type and event.event_type != criteria.event_type:
                continue
            if criteria.severity_min and event.severity.value < criteria.severity_min:
                continue
            if criteria.risk_score_min and event.risk_score < criteria.risk_score_min:
                continue
            if criteria.start_time and event.timestamp < criteria.start_time:
                continue
            if criteria.end_time and event.timestamp > criteria.end_time:
                continue
            if criteria.tenant_id and event.tenant_id != criteria.tenant_id:
                continue
            if criteria.compliance_framework and criteria.compliance_framework not in event.compliance_frameworks:
                continue
            
            filtered_events.append(event)
        
        # Sort by timestamp descending
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply pagination
        start_idx = criteria.offset
        end_idx = start_idx + criteria.limit
        return filtered_events[start_idx:end_idx]
    
    async def get_event_by_id(self, event_id: UUID) -> Optional[AuditEvent]:
        """Get a specific audit event by ID."""
        return self._events.get(event_id)
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[AuditEvent]:
        """Get all events with the same correlation ID."""
        events = self._events_by_correlation.get(correlation_id, [])
        return sorted(events, key=lambda x: x.timestamp)
    
    async def delete_expired_events(self, before_date: datetime) -> int:
        """Delete expired events and return count deleted."""
        expired_ids = [
            event_id for event_id, event in self._events.items()
            if event.timestamp < before_date
        ]
        
        for event_id in expired_ids:
            event = self._events.pop(event_id)
            # Clean up correlation index
            if event.correlation_id in self._events_by_correlation:
                self._events_by_correlation[event.correlation_id].remove(event)
                if not self._events_by_correlation[event.correlation_id]:
                    del self._events_by_correlation[event.correlation_id]
        
        return len(expired_ids)
    
    async def get_event_statistics(self, timeframe: timedelta) -> dict:
        """Get audit event statistics for the given timeframe."""
        start_time = datetime.utcnow() - timeframe
        recent_events = [
            event for event in self._events.values()
            if event.timestamp >= start_time
        ]
        
        if not recent_events:
            return {
                "total_events": 0,
                "unique_users": 0,
                "unique_tenants": 0,
                "avg_risk_score": 0,
                "max_risk_score": 0,
                "critical_events": 0,
                "high_events": 0,
                "medium_events": 0,
                "low_events": 0
            }
        
        unique_users = len(set(event.user_id for event in recent_events if event.user_id))
        unique_tenants = len(set(event.tenant_id for event in recent_events if event.tenant_id))
        
        severity_counts = {}
        for event in recent_events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_events": len(recent_events),
            "unique_users": unique_users,
            "unique_tenants": unique_tenants,
            "avg_risk_score": sum(event.risk_score for event in recent_events) / len(recent_events),
            "max_risk_score": max(event.risk_score for event in recent_events),
            "critical_events": severity_counts.get("CRITICAL", 0),
            "high_events": severity_counts.get("HIGH", 0),
            "medium_events": severity_counts.get("MEDIUM", 0),
            "low_events": severity_counts.get("LOW", 0)
        }
    
    async def verify_event_integrity(self, event_id: UUID) -> bool:
        """Verify the integrity of a specific audit event."""
        event = await self.get_event_by_id(event_id)
        if not event:
            return False
        
        # Recalculate integrity hash and compare
        expected_hash = event.calculate_integrity_hash()
        return event.integrity_hash == expected_hash