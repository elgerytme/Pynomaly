"""Audit event domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass
class AuditEvent:
    """Domain entity for audit events."""
    
    event_type: str
    resource_type: str
    resource_id: str
    action: str
    user_id: str | None = None
    session_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: UUID = field(default_factory=uuid4)
    
    # Event details
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Request context
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    
    # Result information
    success: bool = True
    error_message: str | None = None
    
    def __post_init__(self) -> None:
        """Validate audit event after initialization."""
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        if not self.resource_type:
            raise ValueError("Resource type cannot be empty")
        if not self.action:
            raise ValueError("Action cannot be empty")