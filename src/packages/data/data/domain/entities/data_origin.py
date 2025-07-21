"""Data origin entity."""

from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pydantic import Field, validator
from packages.core.domain.abstractions.base_entity import BaseEntity


class OriginType(str, Enum):
    """Types of data origins."""
    DATABASE = "database"
    FILE_SYSTEM = "file_system" 
    API = "api"
    STREAM = "stream"
    MANUAL_ENTRY = "manual_entry"
    CALCULATED = "calculated"
    DERIVED = "derived"
    IMPORTED = "imported"
    EXTERNAL_SYSTEM = "external_system"


class DataOrigin(BaseEntity):
    """Represents the source or origin of data."""
    
    origin_id: UUID = Field(default_factory=uuid4, description="Unique origin identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Origin name")
    origin_type: OriginType = Field(..., description="Type of data origin")
    description: Optional[str] = Field(None, description="Origin description")
    system_name: Optional[str] = Field(None, description="Source system name")
    connection_string: Optional[str] = Field(None, description="Connection details (sanitized)")
    location: Optional[str] = Field(None, description="Physical or logical location")
    owner: Optional[str] = Field(None, description="Data owner or steward")
    contact_info: Optional[str] = Field(None, description="Contact information")
    is_active: bool = Field(default=True, description="Whether origin is currently active")
    is_trusted: bool = Field(default=True, description="Whether origin is trusted")
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Reliability score (0-1)")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    access_frequency: int = Field(default=0, ge=0, description="Number of times accessed")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Origin configuration")
    security_settings: Dict[str, Any] = Field(default_factory=dict, description="Security configuration")
    compliance_notes: List[str] = Field(default_factory=list, description="Compliance notes")
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate origin name format."""
        if not v.strip():
            raise ValueError("Origin name cannot be empty")
        return v.strip()
    
    @validator('reliability_score')
    def validate_reliability_score(cls, v: float) -> float:
        """Validate reliability score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Reliability score must be between 0.0 and 1.0")
        return v
    
    def record_access(self) -> None:
        """Record an access to this origin."""
        self.last_accessed_at = datetime.utcnow()
        self.access_frequency += 1
        self.updated_at = datetime.utcnow()
    
    def update_reliability_score(self, new_score: float) -> None:
        """Update the reliability score."""
        if not 0.0 <= new_score <= 1.0:
            raise ValueError("Reliability score must be between 0.0 and 1.0")
        
        self.reliability_score = new_score
        self.updated_at = datetime.utcnow()
    
    def deactivate(self, reason: Optional[str] = None) -> None:
        """Deactivate this origin."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
        
        if reason:
            if 'deactivation_reasons' not in self.metadata:
                self.metadata['deactivation_reasons'] = []
            self.metadata['deactivation_reasons'].append({
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def activate(self) -> None:
        """Activate this origin."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def is_database_origin(self) -> bool:
        """Check if this is a database origin."""
        return self.origin_type == OriginType.DATABASE
    
    def is_file_origin(self) -> bool:
        """Check if this is a file system origin."""
        return self.origin_type == OriginType.FILE_SYSTEM
    
    def is_streaming_origin(self) -> bool:
        """Check if this is a streaming data origin."""
        return self.origin_type == OriginType.STREAM
    
    def is_external_origin(self) -> bool:
        """Check if this is an external system origin."""
        return self.origin_type in [OriginType.API, OriginType.EXTERNAL_SYSTEM]
    
    def get_access_frequency_per_day(self, days: int = 30) -> float:
        """Calculate average access frequency per day over specified period."""
        if days <= 0:
            return 0.0
        return self.access_frequency / days
    
    def is_highly_reliable(self, threshold: float = 0.9) -> bool:
        """Check if origin meets high reliability threshold."""
        return self.reliability_score >= threshold
    
    def get_security_level(self) -> str:
        """Get security level based on configuration."""
        return self.security_settings.get('security_level', 'standard')
    
    def requires_authentication(self) -> bool:
        """Check if origin requires authentication."""
        return self.security_settings.get('requires_authentication', False)