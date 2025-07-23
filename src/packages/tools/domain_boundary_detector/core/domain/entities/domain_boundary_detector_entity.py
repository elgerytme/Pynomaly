"""
DomainBoundaryDetectorEntity

Sample domain entity for domain_boundary_detector domain.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from software.core.domain.abstractions.base_entity import BaseEntity

@dataclass
class DomainBoundaryDetectorEntity(BaseEntity):
    """
    Sample domain entity for domain_boundary_detector domain.
    
    This is a template - replace with actual domain entities.
    """
    
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"DomainBoundaryDetectorEntity(id={self.id}, name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (
            f"DomainBoundaryDetectorEntity("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"created_at={self.created_at!r}, "
            f"updated_at={self.updated_at!r}"
            f")"
        )
