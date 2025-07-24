"""
DataRecord

Primary entity for data records.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from data.core.domain.entities.base_data_entity import BaseDataEntity
from data.core.domain.value_objects.data_identifier import DataIdentifier

@dataclass
class DataRecord(BaseDataEntity):
    """
    Primary entity for data records.
    """
    
    # Override id from BaseDataEntity to ensure it's DataIdentifier
    id: DataIdentifier
    name: str = ""
    description: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        # Call BaseDataEntity's __init__ to set the id
        super().__init__(self.id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation"""
        return {
            "id": str(self.id), # Convert DataIdentifier to string for dict representation
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"DataRecord(id={self.id}, name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (
            f"DataRecord("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"created_at={self.created_at!r}, "
            f"updated_at={self.updated_at!r}"
            f")"
        )
