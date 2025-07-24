"""Data identifier value objects."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import hashlib
import uuid


@dataclass(frozen=True)
class DataIdentifier:
    """Unique identifier for data records."""
    
    id: str
    source: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        
        if not self.id:
            raise ValueError("ID cannot be empty")
        if not self.source:
            raise ValueError("Source cannot be empty")
    
    @property
    def fingerprint(self) -> str:
        """Generate a fingerprint hash for this identifier."""
        content = f"{self.id}:{self.source}:{self.timestamp or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @classmethod
    def generate(cls, source: str, metadata: Optional[Dict[str, Any]] = None) -> "DataIdentifier":
        """Generate a new random data identifier."""
        return cls(
            id=str(uuid.uuid4()),
            source=source,
            metadata=metadata
        )
    
    @classmethod
    def from_hash(cls, data: str, source: str) -> "DataIdentifier":
        """Create identifier from data hash."""
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        return cls(
            id=data_hash,
            source=source
        )