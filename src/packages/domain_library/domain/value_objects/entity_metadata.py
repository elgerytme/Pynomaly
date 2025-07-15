"""
Entity Metadata Value Object

Represents rich metadata associated with domain entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EntityMetadata:
    """
    Value object representing metadata for domain entities.
    
    Stores additional descriptive information about entities including
    authorship, source, compliance, and custom properties.
    """
    
    author: Optional[str] = None
    source: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    documentation_url: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate metadata fields."""
        # Ensure compliance_tags and examples are lists
        if not isinstance(self.compliance_tags, list):
            object.__setattr__(self, 'compliance_tags', list(self.compliance_tags) if self.compliance_tags else [])
        
        if not isinstance(self.examples, list):
            object.__setattr__(self, 'examples', list(self.examples) if self.examples else [])
        
        if not isinstance(self.custom_properties, dict):
            object.__setattr__(self, 'custom_properties', dict(self.custom_properties) if self.custom_properties else {})
    
    @classmethod
    def empty(cls) -> EntityMetadata:
        """Create empty metadata."""
        return cls()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityMetadata:
        """Create metadata from dictionary."""
        return cls(
            author=data.get('author'),
            source=data.get('source'),
            compliance_tags=data.get('compliance_tags', []),
            custom_properties=data.get('custom_properties', {}),
            documentation_url=data.get('documentation_url'),
            examples=data.get('examples', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'author': self.author,
            'source': self.source,
            'compliance_tags': list(self.compliance_tags),
            'custom_properties': dict(self.custom_properties),
            'documentation_url': self.documentation_url,
            'examples': list(self.examples)
        }
    
    def with_author(self, author: str) -> EntityMetadata:
        """Create new metadata with updated author."""
        return EntityMetadata(
            author=author,
            source=self.source,
            compliance_tags=self.compliance_tags,
            custom_properties=self.custom_properties,
            documentation_url=self.documentation_url,
            examples=self.examples
        )
    
    def with_source(self, source: str) -> EntityMetadata:
        """Create new metadata with updated source."""
        return EntityMetadata(
            author=self.author,
            source=source,
            compliance_tags=self.compliance_tags,
            custom_properties=self.custom_properties,
            documentation_url=self.documentation_url,
            examples=self.examples
        )
    
    def add_compliance_tag(self, tag: str) -> EntityMetadata:
        """Create new metadata with additional compliance tag."""
        new_tags = list(self.compliance_tags)
        if tag not in new_tags:
            new_tags.append(tag)
        
        return EntityMetadata(
            author=self.author,
            source=self.source,
            compliance_tags=new_tags,
            custom_properties=self.custom_properties,
            documentation_url=self.documentation_url,
            examples=self.examples
        )
    
    def add_custom_property(self, key: str, value: Any) -> EntityMetadata:
        """Create new metadata with additional custom property."""
        new_properties = dict(self.custom_properties)
        new_properties[key] = value
        
        return EntityMetadata(
            author=self.author,
            source=self.source,
            compliance_tags=self.compliance_tags,
            custom_properties=new_properties,
            documentation_url=self.documentation_url,
            examples=self.examples
        )