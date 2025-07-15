"""
Domain Entity - Core entity representing a business domain object.

This module defines the DomainEntity class which represents any business domain
object in the catalog system with full metadata, versioning, and relationship tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..value_objects.entity_id import EntityId
from ..value_objects.entity_metadata import EntityMetadata
from ..value_objects.version_number import VersionNumber
from ..exceptions.entity_exceptions import InvalidEntityError


@dataclass
class DomainEntity:
    """
    Core domain entity representing a business object in the catalog.

    A DomainEntity is the fundamental building block of the domain catalog system.
    It represents any significant business concept, data structure, or operational
    unit that has business meaning and value.

    Attributes:
        id: Unique identifier for this entity
        name: Human-readable name of the entity
        description: Detailed description of the entity's purpose
        category: Business category classification
        metadata: Rich metadata about the entity
        created_at: When this entity was first created
        updated_at: When this entity was last modified
        version: Current version of this entity
        tags: Searchable tags for categorization
        attributes: Key-value pairs of entity attributes
        is_active: Whether this entity is currently active
        business_rules: List of business rules that apply to this entity
    """

    name: str
    description: str
    category: str
    id: EntityId = field(default_factory=EntityId.generate)
    metadata: EntityMetadata = field(default_factory=EntityMetadata.empty)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: VersionNumber = field(default_factory=lambda: VersionNumber("1.0.0"))
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    business_rules: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate entity after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate entity invariants."""
        if not self.name or not self.name.strip():
            raise InvalidEntityError("Entity name cannot be empty")
        
        if not self.description or not self.description.strip():
            raise InvalidEntityError("Entity description cannot be empty")
        
        if not self.category or not self.category.strip():
            raise InvalidEntityError("Entity category cannot be empty")
        
        if len(self.name) > 255:
            raise InvalidEntityError("Entity name cannot exceed 255 characters")
        
        if len(self.description) > 2000:
            raise InvalidEntityError("Entity description cannot exceed 2000 characters")

    def update_description(self, new_description: str) -> None:
        """Update the entity description."""
        if not new_description or not new_description.strip():
            raise InvalidEntityError("Description cannot be empty")
        
        if len(new_description) > 2000:
            raise InvalidEntityError("Description cannot exceed 2000 characters")
        
        self.description = new_description.strip()
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the entity."""
        if not tag or not tag.strip():
            raise InvalidEntityError("Tag cannot be empty")
        
        tag = tag.strip().lower()
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the entity."""
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()

    def add_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute."""
        if not key or not key.strip():
            raise InvalidEntityError("Attribute key cannot be empty")
        
        self.attributes[key.strip()] = value
        self.updated_at = datetime.utcnow()

    def remove_attribute(self, key: str) -> None:
        """Remove an attribute."""
        if key in self.attributes:
            del self.attributes[key]
            self.updated_at = datetime.utcnow()

    def add_business_rule(self, rule: str) -> None:
        """Add a business rule to the entity."""
        if not rule or not rule.strip():
            raise InvalidEntityError("Business rule cannot be empty")
        
        rule = rule.strip()
        if rule not in self.business_rules:
            self.business_rules.append(rule)
            self.updated_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Mark the entity as inactive."""
        self.is_active = False
        self.updated_at = datetime.utcnow()

    def activate(self) -> None:
        """Mark the entity as active."""
        self.is_active = True
        self.updated_at = datetime.utcnow()

    def increment_version(self, version_type: str = "patch") -> None:
        """Increment the entity version."""
        self.version = self.version.increment(version_type)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": str(self.version),
            "tags": self.tags.copy(),
            "attributes": self.attributes.copy(),
            "is_active": self.is_active,
            "business_rules": self.business_rules.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainEntity:
        """Create entity from dictionary representation."""
        return cls(
            id=EntityId(UUID(data["id"])),
            name=data["name"],
            description=data["description"],
            category=data["category"],
            metadata=EntityMetadata.from_dict(data.get("metadata", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=VersionNumber(data["version"]),
            tags=data.get("tags", []),
            attributes=data.get("attributes", {}),
            is_active=data.get("is_active", True),
            business_rules=data.get("business_rules", [])
        )

    def __str__(self) -> str:
        """String representation for users."""
        return f"{self.name} (v{self.version}) - {self.description[:100]}{'...' if len(self.description) > 100 else ''}"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"DomainEntity(id={self.id}, name='{self.name}', "
            f"category='{self.category}', version={self.version}, active={self.is_active})"
        )