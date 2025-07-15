"""
Entity Relationship Domain Entity

Represents relationships between domain entities with rich metadata,
constraints, and lifecycle management capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..value_objects.entity_id import EntityId
from ..value_objects.entity_metadata import EntityMetadata
from ..exceptions.entity_exceptions import (
    EntityValidationError, 
    InvalidEntityError,
    EntityConflictError
)


class RelationshipType(Enum):
    """Enumeration of supported relationship types."""
    
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"
    HIERARCHICAL = "hierarchical"
    COMPOSITIONAL = "compositional"
    AGGREGATIONAL = "aggregational"
    DEPENDENCY = "dependency"
    ASSOCIATION = "association"


class CascadeAction(Enum):
    """Enumeration of cascade actions for relationship lifecycle."""
    
    CASCADE = "cascade"
    RESTRICT = "restrict"
    SET_NULL = "set_null"
    NO_ACTION = "no_action"


@dataclass
class RelationshipConstraint:
    """Represents a constraint on an entity relationship."""
    
    name: str
    description: str
    constraint_type: str  # e.g., "cardinality", "business_rule", "temporal"
    rule_expression: str
    error_message: str
    is_active: bool = True
    
    def validate(self, source_entity: Any, target_entity: Any, context: Dict[str, Any]) -> bool:
        """
        Validate the constraint against the relationship.
        
        This is a placeholder - real implementation would evaluate the rule_expression.
        """
        # Placeholder validation logic
        return True


@dataclass
class EntityRelationship:
    """
    Domain entity representing a relationship between two entities.
    
    Supports various relationship types, constraints, and lifecycle management
    for building rich domain models.
    """
    
    name: str
    description: str
    relationship_type: RelationshipType
    source_entity_id: EntityId
    target_entity_id: EntityId
    id: EntityId = field(default_factory=EntityId.generate)
    metadata: EntityMetadata = field(default_factory=EntityMetadata.empty)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Relationship properties
    is_bidirectional: bool = True
    source_role: Optional[str] = None
    target_role: Optional[str] = None
    
    # Cardinality constraints
    source_min_cardinality: int = 0
    source_max_cardinality: Optional[int] = None  # None means unlimited
    target_min_cardinality: int = 0
    target_max_cardinality: Optional[int] = None
    
    # Lifecycle management
    on_source_delete: CascadeAction = CascadeAction.NO_ACTION
    on_target_delete: CascadeAction = CascadeAction.NO_ACTION
    
    # Additional properties
    constraints: List[RelationshipConstraint] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    is_active: bool = True
    
    def __post_init__(self) -> None:
        """Validate the relationship after initialization."""
        self._validate_relationship()
    
    def _validate_relationship(self) -> None:
        """Validate relationship properties and business rules."""
        if not self.name or not self.name.strip():
            raise InvalidEntityError("Relationship name cannot be empty")
        
        if not self.description or not self.description.strip():
            raise InvalidEntityError("Relationship description cannot be empty")
        
        if self.source_entity_id == self.target_entity_id:
            # Self-relationships are allowed but should be explicitly handled
            pass
        
        # Validate cardinality constraints
        if self.source_min_cardinality < 0:
            raise EntityValidationError("Source minimum cardinality cannot be negative")
        
        if self.target_min_cardinality < 0:
            raise EntityValidationError("Target minimum cardinality cannot be negative")
        
        if (self.source_max_cardinality is not None and 
            self.source_max_cardinality < self.source_min_cardinality):
            raise EntityValidationError("Source maximum cardinality cannot be less than minimum")
        
        if (self.target_max_cardinality is not None and 
            self.target_max_cardinality < self.target_min_cardinality):
            raise EntityValidationError("Target maximum cardinality cannot be less than minimum")
    
    def validate_cardinality(self, source_count: int, target_count: int) -> List[str]:
        """
        Validate current entity counts against cardinality constraints.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check source cardinality
        if source_count < self.source_min_cardinality:
            errors.append(f"Source cardinality {source_count} below minimum {self.source_min_cardinality}")
        
        if (self.source_max_cardinality is not None and 
            source_count > self.source_max_cardinality):
            errors.append(f"Source cardinality {source_count} exceeds maximum {self.source_max_cardinality}")
        
        # Check target cardinality
        if target_count < self.target_min_cardinality:
            errors.append(f"Target cardinality {target_count} below minimum {self.target_min_cardinality}")
        
        if (self.target_max_cardinality is not None and 
            target_count > self.target_max_cardinality):
            errors.append(f"Target cardinality {target_count} exceeds maximum {self.target_max_cardinality}")
        
        return errors
    
    def add_constraint(self, constraint: RelationshipConstraint) -> EntityRelationship:
        """Add a constraint to the relationship."""
        # Check for duplicate constraint names
        existing_names = {c.name for c in self.constraints}
        if constraint.name in existing_names:
            raise EntityConflictError(f"Constraint '{constraint.name}' already exists")
        
        new_constraints = list(self.constraints)
        new_constraints.append(constraint)
        
        return self._create_updated_copy(
            constraints=new_constraints,
            updated_at=datetime.now()
        )
    
    def remove_constraint(self, constraint_name: str) -> EntityRelationship:
        """Remove a constraint from the relationship."""
        new_constraints = [c for c in self.constraints if c.name != constraint_name]
        
        if len(new_constraints) == len(self.constraints):
            raise InvalidEntityError(f"Constraint '{constraint_name}' not found")
        
        return self._create_updated_copy(
            constraints=new_constraints,
            updated_at=datetime.now()
        )
    
    def validate_constraints(self, source_entity: Any, target_entity: Any, 
                           context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate all active constraints against the relationship.
        
        Returns:
            List of constraint violation messages
        """
        if context is None:
            context = {}
        
        violations = []
        for constraint in self.constraints:
            if constraint.is_active:
                try:
                    if not constraint.validate(source_entity, target_entity, context):
                        violations.append(constraint.error_message)
                except Exception as e:
                    violations.append(f"Constraint '{constraint.name}' validation failed: {str(e)}")
        
        return violations
    
    def set_property(self, key: str, value: Any) -> EntityRelationship:
        """Set a custom property on the relationship."""
        new_properties = dict(self.properties)
        new_properties[key] = value
        
        return self._create_updated_copy(
            properties=new_properties,
            updated_at=datetime.now()
        )
    
    def remove_property(self, key: str) -> EntityRelationship:
        """Remove a custom property from the relationship."""
        if key not in self.properties:
            raise InvalidEntityError(f"Property '{key}' not found")
        
        new_properties = dict(self.properties)
        del new_properties[key]
        
        return self._create_updated_copy(
            properties=new_properties,
            updated_at=datetime.now()
        )
    
    def add_tag(self, tag: str) -> EntityRelationship:
        """Add a tag to the relationship."""
        new_tags = set(self.tags)
        new_tags.add(tag.strip().lower())
        
        return self._create_updated_copy(
            tags=new_tags,
            updated_at=datetime.now()
        )
    
    def remove_tag(self, tag: str) -> EntityRelationship:
        """Remove a tag from the relationship."""
        new_tags = set(self.tags)
        new_tags.discard(tag.strip().lower())
        
        return self._create_updated_copy(
            tags=new_tags,
            updated_at=datetime.now()
        )
    
    def get_inverse_relationship(self) -> EntityRelationship:
        """
        Create the inverse relationship (if bidirectional).
        
        Returns:
            New EntityRelationship representing the inverse direction
        """
        if not self.is_bidirectional:
            raise InvalidEntityError("Cannot create inverse of unidirectional relationship")
        
        return EntityRelationship(
            name=f"{self.name}_inverse",
            description=f"Inverse of: {self.description}",
            relationship_type=self.relationship_type,
            source_entity_id=self.target_entity_id,
            target_entity_id=self.source_entity_id,
            is_bidirectional=True,
            source_role=self.target_role,
            target_role=self.source_role,
            source_min_cardinality=self.target_min_cardinality,
            source_max_cardinality=self.target_max_cardinality,
            target_min_cardinality=self.source_min_cardinality,
            target_max_cardinality=self.source_max_cardinality,
            on_source_delete=self.on_target_delete,
            on_target_delete=self.on_source_delete,
            metadata=self.metadata,
            properties=dict(self.properties),
            tags=set(self.tags)
        )
    
    def update_cardinality(self, source_min: int, source_max: Optional[int],
                          target_min: int, target_max: Optional[int]) -> EntityRelationship:
        """Update cardinality constraints."""
        return self._create_updated_copy(
            source_min_cardinality=source_min,
            source_max_cardinality=source_max,
            target_min_cardinality=target_min,
            target_max_cardinality=target_max,
            updated_at=datetime.now()
        )
    
    def deactivate(self) -> EntityRelationship:
        """Deactivate the relationship."""
        return self._create_updated_copy(
            is_active=False,
            updated_at=datetime.now()
        )
    
    def activate(self) -> EntityRelationship:
        """Activate the relationship."""
        return self._create_updated_copy(
            is_active=True,
            updated_at=datetime.now()
        )
    
    def _create_updated_copy(self, **kwargs) -> EntityRelationship:
        """Create an updated copy of the relationship with specified changes."""
        update_data = {
            'name': self.name,
            'description': self.description,
            'relationship_type': self.relationship_type,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'id': self.id,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_bidirectional': self.is_bidirectional,
            'source_role': self.source_role,
            'target_role': self.target_role,
            'source_min_cardinality': self.source_min_cardinality,
            'source_max_cardinality': self.source_max_cardinality,
            'target_min_cardinality': self.target_min_cardinality,
            'target_max_cardinality': self.target_max_cardinality,
            'on_source_delete': self.on_source_delete,
            'on_target_delete': self.on_target_delete,
            'constraints': self.constraints,
            'properties': self.properties,
            'tags': self.tags,
            'is_active': self.is_active
        }
        update_data.update(kwargs)
        
        return EntityRelationship(**update_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'relationship_type': self.relationship_type.value,
            'source_entity_id': str(self.source_entity_id),
            'target_entity_id': str(self.target_entity_id),
            'metadata': self.metadata.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_bidirectional': self.is_bidirectional,
            'source_role': self.source_role,
            'target_role': self.target_role,
            'source_min_cardinality': self.source_min_cardinality,
            'source_max_cardinality': self.source_max_cardinality,
            'target_min_cardinality': self.target_min_cardinality,
            'target_max_cardinality': self.target_max_cardinality,
            'on_source_delete': self.on_source_delete.value,
            'on_target_delete': self.on_target_delete.value,
            'constraints': [
                {
                    'name': c.name,
                    'description': c.description,
                    'constraint_type': c.constraint_type,
                    'rule_expression': c.rule_expression,
                    'error_message': c.error_message,
                    'is_active': c.is_active
                }
                for c in self.constraints
            ],
            'properties': dict(self.properties),
            'tags': list(self.tags),
            'is_active': self.is_active
        }