"""Tests for BaseEntity domain abstraction."""

import pytest
from datetime import datetime
from uuid import UUID
from typing import Any

from pynomaly.domain.abstractions import BaseEntity


class TestEntityImpl(BaseEntity):
    """Test implementation of BaseEntity."""
    
    name: str
    value: int = 0


class TestBaseEntity:
    """Test BaseEntity functionality."""

    def test_entity_creation(self):
        """Test entity creation with default fields."""
        entity = TestEntityImpl(name="test", value=42)
        
        assert entity.name == "test"
        assert entity.value == 42
        assert isinstance(entity.id, UUID)
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)
        assert entity.version == 1
        assert isinstance(entity.metadata, dict)
        assert len(entity.metadata) == 0

    def test_entity_equality(self):
        """Test entity equality based on ID."""
        entity1 = TestEntityImpl(name="test1", value=1)
        entity2 = TestEntityImpl(name="test2", value=2)
        entity3 = TestEntityImpl(id=entity1.id, name="test3", value=3)
        
        # Different entities should not be equal
        assert entity1 != entity2
        
        # Same ID should be equal regardless of other fields
        assert entity1 == entity3
        
        # Entity should not equal non-entity
        assert entity1 != "not an entity"

    def test_entity_hash(self):
        """Test entity hashing based on ID."""
        entity1 = TestEntityImpl(name="test1", value=1)
        entity2 = TestEntityImpl(id=entity1.id, name="test2", value=2)
        
        # Same ID should have same hash
        assert hash(entity1) == hash(entity2)

    def test_entity_repr(self):
        """Test entity string representation."""
        entity = TestEntityImpl(name="test", value=42)
        repr_str = repr(entity)
        
        assert "TestEntityImpl" in repr_str
        assert str(entity.id) in repr_str

    def test_mark_as_updated(self):
        """Test marking entity as updated."""
        entity = TestEntityImpl(name="test", value=42)
        original_updated_at = entity.updated_at
        original_version = entity.version
        
        # Small delay to ensure time difference
        import time
        time.sleep(0.001)
        
        entity.mark_as_updated()
        
        assert entity.updated_at > original_updated_at
        assert entity.version == original_version + 1

    def test_is_new(self):
        """Test checking if entity is new."""
        entity = TestEntityImpl(name="test", value=42)
        
        # New entity should be new
        assert entity.is_new()
        
        # After update, should not be new
        entity.mark_as_updated()
        assert not entity.is_new()

    def test_validate_invariants(self):
        """Test entity invariant validation."""
        entity = TestEntityImpl(name="test", value=42)
        
        # Valid entity should pass validation
        entity.validate_invariants()
        
        # Invalid version should fail validation
        entity.version = 0
        with pytest.raises(ValueError, match="Entity version must be positive"):
            entity.validate_invariants()

    def test_get_identifier_field(self):
        """Test getting identifier field name."""
        assert TestEntityImpl.get_identifier_field() == "id"

    def test_custom_initialization(self):
        """Test custom initialization with validation."""
        # Valid entity should work
        entity = TestEntityImpl(name="test", value=42)
        assert entity.name == "test"
        
        # Entity with custom validation should work
        class ValidatedEntity(BaseEntity):
            name: str
            value: int
            
            def __init__(self, **data: Any) -> None:
                super().__init__(**data)
                self.validate_invariants()
            
            def validate_invariants(self) -> None:
                super().validate_invariants()
                if not self.name:
                    raise ValueError("Name cannot be empty")
                if self.value < 0:
                    raise ValueError("Value must be non-negative")
        
        # Valid entity should pass
        valid_entity = ValidatedEntity(name="test", value=42)
        assert valid_entity.name == "test"
        
        # Invalid entities should fail
        with pytest.raises(ValueError, match="Name cannot be empty"):
            ValidatedEntity(name="", value=42)
        
        with pytest.raises(ValueError, match="Value must be non-negative"):
            ValidatedEntity(name="test", value=-1)

    def test_metadata_operations(self):
        """Test metadata field operations."""
        entity = TestEntityImpl(name="test", value=42)
        
        # Initial metadata should be empty
        assert entity.metadata == {}
        
        # Should be able to add metadata
        entity.metadata["key1"] = "value1"
        entity.metadata["key2"] = 123
        
        assert entity.metadata["key1"] == "value1"
        assert entity.metadata["key2"] == 123

    def test_json_serialization(self):
        """Test entity JSON serialization."""
        entity = TestEntityImpl(name="test", value=42)
        
        # Should be able to serialize to dict
        entity_dict = entity.model_dump()
        
        assert entity_dict["name"] == "test"
        assert entity_dict["value"] == 42
        assert "id" in entity_dict
        assert "created_at" in entity_dict
        assert "updated_at" in entity_dict
        assert "version" in entity_dict
        assert "metadata" in entity_dict

    def test_inheritance_with_additional_fields(self):
        """Test entity inheritance with additional fields."""
        class ExtendedEntity(BaseEntity):
            name: str
            value: int
            description: str = "default"
            tags: list[str] = []
            
            def __init__(self, **data: Any) -> None:
                super().__init__(**data)
                self.validate_invariants()
            
            def validate_invariants(self) -> None:
                super().validate_invariants()
                if len(self.name) < 3:
                    raise ValueError("Name must be at least 3 characters")
        
        # Valid extended entity
        entity = ExtendedEntity(
            name="test_entity",
            value=42,
            description="A test entity",
            tags=["test", "entity"]
        )
        
        assert entity.name == "test_entity"
        assert entity.description == "A test entity"
        assert entity.tags == ["test", "entity"]
        assert entity.version == 1
        
        # Invalid extended entity
        with pytest.raises(ValueError, match="Name must be at least 3 characters"):
            ExtendedEntity(name="ab", value=42)

    def test_entity_lifecycle(self):
        """Test complete entity lifecycle."""
        # Create entity
        entity = TestEntityImpl(name="test", value=42)
        assert entity.is_new()
        assert entity.version == 1
        
        # Update entity multiple times
        for i in range(5):
            entity.mark_as_updated()
            assert entity.version == i + 2
            assert not entity.is_new()
        
        # Final state
        assert entity.version == 6
        assert not entity.is_new()