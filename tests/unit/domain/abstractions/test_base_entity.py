"""Tests for BaseEntity class."""

import pytest
from datetime import datetime
from uuid import UUID, uuid4

from pynomaly.domain.abstractions.base_entity import BaseEntity


class TestEntity(BaseEntity):
    """Test entity for testing BaseEntity."""
    
    name: str
    description: str = "default description"


class TestBaseEntity:
    """Test cases for BaseEntity."""

    def test_base_entity_creation(self):
        """Test basic entity creation."""
        entity = TestEntity(name="test")
        
        assert isinstance(entity.id, UUID)
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)
        assert entity.version == 1
        assert isinstance(entity.metadata, dict)
        assert entity.name == "test"
        assert entity.description == "default description"

    def test_base_entity_with_custom_id(self):
        """Test entity creation with custom ID."""
        custom_id = uuid4()
        entity = TestEntity(name="test", id=custom_id)
        
        assert entity.id == custom_id

    def test_base_entity_equality(self):
        """Test entity equality based on ID."""
        id1 = uuid4()
        entity1 = TestEntity(name="test1", id=id1)
        entity2 = TestEntity(name="test2", id=id1)  # Same ID, different name
        entity3 = TestEntity(name="test1", id=uuid4())  # Different ID, same name
        
        assert entity1 == entity2  # Same ID
        assert entity1 != entity3  # Different ID
        assert entity2 != entity3  # Different ID

    def test_base_entity_hash(self):
        """Test entity hashing based on ID."""
        id1 = uuid4()
        entity1 = TestEntity(name="test1", id=id1)
        entity2 = TestEntity(name="test2", id=id1)  # Same ID
        
        assert hash(entity1) == hash(entity2)

    def test_base_entity_repr(self):
        """Test entity string representation."""
        entity = TestEntity(name="test")
        repr_str = repr(entity)
        
        assert "TestEntity" in repr_str
        assert str(entity.id) in repr_str

    def test_mark_as_updated(self):
        """Test mark_as_updated method."""
        entity = TestEntity(name="test")
        original_updated_at = entity.updated_at
        original_version = entity.version
        
        # Wait a bit to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        entity.mark_as_updated()
        
        assert entity.updated_at > original_updated_at
        assert entity.version == original_version + 1

    def test_is_new(self):
        """Test is_new method."""
        entity = TestEntity(name="test")
        
        assert entity.is_new()
        
        entity.mark_as_updated()
        
        assert not entity.is_new()

    def test_validate_invariants(self):
        """Test validate_invariants method."""
        entity = TestEntity(name="test")
        entity.validate_invariants()  # Should not raise
        
        # Manually set invalid version
        entity.version = 0
        with pytest.raises(ValueError, match="Entity version must be positive"):
            entity.validate_invariants()

    def test_get_identifier_field(self):
        """Test get_identifier_field class method."""
        assert TestEntity.get_identifier_field() == "id"

    def test_metadata_operations(self):
        """Test metadata operations."""
        entity = TestEntity(name="test")
        
        # Initial metadata should be empty
        assert entity.metadata == {}
        
        # Add metadata
        entity.metadata["key1"] = "value1"
        entity.metadata["key2"] = 42
        
        assert entity.metadata["key1"] == "value1"
        assert entity.metadata["key2"] == 42

    def test_json_serialization(self):
        """Test JSON serialization."""
        entity = TestEntity(name="test")
        json_data = entity.dict()
        
        assert "id" in json_data
        assert "created_at" in json_data
        assert "updated_at" in json_data
        assert "version" in json_data
        assert "metadata" in json_data
        assert "name" in json_data
        assert json_data["name"] == "test"

    def test_entity_with_custom_metadata(self):
        """Test entity creation with custom metadata."""
        custom_metadata = {"source": "test", "priority": "high"}
        entity = TestEntity(name="test", metadata=custom_metadata)
        
        assert entity.metadata == custom_metadata

    def test_entity_immutable_during_validation(self):
        """Test that entity validation doesn't change core fields."""
        entity = TestEntity(name="test")
        original_id = entity.id
        original_created_at = entity.created_at
        original_version = entity.version
        
        entity.validate_invariants()
        
        assert entity.id == original_id
        assert entity.created_at == original_created_at
        assert entity.version == original_version
