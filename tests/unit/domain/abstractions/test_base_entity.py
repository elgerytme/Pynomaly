"""Tests for BaseEntity abstraction."""

import pytest
from datetime import datetime
from unittest.mock import patch
from uuid import uuid4

from pynomaly.domain.abstractions.base_entity import BaseEntity


class SampleEntity(BaseEntity):
    """Sample implementation of BaseEntity for testing."""
    
    def __init__(self, name: str, entity_id: str = None):
        super().__init__(entity_id)
        self.name = name
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def validate(self):
        return bool(self.name)


class TestBaseEntity:
    """Test cases for BaseEntity."""
    
    def test_init_with_default_id(self):
        """Test entity initialization with default ID."""
        entity = SampleEntity("test")
        
        assert entity.id is not None
        assert len(entity.id) == 36  # UUID4 length
        assert entity.name == "test"
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)
    
    def test_init_with_custom_id(self):
        """Test entity initialization with custom ID."""
        custom_id = "custom-id-123"
        entity = SampleEntity("test", custom_id)
        
        assert entity.id == custom_id
        assert entity.name == "test"
    
    def test_timestamps_are_set(self):
        """Test that timestamps are properly set."""
        entity = SampleEntity("test")
        
        assert entity.created_at is not None
        assert entity.updated_at is not None
        assert entity.created_at == entity.updated_at
    
    def test_update_timestamp(self):
        """Test updating timestamp."""
        entity = SampleEntity("test")
        original_updated_at = entity.updated_at
        
        # Sleep to ensure time difference
        import time
        time.sleep(0.01)
        
        entity.update_timestamp()
        
        assert entity.updated_at > original_updated_at
        assert entity.created_at != entity.updated_at
    
    def test_equality_based_on_id(self):
        """Test entity equality based on ID."""
        entity1 = SampleEntity("test1")
        entity2 = SampleEntity("test2")
        entity3 = SampleEntity("test3", entity1.id)
        
        assert entity1 == entity3  # Same ID
        assert entity1 != entity2  # Different ID
    
    def test_equality_with_different_type(self):
        """Test equality with different object types."""
        entity = SampleEntity("test")
        
        assert entity != "not an entity"
        assert entity != 123
        assert entity != None
    
    def test_hash_based_on_id(self):
        """Test entity hashing based on ID."""
        entity1 = SampleEntity("test1")
        entity2 = SampleEntity("test2")
        entity3 = SampleEntity("test3", entity1.id)
        
        assert hash(entity1) == hash(entity3)  # Same ID
        assert hash(entity1) != hash(entity2)  # Different ID
    
    def test_hash_allows_set_usage(self):
        """Test that entities can be used in sets."""
        entity1 = SampleEntity("test1")
        entity2 = SampleEntity("test2")
        entity3 = SampleEntity("test3", entity1.id)
        
        entity_set = {entity1, entity2, entity3}
        
        # Should only contain 2 entities since entity1 and entity3 have same ID
        assert len(entity_set) == 2
        assert entity1 in entity_set
        assert entity2 in entity_set
        assert entity3 in entity_set  # Same as entity1
    
    def test_repr(self):
        """Test string representation."""
        entity = SampleEntity("test")
        repr_str = repr(entity)
        
        assert "SampleEntity" in repr_str
        assert entity.id in repr_str
        assert repr_str.startswith("SampleEntity(id=")
        assert repr_str.endswith(")")
    
    def test_to_dict_implementation(self):
        """Test to_dict method implementation."""
        entity = SampleEntity("test")
        result = entity.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == entity.id
        assert result["name"] == entity.name
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_validate_implementation(self):
        """Test validate method implementation."""
        entity1 = SampleEntity("test")
        entity2 = SampleEntity("")
        
        assert entity1.validate() is True
        assert entity2.validate() is False
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because BaseEntity is abstract
            BaseEntity()
    
    def test_multiple_entities_have_unique_ids(self):
        """Test that multiple entities get unique IDs."""
        entities = [SampleEntity(f"test{i}") for i in range(10)]
        ids = [entity.id for entity in entities]
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids)
    
    @patch('pynomaly.domain.abstractions.base_entity.uuid4')
    def test_id_generation_uses_uuid4(self, mock_uuid4):
        """Test that ID generation uses uuid4."""
        mock_uuid = uuid4()
        mock_uuid4.return_value = mock_uuid
        
        entity = SampleEntity("test")
        
        mock_uuid4.assert_called_once()
        assert entity.id == str(mock_uuid)
    
    def test_created_at_immutable_during_updates(self):
        """Test that created_at doesn't change during updates."""
        entity = SampleEntity("test")
        original_created_at = entity.created_at
        
        # Wait a bit and update timestamp
        import time
        time.sleep(0.01)
        entity.update_timestamp()
        
        assert entity.created_at == original_created_at
        assert entity.updated_at > original_created_at
    
    def test_entity_with_none_id_gets_new_id(self):
        """Test that entity with None ID gets a new ID."""
        entity = SampleEntity("test", None)
        
        assert entity.id is not None
        assert len(entity.id) == 36  # UUID4 length
    
    def test_entity_with_empty_string_id_gets_new_id(self):
        """Test that entity with empty string ID gets a new ID."""
        entity = SampleEntity("test", "")
        
        assert entity.id is not None
        assert entity.id != ""
        assert len(entity.id) == 36  # UUID4 length
