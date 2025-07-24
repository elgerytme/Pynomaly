"""
Tests for DataRecord and BaseDataEntity
"""

import pytest
from datetime import datetime
from data.core.domain.entities.data_record import DataRecord
from data.core.domain.entities.base_data_entity import BaseDataEntity
from data.core.domain.value_objects.data_identifier import DataIdentifier

class TestDataRecord:
    """Test suite for DataRecord"""
    
    def test_record_creation(self):
        """Test DataRecord creation"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="Test Record",
            description="Test description"
        )
        
        assert record.id.value == "test-id"
        assert record.name == "Test Record"
        assert record.description == "Test description"
    
    def test_record_to_dict(self):
        """Test DataRecord to dictionary conversion"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="Test Record",
            description="Test description"
        )
        
        result = record.to_dict()
        
        assert result["id"] == "test-id"
        assert result["name"] == "Test Record"
        assert result["description"] == "Test description"
    
    def test_record_string_representation(self):
        """Test DataRecord string representation"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="Test Record"
        )
        
        str_repr = str(record)
        assert "test-id" in str_repr
        assert "Test Record" in str_repr
    
    def test_record_repr(self):
        """Test DataRecord detailed representation"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="Test Record",
            description="Test description"
        )
        
        repr_str = repr(record)
        assert "test-id" in repr_str
        assert "Test Record" in repr_str
        assert "Test description" in repr_str

class TestBaseDataEntity:
    """Test suite for BaseDataEntity"""

    def test_base_data_entity_creation(self):
        """Test BaseDataEntity creation"""
        identifier = DataIdentifier("base-id")
        entity = BaseDataEntity(id=identifier)

        assert entity.id == identifier
        assert entity._id.value == "base-id"

    def test_base_data_entity_equality(self):
        """Test BaseDataEntity equality"""
        id1 = DataIdentifier("id-1")
        id2 = DataIdentifier("id-2")
        entity1 = BaseDataEntity(id=id1)
        entity1_copy = BaseDataEntity(id=id1)
        entity2 = BaseDataEntity(id=id2)

        assert entity1 == entity1_copy
        assert entity1 != entity2

    def test_base_data_entity_hash(self):
        """Test BaseDataEntity hashing"""
        id1 = DataIdentifier("id-a")
        id2 = DataIdentifier("id-b")
        entity1 = BaseDataEntity(id=id1)
        entity2 = BaseDataEntity(id=id2)

        assert hash(entity1) == hash(id1)
        assert hash(entity1) != hash(entity2)

    def test_base_data_entity_id_type_enforcement(self):
        """Test that BaseDataEntity enforces DataIdentifier type for id"""
        with pytest.raises(TypeError, match="id must be an instance of DataIdentifier"):
            BaseDataEntity(id="invalid-id")
