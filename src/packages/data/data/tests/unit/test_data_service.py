"""
Tests for DataService
"""

import pytest
from data.core.domain.services.data_service import DataService
from data.core.domain.entities.data_entity import DataEntity

class TestDataService:
    """Test suite for DataService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = DataService()
    
    def test_process_entity(self):
        """Test entity processing"""
        entity = DataEntity(
            id="test-id",
            name="Test Entity",
            description="Original description"
        )
        
        processed = self.service.process_entity(entity)
        
        assert processed.id == "test-id"
        assert processed.name == "Test Entity"
        assert "Processed:" in processed.description
    
    def test_validate_entity_valid(self):
        """Test entity validation with valid entity"""
        entity = DataEntity(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        result = self.service.validate_entity(entity)
        
        assert result is True
    
    def test_validate_entity_invalid(self):
        """Test entity validation with invalid entity"""
        entity = DataEntity(
            id="test-id",
            name="",  # Empty name should be invalid
            description="Test description"
        )
        
        result = self.service.validate_entity(entity)
        
        assert result is False
    
    def test_get_entity_summary(self):
        """Test entity summary generation"""
        entities = [
            DataEntity(id="1", name="Entity 1", description="Desc 1"),
            DataEntity(id="2", name="Entity 2", description="Desc 2"),
            DataEntity(id="3", name="", description="Desc 3")  # No name
        ]
        
        summary = self.service.get_entity_summary(entities)
        
        assert summary["total_count"] == 3
        assert summary["named_count"] == 2
        assert summary["average_description_length"] > 0
    
    def test_get_entity_summary_empty(self):
        """Test entity summary with empty list"""
        summary = self.service.get_entity_summary([])
        
        assert summary["total_count"] == 0
        assert summary["named_count"] == 0
        assert summary["average_description_length"] == 0
