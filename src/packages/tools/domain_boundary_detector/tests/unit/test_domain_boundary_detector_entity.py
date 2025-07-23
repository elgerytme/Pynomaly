"""
Tests for DomainBoundaryDetectorEntity
"""

import pytest
from datetime import datetime
from domain_boundary_detector.core.domain.entities.domain_boundary_detector_entity import DomainBoundaryDetectorEntity

class TestDomainBoundaryDetectorEntity:
    """Test suite for DomainBoundaryDetectorEntity"""
    
    def test_entity_creation(self):
        """Test entity creation"""
        entity = DomainBoundaryDetectorEntity(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        assert entity.id == "test-id"
        assert entity.name == "Test Entity"
        assert entity.description == "Test description"
    
    def test_entity_to_dict(self):
        """Test entity to dictionary conversion"""
        entity = DomainBoundaryDetectorEntity(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        result = entity.to_dict()
        
        assert result["id"] == "test-id"
        assert result["name"] == "Test Entity"
        assert result["description"] == "Test description"
    
    def test_entity_string_representation(self):
        """Test entity string representation"""
        entity = DomainBoundaryDetectorEntity(
            id="test-id",
            name="Test Entity"
        )
        
        str_repr = str(entity)
        assert "test-id" in str_repr
        assert "Test Entity" in str_repr
    
    def test_entity_repr(self):
        """Test entity detailed representation"""
        entity = DomainBoundaryDetectorEntity(
            id="test-id",
            name="Test Entity",
            description="Test description"
        )
        
        repr_str = repr(entity)
        assert "test-id" in repr_str
        assert "Test Entity" in repr_str
        assert "Test description" in repr_str
