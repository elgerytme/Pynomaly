"""
Tests for DataDomainService
"""

import pytest
from data.core.domain.services.data_domain_service import DataDomainService
from data.core.domain.entities.data_record import DataRecord
from data.core.domain.value_objects.data_identifier import DataIdentifier

class TestDataDomainService:
    """Test suite for DataDomainService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = DataDomainService()
    
    def test_process_record(self):
        """Test record processing"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="Test Record",
            description="Original description"
        )
        
        processed = self.service.process_record(record)
        
        assert processed.id.value == "test-id"
        assert processed.name == "Test Record"
        assert "Processed:" in processed.description
    
    def test_validate_record_valid(self):
        """Test record validation with valid record"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="Test Record",
            description="Test description"
        )
        
        result = self.service.validate_record(record)
        
        assert result is True
    
    def test_validate_record_invalid(self):
        """Test record validation with invalid record"""
        record = DataRecord(
            id=DataIdentifier("test-id"),
            name="",  # Empty name should be invalid
            description="Test description"
        )
        
        result = self.service.validate_record(record)
        
        assert result is False
    
    def test_get_record_summary(self):
        """Test record summary generation"""
        records = [
            DataRecord(id=DataIdentifier("1"), name="Record 1", description="Desc 1"),
            DataRecord(id=DataIdentifier("2"), name="Record 2", description="Desc 2"),
            DataRecord(id=DataIdentifier("3"), name="", description="Desc 3")  # No name
        ]
        
        summary = self.service.get_record_summary(records)
        
        assert summary["total_count"] == 3
        assert summary["named_count"] == 2
        assert summary["average_description_length"] > 0
    
    def test_get_record_summary_empty(self):
        """Test record summary with empty list"""
        summary = self.service.get_record_summary([])
        
        assert summary["total_count"] == 0
        assert summary["named_count"] == 0
        assert summary["average_description_length"] == 0
