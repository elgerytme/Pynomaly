"""Tests for data origin entity."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# TODO: Replace with actual data domain entities when available
# Currently creating test fixtures since the referenced entities don't exist
# Original import: from ....data.domain.entities.data_origin import DataOrigin, OriginType

class OriginType(str, Enum):
    """Mock origin type for testing."""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"

@dataclass
class DataOrigin:
    """Mock data origin entity for testing."""
    id: str = ""
    name: str = ""
    origin_type: OriginType = OriginType.FILE
    source_path: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if not self.id:
            self.id = str(uuid4())


class TestDataOrigin:
    """Test cases for DataOrigin entity."""
    
    def test_create_database_origin(self):
        """Test creating a database data origin."""
        origin = DataOrigin(
            name="production_db",
            origin_type=OriginType.DATABASE,
            description="Production PostgreSQL database",
            system_name="prod-postgres-01",
            owner="dba-team@company.com",
            is_trusted=True
        )
        
        assert origin.name == "production_db"
        assert origin.origin_type == OriginType.DATABASE
        assert origin.description == "Production PostgreSQL database"
        assert origin.system_name == "prod-postgres-01"
        assert origin.owner == "dba-team@company.com"
        assert origin.is_trusted is True
        assert origin.is_active is True
        assert origin.reliability_score == 1.0
    
    def test_create_api_origin(self):
        """Test creating an API data origin."""
        origin = DataOrigin(
            name="customer_api",
            origin_type=OriginType.API,
            description="Customer management API",
            location="https://api.company.com/customers",
            reliability_score=0.95
        )
        
        assert origin.origin_type == OriginType.API
        assert origin.location == "https://api.company.com/customers"
        assert origin.reliability_score == 0.95
    
    def test_name_validation(self):
        """Test name validation."""
        with pytest.raises(ValueError, match="Origin name cannot be empty"):
            DataOrigin(
                name="   ",  # Empty after strip
                origin_type=OriginType.DATABASE
            )
    
    def test_reliability_score_validation(self):
        """Test reliability score validation."""
        with pytest.raises(ValueError, match="Reliability score must be between 0.0 and 1.0"):
            DataOrigin(
                name="test_origin",
                origin_type=OriginType.DATABASE,
                reliability_score=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError, match="Reliability score must be between 0.0 and 1.0"):
            DataOrigin(
                name="test_origin",
                origin_type=OriginType.DATABASE,
                reliability_score=-0.1  # Invalid: < 0.0
            )
    
    def test_record_access(self):
        """Test recording access to origin."""
        origin = DataOrigin(
            name="test_origin",
            origin_type=OriginType.DATABASE
        )
        
        initial_frequency = origin.access_frequency
        initial_accessed_at = origin.last_accessed_at
        
        origin.record_access()
        
        assert origin.access_frequency == initial_frequency + 1
        assert origin.last_accessed_at > initial_accessed_at
        assert origin.updated_at is not None
    
    def test_update_reliability_score(self):
        """Test updating reliability score."""
        origin = DataOrigin(
            name="test_origin",
            origin_type=OriginType.DATABASE,
            reliability_score=1.0
        )
        
        origin.update_reliability_score(0.8)
        
        assert origin.reliability_score == 0.8
        assert origin.updated_at is not None
    
    def test_update_reliability_score_validation(self):
        """Test reliability score validation during update."""
        origin = DataOrigin(
            name="test_origin",
            origin_type=OriginType.DATABASE
        )
        
        with pytest.raises(ValueError, match="Reliability score must be between 0.0 and 1.0"):
            origin.update_reliability_score(2.0)
    
    def test_deactivate_origin(self):
        """Test deactivating an origin."""
        origin = DataOrigin(
            name="test_origin",
            origin_type=OriginType.DATABASE
        )
        
        reason = "Database maintenance"
        origin.deactivate(reason)
        
        assert origin.is_active is False
        assert origin.updated_at is not None
        assert "deactivation_reasons" in origin.metadata
        assert len(origin.metadata["deactivation_reasons"]) == 1
        assert origin.metadata["deactivation_reasons"][0]["reason"] == reason
    
    def test_activate_origin(self):
        """Test activating an origin."""
        origin = DataOrigin(
            name="test_origin",
            origin_type=OriginType.DATABASE,
            is_active=False
        )
        
        origin.activate()
        
        assert origin.is_active is True
        assert origin.updated_at is not None
    
    def test_origin_type_checks(self):
        """Test origin type checking methods."""
        db_origin = DataOrigin(name="db", origin_type=OriginType.DATABASE)
        file_origin = DataOrigin(name="file", origin_type=OriginType.FILE_SYSTEM)
        stream_origin = DataOrigin(name="stream", origin_type=OriginType.STREAM)
        api_origin = DataOrigin(name="api", origin_type=OriginType.API)
        external_origin = DataOrigin(name="ext", origin_type=OriginType.EXTERNAL_SYSTEM)
        
        assert db_origin.is_database_origin()
        assert not db_origin.is_file_origin()
        assert not db_origin.is_streaming_origin()
        assert not db_origin.is_external_origin()
        
        assert file_origin.is_file_origin()
        assert not file_origin.is_database_origin()
        
        assert stream_origin.is_streaming_origin()
        assert not stream_origin.is_database_origin()
        
        assert api_origin.is_external_origin()
        assert external_origin.is_external_origin()
        assert not db_origin.is_external_origin()
    
    def test_access_frequency_calculation(self):
        """Test access frequency per day calculation."""
        origin = DataOrigin(
            name="test_origin",
            origin_type=OriginType.DATABASE,
            access_frequency=60  # 60 accesses
        )
        
        # 60 accesses over 30 days = 2 per day
        assert origin.get_access_frequency_per_day(30) == 2.0
        
        # 60 accesses over 60 days = 1 per day
        assert origin.get_access_frequency_per_day(60) == 1.0
        
        # Edge case: 0 days
        assert origin.get_access_frequency_per_day(0) == 0.0
    
    def test_high_reliability_check(self):
        """Test high reliability checking."""
        high_reliability = DataOrigin(
            name="reliable_origin",
            origin_type=OriginType.DATABASE,
            reliability_score=0.95
        )
        low_reliability = DataOrigin(
            name="unreliable_origin",
            origin_type=OriginType.DATABASE,
            reliability_score=0.8
        )
        
        assert high_reliability.is_highly_reliable()  # Default threshold 0.9
        assert high_reliability.is_highly_reliable(0.9)
        assert not high_reliability.is_highly_reliable(0.98)  # Higher threshold
        
        assert not low_reliability.is_highly_reliable()
    
    def test_security_configuration(self):
        """Test security settings."""
        origin = DataOrigin(
            name="secure_origin",
            origin_type=OriginType.DATABASE,
            security_settings={
                "security_level": "high",
                "requires_authentication": True,
                "encryption_at_rest": True
            }
        )
        
        assert origin.get_security_level() == "high"
        assert origin.requires_authentication() is True
        
        # Test defaults
        basic_origin = DataOrigin(
            name="basic_origin",
            origin_type=OriginType.DATABASE
        )
        
        assert basic_origin.get_security_level() == "standard"
        assert basic_origin.requires_authentication() is False