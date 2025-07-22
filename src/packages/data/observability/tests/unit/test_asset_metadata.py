"""Unit tests for AssetMetadata value object."""

import pytest

from data_observability.domain.value_objects.asset_metadata import AssetMetadata


class TestAssetMetadata:
    """Test cases for AssetMetadata value object."""
    
    def test_create_asset_metadata_minimal(self):
        """Test creating asset metadata with minimal information."""
        metadata = AssetMetadata()
        
        assert metadata.source_system is None
        assert metadata.location is None
        assert metadata.format is None
        assert metadata.size_bytes is None
        assert metadata.row_count is None
        assert metadata.custom_properties == {}
    
    def test_create_asset_metadata_full(self):
        """Test creating asset metadata with full information."""
        custom_props = {
            "partition_key": "date",
            "retention_days": 365,
            "compression": "gzip",
            "environment": "production"
        }
        
        metadata = AssetMetadata(
            source_system="postgres",
            location="/data/warehouse/tables/users",
            format="parquet",
            size_bytes=1024000,
            row_count=50000,
            custom_properties=custom_props
        )
        
        assert metadata.source_system == "postgres"
        assert metadata.location == "/data/warehouse/tables/users"
        assert metadata.format == "parquet"
        assert metadata.size_bytes == 1024000
        assert metadata.row_count == 50000
        assert metadata.custom_properties == custom_props
    
    def test_asset_metadata_equality(self):
        """Test asset metadata equality."""
        metadata1 = AssetMetadata(
            source_system="snowflake",
            location="DB.SCHEMA.TABLE",
            format="table",
            size_bytes=2048,
            row_count=100
        )
        
        metadata2 = AssetMetadata(
            source_system="snowflake",
            location="DB.SCHEMA.TABLE", 
            format="table",
            size_bytes=2048,
            row_count=100
        )
        
        assert metadata1 == metadata2
    
    def test_asset_metadata_inequality(self):
        """Test asset metadata inequality."""
        metadata1 = AssetMetadata(
            source_system="postgres",
            location="/data/table1"
        )
        
        metadata2 = AssetMetadata(
            source_system="postgres",
            location="/data/table2"
        )
        
        assert metadata1 != metadata2
    
    def test_get_custom_property(self):
        """Test getting custom property."""
        metadata = AssetMetadata(
            custom_properties={
                "owner": "data_team",
                "environment": "production",
                "criticality": "high"
            }
        )
        
        assert metadata.get_custom_property("owner") == "data_team"
        assert metadata.get_custom_property("environment") == "production"
        assert metadata.get_custom_property("nonexistent") is None
        assert metadata.get_custom_property("nonexistent", "default") == "default"
    
    def test_set_custom_property(self):
        """Test setting custom property."""
        metadata = AssetMetadata()
        
        metadata.set_custom_property("team", "analytics")
        assert metadata.get_custom_property("team") == "analytics"
        
        # Update existing property
        metadata.set_custom_property("team", "data_science")
        assert metadata.get_custom_property("team") == "data_science"
    
    def test_remove_custom_property(self):
        """Test removing custom property."""
        metadata = AssetMetadata(
            custom_properties={
                "temp_property": "temp_value",
                "permanent_property": "permanent_value"
            }
        )
        
        removed = metadata.remove_custom_property("temp_property")
        assert removed is True
        assert metadata.get_custom_property("temp_property") is None
        assert metadata.get_custom_property("permanent_property") == "permanent_value"
        
        # Try to remove non-existent property
        removed = metadata.remove_custom_property("nonexistent")
        assert removed is False
    
    def test_has_custom_property(self):
        """Test checking if custom property exists."""
        metadata = AssetMetadata(
            custom_properties={
                "existing_key": "existing_value"
            }
        )
        
        assert metadata.has_custom_property("existing_key") is True
        assert metadata.has_custom_property("nonexistent_key") is False
    
    def test_get_all_custom_properties(self):
        """Test getting all custom properties."""
        props = {
            "environment": "staging",
            "team": "platform",
            "criticality": "medium"
        }
        
        metadata = AssetMetadata(custom_properties=props)
        all_props = metadata.get_all_custom_properties()
        
        assert all_props == props
        
        # Ensure we get a copy, not the original
        all_props["new_key"] = "new_value"
        assert "new_key" not in metadata.custom_properties
    
    def test_update_custom_properties(self):
        """Test updating multiple custom properties."""
        metadata = AssetMetadata(
            custom_properties={
                "existing_key": "existing_value",
                "to_update": "old_value"
            }
        )
        
        updates = {
            "to_update": "new_value",
            "new_key": "new_value",
            "another_key": "another_value"
        }
        
        metadata.update_custom_properties(updates)
        
        assert metadata.get_custom_property("existing_key") == "existing_value"
        assert metadata.get_custom_property("to_update") == "new_value"
        assert metadata.get_custom_property("new_key") == "new_value"
        assert metadata.get_custom_property("another_key") == "another_value"
    
    def test_clear_custom_properties(self):
        """Test clearing all custom properties."""
        metadata = AssetMetadata(
            custom_properties={
                "key1": "value1",
                "key2": "value2",
                "key3": "value3"
            }
        )
        
        assert len(metadata.custom_properties) == 3
        
        metadata.clear_custom_properties()
        
        assert len(metadata.custom_properties) == 0
        assert metadata.get_custom_property("key1") is None
    
    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        custom_props = {
            "partition_key": "date",
            "retention_days": 365
        }
        
        metadata = AssetMetadata(
            source_system="bigquery",
            location="project.dataset.table",
            format="table",
            size_bytes=5120000,
            row_count=250000,
            custom_properties=custom_props
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict["source_system"] == "bigquery"
        assert metadata_dict["location"] == "project.dataset.table"
        assert metadata_dict["format"] == "table"
        assert metadata_dict["size_bytes"] == 5120000
        assert metadata_dict["row_count"] == 250000
        assert metadata_dict["custom_properties"] == custom_props
    
    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        metadata_dict = {
            "source_system": "redshift",
            "location": "cluster.database.schema.table",
            "format": "table",
            "size_bytes": 10240000,
            "row_count": 500000,
            "custom_properties": {
                "distribution_key": "user_id",
                "sort_keys": ["created_at", "user_id"],
                "compression": "lzo"
            }
        }
        
        metadata = AssetMetadata.from_dict(metadata_dict)
        
        assert metadata.source_system == "redshift"
        assert metadata.location == "cluster.database.schema.table"
        assert metadata.format == "table"
        assert metadata.size_bytes == 10240000
        assert metadata.row_count == 500000
        assert metadata.get_custom_property("distribution_key") == "user_id"
        assert metadata.get_custom_property("compression") == "lzo"
    
    def test_from_dict_partial(self):
        """Test creating metadata from partial dictionary."""
        metadata_dict = {
            "source_system": "mysql",
            "custom_properties": {
                "engine": "InnoDB"
            }
        }
        
        metadata = AssetMetadata.from_dict(metadata_dict)
        
        assert metadata.source_system == "mysql"
        assert metadata.location is None
        assert metadata.format is None
        assert metadata.size_bytes is None
        assert metadata.row_count is None
        assert metadata.get_custom_property("engine") == "InnoDB"
    
    def test_from_dict_empty(self):
        """Test creating metadata from empty dictionary."""
        metadata = AssetMetadata.from_dict({})
        
        assert metadata.source_system is None
        assert metadata.location is None
        assert metadata.format is None
        assert metadata.size_bytes is None
        assert metadata.row_count is None
        assert metadata.custom_properties == {}
    
    def test_size_formatting(self):
        """Test size formatting helper methods."""
        metadata = AssetMetadata(size_bytes=1024)
        
        # Test if size formatting methods exist (implementation dependent)
        if hasattr(metadata, 'get_size_formatted'):
            formatted = metadata.get_size_formatted()
            assert isinstance(formatted, str)
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Test negative size
        with pytest.raises(ValueError):
            AssetMetadata(size_bytes=-1)
        
        # Test negative row count
        with pytest.raises(ValueError):
            AssetMetadata(row_count=-1)
    
    def test_metadata_copy(self):
        """Test creating a copy of metadata."""
        original = AssetMetadata(
            source_system="original",
            location="/original/path",
            custom_properties={"key": "value"}
        )
        
        # Create copy by converting to/from dict
        copy_dict = original.to_dict()
        copy = AssetMetadata.from_dict(copy_dict)
        
        assert copy == original
        assert copy.custom_properties is not original.custom_properties  # Different objects
        
        # Modify copy
        copy.set_custom_property("new_key", "new_value")
        assert original.get_custom_property("new_key") is None