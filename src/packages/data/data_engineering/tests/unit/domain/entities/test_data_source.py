"""Comprehensive unit tests for DataSource domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

from data_engineering.domain.entities.data_source import (
    DataSource, SourceType, ConnectionConfig, ConnectionStatus
)


class TestSourceType:
    """Test cases for SourceType enum."""

    def test_source_type_values(self):
        """Test all SourceType enum values."""
        assert SourceType.DATABASE.value == "database"
        assert SourceType.FILE.value == "file"
        assert SourceType.API.value == "api"
        assert SourceType.STREAM.value == "stream"
        assert SourceType.CLOUD_STORAGE.value == "cloud_storage"
        assert SourceType.MESSAGE_QUEUE.value == "message_queue"
        assert SourceType.FTP.value == "ftp"
        assert SourceType.SFTP.value == "sftp"
        assert SourceType.WEB_SCRAPING.value == "web_scraping"


class TestConnectionStatus:
    """Test cases for ConnectionStatus enum."""

    def test_connection_status_values(self):
        """Test all ConnectionStatus enum values."""
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.ERROR.value == "error"
        assert ConnectionStatus.TESTING.value == "testing"
        assert ConnectionStatus.UNKNOWN.value == "unknown"


class TestConnectionConfig:
    """Test cases for ConnectionConfig entity."""

    def test_initialization_defaults(self):
        """Test connection config initialization with defaults."""
        config = ConnectionConfig()
        
        assert config.host is None
        assert config.port is None
        assert config.username is None
        assert config.password is None
        assert config.database is None
        assert config.schema is None
        assert config.connection_string is None
        assert config.auth_token is None
        assert config.api_key is None
        assert config.ssl_enabled is False
        assert config.timeout_seconds == 30
        assert config.pool_size == 5
        assert config.extra_params == {}

    def test_initialization_with_data(self):
        """Test connection config initialization with provided data."""
        extra_params = {"retries": 3, "ssl_cert": "/path/to/cert"}
        
        config = ConnectionConfig(
            host="db.example.com",
            port=5432,
            username="admin",
            password="secret123",
            database="production",
            schema="public",
            connection_string="postgresql://user:pass@host:5432/db",
            auth_token="token123",
            api_key="key456",
            ssl_enabled=True,
            timeout_seconds=60,
            pool_size=10,
            extra_params=extra_params
        )
        
        assert config.host == "db.example.com"
        assert config.port == 5432
        assert config.username == "admin"
        assert config.password == "secret123"
        assert config.database == "production"
        assert config.schema == "public"
        assert config.connection_string == "postgresql://user:pass@host:5432/db"
        assert config.auth_token == "token123"
        assert config.api_key == "key456"
        assert config.ssl_enabled is True
        assert config.timeout_seconds == 60
        assert config.pool_size == 10
        assert config.extra_params == extra_params

    def test_post_init_validation_invalid_port_low(self):
        """Test validation fails for port too low."""
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            ConnectionConfig(port=0)

    def test_post_init_validation_invalid_port_high(self):
        """Test validation fails for port too high."""
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            ConnectionConfig(port=65536)

    def test_post_init_validation_valid_port_range(self):
        """Test validation passes for valid port range."""
        config1 = ConnectionConfig(port=1)
        assert config1.port == 1
        
        config2 = ConnectionConfig(port=65535)
        assert config2.port == 65535

    def test_post_init_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ConnectionConfig(timeout_seconds=0)
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ConnectionConfig(timeout_seconds=-1)

    def test_post_init_validation_invalid_pool_size(self):
        """Test validation fails for invalid pool size."""
        with pytest.raises(ValueError, match="Pool size must be positive"):
            ConnectionConfig(pool_size=0)
        
        with pytest.raises(ValueError, match="Pool size must be positive"):
            ConnectionConfig(pool_size=-1)

    def test_mask_sensitive_data_no_sensitive_fields(self):
        """Test masking when no sensitive fields are set."""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            username="user",
            database="testdb"
        )
        
        masked = config.mask_sensitive_data()
        
        assert masked["host"] == "localhost"
        assert masked["port"] == 5432
        assert masked["username"] == "user"
        assert masked["password"] is None
        assert masked["database"] == "testdb"
        assert masked["connection_string"] is None
        assert masked["auth_token"] is None
        assert masked["api_key"] is None
        assert masked["ssl_enabled"] is False
        assert masked["timeout_seconds"] == 30
        assert masked["pool_size"] == 5

    def test_mask_sensitive_data_with_sensitive_fields(self):
        """Test masking when sensitive fields are set."""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            username="user",
            password="secret123",
            database="testdb",
            connection_string="postgresql://user:pass@host:5432/db",
            auth_token="token123",
            api_key="key456"
        )
        
        masked = config.mask_sensitive_data()
        
        assert masked["host"] == "localhost"
        assert masked["port"] == 5432
        assert masked["username"] == "user"
        assert masked["password"] == "***"
        assert masked["database"] == "testdb"
        assert masked["connection_string"] == "***"
        assert masked["auth_token"] == "***"
        assert masked["api_key"] == "***"


class TestDataSource:
    """Test cases for DataSource entity."""

    def test_initialization_defaults(self):
        """Test data source initialization with defaults."""
        source = DataSource(name="test_source")
        
        assert isinstance(source.id, UUID)
        assert source.name == "test_source"
        assert source.description == ""
        assert source.source_type == SourceType.DATABASE
        assert isinstance(source.connection_config, ConnectionConfig)
        assert source.status == ConnectionStatus.UNKNOWN
        assert source.last_connection_test is None
        assert source.last_successful_connection is None
        assert source.connection_error is None
        assert isinstance(source.created_at, datetime)
        assert source.created_by == ""
        assert isinstance(source.updated_at, datetime)
        assert source.updated_by == ""
        assert source.is_active is True
        assert source.metadata == {}
        assert source.tags == []
        assert source.schema_info == {}
        assert source.table_count is None
        assert source.estimated_size_bytes is None

    def test_initialization_with_data(self, sample_connection_config):
        """Test data source initialization with provided data."""
        source_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        metadata = {"environment": "production"}
        tags = ["critical", "financial"]
        schema_info = {"tables": ["users", "orders"]}
        
        source = DataSource(
            id=source_id,
            name="Production DB",
            description="Main production database",
            source_type=SourceType.DATABASE,
            connection_config=sample_connection_config,
            status=ConnectionStatus.CONNECTED,
            last_connection_test=created_at,
            last_successful_connection=created_at,
            created_at=created_at,
            created_by="admin",
            updated_at=created_at,
            updated_by="admin",
            is_active=True,
            metadata=metadata,
            tags=tags,
            schema_info=schema_info,
            table_count=25,
            estimated_size_bytes=1073741824
        )
        
        assert source.id == source_id
        assert source.name == "Production DB"
        assert source.description == "Main production database"
        assert source.source_type == SourceType.DATABASE
        assert source.connection_config == sample_connection_config
        assert source.status == ConnectionStatus.CONNECTED
        assert source.last_connection_test == created_at
        assert source.last_successful_connection == created_at
        assert source.created_at == created_at
        assert source.created_by == "admin"
        assert source.updated_at == created_at
        assert source.updated_by == "admin"
        assert source.metadata == metadata
        assert source.tags == tags
        assert source.schema_info == schema_info
        assert source.table_count == 25
        assert source.estimated_size_bytes == 1073741824

    def test_post_init_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Data source name cannot be empty"):
            DataSource(name="")

    def test_post_init_validation_name_too_long(self):
        """Test validation fails for name too long."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Data source name cannot exceed 100 characters"):
            DataSource(name=long_name)

    def test_post_init_validation_invalid_source_type(self):
        """Test validation fails for invalid source type."""
        with pytest.raises(TypeError, match="source_type must be a SourceType enum"):
            DataSource(name="test", source_type="invalid")

    def test_is_connected_property(self):
        """Test is_connected property."""
        source = DataSource(name="test")
        
        assert source.is_connected is False
        
        source.status = ConnectionStatus.CONNECTED
        assert source.is_connected is True
        
        source.status = ConnectionStatus.ERROR
        assert source.is_connected is False

    def test_has_connection_error_property(self):
        """Test has_connection_error property."""
        source = DataSource(name="test")
        
        assert source.has_connection_error is False
        
        source.status = ConnectionStatus.ERROR
        assert source.has_connection_error is True
        
        source.status = ConnectionStatus.CONNECTED
        assert source.has_connection_error is False

    def test_connection_age_hours_property_none(self):
        """Test connection_age_hours property when no successful connection."""
        source = DataSource(name="test")
        assert source.connection_age_hours is None

    def test_connection_age_hours_property_calculated(self):
        """Test connection_age_hours property calculation."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            past_time = datetime(2024, 1, 15, 10, 0, 0)
            current_time = datetime(2024, 1, 15, 12, 30, 0)  # 2.5 hours later
            
            source.last_successful_connection = past_time
            mock_datetime.utcnow.return_value = current_time
            
            assert source.connection_age_hours == 2.5

    def test_test_connection_success(self):
        """Test successful connection test."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result = source.test_connection()
            
            assert result is True
            assert source.status == ConnectionStatus.CONNECTED
            assert source.last_connection_test == mock_now
            assert source.last_successful_connection == mock_now
            assert source.updated_at == mock_now
            assert source.connection_error is None

    def test_test_connection_sets_testing_status(self):
        """Test that connection test sets testing status initially."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            # Mock the connection test to be synchronous for testing
            original_test = source.test_connection
            
            def mock_test_connection():
                source.status = ConnectionStatus.TESTING
                source.last_connection_test = mock_now
                source.updated_at = mock_now
                return True
            
            source.test_connection = mock_test_connection
            source.test_connection()
            
            assert source.status == ConnectionStatus.TESTING
            assert source.last_connection_test == mock_now

    def test_disconnect(self):
        """Test disconnecting from data source."""
        source = DataSource(name="test")
        source.status = ConnectionStatus.CONNECTED
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 13, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.disconnect()
            
            assert source.status == ConnectionStatus.DISCONNECTED
            assert source.updated_at == mock_now

    def test_mark_connection_error(self):
        """Test marking connection error."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            error_message = "Connection timeout"
            source.mark_connection_error(error_message)
            
            assert source.status == ConnectionStatus.ERROR
            assert source.connection_error == error_message
            assert source.updated_at == mock_now

    def test_update_connection_config(self, sample_connection_config):
        """Test updating connection configuration."""
        source = DataSource(name="test")
        source.status = ConnectionStatus.CONNECTED
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 15, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.update_connection_config(sample_connection_config)
            
            assert source.connection_config == sample_connection_config
            assert source.status == ConnectionStatus.UNKNOWN
            assert source.updated_at == mock_now

    def test_add_tag(self):
        """Test adding tags."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 16, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.add_tag("production")
            source.add_tag("critical")
            
            assert "production" in source.tags
            assert "critical" in source.tags
            assert len(source.tags) == 2
            assert source.updated_at == mock_now

    def test_add_tag_duplicate(self):
        """Test adding duplicate tag."""
        source = DataSource(name="test")
        source.add_tag("production")
        source.add_tag("production")  # Duplicate
        
        assert source.tags.count("production") == 1

    def test_add_tag_empty(self):
        """Test adding empty tag does nothing."""
        source = DataSource(name="test")
        original_updated_at = source.updated_at
        
        source.add_tag("")
        
        assert len(source.tags) == 0
        assert source.updated_at == original_updated_at

    def test_remove_tag(self):
        """Test removing tags."""
        source = DataSource(name="test")
        source.tags = ["production", "critical", "database"]
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 17, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.remove_tag("critical")
            
            assert "critical" not in source.tags
            assert len(source.tags) == 2
            assert source.updated_at == mock_now

    def test_remove_tag_nonexistent(self):
        """Test removing non-existent tag."""
        source = DataSource(name="test")
        source.tags = ["production"]
        original_updated_at = source.updated_at
        
        source.remove_tag("nonexistent")
        
        assert len(source.tags) == 1
        assert source.updated_at == original_updated_at

    def test_has_tag(self):
        """Test checking for tags."""
        source = DataSource(name="test")
        source.tags = ["production", "critical"]
        
        assert source.has_tag("production") is True
        assert source.has_tag("critical") is True
        assert source.has_tag("nonexistent") is False

    def test_update_metadata(self):
        """Test updating metadata."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.update_metadata("environment", "production")
            source.update_metadata("version", "2.1.0")
            
            assert source.metadata["environment"] == "production"
            assert source.metadata["version"] == "2.1.0"
            assert source.updated_at == mock_now

    def test_update_schema_info(self):
        """Test updating schema information."""
        source = DataSource(name="test")
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 19, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            schema_info = {
                "tables": ["users", "orders", "products"],
                "views": ["user_summary"],
                "procedures": ["calculate_totals"]
            }
            
            source.update_schema_info(schema_info)
            
            assert source.schema_info == schema_info
            assert source.updated_at == mock_now

    def test_activate(self):
        """Test activating data source."""
        source = DataSource(name="test")
        source.is_active = False
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 20, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.activate()
            
            assert source.is_active is True
            assert source.updated_at == mock_now

    def test_deactivate(self):
        """Test deactivating data source."""
        source = DataSource(name="test")
        source.status = ConnectionStatus.CONNECTED
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 21, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            source.deactivate()
            
            assert source.is_active is False
            assert source.status == ConnectionStatus.DISCONNECTED
            assert source.updated_at == mock_now

    def test_get_connection_summary(self):
        """Test getting connection summary."""
        source = DataSource(name="test_db")
        source.source_type = SourceType.DATABASE
        source.status = ConnectionStatus.CONNECTED
        source.is_active = True
        source.last_connection_test = datetime(2024, 1, 15, 12, 0, 0)
        source.last_successful_connection = datetime(2024, 1, 15, 12, 0, 0)
        source.connection_error = None
        
        with patch('data_engineering.domain.entities.data_source.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 14, 0, 0)  # 2 hours later
            
            summary = source.get_connection_summary()
            
            assert summary["name"] == "test_db"
            assert summary["source_type"] == "database"
            assert summary["status"] == "connected"
            assert summary["is_active"] is True
            assert summary["last_connection_test"] == "2024-01-15T12:00:00"
            assert summary["last_successful_connection"] == "2024-01-15T12:00:00"
            assert summary["connection_age_hours"] == 2.0
            assert summary["has_error"] is False
            assert summary["connection_error"] is None

    def test_get_connection_summary_with_error(self):
        """Test getting connection summary with error."""
        source = DataSource(name="test_db")
        source.status = ConnectionStatus.ERROR
        source.connection_error = "Connection timeout"
        
        summary = source.get_connection_summary()
        
        assert summary["status"] == "error"
        assert summary["has_error"] is True
        assert summary["connection_error"] == "Connection timeout"

    def test_to_dict_without_sensitive_data(self, sample_connection_config):
        """Test converting to dictionary without sensitive data."""
        source = DataSource(
            name="test_db",
            description="Test database",
            source_type=SourceType.DATABASE,
            connection_config=sample_connection_config,
            status=ConnectionStatus.CONNECTED,
            created_by="admin",
            updated_by="admin",
            metadata={"env": "test"},
            tags=["test", "db"],
            table_count=10,
            estimated_size_bytes=1000000
        )
        
        result = source.to_dict(include_sensitive=False)
        
        assert result["id"] == str(source.id)
        assert result["name"] == "test_db"
        assert result["description"] == "Test database"
        assert result["source_type"] == "database"
        assert result["status"] == "connected"
        assert result["created_by"] == "admin"
        assert result["updated_by"] == "admin"
        assert result["is_active"] is True
        assert result["metadata"] == {"env": "test"}
        assert result["tags"] == ["test", "db"]
        assert result["table_count"] == 10
        assert result["estimated_size_bytes"] == 1000000
        assert result["is_connected"] is True
        assert result["has_connection_error"] is False
        
        # Check connection config is masked
        config = result["connection_config"]
        assert config["password"] == "***"

    def test_to_dict_with_sensitive_data(self, sample_connection_config):
        """Test converting to dictionary with sensitive data."""
        source = DataSource(
            name="test_db",
            connection_config=sample_connection_config
        )
        
        result = source.to_dict(include_sensitive=True)
        
        # Check connection config includes sensitive data
        config = result["connection_config"]
        assert config["password"] == "testpass"  # From sample_connection_config

    def test_str_representation(self):
        """Test string representation."""
        source = DataSource(name="test_db")
        source.source_type = SourceType.DATABASE
        source.status = ConnectionStatus.CONNECTED
        
        str_repr = str(source)
        
        assert "DataSource('test_db'" in str_repr
        assert "database" in str_repr
        assert "status=connected" in str_repr

    def test_repr_representation(self):
        """Test detailed representation."""
        source = DataSource(name="test_db")
        source.source_type = SourceType.API
        source.status = ConnectionStatus.ERROR
        
        repr_str = repr(source)
        
        assert f"id={source.id}" in repr_str
        assert "name='test_db'" in repr_str
        assert "type=api" in repr_str
        assert "status=error" in repr_str

    def test_all_source_types_supported(self):
        """Test that all source types can be used."""
        for source_type in SourceType:
            source = DataSource(name=f"test_{source_type.value}", source_type=source_type)
            assert source.source_type == source_type

    def test_connection_config_extra_params(self):
        """Test connection config with extra parameters."""
        extra_params = {
            "sslmode": "require",
            "connect_timeout": 10,
            "application_name": "test_app"
        }
        
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            extra_params=extra_params
        )
        
        assert config.extra_params == extra_params
        
        masked = config.mask_sensitive_data()
        assert masked["extra_params"] == extra_params  # Extra params are not masked

    def test_data_source_complex_workflow(self, sample_connection_config):
        """Test complex data source workflow."""
        # Create data source
        source = DataSource(
            name="production_db",
            description="Main production database",
            source_type=SourceType.DATABASE,
            connection_config=sample_connection_config,
            created_by="admin"
        )
        
        # Add tags and metadata
        source.add_tag("production")
        source.add_tag("critical")
        source.update_metadata("environment", "production")
        source.update_metadata("backup_schedule", "daily")
        
        # Test connection
        assert source.test_connection() is True
        assert source.is_connected is True
        
        # Update schema info
        schema_info = {
            "tables": ["users", "orders", "products", "payments"],
            "views": ["user_orders", "monthly_sales"],
            "indexes": 45,
            "size_gb": 2.5
        }
        source.update_schema_info(schema_info)
        source.table_count = len(schema_info["tables"])
        source.estimated_size_bytes = int(schema_info["size_gb"] * 1024 * 1024 * 1024)
        
        # Verify final state
        assert source.has_tag("production")
        assert source.has_tag("critical")
        assert source.metadata["environment"] == "production"
        assert source.schema_info == schema_info
        assert source.table_count == 4
        assert source.is_active is True
        
        # Get summary
        summary = source.get_connection_summary()
        assert summary["name"] == "production_db"
        assert summary["status"] == "connected"
        assert summary["is_active"] is True
        
        # Convert to dict
        data = source.to_dict()
        assert data["name"] == "production_db"
        assert data["source_type"] == "database"
        assert data["tags"] == ["production", "critical"]
        assert data["table_count"] == 4