"""Data Source domain entity for data engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


class SourceType(str, Enum):
    """Type of data source."""
    
    DATABASE = "database"
    FILE = "file" 
    API = "api"
    STREAM = "stream"
    CLOUD_STORAGE = "cloud_storage"
    MESSAGE_QUEUE = "message_queue"
    FTP = "ftp"
    SFTP = "sftp"
    WEB_SCRAPING = "web_scraping"


class ConnectionStatus(str, Enum):
    """Connection status."""
    
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"
    UNKNOWN = "unknown"


@dataclass
class ConnectionConfig:
    """Connection configuration for data sources."""
    
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None  # Should be encrypted in production
    database: Optional[str] = None
    schema: Optional[str] = None
    connection_string: Optional[str] = None
    auth_token: Optional[str] = None
    api_key: Optional[str] = None
    ssl_enabled: bool = False
    timeout_seconds: int = 30
    pool_size: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate connection config."""
        if self.port is not None and (self.port < 1 or self.port > 65535):
            raise ValueError("Port must be between 1 and 65535")
        
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.pool_size <= 0:
            raise ValueError("Pool size must be positive")
    
    def mask_sensitive_data(self) -> Dict[str, Any]:
        """Return config with sensitive data masked."""
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "***" if self.password else None,
            "database": self.database,
            "schema": self.schema,
            "connection_string": "***" if self.connection_string else None,
            "auth_token": "***" if self.auth_token else None,
            "api_key": "***" if self.api_key else None,
            "ssl_enabled": self.ssl_enabled,
            "timeout_seconds": self.timeout_seconds,
            "pool_size": self.pool_size,
            "extra_params": self.extra_params,
        }


@dataclass
class DataSource:
    """Data source domain entity representing external data systems."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    source_type: SourceType = SourceType.DATABASE
    connection_config: ConnectionConfig = field(default_factory=ConnectionConfig)
    
    # Status and health
    status: ConnectionStatus = ConnectionStatus.UNKNOWN
    last_connection_test: Optional[datetime] = None
    last_successful_connection: Optional[datetime] = None
    connection_error: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = ""
    
    # Configuration
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    # Data catalog info
    schema_info: Dict[str, Any] = field(default_factory=dict)
    table_count: Optional[int] = None
    estimated_size_bytes: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate data source after initialization."""
        if not self.name:
            raise ValueError("Data source name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Data source name cannot exceed 100 characters")
        
        if not isinstance(self.source_type, SourceType):
            raise TypeError("source_type must be a SourceType enum")
    
    @property
    def is_connected(self) -> bool:
        """Check if data source is currently connected."""
        return self.status == ConnectionStatus.CONNECTED
    
    @property
    def has_connection_error(self) -> bool:
        """Check if data source has connection errors."""
        return self.status == ConnectionStatus.ERROR
    
    @property  
    def connection_age_hours(self) -> Optional[float]:
        """Get hours since last successful connection."""
        if not self.last_successful_connection:
            return None
        
        delta = datetime.utcnow() - self.last_successful_connection
        return delta.total_seconds() / 3600
    
    def test_connection(self) -> bool:
        """Test connection to data source."""
        self.status = ConnectionStatus.TESTING
        self.last_connection_test = datetime.utcnow()
        self.updated_at = self.last_connection_test
        
        # This would be implemented by infrastructure layer
        # For now, simulate a successful test
        success = True  # Would be actual connection test
        
        if success:
            self.status = ConnectionStatus.CONNECTED
            self.last_successful_connection = self.last_connection_test
            self.connection_error = None
        else:
            self.status = ConnectionStatus.ERROR
            self.connection_error = "Connection failed"
        
        return success
    
    def disconnect(self) -> None:
        """Disconnect from data source."""
        self.status = ConnectionStatus.DISCONNECTED
        self.updated_at = datetime.utcnow()
    
    def mark_connection_error(self, error_message: str) -> None:
        """Mark connection as having an error."""
        self.status = ConnectionStatus.ERROR
        self.connection_error = error_message
        self.updated_at = datetime.utcnow()
    
    def update_connection_config(self, config: ConnectionConfig) -> None:
        """Update connection configuration."""
        self.connection_config = config
        self.status = ConnectionStatus.UNKNOWN  # Reset status after config change
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the data source."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the data source."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if data source has a specific tag."""
        return tag in self.tags
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata key-value pair."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def update_schema_info(self, schema_info: Dict[str, Any]) -> None:
        """Update schema information."""
        self.schema_info = schema_info
        self.updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activate the data source."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the data source."""
        self.is_active = False
        self.status = ConnectionStatus.DISCONNECTED
        self.updated_at = datetime.utcnow()
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get connection status summary."""
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "status": self.status.value,
            "is_active": self.is_active,
            "last_connection_test": (
                self.last_connection_test.isoformat() 
                if self.last_connection_test else None
            ),
            "last_successful_connection": (
                self.last_successful_connection.isoformat()
                if self.last_successful_connection else None
            ),
            "connection_age_hours": self.connection_age_hours,
            "has_error": self.has_connection_error,
            "connection_error": self.connection_error,
        }
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert data source to dictionary."""
        connection_config = (
            self.connection_config.__dict__ if include_sensitive
            else self.connection_config.mask_sensitive_data()
        )
        
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type.value,
            "connection_config": connection_config,
            "status": self.status.value,
            "last_connection_test": (
                self.last_connection_test.isoformat() 
                if self.last_connection_test else None
            ),
            "last_successful_connection": (
                self.last_successful_connection.isoformat()
                if self.last_successful_connection else None
            ),
            "connection_error": self.connection_error,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "tags": self.tags,
            "schema_info": self.schema_info,
            "table_count": self.table_count,
            "estimated_size_bytes": self.estimated_size_bytes,
            "connection_age_hours": self.connection_age_hours,
            "is_connected": self.is_connected,
            "has_connection_error": self.has_connection_error,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"DataSource('{self.name}', {self.source_type.value}, status={self.status.value})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"DataSource(id={self.id}, name='{self.name}', "
            f"type={self.source_type.value}, status={self.status.value})"
        )