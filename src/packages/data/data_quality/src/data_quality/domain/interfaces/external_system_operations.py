"""Domain interfaces for external system operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, BinaryIO
from pathlib import Path
from enum import Enum
import pandas as pd


class DataSourceType(Enum):
    """Data source types supported by the system."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    DATABASE = "database"
    API = "api"
    CLOUD_STORAGE = "cloud_storage"
    STREAM = "stream"


class NotificationChannel(Enum):
    """Notification channels supported by the system."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"


class ReportFormat(Enum):
    """Report formats supported by the system."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class DataSourceConfig:
    """Configuration for data source connections."""
    source_type: DataSourceType
    connection_params: Dict[str, Any]
    authentication: Optional[Dict[str, str]] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    channel: NotificationChannel
    recipients: List[str]
    template: Optional[str] = None
    priority: str = "normal"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    format: ReportFormat
    template: Optional[str] = None
    output_path: Optional[str] = None
    include_charts: bool = True
    include_summary: bool = True
    metadata: Optional[Dict[str, Any]] = None


class DataSourcePort(ABC):
    """Port for data source operations."""
    
    @abstractmethod
    async def connect_to_source(self, config: DataSourceConfig) -> str:
        """Connect to a data source and return connection identifier.
        
        Args:
            config: Data source configuration
            
        Returns:
            Connection identifier
        """
        pass
    
    @abstractmethod
    async def read_data(
        self, 
        connection_id: str, 
        query_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Read data from a connected source.
        
        Args:
            connection_id: Data source connection identifier
            query_params: Optional query parameters
            
        Returns:
            Data as pandas DataFrame
        """
        pass
    
    @abstractmethod
    async def write_data(
        self, 
        connection_id: str, 
        data: pd.DataFrame, 
        write_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write data to a connected source.
        
        Args:
            connection_id: Data source connection identifier
            data: Data to write
            write_params: Optional write parameters
            
        Returns:
            True if write successful
        """
        pass
    
    @abstractmethod
    async def get_schema(self, connection_id: str) -> Dict[str, Any]:
        """Get schema information from a data source.
        
        Args:
            connection_id: Data source connection identifier
            
        Returns:
            Schema information
        """
        pass
    
    @abstractmethod
    async def test_connection(self, config: DataSourceConfig) -> bool:
        """Test connection to a data source.
        
        Args:
            config: Data source configuration
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self, connection_id: str) -> bool:
        """Disconnect from a data source.
        
        Args:
            connection_id: Data source connection identifier
            
        Returns:
            True if disconnection successful
        """
        pass
    
    @abstractmethod
    async def list_available_sources(self) -> List[Dict[str, Any]]:
        """List available data sources.
        
        Returns:
            List of available data sources with metadata
        """
        pass


class FileSystemPort(ABC):
    """Port for file system operations."""
    
    @abstractmethod
    async def read_file(self, file_path: str) -> BinaryIO:
        """Read a file from the file system.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as binary stream
        """
        pass
    
    @abstractmethod
    async def write_file(self, file_path: str, content: BinaryIO) -> bool:
        """Write content to a file.
        
        Args:
            file_path: Path to write the file
            content: File content as binary stream
            
        Returns:
            True if write successful
        """
        pass
    
    @abstractmethod
    async def list_files(
        self, 
        directory_path: str, 
        pattern: Optional[str] = None
    ) -> List[str]:
        """List files in a directory.
        
        Args:
            directory_path: Directory to list
            pattern: Optional file pattern filter
            
        Returns:
            List of file paths
        """
        pass
    
    @abstractmethod
    async def create_directory(self, directory_path: str) -> bool:
        """Create a directory.
        
        Args:
            directory_path: Directory path to create
            
        Returns:
            True if creation successful
        """
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if deletion successful
        """
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata
        """
        pass


class NotificationPort(ABC):
    """Port for notification operations."""
    
    @abstractmethod
    async def send_notification(
        self, 
        config: NotificationConfig, 
        subject: str, 
        message: str, 
        attachments: Optional[List[str]] = None
    ) -> bool:
        """Send a notification.
        
        Args:
            config: Notification configuration
            subject: Notification subject
            message: Notification message
            attachments: Optional list of attachment file paths
            
        Returns:
            True if notification sent successfully
        """
        pass
    
    @abstractmethod
    async def send_alert(
        self, 
        config: NotificationConfig, 
        alert_level: str, 
        alert_message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send an alert notification.
        
        Args:
            config: Notification configuration
            alert_level: Alert severity level
            alert_message: Alert message
            metadata: Optional alert metadata
            
        Returns:
            True if alert sent successfully
        """
        pass
    
    @abstractmethod
    async def send_report_notification(
        self, 
        config: NotificationConfig, 
        report_path: str, 
        summary: str
    ) -> bool:
        """Send a report notification.
        
        Args:
            config: Notification configuration
            report_path: Path to the report file
            summary: Report summary
            
        Returns:
            True if notification sent successfully
        """
        pass
    
    @abstractmethod
    async def validate_notification_config(self, config: NotificationConfig) -> bool:
        """Validate notification configuration.
        
        Args:
            config: Notification configuration to validate
            
        Returns:
            True if configuration is valid
        """
        pass


class ReportingPort(ABC):
    """Port for reporting operations."""
    
    @abstractmethod
    async def generate_data_quality_report(
        self, 
        profile_data: Dict[str, Any], 
        config: ReportConfig
    ) -> str:
        """Generate a data quality report.
        
        Args:
            profile_data: Data profile information
            config: Report configuration
            
        Returns:
            Generated report file path
        """
        pass
    
    @abstractmethod
    async def generate_validation_report(
        self, 
        validation_results: List[Dict[str, Any]], 
        config: ReportConfig
    ) -> str:
        """Generate a validation results report.
        
        Args:
            validation_results: List of validation results
            config: Report configuration
            
        Returns:
            Generated report file path
        """
        pass
    
    @abstractmethod
    async def generate_trend_report(
        self, 
        historical_data: List[Dict[str, Any]], 
        config: ReportConfig
    ) -> str:
        """Generate a trend analysis report.
        
        Args:
            historical_data: Historical data quality metrics
            config: Report configuration
            
        Returns:
            Generated report file path
        """
        pass
    
    @abstractmethod
    async def create_dashboard(
        self, 
        dashboard_config: Dict[str, Any]
    ) -> str:
        """Create an interactive dashboard.
        
        Args:
            dashboard_config: Dashboard configuration
            
        Returns:
            Dashboard URL or identifier
        """
        pass
    
    @abstractmethod
    async def export_data(
        self, 
        data: pd.DataFrame, 
        export_config: Dict[str, Any]
    ) -> str:
        """Export data in specified format.
        
        Args:
            data: Data to export
            export_config: Export configuration
            
        Returns:
            Exported file path
        """
        pass


class MetadataPort(ABC):
    """Port for metadata operations."""
    
    @abstractmethod
    async def store_metadata(
        self, 
        entity_type: str, 
        entity_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Store metadata for an entity.
        
        Args:
            entity_type: Type of entity (profile, check, etc.)
            entity_id: Unique identifier for the entity
            metadata: Metadata to store
            
        Returns:
            True if storage successful
        """
        pass
    
    @abstractmethod
    async def retrieve_metadata(
        self, 
        entity_type: str, 
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Unique identifier for the entity
            
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def update_metadata(
        self, 
        entity_type: str, 
        entity_id: str, 
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """Update metadata for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Unique identifier for the entity
            metadata_updates: Metadata updates to apply
            
        Returns:
            True if update successful
        """
        pass
    
    @abstractmethod
    async def delete_metadata(
        self, 
        entity_type: str, 
        entity_id: str
    ) -> bool:
        """Delete metadata for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Unique identifier for the entity
            
        Returns:
            True if deletion successful
        """
        pass
    
    @abstractmethod
    async def search_metadata(
        self, 
        entity_type: str, 
        search_criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search metadata based on criteria.
        
        Args:
            entity_type: Type of entity to search
            search_criteria: Search criteria
            
        Returns:
            List of matching metadata entries
        """
        pass


class CloudStoragePort(ABC):
    """Port for cloud storage operations."""
    
    @abstractmethod
    async def upload_file(
        self, 
        local_path: str, 
        remote_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload a file to cloud storage.
        
        Args:
            local_path: Local file path
            remote_path: Remote storage path
            metadata: Optional file metadata
            
        Returns:
            True if upload successful
        """
        pass
    
    @abstractmethod
    async def download_file(
        self, 
        remote_path: str, 
        local_path: str
    ) -> bool:
        """Download a file from cloud storage.
        
        Args:
            remote_path: Remote storage path
            local_path: Local file path
            
        Returns:
            True if download successful
        """
        pass
    
    @abstractmethod
    async def list_objects(
        self, 
        prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List objects in cloud storage.
        
        Args:
            prefix: Optional prefix filter
            
        Returns:
            List of object information
        """
        pass
    
    @abstractmethod
    async def delete_object(self, remote_path: str) -> bool:
        """Delete an object from cloud storage.
        
        Args:
            remote_path: Remote storage path
            
        Returns:
            True if deletion successful
        """
        pass
    
    @abstractmethod
    async def get_object_metadata(self, remote_path: str) -> Dict[str, Any]:
        """Get metadata for a cloud storage object.
        
        Args:
            remote_path: Remote storage path
            
        Returns:
            Object metadata
        """
        pass