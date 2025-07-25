"""
Data connector interface for ecosystem data exchange.

This module provides interfaces and utilities for standardized
data exchange between platform components and external systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class DataFormat(Enum):
    """Supported data formats for exchange."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    XML = "xml"
    YAML = "yaml"
    BINARY = "binary"
    ARROW = "arrow"
    DELTA = "delta"


class DataTransferMode(Enum):
    """Data transfer modes."""
    BATCH = "batch"
    STREAMING = "streaming"
    REAL_TIME = "real_time"
    MICRO_BATCH = "micro_batch"
    CHANGE_DATA_CAPTURE = "cdc"


class CompressionType(Enum):
    """Compression types for data transfer."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


@dataclass
class DataSchema:
    """Schema definition for data structures."""
    
    # Schema identification
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Schema definition
    fields: List[Dict[str, Any]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    namespace: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Schema name is required")
        if not self.fields:
            raise ValueError("Schema must have at least one field")
    
    def get_field_names(self) -> List[str]:
        """Get list of field names."""
        return [field.get("name", "") for field in self.fields]
    
    def get_field_types(self) -> Dict[str, str]:
        """Get mapping of field names to types."""
        return {
            field.get("name", ""): field.get("type", "")
            for field in self.fields
        }
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check required fields
            field_names = self.get_field_names()
            for field_info in self.fields:
                field_name = field_info.get("name", "")
                if field_info.get("required", False) and field_name not in data:
                    logger.warning(f"Required field '{field_name}' missing from data")
                    return False
            
            # Check data types (basic validation)
            field_types = self.get_field_types()
            for field_name, value in data.items():
                if field_name in field_types:
                    expected_type = field_types[field_name]
                    if not self._validate_field_type(value, expected_type):
                        logger.warning(
                            f"Field '{field_name}' type mismatch",
                            expected=expected_type,
                            actual=type(value).__name__
                        )
                        return False
            
            return True
            
        except Exception as e:
            logger.error("Schema validation failed", error=str(e))
            return False
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type (simplified)."""
        type_mapping = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        if expected_type not in type_mapping:
            return True  # Unknown type, skip validation
        
        return isinstance(value, type_mapping[expected_type])


@dataclass
class DataValidationResult:
    """Result of data validation."""
    
    is_valid: bool
    schema_name: str
    schema_version: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


@dataclass
class DataTransferOptions:
    """Options for data transfer operations."""
    
    # Format options
    format_type: DataFormat = DataFormat.JSON
    compression: CompressionType = CompressionType.NONE
    
    # Transfer options
    mode: DataTransferMode = DataTransferMode.BATCH
    batch_size: int = 1000
    timeout_seconds: int = 300
    
    # Retry options
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Validation options
    validate_schema: bool = True
    strict_validation: bool = False
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


class DataConnectorInterface(ABC):
    """
    Abstract interface for data connectors.
    
    This interface defines the contract for data exchange operations
    between the platform and external data systems.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize data connector."""
        self.name = name
        self.config = config
        self.id = uuid4()
        self.logger = logger.bind(
            connector=name,
            connector_id=str(self.id)
        )
        
        self.logger.info("Data connector initialized")
    
    # Connection management
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to data source/destination.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection to data source/destination.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test the connection.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        pass
    
    # Schema management
    
    @abstractmethod
    async def get_schema(self, source: str) -> Optional[DataSchema]:
        """
        Get schema for data source.
        
        Args:
            source: Data source identifier
            
        Returns:
            DataSchema: Schema definition, None if not available
        """
        pass
    
    @abstractmethod
    async def register_schema(
        self,
        schema: DataSchema,
        source: str
    ) -> bool:
        """
        Register schema for data source.
        
        Args:
            schema: Schema to register
            source: Data source identifier
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def validate_data_against_schema(
        self,
        data: Any,
        schema: DataSchema
    ) -> DataValidationResult:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            DataValidationResult: Validation result
        """
        pass
    
    # Data reading operations
    
    @abstractmethod
    async def read_data(
        self,
        source: str,
        options: Optional[DataTransferOptions] = None
    ) -> Optional[Any]:
        """
        Read data from source.
        
        Args:
            source: Data source identifier
            options: Transfer options
            
        Returns:
            Any: Data from source, None if no data available
        """
        pass
    
    @abstractmethod
    async def read_batch(
        self,
        source: str,
        batch_size: int = 1000,
        offset: int = 0,
        options: Optional[DataTransferOptions] = None
    ) -> Optional[Any]:
        """
        Read batch of data from source.
        
        Args:
            source: Data source identifier
            batch_size: Number of records per batch
            offset: Starting offset
            options: Transfer options
            
        Returns:
            Any: Batch data, None if no data available
        """
        pass
    
    @abstractmethod
    async def read_stream(
        self,
        source: str,
        options: Optional[DataTransferOptions] = None
    ) -> AsyncIterator[Any]:
        """
        Read streaming data from source.
        
        Args:
            source: Data source identifier
            options: Transfer options
            
        Yields:
            Any: Streaming data items
        """
        pass
    
    # Data writing operations
    
    @abstractmethod
    async def write_data(
        self,
        data: Any,
        destination: str,
        options: Optional[DataTransferOptions] = None
    ) -> bool:
        """
        Write data to destination.
        
        Args:
            data: Data to write
            destination: Data destination identifier
            options: Transfer options
            
        Returns:
            bool: True if write successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def write_batch(
        self,
        data_batch: List[Any],
        destination: str,
        options: Optional[DataTransferOptions] = None
    ) -> bool:
        """
        Write batch of data to destination.
        
        Args:
            data_batch: Batch of data to write
            destination: Data destination identifier
            options: Transfer options
            
        Returns:
            bool: True if write successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def write_stream(
        self,
        data_stream: AsyncIterator[Any],
        destination: str,
        options: Optional[DataTransferOptions] = None
    ) -> bool:
        """
        Write streaming data to destination.
        
        Args:
            data_stream: Streaming data to write
            destination: Data destination identifier
            options: Transfer options
            
        Returns:
            bool: True if write successful, False otherwise
        """
        pass
    
    # Data transformation operations
    
    @abstractmethod
    async def transform_data(
        self,
        data: Any,
        transformation_config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Transform data using specified configuration.
        
        Args:
            data: Data to transform
            transformation_config: Transformation configuration
            
        Returns:
            Any: Transformed data, None if transformation failed
        """
        pass
    
    # Metadata operations
    
    @abstractmethod
    async def get_metadata(self, source: str) -> Dict[str, Any]:
        """
        Get metadata for data source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Dict[str, Any]: Metadata information
        """
        pass
    
    @abstractmethod
    async def list_sources(self) -> List[str]:
        """
        List available data sources.
        
        Returns:
            List[str]: Available data source identifiers
        """
        pass
    
    @abstractmethod
    async def get_source_info(self, source: str) -> Dict[str, Any]:
        """
        Get detailed information about data source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Dict[str, Any]: Source information
        """
        pass
    
    # Utility methods
    
    async def get_supported_formats(self) -> List[DataFormat]:
        """
        Get supported data formats.
        
        Returns:
            List[DataFormat]: Supported formats
        """
        # Default implementation - override in subclasses
        return [DataFormat.JSON, DataFormat.CSV]
    
    async def get_supported_modes(self) -> List[DataTransferMode]:
        """
        Get supported transfer modes.
        
        Returns:
            List[DataTransferMode]: Supported modes
        """
        # Default implementation - override in subclasses
        return [DataTransferMode.BATCH]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DataConnector(name={self.name}, id={self.id})"