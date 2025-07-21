"""Data type value objects."""

from enum import Enum
from typing import Optional, Any, Dict
from pydantic import Field, validator
from packages.core.domain.abstractions.base_value_object import BaseValueObject


class PrimitiveDataType(str, Enum):
    """Primitive data types."""
    STRING = "string"
    INTEGER = "integer" 
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    BINARY = "binary"
    JSON = "json"
    UUID = "uuid"


class DataType(BaseValueObject):
    """Represents a data type definition with constraints and metadata."""
    
    primitive_type: PrimitiveDataType = Field(..., description="Base primitive type")
    max_length: Optional[int] = Field(None, ge=0, description="Maximum length for string/binary types")
    precision: Optional[int] = Field(None, ge=1, description="Precision for numeric types")
    scale: Optional[int] = Field(None, ge=0, description="Scale for decimal types")
    format_pattern: Optional[str] = Field(None, description="Regex pattern for format validation")
    nullable: bool = Field(default=True, description="Whether the data can be null")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Additional constraints")
    
    @validator('scale')
    def validate_scale(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Validate that scale doesn't exceed precision."""
        if v is not None and values.get('precision') is not None:
            if v > values['precision']:
                raise ValueError("Scale cannot exceed precision")
        return v
    
    @validator('max_length')
    def validate_max_length_for_type(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        """Validate max_length is only used for appropriate types."""
        primitive_type = values.get('primitive_type')
        if v is not None and primitive_type not in [PrimitiveDataType.STRING, PrimitiveDataType.BINARY]:
            raise ValueError(f"max_length not applicable for type {primitive_type}")
        return v
    
    def is_numeric(self) -> bool:
        """Check if this is a numeric data type."""
        return self.primitive_type in [PrimitiveDataType.INTEGER, PrimitiveDataType.FLOAT]
    
    def is_temporal(self) -> bool:
        """Check if this is a temporal data type."""
        return self.primitive_type in [PrimitiveDataType.DATE, PrimitiveDataType.DATETIME, PrimitiveDataType.TIMESTAMP]
    
    def is_compatible_with(self, other_type: 'DataType') -> bool:
        """Check if this data type is compatible with another."""
        if self.primitive_type == other_type.primitive_type:
            return True
        
        # Numeric compatibility
        numeric_types = {PrimitiveDataType.INTEGER, PrimitiveDataType.FLOAT}
        if self.primitive_type in numeric_types and other_type.primitive_type in numeric_types:
            return True
            
        # Temporal compatibility
        temporal_types = {PrimitiveDataType.DATE, PrimitiveDataType.DATETIME, PrimitiveDataType.TIMESTAMP}
        if self.primitive_type in temporal_types and other_type.primitive_type in temporal_types:
            return True
            
        return False