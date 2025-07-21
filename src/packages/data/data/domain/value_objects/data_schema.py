"""Data schema value objects."""

from typing import Dict, List, Optional, Any
from pydantic import Field, validator
from packages.core.domain.abstractions.base_value_object import BaseValueObject
from .data_type import DataType


class DataFieldSchema(BaseValueObject):
    """Schema definition for a single data field."""
    
    field_name: str = Field(..., min_length=1, max_length=200, description="Field name")
    data_type: DataType = Field(..., description="Data type definition")
    description: Optional[str] = Field(None, description="Field description")
    is_required: bool = Field(default=True, description="Whether field is required")
    is_primary_key: bool = Field(default=False, description="Whether field is primary key")
    is_foreign_key: bool = Field(default=False, description="Whether field is foreign key")
    foreign_key_reference: Optional[str] = Field(None, description="Foreign key reference")
    default_value: Optional[Any] = Field(None, description="Default value")
    enum_values: Optional[List[str]] = Field(None, description="Valid enum values")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('field_name')
    def validate_field_name(cls, v: str) -> str:
        """Validate field name format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Field name must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @validator('foreign_key_reference')
    def validate_foreign_key_reference(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate foreign key reference is provided when is_foreign_key is True."""
        if values.get('is_foreign_key') and not v:
            raise ValueError("foreign_key_reference is required when is_foreign_key is True")
        return v
    
    def is_key_field(self) -> bool:
        """Check if this is a key field (primary or foreign)."""
        return self.is_primary_key or self.is_foreign_key


class DataSchema(BaseValueObject):
    """Complete schema definition for a data structure."""
    
    schema_name: str = Field(..., min_length=1, max_length=200, description="Schema name")
    version: str = Field(default="1.0.0", description="Schema version")
    fields: Dict[str, DataFieldSchema] = Field(..., description="Field definitions")
    description: Optional[str] = Field(None, description="Schema description")
    primary_keys: List[str] = Field(default_factory=list, description="Primary key field names")
    foreign_keys: Dict[str, str] = Field(default_factory=dict, description="Foreign key mappings")
    indexes: List[List[str]] = Field(default_factory=list, description="Index definitions")
    constraints: List[str] = Field(default_factory=list, description="Schema constraints")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('fields')
    def validate_fields_not_empty(cls, v: Dict[str, DataFieldSchema]) -> Dict[str, DataFieldSchema]:
        """Validate that schema has at least one field."""
        if not v:
            raise ValueError("Schema must have at least one field")
        return v
    
    @validator('primary_keys')
    def validate_primary_keys_exist(cls, v: List[str], values: Dict[str, Any]) -> List[str]:
        """Validate primary keys reference existing fields."""
        fields = values.get('fields', {})
        for key in v:
            if key not in fields:
                raise ValueError(f"Primary key '{key}' not found in fields")
        return v
    
    @validator('foreign_keys')
    def validate_foreign_keys_exist(cls, v: Dict[str, str], values: Dict[str, Any]) -> Dict[str, str]:
        """Validate foreign keys reference existing fields."""
        fields = values.get('fields', {})
        for key in v.keys():
            if key not in fields:
                raise ValueError(f"Foreign key '{key}' not found in fields")
        return v
    
    def get_field(self, field_name: str) -> Optional[DataFieldSchema]:
        """Get field schema by name."""
        return self.fields.get(field_name)
    
    def get_required_fields(self) -> List[DataFieldSchema]:
        """Get all required fields."""
        return [field for field in self.fields.values() if field.is_required]
    
    def get_optional_fields(self) -> List[DataFieldSchema]:
        """Get all optional fields."""
        return [field for field in self.fields.values() if not field.is_required]
    
    def get_key_fields(self) -> List[DataFieldSchema]:
        """Get all key fields (primary and foreign keys)."""
        return [field for field in self.fields.values() if field.is_key_field()]
    
    def is_compatible_with(self, other_schema: 'DataSchema') -> bool:
        """Check if this schema is compatible with another schema."""
        # Check if all required fields in other schema exist and are compatible
        for field_name, other_field in other_schema.fields.items():
            if other_field.is_required:
                our_field = self.fields.get(field_name)
                if not our_field:
                    return False
                if not our_field.data_type.is_compatible_with(other_field.data_type):
                    return False
        return True