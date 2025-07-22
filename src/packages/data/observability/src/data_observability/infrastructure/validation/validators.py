"""Input validation utilities for data observability."""

import re
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from ..errors.exceptions import ValidationError


class AssetValidation:
    """Validation utilities for data assets."""
    
    @staticmethod
    def validate_asset_name(name: str) -> str:
        """Validate asset name."""
        if not name or not name.strip():
            raise ValidationError("Asset name cannot be empty", field="name", value=name)
        
        name = name.strip()
        
        # Check length
        if len(name) < 2:
            raise ValidationError("Asset name must be at least 2 characters", field="name", value=name)
        
        if len(name) > 255:
            raise ValidationError("Asset name cannot exceed 255 characters", field="name", value=name)
        
        # Check for valid characters (alphanumeric, underscores, hyphens, dots)
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", name):
            raise ValidationError(
                "Asset name can only contain letters, numbers, underscores, hyphens, and dots",
                field="name",
                value=name
            )
        
        # Cannot start or end with special characters
        if name.startswith(("_", "-", ".")) or name.endswith(("_", "-", ".")):
            raise ValidationError(
                "Asset name cannot start or end with special characters",
                field="name",
                value=name
            )
        
        return name
    
    @staticmethod
    def validate_asset_type(asset_type: str) -> str:
        """Validate asset type."""
        if not asset_type or not asset_type.strip():
            raise ValidationError("Asset type cannot be empty", field="asset_type", value=asset_type)
        
        asset_type = asset_type.strip().lower()
        
        valid_types = {
            "table", "view", "materialized_view", "dataset", "collection",
            "file", "stream", "topic", "queue", "api", "model", "report"
        }
        
        if asset_type not in valid_types:
            raise ValidationError(
                f"Invalid asset type. Must be one of: {', '.join(sorted(valid_types))}",
                field="asset_type",
                value=asset_type
            )
        
        return asset_type
    
    @staticmethod
    def validate_tags(tags: List[str]) -> List[str]:
        """Validate asset tags."""
        if not tags:
            return []
        
        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationError(f"Tag must be a string, got {type(tag)}", field="tags", value=tag)
            
            tag = tag.strip().lower()
            
            if not tag:
                continue  # Skip empty tags
            
            if len(tag) > 50:
                raise ValidationError("Tag cannot exceed 50 characters", field="tags", value=tag)
            
            if not re.match(r"^[a-zA-Z0-9_\-]+$", tag):
                raise ValidationError(
                    "Tag can only contain letters, numbers, underscores, and hyphens",
                    field="tags",
                    value=tag
                )
            
            if tag not in validated_tags:
                validated_tags.append(tag)
        
        return validated_tags
    
    @staticmethod
    def validate_owner(owner: Optional[str]) -> Optional[str]:
        """Validate asset owner."""
        if not owner:
            return None
        
        owner = owner.strip()
        
        if len(owner) > 100:
            raise ValidationError("Owner name cannot exceed 100 characters", field="owner", value=owner)
        
        # Basic email or username validation
        if "@" in owner:
            # Email validation
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, owner):
                raise ValidationError("Invalid email format for owner", field="owner", value=owner)
        else:
            # Username validation
            if not re.match(r"^[a-zA-Z0-9_\-\.]+$", owner):
                raise ValidationError(
                    "Owner username can only contain letters, numbers, underscores, hyphens, and dots",
                    field="owner",
                    value=owner
                )
        
        return owner


class SearchValidation:
    """Validation utilities for search operations."""
    
    @staticmethod
    def validate_query(query: Optional[str]) -> Optional[str]:
        """Validate search query."""
        if not query:
            return None
        
        query = query.strip()
        
        if len(query) < 2:
            raise ValidationError("Search query must be at least 2 characters", field="query", value=query)
        
        if len(query) > 200:
            raise ValidationError("Search query cannot exceed 200 characters", field="query", value=query)
        
        return query
    
    @staticmethod
    def validate_limit(limit: Optional[int]) -> int:
        """Validate search limit."""
        if limit is None:
            return 100  # Default limit
        
        if not isinstance(limit, int):
            raise ValidationError("Limit must be an integer", field="limit", value=limit)
        
        if limit < 1:
            raise ValidationError("Limit must be at least 1", field="limit", value=limit)
        
        if limit > 1000:
            raise ValidationError("Limit cannot exceed 1000", field="limit", value=limit)
        
        return limit
    
    @staticmethod
    def validate_offset(offset: Optional[int]) -> int:
        """Validate search offset."""
        if offset is None:
            return 0  # Default offset
        
        if not isinstance(offset, int):
            raise ValidationError("Offset must be an integer", field="offset", value=offset)
        
        if offset < 0:
            raise ValidationError("Offset cannot be negative", field="offset", value=offset)
        
        return offset


class IDValidation:
    """Validation utilities for IDs."""
    
    @staticmethod
    def validate_uuid(value: Union[str, UUID], field_name: str = "id") -> str:
        """Validate UUID format."""
        if isinstance(value, UUID):
            return str(value)
        
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string or UUID", field=field_name, value=value)
        
        try:
            # This will raise ValueError if invalid
            UUID(value)
            return value
        except ValueError:
            raise ValidationError(f"Invalid UUID format for {field_name}", field=field_name, value=value)
    
    @staticmethod
    def validate_pipeline_id(pipeline_id: str) -> str:
        """Validate pipeline ID format."""
        if not pipeline_id or not pipeline_id.strip():
            raise ValidationError("Pipeline ID cannot be empty", field="pipeline_id", value=pipeline_id)
        
        pipeline_id = pipeline_id.strip()
        
        if len(pipeline_id) > 100:
            raise ValidationError("Pipeline ID cannot exceed 100 characters", field="pipeline_id", value=pipeline_id)
        
        # Allow alphanumeric, underscores, hyphens, dots, and colons for namespace support
        if not re.match(r"^[a-zA-Z0-9_\-\.:]+$", pipeline_id):
            raise ValidationError(
                "Pipeline ID can only contain letters, numbers, underscores, hyphens, dots, and colons",
                field="pipeline_id",
                value=pipeline_id
            )
        
        return pipeline_id


class MetricValidation:
    """Validation utilities for metrics."""
    
    @staticmethod
    def validate_score(score: Optional[float], field_name: str = "score") -> Optional[float]:
        """Validate score value (0.0 to 1.0)."""
        if score is None:
            return None
        
        if not isinstance(score, (int, float)):
            raise ValidationError(f"{field_name} must be a number", field=field_name, value=score)
        
        score = float(score)
        
        if score < 0.0 or score > 1.0:
            raise ValidationError(f"{field_name} must be between 0.0 and 1.0", field=field_name, value=score)
        
        return score
    
    @staticmethod
    def validate_count(count: Optional[int], field_name: str = "count") -> Optional[int]:
        """Validate count value (non-negative integer)."""
        if count is None:
            return None
        
        if not isinstance(count, int):
            raise ValidationError(f"{field_name} must be an integer", field=field_name, value=count)
        
        if count < 0:
            raise ValidationError(f"{field_name} cannot be negative", field=field_name, value=count)
        
        return count
    
    @staticmethod
    def validate_size(size: Optional[int], field_name: str = "size") -> Optional[int]:
        """Validate size value (non-negative integer)."""
        if size is None:
            return None
        
        if not isinstance(size, int):
            raise ValidationError(f"{field_name} must be an integer", field=field_name, value=size)
        
        if size < 0:
            raise ValidationError(f"{field_name} cannot be negative", field=field_name, value=size)
        
        return size


# Pydantic models for request validation
class AssetCreateRequest(BaseModel):
    """Request model for creating assets."""
    
    name: str = Field(..., min_length=2, max_length=255)
    asset_type: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default_factory=list)
    owner: Optional[str] = Field(None, max_length=100)
    
    @validator("name")
    def validate_name(cls, v):
        return AssetValidation.validate_asset_name(v)
    
    @validator("asset_type")
    def validate_type(cls, v):
        return AssetValidation.validate_asset_type(v)
    
    @validator("tags")
    def validate_tags(cls, v):
        return AssetValidation.validate_tags(v)
    
    @validator("owner")
    def validate_owner(cls, v):
        return AssetValidation.validate_owner(v)


class SearchRequest(BaseModel):
    """Request model for search operations."""
    
    query: Optional[str] = Field(None, min_length=2, max_length=200)
    asset_type: Optional[str] = Field(None, max_length=50)
    tags: List[str] = Field(default_factory=list)
    owner: Optional[str] = Field(None, max_length=100)
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    
    @validator("query")
    def validate_query(cls, v):
        return SearchValidation.validate_query(v) if v else v
    
    @validator("asset_type")
    def validate_asset_type(cls, v):
        return AssetValidation.validate_asset_type(v) if v else v
    
    @validator("tags")
    def validate_tags(cls, v):
        return AssetValidation.validate_tags(v)


class PipelineHealthRequest(BaseModel):
    """Request model for pipeline health operations."""
    
    pipeline_id: str = Field(..., min_length=1, max_length=100)
    pipeline_name: Optional[str] = Field(None, max_length=255)
    
    @validator("pipeline_id")
    def validate_pipeline_id(cls, v):
        return IDValidation.validate_pipeline_id(v)