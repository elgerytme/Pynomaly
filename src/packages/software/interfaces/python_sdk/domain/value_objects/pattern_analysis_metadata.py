"""
Processing Metadata Value Object

Represents immutable metadata associated with pattern analysis processing operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass(frozen=True)
class PatternAnalysisMetadata:
    """
    Immutable metadata for pattern analysis processing operations.
    
    This value object contains contextual information about
    the processing request and execution environment.
    """
    
    data_collection_name: Optional[str] = None
    feature_names: Optional[list] = None
    data_source: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: str = "production"
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError("Environment must be 'development', 'staging', or 'production'")
            
        if self.feature_names is not None and not isinstance(self.feature_names, list):
            raise ValueError("Feature names must be a list")
            
        if self.tags is not None and not isinstance(self.tags, dict):
            raise ValueError("Tags must be a dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the value object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "data_collection_name": self.data_collection_name,
            "feature_names": self.feature_names,
            "data_source": self.data_source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "environment": self.environment,
            "tags": self.tags.copy() if self.tags else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternAnalysisMetadata':
        """
        Create PatternAnalysisMetadata from a dictionary.
        
        Args:
            data: Dictionary containing metadata.
            
        Returns:
            PatternAnalysisMetadata: New instance from the dictionary.
        """
        return cls(
            data_collection_name=data.get("data_collection_name"),
            feature_names=data.get("feature_names"),
            data_source=data.get("data_source"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            environment=data.get("environment", "production"),
            tags=data.get("tags")
        )
    
    def with_user(self, user_id: str) -> 'PatternAnalysisMetadata':
        """
        Create new metadata with user ID.
        
        Args:
            user_id: User identifier.
            
        Returns:
            PatternAnalysisMetadata: New instance with user ID.
        """
        return PatternAnalysisMetadata(
            data_collection_name=self.data_collection_name,
            feature_names=self.feature_names,
            data_source=self.data_source,
            user_id=user_id,
            session_id=self.session_id,
            environment=self.environment,
            tags=self.tags
        )
    
    def with_tags(self, **new_tags: str) -> 'PatternAnalysisMetadata':
        """
        Create new metadata with additional tags.
        
        Args:
            **new_tags: Additional tags to include.
            
        Returns:
            PatternAnalysisMetadata: New instance with updated tags.
        """
        updated_tags = (self.tags or {}).copy()
        updated_tags.update(new_tags)
        
        return PatternAnalysisMetadata(
            data_collection_name=self.data_collection_name,
            feature_names=self.feature_names,
            data_source=self.data_source,
            user_id=self.user_id,
            session_id=self.session_id,
            environment=self.environment,
            tags=updated_tags
        )