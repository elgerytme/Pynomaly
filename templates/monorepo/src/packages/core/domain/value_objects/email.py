"""Email value object with validation."""

import re
from typing import Any

from pydantic import BaseModel, validator


class Email(BaseModel):
    """Email value object with validation."""
    
    value: str
    
    @validator('value')
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        if not v:
            raise ValueError('Email cannot be empty')
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        
        # Normalize email
        return v.lower().strip()
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        """Compare emails."""
        if isinstance(other, Email):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash(self.value)
    
    @property
    def domain(self) -> str:
        """Get email domain."""
        return self.value.split('@')[1]
    
    @property
    def local_part(self) -> str:
        """Get local part of email."""
        return self.value.split('@')[0]
    
    class Config:
        """Pydantic configuration."""
        frozen = True