"""
Identifier value objects for the MLOps Marketplace.

Defines strongly-typed identifiers for different entities in the system,
providing type safety and validation.
"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class BaseId(BaseModel):
    """Base class for all identifier value objects."""
    
    value: UUID = Field(default_factory=uuid4)
    
    def __init__(self, value: Any = None, **kwargs):
        """Initialize with value or generate new UUID."""
        if value is None:
            super().__init__(**kwargs)
        elif isinstance(value, str):
            super().__init__(value=UUID(value), **kwargs)
        elif isinstance(value, UUID):
            super().__init__(value=value, **kwargs)
        else:
            super().__init__(value=value, **kwargs)
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.value)
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if isinstance(other, BaseId):
            return self.value == other.value
        if isinstance(other, (str, UUID)):
            return self.value == UUID(str(other))
        return False
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.value)
    
    class Config:
        """Pydantic configuration."""
        frozen = True
        json_encoders = {UUID: str}


class SolutionId(BaseId):
    """Unique identifier for solutions."""
    
    @classmethod
    def generate(cls) -> 'SolutionId':
        """Generate a new solution ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'SolutionId':
        """Create from string representation."""
        return cls(value=value)


class ProviderId(BaseId):
    """Unique identifier for solution providers."""
    
    @classmethod
    def generate(cls) -> 'ProviderId':
        """Generate a new provider ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'ProviderId':
        """Create from string representation."""
        return cls(value=value)


class UserId(BaseId):
    """Unique identifier for marketplace users."""
    
    @classmethod
    def generate(cls) -> 'UserId':
        """Generate a new user ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'UserId':
        """Create from string representation."""
        return cls(value=value)


class SubscriptionId(BaseId):
    """Unique identifier for subscriptions."""
    
    @classmethod
    def generate(cls) -> 'SubscriptionId':
        """Generate a new subscription ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'SubscriptionId':
        """Create from string representation."""
        return cls(value=value)


class TransactionId(BaseId):
    """Unique identifier for transactions."""
    
    @classmethod
    def generate(cls) -> 'TransactionId':
        """Generate a new transaction ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'TransactionId':
        """Create from string representation."""
        return cls(value=value)


class ReviewId(BaseId):
    """Unique identifier for reviews."""
    
    @classmethod
    def generate(cls) -> 'ReviewId':
        """Generate a new review ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'ReviewId':
        """Create from string representation."""
        return cls(value=value)


class CertificationId(BaseId):
    """Unique identifier for certifications."""
    
    @classmethod
    def generate(cls) -> 'CertificationId':
        """Generate a new certification ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'CertificationId':
        """Create from string representation."""
        return cls(value=value)


class DeploymentId(BaseId):
    """Unique identifier for deployments."""
    
    @classmethod
    def generate(cls) -> 'DeploymentId':
        """Generate a new deployment ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'DeploymentId':
        """Create from string representation."""
        return cls(value=value)


class ApiKeyId(BaseId):
    """Unique identifier for API keys."""
    
    @classmethod
    def generate(cls) -> 'ApiKeyId':
        """Generate a new API key ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'ApiKeyId':
        """Create from string representation."""
        return cls(value=value)


class SessionId(BaseId):
    """Unique identifier for user sessions."""
    
    @classmethod
    def generate(cls) -> 'SessionId':
        """Generate a new session ID."""
        return cls()
    
    @classmethod
    def from_string(cls, value: str) -> 'SessionId':
        """Create from string representation."""
        return cls(value=value)