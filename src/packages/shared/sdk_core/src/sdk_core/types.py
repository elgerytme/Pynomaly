"""Type definitions for SDK."""

from typing import Any, Dict, List, Union, TypeVar, Protocol, runtime_checkable
from enum import Enum


# Type aliases
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONDict = Dict[str, JSONValue]
Headers = Dict[str, str]
QueryParams = Dict[str, Union[str, int, float, bool]]

# Generic type variables
T = TypeVar("T")
ResponseT = TypeVar("ResponseT")


class HTTPMethod(Enum):
    """HTTP methods."""
    
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(Enum):
    """Authentication types."""
    
    JWT = "jwt"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"


class LogLevel(Enum):
    """Logging levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create object from dictionary."""
        ...


@runtime_checkable
class AsyncContextManager(Protocol[T]):
    """Protocol for async context managers."""
    
    async def __aenter__(self) -> T:
        ...
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        ...


@runtime_checkable
class SyncContextManager(Protocol[T]):
    """Protocol for sync context managers."""
    
    def __enter__(self) -> T:
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...