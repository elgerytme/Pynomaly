"""Base service abstraction."""

from abc import ABC


class BaseService(ABC):
    """Base service interface."""

    def __init__(self) -> None:
        """Initialize the service."""
        pass

    def get_service_name(self) -> str:
        """Get the name of the service."""
        return self.__class__.__name__
