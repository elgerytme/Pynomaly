"""Repository interface for mathematical functions."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.math_function import MathFunction
from ..value_objects.function_value_objects import FunctionId


class FunctionRepository(ABC):
    """Abstract repository for mathematical functions."""
    
    @abstractmethod
    async def save(self, function: MathFunction) -> MathFunction:
        """Save a mathematical function."""
        pass
    
    @abstractmethod
    async def get_by_id(self, function_id: FunctionId) -> Optional[MathFunction]:
        """Get a function by its ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> List[MathFunction]:
        """Get functions by name."""
        pass
    
    @abstractmethod
    async def get_by_type(self, function_type: str) -> List[MathFunction]:
        """Get functions by type."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[MathFunction]:
        """Get all functions."""
        pass
    
    @abstractmethod
    async def delete(self, function_id: FunctionId) -> bool:
        """Delete a function."""
        pass
    
    @abstractmethod
    async def exists(self, function_id: FunctionId) -> bool:
        """Check if a function exists."""
        pass
    
    @abstractmethod
    async def update(self, function: MathFunction) -> MathFunction:
        """Update a function."""
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[MathFunction]:
        """Search functions by query."""
        pass