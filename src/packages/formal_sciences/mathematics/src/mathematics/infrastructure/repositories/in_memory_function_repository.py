"""In-memory repository implementation for mathematical functions."""

from typing import Dict, List, Optional

from ...domain.entities.math_function import MathFunction
from ...domain.repositories.function_repository import FunctionRepository
from ...domain.value_objects.function_value_objects import FunctionId


class InMemoryFunctionRepository(FunctionRepository):
    """In-memory implementation of function repository."""
    
    def __init__(self):
        self._functions: Dict[str, MathFunction] = {}
    
    async def save(self, function: MathFunction) -> MathFunction:
        """Save a mathematical function."""
        function_id_str = str(function.function_id)
        self._functions[function_id_str] = function
        return function
    
    async def get_by_id(self, function_id: FunctionId) -> Optional[MathFunction]:
        """Get a function by its ID."""
        return self._functions.get(str(function_id))
    
    async def get_by_name(self, name: str) -> List[MathFunction]:
        """Get functions by name."""
        return [
            func for func in self._functions.values()
            if func.metadata.name == name
        ]
    
    async def get_by_type(self, function_type: str) -> List[MathFunction]:
        """Get functions by type."""
        return [
            func for func in self._functions.values()
            if func.properties.function_type.value == function_type
        ]
    
    async def get_all(self) -> List[MathFunction]:
        """Get all functions."""
        return list(self._functions.values())
    
    async def delete(self, function_id: FunctionId) -> bool:
        """Delete a function."""
        function_id_str = str(function_id)
        if function_id_str in self._functions:
            del self._functions[function_id_str]
            return True
        return False
    
    async def exists(self, function_id: FunctionId) -> bool:
        """Check if a function exists."""
        return str(function_id) in self._functions
    
    async def update(self, function: MathFunction) -> MathFunction:
        """Update a function."""
        function_id_str = str(function.function_id)
        if function_id_str in self._functions:
            self._functions[function_id_str] = function
            return function
        else:
            raise ValueError(f"Function with ID {function_id_str} not found")
    
    async def search(self, query: str) -> List[MathFunction]:
        """Search functions by query."""
        query_lower = query.lower()
        results = []
        
        for func in self._functions.values():
            # Search in name, description, expression, and tags
            if (query_lower in func.metadata.name.lower() or
                query_lower in func.metadata.description.lower() or
                query_lower in func.expression.lower() or
                any(query_lower in tag.lower() for tag in func.metadata.tags)):
                results.append(func)
        
        return results
    
    def clear(self) -> None:
        """Clear all functions (for testing)."""
        self._functions.clear()
    
    def count(self) -> int:
        """Get count of functions."""
        return len(self._functions)