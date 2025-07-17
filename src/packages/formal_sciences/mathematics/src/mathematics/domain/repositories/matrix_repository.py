"""Repository interface for matrices."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..entities.matrix import Matrix
from ..value_objects.matrix_value_objects import MatrixId, MatrixType


class MatrixRepository(ABC):
    """Abstract repository for matrices."""
    
    @abstractmethod
    async def save(self, matrix: Matrix) -> Matrix:
        """Save a matrix."""
        pass
    
    @abstractmethod
    async def get_by_id(self, matrix_id: MatrixId) -> Optional[Matrix]:
        """Get a matrix by its ID."""
        pass
    
    @abstractmethod
    async def get_by_type(self, matrix_type: MatrixType) -> List[Matrix]:
        """Get matrices by type."""
        pass
    
    @abstractmethod
    async def get_by_shape(self, shape: Tuple[int, int]) -> List[Matrix]:
        """Get matrices by shape."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Matrix]:
        """Get all matrices."""
        pass
    
    @abstractmethod
    async def delete(self, matrix_id: MatrixId) -> bool:
        """Delete a matrix."""
        pass
    
    @abstractmethod
    async def exists(self, matrix_id: MatrixId) -> bool:
        """Check if a matrix exists."""
        pass
    
    @abstractmethod
    async def update(self, matrix: Matrix) -> Matrix:
        """Update a matrix."""
        pass
    
    @abstractmethod
    async def get_square_matrices(self) -> List[Matrix]:
        """Get all square matrices."""
        pass
    
    @abstractmethod
    async def get_invertible_matrices(self) -> List[Matrix]:
        """Get all invertible matrices."""
        pass