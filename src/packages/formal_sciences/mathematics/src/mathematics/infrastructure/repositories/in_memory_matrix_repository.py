"""In-memory repository implementation for matrices."""

from typing import Dict, List, Optional, Tuple

from ...domain.entities.matrix import Matrix
from ...domain.repositories.matrix_repository import MatrixRepository
from ...domain.value_objects.matrix_value_objects import MatrixId, MatrixType


class InMemoryMatrixRepository(MatrixRepository):
    """In-memory implementation of matrix repository."""
    
    def __init__(self):
        self._matrices: Dict[str, Matrix] = {}
    
    async def save(self, matrix: Matrix) -> Matrix:
        """Save a matrix."""
        matrix_id_str = str(matrix.matrix_id)
        self._matrices[matrix_id_str] = matrix
        return matrix
    
    async def get_by_id(self, matrix_id: MatrixId) -> Optional[Matrix]:
        """Get a matrix by its ID."""
        return self._matrices.get(str(matrix_id))
    
    async def get_by_type(self, matrix_type: MatrixType) -> List[Matrix]:
        """Get matrices by type."""
        return [
            matrix for matrix in self._matrices.values()
            if matrix.properties.matrix_type == matrix_type
        ]
    
    async def get_by_shape(self, shape: Tuple[int, int]) -> List[Matrix]:
        """Get matrices by shape."""
        return [
            matrix for matrix in self._matrices.values()
            if matrix.shape == shape
        ]
    
    async def get_all(self) -> List[Matrix]:
        """Get all matrices."""
        return list(self._matrices.values())
    
    async def delete(self, matrix_id: MatrixId) -> bool:
        """Delete a matrix."""
        matrix_id_str = str(matrix_id)
        if matrix_id_str in self._matrices:
            del self._matrices[matrix_id_str]
            return True
        return False
    
    async def exists(self, matrix_id: MatrixId) -> bool:
        """Check if a matrix exists."""
        return str(matrix_id) in self._matrices
    
    async def update(self, matrix: Matrix) -> Matrix:
        """Update a matrix."""
        matrix_id_str = str(matrix.matrix_id)
        if matrix_id_str in self._matrices:
            self._matrices[matrix_id_str] = matrix
            return matrix
        else:
            raise ValueError(f"Matrix with ID {matrix_id_str} not found")
    
    async def get_square_matrices(self) -> List[Matrix]:
        """Get all square matrices."""
        return [
            matrix for matrix in self._matrices.values()
            if matrix.is_square
        ]
    
    async def get_invertible_matrices(self) -> List[Matrix]:
        """Get all invertible matrices."""
        return [
            matrix for matrix in self._matrices.values()
            if matrix.properties.is_invertible
        ]
    
    async def get_by_property(self, property_name: str, value: bool) -> List[Matrix]:
        """Get matrices by property."""
        results = []
        for matrix in self._matrices.values():
            if hasattr(matrix.properties, property_name):
                if getattr(matrix.properties, property_name) == value:
                    results.append(matrix)
        return results
    
    async def get_symmetric_matrices(self) -> List[Matrix]:
        """Get all symmetric matrices."""
        return await self.get_by_property("is_symmetric", True)
    
    async def get_positive_definite_matrices(self) -> List[Matrix]:
        """Get all positive definite matrices."""
        return await self.get_by_property("is_positive_definite", True)
    
    async def get_orthogonal_matrices(self) -> List[Matrix]:
        """Get all orthogonal matrices."""
        return await self.get_by_property("is_orthogonal", True)
    
    def clear(self) -> None:
        """Clear all matrices (for testing)."""
        self._matrices.clear()
    
    def count(self) -> int:
        """Get count of matrices."""
        return len(self._matrices)