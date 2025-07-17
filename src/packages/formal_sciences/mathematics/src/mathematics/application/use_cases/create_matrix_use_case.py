"""Use case for creating matrices."""

from typing import List
import numpy as np

from ...domain.entities.matrix import Matrix
from ...domain.repositories.matrix_repository import MatrixRepository
from ...domain.value_objects.matrix_value_objects import MatrixType
from ..dto.matrix_dto import MatrixDTO


class CreateMatrixUseCase:
    """Use case for creating matrices."""
    
    def __init__(self, matrix_repository: MatrixRepository):
        self._matrix_repository = matrix_repository
    
    async def execute(self, matrix_dto: MatrixDTO) -> MatrixDTO:
        """
        Create a new matrix.
        
        Args:
            matrix_dto: Matrix data transfer object
            
        Returns:
            Created matrix as DTO
        """
        # Validate input
        if not matrix_dto.data:
            raise ValueError("Matrix data is required")
        
        # Convert to numpy array
        data = np.array(matrix_dto.data)
        
        # Determine matrix type
        matrix_type = MatrixType(matrix_dto.matrix_type) if matrix_dto.matrix_type else MatrixType.GENERAL
        
        # Create matrix using factory method
        matrix = Matrix.from_array(data, matrix_type)
        
        # Update metadata if provided
        if matrix_dto.metadata:
            matrix = matrix.update_metadata(**matrix_dto.metadata)
        
        # Save to repository
        saved_matrix = await self._matrix_repository.save(matrix)
        
        # Convert back to DTO
        return self._to_dto(saved_matrix)
    
    def _to_dto(self, matrix: Matrix) -> MatrixDTO:
        """Convert domain entity to DTO."""
        return MatrixDTO(
            id=str(matrix.matrix_id),
            data=matrix.data.tolist(),
            shape=matrix.shape,
            matrix_type=matrix.properties.matrix_type.value,
            is_square=matrix.is_square,
            is_symmetric=matrix.properties.is_symmetric,
            is_invertible=matrix.properties.is_invertible,
            rank=matrix.properties.rank,
            determinant=matrix.properties.determinant,
            trace=matrix.properties.trace,
            condition_number=matrix.properties.condition_number,
            created_at=matrix.created_at,
            last_modified=matrix.last_modified,
            metadata=matrix.metadata,
        )