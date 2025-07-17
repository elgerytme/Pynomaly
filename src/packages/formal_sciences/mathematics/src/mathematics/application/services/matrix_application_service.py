"""Application service for matrix operations."""

from typing import List, Optional, Tuple

from ...domain.repositories.matrix_repository import MatrixRepository
from ...domain.services.matrix_operations_service import MatrixOperationsService
from ...domain.value_objects.matrix_value_objects import MatrixId, MatrixType
from ..dto.matrix_dto import MatrixDTO, MatrixOperationDTO, MatrixOperationResponseDTO
from ..use_cases.create_matrix_use_case import CreateMatrixUseCase
from ..use_cases.matrix_operations_use_case import MatrixOperationsUseCase


class MatrixApplicationService:
    """Application service for matrix operations."""
    
    def __init__(self, matrix_repository: MatrixRepository,
                 matrix_operations_service: MatrixOperationsService):
        self._matrix_repository = matrix_repository
        self._matrix_operations_service = matrix_operations_service
        self._create_matrix_use_case = CreateMatrixUseCase(matrix_repository)
        self._matrix_operations_use_case = MatrixOperationsUseCase(
            matrix_repository, matrix_operations_service
        )
    
    async def create_matrix(self, matrix_dto: MatrixDTO) -> MatrixDTO:
        """Create a new matrix."""
        return await self._create_matrix_use_case.execute(matrix_dto)
    
    async def execute_operation(self, operation_dto: MatrixOperationDTO) -> MatrixOperationResponseDTO:
        """Execute matrix operation."""
        return await self._matrix_operations_use_case.execute(operation_dto)
    
    async def get_matrix_by_id(self, matrix_id: str) -> Optional[MatrixDTO]:
        """Get matrix by ID."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        return self._to_dto(matrix) if matrix else None
    
    async def get_matrices_by_type(self, matrix_type: str) -> List[MatrixDTO]:
        """Get matrices by type."""
        matrices = await self._matrix_repository.get_by_type(MatrixType(matrix_type))
        return [self._to_dto(m) for m in matrices]
    
    async def get_matrices_by_shape(self, shape: Tuple[int, int]) -> List[MatrixDTO]:
        """Get matrices by shape."""
        matrices = await self._matrix_repository.get_by_shape(shape)
        return [self._to_dto(m) for m in matrices]
    
    async def get_square_matrices(self) -> List[MatrixDTO]:
        """Get all square matrices."""
        matrices = await self._matrix_repository.get_square_matrices()
        return [self._to_dto(m) for m in matrices]
    
    async def get_invertible_matrices(self) -> List[MatrixDTO]:
        """Get all invertible matrices."""
        matrices = await self._matrix_repository.get_invertible_matrices()
        return [self._to_dto(m) for m in matrices]
    
    async def delete_matrix(self, matrix_id: str) -> bool:
        """Delete a matrix."""
        return await self._matrix_repository.delete(MatrixId(matrix_id))
    
    async def solve_linear_system(self, a_id: str, b_id: str) -> MatrixDTO:
        """Solve linear system Ax = b."""
        matrix_a = await self._matrix_repository.get_by_id(MatrixId(a_id))
        matrix_b = await self._matrix_repository.get_by_id(MatrixId(b_id))
        
        if matrix_a is None:
            raise ValueError(f"Matrix with ID {a_id} not found")
        if matrix_b is None:
            raise ValueError(f"Matrix with ID {b_id} not found")
        
        solution = self._matrix_operations_service.solve_linear_system(matrix_a, matrix_b)
        saved_solution = await self._matrix_repository.save(solution)
        
        return self._to_dto(saved_solution)
    
    async def compute_eigenvalues(self, matrix_id: str) -> Tuple[List[complex], List[List[complex]]]:
        """Compute eigenvalues and eigenvectors."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        eigenvals, eigenvecs = self._matrix_operations_service.compute_eigenvalues(matrix)
        
        return eigenvals.tolist(), eigenvecs.tolist()
    
    async def compute_determinant(self, matrix_id: str) -> complex:
        """Compute matrix determinant."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        return matrix.determinant()
    
    async def compute_trace(self, matrix_id: str) -> complex:
        """Compute matrix trace."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        return matrix.trace()
    
    async def compute_rank(self, matrix_id: str) -> int:
        """Compute matrix rank."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        return matrix.rank()
    
    async def compute_condition_number(self, matrix_id: str, norm_type: str = "2") -> float:
        """Compute condition number."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        return self._matrix_operations_service.compute_condition_number(matrix, norm_type)
    
    async def lu_decomposition(self, matrix_id: str) -> dict:
        """Compute LU decomposition."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        decomposition = matrix.lu_decomposition()
        
        return {
            "P": decomposition.factors["P"].tolist(),
            "L": decomposition.factors["L"].tolist(),
            "U": decomposition.factors["U"].tolist(),
            "computation_time": decomposition.computation_time,
        }
    
    async def qr_decomposition(self, matrix_id: str, mode: str = "reduced") -> dict:
        """Compute QR decomposition."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        decomposition = matrix.qr_decomposition(mode)
        
        return {
            "Q": decomposition.factors["Q"].tolist(),
            "R": decomposition.factors["R"].tolist(),
            "computation_time": decomposition.computation_time,
        }
    
    async def svd_decomposition(self, matrix_id: str, full_matrices: bool = True) -> dict:
        """Compute SVD decomposition."""
        matrix = await self._matrix_repository.get_by_id(MatrixId(matrix_id))
        
        if matrix is None:
            raise ValueError(f"Matrix with ID {matrix_id} not found")
        
        decomposition = matrix.svd_decomposition(full_matrices)
        
        return {
            "U": decomposition.factors["U"].tolist(),
            "s": decomposition.factors["s"].tolist(),
            "Vh": decomposition.factors["Vh"].tolist(),
            "computation_time": decomposition.computation_time,
            "condition_number": decomposition.condition_number,
        }
    
    def _to_dto(self, matrix) -> MatrixDTO:
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