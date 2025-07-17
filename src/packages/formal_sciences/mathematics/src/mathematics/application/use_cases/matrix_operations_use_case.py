"""Use case for matrix operations."""

import time
from typing import Optional

from ...domain.repositories.matrix_repository import MatrixRepository
from ...domain.services.matrix_operations_service import MatrixOperationsService
from ...domain.value_objects.matrix_value_objects import MatrixId
from ..dto.matrix_dto import MatrixOperationDTO, MatrixOperationResponseDTO, MatrixDTO


class MatrixOperationsUseCase:
    """Use case for matrix operations."""
    
    def __init__(self, matrix_repository: MatrixRepository, matrix_service: MatrixOperationsService):
        self._matrix_repository = matrix_repository
        self._matrix_service = matrix_service
    
    async def execute(self, operation_dto: MatrixOperationDTO) -> MatrixOperationResponseDTO:
        """
        Execute matrix operations.
        
        Args:
            operation_dto: Operation request
            
        Returns:
            Operation response
        """
        start_time = time.time()
        
        try:
            # Get matrices from repository
            matrix_a_id = MatrixId(operation_dto.matrix_a_id)
            matrix_a = await self._matrix_repository.get_by_id(matrix_a_id)
            
            if matrix_a is None:
                raise ValueError(f"Matrix with ID {operation_dto.matrix_a_id} not found")
            
            matrix_b = None
            if operation_dto.matrix_b_id:
                matrix_b_id = MatrixId(operation_dto.matrix_b_id)
                matrix_b = await self._matrix_repository.get_by_id(matrix_b_id)
                
                if matrix_b is None:
                    raise ValueError(f"Matrix with ID {operation_dto.matrix_b_id} not found")
            
            # Execute operation
            result_matrix = await self._execute_operation(
                operation_dto.operation,
                matrix_a,
                matrix_b,
                operation_dto.scalar_value,
                operation_dto.parameters
            )
            
            # Save result if it's a new matrix
            if result_matrix.matrix_id not in [matrix_a.matrix_id, matrix_b.matrix_id if matrix_b else None]:
                result_matrix = await self._matrix_repository.save(result_matrix)
            
            computation_time = time.time() - start_time
            
            # Build operand IDs list
            operand_ids = [operation_dto.matrix_a_id]
            if operation_dto.matrix_b_id:
                operand_ids.append(operation_dto.matrix_b_id)
            
            return MatrixOperationResponseDTO(
                result_matrix=self._to_dto(result_matrix),
                operation=operation_dto.operation,
                operand_ids=operand_ids,
                computation_time=computation_time
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            return MatrixOperationResponseDTO(
                result_matrix=MatrixDTO(),
                operation=operation_dto.operation,
                operand_ids=[operation_dto.matrix_a_id],
                computation_time=computation_time,
                error=str(e)
            )
    
    async def _execute_operation(self, operation: str, matrix_a, matrix_b=None, 
                                scalar_value=None, parameters=None):
        """Execute specific matrix operation."""
        if operation == "add":
            if matrix_b is None:
                raise ValueError("Matrix addition requires two matrices")
            return matrix_a.add(matrix_b)
        
        elif operation == "subtract":
            if matrix_b is None:
                raise ValueError("Matrix subtraction requires two matrices")
            return matrix_a.subtract(matrix_b)
        
        elif operation == "multiply":
            if matrix_b is not None:
                return matrix_a.multiply(matrix_b)
            elif scalar_value is not None:
                return matrix_a.multiply(scalar_value)
            else:
                raise ValueError("Matrix multiplication requires either another matrix or scalar")
        
        elif operation == "transpose":
            return matrix_a.transpose()
        
        elif operation == "conjugate_transpose":
            return matrix_a.conjugate_transpose()
        
        elif operation == "inverse":
            return matrix_a.inverse()
        
        elif operation == "power":
            power = parameters.get("power", 2) if parameters else 2
            return self._matrix_service.matrix_power(matrix_a, power)
        
        elif operation == "exponential":
            return self._matrix_service.matrix_exponential(matrix_a)
        
        elif operation == "logarithm":
            return self._matrix_service.matrix_logarithm(matrix_a)
        
        elif operation == "square_root":
            return self._matrix_service.matrix_square_root(matrix_a)
        
        elif operation == "pseudoinverse":
            return self._matrix_service.pseudo_inverse(matrix_a)
        
        elif operation == "kronecker":
            if matrix_b is None:
                raise ValueError("Kronecker product requires two matrices")
            return self._matrix_service.kronecker_product(matrix_a, matrix_b)
        
        elif operation == "hadamard":
            if matrix_b is None:
                raise ValueError("Hadamard product requires two matrices")
            return self._matrix_service.hadamard_product(matrix_a, matrix_b)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
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