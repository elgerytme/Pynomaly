"""Use case for evaluating mathematical functions."""

import time
from typing import Optional

from ...domain.repositories.function_repository import FunctionRepository
from ...domain.value_objects.function_value_objects import FunctionId
from ..dto.function_dto import EvaluationRequestDTO, EvaluationResponseDTO


class EvaluateFunctionUseCase:
    """Use case for evaluating mathematical functions."""
    
    def __init__(self, function_repository: FunctionRepository):
        self._function_repository = function_repository
    
    async def execute(self, request: EvaluationRequestDTO) -> EvaluationResponseDTO:
        """
        Evaluate a mathematical function.
        
        Args:
            request: Evaluation request
            
        Returns:
            Evaluation response
        """
        start_time = time.time()
        
        try:
            # Get function from repository
            function_id = FunctionId(request.function_id)
            function = await self._function_repository.get_by_id(function_id)
            
            if function is None:
                raise ValueError(f"Function with ID {request.function_id} not found")
            
            # Check if result is cached
            values_tuple = tuple(request.variable_values[var] for var in function.variables)
            cached_result = function.evaluation_cache.get(values_tuple)
            
            if cached_result is not None:
                evaluation_time = time.time() - start_time
                return EvaluationResponseDTO(
                    result=cached_result,
                    function_id=request.function_id,
                    variable_values=request.variable_values,
                    evaluation_time=evaluation_time,
                    cached=True
                )
            
            # Evaluate function
            result = function.evaluate(**request.variable_values)
            
            # Update cache if requested
            if request.cache_result:
                function.evaluation_cache.put(values_tuple, result)
                await self._function_repository.update(function)
            
            evaluation_time = time.time() - start_time
            
            return EvaluationResponseDTO(
                result=result,
                function_id=request.function_id,
                variable_values=request.variable_values,
                evaluation_time=evaluation_time,
                cached=False
            )
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            return EvaluationResponseDTO(
                result=0.0,
                function_id=request.function_id,
                variable_values=request.variable_values,
                evaluation_time=evaluation_time,
                cached=False,
                error=str(e)
            )