"""Use case for creating mathematical functions."""

from typing import Optional
from datetime import datetime

from ...domain.entities.math_function import MathFunction
from ...domain.repositories.function_repository import FunctionRepository
from ...domain.value_objects.function_value_objects import (
    FunctionId, Domain, FunctionProperties, FunctionMetadata, FunctionType, 
    DifferentiabilityType, EvaluationCache
)
from ..dto.function_dto import FunctionDTO


class CreateFunctionUseCase:
    """Use case for creating mathematical functions."""
    
    def __init__(self, function_repository: FunctionRepository):
        self._function_repository = function_repository
    
    async def execute(self, function_dto: FunctionDTO) -> FunctionDTO:
        """
        Create a new mathematical function.
        
        Args:
            function_dto: Function data transfer object
            
        Returns:
            Created function as DTO
        """
        # Validate input
        if not function_dto.expression:
            raise ValueError("Function expression is required")
        
        if not function_dto.variables:
            raise ValueError("Function variables are required")
        
        # Create domain value object
        domain = Domain(
            lower_bound=function_dto.domain_lower,
            upper_bound=function_dto.domain_upper,
            include_lower=function_dto.include_lower,
            include_upper=function_dto.include_upper,
            excluded_points=function_dto.excluded_points
        )
        
        # Create codomain (simplified - same as domain for now)
        codomain = domain
        
        # Create function properties
        function_type = FunctionType(function_dto.function_type) if function_dto.function_type else FunctionType.GENERAL
        
        properties = FunctionProperties(
            function_type=function_type,
            differentiability=DifferentiabilityType.DIFFERENTIABLE,  # Default
            is_continuous=function_dto.is_continuous,
            is_monotonic=function_dto.is_monotonic,
            is_periodic=function_dto.is_periodic,
            period=function_dto.period,
            is_even=function_dto.is_even,
            is_odd=function_dto.is_odd,
        )
        
        # Create function metadata
        metadata = FunctionMetadata(
            name=function_dto.name,
            description=function_dto.description,
            author=function_dto.author,
            version=function_dto.version,
            tags=function_dto.tags,
            computational_complexity=function_dto.computational_complexity,
        )
        
        # Create domain entity
        function = MathFunction(
            function_id=FunctionId(),
            expression=function_dto.expression,
            variables=function_dto.variables,
            domain=domain,
            codomain=codomain,
            properties=properties,
            metadata=metadata,
            evaluation_cache=EvaluationCache(),
        )
        
        # Save to repository
        saved_function = await self._function_repository.save(function)
        
        # Convert back to DTO
        return self._to_dto(saved_function)
    
    def _to_dto(self, function: MathFunction) -> FunctionDTO:
        """Convert domain entity to DTO."""
        return FunctionDTO(
            id=str(function.function_id),
            expression=function.expression,
            variables=function.variables,
            name=function.metadata.name,
            description=function.metadata.description,
            function_type=function.properties.function_type.value,
            domain_lower=function.domain.lower_bound,
            domain_upper=function.domain.upper_bound,
            include_lower=function.domain.include_lower,
            include_upper=function.domain.include_upper,
            excluded_points=function.domain.excluded_points,
            is_continuous=function.properties.is_continuous,
            is_monotonic=function.properties.is_monotonic,
            is_periodic=function.properties.is_periodic,
            period=function.properties.period,
            is_even=function.properties.is_even,
            is_odd=function.properties.is_odd,
            author=function.metadata.author,
            version=function.metadata.version,
            tags=function.metadata.tags,
            computational_complexity=function.metadata.computational_complexity,
            created_at=function.created_at,
            last_modified=function.last_modified,
        )