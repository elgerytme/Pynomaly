"""Application service for function operations."""

from typing import List, Optional

from ...domain.repositories.function_repository import FunctionRepository
from ...domain.services.mathematical_operations_service import MathematicalOperationsService
from ...domain.value_objects.function_value_objects import FunctionId
from ..dto.function_dto import FunctionDTO, EvaluationRequestDTO, EvaluationResponseDTO
from ..use_cases.create_function_use_case import CreateFunctionUseCase
from ..use_cases.evaluate_function_use_case import EvaluateFunctionUseCase


class FunctionApplicationService:
    """Application service for function operations."""
    
    def __init__(self, function_repository: FunctionRepository, 
                 math_operations_service: MathematicalOperationsService):
        self._function_repository = function_repository
        self._math_operations_service = math_operations_service
        self._create_function_use_case = CreateFunctionUseCase(function_repository)
        self._evaluate_function_use_case = EvaluateFunctionUseCase(function_repository)
    
    async def create_function(self, function_dto: FunctionDTO) -> FunctionDTO:
        """Create a new function."""
        return await self._create_function_use_case.execute(function_dto)
    
    async def evaluate_function(self, request: EvaluationRequestDTO) -> EvaluationResponseDTO:
        """Evaluate a function."""
        return await self._evaluate_function_use_case.execute(request)
    
    async def get_function_by_id(self, function_id: str) -> Optional[FunctionDTO]:
        """Get function by ID."""
        function = await self._function_repository.get_by_id(FunctionId(function_id))
        return self._to_dto(function) if function else None
    
    async def get_functions_by_name(self, name: str) -> List[FunctionDTO]:
        """Get functions by name."""
        functions = await self._function_repository.get_by_name(name)
        return [self._to_dto(f) for f in functions]
    
    async def get_functions_by_type(self, function_type: str) -> List[FunctionDTO]:
        """Get functions by type."""
        functions = await self._function_repository.get_by_type(function_type)
        return [self._to_dto(f) for f in functions]
    
    async def search_functions(self, query: str) -> List[FunctionDTO]:
        """Search functions."""
        functions = await self._function_repository.search(query)
        return [self._to_dto(f) for f in functions]
    
    async def delete_function(self, function_id: str) -> bool:
        """Delete a function."""
        return await self._function_repository.delete(FunctionId(function_id))
    
    async def compose_functions(self, f_id: str, g_id: str) -> FunctionDTO:
        """Compose two functions."""
        f = await self._function_repository.get_by_id(FunctionId(f_id))
        g = await self._function_repository.get_by_id(FunctionId(g_id))
        
        if f is None:
            raise ValueError(f"Function with ID {f_id} not found")
        if g is None:
            raise ValueError(f"Function with ID {g_id} not found")
        
        composed = self._math_operations_service.compose_functions(f, g)
        saved_composed = await self._function_repository.save(composed)
        
        return self._to_dto(saved_composed)
    
    async def add_functions(self, f_id: str, g_id: str) -> FunctionDTO:
        """Add two functions."""
        f = await self._function_repository.get_by_id(FunctionId(f_id))
        g = await self._function_repository.get_by_id(FunctionId(g_id))
        
        if f is None:
            raise ValueError(f"Function with ID {f_id} not found")
        if g is None:
            raise ValueError(f"Function with ID {g_id} not found")
        
        sum_func = self._math_operations_service.add_functions(f, g)
        saved_sum = await self._function_repository.save(sum_func)
        
        return self._to_dto(saved_sum)
    
    async def multiply_functions(self, f_id: str, g_id: str) -> FunctionDTO:
        """Multiply two functions."""
        f = await self._function_repository.get_by_id(FunctionId(f_id))
        g = await self._function_repository.get_by_id(FunctionId(g_id))
        
        if f is None:
            raise ValueError(f"Function with ID {f_id} not found")
        if g is None:
            raise ValueError(f"Function with ID {g_id} not found")
        
        product_func = self._math_operations_service.multiply_functions(f, g)
        saved_product = await self._function_repository.save(product_func)
        
        return self._to_dto(saved_product)
    
    async def differentiate_function(self, function_id: str, variable: str, order: int = 1) -> FunctionDTO:
        """Differentiate a function."""
        function = await self._function_repository.get_by_id(FunctionId(function_id))
        
        if function is None:
            raise ValueError(f"Function with ID {function_id} not found")
        
        derivative = function.derivative(variable, order)
        saved_derivative = await self._function_repository.save(derivative)
        
        return self._to_dto(saved_derivative)
    
    async def integrate_function(self, function_id: str, variable: str, 
                                lower_bound: Optional[float] = None,
                                upper_bound: Optional[float] = None) -> FunctionDTO:
        """Integrate a function."""
        function = await self._function_repository.get_by_id(FunctionId(function_id))
        
        if function is None:
            raise ValueError(f"Function with ID {function_id} not found")
        
        integral = function.integral(variable, lower_bound, upper_bound)
        saved_integral = await self._function_repository.save(integral)
        
        return self._to_dto(saved_integral)
    
    async def find_critical_points(self, function_id: str, variable: str) -> List[float]:
        """Find critical points of a function."""
        function = await self._function_repository.get_by_id(FunctionId(function_id))
        
        if function is None:
            raise ValueError(f"Function with ID {function_id} not found")
        
        return self._math_operations_service.find_critical_points(function, variable)
    
    def _to_dto(self, function) -> FunctionDTO:
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