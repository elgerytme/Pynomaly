"""Mathematical function domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from uuid import UUID, uuid4
import numpy as np
from enum import Enum


class FunctionType(Enum):
    """Types of mathematical functions."""
    POLYNOMIAL = "polynomial"
    TRIGONOMETRIC = "trigonometric"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    RATIONAL = "rational"
    COMPOSITE = "composite"
    PIECEWISE = "piecewise"
    IMPLICIT = "implicit"
    PARAMETRIC = "parametric"
    VECTOR_VALUED = "vector_valued"


class DifferentiabilityType(Enum):
    """Function differentiability classification."""
    NOT_DIFFERENTIABLE = "not_differentiable"
    CONTINUOUS = "continuous"
    DIFFERENTIABLE = "differentiable"
    SMOOTH = "smooth"
    ANALYTIC = "analytic"


@dataclass(frozen=True)
class FunctionId:
    """Unique identifier for mathematical functions."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Domain:
    """Mathematical domain specification."""
    lower_bound: float
    upper_bound: float
    include_lower: bool = True
    include_upper: bool = True
    excluded_points: List[float] = field(default_factory=list)
    
    def contains(self, x: float) -> bool:
        """Check if value is in domain."""
        if x in self.excluded_points:
            return False
            
        if x < self.lower_bound or x > self.upper_bound:
            return False
            
        if x == self.lower_bound and not self.include_lower:
            return False
            
        if x == self.upper_bound and not self.include_upper:
            return False
            
        return True
    
    def intersect(self, other: Domain) -> Optional[Domain]:
        """Compute intersection with another domain."""
        lower = max(self.lower_bound, other.lower_bound)
        upper = min(self.upper_bound, other.upper_bound)
        
        if lower > upper:
            return None
            
        include_lower = (
            self.include_lower if lower == self.lower_bound else other.include_lower
        )
        include_upper = (
            self.include_upper if upper == self.upper_bound else other.include_upper
        )
        
        excluded = list(set(self.excluded_points + other.excluded_points))
        excluded = [x for x in excluded if lower <= x <= upper]
        
        return Domain(
            lower_bound=lower,
            upper_bound=upper,
            include_lower=include_lower,
            include_upper=include_upper,
            excluded_points=excluded
        )


@dataclass(frozen=True)
class FunctionProperties:
    """Properties of a mathematical function."""
    function_type: FunctionType
    differentiability: DifferentiabilityType
    is_continuous: bool
    is_monotonic: bool
    is_periodic: bool
    period: Optional[float] = None
    is_even: bool = False
    is_odd: bool = False
    is_bounded: bool = False
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    has_asymptotes: bool = False
    vertical_asymptotes: List[float] = field(default_factory=list)
    horizontal_asymptotes: List[float] = field(default_factory=list)
    oblique_asymptotes: List[str] = field(default_factory=list)  # Linear functions as strings
    
    def validate(self) -> bool:
        """Validate function properties consistency."""
        if self.is_even and self.is_odd:
            return False  # Cannot be both even and odd (except f(x) = 0)
            
        if self.is_periodic and self.period is None:
            return False  # Periodic functions must have a period
            
        if self.is_bounded and (self.upper_bound is None or self.lower_bound is None):
            return False  # Bounded functions must have bounds
            
        return True


@dataclass(frozen=True)
class FunctionMetadata:
    """Metadata for mathematical functions."""
    name: str
    description: str
    author: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    computational_complexity: str = "O(1)"
    numerical_stability: str = "stable"
    
    def add_tag(self, tag: str) -> FunctionMetadata:
        """Add a tag to the function metadata."""
        if tag not in self.tags:
            new_tags = list(self.tags) + [tag]
            return dataclass.replace(self, tags=new_tags)
        return self
    
    def add_reference(self, reference: str) -> FunctionMetadata:
        """Add a reference to the function metadata."""
        if reference not in self.references:
            new_refs = list(self.references) + [reference]
            return dataclass.replace(self, references=new_refs)
        return self


@dataclass(frozen=True)
class EvaluationCache:
    """Cache for function evaluations."""
    cache: Dict[Tuple[float, ...], float] = field(default_factory=dict)
    max_size: int = 1000
    hit_count: int = 0
    miss_count: int = 0
    
    def get(self, inputs: Tuple[float, ...]) -> Optional[float]:
        """Get cached evaluation result."""
        result = self.cache.get(inputs)
        if result is not None:
            return result
        return None
    
    def put(self, inputs: Tuple[float, ...], result: float) -> EvaluationCache:
        """Add evaluation result to cache."""
        if len(self.cache) >= self.max_size:
            # Simple LRU eviction - remove first item
            cache_copy = dict(self.cache)
            cache_copy.pop(next(iter(cache_copy)))
            cache_copy[inputs] = result
            return dataclass.replace(self, cache=cache_copy)
        else:
            cache_copy = dict(self.cache)
            cache_copy[inputs] = result
            return dataclass.replace(self, cache=cache_copy)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


@dataclass(frozen=True)
class MathFunction:
    """
    Domain entity representing a mathematical function.
    
    A mathematical function is a relation between a set of inputs (domain)
    and a set of possible outputs (codomain) where each input is related
    to exactly one output.
    """
    function_id: FunctionId
    expression: str
    variables: List[str]
    domain: Domain
    codomain: Domain
    properties: FunctionProperties
    metadata: FunctionMetadata
    evaluation_cache: EvaluationCache = field(default_factory=EvaluationCache)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate function after initialization."""
        if not self.properties.validate():
            raise ValueError("Invalid function properties")
            
        if not self.variables:
            raise ValueError("Function must have at least one variable")
            
        if not self.expression.strip():
            raise ValueError("Function expression cannot be empty")
    
    def evaluate(self, **kwargs: float) -> float:
        """
        Evaluate the function at given variable values.
        
        Args:
            **kwargs: Variable values for evaluation
            
        Returns:
            Function evaluation result
            
        Raises:
            ValueError: If required variables are missing or values are outside domain
        """
        # Validate all required variables are provided
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing values for variables: {missing_vars}")
        
        # Extract values in variable order
        values = tuple(kwargs[var] for var in self.variables)
        
        # Check cache first
        cached_result = self.evaluation_cache.get(values)
        if cached_result is not None:
            return cached_result
        
        # Validate inputs are in domain (for single variable functions)
        if len(self.variables) == 1:
            x = values[0]
            if not self.domain.contains(x):
                raise ValueError(f"Value {x} is outside function domain")
        
        # Evaluate function (this would be implemented by specific function types)
        result = self._evaluate_expression(values)
        
        # Cache result
        self.evaluation_cache.put(values, result)
        
        return result
    
    def _evaluate_expression(self, values: Tuple[float, ...]) -> float:
        """
        Internal method to evaluate the mathematical expression.
        
        This would be implemented by specific function implementations
        or use a mathematical expression parser/evaluator.
        """
        # Placeholder implementation - in real implementation, this would
        # use a mathematical expression parser like sympy or custom evaluator
        try:
            # Create variable mapping
            var_dict = dict(zip(self.variables, values))
            
            # Basic expression evaluation (simplified)
            # In production, use sympy or other mathematical expression evaluator
            import ast
            import operator
            
            # Supported operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub, 
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Num):  # Number
                    return node.n
                elif isinstance(node, ast.Name):  # Variable
                    return var_dict[node.id]
                elif isinstance(node, ast.BinOp):  # Binary operation
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):  # Unary operation
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(f"Unsupported expression type: {type(node)}")
            
            # Parse and evaluate expression
            tree = ast.parse(self.expression, mode='eval')
            return eval_expr(tree.body)
            
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{self.expression}': {e}")
    
    def derivative(self, variable: str, order: int = 1) -> MathFunction:
        """
        Compute the derivative of the function with respect to a variable.
        
        Args:
            variable: Variable to differentiate with respect to
            order: Order of derivative (default: 1)
            
        Returns:
            New MathFunction representing the derivative
            
        Raises:
            ValueError: If variable is not in function variables or function is not differentiable
        """
        if variable not in self.variables:
            raise ValueError(f"Variable '{variable}' not in function variables")
            
        if self.properties.differentiability == DifferentiabilityType.NOT_DIFFERENTIABLE:
            raise ValueError("Function is not differentiable")
        
        # This would use symbolic differentiation (e.g., sympy)
        # Placeholder implementation
        derivative_expr = f"d/d{variable}({self.expression})"
        
        # Create derivative function properties
        derivative_properties = dataclass.replace(
            self.properties,
            differentiability=DifferentiabilityType.CONTINUOUS if order > 1 else self.properties.differentiability
        )
        
        # Create derivative metadata
        derivative_metadata = dataclass.replace(
            self.metadata,
            name=f"{self.metadata.name}_derivative_{order}",
            description=f"{order}-order derivative of {self.metadata.description}",
            computational_complexity="O(n)" if order == 1 else f"O(n^{order})"
        )
        
        return MathFunction(
            function_id=FunctionId(),
            expression=derivative_expr,
            variables=self.variables,
            domain=self.domain,
            codomain=self.codomain,  # May need adjustment based on derivative
            properties=derivative_properties,
            metadata=derivative_metadata,
        )
    
    def integral(self, variable: str, lower_bound: Optional[float] = None, 
                upper_bound: Optional[float] = None) -> MathFunction:
        """
        Compute the integral of the function.
        
        Args:
            variable: Variable to integrate with respect to
            lower_bound: Lower bound for definite integral (optional)
            upper_bound: Upper bound for definite integral (optional)
            
        Returns:
            New MathFunction representing the integral (indefinite) or float (definite)
        """
        if variable not in self.variables:
            raise ValueError(f"Variable '{variable}' not in function variables")
        
        # This would use symbolic integration (e.g., sympy)
        # Placeholder implementation
        if lower_bound is not None and upper_bound is not None:
            # Definite integral - would return a number
            integral_expr = f"∫[{lower_bound},{upper_bound}]({self.expression})d{variable}"
        else:
            # Indefinite integral
            integral_expr = f"∫({self.expression})d{variable}"
        
        # Create integral function properties
        integral_properties = dataclass.replace(
            self.properties,
            differentiability=DifferentiabilityType.SMOOTH
        )
        
        # Create integral metadata
        integral_metadata = dataclass.replace(
            self.metadata,
            name=f"{self.metadata.name}_integral",
            description=f"Integral of {self.metadata.description}",
            computational_complexity="O(n log n)"
        )
        
        return MathFunction(
            function_id=FunctionId(),
            expression=integral_expr,
            variables=self.variables,
            domain=self.domain,
            codomain=self.codomain,
            properties=integral_properties,
            metadata=integral_metadata,
        )
    
    def compose(self, other: MathFunction) -> MathFunction:
        """
        Compose this function with another function: f(g(x)).
        
        Args:
            other: Function to compose with
            
        Returns:
            New MathFunction representing the composition
        """
        # Validate composition is possible
        if len(other.variables) != 1 or len(self.variables) != 1:
            raise ValueError("Function composition currently supported only for single-variable functions")
        
        # Check codomain/domain compatibility
        # other's codomain should be subset of self's domain
        
        # Create composition expression
        composition_expr = self.expression.replace(self.variables[0], f"({other.expression})")
        
        # Determine composed function properties
        composed_properties = FunctionProperties(
            function_type=FunctionType.COMPOSITE,
            differentiability=min(self.properties.differentiability, other.properties.differentiability),
            is_continuous=self.properties.is_continuous and other.properties.is_continuous,
            is_monotonic=self.properties.is_monotonic and other.properties.is_monotonic,
            is_periodic=self.properties.is_periodic or other.properties.is_periodic,
            is_even=self.properties.is_even and other.properties.is_even,
            is_odd=self.properties.is_odd and other.properties.is_odd,
            is_bounded=self.properties.is_bounded and other.properties.is_bounded,
        )
        
        # Create composition metadata
        composed_metadata = FunctionMetadata(
            name=f"{self.metadata.name}_composed_{other.metadata.name}",
            description=f"Composition of {self.metadata.description} and {other.metadata.description}",
            author=f"{self.metadata.author}, {other.metadata.author}",
            computational_complexity="O(f(g(n)))"
        )
        
        return MathFunction(
            function_id=FunctionId(),
            expression=composition_expr,
            variables=other.variables,  # Input variables of the outer function
            domain=other.domain,  # Domain of the inner function
            codomain=self.codomain,  # Codomain of the outer function
            properties=composed_properties,
            metadata=composed_metadata,
        )
    
    def is_equal(self, other: MathFunction, tolerance: float = 1e-10) -> bool:
        """
        Check if this function is mathematically equivalent to another function.
        
        Args:
            other: Function to compare with
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if functions are equivalent within tolerance
        """
        # Quick checks
        if self.variables != other.variables:
            return False
            
        if self.expression == other.expression:
            return True
        
        # Numerical comparison at sample points
        # This is a simplified implementation - a complete implementation
        # would use symbolic mathematics to prove equivalence
        sample_points = 100
        for _ in range(sample_points):
            # Generate random point in domain
            test_values = {}
            for var in self.variables:
                # Sample from domain intersection
                domain_intersect = self.domain.intersect(other.domain)
                if domain_intersect is None:
                    return False
                    
                x = np.random.uniform(domain_intersect.lower_bound, domain_intersect.upper_bound)
                test_values[var] = x
            
            try:
                result1 = self.evaluate(**test_values)
                result2 = other.evaluate(**test_values)
                
                if abs(result1 - result2) > tolerance:
                    return False
            except ValueError:
                # If evaluation fails for either function, they're not equivalent
                return False
        
        return True
    
    def update_metadata(self, **kwargs) -> MathFunction:
        """Update function metadata."""
        new_metadata = dataclass.replace(self.metadata, **kwargs)
        return dataclass.replace(
            self, 
            metadata=new_metadata,
            last_modified=datetime.utcnow()
        )
    
    def clear_cache(self) -> MathFunction:
        """Clear the evaluation cache."""
        return dataclass.replace(
            self,
            evaluation_cache=EvaluationCache(),
            last_modified=datetime.utcnow()
        )
    
    def __str__(self) -> str:
        """String representation of the function."""
        return f"f({', '.join(self.variables)}) = {self.expression}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"MathFunction(id={self.function_id}, "
                f"expression='{self.expression}', "
                f"variables={self.variables}, "
                f"type={self.properties.function_type.value})")