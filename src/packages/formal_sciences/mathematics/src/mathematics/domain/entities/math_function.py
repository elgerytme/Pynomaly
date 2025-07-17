"""Mathematical function domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import dataclasses

from ..value_objects.function_value_objects import (
    FunctionId, Domain, FunctionProperties, FunctionMetadata, EvaluationCache,
    FunctionType, DifferentiabilityType
)


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