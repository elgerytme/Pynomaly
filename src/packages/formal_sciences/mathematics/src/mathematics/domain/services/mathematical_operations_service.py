"""Domain service for mathematical operations."""

from typing import List, Optional
import numpy as np

from ..entities.math_function import MathFunction
from ..value_objects.function_value_objects import Domain, FunctionProperties, FunctionType, DifferentiabilityType


class MathematicalOperationsService:
    """Domain service for mathematical operations on functions."""
    
    def compose_functions(self, f: MathFunction, g: MathFunction) -> MathFunction:
        """
        Compose two functions: f(g(x)).
        
        Args:
            f: Outer function
            g: Inner function
            
        Returns:
            Composed function
        """
        return f.compose(g)
    
    def add_functions(self, f: MathFunction, g: MathFunction) -> MathFunction:
        """
        Add two functions: (f + g)(x) = f(x) + g(x).
        
        Args:
            f: First function
            g: Second function
            
        Returns:
            Sum function
        """
        # Validate functions have compatible domains and variables
        if f.variables != g.variables:
            raise ValueError("Functions must have the same variables")
        
        # Compute domain intersection
        domain_intersection = f.domain.intersect(g.domain)
        if domain_intersection is None:
            raise ValueError("Functions have disjoint domains")
        
        # Create sum expression
        sum_expression = f"({f.expression}) + ({g.expression})"
        
        # Determine properties of sum
        sum_properties = FunctionProperties(
            function_type=FunctionType.COMPOSITE,
            differentiability=min(f.properties.differentiability, g.properties.differentiability),
            is_continuous=f.properties.is_continuous and g.properties.is_continuous,
            is_monotonic=False,  # Generally not monotonic
            is_periodic=f.properties.is_periodic and g.properties.is_periodic,
            is_even=f.properties.is_even and g.properties.is_even,
            is_odd=f.properties.is_odd and g.properties.is_odd,
            is_bounded=f.properties.is_bounded and g.properties.is_bounded,
        )
        
        # Create metadata
        sum_metadata = f.metadata.update_metadata(
            name=f"{f.metadata.name}_plus_{g.metadata.name}",
            description=f"Sum of {f.metadata.description} and {g.metadata.description}",
            computational_complexity="O(f(n) + g(n))"
        )
        
        return MathFunction(
            function_id=f.function_id.__class__(),
            expression=sum_expression,
            variables=f.variables,
            domain=domain_intersection,
            codomain=f.codomain,  # Would need proper calculation
            properties=sum_properties,
            metadata=sum_metadata
        )
    
    def multiply_functions(self, f: MathFunction, g: MathFunction) -> MathFunction:
        """
        Multiply two functions: (f * g)(x) = f(x) * g(x).
        
        Args:
            f: First function
            g: Second function
            
        Returns:
            Product function
        """
        # Validate functions have compatible domains and variables
        if f.variables != g.variables:
            raise ValueError("Functions must have the same variables")
        
        # Compute domain intersection
        domain_intersection = f.domain.intersect(g.domain)
        if domain_intersection is None:
            raise ValueError("Functions have disjoint domains")
        
        # Create product expression
        product_expression = f"({f.expression}) * ({g.expression})"
        
        # Determine properties of product
        product_properties = FunctionProperties(
            function_type=FunctionType.COMPOSITE,
            differentiability=min(f.properties.differentiability, g.properties.differentiability),
            is_continuous=f.properties.is_continuous and g.properties.is_continuous,
            is_monotonic=False,  # Generally not monotonic
            is_periodic=f.properties.is_periodic and g.properties.is_periodic,
            is_even=(f.properties.is_even and g.properties.is_even) or 
                    (f.properties.is_odd and g.properties.is_odd),
            is_odd=(f.properties.is_even and g.properties.is_odd) or 
                   (f.properties.is_odd and g.properties.is_even),
            is_bounded=f.properties.is_bounded and g.properties.is_bounded,
        )
        
        # Create metadata
        product_metadata = f.metadata.update_metadata(
            name=f"{f.metadata.name}_times_{g.metadata.name}",
            description=f"Product of {f.metadata.description} and {g.metadata.description}",
            computational_complexity="O(f(n) * g(n))"
        )
        
        return MathFunction(
            function_id=f.function_id.__class__(),
            expression=product_expression,
            variables=f.variables,
            domain=domain_intersection,
            codomain=f.codomain,  # Would need proper calculation
            properties=product_properties,
            metadata=product_metadata
        )
    
    def chain_rule_derivative(self, f: MathFunction, g: MathFunction, variable: str) -> MathFunction:
        """
        Apply chain rule to compute derivative of composite function.
        
        Args:
            f: Outer function
            g: Inner function
            variable: Variable to differentiate with respect to
            
        Returns:
            Derivative using chain rule
        """
        # d/dx[f(g(x))] = f'(g(x)) * g'(x)
        f_prime = f.derivative(g.variables[0])  # f'(g(x))
        g_prime = g.derivative(variable)  # g'(x)
        
        # Compose f' with g
        f_prime_composed = f_prime.compose(g)
        
        # Multiply by g'
        return self.multiply_functions(f_prime_composed, g_prime)
    
    def find_critical_points(self, f: MathFunction, variable: str) -> List[float]:
        """
        Find critical points of a function (where derivative is zero or undefined).
        
        Args:
            f: Function to analyze
            variable: Variable to differentiate with respect to
            
        Returns:
            List of critical points
        """
        # This is a simplified implementation
        # In practice, this would use numerical methods or symbolic computation
        
        if f.properties.differentiability == DifferentiabilityType.NOT_DIFFERENTIABLE:
            return []
        
        # Get derivative
        f_prime = f.derivative(variable)
        
        # Find zeros of derivative (simplified approach)
        # This would use numerical root-finding in practice
        critical_points = []
        
        # Sample points in domain and check where derivative is close to zero
        domain = f.domain
        for x in np.linspace(domain.lower_bound, domain.upper_bound, 1000):
            try:
                if domain.contains(x):
                    derivative_value = f_prime.evaluate(**{variable: x})
                    if abs(derivative_value) < 1e-6:
                        critical_points.append(x)
            except ValueError:
                # Derivative undefined at this point
                critical_points.append(x)
        
        return critical_points
    
    def integrate_function(self, f: MathFunction, variable: str, 
                          lower_bound: Optional[float] = None,
                          upper_bound: Optional[float] = None) -> float:
        """
        Numerically integrate a function.
        
        Args:
            f: Function to integrate
            variable: Variable to integrate over
            lower_bound: Lower integration bound
            upper_bound: Upper integration bound
            
        Returns:
            Integral value
        """
        from scipy.integrate import quad
        
        # Define function for numerical integration
        def integrand(x):
            return f.evaluate(**{variable: x})
        
        # Use domain bounds if not specified
        if lower_bound is None:
            lower_bound = f.domain.lower_bound
        if upper_bound is None:
            upper_bound = f.domain.upper_bound
        
        # Numerical integration
        result, error = quad(integrand, lower_bound, upper_bound)
        return result
    
    def evaluate_limit(self, f: MathFunction, variable: str, 
                      approach_value: float, direction: str = "both") -> float:
        """
        Evaluate limit of function at a point.
        
        Args:
            f: Function
            variable: Variable
            approach_value: Value to approach
            direction: "left", "right", or "both"
            
        Returns:
            Limit value
        """
        epsilon = 1e-6
        
        if direction == "left":
            test_points = [approach_value - epsilon * (10 ** -i) for i in range(5)]
        elif direction == "right":
            test_points = [approach_value + epsilon * (10 ** -i) for i in range(5)]
        else:  # both
            test_points = [approach_value + epsilon * (10 ** -i) * (-1) ** i for i in range(10)]
        
        # Evaluate function at test points
        values = []
        for x in test_points:
            try:
                value = f.evaluate(**{variable: x})
                values.append(value)
            except ValueError:
                continue
        
        if not values:
            raise ValueError("Cannot evaluate limit")
        
        # Check if values converge
        if len(values) >= 2:
            if abs(values[-1] - values[-2]) < epsilon:
                return values[-1]
        
        return sum(values) / len(values)  # Simple average