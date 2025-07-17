"""SymPy adapter for symbolic mathematics."""

from typing import Dict, List, Optional, Any
import sympy as sp

from ...domain.entities.math_function import MathFunction
from ...domain.value_objects.function_value_objects import FunctionType, DifferentiabilityType


class SymPyAdapter:
    """Adapter for SymPy symbolic mathematics library."""
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str) -> sp.Symbol:
        """Get or create a SymPy symbol."""
        if name not in self._symbol_cache:
            self._symbol_cache[name] = sp.Symbol(name)
        return self._symbol_cache[name]
    
    def parse_expression(self, expression: str, variables: List[str]) -> sp.Expr:
        """Parse a mathematical expression string into SymPy expression."""
        # Create symbols for variables
        symbols = {var: self.get_symbol(var) for var in variables}
        
        # Parse the expression
        try:
            expr = sp.sympify(expression, locals=symbols)
            return expr
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{expression}': {e}")
    
    def evaluate_expression(self, expr: sp.Expr, variable_values: Dict[str, float]) -> float:
        """Evaluate a SymPy expression with given variable values."""
        # Create substitution dictionary
        substitutions = {
            self.get_symbol(var): value 
            for var, value in variable_values.items()
        }
        
        # Evaluate expression
        try:
            result = expr.subs(substitutions)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")
    
    def differentiate(self, expr: sp.Expr, variable: str, order: int = 1) -> sp.Expr:
        """Compute derivative of expression."""
        symbol = self.get_symbol(variable)
        try:
            derivative = sp.diff(expr, symbol, order)
            return derivative
        except Exception as e:
            raise ValueError(f"Failed to differentiate: {e}")
    
    def integrate(self, expr: sp.Expr, variable: str, 
                  lower_bound: Optional[float] = None,
                  upper_bound: Optional[float] = None) -> sp.Expr:
        """Compute integral of expression."""
        symbol = self.get_symbol(variable)
        
        try:
            if lower_bound is not None and upper_bound is not None:
                # Definite integral
                integral = sp.integrate(expr, (symbol, lower_bound, upper_bound))
            else:
                # Indefinite integral
                integral = sp.integrate(expr, symbol)
            
            return integral
        except Exception as e:
            raise ValueError(f"Failed to integrate: {e}")
    
    def solve_equation(self, equation: str, variable: str) -> List[float]:
        """Solve equation for a variable."""
        symbol = self.get_symbol(variable)
        
        try:
            eq = sp.sympify(equation)
            solutions = sp.solve(eq, symbol)
            return [float(sol) for sol in solutions if sol.is_real]
        except Exception as e:
            raise ValueError(f"Failed to solve equation: {e}")
    
    def find_critical_points(self, expr: sp.Expr, variable: str) -> List[float]:
        """Find critical points of expression."""
        symbol = self.get_symbol(variable)
        
        try:
            # Compute derivative
            derivative = sp.diff(expr, symbol)
            
            # Find zeros of derivative
            critical_points = sp.solve(derivative, symbol)
            
            # Return real solutions
            return [float(pt) for pt in critical_points if pt.is_real]
        except Exception as e:
            raise ValueError(f"Failed to find critical points: {e}")
    
    def compute_limit(self, expr: sp.Expr, variable: str, 
                     approach_value: float, direction: str = "both") -> float:
        """Compute limit of expression."""
        symbol = self.get_symbol(variable)
        
        try:
            if direction == "left":
                limit = sp.limit(expr, symbol, approach_value, '-')
            elif direction == "right":
                limit = sp.limit(expr, symbol, approach_value, '+')
            else:
                limit = sp.limit(expr, symbol, approach_value)
            
            return float(limit)
        except Exception as e:
            raise ValueError(f"Failed to compute limit: {e}")
    
    def simplify_expression(self, expr: sp.Expr) -> sp.Expr:
        """Simplify expression."""
        try:
            return sp.simplify(expr)
        except Exception as e:
            raise ValueError(f"Failed to simplify expression: {e}")
    
    def expand_expression(self, expr: sp.Expr) -> sp.Expr:
        """Expand expression."""
        try:
            return sp.expand(expr)
        except Exception as e:
            raise ValueError(f"Failed to expand expression: {e}")
    
    def factor_expression(self, expr: sp.Expr) -> sp.Expr:
        """Factor expression."""
        try:
            return sp.factor(expr)
        except Exception as e:
            raise ValueError(f"Failed to factor expression: {e}")
    
    def compose_functions(self, f_expr: sp.Expr, g_expr: sp.Expr, 
                         variable: str) -> sp.Expr:
        """Compose two functions: f(g(x))."""
        symbol = self.get_symbol(variable)
        
        try:
            # Substitute g(x) into f
            composed = f_expr.subs(symbol, g_expr)
            return composed
        except Exception as e:
            raise ValueError(f"Failed to compose functions: {e}")
    
    def analyze_function_properties(self, expr: sp.Expr, variable: str) -> Dict[str, Any]:
        """Analyze properties of a function."""
        symbol = self.get_symbol(variable)
        
        properties = {
            "is_even": False,
            "is_odd": False,
            "is_periodic": False,
            "is_continuous": True,  # Default assumption
            "is_monotonic": False,
            "has_asymptotes": False,
            "vertical_asymptotes": [],
            "horizontal_asymptotes": [],
        }
        
        try:
            # Check if function is even: f(-x) = f(x)
            f_minus_x = expr.subs(symbol, -symbol)
            if sp.simplify(f_minus_x - expr) == 0:
                properties["is_even"] = True
            
            # Check if function is odd: f(-x) = -f(x)
            if sp.simplify(f_minus_x + expr) == 0:
                properties["is_odd"] = True
            
            # Check monotonicity by analyzing derivative
            try:
                derivative = sp.diff(expr, symbol)
                # If derivative is always positive or always negative, function is monotonic
                # This is a simplified check
                if derivative.is_positive or derivative.is_negative:
                    properties["is_monotonic"] = True
            except:
                pass
            
            # Find vertical asymptotes (simplified)
            try:
                # Find where function goes to infinity
                singularities = sp.singularities(expr, symbol)
                properties["vertical_asymptotes"] = [float(s) for s in singularities if s.is_real]
                if properties["vertical_asymptotes"]:
                    properties["has_asymptotes"] = True
            except:
                pass
            
            # Find horizontal asymptotes
            try:
                limit_inf = sp.limit(expr, symbol, sp.oo)
                limit_neg_inf = sp.limit(expr, symbol, -sp.oo)
                
                horizontal_asymptotes = []
                if limit_inf.is_finite:
                    horizontal_asymptotes.append(float(limit_inf))
                if limit_neg_inf.is_finite and limit_neg_inf != limit_inf:
                    horizontal_asymptotes.append(float(limit_neg_inf))
                
                properties["horizontal_asymptotes"] = horizontal_asymptotes
                if horizontal_asymptotes:
                    properties["has_asymptotes"] = True
            except:
                pass
            
        except Exception:
            # If analysis fails, return default properties
            pass
        
        return properties
    
    def get_function_type(self, expr: sp.Expr) -> FunctionType:
        """Determine function type from expression."""
        try:
            # Check if polynomial
            if expr.is_polynomial():
                return FunctionType.POLYNOMIAL
            
            # Check for trigonometric functions
            if any(func in str(expr) for func in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']):
                return FunctionType.TRIGONOMETRIC
            
            # Check for exponential functions
            if any(func in str(expr) for func in ['exp', 'E**']):
                return FunctionType.EXPONENTIAL
            
            # Check for logarithmic functions
            if any(func in str(expr) for func in ['log', 'ln']):
                return FunctionType.LOGARITHMIC
            
            # Check for rational functions
            if expr.is_rational_function():
                return FunctionType.RATIONAL
            
            # Default to composite
            return FunctionType.COMPOSITE
            
        except Exception:
            return FunctionType.COMPOSITE
    
    def expression_to_string(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to string."""
        return str(expr)