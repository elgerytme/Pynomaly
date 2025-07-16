"""
Mathematics Package for Pynomaly

Comprehensive mathematics package providing core mathematical functions,
linear algebra, calculus, and numerical methods for the Pynomaly ecosystem.

This package offers:
- High-performance mathematical computations
- Linear algebra operations (matrices, vectors, eigenvalues)
- Calculus operations (derivatives, integrals, optimization)
- Numerical methods and algorithms
- Integration with NumPy/SciPy ecosystem
"""

__version__ = "1.0.0"
__author__ = "Pynomaly Team"
__email__ = "support@pynomaly.com"

from .domain.entities.math_function import MathFunction
from .domain.entities.matrix import Matrix
from .domain.entities.vector import Vector
from .domain.entities.equation import Equation
from .domain.entities.optimization import Optimization

from .domain.value_objects.complex_number import ComplexNumber
from .domain.value_objects.polynomial import Polynomial
from .domain.value_objects.range import Range
from .domain.value_objects.precision import Precision
from .domain.value_objects.math_result import MathResult

from .application.services.mathematical_computation_service import MathematicalComputationService
from .application.services.linear_algebra_orchestrator import LinearAlgebraOrchestrator
from .application.services.optimization_orchestrator import OptimizationOrchestrator
from .application.services.numerical_analysis_service import NumericalAnalysisService
from .application.services.symbolic_math_service import SymbolicMathService

__all__ = [
    # Domain Entities
    "MathFunction",
    "Matrix", 
    "Vector",
    "Equation",
    "Optimization",
    
    # Value Objects
    "ComplexNumber",
    "Polynomial",
    "Range",
    "Precision",
    "MathResult",
    
    # Application Services
    "MathematicalComputationService",
    "LinearAlgebraOrchestrator",
    "OptimizationOrchestrator", 
    "NumericalAnalysisService",
    "SymbolicMathService",
]