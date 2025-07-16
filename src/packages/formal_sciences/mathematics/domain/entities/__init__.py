"""Domain entities for the mathematics package."""

from .math_function import MathFunction, FunctionId, FunctionProperties, FunctionMetadata
from .matrix import Matrix, MatrixId, MatrixProperties
from .vector import Vector, VectorId, VectorProperties, VectorSpace
from .equation import Equation, EquationId, EquationType, Solution, SolutionMethod
from .optimization import (
    Optimization,
    OptimizationId,
    OptimizationMethod,
    OptimizationSolution,
    OptimizationIteration,
    OptimizationVariable,
    Constraint,
    ConvergenceCriteria,
)

__all__ = [
    # MathFunction
    "MathFunction",
    "FunctionId",
    "FunctionProperties",
    "FunctionMetadata",
    
    # Matrix
    "Matrix",
    "MatrixId", 
    "MatrixProperties",
    
    # Vector
    "Vector",
    "VectorId",
    "VectorProperties",
    "VectorSpace",
    
    # Equation
    "Equation",
    "EquationId",
    "EquationType",
    "Solution",
    "SolutionMethod",
    
    # Optimization
    "Optimization",
    "OptimizationId",
    "OptimizationMethod",
    "OptimizationSolution", 
    "OptimizationIteration",
    "OptimizationVariable",
    "Constraint",
    "ConvergenceCriteria",
]