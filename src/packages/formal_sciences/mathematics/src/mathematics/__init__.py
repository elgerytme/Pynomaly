"""
Mathematics Package

This package provides comprehensive mathematical functions and utilities including:
- Basic arithmetic operations
- Linear algebra operations
- Statistical functions
- Calculus operations
- Optimization algorithms
- Numerical methods
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "support@pynomaly.com"

from typing import Any, Dict, List, Optional, Union
import math

# Basic mathematical operations
class MathOperations:
    """Basic mathematical operations."""
    
    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
        
    @staticmethod
    def subtract(a: float, b: float) -> float:
        """Subtract two numbers."""
        return a - b
        
    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
        
    @staticmethod
    def divide(a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

# Statistics functions
class Statistics:
    """Statistical functions."""
    
    @staticmethod
    def mean(data: List[float]) -> float:
        """Calculate mean of data."""
        return sum(data) / len(data)
        
    @staticmethod
    def median(data: List[float]) -> float:
        """Calculate median of data."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        else:
            return sorted_data[n // 2]

# Main mathematics facade
class Mathematics:
    """Main mathematics facade providing access to all mathematical operations."""
    
    def __init__(self):
        self.operations = MathOperations()
        self.statistics = Statistics()

# Factory function
def create_mathematics() -> Mathematics:
    """Create a mathematics instance."""
    return Mathematics()

__all__ = [
    "MathOperations",
    "Statistics",
    "Mathematics",
    "create_mathematics",
    "__version__",
    "__author__",
    "__email__",
]