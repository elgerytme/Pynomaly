"""Calculator module for mathematical operations."""

from typing import Union

Number = Union[int, float]


class Calculator:
    """A simple calculator class demonstrating package structure."""
    
    def add(self, x: Number, y: Number) -> Number:
        """Add two numbers.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Sum of x and y
        """
        return x + y
    
    def subtract(self, x: Number, y: Number) -> Number:
        """Subtract two numbers.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Difference of x and y
        """
        return x - y
    
    def multiply(self, x: Number, y: Number) -> Number:
        """Multiply two numbers.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Product of x and y
        """
        return x * y
    
    def divide(self, x: Number, y: Number) -> Number:
        """Divide two numbers.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Quotient of x and y
            
        Raises:
            ValueError: If y is zero
        """
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
    
    def power(self, x: Number, y: Number) -> Number:
        """Raise x to the power of y.
        
        Args:
            x: Base number
            y: Exponent
            
        Returns:
            x raised to the power of y
        """
        return x ** y