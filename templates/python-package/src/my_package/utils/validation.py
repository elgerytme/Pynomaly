"""Input validation utilities."""

from typing import Any, Union


def validate_input(value: Any, expected_type: type, min_value: Union[int, float, None] = None) -> bool:
    """Validate input value against expected type and constraints.
    
    Args:
        value: The value to validate
        expected_type: The expected type
        min_value: Optional minimum value for numeric types
        
    Returns:
        True if validation passes, False otherwise
    """
    # Check type
    if not isinstance(value, expected_type):
        return False
    
    # Check minimum value for numeric types
    if min_value is not None and isinstance(value, (int, float)):
        if value < min_value:
            return False
    
    return True


def validate_non_empty_string(value: str) -> bool:
    """Validate that a string is not empty.
    
    Args:
        value: The string to validate
        
    Returns:
        True if string is not empty, False otherwise
    """
    return isinstance(value, str) and len(value.strip()) > 0


def validate_numeric_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
    """Validate that a numeric value is within a specified range.
    
    Args:
        value: The numeric value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        True if value is within range, False otherwise
    """
    if not isinstance(value, (int, float)):
        return False
    
    return min_val <= value <= max_val


def validate_list_length(value: list, min_length: int = 0, max_length: int = float('inf')) -> bool:
    """Validate that a list has a length within specified bounds.
    
    Args:
        value: The list to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        True if list length is within bounds, False otherwise
    """
    if not isinstance(value, list):
        return False
    
    return min_length <= len(value) <= max_length