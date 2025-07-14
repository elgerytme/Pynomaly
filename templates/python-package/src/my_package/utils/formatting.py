"""Output formatting utilities."""

import json
from datetime import datetime
from typing import Any, Dict, List, Union


def format_output(value: Any, format_type: str = "json") -> str:
    """Format output value according to specified format.
    
    Args:
        value: The value to format
        format_type: The format type ("json", "csv", "table")
        
    Returns:
        Formatted string representation
    """
    if format_type == "json":
        return json.dumps(value, indent=2, default=str)
    elif format_type == "csv":
        return _format_as_csv(value)
    elif format_type == "table":
        return _format_as_table(value)
    else:
        return str(value)


def _format_as_csv(value: Any) -> str:
    """Format value as CSV string."""
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            # List of dictionaries
            if not value:
                return ""
            
            headers = list(value[0].keys())
            lines = [",".join(headers)]
            
            for item in value:
                row = [str(item.get(header, "")) for header in headers]
                lines.append(",".join(row))
            
            return "\n".join(lines)
        else:
            # Simple list
            return ",".join(str(item) for item in value)
    elif isinstance(value, dict):
        # Single dictionary
        headers = list(value.keys())
        values = [str(value[header]) for header in headers]
        return ",".join(headers) + "\n" + ",".join(values)
    else:
        return str(value)


def _format_as_table(value: Any) -> str:
    """Format value as ASCII table."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        # List of dictionaries
        headers = list(value[0].keys())
        
        # Calculate column widths
        widths = {header: len(header) for header in headers}
        for item in value:
            for header in headers:
                widths[header] = max(widths[header], len(str(item.get(header, ""))))
        
        # Build table
        lines = []
        
        # Header
        header_line = " | ".join(header.ljust(widths[header]) for header in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for item in value:
            row_line = " | ".join(str(item.get(header, "")).ljust(widths[header]) for header in headers)
            lines.append(row_line)
        
        return "\n".join(lines)
    else:
        return str(value)


def format_timestamp(timestamp: datetime = None) -> str:
    """Format timestamp in ISO format.
    
    Args:
        timestamp: The timestamp to format (defaults to current time)
        
    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.isoformat()


def format_number(value: Union[int, float], decimal_places: int = 2) -> str:
    """Format number with specified decimal places.
    
    Args:
        value: The number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return f"{value:.{decimal_places}f}"
    else:
        return str(value)


def format_percentage(value: Union[int, float], decimal_places: int = 1) -> str:
    """Format number as percentage.
    
    Args:
        value: The value to format (0.0 to 1.0)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"