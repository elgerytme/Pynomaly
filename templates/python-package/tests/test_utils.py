"""Tests for utility functions."""

import pytest
from my_package.utils import validate_input, format_output


class TestValidation:
    """Test cases for validation functions."""
    
    def test_validate_input_type(self):
        """Test type validation."""
        # Valid types
        assert validate_input("hello", str) is True
        assert validate_input(42, int) is True
        assert validate_input(3.14, float) is True
        assert validate_input([1, 2, 3], list) is True
        
        # Invalid types
        assert validate_input("hello", int) is False
        assert validate_input(42, str) is False
        assert validate_input(3.14, int) is False
    
    def test_validate_input_min_value(self):
        """Test minimum value validation."""
        # Valid minimum values
        assert validate_input(10, int, min_value=5) is True
        assert validate_input(10.5, float, min_value=10.0) is True
        assert validate_input(5, int, min_value=5) is True  # Equal to minimum
        
        # Invalid minimum values
        assert validate_input(3, int, min_value=5) is False
        assert validate_input(9.5, float, min_value=10.0) is False
    
    def test_validate_input_non_numeric_with_min(self):
        """Test non-numeric types with minimum value (should ignore min_value)."""
        # String with min_value should ignore min_value and only check type
        assert validate_input("hello", str, min_value=5) is True
        assert validate_input([1, 2, 3], list, min_value=5) is True


class TestFormatting:
    """Test cases for formatting functions."""
    
    def test_format_output_json(self):
        """Test JSON formatting."""
        data = {"name": "test", "value": 42}
        result = format_output(data, "json")
        
        # Should be valid JSON
        import json
        parsed = json.loads(result)
        assert parsed == data
    
    def test_format_output_csv_dict_list(self):
        """Test CSV formatting with list of dictionaries."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        result = format_output(data, "csv")
        
        expected_lines = [
            "name,age",
            "Alice,30",
            "Bob,25"
        ]
        assert result == "\n".join(expected_lines)
    
    def test_format_output_csv_simple_list(self):
        """Test CSV formatting with simple list."""
        data = [1, 2, 3, 4, 5]
        result = format_output(data, "csv")
        assert result == "1,2,3,4,5"
    
    def test_format_output_csv_single_dict(self):
        """Test CSV formatting with single dictionary."""
        data = {"name": "Alice", "age": 30}
        result = format_output(data, "csv")
        
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "name,age"
        assert lines[1] == "Alice,30"
    
    def test_format_output_table(self):
        """Test table formatting."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        result = format_output(data, "table")
        
        # Should contain headers and data
        lines = result.split("\n")
        assert len(lines) >= 4  # Header, separator, and data rows
        assert "name" in lines[0]
        assert "age" in lines[0]
        assert "Alice" in result
        assert "Bob" in result
    
    def test_format_output_default(self):
        """Test default formatting (string conversion)."""
        data = {"test": "value"}
        result = format_output(data, "unknown")
        assert result == str(data)
    
    def test_format_output_empty_list(self):
        """Test formatting empty list."""
        result = format_output([], "csv")
        assert result == ""
    
    def test_format_output_mixed_types(self):
        """Test formatting with mixed data types."""
        data = [
            {"name": "Alice", "score": 95.5, "active": True},
            {"name": "Bob", "score": 87.2, "active": False}
        ]
        result = format_output(data, "csv")
        
        lines = result.split("\n")
        assert "name,score,active" in lines[0]
        assert "Alice,95.5,True" in lines[1]
        assert "Bob,87.2,False" in lines[2]