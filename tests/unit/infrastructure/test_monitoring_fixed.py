"""
Fixed unit tests for monitoring infrastructure.
Tests that demonstrate fixed code quality issues and improved coverage.
"""

import pytest
from unittest.mock import Mock, patch

from pynomaly.infrastructure.monitoring.distributed_tracing import (
    trace_operation,
    start_span,
    end_span,
    add_span_attribute,
    set_span_error,
)


class TestDistributedTracingFixed:
    """Tests for the fixed distributed tracing module."""

    def test_trace_operation_decorator_basic(self):
        """Test trace_operation decorator with basic functionality."""
        @trace_operation("test_operation")
        def test_function(x: int, y: int) -> int:
            return x + y

        result = test_function(3, 4)
        assert result == 7

    def test_trace_operation_with_kwargs(self):
        """Test trace_operation decorator with keyword arguments."""
        @trace_operation("test_operation", metadata={"key": "value"})
        def test_function(name: str) -> str:
            return f"Hello, {name}!"

        result = test_function("World")
        assert result == "Hello, World!"

    def test_trace_operation_with_exception(self):
        """Test trace_operation decorator handles exceptions properly."""
        @trace_operation("failing_operation")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_start_span_no_op(self):
        """Test start_span is a no-op function."""
        # Should not raise any errors
        start_span("test_span")
        start_span("test_span", attribute="value")

    def test_end_span_no_op(self):
        """Test end_span is a no-op function."""
        # Should not raise any errors
        end_span()

    def test_add_span_attribute_no_op(self):
        """Test add_span_attribute is a no-op function."""
        # Should not raise any errors
        add_span_attribute("key", "value")
        add_span_attribute("number", 42)
        add_span_attribute("list", [1, 2, 3])

    def test_set_span_error_no_op(self):
        """Test set_span_error is a no-op function."""
        # Should not raise any errors
        error = ValueError("Test error")
        set_span_error(error)

    def test_trace_operation_preserves_function_metadata(self):
        """Test that trace_operation preserves function metadata."""
        @trace_operation("documented_operation")
        def documented_function(x: int) -> int:
            """This is a documented function."""
            return x * 2

        assert documented_function.__name__ == "documented_function"
        assert "documented function" in documented_function.__doc__

    def test_trace_operation_with_async_function(self):
        """Test trace_operation works with async functions."""
        @trace_operation("async_operation")
        async def async_function(value: str) -> str:
            return f"async_{value}"

        import asyncio
        
        result = asyncio.run(async_function("test"))
        assert result == "async_test"

    def test_multiple_nested_trace_operations(self):
        """Test multiple nested trace operations."""
        @trace_operation("outer_operation")
        def outer_function(x: int) -> int:
            return inner_function(x)

        @trace_operation("inner_operation")
        def inner_function(x: int) -> int:
            return x * 3

        result = outer_function(5)
        assert result == 15

    def test_trace_operation_with_complex_return_types(self):
        """Test trace_operation with complex return types."""
        @trace_operation("complex_operation")
        def complex_function() -> dict[str, list[int]]:
            return {"numbers": [1, 2, 3], "more_numbers": [4, 5, 6]}

        result = complex_function()
        assert result == {"numbers": [1, 2, 3], "more_numbers": [4, 5, 6]}
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())