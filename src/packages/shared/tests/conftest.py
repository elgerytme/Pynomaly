"""
Shared test configuration and fixtures.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from shared.value_objects import Email, Identifier, Timestamp, Money
from shared.types import ValidationError


@pytest.fixture
def valid_email():
    """Valid email for testing."""
    return "test@example.com"


@pytest.fixture
def invalid_email():
    """Invalid email for testing."""
    return "invalid-email"


@pytest.fixture
def sample_identifier():
    """Sample identifier for testing."""
    return Identifier("test-123")


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return Timestamp(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def sample_money():
    """Sample money for testing."""
    return Money(Decimal("100.50"), "USD")


@pytest.fixture
def validation_error():
    """Sample validation error for testing."""
    return ValidationError(
        field="test_field",
        value="test_value",
        message="Test error message",
        code="TEST_ERROR"
    )