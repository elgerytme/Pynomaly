"""
CLI utility for the shared package.
"""

from __future__ import annotations

import sys
from typing import List

from .types import ValidationResult, ValidationError
from .value_objects import Email, Identifier
from .utils import ValidationUtils


def validate() -> None:
    """
    CLI command to validate common value objects.
    
    Usage: shared-validate
    """
    print("Shared Package Validation Utility")
    print("=" * 40)
    
    # Test value objects
    test_cases = [
        ("Email", "test@example.com", lambda x: Email(x)),
        ("Email", "invalid-email", lambda x: Email(x)),
        ("Identifier", "user-123", lambda x: Identifier(x)),
        ("Identifier", "", lambda x: Identifier(x)),
    ]
    
    validator = ValidationUtils()
    all_passed = True
    
    for type_name, value, constructor in test_cases:
        try:
            result = constructor(value)
            print(f"✅ {type_name}('{value}') -> {result}")
        except ValueError as e:
            print(f"❌ {type_name}('{value}') -> Error: {e}")
            if value in ["test@example.com", "user-123"]:  # Expected to pass
                all_passed = False
    
    # Test validation utilities
    print("\nValidation Utilities Test:")
    print("-" * 25)
    
    validation_tests = [
        ("Email validation", "valid@test.com", validator.validate_email),
        ("Email validation", "invalid-email", validator.validate_email),
    ]
    
    for test_name, value, validator_func in validation_tests:
        result = validator_func(value)
        status = "✅" if result.is_valid else "❌"
        print(f"{status} {test_name}: '{value}' -> Valid: {result.is_valid}")
        if not result.is_valid:
            for error in result.errors:
                print(f"   Error: {error.message}")
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some validation tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    validate()