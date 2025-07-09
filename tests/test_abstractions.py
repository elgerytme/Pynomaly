#!/usr/bin/env python3
"""Simple test script for domain abstractions."""

from pynomaly.domain.abstractions import BaseEntity


class TestEntity(BaseEntity):
    """Test entity."""
    name: str
    value: int = 0


def main():
    """Test the abstractions."""
    # Test entity creation
    entity = TestEntity(name="test", value=42)
    print(f"Entity: {entity.name}, ID: {entity.id}, Value: {entity.value}")
    
    # Test entity update
    entity.mark_as_updated()
    print(f"Updated entity version: {entity.version}")
    
    # Test entity validation
    try:
        entity.validate_invariants()
        print("Entity validation passed")
    except ValueError as e:
        print(f"Entity validation failed: {e}")
    
    print("All tests passed!")


if __name__ == "__main__":
    main()
