# Shared Package

Common utilities, types, and cross-cutting concerns for the monorepo.

## Overview

The shared package provides foundational components that are used across multiple domains in the monorepo. It implements common patterns, utilities, and types that enable consistent behavior and reduce code duplication.

## Core Components

### Value Objects
- `Identifier`: Base class for strongly-typed identifiers
- `Email`: Email address validation and handling
- `Timestamp`: Consistent timestamp handling across domains
- `Money`: Financial value representations with currency support

### Common Types
- `Result<T, E>`: Type-safe error handling pattern
- `Optional<T>`: Explicit null handling
- `Paginated<T>`: Standardized pagination responses
- `ValidationResult`: Input validation responses

### Utilities
- `DateTimeUtils`: Date/time manipulation and formatting
- `ValidationUtils`: Common validation patterns
- `SerializationUtils`: JSON and other serialization helpers
- `LoggingUtils`: Structured logging helpers

### Base Classes
- `Entity`: Base class for domain entities
- `ValueObject`: Base class for value objects
- `DomainEvent`: Base class for domain events
- `UseCase`: Base class for application use cases

## Usage Examples

### Using Result Pattern
```python
from shared.types import Result, Success, Failure

def divide(a: float, b: float) -> Result[float, str]:
    if b == 0:
        return Failure("Division by zero")
    return Success(a / b)

result = divide(10, 2)
if result.is_success():
    print(f"Result: {result.value}")
else:
    print(f"Error: {result.error}")
```

### Using Value Objects
```python
from shared.value_objects import Email, Identifier

# Type-safe email handling
email = Email("user@example.com")
print(email.domain)  # "example.com"

# Strong typing for IDs
user_id = Identifier("user-123")
product_id = Identifier("product-456")
# user_id == product_id  # Type checker will warn about this
```

### Using Validation
```python
from shared.utils import ValidationUtils
from shared.types import ValidationResult

validator = ValidationUtils()
result: ValidationResult = validator.validate_email("invalid-email")

if not result.is_valid:
    print(f"Validation errors: {result.errors}")
```

## Architecture Principles

### Domain Isolation
The shared package is designed to be domain-agnostic and contains no business logic specific to any particular domain (AI, Data, Enterprise, etc.).

### Dependency Direction
- Shared package has no dependencies on domain packages
- Domain packages may depend on shared package
- Follows the Dependency Inversion Principle

### Immutability
- Value objects are immutable by default
- Utilities are stateless where possible
- Thread-safe implementations throughout

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shared --cov-report=html

# Run only unit tests
pytest -m unit

# Run performance tests
pytest -m performance
```

## Contributing

1. Follow the established patterns for value objects and utilities
2. Maintain domain independence - no business logic allowed
3. Add comprehensive tests for all new functionality
4. Update documentation for public APIs
5. Ensure backward compatibility or provide migration guides

## API Documentation

For detailed API documentation, see:
- [Value Objects API](docs/api/value_objects.md)
- [Types API](docs/api/types.md)
- [Utilities API](docs/api/utils.md)
- [Base Classes API](docs/api/base_classes.md)