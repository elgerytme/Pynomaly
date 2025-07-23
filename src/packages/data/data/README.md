# 📊 Data Domain Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0+-green.svg)](https://pydantic.dev/)

Core data package for data management functionality

## 🎯 Overview

This package contains the domain logic for data functionality following Domain-Driven Design (DDD) principles.

**Key Features:**
- **🏗️ Clean Architecture**: Strict separation of domain, application, infrastructure, and interface layers
- **📋 Domain-Driven Design**: Rich domain models with ubiquitous language
- **🔒 Type Safety**: Comprehensive type coverage with Pydantic models
- **✅ Domain Compliance**: Automated boundary validation and enforcement
- **🧪 Testing**: Multi-layered testing with high coverage

## 🚀 Quick Start

### Installation

```bash
# Install the data package
cd src/packages/data/data/
pip install -e .

# Install with development dependencies
pip install -e ".[dev,test]"
```

### Basic Usage

```python
# Import core domain entities
from data.core.domain.entities import BaseDataEntity
from data.core.domain.value_objects import DataIdentifier
from data.core.application.services import DataService

# Create a data identifier
data_id = DataIdentifier("user_data_001")

# Use the data service
data_service = DataService()
result = data_service.process_data(data_id)

print(f"Data processed: {result}")
```

## 📚 Usage Examples

### Working with Domain Entities

```python
from data.core.domain.entities import DataRecord
from data.core.domain.value_objects import DataType, DataTimestamp
from datetime import datetime

# Create a data record
record = DataRecord(
    id=DataIdentifier("record_001"),
    data_type=DataType.STRUCTURED,
    content={"key": "value", "count": 42},
    timestamp=DataTimestamp(datetime.now()),
    metadata={"source": "api", "version": "1.0"}
)

print(f"Record ID: {record.id}")
print(f"Data Type: {record.data_type}")
print(f"Content: {record.content}")
```

### Using Application Services

```python
from data.core.application.services import DataProcessingService
from data.core.application.use_cases import ProcessDataUseCase

# Initialize services
processing_service = DataProcessingService()
use_case = ProcessDataUseCase(processing_service)

# Process data through use case
result = use_case.execute(
    data_id="record_001",
    processing_options={
        "validate": True,
        "transform": True,
        "persist": True
    }
)

print(f"Processing result: {result.status}")
print(f"Processed items: {result.item_count}")
```

### Repository Pattern

```python
from data.core.domain.repositories import DataRepositoryInterface
from data.infrastructure.persistence import SqlDataRepository

# Use repository through interface
repository: DataRepositoryInterface = SqlDataRepository()

# Store data
record = DataRecord(...)
saved_record = repository.save(record)

# Retrieve data
retrieved_record = repository.find_by_id(saved_record.id)

# Query data
records = repository.find_by_criteria({
    "data_type": DataType.STRUCTURED,
    "created_after": datetime.now() - timedelta(days=7)
})
```

### Error Handling

```python
from data.core.domain.exceptions import DataValidationError, DataNotFoundError

try:
    # Process data that might fail
    result = data_service.process_data(invalid_data_id)
except DataValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Field errors: {e.field_errors}")
except DataNotFoundError as e:
    print(f"Data not found: {e.data_id}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Structure

```
data/
├── core/
│   ├── domain/              # Domain layer
│   │   ├── entities/        # Domain entities
│   │   ├── services/        # Domain services
│   │   ├── value_objects/   # Value objects
│   │   ├── repositories/    # Repository interfaces
│   │   └── exceptions/      # Domain exceptions
│   ├── application/         # Application layer
│   │   ├── services/        # Application services
│   │   └── use_cases/       # Use cases
│   └── dto/                 # Data transfer objects
├── infrastructure/          # Infrastructure layer
│   ├── adapters/           # External adapters
│   ├── persistence/        # Data persistence
│   └── external/           # External services
├── interfaces/             # Interface layer
│   ├── api/               # REST API endpoints
│   ├── cli/               # Command-line interface
│   ├── web/               # Web interface
│   └── python_sdk/        # Python SDK
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── docs/                   # Documentation
```

## Domain Boundaries

This package follows strict domain boundaries:

### Allowed Concepts
- Data-specific business logic
- Domain entities and value objects
- Domain services and repositories
- Use cases and application services

### Prohibited Concepts
- Generic software infrastructure (belongs in `software/` package)
- Other domain concepts (belongs in respective domain packages)
- Cross-domain references (use interfaces and dependency injection)

## Installation

```bash
pip install data
```

## Development

### Setup
```bash
# Clone repository
git clone https://github.com/monorepo/anomaly-detection-platform.git
cd anomaly-detection-platform

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
python scripts/install_domain_hooks.py
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run domain boundary validation
python scripts/domain_boundary_validator.py
```

### Code Quality
```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

This package follows Clean Architecture principles:

1. **Domain Layer**: Core business logic
2. **Application Layer**: Use cases and application services
3. **Infrastructure Layer**: External concerns
4. **Interface Layer**: User interfaces

## Domain Compliance

This package maintains strict domain boundary compliance:

- **Validation**: Automated domain boundary validation
- **Enforcement**: Pre-commit hooks and CI/CD integration
- **Monitoring**: Continuous compliance monitoring
- **Documentation**: Clear domain boundary rules

## 🔧 API Reference

### Core Domain Classes

#### BaseDataEntity
Base class for all data domain entities.

```python
class BaseDataEntity:
    def __init__(self, id: DataIdentifier):
        """Initialize entity with unique identifier."""
        
    def __eq__(self, other: object) -> bool:
        """Entity equality based on identifier."""
        
    def __hash__(self) -> int:
        """Entity hash based on identifier."""
```

#### DataRecord
Primary entity for data records.

```python
class DataRecord(BaseDataEntity):
    def __init__(
        self,
        id: DataIdentifier,
        data_type: DataType,
        content: Dict[str, Any],
        timestamp: DataTimestamp,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize data record."""
        
    def validate(self) -> None:
        """Validate data record integrity."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
```

#### DataService
Main application service for data operations.

```python
class DataService:
    def process_data(self, data_id: DataIdentifier) -> ProcessingResult:
        """Process data by identifier."""
        
    def validate_data(self, data: DataRecord) -> ValidationResult:
        """Validate data record."""
        
    def transform_data(self, data: DataRecord, rules: TransformationRules) -> DataRecord:
        """Transform data using specified rules."""
```

## 🔍 Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError` when importing data components
```bash
# Solution: Ensure package is properly installed
cd src/packages/data/data/
pip install -e .

# Verify installation
python -c "import data; print('Data package installed successfully')"
```

#### Domain Boundary Violations

**Problem**: Domain boundary validation fails
```bash
# Solution: Run boundary validator to identify issues
python scripts/domain_boundary_validator.py

# Fix violations by moving code to appropriate layers
# Domain layer: Pure business logic only
# Application layer: Use cases and orchestration
# Infrastructure layer: External integrations
# Interface layer: User interfaces and APIs
```

#### Type Checking Errors

**Problem**: MyPy type checking fails
```bash
# Solution: Fix type annotations
mypy src/ --strict

# Common fixes:
# 1. Add return type annotations
# 2. Use proper generic types
# 3. Handle Optional types correctly
```

### FAQ

**Q: How do I add a new domain entity?**
A: Create the entity in `core/domain/entities/` following the BaseDataEntity pattern:

```python
from data.core.domain.entities import BaseDataEntity
from data.core.domain.value_objects import DataIdentifier

class MyEntity(BaseDataEntity):
    def __init__(self, id: DataIdentifier, ...):
        super().__init__(id)
        # Your entity logic here
```

**Q: How do I implement a new repository?**
A: Implement the repository interface in the infrastructure layer:

```python
from data.core.domain.repositories import DataRepositoryInterface
from data.core.domain.entities import DataRecord

class MyDataRepository(DataRepositoryInterface):
    def save(self, record: DataRecord) -> DataRecord:
        # Implementation here
        pass
        
    def find_by_id(self, id: DataIdentifier) -> Optional[DataRecord]:
        # Implementation here
        pass
```

**Q: How do I add validation rules?**
A: Add validation in the domain layer using Pydantic validators:

```python
from pydantic import validator
from data.core.domain.entities import BaseDataEntity

class ValidatedEntity(BaseDataEntity):
    @validator('field_name')
    def validate_field(cls, v):
        if not v:
            raise ValueError('Field cannot be empty')
        return v
```

## Contributing

1. Follow domain boundary rules
2. Add comprehensive tests
3. Update documentation
4. Validate domain compliance
5. Submit pull request

## License

MIT License - see LICENSE file for details.
