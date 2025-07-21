# Pynomaly Data Package

Core data domain package providing foundational data concepts and models for the Pynomaly ecosystem.

## Overview

This package defines the core data domain entities, value objects, and services that represent fundamental data concepts such as data origins, sources, assets, datasets, elements, schemas, and types. It follows Domain-Driven Design (DDD) principles and provides a rich domain model for data governance, lineage, and management.

## Key Concepts

### Domain Entities

- **DataOrigin**: Represents the source or origin of data (database, file system, API, etc.)
- **DataSource**: Specific data source within an origin with access patterns and quality metrics
- **DataAsset**: Logical data asset that can contain multiple datasets with business context
- **DataSet**: Collection of structured data with schema, quality metrics, and validation
- **DataElement**: Individual data field or element within a dataset with type and constraints

### Value Objects

- **DataType**: Comprehensive data type definitions with constraints and validation
- **DataSchema**: Complete schema definitions with fields, relationships, and constraints
- **DataClassification**: Data sensitivity, compliance, and quality requirements

### Domain Services

- **DataValidationService**: Validates data against schemas and business rules
- **DataSchemaService**: Schema generation, evolution, merging, and comparison

## Features

- **Rich Domain Model**: Comprehensive entities with business behavior and validation
- **Type Safety**: Strong typing throughout with Pydantic validation
- **Data Quality**: Built-in quality scoring and metrics tracking
- **Schema Management**: Automatic schema inference, evolution, and validation
- **Data Classification**: Security levels, compliance tags, and access controls
- **Lineage Tracking**: Upstream dependencies and downstream consumers
- **Extensible Design**: Rich metadata support for various data contexts

## Installation

```bash
pip install pynomaly-data
```

## Basic Usage

### Creating Data Types

```python
from packages.data.data.domain.value_objects import DataType, PrimitiveDataType

# Simple string type
string_type = DataType(
    primitive_type=PrimitiveDataType.STRING,
    max_length=255,
    nullable=False
)

# Numeric type with constraints
numeric_type = DataType(
    primitive_type=PrimitiveDataType.FLOAT,
    precision=10,
    scale=2,
    nullable=True
)
```

### Working with Schemas

```python
from packages.data.data.domain.value_objects import DataSchema, DataFieldSchema
from packages.data.data.domain.services import DataSchemaService

# Create schema from sample data
schema_service = DataSchemaService()
sample_data = [
    {"id": 1, "name": "John", "email": "john@example.com"},
    {"id": 2, "name": "Jane", "email": "jane@example.com"}
]

schema = schema_service.generate_schema_from_data(sample_data, "users_schema")
```

### Data Validation

```python
from packages.data.data.domain.services import DataValidationService

validation_service = DataValidationService()
validation_results = validation_service.validate_dataset_against_schema(
    dataset, 
    actual_data
)

if validation_results['is_valid']:
    print("Data validation passed!")
else:
    print(f"Validation issues: {validation_results['schema_violations']}")
```

### Creating Data Assets

```python
from packages.data.data.domain.entities import DataAsset, AssetType
from packages.data.data.domain.value_objects import DataClassification, DataSensitivityLevel

# Create a data asset
asset = DataAsset(
    name="customer_data",
    asset_type=AssetType.TABLE,
    description="Customer information and profiles",
    owner="data-team@company.com",
    business_purpose="Customer analytics and reporting",
    classification=DataClassification(
        sensitivity_level=DataSensitivityLevel.CONFIDENTIAL,
        compliance_tags=[DataComplianceTag.PII, DataComplianceTag.GDPR]
    )
)
```

## Architecture

The package follows clean architecture principles with clear separation of concerns:

```
domain/
├── entities/          # Core business entities
├── value_objects/     # Immutable value objects
├── services/          # Domain services
└── repositories/      # Repository interfaces
```

## Quality Standards

- **Test Coverage**: Minimum 80% required
- **Type Safety**: Full type annotations with mypy validation
- **Code Quality**: Black formatting and Ruff linting
- **Documentation**: Comprehensive docstrings and examples

## Contributing

1. Follow the established DDD patterns and conventions
2. Ensure all tests pass with minimum 80% coverage
3. Use type annotations throughout
4. Include comprehensive docstrings
5. Follow the existing code style (Black + Ruff)

## Dependencies

- **pydantic**: Data validation and settings management
- **python**: 3.11+ required

## License

MIT License - see LICENSE file for details.