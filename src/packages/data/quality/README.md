# Data Quality Package

A comprehensive, self-contained package for data quality assessment, validation, and remediation capabilities.

## Overview

This package provides comprehensive data quality capabilities including validation rules, quality scoring, data cleansing, and quality monitoring for enterprise data pipelines.

## Features

- **Quality Assessment**: Comprehensive data quality scoring and metrics
- **Validation Rules**: Configurable validation rules and constraints
- **Data Cleansing**: Automated data cleaning and standardization
- **Quality Monitoring**: Continuous quality monitoring and alerting
- **Rule Engine**: Flexible rule engine for custom quality checks
- **Error Handling**: Robust error handling with comprehensive logging
- **Monitoring**: Built-in metrics and observability
- **Scalability**: Horizontal scaling support for high-volume data

## Architecture

```
src/quality/
├── domain/                 # Core business logic
├── application/           # Use cases and orchestration  
├── infrastructure/        # External integrations and persistence
└── presentation/          # APIs and CLI interfaces
```

## Quick Start

```python
from quality import QualityService

# Initialize the service
service = QualityService()

# Assess data quality
quality_report = service.assess_quality(dataset)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
# Run all tests
python scripts/test.sh

# Run specific test types
pytest tests/unit/
pytest tests/integration/
```

## Documentation

See the [docs/](docs/) directory for comprehensive documentation including:
- API documentation
- User guides
- Architecture documentation
- Integration examples

## License

MIT License