# Data Quality Package

A comprehensive, self-contained package for Data quality assessment and validation framework.

## Overview

This package provides comprehensive data quality assessment, validation, and profiling capabilities.

## Features

- **Quality Profiling: Automated data quality assessment and profiling**
- **Validation Rules: Flexible validation rule engine**
- **Anomaly Detection: Statistical anomaly detection in data quality**
- **Quality Scoring: Comprehensive data quality scoring and reporting**
- **Remediation: Automated data quality issue remediation**
- **Error Handling**: Robust error handling with comprehensive logging
- **Monitoring**: Built-in metrics and observability
- **Scalability**: Horizontal scaling support for high-volume operations

## Architecture

```
src/data_quality/
├── domain/                 # Core business logic
├── application/           # Use cases and orchestration  
├── infrastructure/        # External integrations and persistence
└── presentation/          # APIs and CLI interfaces
```

## Quick Start

```python
from data_quality import DataqualityService

# Initialize the service
service = DataqualityService()

# Use the service
result = service.process(data)
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