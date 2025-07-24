# Data Profiling Package

A comprehensive, self-contained package for data profiling and statistical analysis capabilities.

## Overview

This package provides comprehensive data profiling capabilities including data quality assessment, statistical analysis, schema inference, and data characterization.

## Features

- **Statistical Profiling**: Comprehensive statistical analysis of datasets
- **Schema Inference**: Automatic schema detection and validation
- **Data Quality Assessment**: Built-in data quality metrics and scoring
- **Pattern Detection**: Identification of data patterns and anomalies
- **Column Analysis**: Detailed column-level profiling and insights
- **Error Handling**: Robust error handling with comprehensive logging
- **Monitoring**: Built-in metrics and observability
- **Scalability**: Horizontal scaling support for large datasets

## Architecture

```
src/profiling/
├── domain/                 # Core business logic
├── application/           # Use cases and orchestration  
├── infrastructure/        # External integrations and persistence
└── presentation/          # APIs and CLI interfaces
```

## Quick Start

```python
from profiling import ProfilingService

# Initialize the service
service = ProfilingService()

# Profile a dataset
profile = service.profile_data(dataset)
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