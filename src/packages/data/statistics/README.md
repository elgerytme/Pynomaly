# Statistics Package

A comprehensive, self-contained package for Statistical analysis and mathematical operations.

## Overview

This package provides comprehensive statistical analysis and mathematical operations for data science applications.

## Features

- **Statistical Methods: Comprehensive statistical analysis methods**
- **Mathematical Operations: Advanced mathematical and numerical operations**
- **Hypothesis Testing: Statistical hypothesis testing framework**
- **Probability Distributions: Support for various probability distributions**
- **Bayesian Analysis: Bayesian statistical analysis capabilities**
- **Error Handling**: Robust error handling with comprehensive logging
- **Monitoring**: Built-in metrics and observability
- **Scalability**: Horizontal scaling support for high-volume operations

## Architecture

```
src/statistics/
├── domain/                 # Core business logic
├── application/           # Use cases and orchestration  
├── infrastructure/        # External integrations and persistence
└── presentation/          # APIs and CLI interfaces
```

## Quick Start

```python
from statistics import StatisticsService

# Initialize the service
service = StatisticsService()

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