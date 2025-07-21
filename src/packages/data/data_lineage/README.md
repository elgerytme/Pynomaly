# Data Lineage Package

A comprehensive, self-contained package for Data lineage tracking and governance capabilities.

## Overview

This package provides comprehensive data lineage tracking, governance, and metadata management capabilities.

## Features

- **Lineage Tracking: Automatic tracking of data flow and transformations**
- **Metadata Management: Rich metadata capture and storage**
- **Impact Analysis: Understand downstream effects of data changes**
- **Compliance Support: Data governance and regulatory compliance features**
- **Visualization: Interactive lineage graphs and dependency maps**
- **Error Handling**: Robust error handling with comprehensive logging
- **Monitoring**: Built-in metrics and observability
- **Scalability**: Horizontal scaling support for high-volume operations

## Architecture

```
src/data_lineage/
├── domain/                 # Core business logic
├── application/           # Use cases and orchestration  
├── infrastructure/        # External integrations and persistence
└── presentation/          # APIs and CLI interfaces
```

## Quick Start

```python
from data_lineage import DatalineageService

# Initialize the service
service = DatalineageService()

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