# Data Ingestion Package

A comprehensive, self-contained package for data ingestion and collection capabilities.

## Overview

This package provides a complete solution for ingesting data from various sources including databases, APIs, files, streaming platforms, and cloud services. It follows Domain-Driven Design principles and provides a clean architecture for scalable data ingestion operations.

## Features

- **Multi-source Data Ingestion**: Support for databases, REST APIs, files, message queues, and cloud services
- **Streaming and Batch Processing**: Real-time and scheduled data ingestion capabilities
- **Data Validation**: Built-in data quality checks during ingestion
- **Schema Management**: Automatic schema detection and validation
- **Error Handling**: Robust error handling with retry mechanisms
- **Monitoring**: Built-in metrics and observability for ingestion processes
- **Scalability**: Horizontal scaling support for high-volume data ingestion

## Architecture

```
src/data_ingestion/
├── domain/                 # Core business logic for data ingestion
├── application/           # Use cases and orchestration
├── infrastructure/        # External integrations and persistence
└── presentation/          # APIs and CLI interfaces
```

## Quick Start

```python
from data_ingestion import DataIngestionService

# Initialize the ingestion service
service = DataIngestionService()

# Ingest data from a source
result = service.ingest_from_source("my_database", "users_table")
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