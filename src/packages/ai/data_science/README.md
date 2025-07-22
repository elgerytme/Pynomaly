# Data Science Package

A comprehensive data science package providing experiment management, feature validation, metrics calculation, and workflow orchestration capabilities.

## Features

- **Experiment Management**: Create, run, and manage data science experiments
- **Feature Validation**: Validate and analyze dataset features
- **Metrics Calculation**: Calculate and compare experiment metrics
- **Workflow Orchestration**: Orchestrate complex data science workflows
- **Performance Monitoring**: Monitor and analyze model performance degradation

## Architecture

This package follows Domain-Driven Design (DDD) principles with a clean architecture:

- **Application Layer**: Services and use cases
- **Domain Layer**: Entities, value objects, and domain services  
- **Infrastructure Layer**: External dependencies and adapters
- **Presentation Layer**: API endpoints, CLI commands, and web interfaces

## Installation

```bash
pip install data-science
```

## Quick Start

### CLI Usage

```bash
# Create a new experiment
data-science experiments create --name "my-experiment" --description "Test experiment"

# Validate features
data-science features validate --dataset /path/to/data.csv

# Calculate metrics
data-science metrics calculate --experiment exp_123
```

### API Usage

Start the server:

```bash
data-science-server
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

### Python SDK

```python
from data_science import IntegratedDataScienceService, FeatureValidator

# Initialize services
service = IntegratedDataScienceService()
validator = FeatureValidator()

# Create and run experiment
experiment = service.create_experiment(name="test", config={})
results = service.run_experiment(experiment.id)

# Validate features
validation_results = validator.validate(dataset_path="/path/to/data.csv")
```

## Configuration

Configuration can be provided via:

1. Environment variables
2. Configuration files (JSON/YAML)
3. CLI arguments

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/data_science --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m performance
```

## Development

### Setup Development Environment

```bash
# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code  
ruff check src/ tests/
mypy src/

# Security scan
bandit -r src/
```

## Documentation

- [API Documentation](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Examples](examples/)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.