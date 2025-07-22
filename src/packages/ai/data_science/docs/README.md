# Data Science Package Documentation

## Table of Contents

- [Architecture Overview](architecture.md)
- [API Reference](api.md) 
- [Development Guide](development.md)
- [Deployment Guide](deployment.md)
- [Configuration](configuration.md)
- [Examples](../examples/)

## Quick Links

- **API Documentation**: Available at `/docs` when running the server
- **Source Code**: Located in `src/data_science/`
- **Tests**: Located in `tests/`
- **Examples**: Located in `examples/`

## Architecture

This package follows Domain-Driven Design (DDD) principles:

```
src/data_science/
├── application/     # Application services and use cases
├── domain/          # Core business logic and entities  
├── infrastructure/  # External dependencies and adapters
└── presentation/    # API endpoints, CLI, and web interfaces
```

## Services Overview

### Application Services

- **IntegratedDataScienceService**: Main orchestration service
- **PerformanceDegradationMonitoringService**: Performance monitoring
- **WorkflowOrchestrationEngine**: Workflow management

### Domain Services

- **FeatureValidator**: Feature validation and analysis
- **MetricsCalculator**: Metrics calculation and comparison
- **ProcessingOrchestrator**: Data processing coordination

## Getting Help

- Check the [examples/](../examples/) directory for usage examples
- Review the API documentation at `/docs` when running the server
- See the [development guide](development.md) for contributing

## Support

For issues and questions:

1. Check existing documentation
2. Review examples and test cases
3. Open an issue in the repository