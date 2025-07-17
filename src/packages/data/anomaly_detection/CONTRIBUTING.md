# Contributing to Anomaly Detection Package

Thank you for your interest in contributing to the Anomaly Detection package! This document provides guidelines for contributing to this specific package within the Pynomaly monorepo.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Package-Specific Guidelines](#package-specific-guidelines)

## Getting Started

The Anomaly Detection package is the core domain package for anomaly detection algorithms and services. It follows clean architecture principles with clear separation between domain, application, and infrastructure layers.

### Package Structure
```
data/anomaly_detection/
├── src/monorepo_detection/     # Main package code
│   ├── domain/                 # Domain entities and business logic
│   ├── application/            # Application services and use cases
│   └── infrastructure/         # Infrastructure adapters
├── tests/                      # Test suite
├── docs/                       # Package documentation
├── examples/                   # Usage examples
└── scripts/                    # Development scripts
```

## Development Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/your-org/pynomaly.git
   cd pynomaly
   ```

2. **Navigate to the package**:
   ```bash
   cd src/packages/data/anomaly_detection
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .[dev,test]
   ```

4. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Code Style

### Python Code Standards
- **PEP 8 compliance**: Use Black for formatting
- **Type hints**: All functions must include type hints
- **Docstrings**: Use Google-style docstrings
- **Import organization**: Use isort for import sorting

### Architecture Guidelines
- **Clean Architecture**: Follow domain-driven design principles
- **Single Responsibility**: Each class/function has one responsibility
- **Dependency Inversion**: Depend on abstractions, not concretions
- **Interface Segregation**: Keep interfaces focused and minimal

### Code Quality Tools
Run these commands before submitting:
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
mypy src/

# Run tests
pytest tests/
```

## Testing

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark algorithm performance
- **Property Tests**: Use Hypothesis for property-based testing

### Test Requirements
- **Coverage**: Maintain >90% test coverage
- **Test Naming**: Use descriptive test names
- **Test Structure**: Follow Arrange-Act-Assert pattern
- **Mock External Dependencies**: Use proper mocking for external services

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=monorepo_detection --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Write tests** for your changes

4. **Update documentation** if needed

5. **Run the full test suite**:
   ```bash
   ./scripts/test.sh
   ```

6. **Create a pull request** with:
   - Clear description of changes
   - Link to relevant issues
   - Test results and coverage reports

## Package-Specific Guidelines

### Anomaly Detection Algorithms
- **Algorithm Interface**: Implement the `AnomalyDetector` interface
- **Performance**: Include benchmark results for new algorithms
- **Documentation**: Provide algorithm description and references
- **Validation**: Include statistical validation of results

### Data Handling
- **Input Validation**: Validate all input data
- **Error Handling**: Use domain-specific exceptions
- **Memory Efficiency**: Consider memory usage for large datasets
- **Type Safety**: Use proper type hints for data structures

### Dependencies
- **Core Dependencies**: Minimize external dependencies
- **Optional Features**: Use optional dependencies for specialized features
- **Version Constraints**: Specify compatible version ranges
- **Security**: Keep dependencies updated for security

### Documentation
- **Algorithm Docs**: Document algorithm theory and implementation
- **Usage Examples**: Provide practical usage examples
- **Performance Characteristics**: Document time/space complexity
- **API Reference**: Keep API documentation current

## Questions and Support

For questions specific to this package:
- Open an issue with the `anomaly-detection` label
- Join our community discussions
- Review existing documentation and examples

Thank you for contributing to the Anomaly Detection package!