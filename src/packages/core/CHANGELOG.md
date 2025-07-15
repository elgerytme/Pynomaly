# Changelog - Core Package

All notable changes to the Pynomaly core package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced domain entity validation and business rules
- Improved value object immutability and type safety
- Additional anomaly score calculation methods
- Extended dataset metadata and validation

### Changed
- Optimized performance for large dataset processing
- Enhanced error handling and exception hierarchy
- Improved domain event system for better decoupling

### Fixed
- Edge cases in contamination rate validation
- Memory optimization for large anomaly collections
- Thread safety improvements in concurrent scenarios

## [1.0.0] - 2025-07-14

### Added
- **Domain Entities**: Core business objects for anomaly detection
  - `Detector`: Configurable anomaly detection models
  - `Dataset`: Structured data containers with metadata
  - `Anomaly`: Individual anomaly instances with scores and metadata
  - `DetectionResult`: Comprehensive detection outcomes
- **Value Objects**: Immutable domain values
  - `ContaminationRate`: Type-safe contamination percentage (0.0-1.0)
  - `AnomalyScore`: Validated anomaly scores with confidence intervals
  - `ConfidenceInterval`: Statistical confidence ranges
  - `Threshold`: Detection thresholds with validation
- **Business Rules**: Domain logic and validation
  - Contamination rate constraints and validation
  - Anomaly score normalization and comparison
  - Dataset size and quality requirements
  - Detection parameter validation
- **Repository Interfaces**: Abstract data access patterns
  - `DatasetRepository`: Dataset persistence interface
  - `DetectorRepository`: Model persistence interface
  - `AnomalyRepository`: Anomaly storage interface
- **Use Cases**: Application layer business operations
  - `DetectAnomaliesUseCase`: Core anomaly detection workflow
  - `TrainDetectorUseCase`: Model training operations
  - `EvaluateDetectorUseCase`: Performance evaluation
  - `ExportResultsUseCase`: Result export functionality

### Core Features
- **Clean Architecture**: Strict layer separation with dependency inversion
- **Domain-Driven Design**: Rich domain model with ubiquitous language
- **Type Safety**: Comprehensive type hints with runtime validation
- **Immutability**: Value objects and entity state protection
- **Extensibility**: Plugin architecture for custom algorithms
- **Validation**: Comprehensive input validation and business rule enforcement

### Performance
- **Memory Efficient**: Optimized for large datasets (1M+ samples)
- **Lazy Loading**: On-demand data loading and processing
- **Batch Processing**: Efficient bulk operations
- **Async Support**: Full async/await compatibility

### Security
- **Input Validation**: Comprehensive data sanitization
- **Safe Defaults**: Secure configuration defaults
- **Error Handling**: Safe error propagation without data leakage
- **Audit Logging**: Domain event tracking for compliance

## [0.9.0] - 2025-06-01

### Added
- Initial domain model design
- Basic entity and value object implementations
- Core repository interfaces
- Foundation use cases

### Changed
- Refined domain boundaries and relationships
- Improved entity lifecycle management

### Fixed
- Initial validation logic improvements
- Enhanced error handling patterns

## [0.1.0] - 2025-01-15

### Added
- Project structure and initial domain concepts
- Basic entity definitions
- Core package foundation

---

## Migration Guide

### Upgrading to 1.0.0

The 1.0.0 release represents the first stable API. Key changes:

```python
# Before (0.9.x)
detector = AnomalyDetector(contamination=0.1)
result = detector.detect(data)

# After (1.0.0)
from pynomaly.core.domain.entities import Detector, Dataset
from pynomaly.core.domain.value_objects import ContaminationRate

detector = Detector.isolation_forest(
    contamination_rate=ContaminationRate(0.1)
)
dataset = Dataset.from_array(data)
result = detector.detect_anomalies(dataset)
```

## Breaking Changes

### 1.0.0
- Renamed `AnomalyDetector` to `Detector`
- Changed contamination parameter to `ContaminationRate` value object
- Updated method signatures for type safety
- Restructured exception hierarchy

## Dependencies

### Runtime Dependencies
- `pydantic>=2.0.0`: Data validation and serialization
- `numpy>=1.24.0`: Numerical computations
- `structlog>=23.0.0`: Structured logging

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `hypothesis>=6.0.0`: Property-based testing
- `mypy>=1.0.0`: Static type checking

## Contributing

When contributing to the core package:

1. **Domain Focus**: Ensure changes align with domain boundaries
2. **Type Safety**: Maintain comprehensive type hints
3. **Immutability**: Preserve value object immutability
4. **Testing**: Add comprehensive unit tests with edge cases
5. **Documentation**: Update docstrings and domain documentation

For detailed contribution guidelines, see [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## Support

- **Package Documentation**: [docs/](docs/)
- **Domain Model Guide**: [docs/domain_model.md](docs/domain_model.md)
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **Issues**: [GitHub Issues](../../../issues)

[Unreleased]: https://github.com/elgerytme/Pynomaly/compare/core-v1.0.0...HEAD
[1.0.0]: https://github.com/elgerytme/Pynomaly/releases/tag/core-v1.0.0
[0.9.0]: https://github.com/elgerytme/Pynomaly/releases/tag/core-v0.9.0
[0.1.0]: https://github.com/elgerytme/Pynomaly/releases/tag/core-v0.1.0