# Core Package Documentation

## Integration Points with Major ML Libraries

- **Scikit-Learn & PyOD**: Utilized in `sklearn_adapter` and `pyod_adapter` for anomaly detection.
- **TensorFlow**: Employed in `tensorflow_adapter` for deep learning models like AutoEncoders and VAE.
- **PyTorch**: Possible usage in performance benchmarking for GPU availability.

## Clean Architecture Layers

- **Domain Layer**: 
  - **Entities**: `Detector`, `Anomaly` representing core domain concepts.

- **Application Layer**:
  - **Services**: `PerformanceBenchmarkingService`, handling benchmarking and evaluation.

- **Infrastructure Layer**: 
  - **Adapters**: `PyODAdapter`, `SklearnAdapter`, `TensorFlowAdapter` facilitate interaction with ML libraries.
  - **Configuration**: Managed by settings files for system settings and dependency injection.

## Domain Entities

- **Detector**: Represents anomaly detectors, includes algorithm-related metadata.
- **Anomaly**: Represents detected anomalies.

## Services

- **PerformanceBenchmarkingService**: Provides CPU/GPU/memory profiling and runtime measurements.

## Adapters

- **Python-based Adapters**: Interfaces for integrating with existing ML libraries, updating application logic.

## Performance, Scalability, and Security Constraints

- **Performance**: 
  - Incorporates vectorization and caching for optimized performance.
  - Designed to handle batch/offline mode for large datasets.

- **Scalability**:
  - Asynchronous tasks and thread management in system monitoring.
  - Settings configurations for adaptable scaling.

- **Security**:
  - Configurations for audit logging, rate limiting, and input sanitization.
  - Security monitoring can be optionally integrated.

## Configuration

- System settings customizable via Pydantic models.
- Dependency Injection via a service registry pattern.

---

This documentation provides an overview of the core functionality, identifying integration points, and elucidating performance, scalability, and security aspects.
