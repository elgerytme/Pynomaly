# Pynomaly Development Master Prompt

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Project

---


You are developing Pynomaly, a state-of-the-art Python anomaly detection package targeting Python 3.11+. This package integrates multiple anomaly detection libraries (PyOD, TODS, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX) through a unified, production-ready interface following clean architecture principles.

## Core Architectural Principles

Follow **Clean Architecture**, **Domain-Driven Design (DDD)**, and **Hexagonal Architecture (Ports & Adapters)**:

1. **Domain Layer**: Pure Python business logic with no external dependencies
   - Entities: `Anomaly`, `Detector`, `Dataset`, `Score`, `DetectionResult`
   - Value Objects: `ContaminationRate`, `ConfidenceInterval`, `AnomalyScore`
   - Domain Services: Core detection logic, scoring algorithms

2. **Application Layer**: Orchestrate use cases without implementation details
   - Use Cases: `DetectAnomalies`, `TrainDetector`, `EvaluateModel`, `ExplainAnomaly`
   - Services: `DetectionService`, `EnsembleService`, `ModelPersistenceService`

3. **Infrastructure Layer**: All external integrations
   - Adapters: `PyODAdapter`, `TODSAdapter`, `PyGODAdapter`, `SklearnAdapter`
   - Data Sources: `CSVLoader`, `ParquetLoader`, `DatabaseLoader`, `StreamingLoader`
   - Persistence: `ModelRepository`, `ResultRepository`

4. **Presentation Layer**: User interfaces
   - REST API (FastAPI)
   - CLI (Click/Typer)
   - Python SDK
   - Progressive Web App (PWA) with HTMX, Tailwind CSS, D3.js, Apache ECharts

## Implementation Guidelines

### Code Quality Standards
- **Type Hints**: 100% coverage with `mypy --strict`
- **Async/Await**: For all I/O operations
- **Protocols**: Define interfaces using Python protocols
- **Dependency Injection**: Use `dependency-injector` or similar
- **Error Handling**: Custom exception hierarchy with context
- **Logging**: Structured logging with `structlog`
- **Configuration**: `pydantic-settings` for type-safe config

### Design Patterns
- **Repository Pattern**: For data access
- **Factory Pattern**: For algorithm creation
- **Strategy Pattern**: For detection algorithms
- **Observer Pattern**: For real-time detection
- **Decorator Pattern**: For feature enhancement
- **Chain of Responsibility**: For data preprocessing

### Testing Requirements
- **Unit Tests**: >90% coverage with pytest
- **Integration Tests**: For all adapters
- **Property-Based Tests**: Using Hypothesis
- **Performance Tests**: Benchmarking suite
- **Mutation Testing**: Ensure test quality
- **Contract Tests**: For adapter interfaces

### Production Features
- **Observability**: OpenTelemetry integration
- **Metrics**: Prometheus metrics
- **Health Checks**: Kubernetes-ready probes
- **Circuit Breakers**: For external services
- **Rate Limiting**: API protection
- **Caching**: Redis/memory caching
- **Security**: Input validation, auth, encryption

### Algorithm Integration
When integrating algorithms:
1. Create adapter implementing `DetectorProtocol`
2. Register in `AlgorithmRegistry` with metadata
3. Support both batch and streaming modes
4. Include hyperparameter schemas
5. Provide performance characteristics
6. Enable GPU acceleration where applicable

### Data Processing
- Support formats: CSV, Parquet, Arrow, HDF5, SQL databases
- Implement streaming with backpressure
- Memory-efficient operations for large datasets
- Data validation with `pandera` or similar
- Feature engineering pipeline
- Data versioning with DVC integration

### State-of-the-Art Features
1. **AutoML**: Automated algorithm selection and tuning
2. **Explainability**: SHAP/LIME integration
3. **Drift Detection**: Monitor model degradation
4. **Active Learning**: Human-in-the-loop capability
5. **Multi-Modal**: Time-series, tabular, graph, text
6. **Ensemble Methods**: Advanced voting strategies
7. **Uncertainty Quantification**: Confidence intervals
8. **Progressive Web App**: Offline-capable, installable web interface
   - Server-side rendering with HTMX for simplicity
   - Modern UI with Tailwind CSS
   - Interactive visualizations with D3.js
   - Statistical charts with Apache ECharts
   - Works offline with service workers
   - Installable on desktop and mobile devices

### Directory Structure
```
pynomaly/
├── src/pynomaly/
│   ├── domain/
│   │   ├── entities/
│   │   ├── value_objects/
│   │   ├── services/
│   │   └── exceptions/
│   ├── application/
│   │   ├── use_cases/
│   │   ├── services/
│   │   └── dto/
│   ├── infrastructure/
│   │   ├── adapters/
│   │   ├── persistence/
│   │   ├── config/
│   │   └── monitoring/
│   ├── presentation/
│   │   ├── api/
│   │   ├── cli/
│   │   ├── sdk/
│   │   └── web/          # Progressive Web App
│   │       ├── static/  # CSS, JS, images
│   │       ├── templates/ # HTMX templates
│   │       └── assets/  # PWA manifest, icons
│   └── shared/
│       ├── protocols/
│       └── utils/
├── tests/
├── docs/
├── examples/
├── benchmarks/
└── docker/
```

## Development Workflow

1. **Always start** with domain models and protocols
2. **Test-first** approach with clear test cases
3. **Document** architectural decisions in ADRs
4. **Benchmark** performance impact of changes
5. **Use** conventional commits for clarity
6. **Maintain** backward compatibility
7. **Follow** semantic versioning strictly

## Key Priorities

1. **Clean, maintainable code** over premature optimization
2. **Extensibility** through well-defined interfaces
3. **Production readiness** from day one
4. **User experience** through intuitive APIs
5. **Performance** with profiling and optimization
6. **Security** by design, not as afterthought
7. **Documentation** as first-class citizen

## Remember

- This is a **production-grade** package, not a research prototype
- Every component should be **independently testable**
- Favor **composition over inheritance**
- Make the **simple case simple**, complex case possible
- **Fail fast** with clear error messages
- Consider **resource constraints** (memory, CPU, GPU)
- Design for **horizontal scalability**