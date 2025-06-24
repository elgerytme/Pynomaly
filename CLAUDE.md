# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

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

## Development Environment

### Python Version
- Python 3.11+ (managed via pyenv-win)
- Virtual environment located at `.venv/`
- Use Poetry for dependency management

### Common Commands

#### Poetry Commands
```bash
poetry install  # Install all dependencies
poetry add <package>  # Add a dependency
poetry add --group dev <package>  # Add dev dependency
poetry run pytest  # Run tests
poetry run mypy src/  # Type checking
poetry build  # Build distribution packages
```

#### Testing and Quality
```bash
poetry run pytest --cov=pynomaly --cov-report=html
poetry run pytest > tests/pytest_output.txt 2>&1  # Capture pytest output in tests/
poetry run pytest -v > tests/pytest_full_output.txt 2>&1  # Verbose output in tests/
poetry run mypy --strict src/
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
poetry run bandit -r src/
```

#### Running the Application
```bash
poetry run pynomaly --help  # CLI
poetry run uvicorn pynomaly.presentation.api:app  # API
poetry run uvicorn pynomaly.presentation.api:app --reload  # API with auto-reload for development
```

#### Web UI Development
```bash
# Install frontend dependencies
npm install -D tailwindcss @tailwindcss/forms @tailwindcss/typography
npm install htmx.org d3 echarts

# Build Tailwind CSS
npm run build-css  # Production build
npm run watch-css  # Development with watch mode

# PWA development
poetry run python -m http.server 8080 --directory src/pynomaly/presentation/web/static  # Test service worker
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

## Claude Behavior Guidelines

- **Remain objective and critical**: Question assumptions, identify potential issues, and provide honest assessments
- **Avoid sycophancy**: Don't exaggerate claims or provide excessive praise for code quality or decisions
- **Be direct**: Point out problems, inefficiencies, or areas for improvement without sugar-coating
- **Focus on facts**: Base recommendations on technical merit, not on trying to please or agree
- **Challenge when appropriate**: If a design decision seems suboptimal, explain why and suggest alternatives

## Project Organization Rules

### File Organization Standards
- **Root Directory**: Keep only essential project files (pyproject.toml, README.md, LICENSE, etc.)
- **scripts/**: All Python scripts, utilities, and CLI tools
  - Development and testing scripts
  - Build and deployment automation
  - Database initialization scripts
  - Benchmarking and performance tools
- **docker/**: All Docker-related files
  - Dockerfiles (main, hardened, testing variants)
  - docker-compose files for different environments
  - Docker-specific Makefiles and configurations
- **tests/**: All testing code organized by test type
- **docs/**: All documentation files
- **examples/**: Sample code and usage demonstrations
- **src/**: Source code following clean architecture structure
- **benchmarks/**: Performance testing and benchmarking code

### Maintenance Rules
- **Never clutter the root directory** with utility scripts or temporary files
- **Group related files** in appropriate subdirectories
- **Use descriptive directory names** that clearly indicate their purpose
- **Maintain consistency** in file naming and organization patterns
- **Regular cleanup**: Remove obsolete files and reorganize as needed

## Changelog Management Rules

### Automatic Changelog Updates
When logical units of work are complete, **ALWAYS** update both CHANGELOG.md and TODO.md following these rules:

#### What Constitutes a "Logical Unit of Work"
- **Feature Implementation**: Complete new features with tests and documentation
- **Bug Fixes**: Resolved issues that affect functionality
- **Infrastructure Changes**: CI/CD, Docker, deployment configuration updates
- **Documentation Phases**: Major documentation additions or restructuring
- **Testing Milestones**: Significant test coverage improvements or test infrastructure
- **Algorithm Implementations**: New ML algorithms, adapters, or detection methods
- **Performance Improvements**: Optimization work with measurable improvements
- **Security Enhancements**: Authentication, authorization, or security fixes
- **API Changes**: Breaking or non-breaking API modifications
- **Dependency Updates**: Major dependency upgrades or additions

#### Changelog Update Process
1. **Immediate Update**: Update CHANGELOG.md immediately when work is complete
2. **Version Numbering**: Follow semantic versioning (MAJOR.MINOR.PATCH)
3. **Entry Format**: Use standardized format with date, version, and categorized changes
4. **Cross-Reference**: Update TODO.md to mark items as completed and reference changelog
5. **Commit Message**: Include changelog update in the same commit as the feature

#### Changelog Entry Categories
- **Added**: New features, capabilities, or functionality
- **Changed**: Changes in existing functionality or behavior
- **Deprecated**: Soon-to-be removed features
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes and issue resolutions
- **Security**: Security-related changes and vulnerability fixes
- **Performance**: Performance improvements and optimizations
- **Documentation**: Documentation additions, improvements, or restructuring
- **Infrastructure**: CI/CD, build system, or deployment changes
- **Testing**: Test additions, improvements, or infrastructure changes

#### Mandatory Changelog Updates
**ALWAYS update the changelog when:**
- Completing any feature from the TODO.md list
- Adding new algorithms, adapters, or detection methods
- Making API changes that affect external usage
- Completing documentation phases or major updates
- Implementing infrastructure or deployment changes
- Achieving testing milestones or coverage improvements
- Fixing bugs or security issues
- Adding new datasets, examples, or analysis tools
- Completing autonomous mode enhancements
- Making performance optimizations

#### Changelog Entry Template
```markdown
## [Version] - YYYY-MM-DD

### Added
- New feature description with context and usage
- Algorithm implementations with performance characteristics
- Documentation sections with scope and audience

### Changed
- Modified functionality with migration guidance
- Updated dependencies with version information
- Improved performance with benchmark results

### Fixed
- Bug fixes with issue references and impact
- Security vulnerabilities with severity assessment
- Compatibility issues with affected systems

### Documentation
- New guides, tutorials, or reference materials
- Updated existing documentation with scope
- API documentation improvements

### Infrastructure
- CI/CD pipeline improvements
- Docker configuration enhancements
- Deployment automation additions

### Testing
- Test coverage improvements with percentages
- New test infrastructure or frameworks
- Performance test additions
```

#### Integration with Development Workflow
1. **Before Starting Work**: Check TODO.md for planned work items
2. **During Development**: Keep notes of changes for changelog entry
3. **Upon Completion**: 
   - Update CHANGELOG.md with detailed entry
   - Mark TODO.md items as completed
   - Cross-reference between both files
   - Commit both files together with the feature
4. **Quality Check**: Ensure changelog entry includes sufficient detail for users

#### Automation Triggers
- **Git Hooks**: Pre-commit hooks to remind about changelog updates
- **PR Templates**: Include changelog update checklist in pull request templates
- **CI Validation**: Check that CHANGELOG.md was modified in feature branches
- **Release Process**: Aggregate changelog entries for version releases

## Important Notes
- Always ensure the virtual environment is activated before installing packages or running code
- The `.gitignore` currently excludes the VS Code workspace file (`Pynomaly.code-workspace`)
- Follow the architectural principles strictly to maintain clean separation of concerns
- **CRITICAL**: Always keep `requirements.txt` in sync with `pyproject.toml` core dependencies. When adding/removing dependencies in `pyproject.toml`, immediately update `requirements.txt` to include only the non-optional dependencies needed for basic functionality

## Web UI Technology Stack
- **HTMX**: Provides dynamic behavior with minimal JavaScript, keeping complexity server-side
- **Tailwind CSS**: Utility-first CSS framework for rapid, consistent UI development
- **D3.js**: For creating custom, interactive anomaly visualizations
- **Apache ECharts**: For statistical charts, time series plots, and dashboards
- **PWA**: Progressive Web App features for offline capability and installability
  - Service workers handle offline caching and background sync
  - Web app manifest enables installation on desktop and mobile
  - IndexedDB for client-side data storage
  - Push API for real-time anomaly notifications