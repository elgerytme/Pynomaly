# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

You are developing Pynomaly, a state-of-the-art Python anomaly detection package targeting Python 3.11+. This package integrates multiple anomaly detection libraries (PyOD, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX) through a unified, production-ready interface following clean architecture principles.

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
   - Adapters: `PyODAdapter`, `PyGODAdapter`, `SklearnAdapter`
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
└── deploy/
    ├── docker/           # All Docker-related files
    └── kubernetes/       # All Kubernetes-related files
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
- **deploy/docker/**: All Docker-related files
  - Dockerfiles (main, hardened, testing variants)
  - docker-compose files for different environments
  - Docker-specific Makefiles and configurations
- **deploy/kubernetes/**: All Kubernetes-related files
  - Pod definitions, services, ingress configurations
  - ConfigMaps, secrets, persistent volumes
  - Helm charts and Kustomize overlays
  - RBAC configurations and network policies
  - **Exception**: Kubernetes files required only for Docker development may remain in docker/
- **tests/**: All testing code organized by test type
- **docs/**: All documentation files
- **examples/**: Sample code and usage demonstrations
- **src/**: Source code following clean architecture structure
- **benchmarks/**: Performance testing and benchmarking code

### Deployment Organization Rules
- **deploy/docker/**: ALL Docker-related files must be placed here
  - Dockerfiles for all environments (main, hardened, testing, UI testing)
  - All docker-compose files (development, production, testing variants)
  - Docker-specific Makefiles and build scripts
  - Docker environment configuration files
  - Docker health check scripts
- **deploy/kubernetes/**: ALL Kubernetes-related files must be placed here
  - Deployment manifests and pod specifications
  - Service definitions and ingress configurations
  - ConfigMaps, secrets, and persistent volume claims
  - Helm charts, values files, and templates
  - Kustomize base configurations and overlays
  - RBAC policies, network policies, and security contexts
  - Kubernetes-specific monitoring and logging configurations
  - **Exception**: Kubernetes files used exclusively for Docker development (e.g., local development clusters) may remain in deploy/docker/

### Maintenance Rules
- **Never clutter the root directory** with utility scripts or temporary files
- **Group related files** in appropriate subdirectories
- **Use descriptive directory names** that clearly indicate their purpose
- **Maintain consistency** in file naming and organization patterns
- **Regular cleanup**: Remove obsolete files and reorganize as needed
- **Deployment separation**: Strictly enforce Docker vs. Kubernetes file placement rules

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

#### TODO.md Synchronization Rules
**MANDATORY: Automatic TODO.md Updates with Claude Code Internal Todos**

Claude Code MUST automatically synchronize the internal todo system with TODO.md following these rules:

**📋 Synchronization Requirements:**
1. **Bidirectional Sync**: Changes in either Claude Code todos or TODO.md must trigger updates in both
2. **Real-Time Updates**: Synchronization occurs immediately when:
   - New Claude Code todos are created using TodoWrite
   - Todo status changes (pending → in_progress → completed)
   - TODO.md is manually edited with new tasks or status updates
   - Logical units of work are completed

**📝 TODO.md Integration Process:**
1. **Task Creation**: When TodoWrite creates new tasks, automatically add them to TODO.md "Current Work in Progress" section
2. **Status Updates**: When todo status changes, update corresponding entries in TODO.md with appropriate symbols:
   - ⏳ **Pending**: Tasks in pending status
   - 🔄 **In Progress**: Tasks actively being worked on
   - ✅ **Completed**: Tasks marked as completed
3. **Completion Documentation**: When tasks are completed, move them to the "Latest Completed" section with:
   - Detailed description of work accomplished
   - Technical specifications and features implemented
   - Cross-references to related files, commits, or documentation

**🔄 Automatic Update Triggers:**
- **Every TodoWrite call**: Update TODO.md current work section
- **Task completion**: Move completed items to "Latest Completed" with full documentation
- **Milestone completion**: Create new "Latest Completed" section with comprehensive summary
- **Session end**: Ensure all todo changes are reflected in TODO.md

**📊 TODO.md Structure Maintenance:**
1. **Current Work Section**: Always reflects active Claude Code todos
2. **Priority Ordering**: High priority todos appear first in each section
3. **Progress Tracking**: Include percentage completion estimates for complex tasks
4. **Cross-References**: Link todos to specific files, commits, or documentation sections

**🎯 Implementation Requirements:**
- Claude Code must check TODO.md at conversation start and sync with any existing tasks
- All todo operations must include automatic TODO.md updates
- TODO.md changes must be committed along with related code changes
- Maintain chronological order in "Latest Completed" sections with most recent first

**Example Integration:**
```markdown
### 🔄 Current Work in Progress
- 🔄 **ACTIVE: Implementing user authentication system** (Priority: High)
  - ✅ Create JWT token service
  - 🔄 Implement login/logout endpoints  
  - ⏳ Add session management
  - ⏳ Create password reset functionality
```

This ensures complete transparency and maintains TODO.md as the authoritative source of project status while leveraging Claude Code's internal todo system for task management.

#### Automation Triggers
- **Git Hooks**: Pre-commit hooks to remind about changelog updates
- **PR Templates**: Include changelog update checklist in pull request templates
- **CI Validation**: Check that CHANGELOG.md was modified in feature branches
- **Release Process**: Aggregate changelog entries for version releases
- **TODO Sync Validation**: Ensure TODO.md reflects all Claude Code todo changes

## Test-Driven Development (TDD) Configuration

### Overview
Pynomaly includes an optional but comprehensive Test-Driven Development (TDD) system that enforces test-first development practices. This system ensures code quality, maintainability, and adherence to TDD principles throughout the development lifecycle.

### TDD System Components

#### 1. Configuration Management
- **Location**: `src/pynomaly/infrastructure/config/tdd_config.py`
- **Purpose**: Centralized TDD settings with flexible configuration options
- **Features**:
  - Enable/disable TDD enforcement globally or per module
  - Configurable coverage thresholds and quality metrics
  - Customizable test naming conventions and patterns
  - Exemption management for legacy code or special cases

#### 2. Test Requirement Tracking
- **Repository**: `src/pynomaly/infrastructure/persistence/tdd_repository.py`
- **Purpose**: Track test requirements and implementation progress
- **Capabilities**:
  - Store test specifications before implementation
  - Monitor test-to-implementation compliance
  - Track coverage metrics and quality indicators
  - Maintain historical compliance data

#### 3. Enforcement Engine
- **Engine**: `src/pynomaly/infrastructure/tdd/enforcement.py`
- **Purpose**: Validate TDD compliance and enforce rules
- **Functions**:
  - Real-time file validation during development
  - Project-wide compliance analysis
  - Automated violation detection and reporting
  - Auto-fix capabilities for common violations

#### 4. Git Integration
- **Hooks**: `src/pynomaly/infrastructure/tdd/git_hooks.py`
- **Purpose**: Integrate TDD validation into git workflow
- **Features**:
  - Pre-commit hooks for immediate feedback
  - Pre-push validation for comprehensive checks
  - Pre-commit framework integration
  - Configurable validation levels

### TDD CLI Commands

#### Basic TDD Management
```bash
# Initialize TDD for the project
pynomaly tdd init --enable --coverage 0.85

# Check current TDD status
pynomaly tdd status --detailed

# Enable/disable TDD enforcement
pynomaly tdd enable --strict --coverage 0.9
pynomaly tdd disable
```

#### Test Requirements Management
```bash
# Create a test requirement
pynomaly tdd require "src/module.py" "function_name" \
  --desc "Test user authentication logic" \
  --spec "Should validate credentials and return user object" \
  --coverage 0.9 \
  --tags "auth,security"

# List test requirements
pynomaly tdd requirements --status pending
pynomaly tdd requirements --module "src/auth.py"
pynomaly tdd requirements --tags "security,critical"
```

#### Validation and Compliance
```bash
# Validate TDD compliance
pynomaly tdd validate --coverage --fix
pynomaly tdd validate --file "src/specific_file.py"

# Run coverage analysis
pynomaly tdd coverage --threshold 0.8

# Generate compliance reports
pynomaly tdd report --output "reports/tdd_compliance.html" --format html
pynomaly tdd report --format json
```

#### Git Integration
```bash
# Install git hooks
pynomaly tdd hooks --install --pre-commit

# Check git hooks status
pynomaly tdd hooks --status

# Uninstall git hooks
pynomaly tdd hooks --uninstall
```

#### Configuration Management
```bash
# View TDD configuration
pynomaly tdd config

# Manage exemptions
pynomaly tdd exempt "migrations/*.py"
pynomaly tdd exempt --list
pynomaly tdd exempt "old_code.py" --remove

# Reset configuration
pynomaly tdd config --reset
```

### TDD Configuration Options

#### Core Settings
```python
# TDD enforcement control
enabled: bool = False                    # Enable TDD enforcement
strict_mode: bool = False               # Enforce strict test-first rules
auto_validation: bool = True            # Automatic compliance validation

# Coverage requirements
min_test_coverage: float = 0.8          # Minimum required coverage
coverage_fail_under: float = 0.7        # Coverage threshold for build failure
branch_coverage_required: bool = True   # Require branch coverage analysis
```

#### File Pattern Configuration
```python
# Test file patterns
test_file_patterns: List[str] = [
    "test_*.py", "*_test.py", "tests/*.py"
]

# Implementation file patterns
implementation_patterns: List[str] = ["*.py"]

# Exemption patterns (files to skip)
exemption_patterns: List[str] = [
    "__init__.py", "setup.py", "conftest.py", 
    "migrations/*", "scripts/*"
]
```

#### Enforcement Scope
```python
# Enforce TDD on specific packages
enforce_on_packages: List[str] = [
    "src/pynomaly/domain",      # Domain layer (required)
    "src/pynomaly/application", # Application layer (required)
    "src/pynomaly/infrastructure/adapters"  # Critical adapters
]

# Module-specific enforcement
enforce_on_modules: List[str] = []  # Empty means all applicable modules
```

#### Naming Conventions
```python
# Test function naming
test_naming_convention: str = "test_{function_name}"

# Test class naming
test_class_naming_convention: str = "Test{ClassName}"

# Require test docstrings
require_test_docstrings: bool = True
```

#### Git and CI Integration
```python
# Git hooks
git_hooks_enabled: bool = True
pre_commit_validation: bool = True
pre_push_validation: bool = False

# CI/CD integration
ci_validation_enabled: bool = True
fail_on_violations: bool = False        # Fail builds on TDD violations
```

#### Development Workflow
```python
# Development flexibility
allow_implementation_first: bool = False  # Allow impl before tests in dev
grace_period_hours: int = 24             # Grace period for compliance
require_test_plan: bool = True           # Require test plan before impl

# Advanced testing features
mutation_testing_enabled: bool = False   # Enable mutation testing
property_testing_enabled: bool = False  # Enable property-based testing
integration_test_ratio: float = 0.2     # Required ratio of integration tests
```

### TDD Workflow Integration

#### 1. Development Process
```bash
# 1. Create test requirement
pynomaly tdd require "src/auth/service.py" "authenticate_user" \
  --desc "Authenticate user with credentials" \
  --spec "Given valid credentials, return user object. Given invalid credentials, raise AuthenticationError"

# 2. Write failing test
# Write test in tests/auth/test_service.py following the specification

# 3. Implement minimal code
# Write just enough code to make the test pass

# 4. Validate compliance
pynomaly tdd validate --file "src/auth/service.py"

# 5. Refactor and improve
# Improve code while maintaining test coverage
```

#### 2. Git Workflow Integration
```bash
# Install hooks for automatic validation
pynomaly tdd hooks --install

# Pre-commit automatically validates:
# - Critical TDD violations (missing tests, parsing errors)
# - Naming convention compliance
# - Test-first violations in strict mode

# Pre-push validates:
# - Comprehensive TDD compliance
# - Coverage thresholds
# - All violation types
```

#### 3. CI/CD Integration
The TDD system integrates with CI/CD pipelines:

```yaml
# .github/workflows/tdd.yml
name: TDD Compliance
on: [push, pull_request]
jobs:
  tdd-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .
      - name: Run TDD validation
        run: |
          pynomaly tdd validate --coverage
          pynomaly tdd report --output tdd_report.json --format json
      - name: Upload TDD report
        uses: actions/upload-artifact@v3
        with:
          name: tdd-compliance-report
          path: tdd_report.json
```

### TDD Best Practices for Pynomaly

#### 1. Domain Layer TDD (Mandatory)
- **All domain entities** must have comprehensive test coverage
- **Value objects** require property-based testing where applicable
- **Domain services** must be tested with various scenarios
- **Business rules** must be validated through tests

#### 2. Application Layer TDD (Mandatory)
- **Use cases** must have test scenarios for all paths
- **Application services** require integration testing
- **DTOs** should have validation testing
- **Error handling** must be thoroughly tested

#### 3. Infrastructure Layer TDD (Selective)
- **Adapters** require comprehensive testing
- **Repositories** need both unit and integration tests
- **External integrations** must have contract tests
- **Configuration** should be validated

#### 4. Presentation Layer TDD (Recommended)
- **API endpoints** require comprehensive testing
- **CLI commands** should have end-to-end tests
- **Web UI** benefits from component and integration tests

### TDD Violation Types and Resolutions

#### Common Violations
1. **missing_test**: Implementation file lacks corresponding test file
   - **Auto-fix**: Generate basic test template
   - **Manual**: Create comprehensive test file

2. **low_coverage**: Test coverage below threshold
   - **Resolution**: Add missing test cases
   - **Analysis**: Use coverage report to identify gaps

3. **implementation_before_test**: Code written before tests (strict mode)
   - **Resolution**: Restructure development process
   - **Prevention**: Use TDD workflow consistently

4. **naming_violation**: Test functions don't follow naming convention
   - **Resolution**: Rename test functions
   - **Configuration**: Adjust naming patterns if needed

5. **missing_test_requirement**: Function lacks test specification
   - **Resolution**: Create test requirement with clear specification
   - **Process**: Define expected behavior before implementation

### TDD Reporting and Metrics

#### Compliance Reports
- **Overall compliance percentage**: Based on requirements completion
- **Module-specific compliance**: Track progress per package
- **Coverage metrics**: Line, branch, and function coverage
- **Violation trends**: Track improvements over time
- **Quality indicators**: Test quality and effectiveness metrics

#### Export Formats
- **JSON**: Machine-readable for CI/CD integration
- **YAML**: Human-readable configuration format
- **HTML**: Rich visual reports for stakeholders
- **CSV**: Data analysis and tracking

### Important Notes
- Always ensure the virtual environment is activated before installing packages or running code
- The `.gitignore` currently excludes the VS Code workspace file (`Pynomaly.code-workspace`)
- Follow the architectural principles strictly to maintain clean separation of concerns
- **CRITICAL**: Always keep `requirements.txt` in sync with `pyproject.toml` core dependencies. When adding/removing dependencies in `pyproject.toml`, immediately update `requirements.txt` to include only the non-optional dependencies needed for basic functionality
- **TDD ENFORCEMENT**: When TDD is enabled, follow test-first development strictly for domain and application layers

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