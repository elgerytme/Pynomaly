# PROJECT STRUCTURE REFERENCE

## Overview
This document serves as the authoritative reference for the Pynomaly project's directory structure and organization. It is used by AI assistants and development tools to maintain consistent project organization.

**⚠️ IMPORTANT NOTE FOR AI ASSISTANTS:**
This project organization is complex and difficult to maintain consistently with AI agents and assistants (such as Claude). Always reference this document when:
- Creating new files or directories
- Moving or reorganizing existing files
- Understanding the architectural boundaries
- Implementing new features

## Root Level Structure

```
/mnt/c/Users/andre/Pynomaly/
├── README.md                    # Main project documentation
├── TODO.md                      # Current tasks and progress
├── CLAUDE.md                    # AI assistant instructions
├── PROJECT_STRUCTURE.md         # This file - structure reference
├── CHANGELOG.md                 # Version history
├── LICENSE                      # Project license
├── Makefile                     # Build automation
├── pyproject.toml              # Python project configuration
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies
├── package-lock.json           # Node.js lock file
├── Pynomaly.code-workspace     # VS Code workspace
└── [Analysis/Strategy Files]    # Various analysis documents
```

## Source Code (`src/`)

### Core Package Structure (`src/pynomaly/`)
**Clean Architecture + DDD Implementation**

```
src/pynomaly/
├── __init__.py
├── __main__.py
├── _version.py
├── py.typed
├── demo_functions.py
├── domain/                      # Business logic (pure)
│   ├── entities/               # Business entities
│   ├── exceptions/             # Domain exceptions
│   ├── services/               # Domain services
│   └── value_objects/          # Value objects
├── application/                # Use cases and orchestration
│   ├── dto/                    # Data Transfer Objects
│   ├── services/               # Application services
│   └── use_cases/              # Business use cases
├── infrastructure/             # External integrations
│   ├── adapters/               # Algorithm adapters
│   ├── auth/                   # Authentication
│   ├── automl/                 # AutoML implementations
│   ├── cache/                  # Caching layer
│   ├── config/                 # Configuration
│   ├── data_loaders/           # Data loading
│   ├── data_processing/        # Data pipelines
│   ├── distributed/            # Distributed computing
│   ├── explainers/             # Explainability
│   ├── logging/                # Logging and observability
│   ├── middleware/             # Middleware components
│   ├── monitoring/             # Health checks and metrics
│   ├── persistence/            # Database operations
│   ├── preprocessing/          # Data preprocessing
│   ├── repositories/           # Data repositories
│   ├── security/               # Security components
│   └── streaming/              # Real-time processing
├── presentation/               # User interfaces
│   ├── api/                    # REST API (FastAPI)
│   ├── cli/                    # Command line interface
│   ├── sdk/                    # Python SDK
│   └── web/                    # Progressive Web App
├── shared/                     # Common utilities
│   ├── protocols/              # Interface definitions
│   └── utils/                  # Utility functions
└── scripts/                    # Initialization scripts
```

## Configuration (`config/`)

```
config/
├── README.md
├── MANIFEST.in
├── docs/                       # Documentation configs
├── environments/               # Environment-specific configs
├── git/                        # Git configuration
├── web/                        # Web application configs
├── advanced_testing_config.json
├── pytest.ini
├── tdd_config.json
└── tox.ini
```

## Documentation (`docs/`)

```
docs/
├── index.md                    # Main documentation entry
├── getting-started/            # Installation and setup
├── user-guides/               # User documentation
├── developer-guides/          # Development documentation
├── reference/                 # API reference
├── examples/                  # Usage examples
├── testing/                   # Testing documentation
├── deployment/                # Deployment guides
├── design-system/             # UI/UX guidelines
├── accessibility/             # Accessibility guides
├── advanced/                  # Advanced topics
├── archive/                   # Historical documentation
└── project/                   # Project management docs
```

## Testing (`tests/`)

```
tests/
├── __init__.py
├── conftest.py                # Pytest configuration
├── unit/                      # Unit tests
├── integration/               # Integration tests
├── e2e/                       # End-to-end tests
├── performance/               # Performance tests
├── security/                  # Security tests
├── ui/                        # UI tests
├── contract/                  # Contract tests
├── mutation/                  # Mutation tests
├── property/                  # Property-based tests
├── domain/                    # Domain layer tests
├── application/               # Application layer tests
├── infrastructure/            # Infrastructure layer tests
├── presentation/              # Presentation layer tests
└── [Test Category Folders]/   # Various test categories
```

## Scripts (`scripts/`)

```
scripts/
├── analysis/                  # Code analysis tools
├── build/                     # Build scripts
├── demo/                      # Demo scripts
├── deploy/                    # Deployment scripts
├── docker/                    # Docker utilities
├── generate/                  # Code generation
├── maintenance/               # Maintenance scripts
├── run/                       # Application runners
├── setup/                     # Setup and installation
├── testing/                   # Testing utilities
└── validation/                # Validation scripts
```

## Deployment (`deploy/`)

```
deploy/
├── README.md
├── docker/                    # Docker configurations
│   ├── Dockerfile.*          # Various Dockerfiles
│   ├── docker-compose.*.yml  # Compose configurations
│   └── config/               # Container configs
├── kubernetes/                # Kubernetes manifests
│   ├── *.yaml               # K8s resource definitions
│   └── Makefile             # K8s deployment automation
├── build-configs/            # Build configurations
└── artifacts/                # Build artifacts
```

## Examples (`examples/`)

```
examples/
├── README.md
├── USAGE_GUIDE.md
├── banking/                   # Banking domain examples
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Example scripts
├── configs/                   # Configuration examples
├── datasets/                  # Sample datasets
├── sample_data/              # Generated sample data
├── sample_datasets/          # Curated datasets
└── storage/                  # Example storage
```

## Storage and Data (`storage/`, `environments/`)

```
storage/                       # Runtime storage
├── analytics/                # Analytics data
├── experiments/              # Experiment data
├── models/                   # Trained models
└── temp/                     # Temporary files

environments/                  # Virtual environments
├── README.md
└── .venv/                    # Virtual environment (dot-prefix)
```

## Templates (`templates/`)

```
templates/
├── README.md
├── TEMPLATE_SYSTEM_GUIDE.md
├── anomaly_test_config/      # Test configurations
├── documentation/            # Documentation templates
├── experiments/              # Experiment templates
├── reporting/                # Report templates
├── scripts/                  # Script templates
└── testing/                  # Testing templates
```

## Reports (`reports/`)

```
reports/
├── coverage/                 # Test coverage reports
├── builds/                   # Build reports
├── [Various Report Files]    # Analysis and validation reports
```

## Key Architectural Boundaries

### 1. Clean Architecture Layers
- **Domain**: Pure business logic, no external dependencies
- **Application**: Use cases, orchestration, DTOs
- **Infrastructure**: External integrations, adapters
- **Presentation**: User interfaces (API, CLI, Web, SDK)

### 2. Dependency Direction
- Outer layers depend on inner layers
- Domain is dependency-free
- All external dependencies in Infrastructure

### 3. File Organization Rules
- Virtual environments: `environments/.venv/` (dot-prefix)
- Configuration files: `config/` directory
- No build artifacts or temp files in root
- Scripts organized by purpose in `scripts/`

### 4. Testing Structure
- Mirror source structure in tests
- Separate by test type (unit, integration, e2e)
- Domain tests are mandatory (>90% coverage)

## AI Assistant Guidelines

### DO:
- ✅ Reference this document before creating files
- ✅ Follow Clean Architecture boundaries
- ✅ Use dot-prefix for virtual environments
- ✅ Organize scripts by purpose
- ✅ Maintain test coverage requirements
- ✅ Update this document when structure changes

### DON'T:
- ❌ Create files without checking structure
- ❌ Mix architectural layers
- ❌ Use non-dot-prefix virtual environments
- ❌ Put scripts in root directory
- ❌ Create temp files in version control
- ❌ Ignore domain purity rules

## Maintenance Notes

This structure is enforced by:
- Pre-commit hooks
- Validation scripts in `scripts/validation/`
- CI/CD pipeline checks
- TDD enforcement for domain/application layers

**Last Updated**: 2025-01-07
**Validation Script**: `scripts/validation/validate_file_organization.py`