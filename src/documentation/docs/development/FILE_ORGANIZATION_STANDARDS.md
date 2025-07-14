# File Organization Standards

**Version:** 1.0  
**Document Type:** Authoritative Standard  
**Status:** Approved  
**Location:** `docs/development/FILE_ORGANIZATION_STANDARDS.md`

---

## Overview

This document establishes the authoritative file organization standards for the Pynomaly project. It provides explicit rules for file placement, directory structure, naming conventions, and automated enforcement mechanisms to maintain a clean, navigable, and professional repository.

## Core Principles

1. **Clean Root Directory**: Only essential project files allowed in repository root
2. **Logical Categorization**: Files grouped by purpose and function
3. **Predictable Structure**: Consistent directory naming and organization
4. **Automated Enforcement**: Standards enforced through tooling and CI/CD
5. **Scalable Architecture**: Structure supports project growth

## Allowed Files in Repository Root

### Essential Project Files
- `README.md` - Primary project documentation
- `LICENSE` - Project license
- `CHANGELOG.md` - Version history and release notes
- `CONTRIBUTING.md` - Contribution guidelines
- `TODO.md` - Project roadmap and tasks (optional)

### Package Configuration
- `pyproject.toml` - Primary Python package configuration
- `setup.py` - Legacy Python setup (if required for compatibility)
- `setup.cfg` - Additional Python setup configuration
- `MANIFEST.in` - Package manifest for distribution

### Requirements Files
- `requirements.txt` - Core dependencies
- `requirements-dev.txt` - Development dependencies
- `requirements-test.txt` - Testing dependencies
- `requirements-minimal.txt` - Minimal installation requirements
- `requirements-server.txt` - Server/API specific dependencies
- `requirements-production.txt` - Production deployment requirements

### Build & Development Configuration
- `Makefile` - Build automation and common tasks
- `package.json` - Node.js dependencies (for web UI components)
- `package-lock.json` - Node.js dependency lock file
- `pytest.ini` - Pytest configuration
- `tox.ini` - Tox testing configuration (if used)

### Version Control Configuration
- `.gitignore` - Git ignore patterns
- `.gitattributes` - Git attributes configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

### CI/CD Configuration
- `.github/` - GitHub Actions workflows and templates
- `docker-compose.yml` - Docker Compose for development (if needed)
- `Dockerfile` - Primary container definition (if needed)

### IDE Configuration (Optional)
- `.vscode/` - VS Code workspace settings
- `Pynomaly.code-workspace` - VS Code workspace file
- `.devcontainer/` - Development container configuration

## Mandatory Top-Level Directories

### Source Code (`src/`)
```
src/
├── pynomaly/                    # Main package
│   ├── __init__.py
│   ├── __main__.py
│   ├── domain/                  # Business logic (clean architecture)
│   │   ├── entities/           # Domain entities
│   │   ├── services/           # Domain services
│   │   ├── repositories/       # Repository interfaces
│   │   └── value_objects/      # Value objects
│   ├── application/            # Application layer
│   │   ├── services/           # Application services
│   │   ├── use_cases/          # Use case implementations
│   │   └── dto/                # Data Transfer Objects
│   ├── infrastructure/         # Infrastructure layer
│   │   ├── adapters/           # External service adapters
│   │   ├── repositories/       # Repository implementations
│   │   ├── persistence/        # Database and storage
│   │   ├── config/             # Configuration management
│   │   └── monitoring/         # Monitoring and logging
│   └── presentation/           # Presentation layer
│       ├── api/                # REST API endpoints
│       ├── cli/                # Command-line interface
│       ├── web/                # Web interface
│       └── sdk/                # SDK components
```

### Testing (`tests/`)
```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests
│   ├── domain/                 # Domain layer tests
│   ├── application/            # Application layer tests
│   ├── infrastructure/         # Infrastructure layer tests
│   └── presentation/           # Presentation layer tests
├── integration/                # Integration tests
│   ├── api/                    # API integration tests
│   ├── database/               # Database integration tests
│   └── external_services/      # External service integration tests
├── e2e/                        # End-to-end tests
│   ├── cli/                    # CLI end-to-end tests
│   ├── web/                    # Web UI end-to-end tests
│   └── api/                    # API end-to-end tests
├── performance/                # Performance and load tests
├── security/                   # Security tests
├── acceptance/                 # Acceptance tests
└── fixtures/                   # Test fixtures and data
```

### Documentation (`docs/`)
```
docs/
├── index.md                    # Documentation homepage
├── getting-started/            # Getting started guides
│   ├── installation.md
│   ├── quickstart.md
│   └── platform-specific/
├── user-guides/                # User documentation
│   ├── basic-usage/
│   ├── advanced-features/
│   └── troubleshooting/
├── developer-guides/           # Developer documentation
│   ├── architecture/
│   ├── contributing/
│   └── api-integration/
├── reference/                  # Reference documentation
│   ├── api/
│   ├── algorithms/
│   └── cli/
├── tutorials/                  # Step-by-step tutorials
├── examples/                   # Documentation examples
├── deployment/                 # Deployment guides
├── security/                   # Security documentation
├── development/                # Development standards (this file)
└── project/                    # Project-specific documentation
```

### Scripts (`scripts/`)
```
scripts/
├── development/                # Development utilities
│   ├── setup/                  # Environment setup scripts
│   ├── run/                    # Application runner scripts
│   └── maintenance/            # Maintenance utilities
├── testing/                    # Test automation scripts
│   ├── unit/                   # Unit test runners
│   ├── integration/            # Integration test runners
│   └── performance/            # Performance test runners
├── deployment/                 # Deployment scripts
│   ├── docker/                 # Docker deployment scripts
│   ├── kubernetes/             # Kubernetes deployment scripts
│   └── production/             # Production deployment scripts
├── analysis/                   # Analysis and reporting scripts
├── validation/                 # Validation and compliance scripts
├── generate/                   # Code and documentation generation
└── build/                      # Build and packaging scripts
```

### Configuration (`config/`)
```
config/
├── development/                # Development configurations
│   ├── database.yaml
│   ├── logging.yaml
│   └── features.yaml
├── production/                 # Production configurations
│   ├── database.yaml
│   ├── logging.yaml
│   └── security.yaml
├── testing/                    # Testing configurations
│   ├── pytest.ini
│   ├── tox.ini
│   └── coverage.ini
├── environments/               # Environment-specific requirements
│   ├── requirements-dev.txt
│   ├── requirements-prod.txt
│   └── requirements-test.txt
└── templates/                  # Configuration templates
    ├── .env.template
    └── config.yaml.template
```

### Reports (`reports/`)
```
reports/
├── coverage/                   # Test coverage reports
│   ├── html/                   # HTML coverage reports
│   └── xml/                    # XML coverage reports
├── security/                   # Security audit reports
│   ├── vulnerability-scans/
│   └── compliance-checks/
├── performance/                # Performance benchmark reports
│   ├── load-tests/
│   └── profiling/
├── quality/                    # Code quality reports
│   ├── linting/
│   ├── complexity/
│   └── duplication/
└── testing/                    # Test execution reports
    ├── unit-tests/
    ├── integration-tests/
    └── e2e-tests/
```

### Deployment (`deploy/`)
```
deploy/
├── docker/                     # Docker configurations
│   ├── Dockerfile.production
│   ├── Dockerfile.development
│   ├── docker-compose.yml
│   └── config/
├── kubernetes/                 # Kubernetes manifests
│   ├── base/
│   ├── overlays/
│   └── helm/
├── terraform/                  # Infrastructure as Code
│   ├── modules/
│   └── environments/
├── ansible/                    # Configuration management
└── artifacts/                  # Build artifacts and releases
```

### Examples (`examples/`)
```
examples/
├── basic/                      # Basic usage examples
│   ├── getting-started.py
│   └── simple-detection.py
├── advanced/                   # Advanced usage examples
│   ├── custom-algorithms.py
│   ├── ensemble-methods.py
│   └── real-time-detection.py
├── notebooks/                  # Jupyter notebooks
│   ├── tutorial.ipynb
│   └── advanced-analysis.ipynb
├── datasets/                   # Sample datasets
│   ├── synthetic/
│   └── real-world/
├── configs/                    # Example configurations
│   ├── autonomous-config.yaml
│   └── production-config.yaml
└── scripts/                    # Example scripts
    ├── batch-processing.py
    └── streaming-analysis.py
```

## Category-to-Directory Mapping

### File Categories and Target Directories

| File Category | Target Directory | Examples |
|---------------|------------------|----------|
| **Source Code** | `src/` | `*.py`, `*.pyx`, `*.pxd` |
| **Unit Tests** | `tests/unit/` | `test_*.py`, `*_test.py` |
| **Integration Tests** | `tests/integration/` | `test_integration_*.py` |
| **End-to-End Tests** | `tests/e2e/` | `test_e2e_*.py` |
| **Performance Tests** | `tests/performance/` | `test_performance_*.py` |
| **Security Tests** | `tests/security/` | `test_security_*.py` |
| **Documentation** | `docs/` | `*.md`, `*.rst`, `*.txt` |
| **Configuration** | `config/` | `*.yaml`, `*.yml`, `*.json`, `*.toml`, `*.ini` |
| **Scripts** | `scripts/` | `*.py`, `*.sh`, `*.ps1`, `*.bat` |
| **Examples** | `examples/` | `example_*.py`, `demo_*.py` |
| **Reports** | `reports/` | `*_report.*`, `*_analysis.*` |
| **Deployment** | `deploy/` | `Dockerfile.*`, `*.yaml`, `*.yml` |
| **Notebooks** | `examples/notebooks/` | `*.ipynb` |
| **Datasets** | `examples/datasets/` | `*.csv`, `*.json`, `*.parquet` |
| **Assets** | `src/*/static/` | `*.css`, `*.js`, `*.png`, `*.svg` |
| **Templates** | `templates/` | `*.j2`, `*.template` |

### Directory Categories and Purposes

| Directory Category | Purpose | Subdirectories |
|-------------------|---------|----------------|
| **Source (`src/`)** | Application source code | `pynomaly/` |
| **Tests (`tests/`)** | All test files | `unit/`, `integration/`, `e2e/`, `performance/` |
| **Documentation (`docs/`)** | Project documentation | `user-guides/`, `developer-guides/`, `reference/` |
| **Configuration (`config/`)** | Configuration files | `development/`, `production/`, `testing/` |
| **Scripts (`scripts/`)** | Automation scripts | `development/`, `testing/`, `deployment/` |
| **Examples (`examples/`)** | Usage examples | `basic/`, `advanced/`, `notebooks/` |
| **Reports (`reports/`)** | Generated reports | `coverage/`, `security/`, `performance/` |
| **Deployment (`deploy/`)** | Deployment artifacts | `docker/`, `kubernetes/`, `terraform/` |

## Treatment of Temporary, Build and Version Artifacts

### Temporary Files
- **Location**: NOT in repository
- **Handling**: Added to `.gitignore`
- **Examples**: 
  - `*.tmp`, `*.temp`, `*.bak`
  - `__pycache__/`, `*.pyc`, `*.pyo`
  - `.mypy_cache/`, `.pytest_cache/`
  - `*.log` (unless specifically needed)

### Build Artifacts
- **Location**: `deploy/artifacts/` OR `.gitignore`
- **Handling**: 
  - Generated during CI/CD
  - Stored in artifact repositories
  - NOT committed to source control
- **Examples**:
  - `dist/`, `build/`, `*.egg-info/`
  - `*.whl`, `*.tar.gz`
  - Docker images, compiled binaries

### Version Control Artifacts
- **Location**: NOT in repository
- **Handling**: Added to `.gitignore`
- **Examples**:
  - `.git/` (obviously)
  - `*.orig`, `*.rej`
  - Merge conflict files

### Runtime Artifacts
- **Location**: `storage/` OR `.gitignore`
- **Handling**: 
  - Development: Local storage, not committed
  - Production: External storage systems
- **Examples**:
  - Log files: `*.log`
  - Database files: `*.db`, `*.sqlite`
  - Cache files: `cache/`, `*.cache`
  - User data: `data/`, `uploads/`

### CI/CD Artifacts
- **Location**: External artifact storage
- **Handling**: 
  - GitHub Actions artifacts
  - Container registries
  - Package repositories
- **Examples**:
  - Test reports (temporary)
  - Coverage reports (temporary)
  - Performance benchmarks (archived)

## Prohibited Files and Directories

### Files NOT Allowed in Root
- `test_*.py` → `tests/`
- `*_test.py` → `tests/`
- `example_*.py` → `examples/`
- `demo_*.py` → `examples/`
- `setup_*.py` → `scripts/setup/`
- `run_*.py` → `scripts/run/`
- `deploy_*.py` → `scripts/deployment/`
- `*.config` → `config/`
- `*.log` → DELETE or `storage/`
- `*.tmp`, `*.temp`, `*.bak` → DELETE
- `requirements_*.txt` (except allowed ones) → `config/environments/`

### Directories NOT Allowed in Root
- `test/`, `tests_*/` → `tests/`
- `example/`, `examples_*/` → `examples/`
- `script/`, `scripts_*/` → `scripts/`
- `doc/`, `documentation/` → `docs/`
- `deploy_*/`, `deployment/` → `deploy/`
- `venv/`, `.venv/`, `env/` → DELETE
- `__pycache__/`, `.mypy_cache/` → DELETE
- `build/`, `dist/`, `*.egg-info/` → DELETE
- `node_modules/` → DELETE (should be in `.gitignore`)
- `coverage/`, `htmlcov/` → DELETE (should be in `reports/`)

## Naming Conventions

### General Rules
1. **Lowercase with underscores**: `my_module.py`
2. **Descriptive names**: `analyze_anomalies.py` not `analyze.py`
3. **Purpose indication**: `test_integration_api.py`
4. **Consistent patterns**: Follow established conventions

### File Naming Patterns
- **Source files**: `module_name.py`
- **Test files**: `test_<module>_<feature>.py`
- **Script files**: `<action>_<purpose>.py`
- **Configuration files**: `<environment>_<service>.yaml`
- **Documentation files**: `<TOPIC>_<TYPE>.md`

### Directory Naming Patterns
- **All lowercase**: `user_guides/`
- **Hyphens for readability**: `getting-started/`
- **Descriptive**: `api-integration/` not `api/`
- **Consistent**: Follow existing patterns

## Enforcement Mechanisms

### 1. Git Configuration
```gitignore
# Root directory protection
/test_*.py
/test_*.sh
/test_*.ps1
/*_test.py
/example_*.py
/demo_*.py
/setup_*.py
/run_*.py
/deploy_*.py

# Temporary files
*.tmp
*.temp
*.bak
*.orig
*.rej

# Build artifacts
/build/
/dist/
/*.egg-info/
__pycache__/
*.pyc
*.pyo

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db
```

### 2. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: file-organization
        name: File Organization Validation
        entry: python scripts/validation/validate_file_organization.py
        language: python
        pass_filenames: false
        always_run: true
```

### 3. CI/CD Validation
```yaml
# .github/workflows/file-organization.yml
name: File Organization Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate File Organization
        run: python scripts/validation/validate_file_organization.py
```

### 4. Development Tools
- `scripts/validation/validate_file_organization.py` - Structure validation
- `scripts/analysis/analyze_project_structure.py` - Structure analysis
- `scripts/validation/auto_fix_root_directory.py` - Automated cleanup

## Validation Commands

### Manual Validation
```bash
# Analyze current structure
python scripts/analysis/analyze_project_structure.py

# Validate organization
python scripts/validation/validate_file_organization.py

# Auto-fix violations (dry run)
python scripts/validation/auto_fix_root_directory.py --dry-run

# Auto-fix violations (execute)
python scripts/validation/auto_fix_root_directory.py --execute
```

### CI/CD Integration
```bash
# Pre-commit validation
pre-commit run file-organization --all-files

# CI validation
python -m pytest tests/structure/test_file_organization.py
```

## Migration Process

### 1. Assessment Phase
- Run structure analysis: `python scripts/analysis/analyze_project_structure.py`
- Review violations report
- Identify files requiring manual review

### 2. Automated Migration
- Run auto-fix tool: `python scripts/validation/auto_fix_root_directory.py`
- Review proposed changes
- Execute approved migrations

### 3. Manual Review
- Handle complex cases requiring human judgment
- Update references and imports
- Verify functionality after migration

### 4. Validation
- Run validation suite
- Execute tests
- Verify CI/CD pipeline

## Quality Gates

### Development
- **Pre-commit**: File organization validation
- **Local testing**: Structure validation tests
- **Code review**: Organization compliance check

### CI/CD Pipeline
- **Structure validation**: Automated organization check
- **Test execution**: Verify moved files function correctly
- **Documentation**: Auto-update documentation links

### Release Process
- **Final validation**: Comprehensive structure check
- **Documentation**: Update organization documentation
- **Tagging**: Tag compliant releases

## Benefits

### For Developers
- **Predictable structure**: Files always in expected locations
- **Reduced cognitive load**: Clear organization patterns
- **Faster navigation**: Intuitive directory structure
- **Better collaboration**: Consistent structure for all contributors

### For Project Management
- **Professional appearance**: Clean, organized repository
- **Automated maintenance**: Reduced manual organization effort
- **Scalability**: Structure supports project growth
- **Compliance**: Consistent standards enforcement

### For Documentation
- **Logical organization**: Docs organized by purpose and audience
- **Easy maintenance**: Predictable locations for updates
- **Comprehensive coverage**: Required documentation enforced
- **Cross-referencing**: Consistent linking structure

## Compliance Checklist

### Repository Structure
- [ ] Only essential files in project root
- [ ] All source code in `src/` directory
- [ ] All tests in `tests/` directory with appropriate subdirectories
- [ ] All documentation in `docs/` directory
- [ ] All scripts in `scripts/` directory
- [ ] All configuration in `config/` directory
- [ ] All examples in `examples/` directory
- [ ] All reports in `reports/` directory
- [ ] All deployment artifacts in `deploy/` directory

### File Organization
- [ ] No test files in root directory
- [ ] No script files in root directory
- [ ] No example files in root directory
- [ ] No temporary files committed
- [ ] No build artifacts committed
- [ ] No virtual environments in repository
- [ ] No IDE-specific files (unless in `.gitignore`)

### Naming Conventions
- [ ] Consistent file naming patterns
- [ ] Descriptive directory names
- [ ] Proper test file naming
- [ ] Consistent configuration file naming

### Automation
- [ ] Pre-commit hooks configured
- [ ] CI/CD validation enabled
- [ ] `.gitignore` patterns updated
- [ ] Validation scripts functional

### Documentation
- [ ] This standards document up-to-date
- [ ] Directory README files present
- [ ] Migration procedures documented
- [ ] Enforcement mechanisms documented

## Related Documentation

- [Contributing Guidelines](../developer-guides/contributing/CONTRIBUTING.md)
- [Development Setup](../developer-guides/DEVELOPMENT_SETUP.md)
- [Project Structure](../project/PROJECT_STRUCTURE.md)
- [CI/CD Pipeline](../deployment/production-deployment.md)
- [Code Quality Standards](../developer-guides/contributing/README.md)

---

**Document Control:**
- **Created**: 2025-01-08
- **Author**: Automated Standards Generation
- **Review Status**: Pending
- **Next Review**: 2025-01-15
- **Approval Required**: Yes
