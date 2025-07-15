# Pynomaly Naming Conventions Guide

This document establishes consistent naming conventions for the Pynomaly project to improve code maintainability, readability, and developer experience.

## 🎯 Overview

Consistent naming conventions are essential for:
- **Developer Productivity**: Easier navigation and understanding
- **Code Maintainability**: Reduced cognitive overhead
- **Tool Integration**: Better IDE and linting support
- **Team Collaboration**: Shared understanding of patterns

## 📝 Python Code Conventions (PEP 8 Compliant)

### Classes
- **Format**: PascalCase
- **Examples**: `AnomalyDetector`, `DetectionResult`, `ConfigurationManager`
- **Pattern**: `class AnomalyDetector:`

### Functions and Methods
- **Format**: snake_case
- **Examples**: `detect_anomalies()`, `calculate_threshold()`, `validate_input()`
- **Pattern**: `def detect_anomalies(data):`

### Variables and Attributes
- **Format**: snake_case
- **Examples**: `detection_result`, `anomaly_score`, `threshold_value`
- **Pattern**: `anomaly_score = 0.85`

### Constants
- **Format**: UPPER_SNAKE_CASE
- **Examples**: `MAX_RETRY_COUNT`, `DEFAULT_THRESHOLD`, `API_VERSION`
- **Pattern**: `MAX_RETRY_COUNT = 3`

### Modules and Packages
- **Format**: snake_case
- **Examples**: `anomaly_detector.py`, `detection_service.py`, `value_objects/`
- **Pattern**: `from pynomaly.domain.entities import detector`

## 📁 File and Directory Conventions

### Python Files
- **Format**: snake_case.py
- **Examples**: `anomaly_detector.py`, `detection_service.py`, `training_automation.py`
- **Pattern**: `src/pynomaly/domain/entities/detector.py`

### Test Files
- **Format**: test_*.py (pytest convention)
- **Examples**: `test_anomaly_detector.py`, `test_detection_service.py`
- **Pattern**: `tests/unit/test_anomaly_detector.py`

### Python Package Directories
- **Format**: snake_case
- **Examples**: `domain/`, `application/`, `infrastructure/`, `value_objects/`
- **Pattern**: `src/pynomaly/domain/entities/`

### Configuration Files
- **Format**: kebab-case with standard extensions
- **YAML**: `.yaml` (preferred over `.yml`)
- **Examples**: `docker-compose.yaml`, `pre-commit-config.yaml`, `ci-config.yaml`
- **Pattern**: `deployment/configs/database-config.yaml`

### Documentation Files
- **Format**: UPPER_KEBAB_CASE.md for important docs, kebab-case.md for others
- **Examples**: `README.md`, `CHANGELOG.md`, `api-reference.md`, `developer-guide.md`
- **Pattern**: `docs/guides/getting-started.md`

### Docker Files
- **Format**: Dockerfile[.suffix] where suffix is snake_case
- **Examples**: `Dockerfile`, `Dockerfile.dev`, `Dockerfile.production`, `Dockerfile.multi_python`
- **Pattern**: `deployment/docker/Dockerfile.api_server`

## 🔧 Specific Project Patterns

### DTO (Data Transfer Objects)
- **Pattern**: `*_dto.py` for modules, `*DTO` for classes
- **Examples**: `detection_request_dto.py` → `DetectionRequestDTO`

### Services
- **Pattern**: `*_service.py` for modules, `*Service` for classes
- **Examples**: `anomaly_detection_service.py` → `AnomalyDetectionService`

### Use Cases
- **Pattern**: `*_use_case.py` for modules, `*UseCase` for classes
- **Examples**: `train_detector_use_case.py` → `TrainDetectorUseCase`

### Repositories
- **Pattern**: `*_repository.py` for modules, `*Repository` for classes
- **Examples**: `detector_repository.py` → `DetectorRepository`

### Value Objects
- **Pattern**: snake_case.py for modules, PascalCase for classes
- **Examples**: `anomaly_score.py` → `AnomalyScore`

## 🚫 Deprecated Patterns to Avoid

### ❌ Incorrect Directory Naming
```
# Avoid kebab-case in Python packages
src/apps/anomaly-detector/  # ❌
src/data-science/           # ❌

# Use snake_case instead
src/apps/anomaly_detector/  # ✅
src/data_science/           # ✅
```

### ❌ Incorrect File Extensions
```
# Avoid mixed YAML extensions
config.yml          # ❌
settings.yml         # ❌

# Use .yaml consistently
config.yaml          # ✅
settings.yaml        # ✅
```

### ❌ Incorrect Test Naming
```
# Avoid missing test_ prefix
anomaly_test.py      # ❌
detector_tests.py    # ❌

# Use test_ prefix
test_anomaly.py      # ✅
test_detector.py     # ✅
```

### ❌ Incorrect Dockerfile Naming
```
# Avoid kebab-case suffixes
Dockerfile.multi-python   # ❌
Dockerfile.ui-testing     # ❌

# Use snake_case suffixes
Dockerfile.multi_python   # ✅
Dockerfile.ui_testing     # ✅
```

## 🔍 Automated Enforcement

### Pre-commit Hooks
We use automated tools to enforce naming conventions:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-naming-conventions
        name: Check naming conventions
        entry: scripts/check_naming_conventions.py
        language: python
        types: [python]
```

### Linting Rules
Naming conventions are enforced through:
- **flake8-naming**: Python naming convention linter
- **ruff**: Fast Python linter with naming rules
- **Custom scripts**: Project-specific validation

### CI/CD Integration
Naming convention checks are part of our CI/CD pipeline:
- **Pre-commit validation**: Local development
- **GitHub Actions**: Pull request validation
- **Quality gates**: Prevents merging non-compliant code

## 📋 Migration Checklist

When updating existing code to follow these conventions:

### File and Directory Updates
- [ ] Rename kebab-case directories to snake_case
- [ ] Update all `.yml` files to `.yaml`
- [ ] Fix Dockerfile suffix naming
- [ ] Ensure test files follow `test_*.py` pattern

### Code Updates
- [ ] Update import statements after file renames
- [ ] Fix any naming violations in Python code
- [ ] Update configuration references
- [ ] Update documentation references

### Documentation Updates
- [ ] Update all documentation to reflect new naming
- [ ] Update examples and tutorials
- [ ] Update API documentation
- [ ] Update deployment guides

## 🏆 Benefits of Consistent Naming

### Developer Experience
- **Faster Navigation**: Predictable file and directory locations
- **Reduced Cognitive Load**: Consistent patterns reduce mental overhead
- **Better IDE Support**: Improved auto-completion and navigation
- **Easier Onboarding**: New developers learn patterns quickly

### Code Quality
- **Improved Maintainability**: Easier to understand and modify code
- **Better Tool Integration**: Linters and formatters work more effectively
- **Reduced Bugs**: Consistent patterns reduce naming-related errors
- **Enhanced Readability**: Code is more self-documenting

### Team Collaboration
- **Shared Understanding**: Everyone follows the same patterns
- **Faster Code Reviews**: Less time spent on style discussions
- **Consistent Contributions**: All contributors follow same standards
- **Professional Appearance**: Code looks polished and well-maintained

## 📚 References

- [PEP 8 – Style Guide for Python Code](https://pep8.org/)
- [pytest naming conventions](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#conventions-for-python-test-discovery)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Code Style](https://black.readthedocs.io/en/stable/)

---

*This document is part of the Pynomaly project quality standards. For questions or suggestions, please create an issue or discussion in the project repository.*