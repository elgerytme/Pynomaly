# Domain Boundary Detector

A powerful tool for detecting and preventing domain boundary violations in monorepo architectures. Enforces clean architecture principles by analyzing cross-package dependencies and ensuring proper domain isolation.

## Overview

The Domain Boundary Detector helps maintain architectural integrity by:

- 🔍 **Scanning** all Python imports across your monorepo
- 🚨 **Detecting** violations of domain boundaries  
- 📊 **Reporting** issues with actionable suggestions
- 🛡️ **Enforcing** clean architecture principles in CI/CD

## Key Features

- **AST-based scanning** for accurate import detection
- **Configurable domain rules** with YAML configuration
- **Multiple output formats** (console, JSON, Markdown)
- **CI/CD integration** with exit codes
- **Exception management** for approved violations
- **Smart detection** of string references and type hints

## Structure

```
domain_boundary_detector/
├── core/
│   ├── domain/              # Domain layer
│   │   ├── entities/        # Domain entities
│   │   ├── services/        # Domain services
│   │   ├── value_objects/   # Value objects
│   │   ├── repositories/    # Repository interfaces
│   │   └── exceptions/      # Domain exceptions
│   ├── application/         # Application layer
│   │   ├── services/        # Application services
│   │   └── use_cases/       # Use cases
│   └── dto/                 # Data transfer objects
├── infrastructure/          # Infrastructure layer
│   ├── adapters/           # External adapters
│   ├── persistence/        # Data persistence
│   └── external/           # External services
├── interfaces/             # Interface layer
│   ├── api/               # REST API endpoints
│   ├── cli/               # Command-line interface
│   ├── web/               # Web interface
│   └── python_sdk/        # Python SDK
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── docs/                   # Documentation
```

## Domain Boundaries

This package follows strict domain boundaries:

### Allowed Concepts
- Domain Boundary Detector-specific business logic
- Domain entities and value objects
- Domain services and repositories
- Use cases and application services

### Prohibited Concepts
- Generic software infrastructure (belongs in `software/` package)
- Other domain concepts (belongs in respective domain packages)
- Cross-domain references (use interfaces and dependency injection)

## Quick Start

### Installation

```bash
# From monorepo root
pip install -e src/packages/tools/domain_boundary_detector
```

### Basic Usage

```bash
# Scan entire monorepo
python -m domain_boundary_detector.cli scan

# Scan specific package
python -m domain_boundary_detector.cli scan --path src/packages/ai/mlops

# Generate report in different formats
python -m domain_boundary_detector.cli scan --format json --output violations.json
python -m domain_boundary_detector.cli scan --format markdown --output violations.md

# Strict mode for CI/CD (exits with error on violations)
python -m domain_boundary_detector.cli scan --strict

# Initialize configuration
python -m domain_boundary_detector.cli init
```

## Example Output

```
Domain Boundary Scan Report
==========================

❌ VIOLATIONS FOUND: 3
  ● Critical: 1
  ● Warning: 1  
  ● Info: 1

CRITICAL VIOLATIONS:
----------------------------------------
1. Cross Domain Import
   File: src/packages/ai/mlops/core/services/billing_integration.py:12
   Import: from finance.billing import calculate_cost
   Violation: ai domain importing from finance domain
   Suggestion: Use event-driven communication or API calls instead of direct imports

WARNING VIOLATIONS:
----------------------------------------
2. Private Module Access
   File: src/packages/data/analytics/utils/helpers.py:8
   Import: from shared._internal import secret_function
   Violation: Accessing private module (starts with _)
   Suggestion: Use public API instead of private implementation details
```

## Configuration

Create a `.domain-boundaries.yaml` file in your monorepo root:

```yaml
domains:
  ai:
    packages:
      - ai/mlops
      - ai/ml_platform
    allowed_dependencies:
      - shared
      - infrastructure
      
  finance:
    packages:
      - finance/billing
      - finance/payments
    allowed_dependencies:
      - shared
      - infrastructure

rules:
  - name: no_cross_domain_imports
    severity: critical
    exceptions:
      - from: ai/mlops
        to: finance/billing
        reason: "Cost calculation for model training"
        expires: "2024-12-31"
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Check Domain Boundaries
  run: |
    pip install -e src/packages/tools/domain_boundary_detector
    python -m domain_boundary_detector.cli scan --strict --format json --output violations.json
  
- name: Upload Violations Report
  if: failure()
  uses: actions/upload-artifact@v2
  with:
    name: domain-violations
    path: violations.json
```

### Pre-commit Hook

```yaml
repos:
  - repo: local
    hooks:
      - id: domain-boundaries
        name: Check domain boundaries
        entry: python -m domain_boundary_detector.cli scan --strict
        language: system
        pass_filenames: false
```

## Development

### Setup
```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
python scripts/install_domain_hooks.py
```

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run domain boundary validation
python scripts/domain_boundary_validator.py
```

### Code Quality
```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

This package follows Clean Architecture principles:

1. **Domain Layer**: Core business logic
2. **Application Layer**: Use cases and application services
3. **Infrastructure Layer**: External concerns
4. **Interface Layer**: User interfaces

## Domain Compliance

This package maintains strict domain boundary compliance:

- **Validation**: Automated domain boundary validation
- **Enforcement**: Pre-commit hooks and CI/CD integration
- **Monitoring**: Continuous compliance monitoring
- **Documentation**: Clear domain boundary rules

## Contributing

1. Follow domain boundary rules
2. Add comprehensive tests
3. Update documentation
4. Validate domain compliance
5. Submit pull request

## License

MIT License - see LICENSE file for details.
