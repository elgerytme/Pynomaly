# 🔧 Tools Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)

Development tools and utilities for maintaining code quality, architecture compliance, and repository governance in the monorepo.

## 🎯 Overview

The `tools` package provides essential development utilities for:

- **Domain Boundary Detection**: Enforcing clean architecture principles and preventing cross-domain violations
- **Repository Analysis**: Scripts for analyzing package structure, dependencies, and architectural compliance
- **Development Automation**: Tools for maintaining code quality and repository governance

This package supports the monorepo's commitment to clean architecture, domain-driven design, and enterprise-grade software engineering practices.

## 📦 Package Contents

### Domain Boundary Detector
- **Purpose**: Validates domain boundaries and prevents architectural violations
- **Location**: `domain_boundary_detector/`
- **Features**: Clean architecture enforcement, dependency analysis, violation reporting

### Analysis Scripts
- **Purpose**: Repository structure and dependency analysis
- **Location**: `scripts/`
- **Features**: Domain boundary checking, separation analysis, integration testing

## 🚀 Quick Start

### Installation

```bash
# Install the tools package
cd src/packages/tools/domain_boundary_detector/
pip install -e .

# Or install with development dependencies
pip install -e ".[dev,test]"
```

### Basic Usage

```python
# Using the Domain Boundary Detector
from domain_boundary_detector.core.domain.services import DomainBoundaryDetectorService
from domain_boundary_detector.core.application.services import ReporterService

# Initialize the detector
detector = DomainBoundaryDetectorService()

# Analyze a package for domain boundary violations
violations = detector.analyze_package("path/to/package")

# Generate a report
reporter = ReporterService()
report = reporter.generate_report(violations)
print(report)
```

### Command Line Usage

```bash
# Check domain boundaries across all packages
python src/packages/tools/scripts/check_domain_boundaries.py

# Run focused separation analysis
python src/packages/tools/scripts/focused_separation_analysis.py

# Perform repository-wide analysis
python src/packages/tools/scripts/repository_separation_analysis.py

# Run integration tests
python src/packages/tools/scripts/integration_test.py
```

## 🏗️ Architecture

The tools package follows clean architecture principles with clear separation of concerns:

```
tools/
├── domain_boundary_detector/           # Core domain boundary detection tool
│   ├── core/                          # Business logic layer
│   │   ├── domain/                    # Domain entities and business rules
│   │   │   ├── entities/              # Domain entities
│   │   │   ├── services/              # Domain services
│   │   │   ├── repositories/          # Repository interfaces
│   │   │   └── value_objects/         # Value objects
│   │   ├── application/               # Application layer
│   │   │   ├── services/              # Application services
│   │   │   └── use_cases/             # Use cases
│   │   └── dto/                       # Data transfer objects
│   ├── infrastructure/                # Infrastructure layer
│   │   ├── adapters/                  # External adapters
│   │   ├── persistence/               # Data persistence
│   │   └── external/                  # External integrations
│   ├── interfaces/                    # Interface layer
│   │   ├── api/                       # REST API endpoints
│   │   ├── cli/                       # Command-line interface
│   │   ├── python_sdk/                # Python SDK
│   │   └── web/                       # Web interface handlers
│   ├── tests/                         # Comprehensive test suite
│   └── pyproject.toml                 # Package configuration
└── scripts/                           # Analysis and utility scripts
    ├── check_domain_boundaries.py     # Main boundary checking script
    ├── focused_separation_analysis.py # Focused analysis tool
    ├── repository_separation_analysis.py # Repository-wide analysis
    ├── simple_domain_check.py         # Simple boundary validation
    └── integration_test.py             # Integration testing script
```

## 🔍 Domain Boundary Detection

### Key Features

- **Violation Detection**: Identifies cross-domain imports and dependencies that violate clean architecture
- **Architectural Compliance**: Ensures packages follow domain-driven design principles
- **Automated Reporting**: Generates detailed reports with recommendations for fixing violations
- **Integration Support**: Works with CI/CD pipelines for continuous architecture validation

### Example Violations Detected

```python
# ❌ Cross-domain violation - Application importing from another domain
from other_domain.entities import SomeEntity

# ❌ Infrastructure layer importing from presentation
from presentation.controllers import SomeController

# ❌ Direct database access from domain layer
import sqlite3

# ✅ Correct - Using dependency injection and interfaces
from .repositories import UserRepositoryInterface
```

### Usage Examples

```python
# Analyze specific package
from domain_boundary_detector.core.domain.services import AnalyzerService

analyzer = AnalyzerService()
violations = analyzer.analyze_directory("/path/to/package")

for violation in violations:
    print(f"Violation: {violation.description}")
    print(f"File: {violation.file_path}")
    print(f"Recommendation: {violation.recommendation}")
```

## 📊 Analysis Scripts

### check_domain_boundaries.py
Comprehensive domain boundary analysis across all packages.

```bash
python scripts/check_domain_boundaries.py
```

**Output:**
- Total packages analyzed
- Violations found per package
- Clean packages identified
- Detailed recommendations

### focused_separation_analysis.py
Targeted analysis for specific architectural concerns.

```bash
python scripts/focused_separation_analysis.py
```

### repository_separation_analysis.py
Repository-wide architectural analysis and reporting.

```bash
python scripts/repository_separation_analysis.py
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=domain_boundary_detector --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Test Structure

```
tests/
├── unit/                              # Unit tests
│   ├── test_analyzer.py               # Analyzer service tests
│   ├── test_domain_boundary_detector_entity.py # Entity tests
│   ├── test_domain_boundary_detector_service.py # Service tests
│   └── test_scanner.py                # Scanner tests
└── integration/                       # Integration tests
    └── test_end_to_end.py             # End-to-end testing
```

## ⚙️ Configuration

### Environment Variables

```bash
# Configure analysis scope
export ANALYSIS_ROOT_PATH="/path/to/packages"
export ANALYSIS_EXCLUDE_PATTERNS="*/tests/*,*/migrations/*"

# Reporting configuration
export REPORT_FORMAT="json"  # json, yaml, text
export REPORT_OUTPUT_PATH="/path/to/reports"
```

### Configuration Files

The domain boundary detector can be configured via `pyproject.toml`:

```toml
[tool.domain_boundary_detector]
# Analysis configuration
exclude_patterns = ["*/tests/*", "*/migrations/*"]
include_patterns = ["*/src/*"]

# Violation levels
strict_mode = true
warning_threshold = 5
error_threshold = 10

# Output configuration
report_format = "json"
generate_recommendations = true
```

## 🚀 Development

### Setting Up Development Environment

```bash
# Clone and navigate to tools package
cd src/packages/tools/domain_boundary_detector/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Run linting
ruff check . --fix

# Run type checking
mypy domain_boundary_detector/

# Run security checks
bandit -r domain_boundary_detector/

# Run tests with coverage
pytest tests/ --cov=domain_boundary_detector --cov-report=html
```

### Adding New Analysis Scripts

1. Create your script in `scripts/`
2. Follow the existing pattern for CLI interface
3. Add comprehensive documentation
4. Include error handling and logging
5. Add corresponding tests

Example script template:

```python
#!/usr/bin/env python3
"""Your analysis script description."""

import argparse
import logging
import sys
from pathlib import Path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--path", type=Path, help="Analysis path")
    parser.add_argument("--output", type=Path, help="Output file")
    
    args = parser.parse_args()
    
    # Your analysis logic here
    
if __name__ == "__main__":
    main()
```

## 🔧 API Reference

### Core Classes

#### DomainBoundaryDetectorService
Main service for detecting domain boundary violations.

```python
class DomainBoundaryDetectorService:
    def analyze_package(self, package_path: str) -> List[Violation]:
        """Analyze a package for domain boundary violations."""
        
    def generate_report(self, violations: List[Violation]) -> str:
        """Generate a formatted report from violations."""
```

#### AnalyzerService
Low-level analysis functionality.

```python
class AnalyzerService:
    def scan_directory(self, path: str) -> List[str]:
        """Scan directory for Python files."""
        
    def analyze_imports(self, file_path: str) -> List[Import]:
        """Analyze imports in a Python file."""
```

#### ReporterService
Report generation and formatting.

```python
class ReporterService:
    def generate_report(self, data: Dict) -> str:
        """Generate formatted report."""
        
    def save_report(self, report: str, output_path: str) -> None:
        """Save report to file."""
```

## 🤝 Contributing

We welcome contributions to the tools package! This package is essential for maintaining architectural quality across the monorepo.

### How to Contribute

1. **Follow Clean Architecture**: Maintain the existing architectural patterns
2. **Add Tests**: Ensure comprehensive test coverage for new features
3. **Update Documentation**: Keep documentation current with changes
4. **Run Quality Checks**: Ensure all quality gates pass before submission

### Areas for Contribution

- **New Analysis Tools**: Additional architectural compliance checks
- **Reporting Enhancements**: Better visualization and reporting formats  
- **IDE Integration**: Plugins for popular development environments
- **Performance Improvements**: Optimization of analysis algorithms
- **Documentation**: Examples, tutorials, and best practices

For detailed guidelines, see the main repository [CONTRIBUTING.md](../../../docs/developer-guides/contributing/CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Documentation

- [Clean Architecture Guidelines](../../../docs/architecture/clean-architecture.md)
- [Domain-Driven Design Principles](../../../docs/architecture/domain-driven-design.md)
- [Repository Governance](../../../docs/governance/repository-structure.md)
- [Development Best Practices](../../../docs/developer-guides/best-practices.md)

---

**Note**: This package is essential for maintaining architectural integrity across the monorepo. Regular use of these tools helps ensure code quality, architectural compliance, and long-term maintainability.