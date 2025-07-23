# 🔍 Data Intelligence Platform - Open Source Monorepo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Clean Architecture](https://img.shields.io/badge/architecture-clean-green.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![Domain Driven Design](https://img.shields.io/badge/design-DDD-orange.svg)](https://www.domainlanguage.com/ddd/)

This is a comprehensive, production-ready open source platform for data intelligence and machine learning across various data types and domains. Built with enterprise-grade architecture, modern Python practices, and designed for scalability, security, and extensibility.

🚀 **Production-Ready** • 🔒 **Enterprise Security** • 📊 **Full Observability** • 🏗️ **Clean Architecture** • 🤖 **Advanced ML**

## 🏗️ Platform Architecture

This repository demonstrates modern software engineering practices with a focus on maintainability, scalability, and clean architecture. It contains multiple domain packages specialized for data intelligence, machine learning, shared infrastructure, and comprehensive development tooling.

```
repository/
├── src/packages/           # 🎯 Domain packages (clean architecture)
│   ├── core/              # Shared domain logic & foundational patterns
│   ├── anomaly_detection/ # Specialized analytics and statistical modeling domain
│   ├── machine_learning/  # ML operations and model lifecycle management
│   ├── people_ops/        # User management and authentication domain
│   ├── mathematics/       # Mathematical computations and statistics
│   ├── data_platform/     # Data processing and quality assurance
│   ├── infrastructure/    # Cross-cutting infrastructure concerns
│   ├── interfaces/        # Presentation layer (CLI, API, Web)
│   ├── enterprise/        # Enterprise governance and compliance
│   ├── services/          # Application service orchestration
│   └── testing/           # Shared testing utilities and frameworks
├── pkg/                   # 🔗 External dependencies
│   ├── vendor_dependencies/ # Vendored third-party packages
│   └── custom_forks/      # Customized package forks
├── scripts/              # 🛠️ Automation and tooling
│   ├── governance/       # Repository structure enforcement
│   ├── analysis/         # Code analysis and metrics
│   └── cleanup/          # Maintenance and cleanup automation
├── templates/            # 📋 Code generation templates
│   └── package/         # Standard package structure templates
├── reports/             # 📊 Generated analysis reports
│   └── analysis/        # Repository health and metrics
├── deployment/          # 🚀 Infrastructure as code
├── configs/            # ⚙️ Shared configuration files
├── docs/               # 📚 Comprehensive documentation
└── tests/              # 🧪 Cross-package integration tests
```

### 📦 Package Organization Principles

All packages follow **Domain-Driven Design** and **Clean Architecture** principles for maximum maintainability and testability:

#### 🏢 **Domain Layer** (Pure Business Logic)
- **`core/`**: Shared domain entities, value objects, and foundational patterns
- **`anomaly_detection/`**: Data analysis domain with detection algorithms and models
- **`mathematics/`**: Mathematical domain with statistical computations and utilities

#### 🚀 **Application Layer** (Use Cases and Orchestration)
- **`machine_learning/`**: ML workflow orchestration and model lifecycle management
- **`people_ops/`**: User management, authentication, and authorization workflows
- **`data_platform/`**: Data processing pipelines and quality assurance workflows
- **`enterprise/`**: Governance, compliance, and multi-tenancy orchestration
- **`services/`**: Cross-domain application services and integration logic

#### 🔧 **Infrastructure Layer** (External Concerns)
- **`infrastructure/`**: Database, messaging, monitoring, and deployment adapters
- **`interfaces/`**: User interfaces (CLI, REST API, Web UI) and external integrations

#### 📋 **Standard Package Structure**
Each package maintains consistent organization:
```
package_name/
├── package_name/         # Source code following clean architecture
│   ├── domain/          # Domain layer (entities, value objects, services)
│   ├── application/     # Application layer (use cases, services, DTOs)
│   ├── infrastructure/  # Infrastructure layer (adapters, repositories)
│   └── presentation/    # Presentation layer (controllers, serializers)
├── tests/              # Comprehensive test suite
├── docs/               # Package-specific documentation
├── README.md           # Package overview and usage
├── pyproject.toml      # Package configuration and dependencies
└── BUCK               # Build system configuration
```

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/monorepo.git
cd monorepo

# Create and activate virtual environment
python -m venv environments/.venv
source environments/.venv/bin/activate  # Linux/macOS
# environments\.venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,test]"

# Verify setup
python -c "import monorepo; print('Setup successful')"
```

### Repository Structure Exploration

```bash
# Validate repository structure
python scripts/governance/package_structure_enforcer.py

# Analyze package dependencies
python scripts/analysis/dependency_analyzer.py

# Check for build artifacts
python scripts/governance/build_artifacts_checker.py

# Run all governance checks
python scripts/governance/root_directory_checker.py
```

### Working with Packages

```python
# Example: Using the core domain patterns
from src.packages.core.domain.entities import BaseEntity
from src.packages.core.domain.value_objects import Identifier

# Example: Data platform usage
from src.packages.data_platform.application.services import DataQualityService
from src.packages.data_platform.domain.entities import Dataset

# Example: Infrastructure patterns
from src.packages.infrastructure.persistence import Repository
from src.packages.infrastructure.monitoring import MetricsCollector
```

## 🏢 Enterprise Architecture Features

This monorepo demonstrates enterprise-ready software engineering practices:

- **🏗️ Clean Architecture**: Strict separation of concerns with domain-driven design
- **🔐 Security**: Comprehensive security patterns and compliance frameworks
- **📊 Observability**: Built-in monitoring, metrics, and distributed tracing
- **🌐 Scalability**: Multi-tenancy patterns and resource management
- **⚙️ DevOps**: Automated testing, deployment, and infrastructure management
- **🔄 Governance**: Repository structure enforcement and dependency management
- **📋 Standards**: Consistent coding standards, documentation, and testing patterns

## 🔧 Development Tooling & Automation

This monorepo includes comprehensive automation for maintaining code quality and repository governance:

### Repository Governance

```bash
# Validate repository structure and organization
python scripts/governance/package_structure_enforcer.py

# Check for build artifacts and cleanup
python scripts/governance/build_artifacts_checker.py

# Validate root directory organization
python scripts/governance/root_directory_checker.py

# Auto-fix common structural issues
python scripts/governance/package_structure_enforcer.py --fix
```

### Code Quality & Security

```bash
# Run linting with ruff (fast, comprehensive)
ruff check src/ tests/

# Run type checking with MyPy (strict mode)
mypy src/packages/

# Run security scan with Bandit
bandit -r src/

# Check for known vulnerabilities
safety check --full-report
pip-audit
```

### Testing & Analysis

```bash
# Run comprehensive test suite
pytest tests/ --cov=src/packages/

# Run package-specific tests
pytest src/packages/core/tests/

# Generate dependency analysis
python scripts/analysis/dependency_analyzer.py

# Performance benchmarking
python scripts/analysis/performance_profiler.py
```

These tools ensure enterprise-grade code quality, security, and maintainability across all packages.

## 🎯 Core Capabilities

### Architecture & Design Patterns

- 🏗️ **Clean Architecture**: Strict layered architecture with dependency inversion
- 🔄 **Domain-Driven Design**: Rich domain models with ubiquitous language
- 🔌 **Hexagonal Architecture**: Ports and adapters pattern for external integrations
- 🏭 **Repository Pattern**: Consistent data access abstraction
- 🎯 **Strategy Pattern**: Pluggable algorithm implementations
- 🔗 **Dependency Injection**: IoC container for loose coupling

### Development Infrastructure

- 🛡️ **Type Safety**: Comprehensive type coverage with mypy strict mode
- ✅ **Testing**: Multi-layered testing with unit, integration, and E2E tests
- 📊 **Coverage**: High test coverage with automated reporting
- 🔍 **Code Quality**: Automated linting, formatting, and static analysis
- 📈 **Performance**: Benchmarking and profiling capabilities
- 🔐 **Security**: Security scanning and vulnerability assessment

### Package Management

- 📦 **Modular Design**: Independent packages with clear boundaries
- 🔗 **Dependency Management**: Centralized dependency configuration
- 🏗️ **Build System**: Hatch-based build system with unified environments
- 📋 **Standards**: Consistent package structure and documentation
- 🔄 **Governance**: Automated structure validation and enforcement

### Developer Experience

- ⚡ **CLI Tools**: Rich command-line interface with comprehensive help
- 🌐 **Web Interface**: Modern web UI with real-time updates
- 🚀 **API**: RESTful API with OpenAPI documentation
- 📚 **Documentation**: Comprehensive guides and API references
- 🔧 **Automation**: Automated setup, testing, and deployment scripts

## 📥 Installation & Setup

### Prerequisites

```bash
# Ensure Python 3.11+ is installed
python --version  # Should show 3.11 or higher

# Git for repository cloning
git --version
```

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/monorepo.git
cd monorepo

# Create virtual environment in organized structure
python -m venv environments/.venv

# Activate environment
# Linux/macOS:
source environments/.venv/bin/activate
# Windows:
environments\.venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Verify installation
python -c "import monorepo; print('Installation successful')"
```

### Package Installation Options

```bash
# Core packages only
pip install -e .

# Full development setup
pip install -e ".[dev,test,lint]"

# Specific package groups
pip install -e ".[ml]"           # Machine learning packages
pip install -e ".[data]"         # Data platform packages  
pip install -e ".[enterprise]"   # Enterprise features
pip install -e ".[interfaces]"   # CLI, API, and web interfaces

# All packages
pip install -e ".[all]"
```

### Environment Organization

This monorepo uses a centralized environment structure to maintain organization:

```bash
environments/
├── .venv/              # Main development environment
├── .test_env/          # Testing environment
├── .prod_env/          # Production-like environment
└── .docs_env/          # Documentation building environment
```

**Benefits**:
- Keeps project root clean and organized
- Centralized environment management
- Clear separation of different environment purposes
- Easy environment switching for different tasks

### Cross-Platform Compatibility

This monorepo is designed to work seamlessly across different operating systems and development environments:

**Supported Platforms:**
- **Linux/Unix**: Full compatibility with bash shell environments
- **macOS**: Complete support for all features and commands  
- **Windows**: Full compatibility with PowerShell and Command Prompt
- **WSL/WSL2**: Tested and verified on Windows Subsystem for Linux

**Development Environment Support:**
- **Virtual Environments**: `venv`, `virtualenv`, `conda`, `pipenv`, `poetry`
- **Python Versions**: 3.11, 3.12, 3.13+
- **Package Managers**: pip, conda, poetry, pipenv
- **IDEs**: VS Code, PyCharm, Vim/Neovim with appropriate configurations

**Deployment Options:**
- **Local Development**: Standard Python development setup
- **Container Deployment**: Docker support for all platforms
- **Cloud Deployment**: AWS, Azure, GCP compatible infrastructure

## 🚀 Quick Start Guide

### Repository Exploration

After installation, explore the monorepo structure and capabilities:

```bash
# Repository structure and governance
python scripts/governance/package_structure_enforcer.py   # Validate structure
python scripts/governance/root_directory_checker.py       # Check organization
python scripts/analysis/dependency_analyzer.py            # Analyze dependencies

# Package management
python scripts/analysis/package_metrics.py                # Package statistics
python scripts/governance/build_artifacts_checker.py      # Check build artifacts

# Development workflow
pytest tests/                                              # Run test suite
ruff check src/                                           # Code quality check
mypy src/packages/                                        # Type checking
```

### Working with Individual Packages

```bash
# Explore package structure
ls src/packages/                    # List all packages
ls src/packages/core/              # Explore core package
ls src/packages/data_platform/     # Explore data platform

# Run package-specific tests
pytest src/packages/core/tests/
pytest src/packages/infrastructure/tests/

# Check package documentation
cat src/packages/core/README.md
cat src/packages/data_platform/README.md
```

### CLI Interface (if available)

```bash
# Show all available commands
monorepo --help

# Basic system information
monorepo version          # Show version info
monorepo status           # System health check

# Package operations
monorepo packages list    # List available packages
monorepo packages info    # Package information

# Development tools
monorepo dev validate     # Validate repository structure
monorepo dev analyze      # Run analysis tools
monorepo dev test         # Run test suites
```

### Python API Examples

```python
# Example: Using core domain patterns
from src.packages.core.domain.entities import BaseEntity
from src.packages.core.domain.value_objects import Identifier
from src.packages.core.application.use_cases import BaseUseCase

# Example: Infrastructure patterns
from src.packages.infrastructure.persistence import Repository
from src.packages.infrastructure.monitoring import MetricsCollector
from src.packages.infrastructure.config import ConfigurationManager

# Example: Working with the data platform
from src.packages.data_platform.domain.entities import Dataset
from src.packages.data_platform.application.services import DataQualityService
from src.packages.data_platform.infrastructure.adapters import DatabaseAdapter

# Example: Enterprise patterns
from src.packages.enterprise.domain.entities import Tenant
from src.packages.enterprise.application.services import GovernanceService
from src.packages.people_ops.domain.entities import User

def demonstrate_architecture():
    """Demonstrate clean architecture patterns."""
    
    # Repository pattern example
    repository = Repository()
    
    # Use case orchestration
    use_case = BaseUseCase(repository)
    
    # Metrics collection
    metrics = MetricsCollector()
    metrics.record_event("example_executed")
    
    print("Clean architecture patterns demonstrated successfully!")

if __name__ == "__main__":
    demonstrate_architecture()
```

### Web Interface & API

If web interfaces are available, access them after starting the development server:

```bash
# Start development server (if available)
uvicorn src.packages.interfaces.api.app:app --reload --port 8000

# Access interfaces
# API Documentation: http://localhost:8000/docs
# Web Interface: http://localhost:8000/app
# Health Check: http://localhost:8000/health
```

**Features** (when implemented):
- **RESTful API**: Clean API design with OpenAPI documentation
- **Web Interface**: Modern UI with responsive design
- **Real-time Updates**: WebSocket support for live data
- **Progressive Web App**: Offline capability and installable interface

## 🏗️ Architecture Overview

This monorepo implements **Clean Architecture**, **Domain-Driven Design (DDD)**, and **Hexagonal Architecture (Ports & Adapters)** across all packages:

```
src/packages/{package_name}/
├── domain/              # Pure business logic (no external dependencies)
│   ├── entities/        # Business entities and aggregate roots
│   ├── value_objects/   # Immutable value objects
│   ├── services/        # Domain services and business rules
│   ├── repositories/    # Repository interfaces (not implementations)
│   └── exceptions/      # Domain-specific exception hierarchy
├── application/         # Orchestrate use cases without implementation details
│   ├── use_cases/       # Application use cases and workflows
│   ├── services/        # Application services
│   ├── dto/             # Data transfer objects
│   └── ports/           # Output port interfaces
├── infrastructure/      # All external integrations and adapters
│   ├── adapters/        # External service adapters (databases, APIs, etc.)
│   ├── persistence/     # Repository implementations
│   ├── config/          # Configuration and dependency injection
│   └── monitoring/      # Observability and health checks
└── presentation/        # User interfaces and external APIs
    ├── api/             # REST API controllers
    ├── cli/             # Command-line interface
    ├── web/             # Web interface (if applicable)
    └── serializers/     # Data serialization/deserialization
```

### Design Patterns Implemented

- **Repository Pattern**: Clean data access abstraction across all packages
- **Factory Pattern**: Object creation and configuration management
- **Strategy Pattern**: Pluggable algorithm and service implementations
- **Observer Pattern**: Event-driven architecture and notifications
- **Decorator Pattern**: Cross-cutting concerns and middleware
- **Chain of Responsibility**: Request processing and validation pipelines
- **Dependency Injection**: IoC container for loose coupling
- **CQRS**: Command Query Responsibility Segregation where applicable

### Key Architectural Principles

- **Separation of Concerns**: Clear boundaries between layers
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Single Responsibility**: Each class has one reason to change
- **Open/Closed Principle**: Open for extension, closed for modification
- **Interface Segregation**: Many specific interfaces are better than one general-purpose interface
- **Domain-Driven Design**: Rich domain models with ubiquitous language

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11+**: Latest Python features and performance improvements
- **Hatch**: Modern Python build system and environment management
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checking with strict mode
- **Pytest**: Comprehensive testing framework

### Development Tools
- **Pre-commit**: Git hooks for code quality
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability checking
- **GitHub Actions**: CI/CD pipeline automation

## 💻 Development Workflow

### Initial Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/monorepo.git
cd monorepo

# Create development environment
python -m venv environments/.venv
source environments/.venv/bin/activate  # Linux/macOS
# environments\.venv\Scripts\activate    # Windows

# Install development dependencies
pip install -e ".[dev,test,lint]"

# Install pre-commit hooks for code quality
pre-commit install

# Verify setup
python -c "import monorepo; print('Setup successful')"
```

### Daily Development

```bash
# Repository governance
python scripts/governance/package_structure_enforcer.py  # Validate structure
python scripts/governance/build_artifacts_checker.py     # Check artifacts
python scripts/governance/root_directory_checker.py      # Check organization

# Code quality
ruff check src/                    # Fast linting
ruff format src/                   # Auto-formatting  
mypy src/packages/                 # Type checking
pytest tests/ --cov=src/packages/ # Run tests with coverage

# Security checks
bandit -r src/                     # Security scan
safety check                      # Dependency vulnerabilities
```

### Package Development

```bash
# Work on specific packages
cd src/packages/core/
pytest tests/                     # Run package tests
ruff check .                      # Check package code quality

# Create new packages
python scripts/templates/create_package.py --name new_package --domain business
```

### Testing Strategy

```bash
# Multi-layered testing approach
pytest tests/unit/                          # Fast unit tests
pytest tests/integration/                   # Integration tests
pytest src/packages/*/tests/                # Package-specific tests
pytest --cov=src/packages/ --cov-report=html # Coverage analysis
```

## 🤝 Contributing

We welcome contributions to this monorepo! This repository serves as an example of enterprise-ready software engineering practices.

### How to Contribute

1. **Fork the repository** and create a feature branch
2. **Follow the existing patterns** and architecture principles
3. **Maintain code quality** with comprehensive tests and documentation
4. **Run all governance checks** before submitting
5. **Submit a pull request** with a clear description

### Areas for Contribution

- **New packages**: Implement additional domain packages following clean architecture
- **Infrastructure improvements**: Enhance monitoring, deployment, or CI/CD
- **Documentation**: Improve guides, tutorials, and architectural documentation
- **Testing**: Increase test coverage and add performance benchmarks
- **Tooling**: Improve development tools and automation scripts

### Development Standards

- Follow clean architecture and DDD principles
- Maintain high test coverage (aim for 85%+)
- Use type hints and pass mypy strict mode
- Follow repository governance rules
- Document all public APIs and architectural decisions

For detailed guidelines, see [CONTRIBUTING.md](docs/developer-guides/contributing/CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This monorepo demonstrates enterprise software engineering practices. The specific domain implementations (anomaly detection, etc.) serve as examples of how to structure complex business logic using clean architecture principles.
