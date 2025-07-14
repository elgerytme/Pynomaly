# Comprehensive Python Templates Collection

A complete collection of modern Python development templates following software architecture and engineering best practices, created using the Pynomaly repository as the architectural foundation.

## 🎯 Template Overview

This collection provides **10 production-ready templates** covering the entire spectrum of modern Python development, from foundational packages to complete SaaS applications.

### 📦 Foundational Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| **[Monorepo](./monorepo/)** | Clean Architecture with DDD patterns | Multi-package projects |
| **[Python Package](./python-package/)** | Modern tooling and CLI integration | Reusable libraries |
| **[Python App](./python-app/)** | Build/deploy structure | Standalone applications |

### 🌐 Web & API Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| **[FastAPI App](./fastapi-app/)** | Authentication and production features | REST APIs |
| **[HTMX + Tailwind](./htmx-tailwind-app/)** | Dynamic web apps with beautiful styling | Modern web applications |

### 🛠️ Developer Tools Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| **[Typer CLI](./typer-cli-app/)** | Rich CLI with extensibility | Command-line tools |
| **[Testing Framework](./testing-template/)** | Comprehensive testing strategies | Quality assurance |

### 🔐 Security Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| **[Authentication System](./auth-template/)** | Enterprise-grade auth with MFA | User management |
| **[Application Security](./app-security-template/)** | Security framework and scanning | Security hardening |

### 🚀 Enterprise Template

| Template | Description | Use Case |
|----------|-------------|----------|
| **[SaaS Application](./saas-app-template/)** | Complete SaaS platform | Multi-tenant SaaS |

## 🏗️ Architecture Principles

All templates follow these core architectural principles:

### Clean Architecture
- **Domain Layer**: Business logic and entities
- **Application Layer**: Use cases and orchestration
- **Infrastructure Layer**: External concerns and adapters
- **Presentation Layer**: APIs, CLIs, and UIs

### Domain-Driven Design (DDD)
- **Value Objects**: Immutable data structures
- **Entities**: Objects with identity
- **Repositories**: Data access abstractions
- **Services**: Domain logic coordination

### Modern Python Practices
- **Python 3.11+**: Latest language features
- **Type Hints**: Comprehensive type safety
- **Async/Await**: Non-blocking operations
- **Pydantic**: Data validation and settings

### Development Workflow
- **Hatch**: Modern Python packaging
- **Ruff**: Fast linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Comprehensive testing
- **Pre-commit**: Quality enforcement

## 🚀 Quick Start Guide

### 1. Choose Your Template

```bash
# For a new Python package
cp -r templates/python-package/ my-new-package

# For a web application
cp -r templates/htmx-tailwind-app/ my-web-app

# For a REST API
cp -r templates/fastapi-app/ my-api

# For a complete SaaS platform
cp -r templates/saas-app-template/ my-saas-app
```

### 2. Initialize Project

```bash
cd my-project

# Install dependencies
pip install -e ".[dev,test]"

# Setup pre-commit hooks
pre-commit install

# Initialize git (if needed)
git init
git add .
git commit -m "Initial commit from template"
```

### 3. Customize Configuration

Each template includes:
- `pyproject.toml` - Project configuration
- `.env.example` - Environment variables
- `README.md` - Detailed usage instructions
- `CHANGELOG.md` - Version history tracking

## 📊 Template Comparison Matrix

| Feature | Package | App | FastAPI | HTMX+Tailwind | Typer CLI | Testing | Auth | Security | SaaS |
|---------|---------|-----|---------|---------------|-----------|---------|------|----------|------|
| **Architecture** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Modern Python** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **CLI Interface** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Web Interface** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **API Endpoints** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Authentication** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Database** | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Security Features** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Multi-tenancy** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Background Tasks** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Payment Processing** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## 🔧 Technology Stack

### Core Technologies
- **Python 3.11+**: Modern language features
- **FastAPI**: High-performance web framework
- **Pydantic**: Data validation and settings
- **SQLAlchemy**: Modern ORM with async support
- **Alembic**: Database migrations

### Frontend Technologies
- **HTMX**: Dynamic web interactions
- **Tailwind CSS**: Utility-first styling
- **Alpine.js**: Lightweight JavaScript
- **Jinja2**: Template engine

### CLI Technologies
- **Typer**: Modern CLI framework
- **Rich**: Beautiful terminal output
- **Click**: Command-line utilities

### Testing Technologies
- **Pytest**: Testing framework
- **Hypothesis**: Property-based testing
- **Factory Boy**: Test data generation
- **Coverage.py**: Code coverage

### Security Technologies
- **JWT**: Token-based authentication
- **bcrypt**: Password hashing
- **OWASP**: Security best practices
- **Bandit**: Security linting

### Development Tools
- **Hatch**: Project management
- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Pre-commit**: Quality hooks
- **GitHub Actions**: CI/CD

### Deployment Technologies
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Terraform**: Infrastructure as code
- **Redis**: Caching and sessions
- **PostgreSQL**: Primary database

## 📖 Usage Examples

### Building a REST API

```bash
# Start with FastAPI template
cp -r templates/fastapi-app/ my-api
cd my-api

# Add authentication
cp -r ../templates/auth-template/src/auth_system/auth/ src/my_api/auth/

# Add security features
cp -r ../templates/app-security-template/src/security_framework/middleware/ src/my_api/middleware/

# Customize and deploy
docker-compose up -d
```

### Creating a SaaS Application

```bash
# Use the complete SaaS template
cp -r templates/saas-app-template/ my-saas
cd my-saas

# Everything is included:
# - Multi-tenancy
# - Authentication
# - Security
# - Billing
# - HTMX frontend
# - CLI management
# - Testing framework

# Just customize and deploy
```

### Developing a Python Package

```bash
# Start with package template
cp -r templates/python-package/ my-package
cd my-package

# Add comprehensive testing
cp -r ../templates/testing-template/tests/ tests/
cp ../templates/testing-template/pyproject.toml pyproject.toml

# Publish to PyPI
hatch build
hatch publish
```

## 🎯 Template Selection Guide

### Choose **Python Package** if you need:
- Reusable library
- CLI utility
- Distribution via PyPI
- Simple project structure

### Choose **FastAPI App** if you need:
- REST API
- Database integration
- Authentication
- API documentation

### Choose **HTMX + Tailwind** if you need:
- Modern web application
- Dynamic interactions
- Beautiful UI
- Minimal JavaScript

### Choose **SaaS Template** if you need:
- Multi-tenant application
- Subscription billing
- Complete platform
- Enterprise features

### Choose **Authentication Template** if you need:
- User management system
- Multi-factor authentication
- OAuth integration
- Security compliance

### Choose **Testing Template** if you need:
- Comprehensive test suite
- Quality assurance
- CI/CD integration
- Test automation

## 🔄 Template Evolution

These templates are based on the production-tested architecture of the Pynomaly project and follow these evolution principles:

### Version 1.0 (Current)
- ✅ Core architectural patterns
- ✅ Modern Python practices
- ✅ Production-ready configurations
- ✅ Comprehensive documentation

### Future Versions
- 🔄 Community feedback integration
- 🔄 New technology adoption
- 🔄 Enhanced automation
- 🔄 Extended plugin ecosystem

## 🤝 Contributing

To contribute to these templates:

1. **Follow Architecture**: Maintain Clean Architecture and DDD principles
2. **Use Modern Python**: Python 3.11+ with comprehensive type hints
3. **Include Tests**: Comprehensive test coverage
4. **Document Thoroughly**: Clear README and code documentation
5. **Security First**: Follow security best practices

## 📚 Documentation Structure

Each template includes:

```
template-name/
├── README.md              # Comprehensive usage guide
├── CHANGELOG.md           # Version history
├── SECURITY.md           # Security policy (for security templates)
├── pyproject.toml        # Modern Python configuration
├── docker-compose.yml    # Development environment
├── Dockerfile           # Production container
├── src/                 # Source code with clean architecture
├── tests/               # Comprehensive test suite
├── docs/                # Additional documentation
├── examples/            # Usage examples
├── scripts/             # Automation scripts
└── .github/             # CI/CD workflows
```

## 🏆 Quality Standards

All templates maintain:

- **100% Type Coverage**: Comprehensive type hints
- **90%+ Test Coverage**: Extensive test suites
- **Security Hardened**: OWASP compliance
- **Performance Optimized**: Efficient implementations
- **Documentation Complete**: Thorough documentation
- **Production Ready**: Deployment configurations

## 📄 License

All templates are released under the MIT License, allowing free use, modification, and distribution.

---

**Created with software architecture and engineering best practices in mind.**
**Based on the production-tested Pynomaly repository architecture.**

🚀 **Ready to build amazing Python applications!**