# Pynomaly Developer Onboarding Guide

Welcome to the Pynomaly development team! This comprehensive guide will help you get up to speed quickly and effectively as a contributor to our enterprise-grade anomaly detection platform.

## 📋 Quick Start Checklist

### ✅ Prerequisites Setup (15 minutes)
- [ ] **Python 3.11+** installed and accessible via `python3 --version`
- [ ] **Git** configured with your name and email
- [ ] **GitHub account** with SSH key configured
- [ ] **VSCode or PyCharm** (recommended IDEs)
- [ ] **Docker** installed (optional, for containerized development)

### ✅ Repository Setup (10 minutes)
- [ ] Fork the repository on GitHub
- [ ] Clone your fork: `git clone git@github.com:your-username/pynomaly.git`
- [ ] Add upstream remote: `git remote add upstream git@github.com:elgerytme/pynomaly.git`
- [ ] Navigate to project: `cd pynomaly`

### ✅ Development Environment (20 minutes)
- [ ] Create virtual environment: `python3 -m venv environments/.venv`
- [ ] Activate environment: `source environments/.venv/bin/activate`
- [ ] Install development dependencies: `pip install -e ".[dev,test]"`
- [ ] Install pre-commit hooks: `pre-commit install`
- [ ] Verify setup: `python -c "import pynomaly; print('Setup successful!')"`

### ✅ Validation (10 minutes)
- [ ] Run basic tests: `pytest tests/unit/domain/ -v`
- [ ] Verify code quality tools: `ruff check src/pynomaly/`
- [ ] Check type annotations: `mypy src/pynomaly/domain/`
- [ ] Test CLI: `python scripts/run/cli.py --help`

## 🏗️ Project Architecture Overview

Pynomaly follows **Clean Architecture** principles with **Domain-Driven Design (DDD)**:

```
src/pynomaly/
├── domain/          # 🎯 Core business logic (no external dependencies)
│   ├── entities/    # Business objects (Anomaly, Detector, Dataset)
│   ├── value_objects/ # Immutable data (AnomalyScore, ContaminationRate)
│   ├── services/    # Business logic that spans entities
│   └── protocols/   # Interfaces for external dependencies
├── application/     # 🔄 Use cases and orchestration
│   ├── use_cases/   # Business workflows (DetectAnomalies, TrainDetector)
│   ├── services/    # Application services
│   └── dto/         # Data transfer objects
├── infrastructure/ # 🔌 External integrations
│   ├── adapters/    # ML library integrations (PyOD, scikit-learn)
│   ├── persistence/ # Data storage
│   └── config/      # Dependency injection
└── presentation/   # 🖥️ User interfaces
    ├── api/         # FastAPI REST endpoints
    ├── cli/         # Command-line interface
    └── web/         # Progressive Web App
```

### 🎯 Key Design Principles

1. **Dependency Inversion**: Dependencies point inward toward the domain
2. **Single Responsibility**: Each component has one reason to change
3. **Interface Segregation**: Small, focused interfaces
4. **Testability**: Pure domain logic with 100% test coverage
5. **Configurability**: Dependency injection for all external services

## 🛠️ Development Workflow

### 1. Feature Development Process

```bash
# 1. Create feature branch
git checkout -b feature/awesome-feature
git push -u origin feature/awesome-feature

# 2. Make changes following TDD
# - Write failing test first
# - Implement minimal code to pass
# - Refactor while keeping tests green

# 3. Ensure quality
ruff format src/ tests/                # Format code
ruff check src/ tests/                 # Lint code
mypy src/pynomaly/                     # Type check
pytest tests/ --cov=src/pynomaly      # Run tests with coverage

# 4. Commit with conventional format
git commit -m "feat(domain): add confidence intervals to anomaly scores

- Add ConfidenceInterval value object with validation
- Update AnomalyScore to include optional intervals
- Add statistical confidence calculation methods

Closes #123"

# 5. Push and create PR
git push origin feature/awesome-feature
# Create PR via GitHub UI
```

### 2. Testing Strategy

#### Test Pyramid
- **Unit Tests** (70%): Fast, isolated, test individual components
- **Integration Tests** (20%): Test component interactions
- **End-to-End Tests** (10%): Test complete workflows

#### Test Organization
```bash
tests/
├── unit/                    # Fast unit tests (< 1s each)
│   ├── domain/             # Domain layer tests (no mocks needed)
│   ├── application/        # Application service tests
│   └── infrastructure/     # Adapter tests (with mocks)
├── integration/            # Integration tests (< 10s each)
│   ├── api/               # API endpoint tests
│   └── persistence/       # Database integration tests
└── e2e/                   # End-to-end tests (< 60s each)
    ├── cli/               # CLI workflow tests
    └── web/               # Web interface tests
```

#### Running Tests
```bash
# Quick validation (30 seconds)
pytest tests/unit/domain/ -v

# Full unit test suite (2-3 minutes)
pytest tests/unit/ -v

# Integration tests (5 minutes)
pytest tests/integration/ -v --tb=short

# Full test suite with coverage (10 minutes)
pytest tests/ --cov=src/pynomaly --cov-report=html

# Performance tests
pytest tests/performance/ -v --benchmark-only
```

### 3. Code Quality Standards

#### Automatic Formatting and Linting
```bash
# Format code (always run before committing)
ruff format src/ tests/

# Check and fix linting issues
ruff check src/ tests/ --fix

# Type checking
mypy src/pynomaly/

# Security scanning
bandit -r src/pynomaly/
```

#### Code Review Checklist
- [ ] **Tests**: All new code has corresponding tests
- [ ] **Types**: All functions have type hints
- [ ] **Documentation**: Public APIs have docstrings
- [ ] **Architecture**: Follows clean architecture principles
- [ ] **Performance**: No obvious performance issues
- [ ] **Security**: No security vulnerabilities introduced

## 📚 Essential Documentation

### Developer Resources
- **[Architecture Decision Records](./architecture/adr/)**: Key design decisions
- **[API Documentation](../api-reference/)**: Complete API reference
- **[Coding Standards](./CODING_STANDARDS.md)**: Style and conventions
- **[Testing Guidelines](./TESTING_GUIDELINES.md)**: Testing best practices

### User Documentation
- **[User Guides](../user-guides/)**: Feature usage guides
- **[Quick Start](../getting-started/quick-start.md)**: 5-minute tutorial
- **[CLI Reference](../api-reference/cli-commands.md)**: Command documentation

## 🔧 Common Development Tasks

### Adding a New Algorithm
```bash
# 1. Create domain interface
# src/pynomaly/domain/protocols/detector_protocols.py
class NewAlgorithmProtocol(Protocol):
    def fit(self, data: DataFrame) -> None: ...
    def predict(self, data: DataFrame) -> List[AnomalyScore]: ...

# 2. Implement adapter
# src/pynomaly/infrastructure/adapters/new_algorithm_adapter.py
class NewAlgorithmAdapter(NewAlgorithmProtocol):
    def fit(self, data: DataFrame) -> None:
        # Implementation here
        pass

# 3. Add tests
# tests/unit/infrastructure/adapters/test_new_algorithm_adapter.py

# 4. Register in container
# src/pynomaly/infrastructure/config/container.py
```

### Adding a New API Endpoint
```bash
# 1. Create request/response DTOs
# src/pynomaly/application/dto/new_feature_dto.py

# 2. Implement use case
# src/pynomaly/application/use_cases/new_feature_use_case.py

# 3. Add API endpoint
# src/pynomaly/presentation/api/endpoints/new_feature.py

# 4. Add tests for all layers
# tests/unit/application/use_cases/test_new_feature_use_case.py
# tests/integration/api/test_new_feature_endpoints.py
```

### Adding a New CLI Command
```bash
# 1. Create command module
# src/pynomaly/presentation/cli/commands/new_command.py

# 2. Register command
# src/pynomaly/presentation/cli/app.py

# 3. Add tests
# tests/integration/cli/test_new_command.py
```

## 🚀 Deployment and Release Process

### Local Development
```bash
# Start development server
uvicorn pynomaly.presentation.api.app:app --reload --port 8000

# Access applications
# API Documentation: http://localhost:8000/docs
# Web Interface: http://localhost:8000/app
```

### Release Process
1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog**: Update `CHANGELOG.md` with changes
3. **Tests**: Ensure all tests pass with `pytest tests/`
4. **Documentation**: Update documentation as needed
5. **Release Branch**: Create `release/vX.Y.Z` branch
6. **Pull Request**: Create PR with changelog and version bump
7. **Merge**: Merge after review and CI passes
8. **Tag**: Create git tag `vX.Y.Z`
9. **Deploy**: Automated deployment via GitHub Actions

## 🆘 Getting Help

### Internal Resources
- **[Troubleshooting Guide](./TROUBLESHOOTING.md)**: Common issues and solutions
- **[FAQ](../troubleshooting/faq.md)**: Frequently asked questions
- **[Architecture Overview](./architecture/overview.md)**: System design deep dive

### Community Support
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Code Review**: Request feedback on complex changes

### Escalation Path
1. Check documentation and troubleshooting guides
2. Search existing GitHub issues
3. Ask in GitHub Discussions
4. Create detailed GitHub issue
5. Contact maintainers directly for urgent issues

## 📈 Growth and Learning Path

### Beginner (Weeks 1-2)
- [ ] Complete this onboarding guide
- [ ] Fix a "good first issue" 
- [ ] Understand the domain model
- [ ] Write first unit tests

### Intermediate (Weeks 3-6)
- [ ] Implement a new algorithm adapter
- [ ] Add a new API endpoint
- [ ] Contribute to architecture decisions
- [ ] Mentor new contributors

### Advanced (Months 2-6)
- [ ] Lead feature development
- [ ] Design major architectural changes
- [ ] Review and approve pull requests
- [ ] Contribute to technical roadmap

## 🎯 Success Metrics

### Technical Metrics
- **Test Coverage**: Maintain >85% line coverage
- **Code Quality**: No ruff violations, mypy clean
- **Performance**: No performance regressions
- **Documentation**: All public APIs documented

### Contribution Metrics
- **Pull Requests**: Regular, focused contributions
- **Code Reviews**: Constructive feedback on others' work
- **Issues**: Help triage and resolve user issues
- **Mentoring**: Support other contributors

---

## 📞 Contact Information

- **Lead Maintainer**: @elgerytme
- **Security Issues**: security@pynomaly.dev
- **General Questions**: discussions@github.com/elgerytme/pynomaly

Welcome to the team! We're excited to have you contribute to making Pynomaly the best anomaly detection platform available. 🚀

---

*Last updated: 2025-01-14*
*Next review: Monthly*