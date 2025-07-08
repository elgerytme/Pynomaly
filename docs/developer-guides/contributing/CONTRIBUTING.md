# Contributing to Pynomaly

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 📄 Contributing

---


Thank you for your interest in contributing to Pynomaly! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use issue templates when available
- Provide clear descriptions and steps to reproduce
- Include system information and error messages

### Suggesting Features

- Open a discussion before implementing major features
- Explain the use case and benefits
- Consider the impact on existing functionality
- Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Commit with clear messages
8. Push to your fork
9. Open a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Git for version control

### Installation

```bash
# Clone your fork
git clone https://github.com/your-username/pynomaly.git
cd pynomaly

# Install dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pynomaly

# Run specific test file
pytest tests/domain/test_entities.py

# Run with verbose output
pytest -v
```

### Code Style

We use several tools to maintain code quality:

```bash
# Format code with black
black src tests

# Sort imports with isort
isort src tests

# Type checking with mypy
mypy src

# Linting with ruff
ruff src tests

# Security checks with bandit
bandit -r src

# All checks
make lint
```

### Documentation

```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# Build API documentation
make apidoc
```

## Architecture Guidelines

### Domain Layer
- Keep domain logic pure and framework-agnostic
- Use value objects for domain concepts
- Ensure entities are self-validating
- Avoid external dependencies

### Application Layer
- Implement use cases as single-purpose classes
- Use DTOs for data transfer
- Keep application services thin
- Handle orchestration and transactions

### Infrastructure Layer
- Implement adapters for external services
- Use protocols for interfaces
- Keep infrastructure details isolated
- Ensure adapters are replaceable

### Presentation Layer
- Keep controllers/endpoints thin
- Use appropriate serialization
- Handle HTTP concerns only
- Delegate business logic to use cases

## Testing Guidelines

### Test Structure
- Mirror the source code structure
- One test file per source file
- Group related tests in classes
- Use descriptive test names

### Test Types
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test efficiency

### Test Best Practices
- Use fixtures for common setup
- Keep tests independent
- Test edge cases
- Mock external dependencies
- Aim for high coverage

## Commit Guidelines

### Commit Messages

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tooling changes

### Examples

```
feat(domain): add confidence intervals to anomaly scores

- Add ConfidenceInterval value object
- Update AnomalyScore to support intervals
- Add validation for interval bounds

Closes #123
```

## Architectural Decision Records (ADRs)

### When an ADR is Required

An ADR must be created for any architectural decision that is:

- **Irreversible**: Decisions that would be difficult or costly to change later
- **High-impact**: Decisions that affect multiple components or the system architecture
- **Significant precedent**: Decisions that establish patterns for future development
- **Technology choices**: Major framework, library, or tool selections
- **API design**: Public interface definitions that affect external integrations
- **Data models**: Core domain model changes or database schema decisions
- **Security architecture**: Authentication, authorization, or encryption approaches
- **Performance considerations**: Decisions that significantly impact system performance
- **Compliance requirements**: Decisions driven by regulatory or security compliance needs

### ADR Authoring & PR Workflow

1. **Draft Creation**: Create a new ADR in `docs/developer-guides/architecture/adr/` with status `PROPOSED`
   ```bash
   # Use the next sequential number
   cp docs/developer-guides/architecture/adr/template.md docs/developer-guides/architecture/adr/ADR-###-descriptive-name.md
   ```

2. **Content Development**: Fill out all sections of the ADR template:
   - Context: Describe the problem and constraints
   - Decision: State the chosen solution
   - Rationale: Explain why this decision was made
   - Alternatives: Document other options considered
   - Consequences: Describe positive and negative impacts
   - Implementation: Outline the implementation approach

3. **Pull Request**: Submit the ADR as a pull request with:
   - Branch name: `adr/###-descriptive-name`
   - Title: `docs(adr): Add ADR-### for [decision topic]`
   - Assign relevant reviewers (architecture team, domain experts)
   - Apply the `ADR` label

4. **Review Process**:
   - **Code Review**: Technical accuracy and completeness
   - **Design Meeting**: Schedule if decision requires broader discussion
   - **Stakeholder Input**: Gather input from affected teams
   - **Consensus Building**: Address concerns and reach agreement

5. **Approval**: Once approved:
   - Update status from `PROPOSED` to `ACCEPTED`
   - Update the approval section with date and reviewers
   - Merge the pull request
   - Update the ADR table of contents

### Superseding ADRs

When an ADR needs to be replaced:

1. **Create New ADR**: Follow the normal process for the new decision
2. **Reference Original**: Link to the ADR being superseded
3. **Update Original**: Change the original ADR's status to `SUPERSEDED`
4. **Update Change Log**: Document the change in CHANGELOG.md
5. **Cross-Reference**: Ensure both ADRs reference each other

### Change Log Updates

All ADR changes must be documented in CHANGELOG.md:

```markdown
## [Unreleased]

### Documentation
- Added ADR-### for [decision topic] (#PR-number)
- Superseded ADR-### with ADR-### for [updated decision] (#PR-number)
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release branch
4. Run full test suite
5. Build and test package
6. Create pull request
7. Merge after approval
8. Tag release
9. Deploy to PyPI

## Getting Help

- Check documentation first
- Search existing issues
- Ask in discussions
- Join our community chat
- Email maintainers

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributors page

Thank you for contributing to Pynomaly!

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
