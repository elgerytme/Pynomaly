# Contributing Guidelines

## Welcome Contributors

Thank you for your interest in contributing to our domain-driven monorepo platform!

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Kubernetes (optional, for full infrastructure testing)
- Make

### Development Setup
```bash
# Clone the repository
git clone https://github.com/your-org/monorepo.git
cd monorepo

# Install dependencies
make install

# Run tests
make test

# Validate package independence
make validate-independence
```

## Contribution Process

### 1. Issue Creation
- Check existing issues before creating new ones
- Use issue templates for bugs, features, and documentation
- Provide clear descriptions with reproduction steps

### 2. Pull Request Process
- Fork the repository
- Create feature branch: `git checkout -b feature/your-feature`
- Follow coding standards and add tests
- Ensure all checks pass: `make ci-check`
- Submit PR with clear description

### 3. Code Review
- All PRs require at least one review
- Address feedback promptly
- Maintain clean commit history

## Development Standards

### Package Development
- Follow domain-driven design principles
- Maintain package independence (validated automatically)
- Include comprehensive tests (unit, integration, security)
- Add monitoring and observability

### Code Quality
- Follow PEP 8 for Python code
- Use type hints and docstrings
- Maintain test coverage > 90%
- Include security considerations

### Documentation
- Update relevant documentation with changes
- Include code examples and usage patterns
- Maintain architecture decision records (ADRs)

## Testing Requirements

### Automated Tests
```bash
# Run all tests
make test

# Run specific package tests
make test-package PACKAGE=ai/mlops

# Run security tests
make security-test

# Run performance tests
make performance-test
```

### Manual Testing
- Test package independence
- Verify deployment configurations
- Validate monitoring and alerting

## Security Guidelines

### Security Review Process
- All PRs undergo automated security scanning
- Critical security changes require security team review
- Follow OWASP guidelines for web security

### Vulnerability Reporting
- Report security vulnerabilities privately
- Email: security@your-org.com
- Include detailed reproduction steps

## Package Contribution Guidelines

### Creating New Packages
```bash
# Use the package generator
python tools/package-generator/main.py --interactive

# Follow the guided setup process
# Validate independence before submitting
make validate-independence
```

### Package Requirements
- Must be self-contained and domain-bounded
- Include complete CI/CD pipeline
- Provide comprehensive documentation
- Implement monitoring and alerting
- Pass all security scans

## Infrastructure Contributions

### Adding Infrastructure Templates
- Test templates thoroughly
- Include security best practices
- Provide clear documentation
- Support multiple deployment scenarios

### Monitoring and Observability
- Include Prometheus metrics
- Add Grafana dashboards
- Configure alerting rules
- Test end-to-end observability

## Documentation Standards

### Documentation Types
- **Architecture Docs**: High-level system design
- **API Docs**: Auto-generated from code
- **User Guides**: Step-by-step instructions
- **Developer Docs**: Implementation details

### Writing Guidelines
- Use clear, concise language
- Include code examples
- Provide context and rationale
- Update with code changes

## Community Participation

### Communication Channels
- GitHub Discussions for general questions
- GitHub Issues for bugs and features
- Slack/Discord for real-time collaboration

### Community Events
- Weekly office hours
- Monthly architecture reviews
- Quarterly contributor meetings

## Recognition

### Contributor Levels
- **Contributor**: Merged PRs
- **Regular Contributor**: 5+ merged PRs
- **Core Contributor**: Significant feature contributions
- **Maintainer**: Code review and release responsibilities

### Acknowledgments
- Contributors listed in CONTRIBUTORS.md
- Recognition in release notes
- Community shoutouts

## Questions and Support

- Check existing documentation first
- Search GitHub issues and discussions
- Join community channels for help
- Contact maintainers for complex questions

Thank you for contributing to making this platform better for everyone!