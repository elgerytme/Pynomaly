# 🤝 Contributing to anomaly_detection

Thank you for your interest in contributing to anomaly_detection! We welcome contributions from the community and are excited to work with you.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Contribution Types](#contribution-types)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (for containerized development)
- PostgreSQL 15+ (for database-dependent features)
- Redis 7+ (for caching features)

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/anomaly_detection.git
   cd anomaly_detection
   git remote add upstream https://github.com/anomaly_detection/anomaly_detection.git
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

### Docker Development Environment

```bash
# Start development environment
docker-compose up -d

# Run tests in container
docker-compose exec anomaly_detection-api pytest

# Access container shell
docker-compose exec anomaly_detection-api bash
```

## 🔄 Development Process

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/**: New features (`feature/add-lstm-detector`)
- **bugfix/**: Bug fixes (`bugfix/fix-memory-leak`)
- **hotfix/**: Critical production fixes

### Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Follow our coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/performance/
   
   # Check code coverage
   pytest --cov=anomaly_detection --cov-report=html
   ```

4. **Lint and format**
   ```bash
   # Format code
   black .
   isort .
   
   # Lint code
   flake8 src/
   mypy src/
   
   # Security scan
   bandit -r src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add LSTM-based anomaly detector
   
   - Implement LSTM autoencoder for time series anomaly detection
   - Add comprehensive unit tests
   - Update documentation with usage examples"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Then create a Pull Request on GitHub
   ```

## 🎯 Contribution Types

### 🐛 Bug Fixes
- **What**: Fix existing functionality that isn't working correctly
- **Requirements**: 
  - Clear description of the bug
  - Steps to reproduce
  - Unit tests that demonstrate the fix
  - Consider backward compatibility

### ✨ New Features
- **What**: Add new functionality or capabilities
- **Requirements**:
  - Feature proposal discussion (create an issue first)
  - Design document for complex features
  - Comprehensive tests
  - Documentation updates
  - Performance considerations

### 📖 Documentation
- **What**: Improve documentation, tutorials, examples
- **Types**:
  - API documentation
  - User guides
  - Code examples
  - Tutorial improvements
  - README enhancements

### 🔧 Infrastructure
- **What**: CI/CD, Docker, deployment, tooling improvements
- **Requirements**:
  - Clear benefits explanation
  - Backward compatibility
  - Testing in isolated environment

### 🧪 Testing
- **What**: Add or improve tests
- **Types**:
  - Unit tests
  - Integration tests
  - Performance tests
  - End-to-end tests

### 🌍 Algorithms
- **What**: Implement new anomaly detection algorithms
- **Requirements**:
  - Literature review and references
  - Comparison with existing algorithms
  - Performance benchmarks
  - Comprehensive documentation

## 📝 Pull Request Guidelines

### PR Checklist

Before submitting a PR, ensure:

- [ ] **Code Quality**
  - [ ] Code follows project style guidelines
  - [ ] All tests pass (`pytest`)
  - [ ] Code coverage is maintained or improved
  - [ ] No lint warnings (`flake8`, `mypy`)
  - [ ] Code is formatted (`black`, `isort`)

- [ ] **Documentation**
  - [ ] Code includes docstrings
  - [ ] README updated (if applicable)
  - [ ] API documentation updated
  - [ ] Examples provided for new features

- [ ] **Testing**
  - [ ] Unit tests for new functionality
  - [ ] Integration tests (if applicable)
  - [ ] Performance tests (for algorithms)
  - [ ] Tests cover edge cases

- [ ] **Security**
  - [ ] Security scan passes (`bandit`)
  - [ ] No hardcoded secrets or sensitive data
  - [ ] Input validation for new endpoints

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code
- [ ] I have made corresponding changes to documentation
- [ ] My changes generate no new warnings
- [ ] New and existing tests pass
```

### PR Size Guidelines

- **Small PRs** (< 400 lines): Preferred, faster review
- **Medium PRs** (400-800 lines): Include detailed description
- **Large PRs** (> 800 lines): Break into smaller PRs when possible

## 🐛 Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug.

## To Reproduce
1. Step 1
2. Step 2
3. See error

## Expected Behavior
What you expected to happen.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.11.0]
- anomaly_detection: [e.g., 1.2.0]

## Additional Context
Any other context about the problem.
```

### Feature Requests

```markdown
## Feature Description
Clear description of the feature.

## Motivation
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other solutions you considered.
```

### Security Issues

For security vulnerabilities, **DO NOT** create a public issue. Instead:
- Email: [security@anomaly_detection.org](mailto:security@anomaly_detection.org)
- Use GitHub's private vulnerability reporting
- Include detailed information about the vulnerability

## 🏷️ Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples
```bash
feat(algorithms): add LSTM autoencoder detector
fix(api): resolve memory leak in detection endpoint
docs(readme): update installation instructions
test(integration): add end-to-end detection tests
```

## 🎯 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 100 characters

### Code Structure

```python
"""Module docstring describing the module purpose."""

from typing import List, Optional
import logging

from src.packages.data.anomaly_detection.core.domain_entities import Dataset
from src.packages.data.anomaly_detection.core.dependency_injection import inject

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detector class docstring.
    
    Args:
        algorithm: The detection algorithm to use
        threshold: Anomaly score threshold
        
    Example:
        >>> detector = AnomalyDetector(algorithm="isolation_forest")
        >>> result = detector.detect(data)
    """
    
    def __init__(self, algorithm: str, threshold: float = 0.5):
        self.algorithm = algorithm
        self.threshold = threshold
        
    def detect(self, data: Dataset) -> List[bool]:
        """Detect anomalies in the provided dataset.
        
        Args:
            data: Input dataset for anomaly detection
            
        Returns:
            List of boolean values indicating anomalies
            
        Raises:
            ValueError: If data is invalid
        """
        logger.info(f"Running anomaly detection with {self.algorithm}")
        # Implementation here
        return []
```

### Testing Standards

```python
import pytest
from unittest.mock import Mock, patch

from src.packages.data.anomaly_detection.algorithms.isolation_forest import IsolationForest


class TestIsolationForest:
    """Test suite for IsolationForest algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = IsolationForest(contamination=0.1)
        self.sample_data = self._create_sample_data()
    
    def test_fit_valid_data(self):
        """Test fitting with valid data."""
        result = self.detector.fit(self.sample_data)
        assert result is not None
        assert self.detector.is_fitted
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error when called before fit."""
        with pytest.raises(ValueError, match="Model not fitted"):
            self.detector.predict(self.sample_data)
    
    @pytest.mark.parametrize("contamination", [0.05, 0.1, 0.2])
    def test_different_contamination_levels(self, contamination):
        """Test detector with different contamination levels."""
        detector = IsolationForest(contamination=contamination)
        detector.fit(self.sample_data)
        result = detector.predict(self.sample_data)
        assert len(result) == len(self.sample_data)
    
    def _create_sample_data(self):
        """Create sample data for testing."""
        # Implementation here
        pass
```

## 🌟 Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Release notes
- Annual contributor report
- Special recognition for significant contributions

### Contribution Levels

- **First-time contributor**: Welcome badge
- **Regular contributor**: 5+ merged PRs
- **Core contributor**: Significant feature contributions
- **Maintainer**: Ongoing maintenance responsibilities

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/anomaly_detection)
- **GitHub Discussions**: Ask questions and discuss ideas
- **Email**: [contributors@anomaly_detection.org](mailto:contributors@anomaly_detection.org)
- **Documentation**: [docs.anomaly_detection.org](https://docs.anomaly_detection.org)

## 🏆 Monthly Recognition

- **Contributor of the Month**: Outstanding contributions
- **Bug Hunter**: Most bugs fixed
- **Documentation Hero**: Best documentation improvements
- **Community Champion**: Most helpful in discussions

---

**Thank you for contributing to anomaly_detection! 🎉**

Your contributions make the open source community an amazing place to learn, inspire, and create.