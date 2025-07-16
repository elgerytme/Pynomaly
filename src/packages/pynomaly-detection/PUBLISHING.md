# Publishing Pynomaly Detection to PyPI

This document explains how to publish the `pynomaly-detection` package to PyPI both manually and automatically using GitHub Actions.

## Package Overview

- **Package Name**: `pynomaly-detection`
- **Version**: 0.1.0
- **Description**: Production-ready Python anomaly detection library with clean architecture, AutoML, and 40+ algorithms

## Manual Publishing

### Prerequisites

1. **PyPI Account**: Create accounts on both [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. **API Tokens**: Generate API tokens for both PyPI and TestPyPI
3. **Tools**: Install required tools:
   ```bash
   pip install build twine
   ```

### Build the Package

```bash
# Navigate to the package directory
cd pynomaly-detection

# Build the package
python -m build

# Verify the build
twine check dist/*
```

This creates:
- `dist/pynomaly_detection-0.1.0-py3-none-any.whl` (wheel)
- `dist/pynomaly_detection-0.1.0.tar.gz` (source distribution)

### Publish to TestPyPI (Recommended First)

```bash
# Upload to TestPyPI first for testing
twine upload --repository testpypi dist/*
```

When prompted, use:
- Username: `__token__`
- Password: Your TestPyPI API token

### Test Installation from TestPyPI

```bash
# Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ pynomaly-detection

# Test the package
python -c "
import pynomaly_detection
import numpy as np
detector = pynomaly_detection.AnomalyDetector()
X = np.random.randn(100, 5)
detector.fit(X)
predictions = detector.predict(X)
print(f'Test successful! Detected {predictions.sum()} anomalies.')
"
```

### Publish to PyPI

Once testing is successful:

```bash
# Upload to PyPI
twine upload dist/*
```

When prompted, use:
- Username: `__token__`
- Password: Your PyPI API token

## Automated Publishing with GitHub Actions

The included `.github/workflows/ci-cd.yml` provides automated CI/CD pipeline.

### Setup GitHub Secrets

1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `PYPI_API_TOKEN`: Your PyPI API token
   - `TEST_PYPI_API_TOKEN`: Your TestPyPI API token

### Workflow Triggers

The GitHub Action automatically:

1. **On every push/PR**: Runs tests, linting, and security checks
2. **On push to main**: Publishes to TestPyPI
3. **On version tags (v*)**: Publishes to PyPI

### Triggering a Release

To publish a new version:

1. Update the version in `pyproject.toml`
2. Commit and push changes
3. Create and push a tag:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

This automatically triggers the full CI/CD pipeline and publishes to PyPI.

## Package Installation

Once published, users can install the package:

### Basic Installation
```bash
pip install pynomaly-detection
```

### With Optional Dependencies
```bash
# With machine learning features
pip install pynomaly-detection[ml]

# With AutoML capabilities
pip install pynomaly-detection[automl]

# With deep learning support
pip install pynomaly-detection[torch]

# Full installation
pip install pynomaly-detection[all]
```

## Usage Examples

### Basic Usage
```python
from pynomaly_detection import AnomalyDetector
import numpy as np

# Generate sample data
X = np.random.randn(1000, 10)
X[0:10] += 5  # Add some outliers

# Create detector
detector = AnomalyDetector()

# Fit and predict
detector.fit(X)
anomalies = detector.predict(X)

print(f"Found {anomalies.sum()} anomalies")
```

### Advanced Usage
```python
from pynomaly_detection import AnomalyDetector
from pynomaly_detection.algorithms import IsolationForestAdapter

# Use specific algorithm
detector = AnomalyDetector(algorithm=IsolationForestAdapter())
detector.fit(X)
results = detector.predict(X)
```

## Version Management

The package uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Monitoring

After publishing, monitor:

1. **PyPI Statistics**: Check download stats on PyPI
2. **GitHub Actions**: Monitor CI/CD pipeline success
3. **Issue Reports**: Watch GitHub issues for user feedback
4. **Security Alerts**: Monitor for security vulnerabilities

## Troubleshooting

### Common Issues

1. **Build Failures**: Check dependencies in `pyproject.toml`
2. **Import Errors**: Verify package structure and `__init__.py` files
3. **Version Conflicts**: Ensure version number is incremented
4. **Permission Errors**: Verify API tokens are correct

### Getting Help

- **Documentation**: Check the README.md
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

## Next Steps

1. Set up monitoring and analytics
2. Create comprehensive documentation
3. Implement additional algorithms
4. Add more comprehensive tests
5. Set up automated dependency updates