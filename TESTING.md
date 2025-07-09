# Pynomaly Reproducible Test Environment

This document describes the comprehensive, reproducible test environment setup for Pynomaly, designed to ensure consistent testing across different platforms and CI/CD systems.

## 🎯 Overview

The reproducible test environment provides:

- **Locked Dependencies**: Using `requirements.lock` for exact version control
- **Comprehensive Test Suite**: Unit, integration, mutation, and E2E UI tests
- **Containerized Environment**: Docker-based development and testing
- **Cross-Platform Compatibility**: Works on Linux, macOS, and Windows
- **CI/CD Ready**: Automated testing with GitHub Actions integration

## 📋 Components

### 1. Requirements Management

#### `requirements.lock`
- Generated from Poetry for exact dependency versions
- Used in CI/CD for reproducible builds
- Ensures consistent behavior across environments

#### `requirements-dev.lock`
- Development dependencies with exact versions
- Includes testing, linting, and development tools

### 2. Test Configuration

#### `tox.ini`
Provides isolated test environments:
- **lint**: Code quality and formatting checks
- **type**: Type checking with mypy
- **unit**: Unit tests with pytest
- **integration**: Integration tests with real services
- **mutation**: Mutation testing for code quality
- **e2e-ui**: End-to-end UI tests with Playwright

#### `pytest.ini`
Configured with:
- **Asyncio support**: `asyncio_mode = auto`
- **Hypothesis profile**: `hypothesis_profile = ci`
- **Comprehensive markers**: For test categorization
- **Coverage reporting**: Integrated coverage analysis

### 3. Development Environment

#### `Dockerfile.dev`
Multi-stage container with:
- **Python 3.11**: Latest stable Python version
- **Node.js 20**: For UI build and testing
- **Chrome/Playwright**: For UI automation testing
- **Development tools**: Poetry, pre-commit, testing frameworks

#### `docker-compose.dev.yml`
Complete development stack:
- **Main service**: Development environment
- **PostgreSQL**: Database for integration tests
- **Redis**: Caching and session management
- **Test services**: Dedicated testing containers

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build and start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests
docker-compose -f docker-compose.dev.yml exec pynomaly-dev poetry run tox

# Interactive development
docker-compose -f docker-compose.dev.yml exec pynomaly-dev bash
```

### Local Development

```bash
# Install dependencies
poetry install --with dev

# Install Playwright browsers
poetry run playwright install

# Run specific test environments
poetry run tox -e lint      # Linting
poetry run tox -e type      # Type checking
poetry run tox -e unit      # Unit tests
poetry run tox -e integration  # Integration tests
poetry run tox -e mutation     # Mutation tests
poetry run tox -e e2e-ui       # UI tests
```

### CI/CD Integration

```bash
# Setup CI environment
./scripts/ci-setup.sh setup

# Run full test suite
./scripts/ci-setup.sh full

# Run quick validation
./scripts/ci-setup.sh quick
```

## 🧪 Test Environments

### Lint Environment
- **Ruff**: Fast Python linter
- **Black**: Code formatting
- **isort**: Import sorting
- **Bandit**: Security analysis
- **Flake8**: Style guide enforcement

### Type Environment
- **mypy**: Static type checking
- **Type stubs**: For external libraries
- **Strict mode**: Comprehensive type validation

### Unit Environment
- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-xdist**: Parallel test execution
- **Hypothesis**: Property-based testing
- **Factory Boy**: Test data factories

### Integration Environment
- **Real services**: PostgreSQL, Redis
- **API testing**: HTTP client integration
- **Database tests**: Real database interactions
- **Network tests**: External service integration

### Mutation Environment
- **mutmut**: Mutation testing framework
- **Code quality**: Validates test effectiveness
- **Timeout handling**: Prevents infinite loops

### E2E UI Environment
- **Playwright**: Cross-browser testing
- **Multiple browsers**: Chromium, Firefox, WebKit
- **Visual testing**: Screenshot comparison
- **Accessibility**: axe-core integration
- **Performance**: Lighthouse integration

## 📊 Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test for isolated components"""
    pass

@pytest.mark.integration
def test_integration_flow():
    """Integration test for component interaction"""
    pass

@pytest.mark.asyncio
async def test_async_functionality():
    """Async test automatically handled"""
    pass

@pytest.mark.property
def test_property_based():
    """Property-based test using Hypothesis"""
    pass
```

## 🔧 Configuration

### Environment Variables

```bash
# Test environment
PYNOMALY_ENVIRONMENT=testing
PYNOMALY_LOG_LEVEL=INFO
PYNOMALY_CACHE_ENABLED=false
PYNOMALY_AUTH_ENABLED=false
PYNOMALY_TESTING=true
PYTHONPATH=src

# UI testing
PLAYWRIGHT_BROWSERS_PATH=/tmp/browsers
HEADLESS=true
```

### Hypothesis Configuration

```ini
# pytest.ini
hypothesis_profile = ci
```

Hypothesis profiles:
- **dev**: Fast feedback during development
- **ci**: Comprehensive testing for CI/CD
- **debug**: Verbose output for debugging

## 📈 Coverage and Reporting

### Coverage Configuration
- **Source**: `src/pynomaly`
- **Branch coverage**: Enabled
- **Minimum coverage**: 90%
- **Reports**: HTML, XML, JSON

### Test Reports
- **JUnit XML**: For CI/CD integration
- **HTML reports**: Human-readable results
- **Coverage reports**: Detailed coverage analysis
- **Performance reports**: Benchmark results

## 🛠️ Development Workflow

### 1. Pre-commit Hooks
```bash
poetry run pre-commit install
```

### 2. Test-Driven Development
```bash
# Write test first
poetry run pytest tests/unit/test_new_feature.py -v

# Implement feature
# Run tests again
poetry run pytest tests/unit/test_new_feature.py -v
```

### 3. Full Test Suite
```bash
# Before committing
poetry run tox
```

### 4. UI Testing
```bash
# Run UI tests
poetry run tox -e e2e-ui

# Debug UI tests
poetry run pytest tests/e2e/ --headed
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.lock
        pip install -r requirements-dev.lock
        pip install tox
    
    - name: Run tests
      run: tox -e py$(echo ${{ matrix.python-version }} | tr -d .)-unit,lint,type
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🐛 Troubleshooting

### Common Issues

1. **Playwright Installation**
   ```bash
   poetry run playwright install-deps
   ```

2. **Permission Issues**
   ```bash
   chmod +x scripts/ci-setup.sh
   ```

3. **Docker Issues**
   ```bash
   docker-compose -f docker-compose.dev.yml down -v
   docker-compose -f docker-compose.dev.yml up --build
   ```

4. **Coverage Issues**
   ```bash
   export COVERAGE_CORE=sysmon
   ```

### Debug Mode

```bash
# Verbose testing
poetry run pytest -v -s tests/

# Debug specific test
poetry run pytest -v -s tests/unit/test_specific.py::test_function

# UI testing with browser
poetry run pytest tests/e2e/ --headed --slowmo=1000
```

## 📚 Best Practices

### 1. Test Organization
- **Unit tests**: `tests/unit/`
- **Integration tests**: `tests/integration/`
- **E2E tests**: `tests/e2e/`
- **Performance tests**: `tests/performance/`

### 2. Test Data
- Use factories for test data
- Avoid hard-coded values
- Clean up after tests

### 3. Async Testing
- Use `@pytest.mark.asyncio` for async tests
- Leverage `pytest-asyncio` plugin
- Handle event loops properly

### 4. UI Testing
- Use page objects pattern
- Add explicit waits
- Test across browsers
- Include accessibility checks

## 🔐 Security

### Security Testing
- **Bandit**: Security linting
- **Safety**: Dependency vulnerability scanning
- **pip-audit**: Additional security checks

### Secrets Management
- Never commit secrets
- Use environment variables
- Validate in CI/CD

## 📄 License

This testing configuration is part of the Pynomaly project and follows the same MIT license.
