# Comprehensive Testing Template

A complete testing framework template with modern Python testing practices, comprehensive coverage strategies, and automation workflows.

## Features

- **Pytest Framework**: Modern testing with fixtures and plugins
- **Test Categories**: Unit, integration, end-to-end, and performance tests
- **Coverage Analysis**: Comprehensive code coverage reporting
- **Test Data Management**: Factories, fixtures, and mock strategies
- **Property-Based Testing**: Hypothesis for robust test generation
- **Mutation Testing**: Code quality assessment with mutation testing
- **Parallel Execution**: Fast test runs with pytest-xdist
- **CI/CD Integration**: GitHub Actions workflows for automated testing
- **Quality Gates**: Automated quality checks and reporting
- **Test Documentation**: Auto-generated test documentation

## Directory Structure

```
testing-template/
├── build/                 # Build artifacts and reports
├── deploy/                # Deployment configurations
├── docs/                  # Testing documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary test files
├── src/                   # Source code under test
│   └── test_framework/
│       ├── core/         # Core functionality
│       ├── utils/        # Utility functions
│       ├── models/       # Data models
│       └── services/     # Business services
├── tests/                # Test suites
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── e2e/             # End-to-end tests
│   ├── performance/     # Performance tests
│   ├── fixtures/        # Test fixtures
│   ├── factories/       # Test data factories
│   ├── mocks/           # Mock objects
│   └── conftest.py      # Pytest configuration
├── pkg/                  # Package metadata
├── examples/             # Testing examples
├── .github/              # CI/CD workflows
├── scripts/              # Test automation scripts
├── pyproject.toml        # Project configuration
├── pytest.ini           # Pytest configuration
├── tox.ini              # Multi-environment testing
├── coverage.ini         # Coverage configuration
├── README.md            # Project documentation
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-testing-project
   cd my-testing-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev,test]"
   ```

3. **Run basic tests**:
   ```bash
   pytest
   ```

4. **Run with coverage**:
   ```bash
   pytest --cov=src/test_framework --cov-report=html
   ```

5. **View coverage report**:
   ```bash
   open htmlcov/index.html
   ```

## Testing Categories

### Unit Tests

Fast, isolated tests for individual components:

```python
# tests/unit/test_calculator.py
import pytest
from test_framework.core.calculator import Calculator

class TestCalculator:
    def test_add_positive_numbers(self):
        calc = Calculator()
        result = calc.add(2, 3)
        assert result == 5
    
    def test_add_negative_numbers(self):
        calc = Calculator()
        result = calc.add(-2, -3)
        assert result == -5
    
    def test_divide_by_zero_raises_error(self):
        calc = Calculator()
        with pytest.raises(ZeroDivisionError):
            calc.divide(5, 0)
```

### Integration Tests

Tests for component interactions:

```python
# tests/integration/test_user_service.py
import pytest
from test_framework.services.user_service import UserService
from test_framework.models.user import User

@pytest.fixture
def user_service():
    return UserService(database_url="sqlite:///:memory:")

class TestUserService:
    def test_create_and_retrieve_user(self, user_service):
        # Create user
        user_data = {"name": "John Doe", "email": "john@example.com"}
        user_id = user_service.create_user(user_data)
        
        # Retrieve user
        retrieved_user = user_service.get_user(user_id)
        
        assert retrieved_user.name == "John Doe"
        assert retrieved_user.email == "john@example.com"
```

### End-to-End Tests

Full system workflow tests:

```python
# tests/e2e/test_user_workflow.py
import pytest
from test_framework.core.app import create_app

@pytest.fixture
def app():
    app = create_app(testing=True)
    return app

@pytest.fixture
def client(app):
    return app.test_client()

class TestUserWorkflow:
    def test_complete_user_registration_flow(self, client):
        # Register new user
        response = client.post('/api/users/register', json={
            'name': 'Jane Doe',
            'email': 'jane@example.com',
            'password': 'secure123'
        })
        assert response.status_code == 201
        
        # Login user
        response = client.post('/api/auth/login', json={
            'email': 'jane@example.com',
            'password': 'secure123'
        })
        assert response.status_code == 200
        assert 'access_token' in response.json
```

### Performance Tests

Load and performance testing:

```python
# tests/performance/test_api_performance.py
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from test_framework.core.app import create_app

class TestAPIPerformance:
    def test_api_response_time(self, client):
        start_time = time.time()
        response = client.get('/api/health')
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Less than 100ms
    
    def test_concurrent_requests(self, client):
        def make_request():
            return client.get('/api/users')
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in results)
```

## Test Data Management

### Factories with Factory Boy

```python
# tests/factories/user_factory.py
import factory
from test_framework.models.user import User

class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    name = factory.Faker('name')
    email = factory.Faker('email')
    age = factory.Faker('pyint', min_value=18, max_value=80)
    is_active = True
    
class AdminUserFactory(UserFactory):
    is_admin = True
    email = factory.Sequence(lambda n: f'admin{n}@example.com')
```

### Fixtures

```python
# tests/conftest.py
import pytest
from test_framework.core.database import Database
from test_framework.models.user import User

@pytest.fixture(scope="session")
def database():
    """Create test database for the session."""
    db = Database("sqlite:///:memory:")
    db.create_tables()
    yield db
    db.close()

@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        name="Test User",
        email="test@example.com",
        age=25
    )

@pytest.fixture
def user_list():
    """Create a list of sample users."""
    return [
        User(name=f"User {i}", email=f"user{i}@example.com", age=20+i)
        for i in range(5)
    ]
```

### Mocking

```python
# tests/mocks/email_service_mock.py
from unittest.mock import Mock, patch
import pytest

@pytest.fixture
def mock_email_service():
    with patch('test_framework.services.email_service.EmailService') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_instance.send_email.return_value = True
        yield mock_instance

class TestUserService:
    def test_user_registration_sends_email(self, user_service, mock_email_service):
        user_data = {"name": "John", "email": "john@example.com"}
        user_service.register_user(user_data)
        
        mock_email_service.send_email.assert_called_once_with(
            to="john@example.com",
            subject="Welcome!",
            template="welcome_email"
        )
```

## Property-Based Testing

Using Hypothesis for robust test generation:

```python
# tests/property/test_calculator_properties.py
from hypothesis import given, strategies as st
from test_framework.core.calculator import Calculator

class TestCalculatorProperties:
    @given(st.integers(), st.integers())
    def test_addition_is_commutative(self, a, b):
        calc = Calculator()
        assert calc.add(a, b) == calc.add(b, a)
    
    @given(st.integers(), st.integers(), st.integers())
    def test_addition_is_associative(self, a, b, c):
        calc = Calculator()
        assert calc.add(calc.add(a, b), c) == calc.add(a, calc.add(b, c))
    
    @given(st.text())
    def test_string_processing_doesnt_crash(self, text):
        from test_framework.utils.text_processor import process_text
        # Should not raise an exception
        result = process_text(text)
        assert isinstance(result, str)
```

## Coverage Analysis

### Basic Coverage

```bash
# Run tests with coverage
pytest --cov=src/test_framework

# Generate HTML report
pytest --cov=src/test_framework --cov-report=html

# Generate XML report for CI
pytest --cov=src/test_framework --cov-report=xml
```

### Advanced Coverage Configuration

```ini
# coverage.ini
[run]
source = src/test_framework
branch = true
omit = 
    */tests/*
    */migrations/*
    */venv/*
    */env/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

show_missing = true
skip_covered = false
precision = 2

[html]
directory = build/coverage/html

[xml]
output = build/coverage/coverage.xml
```

## Mutation Testing

Test the quality of your tests:

```bash
# Install mutation testing
pip install mutmut

# Run mutation testing
mutmut run --paths-to-mutate=src/test_framework

# View results
mutmut results
mutmut show <mutation_id>
```

## Parallel Test Execution

```bash
# Run tests in parallel
pytest -n auto  # Auto-detect CPU count
pytest -n 4     # Use 4 workers

# Parallel with coverage
pytest -n auto --cov=src/test_framework --cov-report=html
```

## Test Configuration

### Pytest Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --strict-config
    --disable-warnings
    --cov=src/test_framework
    --cov-report=term-missing
    --cov-report=html:build/coverage/html
    --cov-report=xml:build/coverage/coverage.xml

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests
    external: Tests that require external services

filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

### Tox Configuration

```ini
# tox.ini
[tox]
envlist = py311, py312, coverage, docs, lint

[testenv]
deps = 
    -e.[test]
commands = pytest {posargs}

[testenv:coverage]
deps = 
    -e.[test]
    coverage[toml]
commands = 
    coverage run -m pytest
    coverage report
    coverage html

[testenv:docs]
deps = 
    -e.[docs]
commands = 
    sphinx-build -b html docs build/docs

[testenv:lint]
deps = 
    -e.[dev]
commands = 
    ruff check src tests
    mypy src tests
    bandit -r src
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run tests
      run: |
        pytest --cov=src/test_framework --cov-report=xml
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint
      run: |
        ruff check src tests
        mypy src tests
        bandit -r src
    
    - name: Security check
      run: safety check
```

## Test Automation Scripts

### Test Runner Script

```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "🧪 Running comprehensive test suite..."

# Unit tests
echo "📦 Running unit tests..."
pytest tests/unit/ -v --tb=short

# Integration tests
echo "🔗 Running integration tests..."
pytest tests/integration/ -v --tb=short

# E2E tests
echo "🌐 Running end-to-end tests..."
pytest tests/e2e/ -v --tb=short

# Performance tests
echo "⚡ Running performance tests..."
pytest tests/performance/ -v --tb=short

# Coverage report
echo "📊 Generating coverage report..."
pytest --cov=src/test_framework --cov-report=html --cov-report=term-missing

echo "✅ All tests completed successfully!"
```

### Quality Gates Script

```bash
#!/bin/bash
# scripts/quality_gates.sh

set -e

echo "🚦 Running quality gates..."

# Minimum coverage threshold
COVERAGE_THRESHOLD=85

# Run tests with coverage
pytest --cov=src/test_framework --cov-report=xml --quiet

# Check coverage threshold
coverage report --fail-under=$COVERAGE_THRESHOLD

# Run mutation testing
echo "🧬 Running mutation testing..."
mutmut run --paths-to-mutate=src/test_framework

# Check mutation score
MUTATION_SCORE=$(mutmut results | grep "Mutation score" | awk '{print $3}' | tr -d '%')
if (( $(echo "$MUTATION_SCORE < 70" | bc -l) )); then
    echo "❌ Mutation score ($MUTATION_SCORE%) below threshold (70%)"
    exit 1
fi

echo "✅ All quality gates passed!"
```

## Test Documentation

### Auto-Generated Test Reports

```python
# scripts/generate_test_docs.py
import pytest
import json
from pathlib import Path

def generate_test_documentation():
    """Generate comprehensive test documentation."""
    
    # Run pytest with json report
    pytest.main([
        "--json-report",
        "--json-report-file=build/test-report.json",
        "tests/"
    ])
    
    # Load test results
    with open("build/test-report.json") as f:
        test_data = json.load(f)
    
    # Generate markdown documentation
    doc_content = generate_markdown_report(test_data)
    
    # Write documentation
    Path("docs/test-report.md").write_text(doc_content)

def generate_markdown_report(test_data):
    """Generate markdown test report."""
    total_tests = test_data["summary"]["total"]
    passed_tests = test_data["summary"]["passed"]
    failed_tests = test_data["summary"]["failed"]
    
    content = f"""# Test Report

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests}
- **Failed**: {failed_tests}
- **Success Rate**: {(passed_tests/total_tests)*100:.1f}%

## Test Results by Category
"""
    
    # Add detailed results
    for test in test_data["tests"]:
        content += f"### {test['nodeid']}\n"
        content += f"**Status**: {test['outcome']}\n"
        content += f"**Duration**: {test['duration']:.3f}s\n\n"
    
    return content

if __name__ == "__main__":
    generate_test_documentation()
```

## Best Practices

### Test Organization

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **One Assertion Per Test**: Keep tests focused
3. **Descriptive Names**: Test names should describe behavior
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Feedback**: Unit tests should run quickly

### Test Data

1. **Use Factories**: Generate test data dynamically
2. **Isolate Data**: Each test should have clean data
3. **Realistic Data**: Use data that resembles production
4. **Edge Cases**: Test boundary conditions
5. **Error Conditions**: Test failure scenarios

### Coverage Goals

1. **Aim for 80%+**: Good coverage baseline
2. **100% Critical Paths**: Cover all critical functionality
3. **Branch Coverage**: Test all code branches
4. **Mutation Testing**: Verify test quality
5. **Regular Monitoring**: Track coverage trends

### Performance Testing

1. **Baseline Metrics**: Establish performance baselines
2. **Load Testing**: Test under realistic load
3. **Stress Testing**: Find breaking points
4. **Resource Monitoring**: Monitor memory and CPU
5. **Regression Testing**: Prevent performance regressions

## Tools and Libraries

### Core Testing
- **pytest**: Modern testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel execution
- **pytest-mock**: Mocking utilities

### Test Data
- **factory-boy**: Test data factories
- **faker**: Realistic fake data
- **hypothesis**: Property-based testing
- **freezegun**: Time mocking

### Quality Assurance
- **mutmut**: Mutation testing
- **bandit**: Security testing
- **safety**: Dependency security
- **codecov**: Coverage tracking

### Performance
- **pytest-benchmark**: Performance benchmarking
- **locust**: Load testing
- **memory-profiler**: Memory usage analysis

## License

MIT License - see LICENSE file for details