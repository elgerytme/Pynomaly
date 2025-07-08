# Testing-Phase Security Guidance

This document outlines best practices for unit, integration, and security testing in the Pynomaly project.

## 1. Unit and Integration Testing

### Use of pytest Fixtures to Inject Mocked Secrets
- **Fixtures**: Use `pytest` fixtures to setup initial state and dependencies for tests.
- **Mocked Secrets**: Use `.env` files or environmental variables to inject mocked secrets using fixtures.

Example:
```python
@pytest.fixture()
def mock_secret():
    return "mock-secret-value"
```

### Test Data Handling and GDPR Considerations
- **Anonymize Data**: Ensure test data is anonymized and does not include any personally identifiable information (PII).
- **Data Storage**: Temporary test data should be securely stored and properly cleaned up after tests.

## 2. Static Analysis

### Bandit and Ruff
- **Bandit**: Use Bandit to find common security issues in Python code.
  ```bash
  bandit -r src/ -f json -o reports/bandit-report.json
  ```
- **Ruff**: Perform linting with Ruff to catch stylistic errors and potential bugs.
  ```bash
  ruff check src/
  ```

Include these in CI/CD to ensure they are executed on each commit.

## 3. Dynamic Analysis

### OWASP ZAP Against FastAPI Endpoints
- **Setup**: OWASP ZAP can be used to perform security testing against API endpoints.
- **Automation**: Integrate running OWASP ZAP in CI to automatically scan APIs.
  ```yaml
  jobs:
    zap:
      runs-on: ubuntu-latest
      steps:
        - name: Run ZAP Baseline Scan
          run: docker run -v $(pwd)/reports/:/zap/wrk/:rw -t owasp/zap2docker-stable zap-baseline.py -t http://api:8000
  ```

## 4. Mutation and Property-Based Testing

### Current Framework
- **Existing Tests**: The project already includes mutation and property-based tests using `Hypothesis`.

Example:
```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.integers())
def test_integer_manipulation(i):
    assert some_function(i) == expected_result
```

### Integration
- Ensure these are integrated into regular test cycles and results are evaluated.

## 5. CI/CD Integration

### Example CI Fragment
Here's an example YAML fragment for integrating security checks in a CI pipeline:
```yaml
dependency-scan:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install Dependencies
      run: |
        pip install -r requirements.txt
    - name: Run Bandit
      run: bandit -r src/
    - name: Run Ruff
      run: ruff check src/
```

# Testing-Phase Security Guidance

This document outlines best practices for unit, integration, and security testing in the Pynomaly project.

## 1. Unit and Integration Testing

### Use of pytest Fixtures to Inject Mocked Secrets
- **Fixtures**: Use `pytest` fixtures to setup initial state and dependencies for tests.
- **Mocked Secrets**: Use `.env` files or environmental variables to inject mocked secrets using fixtures.

Example:
```python
@pytest.fixture()
def mock_secret():
    return "mock-secret-value"
```

### Test Data Handling and GDPR Considerations
- **Anonymize Data**: Ensure test data is anonymized and does not include any personally identifiable information (PII).
- **Data Storage**: Temporary test data should be securely stored and properly cleaned up after tests.

## 2. Static Analysis

### Bandit and Ruff
- **Bandit**: Use Bandit to find common security issues in Python code.
  ```bash
  bandit -r src/ -f json -o reports/bandit-report.json
  ```
- **Ruff**: Perform linting with Ruff to catch stylistic errors and potential bugs.
  ```bash
  ruff check src/
  ```

Embed these in CI/CD to ensure that they are automatically executed on each commit.

## 3. Dynamic Analysis

### OWASP ZAP Against FastAPI Endpoints
- **Setup**: OWASP ZAP can be used to perform security testing against API endpoints.
- **Automation**: Integrate running OWASP ZAP in CI to automatically scan APIs.
  ```yaml
  jobs:
    zap:
      runs-on: ubuntu-latest
      steps:
        - name: Run ZAP Baseline Scan
          run: docker run -v $(pwd)/reports/:/zap/wrk/:rw -t owasp/zap2docker-stable zap-baseline.py -t http://api:8000
  ```

## 4. Mutation and Property-Based Testing

### Current Framework
- **Existing Tests**: The project already includes mutation and property-based tests using `Hypothesis`.

Example:
```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.integers())
def test_integer_manipulation(i):
    assert some_function(i) == expected_result
```

### Integration
- Ensure these are integrated into regular test cycles and results are evaluated.

## 5. CI/CD Integration

### Example CI Fragment
Here's an example YAML fragment for integrating security checks in a CI pipeline:
```yaml
dependency-scan:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install Dependencies
      run: |
        pip install -r requirements.txt
    - name: Run Bandit
      run: bandit -r src/
    - name: Run Ruff
      run: ruff check src/
```
