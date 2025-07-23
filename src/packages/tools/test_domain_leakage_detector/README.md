# Test Domain Leakage Detector

A tool to detect and prevent test domain leakage in monorepo architectures by ensuring that:

1. Package tests only import from their own package (no cross-package imports)
2. System tests don't directly import from domain packages
3. Repository-level tests don't import from specific packages

## Features

- **Package Test Isolation**: Prevents package tests from importing other packages
- **System Test Boundaries**: Ensures system tests use public APIs only
- **Repository Test Separation**: Keeps repository tests isolated from package internals
- **Configurable Rules**: Uses `.domain-boundaries.yaml` for configuration
- **Multiple Output Formats**: Console and JSON reporting
- **CI/CD Integration**: Strict mode for pipeline failures
- **Suggested Fixes**: Provides recommendations for fixing violations

## Installation

```bash
pip install -e .
```

## Usage

### Basic Scanning

```bash
# Scan current directory
test-domain-leakage-detector scan

# Scan specific path
test-domain-leakage-detector scan --path /path/to/project

# Enable verbose output
test-domain-leakage-detector scan --verbose
```

### Output Formats

```bash
# Console output (default)
test-domain-leakage-detector scan --format console

# JSON output
test-domain-leakage-detector scan --format json

# Save to file
test-domain-leakage-detector scan --format json --output report.json
```

### CI/CD Integration

```bash
# Exit with error code on violations (for CI/CD)
test-domain-leakage-detector scan --strict

# Show suggested fixes
test-domain-leakage-detector scan --show-fixes
```

## Configuration

The tool uses the `testing` section in `.domain-boundaries.yaml`:

```yaml
testing:
  rules:
    - name: no_cross_package_imports_in_package_tests
      description: "Package tests must not import from other packages"
      severity: critical
      scope: "src/packages/*/tests/**/*.py"
    
    - name: no_package_imports_in_system_tests  
      description: "System tests must not import from specific packages"
      severity: critical
      scope: "src/packages/system_tests/**/*.py"
    
    - name: no_repo_level_package_imports
      description: "Repository-level tests must not import from packages"
      severity: critical
      scope: "tests/**/*.py"

  exceptions:
    - file: "src/packages/*/tests/conftest.py"
      pattern: "from\\s+\\.\\."
      reason: "Test configuration files may need to import package modules for fixtures"
```

## Violation Types

### 1. Package Cross Import
Package tests importing from other packages:
```python
# ❌ Violation
from data.analytics import something

# ✅ Correct
from ..analytics import something
```

### 2. System Test Package Import
System tests importing domain packages directly:
```python
# ❌ Violation  
from ai.mlops import MLModel

# ✅ Correct
# Use public APIs or test fixtures instead
```

### 3. Repository Test Package Import
Repository tests importing from packages:
```python
# ❌ Violation
from src.packages.data.analytics import something
sys.path.insert(0, 'src/packages/data')

# ✅ Correct
# Use repository-level utilities only
```

## Allowed Imports

The following imports are always allowed in tests:
- Standard library modules (`os`, `sys`, `json`, etc.)
- Testing frameworks (`pytest`, `unittest`, `hypothesis`)  
- Test utilities (`faker`, `freezegun`, `testcontainers`)
- Type annotations (`typing`, `abc`)

## Integration with Existing Tools

This tool complements the existing domain boundary detector by focusing specifically on test isolation. Use both tools together for comprehensive boundary enforcement:

```bash
# Check general domain boundaries
domain-boundary-detector scan

# Check test domain leakage
test-domain-leakage-detector scan
```

## Exit Codes

- `0`: No violations found or non-strict mode
- `1`: Critical violations found in strict mode

## Example Output

```
❌ Found 3 test domain leakage violations:

CRITICAL: 2 violations
  📁 src/packages/ai/mlops/tests/test_model.py:15
     Package tests must only import from their own package - use relative imports
     Import: from data.analytics import DataProcessor
     💡 Suggested fix: from .data.analytics import DataProcessor

  📁 src/packages/system_tests/test_integration.py:8
     System tests must not import directly from domain packages - use public APIs or test fixtures
     Import: from ai.mlops import MLModel

WARNING: 1 violations
  📁 tests/test_repository.py:12
     Repository tests must not import from src.packages - use repository-level utilities only
     Import: from src.packages.data.analytics import something
```