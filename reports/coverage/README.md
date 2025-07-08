# Coverage Reports Directory

This directory contains coverage reports and related artifacts from test execution.

## File Organization

### Runtime-Generated Files (Not Committed)
These files are generated during test runs and are ignored by git:
- `coverage.json` - Detailed coverage data in JSON format
- `*.html` - Coverage report HTML files (e.g., `test.html`)
- `htmlcov/` - HTML coverage report directory

### Committed Files (If Any)
- Static documentation or configuration files
- Report templates
- Coverage baseline or reference files (if needed for comparison)

## Usage

Coverage reports are generated automatically during test execution using:
```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=json --cov-report=term

# Reports will be generated in this directory
```

## Configuration

Coverage configuration is defined in:
- `.coveragerc` - Main coverage configuration
- `pyproject.toml` - Tool-specific settings
- `pytest.ini` - Test runner configuration

## Cleanup

Runtime coverage files are automatically cleaned up by CI/CD processes and are not committed to version control.
