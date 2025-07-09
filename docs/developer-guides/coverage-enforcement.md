# Coverage Enforcement Guide

This guide explains the coverage enforcement system implemented in the Pynomaly project.

## Overview

The Pynomaly project enforces strict code coverage thresholds to ensure high code quality and comprehensive testing. Coverage enforcement is implemented in the CI/CD pipeline and can be tested locally.

## Coverage Thresholds

The project uses different coverage thresholds for different architectural layers:

| Layer | Threshold | Rationale |
|-------|-----------|-----------|
| **Overall Project** | ≥95% | Ensures comprehensive test coverage across the entire codebase |
| **Domain Layer** | ≥98% | Business logic requires the highest coverage standard |
| **Application Layer** | ≥95% | Use cases and orchestration logic need high coverage |
| **Infrastructure Layer** | ≥90% | Adapters and external integrations have moderate requirements |
| **Presentation Layer** | ≥85% | UI and API layers have more flexible requirements |

## CI/CD Integration

### Validation Suite Workflow

The coverage enforcement is integrated into the `validation-suite.yml` workflow:

1. **Test Execution**: Tests are run with coverage collection using `pytest-cov`
2. **JSON Report Generation**: Coverage data is exported to `coverage.json` for robust parsing
3. **Threshold Validation**: Per-package coverage is calculated and validated against thresholds
4. **Pipeline Failure**: If any threshold is not met, the pipeline fails with clear error messages

### Key Features

- **Robust JSON Parsing**: Replaced fragile XML parsing with `coverage json` output
- **Per-Package Analysis**: Individual coverage analysis for each architectural layer
- **Clear Error Messages**: Detailed feedback on which thresholds are failing
- **GitHub Output**: Sets appropriate `$GITHUB_OUTPUT` variables for downstream jobs

## Local Testing

### Coverage Enforcement Test Script

Use the provided test script to validate coverage enforcement locally:

```bash
python scripts/validation/test_coverage_enforcement.py
```

This script:
- Runs tests with coverage collection
- Parses the coverage JSON report
- Validates all thresholds
- Provides clear feedback on pass/fail status

### Manual Testing

You can also test coverage manually:

```bash
# Run tests with coverage
hatch run test:run tests/domain/ tests/application/ -v --tb=short \
  --cov=src/pynomaly --cov-report=json:coverage.json --cov-report=term

# Check overall coverage
jq '.totals.percent_covered' coverage.json

# Check domain coverage
jq '.files | to_entries | map(select(.key | startswith("src/pynomaly/domain"))) | map(.value.summary.percent_covered) | add / length' coverage.json
```

## Implementation Details

### Coverage Collection

The system uses `pytest-cov` to collect coverage data:

```yaml
--cov=src/pynomaly --cov-report=json:coverage.json --cov-report=term
```

### JSON Parsing

Coverage data is parsed using `jq` commands in the CI pipeline:

```bash
# Overall coverage
OVERALL_COVERAGE=$(jq -r '.totals.percent_covered' coverage.json)

# Per-package coverage
DOMAIN_COVERAGE=$(jq -r '.files | to_entries | map(select(.key | startswith("src/pynomaly/domain"))) | map(.value.summary.percent_covered) | add / length' coverage.json)
```

### Threshold Validation

Each threshold is validated using bash arithmetic:

```bash
if (( $(echo "$COVERAGE >= $THRESHOLD" | bc -l) )); then
    echo "✅ Threshold met"
else
    echo "❌ Threshold not met"
    COVERAGE_FAILED=1
fi
```

## Configuration

### Environment Variables

Coverage thresholds are configured as environment variables in the workflow:

```yaml
env:
  COVERAGE_THRESHOLD: 95
  COVERAGE_THRESHOLD_DOMAIN: 98
  COVERAGE_THRESHOLD_APPLICATION: 95
  COVERAGE_THRESHOLD_INFRASTRUCTURE: 90
  COVERAGE_THRESHOLD_PRESENTATION: 85
```

### PyProject Configuration

The `pyproject.toml` file contains coverage configuration:

```toml
[tool.coverage.run]
source = ["src"]
branch = true
parallel = true
data_file = ".coverage"

[tool.coverage.report]
show_missing = true
fail_under = 90
```

## Troubleshooting

### Common Issues

1. **Coverage JSON not found**: Ensure tests are run with `--cov-report=json:coverage.json`
2. **Low coverage**: Add more comprehensive tests for the failing packages
3. **Parsing errors**: Check that `jq` is available in the CI environment

### Debugging Coverage

To debug coverage issues:

1. Run tests locally with coverage report
2. Check the HTML coverage report: `open htmlcov/index.html`
3. Identify uncovered lines and add appropriate tests
4. Use the test script to validate thresholds

## Benefits

### Quality Assurance

- **Consistent Standards**: Enforces consistent testing standards across the codebase
- **Early Detection**: Catches coverage regressions early in the development process
- **Architectural Alignment**: Different thresholds reflect the importance of different layers

### Developer Experience

- **Clear Feedback**: Provides specific guidance on which areas need more testing
- **Local Testing**: Allows developers to validate coverage before pushing
- **Automated Enforcement**: Reduces manual review burden

## Best Practices

### Writing Tests

1. **Focus on Business Logic**: Prioritize testing domain layer components
2. **Test Edge Cases**: Cover error conditions and boundary cases
3. **Use Property-Based Testing**: Leverage hypothesis for comprehensive test coverage
4. **Mock External Dependencies**: Focus tests on your code, not external systems

### Maintaining Coverage

1. **Regular Monitoring**: Check coverage reports regularly
2. **Incremental Improvement**: Gradually increase coverage for existing code
3. **New Code Standards**: Ensure new code meets coverage requirements
4. **Refactoring Safety**: Use coverage as a safety net during refactoring

## Future Enhancements

### Planned Improvements

1. **Dynamic Badges**: Automatically update coverage badges based on actual coverage
2. **Coverage Trends**: Track coverage changes over time
3. **Branch Coverage**: Implement branch coverage requirements
4. **Mutation Testing**: Add mutation testing for coverage quality validation

### Integration Options

1. **IDE Integration**: Coverage reporting in development environments
2. **PR Comments**: Automated coverage comments on pull requests
3. **Slack/Teams**: Coverage alerts for team communication
4. **Dashboard**: Coverage monitoring dashboard for project health

## Conclusion

The coverage enforcement system ensures that Pynomaly maintains high code quality through comprehensive testing. By implementing strict thresholds and automated validation, the project can maintain reliability while supporting rapid development.

For questions or issues with coverage enforcement, refer to the troubleshooting section or contact the development team.
