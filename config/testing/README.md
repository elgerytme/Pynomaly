# Testing Configuration

This directory contains testing configurations for the Pynomaly monorepo.

## Files

- `tox.ini` - Multi-environment testing configuration (moved from src/temporary/)
- `buck2_testing.bzl` - Buck2 integration for tox environments

## Usage

### With Tox (Traditional)
```bash
# Run all test environments
tox

# Run specific environment
tox -e lint
tox -e py311-unit
tox -e integration

# Run tests in parallel
tox -p auto
```

### With Buck2 (Recommended)
```bash
# Run all tests through Buck2
buck2 test //:ai-tests //:data-tests //:enterprise-tests

# Run specific package tests
buck2 test //src/packages/ai/machine_learning:tests

# Run with tox integration
buck2 run //:tox-lint
buck2 run //:tox-integration
```

## Integration with Buck2

The tox configuration is integrated with Buck2 through custom rules that:

1. **Leverage tox environments** for consistent testing across local and CI
2. **Use Buck2's caching** to speed up test execution
3. **Maintain compatibility** with existing CI/CD pipelines
4. **Enable parallel execution** across different Python versions

## Configuration Updates

When the tox configuration was moved from `src/temporary/`, the following updates were made:

- Updated path references to work from repository root
- Integrated with Buck2 test discovery
- Maintained GitHub Actions compatibility
- Preserved all testing environments and configurations

## Test Environments

The configuration includes comprehensive test environments:

- **lint** - Code formatting and style checks (black, ruff, isort)
- **type** - Type checking (mypy)
- **unit** - Unit tests with coverage (pytest)
- **integration** - Integration tests with services
- **mutation** - Mutation testing for test quality
- **e2e-ui** - End-to-end UI testing
- **performance** - Performance and load testing
- **security** - Security scanning and vulnerability testing

Each environment is configured with appropriate dependencies, environment variables, and test commands.