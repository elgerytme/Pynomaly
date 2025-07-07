# Buck2 Incremental Testing System

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📄 Buck2_Incremental_Testing

---


## Overview

The Buck2 Incremental Testing System for Pynomaly provides intelligent, change-based testing that dramatically reduces CI/CD time by running only the tests affected by code changes. This system analyzes Git commits, maps changes to Buck2 targets, assesses impact risk, and executes the optimal testing strategy.

## Key Features

- **Change Detection**: Automatically identifies affected files and their dependencies
- **Impact Analysis**: Assesses risk levels and recommends appropriate testing strategies
- **Incremental Testing**: Runs only necessary tests based on changes
- **Git Integration**: Works seamlessly with Git workflows and commit history
- **CI/CD Integration**: GitHub Actions workflow for automated testing
- **Multiple Strategies**: From minimal to comprehensive testing based on risk assessment

## Architecture

### Core Components

1. **Buck2ChangeDetector** (`scripts/buck2_change_detector.py`)
   - Detects changed files between commits
   - Maps files to Buck2 targets
   - Analyzes dependency relationships

2. **Buck2IncrementalTestRunner** (`scripts/buck2_incremental_test.py`)
   - Executes Buck2 tests in parallel
   - Provides detailed test results and timing
   - Supports dry-run mode

3. **Buck2GitIntegration** (`scripts/buck2_git_integration.py`)
   - Advanced Git operations and commit analysis
   - Branch comparison and validation
   - Bisect functionality for finding breaking commits

4. **Buck2ImpactAnalyzer** (`scripts/buck2_impact_analyzer.py`)
   - Risk assessment based on changed components
   - Test strategy recommendation
   - Component metrics calculation

5. **Buck2Workflow** (`scripts/buck2_workflow.py`)
   - Orchestrates the complete testing workflow
   - Provides multiple workflow types
   - Results tracking and reporting

## Quick Start

### Prerequisites

1. **Buck2 Installation**:
   ```bash
   # Download and install Buck2
   curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu.zst | zstd -d > buck2
   chmod +x buck2
   sudo mv buck2 /usr/local/bin/
   ```

2. **Python Dependencies**:
   ```bash
   # Already included in Pynomaly's poetry dependencies
   poetry install
   ```

### Basic Usage

1. **Test Changes in Current Branch**:
   ```bash
   python scripts/buck2_workflow.py branch --strategy auto
   ```

2. **Test Specific Commit Range**:
   ```bash
   python scripts/buck2_workflow.py standard --base HEAD~3 --target HEAD
   ```

3. **Validate Each Commit**:
   ```bash
   python scripts/buck2_workflow.py validate-commits
   ```

4. **Find Breaking Commit**:
   ```bash
   python scripts/buck2_workflow.py bisect
   ```

## Detailed Usage

### Change Detection

```bash
# Analyze changes between commits
python scripts/buck2_change_detector.py --base HEAD~1 --target HEAD

# Output summary format
python scripts/buck2_change_detector.py --format summary

# Save analysis to JSON
python scripts/buck2_change_detector.py --format json --output analysis.json
```

### Impact Analysis

```bash
# Analyze impact and get test recommendations
python scripts/buck2_impact_analyzer.py --base HEAD~1 --target HEAD

# Run recommended tests automatically
python scripts/buck2_impact_analyzer.py --run-tests

# Save detailed analysis
python scripts/buck2_impact_analyzer.py --output impact_analysis.json
```

### Incremental Testing

```bash
# Run incremental tests with fail-fast
python scripts/buck2_incremental_test.py --fail-fast

# Dry run to see what would be executed
python scripts/buck2_incremental_test.py --dry-run

# Specify parallel jobs and timeout
python scripts/buck2_incremental_test.py --jobs 8 --timeout 600
```

### Git Integration

```bash
# Test current branch against main
python scripts/buck2_git_integration.py test-branch --base main

# Test staged changes
python scripts/buck2_git_integration.py test-staged

# Test each commit individually
python scripts/buck2_git_integration.py test-commits

# Setup Git hooks for automatic testing
python scripts/buck2_git_integration.py setup-hooks

# Get branch information
python scripts/buck2_git_integration.py branch-info
```

## Test Strategies

### Automatic Strategy Selection

The system automatically selects test strategies based on risk assessment:

- **Minimal** (Low Risk): Documentation, examples, configuration changes
- **Standard** (Medium Risk): Regular code changes with good test coverage
- **Comprehensive** (High Risk): Critical components, complex changes
- **Full** (Critical Risk): Domain entities, major architectural changes

### Manual Strategy Override

```bash
# Force minimal testing
python scripts/buck2_workflow.py standard --strategy minimal

# Force comprehensive testing
python scripts/buck2_workflow.py standard --strategy comprehensive
```

## Risk Assessment

### Risk Factors

1. **Component Criticality**:
   - Critical: Domain entities, use cases, core adapters
   - Important: Application services, API endpoints
   - Standard: Utilities, configuration
   - Low Risk: Documentation, examples

2. **Code Metrics**:
   - Lines of code and complexity
   - Test coverage percentage
   - Change frequency
   - Dependency relationships

3. **File Types**:
   - Python code: 1.0x multiplier
   - Configuration: 0.3-0.4x multiplier
   - Documentation: 0.1x multiplier

### Risk Levels

- **Low (0.0-0.3)**: Simple changes, well-tested components
- **Medium (0.3-0.6)**: Standard code changes
- **High (0.6-0.8)**: Complex changes, critical components
- **Critical (0.8-1.0)**: Domain logic, major architectural changes

## CI/CD Integration

### GitHub Actions

The system includes a comprehensive GitHub Actions workflow (`.github/workflows/buck2-incremental-testing.yml`) that:

1. Analyzes changes and determines risk level
2. Runs appropriate tests based on strategy
3. Comments on pull requests with results
4. Runs security scans for high-risk changes
5. Performs performance regression checks

### Usage in CI

```yaml
# Trigger the workflow
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
```

The workflow automatically:
- Detects changes in PRs
- Runs impact analysis
- Executes recommended tests
- Posts results as PR comments
- Fails the build if tests fail

### Manual Workflow Dispatch

```bash
# Trigger via GitHub CLI
gh workflow run buck2-incremental-testing.yml \
  -f test_strategy=comprehensive \
  -f base_ref=main
```

## Configuration

### Buck2 Configuration

The system works with the existing `BUCK` file and `.buckconfig`. Key Buck2 targets:

- **Test Targets**: `test-domain`, `test-application`, `test-infrastructure`, `test-presentation`
- **Build Targets**: `pynomaly-lib`, `pynomaly-cli`, `pynomaly-api`, `pynomaly-web`
- **Quality Targets**: `benchmarks`, `property-tests`, `security-tests`, `mutation-tests`

### Target Mapping

The system maps file patterns to Buck2 targets:

```python
target_map = {
    "src/pynomaly/domain/": {"domain", "test-domain"},
    "src/pynomaly/application/": {"application", "test-application"},
    "src/pynomaly/infrastructure/": {"infrastructure", "test-infrastructure"},
    "src/pynomaly/presentation/": {"presentation", "test-presentation"},
    # ... additional mappings
}
```

## Performance Benefits

### Time Savings

- **Full Test Suite**: ~30 minutes
- **Incremental (Low Risk)**: ~2 minutes
- **Incremental (Medium Risk)**: ~5 minutes
- **Incremental (High Risk)**: ~15 minutes

### Efficiency Gains

- 80-90% reduction in test execution time for typical changes
- Parallel execution with configurable job count
- Smart dependency analysis prevents missing related tests
- Early failure detection with fail-fast option

## Advanced Features

### Commit Validation

Validate each commit individually to find exactly where issues were introduced:

```bash
python scripts/buck2_workflow.py validate-commits
```

### Bisect Functionality

Automatically find the first commit that introduced test failures:

```bash
python scripts/buck2_workflow.py bisect
```

### Performance Monitoring

Track test execution times and identify performance regressions:

```bash
# Run with benchmarks for performance-sensitive changes
python scripts/buck2_impact_analyzer.py --run-tests
```

### Security Integration

Automatic security scanning for high-risk changes:

```bash
# Automatically triggered for critical/high-risk changes in CI
bandit -r src/ -f json -o bandit-report.json
safety check --json --output safety-report.json
```

## Troubleshooting

### Common Issues

1. **Buck2 Not Found**:
   ```bash
   # Ensure Buck2 is installed and in PATH
   buck2 --version
   ```

2. **Import Errors**:
   ```bash
   # Ensure you're in the project root and using the right Python environment
   export PYTHONPATH="$(pwd)/scripts:$PYTHONPATH"
   ```

3. **Git Permission Issues**:
   ```bash
   # Ensure Git hooks have proper permissions
   chmod +x .git/hooks/*
   ```

### Debug Mode

```bash
# Enable verbose logging for troubleshooting
python scripts/buck2_workflow.py standard --verbose
```

### Dry Run

```bash
# See what would be executed without running tests
python scripts/buck2_workflow.py standard --dry-run
```

## Integration with Existing Tools

### Poetry Integration

The system works seamlessly with Poetry for dependency management:

```bash
# Run through Poetry
poetry run python scripts/buck2_workflow.py branch
```

### Pre-commit Hooks

```bash
# Install Git hooks for automatic testing
python scripts/buck2_git_integration.py setup-hooks
```

### IDE Integration

The scripts can be integrated into IDEs for quick testing:

```bash
# VS Code task example
{
  "label": "Buck2 Test Changes",
  "type": "shell",
  "command": "python scripts/buck2_workflow.py branch --strategy auto"
}
```

## Best Practices

### Development Workflow

1. **Before Committing**:
   ```bash
   python scripts/buck2_git_integration.py test-staged
   ```

2. **Before Pushing**:
   ```bash
   python scripts/buck2_workflow.py branch
   ```

3. **Code Review**:
   - Check PR comments for test results
   - Review risk assessment and recommendations
   - Ensure appropriate test coverage

### Maintenance

1. **Update Target Mappings**: Regularly review and update file-to-target mappings as the codebase evolves

2. **Monitor Performance**: Track test execution times and optimize slow tests

3. **Review Risk Assessment**: Adjust risk multipliers based on experience with different types of changes

## Future Enhancements

### Planned Features

1. **Machine Learning**: Learn from historical data to improve risk assessment
2. **Test Prioritization**: Smart ordering of tests based on failure probability
3. **Predictive Analysis**: Predict potential issues before running tests
4. **Integration Testing**: Better handling of integration test dependencies
5. **Coverage Tracking**: Real-time test coverage monitoring and recommendations

### Extension Points

The system is designed to be extensible:

- Custom risk assessment algorithms
- Additional CI/CD platform support
- Integration with other build systems
- Custom test strategies and policies

## Contributing

To contribute to the Buck2 incremental testing system:

1. Follow the existing code patterns and documentation standards
2. Add tests for new functionality
3. Update this documentation for any changes
4. Ensure compatibility with the existing Buck2 configuration

## Support

For issues with the Buck2 incremental testing system:

1. Check the troubleshooting section above
2. Review the verbose logs with `--verbose` flag
3. Validate Buck2 configuration with `buck2 targets //...`
4. Ensure all dependencies are properly installed

## Summary

The Buck2 Incremental Testing System provides intelligent, efficient testing that scales with your codebase. By analyzing changes, assessing risk, and running only necessary tests, it dramatically reduces CI/CD time while maintaining comprehensive test coverage and quality assurance.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
