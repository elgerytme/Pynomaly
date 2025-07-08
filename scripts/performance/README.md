# Performance CLI Utilities

This directory contains CLI utilities for running performance benchmarks and detecting performance regressions in CI/CD pipelines.

## Files

### `run_benchmarks.py`
A comprehensive benchmark runner that:
- Invokes pytest with performance markers
- Runs standalone BenchmarkingService if needed
- Produces `current_results.json` for regression analysis

### `check_regressions.py`
A regression detector that:
- Compares current results against baseline
- Checks for critical performance regressions
- Exits with code 1 if critical regressions are found
- Prints markdown summary to stdout for PR comments

### `performance_config.yml`
Configuration file that defines:
- Severity thresholds for regressions
- Performance thresholds for different metrics
- Algorithm-specific thresholds
- Benchmark configuration
- CI/CD integration settings

## Usage

### Running Benchmarks

```bash
# Run with default configuration
python run_benchmarks.py

# Run with custom configuration and output file
python run_benchmarks.py --config performance_config.yml --output current_results.json --verbose

# Run in CI/CD pipeline
python run_benchmarks.py --config performance_config.yml --output current_results.json
```

### Checking Regressions

```bash
# Check for regressions
python check_regressions.py --baseline baseline.json --current current_results.json --config performance_config.yml

# In CI/CD pipeline with PR comment generation
python check_regressions.py --baseline baseline.json --current current_results.json --config performance_config.yml > pr_comment.md
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Testing

on:
  pull_request:
    branches: [ main ]

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test]
        pip install pytest-json-report pyyaml
    
    - name: Run performance benchmarks
      run: |
        python scripts/performance/run_benchmarks.py --config scripts/performance/performance_config.yml --output current_results.json
    
    - name: Download baseline results
      run: |
        # Download or generate baseline.json from main branch
        # This is implementation-specific
        
    - name: Check for regressions
      id: regression_check
      run: |
        python scripts/performance/check_regressions.py --baseline baseline.json --current current_results.json --config scripts/performance/performance_config.yml > pr_comment.md
        echo "regression_found=$?" >> $GITHUB_OUTPUT
    
    - name: Comment PR
      if: steps.regression_check.outputs.regression_found == '1'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const comment = fs.readFileSync('pr_comment.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

## Configuration

The `performance_config.yml` file allows you to configure:

- **Severity Thresholds**: Define when changes are considered minor, major, or critical
- **Performance Thresholds**: Set limits for execution time, memory usage, throughput, and accuracy
- **Algorithm-Specific Thresholds**: Configure different limits for different algorithms
- **Benchmark Configuration**: Control test parameters like iterations, data sizes, etc.
- **CI Settings**: Configure behavior for CI/CD integration

## Output Format

### `current_results.json`
The benchmark runner produces a JSON file with:
- Metadata about the benchmark run
- Pytest results with performance markers
- Standalone benchmark results
- Combined performance metrics grouped by algorithm
- Summary statistics

### Markdown Summary
The regression checker outputs a markdown report suitable for PR comments:
- Summary of total regressions and improvements
- Detailed breakdown of each regression
- Severity classification
- Metric comparisons with baseline

## Exit Codes

- **`run_benchmarks.py`**: Exits with 1 if benchmarks fail, 0 on success
- **`check_regressions.py`**: Exits with 1 if critical regressions found, 0 otherwise

## Requirements

- Python 3.11+
- pytest with json-report plugin
- PyYAML for configuration loading
- Access to the BenchmarkingService (for standalone benchmarks)

## Integration with Existing Tests

The utilities are designed to work with existing pytest performance tests marked with `@pytest.mark.performance`. They will automatically discover and run these tests as part of the benchmark suite.
