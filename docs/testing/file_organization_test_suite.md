# File Organization Test Suite

## Overview

This document describes the comprehensive test suite for file organization functionality in Pynomaly, covering detection, safe moves/deletes in dry-run mode, and actual filesystem changes with `--fix` mode.

## Test Components

### 1. Core Test File: `tests/integration/test_file_organization.py`

The main test file contains four essential test cases:

#### `test_detect_stray_files(temp_project_dir)`
- **Purpose**: Tests correct detection for each file category
- **Functionality**: 
  - Creates temporary files in wrong locations
  - Uses `detect_stray_files` function to identify issues
  - Validates suggestions for moving and deleting files
- **Categories Tested**: Testing files, temporary files

#### `test_file_organizer_dry_run(temp_project_dir)`  
- **Purpose**: Tests safe moves/deletes in dry-run mode
- **Functionality**:
  - Creates FileOrganizer with `dry_run=True`
  - Analyzes repository structure
  - Plans and executes operations without making actual changes
  - Validates that no filesystem modifications occur

#### `test_file_organizer_fix_mode(temp_project_dir)`
- **Purpose**: Tests actual filesystem changes in fix mode
- **Functionality**:
  - Creates FileOrganizer with `dry_run=False`
  - Executes real file operations
  - Validates that actual moves and deletions occur
  - Marked as `@pytest.mark.slow` for longer execution time

#### `test_file_organizer_categories()`
- **Purpose**: Tests category detection accuracy
- **Functionality**:
  - Validates `categorize_file` function
  - Tests different file types: testing, temporary, scripts, miscellaneous
  - Ensures correct categorization logic

### 2. Validation Script: `scripts/validation/validate_file_organization_suite.py`

Comprehensive validation script that verifies:

- **Test File Existence**: Checks that all required test files are present
- **Pytest Configuration**: Validates `pytest.ini` has proper markers
- **Import Validation**: Tests that all required modules can be imported
- **CI Integration**: Verifies CI workflow includes file organization tests
- **Test Execution**: Runs individual and full test suites
- **Reporting**: Provides detailed validation summary

### 3. CI Integration: `.github/workflows/tests.yml`

Updated CI workflow includes:

#### Python Version Matrix
- **Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Cross-platform**: Ubuntu, Windows, macOS
- **Test Types**: Unit, integration, performance

#### File Organization Specific Job
- **Dedicated Job**: `file-organization-tests`
- **tmpfs Testing**: Uses tmpfs for safe filesystem operations on Linux
- **Validation**: Runs validation script after tests
- **Cleanup**: Proper tmpfs cleanup after completion

#### Integration Test Execution
- **Standard Integration**: Runs with other integration tests
- **Specific Execution**: Dedicated file organization test runs
- **Timeout Protection**: 120-180 second timeouts for safety

## Pytest Configuration

### Markers in `pytest.ini`
```ini
markers =
    integration: Integration tests
    file_organization: File organization tests  
    slow: Slow running tests
```

### Test Discovery
- **Test Path**: `tests/integration/test_file_organization.py`
- **Marker Usage**: `-m "file_organization"`
- **Isolation**: Can run independently or with other integration tests

## Usage Examples

### Run All File Organization Tests
```bash
pytest tests/integration/test_file_organization.py -v
```

### Run Only File Organization Marked Tests
```bash
pytest -m file_organization -v
```

### Run Validation Suite
```bash
python scripts/validation/validate_file_organization_suite.py
```

### CI Local Simulation
```bash
# Create tmpfs (Linux only)
sudo mkdir -p /tmp/pynomaly_test_tmpfs
sudo mount -t tmpfs -o size=100m tmpfs /tmp/pynomaly_test_tmpfs

# Run tests with tmpfs
TMPDIR=/tmp/pynomaly_test_tmpfs pytest tests/integration/test_file_organization.py -v

# Cleanup
sudo umount /tmp/pynomaly_test_tmpfs
```

## Test Coverage Areas

### Detection Accuracy
- ✅ Testing file detection (`test_*.py`)
- ✅ Temporary file detection (`*.tmp`, `*temp*`)
- ✅ Script file detection (`setup_*.py`, `run_*.py`)
- ✅ Miscellaneous file handling

### Safe Operations
- ✅ Dry-run mode validation
- ✅ No filesystem changes in dry-run
- ✅ Operation planning and validation
- ✅ Error handling and reporting

### Fix Mode Operations
- ✅ Actual file moves and deletions
- ✅ Directory creation when needed
- ✅ Backup handling for conflicts
- ✅ Operation execution tracking

### Cross-Platform Compatibility
- ✅ Windows PowerShell support
- ✅ Linux bash support
- ✅ macOS compatibility
- ✅ Path handling across systems

## Quality Assurance

### Automated Validation
- **Import Testing**: Validates all required modules can be imported
- **Configuration Testing**: Ensures pytest configuration is correct
- **Integration Testing**: Verifies CI workflow integration
- **Execution Testing**: Runs actual test scenarios

### Safety Measures
- **tmpfs Usage**: Ensures safe filesystem operations in CI
- **Dry-run Testing**: Validates no unintended changes
- **Timeout Protection**: Prevents hanging tests
- **Error Isolation**: Comprehensive error handling

### Performance Considerations
- **Test Speed**: Fast execution for frequent testing
- **Resource Management**: Efficient temporary directory usage
- **Parallel Safety**: Safe for parallel test execution
- **Memory Efficiency**: Minimal memory footprint

## Maintenance

### Adding New Tests
1. Add test functions to `test_file_organization.py`
2. Use appropriate pytest markers
3. Update validation script if needed
4. Test across Python versions

### Updating Categories
1. Modify `categorize_file` function in `detect_stray_files.py`
2. Update corresponding tests in `test_file_organizer_categories`
3. Run validation suite to ensure compatibility

### CI Updates
1. Update `.github/workflows/tests.yml` for new requirements
2. Test CI changes in feature branch
3. Validate across all Python versions

## Summary

The comprehensive file organization test suite provides:

- **Complete Coverage**: All major functionality tested
- **Cross-Platform**: Works on Python 3.8-3.12 across OS platforms  
- **CI Integration**: Automated testing in continuous integration
- **Safety First**: Dry-run and tmpfs for safe testing
- **Quality Assurance**: Validation scripts and comprehensive checking
- **Documentation**: Full documentation and usage examples

This test suite ensures reliable file organization functionality and maintains high code quality standards throughout the development lifecycle.
