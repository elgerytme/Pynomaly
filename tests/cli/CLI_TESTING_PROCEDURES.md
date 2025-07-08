# Pynomaly CLI Testing Procedures
## Comprehensive Testing Documentation for Fresh Environments

### Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Test Environment Setup](#test-environment-setup)
4. [Running Tests](#running-tests)
5. [Test Results Analysis](#test-results-analysis)
6. [Troubleshooting](#troubleshooting)
7. [Continuous Integration](#continuous-integration)
8. [Best Practices](#best-practices)

---

## Overview

This document provides detailed procedures for testing the Pynomaly CLI in fresh environments. The testing suite validates functionality, performance, and reliability across different platforms and configurations.

### Testing Objectives
- **Functionality Validation**: Ensure all CLI commands work as expected
- **Cross-Platform Compatibility**: Verify consistent behavior across Bash and PowerShell
- **Installation Testing**: Validate clean installation procedures
- **Performance Verification**: Confirm acceptable performance characteristics
- **Error Handling**: Test graceful handling of error conditions
- **User Experience**: Validate ease of use for new users

### Supported Environments
- **Bash**: Linux (Ubuntu 20.04+), macOS (10.15+), WSL2
- **PowerShell**: Windows 10+, Windows Server 2019+, PowerShell Core 7.0+

---

## Prerequisites

### System Requirements

#### For Bash Testing
```bash
# Required software
- Python 3.11 or higher
- Git
- curl or wget
- bc (for calculations)

# Verify prerequisites
python3 --version  # Should show 3.11+
git --version
curl --version
bc --version
```

#### For PowerShell Testing
```powershell
# Required software
- Python 3.11 or higher
- Git for Windows
- PowerShell 5.1+ or PowerShell Core 7.0+

# Verify prerequisites
python --version  # Should show 3.11+
git --version
$PSVersionTable.PSVersion  # PowerShell version
```

### Network Requirements
- Internet connectivity for package installation
- Access to PyPI (pip package index)
- Git repository access (if testing from source)

---

## Test Environment Setup

### Automated Setup
Both test scripts handle environment setup automatically:

1. **Virtual Environment Creation**: Isolated Python environment
2. **Dependency Installation**: Pynomaly and all dependencies
3. **Test Data Generation**: Sample datasets for testing
4. **Temporary Directory**: Clean workspace for test artifacts

### Manual Setup (Optional)
If you prefer manual setup:

```bash
# Bash environment
python3 -m venv test_venv
source test_venv/bin/activate
pip install -e /path/to/pynomaly

# PowerShell environment
python -m venv test_venv
.\test_venv\Scripts\Activate.ps1
pip install -e C:\path\to\pynomaly
```

---

## Running Tests

### Bash Environment Testing

#### Basic Test Execution
```bash
# Navigate to test directory
cd /path/to/pynomaly/tests/cli

# Make script executable
chmod +x test_cli_bash.sh

# Run tests
./test_cli_bash.sh
```

#### Advanced Options
```bash
# Run with custom log level
LOG_LEVEL=DEBUG ./test_cli_bash.sh

# Run in verbose mode
VERBOSE=1 ./test_cli_bash.sh

# Skip performance tests (for faster execution)
SKIP_PERFORMANCE=1 ./test_cli_bash.sh
```

### PowerShell Environment Testing

#### Basic Test Execution
```powershell
# Navigate to test directory
Set-Location "C:\path\to\pynomaly\tests\cli"

# Run tests
.\test_cli_powershell.ps1
```

#### Advanced Options
```powershell
# Run with verbose output
.\test_cli_powershell.ps1 -Verbose

# Skip performance tests
.\test_cli_powershell.ps1 -SkipPerformance

# Save results to specific location
.\test_cli_powershell.ps1 -OutputPath "C:\test_results.json"

# Set custom log level
.\test_cli_powershell.ps1 -LogLevel "Debug"
```

---

## Test Results Analysis

### Understanding Test Output

#### Console Output
```
[2024-06-24 14:30:15] [Info] Starting Pynomaly CLI tests...
[2024-06-24 14:30:16] [Info] Running test: CLI Help
[SUCCESS] CLI Help (0.8s)
[2024-06-24 14:30:17] [Info] Running test: CLI Version
[SUCCESS] CLI Version (0.5s)
...
==============================================
           PYNOMALY CLI TEST SUMMARY
==============================================
Total Tests:    45
Passed:         44
Failed:         1
Success Rate:   97.8%
==============================================
```

#### Test Categories
1. **Basic Commands**: Core CLI functionality (help, version, config, status)
2. **Dataset Management**: Data loading, validation, and information
3. **Detector Management**: Algorithm creation and configuration
4. **Detection Workflow**: Training, detection, and results
5. **Autonomous Mode**: AI-powered automation features
6. **Export Functions**: Results export to various formats
7. **Performance Tests**: Timing validation on larger datasets
8. **Error Handling**: Graceful failure testing

### Result Files

#### JSON Results Format
```json
{
  "test_run": {
    "timestamp": "2024-06-24T14:30:15Z",
    "environment": "bash",
    "platform": "Linux",
    "python_version": "Python 3.11.5",
    "total_tests": 45,
    "passed_tests": 44,
    "failed_tests": 1,
    "success_rate": 97.78
  },
  "test_results": [
    {
      "name": "CLI Help",
      "status": "PASS",
      "duration": 0.8,
      "output": "Usage: pynomaly [OPTIONS] COMMAND..."
    }
  ]
}
```

#### Log File Structure
```
[2024-06-24 14:30:15] [Info] Starting Pynomaly CLI tests...
[2024-06-24 14:30:16] [Info] Setting up test environment...
[2024-06-24 14:30:17] [Info] Creating virtual environment...
[2024-06-24 14:30:18] [Success] Virtual environment created
...
```

### Performance Benchmarks

#### Expected Performance Ranges
| Test Category | Expected Duration | Warning Threshold |
|---------------|-------------------|-------------------|
| Basic Commands | < 5 seconds | > 10 seconds |
| Small Dataset (100 rows) | < 10 seconds | > 20 seconds |
| Medium Dataset (10K rows) | < 60 seconds | > 120 seconds |
| Autonomous Detection | < 30 seconds | > 60 seconds |

---

## Troubleshooting

### Common Issues

#### 1. Python Not Found
**Symptoms**: `python: command not found` or `python3: command not found`

**Solutions**:
```bash
# Linux/macOS
sudo apt-get install python3 python3-pip  # Ubuntu/Debian
brew install python3  # macOS with Homebrew

# Windows
# Download and install from python.org
# Ensure "Add to PATH" is checked during installation
```

#### 2. Virtual Environment Creation Fails
**Symptoms**: `venv: command not found` or permission errors

**Solutions**:
```bash
# Install venv module
sudo apt-get install python3-venv  # Ubuntu/Debian

# Check permissions
ls -la /tmp  # Ensure write access to temp directory

# Alternative location
export TEMP_DIR="$HOME/pynomaly_test"
```

#### 3. Package Installation Errors
**Symptoms**: pip install failures, dependency conflicts

**Solutions**:
```bash
# Update pip
python -m pip install --upgrade pip

# Clear pip cache
python -m pip cache purge

# Install with verbose output
pip install -e . -v

# Use alternative index
pip install -e . -i https://pypi.org/simple/
```

#### 4. CLI Command Not Found
**Symptoms**: `pynomaly: command not found` after installation

**Solutions**:
```bash
# Verify installation
pip list | grep pynomaly

# Check PATH
which pynomaly  # Should show path to executable

# Manual path addition
export PATH="$PATH:$(python -m site --user-base)/bin"

# Reinstall in development mode
pip install -e . --force-reinstall
```

#### 5. Test Data Generation Fails
**Symptoms**: Missing numpy/pandas or data file creation errors

**Solutions**:
```bash
# Install missing dependencies
pip install numpy pandas

# Check file permissions
ls -la tests/cli/test_data/

# Manual data creation
mkdir -p tests/cli/test_data
# Then run test script again
```

### Platform-Specific Issues

#### Windows PowerShell
```powershell
# Execution policy issues
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Path separator issues
$env:PYTHONPATH = "C:\path\to\pynomaly\src"

# Unicode encoding issues
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

#### macOS
```bash
# SSL certificate issues
/Applications/Python\ 3.11/Install\ Certificates.command

# Command line tools
xcode-select --install
```

#### WSL2
```bash
# Windows path issues
export PYTHONPATH="/mnt/c/path/to/pynomaly/src:$PYTHONPATH"

# Memory limitations
export PYTHONHASHSEED=0
```

---

## Continuous Integration

### GitHub Actions Integration

#### Workflow Configuration
```yaml
name: CLI Testing
on: [push, pull_request]

jobs:
  test-cli:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.11, 3.12]

    runs-on: ${{ matrix.os }}

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Run Bash tests
      if: runner.os != 'Windows'
      run: |
        chmod +x tests/cli/test_cli_bash.sh
        tests/cli/test_cli_bash.sh

    - name: Run PowerShell tests
      if: runner.os == 'Windows'
      run: |
        tests/cli/test_cli_powershell.ps1
```

### Local CI Pipeline
```bash
#!/bin/bash
# ci_test.sh - Local continuous integration script

# Test on multiple Python versions
for python_version in 3.11 3.12; do
    echo "Testing with Python $python_version"
    pyenv local $python_version
    ./tests/cli/test_cli_bash.sh
done

# Test with different dependency sets
pip install -e ".[minimal]"
./tests/cli/test_cli_bash.sh

pip install -e ".[all]"
./tests/cli/test_cli_bash.sh
```

---

## Best Practices

### Test Development
1. **Atomic Tests**: Each test should validate one specific functionality
2. **Idempotent Tests**: Tests should produce the same results on repeated runs
3. **Isolated Tests**: Tests should not depend on external state
4. **Clear Assertions**: Test outcomes should be clearly defined
5. **Comprehensive Coverage**: Test both success and failure scenarios

### Environment Management
1. **Clean Environments**: Always test in fresh virtual environments
2. **Dependency Locking**: Use specific versions for reproducible results
3. **Resource Cleanup**: Always clean up temporary files and processes
4. **Error Isolation**: Capture and analyze error conditions properly

### Result Analysis
1. **Trend Monitoring**: Track performance trends over time
2. **Regression Detection**: Identify functionality regressions quickly
3. **Platform Comparison**: Compare results across different platforms
4. **Performance Baselines**: Establish and maintain performance expectations

### Maintenance
1. **Regular Updates**: Keep test scripts updated with new features
2. **Documentation Sync**: Ensure documentation matches actual behavior
3. **Test Optimization**: Continuously improve test execution time
4. **Issue Tracking**: Maintain clear records of test failures and resolutions

---

## Test Script Reference

### Bash Script Functions
```bash
# Core functions
run_test()              # Execute individual test
log()                   # General logging
log_success()           # Success logging
log_error()             # Error logging
setup_test_environment() # Environment setup
generate_test_data()    # Test data creation
install_pynomaly()      # Package installation
cleanup()               # Resource cleanup
```

### PowerShell Script Functions
```powershell
# Core functions
Invoke-Test             # Execute individual test
Write-Log               # General logging
Write-Success           # Success logging
Write-Error             # Error logging
Initialize-TestEnvironment # Environment setup
New-TestData            # Test data creation
Install-Pynomaly        # Package installation
Remove-TestEnvironment  # Resource cleanup
```

### Test Categories Mapping
| Category | Bash Function | PowerShell Function |
|----------|---------------|---------------------|
| Basic | `test_basic_commands()` | `Test-BasicCommands` |
| Datasets | `test_dataset_commands()` | `Test-DatasetCommands` |
| Detectors | `test_detector_commands()` | `Test-DetectorCommands` |
| Detection | `test_detection_commands()` | `Test-DetectionCommands` |
| Autonomous | `test_autonomous_commands()` | `Test-AutonomousCommands` |
| Export | `test_export_commands()` | `Test-ExportCommands` |
| Performance | `test_performance()` | `Test-Performance` |
| Errors | `test_error_handling()` | `Test-ErrorHandling` |

---

## Appendix

### Sample Test Data Specifications
- **Small CSV**: 10 rows, 5 columns, 2 clear anomalies
- **Medium CSV**: 10,000 rows, 5 columns, 10% anomalies, mixed data types
- **JSON**: Nested structure with 5 records
- **Malformed CSV**: Various data quality issues for error testing

### Expected Test Durations
- **Full Test Suite**: 5-10 minutes (without performance tests)
- **Basic Tests Only**: 1-2 minutes
- **Performance Tests**: 2-5 minutes additional

### Resource Requirements
- **Disk Space**: ~100MB for test environment and data
- **Memory**: ~500MB for medium dataset processing
- **CPU**: Single core adequate, multi-core beneficial for performance tests
