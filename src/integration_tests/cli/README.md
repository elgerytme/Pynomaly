# Pynomaly CLI Testing Suite

## Quick Start

### Bash/Linux/macOS Testing
```bash
cd tests/cli
chmod +x test_cli_bash.sh
./test_cli_bash.sh
```

### PowerShell/Windows Testing
```powershell
cd tests\cli
.\test_cli_powershell.ps1
```

## Files in this Directory

| File | Description |
|------|-------------|
| `CLI_TESTING_PLAN.md` | Comprehensive testing strategy and objectives |
| `CLI_TESTING_PROCEDURES.md` | Detailed testing procedures and troubleshooting |
| `test_cli_bash.sh` | Automated test script for Bash environments |
| `test_cli_powershell.ps1` | Automated test script for PowerShell environments |
| `README.md` | This file - quick reference guide |

## Test Coverage

### Core Functionality
- ✅ CLI Help and Version Information
- ✅ Configuration Management
- ✅ System Status Reporting
- ✅ Dataset Loading and Validation
- ✅ Detector Creation and Management
- ✅ Detection Training and Execution
- ✅ Autonomous Mode Operations
- ✅ Results Export Functionality
- ✅ Error Handling and Recovery

### Supported Data Formats
- ✅ CSV files (standard and malformed)
- ✅ JSON/JSONL files
- ✅ Excel files (if dependencies available)
- ✅ Parquet files (if dependencies available)

### Cross-Platform Testing
- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS (10.15+)
- ✅ Windows 10+
- ✅ WSL2
- ✅ PowerShell Core 7.0+

## Quick Reference

### Test Categories
1. **Basic Commands** - Core CLI functionality
2. **Dataset Management** - Data loading and validation
3. **Detector Management** - Algorithm configuration
4. **Detection Workflow** - End-to-end detection process
5. **Autonomous Mode** - AI-powered automation
6. **Export Functions** - Results export capabilities
7. **Performance Tests** - Timing and resource validation
8. **Error Handling** - Graceful failure management

### Expected Results
- **Total Tests**: ~45-50 individual test cases
- **Expected Duration**: 5-10 minutes full suite
- **Success Rate**: >95% in properly configured environments
- **Performance**: <60 seconds for medium datasets (10K rows)

## Troubleshooting Quick Fixes

### Common Issues
```bash
# Python not found
which python3  # Verify Python installation

# Permission denied
chmod +x test_cli_bash.sh

# Virtual environment issues
python3 -m venv --help  # Verify venv availability

# Package installation issues
pip install --upgrade pip
```

### PowerShell Issues
```powershell
# Execution policy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Python not found
python --version  # Verify Python in PATH

# Module import issues
$env:PYTHONPATH = "C:\path\to\pynomaly\src"
```

## Support

For detailed troubleshooting and procedures, see:
- `CLI_TESTING_PROCEDURES.md` - Comprehensive documentation
- `CLI_TESTING_PLAN.md` - Testing strategy and requirements

For issues with the CLI itself, check:
- Main project documentation
- GitHub issues
- CLI command help: `pynomaly --help`
