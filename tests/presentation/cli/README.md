# CLI Testing Infrastructure

## Overview

This directory contains comprehensive tests for the Pynomaly CLI interface, dramatically expanding the testing coverage from 5 simple tests to over 3,700 lines of comprehensive test coverage across all CLI modules.

## Test Files

### 1. `test_cli_comprehensive.py` (1,041 lines)
**Main CLI application testing suite**
- Tests main app commands: version, status, config, quickstart
- Configuration file generation (test, experiment, autonomous modes)
- Help system completeness verification
- Integration workflows
- Error handling and edge cases

### 2. `test_autonomous_cli.py` (689 lines)
**Autonomous anomaly detection CLI testing**
- Data profiling commands and verbose output
- Quick detection with various parameters
- Full autonomous detection workflows
- Configuration file support
- Performance monitoring and error handling
- Multiple data format support (CSV, JSON, Parquet)

### 3. `test_preprocessing_cli.py` (836 lines)
**Data preprocessing CLI testing**
- Data cleaning operations (missing values, outliers, duplicates)
- Data transformation (scaling, encoding, feature selection)
- Pipeline creation and management
- Data validation and analysis
- Feature engineering capabilities
- Integration workflows

### 4. `test_export_cli.py` (766 lines)
**Export functionality CLI testing**
- Multiple export formats (Excel, Power BI, Tableau, CSV, JSON)
- Business intelligence platform integration
- Batch export operations
- Report generation (HTML, PDF)
- Database export capabilities
- Configuration-driven export workflows

### 5. `test_server_cli.py` (610 lines)
**Server management CLI testing**
- Server lifecycle (start, stop, restart, status)
- Configuration management and validation
- Health monitoring and metrics
- Log management and analysis
- SSL/TLS configuration
- Development vs production modes

## Testing Coverage

### Command Categories Tested
- **Main App Commands**: version, status, config, generate-config, quickstart
- **Detector Management**: create, list, show, delete, algorithms, tune
- **Dataset Operations**: load, list, show, describe, export, validate
- **Detection Workflows**: train, run, evaluate, batch, results
- **Autonomous Detection**: detect, profile, quick
- **Data Preprocessing**: clean, transform, pipeline, validate, analyze, features
- **Export Operations**: excel, powerbi, tableau, csv, json, database, batch, report
- **Server Management**: start, stop, restart, status, health, logs, config, metrics

### Test Types Included
- **Unit Tests**: Individual command functionality
- **Integration Tests**: Complete workflow testing
- **Error Handling**: Invalid inputs, service failures, permission errors
- **Configuration Tests**: File-based configuration, validation
- **Mock Testing**: External service interactions
- **Edge Cases**: Large datasets, timeouts, resource limits

### Key Testing Features
- **Comprehensive Mocking**: All external dependencies properly mocked
- **File Operations**: Temporary file handling for safe testing
- **CLI Interaction**: User input simulation and response validation
- **Service Integration**: Mock services for autonomous detection, preprocessing, export
- **Error Scenarios**: Network failures, permission issues, invalid inputs
- **Performance Testing**: Large dataset handling, timeout scenarios

## Usage

### Running All CLI Tests
```bash
pytest tests/presentation/cli/ -v
```

### Running Specific Test Files
```bash
# Main CLI app tests
pytest tests/presentation/cli/test_cli_comprehensive.py -v

# Autonomous detection tests
pytest tests/presentation/cli/test_autonomous_cli.py -v

# Preprocessing tests
pytest tests/presentation/cli/test_preprocessing_cli.py -v

# Export functionality tests
pytest tests/presentation/cli/test_export_cli.py -v

# Server management tests
pytest tests/presentation/cli/test_server_cli.py -v
```

### Running Specific Test Classes
```bash
# Test main CLI app functionality
pytest tests/presentation/cli/test_cli_comprehensive.py::TestMainCLIApp -v

# Test detector CLI commands
pytest tests/presentation/cli/test_cli_comprehensive.py::TestDetectorsCLI -v

# Test autonomous detection
pytest tests/presentation/cli/test_autonomous_cli.py::TestAutonomousCLI -v
```

## Test Architecture

### Fixtures
- **`runner`**: Typer CLI test runner for command execution
- **`mock_container`**: Dependency injection container mocking
- **`mock_services`**: Service layer mocking (preprocessing, autonomous, export)
- **`sample_data_files`**: Temporary test data files (CSV, JSON, Parquet)
- **`mock_external_apis`**: External service mocking (Power BI, Tableau)

### Mocking Strategy
- **Service Layer**: All application services properly mocked
- **External APIs**: Business intelligence platforms mocked
- **File System**: Safe temporary file operations
- **Network Calls**: HTTP requests/responses mocked
- **Database**: Repository patterns mocked

### Test Organization
- **Happy Path**: Normal operation scenarios
- **Error Cases**: Exception handling and error messages
- **Edge Cases**: Boundary conditions and limits
- **Integration**: End-to-end workflow testing
- **Configuration**: File-based and CLI-based configuration

## Benefits Achieved

### Coverage Improvement
- **From**: 5 basic CLI structure tests
- **To**: 3,700+ lines comprehensive CLI testing
- **Coverage**: All CLI commands, options, and workflows
- **Quality**: Production-ready error handling testing

### Testing Quality
- **Comprehensive**: Every CLI command and option tested
- **Realistic**: Mock realistic service interactions
- **Robust**: Error handling and edge cases covered
- **Maintainable**: Clear test organization and documentation

### Development Support
- **Regression Prevention**: Comprehensive test coverage prevents CLI regressions
- **Documentation**: Tests serve as usage documentation
- **Refactoring Safety**: Extensive mocking enables safe refactoring
- **CI/CD Ready**: All tests designed for automated execution

## Future Enhancements

### Additional Test Coverage
- Performance benchmarking tests
- Load testing for large datasets
- Cross-platform compatibility tests
- Internationalization testing

### Test Infrastructure
- Property-based testing with Hypothesis
- Mutation testing for test quality
- Performance regression testing
- Visual regression testing for CLI output

This comprehensive CLI testing infrastructure ensures the Pynomaly CLI is robust, reliable, and thoroughly tested across all functionality areas.