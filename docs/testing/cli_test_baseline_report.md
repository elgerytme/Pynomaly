# CLI Test Baseline Report

## Executive Summary
This report documents the baseline state of CLI tests before converting Click-based modules to Typer.

**Date Generated:** 2025-07-08 09:05
**Python Version:** 3.11.4
**Pytest Version:** 8.4.0
**Total Tests Collected:** 26

## Test Results Summary

### Overall Statistics
- **Total Tests:** 26
- **Passed:** 18 (69.2%)
- **Failed:** 8 (30.8%)
- **Skipped:** 0 (0%)

### Failed Tests (Expected baseline failures)
1. `test_cli_simple.py::test_cli_module_structure` - Test structure issue
2. `test_cli_simple.py::test_typer_cli_structure` - Test structure issue  
3. `test_cli_simple.py::test_cli_help_generation` - Test structure issue
4. `test_cli_simple.py::test_data_loaders` - Test structure issue
5. `test_cli_simple.py::test_autonomous_service` - Test structure issue
6. `test_converted_commands.py::TestConvertedCommands::test_explainability_info` - Exit code assertion
7. `test_converted_commands.py::TestConvertedCommands::test_invalid_choice_options` - Exit code assertion
8. `test_converted_commands.py::test_cli_imports_successfully` - Test structure issue

### Passed Tests
- Multiple tests in `test_converted_commands.py` (18 passing tests)

## Click-Based Modules Identified
The following 12 Click-based modules were found and backed up:

1. `alert.py` (25,331 bytes)
2. `benchmarking.py` (30,917 bytes)  
3. `cost_optimization.py` (28,245 bytes)
4. `dashboard.py` (25,226 bytes)
5. `enhanced_automl.py` (23,566 bytes)
6. `ensemble.py` (24,337 bytes)
7. `explain.py` (33,223 bytes)
8. `governance.py` (36,554 bytes)
9. `quality.py` (17,572 bytes)
10. `security.py` (28,482 bytes)
11. `tenant.py` (22,222 bytes)
12. `training_automation_commands.py` (22,912 bytes)

**Total size of Click modules:** 318,585 bytes (~311 KB)

## Backup Status
✅ All Click-based modules successfully copied to `src/pynomaly/presentation/cli/_click_backup/`

## Test Environment Details
- Platform: Windows 32-bit
- Root Directory: C:\Users\andre\Pynomaly\tests
- Pytest plugins: Faker, anyio, hypothesis, asyncio, bdd, benchmark, cov, mock, timeout, xdist
- Configuration: pytest.ini with comprehensive settings

## Warnings Observed
- 17 Pydantic deprecation warnings (class-based config, json_encoders, min_items/max_items, extra Field kwargs)
- SHAP/LIME dependencies not available
- TensorFlow oneDNN optimization notices

## Notes for Post-Conversion Comparison
This baseline should be compared against test results after Click-to-Typer conversion to ensure:
1. No regression in passing tests
2. Improved test structure for currently failing tests
3. Maintained compatibility with existing CLI functionality

## Risk Assessment
- **Low Risk:** Core test functionality is working (69% pass rate)
- **Medium Risk:** Some test structure issues need attention
- **Mitigation:** Full backup of all Click modules completed

---
*This report serves as the official baseline for the Click-to-Typer conversion project.*
