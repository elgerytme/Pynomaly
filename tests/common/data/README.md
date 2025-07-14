# Test Data Directory

This directory contains **minimal sample data files** needed for test assertions.

## Guidelines

- **Keep only essential data files** required for running tests
- **Data files should be small** and focused on specific test scenarios
- **No test artifacts** (reports, screenshots, coverage files) should be stored here
- **Use meaningful names** that clearly indicate the test purpose

## Current Files

- `small_data.csv` - Sample CSV data for CLI validation tests

## Test Artifacts Location

- **Reports**: All test reports, coverage files, and text outputs are stored in `/reports/`
- **Artifacts**: Screenshots, images, and other test artifacts are stored in `/artifacts/`

## Adding New Test Data

When adding new test data files:
1. Ensure they are minimal and necessary for test assertions
2. Place them in appropriate subdirectories if needed
3. Update this README to document the new files
4. Keep file sizes small to avoid bloating the repository
