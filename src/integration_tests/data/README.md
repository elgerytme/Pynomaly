# Test Data Directory

This directory contains static test fixtures used by the Pynomaly test suite.

## Structure

- `sample/` - Large binary samples and test data files
- Other subdirectories for specific test data types as needed

## Guidelines

- Keep test data files small when possible
- Use realistic but anonymized data for fixtures
- Large binary files should be placed in the `sample/` subdirectory
- Consider using data generators for complex test scenarios instead of static files where appropriate

## Usage

Test fixtures in this directory can be accessed using relative paths from test files:

```python
import os
from pathlib import Path

# Get test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# Access sample data
sample_file = TEST_DATA_DIR / "sample" / "example.bin"
```
