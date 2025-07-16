# Test Directory Structure

This project organizes test files using a pytest-friendly hierarchy as follows:

```
tests/
  unit/
  integration/
  e2e/
  performance/
  ui/
  benchmarks/          # optional micro-benchmarks
  common/              # shared data & helper modules
reports/               # HTML/coverage/perf reports (git-ignored)
artifacts/             # screenshots, DB dumps, etc. (git-ignored)
```

- **tests/unit/**: Unit tests for individual components.
- **tests/integration/**: Integration tests to ensure components work together.
- **tests/e2e/**: End-to-end tests simulating real user scenarios.
- **tests/performance/**: Performance tests to assess speed and scalability.
- **tests/ui/**: User interface tests.
- **tests/benchmarks/**: Micro-benchmark tests to measure performance of small code sections.
- **tests/common/**: Shared data and helper modules used across tests.
- **reports/**: Stores HTML/coverage/performance reports. This directory is git-ignored.
- **artifacts/**: Contains screenshots, DB dumps, and other artifacts. This directory is git-ignored.

The directories for reports and artifacts are not version-controlled to keep the repository clean and focused on source code.
