# Documentation Package

This package contains all documentation, guides, and educational materials for Pynomaly.

## Contents

- `docs/` - Main documentation
- `docs-consolidated/` - Consolidated documentation
- `reports/` - Analysis and status reports
- `training/` - Training materials and workshops
- `migration-backup/` - Historical migration documentation

## Structure

```
documentation/
├── docs/           # Main documentation
├── tests/          # Documentation tests
├── build/          # Build artifacts
├── deploy/         # Deployment configurations
├── scripts/        # Documentation scripts
└── src/            # Source code
    └── documentation/
        └── generators/  # Documentation generators
```

## Usage

The documentation is built using MkDocs and can be served locally for development.

```bash
# Install dependencies
pip install -e .

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```