# Repository Automation Scripts

This directory contains repository-level automation scripts for governance, analysis, and maintenance. These scripts operate on the entire monorepo structure and enforce architectural standards.

## Script Categories

### Repository Governance
- **`repository_governance/`** - Repository structure validation and enforcement
- **`domain_boundary_validator.py`** - Validates domain boundary rules and dependencies
- **`domain_governance.py`** - Comprehensive governance checks and enforcement
- **`package_independence_validator.py`** - Ensures package independence within domains

### Development Standards
- **`best_practices_framework/`** - Code quality and standards enforcement
- **`create_domain_package.py`** - Template-based package creation with proper structure
- **`standardize_package_structure.py`** - Standardizes existing packages to current structure

### Analysis and Monitoring
- **`comprehensive_analysis/`** - Repository health and metrics analysis
- **`performance_validation.py`** - Performance testing and validation
- **`validate_architecture.py`** - Architecture compliance checking

### Migration and Maintenance
- **`migrate_package_structure.py`** - Automated package structure migrations
- **`automated_domain_fixes.py`** - Automated fixes for common domain issues
- **`targeted_migration.py`** - Selective migration utilities

## Key Scripts

### Domain Governance
```bash
# Validate all domain boundaries and dependencies
python scripts/domain_boundary_validator.py

# Run comprehensive governance checks
python scripts/domain_governance.py --check-all

# Validate package independence 
python scripts/package_independence_validator.py
```

### Repository Structure
```bash
# Validate repository structure
python scripts/repository_governance/validate_repository_structure.py

# Create new domain package
python scripts/create_domain_package.py --domain ai --name new_package

# Standardize package structure
python scripts/standardize_package_structure.py --package-path src/packages/ai/example
```

### Analysis and Quality
```bash
# Comprehensive repository analysis
python scripts/comprehensive_analysis/comprehensive_analysis.py

# Performance validation
python scripts/performance_validation.py

# Architecture compliance check
python scripts/validate_architecture.py
```

## Usage Guidelines

### Running Scripts

Most scripts can be run from the repository root:

```bash
# From repository root
python scripts/script_name.py [options]

# With specific parameters
python scripts/domain_governance.py --domain ai --check-boundaries
python scripts/create_domain_package.py --domain data --name analytics_engine
```

### Script Dependencies

Scripts use standard library and repository-specific utilities. No external dependencies beyond what's in `requirements-prod.txt`.

### Output and Logging

- Scripts output results to console with structured logging
- Validation scripts return exit codes (0 = success, 1 = validation failures)  
- Analysis scripts may generate reports in `reports/` directory

## Script Development

### Creating New Scripts

1. **Follow naming conventions**: Use descriptive names with underscores
2. **Include proper documentation**: Docstrings and command-line help
3. **Use standard structure**: Argument parsing, logging, error handling
4. **Test thoroughly**: Validate on actual repository structure
5. **Update this README**: Document new scripts and their purpose

### Script Template

```python
#!/usr/bin/env python3
"""
Script Description

Detailed description of what this script does and when to use it.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--option", help="Option description")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Script logic here
        logger.info("Script execution completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## Important Notes

- **Repository Level Only**: These scripts operate on the entire repository, not individual packages
- **No Package-Level Scripts**: Individual package scripts should be in the package directory
- **Governance Enforcement**: Many scripts enforce architectural rules and standards
- **Safe to Run**: Scripts are designed to be safe and include dry-run options where applicable

## Contributing

When adding new scripts:

1. Ensure they follow the established patterns
2. Include comprehensive documentation
3. Test on the actual repository structure
4. Update this README with usage instructions
5. Consider adding to CI/CD pipeline if appropriate
