# Report Organization Standards

This document defines the standardized approach for organizing reports within the Monorepo repository.

## Overview

Reports are classified into two categories based on their scope:

1. **Repository/Project Level Reports**: Comprehensive reports that cover the entire repository or multiple packages
2. **Package Level Reports**: Reports specific to individual packages or their components

## Directory Structure

### Repository Level Reports
All repository or project-level reports must be placed in:
```
@docs/reports/
```

**Examples of repository-level reports:**
- Architecture summaries spanning multiple packages
- Cross-package dependency analysis
- Repository-wide migration reports
- Overall project completion status
- Comprehensive testing framework summaries
- Domain reorganization reports
- Technical debt analysis

### Package Level Reports
All package-specific reports must be placed in:
```
@<package_path>/docs/reports/
```

**Examples of package-level reports:**
- Package-specific performance metrics
- Individual package test results
- Package migration status
- Package architecture documentation
- Component-specific analysis reports

## File Naming Conventions

### Repository Level Reports
- Use UPPERCASE with underscores: `REPOSITORY_ANALYSIS_REPORT.md`
- Include date for recurring reports: `WEEKLY_BUILD_REPORT_2025_01_21.md`
- Use descriptive, meaningful names that indicate scope

### Package Level Reports
- Use lowercase with underscores: `performance_analysis_report.md`
- Include package name when beneficial: `data_science_migration_report.md`
- Maintain consistency within each package

## Supported Report Formats

The following formats are supported for reports:
- `.md` (Markdown) - **Preferred for documentation**
- `.txt` (Plain text) - For simple reports
- `.json` (JSON) - For structured data reports
- `.html` (HTML) - For rich formatting when needed
- `.pdf` (PDF) - For formal reports requiring fixed formatting

## Validation Rules

### Automated Enforcement
The repository includes automated validation through:

1. **Pre-commit Hook**: Validates report placement on every commit
2. **Buck2 Build Rule**: Integrates validation into the build system
3. **CI/CD Pipeline**: Continuous validation in build processes

### Manual Validation
Run the validation script manually:
```bash
python tools/scripts/validate_report_locations.py --scan-all
```

## Examples

### Correct Placement

✅ **Repository Level:**
```
docs/reports/COMPREHENSIVE_ARCHITECTURE_SUMMARY.md
docs/reports/DOMAIN_CLEANUP_SUMMARY.md
docs/reports/technical_debt_analysis.md
```

✅ **Package Level:**
```
src/packages/data/data_science/docs/reports/performance_metrics.md
src/packages/ai/machine_learning/docs/reports/model_evaluation.json
src/packages/enterprise/auth/docs/reports/security_audit.md
```

### Incorrect Placement

❌ **Package reports in repository folder:**
```
docs/reports/data_science_performance.md  # Should be in package folder
```

❌ **Repository reports in package folder:**
```
src/packages/data/analytics/docs/reports/REPOSITORY_ANALYSIS.md  # Should be in docs/reports/
```

## Migration Guide

If you have existing reports in incorrect locations:

1. **Identify misplaced reports:**
   ```bash
   python tools/scripts/validate_report_locations.py --scan-all --fix
   ```

2. **Move reports to correct locations:**
   ```bash
   # Example: Move package-specific report
   mv docs/reports/package_specific_report.md src/packages/domain/package/docs/reports/
   
   # Example: Move repository-level report
   mv src/packages/some/package/repo_wide_report.md docs/reports/
   ```

3. **Update any references:**
   - Search for file references in documentation
   - Update links in README files
   - Modify any automation that depends on file paths

## Integration with Build System

### Buck2 Integration
The report validation is integrated into Buck2 builds:

```python
# In your BUCK file
load("//tools/buck:report_validation.bzl", "create_report_validation_rule")

create_report_validation_rule(
    name = "validate-reports",
    report_files = glob(["docs/reports/**/*"]),
    package_docs_paths = ["src/packages/*/docs/reports/"],
)
```

### Pre-commit Hook
The validation runs automatically on every commit:

```yaml
- id: validate-report-locations
  name: Validate Report Locations  
  entry: python tools/scripts/validate_report_locations.py
  args: [--scan-all]
  always_run: true
```

## Troubleshooting

### Common Issues

1. **"Report should be in docs/reports/"**
   - The report appears to be repository-wide but is in a package folder
   - Move to `docs/reports/` if it covers multiple packages or the entire project

2. **"Report should be in src/packages/{domain}/{package}/docs/reports/"**
   - The report is package-specific but placed in the repository reports folder
   - Move to the appropriate package's docs/reports folder

3. **False Positives**
   - If the validator incorrectly identifies a file as a report, review the detection logic
   - Consider renaming the file if it's not actually a report

### Getting Help

For questions or issues with report organization:

1. Check this documentation first
2. Review the validation script: `tools/scripts/validate_report_locations.py`
3. Look at existing correctly-placed reports for examples
4. Create an issue if you encounter systematic problems

## Maintenance

This standard should be reviewed and updated as the repository evolves:

- **Quarterly Review**: Assess if the standards meet current needs
- **Tool Updates**: Keep validation scripts current with repository changes
- **Documentation Updates**: Maintain accuracy as the project structure changes

---

**Last Updated:** 2025-01-22  
**Version:** 1.0  
**Maintained By:** Repository Governance Team