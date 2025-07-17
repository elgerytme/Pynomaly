# Repository Rules

## Overview

This document establishes formal rules for the Pynomaly repository to prevent architectural drift and maintain clean code organization.

## Rule 1: Prohibited Folder Creation

### Rule Statement
**RULE**: Replacement folders and files MUST NOT be automatically created when original files/folders cannot be found.

### Prohibited Patterns

#### Folder Names
- `core/` - Generic "core" folders are prohibited anywhere in the repository
- `common/` - Use specific domain names instead
- `shared/` - Use specific shared functionality names
- `utils/` - Use specific utility names (e.g., `validation_utils/`)
- `helpers/` - Use specific helper names
- `lib/` - Use specific library names
- `misc/` - Miscellaneous folders are prohibited

#### File Names
- `core.py` - Use specific module names
- `common.py` - Use specific functionality names
- `utils.py` - Use specific utility names
- `helpers.py` - Use specific helper names
- `misc.py` - Miscellaneous files are prohibited

### Exceptions
The following locations are exempt from this rule:
- `node_modules/` - Third-party dependencies
- `venv/` and `.venv/` - Virtual environments
- Test fixtures that specifically test "core" functionality

### Enforcement

#### Pre-commit Validation
```bash
# Check for prohibited folder creation
if [[ -d "core" || -d "*/core" || -d "*/*/core" ]]; then
    echo "ERROR: 'core' folders are prohibited. Use specific domain names."
    exit 1
fi
```

#### CI/CD Pipeline Check
```bash
# Validate repository structure
python scripts/validate_repository_structure.py
```

## Rule 2: Proper Migration Process

### Rule Statement
**RULE**: When removing or relocating files/folders, ALL references MUST be updated before the removal.

### Process
1. **Identify Dependencies**: Find all references to the structure being removed
2. **Update References**: Update all imports, paths, and configuration
3. **Test Changes**: Verify all references work with new structure
4. **Remove Old Structure**: Only after all references are updated
5. **Validate**: Ensure no automatic recreation occurs

### Migration Checklist
- [ ] All Python imports updated
- [ ] All configuration files updated
- [ ] All documentation updated
- [ ] All scripts updated
- [ ] All test files updated
- [ ] All CI/CD pipelines updated

## Rule 3: Validation Requirements

### Rule Statement
**RULE**: All structural changes MUST be validated to prevent prohibited pattern creation.

### Validation Tools
1. **Repository Structure Validator**: `scripts/validate_repository_structure.py`
2. **Import Validator**: `scripts/validate_imports.py`
3. **Pre-commit Hooks**: `.pre-commit-config.yaml`

### Validation Frequency
- **Before Commit**: Pre-commit hooks
- **Before Merge**: CI/CD pipeline
- **Daily**: Automated repository health check
- **Before Release**: Manual validation

## Rule 4: Documentation Requirements

### Rule Statement
**RULE**: All structural changes MUST be documented with rationale and migration path.

### Documentation Requirements
- **Change Rationale**: Why the change is being made
- **Migration Path**: How to update existing code
- **Impact Assessment**: What will be affected
- **Rollback Plan**: How to revert if needed

## Rule 5: Monitoring and Detection

### Rule Statement
**RULE**: Automated monitoring MUST detect and alert on prohibited structure creation.

### Monitoring Implementation
- **Daily Scans**: Check for prohibited patterns
- **Alert System**: Notify maintainers of violations
- **Automatic Remediation**: Where possible, automatically fix violations
- **Reporting**: Monthly report on repository health

## Enforcement Mechanisms

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-repository-structure
        name: Validate Repository Structure
        entry: python scripts/validate_repository_structure.py
        language: python
        always_run: true
```

### CI/CD Integration
```yaml
# .github/workflows/structure-validation.yml
name: Repository Structure Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Repository Structure
        run: python scripts/validate_repository_structure.py
```

### GitHub Issue Templates
- **Structure Violation Report**: Template for reporting violations
- **Migration Request**: Template for requesting structural changes

## Violation Response

### Immediate Actions
1. **Stop the Process**: Halt any automated creation
2. **Assess Impact**: Determine what was created
3. **Remove Violations**: Delete prohibited structures
4. **Fix Root Cause**: Update references causing the creation
5. **Validate Fix**: Ensure issue won't recur

### Long-term Actions
1. **Update Documentation**: Improve prevention guidelines
2. **Enhance Validation**: Strengthen detection mechanisms
3. **Team Training**: Educate team on proper practices
4. **Process Improvement**: Refine development workflow

## Review and Updates

### Rule Review Process
- **Monthly**: Review rule effectiveness
- **Quarterly**: Update rules based on new patterns
- **Annual**: Comprehensive rule assessment

### Change Management
- All rule changes require team approval
- Changes must be documented and communicated
- Impact assessment required for rule modifications

## Compliance Tracking

### Metrics
- Number of violations detected
- Time to violation resolution
- Recurring violation patterns
- Team compliance rates

### Reporting
- Weekly violation summary
- Monthly compliance report
- Quarterly trend analysis

---

**Note**: These rules are living documents and will be updated as the repository evolves. All team members are responsible for understanding and following these rules.