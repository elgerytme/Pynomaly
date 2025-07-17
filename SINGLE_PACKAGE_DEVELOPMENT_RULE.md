# Single Package Development Rule

## Core Principle
**Only one package may be actively developed at a time** to maintain focus, reduce complexity, and ensure proper testing and validation.

## Rule Definition
1. **Single Package Focus**: All changes in a single commit or pull request must be confined to one package directory under `src/packages/`
2. **Package Boundary**: A package is defined as any directory directly under `src/packages/` (e.g., `src/packages/data/`, `src/packages/software/core/`)
3. **Exception Handling**: Breaking changes that require coordinated updates across packages are the only permitted exception

## Enforcement Mechanisms

### Automated Validation
- Pre-commit hooks validate package isolation
- CI/CD pipeline checks enforce single-package changes
- Automated blocking of multi-package commits

### Detection Rules
```python
def validate_single_package_development(changed_files):
    """Validate that changes are confined to a single package."""
    packages = set()
    for file_path in changed_files:
        if file_path.startswith('src/packages/'):
            # Extract package path (first 3 directory levels)
            parts = file_path.split('/')
            if len(parts) >= 3:
                package = '/'.join(parts[:3])  # src/packages/package_name
                packages.add(package)
    
    return len(packages) <= 1, packages
```

## Breaking Change Exceptions

### When Exceptions Are Allowed
1. **API Contract Changes**: Interface definitions that affect multiple packages
2. **Shared Schema Updates**: Database or data structure changes
3. **Security Updates**: Critical security patches requiring coordinated updates
4. **Major Version Releases**: Planned breaking changes across the ecosystem

### Exception Requirements
1. **Documentation**: Must include `BREAKING_CHANGE_JUSTIFICATION.md` in commit
2. **Planning**: Pre-approved by architecture review
3. **Minimal Scope**: Limit changes to absolute minimum required
4. **Coordinated Testing**: All affected packages must pass tests
5. **Rollback Plan**: Clear rollback strategy documented

### Exception Process
1. **Pre-Planning**: Create RFC (Request for Comments) document
2. **Impact Assessment**: Analyze all affected packages
3. **Approval**: Get architecture team approval
4. **Implementation**: Make minimal coordinated changes
5. **Validation**: Comprehensive testing of all affected packages
6. **Documentation**: Update all relevant documentation

## Implementation Details

### File Structure
```
src/packages/
├── data/anomaly_detection/     # Package A
├── software/core/             # Package B  
├── software/interfaces/       # Package C
└── formal_sciences/mathematics/ # Package D
```

### Validation Logic
- Track changed files by package directory
- Allow changes to root-level files (documentation, configuration)
- Block commits spanning multiple packages
- Provide clear error messages with guidance

### CI/CD Integration
```yaml
# .github/workflows/single-package-check.yml
name: Single Package Development Check
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check Single Package Rule
        run: python tools/validate_single_package_development.py
```

## Benefits
1. **Focused Development**: Clear scope and reduced complexity
2. **Better Testing**: Isolated changes are easier to test
3. **Cleaner History**: Git history shows clear package evolution
4. **Reduced Conflicts**: Fewer merge conflicts between teams
5. **Easier Debugging**: Issues are isolated to specific packages
6. **Simplified Reviews**: Code reviews are more focused and thorough

## Error Messages
- `ERROR: Changes span multiple packages: {package_list}`
- `INFO: To make breaking changes, create BREAKING_CHANGE_JUSTIFICATION.md`
- `HELP: Focus on one package at a time for better development workflow`

## Monitoring and Metrics
- Track exception frequency (should be rare)
- Monitor package development velocity
- Measure code quality improvements
- Track merge conflict reduction

---

This rule ensures disciplined development practices while providing necessary flexibility for legitimate cross-package updates.