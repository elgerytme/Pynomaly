# Deployment Scripts

This directory contains scripts for CI/CD pipeline automation and development workflow enhancement.

## Scripts Overview

### Domain Boundary Violation Detection

**`boundary-violation-check.py`** - Automated detection of domain boundary violations

- **Purpose**: Ensures clean architecture by detecting cross-domain imports that violate domain boundaries
- **Usage**: Can be run locally or in CI/CD pipeline
- **Features**:
  - AST-based Python import analysis
  - Configurable domain boundaries and allowed imports
  - Multiple output formats (console, JSON, GitHub annotations)
  - Severity-based violation classification
  - Detailed reporting with file locations

**Example Usage**:
```bash
# Check all packages for violations
python boundary-violation-check.py src/packages --format console --fail-on-critical

# Generate JSON report
python boundary-violation-check.py src/packages --format json --output violations.json

# CI/CD usage with GitHub annotations
python boundary-violation-check.py src/packages --format github --fail-on-violations
```

### Pre-commit Quality Checks

**`pre-commit-checks.py`** - Comprehensive pre-commit hook for code quality

- **Purpose**: Prevents problematic code from being committed
- **Features**:
  - Domain boundary violation detection
  - Import statement standards validation
  - Security issue scanning
  - Code quality metrics
  - Git hook installation

**Example Usage**:
```bash
# Install pre-commit hook
python pre-commit-checks.py --install

# Run checks on staged files
python pre-commit-checks.py --staged

# Check specific files
python pre-commit-checks.py --files src/packages/data/quality/file.py
```

## CI/CD Integration

### GitHub Actions Workflow

The **`.github/workflows/boundary-check.yml`** workflow automatically:

1. **Runs on every push/PR** to main/develop branches
2. **Detects violations** using the boundary check script  
3. **Fails the build** if critical violations are found
4. **Comments on PRs** with violation details
5. **Uploads reports** as artifacts for detailed analysis
6. **Provides guidance** on how to fix violations

### Workflow Features

- **Automated Detection**: No manual intervention required
- **PR Comments**: Detailed violation reports posted as PR comments
- **Artifact Upload**: JSON reports saved for 30 days
- **Smart Failure**: Only fails on critical violations by default
- **GitHub Annotations**: Violations shown inline in GitHub UI

## Configuration

### Domain Boundary Configuration

The boundary checker uses a configuration that defines:

```python
DOMAIN_CONFIG = {
    "domains": {
        "data": ["anomaly_detection", "data_quality", "data_science", "quality"],
        "ai": ["machine_learning", "mlops", "nlp-advanced", "computer-vision"],
        "enterprise": ["security", "enterprise_auth", "governance"],
        # ... more domains
    },
    "allowed_cross_domain_imports": {
        "shared": ["*"],  # Shared packages can be imported by anyone
        "interfaces": ["*"],
        "data_quality": ["shared", "interfaces", "infrastructure"],
        # ... specific allowances
    }
}
```

### Violation Types

1. **Direct Cross-Domain** (High severity)
   - Direct imports from other domain packages
   - Example: `from src.packages.ai.machine_learning import Model`

2. **Relative Cross-Domain** (Critical severity)  
   - Relative imports that cross domain boundaries
   - Example: `from ....ai.machine_learning import Model`

3. **Suspicious Relative** (Medium severity)
   - Deep relative imports (3+ levels up)
   - May indicate architectural issues

### Security Checks

The pre-commit checker scans for:

- **Hardcoded Credentials**: Passwords, API keys, tokens, secrets
- **Dangerous Functions**: `eval()`, `exec()`, `os.system()`
- **Shell Injection**: Subprocess calls with `shell=True`
- **Code Quality Issues**: Long files, long lines, excessive TODOs

## Best Practices

### For Developers

1. **Run Pre-commit Checks Locally**:
   ```bash
   python src/packages/deployment/scripts/pre-commit-checks.py --staged
   ```

2. **Install Git Hook** (Recommended):
   ```bash
   python src/packages/deployment/scripts/pre-commit-checks.py --install
   ```

3. **Use Cross-Domain Integration Patterns**:
   - Import from `shared.integration` for cross-domain communication
   - Use domain adapters for data transformation
   - Publish/subscribe to domain events instead of direct imports

4. **Fix Violations Immediately**:
   - Don't bypass checks unless absolutely necessary
   - Refactor code to use proper integration patterns
   - Consult architecture documentation for guidance

### For CI/CD Pipeline

1. **Fail Fast**: Boundary checks run early in the pipeline
2. **Detailed Feedback**: Violations are reported with precise locations
3. **Artifact Preservation**: Reports are saved for later analysis
4. **Progressive Enhancement**: Start with warnings, move to failures

## Integration with Development Workflow

### Local Development

```bash
# Before committing
git add .
python src/packages/deployment/scripts/pre-commit-checks.py --staged

# If violations found, fix them
# ... make fixes ...

# Commit when clean
git commit -m "Your commit message"
```

### CI/CD Pipeline

The boundary check automatically runs on:

- **Push to main/develop**: Ensures main branches stay clean
- **Pull Requests**: Prevents violations from being merged
- **Manual Triggers**: Can be run on-demand via workflow_dispatch

### Failure Handling

If violations are detected:

1. **Build Fails**: Pipeline stops with clear error message
2. **PR Comment**: Detailed violation report posted to PR
3. **Artifact Upload**: Full JSON report available for download
4. **Guidance Provided**: Links to documentation and examples

## Troubleshooting

### Common Issues

1. **False Positives**: Update `allowed_cross_domain_imports` configuration
2. **Performance**: Use `--exclude` patterns for large repositories
3. **Configuration**: Check domain mapping in `DOMAIN_CONFIG`

### Configuration Updates

To modify domain boundaries:

1. Update `DOMAIN_CONFIG` in `boundary-violation-check.py`
2. Test changes locally before committing
3. Update documentation to reflect changes

### Script Dependencies

- **Python 3.11+**: Required for AST parsing features
- **ast module**: Built-in Python module for code analysis
- **pathlib**: For path manipulation
- **json**: For report generation

## Architecture Benefits

This automated boundary checking provides:

- **Clean Architecture Enforcement**: Prevents architectural drift
- **Early Problem Detection**: Catches issues before they spread
- **Developer Education**: Teaches proper patterns through feedback
- **Continuous Compliance**: Ensures standards are maintained
- **Technical Debt Prevention**: Stops accumulation of violations

The scripts support the overall goal of maintaining a well-structured, domain-driven monorepo where each package has clear boundaries and responsibilities.