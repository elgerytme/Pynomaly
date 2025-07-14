# Enhanced Validation System

The Enhanced Validation System provides developer-friendly UX enhancements to the existing validation infrastructure, focusing on three key areas:

1. **Colorised console output (rich) grouping violations by severity**
2. **GitHub PR comments with violation summaries and "How to fix" snippets**
3. **Pre-commit hook integration with developer reminders**

## Features

### 🎨 Rich Console Output

The validation system uses the Rich library to provide colorized, well-formatted console output that groups violations by severity level:

- **CRITICAL** (Red): Security issues, critical structural problems
- **HIGH** (Red): Important issues that should be fixed before merge
- **MEDIUM** (Yellow): Code quality issues that should be addressed
- **LOW** (Blue): Minor issues, TODOs, and improvements
- **INFO** (Green): Informational messages and suggestions

```bash
# Run validation with rich output
pynomaly validate run --format rich
```

### 📝 GitHub Integration

When running in CI/CD environments, the system can automatically post comments to GitHub pull requests with:

- Summary of validation results
- First 10 violations with detailed information
- Rule IDs and fix suggestions
- Pre-commit installation reminders

```bash
# Enable GitHub comment posting
pynomaly validate run --github-comment

# Or automatically in CI when environment variables are set
export GITHUB_TOKEN="your-token"
export GITHUB_REPOSITORY="owner/repo"
export GITHUB_PR_NUMBER="123"
```

### 🔧 Pre-commit Integration

The system integrates with pre-commit hooks to provide:

- Automatic validation before commits
- Developer-friendly reminders when validation fails
- Easy installation commands
- Rich output in terminal

```bash
# Check pre-commit status
pynomaly validate check-pre-commit

# Install pre-commit hooks
pynomaly validate install-hooks
```

## Installation

### Prerequisites

```bash
# Install required dependencies
pip install typer[all] rich requests

# Or install with the CLI extra
pip install pynomaly[cli]
```

### Pre-commit Setup

Add to your `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: enhanced-validation
      name: Enhanced Validation with Rich Output
      entry: python -m pynomaly.presentation.cli.app validate run --format rich
      language: python
      pass_filenames: false
      always_run: true
      stages: [pre-commit]
      verbose: true
```

Then install the hooks:

```bash
pre-commit install
```

## Usage

### Command Line Interface

```bash
# Basic validation
pynomaly validate run

# Rich output format (default)
pynomaly validate run --format rich

# JSON output format
pynomaly validate run --format json

# Save report to file
pynomaly validate run --save validation-report.json

# GitHub comment mode
pynomaly validate run --github-comment

# Validate specific path
pynomaly validate run /path/to/project

# Check pre-commit status
pynomaly validate check-pre-commit

# Install pre-commit hooks
pynomaly validate install-hooks
```

### GitHub Actions Integration

Add to your `.github/workflows/validation.yml`:

```yaml
- name: Run enhanced validation
  run: |
    export GITHUB_TOKEN="${{ secrets.GITHUB_TOKEN }}"
    export GITHUB_REPOSITORY="${{ github.repository }}"
    export GITHUB_PR_NUMBER="${{ github.event.number }}"
    
    python -m pynomaly.presentation.cli.app validate run \
      --github-comment \
      --format rich \
      --save validation-report.json
```

### Python API

```python
from pynomaly.presentation.cli.validation import (
    EnhancedValidator,
    RichOutputFormatter,
    GitHubCommentGenerator,
)
from rich.console import Console

# Create validator
validator = EnhancedValidator("/path/to/project")

# Run validation
result = validator.validate_project()

# Display with rich formatting
console = Console()
formatter = RichOutputFormatter(console)
formatter.display_results(result)

# Generate GitHub comment
comment_generator = GitHubCommentGenerator()
comment = comment_generator.generate_comment(result)

# Post to GitHub (requires environment variables)
success = comment_generator.post_to_github(comment)
```

## Validation Rules

### Structure Validation (STRUCT_*)

- **STRUCT_001**: Forbidden directories at root level
- **STRUCT_002**: Missing essential directories

### Code Quality Validation (CODE_*)

- **CODE_001**: TODO/FIXME comments
- **CODE_002**: Debug print statements
- **CODE_003**: File reading/encoding errors

### Documentation Validation (DOCS_*)

- **DOCS_001**: Missing documentation directory
- **DOCS_002**: Missing root README.md

### Security Validation (SEC_*)

- **SEC_001**: Dangerous function usage (eval, exec, os.system, etc.)

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GITHUB_TOKEN` | GitHub personal access token | For PR comments |
| `GITHUB_REPOSITORY` | Repository name (owner/repo) | For PR comments |
| `GITHUB_PR_NUMBER` | Pull request number | For PR comments |
| `CI` | CI environment indicator | Auto-detected |

### Severity Levels

```python
class ViolationSeverity(str, Enum):
    CRITICAL = "critical"  # Blocks merge, security issues
    HIGH = "high"          # Important issues
    MEDIUM = "medium"      # Code quality issues
    LOW = "low"            # Minor issues, TODOs
    INFO = "info"          # Informational
```

## GitHub Comment Example

```markdown
## ❌ Validation Failed

### 📊 Summary
| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 2 | ❌ |
| HIGH | 1 | ❌ |
| MEDIUM | 3 | ⚠️ |
| LOW | 5 | ⚠️ |

### 🔍 Top 10 Violations

#### 1. CRITICAL: Security issue: Use of eval() is dangerous
**File:** `src/insecure.py`
**Line:** 15
**Rule:** SEC_001
**How to fix:** Replace eval() with secure alternative like ast.literal_eval()

#### 2. HIGH: Forbidden directory found: build/
**File:** `build`
**Rule:** STRUCT_001
**How to fix:** Remove or move build/ to appropriate location (e.g., artifacts/, reports/)

### 🔧 Quick Fix
To avoid validation failures in the future, install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Please fix these issues before merging. 🙏
```

## Development

### Running Tests

```bash
# Run enhanced validation tests
pytest tests/integration/test_enhanced_validation.py -v

# Run with coverage
pytest tests/integration/test_enhanced_validation.py --cov=src/pynomaly/presentation/cli/validation
```

### Demo Script

```bash
# Run the interactive demo
python examples/enhanced_validation_demo.py
```

The demo showcases:
- Rich console output with colorized violations
- GitHub comment generation
- Pre-commit integration
- CLI usage examples

### Development Setup

```bash
# Install development dependencies
pip install -e ".[cli,dev]"

# Install pre-commit hooks
pre-commit install

# Run validation locally
pynomaly validate run --format rich
```

## Troubleshooting

### Common Issues

1. **Rich output not displaying colors**
   - Ensure terminal supports ANSI colors
   - Check `TERM` environment variable
   - Use `--format json` as fallback

2. **GitHub comments not posting**
   - Verify environment variables are set
   - Check GitHub token permissions
   - Ensure `requests` library is installed

3. **Pre-commit hooks not running**
   - Run `pre-commit install` in repository root
   - Check `.pre-commit-config.yaml` syntax
   - Verify Python environment has required dependencies

### Debug Mode

```bash
# Enable verbose output
pynomaly validate run --format rich --verbose

# Save detailed report
pynomaly validate run --save debug-report.json

# Check pre-commit status
pynomaly validate check-pre-commit
```

## Contributing

To contribute to the enhanced validation system:

1. Follow the existing code style and patterns
2. Add tests for new validation rules
3. Update documentation for new features
4. Test with the demo script
5. Ensure GitHub Actions integration works

### Adding New Validation Rules

```python
def _validate_new_rule(self):
    """Validate a new rule."""
    # Implementation
    violation = ValidationViolation(
        message="Rule violation message",
        severity=ViolationSeverity.MEDIUM,
        file_path="path/to/file.py",
        line_number=42,
        rule_id="NEW_001",
        fix_suggestion="How to fix this issue"
    )
    self.result.add_violation(violation)
```

## Related Documentation

- [Pre-commit Configuration](../contributing/pre-commit-setup.md)
- [CI/CD Integration](../cicd/github-actions.md)
- [Quality Gates](../quality/quality-gates.md)
- [Developer Setup](../contributing/development-setup.md)

## License

This enhanced validation system is part of the Pynomaly project and is licensed under the MIT License.
