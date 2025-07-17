# Repository Governance System

A comprehensive automated system for monitoring and maintaining repository quality, architecture compliance, and organizational standards.

## Overview

The Repository Governance System provides:

- **Automated Quality Checks**: Monitors tidiness, organization, and architecture compliance
- **Intelligent Fixes**: Automatically resolves common issues while preserving code integrity
- **Comprehensive Reporting**: Multiple output formats including console, HTML, Markdown, and GitHub issues
- **Configurable Rules**: Flexible rule engine for customizing governance policies
- **CI/CD Integration**: Seamless integration with GitHub Actions and other CI systems

## Features

### 🔍 Checkers

#### TidinessChecker
- **Backup Files**: Detects `.bak`, `.backup`, `.orig`, `.old`, `~` files
- **Temporary Files**: Finds `.tmp`, `.temp`, `.swp` files
- **Build Artifacts**: Identifies `__pycache__`, `.pyc`, `node_modules`, `build/`, `dist/` directories
- **Version Control**: Checks for `.DS_Store`, `Thumbs.db`, `.git` artifacts

#### DomainLeakageChecker
- **Monorepo Imports**: Detects imports that breach domain boundaries
- **Cross-Domain Dependencies**: Identifies violations of clean architecture
- **Circular Dependencies**: Finds circular import patterns
- **Architectural Boundaries**: Validates layer separation

#### ArchitectureChecker
- **Clean Architecture**: Validates layer dependencies and boundaries
- **Design Patterns**: Checks for proper implementation of Repository, Service, Factory patterns
- **Anti-Patterns**: Detects god classes, long methods, parameter lists
- **SOLID Principles**: Validates Single Responsibility, Open/Closed, and other principles

### 🔧 Fixers

#### BackupFileFixer
- **Safe Cleanup**: Removes backup files with safety checks
- **Build Artifacts**: Cleans up build directories and cached files
- **Temporary Files**: Removes temporary files and editor artifacts

#### DomainLeakageFixer
- **Import Rewriting**: Replaces monorepo imports with local entities
- **Dependency Injection**: Suggests proper dependency injection patterns
- **Interface Creation**: Generates abstractions for cross-domain dependencies

#### StructureFixer
- **Missing Files**: Creates `__init__.py` files and required structure
- **Directory Organization**: Ensures proper package organization
- **Standard Structure**: Implements consistent directory layouts

### 📊 Reporting

#### Console Reporter
- **Colored Output**: Syntax-highlighted console output
- **Progress Indicators**: Real-time feedback during analysis
- **Summary Statistics**: Comprehensive overview of violations and fixes

#### HTML Reporter
- **Interactive Dashboard**: Rich HTML reports with charts and graphs
- **Detailed Breakdowns**: File-by-file analysis with context
- **Responsive Design**: Mobile-friendly interface

#### Markdown Reporter
- **Documentation Ready**: Clean Markdown format for documentation
- **GitHub Compatible**: Renders perfectly in GitHub and other platforms
- **Table of Contents**: Automatically generated navigation

#### GitHub Issue Reporter
- **Automated Issues**: Creates GitHub issues for violations
- **Batch Processing**: Handles multiple issues efficiently
- **CLI Integration**: Includes shell scripts for GitHub CLI

#### JSON Reporter
- **Machine Readable**: Structured data for tooling integration
- **API Compatible**: Easy integration with external systems
- **Merge Capability**: Combines multiple reports

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Install dependencies
pip install -r scripts/requirements-analysis.txt

# Make the governance runner executable
chmod +x scripts/repository_governance/governance_runner.py
```

## Quick Start

### Basic Usage

```bash
# Run all checks with default configuration
python scripts/repository_governance/governance_runner.py

# Run checks with automatic fixes
python scripts/repository_governance/governance_runner.py --auto-fix

# Run in dry-run mode to see what would be changed
python scripts/repository_governance/governance_runner.py --auto-fix --dry-run

# Generate multiple report formats
python scripts/repository_governance/governance_runner.py --reports console,html,markdown
```

### Configuration

Create a `governance.toml` file in your repository root:

```toml
[general]
enabled = true
dry_run = false
fail_on_violations = false

[checkers.TidinessChecker]
enabled = true
fail_on_violation = false

[checkers.DomainLeakageChecker]
enabled = true
fail_on_violation = true

[checkers.ArchitectureChecker]
enabled = true
fail_on_violation = false

[fixers.BackupFileFixer]
enabled = true
auto_fix = true
dry_run = false

[fixers.DomainLeakageFixer]
enabled = true
auto_fix = false
dry_run = true

[fixers.StructureFixer]
enabled = true
auto_fix = true
dry_run = false

[reporting]
formats = ["console", "html", "markdown"]
output_directory = "reports"
include_charts = true
create_github_issues = false
```

## Advanced Usage

### Custom Rules

Create custom rules using the rules engine:

```python
from scripts.repository_governance.config.rules_engine import RulesEngine, FilePatternRule

engine = RulesEngine()

# Add custom rule
engine.add_rule(FilePatternRule(
    rule_id="no_test_files_in_src",
    description="Test files should not be in src directory",
    pattern=r"src/.*test.*\\.py$",
    should_exist=False,
    severity="high"
))

# Evaluate rules
context = {"root_path": Path(".")}
results = engine.evaluate_all(context)
```

### CI/CD Integration

#### GitHub Actions

Create `.github/workflows/governance.yml`:

```yaml
name: Repository Governance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  governance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r scripts/requirements-analysis.txt
    
    - name: Run governance checks
      run: |
        python scripts/repository_governance/governance_runner.py \\
          --reports console,json,github \\
          --output-dir governance-reports
    
    - name: Upload reports
      uses: actions/upload-artifact@v3
      with:
        name: governance-reports
        path: governance-reports/
    
    - name: Create GitHub issues
      if: failure()
      run: |
        chmod +x governance-reports/create_github_issues.sh
        ./governance-reports/create_github_issues.sh
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Pre-commit Integration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: repository-governance
        name: Repository Governance
        entry: python scripts/repository_governance/governance_runner.py --auto-fix --fail-fast
        language: python
        pass_filenames: false
        always_run: true
```

## Configuration Reference

### General Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable governance system |
| `dry_run` | bool | `false` | Run in dry-run mode (no changes) |
| `fail_on_violations` | bool | `false` | Exit with error code on violations |
| `max_violations` | int | `null` | Maximum allowed violations |

### Checker Configuration

Each checker can be configured with:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable checker |
| `severity_override` | str | `null` | Override default severity |
| `fail_on_violation` | bool | `false` | Fail on any violation |
| `max_violations` | int | `null` | Maximum allowed violations |
| `custom_rules` | dict | `{}` | Checker-specific rules |

### Fixer Configuration

Each fixer can be configured with:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable fixer |
| `auto_fix` | bool | `true` | Automatically apply fixes |
| `dry_run` | bool | `false` | Run in dry-run mode |
| `create_backup` | bool | `true` | Create backups before fixing |
| `max_fixes` | int | `null` | Maximum fixes to apply |

### Reporting Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `formats` | list | `["console"]` | Output formats |
| `output_directory` | str | `null` | Output directory for reports |
| `include_charts` | bool | `true` | Include charts in reports |
| `include_details` | bool | `true` | Include detailed information |
| `create_github_issues` | bool | `false` | Create GitHub issues |

## API Reference

### Main Classes

#### `RepositoryGovernanceRunner`
Main orchestrator class that coordinates checkers, fixers, and reporters.

```python
from scripts.repository_governance.governance_runner import RepositoryGovernanceRunner

runner = RepositoryGovernanceRunner(
    root_path=Path("."),
    config_path=Path("governance.toml")
)

# Run full governance
results = runner.run_full_governance(auto_fix=True, dry_run=False)

# Run only checks
check_results = runner.run_checks()

# Run only fixes
fix_results = runner.run_fixes(dry_run=False)
```

#### `BaseChecker`
Base class for all checkers.

```python
from scripts.repository_governance.checks.base_checker import BaseChecker

class MyChecker(BaseChecker):
    def check(self) -> Dict:
        # Implementation
        pass
```

#### `AutoFixer`
Base class for all fixers.

```python
from scripts.repository_governance.fixes.auto_fixer import AutoFixer

class MyFixer(AutoFixer):
    def can_fix(self, violation: Dict[str, Any]) -> bool:
        # Implementation
        pass
    
    def fix(self, violation: Dict[str, Any]) -> FixResult:
        # Implementation
        pass
```

#### `BaseReporter`
Base class for all reporters.

```python
from scripts.repository_governance.reporting.base_reporter import BaseReporter

class MyReporter(BaseReporter):
    def generate_report(self, check_results: Dict, fix_results: Dict = None) -> str:
        # Implementation
        pass
```

## Best Practices

### 1. Gradual Adoption
- Start with `dry_run = true` to understand impact
- Enable fixers gradually, starting with low-risk ones
- Use `max_violations` to prevent overwhelming output

### 2. Customization
- Create repository-specific rules for your domain
- Override severity levels based on your priorities
- Use custom patterns for your naming conventions

### 3. Integration
- Run governance checks in CI/CD pipelines
- Use pre-commit hooks for immediate feedback
- Schedule regular governance reports

### 4. Team Workflow
- Review fix results before applying automatically
- Use GitHub issues for tracking governance debt
- Include governance metrics in team dashboards

## Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Fix file permissions
chmod +x scripts/repository_governance/governance_runner.py
```

#### Missing Dependencies
```bash
# Install all required dependencies
pip install -r scripts/requirements-analysis.txt
```

#### Configuration Errors
```bash
# Validate configuration
python scripts/repository_governance/governance_runner.py --validate-config
```

#### Performance Issues
```bash
# Run with limited scope
python scripts/repository_governance/governance_runner.py --include-patterns "src/**/*.py"
```

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/repository_governance/governance_runner.py --debug
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r scripts/requirements-analysis.txt

# Run tests
pytest scripts/repository_governance/tests/

# Run governance on itself
python scripts/repository_governance/governance_runner.py --auto-fix
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-org/pynomaly/issues)
- 💬 [Discussions](https://github.com/your-org/pynomaly/discussions)
- 📧 [Email Support](mailto:support@example.com)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a complete history of changes.

## Acknowledgments

- Built with inspiration from industry best practices
- Leverages proven architectural patterns
- Designed for developer productivity and code quality