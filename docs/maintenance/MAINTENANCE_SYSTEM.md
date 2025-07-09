# Automated Maintenance System

This document provides comprehensive documentation for Pynomaly's automated maintenance system, including scheduled workflows, local tool execution, and contributor expectations.

## Overview

The Pynomaly project includes a robust automated maintenance system that ensures code quality, security, and project organization through scheduled workflows and local tooling. This system runs comprehensive quality checks, security scans, and cleanup tasks to maintain high standards across the codebase.

## Scheduled Workflow

### Schedule
- **Frequency**: Weekly (every Monday at 3:00 AM UTC)
- **Trigger**: Automatic via GitHub Actions cron schedule
- **Manual Trigger**: Can be triggered manually through GitHub Actions UI

### Workflow Components

#### 1. Structure Validation
- **Script**: `scripts/validation/validate_structure.py`
- **Purpose**: Validates project structure and organization
- **Checks**:
  - Required directories exist
  - Required files are present
  - Project follows clean architecture patterns
  - File organization standards are maintained

#### 2. Quality Analysis
The workflow runs multiple quality analysis tools to ensure code standards:

##### Ruff (Linting and Formatting)
- **Command**: `ruff check src/ tests/`
- **Purpose**: Fast Python linter with comprehensive rule coverage
- **Output Formats**: GitHub, JSON, SARIF
- **Configuration**: Defined in `pyproject.toml`

##### MyPy (Type Checking)
- **Command**: `mypy src/pynomaly/`
- **Purpose**: Static type checking for Python code
- **Mode**: Strict mode enabled for maximum type safety
- **Output**: Text report with detailed type errors

##### Bandit (Security Scanning)
- **Command**: `bandit -r src/`
- **Purpose**: Security linter for Python code
- **Output Formats**: SARIF, JSON, text
- **Integration**: Results uploaded to GitHub Security tab

##### Safety (Dependency Vulnerability Scanning)
- **Command**: `safety check --full-report`
- **Purpose**: Scans dependencies for known vulnerabilities
- **Output**: JSON report with vulnerability details

##### pip-audit (Package Auditing)
- **Command**: `pip-audit`
- **Purpose**: Audits Python packages for security vulnerabilities
- **Output**: JSON report with detailed vulnerability information

#### 3. Security Integration
- **SARIF Upload**: Automatically uploads security scan results to GitHub Security tab
- **Vulnerability Tracking**: Centralized tracking of security issues
- **Compliance**: Ensures security compliance standards are met

#### 4. Report Generation
- **HTML Reports**: Styled HTML reports for human readability
- **JSON Reports**: Machine-readable reports for automation
- **Artifact Storage**: 90-day retention for historical analysis
- **GitHub Pages**: Automatic deployment of reports for easy access

#### 5. Repository Cleanup
- **Automated Cleanup**: Removes temporary files and cache directories
- **Pull Request Creation**: Automatically creates PRs for cleanup changes
- **Safe Operations**: Only removes files that are safe to delete

#### 6. Notification System
- **Threshold-based Alerts**: Notifications when violations exceed thresholds
- **GitHub Issues**: Automatic issue creation for quality violations
- **Slack Integration**: Optional Slack webhook notifications
- **Cooldown Periods**: Prevents notification spam

## Local Tool Execution

### Prerequisites
Before running maintenance tools locally, ensure you have the required dependencies:

```bash
# Install maintenance tools
pip install ruff mypy bandit safety pip-audit
```

### Running Individual Tools

#### Structure Validation
```bash
python scripts/validation/validate_structure.py
```

#### Linting with Ruff
```bash
# Check for linting issues
ruff check src/ tests/

# Auto-fix issues where possible
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

#### Type Checking with MyPy
```bash
# Run type checking
mypy src/pynomaly/

# Run with strict mode (recommended)
mypy --strict src/pynomaly/
```

#### Security Scanning with Bandit
```bash
# Basic security scan
bandit -r src/

# Detailed scan with JSON output
bandit -r src/ -f json -o security-report.json
```

#### Dependency Vulnerability Scanning
```bash
# Safety scan
safety check --full-report

# pip-audit scan
pip-audit

# Save results to file
safety check --full-report --json --output safety-report.json
pip-audit --format=json --output=pip-audit-report.json
```

### Comprehensive Local Check Script

Create a comprehensive check script for local development:

```bash
#!/bin/bash
# local-maintenance-check.sh

echo "🔧 Running Local Maintenance Checks"
echo "=================================="

# Structure validation
echo "1. Structure Validation..."
python scripts/validation/validate_structure.py

# Linting
echo "2. Linting with Ruff..."
ruff check src/ tests/

# Type checking
echo "3. Type Checking with MyPy..."
mypy src/pynomaly/

# Security scanning
echo "4. Security Scanning..."
bandit -r src/
safety check --full-report
pip-audit

# Test coverage
echo "5. Test Coverage..."
pytest --cov=src/pynomaly --cov-report=html

echo "✅ All checks completed!"
```

## Contributor Expectations

### Pre-Push Checklist
Before pushing any changes, contributors must run the following maintenance checks:

1. **✅ Structure Validation**
   ```bash
   python scripts/validation/validate_structure.py
   ```

2. **✅ Linting and Formatting**
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

3. **✅ Type Checking**
   ```bash
   mypy src/pynomaly/
   ```

4. **✅ Security Scans**
   ```bash
   bandit -r src/
   safety check --full-report
   pip-audit
   ```

5. **✅ Test Coverage**
   ```bash
   pytest --cov=src/pynomaly --cov-report=html
   ```

### Quality Gates
All contributions must meet the following quality standards:

- **Linting**: All Ruff checks must pass
- **Type Checking**: No MyPy errors allowed
- **Security**: No high or medium severity vulnerabilities
- **Test Coverage**: ≥95% code coverage required
- **Tests**: All tests must pass (100% pass rate)

### IDE Integration
Configure your IDE to run these tools automatically:

#### VS Code Configuration
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.banditEnabled": true,
  "python.formatting.provider": "ruff"
}
```

#### PyCharm Configuration
1. Install Ruff, MyPy, and Bandit plugins
2. Configure external tools for security scanning
3. Set up pre-commit hooks for automatic checks

## Maintenance Status Badges

The project README includes maintenance status badges that reflect the current state of the automated workflows:

```markdown
[![Maintenance Status](https://github.com/yourusername/pynomaly/workflows/Scheduled%20Maintenance/badge.svg)](https://github.com/yourusername/pynomaly/actions)
```

### Badge States
- **✅ Passing**: All maintenance checks pass
- **❌ Failing**: One or more maintenance checks fail
- **⚠️ Warning**: Minor issues found but not blocking

## Troubleshooting

### Common Issues

#### High Memory Usage During Scans
```bash
# Reduce MyPy cache size
export MYPY_CACHE_DIR=/tmp/mypy_cache

# Limit Bandit scan scope
bandit -r src/ --exclude tests/
```

#### False Positives in Security Scans
```bash
# Create Bandit configuration file
cat > .bandit << EOF
[bandit]
exclude_dirs = tests/
skips = B101,B601
EOF
```

#### Slow Ruff Performance
```bash
# Use ruff with specific target version
ruff check --target-version py311 src/
```

### Getting Help
- **GitHub Issues**: Report problems with the maintenance system
- **Discussions**: Ask questions about tool configuration
- **Documentation**: Check tool-specific documentation for advanced usage

## Configuration

### Workflow Configuration
The maintenance workflow configuration can be found in:
- **File**: `.github/workflows/maintenance.yml`
- **Schedule**: Cron expression for weekly execution
- **Environment**: Environment variables for tool configuration

### Tool Configuration
Individual tools are configured in:
- **Ruff**: `pyproject.toml` [tool.ruff] section
- **MyPy**: `pyproject.toml` [tool.mypy] section
- **Bandit**: `.bandit` configuration file
- **pytest**: `pyproject.toml` [tool.pytest] section

### Customization
Organizations can customize the maintenance system by:
1. Modifying threshold values for quality gates
2. Adding custom validation scripts
3. Configuring notification channels
4. Extending the cleanup operations

## Best Practices

### For Contributors
1. **Run checks locally** before pushing
2. **Fix issues incrementally** rather than in large batches
3. **Use IDE integration** for real-time feedback
4. **Monitor badge status** to ensure changes don't break maintenance

### For Maintainers
1. **Review maintenance reports** regularly
2. **Adjust thresholds** based on project maturity
3. **Update tool configurations** as needed
4. **Monitor notification channels** for important alerts

### For Organizations
1. **Customize workflows** to match organizational standards
2. **Integrate with existing tools** and processes
3. **Train team members** on maintenance procedures
4. **Establish escalation procedures** for critical issues

## Future Enhancements

### Planned Features
- **Advanced Analytics**: Trend analysis and quality metrics
- **Integration Testing**: Extended testing during maintenance
- **Performance Monitoring**: Performance regression detection
- **Documentation Validation**: Automated documentation checks

### Extensibility
The maintenance system is designed to be extensible:
- **Plugin System**: Add custom validation plugins
- **Integration Points**: Connect with external quality tools
- **Notification Channels**: Add new notification methods
- **Reporting Formats**: Extend reporting capabilities

---

*This documentation is maintained as part of the automated maintenance system and is updated regularly to reflect current practices and configurations.*
