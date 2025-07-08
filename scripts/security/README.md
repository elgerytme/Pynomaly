# Security Scanning Scripts

This directory contains comprehensive security scanning tools for the Pynomaly project.

## Main Script: `run_security_scans.py`

A comprehensive security scanning script that runs multiple security tools and aggregates results into a unified report with GitHub Actions integration.

### Features

- **Environment Detection**: Auto-detects CI environment and activates Hatch environment when needed
- **Multi-Tool Scanning**: Runs safety, bandit, and pip-audit with comprehensive reporting
- **Multiple Output Formats**: Generates JSON, SARIF, and human-readable reports
- **GitHub Integration**: Produces SARIF files for GitHub Security tab upload
- **Severity Analysis**: Aggregates findings by severity level with configurable exit codes
- **Comprehensive Reporting**: Generates detailed security summary with recommendations

### Usage

```bash
# Run all security scans
python scripts/security/run_security_scans.py

# Run in soft mode (don't exit with non-zero code on HIGH/CRITICAL findings)
python scripts/security/run_security_scans.py --soft
```

### Tools Included

#### 1. Safety
- **Purpose**: Scans Python dependencies for known security vulnerabilities
- **Command**: `safety check --full-report --json`
- **Output**: `safety_results.json`, `safety_raw.txt`

#### 2. Bandit
- **Purpose**: Analyzes Python source code for security issues
- **Commands**: 
  - `bandit -r src/ -f json` (JSON output)
  - `bandit -r src/ -f txt` (Text output)
  - `bandit -r src/ -f sarif` (SARIF output for GitHub)
- **Output**: `bandit_results.json`, `bandit_results.txt`, `bandit_results.sarif`

#### 3. Pip-Audit
- **Purpose**: Scans Python packages for known vulnerabilities
- **Commands**:
  - `pip-audit --format json` (JSON output)
  - `pip-audit --format cyclonedx` (CycloneDX SBOM format)
- **Output**: `pip_audit_results.json`, `pip_audit_cyclonedx.json`

### Output Files

All output files are saved to `artifacts/security/`:

| File | Description |
|------|-------------|
| `safety_results.json` | Safety scan results in JSON format |
| `safety_raw.txt` | Raw safety command output |
| `bandit_results.json` | Bandit scan results in JSON format |
| `bandit_results.txt` | Bandit scan results in human-readable format |
| `bandit_results.sarif` | Bandit scan results in SARIF format |
| `pip_audit_results.json` | Pip-audit scan results in JSON format |
| `pip_audit_cyclonedx.json` | Pip-audit results in CycloneDX SBOM format |
| `consolidated_sarif.json` | Consolidated SARIF file for GitHub upload |
| `consolidated_results.json` | All scan results in unified format |
| `security_summary.md` | Human-readable summary report |

### Security Summary Report

The script generates a comprehensive `security_summary.md` report that includes:

- **Overall Summary**: Total issues by severity level
- **Tool-Specific Results**: Detailed breakdown for each security tool
- **Recommendations**: Action items based on findings severity
- **Files Generated**: List of all output files

### Environment Integration

#### Hatch Environment
- Automatically detects if running outside CI
- Activates Hatch environment for consistent tool versions
- Uses project-specific Python environment and dependencies

#### CI Environment Detection
The script detects CI environments using these indicators:
- `CI`
- `CONTINUOUS_INTEGRATION`
- `GITHUB_ACTIONS`
- `GITLAB_CI`
- `JENKINS_URL`
- `TRAVIS`
- `CIRCLECI`
- `BUILDKITE`
- `AZURE_PIPELINES_BUILD_ID`

### Exit Codes

| Exit Code | Condition |
|-----------|-----------|
| 0 | No HIGH/CRITICAL findings OR --soft mode enabled |
| 1 | HIGH/CRITICAL findings detected (unless --soft mode) |

### GitHub Actions Integration

The script generates SARIF files that can be uploaded to GitHub Security tab:

```yaml
- name: Run Security Scans
  run: python scripts/security/run_security_scans.py

- name: Upload SARIF to GitHub Security Tab
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: artifacts/security/consolidated_sarif.json
```

### Severity Levels

| Severity | Description | Exit Impact |
|----------|-------------|-------------|
| Critical | Immediate security risk | Causes exit code 1 |
| High | High-priority security issue | Causes exit code 1 |
| Medium | Medium-priority security issue | No exit impact |
| Low | Low-priority security issue | No exit impact |
| Info | Informational finding | No exit impact |

### Requirements

The script requires the following Python packages:
- `safety` - For dependency vulnerability scanning
- `bandit` - For source code security analysis
- `pip-audit` - For package vulnerability scanning

These should be installed in your development environment or CI pipeline.

### Troubleshooting

#### Common Issues

1. **Hatch not found**: Ensure Hatch is installed and available in PATH
2. **Security tools not found**: Install required tools using pip or ensure they're in the Hatch environment
3. **Permission errors**: Ensure write permissions to `artifacts/security/` directory
4. **Command timeouts**: Security scans have a 10-minute timeout limit

#### Debug Mode

For debugging, you can examine the raw output files:
- `safety_raw.txt` - Raw safety command output
- Check exit codes in JSON result files
- Review stderr output in consolidated results

### Integration Examples

#### Local Development
```bash
# Run before commits
python scripts/security/run_security_scans.py

# Quick check in soft mode
python scripts/security/run_security_scans.py --soft
```

#### CI/CD Pipeline
```yaml
steps:
  - name: Install Security Tools
    run: |
      pip install safety bandit pip-audit
  
  - name: Run Security Scans
    run: python scripts/security/run_security_scans.py
  
  - name: Upload Security Reports
    uses: actions/upload-artifact@v3
    with:
      name: security-reports
      path: artifacts/security/
```

### Customization

To customize the script:

1. **Modify severity estimation**: Update `_estimate_*_severity()` methods
2. **Add new tools**: Extend the `SecurityScanner` class
3. **Change output formats**: Modify the report generation methods
4. **Adjust timeouts**: Update the timeout value in `run_command()`

### Security Best Practices

1. **Regular Scans**: Run security scans on every commit/PR
2. **Dependency Updates**: Keep security tools updated
3. **Review Findings**: Don't ignore security warnings
4. **Severity Thresholds**: Use appropriate exit codes for CI/CD
5. **SARIF Integration**: Upload results to GitHub Security tab
