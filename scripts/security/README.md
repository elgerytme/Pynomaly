# Security Scanning Scripts

This directory contains comprehensive security scanning tools for the Pynomaly project.

## Scripts

### `run_security_scans.py` - Python Code Security Scanning

Comprehensive security scanning script that runs multiple security tools and aggregates results into a unified report with GitHub Actions integration.

### `run_container_scans.py` - Container Security Scanning

Container security scanning script that runs Trivy and Clair scans on Docker images and Dockerfiles, generates SBOMs, and produces unified reports.

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

#### Using Hatch Environment (Recommended)

```bash
# Run comprehensive security scan
hatch run security:scan

# Run quick security check
hatch run security:quick

# Run individual scans
hatch run security:bandit-scan
hatch run security:safety-scan
hatch run security:pip-audit-scan
```

#### Using Make Target

```bash
# Run comprehensive security scan via tox
make security-scan

# Run security scan with CI failure mode
make security-ci
```

#### Direct Script Usage

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

## Container Security Script: `run_container_scans.py`

A comprehensive container security scanning script that runs Trivy and Clair scans on Docker images and Dockerfiles, generates SBOMs, and aggregates results into a unified report.

### Features

- **Image and Dockerfile Support**: Accepts container images or Dockerfiles for scanning
- **Dockerfile Building**: Automatically builds Docker images from Dockerfiles when needed
- **Multi-Tool Scanning**: Runs both Trivy and Clair vulnerability scanners
- **SARIF Output**: Generates SARIF files for GitHub Security tab integration
- **SBOM Generation**: Creates Software Bill of Materials using Trivy with CycloneDX format
- **Unified Reporting**: Aggregates results into comprehensive reports
- **Severity Analysis**: Counts vulnerabilities by severity level with configurable exit codes
- **Flexible Input**: Accepts image lists via CLI arguments or manifest files

### Usage

#### Basic Usage

```bash
# Scan specific images
python scripts/security/run_container_scans.py --images nginx:latest python:3.9-slim

# Scan images from manifest file
python scripts/security/run_container_scans.py --manifest images.txt

# Scan Dockerfiles
python scripts/security/run_container_scans.py --images Dockerfile.prod Dockerfile.dev

# Run in soft mode (don't exit with non-zero code on HIGH/CRITICAL findings)
python scripts/security/run_container_scans.py --images nginx:latest --soft
```

#### Manifest File Format

Create a text file with one image or Dockerfile per line:

```
nginx:latest
python:3.9-slim
ubuntu:20.04
./docker/Dockerfile.prod
./docker/Dockerfile.dev
```

### Tools Included

#### 1. Trivy
- **Purpose**: Comprehensive vulnerability scanner for containers
- **Commands**:
  - `trivy image --format sarif --output [file] [image]` (SARIF output)
  - `trivy image --format json --output [file] [image]` (JSON output)
  - `trivy image --format cyclonedx --output [file] [image]` (SBOM generation)
- **Output**: `*_trivy.sarif`, `*_trivy.json`, `*_sbom.json`

#### 2. Clair
- **Purpose**: Container vulnerability analysis
- **Commands**:
  - `clairctl analyze --format json --output [file] [image]` (preferred)
  - `docker run arminc/clair-local-scan` (fallback)
- **Output**: `*_clair.json`

### Output Files

All output files are saved to `security-results/`:

| File Pattern | Description |
|--------------|-------------|
| `{image}_trivy.sarif` | Trivy scan results in SARIF format |
| `{image}_trivy.json` | Trivy scan results in JSON format |
| `{image}_clair.json` | Clair scan results in JSON format |
| `{image}_sbom.json` | Software Bill of Materials in CycloneDX format |
| `container_consolidated_sarif.json` | Consolidated SARIF file for GitHub upload |
| `container_consolidated_results.json` | All scan results in unified format |
| `container_security_summary.md` | Human-readable summary report |

### Container Security Summary Report

The script generates a comprehensive `container_security_summary.md` report that includes:

- **Overall Summary**: Total scanned images and failed scans
- **Vulnerability Summary**: Total issues by severity level
- **Per-Image Results**: Detailed breakdown for each scanned image
- **Tool-Specific Results**: Separate results for Trivy and Clair
- **SBOM Status**: Whether SBOMs were successfully generated
- **Files Generated**: List of all output files

### Dockerfile Building

The script automatically detects Dockerfiles and builds them before scanning:

```bash
# This will build the Dockerfile and then scan the resulting image
python scripts/security/run_container_scans.py --images ./Dockerfile.prod
```

- Dockerfiles are detected by filename patterns: `Dockerfile*` or `*.dockerfile`
- Images are built with auto-generated tags: `container-scan-{dockerfile-name}`
- Build context is the current directory

### Exit Codes

| Exit Code | Condition |
|-----------|-----------|
| 0 | No HIGH/CRITICAL findings OR --soft mode enabled |
| 1 | HIGH/CRITICAL findings detected (unless --soft mode) |

### GitHub Actions Integration

The script generates SARIF files that can be uploaded to GitHub Security tab:

```yaml
- name: Run Container Security Scans
  run: python scripts/security/run_container_scans.py --images nginx:latest python:3.9

- name: Upload SARIF to GitHub Security Tab
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: security-results/container_consolidated_sarif.json
```

### Requirements

The script requires the following tools to be installed:

- **Docker**: For building Dockerfiles and running containers
- **Trivy**: For vulnerability scanning and SBOM generation
- **Clair**: For additional vulnerability analysis (optional, falls back to Docker-based scan)

#### Installation

```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Install Clair (optional)
go install github.com/quay/clair/v4/cmd/clairctl@latest
```

### Severity Levels

| Severity | Description | Exit Impact |
|----------|-------------|-------------|
| Critical | Immediate security risk | Causes exit code 1 |
| High | High-priority security issue | Causes exit code 1 |
| Medium | Medium-priority security issue | No exit impact |
| Low | Low-priority security issue | No exit impact |
| Info | Informational finding | No exit impact |

### Integration Examples

#### Local Development

```bash
# Scan production images before deployment
python scripts/security/run_container_scans.py --manifest production-images.txt

# Quick check in soft mode
python scripts/security/run_container_scans.py --images myapp:latest --soft

# Build and scan Dockerfiles
python scripts/security/run_container_scans.py --images Dockerfile.prod Dockerfile.dev
```

#### CI/CD Pipeline

```yaml
steps:
  - name: Install Security Tools
    run: |
      curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
  
  - name: Run Container Security Scans
    run: python scripts/security/run_container_scans.py --manifest container-images.txt
  
  - name: Upload Security Reports
    uses: actions/upload-artifact@v3
    with:
      name: container-security-reports
      path: security-results/
```

### Troubleshooting

#### Common Issues

1. **Docker not available**: Ensure Docker is installed and the daemon is running
2. **Trivy not found**: Install Trivy or ensure it's in PATH
3. **Clair not found**: Install clairctl or ensure Docker can pull clair-local-scan image
4. **Permission errors**: Ensure write permissions to `security-results/` directory
5. **Command timeouts**: Container scans have a 10-minute timeout limit
6. **Image pull failures**: Ensure images are accessible and credentials are configured

#### Debug Mode

For debugging, you can examine the individual result files:
- `*_trivy.json` - Detailed Trivy scan results
- `*_clair.json` - Detailed Clair scan results
- `container_consolidated_results.json` - Complete results with exit codes

### Customization

To customize the script:

1. **Modify severity estimation**: Update `_estimate_clair_severity()` method
2. **Add new scanners**: Extend the `ContainerScanner` class
3. **Change output formats**: Modify the report generation methods
4. **Adjust timeouts**: Update the timeout value in `run_command()`
5. **Customize image tags**: Modify the `build_dockerfile()` method

### Security Best Practices

1. **Regular Scans**: Scan container images on every build/deployment
2. **Base Image Updates**: Keep base images updated with security patches
3. **Review Findings**: Don't ignore container security warnings
4. **SBOM Tracking**: Maintain Software Bill of Materials for compliance
5. **Multi-Tool Scanning**: Use both Trivy and Clair for comprehensive coverage
6. **Severity Thresholds**: Use appropriate exit codes for CI/CD pipelines
