# Security Scanning Templates and Configuration

This directory contains comprehensive security scanning templates, configurations, and tools for the domain-bounded monorepo.

## Overview

The security infrastructure implements defense-in-depth with multiple layers of security scanning:

- **SAST (Static Application Security Testing)**: Bandit, Semgrep, CodeQL
- **DAST (Dynamic Application Security Testing)**: OWASP ZAP
- **Container Security**: Trivy, Hadolint, Docker Scout
- **Dependency Scanning**: Safety, pip-audit
- **Secret Detection**: detect-secrets, TruffleHog, GitLeaks
- **License Compliance**: pip-licenses with policy enforcement
- **Supply Chain Security**: SBOM generation with Syft
- **Infrastructure as Code**: Checkov for Terraform/K8s security

## Quick Start

### 1. Run Security Validation

```bash
# Validate security configuration
python security/scripts/validate-security-config.py

# Run comprehensive security scan
.github/workflows/security-scanning.yml
```

### 2. Local Security Scanning

```bash
# SAST scanning
bandit -r src/ -f json -o bandit-results.json
semgrep --config=auto src/

# Dependency scanning
safety check --json --output safety-results.json
pip-audit --format=json --output=pip-audit-results.json

# Container scanning
trivy image anomaly-detection:latest
hadolint Dockerfile

# Secret detection
detect-secrets scan --all-files --force-use-all-plugins
```

## Configuration Files

### Core Configuration
- `security-scanning-config.yaml`: Central security policy and tool configuration
- `.zap/rules.tsv`: OWASP ZAP scanning rules and exceptions

### CI/CD Integration
- `.github/workflows/security-scanning.yml`: Comprehensive security pipeline
- Pre-commit hooks with security tools integration

## Security Tools

### Static Application Security Testing (SAST)

#### Bandit
**Purpose**: Python-specific security issue detection  
**Configuration**: `.bandit` file in repository root  
**Integration**: Pre-commit hooks, CI/CD pipeline

```yaml
# Example bandit configuration
skips: ['B101', 'B601']  # Skip specific tests
exclude_dirs: ['tests', 'migrations']
```

#### Semgrep
**Purpose**: Multi-language static analysis with custom rules  
**Configuration**: `security-scanning-config.yaml`  
**Rules**: OWASP Top 10, security-audit, language-specific

```bash
# Custom Semgrep rules
semgrep --config=p/security-audit --config=p/owasp-security src/
```

#### CodeQL
**Purpose**: GitHub Advanced Security semantic code analysis  
**Languages**: Python, JavaScript  
**Queries**: security-extended, security-and-quality

### Dynamic Application Security Testing (DAST)

#### OWASP ZAP
**Purpose**: Web application security testing  
**Configuration**: `.zap/rules.tsv`  
**Modes**: Baseline scan, Full scan

```bash
# Baseline scan (passive)
zap-baseline.py -t http://localhost:8000

# Full scan (active)
zap-full-scan.py -t http://localhost:8000
```

### Container Security

#### Trivy
**Purpose**: Container vulnerability scanning  
**Features**: OS packages, application dependencies, secrets  
**Formats**: JSON, SARIF, table

```bash
# Scan container image
trivy image --format sarif --output results.sarif anomaly-detection:latest

# Scan filesystem
trivy fs --security-checks vuln,secret,config .
```

#### Hadolint
**Purpose**: Dockerfile linting and security best practices  
**Integration**: CI/CD pipeline, pre-commit hooks

### Dependency Security

#### Safety
**Purpose**: Python dependency vulnerability scanning  
**Database**: PyUp.io vulnerability database  
**Integration**: CI/CD, pre-commit hooks

#### pip-audit
**Purpose**: Python package vulnerability auditing  
**Features**: Requirements file scanning, JSON output

### Secret Detection

#### detect-secrets
**Purpose**: Baseline-driven secret detection  
**Plugins**: AWS, Azure, JWT, private keys, high-entropy strings  
**Baseline**: `.secrets.baseline`

#### TruffleHog
**Purpose**: Advanced secret scanning with verification  
**Features**: Git history scanning, credential verification

#### GitLeaks
**Purpose**: Git-native secret detection  
**Integration**: GitHub Actions, pre-commit hooks

### License Compliance

#### pip-licenses
**Purpose**: Python package license analysis  
**Policy**: Defined in `security-scanning-config.yaml`  
**Formats**: JSON, CSV, HTML

```yaml
# License policy example
allowed_licenses:
  - "MIT"
  - "Apache-2.0"
  - "BSD-3-Clause"

prohibited_licenses:
  - "GPL-3.0"
  - "AGPL-3.0"
```

### Supply Chain Security

#### Syft (SBOM Generation)
**Purpose**: Software Bill of Materials creation  
**Formats**: SPDX, CycloneDX, Syft JSON  
**Integration**: CI/CD pipeline, release process

```bash
# Generate SBOM
syft . -o spdx-json=sbom.spdx.json
syft . -o cyclonedx-json=sbom.cyclonedx.json
```

### Infrastructure as Code Security

#### Checkov
**Purpose**: IaC security scanning  
**Frameworks**: Terraform, Kubernetes, Dockerfile  
**Output**: SARIF for GitHub Security tab integration

## Security Policies

### Risk Tolerance
```yaml
risk_tolerance:
  critical: 0      # No critical vulnerabilities allowed
  high: 2          # Maximum 2 high severity vulnerabilities  
  medium: 10       # Maximum 10 medium severity vulnerabilities
  low: 50          # Maximum 50 low severity vulnerabilities
```

### Quality Gates
- **Block deployment**: Critical vulnerabilities, secrets detected
- **Require approval**: High vulnerability threshold exceeded
- **Automatic pass**: Within risk tolerance thresholds

### Compliance Frameworks
- **OWASP Top 10 2021**: Web application security risks
- **NIST Cybersecurity Framework**: Risk management approach
- **CIS Controls**: Cybersecurity best practices
- **GDPR**: Data protection compliance

## CI/CD Integration

### Security Pipeline Stages

1. **Pre-commit**: Basic security checks (secrets, linting)
2. **SAST**: Static code analysis (Bandit, Semgrep)
3. **Dependency**: Vulnerability scanning (Safety, pip-audit)
4. **Container**: Image security scanning (Trivy, Hadolint)
5. **DAST**: Dynamic security testing (OWASP ZAP)
6. **Compliance**: License and policy validation
7. **Supply Chain**: SBOM generation and validation

### GitHub Security Integration

Results are automatically uploaded to:
- **GitHub Security tab**: SARIF format uploads
- **GitHub Advanced Security**: CodeQL integration
- **Pull Request comments**: Security scan summaries
- **Artifacts**: Detailed reports and evidence

## Custom Security Rules

### Semgrep Custom Rules
Create custom rules in `.semgrep/` directory:

```yaml
# .semgrep/custom-security.yml
rules:
  - id: hardcoded-secret
    pattern: password = "..."
    message: Hardcoded password detected
    languages: [python]
    severity: ERROR
```

### Bandit Custom Configuration
Extend Bandit configuration in `.bandit`:

```ini
[bandit]
exclude_dirs = tests,migrations,docs
skips = B101,B601
```

## Monitoring and Alerting

### Security Metrics
- Vulnerability counts by severity
- Scan coverage and frequency  
- Mean time to remediation
- Policy compliance percentages

### Alerting Channels
- **Slack**: Real-time security notifications
- **Email**: Weekly security summaries
- **GitHub Issues**: Automated vulnerability tracking

## Remediation Workflows

### Critical Vulnerabilities
1. **Immediate**: Block deployment, create incident
2. **Notification**: Alert security team and developers
3. **Remediation**: Patch within 24 hours
4. **Verification**: Re-scan and validate fix

### Non-Critical Issues
1. **Prioritization**: Based on severity and exploitability
2. **Assignment**: To appropriate development team
3. **Timeline**: Based on risk assessment
4. **Tracking**: Through GitHub Issues integration

## Best Practices

### Development
- Run security scans locally before committing
- Address security issues during development
- Follow secure coding guidelines
- Regular security training

### Operations
- Regular security tool updates
- Baseline maintenance for false positives
- Security policy reviews and updates
- Incident response procedures

### Compliance
- Regular compliance audits
- Documentation of security measures
- Evidence collection and retention
- Third-party security assessments

## Troubleshooting

### Common Issues

1. **False Positives**
   - Update tool configurations
   - Add suppression rules
   - Maintain exception baselines

2. **Tool Integration Failures**
   - Verify tool installation and versions
   - Check network connectivity
   - Review authentication credentials

3. **Performance Issues**
   - Adjust scan timeouts
   - Optimize scan scope
   - Use incremental scanning

### Debug Commands

```bash
# Validate security configuration
python security/scripts/validate-security-config.py

# Test individual tools
bandit --version
safety --version  
semgrep --version
trivy --version

# Check CI/CD pipeline
gh workflow run security-scanning.yml
```

## Updates and Maintenance

### Tool Updates
- Monthly security tool version updates
- Quarterly rule and signature updates
- Annual policy and configuration reviews

### Baseline Maintenance
- Weekly false positive reviews
- Monthly baseline updates
- Quarterly policy adjustments

### Compliance Reviews
- Monthly compliance reporting
- Quarterly audit preparations
- Annual policy reviews and updates