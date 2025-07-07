# Security Policy

## Supported Versions

We actively maintain security updates for the following versions of Pynomaly:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take the security of Pynomaly seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Recommended)
   - Go to the [Security tab](https://github.com/pynomaly/pynomaly/security/advisories)
   - Click "Report a vulnerability"
   - Fill out the form with details

2. **Email**
   - Send an email to: security@pynomaly.org
   - Include "SECURITY" in the subject line
   - Provide detailed information about the vulnerability

### What to Include

Please include the following information in your report:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths** of source file(s) related to the manifestation of the issue
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Special configuration** required to reproduce the issue
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact** of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with our assessment
- **Fix Timeline**: 
  - Critical vulnerabilities: Within 14 days
  - High severity: Within 30 days
  - Medium/Low severity: Next regular release cycle

### Security Process

1. **Receipt & Assessment**
   - We acknowledge receipt of your vulnerability report
   - We assess the impact and severity
   - We determine if the issue affects supported versions

2. **Investigation & Fix**
   - We investigate the vulnerability
   - We develop a fix and test it thoroughly
   - We prepare security advisories and documentation

3. **Disclosure**
   - We coordinate with you on disclosure timing
   - We publish security advisories
   - We release patched versions
   - We update our security documentation

### Preferred Languages

We prefer all communications to be in English.

## Security Measures

### Automated Security

Pynomaly employs several automated security measures:

- **Daily vulnerability scanning** with Bandit, Safety, and Semgrep
- **Dependency monitoring** via Dependabot and pip-audit
- **CodeQL static analysis** for advanced security patterns
- **SARIF integration** for GitHub Security tab visibility
- **Automated dependency updates** for security patches

### Development Security

- **Secure development practices** following OWASP guidelines
- **Code review requirements** for all changes
- **Pre-commit security hooks** (planned)
- **Security-focused testing** in CI/CD pipeline
- **Supply chain security** with dependency pinning

### Infrastructure Security

- **GitHub security features** enabled (branch protection, required reviews)
- **Secrets management** via GitHub Secrets
- **Access controls** with principle of least privilege
- **Audit logging** for all repository activities

## Security Best Practices for Users

### Installation

```bash
# Always install from official PyPI
pip install pynomaly

# Verify package integrity (when available)
pip install pynomaly --require-hashes

# Use virtual environments
python -m venv pynomaly-env
source pynomaly-env/bin/activate
pip install pynomaly
```

### Usage

- **Keep dependencies updated**: Regularly update Pynomaly and its dependencies
- **Validate input data**: Always validate and sanitize input data
- **Use secure configurations**: Follow security guidelines in documentation
- **Monitor for updates**: Subscribe to security advisories

### Data Security

- **Sensitive data**: Be cautious when processing sensitive or personal data
- **Data validation**: Always validate data before processing
- **Secure storage**: Follow best practices for data storage and transmission
- **Access controls**: Implement appropriate access controls for your applications

## Security Contact

For security-related questions or concerns:

- **Security Team**: security@pynomaly.org
- **General Questions**: Via GitHub Discussions
- **Documentation**: See [Security Documentation](docs/security.md)

## Acknowledgments

We appreciate security researchers and users who responsibly disclose vulnerabilities. Contributors who report valid security issues will be acknowledged in:

- Security advisories
- Release notes
- Hall of Fame (if desired)

## Security Resources

- [OWASP Python Security](https://owasp.org/www-project-python-security/)
- [Python Security Documentation](https://docs.python.org/3/library/security.html)
- [GitHub Security Features](https://docs.github.com/en/code-security)
- [Supply Chain Security](https://slsa.dev/)

---

**Note**: This security policy is regularly reviewed and updated. Last updated: 2024-07-07