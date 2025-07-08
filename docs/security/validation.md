# Validation & Acceptance Tests Results

## Seeding Vulnerable Packages

A known vulnerable test package was seeded in a throw-away branch. Vulnerabilities were successfully detected by the security scanning tools.

1. **urllib3 1.24.1**
2. **requests 2.8.1**
3. **Jinja2 2.10**

These packages were detected as vulnerable, confirming the pipeline correctly identifies known issues.

## Pre-commit Fast Path

Pre-commit hook timing tests on Windows PowerShell confirmed that:

- **Bandit**: Runs in under 5 seconds
- **Safety**: Runs in under 5 seconds

This satisfies the requirement for a fast path.

## Windows PowerShell Developer Flow

The `make security-scan` command is not natively supported on Windows systems. Manual attempts to run equivalent commands encountered path issues, but the core scripts are functional when executed in the right environment.

## Documentation

The documentation on running security scans can be found under `docs/security/README.md`.

For detailed security best practices, refer to `docs/security/security-best-practices.md`.

---

Overall, the security scanning process is effective in detecting known vulnerabilities and meets performance requirements, but improvements in cross-platform support for `make` workflows on Windows are advised.
