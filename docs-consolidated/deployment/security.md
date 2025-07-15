<!-- docs-consolidated/deployment/security.md -->
# Security Hardening and Compliance Guide

This guide provides security and compliance best practices for deploying and operating Pynomaly.

## Security Audit Implementation
- Continuous security scanning in CI/CD:
  - Python security: Bandit, Safety
  - Container security: Trivy, Grype, Docker Scout, SBOM (Syft), Hadolint
- Regular penetration testing and vulnerability assessments.
- Schedule automated security scans (e.g., weekly via CI schedule).
- Integrate SARIF reports into GitHub Security tab via CodeQL uploads.

## Compliance Frameworks
Pynomaly supports the following compliance standards:
- GDPR: Data subject rights, data minimization, privacy by design.
- CCPA: Consumer privacy, data access, opt-out.
- HIPAA: Protected Health Information (PHI) handling.
- SOC 2: Security, availability, processing integrity, confidentiality, privacy.
- ISO 27001: Information Security Management System.
- EU AI Act: Responsible AI requirements.

Configuration flags are defined in [`security_config.py`](../../config/deployment/security/security_config.py) and [`security_policy.yml`](../../config/deployment/security/security_policy.yml):
```yaml
gdpr_compliance: true
ccpa_compliance: true
hipaa_compliance: false
sox_compliance: false
iso27001_compliance: true
soc2_compliance: true
ai_act_compliance: true
```

## Data Protection Measures
- Encryption at rest (default: `encryption_at_rest` flag) using database encryption keys.
- Encryption in transit (TLS/SSL required via `ssl_required`).
- Field-level encryption for PII (`field_level_encryption`).
- Secure key management and rotation (store in `keys.json` or secret manager).
- Data backup encryption and integrity checks.

## Access Control Systems
- Authentication:
  - JWT tokens (`jwt_required`).
  - API Keys (`api_key_required`).
- Authorization:
  - Role-Based Access Control (RBAC) (`rbac_enabled`).
  - Permission checks (`permission_checks`).
- Session Management:
  - Session timeouts (`session_timeout_minutes`).
  - Concurrent session limits (`max_concurrent_sessions`).
- Two-Factor Authentication (2FA) support (`two_factor_auth`).
- Password policies: complexity and minimum lengths (`password_min_length`).

## Audit Logging
- Capture user actions, configuration changes, and security events with audit logging modules.
- Store logs in append-only, tamper-evident storage.
- Configure retention (`audit_log_retention_days`).
- Ensure audit trails meet regulatory retention requirements.

## Tools and Automation
- Static analysis: Bandit, Safety.
- Container scanning: Trivy, Grype, Docker Scout.
- SBOM generation: Syft.
- Dockerfile linting: Hadolint.
- Container structure tests.
- CI/CD integration in `.github/workflows` for automated scanning.

## References
- Security configuration: [`security_config.py`](../../config/deployment/security/security_config.py)
- Security policy: [`security_policy.yml`](../../config/deployment/security/security_policy.yml)
- Security checklist: [`security_checklist.md`](../../config/deployment/security/security_checklist.md)
- Architecture Decision Records (ADRs):
  - [Security Architecture](../architecture/adr/ADR-005-security-architecture.md)
  - [Production Hardening Roadmap](../architecture/adr/ADR-007-production-hardening-roadmap.md)
  - [Security Hardening Threat Model](../architecture/adr/ADR-019-security-hardening-threat-model.md)