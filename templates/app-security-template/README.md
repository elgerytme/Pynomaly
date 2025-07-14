# Application Security Template

A comprehensive application security framework template with enterprise-grade security controls, vulnerability management, and compliance features.

## Features

- **Security Middleware**: CSRF, XSS, SQL injection protection
- **Input Validation**: Comprehensive data sanitization and validation
- **Dependency Security**: Vulnerability scanning and management
- **Static Analysis**: Code security scanning with SAST tools
- **Dynamic Testing**: DAST and penetration testing automation
- **Security Headers**: Comprehensive HTTP security headers
- **Data Protection**: Encryption at rest and in transit
- **Access Control**: Fine-grained permission systems
- **Audit & Compliance**: Security logging and compliance reporting
- **Incident Response**: Security event handling and alerting
- **Secure Configuration**: Security hardening and best practices
- **Container Security**: Docker and Kubernetes security

## Directory Structure

```
app-security-template/
├── build/                 # Build artifacts and reports
├── deploy/                # Secure deployment configurations
├── docs/                  # Security documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary security files
├── src/                   # Source code
│   └── security_framework/
│       ├── middleware/   # Security middleware
│       ├── validators/   # Input validation
│       ├── scanners/     # Security scanners
│       ├── encryption/   # Cryptographic functions
│       ├── access/       # Access control
│       ├── monitoring/   # Security monitoring
│       ├── incidents/    # Incident response
│       └── compliance/   # Compliance tools
├── tests/                # Security test suites
├── security/             # Security configurations
│   ├── policies/        # Security policies
│   ├── scans/           # Scan results
│   ├── reports/         # Security reports
│   └── certificates/    # SSL/TLS certificates
├── scripts/              # Security automation
├── .github/              # Security workflows
├── pyproject.toml        # Project configuration
├── security.yml         # Security configuration
├── Dockerfile.secure    # Hardened container
├── README.md            # Documentation
├── SECURITY.md          # Security policy
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-secure-app
   cd my-secure-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[security,dev,test]"
   ```

3. **Initialize security**:
   ```bash
   python scripts/init_security.py
   ```

4. **Run security scan**:
   ```bash
   python scripts/security_scan.py
   ```

5. **Start secure application**:
   ```bash
   uvicorn security_framework.main:app --ssl-keyfile=security/certificates/key.pem --ssl-certfile=security/certificates/cert.pem
   ```

## Security Middleware

### CSRF Protection

```python
from security_framework.middleware.csrf import CSRFMiddleware

# CSRF protection middleware
app.add_middleware(
    CSRFMiddleware,
    secret_key="your-csrf-secret",
    cookie_name="csrftoken",
    header_name="X-CSRFToken",
    methods=["POST", "PUT", "DELETE", "PATCH"],
    origins=["https://yourdomain.com"],
    max_age=3600
)
```

### XSS Protection

```python
from security_framework.middleware.xss import XSSProtectionMiddleware

# XSS protection
app.add_middleware(
    XSSProtectionMiddleware,
    auto_escape=True,
    content_security_policy={
        "default-src": ["'self'"],
        "script-src": ["'self'", "'unsafe-inline'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "https:"],
        "font-src": ["'self'", "https://fonts.googleapis.com"],
        "connect-src": ["'self'", "https://api.yourdomain.com"]
    }
)
```

### SQL Injection Prevention

```python
from security_framework.middleware.sql_injection import SQLInjectionMiddleware

# SQL injection protection
app.add_middleware(
    SQLInjectionMiddleware,
    patterns=[
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
        r"(\bunion\b.*\bselect\b)",
        r"(\bor\b.*=.*)",
        r"(;|\-\-|\/\*|\*\/)"
    ],
    log_attempts=True,
    block_requests=True
)
```

### Security Headers

```python
from security_framework.middleware.security_headers import SecurityHeadersMiddleware

# Security headers
app.add_middleware(
    SecurityHeadersMiddleware,
    force_https=True,
    hsts_max_age=31536000,  # 1 year
    hsts_include_subdomains=True,
    hsts_preload=True,
    content_type_options="nosniff",
    frame_options="DENY",
    xss_protection="1; mode=block",
    referrer_policy="strict-origin-when-cross-origin",
    permissions_policy={
        "camera": "none",
        "microphone": "none",
        "geolocation": "self",
        "payment": "self"
    }
)
```

## Input Validation & Sanitization

### Data Validation

```python
from security_framework.validators.input_validator import InputValidator

class UserValidator(InputValidator):
    def validate_email(self, email: str) -> str:
        # Email validation with security checks
        email = self.sanitize_input(email)
        if not self.is_valid_email(email):
            raise ValueError("Invalid email format")
        if self.is_disposable_email(email):
            raise ValueError("Disposable emails not allowed")
        return email
    
    def validate_password(self, password: str) -> str:
        # Password strength validation
        if len(password) < 12:
            raise ValueError("Password must be at least 12 characters")
        if not self.has_complexity(password):
            raise ValueError("Password must include uppercase, lowercase, numbers, and symbols")
        if self.is_common_password(password):
            raise ValueError("Password is too common")
        return password
```

### SQL Injection Prevention

```python
from security_framework.validators.sql_validator import SQLValidator

class DatabaseValidator:
    def __init__(self):
        self.sql_validator = SQLValidator()
    
    def validate_query_params(self, params: dict) -> dict:
        """Validate and sanitize database query parameters."""
        validated_params = {}
        
        for key, value in params.items():
            # Check for SQL injection patterns
            if self.sql_validator.contains_sql_injection(value):
                raise SecurityError(f"Potential SQL injection in parameter: {key}")
            
            # Sanitize the value
            validated_params[key] = self.sql_validator.sanitize(value)
        
        return validated_params
```

### File Upload Security

```python
from security_framework.validators.file_validator import FileValidator

class SecureFileUpload:
    def __init__(self):
        self.file_validator = FileValidator()
    
    def validate_upload(self, file: UploadFile) -> bool:
        # File type validation
        if not self.file_validator.is_allowed_type(file.filename):
            raise ValueError("File type not allowed")
        
        # File size validation
        if file.size > self.file_validator.max_file_size:
            raise ValueError("File too large")
        
        # Virus scanning
        if self.file_validator.contains_malware(file.file):
            raise SecurityError("Malware detected in file")
        
        # Content validation
        if not self.file_validator.validate_content(file.file):
            raise ValueError("Invalid file content")
        
        return True
```

## Vulnerability Scanning

### Dependency Scanning

```python
from security_framework.scanners.dependency_scanner import DependencyScanner

class SecurityScanner:
    def __init__(self):
        self.dependency_scanner = DependencyScanner()
    
    def scan_dependencies(self) -> dict:
        """Scan dependencies for known vulnerabilities."""
        results = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }
        
        # Get installed packages
        packages = self.dependency_scanner.get_installed_packages()
        
        # Check against vulnerability databases
        for package in packages:
            vulnerabilities = self.dependency_scanner.check_vulnerabilities(
                package.name, 
                package.version
            )
            
            for vuln in vulnerabilities:
                results[vuln.severity].append({
                    "package": package.name,
                    "version": package.version,
                    "vulnerability": vuln.id,
                    "description": vuln.description,
                    "fixed_version": vuln.fixed_version
                })
        
        return results
```

### Static Analysis Security Testing (SAST)

```python
from security_framework.scanners.sast_scanner import SASTScanner

class CodeSecurityScanner:
    def __init__(self):
        self.sast_scanner = SASTScanner()
    
    def scan_code(self, target_path: str) -> dict:
        """Perform static analysis security testing."""
        results = {
            "security_issues": [],
            "code_quality": [],
            "best_practices": []
        }
        
        # Security pattern scanning
        security_issues = self.sast_scanner.scan_security_patterns(target_path)
        results["security_issues"] = security_issues
        
        # Code quality analysis
        quality_issues = self.sast_scanner.analyze_code_quality(target_path)
        results["code_quality"] = quality_issues
        
        # Best practices validation
        practices = self.sast_scanner.check_best_practices(target_path)
        results["best_practices"] = practices
        
        return results
```

### Dynamic Application Security Testing (DAST)

```python
from security_framework.scanners.dast_scanner import DASTScanner

class DynamicSecurityScanner:
    def __init__(self):
        self.dast_scanner = DASTScanner()
    
    def scan_application(self, base_url: str) -> dict:
        """Perform dynamic security testing."""
        results = {
            "vulnerabilities": [],
            "misconfigurations": [],
            "compliance_issues": []
        }
        
        # OWASP Top 10 testing
        owasp_results = self.dast_scanner.test_owasp_top10(base_url)
        results["vulnerabilities"].extend(owasp_results)
        
        # Configuration testing
        config_results = self.dast_scanner.test_configuration(base_url)
        results["misconfigurations"] = config_results
        
        # SSL/TLS testing
        ssl_results = self.dast_scanner.test_ssl_configuration(base_url)
        results["ssl_issues"] = ssl_results
        
        return results
```

## Encryption & Cryptography

### Data Encryption

```python
from security_framework.encryption.data_encryption import DataEncryption

class SecureDataHandler:
    def __init__(self):
        self.encryption = DataEncryption()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using AES-256-GCM."""
        return self.encryption.encrypt(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption.decrypt(encrypted_data)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.encryption.hash_password(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return self.encryption.verify_password(password, hashed)
```

### Key Management

```python
from security_framework.encryption.key_management import KeyManager

class SecureKeyManager:
    def __init__(self):
        self.key_manager = KeyManager()
    
    def generate_encryption_key(self) -> str:
        """Generate new encryption key."""
        return self.key_manager.generate_key()
    
    def rotate_keys(self) -> dict:
        """Rotate encryption keys."""
        old_key = self.key_manager.get_current_key()
        new_key = self.key_manager.generate_key()
        
        # Update key in secure storage
        self.key_manager.store_key(new_key)
        
        # Re-encrypt data with new key
        self.key_manager.re_encrypt_data(old_key, new_key)
        
        return {
            "old_key_id": old_key.id,
            "new_key_id": new_key.id,
            "rotation_timestamp": datetime.utcnow()
        }
```

### Digital Signatures

```python
from security_framework.encryption.digital_signatures import DigitalSignature

class DocumentSigner:
    def __init__(self):
        self.signature = DigitalSignature()
    
    def sign_document(self, document: bytes, private_key: str) -> str:
        """Create digital signature for document."""
        signature = self.signature.sign(document, private_key)
        return signature
    
    def verify_signature(self, document: bytes, signature: str, public_key: str) -> bool:
        """Verify document signature."""
        return self.signature.verify(document, signature, public_key)
```

## Access Control & Authorization

### Fine-Grained Permissions

```python
from security_framework.access.permission_manager import PermissionManager

class SecurityAccessControl:
    def __init__(self):
        self.permission_manager = PermissionManager()
    
    def check_permission(self, user_id: int, resource: str, action: str) -> bool:
        """Check if user has permission for specific action on resource."""
        return self.permission_manager.has_permission(user_id, resource, action)
    
    def grant_permission(self, user_id: int, permission: str) -> None:
        """Grant permission to user."""
        self.permission_manager.grant_permission(user_id, permission)
    
    def revoke_permission(self, user_id: int, permission: str) -> None:
        """Revoke permission from user."""
        self.permission_manager.revoke_permission(user_id, permission)
```

### Resource-Based Access Control

```python
from security_framework.access.rbac import ResourceBasedAccessControl

class ResourceAccessControl:
    def __init__(self):
        self.rbac = ResourceBasedAccessControl()
    
    def define_resource_policy(self, resource: str, policy: dict) -> None:
        """Define access policy for resource."""
        self.rbac.set_resource_policy(resource, policy)
    
    def check_resource_access(self, user: dict, resource: str, action: str) -> bool:
        """Check if user can perform action on resource."""
        return self.rbac.evaluate_access(user, resource, action)
```

## Security Monitoring & Incident Response

### Security Event Monitoring

```python
from security_framework.monitoring.security_monitor import SecurityMonitor

class SecurityEventHandler:
    def __init__(self):
        self.monitor = SecurityMonitor()
    
    def log_security_event(self, event_type: str, details: dict) -> None:
        """Log security event for monitoring."""
        self.monitor.log_event({
            "type": event_type,
            "timestamp": datetime.utcnow(),
            "details": details,
            "severity": self._determine_severity(event_type),
            "source_ip": details.get("ip_address"),
            "user_id": details.get("user_id")
        })
    
    def detect_anomalies(self) -> list:
        """Detect security anomalies in recent events."""
        return self.monitor.detect_anomalies()
    
    def generate_alerts(self) -> list:
        """Generate security alerts based on events."""
        return self.monitor.generate_alerts()
```

### Incident Response

```python
from security_framework.incidents.incident_handler import IncidentHandler

class SecurityIncidentResponse:
    def __init__(self):
        self.incident_handler = IncidentHandler()
    
    def create_incident(self, incident_data: dict) -> str:
        """Create new security incident."""
        incident_id = self.incident_handler.create_incident(incident_data)
        
        # Trigger automatic responses
        self._trigger_automatic_response(incident_data)
        
        # Send notifications
        self._send_incident_notifications(incident_id, incident_data)
        
        return incident_id
    
    def escalate_incident(self, incident_id: str) -> None:
        """Escalate security incident."""
        incident = self.incident_handler.get_incident(incident_id)
        
        # Update incident severity
        incident.severity = "HIGH"
        incident.escalated_at = datetime.utcnow()
        
        # Notify security team
        self._notify_security_team(incident)
        
        # Take protective actions
        self._activate_protective_measures(incident)
```

## Compliance & Auditing

### Audit Logging

```python
from security_framework.compliance.audit_logger import AuditLogger

class ComplianceAuditor:
    def __init__(self):
        self.audit_logger = AuditLogger()
    
    def log_data_access(self, user_id: int, resource: str, action: str) -> None:
        """Log data access for compliance."""
        self.audit_logger.log({
            "event_type": "data_access",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "timestamp": datetime.utcnow(),
            "ip_address": self._get_client_ip(),
            "user_agent": self._get_user_agent()
        })
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime) -> dict:
        """Generate compliance report for specified period."""
        return self.audit_logger.generate_report(start_date, end_date)
```

### GDPR Compliance

```python
from security_framework.compliance.gdpr import GDPRCompliance

class DataProtectionCompliance:
    def __init__(self):
        self.gdpr = GDPRCompliance()
    
    def handle_data_request(self, user_id: int, request_type: str) -> dict:
        """Handle GDPR data protection requests."""
        if request_type == "access":
            return self.gdpr.export_user_data(user_id)
        elif request_type == "deletion":
            return self.gdpr.delete_user_data(user_id)
        elif request_type == "portability":
            return self.gdpr.export_portable_data(user_id)
        else:
            raise ValueError("Invalid request type")
    
    def check_consent(self, user_id: int, purpose: str) -> bool:
        """Check if user has given consent for specific purpose."""
        return self.gdpr.has_consent(user_id, purpose)
```

## Container Security

### Docker Security

```dockerfile
# Dockerfile.secure
FROM python:3.11-slim

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set secure permissions
WORKDIR /app
COPY --chown=appuser:appuser . .

# Install dependencies
RUN pip install --no-cache-dir -e .[security]

# Security configurations
RUN chmod -R 750 /app && \
    chmod 640 /app/security/certificates/* && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py

# Run application
CMD ["python", "-m", "security_framework.main"]
```

### Kubernetes Security

```yaml
# k8s-security.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  annotations:
    container.apparmor.security.beta.kubernetes.io/app: runtime/default
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: secure-app:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "512Mi"
        cpu: "500m"
      requests:
        memory: "256Mi"
        cpu: "250m"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: var-tmp
      mountPath: /var/tmp
  volumes:
  - name: tmp
    emptyDir: {}
  - name: var-tmp
    emptyDir: {}
```

## Security Testing

### Penetration Testing

```python
from security_framework.testing.pentest import PenetrationTester

class SecurityTesting:
    def __init__(self):
        self.pentest = PenetrationTester()
    
    def run_vulnerability_scan(self, target: str) -> dict:
        """Run automated vulnerability scan."""
        results = {}
        
        # Network scanning
        results["network"] = self.pentest.scan_network(target)
        
        # Web application testing
        results["web_app"] = self.pentest.test_web_application(target)
        
        # API security testing
        results["api"] = self.pentest.test_api_security(target)
        
        # Configuration testing
        results["config"] = self.pentest.test_configuration(target)
        
        return results
```

### Security Test Automation

```python
import pytest
from security_framework.testing.security_tests import SecurityTestSuite

class TestApplicationSecurity:
    def __init__(self):
        self.security_tests = SecurityTestSuite()
    
    def test_sql_injection_protection(self):
        """Test SQL injection protection."""
        payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users"
        ]
        
        for payload in payloads:
            response = self.security_tests.test_sql_injection(payload)
            assert response.status_code != 200
            assert "error" in response.json()
    
    def test_xss_protection(self):
        """Test XSS protection."""
        payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for payload in payloads:
            response = self.security_tests.test_xss(payload)
            assert payload not in response.text
    
    def test_csrf_protection(self):
        """Test CSRF protection."""
        response = self.security_tests.test_csrf()
        assert "X-CSRFToken" in response.headers
        assert response.cookies.get("csrftoken") is not None
```

## Security Automation Scripts

### Security Scan Script

```bash
#!/bin/bash
# scripts/security_scan.sh

set -e

echo "🔒 Running comprehensive security scan..."

# Dependency vulnerability scan
echo "📦 Scanning dependencies..."
safety check
bandit -r src/

# Static analysis
echo "🔍 Running static analysis..."
semgrep --config=auto src/

# License compliance
echo "📄 Checking license compliance..."
pip-licenses --format=json > build/licenses.json

# Secrets detection
echo "🔐 Scanning for secrets..."
truffleHog filesystem src/ --json > build/secrets-scan.json

# Container security
if [ -f "Dockerfile" ]; then
    echo "🐳 Scanning container..."
    trivy fs .
fi

# Generate security report
echo "📊 Generating security report..."
python scripts/generate_security_report.py

echo "✅ Security scan completed!"
```

### Compliance Check Script

```python
#!/usr/bin/env python3
# scripts/compliance_check.py

import json
from pathlib import Path
from security_framework.compliance.checker import ComplianceChecker

def main():
    """Run compliance checks."""
    checker = ComplianceChecker()
    
    # OWASP Top 10 compliance
    owasp_results = checker.check_owasp_compliance()
    
    # GDPR compliance
    gdpr_results = checker.check_gdpr_compliance()
    
    # Security framework compliance
    framework_results = checker.check_security_framework()
    
    # Generate compliance report
    report = {
        "owasp": owasp_results,
        "gdpr": gdpr_results,
        "security_framework": framework_results,
        "overall_score": checker.calculate_overall_score(),
        "recommendations": checker.get_recommendations()
    }
    
    # Save report
    report_path = Path("build/compliance-report.json")
    report_path.write_text(json.dumps(report, indent=2))
    
    print(f"Compliance report saved to {report_path}")
    
    # Exit with error if compliance issues found
    if report["overall_score"] < 80:
        print("❌ Compliance issues found!")
        exit(1)
    else:
        print("✅ All compliance checks passed!")

if __name__ == "__main__":
    main()
```

## Configuration

### Security Configuration

```yaml
# security.yml
security:
  authentication:
    jwt:
      secret_key: "${JWT_SECRET_KEY}"
      algorithm: "HS256"
      access_token_expire_minutes: 30
      refresh_token_expire_days: 30
    
    mfa:
      enabled: true
      totp_validity_window: 1
      backup_codes_count: 10
  
  authorization:
    rbac:
      enabled: true
      cache_permissions: true
      cache_ttl: 3600
    
    abac:
      enabled: false
      policy_engine: "opa"
  
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    backup_key_count: 3
  
  monitoring:
    security_events: true
    audit_logging: true
    anomaly_detection: true
    retention_days: 365
  
  compliance:
    gdpr:
      enabled: true
      data_retention_days: 1095
      auto_anonymization: true
    
    audit:
      enabled: true
      log_all_access: true
      log_failures_only: false
  
  scanning:
    dependency_scan: true
    static_analysis: true
    dynamic_testing: false
    schedule: "0 2 * * *"  # Daily at 2 AM
```

## Best Practices

1. **Defense in Depth**: Multiple layers of security controls
2. **Zero Trust**: Never trust, always verify
3. **Least Privilege**: Minimal necessary permissions
4. **Secure by Default**: Security-first configuration
5. **Regular Updates**: Keep dependencies current
6. **Monitoring**: Continuous security monitoring
7. **Incident Response**: Prepared response procedures
8. **Compliance**: Meet regulatory requirements

## License

MIT License - see LICENSE file for details