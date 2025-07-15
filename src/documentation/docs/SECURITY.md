# Enhanced Web UI Security Features

This document describes the comprehensive security features implemented in Pynomaly's web interface, providing enterprise-grade protection and compliance capabilities.

## 🔒 Security Architecture Overview

Pynomaly implements a multi-layered security architecture with the following components:

- **Multi-Factor Authentication (MFA)** with TOTP and backup codes
- **Role-Based Access Control (RBAC)** with granular permissions
- **OAuth2 and SAML Integration** for single sign-on
- **Real-time Security Monitoring** with threat analysis
- **Web Application Firewall (WAF)** with pattern matching
- **Advanced Rate Limiting** with burst protection
- **Comprehensive Audit Logging** with compliance framework support
- **Enhanced Session Management** and security headers

## 🛡️ Security Features

### Multi-Factor Authentication (MFA)

#### TOTP (Time-based One-Time Password)
- QR code generation for authenticator apps
- Support for Google Authenticator, Authy, and other TOTP apps
- 30-second time window with clock drift tolerance
- Backup codes for account recovery

#### Implementation
```python
from pynomaly.presentation.web.enhanced_auth import get_auth_service

auth_service = get_auth_service()
# Setup MFA for user
qr_code_url = auth_service.setup_mfa(user_id)
# Verify MFA code
is_valid = auth_service.verify_mfa(user_id, code)
```

#### Configuration
```python
MFA_SETTINGS = {
    "enabled": True,
    "issuer": "Pynomaly",
    "backup_codes_count": 10,
    "qr_code_size": 200,
    "time_window": 30
}
```

### Role-Based Access Control (RBAC)

#### Roles and Permissions
- **Admin**: Full system access, user management, system configuration
- **Data Analyst**: Dataset management, detector configuration, results viewing
- **Viewer**: Read-only access to dashboards and results

#### Permission System
```python
class Permission(Enum):
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Dataset permissions
    DATASET_CREATE = "dataset:create"
    DATASET_READ = "dataset:read"
    DATASET_UPDATE = "dataset:update"
    DATASET_DELETE = "dataset:delete"
    
    # Detector permissions
    DETECTOR_CREATE = "detector:create"
    DETECTOR_READ = "detector:read"
    DETECTOR_UPDATE = "detector:update"
    DETECTOR_DELETE = "detector:delete"
```

#### Usage Example
```python
from pynomaly.presentation.web.enhanced_auth import require_permission

@require_permission(Permission.DATASET_CREATE)
async def create_dataset(request: Request):
    # Only users with dataset creation permission can access
    pass
```

### OAuth2 and SAML Integration

#### Supported Providers
- Google OAuth2
- Microsoft Azure AD
- GitHub OAuth2
- Custom SAML Identity Providers

#### OAuth2 Configuration
```python
OAUTH2_PROVIDERS = {
    "google": {
        "client_id": "your-google-client-id",
        "client_secret": "your-google-client-secret",
        "authorize_url": "https://accounts.google.com/o/oauth2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scopes": ["openid", "email", "profile"]
    }
}
```

#### SAML Configuration
```python
SAML_SETTINGS = {
    "entity_id": "pynomaly-sp",
    "assertion_consumer_service": {
        "url": "https://your-domain.com/auth/saml/acs",
        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
    },
    "single_logout_service": {
        "url": "https://your-domain.com/auth/saml/sls",
        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
    }
}
```

### Real-time Security Monitoring

#### Security Dashboard
The security monitoring dashboard provides real-time visibility into:
- Active security alerts and threats
- Request patterns and anomalies
- Failed authentication attempts
- Rate limiting violations
- WAF blocking events

#### WebSocket Integration
```javascript
// Connect to security monitoring WebSocket
const ws = new WebSocket('ws://localhost:8000/api/security/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'alert') {
        displaySecurityAlert(data.data);
    } else if (data.type === 'metrics') {
        updateSecurityMetrics(data.data);
    }
};
```

#### Alert Types
- Brute force attacks
- DDoS attempts
- Suspicious IP addresses
- Multiple failed logins
- Security policy violations
- Potential data breaches

### Web Application Firewall (WAF)

#### Protection Against
- SQL Injection attacks
- Cross-Site Scripting (XSS)
- Path traversal attempts
- Command injection
- Malicious user agents
- Known attack patterns

#### Pattern Detection
```python
# SQL Injection patterns
sql_patterns = [
    r"(\bUNION\b.*\bSELECT\b)",
    r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b.*\bOR\b.*=.*)",
    r"('.*OR.*'.*'.*=.*')",
]

# XSS patterns
xss_patterns = [
    r"(<script[^>]*>.*?</script>)",
    r"(javascript:)",
    r"(onload\s*=)",
    r"(alert\s*\()",
]
```

#### Configuration
```python
WAF_SETTINGS = {
    "enabled": True,
    "block_malicious_ips": True,
    "threat_threshold": 100,
    "log_all_requests": False,
    "whitelist_ips": ["127.0.0.1", "::1"]
}
```

### Advanced Rate Limiting

#### Endpoint-Specific Rules
```python
RATE_LIMIT_RULES = {
    "/api/auth/login": RateLimitRule(5, 20, 100, 3, 60, 900),
    "/api/auth/register": RateLimitRule(3, 10, 50, 2, 60, 1800),
    "/api/auth/reset-password": RateLimitRule(3, 10, 20, 1, 60, 3600),
    "/api/upload": RateLimitRule(10, 50, 200, 5, 60, 300),
}
```

#### Burst Protection
- Configurable burst limits per endpoint
- Adaptive blocking based on patterns
- IP-based tracking with cleanup
- Exponential backoff for repeated violations

#### Headers
Rate limit information is provided in HTTP headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

### Comprehensive Audit Logging

#### Compliance Framework Support
- SOX (Sarbanes-Oxley Act)
- GDPR (General Data Protection Regulation)
- HIPAA (Health Insurance Portability and Accountability Act)
- PCI-DSS (Payment Card Industry Data Security Standard)
- SOC2 (Service Organization Control 2)
- ISO27001 (Information Security Management)

#### Event Types
```python
class AuditEventType(Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    SECURITY_VIOLATION = "security_violation"
```

#### Usage
```python
from pynomaly.presentation.web.audit_logging import log_user_login

# Log user login
log_user_login(
    user_id="user123",
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
    session_id="session_abc123"
)
```

### Enhanced Session Management

#### Security Features
- Secure session tokens with cryptographic signing
- Session timeout and automatic renewal
- Session fixation prevention
- Cross-Site Request Forgery (CSRF) protection
- Secure cookie attributes

#### Configuration
```python
SESSION_SETTINGS = {
    "secret_key": "your-secret-key",
    "max_age": 3600,  # 1 hour
    "secure": True,   # HTTPS only
    "httponly": True, # No JavaScript access
    "samesite": "strict"
}
```

## 🔧 Security Headers

### Content Security Policy (CSP)
```
Content-Security-Policy: default-src 'self'; 
script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; 
style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com; 
font-src 'self' https://fonts.gstatic.com; 
img-src 'self' data: https:; 
connect-src 'self' ws: wss:; 
frame-ancestors 'none'; 
base-uri 'self'; 
form-action 'self'; 
object-src 'none'; 
upgrade-insecure-requests
```

### Additional Security Headers
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Resource-Policy: same-origin
```

## 🧪 Testing

### Integration Tests
Comprehensive Playwright-based integration tests covering:
- Multi-factor authentication flows
- Role-based access control validation
- OAuth2 and SAML integration
- Security monitoring dashboard
- Rate limiting enforcement
- WAF protection
- Audit logging verification

### Unit Tests
Extensive unit test coverage for:
- Authentication service components
- Authorization middleware
- Rate limiting algorithms
- WAF pattern matching
- Session management
- Audit logging functionality

### Security Testing
- Penetration testing scenarios
- Vulnerability scanning
- Authentication bypass attempts
- Authorization escalation tests
- Input validation testing
- Session management security

## 📋 Configuration

### Environment Variables
```bash
# Authentication
PYNOMALY_AUTH_ENABLED=true
PYNOMALY_JWT_SECRET_KEY=your-jwt-secret
PYNOMALY_SESSION_SECRET_KEY=your-session-secret

# MFA
PYNOMALY_MFA_ENABLED=true
PYNOMALY_MFA_ISSUER=Pynomaly

# OAuth2
PYNOMALY_OAUTH2_GOOGLE_CLIENT_ID=your-client-id
PYNOMALY_OAUTH2_GOOGLE_CLIENT_SECRET=your-client-secret

# SAML
PYNOMALY_SAML_ENTITY_ID=pynomaly-sp
PYNOMALY_SAML_X509_CERT=path/to/cert.pem

# Security
PYNOMALY_WAF_ENABLED=true
PYNOMALY_RATE_LIMITING_ENABLED=true
PYNOMALY_SECURITY_MONITORING_ENABLED=true

# Audit Logging
PYNOMALY_AUDIT_LOGGING_ENABLED=true
PYNOMALY_AUDIT_COMPLIANCE_FRAMEWORKS=SOX,GDPR,HIPAA
```

### Configuration File
```yaml
# config/security.yml
security:
  authentication:
    enabled: true
    mfa:
      enabled: true
      issuer: "Pynomaly"
      backup_codes_count: 10
    oauth2:
      providers:
        - google
        - microsoft
    saml:
      entity_id: "pynomaly-sp"
  
  authorization:
    rbac:
      enabled: true
      default_role: "viewer"
  
  waf:
    enabled: true
    threat_threshold: 100
  
  rate_limiting:
    enabled: true
    default_rules:
      requests_per_minute: 60
      requests_per_hour: 1000
      burst_limit: 10
  
  monitoring:
    enabled: true
    websocket_enabled: true
    alert_thresholds:
      failed_logins_per_minute: 10
      requests_per_minute: 1000
  
  audit_logging:
    enabled: true
    compliance_frameworks:
      - SOX
      - GDPR
      - HIPAA
```

## 🚀 Deployment

### Docker Configuration
```dockerfile
# Security-focused Docker configuration
FROM python:3.11-slim

# Security hardening
RUN adduser --disabled-password --gecos '' pynomaly
USER pynomaly

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=pynomaly:pynomaly . /app
WORKDIR /app

# Security environment
ENV PYNOMALY_AUTH_ENABLED=true
ENV PYNOMALY_WAF_ENABLED=true
ENV PYNOMALY_RATE_LIMITING_ENABLED=true

EXPOSE 8000
CMD ["uvicorn", "pynomaly.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Security
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-web
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: pynomaly
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        env:
        - name: PYNOMALY_AUTH_ENABLED
          value: "true"
        - name: PYNOMALY_WAF_ENABLED
          value: "true"
```

## 📊 Monitoring and Alerting

### Security Metrics
- Authentication success/failure rates
- Active user sessions
- Security violations per minute
- Blocked requests by type
- Geographic distribution of requests

### Alert Conditions
- Brute force attack detection
- Unusual traffic patterns
- High rate of security violations
- Multiple failed authentication attempts
- Suspicious IP addresses

### Integration
- Prometheus metrics export
- Grafana dashboard templates
- Slack/Teams notification integration
- PagerDuty escalation policies

## 🔍 Troubleshooting

### Common Issues

#### MFA Setup Problems
```bash
# Check MFA configuration
python -c "from pynomaly.presentation.web.enhanced_auth import get_auth_service; print(get_auth_service().mfa_enabled)"

# Reset MFA for user
pynomaly auth reset-mfa --user-id user123
```

#### Rate Limiting Issues
```bash
# Check rate limit status
curl -I http://localhost:8000/api/v1/datasets
# Look for X-RateLimit-* headers

# Clear rate limits for IP
pynomaly security clear-rate-limits --ip 192.168.1.100
```

#### WAF False Positives
```bash
# Check WAF logs
tail -f logs/security.log | grep WAF

# Whitelist IP temporarily
pynomaly security whitelist-ip --ip 192.168.1.100 --duration 3600
```

### Log Analysis
```bash
# Security event analysis
grep "SECURITY_VIOLATION" logs/audit.log | jq .

# Authentication analysis
grep "USER_LOGIN" logs/audit.log | jq '.ip_address' | sort | uniq -c
```

## 📚 References

- [OWASP Application Security Verification Standard](https://owasp.org/www-project-application-security-verification-standard/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [RFC 6238 - TOTP Algorithm](https://tools.ietf.org/html/rfc6238)
- [OAuth 2.0 Security Best Practices](https://tools.ietf.org/html/draft-ietf-oauth-security-topics)
- [SAML 2.0 Technical Overview](https://www.oasis-open.org/committees/download.php/27819/sstc-saml-tech-overview-2.0-cd-02.pdf)

---

**Last Updated**: July 10, 2025  
**Version**: 1.0  
**Status**: Production Ready