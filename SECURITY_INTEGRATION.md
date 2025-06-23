# Pynomaly Advanced Security Features Integration Guide

This document provides a comprehensive guide for integrating and using the advanced security features and audit logging system implemented in Pynomaly.

## Overview

The advanced security framework provides:

- **Input Sanitization & Validation**: Comprehensive protection against XSS, injection attacks, and malicious input
- **SQL Injection Protection**: Advanced detection and prevention of SQL injection attacks
- **Data Encryption**: Field-level and data-at-rest encryption capabilities
- **Security Headers**: Comprehensive security headers enforcement
- **Audit Logging**: Complete audit trail with compliance support
- **Security Monitoring**: Real-time threat detection and alerting
- **User Action Tracking**: Detailed logging of all user actions and API access

## Architecture

The security framework follows clean architecture principles and integrates seamlessly with the existing Pynomaly infrastructure:

```
├── infrastructure/security/
│   ├── input_sanitizer.py          # Input validation and sanitization
│   ├── sql_protection.py           # SQL injection protection
│   ├── encryption.py               # Data encryption services
│   ├── security_headers.py         # Security headers middleware
│   ├── audit_logger.py             # Comprehensive audit logging
│   ├── security_monitor.py         # Threat detection and monitoring
│   ├── user_tracking.py            # User action tracking
│   └── middleware_integration.py   # FastAPI integration helpers
```

## Quick Start

### 1. Enable Security Features in Configuration

Update your `.env` file or configuration:

```bash
# Enable security features
PYNOMALY_SECURITY__ENABLE_AUDIT_LOGGING=true
PYNOMALY_SECURITY__ENABLE_SECURITY_MONITORING=true
PYNOMALY_SECURITY__SANITIZATION_LEVEL=moderate
PYNOMALY_SECURITY__ENCRYPTION_ALGORITHM=fernet
PYNOMALY_SECURITY__SECURITY_HEADERS_ENABLED=true

# Set encryption master key (IMPORTANT: Store securely in production)
PYNOMALY_MASTER_KEY=your_base64_encoded_master_key_here
```

### 2. Initialize Security in FastAPI Application

```python
from fastapi import FastAPI
from pynomaly.infrastructure.security import setup_security_middleware, add_security_endpoints

app = FastAPI()

# Setup complete security middleware stack
security_stack = setup_security_middleware(
    app=app,
    enable_security_headers=True,
    enable_user_tracking=True,
    enable_audit_logging=True,
    enable_security_monitoring=True,
    development_mode=False  # Set to True for development
)

# Add security API endpoints
add_security_endpoints(
    app=app,
    security_monitor=security_stack.security_monitor,
    user_tracker=security_stack.user_tracker
)
```

### 3. Using Security Components

#### Input Sanitization

```python
from pynomaly.infrastructure.security import InputSanitizer, SanitizationConfig

# Create sanitizer with custom config
config = SanitizationConfig(
    level="strict",
    max_length=5000,
    allow_html=False
)
sanitizer = InputSanitizer(config)

# Sanitize input data
user_input = "<script>alert('xss')</script>Hello World"
clean_input = sanitizer.sanitize_string(user_input)
# Result: "Hello World" (script removed)

# Sanitize entire dictionaries
data = {
    "name": "<script>evil</script>John",
    "email": "john@example.com",
    "comment": "This is a test with 'quotes'"
}
clean_data = sanitizer.sanitize_dict(data)
```

#### SQL Injection Protection

```python
from pynomaly.infrastructure.security import SQLInjectionProtector, SafeQueryBuilder

# Analyze query for injection risks
protector = SQLInjectionProtector()
query = "SELECT * FROM users WHERE id = ? AND name = ?"
analysis = protector.analyze_query(query, {"id": 1, "name": "John"})

if not analysis.is_safe:
    print(f"Security threats detected: {analysis.detected_threats}")

# Build safe queries
builder = SafeQueryBuilder()
query, params = builder.build_select(
    table_name="users",
    columns=["id", "name", "email"],
    where_conditions={"active": True, "role": "user"},
    limit=100
)
```

#### Data Encryption

```python
from pynomaly.infrastructure.security import DataEncryption

# Initialize encryption service
encryption = DataEncryption()

# Encrypt sensitive data
sensitive_data = {"ssn": "123-45-6789", "credit_card": "4111-1111-1111-1111"}
encrypted = encryption.encrypt_sensitive_data(sensitive_data)

# Decrypt data
decrypted = encryption.decrypt_sensitive_data(encrypted, dict)

# Field-level encryption for database records
encryption.field_encryption.register_encrypted_field("users", "ssn")
encryption.field_encryption.register_encrypted_field("users", "credit_card")

# Encrypt record before saving
record = {"id": 1, "name": "John", "ssn": "123-45-6789"}
encrypted_record = encryption.field_encryption.encrypt_record("users", record)

# Decrypt record after loading
decrypted_record = encryption.field_encryption.decrypt_record("users", encrypted_record)
```

#### Audit Logging

```python
from pynomaly.infrastructure.security import (
    AuditLogger, SecurityEventType, AuditLevel, audit_context
)

# Initialize audit logger
audit_logger = AuditLogger()

# Log security events
audit_logger.log_security_event(
    event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
    message="User logged in successfully",
    level=AuditLevel.INFO,
    details={"user_id": "123", "ip_address": "192.168.1.100"},
    compliance_standards=["SOX", "GDPR"]
)

# Use audit context for correlated events
with audit_context(
    correlation_id="req_12345",
    user_id="user_123",
    ip_address="192.168.1.100"
):
    # All audit events within this context will be correlated
    audit_logger.log_audit_event(
        event_type="data_access",
        action="read",
        resource="user_profile",
        message="User accessed profile data"
    )

# Decorator for automatic action auditing
@audit_action("update_profile", "user_profile")
async def update_user_profile(user_id: str, data: dict):
    # Function implementation
    pass
```

#### Security Monitoring

```python
from pynomaly.infrastructure.security import SecurityMonitor, ThreatLevel

# Initialize security monitor
monitor = SecurityMonitor()
await monitor.start_monitoring()

# Get security status
summary = monitor.get_security_summary()
print(f"Active alerts: {summary['active_alerts_count']}")

# Get active alerts
high_threat_alerts = monitor.get_active_alerts(threat_level=ThreatLevel.HIGH)

# Acknowledge alert
monitor.acknowledge_alert("alert_id_123")

# Custom alert handler
def handle_critical_alert(alert):
    if alert.threat_level == ThreatLevel.CRITICAL:
        # Send email notification, page security team, etc.
        send_security_notification(alert)

monitor.register_alert_handler(handle_critical_alert)
```

## Security API Endpoints

The security framework automatically adds the following API endpoints:

### GET /api/security/status
Get security monitoring status and summary statistics.

```json
{
  "monitoring_enabled": true,
  "active_alerts_count": 3,
  "recent_alerts": {
    "HIGH": 2,
    "MEDIUM": 1
  },
  "recent_metrics": {
    "failed_logins": 15,
    "successful_logins": 234,
    "injection_attempts": 2
  }
}
```

### GET /api/security/alerts
Get active security alerts with optional filtering.

Query parameters:
- `threat_level`: Filter by threat level (LOW, MEDIUM, HIGH, CRITICAL)

### POST /api/security/alerts/{alert_id}/acknowledge
Acknowledge a security alert.

### GET /api/security/users/{user_id}/activity
Get user activity summary for the specified time period.

Query parameters:
- `hours`: Number of hours to look back (default: 24)

## Configuration Options

### Security Settings

```python
class SecuritySettings(BaseModel):
    # Input sanitization
    sanitization_level: str = "moderate"  # strict, moderate, permissive
    max_input_length: int = 10000
    allow_html: bool = False
    
    # Encryption
    encryption_algorithm: str = "fernet"  # fernet, aes_gcm, aes_cbc
    encryption_key_length: int = 32
    enable_key_rotation: bool = True
    key_rotation_days: int = 90
    
    # Audit logging
    enable_audit_logging: bool = True
    enable_compliance_logging: bool = False
    audit_retention_days: int = 2555  # 7 years
    
    # Security monitoring
    enable_security_monitoring: bool = True
    threat_detection_enabled: bool = True
    
    # Rate limiting
    brute_force_max_attempts: int = 5
    brute_force_time_window: int = 300  # 5 minutes
    
    # Headers and CORS
    security_headers_enabled: bool = True
    csp_enabled: bool = True
    hsts_enabled: bool = True
```

### Environment Variables

```bash
# Security configuration
PYNOMALY_SECURITY__SANITIZATION_LEVEL=moderate
PYNOMALY_SECURITY__MAX_INPUT_LENGTH=10000
PYNOMALY_SECURITY__ALLOW_HTML=false
PYNOMALY_SECURITY__ENCRYPTION_ALGORITHM=fernet
PYNOMALY_SECURITY__ENABLE_AUDIT_LOGGING=true
PYNOMALY_SECURITY__ENABLE_COMPLIANCE_LOGGING=false
PYNOMALY_SECURITY__ENABLE_SECURITY_MONITORING=true
PYNOMALY_SECURITY__BRUTE_FORCE_MAX_ATTEMPTS=5
PYNOMALY_SECURITY__SECURITY_HEADERS_ENABLED=true

# Master encryption key (store securely!)
PYNOMALY_MASTER_KEY=your_base64_encoded_key_here
```

## Threat Detection

The security framework includes several built-in threat detectors:

### 1. Brute Force Detector
Detects multiple failed login attempts from the same IP address.

### 2. Anomalous Access Detector
Detects unusual access patterns for users (new IPs, unusual times, etc.).

### 3. Injection Attack Detector
Detects SQL injection and other injection attack attempts.

### Custom Threat Detectors

You can implement custom threat detectors:

```python
from pynomaly.infrastructure.security import ThreatDetector, SecurityAlert, ThreatLevel

class CustomThreatDetector(ThreatDetector):
    def __init__(self):
        super().__init__("custom_threat")
    
    async def analyze(self, event_data: dict) -> Optional[SecurityAlert]:
        # Custom threat detection logic
        if self._detect_custom_threat(event_data):
            return SecurityAlert(
                alert_id=f"custom_{int(time.time())}",
                alert_type="custom_threat",
                threat_level=ThreatLevel.HIGH,
                title="Custom Threat Detected",
                description="Description of the threat",
                timestamp=datetime.now(timezone.utc),
                indicators={"custom_indicator": "value"},
                recommended_actions=["Take action A", "Take action B"]
            )
        return None
    
    def _detect_custom_threat(self, event_data: dict) -> bool:
        # Implement custom detection logic
        return False

# Register custom detector
monitor = SecurityMonitor()
monitor.register_detector(CustomThreatDetector())
```

## Compliance Support

The audit logging system supports various compliance standards:

- **SOX** (Sarbanes-Oxley): Financial reporting controls
- **GDPR** (General Data Protection Regulation): Data privacy
- **HIPAA** (Health Insurance Portability and Accountability Act): Healthcare data
- **PCI DSS** (Payment Card Industry Data Security Standard): Payment data
- **ISO27001**: Information security management
- **NIST**: Cybersecurity framework

Example compliance logging:

```python
from pynomaly.infrastructure.security import ComplianceStandard

audit_logger.log_security_event(
    event_type=SecurityEventType.DATA_ACCESS_READ,
    message="Patient data accessed",
    compliance_standards=[ComplianceStandard.HIPAA, ComplianceStandard.GDPR],
    details={"patient_id": "P12345", "accessed_fields": ["name", "dob"]}
)
```

## Production Deployment

### Security Checklist

1. **Environment Variables**:
   - [ ] Set strong `PYNOMALY_MASTER_KEY`
   - [ ] Configure appropriate sanitization level
   - [ ] Enable compliance logging if required
   - [ ] Set production security headers

2. **Database Security**:
   - [ ] Register encrypted fields for sensitive data
   - [ ] Enable SQL injection protection
   - [ ] Use parameterized queries

3. **Monitoring**:
   - [ ] Configure alert handlers
   - [ ] Set up external alerting (email, Slack, etc.)
   - [ ] Monitor security metrics
   - [ ] Regular security log review

4. **Network Security**:
   - [ ] Enable HTTPS (security headers will enforce HSTS)
   - [ ] Configure proper CORS settings
   - [ ] Set up rate limiting

### Log Management

Audit logs are structured JSON and can be forwarded to external systems:

```python
# Example: Forward to external SIEM
def forward_to_siem(event):
    if isinstance(event, SecurityEvent):
        # Send to SIEM system
        siem_client.send_event(event.dict())

audit_logger.register_compliance_handler(
    ComplianceStandard.SOX,
    forward_to_siem
)
```

## Testing

The security framework includes comprehensive test coverage. To test security features:

```python
# Test input sanitization
def test_xss_protection():
    sanitizer = InputSanitizer()
    malicious_input = "<script>alert('xss')</script>Hello"
    clean_output = sanitizer.sanitize_string(malicious_input)
    assert "<script>" not in clean_output

# Test SQL injection protection
def test_sql_injection_detection():
    protector = SQLInjectionProtector()
    malicious_query = "SELECT * FROM users WHERE id = '1 OR 1=1'"
    analysis = protector.analyze_query(malicious_query)
    assert not analysis.is_safe
    assert "Tautology injection" in analysis.detected_threats

# Test audit logging
def test_audit_logging():
    audit_logger = AuditLogger()
    audit_logger.log_security_event(
        SecurityEventType.AUTH_LOGIN_SUCCESS,
        "Test login",
        details={"user_id": "test"}
    )
    # Verify log was created
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all security dependencies are installed
2. **Encryption Errors**: Verify `PYNOMALY_MASTER_KEY` is set correctly
3. **Performance Impact**: Adjust sanitization level if needed
4. **False Positives**: Tune threat detection thresholds

### Debug Mode

Enable debug logging for security components:

```python
import logging
logging.getLogger("pynomaly.security").setLevel(logging.DEBUG)
```

### Health Checks

Monitor security component health:

```python
# Check if security services are running
status = security_monitor.get_security_summary()
print(f"Security monitoring: {'enabled' if status['monitoring_enabled'] else 'disabled'}")
```

## Performance Considerations

- **Input Sanitization**: Minimal overhead, configurable levels
- **SQL Protection**: Query analysis adds <1ms per query
- **Encryption**: Modern algorithms with hardware acceleration
- **Audit Logging**: Asynchronous logging prevents blocking
- **Monitoring**: Background processing with configurable thresholds

## Security Best Practices

1. **Regular Updates**: Keep security configurations updated
2. **Log Review**: Regularly review audit logs and alerts
3. **Incident Response**: Have procedures for security alerts
4. **Testing**: Regular security testing and penetration testing
5. **Training**: Security awareness for development team

## Support

For security-related questions or incident reporting:

1. Check the audit logs for detailed information
2. Review security monitoring alerts
3. Consult the comprehensive documentation
4. Contact the security team for critical issues

This advanced security framework provides enterprise-grade protection while maintaining the clean architecture principles of Pynomaly. All components are designed to be modular, configurable, and production-ready.