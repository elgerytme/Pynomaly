# Security Configuration Examples

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🔒 [Security](README.md) > 📄 Configuration Examples

---

## Overview

This document provides practical security configuration examples for different deployment scenarios, implementing the security hardening features including TLS enforcement, checksum validation, and client-side encryption.

## Environment-Specific Configurations

### Development Environment

```python
# config/security/development.py
from pynomaly.infrastructure.security.security_hardening import (
    SecurityHardeningConfig,
    TLSVersion,
    ChecksumAlgorithm
)

DEVELOPMENT_SECURITY_CONFIG = SecurityHardeningConfig(
    # TLS Configuration - Relaxed for development
    enforce_tls=False,  # Allow HTTP for local development
    minimum_tls_version=TLSVersion.TLS_1_2,
    verify_ssl_certificates=False,  # Self-signed certs OK
    
    # SDK Version Requirements - Relaxed
    enforce_minimum_sdk_version=False,
    minimum_sdk_version="0.9.0",
    minimum_python_version="3.8",
    
    # Checksum Validation - Enabled for testing
    enable_checksum_validation=True,
    checksum_algorithm=ChecksumAlgorithm.SHA256,
    enable_etag_validation=True,
    enable_md5_validation=True,
    
    # Client-side Encryption - Optional
    enable_client_side_encryption=True,
    require_encrypted_uploads=False,
    
    # Rate Limiting - Generous for development
    max_requests_per_minute=1000,
    max_upload_size_mb=500,
    
    # Audit Settings
    enable_security_audit=True,
    audit_failed_attempts=True,
    audit_checksum_failures=True
)
```

### Staging Environment

```python
# config/security/staging.py
STAGING_SECURITY_CONFIG = SecurityHardeningConfig(
    # TLS Configuration - Enforced
    enforce_tls=True,
    minimum_tls_version=TLSVersion.TLS_1_2,
    verify_ssl_certificates=True,
    
    # SDK Version Requirements - Enforced
    enforce_minimum_sdk_version=True,
    minimum_sdk_version="1.0.0",
    minimum_python_version="3.8",
    
    # Checksum Validation - Fully enabled
    enable_checksum_validation=True,
    checksum_algorithm=ChecksumAlgorithm.SHA256,
    enable_etag_validation=True,
    enable_md5_validation=True,
    
    # Client-side Encryption - Required
    enable_client_side_encryption=True,
    require_encrypted_uploads=True,
    
    # Rate Limiting - Production-like
    max_requests_per_minute=200,
    max_upload_size_mb=100,
    
    # Audit Settings - Full logging
    enable_security_audit=True,
    audit_failed_attempts=True,
    audit_checksum_failures=True
)
```

### Production Environment

```python
# config/security/production.py
PRODUCTION_SECURITY_CONFIG = SecurityHardeningConfig(
    # TLS Configuration - Strict
    enforce_tls=True,
    minimum_tls_version=TLSVersion.TLS_1_3,  # Require TLS 1.3
    verify_ssl_certificates=True,
    ssl_cert_path="/etc/ssl/certs/pynomaly.crt",
    ssl_key_path="/etc/ssl/private/pynomaly.key",
    
    # SDK Version Requirements - Strict
    enforce_minimum_sdk_version=True,
    minimum_sdk_version="1.0.0",
    minimum_python_version="3.9",  # Higher minimum for production
    
    # Checksum Validation - Maximum security
    enable_checksum_validation=True,
    checksum_algorithm=ChecksumAlgorithm.SHA512,  # Stronger algorithm
    enable_etag_validation=True,
    enable_md5_validation=True,
    
    # Client-side Encryption - Mandatory
    enable_client_side_encryption=True,
    require_encrypted_uploads=True,
    
    # Security Headers - Production hardened
    security_headers={
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    },
    
    # Rate Limiting - Conservative
    max_requests_per_minute=100,
    max_upload_size_mb=50,
    
    # Audit Settings - Comprehensive
    enable_security_audit=True,
    audit_failed_attempts=True,
    audit_checksum_failures=True
)
```

## SDK Client Configuration

### Secure Client Setup

```python
# SDK client with security hardening
from pynomaly.presentation.sdk import PynomaliClient
from pynomaly.presentation.sdk.config import SDKConfig, ClientConfig

# Configure secure client
client_config = ClientConfig(
    timeout=30.0,
    verify_ssl=True,
    enforce_tls=True,
    minimum_tls_version="TLSv1.3",
    enable_checksum_validation=True,
    enable_client_side_encryption=True,
    encryption_key="your-encryption-key-id"
)

sdk_config = SDKConfig(
    base_url="https://api.pynomaly.com",
    api_key="your-secure-api-key",
    client=client_config
)

# Initialize secure client
client = PynomaliClient(config=sdk_config)

# Upload with automatic encryption and checksum validation
dataset = client.create_dataset(
    data_source="sensitive_data.csv",
    name="Secure Dataset",
    description="Dataset with client-side encryption"
)
```

### Environment Variables Configuration

```bash
# Environment variables for security configuration
export PYNOMALY_BASE_URL="https://api.pynomaly.com"
export PYNOMALY_API_KEY="your-secure-api-key"
export PYNOMALY_VERIFY_SSL="true"
export PYNOMALY_ENFORCE_TLS="true"
export PYNOMALY_MINIMUM_TLS_VERSION="TLSv1.3"
export PYNOMALY_ENABLE_CHECKSUM_VALIDATION="true"
export PYNOMALY_ENABLE_CLIENT_SIDE_ENCRYPTION="true"
export PYNOMALY_ENCRYPTION_KEY="your-encryption-key-id"

# Security audit settings
export PYNOMALY_SECURITY_AUDIT_ENABLED="true"
export PYNOMALY_AUDIT_LOG_LEVEL="INFO"
```

## Adapter Security Configuration

### Secure Adapter Usage

```python
from pynomaly.infrastructure.adapters.secure_adapter_base import (
    SecureAdapterConfig,
    create_secure_adapter
)
from pynomaly.infrastructure.adapters.enhanced_pyod_adapter import EnhancedPyODAdapter

# Configure secure adapter
adapter_config = SecureAdapterConfig(
    enable_client_encryption=True,
    encryption_key_id="model-encryption-key",
    security_audit_enabled=True,
    parameter_validation_enabled=True,
    integrity_check_enabled=True
)

# Create secure adapter
secure_adapter = create_secure_adapter(
    EnhancedPyODAdapter,
    config=adapter_config,
    algorithm="IsolationForest"
)

# Train with automatic security features
secure_adapter.secure_fit(X_train)

# Get security summary
security_summary = secure_adapter.get_security_summary()
print(f"Security events: {security_summary['security_events_count']}")
```

## Docker Security Configuration

### Secure Docker Compose

```yaml
# docker-compose.security.yml
version: '3.8'

services:
  pynomaly-api:
    image: pynomaly:latest
    environment:
      - PYNOMALY_SECURITY_HARDENING_ENABLED=true
      - PYNOMALY_ENFORCE_TLS=true
      - PYNOMALY_MINIMUM_TLS_VERSION=TLSv1.3
      - PYNOMALY_REQUIRE_ENCRYPTED_UPLOADS=true
      - PYNOMALY_ENABLE_CHECKSUM_VALIDATION=true
      - PYNOMALY_CHECKSUM_ALGORITHM=SHA512
      - PYNOMALY_SECURITY_AUDIT_ENABLED=true
    volumes:
      - ./ssl/certs:/etc/ssl/certs:ro
      - ./ssl/private:/etc/ssl/private:ro
      - ./logs:/var/log/pynomaly
    ports:
      - "443:8443"  # HTTPS only
    networks:
      - pynomaly-secure
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/cache
    user: "1000:1000"

networks:
  pynomaly-secure:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Kubernetes Security Configuration

### Security Policy and Deployment

```yaml
# k8s/security-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-security-policy
spec:
  podSelector:
    matchLabels:
      app: pynomaly
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: pynomaly
    ports:
    - protocol: TCP
      port: 8443
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-secure
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly
  template:
    metadata:
      labels:
        app: pynomaly
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: pynomaly
        image: pynomaly:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        env:
        - name: PYNOMALY_SECURITY_HARDENING_ENABLED
          value: "true"
        - name: PYNOMALY_ENFORCE_TLS
          value: "true"
        - name: PYNOMALY_MINIMUM_TLS_VERSION
          value: "TLSv1.3"
        - name: PYNOMALY_REQUIRE_ENCRYPTED_UPLOADS
          value: "true"
        volumeMounts:
        - name: ssl-certs
          mountPath: /etc/ssl/certs
          readOnly: true
        - name: ssl-private
          mountPath: /etc/ssl/private
          readOnly: true
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: ssl-certs
        secret:
          secretName: pynomaly-tls-certs
      - name: ssl-private
        secret:
          secretName: pynomaly-tls-private
      - name: tmp
        emptyDir: {}
```

## Monitoring and Alerting

### Security Event Monitoring

```python
# monitoring/security_monitor.py
import logging
from pynomaly.infrastructure.security.security_hardening import get_security_hardening_service

# Configure security monitoring
security_service = get_security_hardening_service()

def monitor_security_events():
    """Monitor and alert on security events."""
    
    events = security_service.security_events
    
    # Check for security violations
    violations = [e for e in events if e.get('is_valid') == False]
    if violations:
        logging.critical(f"Security violations detected: {len(violations)}")
        # Send alerts to security team
    
    # Check for excessive failed attempts
    failed_validations = [e for e in events if 'validation' in e.get('event_type', '')]
    if len(failed_validations) > 10:
        logging.warning(f"High number of validation failures: {len(failed_validations)}")
    
    # Generate security summary
    summary = security_service.get_security_summary()
    logging.info(f"Security summary: {summary}")

# Schedule regular monitoring
import schedule
schedule.every(5).minutes.do(monitor_security_events)
```

### Prometheus Metrics

```python
# monitoring/prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Security metrics
SECURITY_EVENTS = Counter('pynomaly_security_events_total', 'Total security events', ['event_type'])
TLS_VALIDATION_DURATION = Histogram('pynomaly_tls_validation_duration_seconds', 'TLS validation duration')
CHECKSUM_VALIDATION_FAILURES = Counter('pynomaly_checksum_validation_failures_total', 'Checksum validation failures')
ENCRYPTION_OPERATIONS = Counter('pynomaly_encryption_operations_total', 'Encryption operations', ['operation'])
ACTIVE_ENCRYPTED_CONNECTIONS = Gauge('pynomaly_active_encrypted_connections', 'Active encrypted connections')

def track_security_metrics(security_service):
    """Track security metrics for Prometheus."""
    
    for event in security_service.security_events:
        SECURITY_EVENTS.labels(event_type=event.get('event_type')).inc()
        
        if event.get('event_type') == 'checksum_validation_failure':
            CHECKSUM_VALIDATION_FAILURES.inc()
            
        if 'encryption' in event.get('event_type', ''):
            ENCRYPTION_OPERATIONS.labels(operation=event.get('event_type')).inc()
```

## Best Practices Summary

### Security Checklist

- [ ] **TLS Enforcement**: Enable mandatory HTTPS/TLS 1.2+ for all environments
- [ ] **SDK Version Control**: Enforce minimum SDK and Python versions
- [ ] **Checksum Validation**: Enable post-upload integrity validation
- [ ] **Client-side Encryption**: Implement encryption keys via adapter parameters
- [ ] **IAM Policies**: Apply least-privilege access controls
- [ ] **Security Auditing**: Enable comprehensive security event logging
- [ ] **Rate Limiting**: Implement appropriate request rate limits
- [ ] **Certificate Management**: Use proper SSL/TLS certificates
- [ ] **Container Security**: Apply security contexts and policies
- [ ] **Monitoring**: Set up security event monitoring and alerting

### Environment-Specific Recommendations

| Feature | Development | Staging | Production |
|---------|-------------|---------|------------|
| TLS Enforcement | Optional | Required | Mandatory |
| SDK Version Check | Disabled | Enabled | Strict |
| Encryption Required | Optional | Recommended | Mandatory |
| Checksum Algorithm | SHA256 | SHA256 | SHA512 |
| Rate Limiting | Generous | Moderate | Conservative |
| Audit Logging | Basic | Full | Comprehensive |

---

📍 **Location**: `docs/security/`  
🏠 **Documentation Home**: [docs/](../README.md)  
🔗 **Security Home**: [Security Guide](README.md)
