# Security Best Practices Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Security

---


## Overview

This comprehensive guide covers security best practices for deploying and operating Pynomaly in production environments. It addresses authentication, authorization, data protection, network security, and compliance considerations for enterprise deployments.

## Table of Contents

1. [Authentication and Authorization](#authentication-and-authorization)
2. [Data Protection and Encryption](#data-protection-and-encryption)
3. [Network Security](#network-security)
4. [API Security](#api-security)
5. [Container and Kubernetes Security](#container-and-kubernetes-security)
6. [Secrets Management](#secrets-management)
7. [Audit Logging and Monitoring](#audit-logging-and-monitoring)
8. [Compliance and Governance](#compliance-and-governance)
9. [Incident Response](#incident-response)
10. [Security Checklist](#security-checklist)

## Authentication and Authorization

### JWT Authentication

Pynomaly uses JSON Web Tokens (JWT) for stateless authentication. Follow these best practices:

#### JWT Configuration

```python
# Secure JWT settings
JWT_SETTINGS = {
    "ALGORITHM": "HS256",  # Use HS256 or RS256
    "SECRET_KEY": "your-256-bit-secret-key-here",  # Must be cryptographically secure
    "ACCESS_TOKEN_EXPIRE_MINUTES": 60,  # Short-lived tokens
    "REFRESH_TOKEN_EXPIRE_DAYS": 7,     # Separate refresh tokens
    "ISSUER": "pynomaly-api",
    "AUDIENCE": ["pynomaly-clients"],
}
```

#### Secure Token Generation

```python
import secrets
import base64

# Generate cryptographically secure secret key
secret_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
print(f"JWT_SECRET_KEY={secret_key}")
```

#### Token Validation

```python
# Implement proper token validation
import jwt
from datetime import datetime, timezone

def validate_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"],
            issuer="pynomaly-api",
            audience="pynomaly-clients",
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
            }
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
```

### Role-Based Access Control (RBAC)

Implement fine-grained access control:

```python
# Define roles and permissions
ROLES_PERMISSIONS = {
    "admin": [
        "detector:create", "detector:read", "detector:update", "detector:delete",
        "dataset:create", "dataset:read", "dataset:update", "dataset:delete",
        "detection:train", "detection:predict", "detection:explain",
        "experiment:create", "experiment:read", "experiment:update", "experiment:delete",
        "user:create", "user:read", "user:update", "user:delete",
        "system:monitor", "system:configure"
    ],
    "data_scientist": [
        "detector:create", "detector:read", "detector:update",
        "dataset:create", "dataset:read", "dataset:update",
        "detection:train", "detection:predict", "detection:explain",
        "experiment:create", "experiment:read", "experiment:update"
    ],
    "analyst": [
        "detector:read",
        "dataset:read",
        "detection:predict", "detection:explain",
        "experiment:read"
    ],
    "viewer": [
        "detector:read",
        "dataset:read",
        "experiment:read"
    ]
}
```

#### Permission Decorator

```python
from functools import wraps
from fastapi import HTTPException, Depends

def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            user_permissions = get_user_permissions(current_user)
            if permission not in user_permissions:
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@router.post("/detectors")
@require_permission("detector:create")
async def create_detector(...):
    pass
```

### Multi-Factor Authentication (MFA)

Implement MFA for enhanced security:

```python
import pyotp
import qrcode

class MFAService:
    def generate_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user."""
        secret = pyotp.random_base32()
        # Store secret securely associated with user
        store_user_mfa_secret(user_id, secret)
        return secret

    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for authenticator app setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name="Pynomaly"
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        # Return base64 encoded QR code image
        return generate_qr_image_base64(qr)

    def verify_token(self, user_id: str, token: str) -> bool:
        """Verify TOTP token."""
        secret = get_user_mfa_secret(user_id)
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
```

## Data Protection and Encryption

### Data Encryption at Rest

Implement encryption for sensitive data:

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: bytes, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher = Fernet(key)
        self.salt = salt

    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Usage for sensitive fields
class EncryptedDetector:
    def __init__(self):
        self.encryption = DataEncryption(get_encryption_key())

    def save_model(self, model_data: bytes, metadata: dict):
        # Encrypt sensitive metadata
        if 'api_key' in metadata:
            metadata['api_key'] = self.encryption.encrypt(metadata['api_key'])

        # Save encrypted data
        save_to_storage(model_data, metadata)
```

### Database Encryption

Configure database-level encryption:

```yaml
# PostgreSQL with encryption
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
data:
  postgresql.conf: |
    # Enable SSL
    ssl = on
    ssl_cert_file = '/etc/ssl/certs/server.crt'
    ssl_key_file = '/etc/ssl/private/server.key'
    ssl_ca_file = '/etc/ssl/certs/ca.crt'

    # Encryption settings
    password_encryption = scram-sha-256
    log_statement = 'none'  # Don't log SQL statements
    log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
```

### Field-Level Encryption

Encrypt specific sensitive fields:

```python
from sqlalchemy_utils import EncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)

    # Encrypted sensitive fields
    api_key = Column(EncryptedType(String, get_encryption_key(), AesEngine, 'pkcs5'))
    personal_info = Column(EncryptedType(JSON, get_encryption_key(), AesEngine, 'pkcs5'))
```

## Network Security

### TLS/SSL Configuration

Implement proper TLS configuration:

```yaml
# Nginx configuration for TLS
server {
    listen 443 ssl http2;
    server_name api.pynomaly.io;

    # SSL certificates
    ssl_certificate /etc/ssl/certs/pynomaly.crt;
    ssl_certificate_key /etc/ssl/private/pynomaly.key;

    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; media-src 'self'; object-src 'none'; child-src 'none'; frame-src 'none'; worker-src 'none'; frame-ancestors 'none'; form-action 'self'; base-uri 'self'; manifest-src 'self';" always;
}
```

### Network Policies

Implement Kubernetes network policies:

```yaml
# Restrict API pod network access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-api-netpol
spec:
  podSelector:
    matchLabels:
      app: pynomaly-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Allow ingress from nginx ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  # Allow ingress from monitoring (Prometheus)
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  # Allow egress to database
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  # Allow egress to cache
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  # Allow DNS resolution
  - to: []
    ports:
    - protocol: UDP
      port: 53
  # Allow HTTPS for external API calls
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

### VPC and Firewall Configuration

Configure cloud network security:

```hcl
# Terraform example for AWS VPC
resource "aws_vpc" "pynomaly_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "pynomaly-vpc"
  }
}

# Private subnets for database
resource "aws_subnet" "private_db" {
  count             = 2
  vpc_id            = aws_vpc.pynomaly_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "pynomaly-private-db-${count.index + 1}"
  }
}

# Security group for API
resource "aws_security_group" "api" {
  name_prefix = "pynomaly-api-"
  vpc_id      = aws_vpc.pynomaly_vpc.id

  # Allow HTTPS from load balancer
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # Allow all outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Security group for database
resource "aws_security_group" "database" {
  name_prefix = "pynomaly-db-"
  vpc_id      = aws_vpc.pynomaly_vpc.id

  # Allow PostgreSQL from API
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}
```

## API Security

### Input Validation and Sanitization

Implement comprehensive input validation:

```python
from pydantic import BaseModel, validator, Field
from typing import Optional
import re

class CreateDetectorRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    algorithm_name: str = Field(..., regex=r'^[A-Za-z][A-Za-z0-9_]*$')
    contamination_rate: float = Field(..., ge=0.0, le=0.5)
    hyperparameters: Optional[dict] = Field(default_factory=dict)
    description: Optional[str] = Field(None, max_length=500)

    @validator('name')
    def validate_name(cls, v):
        # Sanitize name - remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', v.strip())
        if not sanitized:
            raise ValueError('Name cannot be empty after sanitization')
        return sanitized

    @validator('hyperparameters')
    def validate_hyperparameters(cls, v):
        if v is None:
            return {}

        # Validate hyperparameter keys and values
        allowed_keys = {'n_estimators', 'max_samples', 'contamination', 'n_neighbors'}
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f'Invalid hyperparameter: {key}')

        return v

class SQLInjectionProtector:
    """Protect against SQL injection attacks."""

    DANGEROUS_PATTERNS = [
        r"('|(\\')|(;)|(\\;))|(\|)|(\*))",
        r"((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
        r"((\%27)|(\'))union",
        r"exec(\s|\+)+(s|x)p\w+",
        r"union\s+.*select",
        r"insert\s+into",
        r"delete\s+from",
        r"drop\s+table"
    ]

    @classmethod
    def is_safe(cls, input_string: str) -> bool:
        """Check if input string is safe from SQL injection."""
        if not input_string:
            return True

        input_lower = input_string.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return False
        return True

    @classmethod
    def sanitize(cls, input_string: str) -> str:
        """Sanitize input string."""
        if not cls.is_safe(input_string):
            raise ValueError("Potentially dangerous input detected")
        return input_string
```

### Rate Limiting

Implement comprehensive rate limiting:

```python
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# Initialize limiter with Redis backend
redis_client = redis.Redis(host='redis', port=6379, db=0)
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://redis:6379/0"
)

# Different rate limits for different endpoints
@app.post("/api/auth/token")
@limiter.limit("5/minute")  # Strict limit for login attempts
async def login(request: Request, ...):
    pass

@app.get("/api/detectors")
@limiter.limit("100/minute")  # Standard limit for read operations
async def list_detectors(request: Request, ...):
    pass

@app.post("/api/detection/predict")
@limiter.limit("10/minute")  # Lower limit for compute-intensive operations
async def predict(request: Request, ...):
    pass

# User-specific rate limiting
def get_user_id(request: Request):
    # Extract user ID from JWT token
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        payload = decode_jwt(token)
        return payload.get("sub")
    except:
        return get_remote_address(request)

user_limiter = Limiter(key_func=get_user_id)

@app.post("/api/datasets")
@user_limiter.limit("5/hour")  # Per-user upload limit
async def upload_dataset(request: Request, ...):
    pass
```

### CORS Configuration

Configure CORS securely:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pynomaly.io",
        "https://app.pynomaly.io",
        "https://admin.pynomaly.io"
    ],  # Specific origins only, not "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-CSRF-Token"
    ],
    expose_headers=["X-Total-Count"],
    max_age=86400  # 24 hours
)
```

### Security Headers

Implement comprehensive security headers:

```python
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "media-src 'self'; "
            "object-src 'none'; "
            "child-src 'none'; "
            "frame-src 'none'; "
            "worker-src 'none'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'; "
            "manifest-src 'self'"
        )
        response.headers["Content-Security-Policy"] = csp

        # HSTS (only in production with HTTPS)
        if request.headers.get("X-Forwarded-Proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response

app.add_middleware(SecurityHeadersMiddleware)
```

## Container and Kubernetes Security

### Secure Container Images

Create secure Docker images:

```dockerfile
# Use official Python base image with security updates
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r pynomaly && useradd -r -g pynomaly pynomaly

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    dumb-init && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R pynomaly:pynomaly /app
USER pynomaly

# Use dumb-init as entrypoint
ENTRYPOINT ["dumb-init", "--"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "pynomaly.presentation.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Pod Security Standards

Apply pod security standards:

```yaml
# Pod Security Policy
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: pynomaly-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  runAsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
```

### Security Context

Configure secure pod security context:

```yaml
# Secure pod configuration
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 2000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: api
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
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

## Secrets Management

### Kubernetes Secrets

Manage secrets securely in Kubernetes:

```bash
# Create secrets from command line
kubectl create secret generic pynomaly-secrets \
  --from-literal=database-url='postgresql://user:pass@host/db' \
  --from-literal=jwt-secret='your-super-secret-key' \
  --from-literal=redis-url='redis://redis:6379/0'

# Or from files
kubectl create secret generic pynomaly-certs \
  --from-file=tls.crt=server.crt \
  --from-file=tls.key=server.key \
  --from-file=ca.crt=ca.crt
```

### External Secrets Management

Integrate with external secret managers:

```yaml
# Using External Secrets Operator with AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        serviceAccount:
          name: pynomaly-api
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: pynomaly-external-secret
spec:
  refreshInterval: 15s
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: pynomaly-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: pynomaly/production
      property: database_url
  - secretKey: jwt-secret
    remoteRef:
      key: pynomaly/production
      property: jwt_secret
```

### HashiCorp Vault Integration

```python
import hvac

class VaultSecretManager:
    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)

    def get_secret(self, path: str, key: str) -> str:
        """Retrieve secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'][key]
        except Exception as e:
            raise SecurityError(f"Failed to retrieve secret: {e}")

    def rotate_secret(self, path: str, key: str, new_value: str):
        """Rotate a secret in Vault."""
        current_data = self.client.secrets.kv.v2.read_secret_version(path=path)
        current_data['data']['data'][key] = new_value
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=current_data['data']['data']
        )

# Usage
vault = VaultSecretManager('https://vault.company.com', vault_token)
database_url = vault.get_secret('pynomaly/production', 'database_url')
```

## Audit Logging and Monitoring

### Comprehensive Audit Logging

Implement detailed audit logging:

```python
import structlog
from datetime import datetime
from typing import Optional

class AuditLogger:
    def __init__(self):
        self.logger = structlog.get_logger("audit")

    def log_authentication(self, user_id: str, success: bool, ip_address: str, user_agent: str):
        """Log authentication attempts."""
        self.logger.info(
            "authentication_attempt",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow().isoformat()
        )

    def log_data_access(self, user_id: str, resource_type: str, resource_id: str, action: str):
        """Log data access events."""
        self.logger.info(
            "data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            timestamp=datetime.utcnow().isoformat()
        )

    def log_security_event(self, event_type: str, severity: str, details: dict):
        """Log security events."""
        self.logger.warning(
            "security_event",
            event_type=event_type,
            severity=severity,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )

# Middleware for automatic audit logging
class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, audit_logger: AuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Extract user information
        user_id = extract_user_id(request)
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent", "")

        response = await call_next(request)

        # Log the request
        self.audit_logger.log_data_access(
            user_id=user_id,
            resource_type=request.url.path.split('/')[2] if len(request.url.path.split('/')) > 2 else 'unknown',
            resource_id=request.path_params.get('id', 'N/A'),
            action=request.method,
            response_status=response.status_code,
            processing_time=time.time() - start_time,
            ip_address=ip_address,
            user_agent=user_agent
        )

        return response
```

### Security Monitoring

Implement security monitoring and alerting:

```python
class SecurityMonitor:
    def __init__(self, alert_threshold: int = 5, time_window: int = 300):
        self.alert_threshold = alert_threshold
        self.time_window = time_window
        self.failed_attempts = {}

    def check_failed_logins(self, ip_address: str) -> bool:
        """Check for suspicious login patterns."""
        now = time.time()

        # Clean old entries
        self.failed_attempts = {
            ip: times for ip, times in self.failed_attempts.items()
            if any(t > now - self.time_window for t in times)
        }

        # Add current failure
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = []

        self.failed_attempts[ip_address].append(now)

        # Check if threshold exceeded
        recent_failures = [
            t for t in self.failed_attempts[ip_address]
            if t > now - self.time_window
        ]

        if len(recent_failures) >= self.alert_threshold:
            self.send_security_alert(
                "brute_force_attempt",
                f"Multiple failed login attempts from {ip_address}",
                {"ip_address": ip_address, "attempts": len(recent_failures)}
            )
            return True

        return False

    def send_security_alert(self, alert_type: str, message: str, details: dict):
        """Send security alert to monitoring system."""
        # Send to Slack, email, PagerDuty, etc.
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high"
        }

        # Example: Send to Slack
        send_slack_alert(alert_data)

        # Example: Store in database for analysis
        store_security_alert(alert_data)
```

### Prometheus Metrics for Security

Define security-related metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Security metrics
failed_auth_attempts = Counter(
    'failed_authentication_attempts_total',
    'Total number of failed authentication attempts',
    ['ip_address', 'user_agent']
)

successful_auth_attempts = Counter(
    'successful_authentication_attempts_total',
    'Total number of successful authentication attempts',
    ['user_id']
)

security_events = Counter(
    'security_events_total',
    'Total number of security events',
    ['event_type', 'severity']
)

active_sessions = Gauge(
    'active_user_sessions',
    'Number of active user sessions'
)

rate_limit_violations = Counter(
    'rate_limit_violations_total',
    'Total number of rate limit violations',
    ['endpoint', 'ip_address']
)
```

## Compliance and Governance

### GDPR Compliance

Implement GDPR compliance features:

```python
class GDPRCompliance:
    def __init__(self, data_retention_days: int = 365):
        self.data_retention_days = data_retention_days

    def handle_data_subject_request(self, user_id: str, request_type: str):
        """Handle GDPR data subject requests."""
        if request_type == "access":
            return self.export_user_data(user_id)
        elif request_type == "delete":
            return self.delete_user_data(user_id)
        elif request_type == "portability":
            return self.export_portable_data(user_id)
        else:
            raise ValueError(f"Unknown request type: {request_type}")

    def export_user_data(self, user_id: str) -> dict:
        """Export all user data for GDPR access request."""
        user_data = {
            "personal_info": get_user_profile(user_id),
            "detectors": get_user_detectors(user_id),
            "datasets": get_user_datasets(user_id),
            "experiments": get_user_experiments(user_id),
            "audit_logs": get_user_audit_logs(user_id)
        }

        # Log the data export
        audit_logger.log_data_access(
            user_id=user_id,
            resource_type="gdpr_export",
            resource_id=user_id,
            action="export_all_data"
        )

        return user_data

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data for GDPR deletion request."""
        try:
            # Delete user resources
            delete_user_detectors(user_id)
            delete_user_datasets(user_id)
            delete_user_experiments(user_id)

            # Anonymize audit logs (keep for compliance but remove PII)
            anonymize_audit_logs(user_id)

            # Delete user profile
            delete_user_profile(user_id)

            # Log the deletion
            audit_logger.log_security_event(
                "gdpr_deletion",
                "high",
                {"user_id": user_id, "deletion_type": "complete"}
            )

            return True
        except Exception as e:
            audit_logger.log_security_event(
                "gdpr_deletion_failed",
                "high",
                {"user_id": user_id, "error": str(e)}
            )
            return False
```

### Data Classification

Implement data classification:

```python
from enum import Enum

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataClassifier:
    def __init__(self):
        self.classification_rules = {
            "personal_info": DataClassification.RESTRICTED,
            "api_keys": DataClassification.RESTRICTED,
            "model_data": DataClassification.CONFIDENTIAL,
            "experiment_results": DataClassification.INTERNAL,
            "system_logs": DataClassification.INTERNAL
        }

    def classify_data(self, data_type: str) -> DataClassification:
        """Classify data based on type."""
        return self.classification_rules.get(data_type, DataClassification.INTERNAL)

    def get_access_requirements(self, classification: DataClassification) -> dict:
        """Get access requirements for data classification."""
        requirements = {
            DataClassification.PUBLIC: {
                "authentication": False,
                "encryption": False,
                "audit_logging": False
            },
            DataClassification.INTERNAL: {
                "authentication": True,
                "encryption": True,
                "audit_logging": True
            },
            DataClassification.CONFIDENTIAL: {
                "authentication": True,
                "encryption": True,
                "audit_logging": True,
                "mfa_required": True
            },
            DataClassification.RESTRICTED: {
                "authentication": True,
                "encryption": True,
                "audit_logging": True,
                "mfa_required": True,
                "approval_required": True
            }
        }
        return requirements[classification]
```

## Incident Response

### Security Incident Response Plan

```python
class IncidentResponse:
    def __init__(self):
        self.severity_levels = {
            "low": {"response_time": 3600, "escalation_time": 7200},
            "medium": {"response_time": 1800, "escalation_time": 3600},
            "high": {"response_time": 900, "escalation_time": 1800},
            "critical": {"response_time": 300, "escalation_time": 900}
        }

    def handle_security_incident(self, incident_type: str, severity: str, details: dict):
        """Handle security incident according to response plan."""
        incident_id = generate_incident_id()

        # Create incident record
        incident = {
            "id": incident_id,
            "type": incident_type,
            "severity": severity,
            "details": details,
            "timestamp": datetime.utcnow(),
            "status": "open",
            "assigned_to": None
        }

        # Immediate response actions
        if incident_type == "brute_force_attack":
            self.block_ip_address(details.get("ip_address"))
        elif incident_type == "data_breach":
            self.initiate_containment_procedures()
        elif incident_type == "malware_detected":
            self.isolate_affected_systems(details.get("affected_systems"))

        # Notify response team
        self.notify_response_team(incident)

        # Start response timer
        self.start_response_timer(incident_id, severity)

        return incident_id

    def block_ip_address(self, ip_address: str):
        """Block malicious IP address."""
        # Add to firewall rules
        add_firewall_rule(f"DENY {ip_address}")

        # Add to rate limiter blacklist
        add_to_blacklist(ip_address)

        # Log the action
        audit_logger.log_security_event(
            "ip_blocked",
            "medium",
            {"ip_address": ip_address, "reason": "automated_security_response"}
        )

    def initiate_containment_procedures(self):
        """Initiate data breach containment procedures."""
        # Immediately disable external API access
        disable_external_api()

        # Notify legal and compliance teams
        notify_legal_team()

        # Begin forensic data collection
        start_forensic_collection()
```

## Security Checklist

### Pre-Deployment Security Checklist

- [ ] **Authentication and Authorization**
  - [ ] JWT tokens use secure algorithms (HS256/RS256)
  - [ ] Token expiration times are appropriately short
  - [ ] RBAC is properly configured
  - [ ] MFA is enabled for administrative accounts
  - [ ] Password policies enforce strong passwords

- [ ] **Data Protection**
  - [ ] Sensitive data is encrypted at rest
  - [ ] Data in transit uses TLS 1.2+
  - [ ] Database connections are encrypted
  - [ ] PII is properly classified and protected
  - [ ] Data retention policies are implemented

- [ ] **Network Security**
  - [ ] TLS certificates are valid and properly configured
  - [ ] Security headers are implemented
  - [ ] CORS is restrictively configured
  - [ ] Network policies restrict pod communication
  - [ ] Firewalls are properly configured

- [ ] **Container Security**
  - [ ] Images are built from secure base images
  - [ ] Containers run as non-root users
  - [ ] Security contexts are properly configured
  - [ ] Images are scanned for vulnerabilities
  - [ ] Secrets are not embedded in images

- [ ] **API Security**
  - [ ] Input validation is comprehensive
  - [ ] Rate limiting is implemented
  - [ ] SQL injection protection is in place
  - [ ] Error messages don't leak sensitive information
  - [ ] API endpoints require proper authentication

- [ ] **Monitoring and Logging**
  - [ ] Audit logging is comprehensive
  - [ ] Security events are monitored
  - [ ] Alerts are configured for security incidents
  - [ ] Log retention meets compliance requirements
  - [ ] Sensitive data is not logged

### Ongoing Security Maintenance

- [ ] **Regular Security Tasks**
  - [ ] Weekly vulnerability scans
  - [ ] Monthly access reviews
  - [ ] Quarterly penetration testing
  - [ ] Semi-annual security training
  - [ ] Annual security audits

- [ ] **Incident Response**
  - [ ] Incident response plan is documented
  - [ ] Response team contacts are current
  - [ ] Escalation procedures are defined
  - [ ] Recovery procedures are tested
  - [ ] Legal and compliance contacts are ready

This comprehensive security guide provides the foundation for deploying and operating Pynomaly securely in enterprise environments. Regular review and updates of these security measures are essential to maintain protection against evolving threats.

---

## 🔗 **Related Documentation**

- **[User Guides](../user-guides/README.md)** - Feature documentation
- **[Getting Started](../getting-started/README.md)** - Installation and setup
- **[Examples](../examples/README.md)** - Real-world examples
