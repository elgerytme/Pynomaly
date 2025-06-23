# Security Best Practices Guide

This comprehensive guide covers security best practices for Pynomaly deployments, including authentication, authorization, data protection, threat mitigation, and compliance considerations.

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication and Authorization](#authentication-and-authorization)
3. [Data Protection](#data-protection)
4. [Network Security](#network-security)
5. [Input Validation and Sanitization](#input-validation-and-sanitization)
6. [Threat Mitigation](#threat-mitigation)
7. [Audit and Monitoring](#audit-and-monitoring)
8. [Compliance and Standards](#compliance-and-standards)

## Security Overview

Pynomaly implements multiple layers of security following industry best practices and security-by-design principles:

- **Authentication**: JWT tokens, API keys, multi-factor authentication
- **Authorization**: Role-based access control (RBAC), fine-grained permissions
- **Data Protection**: Encryption at rest and in transit, secure key management
- **Network Security**: TLS/SSL, firewall configuration, secure communication
- **Input Validation**: Comprehensive validation, sanitization, rate limiting
- **Monitoring**: Security event logging, threat detection, audit trails

### Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                     │
│  (Web UI, CLI, API Clients, Third-party Integrations)     │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTPS/TLS 1.3
┌─────────────────────▼───────────────────────────────────────┐
│                  Load Balancer / WAF                       │
│           (Rate Limiting, DDoS Protection)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │ Secured Internal Network
┌─────────────────────▼───────────────────────────────────────┐
│                  Pynomaly API Gateway                      │
│    ┌─────────────────────────────────────────────────────┐  │
│    │              Authentication Layer                   │  │
│    │   JWT Validation, API Key Verification, MFA       │  │
│    └─────────────────┬───────────────────────────────────┘  │
│    ┌─────────────────▼───────────────────────────────────┐  │
│    │              Authorization Layer                    │  │
│    │    RBAC, Permissions, Resource Access Control     │  │
│    └─────────────────┬───────────────────────────────────┘  │
│    ┌─────────────────▼───────────────────────────────────┐  │
│    │               Input Validation                      │  │
│    │     Schema Validation, Sanitization, Limits       │  │
│    └─────────────────┬───────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │ Validated Requests
┌─────────────────────▼───────────────────────────────────────┐
│                  Application Services                      │
│              (Business Logic Processing)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ Encrypted Communication
┌─────────────────────▼───────────────────────────────────────┐
│                   Data Layer                               │
│         (Encrypted Database, Secure File Storage)         │
└─────────────────────────────────────────────────────────────┘
```

## Authentication and Authorization

### JWT Authentication Implementation

```python
# infrastructure/auth/jwt_auth.py
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import bcrypt
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)

class JWTAuthManager:
    """Production-ready JWT authentication manager with security best practices."""
    
    def __init__(self, 
                 secret_key: str,
                 algorithm: str = "HS256",
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7):
        
        # Validate secret key strength
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters for security")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        
        # Password hashing context
        self.pwd_context = CryptContext(
            schemes=["bcrypt"], 
            deprecated="auto",
            bcrypt__rounds=12  # Increased rounds for better security
        )
        
        # Token blacklist for logout
        self.blacklisted_tokens = set()
    
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt and salt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str, permissions: List[str], 
                          additional_claims: Dict = None) -> str:
        """Create JWT access token with user permissions."""
        
        now = datetime.utcnow()
        expire = now + self.access_token_expire
        
        # Standard JWT claims
        payload = {
            "sub": user_id,  # Subject (user ID)
            "iat": now,      # Issued at
            "exp": expire,   # Expiration
            "type": "access",
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }
        
        # Security claims
        payload.update({
            "permissions": permissions,
            "scope": "pynomaly_api",
            "iss": "pynomaly_auth",  # Issuer
        })
        
        # Additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Access token created for user {user_id}")
            return token
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token for token renewal."""
        
        now = datetime.utcnow()
        expire = now + self.refresh_token_expire
        
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": expire,
            "type": "refresh",
            "jti": secrets.token_urlsafe(16),
            "scope": "token_refresh",
            "iss": "pynomaly_auth",
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Refresh token created for user {user_id}")
            return token
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict:
        """Verify and decode JWT token with comprehensive validation."""
        
        # Check if token is blacklisted
        if token in self.blacklisted_tokens:
            raise jwt.InvalidTokenError("Token has been revoked")
        
        try:
            # Decode with verification
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["sub", "iat", "exp", "type", "jti"]
                }
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise jwt.InvalidTokenError(f"Invalid token type. Expected {token_type}")
            
            # Verify issuer
            if payload.get("iss") != "pynomaly_auth":
                raise jwt.InvalidTokenError("Invalid token issuer")
            
            logger.debug(f"Token verified successfully for user {payload['sub']}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            raise
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise jwt.InvalidTokenError(f"Token verification failed: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Generate new access token using refresh token."""
        
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, token_type="refresh")
            user_id = payload["sub"]
            
            # Get user permissions (would typically come from database)
            user_permissions = self._get_user_permissions(user_id)
            
            # Create new access token
            new_access_token = self.create_access_token(user_id, user_permissions)
            
            return {
                "access_token": new_access_token,
                "token_type": "bearer",
                "expires_in": int(self.access_token_expire.total_seconds())
            }
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise
    
    def revoke_token(self, token: str):
        """Revoke token by adding to blacklist."""
        try:
            # Decode to get JTI (don't verify expiration for revocation)
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            if jti:
                self.blacklisted_tokens.add(jti)
                logger.info(f"Token revoked: {jti}")
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            # Still add the full token to blacklist as fallback
            self.blacklisted_tokens.add(token)
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions from database."""
        # This would typically query the database
        # For now, return default permissions
        return [
            "read:detectors",
            "write:detectors", 
            "read:datasets",
            "write:datasets",
            "read:results"
        ]


class RoleBasedAccessControl:
    """Role-based access control implementation."""
    
    def __init__(self):
        self.roles = {
            "admin": {
                "permissions": [
                    "read:*", "write:*", "delete:*", 
                    "manage:users", "manage:roles", "manage:system"
                ],
                "description": "Full system access"
            },
            "data_scientist": {
                "permissions": [
                    "read:detectors", "write:detectors",
                    "read:datasets", "write:datasets", 
                    "read:results", "write:results",
                    "read:experiments", "write:experiments"
                ],
                "description": "ML operations and analysis"
            },
            "analyst": {
                "permissions": [
                    "read:detectors", "read:datasets", 
                    "read:results", "read:experiments",
                    "write:datasets"
                ],
                "description": "Data analysis and viewing"
            },
            "viewer": {
                "permissions": [
                    "read:detectors", "read:datasets", "read:results"
                ],
                "description": "Read-only access"
            },
            "api_client": {
                "permissions": [
                    "read:detectors", "write:detectors",
                    "read:datasets", "write:datasets",
                    "read:results"
                ],
                "description": "API automation access"
            }
        }
    
    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a specific role."""
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        return self.roles[role]["permissions"]
    
    def check_permission(self, user_permissions: List[str], 
                        required_permission: str) -> bool:
        """Check if user has required permission."""
        
        # Check for wildcard permissions
        if "read:*" in user_permissions or "write:*" in user_permissions:
            return True
        
        # Check for exact permission match
        if required_permission in user_permissions:
            return True
        
        # Check for resource-level wildcard (e.g., "detectors:*")
        resource = required_permission.split(":")[1] if ":" in required_permission else ""
        wildcard_permission = f"*:{resource}"
        if wildcard_permission in user_permissions:
            return True
        
        return False
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission for endpoint access."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract user permissions from context (implementation depends on framework)
                user_permissions = kwargs.get("user_permissions", [])
                
                if not self.check_permission(user_permissions, required_permission):
                    raise PermissionError(f"Missing required permission: {required_permission}")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# API Key Authentication for service-to-service communication
class APIKeyManager:
    """Secure API key management for service authentication."""
    
    def __init__(self):
        self.api_keys = {}  # In production, store in secure database
    
    def generate_api_key(self, service_name: str, permissions: List[str]) -> str:
        """Generate secure API key for service."""
        
        # Generate cryptographically secure random key
        api_key = f"pyn_{secrets.token_urlsafe(32)}"
        
        # Store key metadata
        self.api_keys[api_key] = {
            "service_name": service_name,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "active": True
        }
        
        logger.info(f"API key generated for service: {service_name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Dict:
        """Validate API key and return metadata."""
        
        if api_key not in self.api_keys:
            raise ValueError("Invalid API key")
        
        key_data = self.api_keys[api_key]
        
        if not key_data["active"]:
            raise ValueError("API key has been deactivated")
        
        # Update last used timestamp
        key_data["last_used"] = datetime.utcnow()
        
        return key_data
    
    def revoke_api_key(self, api_key: str):
        """Revoke API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"API key revoked: {api_key[:10]}...")


# Example FastAPI security integration
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()
jwt_manager = JWTAuthManager(secret_key="your-secret-key-here")
rbac = RoleBasedAccessControl()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FastAPI dependency to get current authenticated user."""
    
    try:
        # Verify JWT token
        payload = jwt_manager.verify_token(credentials.credentials)
        
        user_id = payload["sub"]
        permissions = payload.get("permissions", [])
        
        return {
            "user_id": user_id,
            "permissions": permissions,
            "token_payload": payload
        }
        
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def require_permission(permission: str):
    """FastAPI dependency to require specific permission."""
    
    def permission_checker(current_user: dict = Depends(get_current_user)):
        if not rbac.check_permission(current_user["permissions"], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission}"
            )
        return current_user
    
    return permission_checker
```

### Multi-Factor Authentication (MFA)

```python
# infrastructure/auth/mfa.py
import pyotp
import qrcode
import secrets
from typing import Dict, Optional
import io
import base64
from PIL import Image

class MFAManager:
    """Multi-factor authentication using TOTP (Time-based One-Time Password)."""
    
    def __init__(self, service_name: str = "Pynomaly"):
        self.service_name = service_name
        self.users_mfa = {}  # In production, store in secure database
    
    def enable_mfa_for_user(self, user_id: str, user_email: str) -> Dict[str, str]:
        """Enable MFA for user and return setup information."""
        
        # Generate secret key
        secret = pyotp.random_base32()
        
        # Create TOTP URI
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.service_name
        )
        
        # Generate QR code
        qr_code_data = self._generate_qr_code(totp_uri)
        
        # Store MFA data
        self.users_mfa[user_id] = {
            "secret": secret,
            "enabled": False,  # User must verify first code to enable
            "backup_codes": self._generate_backup_codes(),
            "created_at": datetime.utcnow()
        }
        
        return {
            "secret": secret,
            "qr_code": qr_code_data,
            "backup_codes": self.users_mfa[user_id]["backup_codes"],
            "manual_entry_key": secret
        }
    
    def verify_mfa_code(self, user_id: str, code: str) -> bool:
        """Verify MFA code for user."""
        
        if user_id not in self.users_mfa:
            return False
        
        user_mfa = self.users_mfa[user_id]
        
        # Check TOTP code
        totp = pyotp.TOTP(user_mfa["secret"])
        if totp.verify(code, valid_window=1):  # Allow 30-second window
            # Enable MFA if this is the first successful verification
            if not user_mfa["enabled"]:
                user_mfa["enabled"] = True
                logger.info(f"MFA enabled for user {user_id}")
            return True
        
        # Check backup codes
        if code in user_mfa["backup_codes"]:
            user_mfa["backup_codes"].remove(code)  # One-time use
            logger.info(f"Backup code used for user {user_id}")
            return True
        
        return False
    
    def disable_mfa_for_user(self, user_id: str):
        """Disable MFA for user."""
        if user_id in self.users_mfa:
            del self.users_mfa[user_id]
            logger.info(f"MFA disabled for user {user_id}")
    
    def _generate_qr_code(self, totp_uri: str) -> str:
        """Generate QR code image as base64 string."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA recovery."""
        return [secrets.token_hex(4).upper() for _ in range(count)]
```

## Data Protection

### Encryption at Rest and in Transit

```python
# infrastructure/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import secrets
from typing import Union

class DataEncryption:
    """Production-ready data encryption for sensitive information."""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._generate_master_key()
        
        self.fernet = self._create_fernet()
    
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key."""
        return Fernet.generate_key()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher with PBKDF2 key derivation."""
        # Use PBKDF2 to derive key from master key
        salt = b'pynomaly_salt_v1'  # In production, use random salt per encryption
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # NIST recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = self.fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode('utf-8')
    
    def encrypt_file(self, file_path: str, output_path: str = None):
        """Encrypt file contents."""
        if not output_path:
            output_path = file_path + '.encrypted'
        
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.fernet.encrypt(file_data)
        
        with open(output_path, 'wb') as file:
            file.write(encrypted_data)
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None):
        """Decrypt file contents."""
        if not output_path:
            output_path = encrypted_file_path.replace('.encrypted', '')
        
        with open(encrypted_file_path, 'rb') as file:
            encrypted_data = file.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)


# Database field encryption
class FieldEncryption:
    """Encrypt sensitive database fields."""
    
    def __init__(self, encryption_key: str):
        self.cipher = DataEncryption(encryption_key)
    
    def encrypt_field(self, value: str) -> str:
        """Encrypt database field value."""
        if not value:
            return value
        return self.cipher.encrypt_data(value)
    
    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt database field value."""
        if not encrypted_value:
            return encrypted_value
        return self.cipher.decrypt_data(encrypted_value)


# TLS/SSL Configuration
class TLSConfig:
    """TLS/SSL configuration for secure communication."""
    
    @staticmethod
    def get_ssl_context():
        """Get SSL context for HTTPS connections."""
        import ssl
        
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Disable weak protocols
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        # Set minimum TLS version to 1.2
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Set secure cipher suites
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
    
    @staticmethod
    def get_uvicorn_ssl_config(cert_file: str, key_file: str) -> Dict:
        """Get SSL configuration for Uvicorn ASGI server."""
        return {
            "ssl_keyfile": key_file,
            "ssl_certfile": cert_file,
            "ssl_version": ssl.PROTOCOL_TLSv1_2,
            "ssl_cert_reqs": ssl.CERT_NONE,
            "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
        }
```

## Input Validation and Sanitization

### Comprehensive Input Validation

```python
# infrastructure/security/validation.py
import re
import html
import bleach
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator, Field
import sqlparse
from urllib.parse import urlparse

class SecurityValidator:
    """Comprehensive input validation and sanitization."""
    
    # Security patterns
    SQL_INJECTION_PATTERNS = [
        r"(\'|\")(\s)*(union|select|insert|update|delete|drop|create|alter|exec|execute)",
        r"(union|select|insert|update|delete|drop|create|alter)\s+",
        r"(\-\-)|(\#)",  # SQL comments
        r"(xp_|sp_)",    # SQL Server extended procedures
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload=",
        r"onerror=",
        r"onmouseover=",
        r"onclick=",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.\/",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
    ]
    
    @classmethod
    def validate_sql_input(cls, value: str) -> bool:
        """Check for SQL injection patterns."""
        if not isinstance(value, str):
            return True
        
        value_lower = value.lower()
        
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return False
        
        return True
    
    @classmethod
    def validate_xss_input(cls, value: str) -> bool:
        """Check for XSS patterns."""
        if not isinstance(value, str):
            return True
        
        value_lower = value.lower()
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return False
        
        return True
    
    @classmethod
    def validate_path_traversal(cls, value: str) -> bool:
        """Check for path traversal patterns."""
        if not isinstance(value, str):
            return True
        
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    @classmethod
    def sanitize_html(cls, value: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not isinstance(value, str):
            return value
        
        # Allowed HTML tags and attributes
        allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li']
        allowed_attributes = {}
        
        return bleach.clean(value, tags=allowed_tags, attributes=allowed_attributes)
    
    @classmethod
    def sanitize_sql_string(cls, value: str) -> str:
        """Sanitize string for SQL queries."""
        if not isinstance(value, str):
            return value
        
        # Escape single quotes
        return value.replace("'", "''")
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username format."""
        # Allow alphanumeric, underscore, hyphen, 3-30 characters
        pattern = r'^[a-zA-Z0-9_-]{3,30}$'
        return bool(re.match(pattern, username))
    
    @classmethod
    def validate_file_extension(cls, filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension."""
        if not filename or '.' not in filename:
            return False
        
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in [ext.lower() for ext in allowed_extensions]
    
    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL format and safety."""
        try:
            parsed = urlparse(url)
            
            # Check basic URL structure
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Only allow HTTP and HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Block localhost and private IPs in production
            forbidden_hosts = ['localhost', '127.0.0.1', '0.0.0.0']
            if parsed.hostname in forbidden_hosts:
                return False
            
            return True
            
        except Exception:
            return False


# Pydantic models with security validation
class SecureDatasetInput(BaseModel):
    """Secure dataset input validation."""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    file_path: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = Field(None, max_items=10)
    
    @validator('name')
    def validate_name_security(cls, v):
        if not SecurityValidator.validate_sql_input(v):
            raise ValueError('Invalid characters in name')
        if not SecurityValidator.validate_xss_input(v):
            raise ValueError('Invalid characters in name')
        return v.strip()
    
    @validator('description')
    def validate_description_security(cls, v):
        if v is None:
            return v
        if not SecurityValidator.validate_sql_input(v):
            raise ValueError('Invalid characters in description')
        if not SecurityValidator.validate_xss_input(v):
            raise ValueError('Invalid characters in description')
        return SecurityValidator.sanitize_html(v)
    
    @validator('file_path')
    def validate_file_path_security(cls, v):
        if v is None:
            return v
        if not SecurityValidator.validate_path_traversal(v):
            raise ValueError('Invalid file path')
        return v
    
    @validator('tags')
    def validate_tags_security(cls, v):
        if v is None:
            return v
        
        validated_tags = []
        for tag in v:
            if not SecurityValidator.validate_sql_input(tag):
                raise ValueError(f'Invalid characters in tag: {tag}')
            if not SecurityValidator.validate_xss_input(tag):
                raise ValueError(f'Invalid characters in tag: {tag}')
            validated_tags.append(tag.strip()[:50])  # Limit tag length
        
        return validated_tags


class SecureUserInput(BaseModel):
    """Secure user input validation."""
    
    username: str = Field(..., min_length=3, max_length=30)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def validate_username_format(cls, v):
        if not SecurityValidator.validate_username(v):
            raise ValueError('Invalid username format')
        return v.lower()
    
    @validator('email')
    def validate_email_format(cls, v):
        if not SecurityValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password_strength(cls, v):
        # Password strength requirements
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        
        return v
    
    @validator('full_name')
    def validate_full_name_security(cls, v):
        if v is None:
            return v
        if not SecurityValidator.validate_xss_input(v):
            raise ValueError('Invalid characters in full name')
        return v.strip()


# Rate limiting for API endpoints
class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self):
        self.requests = {}  # In production, use Redis
    
    def is_allowed(self, client_id: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if request is within rate limit."""
        import time
        
        now = time.time()
        client_requests = self.requests.get(client_id, [])
        
        # Remove old requests outside the window
        client_requests = [req_time for req_time in client_requests if now - req_time < window]
        
        # Check if limit exceeded
        if len(client_requests) >= limit:
            return False
        
        # Add current request
        client_requests.append(now)
        self.requests[client_id] = client_requests
        
        return True
    
    def get_rate_limit_info(self, client_id: str, limit: int = 100, window: int = 3600) -> Dict:
        """Get rate limit information for client."""
        import time
        
        now = time.time()
        client_requests = self.requests.get(client_id, [])
        
        # Count requests in current window
        current_requests = len([req_time for req_time in client_requests if now - req_time < window])
        
        # Calculate reset time
        if client_requests:
            oldest_request = min(client_requests)
            reset_time = oldest_request + window
        else:
            reset_time = now + window
        
        return {
            "limit": limit,
            "remaining": max(0, limit - current_requests),
            "reset": int(reset_time),
            "window": window
        }
```

## Network Security

### Firewall and Network Configuration

```yaml
# infrastructure/security/firewall-rules.yaml
# Example firewall configuration for production deployment

# Kubernetes Network Policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-security-policy
  namespace: pynomaly
spec:
  podSelector:
    matchLabels:
      app: pynomaly
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  # Allow traffic from load balancer
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  
  # Allow traffic from monitoring
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000  # Metrics endpoint
  
  egress:
  # Allow database connections
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  
  # Allow Redis connections
  - to:
    - namespaceSelector:
        matchLabels:
          name: cache
    ports:
    - protocol: TCP
      port: 6379
  
  # Allow external API calls (for integrations)
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80

---
# Service-specific security policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-database-policy
  namespace: pynomaly
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes:
  - Ingress
  
  ingress:
  # Only allow connections from Pynomaly application
  - from:
    - podSelector:
        matchLabels:
          app: pynomaly
    ports:
    - protocol: TCP
      port: 5432
```

### Web Application Firewall (WAF) Configuration

```python
# infrastructure/security/waf.py
from typing import List, Dict, Pattern
import re
import logging

class WAFRule:
    """Web Application Firewall rule definition."""
    
    def __init__(self, name: str, pattern: str, action: str = "block", 
                 severity: str = "medium"):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.action = action  # block, log, rate_limit
        self.severity = severity
    
    def matches(self, text: str) -> bool:
        """Check if text matches the rule pattern."""
        return bool(self.pattern.search(text))


class WebApplicationFirewall:
    """Basic WAF implementation for request filtering."""
    
    def __init__(self):
        self.rules = self._load_default_rules()
        self.blocked_ips = set()
        self.rate_limited_ips = {}
        
    def _load_default_rules(self) -> List[WAFRule]:
        """Load default WAF rules."""
        return [
            # SQL Injection
            WAFRule(
                "SQL_INJECTION_UNION",
                r"union\s+(all\s+)?select",
                "block",
                "high"
            ),
            WAFRule(
                "SQL_INJECTION_COMMENT",
                r"(--|#|/\*|\*/)",
                "block",
                "medium"
            ),
            WAFRule(
                "SQL_INJECTION_FUNCTIONS",
                r"(exec|execute|sp_|xp_)\s*\(",
                "block",
                "high"
            ),
            
            # XSS
            WAFRule(
                "XSS_SCRIPT_TAG",
                r"<script[^>]*>.*?</script>",
                "block",
                "high"
            ),
            WAFRule(
                "XSS_EVENT_HANDLERS",
                r"on(load|error|click|mouseover)\s*=",
                "block",
                "medium"
            ),
            WAFRule(
                "XSS_JAVASCRIPT_PROTOCOL",
                r"javascript\s*:",
                "block",
                "medium"
            ),
            
            # Path Traversal
            WAFRule(
                "PATH_TRAVERSAL",
                r"\.\.[\\/]",
                "block",
                "medium"
            ),
            
            # Command Injection
            WAFRule(
                "COMMAND_INJECTION",
                r"[;&|`]",
                "log",
                "medium"
            ),
            
            # File Upload
            WAFRule(
                "MALICIOUS_FILE_EXTENSION",
                r"\.(exe|bat|cmd|com|pif|scr|vbs|js)$",
                "block",
                "high"
            ),
        ]
    
    def analyze_request(self, request_data: Dict) -> Dict[str, Any]:
        """Analyze incoming request for security threats."""
        
        violations = []
        risk_score = 0
        
        # Check all request parameters
        for field, value in request_data.items():
            if isinstance(value, str):
                field_violations = self._check_field(field, value)
                violations.extend(field_violations)
                risk_score += len(field_violations)
        
        # Determine action based on risk score
        action = "allow"
        if risk_score >= 3:
            action = "block"
        elif risk_score >= 1:
            action = "monitor"
        
        return {
            "action": action,
            "risk_score": risk_score,
            "violations": violations,
            "blocked": action == "block"
        }
    
    def _check_field(self, field_name: str, value: str) -> List[Dict]:
        """Check individual field against WAF rules."""
        violations = []
        
        for rule in self.rules:
            if rule.matches(value):
                violations.append({
                    "rule_name": rule.name,
                    "field": field_name,
                    "matched_value": value[:100],  # Truncate for logging
                    "severity": rule.severity,
                    "action": rule.action
                })
                
                logging.warning(f"WAF violation: {rule.name} in field {field_name}")
        
        return violations
    
    def block_ip(self, ip_address: str, duration: int = 3600):
        """Block IP address for specified duration."""
        import time
        
        self.blocked_ips.add(ip_address)
        
        # Schedule unblock (in production, use proper task queue)
        def unblock_later():
            time.sleep(duration)
            self.blocked_ips.discard(ip_address)
        
        import threading
        threading.Thread(target=unblock_later, daemon=True).start()
        
        logging.info(f"IP blocked: {ip_address} for {duration} seconds")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips


# FastAPI middleware integration
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class WAFMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for WAF integration."""
    
    def __init__(self, app, waf: WebApplicationFirewall):
        super().__init__(app)
        self.waf = waf
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        
        # Check if IP is blocked
        if self.waf.is_ip_blocked(client_ip):
            raise HTTPException(status_code=403, detail="IP address blocked")
        
        # Analyze request
        request_data = {}
        
        # Check query parameters
        request_data.update(dict(request.query_params))
        
        # Check form data (if present)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                if "application/json" in request.headers.get("content-type", ""):
                    body = await request.body()
                    if body:
                        import json
                        json_data = json.loads(body)
                        if isinstance(json_data, dict):
                            request_data.update(json_data)
            except:
                pass  # Skip if can't parse body
        
        # Analyze with WAF
        analysis = self.waf.analyze_request(request_data)
        
        if analysis["blocked"]:
            # Log security violation
            logging.error(f"WAF blocked request from {client_ip}: {analysis['violations']}")
            
            # Block IP for repeated violations
            self.waf.block_ip(client_ip, duration=3600)
            
            raise HTTPException(
                status_code=403, 
                detail="Request blocked by security policy"
            )
        
        # Add security headers to response
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response
```

This comprehensive security guide covers authentication, authorization, data protection, network security, input validation, and threat mitigation strategies. The implementation provides production-ready security features that follow industry best practices and security standards.