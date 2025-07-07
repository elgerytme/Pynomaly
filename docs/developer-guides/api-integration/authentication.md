# API Authentication Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 📄 Authentication

---


## Overview

Pynomaly API uses JWT (JSON Web Token) based authentication for securing endpoints. This guide covers authentication setup, token management, and security best practices.

## Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth Service
    
    Client->>API: POST /api/auth/token (username, password)
    API->>Auth Service: Validate credentials
    Auth Service-->>API: User validated
    API-->>Client: JWT token + refresh token
    
    Client->>API: Request with Authorization header
    API->>API: Validate JWT token
    API-->>Client: Protected resource
    
    Client->>API: POST /api/auth/refresh (refresh token)
    API-->>Client: New JWT token
```

## Getting Started

### 1. Obtain Access Token

**Endpoint:** `POST /api/auth/token`

**Request:**
```bash
curl -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "def50200a1b2c3d4e5f6...",
  "scope": "read write"
}
```

### 2. Use Token for API Calls

Include the token in the Authorization header:

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  "http://localhost:8000/api/detectors"
```

## Token Management

### Access Tokens

- **Lifetime:** 1 hour (configurable)
- **Format:** JWT with HS256 signature
- **Claims:** User ID, roles, permissions, expiration

**Token Structure:**
```json
{
  "sub": "user123",
  "iat": 1640995200,
  "exp": 1640998800,
  "iss": "pynomaly-api",
  "aud": ["pynomaly-clients"],
  "roles": ["data_scientist"],
  "permissions": [
    "detector:create",
    "detector:read",
    "dataset:upload",
    "detection:predict"
  ]
}
```

### Refresh Tokens

- **Lifetime:** 7 days (configurable)
- **Purpose:** Obtain new access tokens without re-authentication
- **Storage:** Secure, HttpOnly cookies (recommended)

**Refresh Flow:**
```bash
curl -X POST "http://localhost:8000/api/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "def50200a1b2c3d4e5f6..."}'
```

### Token Validation

The API validates tokens on each request:

1. **Signature verification** - Ensures token integrity
2. **Expiration check** - Rejects expired tokens
3. **Issuer validation** - Confirms token source
4. **Audience validation** - Verifies intended recipient
5. **Permission check** - Validates user permissions for endpoint

## Role-Based Access Control (RBAC)

### Available Roles

| Role | Description | Permissions |
|------|-------------|-------------|
| `admin` | Full system access | All permissions |
| `data_scientist` | ML model development | Create/manage detectors, datasets, experiments |
| `analyst` | Data analysis | Read access, run predictions |
| `viewer` | Read-only access | View detectors, datasets, results |

### Permission System

Permissions use the format: `resource:action`

**Resources:**
- `detector` - Anomaly detection models
- `dataset` - Training and test data
- `detection` - Prediction operations
- `experiment` - ML experiments
- `user` - User management
- `system` - System configuration

**Actions:**
- `create` - Create new resources
- `read` - View existing resources
- `update` - Modify resources
- `delete` - Remove resources
- `execute` - Run operations (train, predict)

### Permission Examples

```json
{
  "admin": [
    "detector:*", "dataset:*", "detection:*", 
    "experiment:*", "user:*", "system:*"
  ],
  "data_scientist": [
    "detector:create", "detector:read", "detector:update",
    "dataset:create", "dataset:read", "dataset:update",
    "detection:train", "detection:predict", "detection:explain",
    "experiment:create", "experiment:read", "experiment:update"
  ],
  "analyst": [
    "detector:read", "dataset:read",
    "detection:predict", "detection:explain",
    "experiment:read"
  ],
  "viewer": [
    "detector:read", "dataset:read", "experiment:read"
  ]
}
```

## API Key Authentication

For service-to-service communication, use API keys:

### Creating API Keys

```bash
curl -X POST "http://localhost:8000/api/auth/api-keys" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Data Pipeline Service",
    "description": "Automated data processing pipeline",
    "permissions": ["dataset:create", "detection:predict"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

**Response:**
```json
{
  "id": "ak_1234567890abcdef",
  "name": "Data Pipeline Service",
  "key": "pyn_live_1234567890abcdef1234567890abcdef",
  "permissions": ["dataset:create", "detection:predict"],
  "created_at": "2024-01-01T00:00:00Z",
  "expires_at": "2024-12-31T23:59:59Z"
}
```

### Using API Keys

```bash
curl -H "Authorization: Bearer pyn_live_1234567890abcdef1234567890abcdef" \
  "http://localhost:8000/api/detectors"
```

## Multi-Factor Authentication (MFA)

### Enabling MFA

1. **Generate MFA Secret:**
```bash
curl -X POST "http://localhost:8000/api/auth/mfa/setup" \
  -H "Authorization: Bearer your_jwt_token"
```

**Response:**
```json
{
  "secret": "JBSWY3DPEHPK3PXP",
  "qr_code": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "backup_codes": [
    "12345678", "87654321", "11223344"
  ]
}
```

2. **Verify MFA Setup:**
```bash
curl -X POST "http://localhost:8000/api/auth/mfa/verify" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{"token": "123456"}'
```

### MFA Login Flow

When MFA is enabled, the login process requires an additional step:

```bash
# Step 1: Initial authentication
curl -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user&password=pass"
```

**Response (MFA required):**
```json
{
  "mfa_required": true,
  "mfa_token": "temp_mfa_token_12345",
  "message": "MFA verification required"
}
```

```bash
# Step 2: MFA verification
curl -X POST "http://localhost:8000/api/auth/mfa/authenticate" \
  -H "Content-Type: application/json" \
  -d '{
    "mfa_token": "temp_mfa_token_12345",
    "mfa_code": "123456"
  }'
```

**Response (success):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "def50200a1b2c3d4e5f6..."
}
```

## Security Best Practices

### Client-Side Security

1. **Secure Token Storage:**
```javascript
// ✅ Good: Store in memory or secure storage
class TokenManager {
  constructor() {
    this.accessToken = null;
    this.refreshToken = null;
  }
  
  setTokens(accessToken, refreshToken) {
    this.accessToken = accessToken;
    // Store refresh token in httpOnly cookie or secure storage
    document.cookie = `refresh_token=${refreshToken}; HttpOnly; Secure; SameSite=Strict`;
  }
  
  getAccessToken() {
    return this.accessToken;
  }
}

// ❌ Bad: Store in localStorage
localStorage.setItem('token', accessToken); // Vulnerable to XSS
```

2. **Automatic Token Refresh:**
```javascript
class ApiClient {
  async makeRequest(url, options = {}) {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Authorization': `Bearer ${this.tokenManager.getAccessToken()}`,
          ...options.headers
        }
      });
      
      if (response.status === 401) {
        await this.refreshToken();
        // Retry original request
        return this.makeRequest(url, options);
      }
      
      return response;
    } catch (error) {
      throw error;
    }
  }
  
  async refreshToken() {
    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include' // Include httpOnly cookie
    });
    
    if (response.ok) {
      const tokens = await response.json();
      this.tokenManager.setTokens(tokens.access_token, tokens.refresh_token);
    } else {
      // Redirect to login
      window.location.href = '/login';
    }
  }
}
```

### Server-Side Security

1. **Token Validation Middleware:**
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256"],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "require": ["exp", "iat", "sub"]
            }
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

2. **Rate Limiting for Auth Endpoints:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/auth/token")
@limiter.limit("5/minute")
async def login(request: Request, ...):
    pass

@app.post("/api/auth/mfa/authenticate")
@limiter.limit("10/minute")
async def mfa_authenticate(request: Request, ...):
    pass
```

## Error Handling

### Authentication Errors

| Status Code | Error Code | Description | Solution |
|-------------|------------|-------------|----------|
| 401 | `TOKEN_MISSING` | No Authorization header | Include Bearer token |
| 401 | `TOKEN_INVALID` | Malformed or invalid token | Get new token |
| 401 | `TOKEN_EXPIRED` | Token has expired | Refresh token |
| 401 | `MFA_REQUIRED` | MFA verification needed | Complete MFA flow |
| 403 | `INSUFFICIENT_PERMISSIONS` | User lacks required permissions | Contact admin |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |

### Error Response Format

```json
{
  "detail": "Token has expired",
  "error_code": "TOKEN_EXPIRED",
  "timestamp": "2024-01-01T12:00:00Z",
  "documentation_url": "https://docs.pynomaly.io/api/authentication#token-expired"
}
```

## SDKs and Examples

### Python SDK

```python
from pynomaly_client import PynomalyClient

# Initialize with credentials
client = PynomalyClient(
    base_url="https://api.pynomaly.io",
    username="your_username",
    password="your_password"
)

# Or with API key
client = PynomalyClient(
    base_url="https://api.pynomaly.io",
    api_key="pyn_live_1234567890abcdef1234567890abcdef"
)

# Automatic token management
detectors = client.detectors.list()  # Handles authentication automatically
```

### JavaScript SDK

```javascript
import { PynomalyClient } from 'pynomaly-js';

const client = new PynomalyClient({
  baseUrl: 'https://api.pynomaly.io',
  username: 'your_username',
  password: 'your_password',
  // Automatic token refresh
  onTokenRefresh: (tokens) => {
    console.log('Tokens refreshed');
  },
  // Handle auth errors
  onAuthError: (error) => {
    console.error('Authentication error:', error);
    // Redirect to login
  }
});

// Usage
const detectors = await client.detectors.list();
```

### cURL Examples

```bash
#!/bin/bash

# Get token
TOKEN=$(curl -s -X POST "https://api.pynomaly.io/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=$USERNAME&password=$PASSWORD" | \
  jq -r '.access_token')

# Use token for API calls
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.pynomaly.io/detectors"

# Function to refresh token when needed
refresh_token() {
  local refresh_token=$1
  curl -s -X POST "https://api.pynomaly.io/auth/refresh" \
    -H "Content-Type: application/json" \
    -d "{\"refresh_token\": \"$refresh_token\"}" | \
    jq -r '.access_token'
}
```

## Testing and Development

### Development Environment

For development, you can disable authentication:

```python
# settings.py
AUTH_ENABLED = False  # Only for development!
```

### Testing with Mock Tokens

```python
import pytest
from fastapi.testclient import TestClient

def create_test_token(user_id: str, roles: list = None):
    """Create test JWT token."""
    if roles is None:
        roles = ["data_scientist"]
    
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "roles": roles
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")

def test_create_detector():
    client = TestClient(app)
    token = create_test_token("test_user", ["data_scientist"])
    
    response = client.post(
        "/api/detectors",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Test Detector", "algorithm_name": "IsolationForest"}
    )
    
    assert response.status_code == 201
```

## Troubleshooting

### Common Issues

1. **"Token has expired" errors:**
   - Implement automatic token refresh
   - Check system clock synchronization
   - Verify token lifetime configuration

2. **"Invalid token" errors:**
   - Check JWT secret key configuration
   - Verify token format and encoding
   - Ensure proper base64 encoding

3. **Permission denied errors:**
   - Verify user roles and permissions
   - Check endpoint permission requirements
   - Contact administrator for role updates

4. **MFA issues:**
   - Ensure clock synchronization on TOTP apps
   - Use backup codes if TOTP fails
   - Check MFA setup configuration

### Debug Mode

Enable debug logging for authentication:

```python
import logging

logging.getLogger("pynomaly.auth").setLevel(logging.DEBUG)
```

This will log detailed authentication information to help troubleshoot issues.

## Migration and Updates

### API Version Updates

When updating API versions, maintain backward compatibility:

```python
# Version-aware token validation
def validate_token_version(token: str, required_version: str = "v1"):
    payload = jwt.decode(token, verify=False)  # Don't verify for version check
    token_version = payload.get("api_version", "v1")
    
    if token_version != required_version:
        # Handle version mismatch
        if is_compatible_version(token_version, required_version):
            # Upgrade token claims if needed
            return upgrade_token_claims(payload)
        else:
            raise HTTPException(status_code=401, detail="Token version not supported")
    
    return validate_token(token)
```

This comprehensive authentication guide provides everything needed to securely implement and manage authentication in Pynomaly deployments.

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
