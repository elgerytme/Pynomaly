# Authentication & Authorization Template

A comprehensive authentication and authorization system template with modern security practices, multi-factor authentication, and enterprise-grade features.

## Features

- **JWT Authentication**: Secure token-based authentication
- **Role-Based Access Control (RBAC)**: Granular permission system
- **Multi-Factor Authentication (MFA)**: TOTP, SMS, and email-based 2FA
- **OAuth2 Integration**: Google, GitHub, Microsoft, and custom providers
- **Session Management**: Secure session handling with Redis
- **Password Security**: Bcrypt hashing with complexity requirements
- **Account Management**: Registration, verification, password reset
- **Audit Logging**: Comprehensive security event logging
- **Rate Limiting**: Brute force protection and API throttling
- **Device Management**: Track and manage user devices
- **Security Headers**: CSRF, XSS, and other attack prevention
- **Compliance Ready**: GDPR, SOC2, and security framework compliance

## Directory Structure

```
auth-template/
├── build/                 # Build artifacts
├── deploy/                # Deployment configurations
├── docs/                  # Documentation
├── env/                   # Environment configurations
├── temp/                  # Temporary files
├── src/                   # Source code
│   └── auth_system/
│       ├── auth/         # Authentication logic
│       ├── authz/        # Authorization logic
│       ├── models/       # Data models
│       ├── services/     # Business services
│       ├── utils/        # Utility functions
│       ├── middleware/   # Security middleware
│       ├── providers/    # OAuth providers
│       └── api/          # API endpoints
├── tests/                # Test suites
├── examples/             # Usage examples
├── scripts/              # Security scripts
├── migrations/           # Database migrations
├── pyproject.toml        # Project configuration
├── docker-compose.yml   # Development environment
├── README.md            # Documentation
├── TODO.md              # Task tracking
└── CHANGELOG.md         # Version history
```

## Quick Start

1. **Clone the template**:
   ```bash
   git clone <template-repo> my-auth-system
   cd my-auth-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev,test]"
   ```

3. **Setup environment**:
   ```bash
   cp env/development/.env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**:
   ```bash
   alembic upgrade head
   ```

5. **Run the application**:
   ```bash
   uvicorn auth_system.main:app --reload
   ```

6. **Access the API**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Admin: http://localhost:8000/admin

## Authentication Features

### JWT Tokens

```python
from auth_system.auth.jwt_handler import JWTHandler

# Generate tokens
jwt_handler = JWTHandler()
access_token = jwt_handler.create_access_token(user_id=123)
refresh_token = jwt_handler.create_refresh_token(user_id=123)

# Verify tokens
payload = jwt_handler.decode_token(access_token)
user_id = payload.get("sub")
```

### Password Security

```python
from auth_system.auth.password_handler import PasswordHandler

# Hash password
password_handler = PasswordHandler()
hashed = password_handler.hash_password("user_password")

# Verify password
is_valid = password_handler.verify_password("user_password", hashed)

# Check password strength
strength = password_handler.check_password_strength("user_password")
```

### Multi-Factor Authentication

```python
from auth_system.auth.mfa_handler import MFAHandler

# Setup TOTP
mfa_handler = MFAHandler()
secret, qr_code = mfa_handler.setup_totp(user_id=123)

# Verify TOTP
is_valid = mfa_handler.verify_totp(user_id=123, token="123456")

# Send SMS code
mfa_handler.send_sms_code(user_id=123, phone_number="+1234567890")

# Verify SMS code
is_valid = mfa_handler.verify_sms_code(user_id=123, code="123456")
```

## Authorization Features

### Role-Based Access Control

```python
from auth_system.authz.rbac import RBACManager

# Create roles and permissions
rbac = RBACManager()
rbac.create_role("admin", description="Administrator role")
rbac.create_permission("user:read", description="Read user data")
rbac.assign_permission_to_role("admin", "user:read")

# Assign roles to users
rbac.assign_role_to_user(user_id=123, role="admin")

# Check permissions
has_permission = rbac.user_has_permission(
    user_id=123, 
    permission="user:read"
)
```

### Decorators for Route Protection

```python
from fastapi import APIRouter
from auth_system.authz.decorators import require_auth, require_permission

router = APIRouter()

@router.get("/protected")
@require_auth
async def protected_endpoint(current_user: User):
    return {"message": "This is protected"}

@router.get("/admin-only")
@require_permission("admin:access")
async def admin_endpoint(current_user: User):
    return {"message": "Admin access only"}
```

### Attribute-Based Access Control

```python
from auth_system.authz.abac import ABACEvaluator

# Define policies
policies = {
    "document_access": {
        "effect": "allow",
        "condition": {
            "and": [
                {"equals": ["user.department", "resource.department"]},
                {"in": ["read", "resource.allowed_actions"]}
            ]
        }
    }
}

# Evaluate access
abac = ABACEvaluator(policies)
access_granted = abac.evaluate(
    policy="document_access",
    user=user_context,
    resource=resource_context,
    action="read"
)
```

## API Endpoints

### Authentication Endpoints

```bash
# User registration
POST /auth/register
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "name": "John Doe"
}

# User login
POST /auth/login
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}

# Token refresh
POST /auth/refresh
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}

# Logout
POST /auth/logout

# Password reset request
POST /auth/password-reset/request
{
  "email": "user@example.com"
}

# Password reset confirm
POST /auth/password-reset/confirm
{
  "token": "reset-token",
  "new_password": "NewSecurePass123!"
}
```

### MFA Endpoints

```bash
# Setup TOTP
POST /auth/mfa/totp/setup

# Verify TOTP setup
POST /auth/mfa/totp/verify-setup
{
  "token": "123456"
}

# Enable MFA
POST /auth/mfa/enable
{
  "mfa_type": "totp"
}

# Verify MFA during login
POST /auth/mfa/verify
{
  "token": "123456",
  "temp_token": "temp-jwt-token"
}
```

### User Management Endpoints

```bash
# Get current user
GET /users/me

# Update user profile
PUT /users/me
{
  "name": "Updated Name",
  "email": "new@example.com"
}

# Change password
POST /users/me/change-password
{
  "current_password": "OldPass123!",
  "new_password": "NewPass123!"
}

# Get user sessions
GET /users/me/sessions

# Revoke session
DELETE /users/me/sessions/{session_id}
```

### Admin Endpoints

```bash
# List users (admin only)
GET /admin/users?page=1&size=10

# Get user details (admin only)
GET /admin/users/{user_id}

# Update user (admin only)
PUT /admin/users/{user_id}
{
  "is_active": false,
  "roles": ["user", "moderator"]
}

# Delete user (admin only)
DELETE /admin/users/{user_id}

# Audit logs (admin only)
GET /admin/audit-logs?user_id=123&action=login
```

## Security Features

### Rate Limiting

```python
from auth_system.middleware.rate_limiting import RateLimitMiddleware

# Configure rate limiting
rate_limits = {
    "login": {"requests": 5, "window": 300},  # 5 attempts per 5 minutes
    "register": {"requests": 3, "window": 3600},  # 3 per hour
    "password_reset": {"requests": 2, "window": 3600}  # 2 per hour
}

app.add_middleware(RateLimitMiddleware, rate_limits=rate_limits)
```

### CSRF Protection

```python
from auth_system.middleware.csrf import CSRFMiddleware

# Add CSRF protection
app.add_middleware(
    CSRFMiddleware,
    secret_key="your-csrf-secret",
    cookie_name="csrftoken",
    header_name="X-CSRFToken"
)
```

### Security Headers

```python
from auth_system.middleware.security_headers import SecurityHeadersMiddleware

# Add security headers
app.add_middleware(
    SecurityHeadersMiddleware,
    hsts_max_age=31536000,  # 1 year
    content_type_options="nosniff",
    frame_options="DENY",
    xss_protection="1; mode=block"
)
```

### Audit Logging

```python
from auth_system.services.audit_service import AuditService

# Log security events
audit = AuditService()

# Login event
audit.log_event(
    user_id=123,
    action="login",
    resource="auth",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0...",
    status="success"
)

# Permission denied event
audit.log_event(
    user_id=123,
    action="access_denied",
    resource="/admin/users",
    ip_address="192.168.1.1",
    status="denied",
    details={"reason": "insufficient_permissions"}
)
```

## OAuth2 Integration

### Google OAuth

```python
from auth_system.providers.google_oauth import GoogleOAuthProvider

# Setup Google OAuth
google_oauth = GoogleOAuthProvider(
    client_id="your-google-client-id",
    client_secret="your-google-client-secret",
    redirect_uri="http://localhost:8000/auth/oauth/google/callback"
)

# Get authorization URL
auth_url = google_oauth.get_authorization_url()

# Exchange code for tokens
tokens = await google_oauth.exchange_code_for_tokens(code)

# Get user info
user_info = await google_oauth.get_user_info(tokens["access_token"])
```

### Custom OAuth Provider

```python
from auth_system.providers.base_oauth import BaseOAuthProvider

class CustomOAuthProvider(BaseOAuthProvider):
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            authorization_url="https://provider.com/oauth/authorize",
            token_url="https://provider.com/oauth/token",
            user_info_url="https://provider.com/api/user"
        )
    
    def parse_user_info(self, user_data: dict) -> dict:
        return {
            "id": user_data["id"],
            "email": user_data["email"],
            "name": user_data["full_name"],
            "avatar_url": user_data["profile_image"]
        }
```

## Session Management

### Redis Session Store

```python
from auth_system.services.session_service import SessionService

# Create session
session_service = SessionService(redis_client)
session_id = await session_service.create_session(
    user_id=123,
    device_info={
        "device_type": "web",
        "browser": "Chrome",
        "os": "Windows 10",
        "ip_address": "192.168.1.1"
    }
)

# Get session
session = await session_service.get_session(session_id)

# Update session activity
await session_service.update_activity(session_id)

# Revoke session
await session_service.revoke_session(session_id)

# Revoke all user sessions
await session_service.revoke_all_user_sessions(user_id=123)
```

### Device Management

```python
from auth_system.services.device_service import DeviceService

# Register device
device_service = DeviceService()
device = await device_service.register_device(
    user_id=123,
    device_info={
        "device_id": "unique-device-id",
        "device_name": "iPhone 12",
        "device_type": "mobile",
        "os": "iOS 15.0",
        "app_version": "1.0.0"
    }
)

# List user devices
devices = await device_service.get_user_devices(user_id=123)

# Revoke device
await device_service.revoke_device(device_id="device-id")
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/authdb

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# Password Configuration
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SYMBOLS=true

# MFA Configuration
MFA_ISSUER_NAME=Your App Name
TOTP_VALIDITY_WINDOW=1

# Rate Limiting
RATE_LIMIT_ENABLED=true
REDIS_URL=redis://localhost:6379

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Security
CSRF_SECRET_KEY=your-csrf-secret
SECURE_COOKIES=true
COOKIE_DOMAIN=.example.com
```

### Settings Configuration

```python
from pydantic_settings import BaseSettings

class AuthSettings(BaseSettings):
    # Database
    database_url: str
    
    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    
    # Password
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    
    # MFA
    mfa_issuer_name: str = "Your App"
    totp_validity_window: int = 1
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    redis_url: str
    
    # Security
    csrf_secret_key: str
    secure_cookies: bool = True
    cookie_domain: str = ".example.com"
    
    class Config:
        env_file = ".env"
```

## Testing

### Authentication Tests

```python
import pytest
from auth_system.auth.jwt_handler import JWTHandler
from auth_system.auth.password_handler import PasswordHandler

class TestAuthentication:
    def test_password_hashing(self):
        handler = PasswordHandler()
        password = "TestPassword123!"
        
        hashed = handler.hash_password(password)
        assert handler.verify_password(password, hashed)
        assert not handler.verify_password("wrong_password", hashed)
    
    def test_jwt_tokens(self):
        handler = JWTHandler()
        user_id = 123
        
        # Test access token
        access_token = handler.create_access_token(user_id)
        payload = handler.decode_token(access_token)
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "access"
        
        # Test refresh token
        refresh_token = handler.create_refresh_token(user_id)
        payload = handler.decode_token(refresh_token)
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "refresh"
```

### Authorization Tests

```python
from auth_system.authz.rbac import RBACManager

class TestAuthorization:
    def test_rbac_permissions(self):
        rbac = RBACManager()
        
        # Setup
        rbac.create_role("editor")
        rbac.create_permission("content:edit")
        rbac.assign_permission_to_role("editor", "content:edit")
        rbac.assign_role_to_user(123, "editor")
        
        # Test
        assert rbac.user_has_permission(123, "content:edit")
        assert not rbac.user_has_permission(123, "admin:delete")
```

### API Tests

```python
from fastapi.testclient import TestClient
from auth_system.main import app

client = TestClient(app)

class TestAuthAPI:
    def test_register_user(self):
        response = client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "TestPass123!",
            "name": "Test User"
        })
        assert response.status_code == 201
        assert "access_token" in response.json()
    
    def test_login_user(self):
        # First register
        client.post("/auth/register", json={
            "email": "test@example.com",
            "password": "TestPass123!",
            "name": "Test User"
        })
        
        # Then login
        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "TestPass123!"
        })
        assert response.status_code == 200
        assert "access_token" in response.json()
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install dependencies
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e .

# Copy application
COPY src/ ./src/
COPY migrations/ ./migrations/

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "auth_system.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  auth-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/authdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=authdb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Security Best Practices

1. **Password Security**:
   - Enforce strong password policies
   - Use bcrypt with proper salt rounds
   - Implement password history
   - Regular password rotation

2. **Token Security**:
   - Short-lived access tokens
   - Secure refresh token storage
   - Token blacklisting
   - Regular key rotation

3. **Session Security**:
   - Secure session cookies
   - Session timeout
   - Concurrent session limits
   - Device fingerprinting

4. **API Security**:
   - Rate limiting and throttling
   - Input validation and sanitization
   - CORS configuration
   - Security headers

5. **Monitoring & Auditing**:
   - Comprehensive audit logging
   - Anomaly detection
   - Failed login monitoring
   - Security event alerting

## Compliance & Standards

### GDPR Compliance
- Data minimization
- Right to be forgotten
- Data portability
- Consent management

### Security Frameworks
- OWASP Top 10 protection
- NIST Cybersecurity Framework
- ISO 27001 alignment
- SOC 2 Type II readiness

## License

MIT License - see LICENSE file for details