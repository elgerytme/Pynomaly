# Enterprise Authentication & Authorization Package

This package provides enterprise-grade authentication and authorization capabilities for Pynomaly, including SSO, SAML, OAuth2, RBAC, and multi-tenancy support.

## Features

### 🔐 **Authentication Methods**
- **Local Authentication**: Username/password with configurable password policies
- **SAML SSO**: SAML 2.0 Service Provider integration
- **OAuth2/OIDC**: Support for major OAuth2 providers (Google, Microsoft, GitHub, etc.)
- **LDAP Integration**: Active Directory and LDAP authentication
- **Multi-Factor Authentication**: TOTP-based MFA with backup codes

### 🏢 **Multi-Tenancy**
- **Tenant Isolation**: Complete data and user isolation between tenants
- **Tenant-Specific Configuration**: Per-tenant authentication policies and settings
- **Subscription Management**: Feature-based access control by subscription plan
- **Custom Domains**: Tenant-specific custom domains and branding

### 🛡️ **Role-Based Access Control (RBAC)**
- **Hierarchical Roles**: Support for role inheritance and hierarchies  
- **Fine-Grained Permissions**: Resource-based and action-specific permissions
- **Custom Roles**: Tenant-specific role definitions
- **Temporal Permissions**: Time-limited role assignments and permissions

### 🔒 **Security Features**
- **Account Lockout**: Configurable failed login attempt protection
- **Session Management**: Concurrent session limits and device tracking
- **Password Policies**: Configurable complexity and rotation requirements
- **Audit Logging**: Comprehensive authentication event logging
- **IP Whitelisting**: Tenant-specific IP access restrictions

## Quick Start

### Installation

```bash
pip install pynomaly-enterprise-auth
```

### Basic Usage

```python
from enterprise_auth import AuthService, AuthConfig
from enterprise_auth.domain.entities import User, Tenant

# Initialize authentication service
config = AuthConfig(
    jwt_secret_key="your-secret-key",
    access_token_expire_minutes=60,
    max_failed_attempts=5
)

auth_service = AuthService(
    user_repository=user_repo,
    tenant_repository=tenant_repo,
    session_repository=session_repo,
    password_service=password_service,
    mfa_service=mfa_service,
    config=config
)

# Authenticate user
from enterprise_auth.application.dto import LoginRequest
login_request = LoginRequest(
    email="user@example.com",
    password="secure_password123"
)

response = await auth_service.authenticate_local(login_request)
if response.success:
    print(f"Login successful! Token: {response.access_token}")
```

### API Integration

```python
from fastapi import FastAPI, Depends
from enterprise_auth.presentation.api import AuthRouter

app = FastAPI()

# Include authentication router
app.include_router(AuthRouter(auth_service), prefix="/auth", tags=["authentication"])

# Protected endpoint example
@app.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello {current_user.email}!"}
```

### CLI Usage

```bash
# Setup authentication
pynomaly-enterprise-auth setup --tenant-id <tenant-id>

# Create user
pynomaly-enterprise-auth user create \
    --email admin@company.com \
    --first-name Admin \
    --last-name User \
    --role tenant_admin

# Setup SAML
pynomaly-enterprise-auth saml configure \
    --tenant-id <tenant-id> \
    --metadata-url https://idp.company.com/metadata

# Enable MFA for user
pynomaly-enterprise-auth mfa setup --user-id <user-id>
```

## Configuration

### Environment Variables

```bash
# JWT Configuration
PYNOMALY_JWT_SECRET_KEY=your-secret-key
PYNOMALY_JWT_ALGORITHM=HS256
PYNOMALY_ACCESS_TOKEN_EXPIRE_MINUTES=60
PYNOMALY_REFRESH_TOKEN_EXPIRE_DAYS=30

# Password Policy
PYNOMALY_MIN_PASSWORD_LENGTH=8
PYNOMALY_REQUIRE_PASSWORD_UPPERCASE=true
PYNOMALY_REQUIRE_PASSWORD_NUMBERS=true
PYNOMALY_REQUIRE_PASSWORD_SPECIAL=true

# Account Security
PYNOMALY_MAX_FAILED_ATTEMPTS=5
PYNOMALY_LOCKOUT_DURATION_MINUTES=30
PYNOMALY_MAX_CONCURRENT_SESSIONS=5

# MFA Settings
PYNOMALY_MFA_ISSUER_NAME="Pynomaly Enterprise"
PYNOMALY_MFA_TOKEN_LIFETIME_SECONDS=300

# Database
PYNOMALY_AUTH_DATABASE_URL=postgresql://user:pass@localhost/pynomaly_auth

# Redis (for sessions)
PYNOMALY_REDIS_URL=redis://localhost:6379/1
```

### Configuration File

Create `auth_config.yaml`:

```yaml
auth:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    access_token_expire_minutes: 60
    refresh_token_expire_days: 30
  
  password_policy:
    min_length: 8
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special: true
    max_age_days: 90
  
  security:
    max_failed_attempts: 5
    lockout_duration_minutes: 30
    max_concurrent_sessions: 5
    session_timeout_minutes: 480
  
  mfa:
    issuer_name: "Pynomaly Enterprise"
    token_lifetime_seconds: 300
    backup_codes_count: 8

# SSO Providers
sso:
  saml:
    enabled: true
    metadata_url: "https://idp.company.com/metadata"
    sp_entity_id: "pynomaly-sp"
  
  oauth2:
    enabled: true
    providers:
      google:
        client_id: "${GOOGLE_CLIENT_ID}"
        client_secret: "${GOOGLE_CLIENT_SECRET}"
        scopes: ["openid", "email", "profile"]
      
      microsoft:
        client_id: "${MICROSOFT_CLIENT_ID}"
        client_secret: "${MICROSOFT_CLIENT_SECRET}"
        tenant_id: "${MICROSOFT_TENANT_ID}"

# Multi-tenancy
tenancy:
  default_plan: "free"
  trial_duration_days: 14
  features_by_plan:
    free:
      - "api_access"
      - "basic_auth"
    starter:
      - "api_access" 
      - "basic_auth"
      - "custom_roles"
    professional:
      - "api_access"
      - "basic_auth"
      - "custom_roles"
      - "sso"
      - "audit_logs"
    enterprise:
      - "api_access"
      - "basic_auth"
      - "custom_roles"
      - "sso"
      - "saml"
      - "ldap"
      - "audit_logs"
      - "priority_support"
```

## Architecture

### Domain-Driven Design

The package follows Domain-Driven Design (DDD) principles:

```
enterprise_auth/
├── domain/                 # Pure business logic
│   ├── entities/          # User, Tenant, Permission, Role
│   ├── services/          # Domain services
│   └── repositories/      # Repository interfaces
├── application/           # Use cases and orchestration
│   ├── use_cases/         # Authentication use cases
│   ├── services/          # Application services
│   └── dto/               # Data transfer objects
├── infrastructure/        # External integrations
│   ├── adapters/          # Database, external service adapters
│   ├── persistence/       # Repository implementations
│   └── external/          # SSO provider integrations
└── presentation/          # User interfaces
    ├── api/               # REST API endpoints
    └── cli/               # Command-line interface
```

### Key Components

#### Domain Entities

- **User**: Core user entity with authentication and profile information
- **Tenant**: Multi-tenant organization entity with subscription management
- **Permission**: Atomic access control units with scope and resource targeting
- **Role**: Permission collections with hierarchy support
- **UserSession**: Session management with device tracking

#### Application Services

- **AuthService**: Main authentication orchestrator
- **TenantService**: Multi-tenant management
- **RBACService**: Role-based access control
- **MFAService**: Multi-factor authentication
- **PasswordService**: Password hashing and validation

#### Infrastructure Adapters

- **SAMLProvider**: SAML 2.0 authentication
- **OAuth2Provider**: OAuth2/OIDC integration
- **LDAPProvider**: LDAP/Active Directory authentication
- **EmailService**: Email notifications and verification

## API Reference

### Authentication Endpoints

```http
POST /auth/login
POST /auth/register  
POST /auth/logout
POST /auth/refresh
POST /auth/forgot-password
POST /auth/reset-password
POST /auth/change-password
```

### Multi-Factor Authentication

```http
POST /auth/mfa/setup
POST /auth/mfa/verify
DELETE /auth/mfa/disable
POST /auth/mfa/backup-codes
```

### SSO Endpoints

```http
GET /auth/saml/login/{tenant_id}
POST /auth/saml/callback
GET /auth/oauth2/login/{provider}
POST /auth/oauth2/callback/{provider}
```

### User Management

```http
GET /users/profile
PUT /users/profile
GET /users/sessions
DELETE /users/sessions/{session_id}
POST /users/bulk-actions
```

### Administration

```http
GET /admin/users
POST /admin/users
PUT /admin/users/{user_id}
DELETE /admin/users/{user_id}
GET /admin/roles
POST /admin/roles
PUT /admin/roles/{role_id}
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# E2E tests
pytest tests/e2e/

# With coverage
pytest --cov=enterprise_auth --cov-report=html
```

### Test Categories

- **Unit Tests**: Domain logic and service tests
- **Integration Tests**: Database and external service integration
- **E2E Tests**: Complete authentication flows
- **Security Tests**: Authentication bypass and security validation
- **Performance Tests**: Load testing for authentication endpoints

### Test Configuration

```python
# conftest.py
import pytest
from enterprise_auth import AuthService, AuthConfig
from enterprise_auth.infrastructure.persistence import InMemoryUserRepository

@pytest.fixture
def auth_config():
    return AuthConfig(
        jwt_secret_key="test-secret",
        access_token_expire_minutes=60
    )

@pytest.fixture
def auth_service(auth_config):
    return AuthService(
        user_repository=InMemoryUserRepository(),
        tenant_repository=InMemoryTenantRepository(),
        session_repository=InMemorySessionRepository(),
        config=auth_config
    )
```

## Security Considerations

### Password Security
- Bcrypt hashing with configurable rounds
- Password complexity requirements
- Password history prevention
- Secure password reset flows

### Session Security
- JWT with short expiration times
- Secure refresh token rotation
- Session invalidation on logout
- Concurrent session management

### Multi-Factor Authentication
- TOTP-based authenticator apps
- Backup codes for recovery
- Rate limiting on MFA attempts
- Secure secret generation

### Audit and Monitoring
- Comprehensive authentication logging
- Failed login attempt tracking
- Session anomaly detection
- Real-time security alerts

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "enterprise_auth.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-enterprise-auth
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly-enterprise-auth
  template:
    metadata:
      labels:
        app: pynomaly-enterprise-auth
    spec:
      containers:
      - name: auth-service
        image: pynomaly/enterprise-auth:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYNOMALY_JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: jwt-secret
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/sso-integration`
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly/src/packages/enterprise/enterprise_auth

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev,test,lint]"

# Run tests
pytest

# Run linting
ruff check .
mypy enterprise_auth/
```

## License

This package is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [https://docs.pynomaly.org/enterprise/auth](https://docs.pynomaly.org/enterprise/auth)
- **Issues**: [GitHub Issues](https://github.com/yourusername/pynomaly/issues)
- **Enterprise Support**: enterprise-support@pynomaly.org

---

**Enterprise Authentication & Authorization Package** - Part of the Pynomaly Enterprise Suite