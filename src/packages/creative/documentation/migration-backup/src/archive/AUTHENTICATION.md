# Pynomaly Authentication & Security System

This document provides a comprehensive guide to the Pynomaly authentication and security system, which includes enterprise-grade features for user management, multi-factor authentication, and security monitoring.

## 🔒 Overview

The Pynomaly authentication system provides:

- **Multi-tenant user management** with role-based access control (RBAC)
- **Multi-factor authentication (MFA)** with TOTP, SMS, email, and backup codes
- **JWT-based authentication** with secure token management
- **API key authentication** for programmatic access
- **Comprehensive audit logging** for security compliance
- **Database persistence** with SQLAlchemy support
- **Email integration** for password reset and user invitations

## 🚀 Quick Start

### 1. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Required: Security settings
PYNOMALY_SECRET_KEY=your-very-secure-random-32-plus-character-secret-key-here
PYNOMALY_APP_ENVIRONMENT=development

# Required: Database (PostgreSQL recommended for production)
DATABASE_URL=postgresql://username:password@localhost:5432/pynomaly

# Optional: Email service (required for password reset and invitations)
PYNOMALY_SECURITY_SMTP_SERVER=smtp.gmail.com
PYNOMALY_SECURITY_SMTP_USERNAME=your-email@gmail.com
PYNOMALY_SECURITY_SMTP_PASSWORD=your-app-password
PYNOMALY_SECURITY_SENDER_EMAIL=your-email@gmail.com
```

### 2. Initialize the System

```python
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.auth.auth_setup import setup_authentication_system

# Load settings
settings = Settings()

# Initialize authentication system
setup_status = setup_authentication_system(settings)
print(f"Setup completed: {setup_status}")
```

### 3. API Usage

#### Authentication Endpoints

- `POST /auth/login` - Login with username/password
- `POST /auth/mfa/verify` - Complete MFA verification
- `POST /auth/register` - Register new user
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - Logout user
- `POST /auth/password-reset` - Request password reset
- `POST /auth/password-reset/confirm` - Confirm password reset

#### MFA Endpoints

- `POST /mfa/totp/setup` - Set up TOTP authentication
- `POST /mfa/totp/verify` - Verify TOTP setup
- `GET /mfa/status` - Get MFA status
- `POST /mfa/backup-codes/regenerate` - Generate new backup codes
- `GET /mfa/trusted-devices` - List trusted devices

#### User Management Endpoints

- `GET /api/users/me` - Get current user profile
- `POST /api/users/invite` - Invite user to tenant
- `GET /api/users` - List users (admin)
- `PUT /api/users/{id}` - Update user (admin)

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer                                │
├─────────────────────────────────────────────────────────────┤
│  Authentication    │   MFA Service   │  User Management     │
│  Middleware        │                 │  Service             │
├─────────────────────────────────────────────────────────────┤
│              Domain Layer (Entities & Repositories)         │
├─────────────────────────────────────────────────────────────┤
│  Database         │   Cache         │   Email Service      │
│  Repositories     │   (Redis)       │   (SMTP)             │
└─────────────────────────────────────────────────────────────┘
```

### Authentication Flow

1. **Initial Login**: User provides username/password
2. **MFA Check**: System checks if user has MFA enabled
3. **MFA Challenge**: If enabled, system sends MFA challenge
4. **Token Issuance**: After successful authentication, JWT tokens are issued
5. **Request Authorization**: Subsequent requests use JWT tokens

### Multi-Tenant Architecture

- **Users** can belong to multiple **Tenants**
- **Roles** are assigned per tenant (tenant-specific permissions)
- **Resource limits** are enforced at the tenant level
- **Usage tracking** for billing and compliance

## 🔐 Security Features

### Multi-Factor Authentication (MFA)

Support for multiple MFA methods:

- **TOTP** (Time-based One-Time Passwords) - Google Authenticator, Authy
- **SMS** codes (requires SMS service configuration)
- **Email** codes
- **Backup codes** for account recovery
- **Trusted devices** (optional MFA bypass)

### Audit Logging

Comprehensive security event logging:

- Authentication attempts (success/failure)
- MFA events (setup, verification, failures)
- User management actions (creation, updates, deletions)
- API key usage
- Permission changes
- Security violations

### Rate Limiting & Brute Force Protection

- **Login rate limiting** to prevent brute force attacks
- **API rate limiting** configurable per endpoint
- **Progressive delays** for repeated failed attempts
- **Account lockout** after maximum attempts

### Security Headers

Automatic security headers when enabled:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security` (HSTS)
- `Content-Security-Policy` (CSP)
- `Referrer-Policy`

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PYNOMALY_SECRET_KEY` | JWT signing key (32+ chars) | ⚠️ Default | ✅ |
| `DATABASE_URL` | Database connection string | SQLite | ❌ |
| `PYNOMALY_SECURITY_AUTH_ENABLED` | Enable authentication | `true` | ❌ |
| `PYNOMALY_SECURITY_JWT_EXPIRATION` | JWT expiration (seconds) | `3600` | ❌ |
| `PYNOMALY_SECURITY_SMTP_SERVER` | SMTP server hostname | `None` | ❌ |
| `PYNOMALY_SECURITY_SMTP_USERNAME` | SMTP username | `None` | ❌ |
| `PYNOMALY_SECURITY_SMTP_PASSWORD` | SMTP password | `None` | ❌ |
| `PYNOMALY_SECURITY_SENDER_EMAIL` | Sender email address | `None` | ❌ |

### Database Configuration

#### SQLite (Development)
```bash
DATABASE_URL=sqlite:///./pynomaly.db
```

#### PostgreSQL (Production)
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/pynomaly
PYNOMALY_DATABASE_POOL_SIZE=10
PYNOMALY_DATABASE_MAX_OVERFLOW=20
```

### Email Configuration

#### Gmail SMTP
```bash
PYNOMALY_SECURITY_SMTP_SERVER=smtp.gmail.com
PYNOMALY_SECURITY_SMTP_PORT=587
PYNOMALY_SECURITY_SMTP_USERNAME=youremail@gmail.com
PYNOMALY_SECURITY_SMTP_PASSWORD=your_app_password  # Use App Password
PYNOMALY_SECURITY_SENDER_EMAIL=youremail@gmail.com
PYNOMALY_SECURITY_SMTP_USE_TLS=true
```

#### Other SMTP Providers
```bash
# SendGrid
PYNOMALY_SECURITY_SMTP_SERVER=smtp.sendgrid.net
PYNOMALY_SECURITY_SMTP_PORT=587
PYNOMALY_SECURITY_SMTP_USERNAME=apikey
PYNOMALY_SECURITY_SMTP_PASSWORD=your_sendgrid_api_key

# Mailgun
PYNOMALY_SECURITY_SMTP_SERVER=smtp.mailgun.org
PYNOMALY_SECURITY_SMTP_PORT=587
PYNOMALY_SECURITY_SMTP_USERNAME=postmaster@your-domain.mailgun.org
PYNOMALY_SECURITY_SMTP_PASSWORD=your_mailgun_password
```

## 🔧 Development

### Running Tests

```bash
# Run authentication tests
pytest tests/test_auth/ -v

# Run MFA tests
pytest tests/test_mfa/ -v

# Run security tests
python tests/security/security_tests.py
```

### Security Testing

The system includes comprehensive security tests:

```python
from tests.security.security_tests import ComprehensiveSecurityTest

# Run security test suite
tester = ComprehensiveSecurityTest("http://localhost:8000")
results = tester.run_all_tests()
print(f"Security tests: {results['summary']['success_rate']:.2f}% pass rate")
```

### Database Migrations

```python
# Create new migration
from pynomaly.infrastructure.persistence.database_repositories import Base
from sqlalchemy import create_engine

engine = create_engine("your_database_url")
Base.metadata.create_all(bind=engine)
```

## 📋 API Reference

### Authentication

#### Login
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=SecurePassword123
```

**Response (No MFA):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Response (MFA Required):**
```json
{
  "mfa_required": true,
  "challenge_id": "challenge_12345_1234567890",
  "available_methods": ["totp", "backup_codes"],
  "message": "MFA verification required"
}
```

#### Complete MFA
```http
POST /auth/mfa/verify
Content-Type: application/json

{
  "challenge_id": "challenge_12345_1234567890",
  "method_type": "totp",
  "verification_code": "123456",
  "remember_device": false
}
```

### User Management

#### Create User
```http
POST /api/users
Authorization: Bearer <token>
Content-Type: application/json

{
  "email": "newuser@example.com",
  "username": "newuser",
  "first_name": "New",
  "last_name": "User",
  "password": "SecurePassword123!",
  "role": "viewer"
}
```

#### Invite User
```http
POST /api/users/invite
Authorization: Bearer <token>
Content-Type: application/json

{
  "email": "invite@example.com",
  "role": "analyst"
}
```

### MFA Management

#### Setup TOTP
```http
POST /mfa/totp/setup
Authorization: Bearer <token>
Content-Type: application/json

{
  "app_name": "Pynomaly",
  "issuer": "Pynomaly Security"
}
```

**Response:**
```json
{
  "secret": "JBSWY3DPEHPK3PXP",
  "qr_code_url": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
  "manual_entry_key": "JBSWY3DPEHPK3PXP",
  "backup_codes": ["12345678", "87654321", ...]
}
```

## 🚨 Security Best Practices

### Production Deployment

1. **Use strong secret keys** (32+ random characters)
2. **Enable HTTPS** and HSTS headers
3. **Configure PostgreSQL** instead of SQLite
4. **Enable audit logging** for compliance
5. **Set up monitoring** for security events
6. **Regular backup** of user data and audit logs
7. **Keep dependencies updated** for security patches

### User Security

1. **Enforce strong passwords** (8+ characters, mixed case, numbers, symbols)
2. **Require MFA** for administrative users
3. **Regular session cleanup** and logout inactive users
4. **Monitor for suspicious activity** (multiple login failures, unusual access patterns)
5. **Implement account lockout** after failed attempts

### API Security

1. **Rate limiting** on all endpoints
2. **Input validation** and sanitization
3. **Proper error handling** (don't leak sensitive information)
4. **CORS configuration** for web applications
5. **API key rotation** for service accounts

## 🐛 Troubleshooting

### Common Issues

#### Email Not Sending
- Check SMTP credentials and server settings
- Verify firewall allows SMTP connections
- Test with a simple SMTP client first
- Check application logs for detailed error messages

#### Database Connection Errors
- Verify DATABASE_URL format and credentials
- Ensure database server is running and accessible
- Check network connectivity and firewall rules
- Verify database user has necessary permissions

#### MFA Setup Issues
- Ensure cache service (Redis) is running for TOTP storage
- Check system time synchronization for TOTP
- Verify QR code generation dependencies are installed

#### JWT Token Issues
- Check secret key configuration
- Verify token expiration settings
- Ensure system clock is synchronized
- Check for token blacklisting if implemented

### Debug Mode

Enable debug logging for troubleshooting:

```bash
PYNOMALY_APP_DEBUG=true
PYNOMALY_MONITORING_LOG_LEVEL=DEBUG
```

### Health Check

Check system status:

```python
from pynomaly.infrastructure.auth.auth_setup import get_security_status

status = get_security_status()
print(f"System health: {status['overall_health']}")
print(f"Components: {status['components']}")
```

## 📞 Support

For authentication and security issues:

1. Check this documentation first
2. Review application logs for error details
3. Verify environment configuration
4. Test with minimal configuration
5. Create an issue with detailed reproduction steps

## 🔄 Updates

The authentication system is actively maintained with:

- Regular security updates
- New authentication methods
- Enhanced audit logging
- Performance improvements
- Additional compliance features

Check the [CHANGELOG.md](CHANGELOG.md) for recent updates and migration guides.