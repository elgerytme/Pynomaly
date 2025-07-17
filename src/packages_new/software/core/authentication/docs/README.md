# Authentication Feature

## Overview

The Authentication feature provides secure user authentication capabilities including login, logout, token management, and user session handling.

## Architecture

This feature follows the Domain → Application → Infrastructure layered architecture:

### Domain Layer
- **Entities**: `User` - Core user entity with authentication properties
- **Value Objects**: `Email` - Email validation and handling
- **Services**: `AuthenticationService` - Password hashing, validation, and authentication logic
- **Repositories**: `UserRepository` - Abstract interface for user data access

### Application Layer
- **Use Cases**: `LoginUserUseCase` - User login workflow
- **DTOs**: Request/Response data transfer objects
- **Services**: `TokenService` - JWT token management
- **User Stories**: Structured user requirements and acceptance criteria

### Infrastructure Layer
- **API**: REST endpoints for authentication (`/auth/login`, `/auth/logout`, etc.)
- **CLI**: Command-line interface for authentication operations
- **Repositories**: SQL implementation of user repository
- **Models**: SQLAlchemy database models

## Features

### 1. User Authentication
- Secure password hashing using PBKDF2-SHA256
- Login with email or username
- Account lockout after failed attempts
- Session management with JWT tokens

### 2. Security Features
- Password strength validation
- Account verification requirement
- Failed login attempt tracking
- Automatic account lockout (5 attempts → 30 min lockout)
- Secure token generation and validation

### 3. Token Management
- JWT access tokens (1 hour expiry)
- JWT refresh tokens (7 days expiry)
- Token validation and refresh
- Token blacklisting support

## API Endpoints

### POST /auth/login
Authenticate user and return JWT tokens.

**Request:**
```json
{
  "identifier": "user@example.com",
  "password": "securepassword",
  "remember_me": false
}
```

**Response:**
```json
{
  "success": true,
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "expires_in": 3600
}
```

### POST /auth/logout
Invalidate current access token.

### POST /auth/refresh
Refresh access token using refresh token.

### GET /auth/me
Get current user information from access token.

## CLI Commands

### auth login
```bash
pynomaly auth login --username user@example.com
```

### auth logout
```bash
pynomaly auth logout
```

### auth whoami
```bash
pynomaly auth whoami
```

### auth validate-token
```bash
pynomaly auth validate-token
```

## Business Rules

1. **Account Verification**: Users must verify their email before login
2. **Account Status**: Only active users can login
3. **Failed Attempts**: Account locks after 5 failed attempts for 30 minutes
4. **Password Policy**: Minimum 8 characters with uppercase, lowercase, digit, and special character
5. **Token Expiry**: Access tokens expire in 1 hour, refresh tokens in 7 days

## Database Schema

### users table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
```

## Testing

### Unit Tests
- Domain entity business logic
- Authentication service methods
- Use case orchestration
- Token service functionality

### Integration Tests
- API endpoint testing
- Database repository testing
- CLI command testing
- End-to-end authentication flow

### Test Examples
```python
# Domain tests
def test_user_can_login_when_active_and_verified():
    user = User(...)
    user.is_active = True
    user.is_verified = True
    assert user.can_login() == True

# Application tests
def test_login_use_case_with_valid_credentials():
    # Test login workflow
    pass

# Infrastructure tests
def test_login_endpoint_returns_tokens():
    # Test API endpoint
    pass
```

## Configuration

### Environment Variables
```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/dbname

# Security Configuration
PASSWORD_MIN_LENGTH=8
ACCOUNT_LOCKOUT_ATTEMPTS=5
ACCOUNT_LOCKOUT_DURATION_MINUTES=30
```

## Security Considerations

1. **Password Security**: Use strong hashing (PBKDF2-SHA256) with salt
2. **Token Security**: Use secure secret keys and appropriate expiry times
3. **Account Lockout**: Prevent brute force attacks with account lockouts
4. **Input Validation**: Validate all inputs and sanitize data
5. **Rate Limiting**: Implement rate limiting on authentication endpoints
6. **HTTPS**: Always use HTTPS for authentication in production
7. **Token Storage**: Store tokens securely (HTTP-only cookies recommended)

## Dependencies

### Domain Layer
- No external dependencies (pure Python)

### Application Layer
- `PyJWT` for token management
- `passlib` for password hashing (optional)

### Infrastructure Layer
- `FastAPI` for API endpoints
- `Click` for CLI commands
- `SQLAlchemy` for database access
- `PostgreSQL` for data storage

## Future Enhancements

1. **Multi-Factor Authentication**: Add TOTP/SMS support
2. **OAuth Integration**: Add Google/GitHub OAuth
3. **Session Management**: Add session storage and management
4. **Audit Logging**: Add comprehensive audit logging
5. **Password Reset**: Add password reset functionality
6. **Social Login**: Add social media authentication
7. **Biometric Authentication**: Add fingerprint/face recognition support