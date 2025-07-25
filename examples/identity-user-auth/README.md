# User Authentication Service Example

A comprehensive authentication service implementing JWT-based authentication with OAuth2 support using Clean Architecture principles.

## Features

- **User Registration & Login**: Secure user account management
- **JWT Token Management**: Access and refresh token handling
- **OAuth2 Integration**: Support for Google, GitHub, Microsoft
- **Role-Based Access Control**: Fine-grained permission management
- **Password Security**: Bcrypt hashing with salt
- **Rate Limiting**: Protection against brute force attacks
- **Audit Logging**: Complete authentication event tracking

## Architecture

```
src/user_auth/
├── domain/              # Core business entities
│   ├── entities/        # User, Role, Permission entities
│   ├── repositories/    # Abstract repository interfaces
│   └── services/        # Domain services
├── application/         # Use cases and business logic
│   ├── use_cases/      # Authentication use cases
│   ├── dto/            # Data transfer objects
│   └── interfaces/     # Application interfaces
├── infrastructure/     # External adapters
│   ├── database/       # Database implementations
│   ├── security/       # JWT and OAuth2 implementations
│   └── monitoring/     # Metrics and logging
└── presentation/       # API controllers
    ├── api/            # REST API endpoints
    └── middleware/     # Authentication middleware
```

## Quick Start

```bash
# Start the service
cd examples/identity-user-auth
docker-compose up -d

# Run tests
make test

# Access the API
curl http://localhost:8000/docs
```

## API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh JWT token
- `POST /auth/logout` - Logout user
- `POST /auth/forgot-password` - Password reset request

### OAuth2
- `GET /auth/oauth2/{provider}` - OAuth2 authorization
- `GET /auth/oauth2/{provider}/callback` - OAuth2 callback

### User Management
- `GET /users/me` - Get current user profile
- `PUT /users/me` - Update user profile
- `GET /users/me/sessions` - List active sessions

## Technology Stack

- **Framework**: FastAPI with async/await
- **Database**: PostgreSQL with SQLAlchemy
- **Authentication**: JWT with PyJWT
- **Password Hashing**: Bcrypt
- **OAuth2**: Authlib
- **Caching**: Redis for session storage
- **Monitoring**: Prometheus metrics
- **Validation**: Pydantic models

## Security Features

### Password Security
- Minimum 8 characters with complexity requirements
- Bcrypt hashing with configurable rounds
- Password history to prevent reuse
- Account lockout after failed attempts

### Token Security
- Short-lived access tokens (15 minutes)
- Long-lived refresh tokens (7 days)
- Token blacklisting for logout
- Automatic token rotation

### Rate Limiting
- Login attempts: 5 per minute per IP
- Registration: 3 per hour per IP
- Password reset: 1 per hour per user

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/auth_db

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# OAuth2
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Redis
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_ENABLED=true
MAX_LOGIN_ATTEMPTS=5
```

## Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Security Tests
```bash
pytest tests/security/ -v
```

## Deployment

### Docker
```bash
docker build -t user-auth .
docker run -p 8000:8000 user-auth
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## Monitoring

- **Metrics**: Authentication success/failure rates, token generation
- **Logging**: All authentication events with user context
- **Alerts**: Failed login thresholds, token validation errors
- **Dashboards**: User activity, security events

## Extensions

This example can be extended with:
- **Multi-Factor Authentication**: TOTP, SMS, email verification
- **Single Sign-On**: SAML integration
- **Advanced RBAC**: Attribute-based access control
- **Compliance**: GDPR, CCPA data handling
- **Fraud Detection**: Behavioral analysis