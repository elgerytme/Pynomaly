# People Operations Package

Comprehensive user management, authentication, authorization, and compliance for the Monorepo platform.

## Overview

This package manages all aspects of user operations, from authentication and authorization to user lifecycle management and compliance. It provides a centralized approach to handling people-related operations across the platform.

## Architecture

```
people_ops/
├── authentication/     # User authentication systems (JWT, OAuth, MFA)
├── authorization/      # Role-based access control and permissions
├── user_management/    # User lifecycle and profile management
├── compliance/         # Audit trails, data privacy, and regulatory compliance
├── sessions/           # Session management and security
└── policies/           # Security policies and governance
```

## Key Features

- **Multi-Factor Authentication**: Support for TOTP, SMS, and hardware tokens
- **Role-Based Access Control**: Granular permissions and role management
- **User Lifecycle Management**: Registration, onboarding, deactivation
- **Audit Trails**: Comprehensive logging of user actions
- **Data Privacy**: GDPR/CCPA compliance and data protection
- **Session Management**: Secure session handling and timeout policies
- **Enterprise Integration**: LDAP, SAML, and SSO support

## Installation

This package is part of the Pynomaly monorepo. Install auth dependencies:

```bash
# Core authentication dependencies
pip install pyjwt passlib bcrypt

# Multi-factor authentication
pip install pyotp qrcode

# Enterprise integration
pip install python-ldap psaml2
```

## Usage

### User Authentication
```python
from people_ops.authentication import AuthService

auth = AuthService()
user = auth.authenticate(username="john.doe", password="secure_password")
token = auth.generate_token(user)
```

### Authorization
```python
from people_ops.authorization import PermissionService

permissions = PermissionService()
if permissions.check_permission(user, "anomaly_detection.read"):
    # User has permission to read anomaly detection data
    pass
```

### User Management
```python
from people_ops.user_management import UserService

user_service = UserService()
new_user = user_service.create_user({
    "username": "jane.smith",
    "email": "jane@company.com",
    "roles": ["analyst"]
})
```

### Compliance Auditing
```python
from people_ops.compliance import AuditService

audit = AuditService()
audit.log_action(user_id=123, action="data_access", resource="sensitive_dataset")
```

## Dependencies

- **Core**: `pyjwt`, `passlib`, `bcrypt`
- **MFA**: `pyotp`, `qrcode`
- **Enterprise**: `python-ldap`, `psaml2`
- **Internal**: `core`, `infrastructure`

## Components

### Authentication
- JWT token management
- Password hashing and validation
- Multi-factor authentication
- OAuth 2.0 and OpenID Connect
- Enterprise SSO integration

### Authorization
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Permission matrices and hierarchies
- Resource-level permissions
- Dynamic policy evaluation

### User Management
- User registration and onboarding
- Profile management and updates
- Account activation/deactivation
- Password reset workflows
- User directory services

### Compliance
- Audit trail generation
- Data access logging
- Privacy controls and consent management
- Regulatory compliance reporting
- Data retention policies

## Security Features

- Secure password storage with bcrypt
- JWT token signing and validation
- Rate limiting for authentication attempts
- Session timeout and refresh handling
- Encrypted data transmission
- Security event monitoring

## Testing

```bash
pytest tests/people_ops/
```

## Contributing

See the main repository CONTRIBUTING.md for guidelines.

## License

MIT License - see main repository LICENSE file.