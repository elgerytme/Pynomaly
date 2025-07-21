# Python Language Conventions & Style Guide

## Overview

This document establishes comprehensive Python language conventions, naming standards, and code style guidelines for the repository. These rules ensure consistency, maintainability, and alignment with clean architecture principles across all packages.

## Core Principles

- **Consistency**: All code follows uniform naming and styling patterns
- **Readability**: Code is self-documenting through clear naming conventions  
- **Architecture Alignment**: Naming reflects clean architecture layers and DDD concepts
- **Python Standards**: Follows PEP 8 and modern Python best practices
- **Domain Clarity**: Names clearly express business concepts and technical intent

## Package and Module Naming

### Package Structure
```python
# Package names: lowercase with underscores for readability
src/packages/anomaly_detection/    # ✓ Good
src/packages/AnomalyDetection/     # ✗ Bad
src/packages/anomalydetection/     # ✗ Bad (hard to read)
```

### Module Organization by Architecture Layer
```python
# Domain layer modules
domain/
├── entities/user_account.py           # Business entities
├── value_objects/email_address.py     # Immutable value objects  
├── services/authentication_service.py # Domain services
├── repositories/user_repository.py    # Repository interfaces
└── exceptions/authentication_error.py # Domain exceptions

# Application layer modules  
application/
├── use_cases/register_user.py         # Use case implementations
├── services/user_management_service.py # Application services
├── dto/user_registration_request.py   # Data transfer objects
└── ports/notification_port.py         # Output port interfaces

# Infrastructure layer modules
infrastructure/
├── adapters/email_adapter.py          # External service adapters
├── persistence/sql_user_repository.py # Repository implementations
├── config/database_config.py          # Configuration
└── monitoring/health_checker.py       # Observability

# Presentation layer modules
presentation/
├── api/user_controller.py             # REST API controllers
├── cli/user_commands.py               # CLI command handlers
├── web/user_views.py                  # Web interface views
└── serializers/user_serializer.py     # Data serialization
```

## Class Naming Conventions

### Domain Layer Classes
```python
# Entities: PascalCase nouns representing business concepts
class UserAccount:           # ✓ Core business entity
class OrderLineItem:         # ✓ Aggregate component
class PaymentTransaction:    # ✓ Business process entity

# Value Objects: PascalCase with descriptive suffix
class EmailAddress:          # ✓ Simple value object
class MonetaryAmount:        # ✓ Complex value object
class GeographicCoordinate:  # ✓ Composite value object

# Domain Services: PascalCase with "Service" suffix
class AuthenticationService:  # ✓ Domain logic service
class PricingCalculationService: # ✓ Business rule service
class InventoryManagementService: # ✓ Domain coordination

# Repository Interfaces: PascalCase with "Repository" suffix
class UserRepository:        # ✓ Abstract repository
class OrderRepository:       # ✓ Domain repository interface
class ProductCatalogRepository: # ✓ Aggregate repository

# Domain Exceptions: PascalCase with "Error" or "Exception" suffix
class InvalidEmailAddressError:     # ✓ Validation error
class InsufficientFundsException:   # ✓ Business rule violation
class UserNotFoundError:            # ✓ Entity not found
```

### Application Layer Classes
```python
# Use Cases: PascalCase verbs describing user actions
class RegisterNewUser:       # ✓ User registration workflow
class ProcessPayment:        # ✓ Payment processing workflow
class GenerateMonthlyReport: # ✓ Reporting workflow

# Application Services: PascalCase with "Service" suffix
class UserManagementService:     # ✓ User lifecycle management
class OrderProcessingService:    # ✓ Order workflow coordination
class NotificationDeliveryService: # ✓ Cross-cutting service

# DTOs: PascalCase with "Request", "Response", or "DTO" suffix
class UserRegistrationRequest:   # ✓ Input DTO
class OrderSummaryResponse:      # ✓ Output DTO
class PaymentProcessingDTO:      # ✓ Internal transfer object

# Ports: PascalCase with "Port" suffix
class EmailNotificationPort:    # ✓ Output port for notifications
class PaymentGatewayPort:       # ✓ External service port
class AuditLoggingPort:         # ✓ Cross-cutting concern port
```

### Infrastructure Layer Classes
```python
# Adapters: PascalCase with "Adapter" suffix indicating technology
class PostgreSQLUserRepository:     # ✓ Database adapter
class SendGridEmailAdapter:         # ✓ External service adapter
class RedisSessionStorageAdapter:   # ✓ Caching adapter

# Configuration: PascalCase with "Config" or "Configuration" suffix
class DatabaseConfiguration:        # ✓ Database settings
class ApiServiceConfig:             # ✓ Service configuration
class MonitoringConfiguration:      # ✓ Observability config

# Monitoring: PascalCase with descriptive purpose
class ApplicationHealthChecker:     # ✓ Health monitoring
class PerformanceMetricsCollector:  # ✓ Metrics collection
class DistributedTracingManager:    # ✓ Tracing coordination
```

### Presentation Layer Classes
```python
# Controllers: PascalCase with "Controller" suffix
class UserController:           # ✓ REST API controller
class OrderManagementController: # ✓ API endpoint group
class ReportingController:      # ✓ Reporting endpoints

# CLI Commands: PascalCase with "Command" suffix
class UserRegistrationCommand:  # ✓ CLI command handler
class DatabaseMigrationCommand: # ✓ Administrative command
class SystemHealthCheckCommand: # ✓ Diagnostic command

# Serializers: PascalCase with "Serializer" suffix
class UserAccountSerializer:    # ✓ Entity serialization
class OrderResponseSerializer:  # ✓ Response formatting
class ErrorMessageSerializer:   # ✓ Error formatting
```

## Method and Function Naming

### Domain Methods
```python
class UserAccount:
    def validate_email_address(self, email: str) -> bool:    # ✓ Validation logic
    def calculate_account_balance(self) -> MonetaryAmount:   # ✓ Business calculation
    def apply_account_suspension(self, reason: str) -> None: # ✓ State change
    
    # Private methods: leading underscore
    def _encrypt_sensitive_data(self, data: str) -> str:     # ✓ Internal logic
    def _audit_account_change(self, change: str) -> None:    # ✓ Internal tracking

class AuthenticationService:
    def authenticate_user_credentials(self, credentials: UserCredentials) -> AuthenticationResult:
    def generate_access_token(self, user: UserAccount) -> AccessToken:
    def revoke_all_user_sessions(self, user_id: UserId) -> None:
```

### Application Methods
```python
class RegisterNewUser:
    def execute(self, request: UserRegistrationRequest) -> UserRegistrationResponse:
    
    # Private orchestration methods
    def _validate_registration_data(self, request: UserRegistrationRequest) -> None:
    def _create_user_account(self, validated_data: dict) -> UserAccount:
    def _send_welcome_notification(self, user: UserAccount) -> None:

class UserManagementService:
    def register_new_user(self, registration_data: dict) -> UserAccount:
    def update_user_profile(self, user_id: UserId, updates: dict) -> UserAccount:
    def deactivate_user_account(self, user_id: UserId, reason: str) -> None:
```

### Infrastructure Methods
```python
class PostgreSQLUserRepository:
    def save(self, user: UserAccount) -> None:
    def find_by_id(self, user_id: UserId) -> Optional[UserAccount]:
    def find_by_email(self, email: EmailAddress) -> Optional[UserAccount]:
    def delete(self, user_id: UserId) -> None:
    
    # Private implementation methods
    def _map_domain_to_database(self, user: UserAccount) -> dict:
    def _map_database_to_domain(self, row: dict) -> UserAccount:

class SendGridEmailAdapter:
    def send_email(self, recipient: EmailAddress, subject: str, body: str) -> None:
    def send_bulk_email(self, recipients: List[EmailAddress], template: EmailTemplate) -> None:
    
    # Private adapter methods
    def _format_sendgrid_request(self, email_data: dict) -> dict:
    def _handle_sendgrid_response(self, response: dict) -> None:
```

## Variable and Property Naming

### Domain Variables
```python
# Entity properties: snake_case reflecting business concepts
class UserAccount:
    def __init__(self):
        self.account_id: UserId = None
        self.email_address: EmailAddress = None
        self.registration_timestamp: datetime = None
        self.account_status: AccountStatus = None
        self.last_login_timestamp: Optional[datetime] = None

# Value object properties: descriptive names
class MonetaryAmount:
    def __init__(self, amount: Decimal, currency: Currency):
        self.amount_value: Decimal = amount
        self.currency_code: Currency = currency
        self.precision_digits: int = 2

# Method variables: clear purpose
def calculate_order_total(self, line_items: List[OrderLineItem]) -> MonetaryAmount:
    subtotal_amount = MonetaryAmount.zero()
    tax_calculation_rate = self._get_applicable_tax_rate()
    
    for current_line_item in line_items:
        item_total_price = current_line_item.calculate_total_price()
        subtotal_amount = subtotal_amount.add(item_total_price)
    
    final_tax_amount = subtotal_amount.multiply(tax_calculation_rate)
    return subtotal_amount.add(final_tax_amount)
```

### Collection Variables
```python
# Collections: plural nouns with descriptive context
active_user_accounts: List[UserAccount] = []
pending_order_requests: List[OrderRequest] = []
cached_product_catalog_items: Dict[ProductId, Product] = {}
failed_payment_transaction_ids: Set[TransactionId] = set()

# Iterators: singular form of collection name
for user_account in active_user_accounts:
    user_account.process_monthly_billing()

for order_request in pending_order_requests:
    order_request.validate_inventory_availability()
```

### Configuration Variables
```python
# Constants: UPPER_SNAKE_CASE
DATABASE_CONNECTION_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS_FOR_FAILED_OPERATIONS = 3
DEFAULT_PAGE_SIZE_FOR_PAGINATION = 50
API_RATE_LIMIT_REQUESTS_PER_MINUTE = 1000

# Configuration class properties
class DatabaseConfiguration:
    def __init__(self):
        self.connection_string: str = None
        self.max_connection_pool_size: int = 20
        self.query_timeout_milliseconds: int = 5000
        self.enable_query_logging: bool = False
```

## Type Hints and Documentation

### Comprehensive Type Annotations
```python
from typing import Optional, List, Dict, Set, Protocol, Union
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass

# Domain entity with complete type annotations
@dataclass(frozen=True)
class UserAccount:
    account_id: UserId
    email_address: EmailAddress
    registration_timestamp: datetime
    account_status: AccountStatus
    profile_data: Optional[UserProfileData] = None
    
    def calculate_account_age_in_days(self) -> int:
        current_timestamp = datetime.now()
        age_delta = current_timestamp - self.registration_timestamp
        return age_delta.days

# Repository protocol with precise types
class UserRepositoryProtocol(Protocol):
    def save(self, user: UserAccount) -> None: ...
    def find_by_id(self, user_id: UserId) -> Optional[UserAccount]: ...
    def find_by_email(self, email: EmailAddress) -> Optional[UserAccount]: ...
    def find_active_users(self, limit: int = 100) -> List[UserAccount]: ...

# Use case with detailed type specifications
class RegisterNewUser:
    def __init__(self, user_repository: UserRepositoryProtocol, 
                 email_service: EmailNotificationPort) -> None:
        self._user_repository = user_repository
        self._email_service = email_service
    
    def execute(self, request: UserRegistrationRequest) -> UserRegistrationResponse:
        # Implementation with type-safe operations
        pass
```

### Documentation Standards
```python
class AuthenticationService:
    """
    Domain service responsible for user authentication and session management.
    
    This service encapsulates the business rules for user authentication,
    including credential validation, session management, and security policies.
    It operates purely in the domain layer without external dependencies.
    """
    
    def authenticate_user_credentials(self, 
                                    credentials: UserCredentials) -> AuthenticationResult:
        """
        Authenticate user credentials against domain security policies.
        
        Args:
            credentials: User-provided authentication credentials containing
                        username/email and password information.
        
        Returns:
            AuthenticationResult containing success status, user account
            information (if successful), and any relevant security metadata.
        
        Raises:
            InvalidCredentialsError: When provided credentials are invalid.
            AccountSuspendedError: When user account is suspended or locked.
            SecurityPolicyViolationError: When authentication violates security rules.
        
        Business Rules:
            - Passwords must meet complexity requirements
            - Failed attempts are tracked for security monitoring
            - Suspended accounts cannot authenticate regardless of credentials
        """
        pass
```

## File and Directory Naming

### Package Structure
```
src/packages/user_management/
├── user_management/
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── user_account.py
│   │   │   └── user_profile.py
│   │   ├── value_objects/
│   │   │   ├── __init__.py
│   │   │   ├── email_address.py
│   │   │   ├── user_id.py
│   │   │   └── account_status.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── authentication_service.py
│   │   │   └── password_policy_service.py
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   └── user_repository.py
│   │   └── exceptions/
│   │       ├── __init__.py
│   │       ├── authentication_error.py
│   │       └── validation_error.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── use_cases/
│   │   │   ├── __init__.py
│   │   │   ├── register_new_user.py
│   │   │   ├── authenticate_user.py
│   │   │   └── update_user_profile.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── user_management_service.py
│   │   ├── dto/
│   │   │   ├── __init__.py
│   │   │   ├── user_registration_request.py
│   │   │   └── user_authentication_response.py
│   │   └── ports/
│   │       ├── __init__.py
│   │       ├── email_notification_port.py
│   │       └── audit_logging_port.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── sendgrid_email_adapter.py
│   │   │   └── cloudwatch_logging_adapter.py
│   │   ├── persistence/
│   │   │   ├── __init__.py
│   │   │   ├── postgresql_user_repository.py
│   │   │   └── redis_session_repository.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── database_configuration.py
│   │   │   └── security_configuration.py
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       ├── authentication_metrics_collector.py
│   │       └── user_activity_tracker.py
│   └── presentation/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── user_controller.py
│       │   └── authentication_controller.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── user_management_commands.py
│       │   └── authentication_commands.py
│       └── serializers/
│           ├── __init__.py
│           ├── user_account_serializer.py
│           └── authentication_result_serializer.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── acceptance/
├── docs/
│   ├── README.md
│   ├── architecture.md
│   └── api_reference.md
├── pyproject.toml
└── BUCK
```

## Import Organization

### Import Statement Order
```python
# Standard library imports
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Protocol
from decimal import Decimal

# Third-party imports
import pydantic
from sqlalchemy import Column, String, DateTime
from fastapi import APIRouter, HTTPException

# Local domain imports (same package)
from ..domain.entities.user_account import UserAccount
from ..domain.value_objects.email_address import EmailAddress
from ..domain.value_objects.user_id import UserId
from ..domain.services.authentication_service import AuthenticationService
from ..domain.exceptions.authentication_error import AuthenticationError

# Local application imports
from ..application.dto.user_registration_request import UserRegistrationRequest
from ..application.ports.email_notification_port import EmailNotificationPort

# Cross-package imports (other domains)
from src.packages.shared.infrastructure.logging import StructuredLogger
from src.packages.shared.domain.value_objects.timestamp import Timestamp
```

### Import Aliasing Guidelines
```python
# Avoid aliasing unless necessary for clarity or conflict resolution
from ..domain.entities.user_account import UserAccount  # ✓ Good
from ..domain.entities.user_account import UserAccount as UA  # ✗ Avoid

# Acceptable aliasing for conflict resolution
from ..domain.services.authentication_service import AuthenticationService
from ..infrastructure.adapters.ldap_authentication_service import LDAPAuthenticationService as LDAPAuthService

# Long module names can be aliased for readability
from ..infrastructure.persistence.postgresql_user_repository import PostgreSQLUserRepository as UserRepo
```

## Error Handling and Exception Naming

### Exception Hierarchy
```python
# Base domain exception
class UserManagementDomainError(Exception):
    """Base exception for all user management domain errors."""
    pass

# Validation exceptions
class UserValidationError(UserManagementDomainError):
    """Raised when user data fails domain validation rules."""
    pass

class InvalidEmailAddressError(UserValidationError):
    """Raised when email address format is invalid."""
    pass

class WeakPasswordError(UserValidationError):
    """Raised when password doesn't meet security requirements."""
    pass

# Business rule exceptions
class BusinessRuleViolationError(UserManagementDomainError):
    """Raised when an operation violates business rules."""
    pass

class DuplicateEmailAddressError(BusinessRuleViolationError):
    """Raised when attempting to register with existing email address."""
    pass

class AccountSuspendedError(BusinessRuleViolationError):
    """Raised when attempting operations on suspended account."""
    pass

# Not found exceptions
class EntityNotFoundError(UserManagementDomainError):
    """Raised when requested entity cannot be found."""
    pass

class UserAccountNotFoundError(EntityNotFoundError):
    """Raised when user account cannot be found by identifier."""
    pass
```

### Error Handling Patterns
```python
class AuthenticationService:
    def authenticate_user_credentials(self, 
                                    credentials: UserCredentials) -> AuthenticationResult:
        try:
            # Validate credentials format
            self._validate_credentials_format(credentials)
            
            # Find user account
            user_account = self._find_user_by_email(credentials.email_address)
            if user_account is None:
                raise UserAccountNotFoundError(f"No account found for email: {credentials.email_address}")
            
            # Check account status
            if user_account.is_suspended():
                raise AccountSuspendedError(f"Account {user_account.account_id} is suspended")
            
            # Verify password
            if not self._verify_password(credentials.password, user_account.password_hash):
                self._record_failed_authentication_attempt(user_account)
                raise InvalidCredentialsError("Invalid password provided")
            
            # Generate authentication result
            return AuthenticationResult.success(user_account)
            
        except UserValidationError as validation_error:
            self._log_validation_error(validation_error)
            raise
        except BusinessRuleViolationError as business_error:
            self._log_business_rule_violation(business_error)
            raise
        except Exception as unexpected_error:
            self._log_unexpected_error(unexpected_error)
            raise AuthenticationSystemError("Unexpected error during authentication") from unexpected_error
```

## Code Quality Standards

### Linting Configuration
```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings  
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "D",    # pydocstyle
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "S",    # flake8-bandit
    "T20",  # flake8-print
]

ignore = [
    "D100",  # Missing docstring in public module
    "D104",  # Missing docstring in public package
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "D"]  # Allow assert statements and skip docstrings in tests

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
```

### Code Formatting Standards
```python
# Maximum line length: 100 characters
# Use trailing commas for multi-line structures
# Use double quotes for strings consistently

# Method chaining: one call per line for readability
result = (
    user_registration_request
    .validate_required_fields()
    .normalize_email_address()
    .apply_security_policies()
    .generate_user_account()
)

# Dictionary formatting for complex structures
user_configuration = {
    "account_settings": {
        "email_notifications_enabled": True,
        "security_alerts_enabled": True,
        "marketing_communications_enabled": False,
    },
    "privacy_settings": {
        "profile_visibility": "private",
        "data_sharing_enabled": False,
        "analytics_tracking_enabled": True,
    },
    "feature_flags": {
        "beta_features_enabled": False,
        "experimental_ui_enabled": False,
        "advanced_reporting_enabled": True,
    },
}

# List formatting for multiple items
required_user_permissions = [
    "user_management:read",
    "user_management:write", 
    "profile_management:read",
    "profile_management:write",
    "authentication:manage_sessions",
]
```

## Domain-Driven Design Naming Patterns

### Ubiquitous Language
```python
# Use business terminology consistently across all layers
class CustomerAccount:  # Not "User" if business calls them "Customers"
    pass

class OrderFulfillment:  # Not "Shipping" if business uses "Fulfillment" 
    pass

class InventoryAdjustment:  # Use precise business terms
    pass

# Avoid technical jargon in domain layer
class CustomerRegistrationWorkflow:  # ✓ Business language
    pass

class CustomerRegistrationDAO:  # ✗ Technical database terminology
    pass
```

### Aggregate and Entity Naming
```python
# Aggregate roots: central business concepts
class CustomerAccount:  # Aggregate root for customer domain
    def __init__(self):
        self.customer_id: CustomerId = None
        self.contact_information: ContactInformation = None
        self.billing_addresses: List[BillingAddress] = []  # Aggregate components
        self.shipping_addresses: List[ShippingAddress] = []
        self.order_history: List[OrderReference] = []  # References to other aggregates

# Entities within aggregates
class BillingAddress:  # Entity within CustomerAccount aggregate
    def __init__(self):
        self.address_id: AddressId = None
        self.address_lines: AddressLines = None
        self.postal_code: PostalCode = None
        self.is_primary_billing_address: bool = False

# Value objects: immutable business concepts
@dataclass(frozen=True)
class PostalCode:
    value: str
    country_code: str
    
    def __post_init__(self):
        if not self._is_valid_format():
            raise InvalidPostalCodeError(f"Invalid postal code format: {self.value}")
```

## Testing Naming Conventions

### Test Organization
```python
# Test files: test_{module_name}.py
tests/
├── unit/
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── test_user_account.py
│   │   │   └── test_order_line_item.py
│   │   ├── value_objects/
│   │   │   ├── test_email_address.py
│   │   │   └── test_monetary_amount.py
│   │   └── services/
│   │       ├── test_authentication_service.py
│   │       └── test_pricing_calculation_service.py
│   ├── application/
│   │   ├── use_cases/
│   │   │   ├── test_register_new_user.py
│   │   │   └── test_process_payment.py
│   │   └── services/
│   │       └── test_user_management_service.py
│   └── infrastructure/
│       ├── persistence/
│       │   └── test_postgresql_user_repository.py
│       └── adapters/
│           └── test_sendgrid_email_adapter.py
├── integration/
│   ├── test_user_registration_workflow.py
│   └── test_payment_processing_workflow.py
└── acceptance/
    ├── test_complete_user_journey.py
    └── test_order_fulfillment_scenarios.py
```

### Test Method Naming
```python
class TestUserAccount:
    """Test suite for UserAccount entity business logic."""
    
    def test_should_create_valid_user_account_with_required_fields(self):
        """Test that user account creation succeeds with all required fields."""
        pass
    
    def test_should_raise_validation_error_when_email_address_is_invalid(self):
        """Test that invalid email address raises appropriate validation error."""
        pass
    
    def test_should_calculate_account_age_correctly_for_recent_registration(self):
        """Test that account age calculation is accurate for recently registered users."""
        pass
    
    def test_should_prevent_account_suspension_when_user_has_active_orders(self):
        """Test that business rule prevents suspension when user has pending orders."""
        pass

class TestAuthenticationService:
    """Test suite for authentication domain service."""
    
    def test_should_authenticate_user_successfully_with_valid_credentials(self):
        """Test successful authentication with correct username and password."""
        pass
    
    def test_should_reject_authentication_when_account_is_suspended(self):
        """Test that suspended accounts cannot authenticate regardless of credentials."""
        pass
    
    def test_should_track_failed_authentication_attempts_for_security_monitoring(self):
        """Test that failed login attempts are recorded for security analysis."""
        pass
```

## Performance and Memory Considerations

### Efficient Naming for Large-Scale Applications
```python
# Use descriptive but concise names for frequently accessed objects
class UserRepo:  # Acceptable abbreviation for very common internal usage
    pass

class UserRepository:  # Preferred full name for public APIs
    pass

# Avoid extremely long names that impact readability
class UserAccountManagementServiceImpl:  # ✗ Too verbose
    pass

class UserManagementService:  # ✓ Appropriately descriptive
    pass

# Cache-friendly naming patterns
_user_cache: Dict[UserId, UserAccount] = {}  # Clear cache purpose
_authentication_session_cache: Dict[SessionId, AuthenticationSession] = {}
```

## Enforcement and Automation

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: local
    hooks:
      - id: naming-convention-check
        name: Check Python naming conventions
        entry: python scripts/governance/naming_convention_checker.py
        language: system
        files: \.py$
```

### Automated Validation Scripts
```python
# scripts/governance/naming_convention_checker.py
"""
Automated validation of Python naming conventions across the monorepo.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any

class NamingConventionValidator:
    """Validates that code follows established naming conventions."""
    
    def validate_package_structure(self, package_path: Path) -> List[str]:
        """Validate package follows clean architecture naming patterns."""
        violations = []
        
        # Check for proper layer directory naming
        expected_layers = {"domain", "application", "infrastructure", "presentation"}
        actual_layers = {d.name for d in package_path.iterdir() if d.is_dir()}
        
        missing_layers = expected_layers - actual_layers
        if missing_layers:
            violations.append(f"Missing architecture layers: {missing_layers}")
        
        return violations
    
    def validate_class_naming(self, source_code: str, file_path: Path) -> List[str]:
        """Validate class names follow convention based on architectural layer."""
        violations = []
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Determine expected naming pattern based on file path
                if "/domain/entities/" in str(file_path):
                    if not self._is_valid_entity_name(class_name):
                        violations.append(f"Entity class '{class_name}' should use PascalCase noun")
                
                elif "/domain/services/" in str(file_path):
                    if not class_name.endswith("Service"):
                        violations.append(f"Domain service '{class_name}' should end with 'Service'")
                
                elif "/application/use_cases/" in str(file_path):
                    if not self._is_valid_use_case_name(class_name):
                        violations.append(f"Use case '{class_name}' should be PascalCase verb phrase")
        
        return violations
    
    def _is_valid_entity_name(self, name: str) -> bool:
        """Check if entity name follows PascalCase noun pattern."""
        return name[0].isupper() and "_" not in name
    
    def _is_valid_use_case_name(self, name: str) -> bool:
        """Check if use case name follows verb phrase pattern."""
        verb_patterns = ["Register", "Process", "Calculate", "Generate", "Update", "Delete", "Create"]
        return any(name.startswith(verb) for verb in verb_patterns)

if __name__ == "__main__":
    validator = NamingConventionValidator()
    # Run validation across entire codebase
    # Report violations and exit with appropriate status code
```

## Summary

This comprehensive language convention guide ensures that all Python code in the repository follows consistent, maintainable, and architecturally-aligned naming patterns. By following these conventions, developers can:

- Write self-documenting code that clearly expresses business intent
- Maintain consistency across all packages and architectural layers  
- Enable efficient code reviews and knowledge transfer
- Support automated tooling and governance validation
- Align code structure with clean architecture and DDD principles

Regular adherence to these conventions, combined with automated validation, ensures the long-term maintainability and clarity of the entire monorepo codebase.