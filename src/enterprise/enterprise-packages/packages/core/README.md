# Enterprise Core Framework

The foundational package for building enterprise applications with clean architecture and domain-driven design principles.

## Features

### Domain Layer Abstractions

- **Base Entity**: Generic entity class with identity, timestamps, and versioning
- **Base Value Object**: Immutable value objects with validation
- **Aggregate Root**: Domain aggregate management with dirty tracking
- **Domain Events**: Event-driven architecture support
- **Specifications**: Business rule encapsulation with composable logic

### Infrastructure Patterns

- **Dependency Injection**: Advanced DI container with service registry
- **Configuration Management**: Environment-aware configuration with secrets support
- **Feature Flags**: Dynamic feature rollouts with context-based evaluation
- **Service Registry**: Centralized service management with health checks
- **Event Bus**: Decoupled event publishing and subscription

### Protocol Definitions

- **Repository**: Data persistence abstraction
- **Unit of Work**: Transaction management
- **Cache**: Caching abstraction with TTL support
- **Message Queue**: Asynchronous messaging
- **Health Checks**: Service monitoring and status reporting
- **Metrics**: Application performance monitoring

### Exception Hierarchy

- **Structured Errors**: Rich error information with context
- **Domain Exceptions**: Business rule violations and validation errors
- **Infrastructure Exceptions**: External service and persistence errors
- **Security Exceptions**: Authentication and authorization failures

## Quick Start

### Installation

```bash
pip install enterprise-core
```

### Basic Usage

```python
from enterprise_core import (
    BaseEntity,
    BaseValueObject,
    Container,
    ConfigurationManager,
    FeatureFlagManager,
)
from uuid import UUID, uuid4

# Define a value object
class Money(BaseValueObject):
    amount: float
    currency: str = "USD"
    
    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(amount=self.amount + other.amount, currency=self.currency)

# Define an entity
class Order(BaseEntity[UUID]):
    customer_id: UUID
    total_amount: Money
    status: str = "pending"
    
    def confirm(self) -> None:
        if self.status != "pending":
            raise ValueError("Can only confirm pending orders")
        self.status = "confirmed"
        self.mark_as_updated()

# Create and use entities
order = Order(
    id=uuid4(),
    customer_id=uuid4(),
    total_amount=Money(amount=99.99),
)

order.confirm()
print(f"Order {order.id} is {order.status}")
```

### Dependency Injection

```python
from enterprise_core import Container, service

# Define a service
@service("email_service")
class EmailService:
    def __init__(self, config: dict):
        self.config = config
    
    async def send_email(self, to: str, subject: str, body: str) -> None:
        # Implementation here
        pass

# Configure container
container = Container()
container.config.from_dict({
    "email": {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
    }
})

# Use the service
email_service = container.email_service()
await email_service.send_email("user@example.com", "Hello", "World")
```

### Feature Flags

```python
from enterprise_core import FeatureFlagManager

# Configure feature flags
flags_config = {
    "new_checkout": {
        "enabled": True,
        "environments": {
            "production": False,
            "staging": True,
        },
        "conditions": [
            {
                "type": "equals",
                "field": "user_type",
                "value": "premium",
                "enabled": True,
            }
        ]
    }
}

flag_manager = FeatureFlagManager(flags_config)

# Check feature flags
if flag_manager.is_enabled("new_checkout", {"user_type": "premium"}):
    # Use new checkout flow
    pass
else:
    # Use old checkout flow
    pass
```

### Configuration Management

```python
from enterprise_core import ConfigurationManager, EnterpriseSettings

# Define application settings
class AppSettings(EnterpriseSettings):
    database_url: str = "sqlite:///app.db"
    redis_url: str = "redis://localhost:6379"
    secret_key: str = "change-me"

# Load configuration
config_manager = ConfigurationManager(AppSettings())

# Access configuration
db_url = config_manager.get("database_url")
is_prod = config_manager.is_production
```

## Architecture Principles

### Clean Architecture

- **Dependency Rule**: Dependencies point inward toward the domain
- **Layer Separation**: Clear boundaries between domain, application, and infrastructure
- **Interface Segregation**: Small, focused interfaces for better testability

### Domain-Driven Design

- **Ubiquitous Language**: Shared vocabulary between developers and domain experts
- **Bounded Contexts**: Clear boundaries around related functionality
- **Aggregate Design**: Consistency boundaries and transaction management

### SOLID Principles

- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes must be substitutable for base types
- **Interface Segregation**: Clients depend only on interfaces they use
- **Dependency Inversion**: Depend on abstractions, not concretions

## Advanced Usage

### Custom Repository Implementation

```python
from enterprise_core import Repository
from typing import Optional, List

class UserRepository(Repository[User, UUID]):
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def save(self, user: User) -> None:
        # Implementation
        pass
    
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        # Implementation
        pass
    
    async def find_by_email(self, email: str) -> Optional[User]:
        # Custom method
        pass
```

### Event-Driven Architecture

```python
from enterprise_core import DomainEvent, EventBus

class OrderConfirmed(DomainEvent):
    event_type: str = "order.confirmed"
    order_id: UUID
    customer_id: UUID

class OrderEventHandler:
    async def handle(self, event: DomainEvent) -> None:
        if isinstance(event, OrderConfirmed):
            # Send confirmation email
            pass

# Usage
event_bus = EventBus()
handler = OrderEventHandler()
event_bus.subscribe("order.confirmed", handler)

order.add_domain_event(OrderConfirmed(
    order_id=order.id,
    customer_id=order.customer_id,
))
```

### Health Checks

```python
from enterprise_core import HealthCheck, HealthStatus

class DatabaseHealthCheck(HealthCheck):
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def check(self) -> HealthStatus:
        try:
            await self.db.execute("SELECT 1")
            return HealthStatus(
                status="healthy",
                message="Database connection is working",
            )
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"Database connection failed: {e}",
                details={"error": str(e)},
            )
```

## Testing

Enterprise Core includes comprehensive testing utilities:

```python
import pytest
from enterprise_core.testing import (
    MockRepository,
    TestContainer,
    EntityBuilder,
)

def test_order_confirmation():
    # Use test utilities
    container = TestContainer()
    order_repo = MockRepository[Order, UUID]()
    
    # Test order confirmation
    order = Order(id=uuid4(), customer_id=uuid4(), total_amount=Money(99.99))
    order.confirm()
    
    assert order.status == "confirmed"
    assert order.version == 2  # Updated
```

## Configuration Options

### Environment Variables

- `ENVIRONMENT`: Application environment (development, staging, production)
- `DEBUG`: Enable debug mode (true/false)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT`: Logging format (json, text)

### Feature Flags

Feature flags can be configured via:

- Configuration files (JSON, YAML)
- Environment variables (`FEATURE_<FLAG_NAME>`)
- Runtime API calls

### Service Registry

Services can be registered programmatically or via configuration:

```yaml
services:
  email_service:
    class: "myapp.services.EmailService"
    config:
      smtp_host: "smtp.example.com"
      smtp_port: 587
  
  cache_service:
    class: "enterprise_adapters.RedisCache"
    config:
      redis_url: "redis://localhost:6379"
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to this package.

## License

MIT License - see [LICENSE](../../LICENSE) file for details.
