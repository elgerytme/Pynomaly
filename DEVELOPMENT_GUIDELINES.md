# Development Guidelines for Feature Architecture

## Overview

This document provides comprehensive guidelines for developing within the domain → package → feature → layer architecture. Follow these guidelines to ensure consistent, maintainable, and well-architected code.

---

## 🏗️ Architecture Principles

### 1. Domain-Driven Design (DDD)
- **Domains** represent major business areas (ai, business, data, software, etc.)
- **Packages** group related functionality within a domain
- **Features** encapsulate specific business capabilities
- **Layers** separate concerns within each feature

### 2. Dependency Direction
```
Infrastructure → Application → Domain
```
- **Domain layer** has no dependencies on other layers
- **Application layer** can depend on Domain layer only
- **Infrastructure layer** can depend on Application and Domain layers

### 3. Feature Isolation
- Features should be self-contained and independent
- Cross-feature communication should go through shared components
- Features should not directly import from other features

---

## 📁 Directory Structure

### Standard Feature Structure
```
src/packages/
├── {domain}/                    # e.g., ai, business, data, software
│   ├── {package}/              # e.g., machine_learning, analytics
│   │   ├── {feature}/          # e.g., model_lifecycle, user_management
│   │   │   ├── domain/         # Pure business logic
│   │   │   │   ├── entities/   # Domain entities
│   │   │   │   ├── services/   # Domain services
│   │   │   │   ├── repositories/  # Repository interfaces
│   │   │   │   └── value_objects/ # Value objects
│   │   │   ├── application/    # Application orchestration
│   │   │   │   ├── use_cases/  # Use case implementations
│   │   │   │   ├── user_stories/  # User story definitions
│   │   │   │   ├── story_maps/ # Story mapping artifacts
│   │   │   │   ├── services/   # Application services
│   │   │   │   └── dto/        # Data transfer objects
│   │   │   ├── infrastructure/ # External interfaces
│   │   │   │   ├── api/        # REST API endpoints
│   │   │   │   ├── cli/        # Command-line interfaces
│   │   │   │   ├── gui/        # Web UI applications
│   │   │   │   ├── adapters/   # External system adapters
│   │   │   │   └── repositories/ # Repository implementations
│   │   │   ├── docs/           # Feature documentation
│   │   │   ├── tests/          # Feature tests
│   │   │   └── scripts/        # Feature automation
│   │   └── shared/             # Package-level shared components
│   └── docs/                   # Domain documentation
```

---

## 🎯 Layer Development Guidelines

### Domain Layer Guidelines

#### **Purpose**
Contains pure business logic, entities, and business rules. No external dependencies.

#### **Do's**
- ✅ Define domain entities with business behavior
- ✅ Create value objects for domain concepts
- ✅ Implement domain services for complex business logic
- ✅ Define repository interfaces for data access
- ✅ Enforce business rules and invariants
- ✅ Use pure Python without external frameworks

#### **Don'ts**
- ❌ Import from application or infrastructure layers
- ❌ Use external frameworks (FastAPI, SQLAlchemy, etc.)
- ❌ Implement infrastructure concerns
- ❌ Handle HTTP requests or database operations
- ❌ Include UI or presentation logic

#### **Example Domain Entity**
```python
# domain/entities/user.py
@dataclass
class User:
    id: UUID
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    
    def can_perform_action(self, action: str) -> bool:
        """Business rule: Active users can perform actions"""
        return self.is_active
    
    def deactivate(self) -> None:
        """Business rule: Users can be deactivated"""
        self.is_active = False
```

#### **Example Domain Service**
```python
# domain/services/user_authorization_service.py
class UserAuthorizationService:
    def authorize_user(self, user: User, required_role: str) -> bool:
        """Business logic for user authorization"""
        if not user.is_active:
            return False
        
        # Complex authorization logic here
        return self._check_role_permissions(user, required_role)
```

### Application Layer Guidelines

#### **Purpose**
Orchestrates domain objects to implement use cases and application workflows.

#### **Do's**
- ✅ Implement use cases that fulfill user stories
- ✅ Create application services for complex workflows
- ✅ Define DTOs for data transfer between layers
- ✅ Document user stories and acceptance criteria
- ✅ Coordinate between domain objects

#### **Don'ts**
- ❌ Import from infrastructure layer
- ❌ Implement infrastructure concerns
- ❌ Handle HTTP requests directly
- ❌ Perform database operations directly
- ❌ Include presentation logic

#### **Example Use Case**
```python
# application/use_cases/create_user_use_case.py
class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    def execute(self, request: CreateUserDto) -> UserDto:
        # Validate business rules
        if self.user_repository.exists_by_email(request.email):
            raise UserAlreadyExistsError()
        
        # Create domain entity
        user = User(
            id=uuid4(),
            email=request.email,
            name=request.name,
            created_at=datetime.now()
        )
        
        # Save through repository
        saved_user = self.user_repository.save(user)
        
        # Return DTO
        return UserDto.from_entity(saved_user)
```

#### **Example User Story**
```yaml
# application/user_stories/create_user_story.yaml
title: "Create New User"
as_a: "System Administrator"
i_want: "to create new user accounts"
so_that: "new team members can access the system"

acceptance_criteria:
  - given: "I am an authenticated administrator"
    when: "I provide valid user details"
    then: "a new user account is created"
    and: "the user receives a welcome email"
```

### Infrastructure Layer Guidelines

#### **Purpose**
Provides external interfaces and implements technical concerns.

#### **Do's**
- ✅ Implement API endpoints for external access
- ✅ Create CLI commands for system administration
- ✅ Build GUI applications for user interaction
- ✅ Implement repository concrete classes
- ✅ Create adapters for external systems
- ✅ Handle HTTP, database, and other I/O operations

#### **Don'ts**
- ❌ Include business logic
- ❌ Implement domain rules
- ❌ Bypass the application layer
- ❌ Create direct dependencies between infrastructure components

#### **Example API Endpoint**
```python
# infrastructure/api/user_endpoints.py
@router.post("/users", response_model=UserDto)
async def create_user(
    request: CreateUserDto,
    use_case: CreateUserUseCase = Depends()
) -> UserDto:
    try:
        return use_case.execute(request)
    except UserAlreadyExistsError:
        raise HTTPException(status_code=409, detail="User already exists")
```

#### **Example CLI Command**
```python
# infrastructure/cli/user_commands.py
@click.command()
@click.option('--email', required=True)
@click.option('--name', required=True)
def create_user(email: str, name: str):
    """Create a new user"""
    use_case = CreateUserUseCase(get_user_repository())
    request = CreateUserDto(email=email, name=name)
    
    try:
        user = use_case.execute(request)
        click.echo(f"✅ User created: {user.email}")
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
```

---

## 🔗 Cross-Layer Communication

### Dependency Injection
Use dependency injection to provide implementations to higher layers:

```python
# In infrastructure layer
class UserController:
    def __init__(self, create_user_use_case: CreateUserUseCase):
        self.create_user_use_case = create_user_use_case

# In application layer
class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

# In domain layer
class UserRepository(ABC):
    @abstractmethod
    def save(self, user: User) -> User:
        pass
```

### Data Transfer Objects (DTOs)
Use DTOs to transfer data between layers:

```python
# application/dto/create_user_dto.py
@dataclass
class CreateUserDto:
    email: str
    name: str
    
    def to_entity(self) -> User:
        return User(
            id=uuid4(),
            email=self.email,
            name=self.name,
            created_at=datetime.now()
        )
```

---

## 🌐 Feature Development Workflow

### 1. Start with User Stories
```yaml
# application/user_stories/feature_story.yaml
title: "Feature Name"
as_a: "User Role"
i_want: "desired functionality"
so_that: "business value"

acceptance_criteria:
  - given: "precondition"
    when: "action"
    then: "expected result"
```

### 2. Design Domain Layer
- Create domain entities
- Define value objects
- Implement domain services
- Define repository interfaces

### 3. Implement Application Layer
- Create use cases based on user stories
- Define DTOs for data transfer
- Implement application services

### 4. Build Infrastructure Layer
- Create API endpoints
- Implement CLI commands
- Build GUI components
- Implement repository concrete classes

### 5. Write Tests
- Unit tests for domain logic
- Integration tests for use cases
- End-to-end tests for complete workflows

### 6. Document the Feature
- Update feature documentation
- Add API documentation
- Create user guides

---

## 🧪 Testing Guidelines

### Test Organization
```
tests/
├── unit/
│   ├── domain/          # Domain layer tests
│   ├── application/     # Application layer tests
│   └── infrastructure/  # Infrastructure layer tests
├── integration/         # Integration tests
├── acceptance/          # Acceptance tests
└── fixtures/           # Test fixtures
```

### Domain Testing
```python
# tests/unit/domain/test_user.py
def test_user_can_be_deactivated():
    user = User(id=uuid4(), email="test@example.com", name="Test")
    user.deactivate()
    assert user.is_active == False
```

### Application Testing
```python
# tests/unit/application/test_create_user_use_case.py
def test_create_user_with_valid_data():
    mock_repo = Mock(spec=UserRepository)
    use_case = CreateUserUseCase(mock_repo)
    
    request = CreateUserDto(email="test@example.com", name="Test")
    result = use_case.execute(request)
    
    assert result.email == "test@example.com"
    mock_repo.save.assert_called_once()
```

### Infrastructure Testing
```python
# tests/unit/infrastructure/test_user_endpoints.py
def test_create_user_endpoint():
    response = client.post("/users", json={
        "email": "test@example.com",
        "name": "Test User"
    })
    
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

---

## 📝 Code Quality Standards

### Naming Conventions
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Features**: `snake_case`

### Documentation
- All public classes and methods must have docstrings
- Use type hints for all function parameters and return values
- Include examples in docstrings when helpful
- Document business rules and domain logic

### Error Handling
- Use custom exceptions for domain errors
- Handle infrastructure errors gracefully
- Provide meaningful error messages
- Log errors appropriately

---

## 🔍 Validation and Enforcement

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: feature-architecture-validation
        name: Feature Architecture Validation
        entry: python scripts/feature_architecture_validator.py
        language: python
        pass_filenames: false
        always_run: true
```

### CI/CD Pipeline
The GitHub Actions workflow validates:
- Feature architecture compliance
- Domain boundary adherence
- Layer dependency rules
- Code quality standards
- Test coverage

### Manual Validation
```bash
# Validate feature architecture
python scripts/feature_architecture_validator.py

# Check domain boundaries
python scripts/updated_domain_boundary_validator.py

# Run tests
pytest src/packages_new/*/tests/
```

---

## 🚀 Best Practices

### 1. Feature Development
- Start with user stories and acceptance criteria
- Design domain layer first (entities, value objects, services)
- Implement application layer use cases
- Build infrastructure layer last
- Write tests at each layer

### 2. Code Organization
- Keep features small and focused
- Use shared components for common functionality
- Minimize cross-feature dependencies
- Follow single responsibility principle

### 3. Testing
- Write unit tests for domain logic
- Test use case orchestration
- Test infrastructure integrations
- Maintain high test coverage (>90%)

### 4. Documentation
- Document all public APIs
- Keep user stories up to date
- Maintain feature documentation
- Include examples and usage guides

### 5. Refactoring
- Refactor within feature boundaries
- Extract common logic to shared components
- Maintain backward compatibility
- Update tests and documentation

---

## 🛠️ Development Tools

### IDE Configuration
Configure your IDE for:
- Python path to include `src/packages_new`
- Auto-formatting with black
- Import sorting with isort
- Type checking with mypy

### Useful Commands
```bash
# Create new feature structure
mkdir -p src/packages_new/domain/package/feature/{domain/{entities,services,repositories,value_objects},application/{use_cases,user_stories,story_maps,services,dto},infrastructure/{api,cli,gui,adapters,repositories},docs,tests,scripts}

# Validate architecture
python scripts/feature_architecture_validator.py

# Run tests
pytest src/packages_new/domain/package/feature/tests/

# Generate documentation
python scripts/generate_feature_docs.py
```

---

## 📚 Additional Resources

- [Architectural Layers Standards](ARCHITECTURAL_LAYERS_STANDARDS.md)
- [Feature Identification Mapping](FEATURE_IDENTIFICATION_MAPPING.md)
- [Migration Plan](MIGRATION_PLAN.md)
- [Domain Boundary Rules](DOMAIN_BOUNDARY_RULES.md)

---

## 🤝 Contributing

When contributing to this codebase:

1. Follow the architecture guidelines
2. Write comprehensive tests
3. Document new features
4. Validate with provided tools
5. Submit pull requests with clear descriptions

Remember: The goal is to create maintainable, testable, and well-architected software that scales with the organization's needs.