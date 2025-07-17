# Architectural Layers Standards

## Domain → Package → Feature → Layer Architecture

This document defines the standards for organizing code within the three-layer architecture: **Domain**, **Application**, and **Infrastructure**.

---

## 🏗️ Architecture Overview

### **Directory Structure Template**
```
src/packages/
├── {domain}/                    # Business domain (e.g., ai, business, data)
│   ├── {package}/              # Domain package (e.g., machine_learning, analytics)
│   │   ├── {feature}/          # Business feature (e.g., model_lifecycle, user_management)
│   │   │   ├── domain/         # Core business logic
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
│   │   │   ├── infrastructure/ # External interfaces & adapters
│   │   │   │   ├── api/        # REST API endpoints
│   │   │   │   ├── cli/        # Command-line interfaces
│   │   │   │   ├── gui/        # Web UI applications
│   │   │   │   ├── adapters/   # External system adapters
│   │   │   │   └── repositories/ # Repository implementations
│   │   │   ├── docs/           # Feature documentation
│   │   │   ├── tests/          # Feature tests
│   │   │   └── scripts/        # Feature automation scripts
│   │   └── shared/             # Package-level shared components
│   └── docs/                   # Domain documentation
```

---

## 🎯 Domain Layer Standards

### **Purpose**
Contains pure business logic, domain entities, and business rules. This layer should have no dependencies on external frameworks or infrastructure.

### **Components**

#### **Entities** (`domain/entities/`)
- **Definition**: Core business objects with identity and lifecycle
- **Responsibilities**: Encapsulate business data and enforce business rules
- **Dependencies**: None (pure domain logic)
- **Example Structure**:
```python
# domain/entities/user.py
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    id: str
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    
    def deactivate(self) -> None:
        """Business rule: Users can be deactivated"""
        self.is_active = False
    
    def can_access_feature(self, feature: str) -> bool:
        """Business rule: Active users can access features"""
        return self.is_active
```

#### **Value Objects** (`domain/value_objects/`)
- **Definition**: Immutable objects that represent domain concepts without identity
- **Responsibilities**: Encapsulate domain concepts and validation rules
- **Dependencies**: None (pure domain logic)
- **Example Structure**:
```python
# domain/value_objects/email.py
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Email:
    value: str
    
    def __post_init__(self):
        if not self._is_valid_email(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
    
    def _is_valid_email(self, email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
```

#### **Domain Services** (`domain/services/`)
- **Definition**: Stateless services that implement domain logic spanning multiple entities
- **Responsibilities**: Complex business operations that don't belong to a single entity
- **Dependencies**: Only other domain components
- **Example Structure**:
```python
# domain/services/user_authorization_service.py
from typing import List
from ..entities.user import User
from ..entities.role import Role

class UserAuthorizationService:
    """Domain service for user authorization logic"""
    
    def can_user_perform_action(self, user: User, action: str, roles: List[Role]) -> bool:
        """Business rule: Authorization based on user roles"""
        if not user.is_active:
            return False
        
        user_roles = {role.name for role in roles if role.user_id == user.id}
        required_roles = self._get_required_roles_for_action(action)
        
        return bool(user_roles.intersection(required_roles))
    
    def _get_required_roles_for_action(self, action: str) -> set:
        """Define role requirements for actions"""
        action_roles = {
            'create_user': {'admin', 'user_manager'},
            'delete_user': {'admin'},
            'view_user': {'admin', 'user_manager', 'viewer'}
        }
        return action_roles.get(action, set())
```

#### **Repository Interfaces** (`domain/repositories/`)
- **Definition**: Abstract interfaces for data persistence
- **Responsibilities**: Define data access contracts without implementation
- **Dependencies**: Only domain entities and value objects
- **Example Structure**:
```python
# domain/repositories/user_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.user import User
from ..value_objects.email import Email

class UserRepository(ABC):
    """Domain repository interface for User entities"""
    
    @abstractmethod
    def save(self, user: User) -> User:
        """Save a user entity"""
        pass
    
    @abstractmethod
    def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID"""
        pass
    
    @abstractmethod
    def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email"""
        pass
    
    @abstractmethod
    def find_active_users(self) -> List[User]:
        """Find all active users"""
        pass
```

---

## 🔄 Application Layer Standards

### **Purpose**
Orchestrates domain objects to implement use cases and application workflows. Contains application-specific logic and coordinates between domain and infrastructure layers.

### **Components**

#### **Use Cases** (`application/use_cases/`)
- **Definition**: Specific application workflows that implement user goals
- **Responsibilities**: Orchestrate domain objects to achieve business outcomes
- **Dependencies**: Domain layer components, repository interfaces
- **Example Structure**:
```python
# application/use_cases/create_user_use_case.py
from typing import Optional
from ..dto.create_user_dto import CreateUserDto
from ..dto.user_dto import UserDto
from ...domain.entities.user import User
from ...domain.value_objects.email import Email
from ...domain.repositories.user_repository import UserRepository
from ...domain.services.user_authorization_service import UserAuthorizationService

class CreateUserUseCase:
    """Use case: Create a new user"""
    
    def __init__(self, 
                 user_repository: UserRepository,
                 authorization_service: UserAuthorizationService):
        self.user_repository = user_repository
        self.authorization_service = authorization_service
    
    def execute(self, request: CreateUserDto, requester: User) -> UserDto:
        """Execute the create user use case"""
        # Validate authorization
        if not self.authorization_service.can_user_perform_action(requester, 'create_user', []):
            raise UnauthorizedError("User not authorized to create users")
        
        # Validate email is unique
        email = Email(request.email)
        existing_user = self.user_repository.find_by_email(email)
        if existing_user:
            raise UserAlreadyExistsError(f"User with email {request.email} already exists")
        
        # Create new user
        user = User(
            id=self._generate_id(),
            email=request.email,
            name=request.name,
            created_at=datetime.now()
        )
        
        # Save user
        saved_user = self.user_repository.save(user)
        
        # Return DTO
        return UserDto.from_entity(saved_user)
    
    def _generate_id(self) -> str:
        """Generate unique user ID"""
        import uuid
        return str(uuid.uuid4())
```

#### **User Stories** (`application/user_stories/`)
- **Definition**: Structured user story definitions with acceptance criteria
- **Responsibilities**: Document user requirements and acceptance criteria
- **Dependencies**: None (documentation)
- **Example Structure**:
```yaml
# application/user_stories/create_user_story.yaml
title: "Create New User"
as_a: "System Administrator"
i_want: "to create new user accounts"
so_that: "new team members can access the system"

acceptance_criteria:
  - given: "I am an authenticated administrator"
    when: "I provide valid user details (name, email)"
    then: "a new user account is created"
    and: "the user receives an activation email"
  
  - given: "I am an authenticated administrator"
    when: "I provide an email that already exists"
    then: "I receive an error message"
    and: "no user account is created"
  
  - given: "I am not an administrator"
    when: "I try to create a user account"
    then: "I receive an authorization error"

technical_notes:
  - "User ID must be unique UUID"
  - "Email must be validated format"
  - "Password must meet security requirements"
  - "Audit log entry must be created"

related_use_cases:
  - "CreateUserUseCase"
  - "SendActivationEmailUseCase"
  - "CreateAuditLogUseCase"
```

#### **Story Maps** (`application/story_maps/`)
- **Definition**: Visual mapping of user journey and feature relationships
- **Responsibilities**: Show user workflow and feature dependencies
- **Dependencies**: User stories
- **Example Structure**:
```yaml
# application/story_maps/user_management_story_map.yaml
name: "User Management Story Map"
description: "Complete user lifecycle management workflow"

user_journey:
  - phase: "User Registration"
    activities:
      - "Submit registration form"
      - "Verify email address"
      - "Set initial password"
    user_stories:
      - "create_user_story"
      - "verify_email_story"
      - "set_password_story"
  
  - phase: "User Authentication"
    activities:
      - "Login to system"
      - "Manage session"
      - "Reset password"
    user_stories:
      - "login_user_story"
      - "manage_session_story"
      - "reset_password_story"
  
  - phase: "Profile Management"
    activities:
      - "Update profile information"
      - "Change password"
      - "Manage preferences"
    user_stories:
      - "update_profile_story"
      - "change_password_story"
      - "manage_preferences_story"

dependencies:
  - "User Registration" → "User Authentication"
  - "User Authentication" → "Profile Management"

metrics:
  - "User registration completion rate"
  - "Login success rate"
  - "Profile update frequency"
```

#### **Application Services** (`application/services/`)
- **Definition**: Coordinate multiple use cases and provide application-level operations
- **Responsibilities**: Handle cross-cutting concerns and orchestrate complex workflows
- **Dependencies**: Use cases, domain services, external service interfaces
- **Example Structure**:
```python
# application/services/user_management_service.py
from typing import List
from ..use_cases.create_user_use_case import CreateUserUseCase
from ..use_cases.activate_user_use_case import ActivateUserUseCase
from ..use_cases.deactivate_user_use_case import DeactivateUserUseCase
from ..dto.user_dto import UserDto
from ..dto.create_user_dto import CreateUserDto
from ...domain.entities.user import User

class UserManagementService:
    """Application service for user management operations"""
    
    def __init__(self,
                 create_user_use_case: CreateUserUseCase,
                 activate_user_use_case: ActivateUserUseCase,
                 deactivate_user_use_case: DeactivateUserUseCase):
        self.create_user_use_case = create_user_use_case
        self.activate_user_use_case = activate_user_use_case
        self.deactivate_user_use_case = deactivate_user_use_case
    
    def onboard_new_user(self, request: CreateUserDto, requester: User) -> UserDto:
        """Complete user onboarding workflow"""
        # Create user
        user = self.create_user_use_case.execute(request, requester)
        
        # Send welcome email (delegated to infrastructure)
        self._send_welcome_email(user)
        
        # Create audit log
        self._create_audit_log('user_created', user.id, requester.id)
        
        return user
    
    def _send_welcome_email(self, user: UserDto) -> None:
        """Send welcome email (implemented by infrastructure)"""
        # This would be injected as an interface
        pass
    
    def _create_audit_log(self, action: str, target_id: str, user_id: str) -> None:
        """Create audit log entry"""
        # This would be injected as an interface
        pass
```

#### **Data Transfer Objects** (`application/dto/`)
- **Definition**: Objects for transferring data between layers
- **Responsibilities**: Serialize/deserialize data and provide layer isolation
- **Dependencies**: Domain entities (for conversion)
- **Example Structure**:
```python
# application/dto/user_dto.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from ...domain.entities.user import User

@dataclass
class UserDto:
    id: str
    email: str
    name: str
    created_at: datetime
    is_active: bool
    
    @classmethod
    def from_entity(cls, user: User) -> 'UserDto':
        """Convert domain entity to DTO"""
        return cls(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
            is_active=user.is_active
        )
    
    def to_entity(self) -> User:
        """Convert DTO to domain entity"""
        return User(
            id=self.id,
            email=self.email,
            name=self.name,
            created_at=self.created_at,
            is_active=self.is_active
        )

@dataclass
class CreateUserDto:
    email: str
    name: str
    password: str
```

---

## 🔌 Infrastructure Layer Standards

### **Purpose**
Provides external interfaces and implements technical concerns. Contains adapters for external systems, user interfaces, and infrastructure implementations.

### **Components**

#### **API Endpoints** (`infrastructure/api/`)
- **Definition**: REST API endpoints for external system integration
- **Responsibilities**: Handle HTTP requests/responses and coordinate with application layer
- **Dependencies**: Application services, DTOs
- **Example Structure**:
```python
# infrastructure/api/user_endpoints.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from ..dto.user_dto import UserDto
from ..dto.create_user_dto import CreateUserDto
from ...application.services.user_management_service import UserManagementService
from ..auth.authentication import get_current_user

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserDto, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserDto,
    current_user: UserDto = Depends(get_current_user),
    user_service: UserManagementService = Depends()
) -> UserDto:
    """Create a new user"""
    try:
        user = user_service.onboard_new_user(request, current_user.to_entity())
        return user
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except UnauthorizedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )

@router.get("/{user_id}", response_model=UserDto)
async def get_user(
    user_id: str,
    current_user: UserDto = Depends(get_current_user),
    user_service: UserManagementService = Depends()
) -> UserDto:
    """Get user by ID"""
    # Implementation here
    pass

@router.get("/", response_model=List[UserDto])
async def list_users(
    current_user: UserDto = Depends(get_current_user),
    user_service: UserManagementService = Depends()
) -> List[UserDto]:
    """List all users"""
    # Implementation here
    pass
```

#### **CLI Commands** (`infrastructure/cli/`)
- **Definition**: Command-line interface for system administration
- **Responsibilities**: Provide CLI access to application functionality
- **Dependencies**: Application services, DTOs
- **Example Structure**:
```python
# infrastructure/cli/user_commands.py
import click
from typing import Optional
from ...application.services.user_management_service import UserManagementService
from ...application.dto.create_user_dto import CreateUserDto
from ..auth.cli_authentication import get_cli_user

@click.group()
def user():
    """User management commands"""
    pass

@user.command()
@click.option('--email', required=True, help='User email address')
@click.option('--name', required=True, help='User full name')
@click.option('--password', required=True, help='User password')
def create(email: str, name: str, password: str):
    """Create a new user"""
    try:
        current_user = get_cli_user()
        user_service = UserManagementService()  # Dependency injection
        
        request = CreateUserDto(
            email=email,
            name=name,
            password=password
        )
        
        user = user_service.onboard_new_user(request, current_user)
        
        click.echo(f"✅ User created successfully:")
        click.echo(f"   ID: {user.id}")
        click.echo(f"   Email: {user.email}")
        click.echo(f"   Name: {user.name}")
        
    except Exception as e:
        click.echo(f"❌ Error creating user: {str(e)}", err=True)

@user.command()
@click.argument('user_id')
def show(user_id: str):
    """Show user details"""
    # Implementation here
    pass

@user.command()
def list():
    """List all users"""
    # Implementation here
    pass
```

#### **GUI Applications** (`infrastructure/gui/`)
- **Definition**: Web-based user interfaces
- **Responsibilities**: Provide interactive user experience
- **Dependencies**: Application services, DTOs
- **Example Structure**:
```python
# infrastructure/gui/user_interface.py
from flask import Flask, render_template, request, jsonify, session
from flask_login import login_required, current_user
from ...application.services.user_management_service import UserManagementService
from ...application.dto.create_user_dto import CreateUserDto

app = Flask(__name__)

@app.route('/users')
@login_required
def users_page():
    """User management page"""
    return render_template('users/index.html')

@app.route('/users/create', methods=['GET', 'POST'])
@login_required
def create_user_page():
    """Create user page"""
    if request.method == 'GET':
        return render_template('users/create.html')
    
    elif request.method == 'POST':
        try:
            user_service = UserManagementService()  # Dependency injection
            
            request_data = CreateUserDto(
                email=request.form['email'],
                name=request.form['name'],
                password=request.form['password']
            )
            
            user = user_service.onboard_new_user(request_data, current_user)
            
            return jsonify({
                'success': True,
                'message': 'User created successfully',
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'name': user.name
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 400

@app.route('/users/<user_id>')
@login_required
def user_detail_page(user_id: str):
    """User detail page"""
    return render_template('users/detail.html', user_id=user_id)
```

#### **Repository Implementations** (`infrastructure/repositories/`)
- **Definition**: Concrete implementations of domain repository interfaces
- **Responsibilities**: Handle data persistence and retrieval
- **Dependencies**: Domain repository interfaces, external databases
- **Example Structure**:
```python
# infrastructure/repositories/sql_user_repository.py
from typing import List, Optional
from sqlalchemy.orm import Session
from ...domain.entities.user import User
from ...domain.value_objects.email import Email
from ...domain.repositories.user_repository import UserRepository
from .models.user_model import UserModel

class SqlUserRepository(UserRepository):
    """SQL implementation of UserRepository"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save(self, user: User) -> User:
        """Save user to database"""
        # Find existing or create new
        user_model = self.session.query(UserModel).filter_by(id=user.id).first()
        
        if user_model is None:
            user_model = UserModel()
        
        # Update model from entity
        user_model.id = user.id
        user_model.email = user.email
        user_model.name = user.name
        user_model.created_at = user.created_at
        user_model.is_active = user.is_active
        
        self.session.add(user_model)
        self.session.commit()
        
        return user
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        """Find user by ID"""
        user_model = self.session.query(UserModel).filter_by(id=user_id).first()
        
        if user_model is None:
            return None
        
        return User(
            id=user_model.id,
            email=user_model.email,
            name=user_model.name,
            created_at=user_model.created_at,
            is_active=user_model.is_active
        )
    
    def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email"""
        user_model = self.session.query(UserModel).filter_by(email=email.value).first()
        
        if user_model is None:
            return None
        
        return User(
            id=user_model.id,
            email=user_model.email,
            name=user_model.name,
            created_at=user_model.created_at,
            is_active=user_model.is_active
        )
    
    def find_active_users(self) -> List[User]:
        """Find all active users"""
        user_models = self.session.query(UserModel).filter_by(is_active=True).all()
        
        return [
            User(
                id=model.id,
                email=model.email,
                name=model.name,
                created_at=model.created_at,
                is_active=model.is_active
            )
            for model in user_models
        ]
```

#### **External Adapters** (`infrastructure/adapters/`)
- **Definition**: Adapters for external services and systems
- **Responsibilities**: Integrate with third-party services
- **Dependencies**: External service interfaces
- **Example Structure**:
```python
# infrastructure/adapters/email_adapter.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailAdapter(ABC):
    """Abstract email adapter"""
    
    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email"""
        pass

class SmtpEmailAdapter(EmailAdapter):
    """SMTP email adapter"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email via SMTP"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
```

---

## 🔗 Dependency Rules

### **Dependency Direction**
```
Infrastructure → Application → Domain
```

### **Allowed Dependencies**
- **Domain**: No external dependencies (pure business logic)
- **Application**: Can depend on Domain layer only
- **Infrastructure**: Can depend on Application and Domain layers

### **Forbidden Dependencies**
- **Domain** → Application (domain should be independent)
- **Domain** → Infrastructure (domain should be pure)
- **Application** → Infrastructure (application should be independent of technical details)

### **Dependency Injection**
- Use interfaces/abstractions for external dependencies
- Implement dependency injection at the infrastructure layer
- Use factory patterns for complex object creation

---

## 🧪 Testing Standards

### **Test Organization**
```
tests/
├── unit/
│   ├── domain/          # Domain layer unit tests
│   ├── application/     # Application layer unit tests
│   └── infrastructure/  # Infrastructure layer unit tests
├── integration/         # Integration tests
├── acceptance/          # Acceptance tests
└── fixtures/           # Test fixtures
```

### **Testing Principles**
- **Domain tests**: Pure unit tests with no external dependencies
- **Application tests**: Test use cases with mocked repositories
- **Infrastructure tests**: Test adapters and external integrations
- **Integration tests**: Test complete workflows across layers

---

## 📊 Quality Metrics

### **Code Quality**
- **Coupling**: Low coupling between layers
- **Cohesion**: High cohesion within features
- **Complexity**: Simple, readable code
- **Coverage**: >90% test coverage

### **Architecture Quality**
- **Dependency violations**: 0 violations of dependency rules
- **Circular dependencies**: 0 circular dependencies
- **Interface segregation**: Small, focused interfaces
- **Single responsibility**: Each class has single responsibility

These standards ensure consistent, maintainable, and testable code organization across all features and domains.