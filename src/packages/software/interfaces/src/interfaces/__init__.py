"""
Interfaces Package - User Interfaces and External APIs

This package provides comprehensive interface capabilities including:
- REST API endpoints
- Command-line interface (CLI)
- Web user interface
- Python SDK
- GraphQL API
- WebSocket communication
- Third-party integrations
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "support@pynomaly.com"

from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod

# API components with fallback implementations
class APIRouter:
    """API router for handling HTTP requests."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.routes: Dict[str, Any] = {}
        
    def add_route(self, path: str, handler: Any, methods: List[str] = None) -> None:
        """Add a route to the router."""
        self.routes[path] = {'handler': handler, 'methods': methods or ['GET']}
        
    def get_routes(self) -> Dict[str, Any]:
        """Get all routes."""
        return self.routes

class SecurityManager:
    """Security manager for authentication and authorization."""
    
    def __init__(self):
        self.users: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}
        
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token."""
        user = self.users.get(username)
        if user and user.get('password') == password:
            import uuid
            token = str(uuid.uuid4())
            self.sessions[token] = {'user': username}
            return token
        return None
        
    def authorize(self, token: str, resource: str) -> bool:
        """Check if user has access to resource."""
        session = self.sessions.get(token)
        return session is not None

class AuthenticationService:
    """Authentication service."""
    
    def __init__(self):
        self.tokens: Dict[str, str] = {}
        
    def login(self, username: str, password: str) -> Optional[str]:
        """Login user."""
        if username and password:  # Basic validation
            import uuid
            token = str(uuid.uuid4())
            self.tokens[token] = username
            return token
        return None
        
    def logout(self, token: str) -> bool:
        """Logout user."""
        return self.tokens.pop(token, None) is not None

class AuthorizationService:
    """Authorization service."""
    
    def __init__(self):
        self.permissions: Dict[str, List[str]] = {}
        
    def has_permission(self, user: str, permission: str) -> bool:
        """Check if user has permission."""
        user_permissions = self.permissions.get(user, [])
        return permission in user_permissions

# CLI components
class CommandRegistry:
    """Command registry for CLI commands."""
    
    def __init__(self):
        self.commands: Dict[str, Any] = {}
        
    def register(self, name: str, handler: Any) -> None:
        """Register a command."""
        self.commands[name] = handler
        
    def get_command(self, name: str) -> Optional[Any]:
        """Get a command by name."""
        return self.commands.get(name)

class CliContainer:
    """CLI dependency injection container."""
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        
    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self.services[name] = service
        
    def get_service(self, name: str) -> Any:
        """Get a service."""
        return self.services.get(name)

class CliConfiguration:
    """CLI configuration management."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value

# Web components
class WebApplication:
    """Web application framework."""
    
    def __init__(self):
        self.routes: Dict[str, Any] = {}
        self.middleware: List[Any] = []
        
    def add_route(self, path: str, handler: Any) -> None:
        """Add a route."""
        self.routes[path] = handler
        
    def add_middleware(self, middleware: Any) -> None:
        """Add middleware."""
        self.middleware.append(middleware)

class WebRouter:
    """Web router for handling HTTP requests."""
    
    def __init__(self):
        self.routes: Dict[str, Any] = {}
        
    def route(self, path: str, methods: List[str] = None):
        """Decorator for adding routes."""
        def decorator(handler):
            self.routes[path] = {'handler': handler, 'methods': methods or ['GET']}
            return handler
        return decorator

class WebSecurityManager:
    """Web security manager."""
    
    def __init__(self):
        self.csrf_tokens: Dict[str, str] = {}
        
    def generate_csrf_token(self, user_id: str) -> str:
        """Generate CSRF token."""
        import uuid
        token = str(uuid.uuid4())
        self.csrf_tokens[user_id] = token
        return token
        
    def validate_csrf_token(self, user_id: str, token: str) -> bool:
        """Validate CSRF token."""
        return self.csrf_tokens.get(user_id) == token

class WebSocketManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        
    def connect(self, client_id: str, websocket: Any) -> None:
        """Connect a client."""
        self.connections[client_id] = websocket
        
    def disconnect(self, client_id: str) -> None:
        """Disconnect a client."""
        self.connections.pop(client_id, None)

# SDK components
class PynomaliClient:
    """Main SDK client."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with the API."""
        # Mock authentication
        return username and password
        
    def detect_anomalies(self, data: Any) -> Dict[str, Any]:
        """Detect anomalies in data."""
        # Mock detection
        return {'anomalies': [], 'status': 'success'}

class DetectionService:
    """Detection service for the SDK."""
    
    def __init__(self, client: PynomaliClient):
        self.client = client
        
    def run_detection(self, data: Any, config: Dict = None) -> Dict[str, Any]:
        """Run anomaly detection."""
        return self.client.detect_anomalies(data)

class ModelService:
    """Model service for the SDK."""
    
    def __init__(self, client: PynomaliClient):
        self.client = client
        self.models: Dict[str, Any] = {}
        
    def create_model(self, name: str, algorithm: str) -> str:
        """Create a new model."""
        import uuid
        model_id = str(uuid.uuid4())
        self.models[model_id] = {'name': name, 'algorithm': algorithm}
        return model_id
        
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get a model."""
        return self.models.get(model_id)

class ExperimentService:
    """Experiment service for the SDK."""
    
    def __init__(self, client: PynomaliClient):
        self.client = client
        self.experiments: Dict[str, Any] = {}
        
    def create_experiment(self, name: str, config: Dict) -> str:
        """Create a new experiment."""
        import uuid
        experiment_id = str(uuid.uuid4())
        self.experiments[experiment_id] = {'name': name, 'config': config}
        return experiment_id

# Shared components
class BaseEntity:
    """Base entity class."""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__

class ErrorHandler:
    """Error handling utilities."""
    
    @staticmethod
    def handle_error(error: Exception) -> Dict[str, Any]:
        """Handle an error."""
        return {
            'error': str(error),
            'type': type(error).__name__,
            'status': 'error'
        }

class ResponseFormatter:
    """Response formatting utilities."""
    
    @staticmethod
    def success(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Format success response."""
        return {
            'status': 'success',
            'message': message,
            'data': data
        }
        
    @staticmethod
    def error(message: str, code: int = 400) -> Dict[str, Any]:
        """Format error response."""
        return {
            'status': 'error',
            'message': message,
            'code': code
        }

class ValidationService:
    """Validation service."""
    
    def __init__(self):
        self.validators: Dict[str, Any] = {}
        
    def validate(self, data: Any, schema: str) -> bool:
        """Validate data against schema."""
        # Mock validation
        return True
        
    def add_validator(self, name: str, validator: Any) -> None:
        """Add a validator."""
        self.validators[name] = validator

# Factory functions
def create_app(config: Dict = None) -> APIRouter:
    """Create API application."""
    return APIRouter()

def create_cli_app(config: Dict = None) -> CommandRegistry:
    """Create CLI application."""
    return CommandRegistry()

__all__ = [
    # API
    "create_app",
    "APIRouter",
    "SecurityManager",
    "AuthenticationService",
    "AuthorizationService",
    
    # CLI
    "create_cli_app",
    "CommandRegistry",
    "CliContainer",
    "CliConfiguration",
    
    # Web
    "WebApplication",
    "WebRouter",
    "WebSecurityManager",
    "WebSocketManager",
    
    # SDK
    "PynomaliClient",
    "DetectionService",
    "ModelService",
    "ExperimentService",
    
    # Shared
    "BaseEntity",
    "ErrorHandler",
    "ResponseFormatter",
    "ValidationService",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]