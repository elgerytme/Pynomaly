"""
Dependency Injection Container for Pynomaly Detection
====================================================

Comprehensive IoC container providing:
- Service registration and resolution
- Singleton and transient lifecycles
- Interface-based dependency injection
- Decorator-based service registration
- Circular dependency detection
- Configuration-driven service setup
"""

import logging
import inspect
from typing import Dict, Any, Type, TypeVar, Callable, Optional, Union, get_type_hints
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceLifetime(Enum):
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

@dataclass
class ServiceDescriptor:
    """Service descriptor for registration."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: Optional[Dict[str, Type]] = None

class ServiceContainer:
    """Dependency injection container."""
    
    def __init__(self):
        """Initialize service container."""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._building: set = set()
        self._lock = threading.RLock()
        
        # Register self
        self.register_instance(ServiceContainer, self)
        
        logger.info("Service container initialized")
    
    def register_transient(self, service_type: Type[T], 
                          implementation_type: Type[T] = None) -> 'ServiceContainer':
        """Register transient service.
        
        Args:
            service_type: Service interface type
            implementation_type: Implementation type
            
        Returns:
            Service container for chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                lifetime=ServiceLifetime.TRANSIENT
            )
            self._services[service_type] = descriptor
            
            logger.debug(f"Registered transient service: {service_type.__name__}")
            return self
    
    def register_singleton(self, service_type: Type[T], 
                          implementation_type: Type[T] = None) -> 'ServiceContainer':
        """Register singleton service.
        
        Args:
            service_type: Service interface type
            implementation_type: Implementation type
            
        Returns:
            Service container for chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                lifetime=ServiceLifetime.SINGLETON
            )
            self._services[service_type] = descriptor
            
            logger.debug(f"Registered singleton service: {service_type.__name__}")
            return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainer':
        """Register service instance.
        
        Args:
            service_type: Service type
            instance: Service instance
            
        Returns:
            Service container for chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                instance=instance,
                lifetime=ServiceLifetime.SINGLETON
            )
            self._services[service_type] = descriptor
            self._instances[service_type] = instance
            
            logger.debug(f"Registered instance: {service_type.__name__}")
            return self
    
    def register_factory(self, service_type: Type[T], 
                        factory: Callable[[], T],
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'ServiceContainer':
        """Register service factory.
        
        Args:
            service_type: Service type
            factory: Factory function
            lifetime: Service lifetime
            
        Returns:
            Service container for chaining
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime
            )
            self._services[service_type] = descriptor
            
            logger.debug(f"Registered factory for: {service_type.__name__}")
            return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service instance.
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered or circular dependency detected
        """
        with self._lock:
            # Check for circular dependency
            if service_type in self._building:
                raise ValueError(f"Circular dependency detected: {service_type.__name__}")
            
            # Check if service is registered
            if service_type not in self._services:
                raise ValueError(f"Service not registered: {service_type.__name__}")
            
            descriptor = self._services[service_type]
            
            # Return existing instance for singletons
            if (descriptor.lifetime == ServiceLifetime.SINGLETON and 
                service_type in self._instances):
                return self._instances[service_type]
            
            # Build service instance
            self._building.add(service_type)
            
            try:
                instance = self._build_instance(descriptor)
                
                # Store singleton instances
                if descriptor.lifetime == ServiceLifetime.SINGLETON:
                    self._instances[service_type] = instance
                
                return instance
                
            finally:
                self._building.discard(service_type)
    
    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve service instance.
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance or None if not registered
        """
        try:
            return self.resolve(service_type)
        except ValueError:
            return None
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service type is registered.
        
        Args:
            service_type: Service type to check
            
        Returns:
            True if registered
        """
        return service_type in self._services
    
    def remove_service(self, service_type: Type) -> bool:
        """Remove service registration.
        
        Args:
            service_type: Service type to remove
            
        Returns:
            True if removed
        """
        with self._lock:
            if service_type in self._services:
                del self._services[service_type]
                
                if service_type in self._instances:
                    del self._instances[service_type]
                
                logger.debug(f"Removed service: {service_type.__name__}")
                return True
            
            return False
    
    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services.
        
        Returns:
            Dictionary of registered services
        """
        return self._services.copy()
    
    def clear(self):
        """Clear all registered services."""
        with self._lock:
            self._services.clear()
            self._instances.clear()
            self._building.clear()
            
            # Re-register self
            self.register_instance(ServiceContainer, self)
            
            logger.info("Service container cleared")
    
    def _build_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Build service instance from descriptor.
        
        Args:
            descriptor: Service descriptor
            
        Returns:
            Service instance
        """
        # Use existing instance
        if descriptor.instance is not None:
            return descriptor.instance
        
        # Use factory
        if descriptor.factory is not None:
            return descriptor.factory()
        
        # Build from implementation type
        if descriptor.implementation_type:
            return self._create_instance(descriptor.implementation_type)
        
        raise ValueError(f"Cannot build service: {descriptor.service_type.__name__}")
    
    def _create_instance(self, impl_type: Type) -> Any:
        """Create instance with dependency injection.
        
        Args:
            impl_type: Implementation type
            
        Returns:
            Instance with injected dependencies
        """
        try:
            # Get constructor signature
            signature = inspect.signature(impl_type.__init__)
            parameters = signature.parameters
            
            # Skip 'self' parameter
            param_names = list(parameters.keys())[1:]
            
            if not param_names:
                # No dependencies
                return impl_type()
            
            # Resolve dependencies
            kwargs = {}
            type_hints = get_type_hints(impl_type.__init__)
            
            for param_name in param_names:
                param = parameters[param_name]
                
                # Get parameter type
                param_type = type_hints.get(param_name)
                if param_type is None:
                    param_type = param.annotation
                
                if param_type and param_type != inspect.Parameter.empty:
                    # Try to resolve dependency
                    dependency = self.try_resolve(param_type)
                    
                    if dependency is not None:
                        kwargs[param_name] = dependency
                    elif param.default == inspect.Parameter.empty:
                        # Required parameter without default value
                        raise ValueError(f"Cannot resolve required dependency: {param_name} ({param_type})")
            
            return impl_type(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create instance of {impl_type.__name__}: {e}")
            raise

# Global container instance
_container: Optional[ServiceContainer] = None
_container_lock = threading.Lock()

def get_container() -> ServiceContainer:
    """Get global service container.
    
    Returns:
        Global service container instance
    """
    global _container
    
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = ServiceContainer()
    
    return _container

def set_container(container: ServiceContainer):
    """Set global service container.
    
    Args:
        container: Service container to set as global
    """
    global _container
    
    with _container_lock:
        _container = container

# Decorator for service registration
def service(service_type: Type = None, 
           lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
    """Decorator for automatic service registration.
    
    Args:
        service_type: Service type (defaults to decorated class)
        lifetime: Service lifetime
        
    Returns:
        Decorated class
    """
    def decorator(cls):
        container = get_container()
        
        if service_type:
            if lifetime == ServiceLifetime.SINGLETON:
                container.register_singleton(service_type, cls)
            else:
                container.register_transient(service_type, cls)
        else:
            if lifetime == ServiceLifetime.SINGLETON:
                container.register_singleton(cls)
            else:
                container.register_transient(cls)
        
        return cls
    
    return decorator

def singleton(service_type: Type = None):
    """Decorator for singleton service registration.
    
    Args:
        service_type: Service type (defaults to decorated class)
        
    Returns:
        Decorated class
    """
    return service(service_type, ServiceLifetime.SINGLETON)

def transient(service_type: Type = None):
    """Decorator for transient service registration.
    
    Args:
        service_type: Service type (defaults to decorated class)
        
    Returns:
        Decorated class
    """
    return service(service_type, ServiceLifetime.TRANSIENT)

# Dependency injection helper
def inject(service_type: Type[T]) -> T:
    """Inject service dependency.
    
    Args:
        service_type: Service type to inject
        
    Returns:
        Service instance
    """
    return get_container().resolve(service_type)

class ConfigurationModule:
    """Configuration module for service registration."""
    
    @abstractmethod
    def configure_services(self, container: ServiceContainer) -> ServiceContainer:
        """Configure services in container.
        
        Args:
            container: Service container
            
        Returns:
            Configured container
        """
        pass

def configure_from_module(module: ConfigurationModule) -> ServiceContainer:
    """Configure services from module.
    
    Args:
        module: Configuration module
        
    Returns:
        Configured container
    """
    container = get_container()
    return module.configure_services(container)