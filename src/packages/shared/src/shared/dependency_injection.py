"""
Dependency Injection framework for cross-package communication.

This module provides a dependency injection container that enables loose
coupling between packages while maintaining proper architectural boundaries.
"""

import inspect
import logging
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from functools import wraps

from interfaces.patterns import Repository, Service, Cache, HealthCheck


logger = logging.getLogger(__name__)
T = TypeVar('T')


class LifecycleScope(Enum):
    """Dependency lifecycle scopes."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class InjectionError(Exception):
    """Exception raised during dependency injection operations."""
    pass


@dataclass
class DependencyRegistration:
    """Registration information for a dependency."""
    interface: Type
    implementation: Type
    lifecycle: LifecycleScope
    factory: Optional[Callable[..., Any]] = None
    instance: Optional[Any] = None
    dependencies: Optional[List[str]] = None


class DIContainer:
    """
    Dependency injection container with support for multiple lifecycle scopes
    and automatic dependency resolution.
    """
    
    def __init__(self):
        self._registrations: Dict[Type, DependencyRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._building_stack: List[Type] = []  # Circular dependency detection
    
    def register_singleton(self, 
                          interface: Type[T], 
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable[..., T]] = None,
                          instance: Optional[T] = None) -> 'DIContainer':
        """Register a singleton dependency."""
        return self._register(interface, implementation, LifecycleScope.SINGLETON, factory, instance)
    
    def register_transient(self, 
                          interface: Type[T], 
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable[..., T]] = None) -> 'DIContainer':
        """Register a transient dependency."""
        return self._register(interface, implementation, LifecycleScope.TRANSIENT, factory)
    
    def register_scoped(self, 
                       interface: Type[T], 
                       implementation: Optional[Type[T]] = None,
                       factory: Optional[Callable[..., T]] = None) -> 'DIContainer':
        """Register a scoped dependency."""
        return self._register(interface, implementation, LifecycleScope.SCOPED, factory)
    
    def _register(self, 
                  interface: Type[T], 
                  implementation: Optional[Type[T]], 
                  lifecycle: LifecycleScope,
                  factory: Optional[Callable[..., T]] = None,
                  instance: Optional[T] = None) -> 'DIContainer':
        """Internal registration method."""
        if implementation is None and factory is None and instance is None:
            implementation = interface
        
        # Extract dependencies from constructor
        dependencies = None
        if implementation:
            try:
                sig = inspect.signature(implementation.__init__)
                dependencies = [
                    param.name for param in sig.parameters.values()
                    if param.name != 'self' and param.annotation != inspect.Parameter.empty
                ]
            except (AttributeError, ValueError):
                dependencies = []
        
        registration = DependencyRegistration(
            interface=interface,
            implementation=implementation,
            lifecycle=lifecycle,
            factory=factory,
            instance=instance,
            dependencies=dependencies
        )
        
        self._registrations[interface] = registration
        logger.debug(f"Registered {interface.__name__} as {lifecycle.value}")
        return self
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a dependency by its interface."""
        if interface in self._building_stack:
            cycle = " -> ".join(cls.__name__ for cls in self._building_stack)
            raise InjectionError(f"Circular dependency detected: {cycle} -> {interface.__name__}")
        
        registration = self._registrations.get(interface)
        if not registration:
            raise InjectionError(f"No registration found for {interface.__name__}")
        
        self._building_stack.append(interface)
        try:
            return self._create_instance(registration)
        finally:
            self._building_stack.pop()
    
    def _create_instance(self, registration: DependencyRegistration) -> Any:
        """Create an instance based on registration."""
        # Check for existing instance based on lifecycle
        if registration.lifecycle == LifecycleScope.SINGLETON:
            if registration.interface in self._singletons:
                return self._singletons[registration.interface]
            if registration.instance is not None:
                self._singletons[registration.interface] = registration.instance
                return registration.instance
        
        elif registration.lifecycle == LifecycleScope.SCOPED:
            if self._current_scope and self._current_scope in self._scoped_instances:
                scoped_cache = self._scoped_instances[self._current_scope]
                if registration.interface in scoped_cache:
                    return scoped_cache[registration.interface]
        
        # Create new instance
        if registration.factory:
            instance = self._invoke_factory(registration.factory)
        else:
            instance = self._create_from_constructor(registration)
        
        # Cache based on lifecycle
        if registration.lifecycle == LifecycleScope.SINGLETON:
            self._singletons[registration.interface] = instance
        elif registration.lifecycle == LifecycleScope.SCOPED and self._current_scope:
            if self._current_scope not in self._scoped_instances:
                self._scoped_instances[self._current_scope] = {}
            self._scoped_instances[self._current_scope][registration.interface] = instance
        
        return instance
    
    def _create_from_constructor(self, registration: DependencyRegistration) -> Any:
        """Create instance using constructor injection."""
        if not registration.implementation:
            raise InjectionError(f"No implementation provided for {registration.interface.__name__}")
        
        constructor = registration.implementation.__init__
        sig = inspect.signature(constructor)
        
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                if param.default == inspect.Parameter.empty:
                    raise InjectionError(
                        f"Parameter '{param_name}' in {registration.implementation.__name__} "
                        f"has no type annotation and no default value"
                    )
                continue
            
            try:
                kwargs[param_name] = self.resolve(param_type)
            except InjectionError:
                if param.default != inspect.Parameter.empty:
                    continue  # Use default value
                raise
        
        return registration.implementation(**kwargs)
    
    def _invoke_factory(self, factory: Callable) -> Any:
        """Invoke a factory function with dependency injection."""
        sig = inspect.signature(factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                continue
            
            try:
                kwargs[param_name] = self.resolve(param_type)
            except InjectionError:
                if param.default != inspect.Parameter.empty:
                    continue
                raise
        
        return factory(**kwargs)
    
    def create_scope(self, scope_name: str) -> 'ScopeContext':
        """Create a new dependency scope."""
        return ScopeContext(self, scope_name)
    
    def _enter_scope(self, scope_name: str) -> None:
        """Enter a dependency scope."""
        self._current_scope = scope_name
        if scope_name not in self._scoped_instances:
            self._scoped_instances[scope_name] = {}
    
    def _exit_scope(self, scope_name: str) -> None:
        """Exit a dependency scope."""
        if scope_name in self._scoped_instances:
            # Cleanup scoped instances
            scoped_cache = self._scoped_instances[scope_name]
            for instance in scoped_cache.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing instance: {e}")
            
            del self._scoped_instances[scope_name]
        
        self._current_scope = None
    
    def is_registered(self, interface: Type) -> bool:
        """Check if an interface is registered."""
        return interface in self._registrations
    
    def get_registrations(self) -> Dict[Type, DependencyRegistration]:
        """Get all registrations."""
        return self._registrations.copy()


class ScopeContext:
    """Context manager for dependency scopes."""
    
    def __init__(self, container: DIContainer, scope_name: str):
        self.container = container
        self.scope_name = scope_name
    
    def __enter__(self):
        self.container._enter_scope(self.scope_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._exit_scope(self.scope_name)


# Decorators for dependency injection
def inject(container: DIContainer):
    """Decorator to enable dependency injection for a function or method."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject dependencies based on type annotations
            sig = inspect.signature(func)
            injected_kwargs = {}
            
            for param_name, param in sig.parameters.items():
                if param_name in kwargs:
                    continue  # Already provided
                
                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    continue
                
                if container.is_registered(param_type):
                    injected_kwargs[param_name] = container.resolve(param_type)
                elif param.default == inspect.Parameter.empty:
                    raise InjectionError(f"Cannot inject parameter '{param_name}' of type {param_type}")
            
            return func(*args, **kwargs, **injected_kwargs)
        return wrapper
    return decorator


def injectable(interface: Optional[Type] = None, lifecycle: LifecycleScope = LifecycleScope.TRANSIENT):
    """Class decorator to mark a class as injectable."""
    def decorator(cls):
        cls._injectable_interface = interface or cls
        cls._injectable_lifecycle = lifecycle
        return cls
    return decorator


# Global container instance
_global_container: DIContainer = None


def get_container() -> DIContainer:
    """Get the global DI container."""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


def configure_container(configurator: Callable[[DIContainer], None]) -> None:
    """Configure the global container."""
    container = get_container()
    configurator(container)


# Common service registration helpers
def register_repository(container: DIContainer, interface: Type[Repository], implementation: Type[Repository]) -> None:
    """Register a repository with standard configuration."""
    container.register_scoped(interface, implementation)


def register_service(container: DIContainer, interface: Type[Service], implementation: Type[Service]) -> None:
    """Register a service with standard configuration."""
    container.register_singleton(interface, implementation)


def register_cache(container: DIContainer, interface: Type[Cache], implementation: Type[Cache]) -> None:
    """Register a cache with standard configuration."""
    container.register_singleton(interface, implementation)


def register_health_check(container: DIContainer, implementation: Type[HealthCheck]) -> None:
    """Register a health check service."""
    container.register_singleton(HealthCheck, implementation)