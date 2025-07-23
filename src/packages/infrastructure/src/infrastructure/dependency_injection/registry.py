"""Service registry for dependency injection."""

from __future__ import annotations

from typing import Any, Dict, Type, TypeVar, Optional, Callable
from abc import ABC

T = TypeVar('T')


class ServiceRegistry:
    """Registry for managing service instances and their interfaces."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a service instance."""
        self._services[interface] = instance
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for creating service instances."""
        self._factories[interface] = factory
    
    def register_singleton(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a singleton factory - instance created once and reused."""
        self._factories[interface] = factory
        # Mark as singleton by also storing in singletons dict
        if interface not in self._singletons:
            self._singletons[interface] = None
    
    def get(self, interface: Type[T]) -> T:
        """Get service instance for interface."""
        # Check for direct instance registration
        if interface in self._services:
            return self._services[interface]
        
        # Check for singleton
        if interface in self._singletons:
            if self._singletons[interface] is None:
                # Create singleton instance
                factory = self._factories[interface]
                instance = factory()
                self._singletons[interface] = instance
            return self._singletons[interface]
        
        # Check for factory
        if interface in self._factories:
            factory = self._factories[interface]
            return factory()
        
        raise ValueError(f"No service registered for interface: {interface}")
    
    def is_registered(self, interface: Type[T]) -> bool:
        """Check if interface is registered."""
        return (
            interface in self._services or 
            interface in self._factories or 
            interface in self._singletons
        )
    
    def unregister(self, interface: Type[T]) -> None:
        """Unregister service for interface."""
        self._services.pop(interface, None)
        self._factories.pop(interface, None)
        self._singletons.pop(interface, None)
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
    
    def list_interfaces(self) -> list[Type]:
        """List all registered interfaces."""
        interfaces = set()
        interfaces.update(self._services.keys())
        interfaces.update(self._factories.keys())
        interfaces.update(self._singletons.keys())
        return list(interfaces)
    
    def get_registration_info(self, interface: Type[T]) -> Dict[str, Any]:
        """Get information about how an interface is registered."""
        info = {
            "interface": interface.__name__,
            "registered": self.is_registered(interface),
            "has_instance": interface in self._services,
            "has_factory": interface in self._factories,
            "is_singleton": interface in self._singletons,
        }
        
        if interface in self._singletons:
            info["singleton_created"] = self._singletons[interface] is not None
        
        return info


# Global registry instance
_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _registry
    _registry = ServiceRegistry()


# Convenience functions
def register(interface: Type[T], instance_or_factory: Any) -> None:
    """Register service in global registry."""
    registry = get_registry()
    
    if callable(instance_or_factory) and not hasattr(instance_or_factory, '__call__'):
        # It's a class or factory function
        registry.register_factory(interface, instance_or_factory)
    else:
        # It's an instance
        registry.register_instance(interface, instance_or_factory)


def register_singleton(interface: Type[T], factory: Callable[[], T]) -> None:
    """Register singleton in global registry."""
    registry = get_registry()
    registry.register_singleton(interface, factory)


def get_service(interface: Type[T]) -> T:
    """Get service from global registry."""
    registry = get_registry()
    return registry.get(interface)


def is_registered(interface: Type[T]) -> bool:
    """Check if interface is registered in global registry."""
    registry = get_registry()
    return registry.is_registered(interface)