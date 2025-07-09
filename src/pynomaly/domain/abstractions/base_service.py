"""Base service abstraction for domain services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from .base_entity import BaseEntity

T = TypeVar("T", bound=BaseEntity)


class BaseService(ABC):
    """Base class for domain services.
    
    Domain services contain domain logic that doesn't naturally fit
    within an entity or value object. They are stateless and focused
    on a single responsibility.
    """

    def __init__(self) -> None:
        """Initialize the service."""
        self._dependencies: dict[str, Any] = {}

    def add_dependency(self, name: str, dependency: Any) -> None:
        """Add a dependency to the service.
        
        Args:
            name: Name of the dependency
            dependency: The dependency object
        """
        self._dependencies[name] = dependency

    def get_dependency(self, name: str) -> Any:
        """Get a dependency by name.
        
        Args:
            name: Name of the dependency
            
        Returns:
            The dependency object
            
        Raises:
            KeyError: If dependency is not found
        """
        return self._dependencies[name]

    def has_dependency(self, name: str) -> bool:
        """Check if a dependency exists.
        
        Args:
            name: Name of the dependency
            
        Returns:
            True if dependency exists
        """
        return name in self._dependencies

    def remove_dependency(self, name: str) -> None:
        """Remove a dependency.
        
        Args:
            name: Name of the dependency to remove
        """
        self._dependencies.pop(name, None)

    def clear_dependencies(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()

    def get_service_name(self) -> str:
        """Get the name of the service.
        
        Returns:
            Service name
        """
        return self.__class__.__name__

    def validate_dependencies(self) -> None:
        """Validate that all required dependencies are present.
        
        Override in subclasses to add dependency validation.
        
        Raises:
            ValueError: If required dependencies are missing
        """
        pass

    def initialize(self) -> None:
        """Initialize the service.
        
        Called after all dependencies are set up.
        Override in subclasses for initialization logic.
        """
        self.validate_dependencies()

    def cleanup(self) -> None:
        """Clean up resources.
        
        Override in subclasses for cleanup logic.
        """
        pass

    def __enter__(self) -> BaseService:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()


class EntityService(BaseService, Generic[T]):
    """Base class for services that operate on specific entity types."""

    def __init__(self, entity_type: type[T]) -> None:
        """Initialize the entity service.
        
        Args:
            entity_type: The type of entity this service operates on
        """
        super().__init__()
        self.entity_type = entity_type

    def get_entity_type(self) -> type[T]:
        """Get the entity type this service operates on.
        
        Returns:
            Entity type
        """
        return self.entity_type

    def is_valid_entity(self, entity: Any) -> bool:
        """Check if an entity is valid for this service.
        
        Args:
            entity: The entity to check
            
        Returns:
            True if entity is valid for this service
        """
        return isinstance(entity, self.entity_type)

    def validate_entity(self, entity: Any) -> None:
        """Validate that an entity is valid for this service.
        
        Args:
            entity: The entity to validate
            
        Raises:
            TypeError: If entity is not of the correct type
        """
        if not self.is_valid_entity(entity):
            raise TypeError(
                f"Entity must be of type {self.entity_type.__name__}, "
                f"got {type(entity).__name__}"
            )


class ValidationService(BaseService):
    """Base class for validation services."""

    @abstractmethod
    def validate(self, entity: Any) -> list[str]:
        """Validate an entity.
        
        Args:
            entity: The entity to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        ...

    def is_valid(self, entity: Any) -> bool:
        """Check if an entity is valid.
        
        Args:
            entity: The entity to check
            
        Returns:
            True if entity is valid
        """
        return len(self.validate(entity)) == 0

    def validate_and_raise(self, entity: Any) -> None:
        """Validate an entity and raise an exception if invalid.
        
        Args:
            entity: The entity to validate
            
        Raises:
            ValueError: If entity is invalid
        """
        errors = self.validate(entity)
        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")


class TransformationService(BaseService, Generic[T]):
    """Base class for transformation services."""

    @abstractmethod
    def transform(self, input_entity: T) -> T:
        """Transform an entity.
        
        Args:
            input_entity: The entity to transform
            
        Returns:
            Transformed entity
        """
        ...

    def can_transform(self, entity: Any) -> bool:
        """Check if an entity can be transformed.
        
        Args:
            entity: The entity to check
            
        Returns:
            True if entity can be transformed
        """
        return True

    def transform_batch(self, entities: list[T]) -> list[T]:
        """Transform multiple entities.
        
        Args:
            entities: List of entities to transform
            
        Returns:
            List of transformed entities
        """
        return [self.transform(entity) for entity in entities]


class CalculationService(BaseService):
    """Base class for calculation services."""

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Any:
        """Perform a calculation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Calculation result
        """
        ...

    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate calculation inputs.
        
        Override in subclasses to add input validation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Raises:
            ValueError: If inputs are invalid
        """
        pass

    def calculate_with_validation(self, *args, **kwargs) -> Any:
        """Perform a calculation with input validation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Calculation result
        """
        self.validate_inputs(*args, **kwargs)
        return self.calculate(*args, **kwargs)


class ComparisonService(BaseService, Generic[T]):
    """Base class for comparison services."""

    @abstractmethod
    def compare(self, entity1: T, entity2: T) -> int:
        """Compare two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            -1 if entity1 < entity2, 0 if equal, 1 if entity1 > entity2
        """
        ...

    def are_equal(self, entity1: T, entity2: T) -> bool:
        """Check if two entities are equal.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entities are equal
        """
        return self.compare(entity1, entity2) == 0

    def is_less_than(self, entity1: T, entity2: T) -> bool:
        """Check if first entity is less than second.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entity1 < entity2
        """
        return self.compare(entity1, entity2) < 0

    def is_greater_than(self, entity1: T, entity2: T) -> bool:
        """Check if first entity is greater than second.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            True if entity1 > entity2
        """
        return self.compare(entity1, entity2) > 0

    def sort_entities(self, entities: list[T]) -> list[T]:
        """Sort entities using the comparison logic.
        
        Args:
            entities: List of entities to sort
            
        Returns:
            Sorted list of entities
        """
        from functools import cmp_to_key
        return sorted(entities, key=cmp_to_key(self.compare))


class AggregationService(BaseService, Generic[T]):
    """Base class for aggregation services."""

    @abstractmethod
    def aggregate(self, entities: list[T]) -> Any:
        """Aggregate multiple entities.
        
        Args:
            entities: List of entities to aggregate
            
        Returns:
            Aggregation result
        """
        ...

    def can_aggregate(self, entities: list[T]) -> bool:
        """Check if entities can be aggregated.
        
        Args:
            entities: List of entities to check
            
        Returns:
            True if entities can be aggregated
        """
        return len(entities) > 0

    def validate_aggregation_inputs(self, entities: list[T]) -> None:
        """Validate aggregation inputs.
        
        Override in subclasses to add input validation.
        
        Args:
            entities: List of entities to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not entities:
            raise ValueError("Cannot aggregate empty list of entities")

    def aggregate_with_validation(self, entities: list[T]) -> Any:
        """Aggregate entities with input validation.
        
        Args:
            entities: List of entities to aggregate
            
        Returns:
            Aggregation result
        """
        self.validate_aggregation_inputs(entities)
        return self.aggregate(entities)


class BusinessRuleService(BaseService):
    """Base class for business rule services."""

    @abstractmethod
    def evaluate_rule(self, context: dict[str, Any]) -> bool:
        """Evaluate a business rule.
        
        Args:
            context: Context data for rule evaluation
            
        Returns:
            True if rule is satisfied
        """
        ...

    def get_rule_name(self) -> str:
        """Get the name of the business rule.
        
        Returns:
            Rule name
        """
        return self.__class__.__name__

    def get_rule_description(self) -> str:
        """Get a description of the business rule.
        
        Override in subclasses to provide rule description.
        
        Returns:
            Rule description
        """
        return f"Business rule: {self.get_rule_name()}"

    def validate_context(self, context: dict[str, Any]) -> None:
        """Validate the context for rule evaluation.
        
        Override in subclasses to add context validation.
        
        Args:
            context: Context data to validate
            
        Raises:
            ValueError: If context is invalid
        """
        pass

    def evaluate_with_validation(self, context: dict[str, Any]) -> bool:
        """Evaluate rule with context validation.
        
        Args:
            context: Context data for rule evaluation
            
        Returns:
            True if rule is satisfied
        """
        self.validate_context(context)
        return self.evaluate_rule(context)


class EventHandlerService(BaseService):
    """Base class for event handler services."""

    @abstractmethod
    def handle_event(self, event: Any) -> None:
        """Handle a domain event.
        
        Args:
            event: The domain event to handle
        """
        ...

    def can_handle_event(self, event: Any) -> bool:
        """Check if this service can handle an event.
        
        Override in subclasses to add event filtering.
        
        Args:
            event: The event to check
            
        Returns:
            True if event can be handled
        """
        return True

    def get_supported_event_types(self) -> list[str]:
        """Get the event types this service can handle.
        
        Override in subclasses to specify supported event types.
        
        Returns:
            List of supported event type names
        """
        return []

    def handle_event_with_validation(self, event: Any) -> None:
        """Handle event with validation.
        
        Args:
            event: The domain event to handle
        """
        if not self.can_handle_event(event):
            raise ValueError(f"Cannot handle event of type {type(event).__name__}")
        
        self.handle_event(event)


class ServiceRegistry:
    """Registry for managing services."""

    def __init__(self) -> None:
        """Initialize the service registry."""
        self._services: dict[str, BaseService] = {}

    def register_service(self, name: str, service: BaseService) -> None:
        """Register a service.
        
        Args:
            name: Name of the service
            service: The service instance
        """
        self._services[name] = service

    def get_service(self, name: str) -> BaseService:
        """Get a service by name.
        
        Args:
            name: Name of the service
            
        Returns:
            The service instance
            
        Raises:
            KeyError: If service is not found
        """
        return self._services[name]

    def has_service(self, name: str) -> bool:
        """Check if a service is registered.
        
        Args:
            name: Name of the service
            
        Returns:
            True if service is registered
        """
        return name in self._services

    def unregister_service(self, name: str) -> None:
        """Unregister a service.
        
        Args:
            name: Name of the service to unregister
        """
        self._services.pop(name, None)

    def get_all_services(self) -> dict[str, BaseService]:
        """Get all registered services.
        
        Returns:
            Dictionary of all services
        """
        return self._services.copy()

    def clear_services(self) -> None:
        """Clear all registered services."""
        self._services.clear()

    def initialize_all_services(self) -> None:
        """Initialize all registered services."""
        for service in self._services.values():
            service.initialize()

    def cleanup_all_services(self) -> None:
        """Clean up all registered services."""
        for service in self._services.values():
            service.cleanup()


# Global service registry instance
_service_registry: ServiceRegistry | None = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry
