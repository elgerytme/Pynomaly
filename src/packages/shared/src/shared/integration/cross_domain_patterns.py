"""Cross-domain integration patterns for secure inter-package communication.

This module provides standardized patterns for communication between domain packages
while maintaining strict domain boundaries and ensuring security, observability,
and compliance requirements are met.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, TypeVar, Generic, 
    Callable, Awaitable, Protocol, runtime_checkable
)
from uuid import uuid4
import structlog

from ..infrastructure.exceptions.base_exceptions import (
    BaseApplicationError, ErrorCategory, ErrorSeverity
)
from ..infrastructure.logging.structured_logging import StructuredLogger


logger = structlog.get_logger()

T = TypeVar('T')
R = TypeVar('R')


class IntegrationError(BaseApplicationError):
    """Cross-domain integration errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class MessageType(Enum):
    """Types of cross-domain messages."""
    COMMAND = "command"
    QUERY = "query" 
    EVENT = "event"
    RESPONSE = "response"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class IntegrationStatus(Enum):
    """Integration operation status."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


@dataclass
class IntegrationContext:
    """Context information for cross-domain operations."""
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    source_domain: str = ""
    target_domain: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossDomainMessage:
    """Standardized cross-domain message format."""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = MessageType.COMMAND
    priority: MessagePriority = MessagePriority.NORMAL
    source_domain: str = ""
    target_domain: str = ""
    operation: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    context: IntegrationContext = field(default_factory=IntegrationContext)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            **asdict(self),
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'context': asdict(self.context)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrossDomainMessage':
        """Create message from dictionary."""
        # Handle datetime parsing
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        
        # Handle enums
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        
        # Handle context
        context_data = data.pop('context', {})
        if 'timestamp' in context_data:
            context_data['timestamp'] = datetime.fromisoformat(context_data['timestamp'])
        data['context'] = IntegrationContext(**context_data)
        
        return cls(**data)


@dataclass
class IntegrationResult(Generic[T]):
    """Result of cross-domain integration operation."""
    status: IntegrationStatus
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None
    execution_time_ms: float = 0.0
    context: Optional[IntegrationContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DomainService(Protocol):
    """Protocol for domain services that can participate in cross-domain integration."""
    
    domain_name: str
    
    async def handle_message(
        self, 
        message: CrossDomainMessage
    ) -> IntegrationResult[Any]:
        """Handle incoming cross-domain message."""
        ...
    
    def get_supported_operations(self) -> List[str]:
        """Get list of operations this service supports."""
        ...


class MessageSerializer:
    """Handles serialization/deserialization of cross-domain messages."""
    
    @staticmethod
    def serialize(message: CrossDomainMessage) -> str:
        """Serialize message to JSON string."""
        try:
            return json.dumps(message.to_dict(), default=str, ensure_ascii=False)
        except Exception as e:
            raise IntegrationError(f"Message serialization failed: {e}") from e
    
    @staticmethod
    def deserialize(data: str) -> CrossDomainMessage:
        """Deserialize message from JSON string."""
        try:
            message_dict = json.loads(data)
            return CrossDomainMessage.from_dict(message_dict)
        except Exception as e:
            raise IntegrationError(f"Message deserialization failed: {e}") from e


class MessageValidator:
    """Validates cross-domain messages."""
    
    @staticmethod
    def validate_message(message: CrossDomainMessage) -> bool:
        """Validate message format and content."""
        try:
            # Basic validation
            if not message.message_id:
                return False
            
            if not message.source_domain or not message.target_domain:
                return False
            
            if not message.operation:
                return False
            
            # Check expiry
            if message.expires_at and datetime.now(timezone.utc) > message.expires_at:
                return False
            
            # Check retry limits
            if message.retry_count > message.max_retries:
                return False
            
            return True
            
        except Exception:
            return False


class CircuitBreaker:
    """Circuit breaker pattern for cross-domain calls."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = 'closed'  # closed, open, half-open
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        if self._state == 'open':
            if self._should_attempt_reset():
                self._state = 'half-open'
            else:
                raise IntegrationError("Circuit breaker is open")
        
        try:
            result = await func()
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self._last_failure_time is not None and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        self._failure_count = 0
        self._state = 'closed'
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = 'open'


class RetryPolicy:
    """Retry policy for failed cross-domain operations."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = min(
            self.base_delay * (self.backoff_multiplier ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        should_retry: Callable[[Exception], bool] = lambda e: True
    ) -> T:
        """Execute function with retry policy."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1 or not should_retry(e):
                    break
                
                delay = self.get_delay(attempt)
                logger.warning(
                    "Operation failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    delay=delay,
                    error=str(e)
                )
                await asyncio.sleep(delay)
        
        raise last_exception


class DomainEventBus:
    """Event bus for cross-domain event publishing and subscription."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[CrossDomainMessage], Awaitable[None]]]] = {}
        self._middleware: List[Callable[[CrossDomainMessage], Awaitable[CrossDomainMessage]]] = []
        self.logger = StructuredLogger({})
    
    def subscribe(
        self, 
        event_pattern: str, 
        handler: Callable[[CrossDomainMessage], Awaitable[None]]
    ) -> None:
        """Subscribe to events matching pattern."""
        if event_pattern not in self._subscribers:
            self._subscribers[event_pattern] = []
        self._subscribers[event_pattern].append(handler)
    
    def add_middleware(
        self, 
        middleware: Callable[[CrossDomainMessage], Awaitable[CrossDomainMessage]]
    ) -> None:
        """Add middleware to event processing pipeline."""
        self._middleware.append(middleware)
    
    async def publish(self, event: CrossDomainMessage) -> None:
        """Publish event to subscribers."""
        try:
            # Apply middleware
            processed_event = event
            for middleware in self._middleware:
                processed_event = await middleware(processed_event)
            
            # Find matching subscribers
            matching_handlers = []
            for pattern, handlers in self._subscribers.items():
                if self._matches_pattern(processed_event.operation, pattern):
                    matching_handlers.extend(handlers)
            
            # Publish to all matching subscribers
            tasks = [handler(processed_event) for handler in matching_handlers]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info(
                "Event published",
                event_id=processed_event.message_id,
                operation=processed_event.operation,
                subscribers_count=len(matching_handlers)
            )
            
        except Exception as e:
            self.logger.error("Event publishing failed", error=str(e))
            raise IntegrationError("Event publishing failed") from e
    
    def _matches_pattern(self, operation: str, pattern: str) -> bool:
        """Check if operation matches subscription pattern."""
        import fnmatch
        return fnmatch.fnmatch(operation, pattern)


class CrossDomainIntegrationManager:
    """Main manager for cross-domain integration operations."""
    
    def __init__(self):
        self._services: Dict[str, DomainService] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._event_bus = DomainEventBus()
        self._active_operations: Dict[str, IntegrationContext] = {}
        self.logger = StructuredLogger({})
        
        # Default retry policy
        self._retry_policy = RetryPolicy()
    
    def register_service(self, service: DomainService) -> None:
        """Register a domain service."""
        self._services[service.domain_name] = service
        self._circuit_breakers[service.domain_name] = CircuitBreaker()
        
        self.logger.info(
            "Domain service registered",
            domain=service.domain_name,
            operations=service.get_supported_operations()
        )
    
    def get_event_bus(self) -> DomainEventBus:
        """Get the event bus instance."""
        return self._event_bus
    
    async def send_command(
        self,
        target_domain: str,
        operation: str,
        payload: Dict[str, Any],
        context: Optional[IntegrationContext] = None
    ) -> IntegrationResult[Any]:
        """Send command to target domain."""
        if context is None:
            context = IntegrationContext()
        
        message = CrossDomainMessage(
            message_type=MessageType.COMMAND,
            source_domain=context.source_domain,
            target_domain=target_domain,
            operation=operation,
            payload=payload,
            context=context
        )
        
        return await self._send_message(message)
    
    async def send_query(
        self,
        target_domain: str,
        operation: str,
        payload: Dict[str, Any],
        context: Optional[IntegrationContext] = None
    ) -> IntegrationResult[Any]:
        """Send query to target domain."""
        if context is None:
            context = IntegrationContext()
        
        message = CrossDomainMessage(
            message_type=MessageType.QUERY,
            source_domain=context.source_domain,
            target_domain=target_domain,
            operation=operation,
            payload=payload,
            context=context
        )
        
        return await self._send_message(message)
    
    async def publish_event(
        self,
        operation: str,
        payload: Dict[str, Any],
        context: Optional[IntegrationContext] = None
    ) -> None:
        """Publish domain event."""
        if context is None:
            context = IntegrationContext()
        
        event = CrossDomainMessage(
            message_type=MessageType.EVENT,
            source_domain=context.source_domain,
            target_domain="*",  # Broadcast
            operation=operation,
            payload=payload,
            context=context
        )
        
        await self._event_bus.publish(event)
    
    async def _send_message(self, message: CrossDomainMessage) -> IntegrationResult[Any]:
        """Send message to target service."""
        start_time = time.time()
        
        try:
            # Validate message
            if not MessageValidator.validate_message(message):
                return IntegrationResult(
                    status=IntegrationStatus.FAILED,
                    error="Invalid message format",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check if target service exists
            target_service = self._services.get(message.target_domain)
            if not target_service:
                return IntegrationResult(
                    status=IntegrationStatus.FAILED,
                    error=f"Target domain not found: {message.target_domain}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Track operation
            self._active_operations[message.message_id] = message.context
            
            try:
                # Use circuit breaker
                circuit_breaker = self._circuit_breakers[message.target_domain]
                
                async def execute_call():
                    return await target_service.handle_message(message)
                
                # Execute with retry policy
                result = await self._retry_policy.execute_with_retry(
                    lambda: circuit_breaker.call(execute_call)
                )
                
                execution_time = (time.time() - start_time) * 1000
                result.execution_time_ms = execution_time
                result.context = message.context
                
                self.logger.info(
                    "Cross-domain operation completed",
                    message_id=message.message_id,
                    source_domain=message.source_domain,
                    target_domain=message.target_domain,
                    operation=message.operation,
                    status=result.status.name,
                    execution_time_ms=execution_time
                )
                
                return result
                
            finally:
                # Clean up tracking
                self._active_operations.pop(message.message_id, None)
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.error(
                "Cross-domain operation failed",
                message_id=message.message_id,
                source_domain=message.source_domain,
                target_domain=message.target_domain,
                operation=message.operation,
                error=str(e),
                execution_time_ms=execution_time
            )
            
            return IntegrationResult(
                status=IntegrationStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
                context=message.context
            )
    
    def get_active_operations(self) -> Dict[str, IntegrationContext]:
        """Get currently active operations."""
        return self._active_operations.copy()
    
    def get_registered_services(self) -> List[str]:
        """Get list of registered domain services."""
        return list(self._services.keys())


# Anti-corruption layer pattern
class AntiCorruptionLayer(ABC):
    """Abstract base class for anti-corruption layers between domains."""
    
    @abstractmethod
    async def translate_inbound(self, message: CrossDomainMessage) -> CrossDomainMessage:
        """Translate incoming message to internal domain format."""
        pass
    
    @abstractmethod
    async def translate_outbound(self, message: CrossDomainMessage) -> CrossDomainMessage:
        """Translate outgoing message to external domain format."""
        pass


# Saga pattern for distributed transactions
@dataclass
class SagaStep:
    """Individual step in a saga."""
    step_id: str
    operation: str
    target_domain: str
    payload: Dict[str, Any]
    compensation_operation: Optional[str] = None
    compensation_payload: Optional[Dict[str, Any]] = None
    completed: bool = False
    compensated: bool = False


class SagaOrchestrator:
    """Orchestrator for distributed saga transactions."""
    
    def __init__(self, integration_manager: CrossDomainIntegrationManager):
        self.integration_manager = integration_manager
        self.logger = StructuredLogger({})
        self._active_sagas: Dict[str, List[SagaStep]] = {}
    
    async def execute_saga(
        self, 
        saga_id: str, 
        steps: List[SagaStep],
        context: Optional[IntegrationContext] = None
    ) -> IntegrationResult[Dict[str, Any]]:
        """Execute a distributed saga transaction."""
        self._active_sagas[saga_id] = steps
        completed_steps = []
        
        try:
            # Execute all steps
            for step in steps:
                result = await self.integration_manager.send_command(
                    target_domain=step.target_domain,
                    operation=step.operation,
                    payload=step.payload,
                    context=context
                )
                
                if result.status != IntegrationStatus.COMPLETED:
                    # Step failed, execute compensation
                    await self._compensate_saga(saga_id, completed_steps)
                    return IntegrationResult(
                        status=IntegrationStatus.FAILED,
                        error=f"Saga step failed: {step.step_id}",
                        metadata={"failed_step": step.step_id}
                    )
                
                step.completed = True
                completed_steps.append(step)
            
            # All steps completed successfully
            del self._active_sagas[saga_id]
            
            return IntegrationResult(
                status=IntegrationStatus.COMPLETED,
                data={"saga_id": saga_id, "completed_steps": len(completed_steps)},
                message="Saga completed successfully"
            )
            
        except Exception as e:
            # Execute compensation for any completed steps
            await self._compensate_saga(saga_id, completed_steps)
            
            return IntegrationResult(
                status=IntegrationStatus.FAILED,
                error=f"Saga execution failed: {e}",
                metadata={"exception_type": type(e).__name__}
            )
    
    async def _compensate_saga(self, saga_id: str, completed_steps: List[SagaStep]) -> None:
        """Execute compensation for completed saga steps."""
        # Execute compensation in reverse order
        for step in reversed(completed_steps):
            if step.compensation_operation and not step.compensated:
                try:
                    await self.integration_manager.send_command(
                        target_domain=step.target_domain,
                        operation=step.compensation_operation,
                        payload=step.compensation_payload or {}
                    )
                    step.compensated = True
                    
                except Exception as e:
                    self.logger.error(
                        "Saga compensation failed",
                        saga_id=saga_id,
                        step_id=step.step_id,
                        error=str(e)
                    )
        
        # Clean up
        if saga_id in self._active_sagas:
            del self._active_sagas[saga_id]


# Global integration manager instance
_integration_manager: Optional[CrossDomainIntegrationManager] = None


def get_integration_manager() -> CrossDomainIntegrationManager:
    """Get global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = CrossDomainIntegrationManager()
    return _integration_manager


# Convenience functions for common integration patterns
async def send_domain_command(
    target_domain: str,
    operation: str,
    payload: Dict[str, Any],
    source_domain: str = "",
    user_id: Optional[str] = None
) -> IntegrationResult[Any]:
    """Send command to another domain."""
    context = IntegrationContext(
        source_domain=source_domain,
        target_domain=target_domain,
        user_id=user_id
    )
    
    manager = get_integration_manager()
    return await manager.send_command(target_domain, operation, payload, context)


async def query_domain(
    target_domain: str,
    operation: str,
    payload: Dict[str, Any],
    source_domain: str = "",
    user_id: Optional[str] = None
) -> IntegrationResult[Any]:
    """Query another domain."""
    context = IntegrationContext(
        source_domain=source_domain,
        target_domain=target_domain,
        user_id=user_id
    )
    
    manager = get_integration_manager()
    return await manager.send_query(target_domain, operation, payload, context)


async def publish_domain_event(
    operation: str,
    payload: Dict[str, Any],
    source_domain: str = "",
    user_id: Optional[str] = None
) -> None:
    """Publish domain event."""
    context = IntegrationContext(
        source_domain=source_domain,
        user_id=user_id
    )
    
    manager = get_integration_manager()
    await manager.publish_event(operation, payload, context)