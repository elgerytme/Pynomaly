"""
Advanced infrastructure implementations for CQRS, Event Sourcing, and Saga patterns.

This module provides production-ready implementations of sophisticated
architectural patterns for complex business scenarios requiring advanced
coordination between packages.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, Callable, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import asdict
import weakref
import uuid

from interfaces.advanced_patterns import (
    Command, Query, CommandResponse, QueryResponse, CommandResult,
    CommandHandler, QueryHandler, CommandBus, QueryBus,
    EventStore, EventStoreEvent, Aggregate, Repository,
    SagaOrchestrator, SagaState, SagaStep, SagaStatus,
    ReadModel, ProjectionManager,
    CQRSConfiguration, EventSourcingConfiguration,
    WorkflowStarted, WorkflowStepCompleted, WorkflowFailed, WorkflowCompleted,
    CompensationRequired, CompensationCompleted
)
from interfaces.events import DomainEvent, EventPriority
from .dependency_injection import DIContainer
from .event_bus import DistributedEventBus


logger = logging.getLogger(__name__)


# =============================================================================
# CQRS Implementation
# =============================================================================

class InMemoryCommandBus(CommandBus):
    """In-memory implementation of command bus."""
    
    def __init__(self, config: CQRSConfiguration, container: DIContainer):
        self.config = config
        self.container = container
        self.handlers: Dict[Type[Command], CommandHandler] = {}
        self.metrics = {
            "commands_processed": 0,
            "commands_failed": 0,
            "avg_execution_time_ms": 0.0,
            "handler_cache_hits": 0,
            "validation_failures": 0
        }
        self.execution_times = deque(maxlen=1000)
        
        # Command batching support
        self.pending_commands: List[Command] = []
        self.batch_timer: Optional[asyncio.Handle] = None
        
        # Circuit breaker for failed commands
        self.failure_counts: Dict[Type[Command], int] = defaultdict(int)
        self.circuit_breakers: Dict[Type[Command], bool] = defaultdict(bool)
    
    async def send(self, command: Command) -> CommandResponse:
        """Send a command for processing."""
        start_time = time.time()
        
        try:
            # Validate command if enabled
            if self.config.enable_command_validation:
                validation_result = await self._validate_command(command)
                if not validation_result:
                    self.metrics["validation_failures"] += 1
                    return CommandResponse(
                        command_id=command.command_id,
                        result=CommandResult.REJECTED,
                        error_message="Command validation failed"
                    )
            
            # Check circuit breaker
            command_type = type(command)
            if self.circuit_breakers[command_type]:
                return CommandResponse(
                    command_id=command.command_id,
                    result=CommandResult.FAILED,
                    error_message="Circuit breaker is open for this command type"
                )
            
            # Batch commands if enabled
            if self.config.enable_command_batching:
                return await self._handle_batched_command(command)
            
            # Handle command directly
            response = await self._execute_command(command)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._update_metrics(command_type, execution_time, response.result == CommandResult.SUCCESS)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing command {command.command_id}: {e}")
            self.metrics["commands_failed"] += 1
            self.failure_counts[command_type] += 1
            
            # Open circuit breaker if too many failures
            if self.failure_counts[command_type] >= 5:
                self.circuit_breakers[command_type] = True
                logger.warning(f"Circuit breaker opened for {command_type.__name__}")
            
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.FAILED,
                error_message=str(e)
            )
    
    def register_handler(self, command_type: Type[Command], handler: CommandHandler) -> None:
        """Register a command handler."""
        self.handlers[command_type] = handler
        logger.info(f"Registered handler for {command_type.__name__}")
    
    async def _validate_command(self, command: Command) -> bool:
        """Validate command before processing."""
        # Basic validation - can be extended
        return (
            command.command_id is not None and
            command.timestamp is not None and
            isinstance(command.metadata, dict)
        )
    
    async def _execute_command(self, command: Command) -> CommandResponse:
        """Execute a single command."""
        command_type = type(command)
        handler = self.handlers.get(command_type)
        
        if not handler:
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.FAILED,
                error_message=f"No handler found for {command_type.__name__}"
            )
        
        try:
            # Set timeout
            response = await asyncio.wait_for(
                handler.handle(command),
                timeout=self.config.command_timeout_seconds
            )
            self.metrics["commands_processed"] += 1
            return response
            
        except asyncio.TimeoutError:
            return CommandResponse(
                command_id=command.command_id,
                result=CommandResult.FAILED,
                error_message="Command execution timeout"
            )
    
    async def _handle_batched_command(self, command: Command) -> CommandResponse:
        """Handle command in batch mode."""
        self.pending_commands.append(command)
        
        # Process batch if size limit reached
        if len(self.pending_commands) >= self.config.batch_size:
            await self._process_command_batch()
        
        # Set timer for batch processing
        elif not self.batch_timer:
            self.batch_timer = asyncio.get_event_loop().call_later(
                self.config.batch_timeout_ms / 1000,
                lambda: asyncio.create_task(self._process_command_batch())
            )
        
        # For now, return pending - in real implementation would track completion
        return CommandResponse(
            command_id=command.command_id,
            result=CommandResult.PENDING,
            metadata={"batched": True}
        )
    
    async def _process_command_batch(self) -> None:
        """Process a batch of commands."""
        if not self.pending_commands:
            return
        
        batch = self.pending_commands.copy()
        self.pending_commands.clear()
        
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Process commands in parallel
        tasks = [self._execute_command(cmd) for cmd in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _update_metrics(self, command_type: Type[Command], execution_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.execution_times.append(execution_time)
        
        # Reset failure count on success
        if success:
            self.failure_counts[command_type] = 0
            self.circuit_breakers[command_type] = False
        
        # Update average execution time
        if self.execution_times:
            self.metrics["avg_execution_time_ms"] = sum(self.execution_times) / len(self.execution_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get command bus metrics."""
        return {
            **self.metrics,
            "registered_handlers": len(self.handlers),
            "pending_commands": len(self.pending_commands),
            "circuit_breakers": {
                cmd_type.__name__: is_open 
                for cmd_type, is_open in self.circuit_breakers.items() 
                if is_open
            }
        }


class InMemoryQueryBus(QueryBus):
    """In-memory implementation of query bus."""
    
    def __init__(self, config: CQRSConfiguration, container: DIContainer):
        self.config = config
        self.container = container
        self.handlers: Dict[Type[Query], QueryHandler] = {}
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.metrics = {
            "queries_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_execution_time_ms": 0.0
        }
        self.execution_times = deque(maxlen=1000)
    
    async def ask(self, query: Query) -> QueryResponse:
        """Execute a query and get response."""
        start_time = time.time()
        
        try:
            # Check cache if enabled
            if self.config.enable_query_caching:
                cache_key = self._get_cache_key(query)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    return cached_result
                self.metrics["cache_misses"] += 1
            
            # Execute query
            response = await self._execute_query(query)
            
            # Cache response if successful
            if self.config.enable_query_caching and response.error_message is None:
                self._cache_response(self._get_cache_key(query), response)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.execution_times.append(execution_time)
            self.metrics["queries_processed"] += 1
            
            if self.execution_times:
                self.metrics["avg_execution_time_ms"] = sum(self.execution_times) / len(self.execution_times)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query {query.query_id}: {e}")
            return QueryResponse(
                query_id=query.query_id,
                error_message=str(e)
            )
    
    def register_handler(self, query_type: Type[Query], handler: QueryHandler) -> None:
        """Register a query handler."""
        self.handlers[query_type] = handler
        logger.info(f"Registered handler for {query_type.__name__}")
    
    async def _execute_query(self, query: Query) -> QueryResponse:
        """Execute a single query."""
        query_type = type(query)
        handler = self.handlers.get(query_type)
        
        if not handler:
            return QueryResponse(
                query_id=query.query_id,
                error_message=f"No handler found for {query_type.__name__}"
            )
        
        try:
            return await asyncio.wait_for(
                handler.handle(query),
                timeout=self.config.query_timeout_seconds
            )
        except asyncio.TimeoutError:
            return QueryResponse(
                query_id=query.query_id,
                error_message="Query execution timeout"
            )
    
    def _get_cache_key(self, query: Query) -> str:
        """Generate cache key for query."""
        # Simple implementation - would be more sophisticated in production
        query_data = asdict(query)
        return f"{type(query).__name__}:{hash(str(sorted(query_data.items())))}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[QueryResponse]:
        """Get response from cache."""
        if cache_key not in self.cache:
            return None
        
        # Check if cache entry is still valid (simple TTL of 5 minutes)
        cached_time = self.cache_timestamps.get(cache_key)
        if cached_time and datetime.utcnow() - cached_time > timedelta(minutes=5):
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        return self.cache[cache_key]
    
    def _cache_response(self, cache_key: str, response: QueryResponse) -> None:
        """Cache a query response."""
        self.cache[cache_key] = response
        self.cache_timestamps[cache_key] = datetime.utcnow()
        
        # Simple cache size management
        if len(self.cache) > 1000:
            oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get query bus metrics."""
        cache_hit_rate = 0.0
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total_requests
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "registered_handlers": len(self.handlers)
        }


# =============================================================================
# Event Sourcing Implementation
# =============================================================================

class InMemoryEventStore(EventStore):
    """In-memory implementation of event store."""
    
    def __init__(self, config: EventSourcingConfiguration):
        self.config = config
        self.events: Dict[str, List[EventStoreEvent]] = defaultdict(list)
        self.global_events: List[EventStoreEvent] = []
        self.snapshots: Dict[str, Any] = {}
        self.version_map: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.metrics = {
            "events_stored": 0,
            "events_read": 0,
            "snapshots_created": 0,
            "snapshots_used": 0
        }
    
    async def append_events(
        self, 
        aggregate_id: str, 
        events: List[DomainEvent], 
        expected_version: int
    ) -> None:
        """Append events to the store."""
        current_version = self.version_map[aggregate_id]
        
        if current_version != expected_version:
            raise ValueError(f"Concurrency conflict: expected version {expected_version}, got {current_version}")
        
        # Convert domain events to store events
        store_events = []
        for i, event in enumerate(events):
            version = current_version + i + 1
            store_event = EventStoreEvent.from_domain_event(
                event, 
                aggregate_type="Unknown",  # Would be determined from aggregate
                version=version
            )
            store_events.append(store_event)
        
        # Store events
        self.events[aggregate_id].extend(store_events)
        self.global_events.extend(store_events)
        self.version_map[aggregate_id] = current_version + len(events)
        self.metrics["events_stored"] += len(events)
        
        # Create snapshot if needed
        if (self.config.enable_snapshots and 
            self.version_map[aggregate_id] % self.config.snapshot_frequency == 0):
            await self._create_snapshot(aggregate_id)
    
    async def get_events(
        self, 
        aggregate_id: str, 
        from_version: int = 0
    ) -> List[EventStoreEvent]:
        """Get events for an aggregate."""
        aggregate_events = self.events.get(aggregate_id, [])
        filtered_events = [
            event for event in aggregate_events 
            if event.version > from_version
        ]
        
        self.metrics["events_read"] += len(filtered_events)
        return filtered_events
    
    async def get_events_by_type(
        self, 
        event_type: str, 
        from_timestamp: Optional[datetime] = None
    ) -> List[EventStoreEvent]:
        """Get all events of a specific type."""
        filtered_events = [
            event for event in self.global_events
            if event.event_type == event_type
        ]
        
        if from_timestamp:
            filtered_events = [
                event for event in filtered_events
                if event.timestamp >= from_timestamp
            ]
        
        self.metrics["events_read"] += len(filtered_events)
        return filtered_events
    
    async def _create_snapshot(self, aggregate_id: str) -> None:
        """Create snapshot for an aggregate."""
        # This would reconstruct the aggregate and save its state
        # For now, just track that a snapshot was created
        self.snapshots[aggregate_id] = {
            "version": self.version_map[aggregate_id],
            "timestamp": datetime.utcnow(),
            "data": {}  # Would contain aggregate state
        }
        self.metrics["snapshots_created"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event store metrics."""
        return {
            **self.metrics,
            "total_aggregates": len(self.events),
            "total_events": len(self.global_events),
            "snapshots_count": len(self.snapshots)
        }


class EventSourcedRepository(Repository[Aggregate]):
    """Repository implementation for event-sourced aggregates."""
    
    def __init__(self, event_store: EventStore, aggregate_factory: Callable[[str], Aggregate]):
        self.event_store = event_store
        self.aggregate_factory = aggregate_factory
        self.aggregate_cache: Dict[str, weakref.ReferenceType] = {}
    
    async def get_by_id(self, aggregate_id: str) -> Optional[Aggregate]:
        """Get aggregate by ID."""
        # Check cache first
        if aggregate_id in self.aggregate_cache:
            cached_aggregate = self.aggregate_cache[aggregate_id]()
            if cached_aggregate:
                return cached_aggregate
        
        # Load from event store
        events = await self.event_store.get_events(aggregate_id)
        if not events:
            return None
        
        # Reconstruct aggregate from events
        domain_events = [event.to_domain_event() for event in events]
        aggregate = self.aggregate_factory(aggregate_id)
        
        for event in domain_events:
            aggregate.apply_event(event)
        
        aggregate.mark_events_as_committed()
        
        # Cache aggregate (weak reference)
        self.aggregate_cache[aggregate_id] = weakref.ref(aggregate)
        
        return aggregate
    
    async def save(self, aggregate: Aggregate) -> None:
        """Save aggregate."""
        uncommitted_events = aggregate.get_uncommitted_events()
        if not uncommitted_events:
            return
        
        await self.event_store.append_events(
            aggregate.aggregate_id,
            uncommitted_events,
            aggregate.version - len(uncommitted_events)
        )
        
        aggregate.mark_events_as_committed()


# =============================================================================
# Saga Implementation
# =============================================================================

class InMemorySagaOrchestrator(SagaOrchestrator):
    """In-memory implementation of saga orchestrator."""
    
    def __init__(self, command_bus: CommandBus, event_bus: DistributedEventBus):
        self.command_bus = command_bus
        self.event_bus = event_bus
        self.sagas: Dict[str, SagaState] = {}
        self.metrics = {
            "sagas_started": 0,
            "sagas_completed": 0,
            "sagas_failed": 0,
            "sagas_compensated": 0,
            "compensation_steps_executed": 0
        }
    
    async def start_saga(self, saga_id: str, steps: List[SagaStep]) -> SagaState:
        """Start a new saga."""
        saga_state = SagaState(
            saga_id=saga_id,
            status=SagaStatus.STARTED,
            current_step=0,
            steps=steps
        )
        
        self.sagas[saga_id] = saga_state
        self.metrics["sagas_started"] += 1
        
        # Publish saga started event
        await self.event_bus.publish(WorkflowStarted(
            event_id=str(uuid.uuid4()),
            event_type="WorkflowStarted",
            aggregate_id=saga_id,
            occurred_at=datetime.utcnow(),
            workflow_id=saga_id,
            workflow_type="saga",
            initiator="saga_orchestrator"
        ))
        
        # Execute first step
        if steps:
            await self._execute_next_step(saga_state)
        
        return saga_state
    
    async def handle_step_completion(
        self, 
        saga_id: str, 
        step_id: str, 
        result: CommandResponse
    ) -> SagaState:
        """Handle completion of a saga step."""
        saga_state = self.sagas.get(saga_id)
        if not saga_state:
            raise ValueError(f"Saga {saga_id} not found")
        
        if result.result == CommandResult.SUCCESS:
            saga_state.completed_steps.append(step_id)
            saga_state.current_step += 1
            saga_state.updated_at = datetime.utcnow()
            
            # Publish step completed event
            await self.event_bus.publish(WorkflowStepCompleted(
                event_id=str(uuid.uuid4()),
                event_type="WorkflowStepCompleted",
                aggregate_id=saga_id,
                occurred_at=datetime.utcnow(),
                workflow_id=saga_id,
                step_id=step_id,
                step_type="saga_step"
            ))
            
            # Check if saga is complete
            if saga_state.current_step >= len(saga_state.steps):
                saga_state.status = SagaStatus.COMPLETED
                self.metrics["sagas_completed"] += 1
                
                await self.event_bus.publish(WorkflowCompleted(
                    event_id=str(uuid.uuid4()),
                    event_type="WorkflowCompleted",
                    aggregate_id=saga_id,
                    occurred_at=datetime.utcnow(),
                    workflow_id=saga_id,
                    final_output={"status": "completed"},
                    execution_time_ms=(datetime.utcnow() - saga_state.started_at).total_seconds() * 1000
                ))
            else:
                # Execute next step
                await self._execute_next_step(saga_state)
        else:
            # Step failed, start compensation
            await self.handle_step_failure(saga_id, step_id, result.error_message or "Unknown error")
        
        return saga_state
    
    async def handle_step_failure(
        self, 
        saga_id: str, 
        step_id: str, 
        error: str
    ) -> SagaState:
        """Handle failure of a saga step."""
        saga_state = self.sagas.get(saga_id)
        if not saga_state:
            raise ValueError(f"Saga {saga_id} not found")
        
        saga_state.status = SagaStatus.COMPENSATING
        saga_state.failed_step = step_id
        saga_state.error_message = error
        saga_state.updated_at = datetime.utcnow()
        
        # Publish failure event
        await self.event_bus.publish(WorkflowFailed(
            event_id=str(uuid.uuid4()),
            event_type="WorkflowFailed",
            aggregate_id=saga_id,
            occurred_at=datetime.utcnow(),
            workflow_id=saga_id,
            failed_step=step_id,
            error_message=error
        ))
        
        # Start compensation
        await self._start_compensation(saga_state)
        
        return saga_state
    
    async def get_saga_state(self, saga_id: str) -> Optional[SagaState]:
        """Get current saga state."""
        return self.sagas.get(saga_id)
    
    async def _execute_next_step(self, saga_state: SagaState) -> None:
        """Execute the next step in the saga."""
        if saga_state.current_step >= len(saga_state.steps):
            return
        
        current_step = saga_state.steps[saga_state.current_step]
        saga_state.status = SagaStatus.IN_PROGRESS
        
        try:
            # Set timeout for step execution
            response = await asyncio.wait_for(
                self.command_bus.send(current_step.command),
                timeout=current_step.timeout_seconds or 30
            )
            
            await self.handle_step_completion(
                saga_state.saga_id,
                current_step.step_id,
                response
            )
        except Exception as e:
            await self.handle_step_failure(
                saga_state.saga_id,
                current_step.step_id,
                str(e)
            )
    
    async def _start_compensation(self, saga_state: SagaState) -> None:
        """Start compensation process."""
        compensation_steps = []
        
        # Collect compensation commands for completed steps (in reverse order)
        for step_id in reversed(saga_state.completed_steps):
            step = next((s for s in saga_state.steps if s.step_id == step_id), None)
            if step and step.compensation_command:
                compensation_steps.append(step.compensation_command)
        
        if compensation_steps:
            await self.event_bus.publish(CompensationRequired(
                event_id=str(uuid.uuid4()),
                event_type="CompensationRequired",
                aggregate_id=saga_state.saga_id,
                occurred_at=datetime.utcnow(),
                saga_id=saga_state.saga_id,
                failed_step=saga_state.failed_step or "unknown",
                compensation_steps=[cmd.command_id for cmd in compensation_steps],
                reason=saga_state.error_message or "Step failure"
            ))
            
            # Execute compensation commands
            for compensation_command in compensation_steps:
                try:
                    await self.command_bus.send(compensation_command)
                    self.metrics["compensation_steps_executed"] += 1
                except Exception as e:
                    logger.error(f"Compensation command failed: {e}")
            
            saga_state.status = SagaStatus.COMPENSATED
            self.metrics["sagas_compensated"] += 1
            
            await self.event_bus.publish(CompensationCompleted(
                event_id=str(uuid.uuid4()),
                event_type="CompensationCompleted",
                aggregate_id=saga_state.saga_id,
                occurred_at=datetime.utcnow(),
                saga_id=saga_state.saga_id,
                compensated_steps=saga_state.completed_steps,
                final_state="compensated"
            ))
        else:
            saga_state.status = SagaStatus.FAILED
            self.metrics["sagas_failed"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get saga orchestrator metrics."""
        return {
            **self.metrics,
            "active_sagas": len([s for s in self.sagas.values() if s.status in [SagaStatus.STARTED, SagaStatus.IN_PROGRESS]]),
            "total_sagas": len(self.sagas)
        }


# =============================================================================
# Projection Manager Implementation
# =============================================================================

class InMemoryProjectionManager(ProjectionManager):
    """In-memory implementation of projection manager."""
    
    def __init__(self, event_store: EventStore, event_bus: DistributedEventBus):
        self.event_store = event_store
        self.event_bus = event_bus
        self.projections: Dict[Type[ReadModel], ReadModel] = {}
        self.projection_positions: Dict[Type[ReadModel], int] = {}
        self.metrics = {
            "events_projected": 0,
            "projections_rebuilt": 0,
            "projection_errors": 0
        }
    
    async def register_projection(self, read_model: ReadModel) -> None:
        """Register a read model projection."""
        read_model_type = type(read_model)
        self.projections[read_model_type] = read_model
        self.projection_positions[read_model_type] = 0
        
        # Subscribe to supported events
        for event_type in read_model.get_supported_events():
            self.event_bus.subscribe(event_type, self._handle_projection_event)
        
        logger.info(f"Registered projection for {read_model_type.__name__}")
    
    async def rebuild_projection(self, read_model_type: Type[ReadModel]) -> None:
        """Rebuild a projection from event store."""
        if read_model_type not in self.projections:
            raise ValueError(f"Projection {read_model_type.__name__} not registered")
        
        read_model = self.projections[read_model_type]
        
        # Get all events that this projection handles
        for event_type in read_model.get_supported_events():
            events = await self.event_store.get_events_by_type(event_type.__name__)
            
            for event_store_event in events:
                try:
                    domain_event = event_store_event.to_domain_event()
                    await read_model.handle_event(domain_event)
                    self.metrics["events_projected"] += 1
                except Exception as e:
                    logger.error(f"Error projecting event {event_store_event.event_id}: {e}")
                    self.metrics["projection_errors"] += 1
        
        self.metrics["projections_rebuilt"] += 1
        logger.info(f"Rebuilt projection for {read_model_type.__name__}")
    
    async def get_projection_status(self, read_model_type: Type[ReadModel]) -> Dict[str, Any]:
        """Get status of a projection."""
        if read_model_type not in self.projections:
            return {"status": "not_registered"}
        
        return {
            "status": "active",
            "position": self.projection_positions.get(read_model_type, 0),
            "supported_events": [
                event_type.__name__ 
                for event_type in self.projections[read_model_type].get_supported_events()
            ]
        }
    
    async def _handle_projection_event(self, event: DomainEvent) -> None:
        """Handle event for all relevant projections."""
        event_type = type(event)
        
        for read_model_type, read_model in self.projections.items():
            if event_type in read_model.get_supported_events():
                try:
                    await read_model.handle_event(event)
                    self.projection_positions[read_model_type] += 1
                    self.metrics["events_projected"] += 1
                except Exception as e:
                    logger.error(f"Error in projection {read_model_type.__name__}: {e}")
                    self.metrics["projection_errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get projection manager metrics."""
        return {
            **self.metrics,
            "registered_projections": len(self.projections),
            "projection_positions": {
                projection_type.__name__: position
                for projection_type, position in self.projection_positions.items()
            }
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_cqrs_infrastructure(
    config: CQRSConfiguration,
    container: DIContainer
) -> tuple[CommandBus, QueryBus]:
    """Create CQRS infrastructure components."""
    command_bus = InMemoryCommandBus(config, container)
    query_bus = InMemoryQueryBus(config, container)
    
    # Register in DI container
    container.register_singleton(CommandBus, factory=lambda: command_bus)
    container.register_singleton(QueryBus, factory=lambda: query_bus)
    
    return command_bus, query_bus


def create_event_sourcing_infrastructure(
    config: EventSourcingConfiguration
) -> tuple[EventStore, ProjectionManager]:
    """Create event sourcing infrastructure components."""
    event_store = InMemoryEventStore(config)
    
    # ProjectionManager needs event bus - would be injected in real implementation
    from .event_bus import get_event_bus
    event_bus = get_event_bus()
    projection_manager = InMemoryProjectionManager(event_store, event_bus)
    
    return event_store, projection_manager


def create_saga_orchestrator(
    command_bus: CommandBus,
    event_bus: DistributedEventBus
) -> SagaOrchestrator:
    """Create saga orchestrator."""
    return InMemorySagaOrchestrator(command_bus, event_bus)


def get_advanced_infrastructure_metrics() -> Dict[str, Any]:
    """Get metrics from all advanced infrastructure components."""
    # This would collect metrics from all registered components
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "advanced_patterns_enabled": True,
        "note": "Individual component metrics available through their respective get_metrics() methods"
    }