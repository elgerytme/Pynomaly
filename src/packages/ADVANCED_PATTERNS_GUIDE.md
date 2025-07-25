# Advanced Patterns Guide: CQRS, Event Sourcing & Saga

This guide covers sophisticated architectural patterns for complex business scenarios requiring advanced coordination between packages.

## 🎯 Overview

The advanced patterns extend the basic package interaction framework with:

- **CQRS (Command Query Responsibility Segregation)**: Separate read and write operations
- **Event Sourcing**: Store state as a sequence of events
- **Saga Pattern**: Manage distributed transactions and workflows
- **Read Models & Projections**: Optimized views for queries
- **Advanced Coordination**: Complex multi-step business processes

## 🏗️ CQRS Pattern

### Core Concepts

CQRS separates commands (write operations) from queries (read operations):

```python
from interfaces.advanced_patterns import (
    Command, Query, CommandResponse, QueryResponse,
    CommandBus, QueryBus, CommandHandler, QueryHandler
)

# Commands change state
@dataclass
class ProcessDataQualityCommand(Command):
    dataset_id: str
    quality_rules: List[str]
    priority: str = "normal"

# Queries read state
@dataclass
class GetDataQualityReportQuery(Query):
    dataset_id: str
    include_details: bool = True
```

### Command Handling

```python
class DataQualityCommandHandler:
    async def handle(self, command: ProcessDataQualityCommand) -> CommandResponse:
        # Execute business logic
        result = await self.process_quality_check(command)
        
        # Publish domain events
        event = DataQualityCheckCompleted(
            dataset_id=command.dataset_id,
            status=result.status,
            overall_score=result.score
        )
        await publish_event(event)
        
        return CommandResponse(
            command_id=command.command_id,
            result=CommandResult.SUCCESS,
            events=[event]
        )
```

### Query Handling

```python
class DataQualityQueryHandler:
    async def handle(self, query: GetDataQualityReportQuery) -> QueryResponse:
        # Read from optimized read models
        report_data = await self.get_quality_report(query.dataset_id)
        
        return QueryResponse(
            query_id=query.query_id,
            data=report_data,
            execution_time_ms=processing_time
        )
```

### Setting Up CQRS

```python
from shared.advanced_infrastructure import create_cqrs_infrastructure

# Configure CQRS
config = CQRSConfiguration(
    enable_command_validation=True,
    enable_query_caching=True,
    command_timeout_seconds=30,
    query_timeout_seconds=10
)

command_bus, query_bus = create_cqrs_infrastructure(config, container)

# Register handlers
command_bus.register_handler(ProcessDataQualityCommand, quality_command_handler)
query_bus.register_handler(GetDataQualityReportQuery, quality_query_handler)

# Execute commands and queries
response = await command_bus.send(quality_command)
query_result = await query_bus.ask(quality_query)
```

## 📦 Event Sourcing Pattern

### Core Concepts

Event sourcing stores state as a sequence of events instead of current state:

```python
from interfaces.advanced_patterns import Aggregate, EventStore, Repository

class DataProcessingWorkflowAggregate(Aggregate):
    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.status = DataProcessingStatus.PENDING
        self.quality_score = 0.0
        self.processing_steps = []
    
    def start_processing(self, dataset_id: str, processing_type: str):
        # Business logic validation
        if self.status != DataProcessingStatus.PENDING:
            raise ValueError(f"Cannot start processing in status {self.status}")
        
        # Raise domain event
        event = DataProcessingStarted(
            aggregate_id=self.aggregate_id,
            dataset_id=dataset_id,
            processing_type=processing_type
        )
        self.raise_event(event)
    
    def _when(self, event: DomainEvent):
        """Apply event to aggregate state"""
        if isinstance(event, DataProcessingStarted):
            self.status = DataProcessingStatus.IN_PROGRESS
            self.processing_steps.append("started")
```

### Event Store Usage

```python
from shared.advanced_infrastructure import InMemoryEventStore, EventSourcedRepository

# Create event store
config = EventSourcingConfiguration(
    enable_snapshots=True,
    snapshot_frequency=100
)
event_store = InMemoryEventStore(config)

# Create repository
repository = EventSourcedRepository(event_store, DataProcessingWorkflowAggregate)

# Save aggregate (stores events)
workflow = DataProcessingWorkflowAggregate("workflow_001")
workflow.start_processing("dataset_123", "full_pipeline")
await repository.save(workflow)

# Load aggregate (replay events)
loaded_workflow = await repository.get_by_id("workflow_001")
```

### Event Store Benefits

- **Complete Audit Trail**: Every state change is recorded
- **Temporal Queries**: Query state at any point in time
- **Event Replay**: Rebuild state from events
- **Debugging**: Understand exactly what happened
- **Analytics**: Analyze business processes over time

## 🔗 Saga Pattern

### Core Concepts

Sagas manage distributed transactions across multiple services:

```python
from interfaces.advanced_patterns import SagaStep, SagaOrchestrator, SagaState

class DataProcessingSaga:
    async def start_data_processing_pipeline(self, dataset_id: str, config: Dict[str, Any]):
        saga_id = f"data_pipeline_{dataset_id}"
        
        # Define steps with compensation
        steps = [
            SagaStep(
                step_id="quality_check",
                command=ProcessDataQualityCommand(dataset_id=dataset_id),
                compensation_command=RollbackQualityCheckCommand(dataset_id=dataset_id),
                timeout_seconds=60
            ),
            SagaStep(
                step_id="anomaly_detection",
                command=RunAnomalyDetectionCommand(dataset_id=dataset_id),
                compensation_command=ClearAnomalyResultsCommand(dataset_id=dataset_id),
                timeout_seconds=120
            )
        ]
        
        return await self.saga_orchestrator.start_saga(saga_id, steps)
```

### Saga Orchestration

```python
from shared.advanced_infrastructure import create_saga_orchestrator

# Create saga orchestrator
saga_orchestrator = create_saga_orchestrator(command_bus, event_bus)

# Start saga
saga_state = await saga_orchestrator.start_saga("saga_001", steps)

# Handle step completion
await saga_orchestrator.handle_step_completion(
    saga_id="saga_001",
    step_id="quality_check",
    result=command_response
)

# Handle step failure (triggers compensation)
await saga_orchestrator.handle_step_failure(
    saga_id="saga_001",
    step_id="anomaly_detection",
    error="Detection service unavailable"
)
```

### Saga Benefits

- **Distributed Transaction Management**: Coordinate across services
- **Automatic Compensation**: Rollback on failure
- **Long-Running Processes**: Handle workflows that take time
- **Fault Tolerance**: Recover from partial failures
- **Workflow Visibility**: Track progress and status

## 📊 Read Models & Projections

### Core Concepts

Read models provide optimized views for queries:

```python
from interfaces.advanced_patterns import ReadModel, ProjectionManager

class DataProcessingDashboardReadModel(ReadModel):
    def __init__(self):
        self.dashboard_data = {
            "total_datasets_processed": 0,
            "average_quality_score": 0.0,
            "recent_activities": [],
            "processing_trends": []
        }
    
    async def handle_event(self, event: DomainEvent):
        """Update read model based on domain events"""
        if isinstance(event, DataQualityCheckCompleted):
            self.dashboard_data["total_datasets_processed"] += 1
            self._update_quality_average(event.overall_score)
            self._add_recent_activity(f"Quality check completed: {event.dataset_id}")
    
    def get_supported_events(self) -> List[type]:
        return [DataQualityCheckCompleted, AnomalyDetected, DataProcessingStarted]
```

### Projection Management

```python
from shared.advanced_infrastructure import InMemoryProjectionManager

# Create projection manager
projection_manager = InMemoryProjectionManager(event_store, event_bus)

# Register read models
dashboard_read_model = DataProcessingDashboardReadModel()
await projection_manager.register_projection(dashboard_read_model)

# Rebuild projection from event history
await projection_manager.rebuild_projection(DataProcessingDashboardReadModel)

# Get projection status
status = await projection_manager.get_projection_status(DataProcessingDashboardReadModel)
```

## 🎭 Integration Patterns

### CQRS + Event Sourcing

Combine CQRS with event sourcing for powerful data management:

```python
class EventSourcedCommandHandler:
    def __init__(self, repository: Repository[DataProcessingWorkflowAggregate]):
        self.repository = repository
    
    async def handle(self, command: ProcessDataQualityCommand) -> CommandResponse:
        # Load aggregate from event store
        workflow = await self.repository.get_by_id(command.dataset_id)
        if not workflow:
            workflow = DataProcessingWorkflowAggregate(command.dataset_id)
        
        # Execute business logic
        workflow.request_quality_check(command.quality_rules)
        
        # Save aggregate (stores new events)
        await self.repository.save(workflow)
        
        return CommandResponse(
            command_id=command.command_id,
            result=CommandResult.SUCCESS,
            events=workflow.get_uncommitted_events()
        )
```

### Saga + CQRS

Use sagas to orchestrate CQRS commands:

```python
class CQRSSagaOrchestrator:
    async def execute_saga_step(self, step: SagaStep):
        # Execute command through command bus
        response = await self.command_bus.send(step.command)
        
        if response.result == CommandResult.SUCCESS:
            await self.handle_step_completion(step.saga_id, step.step_id, response)
        else:
            await self.handle_step_failure(step.saga_id, step.step_id, response.error_message)
```

### Complete Pipeline Example

```python
class AdvancedDataProcessingPipeline:
    """Complete pipeline using all advanced patterns"""
    
    async def process_dataset(self, dataset_id: str, processing_config: Dict[str, Any]):
        # 1. Start with CQRS command
        command = ProcessDatasetCommand(
            dataset_id=dataset_id,
            config=processing_config
        )
        
        # 2. Command handler uses event-sourced aggregate
        workflow = await self.load_or_create_workflow(dataset_id)
        workflow.start_processing(processing_config)
        await self.repository.save(workflow)
        
        # 3. Saga orchestrates the multi-step process
        saga_steps = self.create_processing_steps(dataset_id, processing_config)
        saga_state = await self.saga_orchestrator.start_saga(f"pipeline_{dataset_id}", saga_steps)
        
        # 4. Events update read models for dashboards
        # (Handled automatically by projection manager)
        
        return {
            "workflow_id": workflow.aggregate_id,
            "saga_id": saga_state.saga_id,
            "status": "started"
        }
```

## ⚙️ Configuration

### CQRS Configuration

```python
config = CQRSConfiguration(
    enable_command_validation=True,      # Validate commands before execution
    enable_query_caching=True,           # Cache query results
    command_timeout_seconds=30,          # Command execution timeout
    query_timeout_seconds=10,            # Query execution timeout
    max_retry_attempts=3,                # Retry failed operations
    enable_distributed_tracing=True,     # Add tracing headers
    enable_command_batching=False,       # Batch commands for performance
    batch_size=100,                      # Commands per batch
    batch_timeout_ms=1000               # Batch timeout
)
```

### Event Sourcing Configuration

```python
config = EventSourcingConfiguration(
    event_store_type="memory",           # memory, file, database
    enable_snapshots=True,               # Create aggregate snapshots
    snapshot_frequency=100,              # Snapshot every N events
    compression_enabled=False,           # Compress event data
    encryption_enabled=False,            # Encrypt sensitive events
    batch_size=1000,                     # Events per batch
    consistency_level="strong",          # strong, eventual, session
    conflict_resolution="last_write_wins" # Conflict resolution strategy
)
```

## 📈 Monitoring & Observability

### Advanced Metrics

Each pattern provides detailed metrics:

```python
# CQRS metrics
command_metrics = command_bus.get_metrics()
query_metrics = query_bus.get_metrics()

# Event sourcing metrics
event_store_metrics = event_store.get_metrics()
repository_metrics = repository.get_metrics()

# Saga metrics
saga_metrics = saga_orchestrator.get_metrics()

# Projection metrics
projection_metrics = projection_manager.get_metrics()
```

### Performance Monitoring

```python
from shared.observability import get_metrics_collector

metrics = get_metrics_collector()

# Track advanced pattern usage
metrics.increment_counter("cqrs.commands_processed", tags={"command_type": "ProcessDataQuality"})
metrics.record_histogram("event_sourcing.aggregate_load_time_ms", 150.5)
metrics.set_gauge("saga.active_sagas", 42)
metrics.increment_counter("projections.events_processed", tags={"projection": "Dashboard"})
```

## 🚀 Getting Started

### 1. Basic CQRS Setup

```python
# Define commands and queries
@dataclass
class MyCommand(Command):
    entity_id: str
    data: Dict[str, Any]

@dataclass  
class MyQuery(Query):
    entity_id: str
    filters: Dict[str, Any]

# Create handlers
class MyCommandHandler:
    async def handle(self, command: MyCommand) -> CommandResponse:
        # Execute business logic
        return CommandResponse(command_id=command.command_id, result=CommandResult.SUCCESS)

class MyQueryHandler:
    async def handle(self, query: MyQuery) -> QueryResponse:
        # Read data
        return QueryResponse(query_id=query.query_id, data={"result": "data"})

# Setup infrastructure
config = CQRSConfiguration()
command_bus, query_bus = create_cqrs_infrastructure(config, container)

# Register and use
command_bus.register_handler(MyCommand, MyCommandHandler().handle)
query_bus.register_handler(MyQuery, MyQueryHandler().handle)

response = await command_bus.send(MyCommand(entity_id="123", data={}))
result = await query_bus.ask(MyQuery(entity_id="123", filters={}))
```

### 2. Event Sourcing Setup

```python
# Define aggregate
class MyAggregate(Aggregate):
    def do_something(self, data: Dict[str, Any]):
        event = SomethingHappened(
            aggregate_id=self.aggregate_id,
            data=data
        )
        self.raise_event(event)
    
    def _when(self, event: DomainEvent):
        if isinstance(event, SomethingHappened):
            # Update state based on event
            pass

# Setup infrastructure
config = EventSourcingConfiguration()
event_store, projection_manager = create_event_sourcing_infrastructure(config)
repository = EventSourcedRepository(event_store, MyAggregate)

# Use aggregate
aggregate = MyAggregate("agg_001")
aggregate.do_something({"key": "value"})
await repository.save(aggregate)
```

### 3. Saga Setup

```python
# Define saga steps
steps = [
    SagaStep(
        step_id="step1",
        command=FirstCommand(),
        compensation_command=UndoFirstCommand(),
        timeout_seconds=30
    ),
    SagaStep(
        step_id="step2", 
        command=SecondCommand(),
        compensation_command=UndoSecondCommand(),
        timeout_seconds=60
    )
]

# Start saga
saga_orchestrator = create_saga_orchestrator(command_bus, event_bus)
saga_state = await saga_orchestrator.start_saga("my_saga", steps)
```

## 🎯 Best Practices

### CQRS Best Practices

- **Separate Models**: Use different models for commands and queries
- **Eventual Consistency**: Accept that reads may be slightly behind writes
- **Command Validation**: Validate commands thoroughly before execution
- **Query Optimization**: Optimize read models for specific query patterns
- **Versioning**: Plan for command and query schema evolution

### Event Sourcing Best Practices

- **Event Immutability**: Never modify events once stored
- **Event Versioning**: Plan for event schema changes
- **Snapshot Strategy**: Use snapshots for performance with large event streams
- **Event Naming**: Use descriptive, business-meaningful event names
- **Replay Safety**: Ensure event handlers are idempotent

### Saga Best Practices

- **Compensation Logic**: Design robust compensation for each step
- **Timeout Handling**: Set appropriate timeouts for each step
- **Idempotency**: Ensure saga steps can be safely retried
- **State Management**: Track saga state for monitoring and debugging
- **Error Handling**: Plan for various failure scenarios

## 🔧 Troubleshooting

### Common Issues

**CQRS Command Timeout**
```python
# Increase timeout or optimize handler
config.command_timeout_seconds = 60

# Add retry logic
config.max_retry_attempts = 5
```

**Event Sourcing Memory Usage**
```python
# Enable snapshots
config.enable_snapshots = True
config.snapshot_frequency = 50

# Use external event store
config.event_store_type = "database"
```

**Saga Stuck in Compensation**
```python
# Check saga state
saga_state = await saga_orchestrator.get_saga_state(saga_id)
print(f"Status: {saga_state.status}")
print(f"Failed step: {saga_state.failed_step}")
print(f"Error: {saga_state.error_message}")

# Manual intervention might be needed
```

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger("shared.advanced_infrastructure").setLevel(logging.DEBUG)

# Monitor metrics
metrics = get_advanced_infrastructure_metrics()
print(f"Metrics: {json.dumps(metrics, indent=2)}")

# Check component health
health_status = await get_health_check().check_health()
print(f"Health: {health_status}")
```

## 📚 Additional Resources

- **Advanced Patterns Examples**: [`configurations/examples/advanced_patterns_examples.py`](./configurations/examples/advanced_patterns_examples.py)
- **Infrastructure Implementation**: [`shared/src/shared/advanced_infrastructure.py`](./shared/src/shared/advanced_infrastructure.py)
- **Interface Definitions**: [`interfaces/src/interfaces/advanced_patterns.py`](./interfaces/src/interfaces/advanced_patterns.py)
- **Basic Framework Guide**: [`DEVELOPER_ONBOARDING.md`](./DEVELOPER_ONBOARDING.md)

---

The advanced patterns provide powerful tools for building sophisticated, scalable, and maintainable systems. Start with basic CQRS and gradually adopt event sourcing and sagas as your use cases become more complex.

Remember: **Advanced patterns add complexity** - only use them when the benefits clearly outweigh the costs for your specific use case.