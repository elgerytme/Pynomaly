# Batch Processing Orchestration

A comprehensive batch processing system for handling large datasets efficiently with configurable batch sizes, progress tracking, error handling, and recovery capabilities.

## Overview

The Batch Processing Orchestration system provides:

- **Configurable Batch Processing**: Automatically optimizes batch sizes based on data characteristics and system resources
- **Job Orchestration**: Manages job dependencies, priorities, and concurrent execution
- **Progress Tracking**: Real-time progress monitoring with customizable callbacks
- **Error Handling & Recovery**: Comprehensive failure detection, classification, and recovery strategies
- **System Monitoring**: Resource monitoring with alerts and performance metrics
- **Checkpoint & Resume**: Fault tolerance with checkpoint-based recovery

## Core Components

### 1. BatchProcessingService

The foundation service that handles individual batch processing jobs.

```python
from pynomaly.application.services import BatchProcessingService, BatchConfig

service = BatchProcessingService()

# Register a custom processor
async def my_processor(batch_data, context):
    # Process the batch
    return {"processed": len(batch_data)}

service.register_processor("my_processor", my_processor)

# Create and start a job
job = await service.create_batch_job(
    name="My Batch Job",
    data=my_dataframe,
    processor_name="my_processor",
    config=BatchConfig(batch_size=100, max_concurrent_batches=4)
)

await service.start_batch_job(job.id)
```

### 2. BatchOrchestrator

High-level orchestration of multiple batch jobs with dependency management.

```python
from pynomaly.application.services import BatchOrchestrator, BatchJobRequest, BatchPriority

orchestrator = BatchOrchestrator()

# Submit jobs with dependencies
job_a = await orchestrator.submit_job(BatchJobRequest(
    name="Data Profiling",
    processor_type="data_profiling",
    data_source=data,
    priority=BatchPriority.HIGH
))

job_b = await orchestrator.submit_job(BatchJobRequest(
    name="Anomaly Detection", 
    processor_type="anomaly_detection",
    data_source=data,
    depends_on=[job_a],  # Wait for job_a to complete
    priority=BatchPriority.MEDIUM
))
```

### 3. BatchConfigurationManager

Intelligent configuration optimization based on data and system characteristics.

```python
from pynomaly.application.services import BatchConfigurationManager

config_manager = BatchConfigurationManager()

# Get optimized configuration
result = config_manager.calculate_optimal_batch_config(
    data=my_dataframe,
    processor_name="anomaly_detection",
    target_memory_usage_mb=1000
)

print(f"Recommended batch size: {result.recommended_batch_size}")
print(f"Recommended concurrency: {result.recommended_concurrency}")

# Create optimized config
config = config_manager.create_optimized_config(
    data=my_dataframe,
    processor_name="anomaly_detection"
)
```

### 4. BatchMonitoringService

Real-time monitoring and alerting for batch processing operations.

```python
from pynomaly.application.services import BatchMonitoringService

monitoring = BatchMonitoringService()

# Setup callbacks
def progress_callback(event):
    print(f"Progress: {event.progress_percentage}%")

def alert_callback(alert):
    print(f"Alert: {alert.title} - {alert.message}")

monitoring.add_progress_callback(progress_callback)
monitoring.add_alert_callback(alert_callback)

# Start monitoring
await monitoring.start_monitoring()

# Get dashboard data
dashboard = monitoring.get_monitoring_dashboard_data()
```

### 5. BatchRecoveryService

Comprehensive error handling and recovery mechanisms.

```python
from pynomaly.application.services import BatchRecoveryService, RecoveryConfig

recovery = BatchRecoveryService(monitoring_service)

# Configure recovery strategies
config = RecoveryConfig(
    max_retry_attempts=3,
    enable_checkpointing=True,
    checkpoint_interval_batches=10
)
recovery.configure_recovery(config)

# Handle failures automatically
success = await recovery.handle_failure(job, error, batch_index, context)
```

## Standard Processors

The system includes several built-in processors:

### Anomaly Detection
```python
# Built-in anomaly detection processor
job_request = BatchJobRequest(
    name="Anomaly Detection",
    processor_type="anomaly_detection",
    data_source=dataframe,
    processor_kwargs={"algorithm": "isolation_forest"}
)
```

### Data Quality Assessment
```python
# Built-in data quality processor
job_request = BatchJobRequest(
    name="Data Quality Check",
    processor_type="data_quality", 
    data_source=dataframe
)
```

### Data Profiling
```python
# Built-in data profiling processor
job_request = BatchJobRequest(
    name="Data Profiling",
    processor_type="data_profiling",
    data_source=dataframe
)
```

### Feature Engineering
```python
# Built-in feature engineering processor
job_request = BatchJobRequest(
    name="Feature Engineering",
    processor_type="feature_engineering",
    data_source=dataframe
)
```

## Configuration Options

### BatchConfig

Core configuration for batch processing:

```python
from pynomaly.application.services import BatchConfig

config = BatchConfig(
    batch_size=1000,                    # Items per batch
    max_concurrent_batches=4,           # Concurrent batch limit
    memory_limit_mb=1000.0,             # Memory limit
    timeout_seconds=3600,               # Job timeout
    retry_attempts=3,                   # Retry attempts
    retry_delay_seconds=30.0,           # Delay between retries
    checkpoint_interval=10,             # Batches between checkpoints
    enable_progress_tracking=True,      # Enable progress tracking
    preserve_order=False,               # Preserve batch order
    auto_optimize=True                  # Auto-optimize batch sizes
)
```

### Processing Profiles

Customize processing characteristics:

```python
from pynomaly.application.services import ProcessingProfile

profile = ProcessingProfile(
    processor_name="custom_processor",
    cpu_intensive=True,
    memory_intensive=False,
    io_intensive=False,
    requires_order=False,
    estimated_processing_time_per_row_ms=2.0,
    memory_overhead_factor=1.5
)

config_manager.register_processing_profile(profile)
```

## Job Dependencies

Create complex workflows with job dependencies:

```python
# Create dependency chain
job_a = await orchestrator.submit_job(BatchJobRequest(
    name="Step 1: Data Validation",
    processor_type="data_quality",
    data_source=data
))

job_b = await orchestrator.submit_job(BatchJobRequest(
    name="Step 2: Feature Engineering", 
    processor_type="feature_engineering",
    data_source=data,
    depends_on=[job_a]
))

job_c = await orchestrator.submit_job(BatchJobRequest(
    name="Step 3: Anomaly Detection",
    processor_type="anomaly_detection", 
    data_source=data,
    depends_on=[job_b]
))

# Jobs will execute in sequence: A → B → C
```

## Progress Tracking

Monitor job progress in real-time:

```python
def track_progress(event):
    if event.event_type == "progress":
        print(f"Job {event.job_id}: {event.progress_percentage:.1f}% complete")
        print(f"Message: {event.message}")
    elif event.event_type == "completed":
        print(f"Job {event.job_id} completed successfully!")
    elif event.event_type == "failed":
        print(f"Job {event.job_id} failed: {event.message}")

monitoring.add_progress_callback(track_progress)

# Get job status
status = orchestrator.get_job_status(job_id)
print(f"Progress: {status.get('progress_percentage', 0):.1f}%")
print(f"Items processed: {status.get('processed_items', 0)}")
```

## Error Handling

The system automatically handles various failure types:

- **Processing Errors**: Retry with exponential backoff
- **Memory Errors**: Restart from checkpoint with smaller batches
- **Timeout Errors**: Resume from last checkpoint
- **Network Errors**: Retry with jitter
- **Data Corruption**: Skip failed batches
- **System Errors**: Restart from checkpoint

```python
# Get failure statistics
stats = recovery.get_failure_statistics(hours=24)
print(f"Total failures: {stats['total_failures']}")
print(f"Recovery success rate: {stats['recovery_success_rate']:.1f}%")
print(f"Most common failure: {stats['most_common_failure']}")
```

## Monitoring & Alerts

Comprehensive system monitoring:

```python
# Get system status
status = orchestrator.get_system_status()
print(f"Running jobs: {status['running_jobs']}")
print(f"Scheduled jobs: {status['scheduled_jobs']}")

# Get monitoring dashboard
dashboard = monitoring.get_monitoring_dashboard_data()
print(f"CPU usage: {dashboard['system_summary']['cpu_percent']:.1f}%")
print(f"Memory usage: {dashboard['system_summary']['memory_percent']:.1f}%")

# Get active alerts
alerts = monitoring.get_active_alerts()
for alert in alerts:
    print(f"{alert.level}: {alert.title} - {alert.message}")
```

## Performance Optimization

### Automatic Optimization

The system automatically optimizes batch configurations:

```python
# System considers:
# - Available memory and CPU
# - Data characteristics (size, complexity)
# - Processing requirements (CPU/memory intensive)
# - Current system load

result = config_manager.calculate_optimal_batch_config(
    data=large_dataset,
    processor_name="anomaly_detection"
)

# Provides optimized recommendations
print(f"Batch size: {result.recommended_batch_size}")
print(f"Concurrency: {result.recommended_concurrency}")
print(f"Memory usage: {result.estimated_memory_usage_mb:.1f}MB")
```

### Manual Tuning

Override automatic optimization:

```python
job_request = BatchJobRequest(
    name="Custom Configuration",
    processor_type="anomaly_detection",
    data_source=data,
    config_overrides={
        "batch_size": 500,              # Override batch size
        "max_concurrent_batches": 2,    # Override concurrency
        "memory_limit_mb": 2000         # Override memory limit
    }
)
```

## Best Practices

### 1. Choose Appropriate Batch Sizes
```python
# For memory-intensive operations
config = BatchConfig(batch_size=100, memory_limit_mb=1000)

# For CPU-intensive operations  
config = BatchConfig(batch_size=1000, max_concurrent_batches=2)

# For I/O-intensive operations
config = BatchConfig(batch_size=500, max_concurrent_batches=8)
```

### 2. Use Job Dependencies Wisely
```python
# Good: Logical workflow dependencies
data_validation → feature_engineering → model_training

# Avoid: Unnecessary sequential processing
parallel_task_a, parallel_task_b, parallel_task_c → final_aggregation
```

### 3. Monitor Resource Usage
```python
# Set up monitoring before starting large jobs
await monitoring.start_monitoring()

# Monitor system recommendations
recommendations = config_manager.get_system_recommendations()
for rec in recommendations['recommendations']:
    print(f"{rec['type']}: {rec['message']}")
```

### 4. Handle Failures Gracefully
```python
# Configure appropriate recovery strategies
config = RecoveryConfig(
    max_retry_attempts=3,
    enable_checkpointing=True,
    checkpoint_interval_batches=5,  # More frequent for critical jobs
    preserve_partial_results=True
)
```

### 5. Clean Up Resources
```python
# Regular cleanup of completed jobs
cleaned = await orchestrator.cleanup_old_jobs(hours=24)

# Clean up monitoring data
cleaned_records = await recovery.cleanup_old_records(days=7)
```

## Integration Examples

### With Existing Pynomaly Services

```python
# Integrate with detection service
from pynomaly.application.services import DetectionService

detection_service = DetectionService()

async def anomaly_detection_processor(batch_data, context):
    """Custom processor using DetectionService."""
    results = []
    for _, row in batch_data.iterrows():
        result = await detection_service.detect_anomalies(row.to_dict())
        results.append(result)
    return {"anomalies": results}

# Register and use
orchestrator.batch_service.register_processor(
    "detection_service_processor", 
    anomaly_detection_processor
)
```

### With Message Queue Integration

```python
# Use with existing message queue
from pynomaly.infrastructure.messaging import QueueManager

queue_manager = QueueManager(adapter, settings)
orchestrator = BatchOrchestrator(queue_manager=queue_manager)

# Jobs can be distributed across workers
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```python
   # Reduce batch size
   config_overrides={"batch_size": 50, "memory_limit_mb": 500}
   ```

2. **Slow Processing**
   ```python
   # Increase concurrency
   config_overrides={"max_concurrent_batches": 8}
   ```

3. **Job Failures**
   ```python
   # Check failure history
   failures = recovery.get_job_failure_history(job_id)
   for failure in failures:
       print(f"Failure: {failure.failure_type} - {failure.error_message}")
   ```

4. **High Resource Usage**
   ```python
   # Get system recommendations
   recommendations = config_manager.get_system_recommendations()
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('pynomaly.application.services').setLevel(logging.DEBUG)
```

## API Reference

See the individual service documentation for detailed API reference:

- `BatchProcessingService`: Core batch processing functionality
- `BatchOrchestrator`: High-level job orchestration
- `BatchConfigurationManager`: Configuration optimization
- `BatchMonitoringService`: Real-time monitoring and alerts
- `BatchRecoveryService`: Error handling and recovery

## Example Applications

Complete examples are available in:
- `examples/batch_processing_example.py`: Comprehensive usage examples
- `tests/application/services/test_batch_*.py`: Unit and integration tests

The batch processing orchestration system provides a robust, scalable solution for handling large dataset processing in Pynomaly with enterprise-grade reliability and monitoring capabilities.