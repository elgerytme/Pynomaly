# A-002 Batch Processing Orchestration - Sequence Diagrams

## Overview

This document contains sequence diagrams showing the interactions between `PipelineOrchestrationService`, `BatchProcessingUseCases`, and `BatchProcessor` for the A-002 batch processing orchestration feature.

## Sequence Diagram 1: Batch Processing Job Submission

```mermaid
sequenceDiagram
    participant Client as Client/API
    participant BPU as BatchProcessingUseCases
    participant POS as PipelineOrchestrationService
    participant PO as ProcessingOrchestrator
    participant BP as BatchProcessor
    participant Repo as Repository

    Client->>BPU: createBatchProcessingJob(config)
    BPU->>POS: create_pipeline(name, type=BATCH)
    POS->>Repo: save(pipeline)
    Repo-->>POS: pipeline_id
    POS-->>BPU: pipeline
    
    BPU->>POS: add_pipeline_step(BATCH_PROCESSING)
    POS->>Repo: save(pipeline)
    POS-->>BPU: pipeline_step
    
    BPU->>POS: validate_and_activate_pipeline(pipeline_id)
    POS->>Repo: save(pipeline)
    POS-->>BPU: activated_pipeline
    
    BPU->>PO: start_batch_session(config)
    PO->>BP: create_batch_processor(config)
    BP-->>PO: batch_processor
    PO->>BP: submit_job(name, input_path, output_path)
    BP-->>PO: job_id
    PO-->>BPU: session_id
    
    BPU-->>Client: BatchProcessingJobResponse(job_id, session_id)
```

## Sequence Diagram 2: Batch Processing Execution

```mermaid
sequenceDiagram
    participant BP as BatchProcessor
    participant Engine as Processing Engine
    participant DS as DetectionService
    participant Storage as Storage System
    participant Monitor as Monitoring Service

    BP->>BP: _process_job(job)
    BP->>BP: initialize_engine()
    BP->>Storage: _load_and_chunk_data(job)
    Storage-->>BP: chunks[]
    
    loop For each chunk
        BP->>Engine: process_chunk(chunk)
        Engine->>DS: detect_anomalies(dataset)
        DS-->>Engine: detection_result
        Engine-->>BP: chunk_result
        
        BP->>Monitor: update_progress(job_id, progress)
        
        alt Checkpoint frequency reached
            BP->>Storage: _save_checkpoint(job, partial_results)
        end
    end
    
    BP->>BP: _combine_results(job, results)
    BP->>Storage: _save_result(job, combined_result)
    BP->>Monitor: job_completed(job_id)
```

## Sequence Diagram 3: Batch Processing Monitoring

```mermaid
sequenceDiagram
    participant Client as Client/API
    participant BPU as BatchProcessingUseCases
    participant PO as ProcessingOrchestrator
    participant BP as BatchProcessor
    participant Monitor as Monitoring Service

    Client->>BPU: getBatchProcessingStatus(job_id)
    BPU->>PO: get_session_status(session_id)
    PO->>BP: get_job_status(job_id)
    BP-->>PO: job_status
    PO-->>BPU: session_status
    
    BPU->>Monitor: get_metrics(job_id)
    Monitor-->>BPU: metrics
    
    BPU->>BPU: aggregate_status(session_status, metrics)
    BPU-->>Client: BatchProcessingStatusResponse
    
    Note over Client,Monitor: Real-time updates via WebSocket/SSE
    
    loop Every 5 seconds
        Monitor->>BP: get_job_status(job_id)
        BP-->>Monitor: current_status
        Monitor->>Client: status_update(job_id, status)
    end
```

## Sequence Diagram 4: Batch Processing Pipeline Integration

```mermaid
sequenceDiagram
    participant Client as Client/API
    participant BPU as BatchProcessingUseCases
    participant POS as PipelineOrchestrationService
    participant PO as ProcessingOrchestrator
    participant BP as BatchProcessor
    participant Scheduler as Scheduler

    Client->>BPU: createScheduledBatchPipeline(config)
    BPU->>POS: create_pipeline(name, type=BATCH)
    POS-->>BPU: pipeline
    
    BPU->>POS: add_pipeline_step(BATCH_PROCESSING)
    BPU->>POS: add_pipeline_step(RESULT_VALIDATION)
    BPU->>POS: add_pipeline_step(NOTIFICATION)
    
    BPU->>POS: set_pipeline_schedule(pipeline_id, cron_expression)
    POS-->>BPU: scheduled_pipeline
    
    BPU->>POS: validate_and_activate_pipeline(pipeline_id)
    POS-->>BPU: activated_pipeline
    
    Note over Scheduler: Scheduled time reached
    
    Scheduler->>POS: trigger_pipeline_run(pipeline_id)
    POS->>PO: start_batch_session(config)
    PO->>BP: submit_job(name, input_path, output_path)
    BP-->>PO: job_id
    PO-->>POS: session_id
    
    POS->>POS: track_pipeline_run(run_id, session_id)
    
    Note over BP: Job execution completes
    
    BP->>PO: job_completed(job_id)
    PO->>POS: pipeline_step_completed(run_id, step_id)
    
    POS->>POS: execute_next_step(run_id)
    POS->>POS: complete_pipeline_run(run_id)
```

## Sequence Diagram 5: Batch Processing Error Handling

```mermaid
sequenceDiagram
    participant BP as BatchProcessor
    participant Engine as Processing Engine
    participant DS as DetectionService
    participant Storage as Storage System
    participant Alert as Alert Service
    participant Recovery as Recovery Service

    BP->>Engine: process_chunk(chunk)
    Engine->>DS: detect_anomalies(dataset)
    DS-->>Engine: ERROR: OutOfMemoryError
    Engine-->>BP: ChunkProcessingError
    
    BP->>BP: handle_chunk_error(chunk, error)
    BP->>Storage: save_failed_chunk(chunk, error)
    
    alt Retry count < max_retries
        BP->>BP: increment_retry_count(job)
        BP->>BP: wait(retry_delay)
        BP->>Engine: process_chunk(chunk) [retry]
    else
        BP->>BP: mark_job_failed(job)
        BP->>Alert: send_alert(job_id, error_details)
        BP->>Recovery: initiate_recovery(job_id)
        
        Recovery->>Storage: load_checkpoint(job_id)
        Storage-->>Recovery: checkpoint_data
        Recovery->>BP: resume_from_checkpoint(job_id, checkpoint_data)
        
        alt Recovery successful
            BP->>BP: continue_processing(job)
        else
            BP->>BP: fail_job_permanently(job)
            BP->>Alert: send_failure_notification(job_id)
        end
    end
```

## Sequence Diagram 6: Batch Processing Resource Management

```mermaid
sequenceDiagram
    participant BPU as BatchProcessingUseCases
    participant RM as ResourceManager
    participant PO as ProcessingOrchestrator
    participant BP as BatchProcessor
    participant Monitor as ResourceMonitor

    BPU->>RM: request_resources(job_requirements)
    RM->>Monitor: get_available_resources()
    Monitor-->>RM: resource_status
    
    RM->>RM: calculate_allocation(requirements, availability)
    
    alt Resources available
        RM->>RM: allocate_resources(job_id, allocation)
        RM-->>BPU: resource_allocation
        
        BPU->>PO: start_batch_session(config, allocation)
        PO->>BP: create_batch_processor(config)
        BP->>BP: configure_engine(allocation)
        BP-->>PO: configured_processor
        PO-->>BPU: session_id
        
        Note over BP: Job processing starts
        
        loop During processing
            Monitor->>BP: get_resource_usage()
            BP-->>Monitor: current_usage
            Monitor->>RM: update_usage(job_id, usage)
            
            alt Usage > threshold
                RM->>BP: scale_resources(new_allocation)
                BP->>BP: adjust_workers(new_allocation)
            end
        end
        
        Note over BP: Job completes
        
        BP->>RM: release_resources(job_id)
        RM->>Monitor: deallocate_resources(job_id)
        
    else
        RM->>BPU: queue_job(job_id, requirements)
        BPU-->>Client: JobQueuedResponse(estimated_wait_time)
        
        Note over RM: Wait for resources
        
        RM->>BPU: resources_available(job_id)
        BPU->>PO: start_batch_session(config, allocation)
    end
```

## Component Interactions Summary

### Key Interaction Patterns

1. **Request Flow**: Client → BatchProcessingUseCases → PipelineOrchestrationService → ProcessingOrchestrator → BatchProcessor
2. **Monitoring Flow**: BatchProcessor → Monitor → Client (via WebSocket/SSE)
3. **Error Handling**: BatchProcessor → Recovery Service → Alert Service
4. **Resource Management**: ResourceManager ↔ BatchProcessor ↔ ResourceMonitor

### Data Flow

1. **Input Data**: Client/Storage → BatchProcessor → ProcessingEngine
2. **Processing Results**: ProcessingEngine → DetectionService → BatchProcessor
3. **Output Data**: BatchProcessor → Storage → Client
4. **Monitoring Data**: BatchProcessor → Monitor → Client

### Control Flow

1. **Job Lifecycle**: Created → Queued → Running → Completed/Failed
2. **Resource Lifecycle**: Requested → Allocated → Used → Released
3. **Error Handling**: Detected → Logged → Retried → Recovered/Failed

## Architecture Benefits

### Clean Separation of Concerns
- **Use Cases**: Handle business logic and orchestration
- **Services**: Manage specific domain operations
- **Infrastructure**: Handle technical implementation details

### Scalability
- **Horizontal Scaling**: Multiple BatchProcessor instances
- **Vertical Scaling**: Dynamic resource allocation
- **Load Balancing**: Intelligent job distribution

### Fault Tolerance
- **Checkpointing**: Save intermediate results
- **Retry Logic**: Automatic error recovery
- **Graceful Degradation**: Partial result preservation

### Monitoring and Observability
- **Real-time Status**: Live job progress updates
- **Metrics Collection**: Performance and resource usage
- **Alerting**: Proactive issue notification

---

*Document Version: 1.0*
*Last Updated: January 8, 2025*
*Status: Draft - Part of A-002 Design Documentation*
