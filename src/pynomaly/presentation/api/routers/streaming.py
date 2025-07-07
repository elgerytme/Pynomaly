"""
FastAPI router for streaming data processing endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pynomaly.infrastructure.streaming.streaming_processor import (
    StreamingProcessor, StreamingService, StreamRecord, StreamMetrics, 
    StreamState, WindowType
)
from pynomaly.infrastructure.streaming.connectors import (
    ConnectorFactory, ConnectorConfig, ConnectorType
)
from pynomaly.domain.entities.user import User
from pynomaly.domain.entities.detector import Detector
from pynomaly.shared.exceptions import StreamingError, ValidationError
from pynomaly.shared.types import TenantId, UserId
import json
import asyncio

# Router setup
router = APIRouter(prefix="/api/streaming", tags=["Streaming"])

# Global streaming service instance
streaming_service = StreamingService()

# Request/Response Models
class CreateProcessorRequest(BaseModel):
    """Request to create a streaming processor."""
    processor_id: str = Field(..., description="Unique processor identifier")
    detector_id: str = Field(..., description="Detector to use for anomaly detection")
    window_config: Dict[str, Any] = Field(
        default={
            "type": "tumbling",
            "size_seconds": 60
        },
        description="Window configuration"
    )
    backpressure_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backpressure handling configuration"
    )
    buffer_size: int = Field(default=1000, description="Internal buffer size")
    max_batch_size: int = Field(default=100, description="Maximum batch size")
    processing_timeout: float = Field(default=30.0, description="Processing timeout in seconds")


class ProcessorResponse(BaseModel):
    """Response for processor information."""
    processor_id: str
    state: StreamState
    detector_name: str
    window_config: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime


class StreamMetricsResponse(BaseModel):
    """Response for streaming metrics."""
    processor_id: str
    total_processed: int
    anomalies_detected: int
    processing_rate: float
    avg_latency_ms: float
    error_count: int
    last_processed: Optional[datetime]
    backpressure_events: int
    dropped_records: int


class ConnectorConfigRequest(BaseModel):
    """Request to configure streaming connector."""
    connector_type: ConnectorType
    connection_params: Dict[str, Any]
    serialization_format: str = Field(default="json", description="Data serialization format")
    batch_size: int = Field(default=100, description="Batch size for consumption")
    max_wait_time: float = Field(default=1.0, description="Maximum wait time in seconds")


class StreamRecordRequest(BaseModel):
    """Request to send a single stream record."""
    data: Dict[str, Any]
    tenant_id: str
    metadata: Optional[Dict[str, Any]] = None


class BatchStreamRequest(BaseModel):
    """Request to send batch of stream records."""
    records: List[StreamRecordRequest]


class AnomalyAlert(BaseModel):
    """Anomaly alert for real-time notifications."""
    processor_id: str
    anomaly_id: str
    score: float
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]


# Dependencies
async def get_current_user() -> User:
    """Get current authenticated user."""
    # TODO: Implement authentication
    pass


async def require_streaming_access(tenant_id: UUID, current_user: User = Depends(get_current_user)):
    """Require streaming access to specific tenant."""
    if not (current_user.is_super_admin() or 
            current_user.has_role_in_tenant(TenantId(str(tenant_id)), ["tenant_admin", "data_scientist"])):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied - requires streaming permissions"
        )
    return current_user


async def get_detector_by_id(detector_id: str) -> Detector:
    """Get detector by ID."""
    # TODO: Implement detector retrieval from repository
    # For now, return a mock detector
    from pynomaly.domain.entities.detector import Detector
    from pynomaly.shared.types import DetectorId
    
    return Detector(
        id=DetectorId(detector_id),
        name=f"Detector_{detector_id}",
        algorithm="isolation_forest",
        parameters={},
        is_trained=True,
        created_at=datetime.utcnow()
    )


# Processor Management Endpoints
@router.post("/tenants/{tenant_id}/processors", response_model=ProcessorResponse, status_code=status.HTTP_201_CREATED)
async def create_processor(
    tenant_id: UUID,
    request: CreateProcessorRequest,
    current_user: User = Depends(require_streaming_access)
):
    """Create a new streaming processor."""
    try:
        # Get detector
        detector = await get_detector_by_id(request.detector_id)
        
        # Prepare configuration
        config = {
            "window": request.window_config,
            "backpressure": request.backpressure_config or {},
            "buffer_size": request.buffer_size,
            "max_batch_size": request.max_batch_size,
            "processing_timeout": request.processing_timeout
        }
        
        # Create processor
        processor = await streaming_service.create_processor(
            processor_id=request.processor_id,
            detector=detector,
            config=config
        )
        
        return ProcessorResponse(
            processor_id=request.processor_id,
            state=processor.get_state(),
            detector_name=detector.name,
            window_config=request.window_config,
            metrics=processor.get_metrics().to_dict(),
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create processor: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/processors", response_model=List[ProcessorResponse])
async def list_processors(
    tenant_id: UUID,
    current_user: User = Depends(require_streaming_access)
):
    """List all streaming processors for a tenant."""
    try:
        processor_ids = streaming_service.list_processors()
        processors = []
        
        for processor_id in processor_ids:
            processor = streaming_service.get_processor(processor_id)
            if processor:
                processors.append(ProcessorResponse(
                    processor_id=processor_id,
                    state=processor.get_state(),
                    detector_name=processor.detector.name,
                    window_config=processor.window_config,
                    metrics=processor.get_metrics().to_dict(),
                    created_at=datetime.utcnow()  # TODO: Store actual creation time
                ))
        
        return processors
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list processors: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/processors/{processor_id}", response_model=ProcessorResponse)
async def get_processor(
    tenant_id: UUID,
    processor_id: str,
    current_user: User = Depends(require_streaming_access)
):
    """Get details of a specific processor."""
    try:
        processor = streaming_service.get_processor(processor_id)
        if not processor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processor {processor_id} not found"
            )
        
        return ProcessorResponse(
            processor_id=processor_id,
            state=processor.get_state(),
            detector_name=processor.detector.name,
            window_config=processor.window_config,
            metrics=processor.get_metrics().to_dict(),
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processor: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/processors/{processor_id}/start")
async def start_processor(
    tenant_id: UUID,
    processor_id: str,
    current_user: User = Depends(require_streaming_access)
):
    """Start a streaming processor."""
    try:
        await streaming_service.start_processor(processor_id)
        
        return {"message": f"Processor {processor_id} started successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start processor: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/processors/{processor_id}/stop")
async def stop_processor(
    tenant_id: UUID,
    processor_id: str,
    current_user: User = Depends(require_streaming_access)
):
    """Stop a streaming processor."""
    try:
        await streaming_service.stop_processor(processor_id)
        
        return {"message": f"Processor {processor_id} stopped successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to stop processor: {str(e)}"
        )


@router.delete("/tenants/{tenant_id}/processors/{processor_id}")
async def delete_processor(
    tenant_id: UUID,
    processor_id: str,
    current_user: User = Depends(require_streaming_access)
):
    """Delete a streaming processor."""
    try:
        await streaming_service.remove_processor(processor_id)
        
        return {"message": f"Processor {processor_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete processor: {str(e)}"
        )


# Metrics Endpoints
@router.get("/tenants/{tenant_id}/processors/{processor_id}/metrics", response_model=StreamMetricsResponse)
async def get_processor_metrics(
    tenant_id: UUID,
    processor_id: str,
    current_user: User = Depends(require_streaming_access)
):
    """Get metrics for a specific processor."""
    try:
        processor = streaming_service.get_processor(processor_id)
        if not processor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processor {processor_id} not found"
            )
        
        metrics = processor.get_metrics()
        
        return StreamMetricsResponse(
            processor_id=processor_id,
            total_processed=metrics.total_processed,
            anomalies_detected=metrics.anomalies_detected,
            processing_rate=metrics.processing_rate,
            avg_latency_ms=metrics.avg_latency_ms,
            error_count=metrics.error_count,
            last_processed=metrics.last_processed,
            backpressure_events=metrics.backpressure_events,
            dropped_records=metrics.dropped_records
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/tenants/{tenant_id}/metrics", response_model=List[StreamMetricsResponse])
async def get_all_metrics(
    tenant_id: UUID,
    current_user: User = Depends(require_streaming_access)
):
    """Get metrics for all processors in a tenant."""
    try:
        all_metrics = await streaming_service.get_all_metrics()
        
        response = []
        for processor_id, metrics in all_metrics.items():
            response.append(StreamMetricsResponse(
                processor_id=processor_id,
                total_processed=metrics.total_processed,
                anomalies_detected=metrics.anomalies_detected,
                processing_rate=metrics.processing_rate,
                avg_latency_ms=metrics.avg_latency_ms,
                error_count=metrics.error_count,
                last_processed=metrics.last_processed,
                backpressure_events=metrics.backpressure_events,
                dropped_records=metrics.dropped_records
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


# Data Ingestion Endpoints
@router.post("/tenants/{tenant_id}/processors/{processor_id}/ingest")
async def ingest_record(
    tenant_id: UUID,
    processor_id: str,
    request: StreamRecordRequest,
    current_user: User = Depends(require_streaming_access)
):
    """Ingest a single record into a processor."""
    try:
        processor = streaming_service.get_processor(processor_id)
        if not processor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processor {processor_id} not found"
            )
        
        # Create stream record
        record = StreamRecord(
            id=f"api_record_{int(datetime.utcnow().timestamp() * 1000000)}",
            timestamp=datetime.utcnow(),
            data=request.data,
            tenant_id=TenantId(request.tenant_id),
            metadata=request.metadata or {}
        )
        
        # Process record
        accepted = await processor.process_record(record)
        
        if not accepted:
            return {
                "message": "Record rejected due to backpressure",
                "accepted": False
            }
        
        return {
            "message": "Record ingested successfully",
            "accepted": True,
            "record_id": record.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest record: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/processors/{processor_id}/ingest-batch")
async def ingest_batch(
    tenant_id: UUID,
    processor_id: str,
    request: BatchStreamRequest,
    current_user: User = Depends(require_streaming_access)
):
    """Ingest a batch of records into a processor."""
    try:
        processor = streaming_service.get_processor(processor_id)
        if not processor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processor {processor_id} not found"
            )
        
        # Create stream records
        records = []
        base_timestamp = datetime.utcnow()
        
        for i, req in enumerate(request.records):
            record = StreamRecord(
                id=f"api_batch_{int(base_timestamp.timestamp() * 1000000)}_{i}",
                timestamp=base_timestamp,
                data=req.data,
                tenant_id=TenantId(req.tenant_id),
                metadata=req.metadata or {}
            )
            records.append(record)
        
        # Process batch
        accepted_count = await processor.process_batch(records)
        
        return {
            "message": f"Batch processed: {accepted_count}/{len(records)} records accepted",
            "total_records": len(records),
            "accepted_records": accepted_count,
            "rejected_records": len(records) - accepted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest batch: {str(e)}"
        )


# Real-time Notifications via WebSocket
@router.websocket("/tenants/{tenant_id}/processors/{processor_id}/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    tenant_id: UUID,
    processor_id: str
):
    """WebSocket endpoint for real-time anomaly alerts."""
    await websocket.accept()
    
    try:
        processor = streaming_service.get_processor(processor_id)
        if not processor:
            await websocket.close(code=1008, reason="Processor not found")
            return
        
        # Alert queue for this WebSocket connection
        alert_queue = asyncio.Queue()
        
        # Callback to add alerts to queue
        def anomaly_callback(anomaly):
            alert = AnomalyAlert(
                processor_id=processor_id,
                anomaly_id=str(anomaly.index),
                score=anomaly.score.value,
                timestamp=datetime.utcnow(),
                data=anomaly.features,
                metadata=anomaly.metadata
            )
            
            try:
                alert_queue.put_nowait(alert)
            except asyncio.QueueFull:
                # Queue is full, skip this alert
                pass
        
        # Register callback
        processor.add_anomaly_callback(anomaly_callback)
        
        try:
            while True:
                # Wait for alert or connection close
                try:
                    alert = await asyncio.wait_for(alert_queue.get(), timeout=30.0)
                    
                    # Send alert to client
                    await websocket.send_text(alert.json())
                    
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await websocket.ping()
                    
        except WebSocketDisconnect:
            pass
        finally:
            # Remove callback when connection closes
            if anomaly_callback in processor.anomaly_callbacks:
                processor.anomaly_callbacks.remove(anomaly_callback)
                
    except Exception as e:
        await websocket.close(code=1011, reason=f"Server error: {str(e)}")


# Connector Management Endpoints
@router.get("/connectors/available", response_model=List[str])
async def get_available_connectors():
    """Get list of available streaming connectors."""
    try:
        available = ConnectorFactory.get_available_connectors()
        return [connector.value for connector in available]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available connectors: {str(e)}"
        )


@router.post("/tenants/{tenant_id}/processors/{processor_id}/connect")
async def connect_data_source(
    tenant_id: UUID,
    processor_id: str,
    config: ConnectorConfigRequest,
    current_user: User = Depends(require_streaming_access)
):
    """Connect a data source to a processor."""
    try:
        processor = streaming_service.get_processor(processor_id)
        if not processor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processor {processor_id} not found"
            )
        
        # Create connector config
        connector_config = ConnectorConfig(
            connector_type=config.connector_type,
            connection_params=config.connection_params,
            serialization_format=config.serialization_format,
            batch_size=config.batch_size,
            max_wait_time=config.max_wait_time
        )
        
        # Create and connect
        connector = ConnectorFactory.create_connector(connector_config)
        await connector.connect()
        
        # Start consuming in background task
        async def consume_task():
            try:
                async for record in connector.consume():
                    await processor.process_record(record)
            except Exception as e:
                logger.error(f"Error in consume task: {e}")
            finally:
                await connector.disconnect()
        
        # Store the task (in a real implementation, you'd want to manage these properly)
        asyncio.create_task(consume_task())
        
        return {
            "message": f"Connected {config.connector_type.value} to processor {processor_id}",
            "connector_type": config.connector_type.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to connect data source: {str(e)}"
        )


# Health Check Endpoint
@router.get("/health")
async def streaming_health_check():
    """Health check for streaming service."""
    try:
        processor_count = len(streaming_service.list_processors())
        
        return {
            "status": "healthy",
            "active_processors": processor_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }