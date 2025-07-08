"""Real-time streaming API endpoints."""

from datetime import timedelta
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from pynomaly.application.services.streaming_service import StreamingService
from pynomaly.domain.entities.streaming_session import (
    SessionStatus,
    SessionSummary,
    StreamingAlert,
    StreamingConfiguration,
    StreamingDataSink,
    StreamingDataSource,
    StreamingMetrics,
    StreamingSession,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container
from pynomaly.presentation.api.docs.response_models import (
    ErrorResponse,
    HTTPResponses,
    SuccessResponse,
)

router = APIRouter(
    prefix="/streaming",
    tags=["Real-time Streaming"],
    responses={
        401: HTTPResponses.unauthorized_401(),
        403: HTTPResponses.forbidden_403(),
        404: HTTPResponses.not_found_404(),
        500: HTTPResponses.server_error_500(),
    },
)


class CreateSessionRequest(BaseModel):
    """Request for creating streaming session."""

    name: str = Field(..., description="Session name")
    detector_id: UUID = Field(..., description="Detector to use for anomaly detection")
    data_source: StreamingDataSource = Field(
        ..., description="Input data source configuration"
    )
    configuration: StreamingConfiguration = Field(
        ..., description="Streaming configuration"
    )
    description: str | None = Field(None, description="Session description")
    data_sink: StreamingDataSink | None = Field(
        None, description="Output data sink configuration"
    )
    model_version: str | None = Field(None, description="Specific model version to use")
    max_duration_hours: float | None = Field(
        None, description="Maximum session duration in hours"
    )
    tags: list[str] = Field(default_factory=list, description="Session tags")


class CreateAlertRequest(BaseModel):
    """Request for creating streaming alert."""

    name: str = Field(..., description="Alert name")
    metric_name: str = Field(..., description="Metric to monitor")
    threshold_value: float = Field(..., description="Alert threshold value")
    comparison_operator: str = Field(
        ..., description="Comparison operator (>, <, >=, <=, ==)"
    )
    description: str | None = Field(None, description="Alert description")
    severity: str = Field(default="medium", description="Alert severity")
    duration_threshold_minutes: float = Field(
        default=1.0, description="Duration threshold in minutes"
    )
    notification_channels: list[str] = Field(
        default_factory=list, description="Notification channels"
    )


class ProcessDataRequest(BaseModel):
    """Request for processing streaming data."""

    data: dict[str, Any] = Field(..., description="Input data to process")


async def get_streaming_service(
    container: Container = Depends(get_container),
) -> StreamingService:
    """Get streaming service."""
    # This would be properly injected in a real implementation
    return StreamingService(
        model_repository=container.model_repository(),
        streaming_repository=None,  # Would be injected
        event_repository=None,  # Would be injected
        detector_service=None,  # Would be injected
        notification_service=None,  # Would be injected
    )


@router.post(
    "/sessions",
    response_model=SuccessResponse[StreamingSession],
    summary="Create Streaming Session",
    description="""
    Create a new real-time streaming session for anomaly detection.

    A streaming session continuously processes incoming data through a configured
    anomaly detector, generating real-time anomaly events and metrics.

    **Key Features:**
    - **Real-time Processing**: Continuous data stream processing
    - **Multiple Data Sources**: Support for Kafka, Kinesis, WebSocket, files
    - **Configurable Processing**: Batch, micro-batch, windowed processing modes
    - **Output Sinks**: Send results to various destinations
    - **Monitoring & Alerts**: Real-time metrics and alerting

    **Processing Modes:**
    - `real_time`: Process each message immediately
    - `micro_batch`: Process in small batches for efficiency
    - `sliding_window`: Process data in overlapping time windows
    - `tumbling_window`: Process data in non-overlapping time windows

    **Supported Data Sources:**
    - **Kafka**: Distributed streaming platform
    - **Kinesis**: AWS streaming service
    - **WebSocket**: Real-time web connections
    - **File**: File-based streaming (for testing)

    **Example Configuration:**
    ```json
    {
      "name": "Production Traffic Monitor",
      "detector_id": "detector-uuid",
      "data_source": {
        "source_type": "kafka",
        "connection_config": {
          "bootstrap_servers": ["localhost:9092"],
          "topic": "network-traffic"
        }
      },
      "configuration": {
        "processing_mode": "micro_batch",
        "batch_size": 1000,
        "max_throughput": 10000
      }
    }
    ```
    """,
    responses={
        201: HTTPResponses.created_201("Streaming session created successfully"),
        400: HTTPResponses.bad_request_400("Invalid session configuration"),
    },
)
async def create_streaming_session(
    request: CreateSessionRequest,
    created_by: str = Query(..., description="User creating the session"),
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingSession]:
    """Create a new streaming session."""
    try:
        max_duration = None
        if request.max_duration_hours:
            max_duration = timedelta(hours=request.max_duration_hours)

        session = await streaming_service.create_streaming_session(
            name=request.name,
            detector_id=request.detector_id,
            data_source=request.data_source,
            configuration=request.configuration,
            created_by=created_by,
            description=request.description,
            data_sink=request.data_sink,
            model_version=request.model_version,
            max_duration=max_duration,
            tags=request.tags,
        )

        return SuccessResponse(
            data=session, message="Streaming session created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@router.post(
    "/sessions/{session_id}/start",
    response_model=SuccessResponse[StreamingSession],
    summary="Start Streaming Session",
    description="""
    Start a streaming session to begin processing data.

    Once started, the session will:
    - Connect to the configured data source
    - Begin processing incoming data through the anomaly detector
    - Generate real-time metrics and events
    - Send outputs to configured sinks

    **Status Transitions:**
    `pending` → `starting` → `active`

    **Prerequisites:**
    - Session must be in `pending` status
    - Detector must be available and trained
    - Data source must be accessible
    """,
)
async def start_streaming_session(
    session_id: UUID,
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingSession]:
    """Start a streaming session."""
    try:
        session = await streaming_service.start_streaming_session(session_id)

        return SuccessResponse(
            data=session, message="Streaming session started successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start session: {str(e)}"
        )


@router.post(
    "/sessions/{session_id}/stop",
    response_model=SuccessResponse[StreamingSession],
    summary="Stop Streaming Session",
    description="""
    Stop a streaming session and terminate data processing.

    Stopping a session will:
    - Gracefully shut down data processing
    - Complete processing of any buffered data
    - Generate final metrics and checkpoints
    - Close connections to data sources and sinks

    **Status Transitions:**
    `active` → `stopping` → `stopped`
    """,
)
async def stop_streaming_session(
    session_id: UUID,
    error_message: str | None = Query(None, description="Optional error message"),
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingSession]:
    """Stop a streaming session."""
    try:
        session = await streaming_service.stop_streaming_session(
            session_id, error_message
        )

        return SuccessResponse(
            data=session, message="Streaming session stopped successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")


@router.post(
    "/sessions/{session_id}/pause",
    response_model=SuccessResponse[StreamingSession],
    summary="Pause Streaming Session",
    description="""
    Pause a streaming session temporarily.

    Pausing a session will:
    - Temporarily halt data processing
    - Maintain connections to data sources
    - Preserve session state and metrics
    - Allow resumption without losing context
    """,
)
async def pause_streaming_session(
    session_id: UUID,
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingSession]:
    """Pause a streaming session."""
    try:
        session = await streaming_service.pause_streaming_session(session_id)

        return SuccessResponse(
            data=session, message="Streaming session paused successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to pause session: {str(e)}"
        )


@router.post(
    "/sessions/{session_id}/resume",
    response_model=SuccessResponse[StreamingSession],
    summary="Resume Streaming Session",
    description="""
    Resume a paused streaming session.

    Resuming a session will:
    - Restart data processing from where it left off
    - Continue generating metrics and events
    - Maintain all previous configuration and state
    """,
)
async def resume_streaming_session(
    session_id: UUID,
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingSession]:
    """Resume a streaming session."""
    try:
        session = await streaming_service.resume_streaming_session(session_id)

        return SuccessResponse(
            data=session, message="Streaming session resumed successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to resume session: {str(e)}"
        )


@router.get(
    "/sessions/{session_id}/metrics",
    response_model=SuccessResponse[StreamingMetrics],
    summary="Get Session Metrics",
    description="""
    Get real-time metrics for a streaming session.

    **Metrics Include:**
    - **Throughput**: Messages per second, total processed
    - **Quality**: Anomalies detected, anomaly rate, false positive rate
    - **Latency**: End-to-end, percentile latencies (P50, P95, P99)
    - **Errors**: Failed messages, error rate, retry counts
    - **Resources**: CPU usage, memory usage, network I/O
    - **Backpressure**: Buffer utilization, backpressure events
    - **Windows**: Active/completed windows (for windowed processing)

    Use these metrics for:
    - Performance monitoring and optimization
    - Capacity planning and scaling decisions
    - Quality assessment and tuning
    - Troubleshooting and debugging
    """,
)
async def get_session_metrics(
    session_id: UUID,
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingMetrics]:
    """Get current metrics for a streaming session."""
    try:
        metrics = await streaming_service.get_session_metrics(session_id)

        return SuccessResponse(
            data=metrics, message="Retrieved session metrics successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get(
    "/sessions/{session_id}/summary",
    response_model=SuccessResponse[SessionSummary],
    summary="Get Session Summary",
    description="""
    Get a comprehensive summary of a streaming session.

    The summary provides a high-level overview including:
    - Session status and timing information
    - Key performance indicators
    - Throughput and quality metrics
    - Error rates and system health

    This is ideal for dashboards and monitoring displays.
    """,
)
async def get_session_summary(
    session_id: UUID,
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[SessionSummary]:
    """Get summary for a streaming session."""
    try:
        summary = await streaming_service.get_session_summary(session_id)

        return SuccessResponse(
            data=summary, message="Retrieved session summary successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get(
    "/sessions",
    response_model=SuccessResponse[list[SessionSummary]],
    summary="List Streaming Sessions",
    description="""
    List streaming sessions with optional filtering.

    **Filter Options:**
    - **Status**: Filter by session status (active, stopped, etc.)
    - **Detector**: Filter by detector ID
    - **Creator**: Filter by user who created the session

    **Use Cases:**
    - Monitor all active streaming sessions
    - Find sessions for a specific detector
    - Audit session usage by user
    - Dashboard overview of streaming activity
    """,
)
async def list_streaming_sessions(
    status: SessionStatus | None = Query(None, description="Filter by session status"),
    detector_id: UUID | None = Query(None, description="Filter by detector ID"),
    created_by: str | None = Query(None, description="Filter by creator"),
    limit: int = Query(100, description="Maximum number of results"),
    offset: int = Query(0, description="Result offset"),
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[list[SessionSummary]]:
    """List streaming sessions."""
    try:
        sessions = await streaming_service.list_streaming_sessions(
            status=status,
            detector_id=detector_id,
            created_by=created_by,
            limit=limit,
            offset=offset,
        )

        return SuccessResponse(
            data=sessions, message=f"Retrieved {len(sessions)} streaming sessions"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list sessions: {str(e)}"
        )


@router.post(
    "/sessions/{session_id}/alerts",
    response_model=SuccessResponse[StreamingAlert],
    summary="Create Session Alert",
    description="""
    Create an alert for monitoring streaming session metrics.

    Alerts allow you to be notified when streaming metrics exceed thresholds:

    **Common Alert Examples:**
    - **High Error Rate**: Alert when error_rate > 0.05 (5%)
    - **Low Throughput**: Alert when messages_per_second < 100
    - **High Latency**: Alert when p99_latency > 1000ms
    - **Memory Usage**: Alert when memory_usage > 80%
    - **Anomaly Surge**: Alert when anomaly_rate > 0.1 (10%)

    **Notification Channels:**
    - Email notifications
    - Slack/Teams integration
    - Webhook endpoints
    - SMS alerts (for critical issues)

    **Example:**
    ```json
    {
      "name": "High Error Rate Alert",
      "metric_name": "error_rate",
      "threshold_value": 0.05,
      "comparison_operator": ">",
      "severity": "high",
      "notification_channels": ["email", "slack"]
    }
    ```
    """,
    responses={
        201: HTTPResponses.created_201("Alert created successfully"),
        400: HTTPResponses.bad_request_400("Invalid alert configuration"),
    },
)
async def create_session_alert(
    session_id: UUID,
    request: CreateAlertRequest,
    created_by: str = Query(..., description="User creating the alert"),
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[StreamingAlert]:
    """Create an alert for a streaming session."""
    try:
        duration_threshold = timedelta(minutes=request.duration_threshold_minutes)

        alert = await streaming_service.create_streaming_alert(
            session_id=session_id,
            name=request.name,
            metric_name=request.metric_name,
            threshold_value=request.threshold_value,
            comparison_operator=request.comparison_operator,
            created_by=created_by,
            description=request.description,
            severity=request.severity,
            duration_threshold=duration_threshold,
            notification_channels=request.notification_channels,
        )

        return SuccessResponse(data=alert, message="Alert created successfully")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.post(
    "/sessions/{session_id}/process",
    response_model=SuccessResponse[dict[str, Any]],
    summary="Process Single Data Point",
    description="""
    Process a single data point through the streaming pipeline.

    This endpoint allows you to:
    - Test the streaming pipeline with sample data
    - Process ad-hoc data points for immediate results
    - Debug and validate processing logic

    **Input Requirements:**
    - Data must match the expected schema for the detector
    - Session must be in active status
    - Detector must be available and trained

    **Response Includes:**
    - Anomaly detection results (score, classification)
    - Feature contributions and explanations
    - Processing metadata and timing
    - Any generated events or alerts

    **Example Input:**
    ```json
    {
      "data": {
        "timestamp": "2024-12-25T10:30:00Z",
        "cpu_usage": 85.2,
        "memory_usage": 78.5,
        "network_io": 1024000,
        "response_time": 250
      }
    }
    ```
    """,
)
async def process_streaming_data(
    session_id: UUID,
    request: ProcessDataRequest,
    streaming_service: StreamingService = Depends(get_streaming_service),
) -> SuccessResponse[dict[str, Any]]:
    """Process a single data point through streaming pipeline."""
    try:
        result = await streaming_service.process_streaming_data(
            session_id, request.data
        )

        return SuccessResponse(data=result, message="Data processed successfully")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")


@router.websocket("/sessions/{session_id}/live")
async def streaming_websocket(
    websocket: WebSocket,
    session_id: UUID,
    streaming_service: StreamingService = Depends(get_streaming_service),
):
    """
    WebSocket endpoint for real-time streaming session monitoring.

    This WebSocket provides real-time updates for:
    - Live metrics and performance data
    - Anomaly detection results as they occur
    - Session status changes
    - Alert notifications

    **Message Types:**
    - `metrics`: Real-time metrics update
    - `anomaly`: Anomaly detection event
    - `alert`: Alert notification
    - `status`: Session status change
    - `error`: Error notification

    **Usage:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/streaming/sessions/{session_id}/live');

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log(`Received ${data.type}:`, data.payload);
    };
    ```
    """
    await websocket.accept()

    try:
        # Verify session exists
        summary = await streaming_service.get_session_summary(session_id)

        # Send initial session state
        await websocket.send_json(
            {
                "type": "connected",
                "payload": {
                    "session_id": str(session_id),
                    "session_name": summary.name,
                    "status": summary.status,
                    "message": "Connected to streaming session",
                },
            }
        )

        # Keep connection alive and send periodic updates
        import asyncio

        while True:
            try:
                # Send metrics update every 5 seconds
                metrics = await streaming_service.get_session_metrics(session_id)

                await websocket.send_json(
                    {
                        "type": "metrics",
                        "payload": {
                            "session_id": str(session_id),
                            "timestamp": metrics.measurement_time.isoformat(),
                            "messages_processed": metrics.messages_processed,
                            "messages_per_second": metrics.messages_per_second,
                            "anomalies_detected": metrics.anomalies_detected,
                            "anomaly_rate": metrics.anomaly_rate,
                            "error_rate": metrics.error_rate,
                            "avg_processing_time": metrics.avg_processing_time,
                            "p95_latency": metrics.p95_latency,
                            "cpu_usage": metrics.cpu_usage,
                            "memory_usage": metrics.memory_usage,
                        },
                    }
                )

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await websocket.send_json(
                    {
                        "type": "error",
                        "payload": {
                            "session_id": str(session_id),
                            "error": str(e),
                            "message": "Error retrieving session data",
                        },
                    }
                )
                await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "payload": {
                    "session_id": str(session_id),
                    "error": str(e),
                    "message": "WebSocket connection error",
                },
            }
        )
    finally:
        await websocket.close()
