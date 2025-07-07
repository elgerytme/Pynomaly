"""REST API endpoints for streaming pipeline management."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from pynomaly.application.services.streaming_pipeline_manager import (
    StreamingPipelineManager,
)
from pynomaly.infrastructure.streaming.real_time_anomaly_pipeline import (
    AlertSeverity,
)
from pynomaly.presentation.api.deps import (
    get_current_user,
    get_streaming_pipeline_manager,
    require_write,
)

router = APIRouter(prefix="/streaming", tags=["Streaming Pipelines"])


# ==================== Request/Response Models ====================


class CreatePipelineRequest(BaseModel):
    """Request model for creating a streaming pipeline."""

    pipeline_id: Optional[str] = Field(None, description="Optional custom pipeline ID")
    data_source_type: str = Field(
        ..., description="Data source type (kafka, websocket)"
    )
    data_source_config: Dict[str, Any] = Field(
        ..., description="Data source configuration"
    )
    detector_config: Dict[str, Any] = Field(..., description="Detector configuration")
    streaming_config: Dict[str, Any] = Field(..., description="Streaming configuration")


class CreatePipelineFromTemplateRequest(BaseModel):
    """Request model for creating a pipeline from template."""

    template_name: str = Field(..., description="Template name to use")
    pipeline_id: Optional[str] = Field(None, description="Optional custom pipeline ID")
    override_config: Optional[Dict[str, Any]] = Field(
        None, description="Configuration overrides"
    )


class CreatePipelineResponse(BaseModel):
    """Response model for pipeline creation."""

    pipeline_id: str = Field(..., description="Created pipeline ID")
    message: str = Field(..., description="Success message")


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""

    pipeline_id: str = Field(..., description="Pipeline ID")
    is_running: bool = Field(..., description="Whether pipeline is running")
    start_time: Optional[str] = Field(None, description="Start time")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    buffer_size: int = Field(..., description="Current buffer size")
    buffer_capacity: int = Field(..., description="Buffer capacity")
    detector_trained: bool = Field(..., description="Whether detector is trained")
    samples_processed: int = Field(..., description="Total samples processed")
    metrics: Dict[str, Any] = Field(..., description="Pipeline metrics")


class AggregatedMetricsResponse(BaseModel):
    """Response model for aggregated metrics."""

    total_pipelines: int = Field(..., description="Total number of pipelines")
    running_pipelines: int = Field(..., description="Number of running pipelines")
    total_processed_records: int = Field(..., description="Total processed records")
    total_anomalies_detected: int = Field(..., description="Total anomalies detected")
    total_errors: int = Field(..., description="Total errors")
    overall_anomaly_rate: float = Field(..., description="Overall anomaly rate")
    overall_error_rate: float = Field(..., description="Overall error rate")
    average_processing_latency: float = Field(
        ..., description="Average processing latency"
    )
    total_alerts: int = Field(..., description="Total alerts generated")


class AlertResponse(BaseModel):
    """Response model for alerts."""

    alert_id: str = Field(..., description="Alert ID")
    timestamp: str = Field(..., description="Alert timestamp")
    severity: str = Field(..., description="Alert severity")
    alert_type: str = Field(..., description="Alert type")
    message: str = Field(..., description="Alert message")
    pipeline_id: Optional[str] = Field(None, description="Associated pipeline ID")
    anomaly_score: Optional[float] = Field(None, description="Anomaly score")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class AlertStatisticsResponse(BaseModel):
    """Response model for alert statistics."""

    total_alerts: int = Field(..., description="Total number of alerts")
    alerts_by_severity: Dict[str, int] = Field(..., description="Alerts by severity")
    alerts_by_type: Dict[str, int] = Field(..., description="Alerts by type")
    alerts_by_pipeline: Dict[str, int] = Field(..., description="Alerts by pipeline")
    recent_alert_rate: float = Field(..., description="Recent alert rate per minute")


class TemplateResponse(BaseModel):
    """Response model for pipeline templates."""

    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    data_source_type: str = Field(..., description="Data source type")
    data_source_config: Dict[str, Any] = Field(..., description="Data source config")
    detector_config: Dict[str, Any] = Field(..., description="Detector config")
    streaming_config: Dict[str, Any] = Field(..., description="Streaming config")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    overall_status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    pipeline_health: Dict[str, Any] = Field(
        ..., description="Individual pipeline health"
    )
    issues: List[str] = Field(..., description="Overall issues")


# ==================== Pipeline Management Endpoints ====================


@router.post("/pipelines", response_model=CreatePipelineResponse)
async def create_pipeline(
    request: CreatePipelineRequest,
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> CreatePipelineResponse:
    """Create a new streaming pipeline with custom configuration."""
    try:
        pipeline_id = request.pipeline_id or str(uuid4())

        created_id = await manager.create_pipeline(
            pipeline_id=pipeline_id,
            data_source_type=request.data_source_type,
            data_source_config=request.data_source_config,
            detector_config=request.detector_config,
            streaming_config=request.streaming_config,
        )

        return CreatePipelineResponse(
            pipeline_id=created_id,
            message=f"Pipeline '{created_id}' created successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create pipeline: {str(e)}",
        )


@router.post("/pipelines/from-template", response_model=CreatePipelineResponse)
async def create_pipeline_from_template(
    request: CreatePipelineFromTemplateRequest,
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> CreatePipelineResponse:
    """Create a new streaming pipeline from a predefined template."""
    try:
        pipeline_id = await manager.create_pipeline_from_template(
            template_name=request.template_name,
            pipeline_id=request.pipeline_id,
            override_config=request.override_config,
        )

        return CreatePipelineResponse(
            pipeline_id=pipeline_id,
            message=f"Pipeline '{pipeline_id}' created from template '{request.template_name}'",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create pipeline from template: {str(e)}",
        )


@router.post("/pipelines/{pipeline_id}/start")
async def start_pipeline(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> Dict[str, str]:
    """Start a streaming pipeline."""
    try:
        await manager.start_pipeline(pipeline_id)
        return {"message": f"Pipeline '{pipeline_id}' started successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start pipeline: {str(e)}",
        )


@router.post("/pipelines/{pipeline_id}/stop")
async def stop_pipeline(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> Dict[str, str]:
    """Stop a streaming pipeline."""
    try:
        await manager.stop_pipeline(pipeline_id)
        return {"message": f"Pipeline '{pipeline_id}' stopped successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop pipeline: {str(e)}",
        )


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> Dict[str, str]:
    """Delete a streaming pipeline."""
    try:
        await manager.delete_pipeline(pipeline_id)
        return {"message": f"Pipeline '{pipeline_id}' deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pipeline: {str(e)}",
        )


@router.post("/pipelines/start-all")
async def start_all_pipelines(
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> Dict[str, str]:
    """Start all registered pipelines."""
    try:
        await manager.start_all_pipelines()
        return {
            "message": "All pipelines started (check individual status for details)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start all pipelines: {str(e)}",
        )


@router.post("/pipelines/stop-all")
async def stop_all_pipelines(
    current_user: dict = Depends(get_current_user),
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
    _: None = Depends(require_write),
) -> Dict[str, str]:
    """Stop all running pipelines."""
    try:
        await manager.stop_all_pipelines()
        return {"message": "All pipelines stopped"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop all pipelines: {str(e)}",
        )


# ==================== Status and Monitoring Endpoints ====================


@router.get("/pipelines/{pipeline_id}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline_id: str,
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> PipelineStatusResponse:
    """Get status of a specific pipeline."""
    try:
        status = manager.get_pipeline_status(pipeline_id)
        return PipelineStatusResponse(**status)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline status: {str(e)}",
        )


@router.get("/pipelines/status")
async def get_all_pipeline_status(
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> Dict[str, Any]:
    """Get status of all pipelines."""
    try:
        return manager.get_all_pipeline_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline status: {str(e)}",
        )


@router.get("/pipelines/{pipeline_id}/metrics")
async def get_pipeline_metrics(
    pipeline_id: str,
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> Dict[str, Any]:
    """Get metrics for a specific pipeline."""
    try:
        return manager.get_pipeline_metrics(pipeline_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline metrics: {str(e)}",
        )


@router.get("/metrics/aggregated", response_model=AggregatedMetricsResponse)
async def get_aggregated_metrics(
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> AggregatedMetricsResponse:
    """Get aggregated metrics across all pipelines."""
    try:
        metrics = manager.get_aggregated_metrics()
        return AggregatedMetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get aggregated metrics: {str(e)}",
        )


# ==================== Alert Management Endpoints ====================


@router.get("/alerts", response_model=List[AlertResponse])
async def get_recent_alerts(
    limit: int = 100,
    severity: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> List[AlertResponse]:
    """Get recent alerts with optional filtering."""
    try:
        # Convert severity string to enum if provided
        severity_filter = None
        if severity:
            try:
                severity_filter = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid severity: {severity}",
                )

        alerts = manager.get_recent_alerts(
            limit=limit,
            severity=severity_filter,
            pipeline_id=pipeline_id,
        )

        # Convert to response format
        return [
            AlertResponse(
                alert_id=alert.alert_id,
                timestamp=alert.timestamp.isoformat(),
                severity=alert.severity.value,
                alert_type=alert.alert_type,
                message=alert.message,
                pipeline_id=alert.metadata.get("pipeline_id"),
                anomaly_score=alert.anomaly_score,
                metadata=alert.metadata,
            )
            for alert in alerts
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alerts: {str(e)}",
        )


@router.get("/alerts/statistics", response_model=AlertStatisticsResponse)
async def get_alert_statistics(
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> AlertStatisticsResponse:
    """Get alert statistics."""
    try:
        stats = manager.get_alert_statistics()
        return AlertStatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert statistics: {str(e)}",
        )


# ==================== Template Management Endpoints ====================


@router.get("/templates", response_model=List[TemplateResponse])
async def get_templates(
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> List[TemplateResponse]:
    """Get all available pipeline templates."""
    try:
        templates = manager.get_templates()
        return [
            TemplateResponse(
                name=template.name,
                description=template.description,
                data_source_type=template.data_source_type,
                data_source_config=template.data_source_config,
                detector_config=template.detector_config,
                streaming_config=template.streaming_config,
            )
            for template in templates.values()
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get templates: {str(e)}",
        )


@router.get("/templates/{template_name}", response_model=TemplateResponse)
async def get_template(
    template_name: str,
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> TemplateResponse:
    """Get a specific pipeline template."""
    try:
        templates = manager.get_templates()
        if template_name not in templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template '{template_name}' not found",
            )

        template = templates[template_name]
        return TemplateResponse(
            name=template.name,
            description=template.description,
            data_source_type=template.data_source_type,
            data_source_config=template.data_source_config,
            detector_config=template.detector_config,
            streaming_config=template.streaming_config,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}",
        )


# ==================== Health and Diagnostics Endpoints ====================


@router.get("/health", response_model=HealthCheckResponse)
async def get_health_check(
    manager: StreamingPipelineManager = Depends(get_streaming_pipeline_manager),
) -> HealthCheckResponse:
    """Perform health check on all streaming pipelines."""
    try:
        health = await manager.health_check()
        return HealthCheckResponse(**health)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform health check: {str(e)}",
        )


@router.get("/info")
async def get_streaming_info() -> Dict[str, Any]:
    """Get information about streaming capabilities and configuration."""
    return {
        "service": "streaming_pipelines",
        "version": "1.0.0",
        "supported_data_sources": [
            {
                "type": "kafka",
                "description": "Apache Kafka message streaming",
                "required_config": ["bootstrap_servers", "topic", "group_id"],
                "optional_config": [
                    "auto_offset_reset",
                    "security_protocol",
                    "sasl_mechanism",
                ],
            },
            {
                "type": "websocket",
                "description": "WebSocket real-time data streaming",
                "required_config": ["websocket_url"],
                "optional_config": ["headers", "auth_token"],
            },
        ],
        "supported_algorithms": [
            {
                "name": "isolation_forest",
                "description": "Isolation Forest for anomaly detection",
                "parameters": ["contamination", "n_estimators", "max_samples"],
            },
            {
                "name": "one_class_svm",
                "description": "One-Class SVM for novelty detection",
                "parameters": ["contamination", "kernel", "gamma", "nu"],
            },
            {
                "name": "local_outlier_factor",
                "description": "Local Outlier Factor for outlier detection",
                "parameters": ["contamination", "n_neighbors", "algorithm"],
            },
        ],
        "features": [
            "real_time_processing",
            "sliding_window_detection",
            "automatic_retraining",
            "backpressure_control",
            "alert_management",
            "metrics_collection",
            "health_monitoring",
            "template_management",
        ],
        "limitations": [
            "requires_external_data_sources",
            "memory_bounded_windows",
            "cpu_intensive_detection",
        ],
    }
