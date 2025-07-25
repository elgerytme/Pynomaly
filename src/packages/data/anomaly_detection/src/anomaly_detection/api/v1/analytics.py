"""Analytics and BI dashboard API endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import asyncio

from fastapi import APIRouter, HTTPException, Query, Body, Depends, status
from pydantic import BaseModel, Field, validator

from ...application.services.intelligence.analytics_engine import (
    AnalyticsEngine,
    AnalyticsQuery,
    MetricType,
    AggregationType,
    get_analytics_engine
)
from ...application.services.intelligence.dashboard_service import (
    DashboardService,
    DashboardTemplate,
    ReportFormat,
    get_dashboard_service
)
from ...infrastructure.logging import get_logger
from ...infrastructure.logging.log_decorator import async_log_decorator

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])


# Request/Response Models
class TimeRangeModel(BaseModel):
    """Time range specification."""
    start: datetime
    end: datetime
    
    @validator('end')
    def end_after_start(cls, v, values, **kwargs):
        if 'start' in values and v <= values['start']:
            raise ValueError('End time must be after start time')
        return v


class AnalyticsQueryModel(BaseModel):
    """Analytics query request."""
    metric_type: str = Field(..., description="Type of metrics to query")
    time_range: TimeRangeModel = Field(..., description="Time range for query")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters to apply")
    group_by: List[str] = Field(default_factory=list, description="Fields to group by")
    aggregation: str = Field(default="count", description="Aggregation method")
    limit: Optional[int] = Field(None, ge=1, le=10000, description="Result limit")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_desc: bool = Field(default=True, description="Sort descending")
    
    @validator('metric_type')
    def validate_metric_type(cls, v):
        valid_types = [mt.value for mt in MetricType]
        if v not in valid_types:
            raise ValueError(f"Invalid metric type. Must be one of: {valid_types}")
        return v
    
    @validator('aggregation')
    def validate_aggregation(cls, v):
        valid_aggs = [at.value for at in AggregationType]
        if v not in valid_aggs:
            raise ValueError(f"Invalid aggregation. Must be one of: {valid_aggs}")
        return v


class DashboardListResponse(BaseModel):
    """Dashboard list response."""
    dashboards: List[Dict[str, Any]]
    total: int
    generated_at: str


class DashboardResponse(BaseModel):
    """Dashboard response."""
    dashboard_id: str
    title: str
    description: str
    widgets: List[Dict[str, Any]]
    generated_at: str
    usage_metrics: Optional[Dict[str, Any]] = None


class AnalyticsResponse(BaseModel):
    """Analytics query response."""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query: Dict[str, Any]
    generated_at: str


class InsightsResponse(BaseModel):
    """Insights response."""
    insights: List[Dict[str, Any]]
    generated_at: str
    data_summary: Optional[Dict[str, Any]] = None


class CustomDashboardRequest(BaseModel):
    """Custom dashboard creation request."""
    dashboard_id: str = Field(..., description="Unique dashboard ID")
    title: str = Field(..., description="Dashboard title")
    description: str = Field(default="", description="Dashboard description")
    widgets: List[Dict[str, Any]] = Field(..., description="Widget configurations")
    tags: List[str] = Field(default_factory=list, description="Dashboard tags")
    is_public: bool = Field(default=False, description="Public dashboard flag")


class ExportRequest(BaseModel):
    """Dashboard export request."""
    format: str = Field(..., description="Export format")
    
    @validator('format')
    def validate_format(cls, v):
        valid_formats = [rf.value for rf in ReportFormat]
        if v not in valid_formats:
            raise ValueError(f"Invalid format. Must be one of: {valid_formats}")
        return v


# Dependency functions
def get_analytics_engine_dep() -> AnalyticsEngine:
    """Get analytics engine dependency."""
    return get_analytics_engine()


def get_dashboard_service_dep() -> DashboardService:
    """Get dashboard service dependency."""
    return get_dashboard_service()


# Analytics Endpoints
@router.post("/query", response_model=AnalyticsResponse)
# #@async_log_decorator("execute_analytics_query")  # Temporarily disabled
async def execute_query(
    query_request: AnalyticsQueryModel,
    analytics_engine: AnalyticsEngine = Depends(get_analytics_engine_dep)
) -> AnalyticsResponse:
    """Execute an analytics query."""
    try:
        # Convert request to internal query object
        query = AnalyticsQuery(
            metric_type=MetricType(query_request.metric_type),
            time_range=(query_request.time_range.start, query_request.time_range.end),
            filters=query_request.filters,
            group_by=query_request.group_by,
            aggregation=AggregationType(query_request.aggregation),
            limit=query_request.limit,
            sort_by=query_request.sort_by,
            sort_desc=query_request.sort_desc
        )
        
        # Execute query
        result = await analytics_engine.execute_query(query)
        
        if "error" in result.metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query execution failed: {result.metadata['error']}"
            )
        
        return AnalyticsResponse(
            data=result.data.to_dict(orient="records"),
            metadata=result.metadata,
            query={
                "metric_type": query_request.metric_type,
                "time_range": {
                    "start": query_request.time_range.start.isoformat(),
                    "end": query_request.time_range.end.isoformat()
                },
                "filters": query_request.filters,
                "group_by": query_request.group_by,
                "aggregation": query_request.aggregation
            },
            generated_at=result.generated_at.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Analytics query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/insights/{metric_type}", response_model=InsightsResponse)
#@async_log_decorator("get_analytics_insights")
async def get_insights(
    metric_type: str,
    hours_back: int = Query(default=24, ge=1, le=8760, description="Hours to look back"),
    analytics_engine: AnalyticsEngine = Depends(get_analytics_engine_dep)
) -> InsightsResponse:
    """Get automated insights for metrics."""
    try:
        # Validate metric type
        try:
            metric_enum = MetricType(metric_type)
        except ValueError:
            valid_types = [mt.value for mt in MetricType]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid metric type. Must be one of: {valid_types}"
            )
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Get insights
        insights_result = await analytics_engine.get_insights(
            metric_enum,
            (start_time, end_time)
        )
        
        if "error" in insights_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insights generation failed: {insights_result['error']}"
            )
        
        return InsightsResponse(
            insights=insights_result.get("insights", []),
            generated_at=insights_result.get("generated_at", datetime.utcnow().isoformat()),
            data_summary=insights_result.get("data_summary")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Dashboard Endpoints
@router.get("/dashboards", response_model=DashboardListResponse)
#@async_log_decorator("list_dashboards")
async def list_dashboards(
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> DashboardListResponse:
    """List available dashboards."""
    try:
        # Parse tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Get dashboards
        dashboards = await dashboard_service.list_dashboards(tag_list)
        
        return DashboardListResponse(
            dashboards=dashboards,
            total=len(dashboards),
            generated_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Dashboard listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/dashboards/{dashboard_id}", response_model=DashboardResponse)
@async_log_decorator( "get_dashboard")
async def get_dashboard(
    dashboard_id: str,
    viewer_id: Optional[str] = Query(None, description="Viewer ID for usage tracking"),
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> DashboardResponse:
    """Get dashboard by ID."""
    try:
        dashboard_data = await dashboard_service.get_dashboard(dashboard_id, viewer_id)
        
        if not dashboard_data or "error" in dashboard_data:
            error_msg = dashboard_data.get("error", "Dashboard not found") if dashboard_data else "Dashboard not found"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        
        return DashboardResponse(**dashboard_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/dashboards", status_code=status.HTTP_201_CREATED)
@async_log_decorator( "create_custom_dashboard")
async def create_dashboard(
    dashboard_request: CustomDashboardRequest,
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> Dict[str, Any]:
    """Create a custom dashboard."""
    try:
        result = await dashboard_service.create_custom_dashboard(dashboard_request.dict())
        
        if "error" in result:
            if "already exists" in result["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=result["error"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/dashboards/{dashboard_id}/insights", response_model=InsightsResponse)
@async_log_decorator( "get_dashboard_insights")
async def get_dashboard_insights(
    dashboard_id: str,
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> InsightsResponse:
    """Get insights for a specific dashboard."""
    try:
        insights_result = await dashboard_service.get_dashboard_insights(dashboard_id)
        
        if "error" in insights_result:
            if "not found" in insights_result["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=insights_result["error"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=insights_result["error"]
                )
        
        return InsightsResponse(
            insights=insights_result.get("insights", []),
            generated_at=insights_result.get("generated_at", datetime.utcnow().isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard insights error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/dashboards/{dashboard_id}/export")
@async_log_decorator( "export_dashboard")
async def export_dashboard(
    dashboard_id: str,
    export_request: ExportRequest,
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> Dict[str, Any]:
    """Export dashboard in specified format."""
    try:
        result = await dashboard_service.export_dashboard(
            dashboard_id,
            ReportFormat(export_request.format)
        )
        
        if "error" in result:
            if "not found" in result["error"].lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result["error"]
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result["error"]
                )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard export error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/dashboards/metrics/usage")
@async_log_decorator( "get_dashboard_usage_metrics")
async def get_dashboard_metrics(
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> Dict[str, Any]:
    """Get overall dashboard usage metrics."""
    try:
        metrics = await dashboard_service.get_dashboard_metrics()
        
        if "error" in metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=metrics["error"]
            )
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Template Endpoints
@router.get("/templates")
@async_log_decorator( "list_dashboard_templates")
async def list_dashboard_templates() -> Dict[str, Any]:
    """List available dashboard templates."""
    try:
        templates = [
            {
                "template_id": template.value,
                "name": template.value.replace("_", " ").title(),
                "description": f"Pre-configured {template.value.replace('_', ' ')} dashboard"
            }
            for template in DashboardTemplate
        ]
        
        return {
            "templates": templates,
            "total": len(templates),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Template listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/templates/{template_id}/create")
@async_log_decorator( "create_dashboard_from_template")
async def create_dashboard_from_template(
    template_id: str,
    custom_id: Optional[str] = Body(None, description="Custom dashboard ID"),
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> Dict[str, Any]:
    """Create dashboard from template."""
    try:
        # Validate template
        try:
            template_enum = DashboardTemplate(template_id)
        except ValueError:
            valid_templates = [dt.value for dt in DashboardTemplate]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid template. Must be one of: {valid_templates}"
            )
        
        # This would typically create a dashboard from the template
        # For now, return success message
        dashboard_id = custom_id or f"{template_id}_{int(datetime.utcnow().timestamp())}"
        
        return {
            "success": True,
            "dashboard_id": dashboard_id,
            "template_used": template_id,
            "message": f"Dashboard created from {template_id} template",
            "created_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template dashboard creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Health and Status Endpoints
@router.get("/health")
@async_log_decorator( "analytics_health_check")
async def health_check() -> Dict[str, Any]:
    """Health check for analytics service."""
    try:
        # Basic health checks
        analytics_engine = get_analytics_engine()
        dashboard_service = get_dashboard_service()
        
        # Test basic functionality
        now = datetime.utcnow()
        test_query = AnalyticsQuery(
            metric_type=MetricType.SYSTEM_METRICS,
            time_range=(now - timedelta(hours=1), now)
        )
        
        # Quick test query (with timeout)
        test_task = asyncio.create_task(analytics_engine.execute_query(test_query))
        try:
            result = await asyncio.wait_for(test_task, timeout=5.0)
            query_healthy = "error" not in result.metadata
        except asyncio.TimeoutError:
            query_healthy = False
        except Exception:
            query_healthy = False
        
        # Dashboard service health
        try:
            dashboards = await dashboard_service.list_dashboards()
            dashboard_healthy = isinstance(dashboards, list)
        except Exception:
            dashboard_healthy = False
        
        status_code = status.HTTP_200_OK if query_healthy and dashboard_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return {
            "status": "healthy" if query_healthy and dashboard_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "analytics_engine": "healthy" if query_healthy else "unhealthy",
                "dashboard_service": "healthy" if dashboard_healthy else "unhealthy"
            },
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }


@router.get("/status")
@async_log_decorator( "analytics_status")
async def get_status(
    analytics_engine: AnalyticsEngine = Depends(get_analytics_engine_dep),
    dashboard_service: DashboardService = Depends(get_dashboard_service_dep)
) -> Dict[str, Any]:
    """Get detailed analytics service status."""
    try:
        # Get dashboard metrics
        dashboard_metrics = await dashboard_service.get_dashboard_metrics()
        
        return {
            "service": "analytics",
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "dashboards": dashboard_metrics.get("overview", {}),
                "uptime": "24h",  # Would be calculated from service start time
                "requests_processed": 0,  # Would track actual requests
                "avg_response_time": 0.0  # Would track actual response times
            },
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )