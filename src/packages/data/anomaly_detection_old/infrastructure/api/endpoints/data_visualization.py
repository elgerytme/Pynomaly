"""Data Visualization and Reporting API Endpoints.

This module provides RESTful endpoints for data visualization, chart generation,
and interactive dashboard creation.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from ..security.authorization import require_permissions
from ..dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-visualization", tags=["Data Visualization"])

# Pydantic models for request/response
class ChartGenerationRequest(BaseModel):
    """Request model for chart generation."""
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: str = Field(..., description="Type of chart (bar, line, pie, scatter, histogram, heatmap)")
    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    x_axis: str = Field(..., description="X-axis column name")
    y_axis: Optional[str] = Field(default=None, description="Y-axis column name")
    group_by: Optional[str] = Field(default=None, description="Column to group by")
    aggregation: Optional[str] = Field(default=None, description="Aggregation function (sum, avg, count, min, max)")
    filters: Optional[List[Dict[str, Any]]] = Field(default=None, description="Data filters")
    styling: Optional[Dict[str, Any]] = Field(default=None, description="Chart styling options")        schema_extra = {
            "example": {
                "dataset_id": "sales_data_2024",
                "chart_type": "bar",
                "data": [
                    {"month": "Jan", "sales": 1000, "region": "North"},
                    {"month": "Feb", "sales": 1200, "region": "North"},
                    {"month": "Jan", "sales": 800, "region": "South"},
                    {"month": "Feb", "sales": 900, "region": "South"}
                ],
                "x_axis": "month",
                "y_axis": "sales",
                "group_by": "region",
                "aggregation": "sum",
                "filters": [
                    {"column": "region", "operator": "in", "values": ["North", "South"]}
                ],
                "styling": {
                    "color_scheme": "blue",
                    "title": "Monthly Sales by Region",
                    "width": 800,
                    "height": 600
                }
            }
        }


class ChartGenerationResponse(BaseModel):
    """Response model for chart generation."""
    chart_id: str = Field(..., description="Unique chart identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    chart_type: str = Field(..., description="Chart type")
    chart_config: Dict[str, Any] = Field(..., description="Chart configuration")
    chart_data: Dict[str, Any] = Field(..., description="Processed chart data")
    chart_url: str = Field(..., description="URL to generated chart")
    thumbnail_url: str = Field(..., description="URL to chart thumbnail")
    metadata: Dict[str, Any] = Field(..., description="Chart metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    created_at: str = Field(..., description="Chart creation timestamp")        schema_extra = {
            "example": {
                "chart_id": "chart_123456789",
                "dataset_id": "sales_data_2024",
                "chart_type": "bar",
                "chart_config": {
                    "x_axis": "month",
                    "y_axis": "sales",
                    "group_by": "region",
                    "aggregation": "sum",
                    "title": "Monthly Sales by Region",
                    "color_scheme": "blue"
                },
                "chart_data": {
                    "categories": ["Jan", "Feb"],
                    "series": [
                        {"name": "North", "data": [1000, 1200]},
                        {"name": "South", "data": [800, 900]}
                    ]
                },
                "chart_url": "https://api.example.com/charts/chart_123456789/view",
                "thumbnail_url": "https://api.example.com/charts/chart_123456789/thumbnail",
                "metadata": {
                    "data_points": 4,
                    "series_count": 2,
                    "interactive": True,
                    "export_formats": ["png", "svg", "pdf"]
                },
                "processing_time_ms": 567.8,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class DashboardRequest(BaseModel):
    """Request model for dashboard creation."""
    dashboard_name: str = Field(..., description="Dashboard name")
    description: str = Field(..., description="Dashboard description")
    layout: str = Field(default="grid", description="Dashboard layout type")
    widgets: List[Dict[str, Any]] = Field(..., description="Dashboard widgets")
    refresh_interval: Optional[int] = Field(default=300, description="Refresh interval in seconds")
    filters: Optional[List[Dict[str, Any]]] = Field(default=None, description="Global dashboard filters")
    styling: Optional[Dict[str, Any]] = Field(default=None, description="Dashboard styling")        schema_extra = {
            "example": {
                "dashboard_name": "Sales Performance Dashboard",
                "description": "Real-time sales performance metrics and trends",
                "layout": "grid",
                "widgets": [
                    {
                        "widget_type": "chart",
                        "chart_type": "line",
                        "title": "Sales Trend",
                        "dataset_id": "sales_data_2024",
                        "position": {"x": 0, "y": 0, "width": 6, "height": 4},
                        "config": {
                            "x_axis": "date",
                            "y_axis": "sales",
                            "time_series": True
                        }
                    },
                    {
                        "widget_type": "kpi",
                        "title": "Total Sales",
                        "dataset_id": "sales_data_2024",
                        "position": {"x": 6, "y": 0, "width": 3, "height": 2},
                        "config": {
                            "metric": "sum",
                            "column": "sales",
                            "format": "currency"
                        }
                    }
                ],
                "refresh_interval": 300,
                "filters": [
                    {"column": "date", "type": "date_range", "default": "last_30_days"}
                ],
                "styling": {
                    "theme": "dark",
                    "color_scheme": "corporate"
                }
            }
        }


class DashboardResponse(BaseModel):
    """Response model for dashboard creation."""
    dashboard_id: str = Field(..., description="Unique dashboard identifier")
    dashboard_name: str = Field(..., description="Dashboard name")
    description: str = Field(..., description="Dashboard description")
    layout: str = Field(..., description="Dashboard layout")
    widgets: List[Dict[str, Any]] = Field(..., description="Dashboard widgets")
    dashboard_url: str = Field(..., description="URL to access dashboard")
    embed_url: str = Field(..., description="URL for embedding dashboard")
    refresh_interval: int = Field(..., description="Refresh interval in seconds")
    filters: List[Dict[str, Any]] = Field(..., description="Dashboard filters")
    metadata: Dict[str, Any] = Field(..., description="Dashboard metadata")
    created_by: str = Field(..., description="Dashboard creator")
    created_at: str = Field(..., description="Creation timestamp")        schema_extra = {
            "example": {
                "dashboard_id": "dash_123456789",
                "dashboard_name": "Sales Performance Dashboard",
                "description": "Real-time sales performance metrics and trends",
                "layout": "grid",
                "widgets": [
                    {
                        "widget_id": "widget_001",
                        "widget_type": "chart",
                        "title": "Sales Trend",
                        "status": "active",
                        "last_updated": "2024-01-15T10:30:00Z"
                    }
                ],
                "dashboard_url": "https://api.example.com/dashboards/dash_123456789/view",
                "embed_url": "https://api.example.com/dashboards/dash_123456789/embed",
                "refresh_interval": 300,
                "filters": [
                    {"column": "date", "type": "date_range", "default": "last_30_days"}
                ],
                "metadata": {
                    "widget_count": 5,
                    "data_sources": 3,
                    "last_refresh": "2024-01-15T10:30:00Z",
                    "viewers": 12
                },
                "created_by": "user_123",
                "created_at": "2024-01-15T09:00:00Z"
            }
        }


class ReportGenerationRequest(BaseModel):
    """Request model for report generation."""
    report_name: str = Field(..., description="Report name")
    report_type: str = Field(..., description="Type of report (summary, detailed, executive)")
    datasets: List[str] = Field(..., description="Dataset identifiers")
    sections: List[Dict[str, Any]] = Field(..., description="Report sections")
    format: str = Field(default="pdf", description="Report format (pdf, html, excel)")
    template: Optional[str] = Field(default=None, description="Report template")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Report parameters")        schema_extra = {
            "example": {
                "report_name": "Monthly Sales Report",
                "report_type": "summary",
                "datasets": ["sales_data_2024", "customer_data_2024"],
                "sections": [
                    {
                        "section_type": "summary",
                        "title": "Executive Summary",
                        "content": "kpi_cards"
                    },
                    {
                        "section_type": "chart",
                        "title": "Sales Trends",
                        "chart_type": "line",
                        "dataset_id": "sales_data_2024"
                    },
                    {
                        "section_type": "table",
                        "title": "Top Products",
                        "dataset_id": "sales_data_2024",
                        "columns": ["product", "sales", "units"]
                    }
                ],
                "format": "pdf",
                "template": "corporate_template",
                "parameters": {
                    "date_range": "2024-01-01 to 2024-01-31",
                    "include_forecasts": True
                }
            }
        }


class ReportGenerationResponse(BaseModel):
    """Response model for report generation."""
    report_id: str = Field(..., description="Unique report identifier")
    report_name: str = Field(..., description="Report name")
    report_type: str = Field(..., description="Report type")
    format: str = Field(..., description="Report format")
    status: str = Field(..., description="Report generation status")
    report_url: str = Field(..., description="URL to download report")
    preview_url: str = Field(..., description="URL to preview report")
    metadata: Dict[str, Any] = Field(..., description="Report metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    created_at: str = Field(..., description="Report creation timestamp")


class VisualizationTemplateRequest(BaseModel):
    """Request model for visualization template."""
    template_name: str = Field(..., description="Template name")
    template_type: str = Field(..., description="Template type (chart, dashboard, report)")
    configuration: Dict[str, Any] = Field(..., description="Template configuration")
    description: str = Field(..., description="Template description")
    tags: List[str] = Field(default_factory=list, description="Template tags")        schema_extra = {
            "example": {
                "template_name": "Sales Performance Chart",
                "template_type": "chart",
                "configuration": {
                    "chart_type": "bar",
                    "styling": {
                        "color_scheme": "blue",
                        "title_font_size": 16,
                        "axis_font_size": 12
                    },
                    "default_columns": {
                        "x_axis": "date",
                        "y_axis": "sales"
                    }
                },
                "description": "Standard bar chart template for sales performance",
                "tags": ["sales", "performance", "bar_chart"]
            }
        }


# API Endpoints

@router.post(
    "/charts/generate",
    response_model=ChartGenerationResponse,
    summary="Generate chart",
    description="Generate a chart from data with specified visualization type and styling"
)
@require_permissions(["visualization:create"])
async def generate_chart(
    request: ChartGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate a chart from data."""
    try:
        logger.info(f"Generating {request.chart_type} chart for dataset {request.dataset_id}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Process and aggregate the data
        # 2. Apply filters and transformations
        # 3. Generate chart using visualization library
        # 4. Store chart configuration and data
        # 5. Return chart URLs
        
        # Process data for chart
        processed_data = _process_chart_data(
            request.data, 
            request.x_axis, 
            request.y_axis, 
            request.group_by, 
            request.aggregation
        )
        
        # Generate chart configuration
        chart_config = {
            "x_axis": request.x_axis,
            "y_axis": request.y_axis,
            "group_by": request.group_by,
            "aggregation": request.aggregation,
            "title": request.styling.get("title", f"{request.chart_type.title()} Chart") if request.styling else f"{request.chart_type.title()} Chart",
            "color_scheme": request.styling.get("color_scheme", "default") if request.styling else "default"
        }
        
        # Generate URLs
        chart_id = str(uuid4())
        chart_url = f"https://api.example.com/charts/{chart_id}/view"
        thumbnail_url = f"https://api.example.com/charts/{chart_id}/thumbnail"
        
        # Chart metadata
        metadata = {
            "data_points": len(request.data),
            "series_count": len(processed_data.get("series", [])),
            "interactive": True,
            "export_formats": ["png", "svg", "pdf", "excel"]
        }
        
        return ChartGenerationResponse(
            chart_id=chart_id,
            dataset_id=request.dataset_id,
            chart_type=request.chart_type,
            chart_config=chart_config,
            chart_data=processed_data,
            chart_url=chart_url,
            thumbnail_url=thumbnail_url,
            metadata=metadata,
            processing_time_ms=567.8,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Chart generation failed"
        )


@router.post(
    "/dashboards",
    response_model=DashboardResponse,
    summary="Create dashboard",
    description="Create an interactive dashboard with multiple widgets"
)
@require_permissions(["dashboard:create"])
async def create_dashboard(
    request: DashboardRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new dashboard."""
    try:
        logger.info(f"Creating dashboard: {request.dashboard_name}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Validate widget configurations
        # 2. Create dashboard layout
        # 3. Initialize widgets
        # 4. Set up refresh mechanisms
        # 5. Generate dashboard URLs
        
        # Process widgets
        processed_widgets = []
        for widget in request.widgets:
            processed_widget = {
                "widget_id": str(uuid4()),
                "widget_type": widget["widget_type"],
                "title": widget["title"],
                "status": "active",
                "last_updated": datetime.now().isoformat(),
                "config": widget.get("config", {}),
                "position": widget.get("position", {})
            }
            processed_widgets.append(processed_widget)
        
        # Generate URLs
        dashboard_id = str(uuid4())
        dashboard_url = f"https://api.example.com/dashboards/{dashboard_id}/view"
        embed_url = f"https://api.example.com/dashboards/{dashboard_id}/embed"
        
        # Dashboard metadata
        metadata = {
            "widget_count": len(processed_widgets),
            "data_sources": len(set(w.get("dataset_id") for w in request.widgets if w.get("dataset_id"))),
            "last_refresh": datetime.now().isoformat(),
            "viewers": 0
        }
        
        return DashboardResponse(
            dashboard_id=dashboard_id,
            dashboard_name=request.dashboard_name,
            description=request.description,
            layout=request.layout,
            widgets=processed_widgets,
            dashboard_url=dashboard_url,
            embed_url=embed_url,
            refresh_interval=request.refresh_interval,
            filters=request.filters or [],
            metadata=metadata,
            created_by=current_user["user_id"],
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Dashboard creation failed"
        )


@router.post(
    "/reports/generate",
    response_model=ReportGenerationResponse,
    summary="Generate report",
    description="Generate a formatted report with charts, tables, and analysis"
)
@require_permissions(["report:create"])
async def generate_report(
    request: ReportGenerationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate a formatted report."""
    try:
        logger.info(f"Generating {request.report_type} report: {request.report_name}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Process report sections
        # 2. Generate charts and tables
        # 3. Apply report template
        # 4. Export to specified format
        # 5. Store report file
        
        # Generate URLs
        report_id = str(uuid4())
        report_url = f"https://api.example.com/reports/{report_id}/download"
        preview_url = f"https://api.example.com/reports/{report_id}/preview"
        
        # Report metadata
        metadata = {
            "sections_count": len(request.sections),
            "datasets_count": len(request.datasets),
            "file_size": "2.5MB",
            "pages": 12,
            "generated_by": current_user["user_id"]
        }
        
        return ReportGenerationResponse(
            report_id=report_id,
            report_name=request.report_name,
            report_type=request.report_type,
            format=request.format,
            status="completed",
            report_url=report_url,
            preview_url=preview_url,
            metadata=metadata,
            processing_time_ms=5432.1,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Report generation failed"
        )


@router.get(
    "/charts/{chart_id}",
    response_model=ChartGenerationResponse,
    summary="Get chart",
    description="Get chart by ID with configuration and data"
)
@require_permissions(["visualization:read"])
async def get_chart(
    chart_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get chart by ID."""
    try:
        logger.info(f"Retrieving chart: {chart_id}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database for chart
        # 2. Return chart configuration and data
        
        # Mock response
        return ChartGenerationResponse(
            chart_id=chart_id,
            dataset_id="sample_dataset",
            chart_type="bar",
            chart_config={
                "x_axis": "month",
                "y_axis": "sales",
                "title": "Monthly Sales"
            },
            chart_data={
                "categories": ["Jan", "Feb", "Mar"],
                "series": [{"name": "Sales", "data": [1000, 1200, 1100]}]
            },
            chart_url=f"https://api.example.com/charts/{chart_id}/view",
            thumbnail_url=f"https://api.example.com/charts/{chart_id}/thumbnail",
            metadata={
                "data_points": 3,
                "series_count": 1,
                "interactive": True,
                "export_formats": ["png", "svg", "pdf"]
            },
            processing_time_ms=567.8,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve chart: {e}")
        raise HTTPException(
            status_code=404,
            detail="Chart not found"
        )


@router.get(
    "/dashboards/{dashboard_id}",
    response_model=DashboardResponse,
    summary="Get dashboard",
    description="Get dashboard by ID with widgets and configuration"
)
@require_permissions(["dashboard:read"])
async def get_dashboard(
    dashboard_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get dashboard by ID."""
    try:
        logger.info(f"Retrieving dashboard: {dashboard_id}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database for dashboard
        # 2. Return dashboard configuration and widgets
        
        # Mock response
        return DashboardResponse(
            dashboard_id=dashboard_id,
            dashboard_name="Sales Performance Dashboard",
            description="Real-time sales performance metrics and trends",
            layout="grid",
            widgets=[
                {
                    "widget_id": "widget_001",
                    "widget_type": "chart",
                    "title": "Sales Trend",
                    "status": "active",
                    "last_updated": datetime.now().isoformat()
                }
            ],
            dashboard_url=f"https://api.example.com/dashboards/{dashboard_id}/view",
            embed_url=f"https://api.example.com/dashboards/{dashboard_id}/embed",
            refresh_interval=300,
            filters=[],
            metadata={
                "widget_count": 1,
                "data_sources": 1,
                "last_refresh": datetime.now().isoformat(),
                "viewers": 5
            },
            created_by="user_123",
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve dashboard: {e}")
        raise HTTPException(
            status_code=404,
            detail="Dashboard not found"
        )


@router.get(
    "/charts",
    summary="List charts",
    description="List all charts with filtering options"
)
@require_permissions(["visualization:read"])
async def list_charts(
    dataset_id: Optional[str] = None,
    chart_type: Optional[str] = None,
    created_by: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List charts with optional filtering."""
    try:
        logger.info("Listing charts")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database with filters
        # 2. Return paginated results
        
        # Mock response
        charts = [
            {
                "chart_id": "chart_123456789",
                "dataset_id": "sales_data_2024",
                "chart_type": "bar",
                "title": "Monthly Sales",
                "created_by": "user_123",
                "created_at": "2024-01-15T10:30:00Z",
                "last_updated": "2024-01-15T10:30:00Z",
                "views": 25
            },
            {
                "chart_id": "chart_987654321",
                "dataset_id": "customer_data_2024",
                "chart_type": "pie",
                "title": "Customer Distribution",
                "created_by": "user_456",
                "created_at": "2024-01-14T14:20:00Z",
                "last_updated": "2024-01-14T14:20:00Z",
                "views": 18
            }
        ]
        
        # Apply filters
        if dataset_id:
            charts = [c for c in charts if c["dataset_id"] == dataset_id]
        if chart_type:
            charts = [c for c in charts if c["chart_type"] == chart_type]
        if created_by:
            charts = [c for c in charts if c["created_by"] == created_by]
        
        # Apply pagination
        total_count = len(charts)
        paginated_charts = charts[offset:offset + limit]
        
        return {
            "charts": paginated_charts,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list charts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list charts"
        )


@router.get(
    "/dashboards",
    summary="List dashboards",
    description="List all dashboards with filtering options"
)
@require_permissions(["dashboard:read"])
async def list_dashboards(
    created_by: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List dashboards with optional filtering."""
    try:
        logger.info("Listing dashboards")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database with filters
        # 2. Return paginated results
        
        # Mock response
        dashboards = [
            {
                "dashboard_id": "dash_123456789",
                "dashboard_name": "Sales Performance Dashboard",
                "description": "Real-time sales performance metrics",
                "created_by": "user_123",
                "created_at": "2024-01-15T09:00:00Z",
                "last_updated": "2024-01-15T10:30:00Z",
                "widget_count": 5,
                "viewers": 12
            },
            {
                "dashboard_id": "dash_987654321",
                "dashboard_name": "Customer Analytics Dashboard",
                "description": "Customer behavior and demographics",
                "created_by": "user_456",
                "created_at": "2024-01-14T11:00:00Z",
                "last_updated": "2024-01-14T15:20:00Z",
                "widget_count": 8,
                "viewers": 7
            }
        ]
        
        # Apply filters
        if created_by:
            dashboards = [d for d in dashboards if d["created_by"] == created_by]
        
        # Apply pagination
        total_count = len(dashboards)
        paginated_dashboards = dashboards[offset:offset + limit]
        
        return {
            "dashboards": paginated_dashboards,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list dashboards: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list dashboards"
        )


@router.post(
    "/templates",
    summary="Create visualization template",
    description="Create a reusable visualization template"
)
@require_permissions(["template:create"])
async def create_visualization_template(
    request: VisualizationTemplateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a visualization template."""
    try:
        logger.info(f"Creating visualization template: {request.template_name}")
        
        # Mock implementation - in real implementation, this would:
        # 1. Validate template configuration
        # 2. Store template in database
        # 3. Make template available for use
        
        template_id = str(uuid4())
        
        return {
            "template_id": template_id,
            "template_name": request.template_name,
            "template_type": request.template_type,
            "configuration": request.configuration,
            "description": request.description,
            "tags": request.tags,
            "created_by": current_user["user_id"],
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        
    except Exception as e:
        logger.error(f"Failed to create visualization template: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create visualization template"
        )


@router.get(
    "/templates",
    summary="List visualization templates",
    description="List all available visualization templates"
)
@require_permissions(["template:read"])
async def list_visualization_templates(
    template_type: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """List visualization templates."""
    try:
        logger.info("Listing visualization templates")
        
        # Mock implementation - in real implementation, this would:
        # 1. Query the database with filters
        # 2. Return paginated results
        
        # Mock response
        templates = [
            {
                "template_id": "tmpl_123456789",
                "template_name": "Sales Performance Chart",
                "template_type": "chart",
                "description": "Standard bar chart template for sales performance",
                "tags": ["sales", "performance", "bar_chart"],
                "created_by": "admin",
                "created_at": "2024-01-01T00:00:00Z",
                "usage_count": 25
            },
            {
                "template_id": "tmpl_987654321",
                "template_name": "Executive Dashboard",
                "template_type": "dashboard",
                "description": "Executive-level dashboard with key metrics",
                "tags": ["executive", "kpi", "dashboard"],
                "created_by": "admin",
                "created_at": "2024-01-01T00:00:00Z",
                "usage_count": 12
            }
        ]
        
        # Apply filters
        if template_type:
            templates = [t for t in templates if t["template_type"] == template_type]
        if tags:
            tag_filter = tags.split(",")
            templates = [t for t in templates if any(tag in t["tags"] for tag in tag_filter)]
        
        # Apply pagination
        total_count = len(templates)
        paginated_templates = templates[offset:offset + limit]
        
        return {
            "templates": paginated_templates,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to list visualization templates: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list visualization templates"
        )


# Helper functions

def _process_chart_data(data: List[Dict[str, Any]], x_axis: str, y_axis: str, group_by: str = None, aggregation: str = None) -> Dict[str, Any]:
    """Process data for chart generation."""
    if not data:
        return {"categories": [], "series": []}
    
    # Simple processing for mock implementation
    categories = list(set(record.get(x_axis) for record in data))
    categories.sort()
    
    if group_by:
        # Group by specified column
        groups = {}
        for record in data:
            group_value = record.get(group_by)
            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(record)
        
        # Create series for each group
        series = []
        for group_name, group_data in groups.items():
            series_data = []
            for category in categories:
                category_data = [r for r in group_data if r.get(x_axis) == category]
                if category_data and y_axis:
                    if aggregation == "sum":
                        value = sum(r.get(y_axis, 0) for r in category_data)
                    elif aggregation == "avg":
                        values = [r.get(y_axis, 0) for r in category_data]
                        value = sum(values) / len(values) if values else 0
                    elif aggregation == "count":
                        value = len(category_data)
                    else:
                        value = category_data[0].get(y_axis, 0)
                else:
                    value = 0
                series_data.append(value)
            
            series.append({
                "name": group_name,
                "data": series_data
            })
    else:
        # Single series
        series_data = []
        for category in categories:
            category_data = [r for r in data if r.get(x_axis) == category]
            if category_data and y_axis:
                if aggregation == "sum":
                    value = sum(r.get(y_axis, 0) for r in category_data)
                elif aggregation == "avg":
                    values = [r.get(y_axis, 0) for r in category_data]
                    value = sum(values) / len(values) if values else 0
                elif aggregation == "count":
                    value = len(category_data)
                else:
                    value = category_data[0].get(y_axis, 0)
            else:
                value = len(category_data)
            series_data.append(value)
        
        series = [{
            "name": y_axis or "Count",
            "data": series_data
        }]
    
    return {
        "categories": categories,
        "series": series
    }