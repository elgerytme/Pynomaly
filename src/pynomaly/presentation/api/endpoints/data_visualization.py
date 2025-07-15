"""
Data Visualization API Endpoints

RESTful endpoints for creating interactive data visualizations and dashboards.
"""

from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from uuid import UUID
from datetime import datetime
import logging
import io
import base64

from ....domain.entities.user import User
from ....application.dto.visualization_dto import (
    VisualizationRequestDTO,
    VisualizationResponseDTO,
    DashboardRequestDTO,
    DashboardResponseDTO,
    PlotConfigurationDTO,
    ChartDataRequestDTO,
    ChartDataResponseDTO,
    InteractiveVisualizationRequestDTO,
    InteractiveVisualizationResponseDTO
)
from ....application.use_cases.data_visualization import (
    CreateVisualizationUseCase,
    CreateDashboardUseCase,
    GenerateChartDataUseCase
)
from ....shared.dependencies import get_current_user, get_visualization_use_cases
from ....shared.monitoring import metrics, monitor_endpoint
from ....shared.error_handling import APIError, ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/visualizations", tags=["Data Visualization"])


@router.post(
    "/create",
    response_model=VisualizationResponseDTO,
    summary="Create Data Visualization",
    description="""
    Create interactive data visualizations from dataset features with support
    for multiple chart types and customization options.
    
    Supports visualization types:
    - Statistical plots (histograms, box plots, scatter plots)
    - Time series plots (line charts, area charts)
    - Correlation heatmaps and matrices
    - Distribution plots (density, QQ plots)
    - Categorical plots (bar charts, pie charts)
    - Multi-dimensional plots (3D scatter, parallel coordinates)
    
    **Features:**
    - Interactive plotting with zoom, pan, and hover
    - Customizable themes and styling
    - Export to multiple formats (PNG, SVG, PDF, HTML)
    - Real-time data updates
    - Responsive design for web integration
    """,
    responses={
        200: {"description": "Visualization created successfully"},
        400: {"description": "Invalid visualization parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Incompatible data for visualization type"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("visualization_creation")
async def create_visualization(
    request: VisualizationRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: CreateVisualizationUseCase = Depends(get_visualization_use_cases)
) -> VisualizationResponseDTO:
    """
    Create a data visualization.
    
    Args:
        request: Visualization creation request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Visualization creation use case
        
    Returns:
        Visualization creation results
        
    Raises:
        HTTPException: If visualization creation fails
    """
    try:
        logger.info(
            f"Creating {request.visualization_type} visualization for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.visualization_creation_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Visualization creation failed: {result.error_message}"
            )
        
        metrics.successful_visualizations.inc()
        logger.info(f"Visualization created successfully: {result.visualization_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.visualization_creation_failures.inc()
        logger.error(f"Unexpected error in visualization creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during visualization creation"
        )


@router.post(
    "/dashboard",
    response_model=DashboardResponseDTO,
    summary="Create Interactive Dashboard",
    description="""
    Create interactive dashboards combining multiple visualizations with
    filters, controls, and real-time data updates.
    
    Dashboard features:
    - Multi-panel layouts with drag-and-drop interface
    - Interactive filters and controls
    - Cross-filtering between visualizations
    - Real-time data refresh capabilities
    - Export and sharing functionality
    - Responsive design for mobile and desktop
    
    **Components:**
    - Charts and plots with synchronized axes
    - Statistical summary tables
    - Key performance indicators (KPIs)
    - Interactive data tables
    - Custom HTML components
    """,
    responses={
        200: {"description": "Dashboard created successfully"},
        400: {"description": "Invalid dashboard configuration"},
        404: {"description": "Dataset or visualizations not found"},
        422: {"description": "Incompatible visualization combination"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("dashboard_creation")
async def create_dashboard(
    request: DashboardRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: CreateDashboardUseCase = Depends(get_visualization_use_cases)
) -> DashboardResponseDTO:
    """
    Create an interactive dashboard.
    
    Args:
        request: Dashboard creation request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Dashboard creation use case
        
    Returns:
        Dashboard creation results
        
    Raises:
        HTTPException: If dashboard creation fails
    """
    try:
        logger.info(
            f"Creating dashboard with {len(request.visualizations)} visualizations "
            f"for user {current_user.id}"
        )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.dashboard_creation_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Dashboard creation failed: {result.error_message}"
            )
        
        metrics.successful_dashboards.inc()
        logger.info(f"Dashboard created successfully: {result.dashboard_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.dashboard_creation_failures.inc()
        logger.error(f"Unexpected error in dashboard creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during dashboard creation"
        )


@router.post(
    "/chart-data",
    response_model=ChartDataResponseDTO,
    summary="Generate Chart Data",
    description="""
    Generate optimized chart data for specific visualization types with
    support for data sampling, aggregation, and formatting.
    
    Data optimization features:
    - Intelligent data sampling for large datasets
    - Automatic aggregation for time series data
    - Data type conversion and formatting
    - Outlier handling and filtering
    - Memory-efficient data structures
    
    **Output Formats:**
    - JSON for web applications
    - CSV for data analysis tools
    - Plotly/D3.js compatible formats
    - Custom format specifications
    """,
    responses={
        200: {"description": "Chart data generated successfully"},
        400: {"description": "Invalid data request parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Data incompatible with chart type"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("chart_data_generation")
async def generate_chart_data(
    request: ChartDataRequestDTO,
    current_user: User = Depends(get_current_user),
    use_case: GenerateChartDataUseCase = Depends(get_visualization_use_cases)
) -> ChartDataResponseDTO:
    """
    Generate chart data for visualization.
    
    Args:
        request: Chart data request
        current_user: Authenticated user
        use_case: Chart data generation use case
        
    Returns:
        Chart data response
        
    Raises:
        HTTPException: If data generation fails
    """
    try:
        logger.info(
            f"Generating chart data for dataset {request.dataset_id} "
            f"chart type {request.chart_type}"
        )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.chart_data_generation_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Chart data generation failed: {result.error_message}"
            )
        
        metrics.successful_chart_data_generations.inc()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.chart_data_generation_failures.inc()
        logger.error(f"Unexpected error in chart data generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during chart data generation"
        )


@router.post(
    "/interactive",
    response_model=InteractiveVisualizationResponseDTO,
    summary="Create Interactive Visualization",
    description="""
    Create advanced interactive visualizations with user controls,
    animations, and real-time data updates.
    
    Interactive features:
    - User controls (sliders, dropdowns, date pickers)
    - Animations and transitions
    - Brush and zoom interactions
    - Linked views and cross-filtering
    - Real-time data streaming
    - Custom event handlers
    
    **Advanced Capabilities:**
    - WebGL-accelerated rendering for large datasets
    - Custom D3.js integration
    - Plugin architecture for extensions
    - Mobile-responsive touch interactions
    """,
    responses={
        200: {"description": "Interactive visualization created successfully"},
        400: {"description": "Invalid interaction parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Interaction type not supported for data"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("interactive_visualization_creation")
async def create_interactive_visualization(
    request: InteractiveVisualizationRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: CreateVisualizationUseCase = Depends(get_visualization_use_cases)
) -> InteractiveVisualizationResponseDTO:
    """
    Create an interactive visualization.
    
    Args:
        request: Interactive visualization request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Visualization creation use case
        
    Returns:
        Interactive visualization results
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        logger.info(
            f"Creating interactive visualization for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Convert to general visualization request
        vis_request = VisualizationRequestDTO(
            dataset_id=request.dataset_id,
            visualization_type="interactive",
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            plot_configuration=PlotConfigurationDTO(
                title=request.title,
                theme=request.theme,
                interactive=True,
                custom_options={
                    "interactions": request.interactions,
                    "controls": request.controls,
                    "real_time_updates": request.real_time_updates,
                    "animation_enabled": request.animation_enabled
                }
            )
        )
        
        result = await use_case.execute(vis_request, current_user.id)
        
        if result.status == "failed":
            metrics.interactive_visualization_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Interactive visualization creation failed: {result.error_message}"
            )
        
        metrics.successful_interactive_visualizations.inc()
        
        return InteractiveVisualizationResponseDTO(
            visualization_id=result.visualization_id,
            dataset_id=request.dataset_id,
            interactive_config=request.interactions,
            visualization_html=result.visualization_data.get("html", ""),
            javascript_code=result.visualization_data.get("javascript", ""),
            css_styles=result.visualization_data.get("css", ""),
            api_endpoints=result.visualization_data.get("api_endpoints", []),
            real_time_config=result.visualization_data.get("real_time_config", {}),
            execution_time_seconds=result.execution_time_seconds,
            created_at=result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.interactive_visualization_failures.inc()
        logger.error(f"Unexpected error in interactive visualization creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during interactive visualization creation"
        )


@router.get(
    "/{visualization_id}",
    response_model=VisualizationResponseDTO,
    summary="Get Visualization",
    description="""
    Retrieve a specific visualization with its configuration and data.
    """,
    responses={
        200: {"description": "Visualization retrieved successfully"},
        404: {"description": "Visualization not found"},
        403: {"description": "Access denied to visualization"},
        500: {"description": "Internal server error"}
    }
)
async def get_visualization(
    visualization_id: UUID,
    current_user: User = Depends(get_current_user)
) -> VisualizationResponseDTO:
    """
    Get visualization by ID.
    
    Args:
        visualization_id: Visualization identifier
        current_user: Authenticated user
        
    Returns:
        Visualization data
        
    Raises:
        HTTPException: If visualization not found or access denied
    """
    try:
        logger.info(f"Retrieving visualization {visualization_id}")
        
        # This would be implemented with proper repository queries
        # For now, return a placeholder response
        return VisualizationResponseDTO(
            visualization_id=visualization_id,
            dataset_id=UUID("12345678-1234-5678-9012-123456789012"),
            visualization_type="scatter",
            visualization_data={"placeholder": "data"},
            plot_configuration={},
            status="completed",
            execution_time_seconds=1.5,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving visualization"
        )


@router.get(
    "/{visualization_id}/export",
    summary="Export Visualization",
    description="""
    Export visualization in various formats (PNG, SVG, PDF, HTML, JSON).
    """,
    responses={
        200: {"description": "Visualization exported successfully"},
        404: {"description": "Visualization not found"},
        400: {"description": "Invalid export format"},
        500: {"description": "Internal server error"}
    }
)
async def export_visualization(
    visualization_id: UUID,
    format: str = Query(..., description="Export format (png, svg, pdf, html, json)"),
    width: Optional[int] = Query(None, description="Export width in pixels"),
    height: Optional[int] = Query(None, description="Export height in pixels"),
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    """
    Export visualization in specified format.
    
    Args:
        visualization_id: Visualization identifier
        format: Export format
        width: Optional width
        height: Optional height
        current_user: Authenticated user
        
    Returns:
        Exported visualization file
        
    Raises:
        HTTPException: If export fails
    """
    try:
        logger.info(f"Exporting visualization {visualization_id} as {format}")
        
        # Validate format
        allowed_formats = ["png", "svg", "pdf", "html", "json"]
        if format.lower() not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Allowed formats: {', '.join(allowed_formats)}"
            )
        
        # This would be implemented with actual export logic
        # For now, return a placeholder response
        content = f"Exported {visualization_id} as {format}"
        
        return StreamingResponse(
            io.StringIO(content),
            media_type=f"application/{format}",
            headers={"Content-Disposition": f"attachment; filename=visualization_{visualization_id}.{format}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error exporting visualization"
        )


@router.get(
    "/dashboard/{dashboard_id}",
    response_model=DashboardResponseDTO,
    summary="Get Dashboard",
    description="""
    Retrieve a specific dashboard with all its visualizations and configuration.
    """,
    responses={
        200: {"description": "Dashboard retrieved successfully"},
        404: {"description": "Dashboard not found"},
        403: {"description": "Access denied to dashboard"},
        500: {"description": "Internal server error"}
    }
)
async def get_dashboard(
    dashboard_id: UUID,
    current_user: User = Depends(get_current_user)
) -> DashboardResponseDTO:
    """
    Get dashboard by ID.
    
    Args:
        dashboard_id: Dashboard identifier
        current_user: Authenticated user
        
    Returns:
        Dashboard data
        
    Raises:
        HTTPException: If dashboard not found or access denied
    """
    try:
        logger.info(f"Retrieving dashboard {dashboard_id}")
        
        # This would be implemented with proper repository queries
        # For now, return a placeholder response
        return DashboardResponseDTO(
            dashboard_id=dashboard_id,
            name="Sample Dashboard",
            layout_config={},
            visualization_ids=[],
            filter_config={},
            status="completed",
            execution_time_seconds=2.0,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error retrieving dashboard"
        )


@router.get(
    "/list",
    response_model=List[Dict[str, Any]],
    summary="List User Visualizations",
    description="""
    List all visualizations created by the current user with filtering and pagination.
    """,
    responses={
        200: {"description": "Visualizations listed successfully"},
        400: {"description": "Invalid query parameters"},
        500: {"description": "Internal server error"}
    }
)
async def list_visualizations(
    dataset_id: Optional[UUID] = Query(None, description="Filter by dataset"),
    visualization_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100),
    offset: int = Query(0, description="Results offset", ge=0),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List user's visualizations.
    
    Args:
        dataset_id: Optional dataset filter
        visualization_type: Optional type filter
        limit: Maximum results
        offset: Results offset
        current_user: Authenticated user
        
    Returns:
        List of visualizations
    """
    try:
        logger.info(f"Listing visualizations for user {current_user.id}")
        
        # This would be implemented with proper repository queries
        # For now, return a placeholder response
        return [
            {
                "visualization_id": "vis_1",
                "dataset_id": dataset_id or "dataset_1",
                "type": "scatter",
                "title": "Sample Scatter Plot",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
        
    except Exception as e:
        logger.error(f"Error listing visualizations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error listing visualizations"
        )


@router.delete(
    "/{visualization_id}",
    summary="Delete Visualization",
    description="""
    Delete a visualization. This action cannot be undone.
    """,
    responses={
        200: {"description": "Visualization deleted successfully"},
        404: {"description": "Visualization not found"},
        403: {"description": "Access denied"},
        500: {"description": "Internal server error"}
    }
)
async def delete_visualization(
    visualization_id: UUID,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a visualization.
    
    Args:
        visualization_id: Visualization identifier
        current_user: Authenticated user
        
    Returns:
        Deletion confirmation
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        logger.info(f"Deleting visualization {visualization_id} by user {current_user.id}")
        
        # This would be implemented with proper business logic
        # For now, return a success response
        return {"message": f"Visualization {visualization_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting visualization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error deleting visualization"
        )