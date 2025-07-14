"""
Data Visualization Data Transfer Objects

DTOs for data visualization requests and responses.
"""

from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class VisualizationTypeEnum(str, Enum):
    """Supported visualization types."""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation_matrix"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    PIE = "pie"
    DONUT = "donut"
    AREA = "area"
    BUBBLE = "bubble"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR = "radar"
    SANKEY = "sankey"
    FUNNEL = "funnel"
    WATERFALL = "waterfall"
    CANDLESTICK = "candlestick"
    PLOTLY_3D = "plotly_3d"
    INTERACTIVE = "interactive"


class ChartThemeEnum(str, Enum):
    """Chart theme options."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    COLORFUL = "colorful"
    PROFESSIONAL = "professional"
    SCIENTIFIC = "scientific"
    PRESENTATION = "presentation"


class ExportFormatEnum(str, Enum):
    """Export format options."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class InteractionTypeEnum(str, Enum):
    """Interaction types for visualizations."""
    ZOOM = "zoom"
    PAN = "pan"
    BRUSH = "brush"
    HOVER = "hover"
    CLICK = "click"
    SELECTION = "selection"
    FILTER = "filter"
    CROSSFILTER = "crossfilter"
    ANIMATION = "animation"


# Base Visualization DTOs

class PlotConfigurationDTO(BaseModel):
    """Plot configuration options."""
    title: Optional[str] = Field(None, description="Plot title")
    x_axis_label: Optional[str] = Field(None, description="X-axis label")
    y_axis_label: Optional[str] = Field(None, description="Y-axis label")
    width: int = Field(800, description="Plot width in pixels")
    height: int = Field(600, description="Plot height in pixels")
    theme: ChartThemeEnum = Field(default=ChartThemeEnum.DEFAULT, description="Chart theme")
    color_palette: Optional[List[str]] = Field(None, description="Custom color palette")
    show_legend: bool = Field(True, description="Show legend")
    show_grid: bool = Field(True, description="Show grid lines")
    interactive: bool = Field(True, description="Enable interactivity")
    custom_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom plot options")
    
    @validator("width", "height")
    def validate_dimensions(cls, v):
        if v < 100 or v > 4000:
            raise ValueError("Dimensions must be between 100 and 4000 pixels")
        return v


class VisualizationRequestDTO(BaseModel):
    """Request for creating a visualization."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    visualization_type: VisualizationTypeEnum = Field(..., description="Type of visualization")
    feature_columns: List[str] = Field(..., description="Features to visualize")
    target_column: Optional[str] = Field(None, description="Target column for colored/grouped visualizations")
    plot_configuration: Optional[PlotConfigurationDTO] = Field(
        default_factory=PlotConfigurationDTO,
        description="Plot configuration"
    )
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data filters")
    aggregation: Optional[Dict[str, str]] = Field(None, description="Data aggregation settings")
    
    @validator("feature_columns")
    def validate_feature_columns(cls, v):
        if not v:
            raise ValueError("At least one feature column required")
        return v


class VisualizationResponseDTO(BaseModel):
    """Response for visualization creation."""
    visualization_id: UUID = Field(..., description="Visualization identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    visualization_type: VisualizationTypeEnum = Field(..., description="Visualization type")
    visualization_data: Dict[str, Any] = Field(..., description="Visualization data and configuration")
    plot_configuration: Dict[str, Any] = Field(..., description="Applied plot configuration")
    status: str = Field(..., description="Creation status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")


# Dashboard DTOs

class DashboardLayoutDTO(BaseModel):
    """Dashboard layout configuration."""
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    grid_layout: List[Dict[str, Any]] = Field(..., description="Grid layout specification")
    responsive: bool = Field(True, description="Responsive layout")
    
    @validator("rows", "columns")
    def validate_grid_dimensions(cls, v):
        if v < 1 or v > 12:
            raise ValueError("Grid dimensions must be between 1 and 12")
        return v


class DashboardVisualizationDTO(BaseModel):
    """Visualization configuration for dashboard."""
    visualization_id: Optional[UUID] = Field(None, description="Existing visualization ID")
    visualization_request: Optional[VisualizationRequestDTO] = Field(None, description="New visualization request")
    position: Dict[str, int] = Field(..., description="Position in dashboard grid")
    size: Dict[str, int] = Field(..., description="Size in dashboard grid")
    title: Optional[str] = Field(None, description="Panel title")
    
    @validator("position", "size")
    def validate_grid_position(cls, v):
        required_keys = ["x", "y"] if "position" in cls.__name__ else ["width", "height"]
        if not all(key in v for key in required_keys):
            raise ValueError(f"Must specify {', '.join(required_keys)}")
        return v


class DashboardRequestDTO(BaseModel):
    """Request for creating a dashboard."""
    name: str = Field(..., description="Dashboard name")
    description: Optional[str] = Field(None, description="Dashboard description")
    layout: DashboardLayoutDTO = Field(..., description="Dashboard layout")
    visualizations: List[DashboardVisualizationDTO] = Field(..., description="Dashboard visualizations")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Global dashboard filters")
    refresh_interval: Optional[int] = Field(None, description="Auto-refresh interval in seconds")
    
    @validator("visualizations")
    def validate_visualizations(cls, v):
        if not v:
            raise ValueError("At least one visualization required for dashboard")
        return v
    
    @validator("name")
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Dashboard name cannot be empty")
        return v.strip()


class DashboardResponseDTO(BaseModel):
    """Response for dashboard creation."""
    dashboard_id: UUID = Field(..., description="Dashboard identifier")
    name: str = Field(..., description="Dashboard name")
    layout_config: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
    visualization_ids: List[UUID] = Field(..., description="Visualization identifiers")
    dashboard_url: Optional[str] = Field(None, description="Dashboard access URL")
    embed_url: Optional[str] = Field(None, description="Embeddable dashboard URL")
    filter_config: Dict[str, Any] = Field(..., description="Filter configuration")
    status: str = Field(..., description="Creation status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")


# Chart Data DTOs

class ChartDataRequestDTO(BaseModel):
    """Request for generating chart data."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    chart_type: VisualizationTypeEnum = Field(..., description="Chart type for data optimization")
    feature_columns: List[str] = Field(..., description="Features to include")
    target_column: Optional[str] = Field(None, description="Target column")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data filters")
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    sampling: Optional[Dict[str, Any]] = Field(None, description="Data sampling configuration")
    format: str = Field(default="json", description="Output format")
    
    @validator("format")
    def validate_format(cls, v):
        allowed_formats = ["json", "csv", "plotly", "d3"]
        if v not in allowed_formats:
            raise ValueError(f"Format must be one of: {', '.join(allowed_formats)}")
        return v


class ChartDataResponseDTO(BaseModel):
    """Response for chart data generation."""
    data_id: UUID = Field(..., description="Data identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    chart_data: Dict[str, Any] = Field(..., description="Formatted chart data")
    metadata: Dict[str, Any] = Field(..., description="Data metadata")
    data_summary: Dict[str, Any] = Field(..., description="Data summary statistics")
    sampling_info: Optional[Dict[str, Any]] = Field(None, description="Sampling information")
    status: str = Field(..., description="Generation status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Generation timestamp")


# Interactive Visualization DTOs

class InteractionConfigDTO(BaseModel):
    """Configuration for interactive features."""
    interaction_type: InteractionTypeEnum = Field(..., description="Type of interaction")
    enabled: bool = Field(True, description="Whether interaction is enabled")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Interaction parameters")


class ControlConfigDTO(BaseModel):
    """Configuration for dashboard controls."""
    control_type: str = Field(..., description="Type of control (slider, dropdown, etc.)")
    label: str = Field(..., description="Control label")
    target_feature: str = Field(..., description="Target feature to control")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Control options")
    default_value: Optional[Any] = Field(None, description="Default control value")


class InteractiveVisualizationRequestDTO(BaseModel):
    """Request for creating interactive visualization."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features to visualize")
    target_column: Optional[str] = Field(None, description="Target column")
    title: Optional[str] = Field(None, description="Visualization title")
    theme: ChartThemeEnum = Field(default=ChartThemeEnum.DEFAULT, description="Chart theme")
    interactions: List[InteractionConfigDTO] = Field(..., description="Interaction configurations")
    controls: Optional[List[ControlConfigDTO]] = Field(default_factory=list, description="User controls")
    real_time_updates: bool = Field(False, description="Enable real-time data updates")
    animation_enabled: bool = Field(True, description="Enable animations")
    
    @validator("interactions")
    def validate_interactions(cls, v):
        if not v:
            raise ValueError("At least one interaction configuration required")
        return v


class InteractiveVisualizationResponseDTO(BaseModel):
    """Response for interactive visualization creation."""
    visualization_id: UUID = Field(..., description="Visualization identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    interactive_config: List[InteractionConfigDTO] = Field(..., description="Applied interaction configuration")
    visualization_html: str = Field(..., description="HTML code for visualization")
    javascript_code: str = Field(..., description="JavaScript code for interactivity")
    css_styles: str = Field(..., description="CSS styles")
    api_endpoints: List[str] = Field(default_factory=list, description="API endpoints for data updates")
    real_time_config: Dict[str, Any] = Field(..., description="Real-time update configuration")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")


# Visualization Management DTOs

class VisualizationListRequestDTO(BaseModel):
    """Request for listing visualizations."""
    dataset_id: Optional[UUID] = Field(None, description="Filter by dataset")
    visualization_type: Optional[VisualizationTypeEnum] = Field(None, description="Filter by type")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    tags: Optional[List[str]] = Field(default_factory=list, description="Filter by tags")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")
    limit: int = Field(20, description="Maximum results")
    offset: int = Field(0, description="Results offset")


class VisualizationSummaryDTO(BaseModel):
    """Summary information for a visualization."""
    visualization_id: UUID = Field(..., description="Visualization identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    visualization_type: VisualizationTypeEnum = Field(..., description="Visualization type")
    title: Optional[str] = Field(None, description="Visualization title")
    feature_columns: List[str] = Field(..., description="Features visualized")
    status: str = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Visualization tags")


class VisualizationListResponseDTO(BaseModel):
    """Response for listing visualizations."""
    visualizations: List[VisualizationSummaryDTO] = Field(..., description="Visualization summaries")
    total_count: int = Field(..., description="Total number of visualizations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")


class VisualizationExportRequestDTO(BaseModel):
    """Request for exporting visualization."""
    format: ExportFormatEnum = Field(..., description="Export format")
    width: Optional[int] = Field(None, description="Export width in pixels")
    height: Optional[int] = Field(None, description="Export height in pixels")
    dpi: int = Field(300, description="DPI for raster formats")
    include_data: bool = Field(False, description="Include raw data in export")
    
    @validator("width", "height")
    def validate_export_dimensions(cls, v):
        if v is not None and (v < 100 or v > 8000):
            raise ValueError("Export dimensions must be between 100 and 8000 pixels")
        return v
    
    @validator("dpi")
    def validate_dpi(cls, v):
        if v < 72 or v > 600:
            raise ValueError("DPI must be between 72 and 600")
        return v


class VisualizationExportResponseDTO(BaseModel):
    """Response for visualization export."""
    export_id: UUID = Field(..., description="Export job identifier")
    visualization_id: UUID = Field(..., description="Visualization identifier")
    format: ExportFormatEnum = Field(..., description="Export format")
    file_url: str = Field(..., description="Download URL for exported file")
    file_size_bytes: int = Field(..., description="File size in bytes")
    expires_at: datetime = Field(..., description="Download URL expiration")
    export_time_seconds: float = Field(..., description="Export processing time")
    created_at: datetime = Field(..., description="Export timestamp")


# Advanced Visualization DTOs

class AnimationConfigDTO(BaseModel):
    """Configuration for animated visualizations."""
    enabled: bool = Field(True, description="Enable animations")
    duration_ms: int = Field(1000, description="Animation duration in milliseconds")
    easing: str = Field("ease-in-out", description="Animation easing function")
    loop: bool = Field(False, description="Loop animation")
    auto_play: bool = Field(True, description="Auto-play animation")
    
    @validator("duration_ms")
    def validate_duration(cls, v):
        if v < 100 or v > 10000:
            raise ValueError("Animation duration must be between 100 and 10000 milliseconds")
        return v


class RealTimeConfigDTO(BaseModel):
    """Configuration for real-time data updates."""
    enabled: bool = Field(False, description="Enable real-time updates")
    update_interval_seconds: int = Field(30, description="Update interval in seconds")
    max_data_points: int = Field(1000, description="Maximum data points to keep")
    streaming_endpoint: Optional[str] = Field(None, description="Streaming data endpoint")
    buffer_size: int = Field(100, description="Data buffer size")
    
    @validator("update_interval_seconds")
    def validate_update_interval(cls, v):
        if v < 1 or v > 3600:
            raise ValueError("Update interval must be between 1 and 3600 seconds")
        return v


class ThemeCustomizationDTO(BaseModel):
    """Custom theme configuration."""
    name: str = Field(..., description="Theme name")
    colors: Dict[str, str] = Field(..., description="Color definitions")
    fonts: Dict[str, str] = Field(default_factory=dict, description="Font definitions")
    styles: Dict[str, Any] = Field(default_factory=dict, description="Additional style definitions")


class VisualizationTemplateDTO(BaseModel):
    """Visualization template definition."""
    template_id: UUID = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    visualization_type: VisualizationTypeEnum = Field(..., description="Visualization type")
    default_config: Dict[str, Any] = Field(..., description="Default configuration")
    required_features: List[str] = Field(..., description="Required feature types")
    optional_features: List[str] = Field(default_factory=list, description="Optional feature types")
    created_at: datetime = Field(..., description="Template creation timestamp")
    created_by: UUID = Field(..., description="Template creator ID")