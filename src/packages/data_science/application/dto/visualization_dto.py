"""Data Transfer Objects for data visualization operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime


@dataclass(frozen=True)
class CreateVisualizationRequestDTO:
    """Request DTO for creating data visualizations."""
    dataset_id: UUID
    user_id: UUID
    visualization_type: str
    config: Dict[str, Any]
    title: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class CreateVisualizationResponseDTO:
    """Response DTO for visualization creation."""
    visualization_id: UUID
    visualization_type: str
    status: str
    chart_data: Optional[Dict[str, Any]] = None
    chart_config: Optional[Dict[str, Any]] = None
    created_at: datetime


@dataclass(frozen=True)
class CreateDashboardRequestDTO:
    """Request DTO for creating interactive dashboards."""
    dataset_id: UUID
    user_id: UUID
    dashboard_name: str
    layout_config: Dict[str, Any]
    widgets: List[Dict[str, Any]]
    description: Optional[str] = None


@dataclass(frozen=True)
class CreateDashboardResponseDTO:
    """Response DTO for dashboard creation."""
    dashboard_id: UUID
    dashboard_name: str
    dashboard_url: str
    status: str
    created_at: datetime


@dataclass(frozen=True)
class GenerateReportRequestDTO:
    """Request DTO for generating data analysis reports."""
    analysis_id: UUID
    user_id: UUID
    report_type: str
    format: str = "html"
    include_visualizations: bool = True
    custom_sections: Optional[List[str]] = None


@dataclass(frozen=True)
class GenerateReportResponseDTO:
    """Response DTO for report generation."""
    report_id: UUID
    report_url: str
    format: str
    status: str
    file_size_bytes: Optional[int] = None
    generated_at: datetime


@dataclass(frozen=True)
class PlotConfigDTO:
    """Configuration for statistical plots."""
    plot_type: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    facet_column: Optional[str] = None
    title: Optional[str] = None
    width: int = 800
    height: int = 600
    theme: str = "default"
    color_palette: Optional[str] = None
    show_legend: bool = True
    interactive: bool = True


@dataclass(frozen=True)
class StatisticalPlotsRequestDTO:
    """Request DTO for generating statistical plots."""
    dataset_id: UUID
    user_id: UUID
    plots: List[PlotConfigDTO]
    export_format: str = "html"


@dataclass(frozen=True)
class StatisticalPlotsResponseDTO:
    """Response DTO for statistical plots generation."""
    plots_id: UUID
    generated_plots: List[Dict[str, Any]]
    export_url: Optional[str] = None
    status: str
    created_at: datetime


@dataclass(frozen=True)
class CorrelationHeatmapRequestDTO:
    """Request DTO for correlation heatmap visualization."""
    correlation_matrix_id: UUID
    user_id: UUID
    config: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    color_scheme: str = "RdBu_r"
    show_values: bool = True
    mask_diagonal: bool = True


@dataclass(frozen=True)
class CorrelationHeatmapResponseDTO:
    """Response DTO for correlation heatmap."""
    heatmap_id: UUID
    chart_data: Dict[str, Any]
    chart_config: Dict[str, Any]
    image_url: Optional[str] = None
    interactive_url: Optional[str] = None
    created_at: datetime