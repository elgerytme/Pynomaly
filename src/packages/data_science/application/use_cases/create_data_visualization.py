"""Use case for creating data visualizations."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.visualization_dto import (
    CreateVisualizationRequestDTO,
    CreateVisualizationResponseDTO,
    PlotConfigDTO,
    StatisticalPlotsRequestDTO,
    StatisticalPlotsResponseDTO
)
from ...domain.services.data_visualization_service import IDataVisualizationService


class CreateDataVisualizationUseCase:
    """Use case for creating data visualizations."""
    
    def __init__(self, visualization_service: IDataVisualizationService):
        self._visualization_service = visualization_service
    
    async def execute(self, request: CreateVisualizationRequestDTO) -> CreateVisualizationResponseDTO:
        """Execute visualization creation use case.
        
        Args:
            request: Visualization creation request parameters
            
        Returns:
            Visualization creation response with chart details
            
        Raises:
            VisualizationError: If visualization creation fails
        """
        try:
            visualization_id = uuid4()
            
            # Validate visualization type
            self._validate_visualization_type(request.visualization_type)
            
            # Create visualization based on type
            if request.visualization_type == "histogram":
                chart_data, chart_config = await self._create_histogram(request)
            elif request.visualization_type == "scatter_plot":
                chart_data, chart_config = await self._create_scatter_plot(request)
            elif request.visualization_type == "box_plot":
                chart_data, chart_config = await self._create_box_plot(request)
            elif request.visualization_type == "correlation_heatmap":
                chart_data, chart_config = await self._create_correlation_heatmap(request)
            elif request.visualization_type == "distribution_plot":
                chart_data, chart_config = await self._create_distribution_plot(request)
            elif request.visualization_type == "time_series":
                chart_data, chart_config = await self._create_time_series_plot(request)
            else:
                raise ValueError(f"Unsupported visualization type: {request.visualization_type}")
            
            return CreateVisualizationResponseDTO(
                visualization_id=visualization_id,
                visualization_type=request.visualization_type,
                status="completed",
                chart_data=chart_data,
                chart_config=chart_config,
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            return CreateVisualizationResponseDTO(
                visualization_id=uuid4(),
                visualization_type=request.visualization_type,
                status="failed",
                created_at=datetime.utcnow()
            )
    
    async def create_statistical_plots(self, request: StatisticalPlotsRequestDTO) -> StatisticalPlotsResponseDTO:
        """Create multiple statistical plots."""
        try:
            plots_id = uuid4()
            generated_plots = []
            
            for plot_config in request.plots:
                plot_data = await self._generate_plot_from_config(plot_config, request.dataset_id)
                generated_plots.append(plot_data)
            
            # Generate export URL if requested
            export_url = None
            if request.export_format in ["html", "pdf", "png"]:
                export_url = f"/api/exports/plots/{plots_id}.{request.export_format}"
            
            return StatisticalPlotsResponseDTO(
                plots_id=plots_id,
                generated_plots=generated_plots,
                export_url=export_url,
                status="completed",
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            return StatisticalPlotsResponseDTO(
                plots_id=uuid4(),
                generated_plots=[],
                status="failed",
                created_at=datetime.utcnow()
            )
    
    def _validate_visualization_type(self, visualization_type: str) -> None:
        """Validate visualization type."""
        valid_types = [
            "histogram",
            "scatter_plot", 
            "box_plot",
            "correlation_heatmap",
            "distribution_plot",
            "time_series",
            "bar_chart",
            "line_chart",
            "violin_plot",
            "density_plot"
        ]
        
        if visualization_type not in valid_types:
            raise ValueError(f"Invalid visualization type: {visualization_type}")
    
    async def _create_histogram(self, request: CreateVisualizationRequestDTO) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create histogram visualization."""
        config = request.config
        
        # Mock histogram data
        chart_data = {
            "x": list(range(10)),
            "y": [10, 25, 30, 45, 35, 20, 15, 8, 5, 2],
            "type": "histogram"
        }
        
        chart_config = {
            "title": request.title or "Histogram",
            "xaxis": {"title": config.get("x_column", "Value")},
            "yaxis": {"title": "Frequency"},
            "bins": config.get("bins", 20),
            "color": config.get("color", "#1f77b4")
        }
        
        return chart_data, chart_config
    
    async def _create_scatter_plot(self, request: CreateVisualizationRequestDTO) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create scatter plot visualization."""
        config = request.config
        
        # Mock scatter plot data
        import random
        n_points = 100
        
        chart_data = {
            "x": [random.uniform(0, 100) for _ in range(n_points)],
            "y": [random.uniform(0, 100) for _ in range(n_points)],
            "type": "scatter",
            "mode": "markers"
        }
        
        chart_config = {
            "title": request.title or "Scatter Plot",
            "xaxis": {"title": config.get("x_column", "X Variable")},
            "yaxis": {"title": config.get("y_column", "Y Variable")},
            "marker": {
                "size": config.get("marker_size", 8),
                "color": config.get("color", "#1f77b4"),
                "opacity": config.get("opacity", 0.7)
            }
        }
        
        return chart_data, chart_config
    
    async def _create_box_plot(self, request: CreateVisualizationRequestDTO) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create box plot visualization."""
        config = request.config
        
        # Mock box plot data
        import random
        
        chart_data = {
            "y": [random.gauss(50, 15) for _ in range(100)],
            "type": "box",
            "name": config.get("series_name", "Data")
        }
        
        chart_config = {
            "title": request.title or "Box Plot",
            "yaxis": {"title": config.get("y_column", "Value")},
            "boxmode": config.get("boxmode", "group")
        }
        
        return chart_data, chart_config
    
    async def _create_correlation_heatmap(self, request: CreateVisualizationRequestDTO) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create correlation heatmap visualization."""
        config = request.config
        
        # Mock correlation matrix data
        features = ["Feature A", "Feature B", "Feature C", "Feature D"]
        correlation_matrix = [
            [1.00, 0.65, -0.23, 0.12],
            [0.65, 1.00, 0.12, -0.45],
            [-0.23, 0.12, 1.00, 0.78],
            [0.12, -0.45, 0.78, 1.00]
        ]
        
        chart_data = {
            "z": correlation_matrix,
            "x": features,
            "y": features,
            "type": "heatmap",
            "colorscale": config.get("colorscale", "RdBu"),
            "showscale": True
        }
        
        chart_config = {
            "title": request.title or "Correlation Heatmap",
            "xaxis": {"title": "Features"},
            "yaxis": {"title": "Features"},
            "aspect": "equal"
        }
        
        return chart_data, chart_config
    
    async def _create_distribution_plot(self, request: CreateVisualizationRequestDTO) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create distribution plot visualization."""
        config = request.config
        
        # Mock distribution data
        import random
        import math
        
        x_values = [i * 0.1 for i in range(-50, 51)]
        y_values = [math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi) for x in x_values]
        
        chart_data = {
            "x": x_values,
            "y": y_values,
            "type": "scatter",
            "mode": "lines",
            "fill": "tonexty" if config.get("fill", False) else None
        }
        
        chart_config = {
            "title": request.title or "Distribution Plot",
            "xaxis": {"title": config.get("x_column", "Value")},
            "yaxis": {"title": "Density"},
            "line": {"color": config.get("color", "#1f77b4")}
        }
        
        return chart_data, chart_config
    
    async def _create_time_series_plot(self, request: CreateVisualizationRequestDTO) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create time series plot visualization."""
        config = request.config
        
        # Mock time series data
        import random
        from datetime import datetime, timedelta
        
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(days=i) for i in range(30)]
        values = [random.uniform(0, 100) for _ in range(30)]
        
        chart_data = {
            "x": [d.isoformat() for d in dates],
            "y": values,
            "type": "scatter",
            "mode": "lines+markers"
        }
        
        chart_config = {
            "title": request.title or "Time Series Plot",
            "xaxis": {"title": "Date", "type": "date"},
            "yaxis": {"title": config.get("y_column", "Value")},
            "line": {"color": config.get("color", "#1f77b4")}
        }
        
        return chart_data, chart_config
    
    async def _generate_plot_from_config(self, plot_config: PlotConfigDTO, dataset_id: Any) -> Dict[str, Any]:
        """Generate plot from configuration."""
        # Mock implementation - would integrate with actual plotting library
        return {
            "plot_id": str(uuid4()),
            "plot_type": plot_config.plot_type,
            "title": plot_config.title or f"{plot_config.plot_type.title()} Plot",
            "data": {"mock": "data"},
            "config": {
                "width": plot_config.width,
                "height": plot_config.height,
                "theme": plot_config.theme,
                "interactive": plot_config.interactive
            }
        }