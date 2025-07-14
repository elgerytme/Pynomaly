"""Use case for creating interactive dashboards."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.visualization_dto import (
    CreateDashboardRequestDTO,
    CreateDashboardResponseDTO
)


class CreateInteractiveDashboardUseCase:
    """Use case for creating interactive data exploration dashboards."""
    
    def __init__(self):
        pass
    
    async def execute(self, request: CreateDashboardRequestDTO) -> CreateDashboardResponseDTO:
        """Execute dashboard creation use case.
        
        Args:
            request: Dashboard creation request parameters
            
        Returns:
            Dashboard creation response with dashboard details
            
        Raises:
            DashboardCreationError: If dashboard creation fails
        """
        try:
            dashboard_id = uuid4()
            
            # Validate dashboard configuration
            self._validate_dashboard_config(request.layout_config, request.widgets)
            
            # Generate dashboard HTML/JavaScript
            dashboard_content = await self._generate_dashboard_content(
                request.dashboard_name,
                request.layout_config,
                request.widgets,
                request.dataset_id
            )
            
            # Save dashboard and generate URL
            dashboard_url = await self._save_dashboard(dashboard_id, dashboard_content)
            
            return CreateDashboardResponseDTO(
                dashboard_id=dashboard_id,
                dashboard_name=request.dashboard_name,
                dashboard_url=dashboard_url,
                status="completed",
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            return CreateDashboardResponseDTO(
                dashboard_id=uuid4(),
                dashboard_name=request.dashboard_name,
                dashboard_url="",
                status="failed",
                created_at=datetime.utcnow()
            )
    
    def _validate_dashboard_config(self, layout_config: Dict[str, Any], widgets: List[Dict[str, Any]]) -> None:
        """Validate dashboard configuration."""
        # Validate layout
        required_layout_fields = ["type", "columns", "rows"]
        for field in required_layout_fields:
            if field not in layout_config:
                raise ValueError(f"Missing required layout field: {field}")
        
        # Validate widgets
        if not widgets:
            raise ValueError("Dashboard must have at least one widget")
        
        required_widget_fields = ["type", "position", "config"]
        for i, widget in enumerate(widgets):
            for field in required_widget_fields:
                if field not in widget:
                    raise ValueError(f"Widget {i} missing required field: {field}")
            
            # Validate widget type
            valid_widget_types = [
                "chart", "table", "metric", "filter", "text", "image"
            ]
            if widget["type"] not in valid_widget_types:
                raise ValueError(f"Invalid widget type: {widget['type']}")
    
    async def _generate_dashboard_content(
        self,
        dashboard_name: str,
        layout_config: Dict[str, Any],
        widgets: List[Dict[str, Any]],
        dataset_id: Any
    ) -> str:
        """Generate dashboard HTML/JavaScript content."""
        
        # HTML template for dashboard
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{dashboard_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard-header {{
                    text-align: center;
                    color: #333;
                    margin-bottom: 30px;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat({layout_config.get('columns', 3)}, 1fr);
                    grid-template-rows: repeat({layout_config.get('rows', 2)}, auto);
                    gap: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .widget {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                    min-height: 300px;
                }}
                .widget-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                    border-bottom: 2px solid #007acc;
                    padding-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #007acc;
                    text-align: center;
                    margin: 20px 0;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    text-align: center;
                }}
                .table-container {{
                    overflow-x: auto;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{dashboard_name}</h1>
                <p>Interactive Data Dashboard - Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="dashboard-grid">
                {self._generate_widget_html(widgets)}
            </div>
            
            <script>
                // Dashboard JavaScript
                {self._generate_dashboard_javascript(widgets, dataset_id)}
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_widget_html(self, widgets: List[Dict[str, Any]]) -> str:
        """Generate HTML for dashboard widgets."""
        widget_html = []
        
        for i, widget in enumerate(widgets):
            widget_id = f"widget_{i}"
            widget_type = widget["type"]
            widget_config = widget["config"]
            position = widget["position"]
            
            # Position styling
            position_style = f"grid-column: {position.get('col_start', 1)} / {position.get('col_end', 2)}; grid-row: {position.get('row_start', 1)} / {position.get('row_end', 2)};"
            
            if widget_type == "chart":
                widget_html.append(f"""
                <div class="widget" style="{position_style}">
                    <div class="widget-title">{widget_config.get('title', 'Chart')}</div>
                    <div id="{widget_id}" style="height: 400px;"></div>
                </div>
                """)
            
            elif widget_type == "metric":
                widget_html.append(f"""
                <div class="widget" style="{position_style}">
                    <div class="widget-title">{widget_config.get('title', 'Metric')}</div>
                    <div class="metric-value" id="{widget_id}_value">{widget_config.get('value', '0')}</div>
                    <div class="metric-label">{widget_config.get('label', 'Value')}</div>
                </div>
                """)
            
            elif widget_type == "table":
                widget_html.append(f"""
                <div class="widget" style="{position_style}">
                    <div class="widget-title">{widget_config.get('title', 'Data Table')}</div>
                    <div class="table-container">
                        <table id="{widget_id}">
                            <thead>
                                <tr>
                                    <th>Column 1</th>
                                    <th>Column 2</th>
                                    <th>Column 3</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Sample Data 1</td>
                                    <td>Sample Data 2</td>
                                    <td>Sample Data 3</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                """)
            
            elif widget_type == "text":
                widget_html.append(f"""
                <div class="widget" style="{position_style}">
                    <div class="widget-title">{widget_config.get('title', 'Text')}</div>
                    <div>{widget_config.get('content', 'Text content goes here...')}</div>
                </div>
                """)
        
        return "".join(widget_html)
    
    def _generate_dashboard_javascript(self, widgets: List[Dict[str, Any]], dataset_id: Any) -> str:
        """Generate JavaScript for dashboard interactivity."""
        
        js_code = []
        
        js_code.append("""
        // Dashboard initialization
        $(document).ready(function() {
            initializeDashboard();
        });
        
        function initializeDashboard() {
            console.log('Initializing dashboard...');
        """)
        
        # Generate JavaScript for each widget
        for i, widget in enumerate(widgets):
            widget_id = f"widget_{i}"
            widget_type = widget["type"]
            widget_config = widget["config"]
            
            if widget_type == "chart":
                chart_type = widget_config.get("chart_type", "line")
                
                if chart_type == "line":
                    js_code.append(f"""
                    // Line chart for {widget_id}
                    var trace1 = {{
                        x: [1, 2, 3, 4, 5],
                        y: [10, 11, 12, 13, 14],
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Sample Data'
                    }};
                    
                    var layout = {{
                        title: '{widget_config.get('title', 'Chart')}',
                        xaxis: {{ title: 'X Axis' }},
                        yaxis: {{ title: 'Y Axis' }}
                    }};
                    
                    Plotly.newPlot('{widget_id}', [trace1], layout);
                    """)
                
                elif chart_type == "bar":
                    js_code.append(f"""
                    // Bar chart for {widget_id}
                    var trace1 = {{
                        x: ['Category A', 'Category B', 'Category C', 'Category D'],
                        y: [20, 14, 23, 25],
                        type: 'bar'
                    }};
                    
                    var layout = {{
                        title: '{widget_config.get('title', 'Chart')}'
                    }};
                    
                    Plotly.newPlot('{widget_id}', [trace1], layout);
                    """)
                
                elif chart_type == "pie":
                    js_code.append(f"""
                    // Pie chart for {widget_id}
                    var trace1 = {{
                        values: [19, 26, 55],
                        labels: ['Residential', 'Non-Residential', 'Utility'],
                        type: 'pie'
                    }};
                    
                    var layout = {{
                        title: '{widget_config.get('title', 'Chart')}'
                    }};
                    
                    Plotly.newPlot('{widget_id}', [trace1], layout);
                    """)
            
            elif widget_type == "metric":
                # Add dynamic metric updates
                js_code.append(f"""
                // Update metric {widget_id}
                setInterval(function() {{
                    var randomValue = Math.floor(Math.random() * 1000);
                    $('#{widget_id}_value').text(randomValue);
                }}, 5000);  // Update every 5 seconds
                """)
        
        js_code.append("""
        }
        
        // Dashboard utility functions
        function refreshDashboard() {
            location.reload();
        }
        
        function exportDashboard() {
            window.print();
        }
        """)
        
        return "".join(js_code)
    
    async def _save_dashboard(self, dashboard_id: Any, content: str) -> str:
        """Save dashboard and return URL."""
        # Mock implementation - would save to file system or cloud storage
        dashboard_url = f"/api/dashboards/{dashboard_id}/view"
        
        # In real implementation: save HTML content to file system or database
        
        return dashboard_url