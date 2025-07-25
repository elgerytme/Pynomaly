"""
AIOps Dashboard for Predictive Maintenance Visualization
Real-time dashboard for monitoring system health and predictions
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table
from plotly.subplots import make_subplots

from .predictive_maintenance import AIOpsPredictor

logger = logging.getLogger(__name__)


class AIOpsWebInterface:
    """Web-based dashboard for AIOps monitoring and control"""
    
    def __init__(self, predictor: AIOpsPredictor, config: Dict[str, Any]):
        self.predictor = predictor
        self.config = config
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        # Dashboard state
        self.current_metrics = []
        self.current_predictions = []
        self.maintenance_schedule = []
        
        # Setup dashboard layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("AIOps Predictive Maintenance Dashboard", className="header-title"),
                html.Div([
                    html.Div(id="system-status", className="status-indicator"),
                    html.Div(id="last-update", className="last-update")
                ], className="header-info")
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Button("Refresh Data", id="refresh-btn", n_clicks=0, className="control-btn"),
                    html.Button("Run Analysis", id="analyze-btn", n_clicks=0, className="control-btn"),
                    html.Button("Export Report", id="export-btn", n_clicks=0, className="control-btn"),
                ], className="control-buttons"),
                
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id="time-range-dropdown",
                        options=[
                            {"label": "Last Hour", "value": "1h"},
                            {"label": "Last 6 Hours", "value": "6h"},
                            {"label": "Last 24 Hours", "value": "24h"},
                            {"label": "Last Week", "value": "7d"}
                        ],
                        value="6h",
                        className="control-dropdown"
                    )
                ], className="control-group")
            ], className="control-panel"),
            
            # Main Dashboard Content
            html.Div([
                # Overview Cards
                html.Div([
                    html.Div([
                        html.H3("System Health", className="card-title"),
                        html.Div(id="health-score", className="metric-value"),
                        html.Div("Overall health score", className="metric-label")
                    ], className="overview-card health-card"),
                    
                    html.Div([
                        html.H3("Active Alerts", className="card-title"),
                        html.Div(id="active-alerts", className="metric-value"),
                        html.Div("Critical predictions", className="metric-label")
                    ], className="overview-card alert-card"),
                    
                    html.Div([
                        html.H3("Maintenance Due", className="card-title"),
                        html.Div(id="maintenance-due", className="metric-value"),
                        html.Div("Tasks scheduled", className="metric-label")
                    ], className="overview-card maintenance-card"),
                    
                    html.Div([
                        html.H3("Cost Savings", className="card-title"),
                        html.Div(id="cost-savings", className="metric-value"),
                        html.Div("Prevented failures", className="metric-label")
                    ], className="overview-card savings-card")
                ], className="overview-cards"),
                
                # Charts Row 1
                html.Div([
                    html.Div([
                        dcc.Graph(id="system-metrics-chart")
                    ], className="chart-container", style={"width": "50%"}),
                    
                    html.Div([
                        dcc.Graph(id="prediction-timeline")
                    ], className="chart-container", style={"width": "50%"})
                ], className="charts-row"),
                
                # Charts Row 2
                html.Div([
                    html.Div([
                        dcc.Graph(id="component-health-heatmap")
                    ], className="chart-container", style={"width": "50%"}),
                    
                    html.Div([
                        dcc.Graph(id="maintenance-calendar")
                    ], className="chart-container", style={"width": "50%"})
                ], className="charts-row"),
                
                # Data Tables
                html.Div([
                    html.Div([
                        html.H3("Current Predictions"),
                        dash_table.DataTable(
                            id="predictions-table",
                            columns=[
                                {"name": "Component", "id": "component"},
                                {"name": "Type", "id": "prediction_type"},
                                {"name": "Probability", "id": "probability", "type": "numeric", "format": {"specifier": ".1f"}},
                                {"name": "Severity", "id": "severity"},
                                {"name": "Time", "id": "predicted_time"},
                                {"name": "Actions", "id": "actions"}
                            ],
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{severity} = critical'},
                                    'backgroundColor': '#ffebee',
                                    'color': 'black'
                                },
                                {
                                    'if': {'filter_query': '{severity} = high'},
                                    'backgroundColor': '#fff3e0',
                                    'color': 'black'
                                }
                            ],
                            sort_action="native",
                            page_size=10
                        )
                    ], className="table-container", style={"width": "100%"})
                ], className="tables-row"),
                
                html.Div([
                    html.Div([
                        html.H3("Maintenance Schedule"),
                        dash_table.DataTable(
                            id="maintenance-table",
                            columns=[
                                {"name": "Component", "id": "component"},
                                {"name": "Type", "id": "type"},
                                {"name": "Priority", "id": "priority", "type": "numeric"},
                                {"name": "Deadline", "id": "deadline"},
                                {"name": "Downtime", "id": "estimated_downtime"},
                                {"name": "Cost", "id": "cost_estimate", "type": "numeric", "format": {"specifier": "$.0f"}}
                            ],
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{priority} = 5'},
                                    'backgroundColor': '#ffebee',
                                    'color': 'black'
                                },
                                {
                                    'if': {'filter_query': '{priority} = 4'},
                                    'backgroundColor': '#fff3e0',
                                    'color': 'black'
                                }
                            ],
                            sort_action="native",
                            page_size=10
                        )
                    ], className="table-container", style={"width": "100%"})
                ], className="tables-row"),
                
                # System Log
                html.Div([
                    html.H3("System Activity Log"),
                    html.Div(id="system-log", className="log-container")
                ], className="log-section")
            ], className="main-content"),
            
            # Auto-refresh component
            dcc.Interval(
                id="interval-component",
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            ),
            
            # Store components for data
            dcc.Store(id="metrics-store"),
            dcc.Store(id="predictions-store"),
            dcc.Store(id="maintenance-store")
            
        ], className="dashboard-container")

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output("metrics-store", "data"),
             Output("predictions-store", "data"),
             Output("maintenance-store", "data"),
             Output("last-update", "children")],
            [Input("refresh-btn", "n_clicks"),
             Input("analyze-btn", "n_clicks"),
             Input("interval-component", "n_intervals")]
        )
        def update_data(refresh_clicks, analyze_clicks, n_intervals):
            """Update dashboard data"""
            try:
                # Collect latest metrics
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                metrics = loop.run_until_complete(self.predictor.collect_system_metrics())
                predictions = loop.run_until_complete(self.predictor.analyze_and_predict(metrics))
                maintenance = loop.run_until_complete(self.predictor.generate_maintenance_schedule(predictions))
                
                loop.close()
                
                # Convert to serializable format
                metrics_data = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "component": m.component,
                        "metric_name": m.metric_name,
                        "value": m.value,
                        "labels": m.labels
                    }
                    for m in metrics
                ]
                
                predictions_data = [
                    {
                        "component": p.component,
                        "prediction_type": p.prediction_type,
                        "probability": p.probability,
                        "confidence": p.confidence,
                        "predicted_time": p.predicted_time.isoformat() if p.predicted_time else None,
                        "severity": p.severity,
                        "recommended_actions": p.recommended_actions,
                        "actions": ", ".join(p.recommended_actions[:2])  # First 2 actions for table display
                    }
                    for p in predictions
                ]
                
                maintenance_data = [
                    {
                        "component": m.component,
                        "type": m.type,
                        "priority": m.priority,
                        "deadline": m.deadline.isoformat(),
                        "estimated_downtime": str(m.estimated_downtime),
                        "cost_estimate": m.cost_estimate
                    }
                    for m in maintenance
                ]
                
                last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                return metrics_data, predictions_data, maintenance_data, last_update
                
            except Exception as e:
                logger.error(f"Failed to update dashboard data: {e}")
                return [], [], [], f"Update failed: {str(e)}"

        @self.app.callback(
            [Output("health-score", "children"),
             Output("active-alerts", "children"),
             Output("maintenance-due", "children"),
             Output("cost-savings", "children"),
             Output("system-status", "children"),
             Output("system-status", "className")],
            [Input("predictions-store", "data"),
             Input("maintenance-store", "data")]
        )
        def update_overview_cards(predictions_data, maintenance_data):
            """Update overview cards"""
            try:
                # Calculate health score
                if predictions_data:
                    critical_count = len([p for p in predictions_data if p["severity"] == "critical"])
                    high_count = len([p for p in predictions_data if p["severity"] == "high"])
                    health_score = max(0, 100 - (critical_count * 20 + high_count * 10))
                else:
                    health_score = 100
                
                # Active alerts (critical + high severity)
                active_alerts = len([p for p in predictions_data if p["severity"] in ["critical", "high"]])
                
                # Maintenance due (priority 4+)
                maintenance_due = len([m for m in maintenance_data if m["priority"] >= 4])
                
                # Cost savings (mock calculation)
                cost_savings = f"${len(predictions_data) * 5000:,}"
                
                # System status
                if health_score >= 90:
                    status_text = "● Healthy"
                    status_class = "status-indicator healthy"
                elif health_score >= 70:
                    status_text = "● Warning"
                    status_class = "status-indicator warning"
                else:
                    status_text = "● Critical"
                    status_class = "status-indicator critical"
                
                return (
                    f"{health_score}%",
                    str(active_alerts),
                    str(maintenance_due),
                    cost_savings,
                    status_text,
                    status_class
                )
                
            except Exception as e:
                logger.error(f"Failed to update overview cards: {e}")
                return "N/A", "N/A", "N/A", "N/A", "● Error", "status-indicator error"

        @self.app.callback(
            Output("system-metrics-chart", "figure"),
            [Input("metrics-store", "data"),
             Input("time-range-dropdown", "value")]
        )
        def update_metrics_chart(metrics_data, time_range):
            """Update system metrics chart"""
            try:
                if not metrics_data:
                    return go.Figure().add_annotation(text="No data available", showarrow=False)
                
                df = pd.DataFrame(metrics_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by time range
                now = datetime.now()
                if time_range == "1h":
                    cutoff = now - timedelta(hours=1)
                elif time_range == "6h":
                    cutoff = now - timedelta(hours=6)
                elif time_range == "24h":
                    cutoff = now - timedelta(hours=24)
                else:  # 7d
                    cutoff = now - timedelta(days=7)
                
                df = df[df['timestamp'] >= cutoff]
                
                # Create subplots for different metric types
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('CPU Usage', 'Memory Usage', 'Request Latency', 'Error Rate'),
                    vertical_spacing=0.08
                )
                
                # CPU metrics
                cpu_data = df[df['metric_name'].str.contains('cpu', case=False, na=False)]
                if not cpu_data.empty:
                    fig.add_trace(
                        go.Scatter(x=cpu_data['timestamp'], y=cpu_data['value'], 
                                 mode='lines', name='CPU Usage', line=dict(color='blue')),
                        row=1, col=1
                    )
                
                # Memory metrics
                memory_data = df[df['metric_name'].str.contains('memory', case=False, na=False)]
                if not memory_data.empty:
                    fig.add_trace(
                        go.Scatter(x=memory_data['timestamp'], y=memory_data['value'], 
                                 mode='lines', name='Memory Usage', line=dict(color='green')),
                        row=1, col=2
                    )
                
                # Latency metrics
                latency_data = df[df['metric_name'].str.contains('latency', case=False, na=False)]
                if not latency_data.empty:
                    fig.add_trace(
                        go.Scatter(x=latency_data['timestamp'], y=latency_data['value'], 
                                 mode='lines', name='Latency', line=dict(color='orange')),
                        row=2, col=1
                    )
                
                # Error rate metrics
                error_data = df[df['metric_name'].str.contains('error', case=False, na=False)]
                if not error_data.empty:
                    fig.add_trace(
                        go.Scatter(x=error_data['timestamp'], y=error_data['value'], 
                                 mode='lines', name='Error Rate', line=dict(color='red')),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    title="System Metrics Overview",
                    height=400,
                    showlegend=False
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Failed to update metrics chart: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

        @self.app.callback(
            Output("prediction-timeline", "figure"),
            [Input("predictions-store", "data")]
        )
        def update_prediction_timeline(predictions_data):
            """Update prediction timeline chart"""
            try:
                if not predictions_data:
                    return go.Figure().add_annotation(text="No predictions available", showarrow=False)
                
                df = pd.DataFrame(predictions_data)
                df['predicted_time'] = pd.to_datetime(df['predicted_time'])
                
                # Create timeline chart
                fig = go.Figure()
                
                # Color mapping for severity
                colors = {
                    'critical': 'red',
                    'high': 'orange', 
                    'medium': 'yellow',
                    'low': 'blue'
                }
                
                for severity in ['critical', 'high', 'medium', 'low']:
                    severity_data = df[df['severity'] == severity]
                    if not severity_data.empty:
                        fig.add_trace(go.Scatter(
                            x=severity_data['predicted_time'],
                            y=severity_data['probability'],
                            mode='markers',
                            name=severity.title(),
                            marker=dict(
                                color=colors[severity],
                                size=severity_data['probability'] / 5,
                                opacity=0.7
                            ),
                            text=severity_data['component'],
                            textposition="top center"
                        ))
                
                fig.update_layout(
                    title="Prediction Timeline",
                    xaxis_title="Predicted Time",
                    yaxis_title="Probability (%)",
                    height=400
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Failed to update prediction timeline: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

        @self.app.callback(
            Output("component-health-heatmap", "figure"),
            [Input("metrics-store", "data")]
        )
        def update_component_heatmap(metrics_data):
            """Update component health heatmap"""
            try:
                if not metrics_data:
                    return go.Figure().add_annotation(text="No data available", showarrow=False)
                
                df = pd.DataFrame(metrics_data)
                
                # Calculate health scores by component and metric
                health_matrix = df.pivot_table(
                    index='component', 
                    columns='metric_name', 
                    values='value', 
                    aggfunc='mean'
                ).fillna(0)
                
                # Normalize values to 0-100 scale for health scoring
                health_scores = health_matrix.copy()
                for col in health_scores.columns:
                    if 'error' in col.lower() or 'fail' in col.lower():
                        # For error metrics, lower is better
                        health_scores[col] = 100 - (health_scores[col] / health_scores[col].max() * 100)
                    else:
                        # For other metrics, normalize to reasonable range
                        health_scores[col] = np.clip(health_scores[col] / health_scores[col].max() * 100, 0, 100)
                
                fig = go.Figure(data=go.Heatmap(
                    z=health_scores.values,
                    x=health_scores.columns,
                    y=health_scores.index,
                    colorscale='RdYlGn',
                    reversescale=False,
                    text=health_scores.round(1).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title="Health Score")
                ))
                
                fig.update_layout(
                    title="Component Health Heatmap",
                    height=400,
                    xaxis_title="Metrics",
                    yaxis_title="Components"
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Failed to update component heatmap: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

        @self.app.callback(
            Output("maintenance-calendar", "figure"),
            [Input("maintenance-store", "data")]
        )
        def update_maintenance_calendar(maintenance_data):
            """Update maintenance calendar"""
            try:
                if not maintenance_data:
                    return go.Figure().add_annotation(text="No maintenance scheduled", showarrow=False)
                
                df = pd.DataFrame(maintenance_data)
                df['deadline'] = pd.to_datetime(df['deadline'])
                
                # Create Gantt-style chart
                fig = go.Figure()
                
                # Color mapping for priority
                priority_colors = {5: 'red', 4: 'orange', 3: 'yellow', 2: 'lightblue', 1: 'lightgray'}
                
                for i, row in df.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['deadline']],
                        y=[row['component']],
                        mode='markers',
                        marker=dict(
                            color=priority_colors.get(row['priority'], 'gray'),
                            size=row['priority'] * 5,
                            symbol='square'
                        ),
                        name=f"Priority {row['priority']}",
                        text=f"{row['type']} - ${row['cost_estimate']:.0f}",
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Maintenance Schedule",
                    xaxis_title="Deadline",
                    yaxis_title="Component",
                    height=400
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Failed to update maintenance calendar: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)

        @self.app.callback(
            Output("predictions-table", "data"),
            [Input("predictions-store", "data")]
        )
        def update_predictions_table(predictions_data):
            """Update predictions table"""
            return predictions_data

        @self.app.callback(
            Output("maintenance-table", "data"),
            [Input("maintenance-store", "data")]
        )
        def update_maintenance_table(maintenance_data):
            """Update maintenance table"""
            return maintenance_data

        @self.app.callback(
            Output("system-log", "children"),
            [Input("predictions-store", "data"),
             Input("maintenance-store", "data")]
        )
        def update_system_log(predictions_data, maintenance_data):
            """Update system activity log"""
            try:
                log_entries = []
                
                # Add prediction entries
                for pred in predictions_data[-5:]:  # Last 5 predictions
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    severity_class = f"log-entry {pred['severity']}"
                    log_entries.append(
                        html.Div([
                            html.Span(timestamp, className="log-timestamp"),
                            html.Span(f"PREDICTION: {pred['component']} - {pred['prediction_type']} ({pred['probability']:.1f}%)", 
                                    className="log-message"),
                            html.Span(pred['severity'].upper(), className=f"log-severity {pred['severity']}")
                        ], className=severity_class)
                    )
                
                # Add maintenance entries
                for maint in maintenance_data[:3]:  # Top 3 maintenance items
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    priority_class = f"log-entry priority-{maint['priority']}"
                    log_entries.append(
                        html.Div([
                            html.Span(timestamp, className="log-timestamp"),
                            html.Span(f"MAINTENANCE: {maint['component']} - {maint['type']} scheduled", 
                                    className="log-message"),
                            html.Span(f"P{maint['priority']}", className=f"log-priority priority-{maint['priority']}")
                        ], className=priority_class)
                    )
                
                return log_entries[-10:]  # Show last 10 entries
                
            except Exception as e:
                logger.error(f"Failed to update system log: {e}")
                return [html.Div(f"Log error: {str(e)}", className="log-entry error")]

        @self.app.callback(
            Output("export-btn", "children"),
            [Input("export-btn", "n_clicks")],
            [State("predictions-store", "data"),
             State("maintenance-store", "data")]
        )
        def export_report(n_clicks, predictions_data, maintenance_data):
            """Export dashboard report"""
            if n_clicks > 0:
                try:
                    # Generate report
                    report = {
                        "timestamp": datetime.now().isoformat(),
                        "predictions": predictions_data,
                        "maintenance": maintenance_data,
                        "summary": {
                            "total_predictions": len(predictions_data),
                            "critical_alerts": len([p for p in predictions_data if p["severity"] == "critical"]),
                            "maintenance_tasks": len(maintenance_data),
                            "estimated_cost": sum(m["cost_estimate"] for m in maintenance_data)
                        }
                    }
                    
                    # Save report (in production, implement proper file export)
                    filename = f"aiops_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    return "Report Exported"
                except Exception as e:
                    logger.error(f"Failed to export report: {e}")
                    return "Export Failed"
            
            return "Export Report"

    def run(self, host="0.0.0.0", port=8050, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting AIOps dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# CSS Styles (would typically be in a separate CSS file)
CSS_STYLES = """
.dashboard-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title {
    margin: 0;
    font-size: 28px;
    font-weight: 300;
}

.status-indicator {
    font-size: 18px;
    font-weight: bold;
}

.status-indicator.healthy { color: #4caf50; }
.status-indicator.warning { color: #ff9800; }
.status-indicator.critical { color: #f44336; }
.status-indicator.error { color: #9e9e9e; }

.control-panel {
    background: white;
    padding: 15px;
    margin: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.control-btn {
    background: #667eea;
    color: white;
    border: none;
    padding: 10px 20px;
    margin: 0 5px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}

.control-btn:hover {
    background: #5a6fd8;
}

.overview-cards {
    display: flex;
    gap: 20px;
    margin: 20px;
}

.overview-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    flex: 1;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.card-title {
    margin: 0 0 10px 0;
    font-size: 16px;
    color: #666;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #333;
    margin: 10px 0;
}

.metric-label {
    font-size: 14px;
    color: #999;
}

.health-card { border-top: 4px solid #4caf50; }
.alert-card { border-top: 4px solid #f44336; }
.maintenance-card { border-top: 4px solid #ff9800; }
.savings-card { border-top: 4px solid #2196f3; }

.charts-row {
    display: flex;
    gap: 20px;
    margin: 20px;
}

.chart-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.tables-row {
    margin: 20px;
}

.table-container {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.log-section {
    margin: 20px;
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.log-container {
    max-height: 300px;
    overflow-y: auto;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}

.log-entry {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #e0e0e0;
}

.log-timestamp {
    color: #666;
    width: 80px;
}

.log-message {
    flex: 1;
    padding: 0 10px;
}

.log-severity, .log-priority {
    font-weight: bold;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
}

.log-severity.critical { background: #ffcdd2; color: #d32f2f; }
.log-severity.high { background: #ffe0b2; color: #f57c00; }
.log-severity.medium { background: #fff9c4; color: #f9a825; }
.log-severity.low { background: #e3f2fd; color: #1976d2; }
"""


# Example usage
async def main():
    """Example usage of AIOps Dashboard"""
    from .predictive_maintenance import AIOpsPredictor
    
    config = {
        'kubeconfig_path': '/path/to/kubeconfig',
        'prometheus_url': 'http://prometheus:9090'
    }
    
    # Initialize predictor
    predictor = AIOpsPredictor(config)
    
    # Initialize dashboard
    dashboard = AIOpsWebInterface(predictor, config)
    
    # Run dashboard
    dashboard.run(host="0.0.0.0", port=8050, debug=True)


if __name__ == "__main__":
    asyncio.run(main())