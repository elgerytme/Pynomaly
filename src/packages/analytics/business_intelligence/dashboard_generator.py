"""
Business Intelligence Dashboard Generator

This module creates comprehensive business intelligence dashboards with
real-time analytics, KPI tracking, and interactive visualizations.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from sqlalchemy import create_engine, text
import redis
import boto3
from google.cloud import bigquery
import streamlit as st
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardType(Enum):
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    CUSTOMER = "customer"
    PRODUCT = "product"
    TECHNICAL = "technical"

class MetricType(Enum):
    KPI = "kpi"
    TREND = "trend"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    FORECAST = "forecast"

@dataclass
class KPIDefinition:
    """Key Performance Indicator definition"""
    kpi_id: str
    name: str
    description: str
    query: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    format_string: str = "{:.2f}"
    trend_direction: str = "higher_is_better"  # higher_is_better, lower_is_better
    category: str = "general"

@dataclass
class ChartDefinition:
    """Chart configuration definition"""
    chart_id: str
    title: str
    chart_type: str  # line, bar, pie, scatter, heatmap, etc.
    data_query: str
    x_column: str
    y_column: str
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    aggregation: Optional[str] = None  # sum, avg, count, etc.
    filters: Dict[str, Any] = None
    layout_options: Dict[str, Any] = None

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    kpis: List[KPIDefinition]
    charts: List[ChartDefinition]
    refresh_interval_minutes: int = 5
    data_sources: Dict[str, str] = None
    access_permissions: List[str] = None
    layout_config: Dict[str, Any] = None

class DataProvider:
    """Provides data from various sources for dashboards"""
    
    def __init__(self):
        self.connections = {}
        self.cache = redis.Redis(host='localhost', port=6379, db=1)
        self.cache_ttl = 300  # 5 minutes
    
    def add_connection(self, name: str, connection_string: str, connection_type: str = "postgresql"):
        """Add data source connection"""
        if connection_type == "postgresql":
            self.connections[name] = create_engine(connection_string)
        elif connection_type == "bigquery":
            self.connections[name] = bigquery.Client()
        elif connection_type == "redis":
            self.connections[name] = redis.from_url(connection_string)
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")
    
    async def get_data(self, source: str, query: str, use_cache: bool = True) -> pd.DataFrame:
        """Get data from specified source"""
        cache_key = f"dashboard_data:{source}:{hash(query)}"
        
        # Try cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached data for query: {query[:50]}...")
                return pd.read_json(cached_data)
        
        # Fetch fresh data
        logger.info(f"Fetching data from {source}: {query[:50]}...")
        
        if source not in self.connections:
            raise ValueError(f"Unknown data source: {source}")
        
        connection = self.connections[source]
        
        if isinstance(connection, bigquery.Client):
            df = self._query_bigquery(connection, query)
        else:
            df = pd.read_sql(query, connection)
        
        # Cache the result
        if use_cache:
            self.cache.setex(cache_key, self.cache_ttl, df.to_json())
        
        return df
    
    def _query_bigquery(self, client: bigquery.Client, query: str) -> pd.DataFrame:
        """Query BigQuery and return DataFrame"""
        query_job = client.query(query)
        return query_job.to_dataframe()
    
    async def get_kpi_value(self, kpi: KPIDefinition, source: str) -> Dict[str, Any]:
        """Calculate KPI value and status"""
        df = await self.get_data(source, kpi.query)
        
        if df.empty:
            return {
                "value": 0,
                "formatted_value": "0",
                "status": "critical",
                "trend": "flat",
                "change_percent": 0
            }
        
        current_value = float(df.iloc[0, 0])  # Assume first column is the KPI value
        
        # Determine status
        if kpi.trend_direction == "higher_is_better":
            if current_value >= kpi.target_value:
                status = "good"
            elif current_value >= kpi.warning_threshold:
                status = "warning"
            else:
                status = "critical"
        else:  # lower_is_better
            if current_value <= kpi.target_value:
                status = "good"
            elif current_value <= kpi.warning_threshold:
                status = "warning"
            else:
                status = "critical"
        
        # Calculate trend (simplified - would need historical data)
        trend = "flat"  # up, down, flat
        change_percent = 0
        
        if len(df) > 1:
            previous_value = float(df.iloc[1, 0])
            change_percent = ((current_value - previous_value) / previous_value) * 100
            
            if abs(change_percent) > 5:  # 5% threshold
                trend = "up" if change_percent > 0 else "down"
        
        return {
            "value": current_value,
            "formatted_value": kpi.format_string.format(current_value),
            "status": status,
            "trend": trend,
            "change_percent": change_percent
        }

class ChartGenerator:
    """Generate interactive charts for dashboards"""
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
    
    async def create_chart(self, chart_def: ChartDefinition, data_source: str) -> go.Figure:
        """Create chart based on definition"""
        logger.info(f"Creating chart: {chart_def.title}")
        
        # Get data
        df = await self.data_provider.get_data(data_source, chart_def.data_query)
        
        # Apply filters if specified
        if chart_def.filters:
            df = self._apply_filters(df, chart_def.filters)
        
        # Apply aggregation if specified
        if chart_def.aggregation:
            df = self._apply_aggregation(df, chart_def)
        
        # Create chart based on type
        if chart_def.chart_type == "line":
            fig = self._create_line_chart(df, chart_def)
        elif chart_def.chart_type == "bar":
            fig = self._create_bar_chart(df, chart_def)
        elif chart_def.chart_type == "pie":
            fig = self._create_pie_chart(df, chart_def)
        elif chart_def.chart_type == "scatter":
            fig = self._create_scatter_chart(df, chart_def)
        elif chart_def.chart_type == "heatmap":
            fig = self._create_heatmap(df, chart_def)
        elif chart_def.chart_type == "histogram":
            fig = self._create_histogram(df, chart_def)
        elif chart_def.chart_type == "box":
            fig = self._create_box_plot(df, chart_def)
        elif chart_def.chart_type == "funnel":
            fig = self._create_funnel_chart(df, chart_def)
        else:
            raise ValueError(f"Unsupported chart type: {chart_def.chart_type}")
        
        # Apply layout options
        if chart_def.layout_options:
            fig.update_layout(**chart_def.layout_options)
        
        # Common layout updates
        fig.update_layout(
            title=chart_def.title,
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        for column, filter_value in filters.items():
            if isinstance(filter_value, list):
                df = df[df[column].isin(filter_value)]
            elif isinstance(filter_value, dict):
                if "min" in filter_value:
                    df = df[df[column] >= filter_value["min"]]
                if "max" in filter_value:
                    df = df[df[column] <= filter_value["max"]]
            else:
                df = df[df[column] == filter_value]
        
        return df
    
    def _apply_aggregation(self, df: pd.DataFrame, chart_def: ChartDefinition) -> pd.DataFrame:
        """Apply aggregation to DataFrame"""
        if chart_def.aggregation == "sum":
            return df.groupby(chart_def.x_column)[chart_def.y_column].sum().reset_index()
        elif chart_def.aggregation == "avg":
            return df.groupby(chart_def.x_column)[chart_def.y_column].mean().reset_index()
        elif chart_def.aggregation == "count":
            return df.groupby(chart_def.x_column).size().reset_index(name=chart_def.y_column)
        elif chart_def.aggregation == "max":
            return df.groupby(chart_def.x_column)[chart_def.y_column].max().reset_index()
        elif chart_def.aggregation == "min":
            return df.groupby(chart_def.x_column)[chart_def.y_column].min().reset_index()
        else:
            return df
    
    def _create_line_chart(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create line chart"""
        if chart_def.color_column:
            fig = px.line(df, x=chart_def.x_column, y=chart_def.y_column, 
                         color=chart_def.color_column)
        else:
            fig = px.line(df, x=chart_def.x_column, y=chart_def.y_column)
        
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create bar chart"""
        if chart_def.color_column:
            fig = px.bar(df, x=chart_def.x_column, y=chart_def.y_column, 
                        color=chart_def.color_column)
        else:
            fig = px.bar(df, x=chart_def.x_column, y=chart_def.y_column)
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create pie chart"""
        fig = px.pie(df, names=chart_def.x_column, values=chart_def.y_column)
        return fig
    
    def _create_scatter_chart(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(
            df, 
            x=chart_def.x_column, 
            y=chart_def.y_column,
            color=chart_def.color_column,
            size=chart_def.size_column
        )
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create heatmap"""
        # Pivot data for heatmap
        pivot_df = df.pivot_table(
            index=chart_def.x_column,
            columns=chart_def.color_column or 'category',
            values=chart_def.y_column,
            aggfunc='mean'
        )
        
        fig = px.imshow(pivot_df, aspect="auto")
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create histogram"""
        fig = px.histogram(df, x=chart_def.x_column, color=chart_def.color_column)
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create box plot"""
        fig = px.box(df, x=chart_def.x_column, y=chart_def.y_column, 
                    color=chart_def.color_column)
        return fig
    
    def _create_funnel_chart(self, df: pd.DataFrame, chart_def: ChartDefinition) -> go.Figure:
        """Create funnel chart"""
        fig = go.Figure(go.Funnel(
            y=df[chart_def.x_column],
            x=df[chart_def.y_column]
        ))
        return fig

class DashboardBuilder:
    """Build complete interactive dashboards"""
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.chart_generator = ChartGenerator(data_provider)
        self.apps = {}
    
    async def build_dashboard(self, config: DashboardConfig) -> dash.Dash:
        """Build complete dashboard application"""
        logger.info(f"Building dashboard: {config.name}")
        
        # Create Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Build layout
        app.layout = await self._build_layout(config)
        
        # Register callbacks
        self._register_callbacks(app, config)
        
        # Store app reference
        self.apps[config.dashboard_id] = app
        
        return app
    
    async def _build_layout(self, config: DashboardConfig) -> html.Div:
        """Build dashboard layout"""
        # Header
        header = self._create_header(config)
        
        # KPI section
        kpi_section = await self._create_kpi_section(config)
        
        # Charts section
        charts_section = await self._create_charts_section(config)
        
        # Footer
        footer = self._create_footer()
        
        # Auto-refresh component
        refresh_component = dcc.Interval(
            id='interval-component',
            interval=config.refresh_interval_minutes * 60 * 1000,  # Convert to milliseconds
            n_intervals=0
        )
        
        layout = html.Div([
            refresh_component,
            header,
            kpi_section,
            charts_section,
            footer
        ], style={'padding': '20px'})
        
        return layout
    
    def _create_header(self, config: DashboardConfig) -> dbc.Container:
        """Create dashboard header"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(config.name, className="display-4"),
                    html.P(config.description, className="lead"),
                    html.Hr()
                ])
            ])
        ], fluid=True)
    
    async def _create_kpi_section(self, config: DashboardConfig) -> dbc.Container:
        """Create KPI cards section"""
        kpi_cards = []
        
        for kpi in config.kpis:
            # Get KPI data
            kpi_data = await self.data_provider.get_kpi_value(
                kpi, 
                list(config.data_sources.keys())[0]  # Use first data source
            )
            
            # Create KPI card
            card = self._create_kpi_card(kpi, kpi_data)
            kpi_cards.append(dbc.Col(card, width=12, md=6, lg=3))
        
        return dbc.Container([
            html.H2("Key Performance Indicators"),
            dbc.Row(kpi_cards, className="mb-4")
        ], fluid=True)
    
    def _create_kpi_card(self, kpi: KPIDefinition, kpi_data: Dict[str, Any]) -> dbc.Card:
        """Create individual KPI card"""
        # Determine card color based on status
        color_map = {
            "good": "success",
            "warning": "warning",
            "critical": "danger"
        }
        
        # Trend icon
        trend_icons = {
            "up": "📈",
            "down": "📉",
            "flat": "➡️"
        }
        
        card = dbc.Card([
            dbc.CardBody([
                html.H4(kpi.name, className="card-title"),
                html.H2(
                    [
                        kpi_data["formatted_value"],
                        html.Small(f" {kpi.unit}", className="text-muted")
                    ],
                    className="text-center"
                ),
                html.P([
                    trend_icons.get(kpi_data["trend"], ""),
                    f" {kpi_data['change_percent']:+.1f}%"
                ], className="text-center text-muted"),
                html.P(kpi.description, className="card-text small")
            ])
        ], color=color_map.get(kpi_data["status"], "light"), outline=True)
        
        return card
    
    async def _create_charts_section(self, config: DashboardConfig) -> dbc.Container:
        """Create charts section"""
        chart_rows = []
        
        # Group charts by rows (2 charts per row)
        for i in range(0, len(config.charts), 2):
            row_charts = config.charts[i:i+2]
            chart_cols = []
            
            for chart_def in row_charts:
                # Create chart
                fig = await self.chart_generator.create_chart(
                    chart_def,
                    list(config.data_sources.keys())[0]  # Use first data source
                )
                
                chart_component = dcc.Graph(
                    id=f"chart-{chart_def.chart_id}",
                    figure=fig,
                    style={'height': '400px'}
                )
                
                chart_cols.append(dbc.Col(chart_component, width=12, lg=6))
            
            chart_rows.append(dbc.Row(chart_cols, className="mb-4"))
        
        return dbc.Container([
            html.H2("Analytics"),
            *chart_rows
        ], fluid=True)
    
    def _create_footer(self) -> dbc.Container:
        """Create dashboard footer"""
        return dbc.Container([
            html.Hr(),
            html.P([
                "Last updated: ",
                html.Span(id="last-updated", children=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                " | MLOps Business Intelligence Platform"
            ], className="text-center text-muted")
        ], fluid=True)
    
    def _register_callbacks(self, app: dash.Dash, config: DashboardConfig):
        """Register dashboard callbacks for interactivity"""
        
        @app.callback(
            [Output(f"chart-{chart.chart_id}", "figure") for chart in config.charts] +
            [Output("last-updated", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_dashboard(n_intervals):
            """Update all dashboard components"""
            logger.info(f"Updating dashboard {config.dashboard_id} (interval {n_intervals})")
            
            # Update charts
            updated_figures = []
            for chart_def in config.charts:
                try:
                    # This would normally be async, but Dash callbacks can't be async
                    # In practice, you'd use a different approach for async operations
                    fig = asyncio.run(self.chart_generator.create_chart(
                        chart_def,
                        list(config.data_sources.keys())[0]
                    ))
                    updated_figures.append(fig)
                except Exception as e:
                    logger.error(f"Error updating chart {chart_def.chart_id}: {e}")
                    # Return empty figure on error
                    updated_figures.append(go.Figure())
            
            # Update timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return updated_figures + [timestamp]

class DashboardTemplates:
    """Pre-built dashboard templates for common use cases"""
    
    @staticmethod
    def executive_dashboard() -> DashboardConfig:
        """Executive dashboard template"""
        kpis = [
            KPIDefinition(
                kpi_id="revenue",
                name="Monthly Revenue",
                description="Total revenue for current month",
                query="SELECT SUM(amount) FROM revenue WHERE date >= DATE_TRUNC('month', CURRENT_DATE)",
                target_value=1000000,
                warning_threshold=800000,
                critical_threshold=600000,
                unit="$",
                format_string="${:,.0f}",
                category="financial"
            ),
            KPIDefinition(
                kpi_id="customers",
                name="Active Customers",
                description="Number of active customers",
                query="SELECT COUNT(DISTINCT customer_id) FROM user_sessions WHERE last_activity >= CURRENT_DATE - INTERVAL '30 days'",
                target_value=10000,
                warning_threshold=8000,
                critical_threshold=6000,
                unit="users",
                format_string="{:,.0f}",
                category="customer"
            ),
            KPIDefinition(
                kpi_id="conversion_rate",
                name="Conversion Rate",
                description="Percentage of visitors who convert",
                query="SELECT (COUNT(CASE WHEN converted = true THEN 1 END) * 100.0 / COUNT(*)) FROM user_sessions WHERE date >= CURRENT_DATE - INTERVAL '7 days'",
                target_value=5.0,
                warning_threshold=3.0,
                critical_threshold=2.0,
                unit="%",
                format_string="{:.1f}%",
                category="marketing"
            ),
            KPIDefinition(
                kpi_id="churn_rate",
                name="Customer Churn Rate",
                description="Percentage of customers who churned this month",
                query="SELECT (churned_customers * 100.0 / total_customers) FROM monthly_churn_summary WHERE month = DATE_TRUNC('month', CURRENT_DATE)",
                target_value=2.0,
                warning_threshold=5.0,
                critical_threshold=8.0,
                unit="%",
                format_string="{:.1f}%",
                trend_direction="lower_is_better",
                category="customer"
            )
        ]
        
        charts = [
            ChartDefinition(
                chart_id="revenue_trend",
                title="Revenue Trend (Last 12 Months)",
                chart_type="line",
                data_query="SELECT DATE_TRUNC('month', date) as month, SUM(amount) as revenue FROM revenue WHERE date >= CURRENT_DATE - INTERVAL '12 months' GROUP BY month ORDER BY month",
                x_column="month",
                y_column="revenue"
            ),
            ChartDefinition(
                chart_id="customer_acquisition",
                title="Customer Acquisition by Channel",
                chart_type="bar",
                data_query="SELECT acquisition_channel, COUNT(*) as customers FROM customers WHERE created_at >= CURRENT_DATE - INTERVAL '30 days' GROUP BY acquisition_channel",
                x_column="acquisition_channel",
                y_column="customers"
            ),
            ChartDefinition(
                chart_id="revenue_by_product",
                title="Revenue by Product Category",
                chart_type="pie",
                data_query="SELECT product_category, SUM(amount) as revenue FROM revenue r JOIN products p ON r.product_id = p.id WHERE r.date >= CURRENT_DATE - INTERVAL '30 days' GROUP BY product_category",
                x_column="product_category",
                y_column="revenue"
            ),
            ChartDefinition(
                chart_id="customer_lifetime_value",
                title="Customer Lifetime Value Distribution",
                chart_type="histogram",
                data_query="SELECT customer_lifetime_value FROM customers WHERE created_at >= CURRENT_DATE - INTERVAL '12 months'",
                x_column="customer_lifetime_value",
                y_column="count"
            )
        ]
        
        return DashboardConfig(
            dashboard_id="executive_dashboard",
            name="Executive Dashboard",
            description="High-level business metrics and KPIs for executive decision making",
            dashboard_type=DashboardType.EXECUTIVE,
            kpis=kpis,
            charts=charts,
            refresh_interval_minutes=15,
            data_sources={"main": "postgresql://user:pass@localhost/mlops_analytics"}
        )
    
    @staticmethod
    def operational_dashboard() -> DashboardConfig:
        """Operational dashboard template"""
        kpis = [
            KPIDefinition(
                kpi_id="system_uptime",
                name="System Uptime",
                description="System availability percentage",
                query="SELECT (uptime_minutes * 100.0 / total_minutes) FROM system_uptime WHERE date = CURRENT_DATE",
                target_value=99.9,
                warning_threshold=99.0,
                critical_threshold=95.0,
                unit="%",
                format_string="{:.2f}%",
                category="operational"
            ),
            KPIDefinition(
                kpi_id="api_response_time",
                name="API Response Time",
                description="Average API response time in milliseconds",
                query="SELECT AVG(response_time_ms) FROM api_metrics WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour'",
                target_value=100,
                warning_threshold=200,
                critical_threshold=500,
                unit="ms",
                format_string="{:.0f}ms",
                trend_direction="lower_is_better",
                category="performance"
            ),
            KPIDefinition(
                kpi_id="error_rate",
                name="Error Rate",
                description="Percentage of requests resulting in errors",
                query="SELECT (error_count * 100.0 / total_requests) FROM hourly_error_summary WHERE hour = DATE_TRUNC('hour', CURRENT_TIMESTAMP)",
                target_value=0.1,
                warning_threshold=1.0,
                critical_threshold=5.0,
                unit="%",
                format_string="{:.2f}%",
                trend_direction="lower_is_better",
                category="quality"
            ),
            KPIDefinition(
                kpi_id="active_users",
                name="Active Users (Current Hour)",
                description="Number of users active in the current hour",
                query="SELECT COUNT(DISTINCT user_id) FROM user_activity WHERE timestamp >= DATE_TRUNC('hour', CURRENT_TIMESTAMP)",
                target_value=1000,
                warning_threshold=500,
                critical_threshold=200,
                unit="users",
                format_string="{:,.0f}",
                category="engagement"
            )
        ]
        
        charts = [
            ChartDefinition(
                chart_id="request_volume",
                title="Request Volume (Last 24 Hours)",
                chart_type="line",
                data_query="SELECT DATE_TRUNC('hour', timestamp) as hour, COUNT(*) as requests FROM api_requests WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours' GROUP BY hour ORDER BY hour",
                x_column="hour",
                y_column="requests"
            ),
            ChartDefinition(
                chart_id="response_time_percentiles",
                title="Response Time Percentiles",
                chart_type="line",
                data_query="SELECT hour, p50, p95, p99 FROM response_time_percentiles WHERE hour >= CURRENT_TIMESTAMP - INTERVAL '24 hours' ORDER BY hour",
                x_column="hour",
                y_column="p95"  # Could make this multi-line
            ),
            ChartDefinition(
                chart_id="error_types",
                title="Error Types Distribution",
                chart_type="bar",
                data_query="SELECT error_type, COUNT(*) as count FROM errors WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours' GROUP BY error_type ORDER BY count DESC",
                x_column="error_type",
                y_column="count"
            ),
            ChartDefinition(
                chart_id="system_resources",
                title="System Resource Utilization",
                chart_type="line",
                data_query="SELECT timestamp, cpu_percent, memory_percent FROM system_metrics WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '4 hours' ORDER BY timestamp",
                x_column="timestamp",
                y_column="cpu_percent"
            )
        ]
        
        return DashboardConfig(
            dashboard_id="operational_dashboard",
            name="Operational Dashboard",
            description="Real-time operational metrics and system health monitoring",
            dashboard_type=DashboardType.OPERATIONAL,
            kpis=kpis,
            charts=charts,
            refresh_interval_minutes=1,
            data_sources={"main": "postgresql://user:pass@localhost/mlops_metrics"}
        )

# Usage example and orchestration functions
async def create_dashboard_from_config(config: DashboardConfig, 
                                     data_provider: DataProvider) -> dash.Dash:
    """Create dashboard from configuration"""
    builder = DashboardBuilder(data_provider)
    return await builder.build_dashboard(config)

async def deploy_dashboard_suite():
    """Deploy complete suite of business intelligence dashboards"""
    logger.info("Deploying comprehensive BI dashboard suite")
    
    # Initialize data provider
    data_provider = DataProvider()
    data_provider.add_connection(
        "analytics_db", 
        "postgresql://user:pass@localhost/mlops_analytics"
    )
    data_provider.add_connection(
        "metrics_db", 
        "postgresql://user:pass@localhost/mlops_metrics"
    )
    
    # Create dashboard builder
    builder = DashboardBuilder(data_provider)
    
    # Build executive dashboard
    exec_config = DashboardTemplates.executive_dashboard()
    exec_app = await builder.build_dashboard(exec_config)
    
    # Build operational dashboard
    ops_config = DashboardTemplates.operational_dashboard()
    ops_app = await builder.build_dashboard(ops_config)
    
    # Deploy dashboards (in practice, you'd deploy these to separate endpoints)
    logger.info("BI dashboard suite deployed successfully")
    
    return {
        "executive": exec_app,
        "operational": ops_app
    }

if __name__ == "__main__":
    # Example usage
    async def main():
        apps = await deploy_dashboard_suite()
        
        # Run executive dashboard on port 8050
        apps["executive"].run_server(debug=True, port=8050)
    
    asyncio.run(main())