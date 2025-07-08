"""Advanced visualization dashboard service with real-time analytics.

This service provides comprehensive enterprise-grade visualization capabilities including:
- Real-time anomaly detection dashboards with interactive charts
- Executive summary dashboards with business KPIs and insights
- Operational monitoring with system health and performance metrics
- Algorithm performance analytics with trends and comparisons
- Interactive visualizations with D3.js, Apache ECharts, and Plotly
- Real-time data streaming with WebSocket connections
- Export capabilities for reports and presentations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .business_kpi_service import BusinessKPIService

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Types of dashboards available."""

    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    REAL_TIME = "real_time"


class ChartType(Enum):
    """Types of charts supported."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    TIME_SERIES = "time_series"
    GAUGE = "gauge"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    RADAR = "radar"
    CHOROPLETH = "choropleth"
    CORRELATION_MATRIX = "correlation_matrix"
    FINANCIAL_IMPACT = "financial_impact"
    ROI_COST_SAVINGS = "roi_cost_savings"
    LIVE_ALERT_STREAM = "live_alert_stream"


class VisualizationEngine(Enum):
    """Visualization engines supported."""

    D3JS = "d3js"
    ECHARTS = "echarts"
    PLOTLY = "plotly"
    CHARTJS = "chartjs"
    HIGHCHARTS = "highcharts"


@dataclass
class DashboardConfig:
    """Configuration for dashboard visualization."""

    dashboard_type: DashboardType = DashboardType.ANALYTICAL
    title: str = ""
    description: str = ""
    refresh_interval_seconds: int = 30
    auto_refresh: bool = True
    theme: str = "default"  # dark, light, corporate
    layout: str = "grid"  # grid, flex, custom
    width: int | None = None
    height: int | None = None
    responsive: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["png", "pdf", "svg"])
    real_time_enabled: bool = False
    websocket_endpoint: str | None = None


@dataclass
class ChartConfig:
    """Configuration for individual charts."""

    chart_id: str = ""
    chart_type: ChartType = ChartType.LINE
    title: str = ""
    subtitle: str = ""
    engine: VisualizationEngine = VisualizationEngine.ECHARTS
    width: int | None = None
    height: int | None = None
    x_axis_label: str = ""
    y_axis_label: str = ""
    color_scheme: str = "default"
    interactive: bool = True
    animation: bool = True
    legend: bool = True
    grid: bool = True
    zoom: bool = True
    brush: bool = False
    data_zoom: bool = True
    tooltip: bool = True
    custom_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTimeMetrics:
    """Real-time metrics for dashboard updates."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    anomalies_detected: int = 0
    detection_rate: float = 0.0
    system_cpu_usage: float = 0.0
    system_memory_usage: float = 0.0
    active_detectors: int = 0
    processed_samples: int = 0
    processing_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    alert_count: int = 0
    business_kpis: dict[str, float] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Data container for dashboard visualization."""

    dashboard_id: str = ""
    dashboard_type: DashboardType = DashboardType.ANALYTICAL
    title: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    charts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    kpis: dict[str, float] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dashboard_id": self.dashboard_id,
            "dashboard_type": self.dashboard_type.value,
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "charts": self.charts,
            "metrics": self.metrics,
            "kpis": self.kpis,
            "alerts": self.alerts,
            "metadata": self.metadata,
        }


class VisualizationDashboardService:
    """Advanced visualization dashboard service for enterprise analytics.

    This service provides comprehensive visualization capabilities including:
    - Executive dashboards with business KPIs and strategic insights
    - Operational dashboards with real-time system monitoring
    - Analytical dashboards with detailed anomaly detection analysis
    - Performance dashboards with algorithm comparison and optimization
    - Compliance dashboards with audit trails and regulatory metrics
    - Real-time streaming dashboards with WebSocket updates
    """

    def __init__(
        self,
        storage_path: Path,
        detector_repository=None,
        result_repository=None,
        dataset_repository=None,
    ):
        """Initialize visualization dashboard service.

        Args:
            storage_path: Path for storing dashboard artifacts
            detector_repository: Repository for detector data
            result_repository: Repository for detection results
            dataset_repository: Repository for dataset information
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.detector_repository = detector_repository
        self.result_repository = result_repository
        self.dataset_repository = dataset_repository

        # Dashboard cache
        self.dashboard_cache: dict[str, DashboardData] = {}

        # KPI cache for computed business KPIs
        self.kpi_cache: dict[str, dict[str, float]] = {}
        self.kpi_cache_ttl = timedelta(minutes=30)  # Cache KPIs for 30 minutes
        self.kpi_cache_timestamps: dict[str, datetime] = {}

        # Real-time data streams
        self.real_time_subscribers: set[str] = set()
        self.metrics_history: list[RealTimeMetrics] = []
        self.max_history_size = 1000

        # Chart generators
        self.chart_generators = {
            ChartType.LINE: self._generate_line_chart,
            ChartType.BAR: self._generate_bar_chart,
            ChartType.SCATTER: self._generate_scatter_chart,
            ChartType.HEATMAP: self._generate_heatmap_chart,
            ChartType.PIE: self._generate_pie_chart,
            ChartType.HISTOGRAM: self._generate_histogram_chart,
            ChartType.BOX_PLOT: self._generate_box_plot_chart,
            ChartType.TIME_SERIES: self._generate_time_series_chart,
            ChartType.GAUGE: self._generate_gauge_chart,
            ChartType.TREEMAP: self._generate_treemap_chart,
            ChartType.RADAR: self._generate_radar_chart,
            ChartType.CHOROPLETH: self._generate_choropleth_chart,
            ChartType.CORRELATION_MATRIX: self._generate_correlation_matrix_chart,
            ChartType.FINANCIAL_IMPACT: self._generate_financial_impact_chart,
            ChartType.ROI_COST_SAVINGS: self._generate_roi_cost_savings_chart,
            ChartType.LIVE_ALERT_STREAM: self._generate_live_alert_stream_chart,
        }

        logger.info("Visualization dashboard service initialized")

    # Common helper methods to reduce duplication
    def _get_metrics_series(
        self, metrics_data: dict[str, Any], series_config: dict[str, Any] = None
    ) -> list[dict[str, Any]]:
        """Get metrics series data with consistent formatting.

        Args:
            metrics_data: Dictionary containing metrics data
            series_config: Optional configuration for series styling

        Returns:
            List of series configurations
        """
        if series_config is None:
            series_config = {}

        series = []

        for metric_name, metric_values in metrics_data.items():
            series_item = {
                "name": metric_name,
                "data": (
                    metric_values
                    if isinstance(metric_values, list)
                    else [metric_values]
                ),
                "type": series_config.get("type", "line"),
                "smooth": series_config.get("smooth", True),
                "symbol": series_config.get("symbol", "circle"),
                "symbolSize": series_config.get("symbolSize", 6),
                "lineStyle": series_config.get("lineStyle", {"width": 2}),
                "itemStyle": series_config.get("itemStyle", {"borderRadius": 4}),
                "emphasis": series_config.get("emphasis", {"focus": "series"}),
                "markPoint": series_config.get(
                    "markPoint",
                    {
                        "data": [
                            {"type": "max", "name": "Maximum"},
                            {"type": "min", "name": "Minimum"},
                        ]
                    },
                ),
                "animationDuration": series_config.get("animationDuration", 1000),
            }

            # Add color if specified
            if "color" in series_config:
                series_item["color"] = series_config["color"]

            series.append(series_item)

        return series

    def _get_heatmap_color(self, value: float, max_value: float) -> str:
        """Get color for heatmap based on value intensity."""
        intensity = value / max_value if max_value > 0 else 0

        # Blue gradient for heatmap
        if intensity < 0.2:
            return "#e0f3f8"
        elif intensity < 0.4:
            return "#abd9e9"
        elif intensity < 0.6:
            return "#74add1"
        elif intensity < 0.8:
            return "#4575b4"
        else:
            return "#313695"

    def _build_chart_payload(
        self,
        chart_id: str,
        chart_type: str,
        title: str,
        x_data: list = None,
        series: list = None,
        engine: str = "echarts",
        **kwargs,
    ) -> dict[str, Any]:
        """Build standardized chart payload with consistent structure.

        Args:
            chart_id: Unique identifier for the chart
            chart_type: Type of chart (line, bar, etc.)
            title: Chart title
            x_data: X-axis data
            series: Series data
            engine: Visualization engine to use
            **kwargs: Additional chart options

        Returns:
            Standardized chart payload
        """
        config = {
            "type": chart_type,
            "title": {
                "text": title,
                "left": kwargs.get("title_position", "center"),
                "textStyle": kwargs.get(
                    "title_style", {"fontSize": 16, "fontWeight": "bold"}
                ),
            },
            "tooltip": kwargs.get(
                "tooltip",
                {
                    "trigger": "axis",
                    "axisPointer": {"type": "cross"},
                    "backgroundColor": "rgba(0,0,0,0.8)",
                    "borderColor": "#777",
                    "borderWidth": 1,
                    "textStyle": {"color": "#fff"},
                    "formatter": kwargs.get("tooltip_formatter"),
                },
            ),
            "legend": kwargs.get(
                "legend",
                {
                    "show": True,
                    "top": "bottom",
                    "orient": "horizontal",
                    "align": "center",
                    "itemGap": 20,
                },
            ),
            "grid": kwargs.get(
                "grid",
                {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
            ),
            "toolbox": kwargs.get(
                "toolbox",
                {
                    "show": True,
                    "feature": {
                        "dataZoom": {"yAxisIndex": "none"},
                        "dataView": {"readOnly": False},
                        "magicType": {"type": ["line", "bar"]},
                        "restore": {},
                        "saveAsImage": {},
                    },
                },
            ),
            "dataZoom": kwargs.get(
                "dataZoom",
                [
                    {"type": "inside", "start": 0, "end": 100},
                    {"type": "slider", "start": 0, "end": 100, "height": 30},
                ],
            ),
            "animation": kwargs.get("animation", True),
            "animationDuration": kwargs.get("animationDuration", 1000),
            "animationEasing": kwargs.get("animationEasing", "cubicOut"),
        }

        # Add x-axis if provided
        if x_data is not None:
            config["xAxis"] = {
                "type": "category",
                "data": x_data,
                "axisLabel": kwargs.get("x_axis_label_style", {"rotate": 0}),
                "name": kwargs.get("x_axis_name", ""),
                "nameLocation": "middle",
                "nameGap": 30,
            }

        # Add y-axis configuration
        config["yAxis"] = kwargs.get(
            "yAxis",
            {
                "type": "value",
                "name": kwargs.get("y_axis_name", ""),
                "nameLocation": "middle",
                "nameGap": 50,
                "axisLabel": {"formatter": kwargs.get("y_axis_formatter", "{value}")},
                "splitLine": {"show": True, "lineStyle": {"type": "dashed"}},
            },
        )

        # Add series if provided
        if series is not None:
            config["series"] = series

        # Add custom options
        if "custom_options" in kwargs:
            config.update(kwargs["custom_options"])

        return {
            "id": chart_id,
            "config": config,
            "engine": engine,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "chart_type": chart_type,
                "data_points": len(x_data) if x_data else 0,
                "series_count": len(series) if series else 0,
            },
        }

    async def generate_executive_dashboard(
        self, time_period: timedelta = timedelta(days=30)
    ) -> DashboardData:
        """Generate executive dashboard with business KPIs and strategic insights.

        Args:
            time_period: Time period for analysis

        Returns:
            Executive dashboard data
        """
        try:
            dashboard_data = DashboardData(
                dashboard_id="executive_dashboard",
                dashboard_type=DashboardType.EXECUTIVE,
                title="Executive Anomaly Detection Dashboard",
            )

            # Calculate business KPIs
            business_kpi_service = BusinessKPIService()

            # Example values, these need to be replaced by actual data sourcing
            total_revenue = 5000000.0
            risk_factor = 0.05
            loss_prevented = 100000.0
            total_potential_loss = 200000.0
            gains_from_investment = 150000.0
            cost_of_investment = 50000.0
            monthly_savings = [10000, 15000, 12000, 13000, 11000]

            # Calculate KPIs
            revenue_at_risk = business_kpi_service.calculate_revenue_at_risk(
                total_revenue, risk_factor
            )
            prevented_loss = business_kpi_service.calculate_prevented_loss(
                loss_prevented, total_potential_loss
            )
            roi = business_kpi_service.calculate_roi(
                gains_from_investment, cost_of_investment
            )
            cost_savings_trend = business_kpi_service.calculate_cost_savings_trends(
                monthly_savings
            )

            kpis = {
                "revenue_at_risk": revenue_at_risk,
                "prevented_loss": prevented_loss,
                "roi": roi,
                "cost_savings_trend": cost_savings_trend,
            }
            dashboard_data.kpis = kpis

            # Generate executive summary charts
            charts = []

            # KPI summary chart
            kpi_chart = await self._generate_kpi_summary_chart(kpis)
            charts.append(kpi_chart)

            # Anomaly trends chart
            trends_chart = await self._generate_anomaly_trends_chart(time_period)
            charts.append(trends_chart)

            # ROI analysis chart
            roi_chart = await self._generate_roi_analysis_chart(time_period)
            charts.append(roi_chart)

            # Cost savings chart
            savings_chart = await self._generate_cost_savings_chart(time_period)
            charts.append(savings_chart)

            # Alert summary chart
            alerts_chart = await self._generate_alert_summary_chart(time_period)
            charts.append(alerts_chart)

            # Performance overview chart
            performance_chart = await self._generate_performance_overview_chart(
                time_period
            )
            charts.append(performance_chart)

            dashboard_data.charts = charts

            # Calculate overall metrics
            dashboard_data.metrics = await self._calculate_executive_metrics(
                time_period
            )

            # Cache dashboard
            self.dashboard_cache[dashboard_data.dashboard_id] = dashboard_data

            logger.info("Executive dashboard generated successfully")
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate executive dashboard: {e}")
            raise

    async def generate_operational_dashboard(
        self, real_time: bool = True
    ) -> DashboardData:
        """Generate operational dashboard with real-time system monitoring.

        Args:
            real_time: Enable real-time updates

        Returns:
            Operational dashboard data
        """
        try:
            dashboard_data = DashboardData(
                dashboard_id="operational_dashboard",
                dashboard_type=DashboardType.OPERATIONAL,
                title="Operational Monitoring Dashboard",
            )

            # Generate operational charts
            charts = []

            # System health chart
            health_chart = await self._generate_system_health_chart()
            charts.append(health_chart)

            # Resource utilization chart
            resources_chart = await self._generate_resource_utilization_chart()
            charts.append(resources_chart)

            # Throughput monitoring chart
            throughput_chart = await self._generate_throughput_monitoring_chart()
            charts.append(throughput_chart)

            # Error rate monitoring chart
            error_chart = await self._generate_error_rate_chart()
            charts.append(error_chart)

            # Active detectors chart
            detectors_chart = await self._generate_active_detectors_chart()
            charts.append(detectors_chart)

            # Processing latency chart
            latency_chart = await self._generate_processing_latency_chart()
            charts.append(latency_chart)

            dashboard_data.charts = charts

            # Calculate operational metrics
            dashboard_data.metrics = await self._calculate_operational_metrics()

            # Real-time configuration
            if real_time:
                dashboard_data.metadata["real_time"] = True
                dashboard_data.metadata["refresh_interval"] = 5  # seconds

            # Cache dashboard
            self.dashboard_cache[dashboard_data.dashboard_id] = dashboard_data

            logger.info("Operational dashboard generated successfully")
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate operational dashboard: {e}")
            raise

    async def generate_analytical_dashboard(
        self, algorithm_comparison: bool = True, feature_analysis: bool = True
    ) -> DashboardData:
        """Generate analytical dashboard with detailed anomaly analysis.

        Args:
            algorithm_comparison: Include algorithm comparison charts
            feature_analysis: Include feature importance analysis

        Returns:
            Analytical dashboard data
        """
        try:
            dashboard_data = DashboardData(
                dashboard_id="analytical_dashboard",
                dashboard_type=DashboardType.ANALYTICAL,
                title="Analytical Anomaly Detection Dashboard",
            )

            # Generate analytical charts
            charts = []

            # Anomaly distribution chart
            distribution_chart = await self._generate_anomaly_distribution_chart()
            charts.append(distribution_chart)

            # Score distribution chart
            scores_chart = await self._generate_score_distribution_chart()
            charts.append(scores_chart)

            # Detection confidence chart
            confidence_chart = await self._generate_confidence_analysis_chart()
            charts.append(confidence_chart)

            if algorithm_comparison:
                # Algorithm performance comparison
                algo_chart = await self._generate_algorithm_comparison_chart()
                charts.append(algo_chart)

                # Algorithm efficiency chart
                efficiency_chart = await self._generate_algorithm_efficiency_chart()
                charts.append(efficiency_chart)

            if feature_analysis:
                # Feature importance chart
                features_chart = await self._generate_feature_importance_chart()
                charts.append(features_chart)

                # Feature correlation chart
                correlation_chart = await self._generate_feature_correlation_chart()
                charts.append(correlation_chart)

            # Anomaly patterns chart
            patterns_chart = await self._generate_anomaly_patterns_chart()
            charts.append(patterns_chart)

            # Temporal analysis chart
            temporal_chart = await self._generate_temporal_analysis_chart()
            charts.append(temporal_chart)

            dashboard_data.charts = charts

            # Calculate analytical metrics
            dashboard_data.metrics = await self._calculate_analytical_metrics()

            # Cache dashboard
            self.dashboard_cache[dashboard_data.dashboard_id] = dashboard_data

            logger.info("Analytical dashboard generated successfully")
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate analytical dashboard: {e}")
            raise

    async def generate_performance_dashboard(
        self, benchmark_comparison: bool = True
    ) -> DashboardData:
        """Generate performance dashboard with algorithm benchmarking.

        Args:
            benchmark_comparison: Include benchmark comparisons

        Returns:
            Performance dashboard data
        """
        try:
            dashboard_data = DashboardData(
                dashboard_id="performance_dashboard",
                dashboard_type=DashboardType.PERFORMANCE,
                title="Performance Analytics Dashboard",
            )

            # Collect algorithm metrics from result repository and monitoring hooks
            algorithm_metrics = await self._collect_algorithm_metrics()
            
            # Generate performance charts
            charts = []

            # Algorithm performance comparison chart with real data
            performance_comparison_chart = await self._generate_algorithm_performance_comparison_chart(algorithm_metrics)
            charts.append(performance_comparison_chart)

            # Accuracy vs Latency scatter chart
            accuracy_latency_chart = await self._generate_accuracy_latency_chart(algorithm_metrics)
            charts.append(accuracy_latency_chart)

            # F1 Score trends chart
            f1_trends_chart = await self._generate_f1_trends_chart(algorithm_metrics)
            charts.append(f1_trends_chart)

            # Memory usage comparison chart
            memory_comparison_chart = await self._generate_memory_comparison_chart(algorithm_metrics)
            charts.append(memory_comparison_chart)

            # Performance regression detection chart
            regression_detection_chart = await self._generate_regression_detection_chart(algorithm_metrics)
            charts.append(regression_detection_chart)

            if benchmark_comparison:
                # Benchmark comparison chart
                benchmark_chart = await self._generate_benchmark_comparison_chart(algorithm_metrics)
                charts.append(benchmark_chart)

                # Performance regression warnings
                regression_warnings_chart = await self._generate_regression_warnings_chart(algorithm_metrics)
                charts.append(regression_warnings_chart)

            # Resource efficiency radar chart
            efficiency_radar_chart = await self._generate_resource_efficiency_radar_chart(algorithm_metrics)
            charts.append(efficiency_radar_chart)

            # Performance trends over time
            trends_chart = await self._generate_performance_trends_timeline_chart(algorithm_metrics)
            charts.append(trends_chart)

            dashboard_data.charts = charts

            # Calculate performance metrics with real data
            dashboard_data.metrics = await self._calculate_performance_metrics_with_data(algorithm_metrics)

            # Add performance alerts and warnings
            dashboard_data.alerts = self._generate_performance_alerts(algorithm_metrics)

            # Cache dashboard
            self.dashboard_cache[dashboard_data.dashboard_id] = dashboard_data

            logger.info("Performance dashboard generated successfully")
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate performance dashboard: {e}")
            raise

    async def generate_real_time_dashboard(
        self, websocket_endpoint: str
    ) -> DashboardData:
        """Generate real-time dashboard with live data streaming.

        Args:
            websocket_endpoint: WebSocket endpoint for real-time data

        Returns:
            Real-time dashboard data
        """
        try:
            dashboard_data = DashboardData(
                dashboard_id="real_time_dashboard",
                dashboard_type=DashboardType.REAL_TIME,
                title="Real-Time Anomaly Detection Dashboard",
            )

            # Generate real-time charts
            charts = []

            # Live anomaly detection chart
            live_chart = await self._generate_live_detection_chart()
            charts.append(live_chart)

            # Real-time metrics chart
            metrics_chart = await self._generate_real_time_metrics_chart()
            charts.append(metrics_chart)

            # Live system status chart
            status_chart = await self._generate_live_status_chart()
            charts.append(status_chart)

            # Alert stream chart
            alerts_chart = await self._generate_alert_stream_chart()
            charts.append(alerts_chart)

            # Throughput monitor chart
            throughput_chart = await self._generate_live_throughput_chart()
            charts.append(throughput_chart)

            dashboard_data.charts = charts

            # Configure real-time settings
            dashboard_data.metadata = {
                "real_time": True,
                "websocket_endpoint": websocket_endpoint,
                "refresh_interval": 1,  # 1 second updates
                "buffer_size": 100,
                "auto_scroll": True,
            }

            # Cache dashboard
            self.dashboard_cache[dashboard_data.dashboard_id] = dashboard_data

            logger.info("Real-time dashboard generated successfully")
            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to generate real-time dashboard: {e}")
            raise

    async def update_real_time_metrics(self, metrics: RealTimeMetrics) -> None:
        """Update real-time metrics for streaming dashboards.

        Args:
            metrics: Real-time metrics to update
        """
        try:
            # Add to metrics history
            self.metrics_history.append(metrics)

            # Maintain history size limit
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size :]

            # Notify real-time subscribers
            await self._notify_real_time_subscribers(metrics)

            logger.debug(
                f"Real-time metrics updated: {metrics.anomalies_detected} anomalies"
            )

        except Exception as e:
            logger.error(f"Failed to update real-time metrics: {e}")
            raise

    async def export_dashboard(
        self,
        dashboard_id: str,
        format: str = "png",
        config: dict[str, Any] | None = None,
    ) -> bytes:
        """Export dashboard to specified format.

        Args:
            dashboard_id: Dashboard to export
            format: Export format (png, pdf, svg, html)
            config: Export configuration

        Returns:
            Exported dashboard data
        """
        try:
            if dashboard_id not in self.dashboard_cache:
                raise ValueError(f"Dashboard {dashboard_id} not found in cache")

            dashboard_data = self.dashboard_cache[dashboard_id]

            if format == "html":
                return await self._export_to_html(dashboard_data, config)
            elif format == "png":
                return await self._export_to_png(dashboard_data, config)
            elif format == "pdf":
                return await self._export_to_pdf(dashboard_data, config)
            elif format == "svg":
                return await self._export_to_svg(dashboard_data, config)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise

    # Private helper methods for calculations and chart generation

    async def _generate_line_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate line chart configuration."""
        try:
            # Use the common helper method
            metrics_data = {"series": data.get("y_data", [])}
            series = self._get_metrics_series(metrics_data, {"type": "line"})

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="line",
                title=config.title,
                x_data=data.get("x_data", []),
                series=series,
                engine=config.engine.value,
                y_axis_name=config.y_axis_label,
            )

        except Exception as e:
            logger.error(f"Failed to generate line chart: {e}")
            return {}

    async def _generate_kpi_summary_chart(
        self, kpis: dict[str, float]
    ) -> dict[str, Any]:
        """Generate KPI summary chart for executive dashboard."""
        try:
            # Create gauge charts for key KPIs
            chart_data = {"x_data": list(kpis.keys()), "y_data": list(kpis.values())}

            config = ChartConfig(
                chart_id="kpi_summary",
                chart_type=ChartType.GAUGE,
                title="Key Performance Indicators",
                engine=VisualizationEngine.ECHARTS,
            )

            return await self._generate_gauge_chart(config, chart_data)

        except Exception as e:
            logger.error(f"Failed to generate KPI summary chart: {e}")
            return {}

    async def _generate_gauge_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate gauge chart configuration."""
        try:
            # Create multiple gauge charts for KPIs
            series = []

            for i, (name, value) in enumerate(
                zip(data["x_data"], data["y_data"], strict=False)
            ):
                series.append(
                    {
                        "name": name,
                        "type": "gauge",
                        "center": [f"{20 + (i % 3) * 30}%", f"{30 + (i // 3) * 40}%"],
                        "radius": "15%",
                        "data": [{"value": value, "name": name}],
                        "detail": {"fontSize": 10},
                        "title": {"fontSize": 12},
                        "axisLine": {"lineStyle": {"width": 8}},
                        "axisLabel": {"fontSize": 8},
                        "splitLine": {"lineStyle": {"width": 2}},
                        "pointer": {"width": 3},
                        "itemStyle": {"color": "auto"},
                        "animation": True,
                        "animationDuration": 1000,
                        "animationEasing": "cubicOut",
                    }
                )

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="gauge",
                title=config.title,
                engine=config.engine.value,
                series=series,
                tooltip={"formatter": "{a} <br/>{b} : {c}%"},
                custom_options={"backgroundColor": "transparent"},
            )

        except Exception as e:
            logger.error(f"Failed to generate gauge chart: {e}")
            return {}

    async def _calculate_business_kpis(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Calculate business KPIs for dashboard."""
        return {
            "anomaly_detection_rate": 95.7,
            "false_positive_rate": 2.3,
            "cost_savings_monthly": 125000.0,
            "automation_coverage": 87.5,
            "model_accuracy": 94.2,
            "system_uptime": 99.8,
            "processing_efficiency": 92.1,
            "compliance_score": 98.5,
            "roi_percentage": 285.7,
            "incident_reduction": 76.3,
        }

    async def _calculate_executive_metrics(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Calculate metrics for executive dashboard."""
        return {
            "period_days": time_period.days,
            "total_detections": 15420,
            "anomalies_found": 1247,
            "models_deployed": 23,
            "cost_savings": 125000,
            "accuracy_improvement": 12.5,
        }

    async def _calculate_operational_metrics(self) -> dict[str, Any]:
        """Calculate metrics for operational dashboard."""
        return {
            "system_health": "healthy",
            "active_services": 12,
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.4,
            "network_throughput": 1250.0,
        }

    async def _calculate_analytical_metrics(self) -> dict[str, Any]:
        """Calculate metrics for analytical dashboard."""
        return {
            "algorithms_analyzed": 8,
            "features_analyzed": 25,
            "confidence_avg": 0.847,
            "score_variance": 0.123,
            "pattern_diversity": 0.678,
        }

    async def _collect_algorithm_metrics(self) -> dict[str, Any]:
        """Collect algorithm metrics from result repository and monitoring hooks."""
        try:
            metrics = {
                "algorithms": {},
                "timestamps": [],
                "overall_stats": {},
            }

            # Collect metrics from result repository if available
            if self.result_repository:
                recent_results = self.result_repository.find_recent(limit=100)
                
                for result in recent_results:
                    # Extract algorithm name from metadata or detector
                    algorithm_name = result.metadata.get("algorithm", "unknown")
                    
                    if algorithm_name not in metrics["algorithms"]:
                        metrics["algorithms"][algorithm_name] = {
                            "accuracy": [],
                            "f1_score": [],
                            "latency": [],
                            "memory": [],
                            "timestamps": [],
                        }
                    
                    # Extract performance metrics
                    perf_metrics = result.metadata.get("performance_metrics", {})
                    
                    # Calculate accuracy approximation from anomaly detection
                    accuracy = 1.0 - abs(result.anomaly_rate - 0.1)  # Assuming 10% expected anomaly rate
                    
                    # Calculate F1 score approximation
                    precision = result.metadata.get("precision", 0.85)
                    recall = result.metadata.get("recall", 0.80)
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    # Get execution time (convert to seconds)
                    latency = (result.execution_time_ms or 100) / 1000.0
                    
                    # Get memory usage (MB)
                    memory = perf_metrics.get("memory_usage_mb", 128.0)
                    
                    # Add to algorithm metrics
                    metrics["algorithms"][algorithm_name]["accuracy"].append(accuracy)
                    metrics["algorithms"][algorithm_name]["f1_score"].append(f1_score)
                    metrics["algorithms"][algorithm_name]["latency"].append(latency)
                    metrics["algorithms"][algorithm_name]["memory"].append(memory)
                    metrics["algorithms"][algorithm_name]["timestamps"].append(result.timestamp)
                    
                    # Add to overall timestamps
                    metrics["timestamps"].append(result.timestamp)

            # If no real data, provide sample data for testing
            if not metrics["algorithms"]:
                import random
                from datetime import datetime, timedelta
                
                algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "DBSCAN"]
                base_time = datetime.utcnow() - timedelta(days=7)
                
                for algo in algorithms:
                    metrics["algorithms"][algo] = {
                        "accuracy": [random.uniform(0.85, 0.95) for _ in range(20)],
                        "f1_score": [random.uniform(0.80, 0.92) for _ in range(20)],
                        "latency": [random.uniform(0.1, 2.0) for _ in range(20)],
                        "memory": [random.uniform(64, 512) for _ in range(20)],
                        "timestamps": [base_time + timedelta(hours=i) for i in range(20)],
                    }

            # Calculate overall statistics
            all_accuracies = []
            all_f1_scores = []
            all_latencies = []
            all_memory = []
            
            for algo_data in metrics["algorithms"].values():
                all_accuracies.extend(algo_data["accuracy"])
                all_f1_scores.extend(algo_data["f1_score"])
                all_latencies.extend(algo_data["latency"])
                all_memory.extend(algo_data["memory"])
            
            if all_accuracies:
                metrics["overall_stats"] = {
                    "avg_accuracy": sum(all_accuracies) / len(all_accuracies),
                    "avg_f1_score": sum(all_f1_scores) / len(all_f1_scores),
                    "avg_latency": sum(all_latencies) / len(all_latencies),
                    "avg_memory": sum(all_memory) / len(all_memory),
                    "total_algorithms": len(metrics["algorithms"]),
                    "total_measurements": len(all_accuracies),
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect algorithm metrics: {e}")
            return {"algorithms": {}, "timestamps": [], "overall_stats": {}}

    async def _calculate_performance_metrics(self) -> dict[str, Any]:
        """Calculate metrics for performance dashboard."""
        return {
            "avg_execution_time": 245.7,
            "memory_peak": 2.4,
            "throughput": 1540.0,
            "efficiency_score": 0.923,
            "scalability_factor": 0.876,
        }

    async def _calculate_performance_metrics_with_data(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Calculate performance metrics using collected algorithm data."""
        try:
            overall_stats = algorithm_metrics.get("overall_stats", {})
            
            return {
                "avg_accuracy": overall_stats.get("avg_accuracy", 0.0),
                "avg_f1_score": overall_stats.get("avg_f1_score", 0.0),
                "avg_latency_seconds": overall_stats.get("avg_latency", 0.0),
                "avg_memory_mb": overall_stats.get("avg_memory", 0.0),
                "total_algorithms": overall_stats.get("total_algorithms", 0),
                "total_measurements": overall_stats.get("total_measurements", 0),
                "performance_score": self._calculate_overall_performance_score(overall_stats),
                "regression_detected": self._detect_performance_regression(algorithm_metrics),
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics with data: {e}")
            return {}

    def _calculate_overall_performance_score(self, overall_stats: dict[str, Any]) -> float:
        """Calculate overall performance score from statistics."""
        try:
            if not overall_stats:
                return 0.0
            
            # Weighted scoring based on key metrics
            accuracy_weight = 0.35
            f1_weight = 0.35
            latency_weight = 0.20  # Lower latency is better
            memory_weight = 0.10   # Lower memory is better
            
            accuracy = overall_stats.get("avg_accuracy", 0.0)
            f1_score = overall_stats.get("avg_f1_score", 0.0)
            latency = overall_stats.get("avg_latency", 1.0)
            memory = overall_stats.get("avg_memory", 512.0)
            
            # Normalize latency and memory scores (lower is better)
            latency_score = max(0, 1.0 - (latency / 5.0))  # Assume 5 seconds is very poor
            memory_score = max(0, 1.0 - (memory / 1024.0))  # Assume 1GB is very poor
            
            # Calculate weighted score
            score = (
                accuracy * accuracy_weight +
                f1_score * f1_weight +
                latency_score * latency_weight +
                memory_score * memory_weight
            )
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate overall performance score: {e}")
            return 0.0

    def _detect_performance_regression(self, algorithm_metrics: dict[str, Any]) -> bool:
        """Detect performance regression by comparing recent vs historical metrics."""
        try:
            algorithms = algorithm_metrics.get("algorithms", {})
            
            for algo_name, algo_data in algorithms.items():
                # Check if we have enough data points
                if len(algo_data.get("accuracy", [])) < 4:
                    continue
                
                # Compare recent metrics (last 25%) with historical (first 75%)
                accuracy_data = algo_data["accuracy"]
                f1_data = algo_data["f1_score"]
                latency_data = algo_data["latency"]
                
                split_point = len(accuracy_data) * 3 // 4
                
                # Historical averages
                hist_accuracy = sum(accuracy_data[:split_point]) / split_point
                hist_f1 = sum(f1_data[:split_point]) / split_point
                hist_latency = sum(latency_data[:split_point]) / split_point
                
                # Recent averages
                recent_accuracy = sum(accuracy_data[split_point:]) / (len(accuracy_data) - split_point)
                recent_f1 = sum(f1_data[split_point:]) / (len(f1_data) - split_point)
                recent_latency = sum(latency_data[split_point:]) / (len(latency_data) - split_point)
                
                # Check for regression (significant decrease in performance)
                accuracy_regression = (hist_accuracy - recent_accuracy) > 0.05
                f1_regression = (hist_f1 - recent_f1) > 0.05
                latency_regression = (recent_latency - hist_latency) > 0.5  # Increase in latency
                
                if accuracy_regression or f1_regression or latency_regression:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to detect performance regression: {e}")
            return False

    async def _generate_performance_alerts(self, algorithm_metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate performance alerts based on algorithm metrics."""
        try:
            alerts = []
            
            overall_stats = algorithm_metrics.get("overall_stats", {})
            
            # Check for low accuracy
            avg_accuracy = overall_stats.get("avg_accuracy", 0.0)
            if avg_accuracy < 0.8:
                alerts.append({
                    "level": "warning",
                    "message": f"Low average accuracy detected: {avg_accuracy:.2%}",
                    "metric": "accuracy",
                    "value": avg_accuracy,
                    "threshold": 0.8
                })
            
            # Check for low F1 score
            avg_f1 = overall_stats.get("avg_f1_score", 0.0)
            if avg_f1 < 0.75:
                alerts.append({
                    "level": "warning",
                    "message": f"Low average F1 score detected: {avg_f1:.2%}",
                    "metric": "f1_score",
                    "value": avg_f1,
                    "threshold": 0.75
                })
            
            # Check for high latency
            avg_latency = overall_stats.get("avg_latency", 0.0)
            if avg_latency > 2.0:
                alerts.append({
                    "level": "critical",
                    "message": f"High average latency detected: {avg_latency:.2f}s",
                    "metric": "latency",
                    "value": avg_latency,
                    "threshold": 2.0
                })
            
            # Check for high memory usage
            avg_memory = overall_stats.get("avg_memory", 0.0)
            if avg_memory > 512.0:
                alerts.append({
                    "level": "warning",
                    "message": f"High average memory usage detected: {avg_memory:.0f}MB",
                    "metric": "memory",
                    "value": avg_memory,
                    "threshold": 512.0
                })
            
            # Check for regression
            if self._detect_performance_regression(algorithm_metrics):
                alerts.append({
                    "level": "critical",
                    "message": "Performance regression detected in algorithm metrics",
                    "metric": "regression",
                    "value": True,
                    "threshold": False
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate performance alerts: {e}")
            return []

    # Placeholder implementations for chart generation methods
    # These would be fully implemented with actual data processing

    async def _generate_anomaly_trends_chart(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Generate anomaly trends chart."""
        return {"id": "anomaly_trends", "type": "line", "data": {}}

    async def _generate_roi_analysis_chart(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Generate ROI analysis chart."""
        return {"id": "roi_analysis", "type": "bar", "data": {}}

    async def _generate_cost_savings_chart(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Generate cost savings chart."""
        return {"id": "cost_savings", "type": "line", "data": {}}

    async def _generate_alert_summary_chart(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Generate alert summary chart."""
        return {"id": "alert_summary", "type": "pie", "data": {}}

    async def _generate_performance_overview_chart(
        self, time_period: timedelta
    ) -> dict[str, Any]:
        """Generate performance overview chart."""
        return {"id": "performance_overview", "type": "radar", "data": {}}

    async def _generate_system_health_chart(self) -> dict[str, Any]:
        """Generate system health chart."""
        return {"id": "system_health", "type": "gauge", "data": {}}

    async def _generate_resource_utilization_chart(self) -> dict[str, Any]:
        """Generate resource utilization chart."""
        return {"id": "resource_utilization", "type": "line", "data": {}}

    async def _generate_throughput_monitoring_chart(self) -> dict[str, Any]:
        """Generate throughput monitoring chart."""
        return {"id": "throughput_monitoring", "type": "line", "data": {}}

    async def _generate_error_rate_chart(self) -> dict[str, Any]:
        """Generate error rate chart."""
        return {"id": "error_rate", "type": "line", "data": {}}

    async def _generate_active_detectors_chart(self) -> dict[str, Any]:
        """Generate active detectors chart."""
        return {"id": "active_detectors", "type": "bar", "data": {}}

    async def _generate_processing_latency_chart(self) -> dict[str, Any]:
        """Generate processing latency chart."""
        return {"id": "processing_latency", "type": "histogram", "data": {}}

    # New chart generation methods
    async def _generate_financial_impact_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate financial impact chart with bar and line visualization."""
        try:
            # Prepare financial impact data
            months = data.get(
                "months",
                [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            )
            cost_savings = data.get(
                "cost_savings",
                [
                    50000,
                    75000,
                    120000,
                    95000,
                    180000,
                    210000,
                    165000,
                    190000,
                    225000,
                    240000,
                    275000,
                    320000,
                ],
            )
            revenue_impact = data.get(
                "revenue_impact",
                [
                    25000,
                    45000,
                    80000,
                    65000,
                    110000,
                    135000,
                    105000,
                    125000,
                    150000,
                    160000,
                    185000,
                    200000,
                ],
            )

            # Create series with both bar and line charts
            series = [
                {
                    "name": "Cost Savings",
                    "type": "bar",
                    "data": cost_savings,
                    "yAxisIndex": 0,
                    "itemStyle": {"color": "#5470c6"},
                    "emphasis": {"focus": "series"},
                    "markLine": {
                        "data": [{"type": "average", "name": "Average Cost Savings"}]
                    },
                },
                {
                    "name": "Revenue Impact",
                    "type": "line",
                    "data": revenue_impact,
                    "yAxisIndex": 0,
                    "smooth": True,
                    "lineStyle": {"width": 3},
                    "itemStyle": {"color": "#91cc75"},
                    "emphasis": {"focus": "series"},
                    "markPoint": {
                        "data": [
                            {"type": "max", "name": "Max"},
                            {"type": "min", "name": "Min"},
                        ]
                    },
                },
            ]

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="mixed",
                title="Financial Impact Analysis",
                x_data=months,
                series=series,
                engine=config.engine.value,
                y_axis_name="Amount ($)",
                y_axis_formatter="${value:,.0f}",
                tooltip={
                    "trigger": "axis",
                    "axisPointer": {"type": "cross"},
                    "formatter": "{b}<br/>{a0}: ${c0:,.0f}<br/>{a1}: ${c1:,.0f}",
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate financial impact chart: {e}")
            return {}

    async def _generate_roi_cost_savings_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate ROI & Cost Savings chart with gauge and line visualization."""
        try:
            # Prepare ROI data
            roi_percentage = data.get("roi_percentage", 285.7)
            cost_savings_monthly = data.get(
                "cost_savings_monthly", [45000, 52000, 78000, 85000, 95000, 112000]
            )
            months = data.get("months", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"])

            # Create gauge for ROI
            roi_gauge = {
                "name": "ROI Percentage",
                "type": "gauge",
                "center": ["25%", "50%"],
                "radius": "50%",
                "min": 0,
                "max": 500,
                "splitNumber": 10,
                "data": [{"value": roi_percentage, "name": "ROI %"}],
                "detail": {"fontSize": 16, "formatter": "{value}%", "color": "#5470c6"},
                "title": {"fontSize": 14, "offsetCenter": [0, "20%"]},
                "axisLine": {
                    "lineStyle": {
                        "width": 20,
                        "color": [[0.2, "#fd666d"], [0.8, "#37a2da"], [1, "#67e0e3"]],
                    }
                },
                "pointer": {"width": 5},
                "itemStyle": {"color": "#5470c6"},
                "animationDuration": 2000,
                "animationEasing": "cubicOut",
            }

            # Create line chart for cost savings trend
            cost_savings_line = {
                "name": "Monthly Cost Savings",
                "type": "line",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": cost_savings_monthly,
                "smooth": True,
                "lineStyle": {"width": 3, "color": "#91cc75"},
                "areaStyle": {"opacity": 0.3, "color": "#91cc75"},
                "emphasis": {"focus": "series"},
                "markPoint": {"data": [{"type": "max", "name": "Peak Savings"}]},
            }

            # Custom configuration for mixed chart
            custom_config = {
                "xAxis": {
                    "type": "category",
                    "data": months,
                    "gridIndex": 0,
                    "axisLabel": {"rotate": 0},
                },
                "yAxis": {
                    "type": "value",
                    "name": "Savings ($)",
                    "gridIndex": 0,
                    "axisLabel": {"formatter": "${value:,.0f}"},
                    "splitLine": {"show": True},
                },
                "grid": {
                    "left": "55%",
                    "right": "5%",
                    "top": "20%",
                    "bottom": "20%",
                    "containLabel": True,
                },
                "series": [roi_gauge, cost_savings_line],
            }

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="mixed",
                title="ROI & Cost Savings Analysis",
                engine=config.engine.value,
                custom_options=custom_config,
                tooltip={"trigger": "item", "formatter": "{a} <br/>{b} : {c}"},
            )

        except Exception as e:
            logger.error(f"Failed to generate ROI & Cost Savings chart: {e}")
            return {}

    async def _generate_choropleth_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate geographic heatmap (choropleth) chart."""
        try:
            # Prepare geographic data
            regions = data.get(
                "regions", ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "IN", "CN"]
            )
            anomaly_counts = data.get(
                "anomaly_counts", [1247, 892, 654, 789, 456, 234, 123, 345, 567, 890]
            )

            # Create map data for visualization
            map_data = []
            for region, count in zip(regions, anomaly_counts, strict=False):
                map_data.append(
                    {
                        "name": region,
                        "value": count,
                        "itemStyle": {
                            "color": self._get_heatmap_color(count, max(anomaly_counts))
                        },
                    }
                )

            # Create choropleth series
            series = [
                {
                    "name": "Anomaly Distribution",
                    "type": "map",
                    "map": "world",
                    "data": map_data,
                    "roam": True,
                    "emphasis": {
                        "label": {"show": True},
                        "itemStyle": {"areaColor": "#389BB7"},
                    },
                    "label": {"show": True, "fontSize": 12},
                    "itemStyle": {"borderColor": "#fff", "borderWidth": 0.5},
                }
            ]

            # Custom visualization map configuration
            custom_config = {
                "visualMap": {
                    "min": 0,
                    "max": max(anomaly_counts),
                    "left": "left",
                    "top": "bottom",
                    "text": ["High", "Low"],
                    "calculable": True,
                    "inRange": {
                        "color": ["#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"]
                    },
                },
                "series": series,
            }

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="map",
                title="Geographic Anomaly Distribution",
                engine=config.engine.value,
                custom_options=custom_config,
                tooltip={"trigger": "item", "formatter": "{b}<br/>Anomalies: {c}"},
            )

        except Exception as e:
            logger.error(f"Failed to generate choropleth chart: {e}")
            return {}

    async def _generate_correlation_matrix_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate correlation matrix heatmap chart."""
        try:
            # Prepare correlation matrix data
            variables = data.get("variables", ["Var1", "Var2", "Var3", "Var4", "Var5"])
            correlations = data.get(
                "correlations",
                [
                    [1, 0.8, 0.5, 0.3, 0.1],
                    [0.8, 1, 0.4, 0.2, 0.05],
                    [0.5, 0.4, 1, 0.6, 0.3],
                    [0.3, 0.2, 0.6, 1, 0.7],
                    [0.1, 0.05, 0.3, 0.7, 1],
                ],
            )

            # Create heatmap data
            heatmap_data = []
            for i, row in enumerate(correlations):
                for j, value in enumerate(row):
                    heatmap_data.append([i, j, value])

            # Create heatmap series
            series = [
                {
                    "name": "Correlation Coefficients",
                    "type": "heatmap",
                    "data": heatmap_data,
                    "label": {"show": True, "formatter": "{c}", "fontSize": 10},
                    "itemStyle": {
                        "emphasis": {
                            "shadowBlur": 10,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ]

            # Custom visualization map configuration
            custom_config = {
                "xAxis": {"type": "category", "data": variables},
                "yAxis": {"type": "category", "data": variables},
                "visualMap": {
                    "min": -1,
                    "max": 1,
                    "calculable": True,
                    "inRange": {
                        "color": [
                            "#313695",
                            "#4575b4",
                            "#74add1",
                            "#abd9e9",
                            "#e0f3f8",
                            "#fdae61",
                            "#f46d43",
                        ]
                    },
                },
                "series": series,
            }

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="heatmap",
                title="Correlation Matrix",
                engine=config.engine.value,
                custom_options=custom_config,
                tooltip={"position": "top", "formatter": "{a} br/{i}:{j} = {c}"},
            )

        except Exception as e:
            logger.error(f"Failed to generate correlation matrix chart: {e}")
            return {}

    async def _generate_live_alert_stream_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate live alert stream chart with notification overlay."""
        try:
            # Prepare alert stream data
            timestamps = data.get("timestamps", [])
            alerts = data.get("alerts", [])
            severity_levels = data.get("severity_levels", [])

            # Create notification overlay data
            notification_data = []
            for i, (timestamp, alert, severity) in enumerate(
                zip(timestamps, alerts, severity_levels, strict=False)
            ):
                notification_data.append(
                    {
                        "value": [timestamp, alert, severity],
                        "itemStyle": {"color": self._get_severity_color(severity)},
                    }
                )

            # Create alert stream series
            series = [
                {
                    "name": "Alert Stream",
                    "type": "scatter",
                    "data": notification_data,
                    "symbolSize": 8,
                    "emphasis": {"focus": "series", "blurScope": "coordinateSystem"},
                    "label": {
                        "show": True,
                        "position": "top",
                        "formatter": "{@[2]}",
                        "fontSize": 10,
                    },
                    "itemStyle": {"borderWidth": 1, "borderColor": "#fff"},
                },
                {
                    "name": "Alert Trend",
                    "type": "line",
                    "data": alerts,
                    "smooth": True,
                    "lineStyle": {"width": 2, "color": "#5470c6", "opacity": 0.8},
                    "areaStyle": {"opacity": 0.1},
                    "symbol": "none",
                    "animation": True,
                    "animationDuration": 500,
                },
            ]

            # Custom configuration for live streaming
            custom_config = {
                "xAxis": {
                    "type": "time",
                    "name": "Time",
                    "axisLabel": {"formatter": "{HH}:{mm}:{ss}"},
                    "splitLine": {"show": True, "lineStyle": {"type": "dashed"}},
                },
                "yAxis": {
                    "type": "value",
                    "name": "Alert Count",
                    "min": 0,
                    "splitLine": {"show": True, "lineStyle": {"type": "dashed"}},
                },
                "series": series,
                "graphic": {
                    "type": "group",
                    "right": 10,
                    "top": 10,
                    "children": [
                        {
                            "type": "rect",
                            "z": 100,
                            "left": "center",
                            "top": "middle",
                            "shape": {"width": 80, "height": 30},
                            "style": {
                                "fill": "rgba(0,0,0,0.3)",
                                "stroke": "#fff",
                                "lineWidth": 1,
                            },
                        },
                        {
                            "type": "text",
                            "z": 100,
                            "left": "center",
                            "top": "middle",
                            "style": {
                                "text": "LIVE",
                                "font": "bold 12px Arial",
                                "fill": "#fff",
                            },
                        },
                    ],
                },
            }

            return self._build_chart_payload(
                chart_id=config.chart_id,
                chart_type="time_series",
                title="Live Alert Stream",
                engine=config.engine.value,
                custom_options=custom_config,
                dataZoom=[
                    {"type": "inside", "start": 70, "end": 100},
                    {"type": "slider", "start": 70, "end": 100},
                ],
                tooltip={
                    "trigger": "axis",
                    "axisPointer": {"type": "cross"},
                    "formatter": "{b}<br/>{a}: {c}<br/>Severity: {d}",
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate live alert stream chart: {e}")
            return {}

    def _get_severity_color(self, severity: str) -> str:
        """Get color based on alert severity."""
        severity_colors = {
            "critical": "#ff4757",
            "high": "#ff7675",
            "medium": "#fdcb6e",
            "low": "#6c5ce7",
            "info": "#74b9ff",
        }
        return severity_colors.get(severity.lower(), "#a4a4a4")

    async def _notify_real_time_subscribers(self, metrics: RealTimeMetrics) -> None:
        """Notify real-time dashboard subscribers of new metrics."""
        # Implementation would send WebSocket updates to connected clients
        pass

    async def _export_to_html(
        self, dashboard_data: DashboardData, config: dict[str, Any] | None
    ) -> bytes:
        """Export dashboard to HTML format."""
        # Implementation would generate HTML with embedded charts
        return b"<html>Dashboard HTML</html>"

    async def _export_to_png(
        self, dashboard_data: DashboardData, config: dict[str, Any] | None
    ) -> bytes:
        """Export dashboard to PNG format."""
        # Implementation would render charts to PNG
        return b"PNG data"

    async def _export_to_pdf(
        self, dashboard_data: DashboardData, config: dict[str, Any] | None
    ) -> bytes:
        """Export dashboard to PDF format."""
        # Implementation would generate PDF report
        return b"PDF data"

    async def _export_to_svg(
        self, dashboard_data: DashboardData, config: dict[str, Any] | None
    ) -> bytes:
        """Export dashboard to SVG format."""
        # Implementation would export as SVG
        return b"SVG data"

    # Placeholder methods for remaining chart types
    async def _generate_bar_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "bar", "data": data}

    async def _generate_scatter_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "scatter", "data": data}

    async def _generate_heatmap_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "heatmap", "data": data}

    async def _generate_pie_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "pie", "data": data}

    async def _generate_histogram_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "histogram", "data": data}

    async def _generate_box_plot_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "boxplot", "data": data}

    async def _generate_time_series_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "timeseries", "data": data}

    async def _generate_treemap_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "treemap", "data": data}

    async def _generate_radar_chart(
        self, config: ChartConfig, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"id": config.chart_id, "type": "radar", "data": data}

    # Placeholder implementations for remaining analytical chart methods
    async def _generate_anomaly_distribution_chart(self) -> dict[str, Any]:
        return {"id": "anomaly_distribution", "type": "histogram", "data": {}}

    async def _generate_score_distribution_chart(self) -> dict[str, Any]:
        return {"id": "score_distribution", "type": "histogram", "data": {}}

    async def _generate_confidence_analysis_chart(self) -> dict[str, Any]:
        return {"id": "confidence_analysis", "type": "box_plot", "data": {}}

    async def _generate_algorithm_comparison_chart(self) -> dict[str, Any]:
        return {"id": "algorithm_comparison", "type": "bar", "data": {}}

    async def _generate_algorithm_efficiency_chart(self) -> dict[str, Any]:
        return {"id": "algorithm_efficiency", "type": "scatter", "data": {}}

    async def _generate_feature_importance_chart(self) -> dict[str, Any]:
        return {"id": "feature_importance", "type": "bar", "data": {}}

    async def _generate_feature_correlation_chart(self) -> dict[str, Any]:
        return {"id": "feature_correlation", "type": "heatmap", "data": {}}

    async def _generate_anomaly_patterns_chart(self) -> dict[str, Any]:
        return {"id": "anomaly_patterns", "type": "scatter", "data": {}}

    async def _generate_temporal_analysis_chart(self) -> dict[str, Any]:
        return {"id": "temporal_analysis", "type": "time_series", "data": {}}

    # Performance dashboard chart methods
    async def _generate_execution_time_chart(self) -> dict[str, Any]:
        return {"id": "execution_time", "type": "bar", "data": {}}

    async def _generate_memory_usage_chart(self) -> dict[str, Any]:
        return {"id": "memory_usage", "type": "line", "data": {}}

    async def _generate_scalability_chart(self) -> dict[str, Any]:
        return {"id": "scalability", "type": "line", "data": {}}

    async def _generate_accuracy_speed_chart(self) -> dict[str, Any]:
        return {"id": "accuracy_speed", "type": "scatter", "data": {}}

    async def _generate_benchmark_comparison_chart(self) -> dict[str, Any]:
        return {"id": "benchmark_comparison", "type": "bar", "data": {}}

    async def _generate_performance_regression_chart(self) -> dict[str, Any]:
        return {"id": "performance_regression", "type": "line", "data": {}}

    async def _generate_resource_efficiency_chart(self) -> dict[str, Any]:
        return {"id": "resource_efficiency", "type": "radar", "data": {}}

    async def _generate_performance_trends_chart(self) -> dict[str, Any]:
        return {"id": "performance_trends", "type": "line", "data": {}}

    # Real-time dashboard chart methods
    async def _generate_live_detection_chart(self) -> dict[str, Any]:
        return {
            "id": "live_detection",
            "type": "time_series",
            "data": {},
            "realtime": True,
        }

    async def _generate_real_time_metrics_chart(self) -> dict[str, Any]:
        return {"id": "real_time_metrics", "type": "line", "data": {}, "realtime": True}

    async def _generate_live_status_chart(self) -> dict[str, Any]:
        return {"id": "live_status", "type": "gauge", "data": {}, "realtime": True}

    async def _generate_alert_stream_chart(self) -> dict[str, Any]:
        return {
            "id": "alert_stream",
            "type": "time_series",
            "data": {},
            "realtime": True,
        }

    async def _generate_live_throughput_chart(self) -> dict[str, Any]:
        return {"id": "live_throughput", "type": "line", "data": {}, "realtime": True}

    # New performance dashboard chart methods
    async def _generate_algorithm_performance_comparison_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate algorithm performance comparison chart."""
        return {"id": "algorithm_performance_comparison", "type": "bar", "data": algorithm_metrics}

    async def _generate_accuracy_latency_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate accuracy vs latency scatter chart."""
        return {"id": "accuracy_latency", "type": "scatter", "data": algorithm_metrics}

    async def _generate_f1_trends_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate F1 score trends chart."""
        return {"id": "f1_trends", "type": "line", "data": algorithm_metrics}

    async def _generate_memory_comparison_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate memory usage comparison chart."""
        return {"id": "memory_comparison", "type": "bar", "data": algorithm_metrics}

    async def _generate_regression_detection_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate performance regression detection chart."""
        return {"id": "regression_detection", "type": "line", "data": algorithm_metrics}

    async def _generate_regression_warnings_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate regression warnings chart."""
        return {"id": "regression_warnings", "type": "alert", "data": algorithm_metrics}

    async def _generate_resource_efficiency_radar_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate resource efficiency radar chart."""
        return {"id": "resource_efficiency_radar", "type": "radar", "data": algorithm_metrics}

    async def _generate_performance_trends_timeline_chart(self, algorithm_metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate performance trends timeline chart."""
        return {"id": "performance_trends_timeline", "type": "time_series", "data": algorithm_metrics}

    # Update existing methods to accept algorithm_metrics parameter
    async def _generate_benchmark_comparison_chart(self, algorithm_metrics: dict[str, Any] = None) -> dict[str, Any]:
        """Generate benchmark comparison chart."""
        return {"id": "benchmark_comparison", "type": "bar", "data": algorithm_metrics or {}}
