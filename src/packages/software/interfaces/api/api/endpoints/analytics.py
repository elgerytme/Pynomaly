#!/usr/bin/env python3
"""
Advanced Analytics API Endpoints
REST API endpoints for analytics and business intelligence features
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pynomaly_detection.application.services.advanced_analytics_service import (
    AdvancedAnalyticsService,
    AnalyticsQuery,
    AnalyticsTimeframe,
    ChartType,
    MetricType,
)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


# Dependency injection
def get_analytics_service() -> AdvancedAnalyticsService:
    return AdvancedAnalyticsService()


# Pydantic models for API
class AnalyticsQueryRequest(BaseModel):
    """Request model for analytics queries"""

    metric_type: MetricType
    timeframe: AnalyticsTimeframe
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Filtering
    filters: dict[str, Any] = Field(default_factory=dict)
    group_by: list[str] = Field(default_factory=list)

    # Aggregation
    aggregation: str = Field(default="sum", regex="^(sum|avg|count|min|max)$")

    # Visualization
    chart_type: ChartType = ChartType.LINE

    # Advanced options
    include_forecast: bool = False
    include_anomalies: bool = False
    include_trends: bool = True


class AnalyticsResultResponse(BaseModel):
    """Response model for analytics results"""

    query: dict[str, Any]
    data: list[dict[str, Any]]
    metadata: dict[str, Any]
    trends: dict[str, Any]
    anomalies: list[dict[str, Any]]
    forecasts: dict[str, Any]
    chart_data: dict[str, Any]


class BusinessInsightResponse(BaseModel):
    """Response model for business insights"""

    insight_id: str
    title: str
    description: str
    category: str
    severity: str
    confidence: float
    metrics: dict[str, Any]
    recommendations: list[str]
    time_period: str
    affected_entities: list[str]
    generated_at: datetime
    expires_at: datetime | None


class DashboardResponse(BaseModel):
    """Response model for dashboard data"""

    summary: dict[str, Any]
    charts: list[dict[str, Any]]
    insights: list[dict[str, Any]]
    kpis: dict[str, Any]


class CustomReportRequest(BaseModel):
    """Request model for custom reports"""

    title: str = "Custom Analytics Report"
    queries: list[AnalyticsQueryRequest]


class CustomReportResponse(BaseModel):
    """Response model for custom reports"""

    title: str
    generated_at: str
    sections: list[dict[str, Any]]


# API Endpoints
@router.post("/query", response_model=AnalyticsResultResponse)
async def execute_analytics_query(
    request: AnalyticsQueryRequest,
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
):
    """
    Execute an analytics query and return comprehensive results.

    This endpoint allows you to query various metrics and get detailed analytics
    including trends, anomalies, forecasts, and chart data.
    """
    try:
        # Convert request to AnalyticsQuery
        query = AnalyticsQuery(
            metric_type=request.metric_type,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            filters=request.filters,
            group_by=request.group_by,
            aggregation=request.aggregation,
            chart_type=request.chart_type,
            include_forecast=request.include_forecast,
            include_anomalies=request.include_anomalies,
            include_trends=request.include_trends,
        )

        # Execute query
        result = await analytics_service.execute_analytics_query(query)

        # Convert DataFrame to list of dictionaries
        data_records = []
        if not result.data.empty:
            data_records = result.data.reset_index().to_dict("records")

            # Convert datetime objects to strings for JSON serialization
            for record in data_records:
                for key, value in record.items():
                    if isinstance(value, datetime):
                        record[key] = value.isoformat()

        return AnalyticsResultResponse(
            query={
                "metric_type": query.metric_type.value,
                "timeframe": query.timeframe.value,
                "chart_type": query.chart_type.value,
                "aggregation": query.aggregation,
                "filters": query.filters,
                "group_by": query.group_by,
            },
            data=data_records,
            metadata=result.metadata,
            trends=result.trends,
            anomalies=result.anomalies,
            forecasts=result.forecasts,
            chart_data=result.chart_data,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {str(e)}")


@router.get("/insights", response_model=list[BusinessInsightResponse])
async def get_business_insights(
    timeframe: AnalyticsTimeframe = Query(
        AnalyticsTimeframe.WEEK, description="Time period for insights"
    ),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
):
    """
    Get automated business insights for the specified timeframe.

    Returns a list of automatically generated insights based on data analysis,
    including trends, anomalies, and recommendations.
    """
    try:
        insights = await analytics_service.generate_business_insights(timeframe)

        return [
            BusinessInsightResponse(
                insight_id=insight.insight_id,
                title=insight.title,
                description=insight.description,
                category=insight.category,
                severity=insight.severity,
                confidence=insight.confidence,
                metrics=insight.metrics,
                recommendations=insight.recommendations,
                time_period=insight.time_period,
                affected_entities=insight.affected_entities,
                generated_at=insight.generated_at,
                expires_at=insight.expires_at,
            )
            for insight in insights
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate insights: {str(e)}"
        )


@router.get("/dashboard", response_model=DashboardResponse)
async def get_executive_dashboard(
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
):
    """
    Get executive dashboard with key business metrics and insights.

    Returns a comprehensive dashboard view with KPIs, charts, and automated insights
    designed for executive and management consumption.
    """
    try:
        dashboard_data = await analytics_service.create_executive_dashboard()

        return DashboardResponse(
            summary=dashboard_data.get("summary", {}),
            charts=dashboard_data.get("charts", []),
            insights=dashboard_data.get("insights", []),
            kpis=dashboard_data.get("kpis", {}),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create dashboard: {str(e)}"
        )


@router.post("/report", response_model=CustomReportResponse)
async def generate_custom_report(
    request: CustomReportRequest,
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
):
    """
    Generate a custom analytics report with multiple queries.

    Allows creating comprehensive reports by combining multiple analytics queries
    into a single report with multiple sections.
    """
    try:
        # Convert request queries to AnalyticsQuery objects
        queries = []
        for query_request in request.queries:
            query = AnalyticsQuery(
                metric_type=query_request.metric_type,
                timeframe=query_request.timeframe,
                start_date=query_request.start_date,
                end_date=query_request.end_date,
                filters=query_request.filters,
                group_by=query_request.group_by,
                aggregation=query_request.aggregation,
                chart_type=query_request.chart_type,
                include_forecast=query_request.include_forecast,
                include_anomalies=query_request.include_anomalies,
                include_trends=query_request.include_trends,
            )
            queries.append(query)

        # Generate report
        report = await analytics_service.generate_custom_report(queries, request.title)

        return CustomReportResponse(
            title=report["title"],
            generated_at=report["generated_at"],
            sections=report["sections"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/metrics/types")
async def get_metric_types():
    """
    Get available metric types for analytics queries.

    Returns a list of all available metric types that can be used in analytics queries.
    """
    return {
        "metric_types": [
            {
                "value": metric.value,
                "name": metric.name,
                "description": _get_metric_description(metric),
            }
            for metric in MetricType
        ]
    }


@router.get("/timeframes")
async def get_timeframes():
    """
    Get available timeframes for analytics queries.

    Returns a list of all available timeframes that can be used in analytics queries.
    """
    return {
        "timeframes": [
            {
                "value": timeframe.value,
                "name": timeframe.name,
                "description": _get_timeframe_description(timeframe),
            }
            for timeframe in AnalyticsTimeframe
        ]
    }


@router.get("/charts/types")
async def get_chart_types():
    """
    Get available chart types for data visualization.

    Returns a list of all available chart types that can be used for visualizing analytics data.
    """
    return {
        "chart_types": [
            {
                "value": chart.value,
                "name": chart.name,
                "description": _get_chart_description(chart),
            }
            for chart in ChartType
        ]
    }


@router.get("/metrics/{metric_type}/sample")
async def get_sample_data(
    metric_type: MetricType,
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
):
    """
    Get sample data for a specific metric type.

    Useful for understanding the structure and format of data for each metric type.
    """
    try:
        # Create a sample query
        sample_query = AnalyticsQuery(
            metric_type=metric_type,
            timeframe=AnalyticsTimeframe.DAY,
            include_trends=False,
            include_anomalies=False,
            include_forecast=False,
        )

        # Get sample data (limited to 10 records)
        result = await analytics_service.execute_analytics_query(sample_query)

        sample_data = []
        if not result.data.empty:
            sample_data = result.data.head(10).reset_index().to_dict("records")

            # Convert datetime objects to strings
            for record in sample_data:
                for key, value in record.items():
                    if isinstance(value, datetime):
                        record[key] = value.isoformat()

        return {
            "metric_type": metric_type.value,
            "sample_data": sample_data,
            "columns": list(result.data.columns) if not result.data.empty else [],
            "total_records": len(result.data),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get sample data: {str(e)}"
        )


# Utility functions for descriptions
def _get_metric_description(metric: MetricType) -> str:
    """Get description for metric type"""
    descriptions = {
        MetricType.DETECTION_VOLUME: "Number of anomaly detections performed over time",
        MetricType.ANOMALY_RATE: "Percentage of data points identified as anomalous",
        MetricType.MODEL_PERFORMANCE: "ML model accuracy, precision, recall, and other performance metrics",
        MetricType.SYSTEM_HEALTH: "System resource usage, response times, and health metrics",
        MetricType.USER_ENGAGEMENT: "User activity, API usage, and engagement metrics",
        MetricType.DATA_QUALITY: "Data completeness, accuracy, and quality scores",
        MetricType.BUSINESS_KPI: "Business key performance indicators and revenue metrics",
    }
    return descriptions.get(metric, "Analytics metric")


def _get_timeframe_description(timeframe: AnalyticsTimeframe) -> str:
    """Get description for timeframe"""
    descriptions = {
        AnalyticsTimeframe.HOUR: "Last hour of data",
        AnalyticsTimeframe.DAY: "Last 24 hours of data",
        AnalyticsTimeframe.WEEK: "Last 7 days of data",
        AnalyticsTimeframe.MONTH: "Last 30 days of data",
        AnalyticsTimeframe.QUARTER: "Last 90 days of data",
        AnalyticsTimeframe.YEAR: "Last 365 days of data",
        AnalyticsTimeframe.CUSTOM: "Custom date range",
    }
    return descriptions.get(timeframe, "Time period")


def _get_chart_description(chart: ChartType) -> str:
    """Get description for chart type"""
    descriptions = {
        ChartType.LINE: "Line chart for time series data",
        ChartType.BAR: "Bar chart for categorical data",
        ChartType.PIE: "Pie chart for proportional data",
        ChartType.SCATTER: "Scatter plot for correlation analysis",
        ChartType.HEATMAP: "Heatmap for correlation matrices",
        ChartType.HISTOGRAM: "Histogram for distribution analysis",
        ChartType.BOX_PLOT: "Box plot for statistical distribution",
        ChartType.AREA: "Area chart for cumulative data",
        ChartType.CANDLESTICK: "Candlestick chart for OHLC data",
        ChartType.CORRELATION_MATRIX: "Correlation matrix visualization",
    }
    return descriptions.get(chart, "Chart visualization")


# Health check endpoint
@router.get("/health")
async def analytics_health():
    """Health check endpoint for analytics service"""
    return {
        "status": "healthy",
        "service": "advanced_analytics",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }
