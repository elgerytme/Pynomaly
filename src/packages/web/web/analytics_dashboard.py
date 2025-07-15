#!/usr/bin/env python3
"""
Advanced Analytics Dashboard for Pynomaly production deployment.
This module provides comprehensive analytics, insights, and business intelligence.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for analytics
class AnalyticsQuery(BaseModel):
    """Analytics query parameters."""

    start_date: datetime
    end_date: datetime
    metric_type: str
    aggregation: str = "daily"
    filters: dict[str, Any] | None = None


class AnalyticsResponse(BaseModel):
    """Analytics response data."""

    data: list[dict[str, Any]]
    summary: dict[str, Any]
    charts: list[dict[str, Any]]
    insights: list[str]
    recommendations: list[str]


class DashboardMetrics(BaseModel):
    """Dashboard metrics data."""

    total_detections: int
    anomaly_rate: float
    avg_response_time: float
    system_uptime: float
    data_processed_mb: float
    active_users: int
    alert_count: int


# Analytics service
class AnalyticsService:
    """Advanced analytics service for Pynomaly."""

    def __init__(self):
        """Initialize analytics service."""
        self.cache = {}
        self.metrics_history = []

    async def get_detection_analytics(self, query: AnalyticsQuery) -> dict[str, Any]:
        """Get anomaly detection analytics."""
        logger.info(
            f"Generating detection analytics for {query.start_date} to {query.end_date}"
        )

        # Simulate analytics data (in production, this would query actual databases)
        date_range = pd.date_range(start=query.start_date, end=query.end_date, freq="D")

        # Generate synthetic data for demonstration
        detection_data = []
        for date in date_range:
            daily_detections = np.random.poisson(100)  # Average 100 detections per day
            anomaly_rate = np.random.uniform(0.02, 0.08)  # 2-8% anomaly rate

            detection_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "total_detections": daily_detections,
                    "anomalies_found": int(daily_detections * anomaly_rate),
                    "anomaly_rate": anomaly_rate,
                    "avg_confidence": np.random.uniform(0.7, 0.95),
                    "processing_time_ms": np.random.uniform(50, 200),
                }
            )

        # Calculate summary statistics
        df = pd.DataFrame(detection_data)
        summary = {
            "total_detections": int(df["total_detections"].sum()),
            "total_anomalies": int(df["anomalies_found"].sum()),
            "avg_anomaly_rate": float(df["anomaly_rate"].mean()),
            "avg_confidence": float(df["avg_confidence"].mean()),
            "avg_processing_time": float(df["processing_time_ms"].mean()),
            "period_days": len(date_range),
        }

        # Generate insights
        insights = self._generate_detection_insights(df, summary)

        # Generate recommendations
        recommendations = self._generate_detection_recommendations(df, summary)

        return {
            "data": detection_data,
            "summary": summary,
            "insights": insights,
            "recommendations": recommendations,
        }

    async def get_performance_analytics(self, query: AnalyticsQuery) -> dict[str, Any]:
        """Get system performance analytics."""
        logger.info(
            f"Generating performance analytics for {query.start_date} to {query.end_date}"
        )

        date_range = pd.date_range(start=query.start_date, end=query.end_date, freq="H")

        # Generate synthetic performance data
        performance_data = []
        for timestamp in date_range:
            cpu_usage = np.random.uniform(20, 80)
            memory_usage = np.random.uniform(30, 85)
            response_time = np.random.lognormal(
                4, 0.5
            )  # Log-normal distribution for response times

            performance_data.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "response_time_ms": response_time,
                    "throughput_rps": np.random.uniform(50, 200),
                    "error_rate": np.random.uniform(0, 0.05),
                    "disk_usage": np.random.uniform(40, 70),
                }
            )

        # Calculate summary statistics
        df = pd.DataFrame(performance_data)
        summary = {
            "avg_cpu_usage": float(df["cpu_usage"].mean()),
            "max_cpu_usage": float(df["cpu_usage"].max()),
            "avg_memory_usage": float(df["memory_usage"].mean()),
            "max_memory_usage": float(df["memory_usage"].max()),
            "avg_response_time": float(df["response_time_ms"].mean()),
            "p95_response_time": float(df["response_time_ms"].quantile(0.95)),
            "avg_throughput": float(df["throughput_rps"].mean()),
            "avg_error_rate": float(df["error_rate"].mean()),
            "uptime_percentage": 99.8,  # Simulated uptime
        }

        # Generate insights
        insights = self._generate_performance_insights(df, summary)

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(df, summary)

        return {
            "data": performance_data,
            "summary": summary,
            "insights": insights,
            "recommendations": recommendations,
        }

    async def get_business_analytics(self, query: AnalyticsQuery) -> dict[str, Any]:
        """Get business intelligence analytics."""
        logger.info(
            f"Generating business analytics for {query.start_date} to {query.end_date}"
        )

        date_range = pd.date_range(start=query.start_date, end=query.end_date, freq="D")

        # Generate synthetic business data
        business_data = []
        for date in date_range:
            active_users = np.random.poisson(50)
            data_processed = np.random.uniform(100, 1000)  # MB
            cost_per_gb = 0.05  # $0.05 per GB

            business_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "active_users": active_users,
                    "data_processed_mb": data_processed,
                    "processing_cost": (data_processed / 1024) * cost_per_gb,
                    "api_calls": np.random.poisson(1000),
                    "storage_used_gb": np.random.uniform(10, 100),
                    "bandwidth_used_gb": np.random.uniform(5, 50),
                }
            )

        # Calculate summary statistics
        df = pd.DataFrame(business_data)
        summary = {
            "total_active_users": int(df["active_users"].sum()),
            "avg_daily_users": float(df["active_users"].mean()),
            "total_data_processed_gb": float(df["data_processed_mb"].sum() / 1024),
            "total_processing_cost": float(df["processing_cost"].sum()),
            "total_api_calls": int(df["api_calls"].sum()),
            "avg_storage_used": float(df["storage_used_gb"].mean()),
            "total_bandwidth": float(df["bandwidth_used_gb"].sum()),
        }

        # Generate insights
        insights = self._generate_business_insights(df, summary)

        # Generate recommendations
        recommendations = self._generate_business_recommendations(df, summary)

        return {
            "data": business_data,
            "summary": summary,
            "insights": insights,
            "recommendations": recommendations,
        }

    def _generate_detection_insights(
        self, df: pd.DataFrame, summary: dict[str, Any]
    ) -> list[str]:
        """Generate insights from detection data."""
        insights = []

        if summary["avg_anomaly_rate"] > 0.05:
            insights.append(
                "High anomaly rate detected - may indicate data quality issues or model drift"
            )

        if summary["avg_confidence"] < 0.8:
            insights.append(
                "Average confidence is below threshold - consider model retraining"
            )

        if summary["avg_processing_time"] > 150:
            insights.append(
                "Processing time is above optimal - consider performance optimization"
            )

        # Trend analysis
        if len(df) >= 7:
            recent_rate = df.tail(7)["anomaly_rate"].mean()
            earlier_rate = df.head(7)["anomaly_rate"].mean()

            if recent_rate > earlier_rate * 1.2:
                insights.append(
                    "Anomaly rate is trending upward - requires investigation"
                )
            elif recent_rate < earlier_rate * 0.8:
                insights.append(
                    "Anomaly rate is trending downward - system performance improving"
                )

        if not insights:
            insights.append("Detection system operating within normal parameters")

        return insights

    def _generate_performance_insights(
        self, df: pd.DataFrame, summary: dict[str, Any]
    ) -> list[str]:
        """Generate insights from performance data."""
        insights = []

        if summary["avg_cpu_usage"] > 70:
            insights.append("High CPU usage detected - consider scaling resources")

        if summary["avg_memory_usage"] > 80:
            insights.append(
                "High memory usage detected - potential memory leak or need for optimization"
            )

        if summary["p95_response_time"] > 1000:
            insights.append(
                "95th percentile response time exceeds 1 second - performance optimization needed"
            )

        if summary["avg_error_rate"] > 0.01:
            insights.append("Error rate above 1% - investigate and fix error sources")

        if summary["uptime_percentage"] < 99.5:
            insights.append(
                "System uptime below target - implement reliability improvements"
            )

        if not insights:
            insights.append("System performance is within acceptable parameters")

        return insights

    def _generate_business_insights(
        self, df: pd.DataFrame, summary: dict[str, Any]
    ) -> list[str]:
        """Generate insights from business data."""
        insights = []

        user_growth = df["active_users"].pct_change().mean()
        if user_growth > 0.1:
            insights.append("User base is growing rapidly - prepare for scaling")
        elif user_growth < -0.1:
            insights.append("User base is declining - investigate retention issues")

        if summary["total_processing_cost"] > 1000:
            insights.append(
                "Processing costs are high - consider cost optimization strategies"
            )

        cost_per_user = summary["total_processing_cost"] / summary["total_active_users"]
        if cost_per_user > 5:
            insights.append("High cost per user - optimize processing efficiency")

        if summary["avg_storage_used"] > 80:
            insights.append("Storage usage is high - implement data archiving strategy")

        if not insights:
            insights.append("Business metrics are healthy and within expected ranges")

        return insights

    def _generate_detection_recommendations(
        self, df: pd.DataFrame, summary: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for detection improvements."""
        recommendations = []

        if summary["avg_confidence"] < 0.85:
            recommendations.append(
                "Retrain models with recent data to improve confidence scores"
            )

        if summary["avg_processing_time"] > 100:
            recommendations.append(
                "Optimize model inference pipeline for faster processing"
            )

        if summary["avg_anomaly_rate"] > 0.06:
            recommendations.append(
                "Review data preprocessing steps and model thresholds"
            )

        recommendations.extend(
            [
                "Implement A/B testing for model improvements",
                "Set up automated model retraining pipeline",
                "Add more diverse training data sources",
                "Implement ensemble methods for better accuracy",
            ]
        )

        return recommendations

    def _generate_performance_recommendations(
        self, df: pd.DataFrame, summary: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for performance improvements."""
        recommendations = []

        if summary["avg_cpu_usage"] > 60:
            recommendations.append("Scale horizontally or upgrade CPU resources")

        if summary["avg_memory_usage"] > 75:
            recommendations.append("Optimize memory usage or increase available RAM")

        if summary["p95_response_time"] > 500:
            recommendations.append("Implement caching and database query optimization")

        recommendations.extend(
            [
                "Set up auto-scaling based on resource usage",
                "Implement connection pooling for database connections",
                "Add CDN for static asset delivery",
                "Optimize database indexes for common queries",
            ]
        )

        return recommendations

    def _generate_business_recommendations(
        self, df: pd.DataFrame, summary: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for business improvements."""
        recommendations = []

        if summary["total_active_users"] > 0:
            cost_per_user = (
                summary["total_processing_cost"] / summary["total_active_users"]
            )
            if cost_per_user > 2:
                recommendations.append(
                    "Implement cost optimization strategies to reduce per-user costs"
                )

        recommendations.extend(
            [
                "Implement usage-based pricing model",
                "Add user analytics to improve retention",
                "Optimize resource allocation based on usage patterns",
                "Implement data lifecycle management policies",
                "Add business intelligence dashboards for stakeholders",
            ]
        )

        return recommendations


# Initialize analytics service
analytics_service = AnalyticsService()

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Get the main analytics dashboard."""
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pynomaly Analytics Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: #667eea;
            }
            .metric-label {
                color: #666;
                font-size: 0.9rem;
                margin-top: 5px;
            }
            .chart-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .insights-panel {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .insight-item {
                background: #f8f9fa;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }
            .recommendation-item {
                background: #fff3cd;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }
            .refresh-btn {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1rem;
            }
            .refresh-btn:hover {
                background: #5a67d8;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Pynomaly Analytics Dashboard</h1>
            <p>Real-time analytics and insights for your anomaly detection system</p>
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh Data</button>
        </div>

        <div class="metrics-grid" id="metrics-grid">
            <div class="loading">Loading metrics...</div>
        </div>

        <div class="chart-container">
            <h3>Detection Analytics</h3>
            <div id="detection-chart"></div>
        </div>

        <div class="chart-container">
            <h3>Performance Analytics</h3>
            <div id="performance-chart"></div>
        </div>

        <div class="chart-container">
            <h3>Business Analytics</h3>
            <div id="business-chart"></div>
        </div>

        <div class="insights-panel">
            <h3>📊 Insights & Recommendations</h3>
            <div id="insights-content">
                <div class="loading">Loading insights...</div>
            </div>
        </div>

        <script>
            // Dashboard JavaScript
            let dashboardData = {};

            async function refreshDashboard() {
                try {
                    // Load metrics
                    await loadMetrics();

                    // Load analytics data
                    await loadAnalytics();

                    // Load insights
                    await loadInsights();

                    console.log('Dashboard refreshed successfully');
                } catch (error) {
                    console.error('Dashboard refresh failed:', error);
                }
            }

            async function loadMetrics() {
                try {
                    const response = await fetch('/analytics/metrics');
                    const data = await response.json();

                    const metricsGrid = document.getElementById('metrics-grid');
                    metricsGrid.innerHTML = `
                        <div class="metric-card">
                            <div class="metric-value">${data.total_detections.toLocaleString()}</div>
                            <div class="metric-label">Total Detections</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(data.anomaly_rate * 100).toFixed(1)}%</div>
                            <div class="metric-label">Anomaly Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.avg_response_time.toFixed(0)}ms</div>
                            <div class="metric-label">Avg Response Time</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.system_uptime.toFixed(1)}%</div>
                            <div class="metric-label">System Uptime</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(data.data_processed_mb / 1024).toFixed(1)}GB</div>
                            <div class="metric-label">Data Processed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.active_users}</div>
                            <div class="metric-label">Active Users</div>
                        </div>
                    `;
                } catch (error) {
                    console.error('Failed to load metrics:', error);
                }
            }

            async function loadAnalytics() {
                try {
                    const endDate = new Date();
                    const startDate = new Date(endDate.getTime() - 7 * 24 * 60 * 60 * 1000); // 7 days ago

                    // Load detection analytics
                    const detectionResponse = await fetch(`/analytics/detection?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}&metric_type=daily&aggregation=daily`);
                    const detectionData = await detectionResponse.json();

                    // Create detection chart
                    const detectionChart = {
                        data: [
                            {
                                x: detectionData.data.map(d => d.date),
                                y: detectionData.data.map(d => d.total_detections),
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: 'Total Detections',
                                line: {color: '#667eea'}
                            },
                            {
                                x: detectionData.data.map(d => d.date),
                                y: detectionData.data.map(d => d.anomalies_found),
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: 'Anomalies Found',
                                line: {color: '#f093fb'}
                            }
                        ],
                        layout: {
                            title: 'Detection Trends (Last 7 Days)',
                            xaxis: {title: 'Date'},
                            yaxis: {title: 'Count'},
                            showlegend: true
                        }
                    };

                    Plotly.newPlot('detection-chart', detectionChart.data, detectionChart.layout);

                    // Load performance analytics
                    const performanceResponse = await fetch(`/analytics/performance?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}&metric_type=hourly&aggregation=hourly`);
                    const performanceData = await performanceResponse.json();

                    // Create performance chart (sample the data to avoid too many points)
                    const sampledPerformance = performanceData.data.filter((_, index) => index % 6 === 0); // Every 6th hour

                    const performanceChart = {
                        data: [
                            {
                                x: sampledPerformance.map(d => d.timestamp),
                                y: sampledPerformance.map(d => d.response_time_ms),
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Response Time (ms)',
                                line: {color: '#667eea'}
                            }
                        ],
                        layout: {
                            title: 'Performance Trends (Last 7 Days)',
                            xaxis: {title: 'Time'},
                            yaxis: {title: 'Response Time (ms)'},
                            showlegend: true
                        }
                    };

                    Plotly.newPlot('performance-chart', performanceChart.data, performanceChart.layout);

                    // Load business analytics
                    const businessResponse = await fetch(`/analytics/business?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}&metric_type=daily&aggregation=daily`);
                    const businessData = await businessResponse.json();

                    // Create business chart
                    const businessChart = {
                        data: [
                            {
                                x: businessData.data.map(d => d.date),
                                y: businessData.data.map(d => d.active_users),
                                type: 'bar',
                                name: 'Active Users',
                                marker: {color: '#667eea'}
                            }
                        ],
                        layout: {
                            title: 'Business Metrics (Last 7 Days)',
                            xaxis: {title: 'Date'},
                            yaxis: {title: 'Active Users'},
                            showlegend: true
                        }
                    };

                    Plotly.newPlot('business-chart', businessChart.data, businessChart.layout);

                    // Store data for insights
                    dashboardData = {
                        detection: detectionData,
                        performance: performanceData,
                        business: businessData
                    };

                } catch (error) {
                    console.error('Failed to load analytics:', error);
                }
            }

            async function loadInsights() {
                try {
                    const insightsContent = document.getElementById('insights-content');

                    if (!dashboardData.detection) {
                        insightsContent.innerHTML = '<div class="loading">Loading insights...</div>';
                        return;
                    }

                    let content = '<h4>🔍 Key Insights</h4>';

                    // Add detection insights
                    dashboardData.detection.insights.forEach(insight => {
                        content += `<div class="insight-item">📊 ${insight}</div>`;
                    });

                    // Add performance insights
                    dashboardData.performance.insights.forEach(insight => {
                        content += `<div class="insight-item">⚡ ${insight}</div>`;
                    });

                    // Add business insights
                    dashboardData.business.insights.forEach(insight => {
                        content += `<div class="insight-item">💼 ${insight}</div>`;
                    });

                    content += '<h4>💡 Recommendations</h4>';

                    // Add recommendations
                    const allRecommendations = [
                        ...dashboardData.detection.recommendations.slice(0, 3),
                        ...dashboardData.performance.recommendations.slice(0, 3),
                        ...dashboardData.business.recommendations.slice(0, 3)
                    ];

                    allRecommendations.forEach(rec => {
                        content += `<div class="recommendation-item">💡 ${rec}</div>`;
                    });

                    insightsContent.innerHTML = content;

                } catch (error) {
                    console.error('Failed to load insights:', error);
                }
            }

            // Initialize dashboard
            refreshDashboard();

            // Auto-refresh every 5 minutes
            setInterval(refreshDashboard, 5 * 60 * 1000);
        </script>
    </body>
    </html>
    """

    return dashboard_html


@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get current dashboard metrics."""
    # In production, this would query actual metrics from monitoring systems
    return DashboardMetrics(
        total_detections=np.random.randint(10000, 50000),
        anomaly_rate=np.random.uniform(0.02, 0.08),
        avg_response_time=np.random.uniform(50, 200),
        system_uptime=np.random.uniform(99.5, 99.9),
        data_processed_mb=np.random.uniform(1000, 10000),
        active_users=np.random.randint(10, 100),
        alert_count=np.random.randint(0, 5),
    )


@router.get("/detection", response_model=AnalyticsResponse)
async def get_detection_analytics(
    start_date: datetime = Query(..., description="Start date for analytics"),
    end_date: datetime = Query(..., description="End date for analytics"),
    metric_type: str = Query(
        "daily", description="Metric type (daily, hourly, weekly)"
    ),
    aggregation: str = Query("daily", description="Aggregation level"),
):
    """Get detection analytics."""
    query = AnalyticsQuery(
        start_date=start_date,
        end_date=end_date,
        metric_type=metric_type,
        aggregation=aggregation,
    )

    result = await analytics_service.get_detection_analytics(query)

    return AnalyticsResponse(
        data=result["data"],
        summary=result["summary"],
        charts=[],  # Charts are generated in frontend
        insights=result["insights"],
        recommendations=result["recommendations"],
    )


@router.get("/performance", response_model=AnalyticsResponse)
async def get_performance_analytics(
    start_date: datetime = Query(..., description="Start date for analytics"),
    end_date: datetime = Query(..., description="End date for analytics"),
    metric_type: str = Query(
        "hourly", description="Metric type (daily, hourly, weekly)"
    ),
    aggregation: str = Query("hourly", description="Aggregation level"),
):
    """Get performance analytics."""
    query = AnalyticsQuery(
        start_date=start_date,
        end_date=end_date,
        metric_type=metric_type,
        aggregation=aggregation,
    )

    result = await analytics_service.get_performance_analytics(query)

    return AnalyticsResponse(
        data=result["data"],
        summary=result["summary"],
        charts=[],  # Charts are generated in frontend
        insights=result["insights"],
        recommendations=result["recommendations"],
    )


@router.get("/business", response_model=AnalyticsResponse)
async def get_business_analytics(
    start_date: datetime = Query(..., description="Start date for analytics"),
    end_date: datetime = Query(..., description="End date for analytics"),
    metric_type: str = Query(
        "daily", description="Metric type (daily, hourly, weekly)"
    ),
    aggregation: str = Query("daily", description="Aggregation level"),
):
    """Get business analytics."""
    query = AnalyticsQuery(
        start_date=start_date,
        end_date=end_date,
        metric_type=metric_type,
        aggregation=aggregation,
    )

    result = await analytics_service.get_business_analytics(query)

    return AnalyticsResponse(
        data=result["data"],
        summary=result["summary"],
        charts=[],  # Charts are generated in frontend
        insights=result["insights"],
        recommendations=result["recommendations"],
    )


@router.get("/export/{format}")
async def export_analytics(
    format: str,
    start_date: datetime = Query(..., description="Start date for export"),
    end_date: datetime = Query(..., description="End date for export"),
    metric_type: str = Query("daily", description="Metric type to export"),
):
    """Export analytics data in various formats."""
    if format not in ["json", "csv", "xlsx"]:
        raise HTTPException(status_code=400, detail="Unsupported format")

    query = AnalyticsQuery(
        start_date=start_date, end_date=end_date, metric_type=metric_type
    )

    # Get all analytics data
    detection_data = await analytics_service.get_detection_analytics(query)
    performance_data = await analytics_service.get_performance_analytics(query)
    business_data = await analytics_service.get_business_analytics(query)

    export_data = {
        "export_info": {
            "generated_at": datetime.now().isoformat(),
            "period": f"{start_date.date()} to {end_date.date()}",
            "metric_type": metric_type,
        },
        "detection_analytics": detection_data,
        "performance_analytics": performance_data,
        "business_analytics": business_data,
    }

    if format == "json":
        return JSONResponse(content=export_data)
    elif format == "csv":
        # Convert to CSV format (simplified)
        csv_data = "category,metric,value\n"
        for category, data in export_data.items():
            if isinstance(data, dict) and "summary" in data:
                for metric, value in data["summary"].items():
                    csv_data += f"{category},{metric},{value}\n"

        return Response(content=csv_data, media_type="text/csv")
    elif format == "xlsx":
        # In production, you would use pandas to create Excel files
        return JSONResponse(content={"message": "Excel export not implemented yet"})


# Make router available for import
__all__ = ["router", "analytics_service"]
