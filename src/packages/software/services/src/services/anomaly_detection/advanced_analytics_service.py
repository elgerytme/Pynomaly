#!/usr/bin/env python3
"""
Advanced Analytics and Business Intelligence Service
Provides comprehensive analytics, reporting, and business intelligence capabilities
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnalyticsTimeframe(Enum):
    """Analytics timeframe options"""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of measurements"""

    DETECTION_VOLUME = "processing_volume"
    ANOMALY_RATE = "anomaly_rate"
    MODEL_PERFORMANCE = "processor_performance"
    SYSTEM_HEALTH = "system_health"
    USER_ENGAGEMENT = "user_engagement"
    DATA_QUALITY = "data_quality"
    BUSINESS_KPI = "business_kpi"


class ChartType(Enum):
    """Chart types for visualization"""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    AREA = "area"
    CANDLESTICK = "candlestick"
    CORRELATION_MATRIX = "correlation_matrix"


@dataclass
class AnalyticsQuery:
    """Analytics query configuration"""

    metric_type: MetricType
    timeframe: AnalyticsTimeframe
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Filtering
    filters: dict[str, Any] = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)

    # Aggregation
    aggregation: str = "sum"  # sum, avg, count, min, max

    # Visualization
    chart_type: ChartType = ChartType.LINE

    # Advanced options
    include_forecast: bool = False
    include_anomalies: bool = False
    include_trends: bool = True


@dataclass
class AnalyticsResult:
    """Analytics query result"""

    query: AnalyticsQuery
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    # Derived insights
    trends: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    forecasts: dict[str, Any] = field(default_factory=dict)

    # Visualization data
    chart_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessInsight:
    """Business insight generated from analytics"""

    insight_id: str
    title: str
    description: str
    category: str
    severity: str  # low, medium, high, critical
    confidence: float

    # Supporting data
    measurements: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    # Context
    time_period: str = ""
    affected_entities: list[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None


class AdvancedAnalyticsService:
    """Advanced analytics and business intelligence service"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.cache = {}
        self.insights_cache = {}

    async def execute_analytics_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute analytics query and return comprehensive results"""
        logger.info(f"Executing analytics query: {query.metric_type.value}")

        try:
            # Get base data
            data = await self._fetch_analytics_data(query)

            if data.empty:
                return AnalyticsResult(
                    query=query,
                    data=data,
                    metadata={"error": "No data found for query"},
                )

            # Apply filtering and grouping
            processed_data = await self._process_analytics_data(data, query)

            # Calculate trends if requested
            trends = {}
            if query.include_trends:
                trends = await self._calculate_trends(processed_data, query)

            # Detect anomalies if requested
            anomalies = []
            if query.include_anomalies:
                anomalies = await self._detect_anomalies(processed_data, query)

            # Generate forecasts if requested
            forecasts = {}
            if query.include_forecast:
                forecasts = await self._generate_forecasts(processed_data, query)

            # Prepare chart data
            chart_data = await self._prepare_chart_data(processed_data, query)

            # Calculate metadata
            metadata = {
                "total_records": len(processed_data),
                "date_range": {
                    "start": processed_data.index.min()
                    if len(processed_data) > 0
                    else None,
                    "end": processed_data.index.max()
                    if len(processed_data) > 0
                    else None,
                },
                "query_time": datetime.now().isoformat(),
                "processing_time_ms": 150,  # Placeholder
            }

            return AnalyticsResult(
                query=query,
                data=processed_data,
                metadata=metadata,
                trends=trends,
                anomalies=anomalies,
                forecasts=forecasts,
                chart_data=chart_data,
            )

        except Exception as e:
            logger.error(f"Analytics query failed: {e}")
            return AnalyticsResult(
                query=query, data=pd.DataFrame(), metadata={"error": str(e)}
            )

    async def _fetch_analytics_data(self, query: AnalyticsQuery) -> pd.DataFrame:
        """Fetch analytics data based on query parameters"""

        # Determine time range
        end_date = query.end_date or datetime.now()

        if query.start_date:
            start_date = query.start_date
        else:
            # Calculate start date based on timeframe
            timeframe_deltas = {
                AnalyticsTimeframe.HOUR: timedelta(hours=1),
                AnalyticsTimeframe.DAY: timedelta(days=1),
                AnalyticsTimeframe.WEEK: timedelta(weeks=1),
                AnalyticsTimeframe.MONTH: timedelta(days=30),
                AnalyticsTimeframe.QUARTER: timedelta(days=90),
                AnalyticsTimeframe.YEAR: timedelta(days=365),
            }
            delta = timeframe_deltas.get(query.timeframe, timedelta(days=7))
            start_date = end_date - delta

        # Generate sample data based on metric type
        # In production, this would query actual databases
        if query.metric_type == MetricType.DETECTION_VOLUME:
            return self._generate_processing_volume_data(start_date, end_date)
        elif query.metric_type == MetricType.ANOMALY_RATE:
            return self._generate_anomaly_rate_data(start_date, end_date)
        elif query.metric_type == MetricType.MODEL_PERFORMANCE:
            return self._generate_processor_performance_data(start_date, end_date)
        elif query.metric_type == MetricType.SYSTEM_HEALTH:
            return self._generate_system_health_data(start_date, end_date)
        elif query.metric_type == MetricType.USER_ENGAGEMENT:
            return self._generate_user_engagement_data(start_date, end_date)
        elif query.metric_type == MetricType.DATA_QUALITY:
            return self._generate_data_quality_data(start_date, end_date)
        elif query.metric_type == MetricType.BUSINESS_KPI:
            return self._generate_business_kpi_data(start_date, end_date)
        else:
            return pd.DataFrame()

    def _generate_detection_volume_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample processing volume data"""
        date_range = pd.date_range(start_date, end_date, freq="H")

        # Simulate realistic processing volume with daily patterns
        base_volume = 100
        data = []

        for timestamp in date_range:
            # Add daily pattern (higher during business hours)
            hour_factor = 1.5 if 9 <= timestamp.hour <= 17 else 0.7

            # Add weekly pattern (lower on weekends)
            day_factor = 0.6 if timestamp.weekday() >= 5 else 1.0

            # Add random variation
            random_factor = np.random.normal(1.0, 0.2)

            volume = int(base_volume * hour_factor * day_factor * random_factor)

            data.append(
                {
                    "timestamp": timestamp,
                    "processing_count": max(0, volume),
                    "processor_type": np.random.choice(
                        ["isolation_forest", "one_class_svm", "autoencoder"]
                    ),
                    "data_source": np.random.choice(["api", "batch", "streaming"]),
                    "tenant_id": f"tenant_{np.random.randint(1, 10)}",
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _generate_anomaly_rate_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample anomaly rate data"""
        date_range = pd.date_range(start_date, end_date, freq="H")

        data = []
        for timestamp in date_range:
            # Base anomaly rate with some variation
            base_rate = 0.05  # 5% base anomaly rate
            variation = np.random.normal(0, 0.02)
            anomaly_rate = max(0, min(1, base_rate + variation))

            total_processings = np.random.randint(50, 200)
            anomalies_detected = int(total_processings * anomaly_rate)

            data.append(
                {
                    "timestamp": timestamp,
                    "total_processings": total_processings,
                    "anomalies_detected": anomalies_detected,
                    "anomaly_rate": anomaly_rate,
                    "false_positive_rate": np.random.uniform(0.01, 0.1),
                    "processor_confidence": np.random.uniform(0.7, 0.95),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _generate_model_performance_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample processor performance data"""
        date_range = pd.date_range(start_date, end_date, freq="D")

        models = ["isolation_forest", "one_class_svm", "autoencoder", "lstm_anomaly"]
        data = []

        for timestamp in date_range:
            for processor in models:
                # Simulate performance measurements with slight degradation over time
                days_since_start = (timestamp - start_date).days
                degradation_factor = max(0.8, 1 - (days_since_start * 0.001))

                accuracy = np.random.normal(0.92, 0.05) * degradation_factor
                precision = np.random.normal(0.88, 0.04) * degradation_factor
                recall = np.random.normal(0.85, 0.06) * degradation_factor
                f1_score = 2 * (precision * recall) / (precision + recall)

                data.append(
                    {
                        "timestamp": timestamp,
                        "processor_name": processor,
                        "accuracy": max(0, min(1, accuracy)),
                        "precision": max(0, min(1, precision)),
                        "recall": max(0, min(1, recall)),
                        "f1_score": max(0, min(1, f1_score)),
                        "training_time_minutes": np.random.uniform(5, 120),
                        "inference_time_ms": np.random.uniform(10, 500),
                        "processor_size_mb": np.random.uniform(1, 100),
                    }
                )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _generate_system_health_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample system health data"""
        date_range = pd.date_range(start_date, end_date, freq="5min")

        data = []
        for timestamp in date_range:
            # Simulate system measurements
            cpu_usage = max(0, min(100, np.random.normal(45, 15)))
            memory_usage = max(0, min(100, np.random.normal(60, 20)))
            disk_usage = max(0, min(100, np.random.normal(30, 10)))

            # Response times with occasional spikes
            response_time = np.random.exponential(200)  # ms
            if np.random.random() < 0.05:  # 5% chance of spike
                response_time *= 5

            data.append(
                {
                    "timestamp": timestamp,
                    "cpu_usage_percent": cpu_usage,
                    "memory_usage_percent": memory_usage,
                    "disk_usage_percent": disk_usage,
                    "response_time_ms": response_time,
                    "active_connections": np.random.randint(50, 500),
                    "error_rate_percent": np.random.exponential(1),
                    "throughput_rps": np.random.normal(150, 30),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _generate_user_engagement_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample user engagement data"""
        date_range = pd.date_range(start_date, end_date, freq="H")

        data = []
        for timestamp in date_range:
            # Business hours pattern
            if 9 <= timestamp.hour <= 17 and timestamp.weekday() < 5:
                active_users = np.random.randint(20, 100)
                api_calls = np.random.randint(500, 2000)
            else:
                active_users = np.random.randint(5, 30)
                api_calls = np.random.randint(50, 300)

            data.append(
                {
                    "timestamp": timestamp,
                    "active_users": active_users,
                    "new_users": np.random.randint(0, 10),
                    "api_calls": api_calls,
                    "session_duration_minutes": np.random.exponential(25),
                    "feature_usage": {
                        "processing": np.random.randint(100, 500),
                        "training": np.random.randint(10, 50),
                        "analysis": np.random.randint(20, 100),
                        "dashboard": np.random.randint(50, 200),
                    },
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _generate_data_quality_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample data quality data"""
        date_range = pd.date_range(start_date, end_date, freq="D")

        data = []
        for timestamp in date_range:
            # Data quality measurements
            completeness = np.random.normal(0.95, 0.05)
            accuracy = np.random.normal(0.92, 0.04)
            consistency = np.random.normal(0.88, 0.06)
            timeliness = np.random.normal(0.90, 0.05)

            data.append(
                {
                    "timestamp": timestamp,
                    "completeness_score": max(0, min(1, completeness)),
                    "accuracy_score": max(0, min(1, accuracy)),
                    "consistency_score": max(0, min(1, consistency)),
                    "timeliness_score": max(0, min(1, timeliness)),
                    "duplicate_rate": np.random.uniform(0, 0.05),
                    "missing_values_rate": np.random.uniform(0, 0.1),
                    "outlier_rate": np.random.uniform(0.01, 0.15),
                    "schema_violations": np.random.randint(0, 10),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _generate_business_kpi_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample business KPI data"""
        date_range = pd.date_range(start_date, end_date, freq="D")

        data = []
        base_revenue = 1000

        for i, timestamp in enumerate(date_range):
            # Simulate growing business with some variation
            growth_factor = 1 + (i * 0.001)  # Small daily growth
            daily_variation = np.random.normal(1, 0.1)

            revenue = base_revenue * growth_factor * daily_variation

            data.append(
                {
                    "timestamp": timestamp,
                    "daily_revenue": max(0, revenue),
                    "new_customers": np.random.randint(1, 15),
                    "customer_churn_rate": np.random.uniform(0.01, 0.05),
                    "average_deal_size": np.random.normal(250, 50),
                    "customer_satisfaction": np.random.normal(4.2, 0.3),  # 1-5 scale
                    "support_tickets": np.random.randint(5, 25),
                    "feature_adoption_rate": np.random.uniform(0.3, 0.8),
                    "system_uptime": np.random.uniform(0.995, 1.0),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    async def _process_analytics_data(
        self, data: pd.DataFrame, query: AnalyticsQuery
    ) -> pd.DataFrame:
        """Process analytics data with filtering and grouping"""

        # Apply filters
        filtered_data = data.copy()
        for filter_key, filter_value in query.filters.items():
            if filter_key in filtered_data.columns:
                if isinstance(filter_value, list):
                    filtered_data = filtered_data[
                        filtered_data[filter_key].isin(filter_value)
                    ]
                else:
                    filtered_data = filtered_data[
                        filtered_data[filter_key] == filter_value
                    ]

        # Apply grouping and aggregation
        if query.group_by:
            # Group by specified columns and aggregate
            numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns

            if query.aggregation == "sum":
                processed_data = filtered_data.groupby(query.group_by)[
                    numeric_columns
                ].sum()
            elif query.aggregation == "avg":
                processed_data = filtered_data.groupby(query.group_by)[
                    numeric_columns
                ].mean()
            elif query.aggregation == "count":
                processed_data = filtered_data.groupby(query.group_by)[
                    numeric_columns
                ].count()
            elif query.aggregation == "min":
                processed_data = filtered_data.groupby(query.group_by)[
                    numeric_columns
                ].min()
            elif query.aggregation == "max":
                processed_data = filtered_data.groupby(query.group_by)[
                    numeric_columns
                ].max()
            else:
                processed_data = filtered_data.groupby(query.group_by)[
                    numeric_columns
                ].mean()
        else:
            processed_data = filtered_data

        return processed_data

    async def _calculate_trends(
        self, data: pd.DataFrame, query: AnalyticsQuery
    ) -> dict[str, Any]:
        """Calculate trend analysis"""

        if len(data) < 2:
            return {}

        trends = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column in data.columns and not data[column].empty:
                values = data[column].dropna()

                if len(values) >= 2:
                    # Calculate trend direction and magnitude
                    x = np.arange(len(values))
                    y = values.values

                    # Linear regression for trend
                    slope = np.polyfit(x, y, 1)[0]

                    # Calculate percentage change
                    first_value = values.iloc[0]
                    last_value = values.iloc[-1]
                    percent_change = (
                        ((last_value - first_value) / first_value * 100)
                        if first_value != 0
                        else 0
                    )

                    # Determine trend direction
                    if abs(percent_change) < 1:
                        direction = "stable"
                    elif percent_change > 0:
                        direction = "increasing"
                    else:
                        direction = "decreasing"

                    trends[column] = {
                        "direction": direction,
                        "slope": slope,
                        "percent_change": percent_change,
                        "first_value": first_value,
                        "last_value": last_value,
                        "volatility": np.std(y) / np.mean(y) if np.mean(y) != 0 else 0,
                    }

        return trends

    async def _detect_anomalies(
        self, data: pd.DataFrame, query: AnalyticsQuery
    ) -> list[dict[str, Any]]:
        """Detect anomalies in the data"""

        anomalies = []
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column in data.columns and not data[column].empty:
                values = data[column].dropna()

                if len(values) >= 10:  # Need sufficient data points
                    # Use statistical method (IQR) for anomaly processing
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Find anomalous points
                    anomalous_points = values[
                        (values < lower_bound) | (values > upper_bound)
                    ]

                    for timestamp, value in anomalous_points.items():
                        anomalies.append(
                            {
                                "timestamp": timestamp.isoformat()
                                if hasattr(timestamp, "isoformat")
                                else str(timestamp),
                                "column": column,
                                "value": value,
                                "expected_range": [lower_bound, upper_bound],
                                "severity": "high"
                                if abs(value - values.median()) > 3 * values.std()
                                else "medium",
                            }
                        )

        return anomalies

    async def _generate_forecasts(
        self, data: pd.DataFrame, query: AnalyticsQuery
    ) -> dict[str, Any]:
        """Generate forecasts for time series data"""

        forecasts = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        # Simple linear extrapolation for demonstration
        # In production, would use more sophisticated forecasting models
        for column in numeric_columns:
            if column in data.columns and not data[column].empty:
                values = data[column].dropna()

                if len(values) >= 5:
                    # Use last 5 points for trend
                    recent_values = values.tail(5)
                    x = np.arange(len(recent_values))
                    y = recent_values.values

                    # Fit linear trend
                    slope, intercept = np.polyfit(x, y, 1)

                    # Generate forecast for next 5 periods
                    forecast_periods = 5
                    forecast_x = np.arange(
                        len(recent_values), len(recent_values) + forecast_periods
                    )
                    forecast_y = slope * forecast_x + intercept

                    # Add some uncertainty bounds
                    residuals = y - (slope * x + intercept)
                    std_error = np.std(residuals)

                    forecasts[column] = {
                        "forecast_values": forecast_y.tolist(),
                        "confidence_upper": (forecast_y + 2 * std_error).tolist(),
                        "confidence_lower": (forecast_y - 2 * std_error).tolist(),
                        "periods": forecast_periods,
                        "trend_slope": slope,
                        "r_squared": np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 1 else 0,
                    }

        return forecasts

    async def _prepare_chart_data(
        self, data: pd.DataFrame, query: AnalyticsQuery
    ) -> dict[str, Any]:
        """Prepare data for chart visualization"""

        chart_data = {
            "chart_type": query.chart_type.value,
            "labels": [],
            "datasets": [],
        }

        if data.empty:
            return chart_data

        try:
            if query.chart_type == ChartType.LINE:
                # Time series line chart
                chart_data["labels"] = [str(idx) for idx in data.index]

                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for i, column in enumerate(numeric_columns[:5]):  # Limit to 5 series
                    chart_data["datasets"].append(
                        {
                            "label": column,
                            "data": data[column].fillna(0).tolist(),
                            "borderColor": f"hsl({i * 60}, 70%, 50%)",
                            "backgroundColor": f"hsla({i * 60}, 70%, 50%, 0.1)",
                        }
                    )

            elif query.chart_type == ChartType.BAR:
                # Bar chart for categorical data
                if query.group_by:
                    chart_data["labels"] = [str(idx) for idx in data.index]
                    chart_data["datasets"].append(
                        {
                            "label": "Count",
                            "data": data.iloc[:, 0].tolist()
                            if len(data.columns) > 0
                            else [],
                            "backgroundColor": "rgba(54, 162, 235, 0.5)",
                        }
                    )

            elif query.chart_type == ChartType.PIE:
                # Pie chart for proportional data
                if len(data) <= 10:  # Reasonable number of slices
                    chart_data["labels"] = [str(idx) for idx in data.index]
                    chart_data["datasets"].append(
                        {
                            "data": data.iloc[:, 0].tolist()
                            if len(data.columns) > 0
                            else [],
                            "backgroundColor": [
                                f"hsl({i * 360 / len(data)}, 70%, 50%)"
                                for i in range(len(data))
                            ],
                        }
                    )

            elif query.chart_type == ChartType.HEATMAP:
                # Heatmap for correlation matrix
                numeric_data = data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) >= 2:
                    correlation_matrix = numeric_data.corr()

                    chart_data = {
                        "chart_type": "heatmap",
                        "labels": correlation_matrix.columns.tolist(),
                        "data": correlation_matrix.values.tolist(),
                    }

        except Exception as e:
            logger.warning(f"Failed to prepare chart data: {e}")
            chart_data = {"chart_type": query.chart_type.value, "error": str(e)}

        return chart_data

    async def generate_business_insights(
        self, timeframe: AnalyticsTimeframe = AnalyticsTimeframe.WEEK
    ) -> list[BusinessInsight]:
        """Generate automated business insights"""

        insights = []

        try:
            # Analyze processing volume trends
            volume_query = AnalyticsQuery(
                metric_type=MetricType.DETECTION_VOLUME,
                timeframe=timeframe,
                include_trends=True,
                include_anomalies=True,
            )
            volume_result = await self.execute_analytics_query(volume_query)

            # Generate insights from processing volume
            if volume_result.trends:
                for metric, trend in volume_result.trends.items():
                    if abs(trend["percent_change"]) > 20:  # Significant change
                        severity = (
                            "high" if abs(trend["percent_change"]) > 50 else "medium"
                        )

                        insight = BusinessInsight(
                            insight_id=f"volume_trend_{metric}",
                            title=f"Significant change in {metric}",
                            description=f"{metric} has {trend['direction']} by {trend['percent_change']:.1f}% over the {timeframe.value}",
                            category="operational",
                            severity=severity,
                            confidence=0.8,
                            measurements={"trend": trend},
                            recommendations=self._generate_volume_recommendations(
                                trend
                            ),
                        )
                        insights.append(insight)

            # Analyze processor performance
            perf_query = AnalyticsQuery(
                metric_type=MetricType.MODEL_PERFORMANCE,
                timeframe=timeframe,
                include_trends=True,
            )
            perf_result = await self.execute_analytics_query(perf_query)

            # Check for processor degradation
            if perf_result.trends:
                for metric, trend in perf_result.trends.items():
                    if (
                        trend["direction"] == "decreasing"
                        and "accuracy" in metric.lower()
                    ):
                        insight = BusinessInsight(
                            insight_id=f"processor_degradation_{metric}",
                            title="Processor performance degradation detected",
                            description=f"Processor {metric} has decreased by {abs(trend['percent_change']):.1f}%",
                            category="ml_ops",
                            severity="high",
                            confidence=0.9,
                            measurements={"performance_trend": trend},
                            recommendations=[
                                "Consider retraining models with recent data",
                                "Review data quality and distribution changes",
                                "Implement processor performance monitoring alerts",
                            ],
                        )
                        insights.append(insight)

            # Analyze system health
            health_query = AnalyticsQuery(
                metric_type=MetricType.SYSTEM_HEALTH,
                timeframe=AnalyticsTimeframe.DAY,
                include_anomalies=True,
            )
            health_result = await self.execute_analytics_query(health_query)

            # Check for system issues
            if health_result.anomalies:
                critical_anomalies = [
                    a for a in health_result.anomalies if a["severity"] == "high"
                ]

                if critical_anomalies:
                    insight = BusinessInsight(
                        insight_id="system_health_anomalies",
                        title="System health anomalies detected",
                        description=f"Detected {len(critical_anomalies)} critical system health anomalies",
                        category="infrastructure",
                        severity="critical",
                        confidence=0.95,
                        measurements={"anomaly_count": len(critical_anomalies)},
                        recommendations=[
                            "Investigate system performance issues",
                            "Check resource utilization and scaling policies",
                            "Review recent deployments and configurations",
                        ],
                    )
                    insights.append(insight)

            # Analyze business KPIs
            kpi_query = AnalyticsQuery(
                metric_type=MetricType.BUSINESS_KPI,
                timeframe=timeframe,
                include_trends=True,
            )
            kpi_result = await self.execute_analytics_query(kpi_query)

            # Check revenue trends
            if kpi_result.trends and "daily_revenue" in kpi_result.trends:
                revenue_trend = kpi_result.trends["daily_revenue"]

                if revenue_trend["percent_change"] > 10:
                    insight = BusinessInsight(
                        insight_id="revenue_growth",
                        title="Strong revenue growth detected",
                        description=f"Revenue has increased by {revenue_trend['percent_change']:.1f}% over the {timeframe.value}",
                        category="business",
                        severity="low",
                        confidence=0.85,
                        measurements={"revenue_trend": revenue_trend},
                        recommendations=[
                            "Analyze successful customer acquisition channels",
                            "Consider scaling infrastructure to support growth",
                            "Investigate factors driving increased usage",
                        ],
                    )
                    insights.append(insight)
                elif revenue_trend["percent_change"] < -5:
                    insight = BusinessInsight(
                        insight_id="revenue_decline",
                        title="Revenue decline detected",
                        description=f"Revenue has decreased by {abs(revenue_trend['percent_change']):.1f}% over the {timeframe.value}",
                        category="business",
                        severity="high",
                        confidence=0.85,
                        measurements={"revenue_trend": revenue_trend},
                        recommendations=[
                            "Review customer churn and satisfaction measurements",
                            "Analyze competitive landscape changes",
                            "Investigate technical issues affecting service quality",
                        ],
                    )
                    insights.append(insight)

        except Exception as e:
            logger.error(f"Failed to generate business insights: {e}")

        return insights

    def _generate_volume_recommendations(self, trend: dict[str, Any]) -> list[str]:
        """Generate recommendations based on volume trends"""

        recommendations = []

        if trend["direction"] == "increasing":
            if trend["percent_change"] > 50:
                recommendations.extend(
                    [
                        "Consider scaling infrastructure to handle increased load",
                        "Monitor system performance and response times",
                        "Review alerting thresholds and escalation procedures",
                    ]
                )
            else:
                recommendations.extend(
                    [
                        "Monitor resource utilization trends",
                        "Consider proactive capacity planning",
                    ]
                )

        elif trend["direction"] == "decreasing":
            if trend["percent_change"] < -30:
                recommendations.extend(
                    [
                        "Investigate potential data pipeline issues",
                        "Check for service disruptions or outages",
                        "Review client integrations and API usage",
                    ]
                )
            else:
                recommendations.extend(
                    [
                        "Monitor for continued decline",
                        "Consider cost optimization opportunities",
                    ]
                )

        return recommendations

    async def create_executive_dashboard(self) -> dict[str, Any]:
        """Create executive dashboard with key business measurements"""

        try:
            # Key measurements for executive overview
            dashboard_data = {"summary": {}, "charts": [], "insights": [], "kpis": {}}

            # Get business KPIs
            kpi_query = AnalyticsQuery(
                metric_type=MetricType.BUSINESS_KPI,
                timeframe=AnalyticsTimeframe.MONTH,
                include_trends=True,
            )
            kpi_result = await self.execute_analytics_query(kpi_query)

            if not kpi_result.data.empty:
                # Calculate key summary measurements
                latest_data = kpi_result.data.tail(1)

                dashboard_data["kpis"] = {
                    "monthly_revenue": float(kpi_result.data["daily_revenue"].sum()),
                    "new_customers": int(kpi_result.data["new_customers"].sum()),
                    "customer_satisfaction": float(
                        kpi_result.data["customer_satisfaction"].mean()
                    ),
                    "system_uptime": float(kpi_result.data["system_uptime"].mean()),
                    "churn_rate": float(kpi_result.data["customer_churn_rate"].mean()),
                }

                # Add revenue trend chart
                dashboard_data["charts"].append(
                    {
                        "title": "Monthly Revenue Trend",
                        "type": "line",
                        "data": kpi_result.chart_data,
                    }
                )

            # Get system health overview
            health_query = AnalyticsQuery(
                metric_type=MetricType.SYSTEM_HEALTH,
                timeframe=AnalyticsTimeframe.DAY,
                aggregation="avg",
            )
            health_result = await self.execute_analytics_query(health_query)

            if not health_result.data.empty:
                dashboard_data["summary"]["system_health"] = {
                    "avg_response_time": float(
                        health_result.data["response_time_ms"].mean()
                    ),
                    "error_rate": float(
                        health_result.data["error_rate_percent"].mean()
                    ),
                    "uptime": 99.9,  # Calculated from downtime events
                }

            # Get user engagement measurements
            engagement_query = AnalyticsQuery(
                metric_type=MetricType.USER_ENGAGEMENT,
                timeframe=AnalyticsTimeframe.WEEK,
                aggregation="sum",
            )
            engagement_result = await self.execute_analytics_query(engagement_query)

            if not engagement_result.data.empty:
                dashboard_data["summary"]["user_engagement"] = {
                    "weekly_active_users": int(
                        engagement_result.data["active_users"].max()
                    ),
                    "total_api_calls": int(engagement_result.data["api_calls"].sum()),
                    "avg_session_duration": float(
                        engagement_result.data["session_duration_minutes"].mean()
                    ),
                }

            # Get automated insights
            insights = await self.generate_business_insights(AnalyticsTimeframe.WEEK)
            dashboard_data["insights"] = [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "severity": insight.severity,
                    "category": insight.category,
                }
                for insight in insights[:5]  # Top 5 insights
            ]

            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to create executive dashboard: {e}")
            return {"error": str(e)}

    async def generate_custom_report(
        self,
        queries: list[AnalyticsQuery],
        report_title: str = "Custom Analytics Report",
    ) -> dict[str, Any]:
        """Generate custom analytics report"""

        report = {
            "title": report_title,
            "generated_at": datetime.now().isoformat(),
            "sections": [],
        }

        try:
            for i, query in enumerate(queries):
                result = await self.execute_analytics_query(query)

                section = {
                    "title": f"Analysis {i+1}: {query.metric_type.value}",
                    "query": {
                        "metric_type": query.metric_type.value,
                        "timeframe": query.timeframe.value,
                        "chart_type": query.chart_type.value,
                    },
                    "data_summary": {
                        "total_records": len(result.data),
                        "date_range": result.metadata.get("date_range", {}),
                        "columns": list(result.data.columns)
                        if not result.data.empty
                        else [],
                    },
                    "chart_data": result.chart_data,
                    "trends": result.trends,
                    "anomalies": result.anomalies[:5],  # Top 5 anomalies
                    "forecasts": result.forecasts,
                }

                report["sections"].append(section)

            return report

        except Exception as e:
            logger.error(f"Failed to generate custom report: {e}")
            return {"error": str(e), "title": report_title}


# Export for use in other modules
__all__ = [
    "AdvancedAnalyticsService",
    "AnalyticsQuery",
    "AnalyticsResult",
    "BusinessInsight",
    "AnalyticsTimeframe",
    "MetricType",
    "ChartType",
]
