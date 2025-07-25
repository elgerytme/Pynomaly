"""Comprehensive analytics engine for business intelligence and data insights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict
import json
import math
from concurrent.futures import ThreadPoolExecutor
import threading

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ....infrastructure.logging import get_logger

logger = get_logger(__name__)

# Lazy import metrics collector to avoid None issues
def get_safe_metrics_collector():
    """Get metrics collector with safe fallback."""
    try:
        from ....infrastructure.monitoring import get_metrics_collector
        return get_metrics_collector()
    except Exception:
        class MockMetricsCollector:
            def record_metric(self, *args, **kwargs):
                pass
        return MockMetricsCollector()


class MetricType(Enum):
    """Types of metrics for analytics."""
    DETECTION_METRICS = "detection_metrics"
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_METRICS = "system_metrics"
    BUSINESS_METRICS = "business_metrics"
    SECURITY_METRICS = "security_metrics"


class ChartType(Enum):
    """Chart types for visualization."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    PIE = "pie"
    BOX = "box"
    VIOLIN = "violin"
    CANDLESTICK = "candlestick"
    TREEMAP = "treemap"


class AggregationType(Enum):
    """Data aggregation types."""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    PERCENTILE = "percentile"


@dataclass
class AnalyticsQuery:
    """Analytics query specification."""
    metric_type: MetricType
    time_range: Tuple[datetime, datetime]
    filters: Dict[str, Any] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    aggregation: AggregationType = AggregationType.COUNT
    limit: Optional[int] = None
    sort_by: Optional[str] = None
    sort_desc: bool = True


@dataclass
class AnalyticsResult:
    """Analytics query result."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    query: AnalyticsQuery
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "data": self.data.to_dict(orient="records"),
            "metadata": self.metadata,
            "query": {
                "metric_type": self.query.metric_type.value,
                "time_range": [
                    self.query.time_range[0].isoformat(),
                    self.query.time_range[1].isoformat()
                ],
                "filters": self.query.filters,
                "group_by": self.query.group_by,
                "aggregation": self.query.aggregation.value
            },
            "generated_at": self.generated_at.isoformat()
        }


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    chart_type: ChartType
    query: AnalyticsQuery
    refresh_interval: int = 300  # seconds
    width: int = 6  # 1-12 grid columns
    height: int = 300  # pixels
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    is_public: bool = False


class DataProcessor:
    """Advanced data processing for analytics."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.lock = threading.Lock()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        # Remove duplicates
        df_clean = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns
        
        # Fill numeric NaNs with median
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill categorical NaNs with mode
        for col in categorical_columns:
            mode_value = df_clean[col].mode()
            if len(mode_value) > 0:
                df_clean[col] = df_clean[col].fillna(mode_value[0])
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_outliers = df.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        df_outliers["is_outlier"] = outlier_mask
        return df_outliers
    
    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform clustering analysis."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            return df, {"error": "No numeric columns for clustering"}
        
        # Prepare data
        X = df[numeric_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            silhouette_scores = []
            K_range = range(2, min(11, len(df) // 2))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            n_clusters = K_range[np.argmax(silhouette_scores)]
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered["cluster"] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = df_clustered[df_clustered["cluster"] == i][numeric_columns]
            cluster_stats[f"cluster_{i}"] = {
                "size": len(cluster_data),
                "centroid": cluster_data.mean().to_dict(),
                "std": cluster_data.std().to_dict()
            }
        
        metadata = {
            "n_clusters": n_clusters,
            "cluster_stats": cluster_stats,
            "silhouette_score": silhouette_score(X_scaled, cluster_labels),
            "inertia": kmeans.inertia_
        }
        
        return df_clustered, metadata
    
    def time_series_analysis(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, Any]:
        """Perform time series analysis."""
        if time_col not in df.columns or value_col not in df.columns:
            return {"error": "Required columns not found"}
        
        # Ensure datetime index
        df_ts = df.copy()
        df_ts[time_col] = pd.to_datetime(df_ts[time_col])
        df_ts = df_ts.set_index(time_col).sort_index()
        
        # Basic statistics
        ts_data = df_ts[value_col]
        stats = {
            "mean": ts_data.mean(),
            "std": ts_data.std(),
            "min": ts_data.min(),
            "max": ts_data.max(),
            "trend": self._calculate_trend(ts_data),
            "seasonality": self._detect_seasonality(ts_data),
            "volatility": ts_data.std() / ts_data.mean() if ts_data.mean() != 0 else 0
        }
        
        # Moving averages
        stats["ma_7d"] = ts_data.rolling(window=7).mean().iloc[-1] if len(ts_data) >= 7 else None
        stats["ma_30d"] = ts_data.rolling(window=30).mean().iloc[-1] if len(ts_data) >= 30 else None
        
        return stats
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction."""
        if len(series) < 2:
            return "insufficient_data"
        
        x = np.arange(len(series))
        y = series.values
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _detect_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Detect seasonality patterns."""
        if len(series) < 14:
            return {"detected": False, "reason": "insufficient_data"}
        
        # Simple autocorrelation check
        try:
            autocorr_7d = series.autocorr(lag=7) if len(series) >= 14 else 0
            autocorr_24h = series.autocorr(lag=24) if len(series) >= 48 else 0
            
            return {
                "detected": max(abs(autocorr_7d), abs(autocorr_24h)) > 0.3,
                "weekly_pattern": abs(autocorr_7d) > 0.3,
                "daily_pattern": abs(autocorr_24h) > 0.3,
                "weekly_correlation": autocorr_7d,
                "daily_correlation": autocorr_24h
            }
        except Exception:
            return {"detected": False, "reason": "calculation_error"}


class ChartGenerator:
    """Generate interactive charts using Plotly."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_chart(self, data: pd.DataFrame, chart_type: ChartType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create chart based on type and configuration."""
        try:
            if chart_type == ChartType.LINE:
                return self._create_line_chart(data, config)
            elif chart_type == ChartType.BAR:
                return self._create_bar_chart(data, config)
            elif chart_type == ChartType.SCATTER:
                return self._create_scatter_chart(data, config)
            elif chart_type == ChartType.HISTOGRAM:
                return self._create_histogram(data, config)
            elif chart_type == ChartType.HEATMAP:
                return self._create_heatmap(data, config)
            elif chart_type == ChartType.PIE:
                return self._create_pie_chart(data, config)
            elif chart_type == ChartType.BOX:
                return self._create_box_plot(data, config)
            elif chart_type == ChartType.TREEMAP:
                return self._create_treemap(data, config)
            else:
                return {"error": f"Unsupported chart type: {chart_type}"}
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return {"error": f"Chart generation failed: {str(e)}"}
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        color_col = config.get("color_column")
        
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            return {"error": "Missing required columns"}
        
        fig = px.line(
            data,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col in data.columns else None,
            title=config.get("title", "Line Chart")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "line"}
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            return {"error": "Missing required columns"}
        
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=config.get("title", "Bar Chart"),
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "bar"}
    
    def _create_scatter_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create scatter plot."""
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        size_col = config.get("size_column")
        color_col = config.get("color_column")
        
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            return {"error": "Missing required columns"}
        
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            size=size_col if size_col and size_col in data.columns else None,
            color=color_col if color_col and color_col in data.columns else None,
            title=config.get("title", "Scatter Plot")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "scatter"}
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create histogram."""
        x_col = config.get("x_column")
        
        if not x_col or x_col not in data.columns:
            return {"error": "Missing required column"}
        
        fig = px.histogram(
            data,
            x=x_col,
            nbins=config.get("bins", 30),
            title=config.get("title", "Histogram")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "histogram"}
    
    def _create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create heatmap."""
        # Assume data is already in matrix format or create correlation matrix
        if "correlation" in config and config["correlation"]:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return {"error": "Insufficient numeric columns for correlation"}
            
            corr_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title=config.get("title", "Correlation Heatmap"),
                aspect="auto"
            )
        else:
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            z_col = config.get("z_column")
            
            if not all([x_col, y_col, z_col]) or not all([col in data.columns for col in [x_col, y_col, z_col]]):
                return {"error": "Missing required columns for heatmap"}
            
            pivot_data = data.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
            
            fig = px.imshow(
                pivot_data,
                title=config.get("title", "Heatmap"),
                aspect="auto"
            )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "heatmap"}
    
    def _create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pie chart."""
        values_col = config.get("values_column")
        names_col = config.get("names_column")
        
        if not values_col or not names_col or values_col not in data.columns or names_col not in data.columns:
            return {"error": "Missing required columns"}
        
        fig = px.pie(
            data,
            values=values_col,
            names=names_col,
            title=config.get("title", "Pie Chart")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "pie"}
    
    def _create_box_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create box plot."""
        y_col = config.get("y_column")
        x_col = config.get("x_column")  # Optional grouping
        
        if not y_col or y_col not in data.columns:
            return {"error": "Missing required column"}
        
        fig = px.box(
            data,
            y=y_col,
            x=x_col if x_col and x_col in data.columns else None,
            title=config.get("title", "Box Plot")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "box"}
    
    def _create_treemap(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create treemap."""
        path_cols = config.get("path_columns", [])
        values_col = config.get("values_column")
        
        if not path_cols or not values_col or values_col not in data.columns:
            return {"error": "Missing required columns"}
        
        # Ensure all path columns exist
        valid_path_cols = [col for col in path_cols if col in data.columns]
        if not valid_path_cols:
            return {"error": "No valid path columns found"}
        
        fig = px.treemap(
            data,
            path=valid_path_cols,
            values=values_col,
            title=config.get("title", "Treemap")
        )
        
        fig.update_layout(
            template="plotly_white",
            height=config.get("height", 400)
        )
        
        return {"chart": fig.to_json(), "type": "treemap"}


class AnalyticsEngine:
    """Main analytics engine for business intelligence."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.chart_generator = ChartGenerator()
        self.dashboards: Dict[str, Dashboard] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics_collector = get_safe_metrics_collector()
        
        # Sample data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_lock = threading.Lock()
        
        logger.info("Analytics engine initialized")
    
    async def execute_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute analytics query."""
        try:
            # Get data based on metric type
            data = await self._get_data_for_query(query)
            
            if data.empty:
                return AnalyticsResult(
                    data=data,
                    metadata={"warning": "No data found for query"},
                    query=query
                )
            
            # Apply filters
            filtered_data = self._apply_filters(data, query.filters)
            
            # Apply grouping and aggregation
            aggregated_data = self._apply_aggregation(filtered_data, query)
            
            # Apply sorting and limiting
            final_data = self._apply_sorting_and_limiting(aggregated_data, query)
            
            # Generate metadata
            metadata = {
                "total_rows": len(data),
                "filtered_rows": len(filtered_data),
                "final_rows": len(final_data),
                "columns": final_data.columns.tolist(),
                "data_types": final_data.dtypes.to_dict()
            }
            
            # Record metrics
            self.metrics_collector.record_metric(
                "analytics.query.executed",
                1,
                {
                    "metric_type": query.metric_type.value,
                    "rows_processed": len(data)
                }
            )
            
            return AnalyticsResult(
                data=final_data,
                metadata=metadata,
                query=query
            )
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            
            self.metrics_collector.record_metric(
                "analytics.query.failed",
                1,
                {"metric_type": query.metric_type.value}
            )
            
            return AnalyticsResult(
                data=pd.DataFrame(),
                metadata={"error": str(e)},
                query=query
            )
    
    async def _get_data_for_query(self, query: AnalyticsQuery) -> pd.DataFrame:
        """Get data based on query metric type."""
        # In a real implementation, this would connect to databases
        # For now, generate sample data based on metric type
        
        if query.metric_type == MetricType.DETECTION_METRICS:
            return self._generate_detection_data(query.time_range)
        elif query.metric_type == MetricType.PERFORMANCE_METRICS:
            return self._generate_performance_data(query.time_range)
        elif query.metric_type == MetricType.SYSTEM_METRICS:
            return self._generate_system_data(query.time_range)
        elif query.metric_type == MetricType.BUSINESS_METRICS:
            return self._generate_business_data(query.time_range)
        elif query.metric_type == MetricType.SECURITY_METRICS:
            return self._generate_security_data(query.time_range)
        else:
            return pd.DataFrame()
    
    def _generate_detection_data(self, time_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Generate sample detection metrics data."""
        start, end = time_range
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = pd.date_range(start, end, periods=min(hours, 1000))
        
        data = []
        algorithms = ["isolation_forest", "local_outlier_factor", "one_class_svm", "elliptic_envelope"]
        
        for ts in timestamps:
            for algo in algorithms:
                data.append({
                    "timestamp": ts,
                    "algorithm": algo,
                    "anomalies_detected": np.random.poisson(5),
                    "processing_time": np.random.gamma(2, 0.5),
                    "accuracy": np.random.beta(8, 2),
                    "precision": np.random.beta(7, 3),
                    "recall": np.random.beta(6, 4),
                    "f1_score": np.random.beta(7, 3),
                    "data_points_processed": np.random.randint(100, 10000)
                })
        
        return pd.DataFrame(data)
    
    def _generate_performance_data(self, time_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Generate sample performance metrics data."""
        start, end = time_range
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = pd.date_range(start, end, periods=min(hours, 1000))
        
        data = []
        endpoints = ["/api/v1/detection/detect", "/api/v1/health", "/api/v1/models"]
        
        for ts in timestamps:
            for endpoint in endpoints:
                base_latency = {"detect": 500, "health": 50, "models": 100}.get(
                    endpoint.split("/")[-1], 200
                )
                
                data.append({
                    "timestamp": ts,
                    "endpoint": endpoint,
                    "response_time": np.random.gamma(base_latency/100, 100),
                    "requests_per_second": np.random.poisson(10),
                    "error_rate": np.random.beta(1, 19),  # Low error rate
                    "cpu_usage": np.random.beta(3, 7),
                    "memory_usage": np.random.beta(4, 6),
                    "throughput": np.random.gamma(50, 0.5)
                })
        
        return pd.DataFrame(data)
    
    def _generate_system_data(self, time_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Generate sample system metrics data."""
        start, end = time_range
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = pd.date_range(start, end, periods=min(hours, 1000))
        
        data = []
        services = ["anomaly_detection", "api_gateway", "database", "redis"]
        
        for ts in timestamps:
            for service in services:
                data.append({
                    "timestamp": ts,
                    "service": service,
                    "cpu_usage": np.random.beta(3, 7),
                    "memory_usage": np.random.beta(4, 6),
                    "disk_usage": np.random.beta(2, 8),
                    "network_io": np.random.gamma(100, 10),
                    "connections": np.random.poisson(50),
                    "uptime": np.random.uniform(0.95, 1.0),
                    "error_count": np.random.poisson(2),
                    "restart_count": np.random.poisson(0.1)
                })
        
        return pd.DataFrame(data)
    
    def _generate_business_data(self, time_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Generate sample business metrics data."""
        start, end = time_range
        days = int((end - start).days) + 1
        
        timestamps = pd.date_range(start, end, periods=min(days, 365))
        
        data = []
        products = ["anomaly_detection_api", "security_monitoring", "data_analytics"]
        regions = ["us-east", "us-west", "eu-central", "asia-pacific"]
        
        for ts in timestamps:
            for product in products:
                for region in regions:
                    data.append({
                        "timestamp": ts,
                        "product": product,
                        "region": region,
                        "revenue": np.random.gamma(1000, 5),
                        "users": np.random.poisson(100),
                        "api_calls": np.random.poisson(10000),
                        "cost": np.random.gamma(500, 2),
                        "customer_satisfaction": np.random.beta(8, 2),
                        "churn_rate": np.random.beta(1, 19),
                        "conversion_rate": np.random.beta(2, 8)
                    })
        
        return pd.DataFrame(data)
    
    def _generate_security_data(self, time_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Generate sample security metrics data."""
        start, end = time_range
        hours = int((end - start).total_seconds() / 3600)
        
        timestamps = pd.date_range(start, end, periods=min(hours, 1000))
        
        data = []
        threat_types = ["brute_force", "sql_injection", "xss", "dos_attack", "unauthorized_access"]
        
        for ts in timestamps:
            for threat_type in threat_types:
                data.append({
                    "timestamp": ts,
                    "threat_type": threat_type,
                    "incidents_detected": np.random.poisson(2),
                    "blocked_attempts": np.random.poisson(10),
                    "false_positives": np.random.poisson(1),
                    "response_time": np.random.gamma(10, 5),
                    "severity": np.random.choice(["low", "medium", "high", "critical"], 
                                                p=[0.4, 0.3, 0.2, 0.1]),
                    "source_ip_count": np.random.poisson(5),
                    "mitigation_success": np.random.beta(9, 1)
                })
        
        return pd.DataFrame(data)
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to data."""
        filtered_data = data.copy()
        
        for column, filter_value in filters.items():
            if column not in data.columns:
                continue
            
            if isinstance(filter_value, dict):
                # Range filter
                if "min" in filter_value:
                    filtered_data = filtered_data[filtered_data[column] >= filter_value["min"]]
                if "max" in filter_value:
                    filtered_data = filtered_data[filtered_data[column] <= filter_value["max"]]
            elif isinstance(filter_value, list):
                # Include filter
                filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
            else:
                # Equal filter
                filtered_data = filtered_data[filtered_data[column] == filter_value]
        
        return filtered_data
    
    def _apply_aggregation(self, data: pd.DataFrame, query: AnalyticsQuery) -> pd.DataFrame:
        """Apply grouping and aggregation."""
        if not query.group_by:
            return data
        
        # Check if all group_by columns exist
        valid_group_cols = [col for col in query.group_by if col in data.columns]
        if not valid_group_cols:
            return data
        
        # Get numeric columns for aggregation
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            # If no numeric columns, just return grouped data
            return data.groupby(valid_group_cols).size().reset_index(name="count")
        
        # Apply aggregation
        if query.aggregation == AggregationType.SUM:
            return data.groupby(valid_group_cols)[numeric_cols].sum().reset_index()
        elif query.aggregation == AggregationType.MEAN:
            return data.groupby(valid_group_cols)[numeric_cols].mean().reset_index()
        elif query.aggregation == AggregationType.MEDIAN:
            return data.groupby(valid_group_cols)[numeric_cols].median().reset_index()
        elif query.aggregation == AggregationType.MIN:
            return data.groupby(valid_group_cols)[numeric_cols].min().reset_index()
        elif query.aggregation == AggregationType.MAX:
            return data.groupby(valid_group_cols)[numeric_cols].max().reset_index()
        elif query.aggregation == AggregationType.STD:
            return data.groupby(valid_group_cols)[numeric_cols].std().reset_index()
        else:  # COUNT
            return data.groupby(valid_group_cols).size().reset_index(name="count")
    
    def _apply_sorting_and_limiting(self, data: pd.DataFrame, query: AnalyticsQuery) -> pd.DataFrame:
        """Apply sorting and row limiting."""
        result_data = data.copy()
        
        # Apply sorting
        if query.sort_by and query.sort_by in data.columns:
            result_data = result_data.sort_values(
                by=query.sort_by,
                ascending=not query.sort_desc
            )
        
        # Apply limit
        if query.limit and query.limit > 0:
            result_data = result_data.head(query.limit)
        
        return result_data
    
    async def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create a new dashboard."""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            
            self.metrics_collector.record_metric(
                "analytics.dashboard.created",
                1,
                {"dashboard_id": dashboard.dashboard_id}
            )
            
            logger.info(f"Dashboard created: {dashboard.dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard creation error: {e}")
            return False
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self.dashboards.get(dashboard_id)
    
    async def render_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Render dashboard with current data."""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        try:
            rendered_widgets = []
            
            # Process widgets concurrently
            tasks = []
            for widget in dashboard.widgets:
                task = self._render_widget(widget)
                tasks.append(task)
            
            widget_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(widget_results):
                if isinstance(result, Exception):
                    logger.error(f"Widget rendering error: {result}")
                    rendered_widgets.append({
                        "widget_id": dashboard.widgets[i].widget_id,
                        "error": str(result)
                    })
                else:
                    rendered_widgets.append(result)
            
            return {
                "dashboard_id": dashboard_id,
                "title": dashboard.title,
                "description": dashboard.description,
                "widgets": rendered_widgets,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard rendering error: {e}")
            return {"error": str(e)}
    
    async def _render_widget(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Render individual widget."""
        try:
            # Execute query
            result = await self.execute_query(widget.query)
            
            if result.data.empty:
                return {
                    "widget_id": widget.widget_id,
                    "title": widget.title,
                    "warning": "No data available"
                }
            
            # Generate chart
            chart_result = self.chart_generator.create_chart(
                result.data,
                widget.chart_type,
                widget.config
            )
            
            return {
                "widget_id": widget.widget_id,
                "title": widget.title,
                "chart_type": widget.chart_type.value,
                "chart": chart_result.get("chart"),
                "width": widget.width,
                "height": widget.height,
                "data_summary": {
                    "rows": len(result.data),
                    "columns": len(result.data.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Widget rendering error: {e}")
            return {
                "widget_id": widget.widget_id,
                "title": widget.title,
                "error": str(e)
            }
    
    async def get_insights(self, metric_type: MetricType, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate automated insights for metrics."""
        try:
            # Get data
            query = AnalyticsQuery(
                metric_type=metric_type,
                time_range=time_range
            )
            
            result = await self.execute_query(query)
            
            if result.data.empty:
                return {"insights": [], "message": "No data available for insights"}
            
            # Generate insights
            insights = []
            
            # Time series insights
            if "timestamp" in result.data.columns:
                ts_insights = self._generate_time_series_insights(result.data)
                insights.extend(ts_insights)
            
            # Anomaly insights
            anomaly_insights = self._generate_anomaly_insights(result.data)
            insights.extend(anomaly_insights)
            
            # Correlation insights
            correlation_insights = self._generate_correlation_insights(result.data)
            insights.extend(correlation_insights)
            
            return {
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat(),
                "data_summary": result.metadata
            }
            
        except Exception as e:
            logger.error(f"Insights generation error: {e}")
            return {"error": str(e)}
    
    def _generate_time_series_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate time series insights."""
        insights = []
        
        if "timestamp" not in data.columns:
            return insights
        
        # Find numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            try:
                ts_analysis = self.data_processor.time_series_analysis(data, "timestamp", col)
                
                if "error" not in ts_analysis:
                    insight = {
                        "type": "trend",
                        "metric": col,
                        "trend": ts_analysis.get("trend", "unknown"),
                        "volatility": ts_analysis.get("volatility", 0),
                        "confidence": "medium"
                    }
                    
                    # Add description based on trend
                    if ts_analysis.get("trend") == "increasing":
                        insight["description"] = f"{col} shows an increasing trend over time"
                    elif ts_analysis.get("trend") == "decreasing":
                        insight["description"] = f"{col} shows a decreasing trend over time"
                    else:
                        insight["description"] = f"{col} remains relatively stable over time"
                    
                    insights.append(insight)
                    
            except Exception as e:
                logger.warning(f"Time series insight error for {col}: {e}")
        
        return insights
    
    def _generate_anomaly_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate anomaly insights."""
        insights = []
        
        try:
            # Detect outliers
            outlier_data = self.data_processor.detect_outliers(data)
            
            if "is_outlier" in outlier_data.columns:
                outlier_count = outlier_data["is_outlier"].sum()
                total_count = len(outlier_data)
                outlier_rate = outlier_count / total_count if total_count > 0 else 0
                
                if outlier_rate > 0.05:  # More than 5% outliers
                    insights.append({
                        "type": "anomaly",
                        "description": f"High outlier rate detected: {outlier_rate:.1%} of data points",
                        "outlier_count": int(outlier_count),
                        "total_count": int(total_count),
                        "confidence": "high" if outlier_rate > 0.1 else "medium"
                    })
                    
        except Exception as e:
            logger.warning(f"Anomaly insight error: {e}")
        
        return insights
    
    def _generate_correlation_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate correlation insights."""
        insights = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return insights
            
            # Calculate correlations
            corr_matrix = data[numeric_cols].corr()
            
            # Find strong correlations (excluding self-correlations)
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": corr_value
                        })
            
            # Generate insights for strong correlations
            for corr in strong_correlations[:3]:  # Limit to top 3
                correlation_type = "positive" if corr["correlation"] > 0 else "negative"
                strength = "very strong" if abs(corr["correlation"]) > 0.9 else "strong"
                
                insights.append({
                    "type": "correlation",
                    "description": f"{strength.title()} {correlation_type} correlation between {corr['var1']} and {corr['var2']}",
                    "correlation_value": corr["correlation"],
                    "variables": [corr["var1"], corr["var2"]],
                    "confidence": "high"
                })
                
        except Exception as e:
            logger.warning(f"Correlation insight error: {e}")
        
        return insights


# Global analytics engine instance
_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get the global analytics engine instance."""
    global _analytics_engine
    
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    
    return _analytics_engine


def initialize_analytics_engine() -> AnalyticsEngine:
    """Initialize the global analytics engine."""
    global _analytics_engine
    _analytics_engine = AnalyticsEngine()
    return _analytics_engine