"""Tests for analytics engine and BI dashboard functionality."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.anomaly_detection.application.services.intelligence.analytics_engine import (
    AnalyticsEngine,
    AnalyticsQuery,
    MetricType,
    AggregationType,
    ChartType,
    Dashboard,
    DashboardWidget,
    DataProcessor,
    ChartGenerator,
    get_analytics_engine,
    initialize_analytics_engine
)
from src.anomaly_detection.application.services.intelligence.dashboard_service import (
    DashboardService,
    DashboardTemplateFactory,
    get_dashboard_service,
    initialize_dashboard_service
)


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_clean_data_basic(self):
        """Test basic data cleaning."""
        processor = DataProcessor()
        
        # Create test data with duplicates and missing values
        data = {
            'A': [1, 2, 2, 4, np.nan],
            'B': ['x', 'y', 'y', 'z', None],
            'C': [1.1, 2.2, 2.2, 4.4, 5.5]
        }
        df = pd.DataFrame(data)
        
        cleaned_df = processor.clean_data(df)
        
        # Check duplicates removed
        assert len(cleaned_df) == 4  # One duplicate removed
        
        # Check no missing values
        assert not cleaned_df.isnull().any().any()
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        processor = DataProcessor()
        
        # Create data with clear outliers
        data = {
            'value': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]  # 100 is an outlier
        }
        df = pd.DataFrame(data)
        
        outlier_df = processor.detect_outliers(df, ['value'])
        
        assert 'is_outlier' in outlier_df.columns
        assert outlier_df['is_outlier'].sum() > 0  # At least one outlier detected
    
    def test_perform_clustering(self):
        """Test clustering analysis."""
        processor = DataProcessor()
        
        # Create clusterable data
        np.random.seed(42)
        data = {
            'x': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
            'y': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])
        }
        df = pd.DataFrame(data)
        
        clustered_df, metadata = processor.perform_clustering(df, n_clusters=2)
        
        assert 'cluster' in clustered_df.columns
        assert len(clustered_df['cluster'].unique()) == 2
        assert 'n_clusters' in metadata
        assert 'silhouette_score' in metadata
        assert metadata['n_clusters'] == 2
    
    def test_time_series_analysis(self):
        """Test time series analysis."""
        processor = DataProcessor()
        
        # Create time series data with trend
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = np.arange(100) + np.random.normal(0, 0.1, 100)  # Upward trend with noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        
        stats = processor.time_series_analysis(df, 'timestamp', 'value')
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'trend' in stats
        assert stats['trend'] == 'increasing'  # Should detect upward trend


class TestChartGenerator:
    """Test chart generation functionality."""
    
    def test_create_line_chart(self):
        """Test line chart creation."""
        generator = ChartGenerator()
        
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        config = {
            'x_column': 'x',
            'y_column': 'y',
            'title': 'Test Line Chart'
        }
        
        result = generator.create_chart(data, ChartType.LINE, config)
        
        assert 'chart' in result
        assert result['type'] == 'line'
        assert 'error' not in result
    
    def test_create_bar_chart(self):
        """Test bar chart creation."""
        generator = ChartGenerator()
        
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 15]
        })
        
        config = {
            'x_column': 'category',
            'y_column': 'value',
            'title': 'Test Bar Chart'
        }
        
        result = generator.create_chart(data, ChartType.BAR, config)
        
        assert 'chart' in result
        assert result['type'] == 'bar'
        assert 'error' not in result
    
    def test_create_chart_missing_columns(self):
        """Test chart creation with missing columns."""
        generator = ChartGenerator()
        
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        
        config = {
            'x_column': 'missing_x',
            'y_column': 'y',
            'title': 'Test Chart'
        }
        
        result = generator.create_chart(data, ChartType.LINE, config)
        
        assert 'error' in result
        assert 'Missing required columns' in result['error']


class TestAnalyticsEngine:
    """Test analytics engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics_engine = AnalyticsEngine()
    
    @pytest.mark.asyncio
    async def test_execute_query_detection_metrics(self):
        """Test query execution for detection metrics."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=1), now)
        
        query = AnalyticsQuery(
            metric_type=MetricType.DETECTION_METRICS,
            time_range=time_range,
            aggregation=AggregationType.COUNT
        )
        
        result = await self.analytics_engine.execute_query(query)
        
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert result.metadata is not None
        assert 'total_rows' in result.metadata
    
    @pytest.mark.asyncio
    async def test_execute_query_with_filters(self):
        """Test query execution with filters."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=1), now)
        
        query = AnalyticsQuery(
            metric_type=MetricType.SYSTEM_METRICS,
            time_range=time_range,
            filters={'service': ['anomaly_detection']},
            group_by=['service'],
            aggregation=AggregationType.MEAN
        )
        
        result = await self.analytics_engine.execute_query(query)
        
        assert result.data is not None
        assert 'filtered_rows' in result.metadata
    
    @pytest.mark.asyncio
    async def test_create_dashboard(self):
        """Test dashboard creation."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=1), now)
        
        widget = DashboardWidget(
            widget_id="test_widget",
            title="Test Widget",
            chart_type=ChartType.LINE,
            query=AnalyticsQuery(
                metric_type=MetricType.SYSTEM_METRICS,
                time_range=time_range
            )
        )
        
        dashboard = Dashboard(
            dashboard_id="test_dashboard",
            title="Test Dashboard",
            description="Test dashboard",
            widgets=[widget]
        )
        
        success = await self.analytics_engine.create_dashboard(dashboard)
        assert success is True
        
        # Verify dashboard can be retrieved
        retrieved = await self.analytics_engine.get_dashboard("test_dashboard")
        assert retrieved is not None
        assert retrieved.dashboard_id == "test_dashboard"
    
    @pytest.mark.asyncio
    async def test_render_dashboard(self):
        """Test dashboard rendering."""
        # First create a dashboard
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=1), now)
        
        widget = DashboardWidget(
            widget_id="test_widget",
            title="Test Widget",
            chart_type=ChartType.LINE,
            query=AnalyticsQuery(
                metric_type=MetricType.SYSTEM_METRICS,
                time_range=time_range
            ),
            config={
                'x_column': 'timestamp',
                'y_column': 'cpu_usage',
                'title': 'CPU Usage'
            }
        )
        
        dashboard = Dashboard(
            dashboard_id="test_render_dashboard",
            title="Test Render Dashboard",
            description="Test dashboard rendering",
            widgets=[widget]
        )
        
        await self.analytics_engine.create_dashboard(dashboard)
        
        # Render the dashboard
        rendered = await self.analytics_engine.render_dashboard("test_render_dashboard")
        
        assert 'dashboard_id' in rendered
        assert 'widgets' in rendered
        assert len(rendered['widgets']) == 1
        assert rendered['widgets'][0]['widget_id'] == 'test_widget'
    
    @pytest.mark.asyncio
    async def test_get_insights(self):
        """Test insights generation."""
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=24), now)
        
        insights = await self.analytics_engine.get_insights(
            MetricType.DETECTION_METRICS,
            time_range
        )
        
        assert 'insights' in insights
        assert isinstance(insights['insights'], list)
        assert 'generated_at' in insights


class TestDashboardTemplateFactory:
    """Test dashboard template factory."""
    
    def test_create_system_overview_dashboard(self):
        """Test system overview dashboard creation."""
        dashboard = DashboardTemplateFactory.create_system_overview_dashboard()
        
        assert dashboard.dashboard_id == "system_overview"
        assert dashboard.title == "System Overview"
        assert len(dashboard.widgets) > 0
        assert "system" in dashboard.tags
        
        # Check widget configurations
        for widget in dashboard.widgets:
            assert widget.widget_id is not None
            assert widget.title is not None
            assert widget.query.metric_type == MetricType.SYSTEM_METRICS
    
    def test_create_anomaly_detection_dashboard(self):
        """Test anomaly detection dashboard creation."""
        dashboard = DashboardTemplateFactory.create_anomaly_detection_dashboard()
        
        assert dashboard.dashboard_id == "anomaly_detection"
        assert dashboard.title == "Anomaly Detection Analytics"
        assert len(dashboard.widgets) > 0
        assert "anomaly" in dashboard.tags
        
        # Check widget configurations
        for widget in dashboard.widgets:
            assert widget.widget_id is not None
            assert widget.title is not None
            assert widget.query.metric_type == MetricType.DETECTION_METRICS
    
    def test_create_security_monitoring_dashboard(self):
        """Test security monitoring dashboard creation."""
        dashboard = DashboardTemplateFactory.create_security_monitoring_dashboard()
        
        assert dashboard.dashboard_id == "security_monitoring"
        assert dashboard.title == "Security Monitoring"
        assert len(dashboard.widgets) > 0
        assert "security" in dashboard.tags
        
        # Check widget configurations
        for widget in dashboard.widgets:
            assert widget.widget_id is not None
            assert widget.title is not None
            assert widget.query.metric_type == MetricType.SECURITY_METRICS


class TestDashboardService:
    """Test dashboard service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dashboard_service = DashboardService()
    
    @pytest.mark.asyncio
    async def test_list_dashboards(self):
        """Test dashboard listing."""
        dashboards = await self.dashboard_service.list_dashboards()
        
        assert isinstance(dashboards, list)
        assert len(dashboards) > 0  # Should have default dashboards
        
        # Check dashboard structure
        for dashboard in dashboards:
            assert 'dashboard_id' in dashboard
            assert 'title' in dashboard
            assert 'description' in dashboard
            assert 'tags' in dashboard
    
    @pytest.mark.asyncio
    async def test_list_dashboards_with_tags(self):
        """Test dashboard listing with tag filtering."""
        # List system dashboards
        system_dashboards = await self.dashboard_service.list_dashboards(['system'])
        
        assert isinstance(system_dashboards, list)
        for dashboard in system_dashboards:
            assert any('system' in tag for tag in dashboard['tags'])
    
    @pytest.mark.asyncio
    async def test_get_dashboard(self):
        """Test getting a specific dashboard."""
        # Get system overview dashboard (should exist by default)
        dashboard_data = await self.dashboard_service.get_dashboard('system_overview', 'test_viewer')
        
        if 'error' not in dashboard_data:
            assert 'dashboard_id' in dashboard_data
            assert 'title' in dashboard_data
            assert 'widgets' in dashboard_data
            assert dashboard_data['dashboard_id'] == 'system_overview'
    
    @pytest.mark.asyncio
    async def test_create_custom_dashboard(self):
        """Test custom dashboard creation."""
        dashboard_config = {
            'dashboard_id': 'test_custom_dashboard',
            'title': 'Test Custom Dashboard',
            'description': 'A test custom dashboard',
            'widgets': [
                {
                    'widget_id': 'test_widget',
                    'title': 'Test Widget',
                    'chart_type': 'line',
                    'query': {
                        'metric_type': 'system_metrics',
                        'time_range': [
                            (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                            datetime.utcnow().isoformat()
                        ],
                        'group_by': [],
                        'aggregation': 'mean'
                    },
                    'config': {
                        'x_column': 'timestamp',
                        'y_column': 'cpu_usage',
                        'title': 'CPU Usage'
                    }
                }
            ],
            'tags': ['custom', 'test'],
            'is_public': False
        }
        
        result = await self.dashboard_service.create_custom_dashboard(dashboard_config)
        
        if 'error' not in result:
            assert result['success'] is True
            assert result['dashboard_id'] == 'test_custom_dashboard'
    
    @pytest.mark.asyncio
    async def test_get_dashboard_insights(self):
        """Test dashboard insights generation."""
        # Try to get insights for system overview dashboard
        insights = await self.dashboard_service.get_dashboard_insights('system_overview')
        
        if 'error' not in insights:
            assert 'insights' in insights
            assert isinstance(insights['insights'], list)
            assert 'generated_at' in insights
    
    @pytest.mark.asyncio
    async def test_export_dashboard_json(self):
        """Test dashboard export in JSON format."""
        from src.anomaly_detection.application.services.intelligence.dashboard_service import ReportFormat
        
        result = await self.dashboard_service.export_dashboard('system_overview', ReportFormat.JSON)
        
        if 'error' not in result:
            assert result['success'] is True
            assert result['format'] == 'json'
            assert 'data' in result
    
    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self):
        """Test dashboard usage metrics."""
        metrics = await self.dashboard_service.get_dashboard_metrics()
        
        if 'error' not in metrics:
            assert 'overview' in metrics
            assert 'top_dashboards' in metrics
            assert 'generated_at' in metrics
            
            overview = metrics['overview']
            assert 'total_dashboards' in overview
            assert 'total_views' in overview


class TestGlobalInstances:
    """Test global instance management."""
    
    def test_get_analytics_engine_singleton(self):
        """Test analytics engine singleton."""
        engine1 = get_analytics_engine()
        engine2 = get_analytics_engine()
        
        assert engine1 is engine2  # Should be the same instance
    
    def test_initialize_analytics_engine(self):
        """Test analytics engine initialization."""
        engine = initialize_analytics_engine()
        
        assert engine is not None
        assert isinstance(engine, AnalyticsEngine)
    
    def test_get_dashboard_service_singleton(self):
        """Test dashboard service singleton."""
        service1 = get_dashboard_service()
        service2 = get_dashboard_service()
        
        assert service1 is service2  # Should be the same instance
    
    def test_initialize_dashboard_service(self):
        """Test dashboard service initialization."""
        service = initialize_dashboard_service()
        
        assert service is not None
        assert isinstance(service, DashboardService)


class TestIntegration:
    """Integration tests for analytics and dashboard functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics_engine = AnalyticsEngine()
        self.dashboard_service = DashboardService()
    
    @pytest.mark.asyncio
    async def test_end_to_end_dashboard_workflow(self):
        """Test complete dashboard workflow."""
        # 1. Create a dashboard
        now = datetime.utcnow()
        time_range = (now - timedelta(hours=1), now)
        
        widget = DashboardWidget(
            widget_id="integration_widget",
            title="Integration Test Widget",
            chart_type=ChartType.BAR,
            query=AnalyticsQuery(
                metric_type=MetricType.PERFORMANCE_METRICS,
                time_range=time_range,
                group_by=['endpoint'],
                aggregation=AggregationType.MEAN
            ),
            config={
                'x_column': 'endpoint',
                'y_column': 'response_time',
                'title': 'Average Response Time by Endpoint'
            }
        )
        
        dashboard = Dashboard(
            dashboard_id="integration_test_dashboard",
            title="Integration Test Dashboard",
            description="End-to-end integration test",
            widgets=[widget],
            tags=['integration', 'test']
        )
        
        # 2. Create dashboard
        success = await self.analytics_engine.create_dashboard(dashboard)
        assert success is True
        
        # 3. Render dashboard
        rendered = await self.analytics_engine.render_dashboard("integration_test_dashboard")
        assert 'dashboard_id' in rendered
        assert len(rendered['widgets']) == 1
        
        # 4. Get insights
        insights = await self.analytics_engine.get_insights(
            MetricType.PERFORMANCE_METRICS,
            time_range
        )
        assert 'insights' in insights
        
        # 5. Test dashboard service integration
        dashboard_data = await self.dashboard_service.get_dashboard("integration_test_dashboard")
        if 'error' not in dashboard_data:
            assert dashboard_data['dashboard_id'] == "integration_test_dashboard"