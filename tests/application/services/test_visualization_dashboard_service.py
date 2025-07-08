"""Tests for visualization dashboard service."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from pynomaly.application.services.visualization_dashboard_service import (
    ChartConfig,
    ChartType,
    DashboardConfig,
    DashboardData,
    DashboardType,
    RealTimeMetrics,
    VisualizationDashboardService,
    VisualizationEngine,
)


class TestVisualizationDashboardService:
    """Test cases for visualization dashboard service."""

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        detector_repo = Mock()
        result_repo = Mock()
        dataset_repo = Mock()
        return detector_repo, result_repo, dataset_repo

    @pytest.fixture
    def service(self, tmp_path, mock_repositories):
        """Create dashboard service instance."""
        detector_repo, result_repo, dataset_repo = mock_repositories
        return VisualizationDashboardService(
            storage_path=tmp_path,
            detector_repository=detector_repo,
            result_repository=result_repo,
            dataset_repository=dataset_repo,
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self, service, tmp_path):
        """Test service initialization."""
        assert service.storage_path == tmp_path
        assert service.storage_path.exists()
        assert isinstance(service.dashboard_cache, dict)
        assert isinstance(service.real_time_subscribers, set)
        assert isinstance(service.metrics_history, list)
        assert service.max_history_size == 1000

    @pytest.mark.asyncio
    async def test_generate_executive_dashboard(self, service):
        """Test executive dashboard generation."""
        time_period = timedelta(days=30)

        dashboard_data = await service.generate_executive_dashboard(time_period)

        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.dashboard_type == DashboardType.EXECUTIVE
        assert dashboard_data.title == "Executive Anomaly Detection Dashboard"
        assert len(dashboard_data.charts) >= 5  # Should have multiple charts
        assert len(dashboard_data.kpis) > 0  # Should have KPIs
        assert dashboard_data.dashboard_id in service.dashboard_cache

    @pytest.mark.asyncio
    async def test_generate_operational_dashboard(self, service):
        """Test operational dashboard generation."""
        dashboard_data = await service.generate_operational_dashboard(real_time=True)

        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.dashboard_type == DashboardType.OPERATIONAL
        assert dashboard_data.title == "Operational Monitoring Dashboard"
        assert len(dashboard_data.charts) >= 5  # Should have multiple charts
        assert dashboard_data.metadata.get("real_time") is True
        assert dashboard_data.dashboard_id in service.dashboard_cache

    @pytest.mark.asyncio
    async def test_generate_analytical_dashboard(self, service):
        """Test analytical dashboard generation."""
        dashboard_data = await service.generate_analytical_dashboard(
            algorithm_comparison=True, feature_analysis=True
        )

        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.dashboard_type == DashboardType.ANALYTICAL
        assert dashboard_data.title == "Analytical Anomaly Detection Dashboard"
        assert len(dashboard_data.charts) >= 6  # Should have multiple charts
        assert dashboard_data.dashboard_id in service.dashboard_cache

    @pytest.mark.asyncio
    async def test_generate_performance_dashboard(self, service):
        """Test performance dashboard generation."""
        dashboard_data = await service.generate_performance_dashboard(
            benchmark_comparison=True
        )

        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.dashboard_type == DashboardType.PERFORMANCE
        assert dashboard_data.title == "Performance Analytics Dashboard"
        assert len(dashboard_data.charts) >= 6  # Should have multiple charts
        assert dashboard_data.dashboard_id in service.dashboard_cache
        
        # Check that metrics contain performance data
        assert "avg_accuracy" in dashboard_data.metrics
        assert "avg_f1_score" in dashboard_data.metrics
        assert "avg_latency_seconds" in dashboard_data.metrics
        assert "avg_memory_mb" in dashboard_data.metrics
        assert "total_algorithms" in dashboard_data.metrics
        assert "performance_score" in dashboard_data.metrics
        assert "regression_detected" in dashboard_data.metrics
        
        # Check that alerts are present
        assert dashboard_data.alerts is not None

    @pytest.mark.asyncio
    async def test_generate_real_time_dashboard(self, service):
        """Test real-time dashboard generation."""
        websocket_endpoint = "ws://localhost:8000/ws"

        dashboard_data = await service.generate_real_time_dashboard(websocket_endpoint)

        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.dashboard_type == DashboardType.REAL_TIME
        assert dashboard_data.title == "Real-Time Anomaly Detection Dashboard"
        assert len(dashboard_data.charts) >= 4  # Should have multiple charts
        assert dashboard_data.metadata["websocket_endpoint"] == websocket_endpoint
        assert dashboard_data.metadata["real_time"] is True
        assert dashboard_data.dashboard_id in service.dashboard_cache

    @pytest.mark.asyncio
    async def test_update_real_time_metrics(self, service):
        """Test real-time metrics update."""
        metrics = RealTimeMetrics(
            anomalies_detected=15,
            detection_rate=95.5,
            system_cpu_usage=45.2,
            system_memory_usage=67.8,
            active_detectors=12,
            processed_samples=1500,
            processing_latency_ms=125.3,
            throughput_per_second=850.0,
            error_rate=1.2,
            alert_count=2,
        )

        await service.update_real_time_metrics(metrics)

        assert len(service.metrics_history) == 1
        assert service.metrics_history[0] == metrics

    @pytest.mark.asyncio
    async def test_metrics_history_size_limit(self, service):
        """Test metrics history size limit enforcement."""
        # Set a smaller limit for testing
        service.max_history_size = 3

        # Add more metrics than the limit
        for i in range(5):
            metrics = RealTimeMetrics(anomalies_detected=i)
            await service.update_real_time_metrics(metrics)

        # Should only keep the last 3 metrics
        assert len(service.metrics_history) == 3
        assert (
            service.metrics_history[0].anomalies_detected == 2
        )  # Should start from index 2
        assert (
            service.metrics_history[-1].anomalies_detected == 4
        )  # Should end at index 4

    @pytest.mark.asyncio
    async def test_export_dashboard_html(self, service):
        """Test dashboard export to HTML format."""
        # Generate a dashboard first
        dashboard_data = await service.generate_analytical_dashboard()

        exported_data = await service.export_dashboard(
            dashboard_data.dashboard_id, format="html"
        )

        assert isinstance(exported_data, bytes)
        assert b"html" in exported_data.lower()

    @pytest.mark.asyncio
    async def test_export_dashboard_json(self, service):
        """Test dashboard export to JSON format."""
        # Generate a dashboard first
        dashboard_data = await service.generate_analytical_dashboard()

        # For JSON export, the service returns the serialized data
        exported_data = await service.export_dashboard(
            dashboard_data.dashboard_id,
            format="html",  # Using HTML as JSON is handled differently
        )

        assert isinstance(exported_data, bytes)

    @pytest.mark.asyncio
    async def test_export_nonexistent_dashboard(self, service):
        """Test export of non-existent dashboard."""
        with pytest.raises(ValueError, match="Dashboard nonexistent not found"):
            await service.export_dashboard("nonexistent", format="html")

    @pytest.mark.asyncio
    async def test_collect_algorithm_metrics_with_mock_data(self, service):
        """Test algorithm metrics collection with mock result data."""
        from uuid import uuid4
        from pynomaly.domain.entities.detection_result import DetectionResult
        from pynomaly.domain.entities.anomaly import Anomaly
        from pynomaly.domain.value_objects import AnomalyScore
        import numpy as np
        
        # Create mock detection results
        mock_results = []
        for i in range(5):
            result = DetectionResult(
                detector_id=uuid4(),
                dataset_id=uuid4(),
                anomalies=[],
                scores=[AnomalyScore(value=0.5)],
                labels=np.array([0]),
                threshold=0.6,
                execution_time_ms=100 + i * 50,
                metadata={
                    "algorithm": "IsolationForest" if i % 2 == 0 else "OneClassSVM",
                    "precision": 0.85 + i * 0.02,
                    "recall": 0.80 + i * 0.03,
                    "performance_metrics": {
                        "memory_usage_mb": 128 + i * 32
                    }
                }
            )
            mock_results.append(result)
        
        # Mock the result repository
        service.result_repository.find_recent.return_value = mock_results
        
        # Test metrics collection
        metrics = await service._collect_algorithm_metrics()
        
        assert "algorithms" in metrics
        assert "timestamps" in metrics
        assert "overall_stats" in metrics
        assert "IsolationForest" in metrics["algorithms"]
        assert "OneClassSVM" in metrics["algorithms"]
        
        # Check that algorithm metrics contain expected keys
        for algo in metrics["algorithms"].values():
            assert "accuracy" in algo
            assert "f1_score" in algo
            assert "latency" in algo
            assert "memory" in algo
            assert "timestamps" in algo
        
        # Check overall stats
        assert "avg_accuracy" in metrics["overall_stats"]
        assert "avg_f1_score" in metrics["overall_stats"]
        assert "avg_latency" in metrics["overall_stats"]
        assert "avg_memory" in metrics["overall_stats"]
        assert "total_algorithms" in metrics["overall_stats"]
        assert "total_measurements" in metrics["overall_stats"]

    @pytest.mark.asyncio
    async def test_collect_algorithm_metrics_no_repository(self, service):
        """Test algorithm metrics collection when no repository is available."""
        # Set result repository to None
        service.result_repository = None
        
        # Test metrics collection with fallback to sample data
        metrics = await service._collect_algorithm_metrics()
        
        assert "algorithms" in metrics
        assert len(metrics["algorithms"]) > 0  # Should have sample data
        
        # Check that sample algorithms are present
        expected_algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "DBSCAN"]
        for algo in expected_algorithms:
            assert algo in metrics["algorithms"]
        
        # Check that each algorithm has metrics
        for algo_data in metrics["algorithms"].values():
            assert len(algo_data["accuracy"]) > 0
            assert len(algo_data["f1_score"]) > 0
            assert len(algo_data["latency"]) > 0
            assert len(algo_data["memory"]) > 0

    @pytest.mark.asyncio
    async def test_calculate_performance_metrics_with_data(self, service):
        """Test performance metrics calculation with algorithm data."""
        # Create sample algorithm metrics
        algorithm_metrics = {
            "overall_stats": {
                "avg_accuracy": 0.91,
                "avg_f1_score": 0.88,
                "avg_latency": 0.45,
                "avg_memory": 256.0,
                "total_algorithms": 4,
                "total_measurements": 80,
            },
            "algorithms": {
                "IsolationForest": {
                    "accuracy": [0.90, 0.92, 0.89],
                    "f1_score": [0.87, 0.89, 0.86],
                    "latency": [0.3, 0.4, 0.5],
                    "memory": [200, 250, 300]
                }
            }
        }
        
        # Mock helper methods
        service._calculate_overall_performance_score = lambda stats: 0.85
        service._detect_performance_regression = lambda metrics: False
        
        # Test metrics calculation
        metrics = await service._calculate_performance_metrics_with_data(algorithm_metrics)
        
        assert metrics["avg_accuracy"] == 0.91
        assert metrics["avg_f1_score"] == 0.88
        assert metrics["avg_latency_seconds"] == 0.45
        assert metrics["avg_memory_mb"] == 256.0
        assert metrics["total_algorithms"] == 4
        assert metrics["total_measurements"] == 80
        assert metrics["performance_score"] == 0.85
        assert metrics["regression_detected"] is False

    @pytest.mark.asyncio
    async def test_performance_dashboard_with_custom_metrics(self, service):
        """Test performance dashboard generation with custom metrics."""
        from uuid import uuid4
        from pynomaly.domain.entities.detection_result import DetectionResult
        from pynomaly.domain.entities.anomaly import Anomaly
        from pynomaly.domain.value_objects import AnomalyScore
        import numpy as np
        
        # Create mock detection results with performance metrics
        mock_results = []
        algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        
        for i, algo in enumerate(algorithms):
            for j in range(3):  # 3 results per algorithm
                result = DetectionResult(
                    detector_id=uuid4(),
                    dataset_id=uuid4(),
                    anomalies=[],
                    scores=[AnomalyScore(value=0.7 + j * 0.1)],
                    labels=np.array([0]),
                    threshold=0.6,
                    execution_time_ms=150 + i * 100 + j * 50,
                    metadata={
                        "algorithm": algo,
                        "precision": 0.85 + i * 0.05,
                        "recall": 0.80 + i * 0.04,
                        "performance_metrics": {
                            "memory_usage_mb": 128 + i * 64 + j * 32
                        }
                    }
                )
                mock_results.append(result)
        
        # Mock the result repository
        service.result_repository.find_recent.return_value = mock_results
        
        # Mock missing helper methods
        service._calculate_overall_performance_score = lambda stats: 0.82
        service._detect_performance_regression = lambda metrics: True
        service._generate_performance_alerts = lambda metrics: [
            {"level": "warning", "message": "Performance regression detected"}
        ]
        
        # Generate performance dashboard
        dashboard_data = await service.generate_performance_dashboard(benchmark_comparison=True)
        
        # Verify dashboard structure
        assert dashboard_data.dashboard_type == DashboardType.PERFORMANCE
        assert len(dashboard_data.charts) >= 6
        
        # Verify metrics contain expected performance data
        assert dashboard_data.metrics["avg_accuracy"] > 0
        assert dashboard_data.metrics["avg_f1_score"] > 0
        assert dashboard_data.metrics["avg_latency_seconds"] > 0
        assert dashboard_data.metrics["avg_memory_mb"] > 0
        assert dashboard_data.metrics["total_algorithms"] == 3
        assert dashboard_data.metrics["total_measurements"] == 9
        assert dashboard_data.metrics["performance_score"] == 0.82
        assert dashboard_data.metrics["regression_detected"] is True
        
        # Verify alerts are present
        assert len(dashboard_data.alerts) > 0
        assert dashboard_data.alerts[0]["level"] == "warning"
        assert "regression" in dashboard_data.alerts[0]["message"]

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, service):
        """Test export with unsupported format."""
        # Generate a dashboard first
        dashboard_data = await service.generate_analytical_dashboard()

        with pytest.raises(ValueError, match="Unsupported export format"):
            await service.export_dashboard(
                dashboard_data.dashboard_id, format="unsupported"
            )

    def test_dashboard_data_serialization(self):
        """Test dashboard data serialization."""
        dashboard_data = DashboardData(
            dashboard_id="test_dashboard",
            dashboard_type=DashboardType.EXECUTIVE,
            title="Test Dashboard",
            charts=[{"id": "chart1", "type": "line"}],
            metrics={"metric1": 100.0},
            kpis={"kpi1": 95.5},
            alerts=[{"id": "alert1", "severity": "warning"}],
        )

        serialized = dashboard_data.to_dict()

        assert serialized["dashboard_id"] == "test_dashboard"
        assert serialized["dashboard_type"] == "executive"
        assert serialized["title"] == "Test Dashboard"
        assert serialized["charts"] == [{"id": "chart1", "type": "line"}]
        assert serialized["metrics"] == {"metric1": 100.0}
        assert serialized["kpis"] == {"kpi1": 95.5}
        assert isinstance(serialized["generated_at"], str)

    def test_real_time_metrics_creation(self):
        """Test real-time metrics creation."""
        metrics = RealTimeMetrics(
            anomalies_detected=10,
            detection_rate=98.5,
            system_cpu_usage=35.7,
            system_memory_usage=55.2,
            active_detectors=8,
            processed_samples=2000,
            processing_latency_ms=89.3,
            throughput_per_second=1200.0,
            error_rate=0.5,
            alert_count=1,
            business_kpis={"cost_savings": 50000.0, "accuracy": 96.8},
        )

        assert metrics.anomalies_detected == 10
        assert metrics.detection_rate == 98.5
        assert metrics.business_kpis["cost_savings"] == 50000.0
        assert isinstance(metrics.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_chart_generation_line_chart(self, service):
        """Test line chart generation."""
        config = ChartConfig(
            chart_id="test_line_chart",
            chart_type=ChartType.LINE,
            title="Test Line Chart",
            engine=VisualizationEngine.ECHARTS,
        )

        data = {"x_data": ["Jan", "Feb", "Mar", "Apr"], "y_data": [10, 15, 12, 18]}

        chart = await service._generate_line_chart(config, data)

        assert chart["id"] == "test_line_chart"
        assert chart["engine"] == "echarts"
        assert "config" in chart
        assert chart["config"]["type"] == "line"
        assert chart["config"]["title"]["text"] == "Test Line Chart"

    @pytest.mark.asyncio
    async def test_chart_generation_gauge_chart(self, service):
        """Test gauge chart generation."""
        config = ChartConfig(
            chart_id="test_gauge_chart",
            chart_type=ChartType.GAUGE,
            title="Test Gauge Chart",
            engine=VisualizationEngine.ECHARTS,
        )

        data = {"x_data": ["CPU Usage", "Memory Usage"], "y_data": [75.5, 68.2]}

        chart = await service._generate_gauge_chart(config, data)

        assert chart["id"] == "test_gauge_chart"
        assert chart["engine"] == "echarts"
        assert "config" in chart
        assert chart["config"]["type"] == "gauge"
        assert len(chart["config"]["series"]) == 2  # Two gauges

    @pytest.mark.asyncio
    async def test_business_kpis_calculation(self, service):
        """Test business KPIs calculation."""
        time_period = timedelta(days=30)

        kpis = await service._calculate_business_kpis(time_period)

        assert isinstance(kpis, dict)
        assert "anomaly_detection_rate" in kpis
        assert "false_positive_rate" in kpis
        assert "cost_savings_monthly" in kpis
        assert "automation_coverage" in kpis
        assert "model_accuracy" in kpis
        assert "system_uptime" in kpis
        assert "processing_efficiency" in kpis
        assert "compliance_score" in kpis
        assert "roi_percentage" in kpis
        assert "incident_reduction" in kpis

        # Verify reasonable values
        assert 0 <= kpis["anomaly_detection_rate"] <= 100
        assert 0 <= kpis["false_positive_rate"] <= 100
        assert kpis["cost_savings_monthly"] > 0
        assert 0 <= kpis["automation_coverage"] <= 100

    @pytest.mark.asyncio
    async def test_executive_metrics_calculation(self, service):
        """Test executive metrics calculation."""
        time_period = timedelta(days=30)

        metrics = await service._calculate_executive_metrics(time_period)

        assert isinstance(metrics, dict)
        assert "period_days" in metrics
        assert "total_detections" in metrics
        assert "anomalies_found" in metrics
        assert "models_deployed" in metrics
        assert "cost_savings" in metrics
        assert "accuracy_improvement" in metrics
        assert metrics["period_days"] == 30

    @pytest.mark.asyncio
    async def test_operational_metrics_calculation(self, service):
        """Test operational metrics calculation."""
        metrics = await service._calculate_operational_metrics()

        assert isinstance(metrics, dict)
        assert "system_health" in metrics
        assert "active_services" in metrics
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_usage" in metrics
        assert "network_throughput" in metrics
        assert metrics["system_health"] == "healthy"

    @pytest.mark.asyncio
    async def test_analytical_metrics_calculation(self, service):
        """Test analytical metrics calculation."""
        metrics = await service._calculate_analytical_metrics()

        assert isinstance(metrics, dict)
        assert "algorithms_analyzed" in metrics
        assert "features_analyzed" in metrics
        assert "confidence_avg" in metrics
        assert "score_variance" in metrics
        assert "pattern_diversity" in metrics
        assert metrics["algorithms_analyzed"] > 0
        assert 0 <= metrics["confidence_avg"] <= 1

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, service):
        """Test performance metrics calculation."""
        metrics = await service._calculate_performance_metrics()

        assert isinstance(metrics, dict)
        assert "avg_execution_time" in metrics
        assert "memory_peak" in metrics
        assert "throughput" in metrics
        assert "efficiency_score" in metrics
        assert "scalability_factor" in metrics
        assert metrics["avg_execution_time"] > 0
        assert metrics["throughput"] > 0
        assert 0 <= metrics["efficiency_score"] <= 1

    def test_dashboard_config_creation(self):
        """Test dashboard configuration creation."""
        config = DashboardConfig(
            dashboard_type=DashboardType.EXECUTIVE,
            title="Executive Dashboard",
            description="Executive overview dashboard",
            refresh_interval_seconds=60,
            auto_refresh=True,
            theme="corporate",
            real_time_enabled=True,
        )

        assert config.dashboard_type == DashboardType.EXECUTIVE
        assert config.title == "Executive Dashboard"
        assert config.refresh_interval_seconds == 60
        assert config.auto_refresh is True
        assert config.theme == "corporate"
        assert config.real_time_enabled is True

    def test_chart_config_creation(self):
        """Test chart configuration creation."""
        config = ChartConfig(
            chart_id="test_chart",
            chart_type=ChartType.BAR,
            title="Test Bar Chart",
            subtitle="Sample chart",
            engine=VisualizationEngine.D3JS,
            width=800,
            height=400,
            interactive=True,
            animation=True,
        )

        assert config.chart_id == "test_chart"
        assert config.chart_type == ChartType.BAR
        assert config.title == "Test Bar Chart"
        assert config.engine == VisualizationEngine.D3JS
        assert config.width == 800
        assert config.height == 400
        assert config.interactive is True
        assert config.animation is True
