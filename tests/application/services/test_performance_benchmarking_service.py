"""Tests for performance benchmarking service."""

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.performance_benchmarking_service import (
    BenchmarkConfig,
    BenchmarkSuite,
    PerformanceBenchmarkingService,
    PerformanceMetrics,
    SystemMonitor,
)


class TestPerformanceBenchmarkingService:
    """Test cases for performance benchmarking service."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create benchmarking service instance."""
        return PerformanceBenchmarkingService(storage_path=tmp_path)

    @pytest.fixture
    def sample_config(self):
        """Create sample benchmark configuration."""
        return BenchmarkConfig(
            benchmark_name="Test Benchmark",
            description="Test benchmark configuration",
            dataset_sizes=[100, 500],
            feature_dimensions=[5, 10],
            contamination_rates=[0.1, 0.2],
            algorithms=["TestAlgorithm1", "TestAlgorithm2"],
            iterations=2,
            timeout_seconds=60.0,
        )

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])
        df["label"] = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        return df

    @pytest.mark.asyncio
    async def test_service_initialization(self, service, tmp_path):
        """Test service initialization."""
        assert service.storage_path == tmp_path
        assert service.storage_path.exists()
        assert isinstance(service.benchmark_results, dict)
        assert isinstance(service.active_benchmarks, set)
        assert isinstance(service.performance_history, list)
        assert isinstance(service.baseline_metrics, dict)

    @pytest.mark.asyncio
    async def test_create_benchmark_suite(self, service, sample_config):
        """Test benchmark suite creation."""
        suite_id = await service.create_benchmark_suite(
            suite_name="Test Suite",
            description="Test suite description",
            config=sample_config,
        )

        assert suite_id in service.benchmark_results
        suite = service.benchmark_results[suite_id]

        assert suite.suite_name == "Test Suite"
        assert suite.description == "Test suite description"
        assert suite.config == sample_config
        assert isinstance(suite.test_environment, dict)
        assert isinstance(suite.system_info, dict)

    @pytest.mark.asyncio
    async def test_benchmark_single_run(self, service, sample_dataset):
        """Test single benchmark run."""
        with patch.object(
            service, "_run_detection_algorithm", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = {"anomalies": np.zeros(100)}

            metrics = await service._benchmark_single_run(
                algorithm_name="TestAlgorithm",
                dataset=sample_dataset,
                dataset_size=100,
                feature_dimension=5,
                contamination_rate=0.1,
            )

            assert metrics.algorithm_name == "TestAlgorithm"
            assert metrics.dataset_size == 100
            assert metrics.feature_dimension == 5
            assert metrics.contamination_rate == 0.1
            assert metrics.execution_time_seconds > 0
            assert metrics.success
            assert metrics.error_message is None

    @pytest.mark.asyncio
    async def test_benchmark_single_run_with_error(self, service, sample_dataset):
        """Test single benchmark run with error handling."""
        with patch.object(
            service, "_run_detection_algorithm", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = Exception("Test error")

            metrics = await service._benchmark_single_run(
                algorithm_name="TestAlgorithm",
                dataset=sample_dataset,
                dataset_size=100,
                feature_dimension=5,
                contamination_rate=0.1,
            )

            assert not metrics.success
            assert metrics.error_message == "Test error"

    @pytest.mark.asyncio
    async def test_generate_synthetic_dataset(self, service):
        """Test synthetic dataset generation."""
        dataset = await service._generate_synthetic_dataset(
            size=1000, features=10, contamination=0.1
        )

        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) == 1000
        assert dataset.shape[1] == 11  # 10 features + 1 label
        assert "label" in dataset.columns

        # Check contamination rate
        anomaly_rate = dataset["label"].sum() / len(dataset)
        assert abs(anomaly_rate - 0.1) < 0.05  # Allow some variance

    @pytest.mark.asyncio
    async def test_calculate_average_metrics(self, service):
        """Test average metrics calculation."""
        metrics_list = [
            PerformanceMetrics(
                algorithm_name="Test",
                execution_time_seconds=1.0,
                peak_memory_mb=100.0,
                accuracy_score=0.8,
                success=True,
            ),
            PerformanceMetrics(
                algorithm_name="Test",
                execution_time_seconds=2.0,
                peak_memory_mb=200.0,
                accuracy_score=0.9,
                success=True,
            ),
        ]

        avg_metrics = await service._calculate_average_metrics(metrics_list)

        assert avg_metrics.algorithm_name == "Test"
        assert avg_metrics.execution_time_seconds == 1.5
        assert avg_metrics.peak_memory_mb == 150.0
        assert avg_metrics.accuracy_score == 0.85

    @pytest.mark.asyncio
    async def test_calculate_average_metrics_with_failures(self, service):
        """Test average metrics calculation with failed runs."""
        metrics_list = [
            PerformanceMetrics(
                algorithm_name="Test",
                execution_time_seconds=1.0,
                peak_memory_mb=100.0,
                accuracy_score=0.8,
                success=True,
            ),
            PerformanceMetrics(
                algorithm_name="Test", error_message="Failed", success=False
            ),
        ]

        avg_metrics = await service._calculate_average_metrics(metrics_list)

        # Should only consider successful runs
        assert avg_metrics.execution_time_seconds == 1.0
        assert avg_metrics.peak_memory_mb == 100.0
        assert avg_metrics.accuracy_score == 0.8

    @pytest.mark.asyncio
    async def test_scalability_test(self, service):
        """Test scalability testing functionality."""
        with patch.object(
            service, "_benchmark_single_run", new_callable=AsyncMock
        ) as mock_benchmark:
            # Mock different execution times for different scales
            def mock_benchmark_response(
                algorithm_name, dataset, dataset_size, **kwargs
            ):
                # Simulate linear scaling (time proportional to size)
                time_per_sample = 0.001
                execution_time = dataset_size * time_per_sample

                return PerformanceMetrics(
                    algorithm_name=algorithm_name,
                    dataset_size=dataset_size,
                    execution_time_seconds=execution_time,
                    peak_memory_mb=dataset_size * 0.1,
                    success=True,
                )

            mock_benchmark.side_effect = mock_benchmark_response

            results = await service.run_scalability_test(
                algorithm_name="TestAlgorithm",
                base_dataset_size=100,
                scale_factors=[1, 2, 4],
                feature_dimension=10,
            )

            assert results["algorithm"] == "TestAlgorithm"
            assert len(results["results"]) == 3

            # Check that execution time scales appropriately
            times = [r.execution_time_seconds for r in results["results"]]
            assert times[1] > times[0]  # 2x scale should take more time
            assert times[2] > times[1]  # 4x scale should take even more time

            assert "scalability_summary" in results

    @pytest.mark.asyncio
    async def test_memory_stress_test(self, service):
        """Test memory stress testing."""
        with patch.object(
            service, "_benchmark_single_run", new_callable=AsyncMock
        ) as mock_benchmark:
            # Mock increasing memory usage
            def mock_benchmark_response(
                algorithm_name, dataset, dataset_size, **kwargs
            ):
                memory_usage = dataset_size * 0.1  # 0.1 MB per sample

                return PerformanceMetrics(
                    algorithm_name=algorithm_name,
                    dataset_size=dataset_size,
                    execution_time_seconds=1.0,
                    peak_memory_mb=memory_usage,
                    success=True,
                )

            mock_benchmark.side_effect = mock_benchmark_response

            results = await service.run_memory_stress_test(
                algorithm_name="TestAlgorithm",
                max_dataset_size=10000,
                memory_limit_mb=500.0,
            )

            assert results["algorithm"] == "TestAlgorithm"
            assert results["memory_limit_mb"] == 500.0
            assert len(results["results"]) > 0
            assert "memory_analysis" in results

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, service):
        """Test throughput benchmarking."""
        with patch.object(
            service, "_measure_throughput", new_callable=AsyncMock
        ) as mock_throughput:
            mock_throughput.return_value = {
                "algorithm": "TestAlgorithm",
                "dataset_size": 1000,
                "duration_seconds": 60.0,
                "samples_processed": 10000,
                "iterations": 10,
                "throughput_samples_per_second": 166.7,
                "throughput_datasets_per_second": 0.167,
            }

            results = await service.run_throughput_benchmark(
                algorithms=["TestAlgorithm"], dataset_sizes=[1000], duration_seconds=60
            )

            assert "results" in results
            assert "TestAlgorithm" in results["results"]
            assert "throughput_analysis" in results

    @pytest.mark.asyncio
    async def test_algorithm_comparison(self, service):
        """Test algorithm comparison functionality."""
        with patch.object(
            service, "_benchmark_single_run", new_callable=AsyncMock
        ) as mock_benchmark:

            def mock_benchmark_response(
                algorithm_name, dataset, dataset_size, **kwargs
            ):
                # Mock different performance for different algorithms
                if algorithm_name == "FastAlgorithm":
                    execution_time = 0.5
                    accuracy = 0.8
                else:
                    execution_time = 1.0
                    accuracy = 0.9

                return PerformanceMetrics(
                    algorithm_name=algorithm_name,
                    dataset_size=dataset_size,
                    execution_time_seconds=execution_time,
                    peak_memory_mb=100.0,
                    accuracy_score=accuracy,
                    training_throughput=dataset_size / execution_time,
                    success=True,
                )

            mock_benchmark.side_effect = mock_benchmark_response

            results = await service.compare_algorithms(
                algorithms=["FastAlgorithm", "AccurateAlgorithm"],
                dataset_sizes=[1000],
                metrics=["execution_time", "accuracy"],
            )

            assert "algorithms" in results
            assert "results" in results
            assert "analysis" in results

            # Check that both algorithms were tested
            assert "FastAlgorithm" in results["results"]
            assert "AccurateAlgorithm" in results["results"]

    @pytest.mark.asyncio
    async def test_performance_trends_empty_history(self, service):
        """Test performance trends with empty history."""
        results = await service.get_performance_trends(
            algorithm_name="TestAlgorithm", days=30
        )

        assert "message" in results
        assert "No historical data available" in results["message"]

    @pytest.mark.asyncio
    async def test_performance_trends_with_data(self, service):
        """Test performance trends with historical data."""
        from datetime import datetime, timedelta

        # Add some historical data
        base_time = datetime.utcnow() - timedelta(days=10)

        for i in range(10):
            metrics = PerformanceMetrics(
                algorithm_name="TestAlgorithm",
                execution_time_seconds=1.0 + i * 0.1,  # Increasing trend
                peak_memory_mb=100.0,
                accuracy_score=0.9 - i * 0.01,  # Decreasing trend
                timestamp=base_time + timedelta(days=i),
            )
            service.performance_history.append(metrics)

        results = await service.get_performance_trends(
            algorithm_name="TestAlgorithm", days=30
        )

        assert "trends" in results
        assert "recommendations" in results
        assert results["data_points"] == 10

        trends = results["trends"]
        assert "execution_time_trend" in trends
        assert "accuracy_trend" in trends

    @pytest.mark.asyncio
    async def test_generate_benchmark_datasets(self, service, sample_config):
        """Test benchmark dataset generation."""
        datasets = await service._generate_benchmark_datasets(sample_config)

        # Should generate datasets for each size/dimension/contamination combination
        expected_count = len(sample_config.dataset_sizes) * len(
            sample_config.feature_dimensions
        ) * len(sample_config.contamination_rates)
        assert len(datasets) == expected_count

        for dataset in datasets:
            assert isinstance(dataset, pd.DataFrame)
            assert "label" in dataset.columns

    @pytest.mark.asyncio
    async def test_estimate_time_complexity(self, service):
        """Test time complexity estimation."""
        # Test linear complexity
        scales = [1, 2, 4, 8]
        times = [1.0, 2.0, 4.0, 8.0]  # Linear scaling
        complexity = service._estimate_time_complexity(scales, times)
        assert complexity == "O(n)"

        # Test quadratic complexity
        times = [1.0, 4.0, 16.0, 64.0]  # Quadratic scaling
        complexity = service._estimate_time_complexity(scales, times)
        assert complexity in ["O(n²)", "O(n³) or worse"]

    @pytest.mark.asyncio
    async def test_calculate_scalability_grade(self, service):
        """Test scalability grade calculation."""
        # Excellent efficiency
        efficiency_ratios = [0.95, 0.92, 0.90]
        grade = service._calculate_scalability_grade(efficiency_ratios)
        assert grade == "A"

        # Poor efficiency
        efficiency_ratios = [0.5, 0.3, 0.2]
        grade = service._calculate_scalability_grade(efficiency_ratios)
        assert grade in ["C", "D"]

    @pytest.mark.asyncio
    async def test_memory_scalability_assessment(self, service):
        """Test memory scalability assessment."""
        # Linear memory growth
        sizes = [1000, 2000, 4000]
        memory = [100, 200, 400]
        scalability = service._assess_memory_scalability(sizes, memory)
        assert scalability == "linear"

        # Exponential memory growth
        memory = [100, 400, 1600]
        scalability = service._assess_memory_scalability(sizes, memory)
        assert scalability == "exponential"

    @pytest.mark.asyncio
    async def test_calculate_trend(self, service):
        """Test trend calculation."""
        # Increasing trend
        values = [1.0, 1.1, 1.2, 1.8, 1.9, 2.0]
        trend = service._calculate_trend(values)
        assert trend["direction"] == "increasing"
        assert trend["change_percent"] > 0

        # Decreasing trend
        values = [2.0, 1.9, 1.8, 1.2, 1.1, 1.0]
        trend = service._calculate_trend(values)
        assert trend["direction"] == "decreasing"
        assert trend["change_percent"] < 0

        # Stable trend
        values = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0]
        trend = service._calculate_trend(values)
        assert trend["direction"] == "stable"

    @pytest.mark.asyncio
    async def test_get_test_environment(self, service):
        """Test test environment information collection."""
        env_info = await service._get_test_environment()

        assert "timestamp" in env_info
        assert "python_version" in env_info
        assert "platform" in env_info
        assert "architecture" in env_info

    @pytest.mark.asyncio
    async def test_get_system_info(self, service):
        """Test system information collection."""
        sys_info = await service._get_system_info()

        assert "cpu_cores" in sys_info
        assert "cpu_logical" in sys_info
        assert "memory_total_gb" in sys_info
        assert "memory_available_gb" in sys_info
        assert "disk_space_gb" in sys_info

        # Validate that values are reasonable
        assert sys_info["cpu_cores"] > 0
        assert sys_info["memory_total_gb"] > 0
        assert sys_info["disk_space_gb"] > 0


class TestSystemMonitor:
    """Test cases for system monitor."""

    @pytest.fixture
    def monitor(self):
        """Create system monitor instance."""
        return SystemMonitor()

    @pytest.mark.asyncio
    async def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert not monitor.monitoring_active
        assert isinstance(monitor.monitoring_data, list)
        assert len(monitor.monitoring_data) == 0

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """Test complete monitoring lifecycle."""
        # Start monitoring
        monitor_id = await monitor.start_monitoring()
        assert isinstance(monitor_id, str)
        assert monitor.monitoring_active

        # Let it collect some data
        await asyncio.sleep(1.5)  # Should collect 2-3 data points

        # Stop monitoring
        summary = await monitor.stop_monitoring(monitor_id)
        assert not monitor.monitoring_active

        # Validate summary
        assert "monitor_id" in summary
        assert "data_points" in summary
        assert "peak_memory_mb" in summary
        assert "avg_memory_mb" in summary
        assert "avg_cpu_percent" in summary
        assert "max_cpu_percent" in summary

        assert summary["monitor_id"] == monitor_id
        assert summary["data_points"] > 0
        assert summary["peak_memory_mb"] >= 0
        assert summary["avg_memory_mb"] >= 0

    @pytest.mark.asyncio
    async def test_monitor_data_clearing(self, monitor):
        """Test that monitoring data is cleared between sessions."""
        # First monitoring session
        monitor_id1 = await monitor.start_monitoring()
        await asyncio.sleep(0.6)
        summary1 = await monitor.stop_monitoring(monitor_id1)

        # Second monitoring session
        monitor_id2 = await monitor.start_monitoring()
        await asyncio.sleep(0.6)
        summary2 = await monitor.stop_monitoring(monitor_id2)

        # IDs should be different
        assert monitor_id1 != monitor_id2

        # Both should have data
        assert summary1["data_points"] > 0
        assert summary2["data_points"] > 0


class TestBenchmarkConfig:
    """Test cases for benchmark configuration."""

    def test_benchmark_config_defaults(self):
        """Test benchmark configuration with defaults."""
        config = BenchmarkConfig()

        assert config.benchmark_name == ""
        assert config.description == ""
        assert len(config.dataset_sizes) == 4
        assert len(config.feature_dimensions) == 4
        assert len(config.contamination_rates) == 4
        assert config.max_execution_time_seconds == 300.0
        assert config.iterations == 5
        assert config.enable_memory_profiling
        assert config.enable_cpu_profiling
        assert config.save_detailed_results

    def test_benchmark_config_custom(self):
        """Test benchmark configuration with custom values."""
        config = BenchmarkConfig(
            benchmark_name="Custom Benchmark",
            dataset_sizes=[100, 200],
            iterations=3,
            timeout_seconds=120.0,
        )

        assert config.benchmark_name == "Custom Benchmark"
        assert config.dataset_sizes == [100, 200]
        assert config.iterations == 3
        assert config.timeout_seconds == 120.0


class TestPerformanceMetrics:
    """Test cases for performance metrics."""

    def test_performance_metrics_defaults(self):
        """Test performance metrics with defaults."""
        metrics = PerformanceMetrics()

        assert metrics.algorithm_name == ""
        assert metrics.dataset_size == 0
        assert metrics.execution_time_seconds == 0.0
        assert metrics.peak_memory_mb == 0.0
        assert metrics.accuracy_score == 0.0
        assert metrics.success
        assert metrics.error_message is None

    def test_performance_metrics_custom(self):
        """Test performance metrics with custom values."""
        metrics = PerformanceMetrics(
            algorithm_name="TestAlgorithm",
            dataset_size=1000,
            execution_time_seconds=2.5,
            peak_memory_mb=256.0,
            accuracy_score=0.85,
            success=True,
        )

        assert metrics.algorithm_name == "TestAlgorithm"
        assert metrics.dataset_size == 1000
        assert metrics.execution_time_seconds == 2.5
        assert metrics.peak_memory_mb == 256.0
        assert metrics.accuracy_score == 0.85
        assert metrics.success

    def test_performance_metrics_failed_run(self):
        """Test performance metrics for failed run."""
        metrics = PerformanceMetrics(
            algorithm_name="TestAlgorithm",
            error_message="Algorithm failed",
            success=False,
        )

        assert not metrics.success
        assert metrics.error_message == "Algorithm failed"


class TestBenchmarkSuite:
    """Test cases for benchmark suite."""

    def test_benchmark_suite_defaults(self):
        """Test benchmark suite with defaults."""
        suite = BenchmarkSuite()

        assert suite.suite_name == ""
        assert suite.description == ""
        assert isinstance(suite.config, BenchmarkConfig)
        assert isinstance(suite.individual_results, list)
        assert isinstance(suite.summary_stats, dict)
        assert isinstance(suite.comparative_analysis, dict)
        assert suite.end_time is None
        assert suite.total_duration_seconds == 0.0
        assert suite.overall_score == 0.0
        assert suite.performance_grade == "B"

    def test_benchmark_suite_custom(self):
        """Test benchmark suite with custom values."""
        config = BenchmarkConfig(benchmark_name="Test Config")

        suite = BenchmarkSuite(
            suite_name="Test Suite", description="Test description", config=config
        )

        assert suite.suite_name == "Test Suite"
        assert suite.description == "Test description"
        assert suite.config.benchmark_name == "Test Config"
