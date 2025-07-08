"""Tests for performance testing service."""

import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from pynomaly.application.services.performance_testing_service import (
    BenchmarkResult,
    BenchmarkSuite,
    PerformanceMetrics,
    PerformanceTestingService,
    StressTestConfig,
    SystemMonitor,
)
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector


class TestPerformanceTestingService:
    """Test cases for performance testing service."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create performance testing service instance."""
        return PerformanceTestingService(
            storage_path=tmp_path, cache_results=True, enable_profiling=True
        )

    @pytest.fixture
    def mock_detector(self):
        """Create mock detector for testing."""
        detector = Mock(spec=Detector)
        detector.fit = Mock()
        detector.predict = Mock(return_value=np.array([1, -1, 1, -1, 1]))
        detector.decision_function = Mock(
            return_value=np.array([0.1, -0.5, 0.3, -0.2, 0.4])
        )
        return detector

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )
        labels = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

        return Dataset(
            name="test_dataset",
            features=features,
            labels=labels,
            metadata={"contamination_rate": 0.1},
        )

    def test_service_initialization(self, service, tmp_path):
        """Test service initialization."""
        assert service.storage_path == tmp_path
        assert service.storage_path.exists()
        assert service.cache_results is True
        assert service.enable_profiling is True
        assert isinstance(service.benchmark_results, dict)
        assert isinstance(service.performance_history, list)
        assert isinstance(service.benchmark_suites, dict)

        # Check default suites
        assert "quick" in service.benchmark_suites
        assert "comprehensive" in service.benchmark_suites
        assert "scalability" in service.benchmark_suites

    def test_default_benchmark_suites(self, service):
        """Test default benchmark suite configuration."""
        quick_suite = service.benchmark_suites["quick"]
        assert quick_suite.name == "Quick Performance"
        assert len(quick_suite.algorithms) >= 3
        assert quick_suite.iterations == 3
        assert quick_suite.timeout_seconds == 60

        comprehensive_suite = service.benchmark_suites["comprehensive"]
        assert comprehensive_suite.name == "Comprehensive Benchmark"
        assert len(comprehensive_suite.algorithms) >= 3
        assert comprehensive_suite.iterations == 5
        assert comprehensive_suite.timeout_seconds == 300

        scalability_suite = service.benchmark_suites["scalability"]
        assert scalability_suite.name == "Scalability Testing"
        assert len(scalability_suite.scalability_sizes) > 0
        assert len(scalability_suite.scalability_features) > 0

    @pytest.mark.asyncio
    async def test_run_single_benchmark(self, service, mock_detector, sample_dataset):
        """Test running single algorithm benchmark."""
        result = await service._run_single_benchmark(
            detector=mock_detector,
            dataset=sample_dataset,
            algorithm_name="TestAlgorithm",
            timeout=60,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.algorithm_name == "TestAlgorithm"
        assert result.dataset_name == "test_dataset"
        assert result.metrics.training_time >= 0
        assert result.metrics.prediction_time >= 0
        assert result.metrics.total_time >= 0
        assert result.metrics.throughput > 0
        assert result.metrics.dataset_size == 100
        assert result.metrics.feature_count == 3

        # Verify detector was called
        mock_detector.fit.assert_called_once()
        mock_detector.predict.assert_called_once()
        mock_detector.decision_function.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_single_benchmark_with_labels(
        self, service, mock_detector, sample_dataset
    ):
        """Test benchmark with quality metrics calculation."""
        result = await service._run_single_benchmark(
            detector=mock_detector,
            dataset=sample_dataset,
            algorithm_name="TestAlgorithm",
        )

        # Should have calculated quality metrics
        assert result.metrics.roc_auc >= 0
        assert result.metrics.f1_score >= 0
        assert result.metrics.precision >= 0
        assert result.metrics.recall >= 0

    @pytest.mark.asyncio
    async def test_generate_synthetic_dataset(self, service):
        """Test synthetic dataset generation."""
        dataset = await service._generate_synthetic_dataset(
            n_samples=1000, n_features=5, contamination=0.1
        )

        assert isinstance(dataset, Dataset)
        assert dataset.name.startswith("synthetic_")
        assert len(dataset.features) == 1000
        assert dataset.features.shape[1] == 5
        assert len(dataset.labels) == 1000

        # Check contamination rate
        anomaly_rate = np.mean(dataset.labels)
        assert abs(anomaly_rate - 0.1) < 0.02  # Allow small variance

    @pytest.mark.asyncio
    async def test_generate_test_datasets(self, service):
        """Test test dataset generation from configurations."""
        configs = [
            {"type": "synthetic", "samples": 500, "features": 3, "contamination": 0.05},
            {"type": "synthetic", "samples": 1000, "features": 5, "contamination": 0.1},
        ]

        datasets = await service._generate_test_datasets(configs)

        assert len(datasets) == 2
        assert all(isinstance(d, Dataset) for d in datasets)
        assert datasets[0].features.shape == (500, 3)
        assert datasets[1].features.shape == (1000, 5)

    def test_aggregate_benchmark_results(self, service):
        """Test benchmark result aggregation."""
        # Create sample results
        results = []
        for i in range(3):
            result = BenchmarkResult(algorithm_name="TestAlg", dataset_name="TestData")
            result.metrics.training_time = 1.0 + i * 0.1
            result.metrics.roc_auc = 0.8 + i * 0.01
            result.metrics.f1_score = 0.7 + i * 0.01
            results.append(result)

        aggregated = service._aggregate_benchmark_results(
            results, confidence_level=0.95
        )

        assert isinstance(aggregated, BenchmarkResult)
        assert aggregated.algorithm_name == "TestAlg"
        assert aggregated.dataset_name == "TestData"

        # Check aggregated metrics
        assert 1.0 <= aggregated.metrics.training_time <= 1.2
        assert 0.8 <= aggregated.metrics.roc_auc <= 0.82
        assert aggregated.statistical_significance is True
        assert len(aggregated.confidence_interval) == 2

    def test_aggregate_single_result(self, service):
        """Test aggregation with single result."""
        result = BenchmarkResult(algorithm_name="TestAlg", dataset_name="TestData")
        result.metrics.training_time = 1.5

        aggregated = service._aggregate_benchmark_results([result])

        assert aggregated.algorithm_name == "TestAlg"
        assert aggregated.metrics.training_time == 1.5
        assert aggregated.statistical_significance is False

    def test_aggregate_empty_results(self, service):
        """Test aggregation with empty results list."""
        with pytest.raises(ValueError, match="No results to aggregate"):
            service._aggregate_benchmark_results([])

    @pytest.mark.asyncio
    async def test_run_benchmark_suite(self, service):
        """Test running complete benchmark suite."""
        # Mock detectors
        detectors = {"MockAlg1": Mock(spec=Detector), "MockAlg2": Mock(spec=Detector)}

        for detector in detectors.values():
            detector.fit = Mock()
            detector.predict = Mock(return_value=np.array([1, -1, 1]))
            detector.decision_function = Mock(return_value=np.array([0.1, -0.5, 0.3]))

        # Mock dataset generation
        with patch.object(service, "_generate_test_datasets") as mock_gen:
            mock_gen.return_value = [Mock(spec=Dataset, name="test_dataset")]

            with patch.object(service, "_run_single_benchmark") as mock_bench:
                mock_result = BenchmarkResult(
                    algorithm_name="MockAlg", dataset_name="test"
                )
                mock_bench.return_value = mock_result

                results = await service.run_benchmark_suite(
                    suite_name="quick", detectors=detectors
                )

        assert "suite_id" in results
        assert "suite_name" in results
        assert "results" in results
        assert "summary" in results
        assert results["suite_name"] == "Quick Performance"

    @pytest.mark.asyncio
    async def test_run_scalability_analysis(self, service, mock_detector):
        """Test scalability analysis."""
        with patch.object(service, "_generate_synthetic_dataset") as mock_gen:
            mock_gen.return_value = Mock(spec=Dataset, name="test_dataset")

            with patch.object(service, "_run_single_benchmark") as mock_bench:
                mock_result = BenchmarkResult(
                    algorithm_name="TestAlg", dataset_name="test"
                )
                mock_result.metrics.training_time = 1.0
                mock_result.metrics.peak_memory_mb = 50.0
                mock_result.metrics.throughput = 100.0
                mock_bench.return_value = mock_result

                results = await service.run_scalability_analysis(
                    detector=mock_detector,
                    algorithm_name="TestAlgorithm",
                    size_range=(1000, 10000),
                    feature_range=(10, 50),
                    steps=3,
                )

        assert "algorithm" in results
        assert "analysis_id" in results
        assert "size_scaling" in results
        assert "feature_scaling" in results
        assert "complexity_analysis" in results
        assert "recommendations" in results

        assert results["algorithm"] == "TestAlgorithm"
        assert len(results["size_scaling"]) == 3
        assert len(results["feature_scaling"]) == 3

    @pytest.mark.asyncio
    async def test_run_stress_test(self, service, mock_detector):
        """Test stress testing."""
        config = StressTestConfig(
            concurrent_requests=5,
            request_duration=30,
            memory_pressure_mb=100,
            cpu_intensive_operations=500,
            endurance_duration_hours=0,
        )

        results = await service.run_stress_test(
            detector=mock_detector, algorithm_name="TestAlgorithm", config=config
        )

        assert "test_id" in results
        assert "algorithm" in results
        assert "configuration" in results
        assert "load_test" in results
        assert "memory_stress" in results
        assert "cpu_stress" in results
        assert "endurance_test" in results
        assert "overall_stability" in results

        assert results["algorithm"] == "TestAlgorithm"
        assert isinstance(results["overall_stability"], float)

    @pytest.mark.asyncio
    async def test_compare_algorithms(self, service):
        """Test algorithm comparison."""
        # Mock detectors
        detectors = {"Alg1": Mock(spec=Detector), "Alg2": Mock(spec=Detector)}

        for detector in detectors.values():
            detector.fit = Mock()
            detector.predict = Mock(return_value=np.array([1, -1]))
            detector.decision_function = Mock(return_value=np.array([0.1, -0.5]))

        # Mock datasets
        datasets = [
            Mock(spec=Dataset, name="dataset1"),
            Mock(spec=Dataset, name="dataset2"),
        ]

        with patch.object(service, "_run_single_benchmark") as mock_bench:
            mock_result = BenchmarkResult(algorithm_name="TestAlg", dataset_name="test")
            mock_result.metrics.roc_auc = 0.8
            mock_result.metrics.training_time = 1.0
            mock_result.metrics.peak_memory_mb = 50.0
            mock_result.metrics.throughput = 100.0
            mock_bench.return_value = mock_result

            results = await service.compare_algorithms(
                detectors=detectors,
                datasets=datasets,
                metrics=["roc_auc", "training_time"],
            )

        assert "comparison_id" in results
        assert "algorithms" in results
        assert "datasets" in results
        assert "results" in results
        assert "rankings" in results
        assert "statistical_analysis" in results

        assert len(results["algorithms"]) == 2
        assert len(results["datasets"]) == 2

    def test_analyze_algorithmic_complexity(self, service):
        """Test algorithmic complexity analysis."""
        size_scaling = [
            {"size": 1000, "training_time": 1.0},
            {"size": 5000, "training_time": 2.5},
            {"size": 10000, "training_time": 5.2},
        ]

        feature_scaling = [
            {"features": 10, "training_time": 1.0},
            {"features": 50, "training_time": 1.8},
            {"features": 100, "training_time": 3.2},
        ]

        analysis = service._analyze_algorithmic_complexity(
            size_scaling, feature_scaling
        )

        assert "time_complexity" in analysis
        assert "space_complexity" in analysis
        assert "scalability_rating" in analysis

        assert isinstance(analysis["time_complexity"], str)
        assert isinstance(analysis["scalability_rating"], str)

    def test_generate_scalability_recommendations(self, service):
        """Test scalability recommendation generation."""
        complexity_analysis = {
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "scalability_rating": "good",
        }

        recommendations = service._generate_scalability_recommendations(
            complexity_analysis
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_calculate_overall_stability(self, service):
        """Test overall stability calculation."""
        stress_results = {
            "load_test": {"success_rate": 0.95},
            "memory_stress": {"memory_leaks_detected": False},
            "cpu_stress": {"performance_degradation": 0.1},
        }

        stability = service._calculate_overall_stability(stress_results)

        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    @pytest.mark.asyncio
    async def test_save_benchmark_results(self, service, tmp_path):
        """Test saving benchmark results."""
        results = {
            "suite_name": "test_suite",
            "results": [{"algorithm": "TestAlg", "score": 0.8}],
        }

        await service._save_benchmark_results("test_suite", results)

        # Check if file was created
        result_files = list(tmp_path.glob("benchmark_test_suite_*.json"))
        assert len(result_files) == 1

        # Check file content
        import json

        with open(result_files[0]) as f:
            saved_results = json.load(f)

        assert saved_results["suite_name"] == "test_suite"
        assert len(saved_results["results"]) == 1


class TestSystemMonitor:
    """Test cases for system monitor."""

    def test_system_monitor_initialization(self):
        """Test system monitor initialization."""
        monitor = SystemMonitor()

        assert monitor.monitoring is False
        assert isinstance(monitor.stats, list)
        assert len(monitor.stats) == 0

    @pytest.mark.asyncio
    async def test_start_and_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = SystemMonitor()

        # Start monitoring task
        monitoring_task = asyncio.create_task(monitor.start_monitoring())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop monitoring
        monitor.stop_monitoring()

        # Wait for task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except TimeoutError:
            monitoring_task.cancel()

        assert monitor.monitoring is False

    def test_get_summary_empty(self):
        """Test getting summary with no stats."""
        monitor = SystemMonitor()
        summary = monitor.get_summary()

        assert summary == {}

    def test_get_summary_with_stats(self):
        """Test getting summary with stats."""
        monitor = SystemMonitor()

        # Add some fake stats
        monitor.stats = [
            {"cpu_percent": 50.0, "memory_percent": 60.0},
            {"cpu_percent": 55.0, "memory_percent": 65.0},
            {"cpu_percent": 45.0, "memory_percent": 55.0},
        ]

        summary = monitor.get_summary()

        assert "duration_seconds" in summary
        assert "avg_cpu_percent" in summary
        assert "max_cpu_percent" in summary
        assert "avg_memory_percent" in summary
        assert "max_memory_percent" in summary
        assert "samples_collected" in summary

        assert summary["duration_seconds"] == 3
        assert summary["avg_cpu_percent"] == 50.0
        assert summary["max_cpu_percent"] == 55.0
        assert summary["avg_memory_percent"] == 60.0
        assert summary["max_memory_percent"] == 65.0
        assert summary["samples_collected"] == 3


class TestPerformanceMetrics:
    """Test cases for performance metrics."""

    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization."""
        metrics = PerformanceMetrics()

        assert metrics.training_time == 0.0
        assert metrics.prediction_time == 0.0
        assert metrics.total_time == 0.0
        assert metrics.throughput == 0.0
        assert metrics.peak_memory_mb == 0.0
        assert metrics.cpu_percent == 0.0
        assert metrics.roc_auc == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.dataset_size == 0
        assert metrics.feature_count == 0


class TestBenchmarkResult:
    """Test cases for benchmark result."""

    def test_benchmark_result_initialization(self):
        """Test benchmark result initialization."""
        result = BenchmarkResult(
            algorithm_name="TestAlgorithm", dataset_name="TestDataset"
        )

        assert result.algorithm_name == "TestAlgorithm"
        assert result.dataset_name == "TestDataset"
        assert isinstance(result.benchmark_id, type(result.benchmark_id))
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.metrics, PerformanceMetrics)
        assert isinstance(result.parameters, dict)
        assert isinstance(result.system_info, dict)

    def test_get_summary(self):
        """Test benchmark result summary generation."""
        result = BenchmarkResult(
            algorithm_name="TestAlgorithm", dataset_name="TestDataset"
        )
        result.metrics.training_time = 1.5
        result.metrics.roc_auc = 0.85
        result.metrics.peak_memory_mb = 100.0

        summary = result.get_summary()

        assert "benchmark_id" in summary
        assert "algorithm" in summary
        assert "dataset" in summary
        assert "performance" in summary
        assert "efficiency" in summary

        assert summary["algorithm"] == "TestAlgorithm"
        assert summary["dataset"] == "TestDataset"
        assert summary["performance"]["training_time"] == 1.5
        assert summary["performance"]["roc_auc"] == 0.85


class TestBenchmarkSuite:
    """Test cases for benchmark suite."""

    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        suite = BenchmarkSuite(name="Test Suite", description="Test benchmark suite")

        assert suite.name == "Test Suite"
        assert suite.description == "Test benchmark suite"
        assert isinstance(suite.suite_id, type(suite.suite_id))
        assert isinstance(suite.created_at, datetime)
        assert isinstance(suite.algorithms, list)
        assert isinstance(suite.datasets, list)
        assert suite.iterations == 5
        assert suite.timeout_seconds == 300
        assert suite.confidence_level == 0.95


class TestStressTestConfig:
    """Test cases for stress test configuration."""

    def test_stress_test_config_initialization(self):
        """Test stress test configuration initialization."""
        config = StressTestConfig(
            concurrent_requests=20, request_duration=120, memory_pressure_mb=1000
        )

        assert config.concurrent_requests == 20
        assert config.request_duration == 120
        assert config.memory_pressure_mb == 1000
        assert config.ramp_up_time == 30
        assert config.parallel_workers == 4
        assert config.endurance_duration_hours == 2
        assert config.periodic_gc is True


class TestPerformanceTestingIntegration:
    """Integration tests for performance testing service."""

    @pytest.mark.asyncio
    async def test_end_to_end_benchmark(self, tmp_path):
        """Test end-to-end benchmark workflow."""
        service = PerformanceTestingService(storage_path=tmp_path)

        # Create mock detector
        detector = Mock(spec=Detector)
        detector.fit = Mock()
        detector.predict = Mock(return_value=np.array([1, -1, 1, -1, 1]))
        detector.decision_function = Mock(
            return_value=np.array([0.1, -0.5, 0.3, -0.2, 0.4])
        )

        # Create test dataset
        dataset = await service._generate_synthetic_dataset(
            n_samples=100, n_features=5, contamination=0.1
        )

        # Run single benchmark
        result = await service._run_single_benchmark(
            detector=detector, dataset=dataset, algorithm_name="TestAlgorithm"
        )

        # Verify result
        assert isinstance(result, BenchmarkResult)
        assert result.metrics.dataset_size == 100
        assert result.metrics.feature_count == 5
        assert result.metrics.total_time > 0

        # Verify detector was used
        detector.fit.assert_called_once()
        detector.predict.assert_called_once()
        detector.decision_function.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_comparison_workflow(self, tmp_path):
        """Test algorithm comparison workflow."""
        service = PerformanceTestingService(storage_path=tmp_path)

        # Create mock detectors
        detectors = {}
        for i, alg_name in enumerate(["Alg1", "Alg2"]):
            detector = Mock(spec=Detector)
            detector.fit = Mock()
            detector.predict = Mock(return_value=np.array([1, -1, 1]))
            detector.decision_function = Mock(
                return_value=np.array([0.1 + i * 0.1, -0.5, 0.3])
            )
            detectors[alg_name] = detector

        # Create test datasets
        datasets = []
        for size in [50, 100]:
            dataset = await service._generate_synthetic_dataset(
                n_samples=size, n_features=3, contamination=0.1
            )
            datasets.append(dataset)

        # Run comparison
        results = await service.compare_algorithms(
            detectors=detectors, datasets=datasets
        )

        # Verify results structure
        assert "comparison_id" in results
        assert "algorithms" in results
        assert "datasets" in results
        assert "results" in results

        assert set(results["algorithms"]) == {"Alg1", "Alg2"}
        assert len(results["results"]) == 2  # 2 algorithms

        # Verify all detectors were used
        for detector in detectors.values():
            assert detector.fit.call_count == 2  # Called for each dataset
            assert detector.predict.call_count == 2
            assert detector.decision_function.call_count == 2
