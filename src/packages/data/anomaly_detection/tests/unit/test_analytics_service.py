"""Unit tests for AnalyticsService."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import deque

import numpy as np

from anomaly_detection.domain.services.analytics_service import (
    AnalyticsService,
    PerformanceMetrics,
    AlgorithmStats,
    DataQualityMetrics
)
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.infrastructure.monitoring.metrics_collector import MetricsCollector


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of PerformanceMetrics."""
        metrics = PerformanceMetrics()
        assert metrics.total_detections == 0
        assert metrics.total_anomalies == 0
        assert metrics.average_detection_time == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.throughput == 0.0
        assert metrics.error_rate == 0.0
    
    def test_custom_initialization(self):
        """Test custom initialization of PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_detections=100,
            total_anomalies=15,
            average_detection_time=1.5,
            success_rate=95.0,
            throughput=10.0,
            error_rate=5.0
        )
        assert metrics.total_detections == 100
        assert metrics.total_anomalies == 15
        assert metrics.average_detection_time == 1.5
        assert metrics.success_rate == 95.0
        assert metrics.throughput == 10.0
        assert metrics.error_rate == 5.0


class TestAlgorithmStats:
    """Test suite for AlgorithmStats dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of AlgorithmStats."""
        stats = AlgorithmStats(algorithm="isolation_forest")
        assert stats.algorithm == "isolation_forest"
        assert stats.detections_count == 0
        assert stats.anomalies_found == 0
        assert stats.average_score == 0.0
        assert stats.average_time == 0.0
        assert stats.success_rate == 0.0
        assert stats.last_used is None
    
    def test_custom_initialization(self):
        """Test custom initialization of AlgorithmStats."""
        last_used = datetime.now()
        stats = AlgorithmStats(
            algorithm="lof",
            detections_count=50,
            anomalies_found=10,
            average_score=0.8,
            average_time=2.1,
            success_rate=98.0,
            last_used=last_used
        )
        assert stats.algorithm == "lof"
        assert stats.detections_count == 50
        assert stats.anomalies_found == 10
        assert stats.average_score == 0.8
        assert stats.average_time == 2.1
        assert stats.success_rate == 98.0
        assert stats.last_used == last_used


class TestDataQualityMetrics:
    """Test suite for DataQualityMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of DataQualityMetrics."""
        metrics = DataQualityMetrics()
        assert metrics.total_samples == 0
        assert metrics.missing_values == 0
        assert metrics.duplicate_samples == 0
        assert metrics.outliers_count == 0
        assert metrics.data_drift_events == 0
        assert metrics.quality_score == 0.0
    
    def test_custom_initialization(self):
        """Test custom initialization of DataQualityMetrics."""
        metrics = DataQualityMetrics(
            total_samples=1000,
            missing_values=5,
            duplicate_samples=3,
            outliers_count=15,
            data_drift_events=2,
            quality_score=85.0
        )
        assert metrics.total_samples == 1000
        assert metrics.missing_values == 5
        assert metrics.duplicate_samples == 3
        assert metrics.outliers_count == 15
        assert metrics.data_drift_events == 2
        assert metrics.quality_score == 85.0


class TestAnalyticsService:
    """Test suite for AnalyticsService."""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector fixture."""
        collector = Mock(spec=MetricsCollector)
        collector.get_system_metrics.return_value = {
            'cpu_usage': 45.2,
            'memory_usage': 60.8,
            'disk_usage': 30.1
        }
        return collector
    
    @pytest.fixture
    def analytics_service(self, mock_metrics_collector):
        """Analytics service fixture."""
        return AnalyticsService(metrics_collector=mock_metrics_collector)
    
    @pytest.fixture
    def sample_detection_result(self):
        """Sample detection result fixture."""
        return DetectionResult(
            algorithm="isolation_forest",
            predictions=np.array([1, -1, 1, 1, -1]),
            scores=np.array([0.1, 0.9, 0.2, 0.3, 0.8]),
            total_samples=5,
            anomaly_count=2,
            processing_time=1.5,
            success=True,
            metadata={'contamination': 0.1}
        )
    
    def test_initialization(self, mock_metrics_collector):
        """Test AnalyticsService initialization."""
        service = AnalyticsService(metrics_collector=mock_metrics_collector)
        
        assert service.metrics_collector == mock_metrics_collector
        assert isinstance(service.detection_history, deque)
        assert service.detection_history.maxlen == 10000
        assert isinstance(service.algorithm_stats, dict)
        assert isinstance(service.performance_history, deque)
        assert service.performance_history.maxlen == 1000
        assert isinstance(service.data_quality_history, deque)
        assert service.data_quality_history.maxlen == 1000
        assert service.system_health['status'] == 'healthy'
        assert 'uptime' in service.system_health
        assert 'last_check' in service.system_health
        assert 'issues' in service.system_health
    
    def test_initialization_without_metrics_collector(self):
        """Test AnalyticsService initialization without metrics collector."""
        with patch('anomaly_detection.domain.services.analytics_service.get_metrics_collector') as mock_get:
            mock_collector = Mock()
            mock_get.return_value = mock_collector
            
            service = AnalyticsService()
            assert service.metrics_collector == mock_collector
            mock_get.assert_called_once()
    
    def test_record_detection(self, analytics_service, sample_detection_result):
        """Test recording a detection result."""
        algorithm = "isolation_forest"
        processing_time = 1.5
        data_quality = {'quality_score': 95.0}
        
        analytics_service.record_detection(
            algorithm=algorithm,
            result=sample_detection_result,
            processing_time=processing_time,
            data_quality=data_quality
        )
        
        # Check detection history
        assert len(analytics_service.detection_history) == 1
        record = analytics_service.detection_history[-1]
        assert record['algorithm'] == algorithm
        assert record['total_samples'] == sample_detection_result.total_samples
        assert record['anomalies_found'] == sample_detection_result.anomaly_count
        assert record['processing_time'] == processing_time
        assert record['success'] == sample_detection_result.success
        assert record['data_quality'] == data_quality
        assert isinstance(record['timestamp'], datetime)
        
        # Check algorithm stats
        assert algorithm in analytics_service.algorithm_stats
        stats = analytics_service.algorithm_stats[algorithm]
        assert stats.algorithm == algorithm
        assert stats.detections_count == 1
        assert stats.anomalies_found == sample_detection_result.anomaly_count
        assert isinstance(stats.last_used, datetime)
    
    def test_record_detection_without_data_quality(self, analytics_service, sample_detection_result):
        """Test recording a detection result without data quality."""
        algorithm = "lof"
        processing_time = 2.0
        
        analytics_service.record_detection(
            algorithm=algorithm,
            result=sample_detection_result,
            processing_time=processing_time
        )
        
        record = analytics_service.detection_history[-1]
        assert record['data_quality'] == {}
    
    def test_get_performance_metrics(self, analytics_service, sample_detection_result):
        """Test getting performance metrics."""
        # Record some detections
        algorithms = ["isolation_forest", "lof", "one_class_svm"]
        for i, algorithm in enumerate(algorithms):
            result = DetectionResult(
                algorithm=algorithm,
                predictions=np.array([1, -1]),
                scores=np.array([0.1, 0.9]),
                total_samples=2,
                anomaly_count=1,
                processing_time=1.0 + i * 0.5,
                success=i < 2,  # First two succeed, third fails
                metadata={}
            )
            analytics_service.record_detection(algorithm, result, 1.0 + i * 0.5)
        
        metrics = analytics_service.get_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_detections == 3
        assert metrics.total_anomalies == 2  # Only successful detections count
        assert metrics.success_rate == 66.67  # 2/3 * 100
        assert metrics.error_rate == 33.33   # 1/3 * 100
        assert metrics.average_detection_time > 0
        assert metrics.throughput >= 0
    
    def test_get_performance_metrics_empty(self, analytics_service):
        """Test getting performance metrics with no data."""
        metrics = analytics_service.get_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_detections == 0
        assert metrics.total_anomalies == 0
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.average_detection_time == 0.0
        assert metrics.throughput == 0.0
    
    def test_get_algorithm_performance(self, analytics_service, sample_detection_result):
        """Test getting algorithm performance statistics."""
        # Record detections for multiple algorithms
        algorithms = ["isolation_forest", "lof"]
        for algorithm in algorithms:
            for i in range(3):
                result = DetectionResult(
                    algorithm=algorithm,
                    predictions=np.array([1, -1]),
                    scores=np.array([0.1, 0.9]),
                    total_samples=2,
                    anomaly_count=1,
                    processing_time=1.0,
                    success=True,
                    metadata={}
                )
                analytics_service.record_detection(algorithm, result, 1.0)
        
        performance = analytics_service.get_algorithm_performance()
        
        assert isinstance(performance, list)
        assert len(performance) == 2
        
        for alg_perf in performance:
            assert isinstance(alg_perf, AlgorithmStats)
            assert alg_perf.algorithm in algorithms
            assert alg_perf.detections_count == 3
            assert alg_perf.anomalies_found == 3
            assert alg_perf.success_rate == 100.0
            assert alg_perf.average_time == 1.0
            assert isinstance(alg_perf.last_used, datetime)
    
    def test_get_algorithm_performance_empty(self, analytics_service):
        """Test getting algorithm performance with no data."""
        performance = analytics_service.get_algorithm_performance()
        
        assert isinstance(performance, list)
        assert len(performance) == 0
    
    def test_get_data_quality_metrics(self, analytics_service):
        """Test getting data quality metrics."""
        # Mock some data quality history
        analytics_service.data_quality_history.extend([
            {'quality_score': 95.0, 'missing_values': 5, 'duplicates': 2},
            {'quality_score': 87.0, 'missing_values': 8, 'duplicates': 1},
            {'quality_score': 92.0, 'missing_values': 3, 'duplicates': 0}
        ])
        
        metrics = analytics_service.get_data_quality_metrics()
        
        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.quality_score == 91.33  # Average of 95, 87, 92
        # Other fields depend on implementation details
    
    def test_get_data_quality_metrics_empty(self, analytics_service):
        """Test getting data quality metrics with no data."""
        metrics = analytics_service.get_data_quality_metrics()
        
        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.quality_score == 0.0
    
    def test_get_detection_timeline(self, analytics_service, sample_detection_result):
        """Test getting detection timeline data."""
        # Record detections across different times
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(5):
            with patch('anomaly_detection.domain.services.analytics_service.datetime') as mock_dt:
                mock_dt.now.return_value = base_time + timedelta(hours=i * 6)
                analytics_service.record_detection("isolation_forest", sample_detection_result, 1.0)
        
        timeline = analytics_service.get_detection_timeline(hours=24)
        
        assert isinstance(timeline, list)
        assert len(timeline) > 0
        
        for point in timeline:
            assert 'timestamp' in point
            assert 'detections' in point
            assert 'anomalies' in point
    
    def test_get_detection_timeline_empty(self, analytics_service):
        """Test getting detection timeline with no data."""
        timeline = analytics_service.get_detection_timeline(hours=24)
        
        assert isinstance(timeline, list)
        # Should return empty list or default time points
    
    def test_get_algorithm_distribution(self, analytics_service, sample_detection_result):
        """Test getting algorithm distribution data."""
        algorithms = ["isolation_forest", "lof", "one_class_svm"]
        counts = [5, 3, 2]
        
        for algorithm, count in zip(algorithms, counts):
            for _ in range(count):
                analytics_service.record_detection(algorithm, sample_detection_result, 1.0)
        
        distribution = analytics_service.get_algorithm_distribution()
        
        assert isinstance(distribution, list)
        assert len(distribution) == 3
        
        for item in distribution:
            assert 'algorithm' in item
            assert 'count' in item
            assert 'percentage' in item
            assert item['algorithm'] in algorithms
        
        # Check that percentages sum to ~100
        total_percentage = sum(item['percentage'] for item in distribution)
        assert abs(total_percentage - 100.0) < 0.01
    
    def test_get_algorithm_distribution_empty(self, analytics_service):
        """Test getting algorithm distribution with no data."""
        distribution = analytics_service.get_algorithm_distribution()
        
        assert isinstance(distribution, list)
        assert len(distribution) == 0
    
    def test_get_system_status(self, analytics_service, mock_metrics_collector):
        """Test getting system status."""
        system_metrics = {
            'cpu_usage': 45.2,
            'memory_usage': 60.8,
            'disk_usage': 30.1
        }
        mock_metrics_collector.get_system_metrics.return_value = system_metrics
        
        status = analytics_service.get_system_status()
        
        assert isinstance(status, dict)
        assert 'overall_status' in status
        assert 'api_status' in status
        assert 'database_status' in status
        assert 'memory_usage' in status
        assert 'cpu_usage' in status
        assert 'disk_usage' in status
        assert 'active_operations' in status
        assert 'success_rate' in status
        assert 'last_check' in status
        
        mock_metrics_collector.get_system_metrics.assert_called_once()
    
    def test_get_dashboard_stats(self, analytics_service, sample_detection_result):
        """Test getting dashboard statistics."""
        # Record some data
        for i in range(10):
            result = DetectionResult(
                algorithm="isolation_forest",
                predictions=np.array([1, -1]),
                scores=np.array([0.1, 0.9]),
                total_samples=100,
                anomaly_count=i % 3,  # Varying anomaly counts
                processing_time=1.0,
                success=True,
                metadata={}
            )
            analytics_service.record_detection("isolation_forest", result, 1.0)
        
        stats = analytics_service.get_dashboard_stats()
        
        assert isinstance(stats, dict)
        assert 'total_detections' in stats
        assert 'total_anomalies' in stats
        assert 'active_algorithms' in stats
        assert 'average_detection_time' in stats
        assert 'system_status' in stats
        assert 'success_rate' in stats
        
        assert stats['total_detections'] == 10
        assert stats['total_anomalies'] > 0
        assert stats['active_algorithms'] >= 1
        assert stats['average_detection_time'] > 0
    
    def test_get_dashboard_stats_empty(self, analytics_service):
        """Test getting dashboard statistics with no data."""
        stats = analytics_service.get_dashboard_stats()
        
        assert isinstance(stats, dict)
        assert stats['total_detections'] == 0
        assert stats['total_anomalies'] == 0
        assert stats['active_algorithms'] == 0
        assert stats['average_detection_time'] == 0.0
    
    def test_simulate_detection(self, analytics_service):
        """Test simulating a detection."""
        result = analytics_service.simulate_detection()
        
        assert isinstance(result, dict)
        assert 'algorithm' in result
        assert 'total_samples' in result
        assert 'anomalies_found' in result
        assert 'anomaly_rate' in result
        assert 'processing_time' in result
        
        assert result['total_samples'] > 0
        assert result['anomalies_found'] >= 0
        assert 0 <= result['anomaly_rate'] <= 100
        assert result['processing_time'] > 0
    
    def test_get_performance_trend(self, analytics_service, sample_detection_result):
        """Test getting performance trend data."""
        # Record performance data
        for i in range(5):
            analytics_service.performance_history.append({
                'timestamp': datetime.now() - timedelta(hours=i),
                'processing_time': 1.0 + i * 0.1,
                'throughput': 10.0 - i,
                'success_rate': 95.0 - i * 2
            })
        
        trend = analytics_service.get_performance_trend(hours=24)
        
        assert isinstance(trend, list)
        assert len(trend) > 0
        
        for point in trend:
            assert 'timestamp' in point
            assert 'processing_time' in point
            assert 'throughput' in point
            assert 'success_rate' in point
    
    def test_get_performance_trend_empty(self, analytics_service):
        """Test getting performance trend with no data."""
        trend = analytics_service.get_performance_trend(hours=24)
        
        assert isinstance(trend, list)
        # Should return empty list or default values
    
    def test_health_check_integration(self, analytics_service, mock_metrics_collector):
        """Test integration with health monitoring."""
        # This tests that the analytics service properly integrates with system health
        mock_metrics_collector.get_system_metrics.return_value = {
            'cpu_usage': 85.0,  # High CPU usage
            'memory_usage': 90.0,  # High memory usage
            'disk_usage': 95.0   # High disk usage
        }
        
        status = analytics_service.get_system_status()
        
        # The system should detect high resource usage
        assert status['cpu_usage'] == '85.0%'
        assert status['memory_usage'] == '90.0%'
        assert status['disk_usage'] == '95.0%'
    
    def test_concurrent_access_safety(self, analytics_service, sample_detection_result):
        """Test thread safety of analytics service operations."""
        import threading
        import time
        
        def record_detections():
            for i in range(10):
                analytics_service.record_detection(f"algo_{i%3}", sample_detection_result, 1.0)
                time.sleep(0.001)  # Small delay to increase chance of race conditions
        
        # Start multiple threads
        threads = [threading.Thread(target=record_detections) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Verify data integrity
        assert len(analytics_service.detection_history) == 30
        performance = analytics_service.get_performance_metrics()
        assert performance.total_detections == 30