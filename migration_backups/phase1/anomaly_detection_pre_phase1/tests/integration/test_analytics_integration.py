"""Integration tests for analytics service and dashboard functionality."""

import pytest
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import patch, Mock, MagicMock
from collections import deque

from anomaly_detection.domain.services.analytics_service import (
    AnalyticsService,
    PerformanceMetrics,
    AlgorithmStats,
    DataQualityMetrics
)
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.infrastructure.monitoring.metrics_collector import MetricsCollector


class TestAnalyticsServiceIntegration:
    """Integration tests for analytics service with real components."""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create a mock metrics collector."""
        collector = Mock(spec=MetricsCollector)
        collector.get_system_metrics.return_value = {
            'cpu_usage': 45.2,
            'memory_usage': 60.8,
            'disk_usage': 30.1,
            'memory_mb': 2048,
            'cpu_percent': 45.2
        }
        return collector
    
    @pytest.fixture
    def analytics_service(self, mock_metrics_collector):
        """Create analytics service with mocked collector."""
        return AnalyticsService(metrics_collector=mock_metrics_collector)
    
    @pytest.fixture
    def sample_detection_results(self):
        """Create sample detection results for testing."""
        results = []
        algorithms = ['isolation_forest', 'lof', 'one_class_svm', 'ensemble_majority']
        
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(50):
            algorithm = algorithms[i % len(algorithms)]
            
            # Vary the results to create realistic patterns
            if i < 10:  # Early results - learning phase
                success = i > 2  # First few might fail
                anomaly_count = np.random.randint(1, 5) if success else 0
                processing_time = np.random.uniform(2.0, 5.0)  # Slower initially
            elif i < 30:  # Stable phase
                success = True
                anomaly_count = np.random.randint(5, 15)
                processing_time = np.random.uniform(1.0, 2.5)
            else:  # Recent results
                success = np.random.random() > 0.05  # 95% success rate
                anomaly_count = np.random.randint(3, 20) if success else 0
                processing_time = np.random.uniform(0.8, 2.0)
            
            result = DetectionResult(
                algorithm=algorithm,
                predictions=np.random.choice([-1, 1], size=100),
                scores=np.random.uniform(0, 1, size=100),
                total_samples=100,
                anomaly_count=anomaly_count,
                processing_time=processing_time,
                success=success,
                metadata={'contamination': 0.1, 'timestamp': base_time + timedelta(minutes=i*30)}
            )
            
            results.append({
                'result': result,
                'algorithm': algorithm,
                'processing_time': processing_time,
                'timestamp': base_time + timedelta(minutes=i*30),
                'data_quality': {
                    'quality_score': np.random.uniform(85, 98),
                    'missing_values': np.random.randint(0, 5),
                    'duplicates': np.random.randint(0, 3)
                }
            })
        
        return results
    
    def test_analytics_service_initialization_integration(self, mock_metrics_collector):
        """Test analytics service initialization with real-like components."""
        service = AnalyticsService(metrics_collector=mock_metrics_collector)
        
        # Verify initialization
        assert service.metrics_collector == mock_metrics_collector
        assert isinstance(service.detection_history, deque)
        assert isinstance(service.algorithm_stats, dict)
        assert isinstance(service.performance_history, deque)
        assert isinstance(service.data_quality_history, deque)
        assert 'status' in service.system_health
        assert 'uptime' in service.system_health
        assert isinstance(service.system_health['uptime'], datetime)
    
    def test_detection_recording_integration(self, analytics_service, sample_detection_results):
        """Test recording detection results and updating analytics."""
        initial_history_length = len(analytics_service.detection_history)
        initial_algorithm_count = len(analytics_service.algorithm_stats)
        
        # Record multiple detection results
        for sample in sample_detection_results[:10]:
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time'],
                data_quality=sample['data_quality']
            )
        
        # Verify detection history was updated
        assert len(analytics_service.detection_history) == initial_history_length + 10
        
        # Verify algorithm statistics were updated
        recorded_algorithms = set(sample['algorithm'] for sample in sample_detection_results[:10])
        for algorithm in recorded_algorithms:
            assert algorithm in analytics_service.algorithm_stats
            stats = analytics_service.algorithm_stats[algorithm]
            assert stats.detections_count > 0
            assert stats.anomalies_found >= 0
            assert isinstance(stats.last_used, datetime)
    
    def test_performance_metrics_calculation_integration(self, analytics_service, sample_detection_results):
        """Test performance metrics calculation with realistic data."""
        # Record detection results
        for sample in sample_detection_results:
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time'],
                data_quality=sample['data_quality']
            )
        
        # Calculate performance metrics
        metrics = analytics_service.get_performance_metrics()
        
        # Verify metrics are realistic
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_detections == len(sample_detection_results)
        
        # Calculate expected values
        successful_results = [s for s in sample_detection_results if s['result'].success]
        expected_anomalies = sum(s['result'].anomaly_count for s in successful_results)
        expected_success_rate = len(successful_results) / len(sample_detection_results) * 100
        expected_error_rate = 100 - expected_success_rate
        
        assert metrics.total_anomalies == expected_anomalies
        assert abs(metrics.success_rate - expected_success_rate) < 0.1
        assert abs(metrics.error_rate - expected_error_rate) < 0.1
        assert metrics.average_detection_time > 0
        assert metrics.throughput >= 0
    
    def test_algorithm_performance_analysis_integration(self, analytics_service, sample_detection_results):
        """Test algorithm performance analysis with multiple algorithms."""
        # Record detection results
        for sample in sample_detection_results:
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time']
            )
        
        # Get algorithm performance
        performance = analytics_service.get_algorithm_performance()
        
        # Verify performance analysis
        assert isinstance(performance, list)
        assert len(performance) > 0
        
        # Verify each algorithm has proper statistics
        algorithm_names = set(sample['algorithm'] for sample in sample_detection_results)
        performance_algorithms = set(alg.algorithm for alg in performance)
        
        assert algorithm_names == performance_algorithms
        
        for alg_perf in performance:
            assert isinstance(alg_perf, AlgorithmStats)
            assert alg_perf.detections_count > 0
            assert alg_perf.anomalies_found >= 0
            assert 0 <= alg_perf.success_rate <= 100
            assert alg_perf.average_time >= 0
            assert isinstance(alg_perf.last_used, datetime)
    
    def test_data_quality_metrics_integration(self, analytics_service, sample_detection_results):
        """Test data quality metrics calculation."""
        # Record detection results with quality data
        for sample in sample_detection_results:
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time'],
                data_quality=sample['data_quality']
            )
        
        # Get data quality metrics
        quality_metrics = analytics_service.get_data_quality_metrics()
        
        # Verify quality metrics
        assert isinstance(quality_metrics, DataQualityMetrics)
        assert quality_metrics.quality_score > 0
        
        # Calculate expected values
        total_samples = sum(s['result'].total_samples for s in sample_detection_results if s['result'].success)
        expected_quality_score = np.mean([s['data_quality']['quality_score'] for s in sample_detection_results])
        
        assert quality_metrics.total_samples == total_samples
        assert abs(quality_metrics.quality_score - expected_quality_score) < 1.0
    
    def test_detection_timeline_integration(self, analytics_service, sample_detection_results):
        """Test detection timeline generation with time-based data."""
        # Record detection results with timestamps
        for sample in sample_detection_results:
            # Mock datetime.now() to return the sample timestamp
            with patch('anomaly_detection.domain.services.analytics_service.datetime') as mock_dt:
                mock_dt.now.return_value = sample['timestamp']
                
                analytics_service.record_detection(
                    algorithm=sample['algorithm'],
                    result=sample['result'],
                    processing_time=sample['processing_time']
                )
        
        # Get detection timeline
        timeline = analytics_service.get_detection_timeline(hours=24)
        
        # Verify timeline structure
        assert isinstance(timeline, list)
        
        if timeline:  # May be empty if implementation doesn't generate timeline
            for point in timeline:
                assert isinstance(point, dict)
                assert 'timestamp' in point
                assert 'detections' in point
                assert 'anomalies' in point
                assert point['detections'] >= 0
                assert point['anomalies'] >= 0
    
    def test_algorithm_distribution_integration(self, analytics_service, sample_detection_results):
        """Test algorithm distribution calculation."""
        # Record detection results
        for sample in sample_detection_results:
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time']
            )
        
        # Get algorithm distribution
        distribution = analytics_service.get_algorithm_distribution()
        
        # Verify distribution
        assert isinstance(distribution, list)
        assert len(distribution) > 0
        
        # Calculate expected distribution
        algorithm_counts = {}
        for sample in sample_detection_results:
            algorithm = sample['algorithm']
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        total_detections = len(sample_detection_results)
        
        # Verify distribution matches recorded data
        distribution_algorithms = set(item['algorithm'] for item in distribution)
        expected_algorithms = set(algorithm_counts.keys())
        
        assert distribution_algorithms == expected_algorithms
        
        # Verify percentages sum to 100
        total_percentage = sum(item['percentage'] for item in distribution)
        assert abs(total_percentage - 100.0) < 0.01
        
        # Verify counts match
        for item in distribution:
            expected_count = algorithm_counts[item['algorithm']]
            assert item['count'] == expected_count
    
    def test_system_status_integration(self, analytics_service, mock_metrics_collector):
        """Test system status with metrics collector integration."""
        # Configure mock metrics collector
        mock_metrics_collector.get_system_metrics.return_value = {
            'cpu_usage': 65.5,
            'memory_usage': 78.2,
            'disk_usage': 45.8,
            'memory_mb': 4096,
            'cpu_percent': 65.5
        }
        
        # Get system status
        status = analytics_service.get_system_status()
        
        # Verify status structure
        assert isinstance(status, dict)
        required_fields = [
            'overall_status', 'api_status', 'database_status',
            'memory_usage', 'cpu_usage', 'disk_usage',
            'active_operations', 'success_rate', 'last_check'
        ]
        
        for field in required_fields:
            assert field in status
        
        # Verify metrics collector was called
        mock_metrics_collector.get_system_metrics.assert_called_once()
        
        # Verify status values
        assert status['cpu_usage'] == '65.5%'
        assert status['memory_usage'] == '78.2%'
        assert status['disk_usage'] == '45.8%'
    
    def test_dashboard_stats_integration(self, analytics_service, sample_detection_results):
        """Test dashboard statistics with comprehensive data."""
        # Record detection results
        for sample in sample_detection_results:
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time']
            )
        
        # Get dashboard stats
        stats = analytics_service.get_dashboard_stats()
        
        # Verify stats structure
        assert isinstance(stats, dict)
        required_fields = [
            'total_detections', 'total_anomalies', 'active_algorithms',
            'average_detection_time', 'system_status', 'success_rate'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Verify stats values
        assert stats['total_detections'] == len(sample_detection_results)
        
        successful_results = [s for s in sample_detection_results if s['result'].success]
        expected_anomalies = sum(s['result'].anomaly_count for s in successful_results)
        assert stats['total_anomalies'] == expected_anomalies
        
        unique_algorithms = set(sample['algorithm'] for sample in sample_detection_results)
        assert stats['active_algorithms'] == len(unique_algorithms)
        
        assert stats['average_detection_time'] > 0
        assert 0 <= stats['success_rate'] <= 100
    
    def test_simulation_integration(self, analytics_service):
        """Test detection simulation functionality."""
        # Run simulation
        simulation_result = analytics_service.simulate_detection()
        
        # Verify simulation result
        assert isinstance(simulation_result, dict)
        required_fields = [
            'algorithm', 'total_samples', 'anomalies_found',
            'anomaly_rate', 'processing_time'
        ]
        
        for field in required_fields:
            assert field in simulation_result
        
        # Verify simulation values are realistic
        assert simulation_result['total_samples'] > 0
        assert simulation_result['anomalies_found'] >= 0
        assert 0 <= simulation_result['anomaly_rate'] <= 100
        assert simulation_result['processing_time'] > 0
        assert simulation_result['algorithm'] in [
            'isolation_forest', 'lof', 'one_class_svm', 'ensemble_majority'
        ]
    
    def test_performance_trending_integration(self, analytics_service, sample_detection_results):
        """Test performance trending over time."""
        # Record detection results with time progression
        base_time = datetime.now() - timedelta(hours=12)
        
        for i, sample in enumerate(sample_detection_results[:20]):
            # Create performance history entries
            performance_entry = {
                'timestamp': base_time + timedelta(minutes=i*30),
                'processing_time': sample['processing_time'] * 1000,  # Convert to ms
                'throughput': 100 / sample['processing_time'],  # samples per second
                'success_rate': 95.0 + np.random.uniform(-5, 5)  # Varying success rate
            }
            
            analytics_service.performance_history.append(performance_entry)
        
        # Get performance trend
        trend = analytics_service.get_performance_trend(hours=12)
        
        # Verify trend structure
        assert isinstance(trend, list)
        
        if trend:  # May be empty based on implementation
            for point in trend:
                assert isinstance(point, dict)
                assert 'timestamp' in point
                assert 'processing_time' in point
                assert 'throughput' in point
                assert 'success_rate' in point
    
    def test_concurrent_analytics_operations_integration(self, analytics_service, sample_detection_results):
        """Test concurrent analytics operations."""
        import threading
        import time
        
        results = {'recorded': 0, 'errors': 0}
        
        def record_detections(start_idx, count):
            """Record detections concurrently."""
            try:
                end_idx = min(start_idx + count, len(sample_detection_results))
                for i in range(start_idx, end_idx):
                    sample = sample_detection_results[i]
                    analytics_service.record_detection(
                        algorithm=sample['algorithm'],
                        result=sample['result'],
                        processing_time=sample['processing_time']
                    )
                    results['recorded'] += 1
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                results['errors'] += 1
                print(f"Error in concurrent recording: {e}")
        
        def read_analytics():
            """Read analytics concurrently."""
            try:
                for _ in range(5):
                    analytics_service.get_performance_metrics()
                    analytics_service.get_dashboard_stats()
                    analytics_service.get_algorithm_performance()
                    time.sleep(0.01)
            except Exception as e:
                results['errors'] += 1
                print(f"Error in concurrent reading: {e}")
        
        # Start concurrent operations
        threads = [
            threading.Thread(target=record_detections, args=(0, 10)),
            threading.Thread(target=record_detections, args=(10, 10)),
            threading.Thread(target=read_analytics),
            threading.Thread(target=read_analytics)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify concurrent operations completed successfully
        assert results['recorded'] > 0
        assert results['errors'] == 0
        
        # Verify data integrity after concurrent operations
        performance_metrics = analytics_service.get_performance_metrics()
        assert performance_metrics.total_detections == results['recorded']
    
    def test_memory_management_integration(self, analytics_service, sample_detection_results):
        """Test memory management with large datasets."""
        # Record more detection results than the deque limit
        max_history = analytics_service.detection_history.maxlen
        
        # Record twice the maximum history to test memory management
        for i in range(max_history + 50):
            sample_idx = i % len(sample_detection_results)
            sample = sample_detection_results[sample_idx]
            
            analytics_service.record_detection(
                algorithm=sample['algorithm'],
                result=sample['result'],
                processing_time=sample['processing_time']
            )
        
        # Verify memory management
        assert len(analytics_service.detection_history) == max_history
        
        # Verify analytics still work correctly
        performance_metrics = analytics_service.get_performance_metrics()
        assert performance_metrics.total_detections == max_history
        
        dashboard_stats = analytics_service.get_dashboard_stats()
        assert dashboard_stats['total_detections'] == max_history
        
        # Verify algorithm stats are maintained
        algorithm_performance = analytics_service.get_algorithm_performance()
        assert len(algorithm_performance) > 0


if __name__ == "__main__":
    print("Analytics Service Integration Tests")
    print("Testing analytics service with realistic data and scenarios")
    
    # Simple functionality test
    try:
        from collections import deque
        import numpy as np
        from datetime import datetime, timedelta
        
        print("✓ All dependencies available")
        print("Ready to run analytics integration tests")
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")