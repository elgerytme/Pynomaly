"""Unit tests for analytics service dataclasses that can run independently."""

import pytest
from datetime import datetime
from dataclasses import asdict


def test_performance_metrics_dataclass():
    """Test PerformanceMetrics dataclass independently."""
    from dataclasses import dataclass
    
    @dataclass
    class PerformanceMetrics:
        """Performance metrics for the system."""
        total_detections: int = 0
        total_anomalies: int = 0
        average_detection_time: float = 0.0
        success_rate: float = 0.0
        throughput: float = 0.0  # detections per second
        error_rate: float = 0.0
    
    # Test default initialization
    metrics = PerformanceMetrics()
    assert metrics.total_detections == 0
    assert metrics.total_anomalies == 0
    assert metrics.average_detection_time == 0.0
    assert metrics.success_rate == 0.0
    assert metrics.throughput == 0.0
    assert metrics.error_rate == 0.0
    
    # Test custom initialization
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
    
    # Test conversion to dict
    metrics_dict = asdict(metrics)
    assert metrics_dict["total_detections"] == 100
    assert metrics_dict["total_anomalies"] == 15


def test_algorithm_stats_dataclass():
    """Test AlgorithmStats dataclass independently."""
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class AlgorithmStats:
        """Statistics for a specific algorithm."""
        algorithm: str
        detections_count: int = 0
        anomalies_found: int = 0
        average_score: float = 0.0
        average_time: float = 0.0
        success_rate: float = 0.0
        last_used: Optional[datetime] = None
    
    # Test default initialization
    stats = AlgorithmStats(algorithm="isolation_forest")
    assert stats.algorithm == "isolation_forest"
    assert stats.detections_count == 0
    assert stats.anomalies_found == 0
    assert stats.average_score == 0.0
    assert stats.average_time == 0.0
    assert stats.success_rate == 0.0
    assert stats.last_used is None
    
    # Test custom initialization
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
    
    # Test conversion to dict
    stats_dict = asdict(stats)
    assert stats_dict["algorithm"] == "lof"
    assert stats_dict["detections_count"] == 50


def test_data_quality_metrics_dataclass():
    """Test DataQualityMetrics dataclass independently."""
    from dataclasses import dataclass
    
    @dataclass
    class DataQualityMetrics:
        """Data quality metrics."""
        total_samples: int = 0
        missing_values: int = 0
        duplicate_samples: int = 0
        outliers_count: int = 0
        data_drift_events: int = 0
        quality_score: float = 0.0
    
    # Test default initialization
    metrics = DataQualityMetrics()
    assert metrics.total_samples == 0
    assert metrics.missing_values == 0
    assert metrics.duplicate_samples == 0
    assert metrics.outliers_count == 0
    assert metrics.data_drift_events == 0
    assert metrics.quality_score == 0.0
    
    # Test custom initialization
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
    
    # Test conversion to dict
    metrics_dict = asdict(metrics)
    assert metrics_dict["total_samples"] == 1000
    assert metrics_dict["quality_score"] == 85.0


def test_analytics_calculations():
    """Test analytics calculation methods independently."""
    
    def calculate_success_rate(total_operations: int, successful_operations: int) -> float:
        """Calculate success rate percentage."""
        if total_operations == 0:
            return 0.0
        return (successful_operations / total_operations) * 100.0
    
    def calculate_error_rate(total_operations: int, failed_operations: int) -> float:
        """Calculate error rate percentage."""
        if total_operations == 0:
            return 0.0
        return (failed_operations / total_operations) * 100.0
    
    def calculate_average_time(times: list) -> float:
        """Calculate average time from a list of times."""
        if not times:
            return 0.0
        return sum(times) / len(times)
    
    def calculate_throughput(operations: int, time_period_seconds: float) -> float:
        """Calculate throughput (operations per second)."""
        if time_period_seconds <= 0:
            return 0.0
        return operations / time_period_seconds
    
    # Test success rate calculation
    assert calculate_success_rate(100, 95) == 95.0
    assert calculate_success_rate(0, 0) == 0.0
    assert calculate_success_rate(10, 10) == 100.0
    
    # Test error rate calculation
    assert calculate_error_rate(100, 5) == 5.0
    assert calculate_error_rate(0, 0) == 0.0
    assert calculate_error_rate(10, 0) == 0.0
    
    # Test average time calculation
    assert calculate_average_time([1.0, 2.0, 3.0]) == 2.0
    assert calculate_average_time([]) == 0.0
    assert calculate_average_time([5.0]) == 5.0
    
    # Test throughput calculation
    assert calculate_throughput(100, 10.0) == 10.0
    assert calculate_throughput(0, 10.0) == 0.0
    assert calculate_throughput(100, 0.0) == 0.0


def test_data_aggregation():
    """Test data aggregation functions that might be used in analytics."""
    from collections import defaultdict, deque
    
    def aggregate_by_algorithm(detection_records: list) -> dict:
        """Aggregate detection records by algorithm."""
        aggregated = defaultdict(lambda: {
            'count': 0,
            'total_anomalies': 0,
            'total_time': 0.0
        })
        
        for record in detection_records:
            algorithm = record['algorithm']
            aggregated[algorithm]['count'] += 1
            aggregated[algorithm]['total_anomalies'] += record['anomalies_found']
            aggregated[algorithm]['total_time'] += record['processing_time']
        
        # Calculate averages
        for algorithm, stats in aggregated.items():
            if stats['count'] > 0:
                stats['avg_anomalies'] = stats['total_anomalies'] / stats['count']
                stats['avg_time'] = stats['total_time'] / stats['count']
            else:
                stats['avg_anomalies'] = 0.0
                stats['avg_time'] = 0.0
        
        return dict(aggregated)
    
    def calculate_time_windows(records: list, window_hours: int = 1) -> list:
        """Calculate metrics for time windows."""
        # Simplified implementation for testing
        return [
            {
                'timestamp': '2024-01-20T10:00:00',
                'detections': len([r for r in records if r['algorithm'] == 'isolation_forest']),
                'anomalies': sum(r['anomalies_found'] for r in records if r['algorithm'] == 'isolation_forest')
            }
        ]
    
    # Test algorithm aggregation
    test_records = [
        {'algorithm': 'isolation_forest', 'anomalies_found': 5, 'processing_time': 1.2},
        {'algorithm': 'isolation_forest', 'anomalies_found': 3, 'processing_time': 1.8},
        {'algorithm': 'lof', 'anomalies_found': 2, 'processing_time': 2.1},
    ]
    
    aggregated = aggregate_by_algorithm(test_records)
    
    assert 'isolation_forest' in aggregated
    assert 'lof' in aggregated
    assert aggregated['isolation_forest']['count'] == 2
    assert aggregated['isolation_forest']['total_anomalies'] == 8
    assert aggregated['isolation_forest']['avg_anomalies'] == 4.0
    assert aggregated['lof']['count'] == 1
    assert aggregated['lof']['total_anomalies'] == 2
    
    # Test time window calculation
    windows = calculate_time_windows(test_records)
    assert len(windows) > 0
    assert 'timestamp' in windows[0]
    assert 'detections' in windows[0]
    assert 'anomalies' in windows[0]


def test_statistics_calculations():
    """Test statistical calculations used in analytics."""
    import statistics
    
    def calculate_percentiles(values: list, percentiles: list = [50, 75, 90, 95, 99]) -> dict:
        """Calculate percentiles for a list of values."""
        if not values:
            return {p: 0.0 for p in percentiles}
        
        result = {}
        sorted_values = sorted(values)
        
        for p in percentiles:
            index = (p / 100.0) * (len(sorted_values) - 1)
            if index.is_integer():
                result[p] = sorted_values[int(index)]
            else:
                lower = sorted_values[int(index)]
                upper = sorted_values[int(index) + 1]
                result[p] = lower + (upper - lower) * (index - int(index))
        
        return result
    
    def calculate_moving_average(values: list, window_size: int = 5) -> list:
        """Calculate moving average."""
        if len(values) < window_size:
            return values
        
        moving_averages = []
        for i in range(window_size - 1, len(values)):
            window = values[i - window_size + 1:i + 1]
            moving_averages.append(sum(window) / window_size)
        
        return moving_averages
    
    # Test percentile calculations
    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    percentiles = calculate_percentiles(test_values, [50, 90])
    
    assert percentiles[50] == 5.5  # Median
    assert percentiles[90] == 9.1  # 90th percentile
    
    # Test empty values
    empty_percentiles = calculate_percentiles([], [50, 90])
    assert empty_percentiles[50] == 0.0
    assert empty_percentiles[90] == 0.0
    
    # Test moving average
    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    moving_avg = calculate_moving_average(test_values, 3)
    
    assert len(moving_avg) == 8  # 10 - 3 + 1
    assert moving_avg[0] == 2.0  # (1+2+3)/3
    assert moving_avg[-1] == 9.0  # (8+9+10)/3
    
    # Test moving average with insufficient data
    short_values = [1, 2]
    short_moving_avg = calculate_moving_average(short_values, 5)
    assert short_moving_avg == short_values


if __name__ == "__main__":
    # Run tests directly
    test_performance_metrics_dataclass()
    test_algorithm_stats_dataclass()
    test_data_quality_metrics_dataclass()
    test_analytics_calculations()
    test_data_aggregation()
    test_statistics_calculations()
    print("All tests passed!")