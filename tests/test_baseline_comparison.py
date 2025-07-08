#!/usr/bin/env python3
"""Test baseline comparison functionality."""

import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from pynomaly.application.services.performance_benchmarking_service import (
    PerformanceBenchmarkingService, 
    BenchmarkSuite, 
    PerformanceMetrics, 
    SeverityThresholds
)
from datetime import datetime
from uuid import uuid4

def test_baseline_comparison():
    # Create test data
    service = PerformanceBenchmarkingService(Path(tempfile.mkdtemp()))

    baseline = {
        'timestamp': '2025-06-25T00:00:00.000000',
        'performance_metrics': {
            'basic_workflow_time': 450.0,
            'peak_memory_mb': 145.0
        }
    }

    current = BenchmarkSuite(
        suite_name='Current Suite',
        individual_results=[
            PerformanceMetrics(
                algorithm_name='TestAlgorithm',
                execution_time_seconds=0.5,
                peak_memory_mb=150.0,
                accuracy_score=0.88,
                training_throughput=120.0,
                dataset_size=1000,
                feature_dimension=10,
                contamination_rate=0.1
            )
        ],
        start_time=datetime.utcnow()
    )

    # Test the method
    result = service.compare_to_baseline(baseline, current)
    print('SUCCESS: Baseline comparison method works!')
    print(f'Regressions: {len(result["regressions"])}')
    print(f'Improvements: {len(result["improvements"])}')
    print(f'Summary: {result["summary"]}')
    
    # Test the file was created
    diff_path = service.storage_path / f"benchmark_diff_{current.suite_id}.json"
    print(f'Diff file created: {diff_path.exists()}')
    
    return result

if __name__ == "__main__":
    test_baseline_comparison()
