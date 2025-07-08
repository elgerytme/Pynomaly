#!/usr/bin/env python3
"""Direct test of baseline functionality."""

import json
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Test direct import
from pynomaly.application.services.performance_benchmarking_service import (
    PerformanceBenchmarkingService,
    BenchmarkSuite,
    PerformanceMetrics,
    SeverityThresholds,
)

def test_direct():
    """Test the baseline comparison functionality."""
    
    # Create a temporary directory for the service
    temp_dir = Path.cwd() / 'temp_test'
    temp_dir.mkdir(exist_ok=True)
    service = PerformanceBenchmarkingService(temp_dir)
    
    # Create a baseline dictionary
    baseline = {
        'timestamp': '2025-06-25T00:00:00.000000',
        'performance_metrics': {
            'basic_workflow_time': 450.0,
            'peak_memory_mb': 145.0
        },
        'git_ref': 'main',
        'git_commit': 'baseline',
        'description': 'Test baseline'
    }
    
    # Create current results
    current_results = [
        PerformanceMetrics(
            algorithm_name='TestAlgorithm',
            execution_time_seconds=0.5,  # Much faster than baseline
            peak_memory_mb=150.0,  # Slightly higher than baseline
            accuracy_score=0.88,
            training_throughput=120.0,
            dataset_size=1000,
            feature_dimension=10,
            contamination_rate=0.1
        )
    ]
    
    current_suite = BenchmarkSuite(
        suite_name='Current Suite',
        individual_results=current_results,
        start_time=datetime.utcnow()
    )
    
    # Create custom thresholds
    thresholds = SeverityThresholds(
        minor_threshold=0.05,
        major_threshold=0.15,
        critical_threshold=0.30
    )
    
    # Test the comparison
    result = service.compare_to_baseline(baseline, current_suite, thresholds)
    
    print("SUCCESS: Baseline comparison method works!")
    print(f"Regressions: {len(result['regressions'])}")
    print(f"Improvements: {len(result['improvements'])}")
    
    # Print details
    for regression in result['regressions']:
        print(f"  Regression: {regression['metric']} changed by {regression['percent_change']:.1f}% ({regression['severity']})")
    
    for improvement in result['improvements']:
        print(f"  Improvement: {improvement['metric']} changed by {improvement['percent_change']:.1f}% ({improvement['severity']})")
    
    print(f"Summary: {result['summary']}")
    
    # Verify the diff file was created
    diff_path = temp_dir / f"benchmark_diff_{current_suite.suite_id}.json"
    print(f"Diff file created: {diff_path.exists()}")
    
    if diff_path.exists():
        with open(diff_path, 'r') as f:
            saved_data = json.load(f)
        print(f"Diff file contains {len(saved_data)} keys")
        print(f"File structure: {list(saved_data.keys())}")
    
    return result

if __name__ == "__main__":
    test_direct()
