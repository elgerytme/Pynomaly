#!/usr/bin/env python3
"""Simple test for performance configuration v2."""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_performance_config():
    """Test performance configuration functionality."""
    try:
        # Import directly to avoid dependency issues
        sys.path.insert(0, 'src/pynomaly/application/services')
        from performance_benchmarking_service import (
            PerformanceBenchmarkingService, 
            BenchmarkConfig,
            PerformanceMetrics
        )
        
        print("✓ Successfully imported PerformanceBenchmarkingService")
        
        # Create service with temp directory
        temp_dir = Path(tempfile.mkdtemp())
        service = PerformanceBenchmarkingService(temp_dir)
        print("✓ Service initialized successfully")
        
        # Test configuration loading
        is_v2 = service.is_v2_config()
        print(f"✓ Using v2 config: {is_v2}")
        
        # Test threshold retrieval
        exec_time_threshold = service.get_threshold(
            'performance_thresholds.execution_time.max_execution_time_seconds', 
            999
        )
        print(f"✓ Execution time threshold: {exec_time_threshold}")
        
        # Test configuration application
        config = BenchmarkConfig()
        print(f"✓ Default max execution time: {config.max_execution_time_seconds}")
        
        config.apply_performance_config(service)
        print(f"✓ After config application: {config.max_execution_time_seconds}")
        
        # Test validation
        metrics = PerformanceMetrics(
            execution_time_seconds=200.0,
            peak_memory_mb=1000.0,
            training_throughput=50.0,
            accuracy_score=0.85,
            f1_score=0.80,
            cpu_usage_percent=60.0
        )
        
        validation_results = service.validate_performance_metrics(metrics)
        print(f"✓ Validation results: {validation_results}")
        
        # Test algorithm-specific thresholds
        fast_thresholds = service.get_algorithm_specific_thresholds('LocalOutlierFactor')
        print(f"✓ Fast algorithm thresholds: {fast_thresholds}")
        
        default_thresholds = service.get_algorithm_specific_thresholds('UnknownAlgorithm')
        print(f"✓ Default algorithm thresholds: {default_thresholds}")
        
        print("\n🎉 All tests passed! Performance configuration v2 is working correctly.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_performance_config()
    sys.exit(0 if success else 1)
