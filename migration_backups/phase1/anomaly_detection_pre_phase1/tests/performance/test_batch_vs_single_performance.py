#!/usr/bin/env python3
"""
Test for the specific batch vs single performance issue mentioned in #859.

This test verifies that batch processing is significantly faster than individual
processing, addressing the performance bottleneck that was fixed.
"""

import pytest
import time
import numpy as np
from typing import Dict, Any

from anomaly_detection.domain.services.detection_service import DetectionService


class TestBatchVsSinglePerformance:
    """Test batch processing performance vs individual sample processing."""
    
    @pytest.fixture
    def detection_service(self):
        """Create detection service for testing."""
        return DetectionService()
    
    @pytest.fixture
    def test_dataset(self):
        """Generate test dataset for performance testing."""
        np.random.seed(42)
        # Generate dataset with clear normal and anomaly patterns
        normal_data = np.random.randn(1000, 10)
        anomaly_data = np.random.randn(100, 10) * 3 + 5  # Shifted and scaled anomalies
        return np.vstack([normal_data, anomaly_data])
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure function execution time and success."""
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        return {
            'processing_time': processing_time,
            'success': success,
            'result': result,
            'error': error
        }
    
    def test_batch_processing_performance(self, detection_service, test_dataset):
        """Test that batch processing is significantly faster than individual processing.
        
        This is the main test that was failing in issue #859.
        The batch processing should be at least 5x faster than processing samples individually.
        """
        # Test batch processing (single large call)
        batch_perf = self.measure_performance(
            detection_service.detect_anomalies,
            data=test_dataset,
            algorithm='iforest',
            contamination=0.1,
            n_estimators=50,
            random_state=42
        )
        
        assert batch_perf['success'], f"Batch processing failed: {batch_perf['error']}"
        
        batch_samples_per_sec = len(test_dataset) / batch_perf['processing_time']
        print(f"\nBatch processing: {batch_perf['processing_time']:.2f}s ({batch_samples_per_sec:.0f} samples/sec)")
        print(f"Anomalies detected: {batch_perf['result'].anomaly_count}")
        
        # Test individual processing (multiple small calls)
        # Process dataset in smaller chunks to simulate individual processing
        chunk_size = 100
        num_chunks = min(10, len(test_dataset) // chunk_size)
        
        individual_times = []
        total_anomalies = 0
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = test_dataset[start_idx:end_idx]
            
            chunk_perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=chunk,
                algorithm='iforest',
                contamination=0.1,
                n_estimators=50,
                random_state=42 + i
            )
            
            assert chunk_perf['success'], f"Individual chunk {i} failed: {chunk_perf['error']}"
            individual_times.append(chunk_perf['processing_time'])
            total_anomalies += chunk_perf['result'].anomaly_count
        
        # Calculate individual processing metrics
        total_individual_time = sum(individual_times)
        samples_processed = num_chunks * chunk_size
        individual_throughput = samples_processed / total_individual_time
        
        print(f"Individual processing: {total_individual_time:.2f}s ({individual_throughput:.0f} samples/sec)")
        print(f"Total anomalies detected: {total_anomalies}")
        
        # Calculate efficiency ratio
        efficiency_ratio = batch_samples_per_sec / individual_throughput
        print(f"Batch efficiency: {efficiency_ratio:.1f}x faster than individual processing")
        
        # This is the critical assertion that was failing before the fix
        assert efficiency_ratio >= 5.0, (
            f"Batch processing not efficient enough: {efficiency_ratio:.1f}x speedup "
            f"(expected at least 5.0x). This indicates a performance problem."
        )
        
        # Additional performance assertions
        assert batch_perf['processing_time'] < 10.0, (
            f"Batch processing too slow: {batch_perf['processing_time']:.2f}s "
            f"(expected < 10s for {len(test_dataset)} samples)"
        )
        
        assert batch_samples_per_sec > 500, (
            f"Batch throughput too low: {batch_samples_per_sec:.0f} samples/sec "
            f"(expected > 500 samples/sec)"
        )
    
    def test_scalability_performance(self, detection_service):
        """Test that performance scales reasonably with dataset size."""
        sizes = [500, 1000, 2000]
        throughputs = []
        
        for size in sizes:
            np.random.seed(42)
            data = np.random.randn(size, 8)
            
            perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=data,
                algorithm='iforest',
                contamination=0.1,
                n_estimators=30,  # Reduced for faster testing
                random_state=42
            )
            
            assert perf['success'], f"Scalability test failed for size {size}: {perf['error']}"
            
            throughput = size / perf['processing_time']
            throughputs.append(throughput)
            
            print(f"Size {size}: {perf['processing_time']:.2f}s ({throughput:.0f} samples/sec)")
        
        # Check that throughput doesn't degrade dramatically with size
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        throughput_ratio = max_throughput / min_throughput
        
        print(f"Throughput ratio (max/min): {throughput_ratio:.1f}x")
        
        # Throughput shouldn't vary by more than 3x across different sizes
        assert throughput_ratio < 3.0, (
            f"Poor scalability: throughput varies by {throughput_ratio:.1f}x "
            f"across dataset sizes (expected < 3.0x)"
        )
    
    def test_algorithm_performance_comparison(self, detection_service):
        """Test performance of different algorithms."""
        np.random.seed(42)
        test_data = np.random.randn(800, 12)
        
        algorithms = [
            ('iforest', {'n_estimators': 50}),
            ('lof', {'n_neighbors': 20})
        ]
        
        results = {}
        
        for algo_name, params in algorithms:
            perf = self.measure_performance(
                detection_service.detect_anomalies,
                data=test_data,
                algorithm=algo_name,
                contamination=0.1,
                **params
            )
            
            if perf['success']:
                throughput = len(test_data) / perf['processing_time']
                results[algo_name] = {
                    'throughput': throughput,
                    'time': perf['processing_time'],
                    'anomalies': perf['result'].anomaly_count
                }
                print(f"{algo_name}: {perf['processing_time']:.2f}s ({throughput:.0f} samples/sec)")
        
        # At least one algorithm should work
        assert len(results) > 0, "No algorithms completed successfully"
        
        # All successful algorithms should have reasonable performance
        for algo_name, metrics in results.items():
            assert metrics['throughput'] > 100, (
                f"{algo_name} throughput too low: {metrics['throughput']:.0f} samples/sec"
            )


if __name__ == "__main__":
    print("Batch vs Single Performance Test")
    print("=" * 40)
    print("Testing the performance fix for issue #859")
    print()
    
    # Quick smoke test
    service = DetectionService()
    np.random.seed(42)
    test_data = np.random.randn(200, 5)
    
    try:
        start = time.perf_counter()
        result = service.detect_anomalies(
            data=test_data,
            algorithm='iforest',
            contamination=0.1,
            n_estimators=30
        )
        end = time.perf_counter()
        
        throughput = len(test_data) / (end - start)
        print(f"✓ Quick test passed: {result.anomaly_count} anomalies detected")
        print(f"  Performance: {throughput:.0f} samples/sec")
        print("Ready to run comprehensive batch performance tests")
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        print("Performance tests may not run properly")