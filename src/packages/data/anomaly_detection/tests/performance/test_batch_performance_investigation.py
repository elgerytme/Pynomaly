#!/usr/bin/env python3
"""
Investigation of batch processing performance issues.

This test focuses on identifying and fixing the specific performance bottleneck
mentioned in issue #859.
"""

import time
import numpy as np
import pytest
from typing import Dict, Any

from anomaly_detection.domain.services.detection_service import DetectionService


class TestBatchPerformanceInvestigation:
    """Focused performance investigation for batch processing."""
    
    @pytest.fixture
    def detection_service(self):
        """Create detection service for testing."""
        return DetectionService()
    
    @pytest.fixture 
    def test_dataset(self):
        """Generate test dataset for performance investigation."""
        np.random.seed(42)
        # Create a reasonably-sized dataset for performance testing
        normal_data = np.random.randn(8000, 15)
        # Add some clear anomalies
        anomaly_data = np.random.randn(200, 15) * 3 + 10  # Shifted and scaled
        return np.vstack([normal_data, anomaly_data])
    
    def measure_processing_time(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure processing time for a function call."""
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
    
    def test_current_batch_vs_individual_performance(self, detection_service, test_dataset):
        """Test current implementation: batch vs individual processing."""
        print("\n=== Current Implementation Performance Test ===")
        
        # Test batch processing (current approach)
        print("Testing batch processing (single large call)...")
        batch_perf = self.measure_processing_time(
            detection_service.detect_anomalies,
            data=test_dataset,
            algorithm='iforest',
            contamination=0.05,
            n_estimators=50,
            random_state=42
        )
        
        if batch_perf['success']:
            batch_samples_per_sec = len(test_dataset) / batch_perf['processing_time']
            print(f"  Batch processing: {batch_perf['processing_time']:.2f}s")
            print(f"  Throughput: {batch_samples_per_sec:.0f} samples/sec")
            print(f"  Anomalies detected: {batch_perf['result'].anomaly_count}")
        else:
            print(f"  Batch processing failed: {batch_perf['error']}")
            return
        
        # Test individual processing (simulating multiple small calls)
        print("\nTesting individual processing (multiple small calls)...")
        
        # Break dataset into smaller chunks
        chunk_size = 100
        num_chunks = min(10, len(test_dataset) // chunk_size)  # Limit to 10 chunks for testing
        
        individual_times = []
        total_anomalies = 0
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = test_dataset[start_idx:end_idx]
            
            chunk_perf = self.measure_processing_time(
                detection_service.detect_anomalies,
                data=chunk,
                algorithm='iforest',
                contamination=0.05,
                n_estimators=50,
                random_state=42 + i
            )
            
            if chunk_perf['success']:
                individual_times.append(chunk_perf['processing_time'])
                total_anomalies += chunk_perf['result'].anomaly_count
            else:
                print(f"  Chunk {i} failed: {chunk_perf['error']}")
        
        if individual_times:
            total_individual_time = sum(individual_times)
            avg_chunk_time = total_individual_time / len(individual_times)
            samples_processed = num_chunks * chunk_size
            individual_throughput = samples_processed / total_individual_time
            
            print(f"  Individual processing: {total_individual_time:.2f}s total")
            print(f"  Average per chunk: {avg_chunk_time:.3f}s")
            print(f"  Throughput: {individual_throughput:.0f} samples/sec")
            print(f"  Total anomalies detected: {total_anomalies}")
            
            # Calculate efficiency ratio
            efficiency_ratio = batch_samples_per_sec / individual_throughput
            print(f"\nEfficiency Analysis:")
            print(f"  Batch is {efficiency_ratio:.1f}x faster than individual processing")
            
            # The test that's likely failing: batch should be significantly faster
            if efficiency_ratio < 5.0:
                print(f"  ❌ PERFORMANCE ISSUE: Batch efficiency is only {efficiency_ratio:.1f}x")
                print(f"     Expected: >5x faster")
                print(f"     This is likely the failing test mentioned in issue #859")
            else:
                print(f"  ✅ GOOD: Batch processing is adequately faster")
        
        # Analyze the root cause
        print(f"\n=== Root Cause Analysis ===")
        print(f"Current implementation calls sklearn's fit_predict() for each request.")
        print(f"This means:")
        print(f"  - Batch processing: 1 model training + 1 prediction")
        print(f"  - Individual processing: {num_chunks} model trainings + {num_chunks} predictions")
        print(f"  - The difference should be significant due to training overhead")
        
        return {
            'batch_time': batch_perf['processing_time'],
            'individual_time': total_individual_time if individual_times else 0,
            'efficiency_ratio': efficiency_ratio if individual_times else 0,
            'performance_issue': efficiency_ratio < 5.0 if individual_times else True
        }
    
    def test_identify_bottlenecks(self, detection_service, test_dataset):
        """Identify specific bottlenecks in the detection pipeline."""
        print("\n=== Bottleneck Identification ===")
        
        # Test different aspects of the pipeline
        dataset_subset = test_dataset[:2000]  # Smaller dataset for detailed analysis
        
        # Test with different n_estimators values
        estimator_counts = [10, 50, 100, 200]
        
        print("Testing impact of n_estimators on performance:")
        for n_est in estimator_counts:
            perf = self.measure_processing_time(
                detection_service.detect_anomalies,
                data=dataset_subset,
                algorithm='iforest',
                contamination=0.05,
                n_estimators=n_est,
                random_state=42
            )
            
            if perf['success']:
                throughput = len(dataset_subset) / perf['processing_time']
                print(f"  n_estimators={n_est}: {perf['processing_time']:.2f}s ({throughput:.0f} samples/sec)")
        
        # Test different dataset sizes
        print("\nTesting scalability with dataset size:")
        sizes = [500, 1000, 2000, 4000]
        
        for size in sizes:
            data_subset = test_dataset[:size]
            perf = self.measure_processing_time(
                detection_service.detect_anomalies,
                data=data_subset,
                algorithm='iforest',
                contamination=0.05,
                n_estimators=50,
                random_state=42
            )
            
            if perf['success']:
                throughput = size / perf['processing_time']
                print(f"  {size} samples: {perf['processing_time']:.2f}s ({throughput:.0f} samples/sec)")
    
    def test_proposed_batch_optimization(self, detection_service, test_dataset):
        """Test a proposed optimization for batch processing."""
        print("\n=== Proposed Batch Optimization Test ===")
        
        # Current approach: multiple separate detect_anomalies calls
        dataset_chunks = [
            test_dataset[i:i+1000] for i in range(0, min(5000, len(test_dataset)), 1000)
        ]
        
        print("Current approach (multiple separate calls):")
        start_time = time.perf_counter()
        
        current_results = []
        for i, chunk in enumerate(dataset_chunks):
            result = detection_service.detect_anomalies(
                data=chunk,
                algorithm='iforest',
                contamination=0.05,
                n_estimators=50,
                random_state=42 + i
            )
            current_results.append(result)
        
        current_time = time.perf_counter() - start_time
        total_samples = sum(len(chunk) for chunk in dataset_chunks)
        current_throughput = total_samples / current_time
        
        print(f"  Total time: {current_time:.2f}s")
        print(f"  Throughput: {current_throughput:.0f} samples/sec")
        print(f"  Total anomalies: {sum(r.anomaly_count for r in current_results)}")
        
        # Proposed optimization: single large call
        print("\nProposed optimization (single large call):")
        combined_data = np.vstack(dataset_chunks)
        
        start_time = time.perf_counter()
        optimized_result = detection_service.detect_anomalies(
            data=combined_data,
            algorithm='iforest', 
            contamination=0.05,
            n_estimators=50,
            random_state=42
        )
        optimized_time = time.perf_counter() - start_time
        optimized_throughput = len(combined_data) / optimized_time
        
        print(f"  Total time: {optimized_time:.2f}s")
        print(f"  Throughput: {optimized_throughput:.0f} samples/sec")
        print(f"  Total anomalies: {optimized_result.anomaly_count}")
        
        # Calculate improvement
        improvement_ratio = optimized_throughput / current_throughput
        time_saved = current_time - optimized_time
        
        print(f"\nOptimization Results:")
        print(f"  Improvement: {improvement_ratio:.1f}x faster")
        print(f"  Time saved: {time_saved:.2f}s ({time_saved/current_time*100:.1f}%)")
        
        if improvement_ratio > 2.0:
            print(f"  ✅ SIGNIFICANT IMPROVEMENT: {improvement_ratio:.1f}x speedup")
        else:
            print(f"  ⚠️  MODEST IMPROVEMENT: {improvement_ratio:.1f}x speedup")
        
        return {
            'current_throughput': current_throughput,
            'optimized_throughput': optimized_throughput,
            'improvement_ratio': improvement_ratio,
            'time_saved_percent': time_saved/current_time*100
        }
    
    def test_memory_efficiency_investigation(self, detection_service, test_dataset):
        """Investigate memory usage patterns."""
        print("\n=== Memory Efficiency Investigation ===")
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            print("psutil not available - skipping memory analysis")
            return
        
        # Test memory usage with different approaches
        dataset_subset = test_dataset[:3000]
        
        # Measure memory before
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Test multiple small calls
        print("\nTesting memory usage: multiple small calls")
        chunk_size = 300
        chunks = [dataset_subset[i:i+chunk_size] for i in range(0, len(dataset_subset), chunk_size)]
        
        memory_readings = [initial_memory]
        
        for i, chunk in enumerate(chunks):
            detection_service.detect_anomalies(
                data=chunk,
                algorithm='iforest',
                contamination=0.05,
                n_estimators=50,
                random_state=42 + i
            )
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            
            if i % 3 == 0:  # Print every 3rd reading
                print(f"  After chunk {i+1}: {current_memory:.1f} MB")
        
        peak_memory_small = max(memory_readings)
        final_memory_small = memory_readings[-1]
        
        # Test single large call
        print(f"\nTesting memory usage: single large call")
        pre_large_memory = process.memory_info().rss / 1024 / 1024
        
        detection_service.detect_anomalies(
            data=dataset_subset,
            algorithm='iforest',
            contamination=0.05,
            n_estimators=50,
            random_state=42
        )
        
        post_large_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"  Before large call: {pre_large_memory:.1f} MB")
        print(f"  After large call: {post_large_memory:.1f} MB")
        
        print(f"\nMemory Analysis:")
        print(f"  Small calls - Peak: {peak_memory_small:.1f} MB, Final: {final_memory_small:.1f} MB")
        print(f"  Large call - Peak: {post_large_memory:.1f} MB")
        print(f"  Memory efficiency: {peak_memory_small/post_large_memory:.2f}x ratio")


if __name__ == "__main__":
    print("Batch Processing Performance Investigation")
    print("=" * 50)
    print("This test investigates the performance issues mentioned in issue #859")
    print()
    
    # Run a quick test
    service = DetectionService()
    np.random.seed(42)
    test_data = np.random.randn(1000, 10)
    
    try:
        result = service.detect_anomalies(
            data=test_data,
            algorithm='iforest',
            contamination=0.1
        )
        print(f"✓ Quick test passed: {result.anomaly_count} anomalies detected")
        print("Ready to run comprehensive performance investigation")
    except Exception as e:
        print(f"✗ Quick test failed: {e}")