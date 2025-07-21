"""Performance and benchmark tests for pynomaly_detection."""

import time
import pytest
import numpy as np

from pynomaly_detection import AnomalyDetector


class TestPerformance:
    """Performance tests for anomaly detection."""

    def setup_method(self):
        """Setup test data of various sizes."""
        np.random.seed(42)
        
        # Different dataset sizes for performance testing
        self.small_data = np.random.randn(100, 5)
        self.medium_data = np.random.randn(1000, 10)
        self.large_data = np.random.randn(5000, 20)
        
        # Add some outliers
        self.small_data[:10] += 3
        self.medium_data[:100] += 3
        self.large_data[:500] += 3

    def test_small_dataset_performance(self):
        """Test performance with small dataset."""
        detector = AnomalyDetector()
        
        start_time = time.time()
        predictions = detector.detect(self.small_data, contamination=0.1)
        elapsed_time = time.time() - start_time
        
        assert len(predictions) == len(self.small_data)
        assert elapsed_time < 1.0  # Should complete in less than 1 second

    def test_medium_dataset_performance(self):
        """Test performance with medium dataset."""
        detector = AnomalyDetector()
        
        start_time = time.time()
        predictions = detector.detect(self.medium_data, contamination=0.1)
        elapsed_time = time.time() - start_time
        
        assert len(predictions) == len(self.medium_data)
        assert elapsed_time < 5.0  # Should complete in less than 5 seconds

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        detector = AnomalyDetector()
        
        start_time = time.time()
        predictions = detector.detect(self.large_data, contamination=0.1)
        elapsed_time = time.time() - start_time
        
        assert len(predictions) == len(self.large_data)
        assert elapsed_time < 15.0  # Should complete in less than 15 seconds

    def test_fit_predict_vs_detect_performance(self):
        """Compare performance of fit+predict vs detect."""
        detector1 = AnomalyDetector()
        detector2 = AnomalyDetector()
        
        # Test fit + predict
        start_time = time.time()
        detector1.fit(self.medium_data, contamination=0.1)
        pred1 = detector1.predict(self.medium_data)
        fit_predict_time = time.time() - start_time
        
        # Test detect
        start_time = time.time()
        pred2 = detector2.detect(self.medium_data, contamination=0.1)
        detect_time = time.time() - start_time
        
        # Both should produce similar results
        assert len(pred1) == len(pred2)
        
        # Detect might be slightly slower due to internal fit_predict call
        assert abs(fit_predict_time - detect_time) < 2.0

    def test_repeated_predictions_performance(self):
        """Test performance of repeated predictions on same model."""
        detector = AnomalyDetector()
        detector.fit(self.medium_data, contamination=0.1)
        
        # Time multiple predictions
        start_time = time.time()
        for _ in range(10):
            predictions = detector.predict(self.medium_data)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 5.0  # Should complete 10 predictions in less than 5 seconds

    def test_dimensionality_performance(self):
        """Test performance with different dimensionalities."""
        detector = AnomalyDetector()
        
        # Test with increasing dimensions
        dimensions = [5, 10, 20, 50]
        times = []
        
        for dim in dimensions:
            data = np.random.randn(500, dim)
            data[:50] += 2  # Add outliers
            
            start_time = time.time()
            predictions = detector.detect(data, contamination=0.1)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            assert len(predictions) == 500
            assert elapsed_time < 10.0  # Should complete in reasonable time
        
        # Time should scale reasonably with dimensionality
        # (not testing strict scaling as it depends on algorithm internals)

    def test_contamination_parameter_performance(self):
        """Test that contamination parameter doesn't significantly affect performance."""
        detector = AnomalyDetector()
        
        contamination_levels = [0.05, 0.1, 0.2, 0.3]
        times = []
        
        for contamination in contamination_levels:
            start_time = time.time()
            predictions = detector.detect(self.medium_data, contamination=contamination)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            assert len(predictions) == len(self.medium_data)
        
        # All should complete in reasonable time
        assert all(t < 5.0 for t in times)
        
        # Times should be relatively similar
        assert max(times) - min(times) < 2.0

    def test_memory_usage_basic(self):
        """Basic memory usage test."""
        detector = AnomalyDetector()
        
        # This is a basic memory test - just ensure it doesn't crash
        predictions = detector.detect(self.large_data, contamination=0.1)
        
        assert len(predictions) == len(self.large_data)

    def test_batch_processing_performance(self):
        """Test performance of batch processing."""
        detector = AnomalyDetector()
        
        # Split large dataset into batches
        batch_size = 1000
        batches = [
            self.large_data[i:i+batch_size] 
            for i in range(0, len(self.large_data), batch_size)
        ]
        
        # Process all batches
        start_time = time.time()
        all_predictions = []
        for batch in batches:
            if len(batch) > 0:
                predictions = detector.detect(batch, contamination=0.1)
                all_predictions.extend(predictions)
        batch_time = time.time() - start_time
        
        # Compare with single large processing
        start_time = time.time()
        single_predictions = detector.detect(self.large_data, contamination=0.1)
        single_time = time.time() - start_time
        
        assert len(all_predictions) == len(single_predictions)
        assert len(all_predictions) == len(self.large_data)
        
        # Batch processing might be slower due to overhead
        # but should still complete in reasonable time
        assert batch_time < single_time * 5  # More lenient for CI environments

    def test_random_state_performance(self):
        """Test that random state doesn't affect performance."""
        detector = AnomalyDetector()
        
        # Test with random state
        start_time = time.time()
        pred1 = detector.detect(self.medium_data, contamination=0.1, random_state=42)
        time_with_random_state = time.time() - start_time
        
        # Test without random state
        start_time = time.time()
        pred2 = detector.detect(self.medium_data, contamination=0.1)
        time_without_random_state = time.time() - start_time
        
        assert len(pred1) == len(pred2)
        
        # Times should be similar
        assert abs(time_with_random_state - time_without_random_state) < 1.0

    def test_concurrent_detectors_performance(self):
        """Test performance with multiple concurrent detectors."""
        import threading
        
        results = []
        
        def run_detection(data, result_list):
            detector = AnomalyDetector()
            predictions = detector.detect(data, contamination=0.1)
            result_list.append(predictions)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            data_subset = self.medium_data[i*300:(i+1)*300]
            thread = threading.Thread(target=run_detection, args=(data_subset, results))
            threads.append(thread)
        
        # Run concurrently
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        elapsed_time = time.time() - start_time
        
        assert len(results) == 3
        assert all(len(r) == 300 for r in results)
        assert elapsed_time < 10.0  # Should complete in reasonable time

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000])
    def test_scaling_performance(self, n_samples):
        """Test performance scaling with different sample sizes."""
        data = np.random.randn(n_samples, 10)
        data[:max(1, n_samples//10)] += 3  # Add outliers
        
        detector = AnomalyDetector()
        
        start_time = time.time()
        predictions = detector.detect(data, contamination=0.1)
        elapsed_time = time.time() - start_time
        
        assert len(predictions) == n_samples
        
        # Time should scale reasonably (not testing strict O(n) as it depends on algorithm)
        expected_max_time = n_samples / 100  # Rough heuristic
        assert elapsed_time < max(1.0, expected_max_time)

    def test_stress_test_repeated_operations(self):
        """Stress test with repeated operations."""
        detector = AnomalyDetector()
        
        # Perform many repeated operations
        start_time = time.time()
        for i in range(50):
            # Vary the data slightly each time
            data = self.small_data + np.random.randn(*self.small_data.shape) * 0.1
            predictions = detector.detect(data, contamination=0.1)
            assert len(predictions) == len(self.small_data)
        
        elapsed_time = time.time() - start_time
        
        # Should complete 50 operations in reasonable time
        assert elapsed_time < 30.0

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up."""
        # Create and destroy many detectors
        for i in range(10):
            detector = AnomalyDetector()
            predictions = detector.detect(self.medium_data, contamination=0.1)
            assert len(predictions) == len(self.medium_data)
            del detector
            
        # If we reach here without memory issues, test passes
        assert True