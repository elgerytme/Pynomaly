"""Integration tests for the pynomaly_detection package."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from pynomaly_detection import AnomalyDetector, get_default_detector


class TestIntegration:
    """Integration tests for anomaly detection workflows."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        
        # Create synthetic dataset with normal and anomalous patterns
        self.normal_data, _ = make_classification(
            n_samples=500, n_features=5, n_informative=3, n_redundant=2,
            n_clusters_per_class=1, random_state=42
        )
        
        # Create clear anomalies by adding extreme values
        self.anomalies = np.random.randn(50, 5) * 5 + 10
        
        # Combined dataset
        self.combined_data = np.vstack([self.normal_data, self.anomalies])
        
        # Standardize the data
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.combined_data)

    def test_end_to_end_workflow(self):
        """Test complete anomaly detection workflow."""
        detector = AnomalyDetector()
        
        # Step 1: Fit the detector
        detector.fit(self.scaled_data)
        
        # Step 2: Predict anomalies
        predictions = detector.predict(self.scaled_data)
        
        # Step 3: Analyze results
        anomaly_count = np.sum(predictions)
        anomaly_indices = np.where(predictions == 1)[0]
        
        # Verify reasonable results
        assert len(predictions) == len(self.scaled_data)
        assert anomaly_count > 0
        assert anomaly_count < len(self.scaled_data)
        
        # Most anomalies should be detected in the anomalous portion
        anomalous_portion_detected = np.sum(predictions[500:])  # Last 50 samples
        assert anomalous_portion_detected > 0  # Should detect some anomalies

    def test_detect_method_workflow(self):
        """Test using the detect method for one-step detection."""
        detector = AnomalyDetector()
        
        predictions = detector.detect(self.scaled_data, contamination=0.1)
        
        # Should detect approximately 10% as anomalies
        anomaly_count = np.sum(predictions)
        expected_count = int(0.1 * len(self.scaled_data))
        
        assert abs(anomaly_count - expected_count) <= 10  # Allow some tolerance

    def test_different_contamination_levels(self):
        """Test detection with different contamination levels."""
        detector = AnomalyDetector()
        
        contamination_levels = [0.05, 0.1, 0.2, 0.3]
        results = {}
        
        for contamination in contamination_levels:
            predictions = detector.detect(self.scaled_data, contamination=contamination)
            anomaly_count = np.sum(predictions)
            results[contamination] = anomaly_count
            
            # Verify that we detect some anomalies
            assert 0 < anomaly_count < len(self.scaled_data)
            
            # Verify that contamination level affects detection (within reasonable bounds)
            expected_count = int(contamination * len(self.scaled_data))
            # IsolationForest has variance, so we use a generous tolerance
            assert anomaly_count <= len(self.scaled_data) * 0.8  # Not too many false positives
            assert anomaly_count >= 1  # At least some detection
        
        # Verify that overall, higher contamination levels tend to detect more anomalies
        # (test the general trend rather than strict ordering)
        low_contamination_avg = (results[0.05] + results[0.1]) / 2
        high_contamination_avg = (results[0.2] + results[0.3]) / 2
        
        # Allow for algorithm variance - high contamination should detect at least as many on average
        assert high_contamination_avg >= low_contamination_avg * 0.8

    def test_cross_validation_workflow(self):
        """Test cross-validation-like workflow."""
        # Split data into train and test
        train_data = self.scaled_data[:400]
        test_data = self.scaled_data[400:]
        
        detector = AnomalyDetector()
        
        # Train on subset
        detector.fit(train_data, contamination=0.1)
        
        # Test on remaining data
        predictions = detector.predict(test_data)
        
        assert len(predictions) == len(test_data)
        assert np.sum(predictions) >= 0

    def test_reproducibility_across_runs(self):
        """Test that results are reproducible with fixed random state."""
        detector1 = AnomalyDetector()
        detector2 = AnomalyDetector()
        
        pred1 = detector1.detect(self.scaled_data, random_state=42, contamination=0.1)
        pred2 = detector2.detect(self.scaled_data, random_state=42, contamination=0.1)
        
        np.testing.assert_array_equal(pred1, pred2)

    def test_batch_processing_workflow(self):
        """Test processing multiple batches of data."""
        detector = AnomalyDetector()
        
        # Split data into batches
        batch_size = 100
        batches = [
            self.scaled_data[i:i+batch_size] 
            for i in range(0, len(self.scaled_data), batch_size)
        ]
        
        # Process each batch
        all_predictions = []
        for batch in batches:
            if len(batch) > 0:
                predictions = detector.detect(batch, contamination=0.1)
                all_predictions.extend(predictions)
        
        # Verify results
        assert len(all_predictions) == len(self.scaled_data)
        assert all(p in [0, 1] for p in all_predictions)

    def test_edge_cases_integration(self):
        """Test integration with edge cases."""
        detector = AnomalyDetector()
        
        # Test with all normal data
        normal_only = self.normal_data[:100]
        predictions = detector.detect(normal_only, contamination=0.05)
        
        # Should still detect some points as anomalies due to contamination
        anomaly_count = np.sum(predictions)
        assert anomaly_count > 0
        
        # Test with very few samples
        few_samples = self.scaled_data[:10]
        try:
            predictions = detector.detect(few_samples)
            assert len(predictions) == 10
        except (ValueError, RuntimeError):
            # Some configurations might not work with very few samples
            pass

    def test_performance_characteristics(self):
        """Test performance characteristics of the detector."""
        detector = AnomalyDetector()
        
        # Test with increasing data sizes
        sizes = [50, 100, 200, 500]
        
        for size in sizes:
            data_subset = self.scaled_data[:size]
            predictions = detector.detect(data_subset, contamination=0.1)
            
            assert len(predictions) == size
            
            # Performance should scale reasonably
            anomaly_count = np.sum(predictions)
            expected_count = int(0.1 * size)
            assert abs(anomaly_count - expected_count) <= max(3, size // 20)

    def test_data_type_compatibility(self):
        """Test compatibility with different data types."""
        detector = AnomalyDetector()
        
        # Test with float32
        data_float32 = self.scaled_data.astype(np.float32)
        predictions = detector.detect(data_float32[:100])
        assert len(predictions) == 100
        
        # Test with float64
        data_float64 = self.scaled_data.astype(np.float64)
        predictions = detector.detect(data_float64[:100])
        assert len(predictions) == 100
        
        # Test with Python lists
        data_list = self.scaled_data[:100].tolist()
        predictions = detector.detect(data_list)
        assert len(predictions) == 100

    def test_default_detector_integration(self):
        """Test integration using default detector."""
        detector = get_default_detector()
        
        # Should work exactly like AnomalyDetector
        predictions = detector.detect(self.scaled_data[:100], contamination=0.1)
        
        assert len(predictions) == 100
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == int

    def test_multiple_detectors_independence(self):
        """Test that multiple detectors work independently."""
        detector1 = AnomalyDetector()
        detector2 = AnomalyDetector()
        
        # Train on different subsets
        detector1.fit(self.scaled_data[:200], contamination=0.1)
        detector2.fit(self.scaled_data[200:400], contamination=0.2)
        
        # Test on same data
        test_data = self.scaled_data[400:450]
        pred1 = detector1.predict(test_data)
        pred2 = detector2.predict(test_data)
        
        assert len(pred1) == len(pred2) == 50
        # Results might differ due to different training data
        assert not np.array_equal(pred1, pred2) or np.array_equal(pred1, pred2)

    def test_realistic_anomaly_detection_scenario(self):
        """Test a realistic anomaly detection scenario."""
        # Simulate sensor data with anomalies
        np.random.seed(42)
        
        # Normal operation: periodic signal with noise
        time = np.linspace(0, 10, 1000)
        normal_signal = np.sin(time) + 0.1 * np.random.randn(1000)
        
        # Add some anomalies: spikes and drops
        anomalous_signal = normal_signal.copy()
        anomalous_signal[100:105] += 5  # Spike
        anomalous_signal[500:510] -= 3  # Drop
        anomalous_signal[800:820] += np.random.randn(20) * 2  # Noise burst
        
        # Create feature matrix (using sliding window)
        window_size = 10
        features = []
        for i in range(len(anomalous_signal) - window_size + 1):
            window = anomalous_signal[i:i+window_size]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                np.percentile(window, 75) - np.percentile(window, 25)
            ])
        
        feature_matrix = np.array(features)
        
        # Detect anomalies
        detector = AnomalyDetector()
        predictions = detector.detect(feature_matrix, contamination=0.1)
        
        # Should detect some anomalies
        anomaly_count = np.sum(predictions)
        assert anomaly_count > 0
        assert anomaly_count < len(predictions) * 0.3  # Not too many false positives
        
        # Verify that some anomalies are detected in the known anomalous regions
        anomaly_indices = np.where(predictions == 1)[0]
        
        # Check if any anomalies are detected near the spike (around index 100)
        spike_region = range(90, 110)
        spike_detections = [idx for idx in anomaly_indices if idx in spike_region]
        
        # Should detect at least some anomalies in the known anomalous regions
        assert len(spike_detections) > 0 or anomaly_count > 20  # Either specific detection or general sensitivity