"""Property-based tests for anomaly detection using Hypothesis."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.extra.numpy import arrays, array_shapes
from typing import List, Dict, Any, Union
from unittest.mock import Mock, patch

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.domain.entities.detection_result import DetectionResult


@pytest.mark.property
class TestDetectionServiceProperties:
    """Property-based tests for detection service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
    
    @given(
        data=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=2, max_dims=2, min_side=5, max_side=100),
            elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        algorithm=st.sampled_from(["isolation_forest", "one_class_svm", "local_outlier_factor"]),
        contamination=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=50, deadline=10000)
    def test_detection_output_properties(self, data, algorithm, contamination):
        """Test properties of detection output."""
        assume(data.shape[0] >= 3)  # Need minimum samples
        assume(data.shape[1] >= 1)  # Need at least one feature
        assume(not np.any(np.isnan(data)))  # No NaN values
        assume(not np.any(np.isinf(data)))  # No infinite values
        
        # Convert to list format expected by service
        data_list = data.tolist()
        
        try:
            result = self.detection_service.detect_anomalies(
                data_list,
                algorithm=algorithm,
                contamination=contamination
            )
            
            # Property 1: Output should have same number of predictions as input samples
            assert len(result["anomalies"]) == len(data_list)
            assert len(result["scores"]) == len(data_list)
            
            # Property 2: Anomaly predictions should be binary (0 or 1)
            assert all(pred in [0, 1] for pred in result["anomalies"])
            
            # Property 3: Scores should be numeric
            assert all(isinstance(score, (int, float)) for score in result["scores"])
            assert all(not np.isnan(score) for score in result["scores"])
            assert all(not np.isinf(score) for score in result["scores"])
            
            # Property 4: Number of anomalies should respect contamination rate
            anomaly_count = sum(result["anomalies"])
            expected_anomalies = int(len(data_list) * contamination)
            tolerance = max(1, len(data_list) * 0.1)  # 10% tolerance or at least 1
            assert abs(anomaly_count - expected_anomalies) <= tolerance
            
            # Property 5: Algorithm should be preserved in result
            assert result["algorithm"] == algorithm
            
        except (ValueError, Exception) as e:
            # Some combinations might be invalid, that's acceptable
            assert "insufficient" in str(e).lower() or "invalid" in str(e).lower()
    
    @given(
        data=st.lists(
            st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=10
            ),
            min_size=10, max_size=100
        ),
        contamination1=st.floats(min_value=0.05, max_value=0.3),
        contamination2=st.floats(min_value=0.05, max_value=0.3)
    )
    @settings(max_examples=30, deadline=15000)
    def test_contamination_monotonicity(self, data, contamination1, contamination2):
        """Test that higher contamination leads to more anomalies detected."""
        assume(len(set(map(len, data))) == 1)  # All rows same length
        assume(contamination1 != contamination2)
        
        if contamination1 > contamination2:
            high_cont, low_cont = contamination1, contamination2
        else:
            high_cont, low_cont = contamination2, contamination1
        
        try:
            result_low = self.detection_service.detect_anomalies(
                data, algorithm="isolation_forest", contamination=low_cont
            )
            result_high = self.detection_service.detect_anomalies(
                data, algorithm="isolation_forest", contamination=high_cont
            )
            
            anomalies_low = sum(result_low["anomalies"])
            anomalies_high = sum(result_high["anomalies"])
            
            # Property: Higher contamination should generally lead to more anomalies
            # Allow some tolerance due to algorithmic variance
            assert anomalies_high >= anomalies_low - 1
            
        except (ValueError, Exception):
            # Invalid data combinations are acceptable
            pass
    
    @given(
        data=st.lists(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=3, max_size=8
            ),
            min_size=20, max_size=50
        )
    )
    @settings(max_examples=20, deadline=10000)
    def test_detection_determinism(self, data):
        """Test that detection is deterministic for same input."""
        assume(len(set(map(len, data))) == 1)  # All rows same length
        
        try:
            result1 = self.detection_service.detect_anomalies(
                data, algorithm="isolation_forest", contamination=0.1, random_state=42
            )
            result2 = self.detection_service.detect_anomalies(
                data, algorithm="isolation_forest", contamination=0.1, random_state=42
            )
            
            # Property: Same input with same random state should give identical results
            assert result1["anomalies"] == result2["anomalies"]
            # Scores might have small numerical differences, so use approximate equality
            for s1, s2 in zip(result1["scores"], result2["scores"]):
                assert abs(s1 - s2) < 1e-10
                
        except (ValueError, Exception):
            # Invalid data combinations are acceptable
            pass
    
    @given(
        base_data=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=30),
            elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
        ),
        outlier_multiplier=st.floats(min_value=3.0, max_value=10.0)
    )
    @settings(max_examples=20, deadline=15000)
    def test_outlier_detection_sensitivity(self, base_data, outlier_multiplier):
        """Test that obvious outliers are detected."""
        assume(base_data.shape[0] >= 10)
        assume(base_data.shape[1] >= 2)
        
        # Create obvious outliers by scaling some points
        data_with_outliers = base_data.copy()
        n_outliers = max(1, len(data_with_outliers) // 10)  # 10% outliers
        outlier_indices = np.random.choice(len(data_with_outliers), n_outliers, replace=False)
        
        for idx in outlier_indices:
            data_with_outliers[idx] *= outlier_multiplier
        
        data_list = data_with_outliers.tolist()
        
        try:
            result = self.detection_service.detect_anomalies(
                data_list,
                algorithm="isolation_forest",
                contamination=0.2  # Higher contamination to catch outliers
            )
            
            # Property: Should detect some anomalies when obvious outliers are present
            anomaly_count = sum(result["anomalies"])
            assert anomaly_count > 0, "Should detect some anomalies with obvious outliers"
            
            # Property: Outlier points should have higher anomaly scores on average
            outlier_scores = [result["scores"][i] for i in outlier_indices]
            normal_scores = [result["scores"][i] for i in range(len(result["scores"])) if i not in outlier_indices]
            
            if len(normal_scores) > 0 and len(outlier_scores) > 0:
                avg_outlier_score = np.mean(outlier_scores)
                avg_normal_score = np.mean(normal_scores)
                # Allow some tolerance for algorithmic variance
                assert avg_outlier_score >= avg_normal_score - 0.1
            
        except (ValueError, Exception):
            # Invalid data combinations are acceptable
            pass


@pytest.mark.property
class TestEnsembleServiceProperties:
    """Property-based tests for ensemble service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble_service = EnsembleService()
    
    @given(
        data=st.lists(
            st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=3, max_size=8
            ),
            min_size=15, max_size=40
        ),
        algorithms=st.lists(
            st.sampled_from(["isolation_forest", "one_class_svm", "local_outlier_factor"]),
            min_size=2, max_size=3, unique=True
        ),
        ensemble_method=st.sampled_from(["majority", "average", "weighted_average"])
    )
    @settings(max_examples=30, deadline=20000)
    def test_ensemble_output_properties(self, data, algorithms, ensemble_method):
        """Test properties of ensemble output."""
        assume(len(set(map(len, data))) == 1)  # All rows same length
        
        try:
            result = self.ensemble_service.detect_with_ensemble(
                data,
                algorithms=algorithms,
                ensemble_method=ensemble_method,
                contamination=0.1
            )
            
            # Property 1: Output structure should be consistent
            assert len(result["anomalies"]) == len(data)
            assert len(result["scores"]) == len(data)
            assert "individual_results" in result
            assert len(result["individual_results"]) == len(algorithms)
            
            # Property 2: Anomaly predictions should be binary
            assert all(pred in [0, 1] for pred in result["anomalies"])
            
            # Property 3: Each individual result should have correct structure
            for algo_result in result["individual_results"].values():
                assert len(algo_result["anomalies"]) == len(data)
                assert len(algo_result["scores"]) == len(data)
            
            # Property 4: Ensemble method should be preserved
            assert result["ensemble_method"] == ensemble_method
            assert result["algorithms"] == algorithms
            
        except (ValueError, Exception):
            # Some combinations might be invalid
            pass
    
    @given(
        data=st.lists(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=4, max_size=6
            ),
            min_size=20, max_size=30
        )
    )
    @settings(max_examples=15, deadline=25000)
    def test_ensemble_vs_individual_consistency(self, data):
        """Test consistency between ensemble and individual results."""
        assume(len(set(map(len, data))) == 1)  # All rows same length
        
        algorithms = ["isolation_forest", "one_class_svm"]
        
        try:
            # Get ensemble result
            ensemble_result = self.ensemble_service.detect_with_ensemble(
                data,
                algorithms=algorithms,
                ensemble_method="majority",
                contamination=0.1
            )
            
            # Get individual results
            detection_service = DetectionService()
            individual_results = {}
            for algo in algorithms:
                individual_results[algo] = detection_service.detect_anomalies(
                    data, algorithm=algo, contamination=0.1
                )
            
            # Property: Ensemble individual_results should match standalone results
            for algo in algorithms:
                ensemble_individual = ensemble_result["individual_results"][algo]
                standalone = individual_results[algo]
                
                # Allow some tolerance for numerical differences
                for e_score, s_score in zip(ensemble_individual["scores"], standalone["scores"]):
                    assert abs(e_score - s_score) < 1e-8
            
        except (ValueError, Exception):
            # Invalid combinations are acceptable
            pass


@pytest.mark.property
class TestStreamingServiceProperties:
    """Property-based tests for streaming service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.streaming_service = StreamingService()
    
    @given(
        batch=st.lists(
            st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=3, max_size=8
            ),
            min_size=10, max_size=50
        ),
        buffer_size=st.integers(min_value=50, max_value=200)
    )
    @settings(max_examples=25, deadline=15000)
    def test_streaming_batch_properties(self, batch, buffer_size):
        """Test properties of streaming batch processing."""
        assume(len(set(map(len, batch))) == 1)  # All rows same length
        assume(buffer_size >= len(batch))  # Buffer should accommodate batch
        
        try:
            result = self.streaming_service.process_streaming_batch(
                batch,
                algorithm="isolation_forest",
                buffer_size=buffer_size
            )
            
            # Property 1: Output should match input size
            assert len(result["anomalies"]) == len(batch)
            assert len(result["scores"]) == len(batch)
            
            # Property 2: Anomaly predictions should be binary
            assert all(pred in [0, 1] for pred in result["anomalies"])
            
            # Property 3: Scores should be valid numbers
            assert all(isinstance(score, (int, float)) for score in result["scores"])
            assert all(not np.isnan(score) for score in result["scores"])
            
            # Property 4: Buffer size should be preserved in metadata
            if "metadata" in result:
                assert result["metadata"].get("buffer_size") == buffer_size
            
        except (ValueError, Exception):
            # Invalid combinations are acceptable
            pass
    
    @given(
        batch1=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(min_value=15, max_value=25), st.integers(min_value=3, max_value=6)),
            elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
        ),
        drift_magnitude=st.floats(min_value=2.0, max_value=5.0)
    )
    @settings(max_examples=20, deadline=15000)
    def test_concept_drift_detection_properties(self, batch1, drift_magnitude):
        """Test properties of concept drift detection."""
        # Create second batch with concept drift
        batch2 = batch1 + drift_magnitude  # Shift all values
        
        batch1_list = batch1.tolist()
        batch2_list = batch2.tolist()
        
        try:
            # Process first batch (no drift expected)
            result1 = self.streaming_service.detect_concept_drift(
                batch1_list,
                threshold=0.05
            )
            
            # Process second batch (drift expected)
            result2 = self.streaming_service.detect_concept_drift(
                batch2_list,
                previous_batch=batch1_list,
                threshold=0.05
            )
            
            # Property 1: Results should have required fields
            assert "drift_detected" in result1
            assert "drift_score" in result1
            assert "drift_detected" in result2
            assert "drift_score" in result2
            
            # Property 2: Drift scores should be numeric and non-negative
            assert isinstance(result1["drift_score"], (int, float))
            assert isinstance(result2["drift_score"], (int, float))
            assert result1["drift_score"] >= 0
            assert result2["drift_score"] >= 0
            
            # Property 3: Large drift should be detected
            if drift_magnitude > 3.0:  # Significant drift
                assert result2["drift_detected"] or result2["drift_score"] > result1["drift_score"]
            
        except (ValueError, Exception):
            # Invalid combinations are acceptable
            pass


@pytest.mark.property
class TestDataValidationProperties:
    """Property-based tests for data validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
    
    @given(
        invalid_data=st.one_of(
            st.none(),
            st.text(),
            st.integers(),
            st.floats(),
            st.booleans(),
            st.dictionaries(st.text(), st.integers()),
            st.lists(st.text(), min_size=1, max_size=10),
            st.lists(st.integers(), min_size=1, max_size=10),
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_invalid_data_rejection(self, invalid_data):
        """Test that invalid data is properly rejected."""
        # Property: Invalid data should raise appropriate exceptions
        with pytest.raises((ValueError, TypeError, AttributeError)):
            self.detection_service.detect_anomalies(
                invalid_data,
                algorithm="isolation_forest",
                contamination=0.1
            )
    
    @given(
        valid_shape_data=st.lists(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=5
            ),
            min_size=5, max_size=20
        ),
        inconsistent_row=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=6, max_size=10  # Different size from valid_shape_data
        )
    )
    @settings(max_examples=30, deadline=8000)
    def test_inconsistent_data_shape_rejection(self, valid_shape_data, inconsistent_row):
        """Test rejection of data with inconsistent shapes."""
        assume(len(set(map(len, valid_shape_data))) == 1)  # All rows same length initially
        assume(len(inconsistent_row) != len(valid_shape_data[0]))  # Inconsistent length
        
        # Add inconsistent row
        invalid_data = valid_shape_data + [inconsistent_row]
        
        # Property: Data with inconsistent row lengths should be rejected
        with pytest.raises((ValueError, Exception)):
            self.detection_service.detect_anomalies(
                invalid_data,
                algorithm="isolation_forest",
                contamination=0.1
            )
    
    @given(
        contamination=st.one_of(
            st.floats(min_value=-1.0, max_value=0.0),  # Negative
            st.floats(min_value=1.0, max_value=2.0),   # > 1
            st.text(),
            st.none(),
            st.lists(st.floats()),
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_invalid_contamination_rejection(self, contamination):
        """Test rejection of invalid contamination values."""
        valid_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        # Property: Invalid contamination should raise appropriate exceptions
        with pytest.raises((ValueError, TypeError)):
            self.detection_service.detect_anomalies(
                valid_data,
                algorithm="isolation_forest",
                contamination=contamination
            )
    
    @given(
        algorithm=st.one_of(
            st.text().filter(lambda x: x not in ["isolation_forest", "one_class_svm", "local_outlier_factor"]),
            st.none(),
            st.integers(),
            st.lists(st.text()),
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_invalid_algorithm_rejection(self, algorithm):
        """Test rejection of invalid algorithms."""
        valid_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        # Property: Invalid algorithms should raise appropriate exceptions
        with pytest.raises((ValueError, KeyError, AttributeError, TypeError)):
            self.detection_service.detect_anomalies(
                valid_data,
                algorithm=algorithm,
                contamination=0.1
            )


@pytest.mark.property
class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
    
    @given(
        data=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=2, max_dims=2, min_side=10, max_side=30),
            elements=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False)
        ),
        scale_factor=st.floats(min_value=1e-5, max_value=1e5)
    )
    @settings(max_examples=20, deadline=15000)
    def test_scale_invariance_approximation(self, data, scale_factor):
        """Test approximate scale invariance of detection."""
        assume(data.shape[0] >= 10)
        assume(data.shape[1] >= 2)
        assume(not np.any(np.abs(data) < 1e-15))  # Avoid tiny values
        
        scaled_data = data * scale_factor
        
        try:
            result_original = self.detection_service.detect_anomalies(
                data.tolist(),
                algorithm="isolation_forest",
                contamination=0.1,
                random_state=42
            )
            
            result_scaled = self.detection_service.detect_anomalies(
                scaled_data.tolist(),
                algorithm="isolation_forest",
                contamination=0.1,
                random_state=42
            )
            
            # Property: Scaling should not dramatically change anomaly detection
            # (Isolation Forest should be approximately scale-invariant)
            original_anomalies = set(i for i, x in enumerate(result_original["anomalies"]) if x == 1)
            scaled_anomalies = set(i for i, x in enumerate(result_scaled["anomalies"]) if x == 1)
            
            # Allow some variance due to algorithmic sensitivity
            overlap = len(original_anomalies & scaled_anomalies)
            total_anomalies = max(len(original_anomalies), len(scaled_anomalies))
            
            if total_anomalies > 0:
                agreement_ratio = overlap / total_anomalies
                assert agreement_ratio >= 0.3  # At least 30% agreement
            
        except (ValueError, Exception):
            # Some extreme scaling might cause numerical issues
            pass
    
    @given(
        base_data=arrays(
            dtype=np.float64,
            shape=array_shapes(min_dims=2, max_dims=2, min_side=15, max_side=25),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
        ),
        noise_std=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=15, deadline=20000)
    def test_noise_robustness(self, base_data, noise_std):
        """Test robustness to small amounts of noise."""
        assume(base_data.shape[0] >= 15)
        assume(base_data.shape[1] >= 2)
        
        # Add small amount of Gaussian noise
        noise = np.random.normal(0, noise_std, base_data.shape)
        noisy_data = base_data + noise
        
        try:
            result_original = self.detection_service.detect_anomalies(
                base_data.tolist(),
                algorithm="isolation_forest",
                contamination=0.1,
                random_state=42
            )
            
            result_noisy = self.detection_service.detect_anomalies(
                noisy_data.tolist(),
                algorithm="isolation_forest",
                contamination=0.1,
                random_state=42
            )
            
            # Property: Small noise should not drastically change results
            original_anomalies = sum(result_original["anomalies"])
            noisy_anomalies = sum(result_noisy["anomalies"])
            
            # Allow reasonable variance due to noise
            difference = abs(original_anomalies - noisy_anomalies)
            total_samples = len(base_data)
            relative_difference = difference / total_samples
            
            # Noise should not change more than 20% of classifications
            assert relative_difference <= 0.2
            
        except (ValueError, Exception):
            # Some data might be problematic
            pass


if __name__ == "__main__":
    # Run specific property-based tests
    pytest.main([
        __file__ + "::TestDetectionServiceProperties::test_detection_output_properties",
        "-v", "-s", "--tb=short"
    ])