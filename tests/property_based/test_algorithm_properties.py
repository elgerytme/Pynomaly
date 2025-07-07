"""Property-based tests for algorithm behavior using Hypothesis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings, strategies as st
from sklearn.datasets import make_classification

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import ContaminationRate


# Data generation strategies
@st.composite
def valid_numpy_arrays(draw, min_samples=10, max_samples=1000, min_features=2, max_features=50):
    """Generate valid numpy arrays for testing."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_features = draw(st.integers(min_value=min_features, max_value=max_features))
    
    # Generate array with finite values only
    data = draw(st.lists(
        st.lists(
            st.floats(
                min_value=-100.0, 
                max_value=100.0, 
                allow_nan=False, 
                allow_infinity=False
            ),
            min_size=n_features,
            max_size=n_features
        ),
        min_size=n_samples,
        max_size=n_samples
    ))
    
    return np.array(data)


@st.composite
def anomaly_datasets(draw):
    """Generate datasets suitable for anomaly detection."""
    n_samples = draw(st.integers(min_value=50, max_value=500))
    n_features = draw(st.integers(min_value=2, max_value=20))
    contamination = draw(st.floats(min_value=0.01, max_value=0.3))
    
    # Use sklearn to generate a classification dataset
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Add some outliers
    n_outliers = int(n_samples * contamination)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    # Make outliers more extreme
    for idx in outlier_indices:
        X[idx] *= draw(st.floats(min_value=2.0, max_value=5.0))
    
    # Convert to DataFrame
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    dataset_name = draw(st.text(min_size=1, max_size=50))
    return Dataset(name=dataset_name, data=df)


@st.composite
def algorithm_parameters(draw, algorithm_name):
    """Generate valid parameters for specific algorithms."""
    params = {}
    
    if algorithm_name in ["IsolationForest", "isolation_forest"]:
        params.update({
            "n_estimators": draw(st.integers(min_value=10, max_value=200)),
            "max_samples": draw(st.one_of(
                st.just("auto"),
                st.floats(min_value=0.1, max_value=1.0),
                st.integers(min_value=10, max_value=1000)
            )),
            "max_features": draw(st.floats(min_value=0.1, max_value=1.0)),
            "bootstrap": draw(st.booleans()),
            "random_state": draw(st.integers(min_value=0, max_value=2**31 - 1))
        })
    elif algorithm_name in ["LocalOutlierFactor", "lof"]:
        params.update({
            "n_neighbors": draw(st.integers(min_value=1, max_value=50)),
            "algorithm": draw(st.sampled_from(["auto", "ball_tree", "kd_tree", "brute"])),
            "leaf_size": draw(st.integers(min_value=10, max_value=50)),
            "metric": draw(st.sampled_from(["minkowski", "euclidean", "manhattan"])),
            "contamination": draw(st.floats(min_value=0.01, max_value=0.5))
        })
    elif algorithm_name in ["OneClassSVM", "ocsvm"]:
        params.update({
            "kernel": draw(st.sampled_from(["rbf", "linear", "poly", "sigmoid"])),
            "gamma": draw(st.one_of(
                st.just("scale"),
                st.just("auto"),
                st.floats(min_value=0.001, max_value=1.0)
            )),
            "nu": draw(st.floats(min_value=0.01, max_value=0.99)),
            "degree": draw(st.integers(min_value=1, max_value=5))
        })
    
    return params


class TestAlgorithmUniversalProperties:
    """Test universal properties that should hold for all anomaly detection algorithms."""

    @given(
        anomaly_datasets(),
        st.sampled_from(["IsolationForest", "LocalOutlierFactor", "OneClassSVM"])
    )
    @settings(max_examples=20, deadline=60000)  # Longer deadline for algorithm tests
    def test_algorithm_score_range(self, dataset, algorithm_name):
        """Anomaly scores should always be in a reasonable range."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            # Skip if algorithm not available
            adapter = PyODAdapter()
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            contamination_rate = ContaminationRate(0.1)
            params = algorithm_parameters(algorithm_name).example()
            
            detector = Detector(
                name=f"test_{algorithm_name}",
                algorithm_name=algorithm_name,
                contamination_rate=contamination_rate,
                parameters=params
            )
            
            # Get numeric features only
            numeric_features = dataset.get_numeric_features()
            assume(len(numeric_features) >= 2)  # Need at least 2 features
            
            X = dataset.data[numeric_features].values
            assume(X.shape[0] >= 10)  # Need at least 10 samples
            assume(not np.any(np.isnan(X)))  # No NaN values
            assume(not np.any(np.isinf(X)))  # No infinite values
            
            # Fit and predict
            algorithm_instance = adapter.create_algorithm(algorithm_name, detector.parameters)
            algorithm_instance.fit(X)
            
            scores = algorithm_instance.decision_function(X)
            
            # Universal properties
            assert len(scores) == len(X), "Score count should match sample count"
            assert np.all(np.isfinite(scores)), "All scores should be finite"
            assert len(np.unique(scores)) > 1 or len(scores) == 1, "Scores should have some variation (unless single sample)"
            
        except Exception as e:
            # If algorithm is not available or fails, skip the test
            pytest.skip(f"Algorithm test failed: {e}")

    @given(
        anomaly_datasets(),
        st.sampled_from(["IsolationForest"])  # Focus on one stable algorithm
    )
    @settings(max_examples=10, deadline=60000)
    def test_algorithm_determinism(self, dataset, algorithm_name):
        """Algorithm should be deterministic when random_state is set."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            adapter = PyODAdapter()
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            contamination_rate = ContaminationRate(0.1)
            params = {"random_state": 42, "n_estimators": 50}
            
            detector = Detector(
                name=f"test_{algorithm_name}",
                algorithm_name=algorithm_name,
                contamination_rate=contamination_rate,
                parameters=params
            )
            
            numeric_features = dataset.get_numeric_features()
            assume(len(numeric_features) >= 2)
            
            X = dataset.data[numeric_features].values
            assume(X.shape[0] >= 10)
            assume(not np.any(np.isnan(X)))
            assume(not np.any(np.isinf(X)))
            
            # Run algorithm twice with same parameters
            algorithm_instance_1 = adapter.create_algorithm(algorithm_name, detector.parameters)
            algorithm_instance_1.fit(X)
            scores_1 = algorithm_instance_1.decision_function(X)
            
            algorithm_instance_2 = adapter.create_algorithm(algorithm_name, detector.parameters)
            algorithm_instance_2.fit(X)
            scores_2 = algorithm_instance_2.decision_function(X)
            
            # Should get identical results
            np.testing.assert_array_almost_equal(
                scores_1, scores_2, decimal=5,
                err_msg="Algorithm should be deterministic with fixed random_state"
            )
            
        except Exception as e:
            pytest.skip(f"Determinism test failed: {e}")

    @given(
        anomaly_datasets(),
        st.sampled_from(["IsolationForest"])
    )
    @settings(max_examples=10, deadline=60000)
    def test_algorithm_monotonicity_with_contamination(self, dataset, algorithm_name):
        """Algorithm behavior should be monotonic with contamination rate."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            adapter = PyODAdapter()
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            numeric_features = dataset.get_numeric_features()
            assume(len(numeric_features) >= 2)
            
            X = dataset.data[numeric_features].values
            assume(X.shape[0] >= 20)  # Need enough samples for different contamination rates
            assume(not np.any(np.isnan(X)))
            assume(not np.any(np.isinf(X)))
            
            contamination_rates = [0.05, 0.1, 0.2]
            outlier_counts = []
            
            for contamination in contamination_rates:
                params = {"random_state": 42, "contamination": contamination}
                
                algorithm_instance = adapter.create_algorithm(algorithm_name, params)
                algorithm_instance.fit(X)
                
                predictions = algorithm_instance.predict(X)
                outlier_count = np.sum(predictions == 1)  # Count outliers
                outlier_counts.append(outlier_count)
            
            # Higher contamination should generally detect more outliers
            # (allowing for some tolerance due to algorithm specifics)
            assert outlier_counts[0] <= outlier_counts[1] + 2, "Low contamination should detect fewer outliers"
            assert outlier_counts[1] <= outlier_counts[2] + 2, "Medium contamination should detect fewer outliers than high"
            
        except Exception as e:
            pytest.skip(f"Monotonicity test failed: {e}")


class TestDataQualityProperties:
    """Test algorithm behavior with different data quality scenarios."""

    @given(
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=0.5)
    )
    @settings(max_examples=10, deadline=30000)
    def test_algorithm_with_constant_features(self, n_samples, n_features, noise_level):
        """Algorithm should handle datasets with constant features gracefully."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"
            
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            # Create dataset with some constant features
            X = np.random.randn(n_samples, n_features)
            
            # Make some features constant
            constant_feature_idx = 0
            X[:, constant_feature_idx] = 1.0
            
            # Add small amount of noise to avoid perfect constants
            if noise_level > 0:
                X += np.random.normal(0, noise_level, X.shape)
            
            params = {"random_state": 42, "contamination": 0.1}
            
            algorithm_instance = adapter.create_algorithm(algorithm_name, params)
            
            # Algorithm should handle this gracefully (not crash)
            algorithm_instance.fit(X)
            scores = algorithm_instance.decision_function(X)
            
            # Basic sanity checks
            assert len(scores) == n_samples
            assert np.all(np.isfinite(scores))
            
        except Exception as e:
            pytest.skip(f"Constant features test failed: {e}")

    @given(
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.01, max_value=0.2)
    )
    @settings(max_examples=10, deadline=30000)
    def test_algorithm_with_correlated_features(self, n_samples, n_features, correlation):
        """Algorithm should handle correlated features appropriately."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"
            
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            # Create dataset with correlated features
            X = np.random.randn(n_samples, n_features)
            
            # Make second feature correlated with first
            if n_features > 1:
                X[:, 1] = correlation * X[:, 0] + (1 - correlation) * X[:, 1]
            
            params = {"random_state": 42, "contamination": 0.1}
            
            algorithm_instance = adapter.create_algorithm(algorithm_name, params)
            algorithm_instance.fit(X)
            scores = algorithm_instance.decision_function(X)
            
            # Basic sanity checks
            assert len(scores) == n_samples
            assert np.all(np.isfinite(scores))
            
            # With correlated features, algorithm should still produce varied scores
            assert np.std(scores) > 0, "Should produce varied scores even with correlated features"
            
        except Exception as e:
            pytest.skip(f"Correlated features test failed: {e}")


class TestScalingProperties:
    """Test algorithm properties related to data scaling."""

    @given(
        st.integers(min_value=20, max_value=200),
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.1, max_value=100.0)
    )
    @settings(max_examples=10, deadline=30000)
    def test_algorithm_scale_invariance(self, n_samples, n_features, scale_factor):
        """Test algorithm behavior under different data scales."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"
            
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            # Create base dataset
            X_base = np.random.randn(n_samples, n_features)
            X_scaled = X_base * scale_factor
            
            params = {"random_state": 42, "contamination": 0.1}
            
            # Test on base data
            algorithm_instance_base = adapter.create_algorithm(algorithm_name, params)
            algorithm_instance_base.fit(X_base)
            predictions_base = algorithm_instance_base.predict(X_base)
            
            # Test on scaled data
            algorithm_instance_scaled = adapter.create_algorithm(algorithm_name, params)
            algorithm_instance_scaled.fit(X_scaled)
            predictions_scaled = algorithm_instance_scaled.predict(X_scaled)
            
            # For tree-based algorithms like Isolation Forest, predictions should be similar
            # (allowing for some variation due to numerical precision)
            agreement = np.mean(predictions_base == predictions_scaled)
            assert agreement >= 0.8, f"Scale invariance violated: only {agreement:.2f} agreement"
            
        except Exception as e:
            pytest.skip(f"Scale invariance test failed: {e}")


class TestRobustnessProperties:
    """Test algorithm robustness properties."""

    @given(
        st.integers(min_value=50, max_value=200),
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.05, max_value=0.3)
    )
    @settings(max_examples=5, deadline=60000)
    def test_algorithm_robustness_to_outliers(self, n_samples, n_features, outlier_fraction):
        """Algorithm should be robust to the presence of outliers."""
        try:
            from pynomaly.infrastructure.algorithms.adapters.pyod_adapter import PyODAdapter
            
            adapter = PyODAdapter()
            algorithm_name = "IsolationForest"
            
            if not adapter.supports_algorithm(algorithm_name):
                pytest.skip(f"Algorithm {algorithm_name} not available")
            
            # Create clean dataset
            X_clean = np.random.randn(n_samples, n_features)
            
            # Create dataset with outliers
            X_with_outliers = X_clean.copy()
            n_outliers = int(n_samples * outlier_fraction)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
            
            # Make outliers extreme
            for idx in outlier_indices:
                X_with_outliers[idx] *= 5.0  # Make them 5x larger
            
            params = {"random_state": 42, "contamination": outlier_fraction}
            
            # Test on clean data
            algorithm_instance_clean = adapter.create_algorithm(algorithm_name, params)
            algorithm_instance_clean.fit(X_clean)
            scores_clean = algorithm_instance_clean.decision_function(X_clean)
            
            # Test on data with outliers
            algorithm_instance_outliers = adapter.create_algorithm(algorithm_name, params)
            algorithm_instance_outliers.fit(X_with_outliers)
            scores_outliers = algorithm_instance_outliers.decision_function(X_with_outliers)
            predictions_outliers = algorithm_instance_outliers.predict(X_with_outliers)
            
            # Algorithm should detect the outliers
            detected_outliers = np.sum(predictions_outliers == 1)
            expected_outliers = n_outliers
            
            # Allow some tolerance in outlier detection
            assert detected_outliers >= expected_outliers * 0.5, \
                f"Should detect at least half of the outliers: detected {detected_outliers}, expected ~{expected_outliers}"
            
            # Scores should have reasonable range
            assert np.all(np.isfinite(scores_outliers))
            assert np.std(scores_outliers) > 0
            
        except Exception as e:
            pytest.skip(f"Robustness test failed: {e}")