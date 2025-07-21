"""Test simplified PyOD adapter functionality."""

import pytest
import numpy as np
from algorithms.adapters.simple_pyod_adapter import SimplePyODAdapter, create_pyod_detector


class TestSimplePyODAdapter:
    """Test the simplified PyOD adapter."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create data with clear anomalies
        normal_data = np.random.randn(90, 5)
        anomalous_data = np.random.randn(10, 5) + 3
        self.test_data = np.vstack([normal_data, anomalous_data])
        
        # Smaller dataset for quick tests
        self.small_data = np.random.randn(20, 3)
        self.small_data[:2] += 2

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = SimplePyODAdapter()
        assert adapter.algorithm == "iforest"
        assert adapter.contamination == 0.1
        assert not adapter._fitted

    def test_initialization_with_params(self):
        """Test adapter initialization with parameters."""
        adapter = SimplePyODAdapter(
            algorithm="lof",
            contamination=0.2,
            n_neighbors=10
        )
        assert adapter.algorithm == "lof"
        assert adapter.contamination == 0.2
        assert adapter.kwargs["n_neighbors"] == 10

    def test_list_algorithms(self):
        """Test listing available algorithms."""
        algorithms = SimplePyODAdapter.list_algorithms()
        assert isinstance(algorithms, list)
        assert len(algorithms) > 20  # Should have many algorithms
        assert "iforest" in algorithms
        assert "lof" in algorithms
        assert "pca" in algorithms

    def test_iforest_algorithm(self):
        """Test IsolationForest algorithm."""
        adapter = SimplePyODAdapter(algorithm="iforest", contamination=0.1)
        
        # Test fit
        adapter.fit(self.test_data)
        assert adapter._fitted
        assert adapter._model is not None
        
        # Test predict
        predictions = adapter.predict(self.test_data)
        assert len(predictions) == len(self.test_data)
        assert predictions.dtype == np.integer
        assert all(p in [0, 1] for p in predictions)
        
        # Should detect some anomalies
        anomaly_count = np.sum(predictions)
        assert 0 < anomaly_count < len(self.test_data)

    def test_lof_algorithm(self):
        """Test LocalOutlierFactor algorithm."""
        adapter = SimplePyODAdapter(algorithm="lof", contamination=0.1)
        
        # Test fit_predict
        predictions = adapter.fit_predict(self.small_data)
        assert len(predictions) == len(self.small_data)
        assert predictions.dtype == np.integer
        assert all(p in [0, 1] for p in predictions)

    def test_pca_algorithm(self):
        """Test PCA algorithm."""
        adapter = SimplePyODAdapter(algorithm="pca", contamination=0.15)
        
        adapter.fit(self.small_data)
        predictions = adapter.predict(self.small_data)
        
        assert len(predictions) == len(self.small_data)
        assert predictions.dtype == np.integer

    def test_decision_function(self):
        """Test anomaly scoring functionality."""
        adapter = SimplePyODAdapter(algorithm="iforest")
        adapter.fit(self.test_data)
        
        scores = adapter.decision_function(self.test_data)
        assert len(scores) == len(self.test_data)
        assert scores.dtype == np.floating
        
        # Higher scores should indicate more anomalous samples
        assert np.any(scores > 0)

    def test_fit_predict_workflow(self):
        """Test the fit_predict method."""
        adapter = SimplePyODAdapter(algorithm="iforest", contamination=0.1)
        
        predictions = adapter.fit_predict(self.test_data)
        assert len(predictions) == len(self.test_data)
        assert adapter._fitted
        
        # Should be able to predict on same data after fit_predict
        new_predictions = adapter.predict(self.test_data[:20])  # Use same feature count
        assert len(new_predictions) == 20

    def test_prediction_without_fit_raises_error(self):
        """Test that prediction without fitting raises an error."""
        adapter = SimplePyODAdapter()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            adapter.predict(self.test_data)

    def test_decision_function_without_fit_raises_error(self):
        """Test that scoring without fitting raises an error."""
        adapter = SimplePyODAdapter()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            adapter.decision_function(self.test_data)

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm name raises an error."""
        adapter = SimplePyODAdapter(algorithm="invalid_algorithm")
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            adapter.fit(self.test_data)

    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        adapter = SimplePyODAdapter(algorithm="lof", contamination=0.2, n_neighbors=15)
        
        info = adapter.get_algorithm_info()
        assert info["algorithm"] == "lof"
        assert info["contamination"] == 0.2
        assert info["parameters"]["n_neighbors"] == 15
        assert not info["fitted"]
        assert isinstance(info["available_algorithms"], list)
        
        # After fitting
        adapter.fit(self.small_data)
        info_after = adapter.get_algorithm_info()
        assert info_after["fitted"]

    def test_create_pyod_detector_convenience_function(self):
        """Test the convenience function for creating detectors."""
        detector = create_pyod_detector("iforest", contamination=0.15)
        
        assert isinstance(detector, SimplePyODAdapter)
        assert detector.algorithm == "iforest"
        assert detector.contamination == 0.15
        
        # Should work normally
        predictions = detector.fit_predict(self.test_data)
        assert len(predictions) == len(self.test_data)

    def test_different_contamination_levels(self):
        """Test different contamination levels."""
        for contamination in [0.05, 0.1, 0.2, 0.3]:
            adapter = SimplePyODAdapter(algorithm="iforest", contamination=contamination)
            predictions = adapter.fit_predict(self.test_data)
            
            anomaly_rate = np.sum(predictions) / len(predictions)
            # Allow some tolerance around the contamination level
            assert 0.0 <= anomaly_rate <= 0.5

    @pytest.mark.parametrize("algorithm", ["iforest", "lof", "pca", "ocsvm", "hbos"])
    def test_multiple_algorithms(self, algorithm):
        """Test multiple PyOD algorithms."""
        adapter = SimplePyODAdapter(algorithm=algorithm, contamination=0.1)
        
        # Should not raise an error
        predictions = adapter.fit_predict(self.small_data)
        assert len(predictions) == len(self.small_data)
        assert all(p in [0, 1] for p in predictions)