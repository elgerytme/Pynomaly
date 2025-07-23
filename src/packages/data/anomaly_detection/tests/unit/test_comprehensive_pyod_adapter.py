"""Comprehensive test suite for ComprehensivePyODAdapter."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import pickle

from anomaly_detection.infrastructure.adapters.comprehensive_pyod_adapter import (
    ComprehensivePyODAdapter,
    AlgorithmCategory,
    AlgorithmInfo,
    PYOD_AVAILABLE
)


class TestComprehensivePyODAdapter:
    """Test suite for ComprehensivePyODAdapter."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        return np.random.randn(100, 5).astype(np.float32)
    
    @pytest.fixture
    def test_data(self):
        """Create sample test data."""
        np.random.seed(43)
        return np.random.randn(50, 5).astype(np.float32)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_init_default_algorithm(self):
        """Test initialization with default algorithm."""
        adapter = ComprehensivePyODAdapter()
        
        assert adapter.algorithm == "iforest"
        assert adapter._fitted is False
        assert adapter.model is None
        assert adapter._algorithm_info is not None
        assert adapter._algorithm_info.name == "iforest"
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_init_custom_algorithm(self):
        """Test initialization with custom algorithm."""
        adapter = ComprehensivePyODAdapter(
            algorithm="lof",
            n_neighbors=15,
            contamination=0.05
        )
        
        assert adapter.algorithm == "lof"
        assert adapter.parameters["n_neighbors"] == 15
        assert adapter.parameters["contamination"] == 0.05
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_init_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ComprehensivePyODAdapter(algorithm="invalid_algo")
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_algorithm_registry_initialization(self):
        """Test algorithm registry is properly initialized."""
        adapter = ComprehensivePyODAdapter()
        
        # Check that registry contains expected categories
        categories = set(info.category for info in adapter.available_algorithms.values())
        expected_categories = {
            AlgorithmCategory.PROXIMITY_BASED,
            AlgorithmCategory.LINEAR_MODELS,
            AlgorithmCategory.PROBABILISTIC,
            AlgorithmCategory.NEURAL_NETWORKS,
            AlgorithmCategory.ENSEMBLE
        }
        
        assert expected_categories.issubset(categories)
        
        # Check specific algorithms
        assert "iforest" in adapter.available_algorithms
        assert "lof" in adapter.available_algorithms
        assert "ocsvm" in adapter.available_algorithms
        assert "pca" in adapter.available_algorithms
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_set_defaults(self):
        """Test default parameter setting."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        # Check that defaults are set
        assert "contamination" in adapter.parameters
        assert "n_estimators" in adapter.parameters
        assert adapter.parameters["contamination"] == 0.1
        assert adapter.parameters["n_estimators"] == 100
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_fit(self, sample_data):
        """Test model fitting."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        result = adapter.fit(sample_data)
        
        assert result is adapter  # Method chaining
        assert adapter._fitted is True
        assert adapter.model is not None
        assert adapter._training_data is not None
        assert np.array_equal(adapter._training_data, sample_data)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_predict(self, sample_data, test_data):
        """Test prediction functionality."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        predictions = adapter.predict(test_data)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (test_data.shape[0],)
        assert np.all(np.isin(predictions, [0, 1]))  # Binary predictions
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available") 
    def test_predict_not_fitted(self, test_data):
        """Test prediction fails when model not fitted."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            adapter.predict(test_data)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_fit_predict(self, sample_data):
        """Test fit and predict in one step."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        predictions = adapter.fit_predict(sample_data)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (sample_data.shape[0],)
        assert adapter._fitted is True
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_decision_function(self, sample_data, test_data):
        """Test decision function (anomaly scores)."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        scores = adapter.decision_function(test_data)
        
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (test_data.shape[0],)
        assert np.all(np.isfinite(scores))
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_predict_proba(self, sample_data, test_data):
        """Test prediction probabilities."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        probas = adapter.predict_proba(test_data)
        
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (test_data.shape[0],)
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_predict_confidence(self, sample_data, test_data):
        """Test prediction confidence scores."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        confidence = adapter.predict_confidence(test_data)
        
        assert isinstance(confidence, np.ndarray)
        assert confidence.shape == (test_data.shape[0],)
        assert np.all(np.isfinite(confidence))
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        adapter = ComprehensivePyODAdapter(algorithm="lof")
        
        info = adapter.get_algorithm_info()
        
        assert isinstance(info, AlgorithmInfo)
        assert info.name == "lof"
        assert info.display_name == "Local Outlier Factor"
        assert info.category == AlgorithmCategory.PROXIMITY_BASED
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_parameters(self):
        """Test getting current parameters."""
        adapter = ComprehensivePyODAdapter(
            algorithm="iforest",
            n_estimators=200,
            contamination=0.05
        )
        
        params = adapter.get_parameters()
        
        assert isinstance(params, dict)
        assert params["n_estimators"] == 200
        assert params["contamination"] == 0.05
        
        # Ensure it's a copy
        params["n_estimators"] = 300
        assert adapter.parameters["n_estimators"] == 200
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_set_parameters(self, sample_data):
        """Test setting parameters."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        # Parameters should reset fitted state
        result = adapter.set_parameters(n_estimators=150, contamination=0.08)
        
        assert result is adapter  # Method chaining
        assert adapter.parameters["n_estimators"] == 150
        assert adapter.parameters["contamination"] == 0.08
        assert adapter._fitted is False
        assert adapter.model is None
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_model_info_not_fitted(self):
        """Test getting model info when not fitted."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        info = adapter.get_model_info()
        
        assert info["fitted"] is False
        assert info["algorithm"] == "iforest"
        assert "parameters" in info
        assert "algorithm_info" in info
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_model_info_fitted(self, sample_data):
        """Test getting model info when fitted."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest", contamination=0.1)
        adapter.fit(sample_data)
        
        info = adapter.get_model_info()
        
        assert info["fitted"] is True
        assert info["algorithm"] == "iforest"
        assert "contamination" in info
        assert info["algorithm_info"]["display_name"] == "Isolation Forest"
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_list_available_algorithms(self):
        """Test listing available algorithms."""
        algorithms = ComprehensivePyODAdapter.list_available_algorithms()
        
        assert isinstance(algorithms, dict)
        assert "iforest" in algorithms
        assert "lof" in algorithms
        
        # Check algorithm info structure
        iforest_info = algorithms["iforest"]
        assert "display_name" in iforest_info
        assert "category" in iforest_info
        assert "description" in iforest_info
        assert "default_parameters" in iforest_info
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_algorithms_by_category(self):
        """Test getting algorithms by category."""
        proximity_algorithms = ComprehensivePyODAdapter.get_algorithms_by_category(
            AlgorithmCategory.PROXIMITY_BASED
        )
        
        assert isinstance(proximity_algorithms, list)
        assert "iforest" in proximity_algorithms
        assert "lof" in proximity_algorithms
        assert "knn" in proximity_algorithms
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_recommended_algorithms(self):
        """Test algorithm recommendations."""
        # Test for small data with low complexity preference
        recommendations = ComprehensivePyODAdapter.get_recommended_algorithms(
            data_size="small",
            complexity_preference="low",
            interpretability_required=True
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Neural network algorithms should be filtered out due to interpretability requirement
        adapter = ComprehensivePyODAdapter()
        nn_algorithms = [
            name for name, info in adapter.available_algorithms.items()
            if info.category == AlgorithmCategory.NEURAL_NETWORKS
        ]
        
        for nn_algo in nn_algorithms:
            assert nn_algo not in recommendations
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_evaluate_algorithm_suitability(self):
        """Test algorithm suitability evaluation."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        # Test with reasonable data size
        evaluation = adapter.evaluate_algorithm_suitability(
            data_shape=(1000, 10),
            has_labels=False,
            streaming=False
        )
        
        assert "suitability_score" in evaluation
        assert "warnings" in evaluation
        assert "recommendations" in evaluation
        assert "algorithm_properties" in evaluation
        assert 0 <= evaluation["suitability_score"] <= 100
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_evaluate_algorithm_suitability_large_data(self):
        """Test suitability evaluation with large dataset."""
        adapter = ComprehensivePyODAdapter(algorithm="abod")  # High complexity algorithm
        
        evaluation = adapter.evaluate_algorithm_suitability(
            data_shape=(200000, 1000),  # Large, high-dimensional data
            streaming=True
        )
        
        # Should have warnings and lower suitability score
        assert evaluation["suitability_score"] < 100
        assert len(evaluation["warnings"]) > 0
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_save_load_model(self, sample_data):
        """Test model saving and loading."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest", n_estimators=50)
        adapter.fit(sample_data)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = Path(f.name)
        
        try:
            # Save model
            adapter.save_model(str(model_path))
            assert model_path.exists()
            
            # Load model
            loaded_adapter = ComprehensivePyODAdapter.load_model(str(model_path))
            
            assert loaded_adapter.algorithm == "iforest"
            assert loaded_adapter.parameters["n_estimators"] == 50
            assert loaded_adapter._fitted is True
            
            # Test that loaded model can make predictions
            test_data = np.random.randn(10, 5).astype(np.float32)
            predictions = loaded_adapter.predict(test_data)
            assert predictions.shape == (10,)
        
        finally:
            model_path.unlink(missing_ok=True)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_save_model_not_fitted(self):
        """Test saving unfitted model fails."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            adapter.save_model("test.pkl")
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_multiple_algorithms(self, sample_data):
        """Test multiple different algorithms."""
        algorithms_to_test = ["iforest", "lof", "pca", "hbos"]
        
        for algorithm in algorithms_to_test:
            adapter = ComprehensivePyODAdapter(algorithm=algorithm)
            
            # Should be able to fit and predict
            adapter.fit(sample_data)
            predictions = adapter.predict(sample_data[:10])
            
            assert adapter._fitted is True
            assert predictions.shape == (10,)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_neural_network_algorithms(self, sample_data):
        """Test neural network algorithms if available."""
        nn_algorithms = ComprehensivePyODAdapter.get_algorithms_by_category(
            AlgorithmCategory.NEURAL_NETWORKS
        )
        
        if nn_algorithms:
            # Test one neural network algorithm
            algorithm = nn_algorithms[0]
            adapter = ComprehensivePyODAdapter(
                algorithm=algorithm,
                epochs=2,  # Reduced for testing
                verbose=0
            )
            
            # Fit with small dataset for speed
            small_data = sample_data[:20]
            adapter.fit(small_data)
            
            predictions = adapter.predict(small_data[:5])
            assert predictions.shape == (5,)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_ensemble_algorithms(self, sample_data):
        """Test ensemble algorithms."""
        ensemble_algorithms = ComprehensivePyODAdapter.get_algorithms_by_category(
            AlgorithmCategory.ENSEMBLE
        )
        
        if ensemble_algorithms:
            algorithm = ensemble_algorithms[0]
            adapter = ComprehensivePyODAdapter(algorithm=algorithm)
            
            adapter.fit(sample_data)
            predictions = adapter.predict(sample_data[:10])
            
            assert predictions.shape == (10,)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_get_feature_importances(self, sample_data):
        """Test feature importance extraction."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        # Should return None when not fitted
        importances = adapter.get_feature_importances()
        assert importances is None
        
        # Fit model
        adapter.fit(sample_data)
        importances = adapter.get_feature_importances()
        
        # Isolation Forest doesn't typically have feature importances
        # but the method should not fail
        assert importances is None or isinstance(importances, np.ndarray)
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_error_handling_in_fit(self, sample_data):
        """Test error handling during model fitting."""
        # Create adapter with invalid parameters that should cause fitting to fail
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        
        # Mock the model creation to raise an exception
        with patch.object(adapter, '_create_model') as mock_create:
            mock_create.side_effect = ValueError("Invalid parameters")
            
            with pytest.raises(ValueError, match="Invalid parameters"):
                adapter.fit(sample_data)
            
            # Should not be marked as fitted
            assert adapter._fitted is False
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_error_handling_in_predict(self, sample_data, test_data):
        """Test error handling during prediction."""
        adapter = ComprehensivePyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        # Mock model predict to raise exception
        with patch.object(adapter.model, 'predict') as mock_predict:
            mock_predict.side_effect = RuntimeError("Prediction failed")
            
            with pytest.raises(RuntimeError, match="Prediction failed"):
                adapter.predict(test_data)
    
    def test_pyod_not_available(self):
        """Test behavior when PyOD is not available."""
        with patch('anomaly_detection.infrastructure.adapters.comprehensive_pyod_adapter.PYOD_AVAILABLE', False):
            with pytest.raises(ImportError, match="PyOD is required"):
                ComprehensivePyODAdapter()
    
    def test_list_algorithms_pyod_not_available(self):
        """Test algorithm listing when PyOD not available."""
        with patch('anomaly_detection.infrastructure.adapters.comprehensive_pyod_adapter.PYOD_AVAILABLE', False):
            result = ComprehensivePyODAdapter.list_available_algorithms()
            assert "error" in result
            assert "PyOD library not available" in result["error"]
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_algorithm_categories_complete(self):
        """Test that all algorithm categories are represented."""
        adapter = ComprehensivePyODAdapter()
        
        categories_found = set()
        for info in adapter.available_algorithms.values():
            categories_found.add(info.category)
        
        # Should have multiple categories
        assert len(categories_found) >= 4
        assert AlgorithmCategory.PROXIMITY_BASED in categories_found
        assert AlgorithmCategory.LINEAR_MODELS in categories_found
        assert AlgorithmCategory.PROBABILISTIC in categories_found
    
    @pytest.mark.skipif(not PYOD_AVAILABLE, reason="PyOD not available")
    def test_algorithm_info_completeness(self):
        """Test that algorithm info is complete."""
        adapter = ComprehensivePyODAdapter()
        
        for name, info in adapter.available_algorithms.items():
            assert isinstance(info, AlgorithmInfo)
            assert info.name == name
            assert info.display_name
            assert info.description
            assert isinstance(info.parameters, dict)
            assert info.computational_complexity in ["low", "medium", "high", "very_high"]
            assert info.memory_usage in ["low", "medium", "high", "very_high"]
            assert isinstance(info.requires_scaling, bool)
            assert isinstance(info.supports_streaming, bool)