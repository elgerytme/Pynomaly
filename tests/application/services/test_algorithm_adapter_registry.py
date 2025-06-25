"""Tests for Algorithm Adapter Registry with real PyOD integration."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from pynomaly.application.services.algorithm_adapter_registry import (
    AlgorithmAdapterRegistry,
    PyODAlgorithmAdapter,
    SklearnAlgorithmAdapter
)
from pynomaly.domain.entities import Detector, Dataset
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore
from pynomaly.domain.exceptions import InvalidAlgorithmError, FittingError


class TestPyODAlgorithmAdapter:
    """Test PyOD algorithm adapter with real algorithms."""

    @pytest.fixture
    def adapter(self):
        """PyOD algorithm adapter."""
        return PyODAlgorithmAdapter()

    @pytest.fixture
    def detector(self):
        """Isolation Forest detector."""
        return Detector(
            name="Test IForest",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 100, "random_state": 42}
        )

    @pytest.fixture
    def dataset(self):
        """Test dataset with outliers."""
        # Create dataset with clear outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 2))
        outliers = np.random.normal(5, 1, (10, 2))  # Clear outliers
        
        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=['feature1', 'feature2'])
        
        return Dataset(name="Test Data", data=df)

    def test_create_isolation_forest_instance(self, adapter, detector):
        """Test creating PyOD Isolation Forest instance."""
        # Act
        algorithm = adapter._create_algorithm_instance(detector)
        
        # Assert
        assert algorithm is not None
        assert hasattr(algorithm, 'fit')
        assert hasattr(algorithm, 'predict')
        assert hasattr(algorithm, 'decision_function')
        
        # Check parameters were set correctly
        assert algorithm.contamination == 0.1
        assert algorithm.n_estimators == 100
        assert algorithm.random_state == 42

    def test_fit_algorithm_with_real_data(self, adapter, detector, dataset):
        """Test fitting PyOD algorithm with real data."""
        # Arrange
        algorithm = adapter._create_algorithm_instance(detector)
        feature_data = adapter._prepare_data(dataset)
        
        # Act
        fitted_algorithm = adapter._fit_algorithm(algorithm, feature_data)
        
        # Assert
        assert fitted_algorithm is not None
        assert hasattr(fitted_algorithm, 'decision_function')
        assert hasattr(fitted_algorithm, 'predict')

    def test_predict_with_real_algorithm(self, adapter, detector, dataset):
        """Test prediction with real PyOD algorithm."""
        # Arrange
        adapter.fit(detector, dataset)
        
        # Act
        predictions = adapter.predict(detector, dataset)
        
        # Assert
        assert len(predictions) == len(dataset.data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should detect some outliers (we have clear outliers in the data)
        assert sum(predictions) > 0
        assert sum(predictions) < len(predictions)  # Not all should be outliers

    def test_score_with_real_algorithm(self, adapter, detector, dataset):
        """Test scoring with real PyOD algorithm."""
        # Arrange
        adapter.fit(detector, dataset)
        
        # Act
        scores = adapter.score(detector, dataset)
        
        # Assert
        assert len(scores) == len(dataset.data)
        assert all(isinstance(score, AnomalyScore) for score in scores)
        assert all(0.0 <= score.value <= 1.0 for score in scores)
        
        # Scores should have some variation
        score_values = [score.value for score in scores]
        assert max(score_values) > min(score_values)

    def test_unsupported_algorithm_raises_error(self, adapter):
        """Test that unsupported algorithm raises InvalidAlgorithmError."""
        # Arrange
        detector = Detector(
            name="Unsupported",
            algorithm_name="UnsupportedAlgorithm",
            contamination_rate=ContaminationRate(0.1)
        )
        
        # Act & Assert
        with pytest.raises(InvalidAlgorithmError):
            adapter._create_algorithm_instance(detector)

    def test_fit_without_numeric_features_raises_error(self, adapter, detector):
        """Test that fitting without numeric features raises error."""
        # Arrange
        text_data = pd.DataFrame({'text': ['hello', 'world', 'test']})
        dataset = Dataset(name="Text Data", data=text_data)
        
        # Act & Assert
        with pytest.raises(ValueError, match="No numeric features found"):
            adapter.fit(detector, dataset)

    def test_predict_before_fit_raises_error(self, adapter, detector, dataset):
        """Test that predicting before fitting raises error."""
        # Act & Assert
        with pytest.raises(FittingError, match="not fitted"):
            adapter.predict(detector, dataset)

    def test_multiple_algorithms_supported(self, adapter):
        """Test that multiple PyOD algorithms are supported."""
        # Test a few key algorithms
        supported_algorithms = ["IsolationForest", "LOF", "OneClassSVM", "ABOD"]
        
        for algorithm_name in supported_algorithms:
            detector = Detector(
                name=f"Test {algorithm_name}",
                algorithm_name=algorithm_name,
                contamination_rate=ContaminationRate(0.1)
            )
            
            # Should not raise exception
            try:
                algorithm = adapter._create_algorithm_instance(detector)
                assert algorithm is not None
            except ImportError:
                # Some algorithms might not be available in all PyOD installations
                pytest.skip(f"{algorithm_name} not available in this PyOD installation")


class TestSklearnAlgorithmAdapter:
    """Test sklearn algorithm adapter."""

    @pytest.fixture
    def adapter(self):
        """Sklearn algorithm adapter."""
        return SklearnAlgorithmAdapter()

    @pytest.fixture
    def detector(self):
        """Sklearn Isolation Forest detector."""
        return Detector(
            name="Test Sklearn IForest",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42}
        )

    @pytest.fixture
    def dataset(self):
        """Test dataset."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 2))
        outliers = np.random.normal(4, 1, (10, 2))
        
        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=['x', 'y'])
        
        return Dataset(name="Test Data", data=df)

    def test_sklearn_isolation_forest_integration(self, adapter, detector, dataset):
        """Test sklearn Isolation Forest integration."""
        # Arrange & Act
        adapter.fit(detector, dataset)
        predictions = adapter.predict(detector, dataset)
        scores = adapter.score(detector, dataset)
        
        # Assert
        assert len(predictions) == len(dataset.data)
        assert len(scores) == len(dataset.data)
        assert sum(predictions) > 0  # Should detect some outliers
        
        # Predictions should be 0/1
        assert all(pred in [0, 1] for pred in predictions)

    def test_sklearn_prediction_conversion(self, adapter, detector, dataset):
        """Test that sklearn predictions are correctly converted from -1/1 to 0/1."""
        # Arrange
        adapter.fit(detector, dataset)
        
        # Act
        predictions = adapter.predict(detector, dataset)
        
        # Assert - sklearn returns -1 for outliers, 1 for inliers
        # We convert to 1 for outliers, 0 for inliers
        assert all(pred in [0, 1] for pred in predictions)


class TestAlgorithmAdapterRegistry:
    """Test the complete algorithm adapter registry."""

    @pytest.fixture
    def registry(self):
        """Algorithm adapter registry."""
        return AlgorithmAdapterRegistry()

    @pytest.fixture
    def dataset(self):
        """Test dataset with outliers."""
        np.random.seed(42)
        # Generate normal data
        normal_data = np.random.normal(0, 1, (200, 3))
        # Generate clear outliers
        outliers = np.random.normal(6, 0.5, (20, 3))
        
        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        
        return Dataset(name="Integration Test Data", data=df)

    @pytest.mark.integration
    def test_end_to_end_pyod_isolation_forest(self, registry, dataset):
        """Test complete end-to-end workflow with PyOD Isolation Forest."""
        # Arrange
        detector = Detector(
            name="E2E IForest",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 100, "random_state": 42}
        )
        
        # Act - Fit
        registry.fit_detector(detector, dataset)
        
        # Act - Predict
        predictions = registry.predict_with_detector(detector, dataset)
        
        # Act - Score
        scores = registry.score_with_detector(detector, dataset)
        
        # Assert
        assert len(predictions) == len(dataset.data)
        assert len(scores) == len(dataset.data)
        
        # Should detect outliers (we have 20 outliers out of 220 samples)
        outlier_count = sum(predictions)
        assert 10 <= outlier_count <= 40  # Reasonable range
        
        # Scores should be normalized
        score_values = [score.value for score in scores]
        assert all(0.0 <= score <= 1.0 for score in score_values)
        assert max(score_values) > min(score_values)

    @pytest.mark.integration  
    def test_algorithm_comparison(self, registry, dataset):
        """Test multiple algorithms on the same dataset."""
        algorithms_to_test = [
            ("IsolationForest", {"n_estimators": 50, "random_state": 42}),
            ("LOF", {"n_neighbors": 20}),
            ("OneClassSVM", {"nu": 0.1})
        ]
        
        results = {}
        
        for algorithm_name, params in algorithms_to_test:
            try:
                # Create detector
                detector = Detector(
                    name=f"Test {algorithm_name}",
                    algorithm_name=algorithm_name,
                    contamination_rate=ContaminationRate(0.1),
                    parameters=params
                )
                
                # Fit and predict
                registry.fit_detector(detector, dataset)
                predictions = registry.predict_with_detector(detector, dataset)
                scores = registry.score_with_detector(detector, dataset)
                
                results[algorithm_name] = {
                    "outliers_detected": sum(predictions),
                    "avg_score": np.mean([s.value for s in scores]),
                    "max_score": max(s.value for s in scores)
                }
                
            except (ImportError, InvalidAlgorithmError):
                # Algorithm might not be available
                pytest.skip(f"{algorithm_name} not available")
        
        # Assert we got results for at least one algorithm
        assert len(results) > 0
        
        # Each algorithm should detect some outliers
        for algorithm_name, result in results.items():
            assert result["outliers_detected"] > 0, f"{algorithm_name} detected no outliers"
            assert 0.0 < result["avg_score"] < 1.0, f"{algorithm_name} avg score out of range"

    def test_get_supported_algorithms(self, registry):
        """Test getting supported algorithms."""
        # Act
        algorithms = registry.get_supported_algorithms()
        
        # Assert
        assert len(algorithms) > 0
        assert "IsolationForest" in algorithms
        
        # Should have both PyOD and sklearn algorithms
        pyod_algorithms = [alg for alg in algorithms if not alg.startswith("sklearn_")]
        sklearn_algorithms = [alg for alg in algorithms if alg.startswith("sklearn_")]
        
        assert len(pyod_algorithms) > 0
        assert len(sklearn_algorithms) > 0

    def test_invalid_algorithm_error(self, registry, dataset):
        """Test error handling for invalid algorithms."""
        # Arrange
        detector = Detector(
            name="Invalid",
            algorithm_name="NonExistentAlgorithm",
            contamination_rate=ContaminationRate(0.1)
        )
        
        # Act & Assert
        with pytest.raises(InvalidAlgorithmError):
            registry.fit_detector(detector, dataset)

    def test_adapter_info_retrieval(self, registry):
        """Test retrieving adapter information."""
        # Act
        pyod_info = registry.get_adapter_info("pyod")
        sklearn_info = registry.get_adapter_info("sklearn")
        
        # Assert
        assert pyod_info is not None
        assert pyod_info["name"] == "PyOD"
        assert len(pyod_info["supported_algorithms"]) > 0
        
        assert sklearn_info is not None
        assert len(sklearn_info["supported_algorithms"]) > 0

    @pytest.mark.performance
    def test_large_dataset_performance(self, registry):
        """Test performance with larger datasets."""
        # Arrange - Create larger dataset
        np.random.seed(42)
        large_data = np.random.normal(0, 1, (5000, 5))
        outliers = np.random.normal(5, 1, (500, 5))
        
        data = np.vstack([large_data, outliers])
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
        dataset = Dataset(name="Large Dataset", data=df)
        
        detector = Detector(
            name="Performance Test",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42}
        )
        
        # Act
        import time
        start_time = time.time()
        
        registry.fit_detector(detector, dataset)
        predictions = registry.predict_with_detector(detector, dataset)
        scores = registry.score_with_detector(detector, dataset)
        
        end_time = time.time()
        
        # Assert
        assert len(predictions) == len(dataset.data)
        assert len(scores) == len(dataset.data)
        
        # Performance should be reasonable (less than 10 seconds for 5500 samples)
        processing_time = end_time - start_time
        assert processing_time < 10.0, f"Processing took too long: {processing_time}s"
        
        # Should detect a reasonable number of outliers
        outlier_count = sum(predictions)
        assert 200 <= outlier_count <= 1000  # Reasonable range for 500 injected outliers