"""
ML Adapter Integration Testing Suite
Integration tests for all ML adapters ensuring proper interface compliance and cross-adapter compatibility.
"""

import os
import sys
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol


class TestAdapterInterfaceCompliance:
    """Test that all adapters comply with the DetectorProtocol interface."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for interface testing."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 8))
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(8)])

        dataset = Mock(spec=Dataset)
        dataset.id = "interface_test_dataset"
        dataset.name = "Interface Test Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(UTC)
        dataset.get_numeric_features.return_value = list(df.columns)

        return dataset

    def test_pyod_adapter_interface_compliance(self, sample_dataset):
        """Test PyOD adapter interface compliance."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            # Mock PyOD model
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0] * 90 + [1] * 10)
            mock_model.decision_function.return_value = np.random.normal(0, 1, 100)
            mock_model_class.return_value = mock_model

            mock_module = Mock()
            mock_module.IForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Test interface compliance
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
            assert hasattr(adapter, "score")
            assert hasattr(adapter, "get_params")
            assert hasattr(adapter, "set_params")

            assert callable(adapter.fit)
            assert callable(adapter.detect)
            assert callable(adapter.score)
            assert callable(adapter.get_params)
            assert callable(adapter.set_params)

            # Test method signatures and return types
            adapter.fit(sample_dataset)
            assert adapter.is_fitted is True

            result = adapter.detect(sample_dataset)
            assert isinstance(result, DetectionResult)

            scores = adapter.score(sample_dataset)
            assert isinstance(scores, list)
            assert all(isinstance(score, AnomalyScore) for score in scores)

            params = adapter.get_params()
            assert isinstance(params, dict)

    def test_sklearn_adapter_interface_compliance(self, sample_dataset):
        """Test sklearn adapter interface compliance."""
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_import:
            # Mock sklearn model
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([-1] * 10 + [1] * 90)
            mock_model.score_samples.return_value = np.random.normal(0, 1, 100)
            mock_model.get_params.return_value = {"contamination": 0.1}
            mock_model.set_params.return_value = None
            mock_model_class.return_value = mock_model

            mock_module = Mock()
            mock_module.IsolationForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            adapter = SklearnAdapter(algorithm_name="IsolationForest")

            # Test interface compliance
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
            assert hasattr(adapter, "score")
            assert hasattr(adapter, "get_params")
            assert hasattr(adapter, "set_params")

            # Test functionality
            adapter.fit(sample_dataset)
            result = adapter.detect(sample_dataset)
            scores = adapter.score(sample_dataset)

            assert isinstance(result, DetectionResult)
            assert isinstance(scores, list)
            assert len(result.anomalies) == 10  # Number of -1 predictions

    def test_adapter_protocol_compliance(self):
        """Test that adapters implement DetectorProtocol correctly."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = Mock()
            mock_import.return_value.IForest = Mock()

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Verify protocol compliance
            assert isinstance(adapter, DetectorProtocol)

            # Check required attributes
            assert hasattr(adapter, "id")
            assert hasattr(adapter, "name")
            assert hasattr(adapter, "algorithm_name")
            assert hasattr(adapter, "is_fitted")
            assert hasattr(adapter, "contamination_rate")
            assert hasattr(adapter, "parameters")
            assert hasattr(adapter, "metadata")


class TestCrossAdapterCompatibility:
    """Test compatibility between different ML adapters."""

    @pytest.fixture
    def standardized_dataset(self):
        """Create standardized dataset for cross-adapter testing."""
        np.random.seed(42)

        # Create dataset with known anomalies
        normal_data = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4), 900)
        anomaly_data = np.random.multivariate_normal([3, 3, 3, 3], np.eye(4) * 0.5, 100)

        all_data = np.vstack([normal_data, anomaly_data])
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]

        df = pd.DataFrame(all_data, columns=["x1", "x2", "x3", "x4"])

        dataset = Mock(spec=Dataset)
        dataset.id = "cross_adapter_dataset"
        dataset.name = "Cross Adapter Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(UTC)
        dataset.get_numeric_features.return_value = ["x1", "x2", "x3", "x4"]

        return dataset

    def test_adapter_result_consistency(self, standardized_dataset):
        """Test that different adapters produce consistent result formats."""
        # Mock PyOD adapter
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_pyod_import:
            mock_pyod_import.return_value = Mock()
            mock_pyod_import.return_value.IForest = Mock()

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            pyod_adapter = PyODAdapter(algorithm_name="IsolationForest")

        # Mock sklearn adapter
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_sklearn_import:
            mock_sklearn_import.return_value = Mock()
            mock_sklearn_import.return_value.IsolationForest = Mock()

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            sklearn_adapter = SklearnAdapter(algorithm_name="IsolationForest")

        adapters = [pyod_adapter, sklearn_adapter]
        results = []

        for adapter in adapters:
            # Mock the underlying model behavior
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0] * 900 + [1] * 100)
            mock_model.decision_function.return_value = np.random.normal(0, 1, 1000)
            if hasattr(adapter, "_model_class"):
                adapter._model_class.return_value = mock_model

            # Fit and detect
            adapter.fit(standardized_dataset)
            adapter._model = mock_model  # Ensure model is set
            result = adapter.detect(standardized_dataset)

            results.append(result)

        # Verify consistent result structure
        for result in results:
            assert isinstance(result, DetectionResult)
            assert hasattr(result, "detector_id")
            assert hasattr(result, "dataset_id")
            assert hasattr(result, "anomalies")
            assert hasattr(result, "scores")
            assert hasattr(result, "labels")
            assert hasattr(result, "threshold")
            assert hasattr(result, "execution_time_ms")
            assert hasattr(result, "metadata")

            # Verify data types
            assert isinstance(result.anomalies, list)
            assert isinstance(result.scores, list)
            assert isinstance(result.labels, list | np.ndarray)
            assert isinstance(result.threshold, int | float)
            assert isinstance(result.execution_time_ms, int | float)
            assert isinstance(result.metadata, dict)

    def test_adapter_score_normalization(self, standardized_dataset):
        """Test that all adapters normalize scores to [0, 1] range."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            # Mock PyOD with various score ranges
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.uniform(
                -100, 100, 1000
            )
            mock_model_class.return_value = mock_model

            mock_module = Mock()
            mock_module.IForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")
            adapter.fit(standardized_dataset)
            scores = adapter.score(standardized_dataset)

            # Verify score normalization
            assert all(0 <= score.value <= 1 for score in scores)
            assert all(isinstance(score, AnomalyScore) for score in scores)

    def test_adapter_parameter_compatibility(self):
        """Test parameter compatibility across adapters."""
        common_params = {
            "contamination_rate": ContaminationRate(0.1),
            "random_state": 42,
        }

        # Test PyOD adapter
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = Mock()
            mock_import.return_value.IForest = Mock()

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            pyod_adapter = PyODAdapter(
                algorithm_name="IsolationForest", **common_params
            )

            assert pyod_adapter.contamination_rate.value == 0.1
            assert pyod_adapter.parameters.get("random_state") == 42

        # Test sklearn adapter
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = Mock()
            mock_import.return_value.IsolationForest = Mock()

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            sklearn_adapter = SklearnAdapter(
                algorithm_name="IsolationForest", **common_params
            )

            assert sklearn_adapter.contamination_rate.value == 0.1
            assert sklearn_adapter.parameters.get("random_state") == 42


class TestAdapterErrorHandlingConsistency:
    """Test consistent error handling across all adapters."""

    def test_adapter_not_fitted_errors(self):
        """Test that all adapters handle not-fitted state consistently."""
        # Test PyOD adapter
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = Mock()
            mock_import.return_value.IForest = Mock()

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Create mock dataset
            mock_dataset = Mock(spec=Dataset)
            mock_dataset.features = pd.DataFrame(np.random.randn(10, 5))
            mock_dataset.get_numeric_features.return_value = [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
            ]

            # Test detect without fitting
            with pytest.raises(DetectorNotFittedError) as exc_info:
                adapter.detect(mock_dataset)

            assert "PyOD_IsolationForest" in str(exc_info.value)
            assert "detect" in str(exc_info.value)

            # Test score without fitting
            with pytest.raises(DetectorNotFittedError) as exc_info:
                adapter.score(mock_dataset)

            assert "score" in str(exc_info.value)

    def test_adapter_invalid_algorithm_errors(self):
        """Test that adapters handle invalid algorithms consistently."""
        # Test PyOD adapter
        with pytest.raises(InvalidAlgorithmError) as exc_info:
            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            PyODAdapter(algorithm_name="NonExistentAlgorithm")

        assert "NonExistentAlgorithm" in str(exc_info.value)
        assert "available_algorithms" in str(exc_info.value)

        # Test sklearn adapter
        with pytest.raises(InvalidAlgorithmError) as exc_info:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            SklearnAdapter(algorithm_name="InvalidSklearnAlgorithm")

        assert "InvalidSklearnAlgorithm" in str(exc_info.value)

    def test_adapter_fitting_error_handling(self):
        """Test adapter fitting error handling."""
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_import:
            # Mock sklearn model that raises error during fit
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.side_effect = ValueError("Fitting error occurred")
            mock_model_class.return_value = mock_model

            mock_module = Mock()
            mock_module.IsolationForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            adapter = SklearnAdapter(algorithm_name="IsolationForest")

            # Create mock dataset
            mock_dataset = Mock(spec=Dataset)
            mock_dataset.name = "Error Test Dataset"
            mock_dataset.features = pd.DataFrame(np.random.randn(10, 5))
            mock_dataset.get_numeric_features.return_value = [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
            ]
            mock_dataset.created_at = datetime.now(UTC)
            mock_dataset.n_samples = 10

            with pytest.raises(FittingError) as exc_info:
                adapter.fit(mock_dataset)

            assert "Fitting error occurred" in str(exc_info.value)
            assert adapter.name in str(exc_info.value)
            assert "Error Test Dataset" in str(exc_info.value)


class TestAdapterPerformanceCharacteristics:
    """Test performance characteristics and metadata across adapters."""

    def test_adapter_metadata_consistency(self):
        """Test that adapters provide consistent metadata."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_import.return_value = Mock()
            mock_import.return_value.IForest = Mock()

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Check required metadata fields
            assert "category" in adapter.metadata
            assert "time_complexity" in adapter.metadata
            assert "space_complexity" in adapter.metadata

            # Check metadata values
            assert adapter.metadata["category"] == "ensemble"
            assert adapter.metadata["time_complexity"] == "O(n log n)"
            assert adapter.metadata["space_complexity"] == "O(n)"

    def test_adapter_training_metadata_updates(self):
        """Test that adapters update training metadata correctly."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            # Mock PyOD model
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model_class.return_value = mock_model

            mock_module = Mock()
            mock_module.IForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Create mock dataset
            mock_dataset = Mock(spec=Dataset)
            mock_dataset.features = pd.DataFrame(np.random.randn(100, 8))
            mock_dataset.get_numeric_features.return_value = [f"f{i}" for i in range(8)]
            mock_dataset.created_at = datetime.now(UTC)
            mock_dataset.n_samples = 100
            mock_dataset.name = "Training Metadata Test"

            # Initial metadata state
            assert "training_time_ms" not in adapter.metadata
            assert "training_samples" not in adapter.metadata
            assert "training_features" not in adapter.metadata

            # Fit adapter
            adapter.fit(mock_dataset)

            # Check updated metadata
            assert "training_time_ms" in adapter.metadata
            assert "training_samples" in adapter.metadata
            assert "training_features" in adapter.metadata
            assert adapter.metadata["training_samples"] == 100
            assert adapter.metadata["training_features"] == 8
            assert isinstance(adapter.metadata["training_time_ms"], int | float)

    def test_adapter_scalability_indicators(self):
        """Test adapter scalability indicators and complexity information."""
        algorithms_and_complexities = [
            ("IsolationForest", "O(n log n)", "O(n)"),
            ("LOF", "O(n²)", "O(n)"),
            ("KNN", "O(n log n)", "O(n)"),
            ("COPOD", "O(n*p)", "O(n*p)"),
            ("ECOD", "O(n*p)", "O(n*p)"),
        ]

        for algorithm, expected_time, expected_space in algorithms_and_complexities:
            with patch(
                "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
            ) as mock_import:
                mock_import.return_value = Mock()
                setattr(
                    mock_import.return_value,
                    algorithm.replace("IsolationForest", "IForest"),
                    Mock(),
                )

                from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

                adapter = PyODAdapter(algorithm_name=algorithm)

                if algorithm in ["IsolationForest", "LOF", "KNN", "COPOD", "ECOD"]:
                    assert adapter.metadata.get("time_complexity") == expected_time
                    assert adapter.metadata.get("space_complexity") == expected_space


class TestAdapterEnsembleCompatibility:
    """Test adapter compatibility for ensemble usage."""

    @pytest.fixture
    def ensemble_dataset(self):
        """Create dataset suitable for ensemble testing."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (200, 6))
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(6)])

        dataset = Mock(spec=Dataset)
        dataset.id = "ensemble_dataset"
        dataset.name = "Ensemble Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(UTC)
        dataset.get_numeric_features.return_value = list(df.columns)

        return dataset

    def test_multiple_adapter_ensemble(self, ensemble_dataset):
        """Test using multiple adapters in an ensemble."""
        adapters = []

        # Create PyOD adapter
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_pyod_import:
            mock_pyod_import.return_value = Mock()
            mock_pyod_import.return_value.IForest = Mock()

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            pyod_adapter = PyODAdapter(
                algorithm_name="IsolationForest", name="PyOD_Ensemble_Member"
            )
            adapters.append(pyod_adapter)

        # Create sklearn adapter
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_sklearn_import:
            mock_sklearn_import.return_value = Mock()
            mock_sklearn_import.return_value.IsolationForest = Mock()

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            sklearn_adapter = SklearnAdapter(
                algorithm_name="IsolationForest", name="Sklearn_Ensemble_Member"
            )
            adapters.append(sklearn_adapter)

        # Mock training and prediction for ensemble
        ensemble_results = []

        for _i, adapter in enumerate(adapters):
            # Mock model behavior
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.random.binomial(1, 0.1, 200)
            mock_model.decision_function.return_value = np.random.normal(0, 1, 200)
            mock_model.score_samples.return_value = np.random.normal(0, 1, 200)

            adapter._model_class.return_value = mock_model

            # Fit adapter
            adapter.fit(ensemble_dataset)
            adapter._model = mock_model

            # Get ensemble member result
            result = adapter.detect(ensemble_dataset)
            ensemble_results.append(result)

        # Verify ensemble compatibility
        assert len(ensemble_results) == 2

        for result in ensemble_results:
            assert isinstance(result, DetectionResult)
            assert len(result.scores) == 200
            assert all(0 <= score.value <= 1 for score in result.scores)

        # Test score aggregation compatibility
        all_scores = []
        for result in ensemble_results:
            scores = [score.value for score in result.scores]
            all_scores.append(scores)

        # Verify scores can be aggregated
        mean_scores = np.mean(all_scores, axis=0)
        assert len(mean_scores) == 200
        assert all(0 <= score <= 1 for score in mean_scores)

    def test_adapter_weight_compatibility(self, ensemble_dataset):
        """Test adapter compatibility with weighted ensemble approaches."""
        # Test different contamination rates (could be used as weights)
        contamination_rates = [0.05, 0.1, 0.15]
        adapters = []

        for rate in contamination_rates:
            with patch(
                "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
            ) as mock_import:
                mock_import.return_value = Mock()
                mock_import.return_value.IForest = Mock()

                from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

                adapter = PyODAdapter(
                    algorithm_name="IsolationForest",
                    contamination_rate=ContaminationRate(rate),
                    name=f"Adapter_Rate_{rate}",
                )
                adapters.append(adapter)

        # Verify contamination rates are preserved
        for i, adapter in enumerate(adapters):
            assert adapter.contamination_rate.value == contamination_rates[i]
            assert adapter.name == f"Adapter_Rate_{contamination_rates[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
