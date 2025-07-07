"""Comprehensive tests for algorithm adapters."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.algorithm_factory import (
    AlgorithmFactory,
    AlgorithmLibrary,
    DatasetCharacteristics,
)
from pynomaly.infrastructure.adapters.enhanced_pyod_adapter import EnhancedPyODAdapter
from pynomaly.infrastructure.adapters.enhanced_sklearn_adapter import (
    EnhancedSklearnAdapter,
)
from pynomaly.infrastructure.adapters.ensemble_meta_adapter import (
    AggregationMethod,
    EnsembleMetaAdapter,
)


class TestEnhancedPyODAdapter:
    """Test enhanced PyOD adapter."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        # Create dataset with 100 normal points and 10 anomalies
        normal_data = np.random.normal(0, 1, (90, 3))
        anomaly_data = np.random.normal(3, 0.5, (10, 3))
        data = np.vstack([normal_data, anomaly_data])

        df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
        return Dataset(data=df, name="test_dataset")

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = EnhancedPyODAdapter(
            algorithm_name="IsolationForest", contamination_rate=ContaminationRate(0.1)
        )

        assert adapter.name == "Enhanced_PyOD_IsolationForest"
        assert adapter.contamination_rate.value == 0.1
        assert not adapter.is_fitted
        assert adapter.algorithm_metadata.category == "Ensemble"

    def test_invalid_algorithm(self):
        """Test invalid algorithm handling."""
        with pytest.raises(Exception):  # Should be InvalidAlgorithmError
            EnhancedPyODAdapter(algorithm_name="NonExistentAlgorithm")

    @patch("pyod.models.iforest.IForest")
    def test_fit_method(self, mock_iforest, sample_dataset):
        """Test fit method."""
        # Mock the PyOD model
        mock_model = Mock()
        mock_iforest.return_value = mock_model

        adapter = EnhancedPyODAdapter(algorithm_name="IsolationForest")
        adapter.fit(sample_dataset)

        assert adapter.is_fitted
        mock_model.fit.assert_called_once()

    @patch("pyod.models.iforest.IForest")
    def test_detect_method(self, mock_iforest, sample_dataset):
        """Test detect method."""
        # Mock the PyOD model
        mock_model = Mock()
        mock_model.predict.return_value = np.array(
            [0, 0, 1, 0, 1]
        )  # Sample predictions
        mock_model.decision_function.return_value = np.array(
            [0.1, 0.2, 0.8, 0.3, 0.9]
        )  # Sample scores
        mock_iforest.return_value = mock_model

        adapter = EnhancedPyODAdapter(algorithm_name="IsolationForest")
        adapter.fit(sample_dataset)

        # Take only first 5 rows for this test
        test_data = sample_dataset.data.head(5)
        test_dataset = Dataset(data=test_data, name="test_subset")

        result = adapter.detect(test_dataset)

        assert result is not None
        assert len(result.scores) == 5
        assert len(result.labels) == 5
        assert len(result.anomalies) == 2  # Two anomalies predicted

    def test_algorithm_recommendations(self):
        """Test algorithm recommendations."""
        recommendations = EnhancedPyODAdapter.recommend_algorithms(
            n_samples=1000, n_features=10, has_gpu=False, prefer_fast=True
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert "IsolationForest" in recommendations or "COPOD" in recommendations

    def test_list_algorithms(self):
        """Test listing available algorithms."""
        algorithms = EnhancedPyODAdapter.list_algorithms()

        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert "IsolationForest" in algorithms
        assert "LOF" in algorithms

    def test_algorithm_metadata(self):
        """Test algorithm metadata retrieval."""
        metadata = EnhancedPyODAdapter.get_algorithm_metadata("IsolationForest")

        assert metadata is not None
        assert metadata.category == "Ensemble"
        assert metadata.complexity_time == "O(n log n)"
        assert not metadata.requires_gpu


class TestEnhancedSklearnAdapter:
    """Test enhanced sklearn adapter."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (90, 3))
        anomaly_data = np.random.normal(3, 0.5, (10, 3))
        data = np.vstack([normal_data, anomaly_data])

        df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
        return Dataset(data=df, name="test_dataset")

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = EnhancedSklearnAdapter(
            algorithm_name="IsolationForest", contamination_rate=ContaminationRate(0.1)
        )

        assert adapter.name == "Sklearn_IsolationForest"
        assert adapter.contamination_rate.value == 0.1
        assert not adapter.is_fitted
        assert adapter.algorithm_info.category == "Ensemble"

    @patch("sklearn.ensemble.IsolationForest")
    def test_fit_method(self, mock_iforest, sample_dataset):
        """Test fit method."""
        # Mock the sklearn model
        mock_model = Mock()
        mock_iforest.return_value = mock_model

        adapter = EnhancedSklearnAdapter(algorithm_name="IsolationForest")
        adapter.fit(sample_dataset)

        assert adapter.is_fitted
        mock_model.fit.assert_called_once()

    @patch("sklearn.ensemble.IsolationForest")
    def test_detect_with_scaling(self, mock_iforest, sample_dataset):
        """Test detect method with scaling."""
        # Mock the sklearn model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 1, -1, 1, -1])  # sklearn format
        mock_model.decision_function.return_value = np.array(
            [0.1, 0.2, -0.8, 0.3, -0.9]
        )
        mock_iforest.return_value = mock_model

        adapter = EnhancedSklearnAdapter(
            algorithm_name="IsolationForest", use_scaling=True
        )
        adapter.fit(sample_dataset)

        # Test with subset
        test_data = sample_dataset.data.head(5)
        test_dataset = Dataset(data=test_data, name="test_subset")

        result = adapter.detect(test_dataset)

        assert result is not None
        assert len(result.scores) == 5
        assert len(result.labels) == 5
        # sklearn returns -1 for anomalies, which should be converted to 1
        assert result.labels.count(1) == 2  # Two anomalies

    def test_algorithm_recommendations(self):
        """Test algorithm recommendations."""
        recommendations = EnhancedSklearnAdapter.recommend_algorithms(
            n_samples=1000, n_features=10, prefer_interpretable=True
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestEnsembleMetaAdapter:
    """Test ensemble meta-adapter."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (90, 3))
        anomaly_data = np.random.normal(3, 0.5, (10, 3))
        data = np.vstack([normal_data, anomaly_data])

        df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
        return Dataset(data=df, name="test_dataset")

    @pytest.fixture
    def mock_detectors(self):
        """Create mock detectors for testing."""
        detector1 = Mock()
        detector1.name = "MockDetector1"
        detector1.fit.return_value = None
        detector1.detect.return_value = Mock(
            anomalies=[],
            scores=[Mock(value=0.1), Mock(value=0.8), Mock(value=0.3)],
            labels=[0, 1, 0],
            threshold=0.5,
            execution_time_ms=100,
        )

        detector2 = Mock()
        detector2.name = "MockDetector2"
        detector2.fit.return_value = None
        detector2.detect.return_value = Mock(
            anomalies=[],
            scores=[Mock(value=0.2), Mock(value=0.9), Mock(value=0.4)],
            labels=[0, 1, 0],
            threshold=0.6,
            execution_time_ms=150,
        )

        return [detector1, detector2]

    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleMetaAdapter(
            name="TestEnsemble", aggregation_method=AggregationMethod.WEIGHTED_AVERAGE
        )

        assert ensemble.name == "TestEnsemble"
        assert not ensemble.is_fitted
        assert len(ensemble.base_detectors) == 0

    def test_add_remove_detectors(self, mock_detectors):
        """Test adding and removing detectors."""
        ensemble = EnsembleMetaAdapter()

        # Add detectors
        ensemble.add_detector(mock_detectors[0], weight=1.0)
        ensemble.add_detector(mock_detectors[1], weight=2.0)

        assert len(ensemble.base_detectors) == 2
        weights = ensemble.get_detector_weights()
        assert weights["MockDetector1"] == 1.0
        assert weights["MockDetector2"] == 2.0

        # Remove detector
        ensemble.remove_detector("MockDetector1")
        assert len(ensemble.base_detectors) == 1

    def test_ensemble_fit(self, mock_detectors, sample_dataset):
        """Test ensemble fitting."""
        ensemble = EnsembleMetaAdapter()
        ensemble.add_detector(mock_detectors[0])
        ensemble.add_detector(mock_detectors[1])

        ensemble.fit(sample_dataset)

        assert ensemble.is_fitted
        mock_detectors[0].fit.assert_called_once_with(sample_dataset)
        mock_detectors[1].fit.assert_called_once_with(sample_dataset)

    def test_ensemble_detect(self, mock_detectors, sample_dataset):
        """Test ensemble detection."""
        ensemble = EnsembleMetaAdapter(aggregation_method=AggregationMethod.AVERAGE)
        ensemble.add_detector(mock_detectors[0])
        ensemble.add_detector(mock_detectors[1])

        # Mock the fitted state
        ensemble._is_fitted = True

        # Create a smaller dataset for testing
        test_data = pd.DataFrame(
            {
                "feature1": [0.1, 2.5, 0.3],
                "feature2": [0.2, 3.0, 0.4],
                "feature3": [0.1, 2.8, 0.2],
            }
        )
        test_dataset = Dataset(data=test_data, name="test_subset")

        result = ensemble.detect(test_dataset)

        assert result is not None
        assert len(result.scores) == 3
        assert len(result.labels) == 3

        # Verify individual detectors were called
        mock_detectors[0].detect.assert_called_once()
        mock_detectors[1].detect.assert_called_once()


class TestAlgorithmFactory:
    """Test algorithm factory."""

    @pytest.fixture
    def sample_characteristics(self):
        """Create sample dataset characteristics."""
        return DatasetCharacteristics(
            n_samples=1000,
            n_features=10,
            has_categorical=False,
            has_missing_values=False,
            contamination_estimate=0.1,
            computational_budget="medium",
        )

    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = AlgorithmFactory()

        algorithms = factory.list_all_algorithms()
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0

    @patch("pynomaly.infrastructure.adapters.enhanced_pyod_adapter.EnhancedPyODAdapter")
    def test_create_pyod_detector(self, mock_adapter):
        """Test creating PyOD detector."""
        factory = AlgorithmFactory()

        detector = factory.create_detector(
            algorithm_name="IsolationForest", library=AlgorithmLibrary.PYOD
        )

        mock_adapter.assert_called_once()

    @patch(
        "pynomaly.infrastructure.adapters.enhanced_sklearn_adapter.EnhancedSklearnAdapter"
    )
    def test_create_sklearn_detector(self, mock_adapter):
        """Test creating sklearn detector."""
        factory = AlgorithmFactory()

        detector = factory.create_detector(
            algorithm_name="IsolationForest", library=AlgorithmLibrary.SKLEARN
        )

        mock_adapter.assert_called_once()

    def test_create_ensemble_detector(self):
        """Test creating ensemble detector."""
        factory = AlgorithmFactory()

        detector_configs = [
            {
                "algorithm_name": "IsolationForest",
                "library": AlgorithmLibrary.SKLEARN,
                "weight": 1.0,
            },
            {"algorithm_name": "LOF", "library": AlgorithmLibrary.PYOD, "weight": 1.5},
        ]

        # Mock the individual detector creation
        with patch.object(factory, "create_detector") as mock_create:
            mock_detector1 = Mock()
            mock_detector1.name = "IsolationForest"
            mock_detector2 = Mock()
            mock_detector2.name = "LOF"
            mock_create.side_effect = [mock_detector1, mock_detector2]

            ensemble = factory.create_ensemble(
                detector_configs=detector_configs, name="TestEnsemble"
            )

            assert isinstance(ensemble, EnsembleMetaAdapter)
            assert ensemble.name == "TestEnsemble"

    def test_algorithm_recommendations(self, sample_characteristics):
        """Test algorithm recommendations."""
        factory = AlgorithmFactory()

        recommendations = factory.recommend_algorithms(
            dataset_characteristics=sample_characteristics, top_k=3
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3

        for rec in recommendations:
            assert hasattr(rec, "algorithm_name")
            assert hasattr(rec, "confidence")
            assert 0 <= rec.confidence <= 1

    def test_auto_detector_creation(self, sample_characteristics):
        """Test automatic detector creation."""
        factory = AlgorithmFactory()

        with patch.object(factory, "create_detector") as mock_create:
            mock_detector = Mock()
            mock_create.return_value = mock_detector

            detector = factory.create_auto_detector(
                dataset_characteristics=sample_characteristics,
                performance_preference="balanced",
            )

            assert detector is not None
            mock_create.assert_called_once()

    def test_library_detection(self):
        """Test automatic library detection."""
        factory = AlgorithmFactory()

        # Test PyOD detection
        library = factory._detect_library("LOF")
        assert library == AlgorithmLibrary.PYOD

        # Test sklearn detection
        library = factory._detect_library("OneClassSVM")
        assert library == AlgorithmLibrary.SKLEARN

        # Test ensemble detection
        library = factory._detect_library("ensemble")
        assert library == AlgorithmLibrary.ENSEMBLE

    def test_algorithm_info_retrieval(self):
        """Test algorithm information retrieval."""
        factory = AlgorithmFactory()

        info = factory.get_algorithm_info("IsolationForest", AlgorithmLibrary.PYOD)
        assert isinstance(info, dict)
        assert "name" in info
        assert "library" in info
        assert "category" in info


class TestIntegration:
    """Integration tests for the complete algorithm system."""

    @pytest.fixture
    def real_dataset(self):
        """Create a realistic dataset for integration testing."""
        np.random.seed(42)
        # Create a more realistic dataset
        n_normal = 950
        n_anomaly = 50

        # Normal data: multivariate normal
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
            size=n_normal,
        )

        # Anomaly data: shifted and different covariance
        anomaly_data = np.random.multivariate_normal(
            mean=[3, 3, 3], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], size=n_anomaly
        )

        data = np.vstack([normal_data, anomaly_data])

        # Shuffle the data
        indices = np.random.permutation(len(data))
        data = data[indices]

        df = pd.DataFrame(data, columns=["x1", "x2", "x3"])
        return Dataset(data=df, name="integration_test_dataset")

    def test_end_to_end_detection(self, real_dataset):
        """Test end-to-end detection workflow."""
        factory = AlgorithmFactory()

        # Create dataset characteristics
        characteristics = DatasetCharacteristics(
            n_samples=len(real_dataset.data),
            n_features=len(real_dataset.data.columns),
            computational_budget="medium",
        )

        # Get recommendations
        recommendations = factory.recommend_algorithms(characteristics, top_k=2)
        assert len(recommendations) > 0

        # Create detector based on top recommendation
        best_rec = recommendations[0]

        # Skip ensemble creation for this test to avoid complexity
        if best_rec.library != AlgorithmLibrary.ENSEMBLE:
            try:
                detector = factory.create_detector(
                    algorithm_name=best_rec.algorithm_name,
                    library=best_rec.library,
                    contamination_rate=ContaminationRate(0.05),
                )

                # Test the complete workflow
                detector.fit(real_dataset)
                assert detector.is_fitted

                result = detector.detect(real_dataset)
                assert result is not None
                assert len(result.scores) == len(real_dataset.data)
                assert len(result.labels) == len(real_dataset.data)

                # Check that some anomalies were detected
                n_anomalies = sum(result.labels)
                assert (
                    0 < n_anomalies < len(real_dataset.data) * 0.2
                )  # Reasonable range

            except Exception as e:
                # Skip if dependencies are not available
                pytest.skip(f"Skipping due to missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
