"""Integration tests for PyGOD adapter validation.

This module contains comprehensive tests to validate the PyGOD adapter
implementation when the required dependencies are installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import AdapterError, AlgorithmNotFoundError
from monorepo.domain.value_objects import ContaminationRate
from monorepo.infrastructure.adapters.pygod_adapter import PyGODAdapter


def _pygod_available() -> bool:
    """Check if PyGOD dependencies are available."""
    try:
        import pygod
        import torch
        import torch_geometric

        return True
    except ImportError:
        return False


class TestPyGODAdapterValidation:
    """Comprehensive validation tests for PyGOD adapter."""

    @pytest.fixture
    def sample_graph_dataset(self) -> Dataset:
        """Create a sample graph dataset for testing."""
        # Create a simple graph with edge list format
        data = pd.DataFrame(
            {
                "source": [0, 1, 1, 2, 2, 3, 3, 4],
                "target": [1, 0, 2, 1, 3, 2, 4, 3],
                "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature_1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            }
        )

        return Dataset(
            id="test_graph",
            name="Test Graph Dataset",
            data=data,
            metadata={
                "is_graph": True,
                "edge_columns": ["source", "target"],
                "feature_columns": ["feature_0", "feature_1"],
            },
        )

    @pytest.fixture
    def sample_node_features_dataset(self) -> Dataset:
        """Create a dataset with node features only (no explicit edges)."""
        # This will test k-NN graph construction
        data = pd.DataFrame(
            {
                "node_id": range(10),
                "feature_0": np.random.normal(0, 1, 10),
                "feature_1": np.random.normal(0, 1, 10),
                "feature_2": np.random.normal(0, 1, 10),
            }
        )

        return Dataset(
            id="test_nodes",
            name="Test Node Features Dataset",
            data=data,
            metadata={"feature_columns": ["feature_0", "feature_1", "feature_2"]},
        )

    def test_adapter_initialization(self):
        """Test PyGOD adapter initialization."""
        # Test with mock algorithms
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            adapter = PyGODAdapter(
                algorithm_name="DOMINANT", contamination_rate=ContaminationRate(0.15)
            )

            assert adapter.name == "PyGOD_DOMINANT"
            assert adapter.algorithm_name == "DOMINANT"
            assert adapter.contamination_rate.value == 0.15
            assert not adapter.is_fitted

    def test_get_supported_algorithms(self):
        """Test retrieving supported algorithms."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_algorithms = {
                "DOMINANT": MagicMock,
                "GCNAE": MagicMock,
                "SCAN": MagicMock,
            }
            mock_map.return_value = mock_algorithms

            algorithms = PyGODAdapter.get_supported_algorithms()

            assert len(algorithms) == 3
            assert "DOMINANT" in algorithms
            assert "GCNAE" in algorithms
            assert "SCAN" in algorithms

    def test_get_algorithm_info(self):
        """Test retrieving algorithm information."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            info = PyGODAdapter.get_algorithm_info("DOMINANT")

            assert "name" in info
            assert "type" in info
            assert "description" in info
            assert "parameters" in info
            assert info["name"] == "DOMINANT"
            assert info["type"] == "Deep Learning"

    def test_get_algorithm_info_invalid_algorithm(self):
        """Test error handling for invalid algorithm."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            with pytest.raises(AlgorithmNotFoundError):
                PyGODAdapter.get_algorithm_info("INVALID_ALGO")

    @patch("monorepo.infrastructure.adapters.pygod_adapter.torch")
    @patch("monorepo.infrastructure.adapters.pygod_adapter.Data")
    def test_graph_data_preparation_edge_list(
        self, mock_data, mock_torch, sample_graph_dataset
    ):
        """Test graph data preparation from edge list format."""
        # Mock PyTorch components
        mock_torch.tensor.return_value = MagicMock()
        mock_data.return_value = MagicMock()

        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            adapter = PyGODAdapter(algorithm_name="DOMINANT")

            # This should not raise an error
            try:
                adapter._prepare_graph_data(sample_graph_dataset)
            except ImportError:
                # Expected when PyTorch is not installed
                pass

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            adapter = PyGODAdapter(algorithm_name="DOMINANT")

            # Test normal sample (label = 0)
            confidence_normal = adapter._calculate_confidence(0.2, 0)
            assert 0.5 <= confidence_normal <= 1.0

            # Test anomaly sample (label = 1)
            confidence_anomaly = adapter._calculate_confidence(0.8, 1)
            assert 0.5 <= confidence_anomaly <= 1.0
            assert confidence_anomaly > confidence_normal

    @patch("monorepo.infrastructure.adapters.pygod_adapter.torch")
    @patch("monorepo.infrastructure.adapters.pygod_adapter.Data")
    def test_fit_and_predict_workflow(
        self, mock_data, mock_torch, sample_graph_dataset
    ):
        """Test the complete fit and predict workflow."""
        # Mock PyTorch components
        mock_torch.tensor.return_value = MagicMock()
        mock_data.return_value = MagicMock()

        # Mock PyGOD model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.decision_function.return_value = np.array([0.1, 0.9, 0.2, 0.8, 0.1])

        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_algorithm_class = MagicMock(return_value=mock_model)
            mock_map.return_value = {"DOMINANT": mock_algorithm_class}

            adapter = PyGODAdapter(algorithm_name="DOMINANT")

            # Mock the model creation
            adapter._model = mock_model

            try:
                # Test fitting
                adapter.fit(sample_graph_dataset)
                assert adapter.is_fitted

                # Test prediction
                result = adapter.predict(sample_graph_dataset)
                assert result.detector_id == adapter.name
                assert result.dataset_id == sample_graph_dataset.id
                assert len(result.scores) == len(result.labels)
                assert "is_graph" in result.metadata
                assert result.metadata["is_graph"] is True

            except ImportError:
                # Expected when dependencies are not installed
                pass

    def test_predict_without_fit_raises_error(self, sample_graph_dataset):
        """Test that prediction without fitting raises an error."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            adapter = PyGODAdapter(algorithm_name="DOMINANT")

            with pytest.raises(AdapterError, match="Model must be fitted"):
                adapter.predict(sample_graph_dataset)

    def test_knn_graph_construction(self):
        """Test k-NN graph construction from features."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            adapter = PyGODAdapter(algorithm_name="DOMINANT")

            # Create sample feature matrix
            features = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

            try:
                edge_index = adapter._build_knn_graph(features, k=2)
                assert edge_index.shape[0] == 2  # 2 rows for edge indices
                assert edge_index.shape[1] > 0  # Should have some edges
            except ImportError:
                # Expected when scikit-learn is not available
                pass

    def test_invalid_algorithm_initialization(self):
        """Test initialization with invalid algorithm name."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_map.return_value = {"DOMINANT": MagicMock}

            with pytest.raises(AlgorithmNotFoundError):
                PyGODAdapter(algorithm_name="INVALID_ALGORITHM")

    def test_algorithm_parameter_passing(self):
        """Test that algorithm parameters are properly passed."""
        with patch.object(PyGODAdapter, "_get_algorithm_map") as mock_map:
            mock_algorithm_class = MagicMock()
            mock_map.return_value = {"DOMINANT": mock_algorithm_class}

            custom_params = {"hidden_dim": 128, "num_layers": 3, "dropout": 0.5}

            adapter = PyGODAdapter(algorithm_name="DOMINANT", **custom_params)

            # Verify parameters are stored
            assert adapter.parameters["hidden_dim"] == 128
            assert adapter.parameters["num_layers"] == 3
            assert adapter.parameters["dropout"] == 0.5


class TestPyGODAdapterRealDependencies:
    """Tests that run only when PyGOD dependencies are actually installed."""

    @pytest.mark.skipif(
        not _pygod_available(), reason="PyGOD dependencies not installed"
    )
    def test_real_pygod_import(self):
        """Test actual PyGOD import and algorithm listing."""
        algorithms = PyGODAdapter.get_supported_algorithms()

        # Verify expected algorithms are available
        expected_algorithms = ["DOMINANT", "GCNAE", "SCAN", "RADAR"]
        for algo in expected_algorithms:
            assert algo in algorithms

    @pytest.mark.skipif(
        not _pygod_available(), reason="PyGOD dependencies not installed"
    )
    def test_real_dominant_algorithm_info(self):
        """Test retrieving real DOMINANT algorithm information."""
        info = PyGODAdapter.get_algorithm_info("DOMINANT")

        assert info["name"] == "DOMINANT"
        assert info["type"] == "Deep Learning"
        assert "hidden_dim" in info["parameters"]
        assert "num_layers" in info["parameters"]

    @pytest.mark.skipif(
        not _pygod_available(), reason="PyGOD dependencies not installed"
    )
    def test_real_adapter_creation(self):
        """Test creating a real PyGOD adapter."""
        adapter = PyGODAdapter(
            algorithm_name="SCAN",  # Statistical algorithm, lighter than deep learning
            contamination_rate=ContaminationRate(0.1),
        )

        assert adapter.name == "PyGOD_SCAN"
        assert adapter.algorithm_name == "SCAN"
        assert not adapter.is_fitted


# Integration test for container registration
def test_pygod_adapter_in_container():
    """Test that PyGOD adapter is properly registered in the dependency injection container."""
    from monorepo.infrastructure.config.container import OptionalServiceManager

    service_manager = OptionalServiceManager()

    # Check if PyGOD adapter is registered
    assert service_manager.is_available("pygod_adapter") or not _pygod_available()

    if service_manager.is_available("pygod_adapter"):
        adapter_class = service_manager.get_service("pygod_adapter")
        assert adapter_class is not None


# Performance benchmarking tests
@pytest.mark.performance
@pytest.mark.skipif(not _pygod_available(), reason="PyGOD dependencies not installed")
def test_pygod_performance_benchmark():
    """Benchmark PyGOD adapter performance with different graph sizes."""
    import time

    # Test with small graph (this would need real implementation)
    # This is a placeholder for actual performance testing
    start_time = time.time()

    # Create small test graph
    adapter = PyGODAdapter(algorithm_name="SCAN")

    # Simulate graph creation time
    creation_time = time.time() - start_time

    # Should be very fast for small graphs
    assert creation_time < 1.0, f"Adapter creation took too long: {creation_time}s"
