"""
Branch Coverage Enhancement - Algorithm-Specific Branch Testing
Comprehensive tests targeting algorithm-specific conditional branches and decision paths.
"""

import os
import sys
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from pynomaly.domain.entities import DetectionResult
from pynomaly.domain.exceptions import (
    InvalidAlgorithmError,
    ValidationError,
)
from pynomaly.domain.value_objects import ContaminationRate


class TestAlgorithmSelectionBranches:
    """Test algorithm selection and configuration branches."""

    def test_algorithm_availability_branches(self):
        """Test algorithm availability checking branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            # Test available algorithm
            mock_module = Mock()
            mock_module.IForest = Mock()
            mock_import.return_value = mock_module

            adapter = PyODAdapter(algorithm_name="IsolationForest")
            assert adapter.algorithm_name == "IsolationForest"

            # Test algorithm with missing dependencies
            mock_import.side_effect = ImportError("Module not found")

            with pytest.raises(InvalidAlgorithmError):
                PyODAdapter(algorithm_name="NonExistentAlgorithm")

            # Test algorithm with version compatibility issues
            mock_import.side_effect = None
            mock_import.return_value = mock_module

            # Mock version check
            with patch.object(
                PyODAdapter, "_check_algorithm_compatibility"
            ) as mock_check:
                mock_check.return_value = False

                with pytest.raises(InvalidAlgorithmError, match="incompatible"):
                    adapter = PyODAdapter(algorithm_name="IsolationForest")
                    adapter._check_algorithm_compatibility()

    def test_parameter_validation_branches(self):
        """Test algorithm parameter validation branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.IForest = Mock()
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            # Test valid parameters
            valid_adapter = PyODAdapter(
                algorithm_name="IsolationForest",
                n_estimators=100,
                contamination=0.1,
                random_state=42,
            )
            assert valid_adapter.parameters["n_estimators"] == 100

            # Test parameter type validation
            with pytest.raises(ValidationError):
                PyODAdapter(
                    algorithm_name="IsolationForest",
                    n_estimators="invalid_type",  # Should be int
                )

            # Test parameter range validation
            with pytest.raises(ValidationError):
                PyODAdapter(
                    algorithm_name="IsolationForest",
                    n_estimators=-1,  # Should be positive
                )

            # Test algorithm-specific parameter validation
            lof_adapter = PyODAdapter(algorithm_name="LOF")

            # Valid LOF parameters
            lof_adapter.set_params(n_neighbors=20, metric="euclidean")
            assert lof_adapter.parameters["n_neighbors"] == 20

            # Invalid LOF parameters
            with pytest.raises(ValidationError):
                lof_adapter.set_params(n_neighbors=0)  # Should be > 0

    def test_contamination_rate_branches(self):
        """Test contamination rate handling branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.IForest = Mock()
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            # Test auto contamination rate
            auto_adapter = PyODAdapter(
                algorithm_name="IsolationForest",
                contamination_rate=ContaminationRate.auto(),
            )
            assert auto_adapter.contamination_rate.is_auto
            assert auto_adapter.contamination_rate.value == 0.1  # Default

            # Test explicit contamination rate
            explicit_adapter = PyODAdapter(
                algorithm_name="IsolationForest",
                contamination_rate=ContaminationRate(0.05),
            )
            assert not explicit_adapter.contamination_rate.is_auto
            assert explicit_adapter.contamination_rate.value == 0.05

            # Test contamination rate inference from data
            sample_data = pd.DataFrame(
                {
                    "feature1": np.concatenate(
                        [
                            np.random.normal(0, 1, 950),  # Normal data
                            np.random.normal(
                                5, 1, 50
                            ),  # Anomalous data (5% contamination)
                        ]
                    ),
                    "feature2": np.concatenate(
                        [np.random.normal(0, 1, 950), np.random.normal(5, 1, 50)]
                    ),
                }
            )

            dataset = Mock()
            dataset.features = sample_data
            dataset.get_numeric_features.return_value = ["feature1", "feature2"]
            dataset.created_at = datetime.now(UTC)
            dataset.n_samples = 1000
            dataset.name = "test_dataset"

            inferred_adapter = PyODAdapter(
                algorithm_name="IsolationForest",
                contamination_rate=ContaminationRate.auto(),
            )

            # Mock model fitting with contamination inference
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.contamination_ = 0.05  # Inferred contamination

            inferred_adapter._model_class.return_value = mock_model
            inferred_adapter.fit(dataset)

            # Check if contamination was updated based on inference
            if hasattr(inferred_adapter, "_update_contamination_from_model"):
                inferred_adapter._update_contamination_from_model()
                assert abs(inferred_adapter.contamination_rate.value - 0.05) < 0.01


class TestAlgorithmSpecificBranches:
    """Test algorithm-specific conditional branches."""

    def test_isolation_forest_branches(self):
        """Test IsolationForest-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.IForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Test different bootstrap configurations
            bootstrap_configs = [True, False]
            for bootstrap in bootstrap_configs:
                adapter.set_params(bootstrap=bootstrap)
                assert adapter.parameters["bootstrap"] == bootstrap

            # Test different max_samples configurations
            max_samples_configs = [None, 256, 0.5, "auto"]
            for max_samples in max_samples_configs:
                if max_samples == "auto":
                    # Special handling for auto
                    adapter.set_params(max_samples=max_samples)
                    # Should convert to appropriate value
                    assert adapter.parameters["max_samples"] in [None, "auto"]
                else:
                    adapter.set_params(max_samples=max_samples)
                    assert adapter.parameters["max_samples"] == max_samples

            # Test warm_start functionality
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.warm_start = False
            mock_model_class.return_value = mock_model

            # Test without warm start
            adapter.set_params(warm_start=False)
            sample_dataset = self._create_sample_dataset()
            adapter.fit(sample_dataset)
            assert mock_model.warm_start == False

            # Test with warm start
            adapter.set_params(warm_start=True)
            mock_model.warm_start = True
            adapter.fit(sample_dataset)
            assert mock_model.warm_start == True

    def test_lof_algorithm_branches(self):
        """Test LOF-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.LOF = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="LOF")

            # Test different distance metrics
            metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
            for metric in metrics:
                adapter.set_params(metric=metric)
                assert adapter.parameters["metric"] == metric

            # Test different algorithms for neighbor search
            algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
            for algorithm in algorithms:
                adapter.set_params(algorithm=algorithm)
                assert adapter.parameters["algorithm"] == algorithm

            # Test novelty detection mode
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model_class.return_value = mock_model

            # Test with novelty=False (default)
            adapter.set_params(novelty=False)
            sample_dataset = self._create_sample_dataset()
            adapter.fit(sample_dataset)

            # For novelty=False, predict should work on training data
            mock_model.predict.return_value = np.array([1, -1, 1, -1, 1])
            mock_model.decision_function.return_value = np.array(
                [0.1, 0.9, 0.2, 0.8, 0.3]
            )

            result = adapter.detect(sample_dataset)
            assert isinstance(result, DetectionResult)

            # Test with novelty=True
            adapter.set_params(novelty=True)
            adapter.fit(sample_dataset)

            # With novelty=True, should be able to predict on new data
            new_dataset = self._create_sample_dataset(n_samples=10)
            mock_model.predict.return_value = np.array(
                [1, -1, 1, -1, 1, 1, -1, 1, -1, 1]
            )
            mock_model.decision_function.return_value = np.random.random(10)

            new_result = adapter.detect(new_dataset)
            assert isinstance(new_result, DetectionResult)

    def test_ocsvm_algorithm_branches(self):
        """Test OneClassSVM-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.OneClassSVM = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            adapter = SklearnAdapter(algorithm_name="OneClassSVM")

            # Test different kernel configurations
            kernels = ["rbf", "linear", "poly", "sigmoid"]
            for kernel in kernels:
                adapter.set_params(kernel=kernel)
                assert adapter.parameters["kernel"] == kernel

            # Test gamma parameter branches
            gamma_values = ["scale", "auto", 0.001, 0.1, 1.0]
            for gamma in gamma_values:
                adapter.set_params(gamma=gamma)
                assert adapter.parameters["gamma"] == gamma

            # Test polynomial kernel specific parameters
            adapter.set_params(kernel="poly")

            # Test different polynomial degrees
            degrees = [1, 2, 3, 4, 5]
            for degree in degrees:
                adapter.set_params(degree=degree)
                assert adapter.parameters["degree"] == degree

            # Test coef0 parameter for poly and sigmoid kernels
            coef0_values = [0.0, 0.1, 1.0, -0.5]
            for coef0 in coef0_values:
                adapter.set_params(coef0=coef0)
                assert adapter.parameters["coef0"] == coef0

            # Test cache_size optimization
            cache_sizes = [100, 200, 500, 1000]
            for cache_size in cache_sizes:
                adapter.set_params(cache_size=cache_size)
                assert adapter.parameters["cache_size"] == cache_size

    def test_autoencoder_branches(self):
        """Test autoencoder-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.AutoEncoder = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="AutoEncoder")

            # Test different hidden layer configurations
            hidden_configs = [
                [64, 32, 16, 32, 64],  # Symmetric
                [100, 50, 25],  # Decreasing
                [10, 20, 30, 20, 10],  # Increasing then decreasing
                [128],  # Single hidden layer
            ]

            for hidden_neurons in hidden_configs:
                adapter.set_params(hidden_neurons=hidden_neurons)
                assert adapter.parameters["hidden_neurons"] == hidden_neurons

            # Test different activation functions
            activations = ["relu", "sigmoid", "tanh", "linear"]
            for activation in activations:
                adapter.set_params(activation=activation)
                assert adapter.parameters["activation"] == activation

            # Test different optimizers
            optimizers = ["adam", "sgd", "rmsprop", "adagrad"]
            for optimizer in optimizers:
                adapter.set_params(optimizer=optimizer)
                assert adapter.parameters["optimizer"] == optimizer

            # Test learning rate scheduling
            learning_rates = [0.001, 0.01, 0.1, 0.0001]
            for lr in learning_rates:
                adapter.set_params(learning_rate=lr)
                assert adapter.parameters["learning_rate"] == lr

            # Test batch size configurations
            batch_sizes = [16, 32, 64, 128, 256]
            for batch_size in batch_sizes:
                adapter.set_params(batch_size=batch_size)
                assert adapter.parameters["batch_size"] == batch_size

            # Test regularization options
            adapter.set_params(l2_regularizer=0.01)
            assert adapter.parameters["l2_regularizer"] == 0.01

            adapter.set_params(dropout_rate=0.2)
            assert adapter.parameters["dropout_rate"] == 0.2

    def _create_sample_dataset(self, n_samples=100, n_features=5):
        """Helper method to create sample dataset."""
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        dataset = Mock()
        dataset.features = data
        dataset.get_numeric_features.return_value = list(data.columns)
        dataset.created_at = datetime.now(UTC)
        dataset.n_samples = n_samples
        dataset.name = "sample_dataset"

        return dataset


class TestEnsembleAlgorithmBranches:
    """Test ensemble algorithm-specific branches."""

    def test_feature_bagging_branches(self):
        """Test FeatureBagging-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.FeatureBagging = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="FeatureBagging")

            # Test different base estimators
            base_estimators = [
                {"class": "LOF", "params": {"n_neighbors": 5}},
                {"class": "IForest", "params": {"n_estimators": 50}},
                {"class": "PCA", "params": {"n_components": 2}},
            ]

            for base_estimator in base_estimators:
                adapter.set_params(base_estimator=base_estimator)
                assert adapter.parameters["base_estimator"] == base_estimator

            # Test different feature sampling strategies
            max_features_configs = [0.5, 0.8, 10, None]
            for max_features in max_features_configs:
                adapter.set_params(max_features=max_features)
                assert adapter.parameters["max_features"] == max_features

            # Test bootstrap feature sampling
            bootstrap_features_configs = [True, False]
            for bootstrap_features in bootstrap_features_configs:
                adapter.set_params(bootstrap_features=bootstrap_features)
                assert adapter.parameters["bootstrap_features"] == bootstrap_features

            # Test different combination methods
            combination_methods = ["average", "max", "median"]
            for method in combination_methods:
                adapter.set_params(combination=method)
                assert adapter.parameters["combination"] == method

    def test_lscp_ensemble_branches(self):
        """Test LSCP ensemble-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.LSCP = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="LSCP")

            # Test different detector lists
            detector_lists = [
                ["LOF", "IForest", "OCSVM"],
                ["PCA", "MCD", "COPOD"],
                ["AutoEncoder", "VAE"],
            ]

            for detector_list in detector_lists:
                adapter.set_params(detector_list=detector_list)
                assert adapter.parameters["detector_list"] == detector_list

            # Test different local region configurations
            local_region_configs = [5, 10, 15, 20]
            for local_region_size in local_region_configs:
                adapter.set_params(local_region_size=local_region_size)
                assert adapter.parameters["local_region_size"] == local_region_size

            # Test different local region methods
            local_methods = ["knn", "radius", "clustering"]
            for method in local_methods:
                adapter.set_params(local_region_method=method)
                assert adapter.parameters["local_region_method"] == method

            # Test parallel execution options
            n_jobs_configs = [1, 2, 4, -1]
            for n_jobs in n_jobs_configs:
                adapter.set_params(n_jobs=n_jobs)
                assert adapter.parameters["n_jobs"] == n_jobs

    def test_suod_ensemble_branches(self):
        """Test SUOD ensemble-specific branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.SUOD = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="SUOD")

            # Test different base detector configurations
            base_detectors = [
                ["IForest", "LOF", "OCSVM", "PCA"],
                ["COPOD", "ECOD", "HBOS"],
                ["AutoEncoder", "VAE", "SO_GAAL"],
            ]

            for detectors in base_detectors:
                adapter.set_params(base_detectors=detectors)
                assert adapter.parameters["base_detectors"] == detectors

            # Test different approximation configurations
            approximation_configs = [True, False]
            for use_approximation in approximation_configs:
                adapter.set_params(approximation_flag=use_approximation)
                assert adapter.parameters["approximation_flag"] == use_approximation

            # Test different combination strategies
            combination_strategies = ["average", "maximization", "majority_vote"]
            for strategy in combination_strategies:
                adapter.set_params(combination=strategy)
                assert adapter.parameters["combination"] == strategy

            # Test batch processing configurations
            batch_sizes = [100, 500, 1000, 5000]
            for batch_size in batch_sizes:
                adapter.set_params(bps_flag=True, batch_size=batch_size)
                assert adapter.parameters["batch_size"] == batch_size

            # Test different worker configurations
            n_jobs_configs = [1, 2, 4, 8, -1]
            for n_jobs in n_jobs_configs:
                adapter.set_params(n_jobs=n_jobs)
                assert adapter.parameters["n_jobs"] == n_jobs


class TestAlgorithmFittingBranches:
    """Test algorithm fitting process branches."""

    def test_fitting_with_different_data_sizes(self):
        """Test fitting branches with different data sizes."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model_class.return_value = mock_model
            mock_module.IForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Test fitting with very small dataset
            small_data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
            small_dataset = self._create_dataset(small_data)

            # Small dataset might require parameter adjustment
            if hasattr(adapter, "_adjust_parameters_for_small_data"):
                adapter._adjust_parameters_for_small_data(small_dataset)

            adapter.fit(small_dataset)
            assert adapter.is_fitted

            # Test fitting with large dataset
            large_data = pd.DataFrame(
                np.random.randn(10000, 50), columns=[f"feature_{i}" for i in range(50)]
            )
            large_dataset = self._create_dataset(large_data)

            # Large dataset might trigger memory optimization
            if hasattr(adapter, "_optimize_for_large_data"):
                adapter._optimize_for_large_data(large_dataset)

            adapter.fit(large_dataset)
            assert adapter.is_fitted

            # Test fitting with wide dataset (many features)
            wide_data = pd.DataFrame(
                np.random.randn(100, 1000),
                columns=[f"feature_{i}" for i in range(1000)],
            )
            wide_dataset = self._create_dataset(wide_data)

            # Wide dataset might require feature selection
            if hasattr(adapter, "_handle_high_dimensional_data"):
                adapter._handle_high_dimensional_data(wide_dataset)

            adapter.fit(wide_dataset)
            assert adapter.is_fitted

    def test_fitting_convergence_branches(self):
        """Test fitting convergence handling branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.AutoEncoder = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="AutoEncoder")

            # Test successful convergence
            mock_model_converged = Mock()
            mock_model_converged.fit.return_value = None
            mock_model_converged.converged_ = True
            mock_model_converged.n_iter_ = 50
            mock_model_class.return_value = mock_model_converged

            sample_dataset = self._create_sample_dataset()
            adapter.fit(sample_dataset)

            if hasattr(mock_model_converged, "converged_"):
                assert adapter.metadata.get("converged") == True
                assert adapter.metadata.get("n_iterations") == 50

            # Test non-convergence warning
            mock_model_not_converged = Mock()
            mock_model_not_converged.fit.return_value = None
            mock_model_not_converged.converged_ = False
            mock_model_not_converged.n_iter_ = 1000  # Max iterations reached
            mock_model_class.return_value = mock_model_not_converged

            adapter_nc = PyODAdapter(algorithm_name="AutoEncoder")

            with pytest.warns(UserWarning, match="did not converge"):
                adapter_nc.fit(sample_dataset)

            if hasattr(mock_model_not_converged, "converged_"):
                assert adapter_nc.metadata.get("converged") == False

    def test_incremental_learning_branches(self):
        """Test incremental learning branches."""
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_module.SGDOneClassSVM = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            adapter = SklearnAdapter(algorithm_name="SGDOneClassSVM")

            # Test initial fitting
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.partial_fit = Mock()
            mock_model_class.return_value = mock_model

            initial_dataset = self._create_sample_dataset(n_samples=1000)
            adapter.fit(initial_dataset)
            assert adapter.is_fitted

            # Test incremental update
            if hasattr(mock_model, "partial_fit"):
                new_data_dataset = self._create_sample_dataset(n_samples=100)

                if hasattr(adapter, "partial_fit"):
                    adapter.partial_fit(new_data_dataset)
                    mock_model.partial_fit.assert_called()
                else:
                    # Fallback to full refit
                    adapter.fit(new_data_dataset)
                    assert mock_model.fit.call_count >= 2

    def _create_dataset(self, data):
        """Helper method to create dataset from DataFrame."""
        dataset = Mock()
        dataset.features = data
        dataset.get_numeric_features.return_value = list(data.columns)
        dataset.created_at = datetime.now(UTC)
        dataset.n_samples = len(data)
        dataset.name = "test_dataset"
        return dataset

    def _create_sample_dataset(self, n_samples=100, n_features=5):
        """Helper method to create sample dataset."""
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        return self._create_dataset(data)


class TestDetectionBranches:
    """Test detection process branches."""

    def test_score_normalization_branches(self):
        """Test score normalization branches."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model_class.return_value = mock_model
            mock_module.IForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            adapter = PyODAdapter(algorithm_name="IsolationForest")

            # Test with normal score range
            mock_model.decision_function.return_value = np.array(
                [0.1, 0.5, 0.9, 0.2, 0.8]
            )
            mock_model.predict.return_value = np.array([0, 1, 1, 0, 1])

            sample_dataset = self._create_sample_dataset(n_samples=5)
            adapter.fit(sample_dataset)
            adapter._model = mock_model

            result = adapter.detect(sample_dataset)

            # Scores should be normalized to [0, 1]
            for score in result.scores:
                assert 0 <= score.value <= 1

            # Test with extreme score range
            mock_model.decision_function.return_value = np.array(
                [-1000, -500, 0, 500, 1000]
            )

            extreme_result = adapter.detect(sample_dataset)

            # Should still be normalized
            for score in extreme_result.scores:
                assert 0 <= score.value <= 1

            # Test with constant scores (edge case)
            mock_model.decision_function.return_value = np.array(
                [0.5, 0.5, 0.5, 0.5, 0.5]
            )

            constant_result = adapter.detect(sample_dataset)

            # Should handle constant scores gracefully
            for score in constant_result.scores:
                assert 0 <= score.value <= 1

    def test_threshold_calculation_branches(self):
        """Test threshold calculation branches."""
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_model_class = Mock()
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model_class.return_value = mock_model
            mock_module.IsolationForest = mock_model_class
            mock_import.return_value = mock_module

            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            # Test with different contamination rates
            contamination_rates = [0.01, 0.05, 0.1, 0.2, 0.4]

            for contamination in contamination_rates:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    contamination_rate=ContaminationRate(contamination),
                )

                # Mock scores that would result in the expected contamination
                n_samples = 1000
                n_anomalies = int(n_samples * contamination)

                # Create scores where top n_anomalies have high scores
                scores = np.random.random(n_samples)
                scores[:n_anomalies] = np.random.uniform(0.8, 1.0, n_anomalies)
                scores[n_anomalies:] = np.random.uniform(
                    0.0, 0.7, n_samples - n_anomalies
                )

                mock_model.score_samples.return_value = (
                    -scores
                )  # sklearn uses negative scores
                mock_model.predict.return_value = np.concatenate(
                    [
                        [-1] * n_anomalies,  # Anomalies
                        [1] * (n_samples - n_anomalies),  # Normal
                    ]
                )

                sample_dataset = self._create_sample_dataset(n_samples=n_samples)
                adapter.fit(sample_dataset)
                adapter._model = mock_model

                result = adapter.detect(sample_dataset)

                # Check that threshold roughly corresponds to contamination rate
                expected_threshold_idx = int(n_samples * (1 - contamination))
                sorted_scores = sorted([s.value for s in result.scores])
                expected_threshold = sorted_scores[expected_threshold_idx]

                # Allow some tolerance due to randomness
                assert abs(result.threshold - expected_threshold) < 0.1

    def _create_sample_dataset(self, n_samples=100, n_features=5):
        """Helper method to create sample dataset."""
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        dataset = Mock()
        dataset.features = data
        dataset.get_numeric_features.return_value = list(data.columns)
        dataset.created_at = datetime.now(UTC)
        dataset.n_samples = n_samples
        dataset.name = "sample_dataset"

        return dataset


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
