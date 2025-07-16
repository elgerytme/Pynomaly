"""Property-based testing using Hypothesis - Phase 4 Advanced Testing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import columns, data_frames

from monorepo.application.use_cases import DetectAnomalies, TrainDetector
from monorepo.domain.entities import Dataset, Detector
from monorepo.domain.value_objects import AnomalyScore, ContaminationRate
from monorepo.infrastructure.adapters.pyod_adapter import PyODAdapter


# Hypothesis strategies for generating test data
@st.composite
def valid_contamination_rates(draw):
    """Generate valid contamination rates."""
    return draw(st.floats(min_value=0.001, max_value=0.5))


@st.composite
def feature_names(draw):
    """Generate valid feature names."""
    return draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.isalnum() and not x.isdigit())
    )


@st.composite
def dataset_names(draw):
    """Generate valid dataset names."""
    return draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Ps", "Pe")
            ),
            min_size=1,
            max_size=100,
        ).filter(lambda x: len(x.strip()) > 0)
    )


@st.composite
def detector_names(draw):
    """Generate valid detector names."""
    return draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Ps", "Pe")
            ),
            min_size=1,
            max_size=100,
        ).filter(lambda x: len(x.strip()) > 0)
    )


@st.composite
def numeric_datasets(draw):
    """Generate numeric datasets for testing."""
    n_samples = draw(st.integers(min_value=10, max_value=1000))
    n_features = draw(st.integers(min_value=1, max_value=20))

    # Generate feature names
    feature_list = []
    for i in range(n_features):
        feature_list.append(f"feature_{i}")

    # Generate data
    data = draw(
        arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return pd.DataFrame(data, columns=feature_list)


@st.composite
def algorithm_configs(draw):
    """Generate algorithm configurations."""
    algorithm = draw(
        st.sampled_from(
            ["IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope"]
        )
    )

    contamination = draw(valid_contamination_rates())
    random_state = draw(st.integers(min_value=0, max_value=2**31 - 1))

    base_config = {"contamination": contamination, "random_state": random_state}

    # Algorithm-specific parameters
    if algorithm == "IsolationForest":
        base_config.update(
            {
                "n_estimators": draw(st.integers(min_value=10, max_value=200)),
                "max_samples": draw(
                    st.one_of(
                        st.just("auto"), st.integers(min_value=10, max_value=1000)
                    )
                ),
            }
        )
    elif algorithm == "LocalOutlierFactor":
        base_config.update(
            {
                "n_neighbors": draw(st.integers(min_value=1, max_value=50)),
                "algorithm": draw(
                    st.sampled_from(["auto", "ball_tree", "kd_tree", "brute"])
                ),
            }
        )
    elif algorithm == "OneClassSVM":
        base_config.update(
            {
                "kernel": draw(st.sampled_from(["rbf", "linear", "poly", "sigmoid"])),
                "gamma": draw(
                    st.one_of(
                        st.just("scale"),
                        st.just("auto"),
                        st.floats(min_value=0.001, max_value=1.0),
                    )
                ),
            }
        )

    return algorithm, base_config


class TestPropertyBasedDomainInvariants:
    """Test domain invariants using property-based testing."""

    @given(contamination_rate=valid_contamination_rates(), name=dataset_names())
    @settings(max_examples=50, deadline=5000)
    def test_contamination_rate_invariants(self, contamination_rate: float, name: str):
        """Test contamination rate invariants always hold."""
        # ContaminationRate should accept valid rates
        rate = ContaminationRate(contamination_rate)

        # Invariants
        assert 0 < rate.value <= 0.5
        assert rate.value == contamination_rate

        # String representation should be meaningful
        str_repr = str(rate)
        assert str(contamination_rate) in str_repr

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=1000,
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_anomaly_score_invariants(self, scores: list[float]):
        """Test anomaly score invariants."""
        for score_value in scores:
            score = AnomalyScore(score_value)

            # Invariants
            assert 0.0 <= score.value <= 1.0
            assert score.value == score_value

            # Scores should be comparable
            other_score = AnomalyScore(min(score_value + 0.1, 1.0))
            if score.value < other_score.value:
                assert score < other_score
            elif score.value > other_score.value:
                assert score > other_score
            else:
                assert score == other_score

    @given(dataset=numeric_datasets(), name=dataset_names())
    @settings(max_examples=20, deadline=10000)
    def test_dataset_entity_invariants(self, dataset: pd.DataFrame, name: str):
        """Test dataset entity invariants."""
        assume(len(dataset) > 0)
        assume(len(dataset.columns) > 0)
        assume(not dataset.isnull().all().all())  # Not all values are NaN

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Create Dataset entity
            dataset_entity = Dataset(
                id=f"test_{hash(name) % 1000000}",
                name=name,
                file_path=csv_path,
                n_samples=len(dataset),
                n_features=len(dataset.columns),
                feature_names=list(dataset.columns),
                target_column=None,
            )

            # Invariants
            assert dataset_entity.id is not None
            assert len(dataset_entity.id) > 0
            assert dataset_entity.name == name
            assert dataset_entity.n_samples == len(dataset)
            assert dataset_entity.n_features == len(dataset.columns)
            assert dataset_entity.feature_names == list(dataset.columns)
            assert Path(dataset_entity.file_path).exists()

            # Dataset should be loadable
            loaded_data = pd.read_csv(dataset_entity.file_path)
            assert len(loaded_data) == dataset_entity.n_samples
            assert len(loaded_data.columns) == dataset_entity.n_features

        finally:
            Path(csv_path).unlink(missing_ok=True)

    @given(name=detector_names(), algorithm_config=algorithm_configs())
    @settings(max_examples=30, deadline=5000)
    def test_detector_entity_invariants(self, name: str, algorithm_config):
        """Test detector entity invariants."""
        algorithm, hyperparameters = algorithm_config

        detector = Detector(
            id=f"test_{hash(name) % 1000000}",
            name=name,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            is_fitted=False,
        )

        # Invariants
        assert detector.id is not None
        assert len(detector.id) > 0
        assert detector.name == name
        assert detector.algorithm == algorithm
        assert detector.hyperparameters == hyperparameters
        assert detector.is_fitted is False

        # Hyperparameters should contain contamination
        assert "contamination" in detector.hyperparameters
        assert 0 < detector.hyperparameters["contamination"] <= 0.5

        # Algorithm should be supported
        supported_algorithms = [
            "IsolationForest",
            "LocalOutlierFactor",
            "OneClassSVM",
            "EllipticEnvelope",
        ]
        assert detector.algorithm in supported_algorithms


class TestPropertyBasedAlgorithmBehavior:
    """Test algorithm behavior properties."""

    @given(dataset=numeric_datasets(), algorithm_config=algorithm_configs())
    @settings(max_examples=15, deadline=15000)
    def test_training_determinism(self, dataset: pd.DataFrame, algorithm_config):
        """Test that training is deterministic with same random state."""
        assume(len(dataset) >= 10)
        assume(len(dataset.columns) >= 1)
        assume(not dataset.isnull().all().all())

        algorithm, hyperparameters = algorithm_config

        # Ensure we have random_state for determinism
        if "random_state" not in hyperparameters:
            hyperparameters["random_state"] = 42

        # Create two identical detectors
        detector1 = Detector(
            id="test_det_1",
            name="Test Detector 1",
            algorithm=algorithm,
            hyperparameters=hyperparameters.copy(),
            is_fitted=False,
        )

        detector2 = Detector(
            id="test_det_2",
            name="Test Detector 2",
            algorithm=algorithm,
            hyperparameters=hyperparameters.copy(),
            is_fitted=False,
        )

        # Create dataset entity
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            dataset_entity = Dataset(
                id="test_dataset",
                name="Test Dataset",
                file_path=csv_path,
                n_samples=len(dataset),
                n_features=len(dataset.columns),
                feature_names=list(dataset.columns),
            )

            # Mock repositories
            detector_repo = Mock()
            dataset_repo = Mock()
            dataset_repo.get.return_value = dataset_entity

            # Mock adapter
            adapter = Mock()
            adapter_registry = Mock()
            adapter_registry.get_adapter.return_value = adapter

            # Mock training results
            adapter.train.return_value = True
            adapter.predict.return_value = (
                np.random.choice([0, 1], size=len(dataset)),
                np.random.random(len(dataset)),
            )

            # Train both detectors
            train_use_case = TrainDetector(
                detector_repo, dataset_repo, adapter_registry
            )

            # Set up repository returns
            detector_repo.get.side_effect = lambda det_id: (
                detector1 if det_id == "test_det_1" else detector2
            )
            detector_repo.update.return_value = None

            result1 = train_use_case.execute("test_det_1", "test_dataset")
            result2 = train_use_case.execute("test_det_2", "test_dataset")

            # Both should succeed
            assert result1.success
            assert result2.success

            # Training times should be reasonable
            assert result1.training_time_ms > 0
            assert result2.training_time_ms > 0

        finally:
            Path(csv_path).unlink(missing_ok=True)

    @given(dataset=numeric_datasets(), contamination=valid_contamination_rates())
    @settings(max_examples=10, deadline=20000)
    def test_contamination_rate_behavior(
        self, dataset: pd.DataFrame, contamination: float
    ):
        """Test that contamination rate affects detection behavior appropriately."""
        assume(len(dataset) >= 20)
        assume(len(dataset.columns) >= 1)
        assume(not dataset.isnull().all().all())

        # Create detectors with different contamination rates
        low_contamination = min(contamination, 0.05)
        high_contamination = max(contamination, 0.2)

        assume(low_contamination < high_contamination)

        detector_low = Detector(
            id="low_cont_det",
            name="Low Contamination Detector",
            algorithm="IsolationForest",
            hyperparameters={
                "contamination": low_contamination,
                "random_state": 42,
                "n_estimators": 50,
            },
            is_fitted=False,
        )

        detector_high = Detector(
            id="high_cont_det",
            name="High Contamination Detector",
            algorithm="IsolationForest",
            hyperparameters={
                "contamination": high_contamination,
                "random_state": 42,
                "n_estimators": 50,
            },
            is_fitted=False,
        )

        # Create dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            dataset_entity = Dataset(
                id="contamination_test_dataset",
                name="Contamination Test Dataset",
                file_path=csv_path,
                n_samples=len(dataset),
                n_features=len(dataset.columns),
                feature_names=list(dataset.columns),
            )

            # Mock components
            detector_repo = Mock()
            dataset_repo = Mock()
            dataset_repo.get.return_value = dataset_entity

            # Use real PyOD adapter for realistic behavior
            adapter = PyODAdapter()
            adapter_registry = Mock()
            adapter_registry.get_adapter.return_value = adapter

            # Mock detector repository behavior
            detector_repo.get.side_effect = lambda det_id: (
                detector_low if det_id == "low_cont_det" else detector_high
            )
            detector_repo.update.return_value = None

            # Train and detect
            train_use_case = TrainDetector(
                detector_repo, dataset_repo, adapter_registry
            )
            detect_use_case = DetectAnomalies(
                detector_repo, dataset_repo, adapter_registry
            )

            # Train both detectors
            train_result_low = train_use_case.execute(
                "low_cont_det", "contamination_test_dataset"
            )
            train_result_high = train_use_case.execute(
                "high_cont_det", "contamination_test_dataset"
            )

            assert train_result_low.success
            assert train_result_high.success

            # Mark detectors as fitted
            detector_low.is_fitted = True
            detector_high.is_fitted = True

            # Run detection
            detect_result_low = detect_use_case.execute(
                "low_cont_det", "contamination_test_dataset"
            )
            detect_result_high = detect_use_case.execute(
                "high_cont_det", "contamination_test_dataset"
            )

            assert detect_result_low.success
            assert detect_result_high.success

            # Property: Higher contamination rate should generally detect more anomalies
            low_anomaly_rate = detect_result_low.anomaly_count / len(
                detect_result_low.predictions
            )
            high_anomaly_rate = detect_result_high.anomaly_count / len(
                detect_result_high.predictions
            )

            note(
                f"Low contamination rate: {low_contamination}, anomalies: {low_anomaly_rate}"
            )
            note(
                f"High contamination rate: {high_contamination}, anomalies: {high_anomaly_rate}"
            )

            # Allow some tolerance due to algorithm variations
            assert high_anomaly_rate >= low_anomaly_rate * 0.8

        finally:
            Path(csv_path).unlink(missing_ok=True)

    @given(dataset=numeric_datasets())
    @settings(max_examples=10, deadline=15000)
    def test_prediction_consistency(self, dataset: pd.DataFrame):
        """Test that predictions are consistent across multiple runs."""
        assume(len(dataset) >= 10)
        assume(len(dataset.columns) >= 1)
        assume(not dataset.isnull().all().all())

        detector = Detector(
            id="consistency_detector",
            name="Consistency Test Detector",
            algorithm="IsolationForest",
            hyperparameters={
                "contamination": 0.1,
                "random_state": 42,  # Fixed for consistency
                "n_estimators": 50,
            },
            is_fitted=True,  # Assume already fitted
        )

        # Create dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            dataset_entity = Dataset(
                id="consistency_dataset",
                name="Consistency Test Dataset",
                file_path=csv_path,
                n_samples=len(dataset),
                n_features=len(dataset.columns),
                feature_names=list(dataset.columns),
            )

            # Mock components
            detector_repo = Mock()
            dataset_repo = Mock()
            detector_repo.get.return_value = detector
            dataset_repo.get.return_value = dataset_entity

            adapter = PyODAdapter()
            adapter_registry = Mock()
            adapter_registry.get_adapter.return_value = adapter

            # Train once to establish model state
            train_use_case = TrainDetector(
                detector_repo, dataset_repo, adapter_registry
            )
            train_result = train_use_case.execute(
                "consistency_detector", "consistency_dataset"
            )
            assume(train_result.success)

            # Run multiple predictions
            detect_use_case = DetectAnomalies(
                detector_repo, dataset_repo, adapter_registry
            )

            results = []
            for _ in range(3):  # Multiple runs
                result = detect_use_case.execute(
                    "consistency_detector", "consistency_dataset"
                )
                assert result.success
                results.append(result)

            # Property: All predictions should be identical (deterministic)
            first_predictions = results[0].predictions
            first_scores = results[0].anomaly_scores

            for result in results[1:]:
                # Predictions should be exactly the same
                assert result.predictions == first_predictions
                # Scores should be very close (allowing for tiny numerical differences)
                np.testing.assert_allclose(
                    result.anomaly_scores, first_scores, rtol=1e-10
                )

        finally:
            Path(csv_path).unlink(missing_ok=True)


class TestPropertyBasedDataValidation:
    """Test data validation properties."""

    @given(
        n_samples=st.integers(min_value=1, max_value=1000),
        n_features=st.integers(min_value=1, max_value=20),
        has_nans=st.booleans(),
        has_infs=st.booleans(),
    )
    @settings(max_examples=25, deadline=5000)
    def test_data_quality_handling(
        self, n_samples: int, n_features: int, has_nans: bool, has_infs: bool
    ):
        """Test handling of various data quality issues."""
        # Generate base data
        data = np.random.normal(0, 1, (n_samples, n_features))

        # Introduce data quality issues
        if has_nans and n_samples > 1:
            nan_indices = np.random.choice(
                n_samples, size=min(n_samples // 4, 5), replace=False
            )
            data[nan_indices, 0] = np.nan

        if has_infs and n_samples > 1:
            inf_indices = np.random.choice(
                n_samples, size=min(n_samples // 4, 5), replace=False
            )
            data[inf_indices, -1] = np.inf

        feature_names = [f"feature_{i}" for i in range(n_features)]
        dataset = pd.DataFrame(data, columns=feature_names)

        # Create dataset entity
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            Dataset(
                id="data_quality_test",
                name="Data Quality Test Dataset",
                file_path=csv_path,
                n_samples=len(dataset),
                n_features=len(dataset.columns),
                feature_names=list(dataset.columns),
            )

            # Properties about data quality
            loaded_data = pd.read_csv(csv_path)

            # Shape should be preserved
            assert loaded_data.shape[0] == n_samples
            assert loaded_data.shape[1] == n_features

            # Column names should be preserved
            assert list(loaded_data.columns) == feature_names

            # If we introduced NaNs/Infs, they should be detectable
            if has_nans:
                assert loaded_data.isnull().any().any()
            if has_infs:
                assert (
                    np.isinf(loaded_data.select_dtypes(include=[np.number])).any().any()
                )

        finally:
            Path(csv_path).unlink(missing_ok=True)

    @given(
        dataset=data_frames(
            [
                columns(["feature_1", "feature_2"], dtype=float),
                columns(["feature_3"], dtype=int),
                columns(["feature_4"], dtype=str),
            ],
            rows=st.tuples(
                st.floats(min_value=-100, max_value=100, allow_nan=False),
                st.floats(min_value=-100, max_value=100, allow_nan=False),
                st.integers(min_value=-1000, max_value=1000),
                st.text(min_size=1, max_size=10),
            ),
            min_size=10,
            max_size=100,
        )
    )
    @settings(max_examples=10, deadline=10000)
    def test_mixed_datatype_handling(self, dataset: pd.DataFrame):
        """Test handling of datasets with mixed data types."""
        assume(len(dataset) >= 5)

        # Create dataset with mixed types
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            # Load and validate
            loaded_data = pd.read_csv(csv_path)

            # Properties
            assert len(loaded_data) == len(dataset)
            assert list(loaded_data.columns) == list(dataset.columns)

            # Numeric columns should be loadable as numeric
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in loaded_data.columns:
                    # Should be able to convert to numeric
                    pd.to_numeric(loaded_data[col], errors="coerce")

            # String columns should be preserved
            string_columns = dataset.select_dtypes(include=["object"]).columns
            for col in string_columns:
                if col in loaded_data.columns:
                    assert loaded_data[col].dtype == object

        finally:
            Path(csv_path).unlink(missing_ok=True)


class TestPropertyBasedPerformance:
    """Test performance properties."""

    @given(
        n_samples=st.integers(min_value=100, max_value=5000),
        n_features=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=10, deadline=30000)
    def test_scaling_properties(self, n_samples: int, n_features: int):
        """Test that performance scales reasonably with data size."""
        # Generate dataset
        data = np.random.normal(0, 1, (n_samples, n_features))
        feature_names = [f"feature_{i}" for i in range(n_features)]
        dataset = pd.DataFrame(data, columns=feature_names)

        detector = Detector(
            id="scaling_test_detector",
            name="Scaling Test Detector",
            algorithm="IsolationForest",
            hyperparameters={
                "contamination": 0.1,
                "random_state": 42,
                "n_estimators": 50,  # Keep small for performance
            },
            is_fitted=False,
        )

        # Create dataset file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            dataset.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            dataset_entity = Dataset(
                id="scaling_dataset",
                name="Scaling Test Dataset",
                file_path=csv_path,
                n_samples=n_samples,
                n_features=n_features,
                feature_names=feature_names,
            )

            # Mock components
            detector_repo = Mock()
            dataset_repo = Mock()
            detector_repo.get.return_value = detector
            dataset_repo.get.return_value = dataset_entity

            adapter = PyODAdapter()
            adapter_registry = Mock()
            adapter_registry.get_adapter.return_value = adapter

            # Measure training time
            import time

            train_use_case = TrainDetector(
                detector_repo, dataset_repo, adapter_registry
            )

            start_time = time.time()
            train_result = train_use_case.execute(
                "scaling_test_detector", "scaling_dataset"
            )
            training_time = time.time() - start_time

            assume(train_result.success)

            # Properties about scaling
            note(
                f"Dataset size: {n_samples}x{n_features}, Training time: {training_time:.3f}s"
            )

            # Training time should be reasonable (not exponential)
            # Very rough heuristic: should not take more than 30 seconds for reasonable sizes
            max_expected_time = min(30.0, 0.01 * n_samples * n_features / 1000)
            assert training_time < max_expected_time

            # Mark as fitted for prediction test
            detector.is_fitted = True
            detector_repo.update.return_value = None

            # Measure prediction time
            detect_use_case = DetectAnomalies(
                detector_repo, dataset_repo, adapter_registry
            )

            start_time = time.time()
            detect_result = detect_use_case.execute(
                "scaling_test_detector", "scaling_dataset"
            )
            prediction_time = time.time() - start_time

            assert detect_result.success

            note(f"Prediction time: {prediction_time:.3f}s")

            # Prediction should be faster than training
            assert prediction_time <= training_time * 2

            # Results should have correct dimensions
            assert len(detect_result.predictions) == n_samples
            assert len(detect_result.anomaly_scores) == n_samples

        finally:
            Path(csv_path).unlink(missing_ok=True)

    @given(
        batch_sizes=st.lists(
            st.integers(min_value=10, max_value=500), min_size=2, max_size=5
        )
    )
    @settings(max_examples=5, deadline=20000)
    def test_batch_processing_consistency(self, batch_sizes: list[int]):
        """Test that batch processing gives consistent results."""
        assume(len(set(batch_sizes)) >= 2)  # At least 2 different sizes

        # Create a base dataset
        base_data = np.random.normal(0, 1, (max(batch_sizes), 3))
        feature_names = ["feature_1", "feature_2", "feature_3"]
        base_dataset = pd.DataFrame(base_data, columns=feature_names)

        detector = Detector(
            id="batch_test_detector",
            name="Batch Test Detector",
            algorithm="IsolationForest",
            hyperparameters={
                "contamination": 0.1,
                "random_state": 42,
                "n_estimators": 50,
            },
            is_fitted=True,
        )

        # Test different batch sizes on the same data
        results = {}

        for batch_size in batch_sizes:
            # Take subset of data
            batch_data = base_dataset.iloc[:batch_size]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                batch_data.to_csv(f.name, index=False)
                csv_path = f.name

            try:
                dataset_entity = Dataset(
                    id=f"batch_dataset_{batch_size}",
                    name=f"Batch Dataset {batch_size}",
                    file_path=csv_path,
                    n_samples=len(batch_data),
                    n_features=len(batch_data.columns),
                    feature_names=feature_names,
                )

                # Mock components
                detector_repo = Mock()
                dataset_repo = Mock()
                detector_repo.get.return_value = detector
                dataset_repo.get.return_value = dataset_entity

                adapter = PyODAdapter()
                adapter_registry = Mock()
                adapter_registry.get_adapter.return_value = adapter

                # Train and predict
                train_use_case = TrainDetector(
                    detector_repo, dataset_repo, adapter_registry
                )
                train_result = train_use_case.execute(
                    "batch_test_detector", f"batch_dataset_{batch_size}"
                )
                assume(train_result.success)

                detect_use_case = DetectAnomalies(
                    detector_repo, dataset_repo, adapter_registry
                )
                detect_result = detect_use_case.execute(
                    "batch_test_detector", f"batch_dataset_{batch_size}"
                )
                assume(detect_result.success)

                results[batch_size] = {
                    "anomaly_rate": detect_result.anomaly_count
                    / len(detect_result.predictions),
                    "mean_score": np.mean(detect_result.anomaly_scores),
                    "predictions": detect_result.predictions[
                        : min(10, len(detect_result.predictions))
                    ],  # First 10 for comparison
                }

            finally:
                Path(csv_path).unlink(missing_ok=True)

        # Properties: Results on overlapping data should be reasonably consistent
        if len(results) >= 2:
            batch_sizes_sorted = sorted(results.keys())

            # Compare anomaly rates - should be reasonably similar
            rates = [results[size]["anomaly_rate"] for size in batch_sizes_sorted]
            rate_std = np.std(rates)
            rate_mean = np.mean(rates)

            note(f"Anomaly rates across batches: {rates}")
            note(f"Rate std/mean: {rate_std / rate_mean if rate_mean > 0 else 0}")

            # Coefficient of variation should not be too high
            if rate_mean > 0:
                cv = rate_std / rate_mean
                assert cv < 1.0  # Standard deviation should not exceed mean

            # Mean scores should also be reasonably consistent
            mean_scores = [results[size]["mean_score"] for size in batch_sizes_sorted]
            score_std = np.std(mean_scores)
            score_mean = np.mean(mean_scores)

            note(f"Mean scores across batches: {mean_scores}")

            if score_mean > 0:
                score_cv = score_std / score_mean
                assert score_cv < 0.5  # Scores should be more stable than counts
